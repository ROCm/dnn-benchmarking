# Measurement methodology

This page tells you what each number in a result means and how the tool
measures it. The code is `src/dnn_benchmarking/execution/timing.py`
(`measure`), `reporting/statistics.py` and `execution/suite_runner.py`.

## Summary

- The headline number is `kernel_med`: the median device time of one launch
  of the graph, in the table and in `ootb.kernel.median_ms`.
- One loop implementation (`timing.measure`) times hipDNN engines, the
  PyTorch runtime, the rocKE runtime and the AITER runtime. The runtimes get
  the same warmup, stop rule, cache mode and statistics. rocKE rows launch
  with `rocke.runtime.no_fence`, because a fenced rocKE launch synchronizes
  the host after each call. AITER rows launch on the torch default stream.
- Each plan records how it was measured in `timing`: mode, timer,
  cache mode, warmup count, first-call cost, cap and fallback reason.
- TFLOP/s and GB/s use the median. The tool reports the spread (IQR, CV) and
  flags noise. It never removes samples.

## What `kernel_med` measures

The default mode is `staged`. Each timed iteration does these steps on the
execution stream:

1. Arm a stall gate (`hipStreamWaitValue32`). The stream stops at the gate.
2. Record the start event.
3. Read the host clock, call `enqueue()` (one launch of the graph), read the
   host clock again.
4. Record the stop event.
5. Release the gate. The device runs the start event, the graph and the
   stop event without a pause.
6. Wait for the stop event.

The device time is the span from the start event to the stop event. The
host enqueues all work before the gate opens. Thus the span has no host
launch gaps between the kernels of a multi-kernel graph. It is the device
time of one launch, with the kernels back to back.

The `staged` mode needs the HIP timer (`hipdnn_frontend` with the
`HipStallGate` binding, including its watchdog state `timed_out`) and a
device with stream wait-value support. When this is not available, the tool
uses `events` mode and records the reason in `timing.fallback_reason`:

| Reason | Cause |
|---|---|
| `staged timing requires the hip timer` | PyTorch runtime on CUDA, or on ROCm without `hipdnn_frontend`. |
| `hipdnn_frontend is missing staging bindings: ...` | An old `hipdnn_frontend` (for example, one without `HipStallGate.timed_out`). |
| `device does not support hipStreamWaitValue32` | The device or driver has no stream wait-value support. |
| `host sync in enqueue: ...` | The PyTorch graph reads a device value on the host. A stalled stream would then never complete. |
| `stall gate failed in this graph; every row remeasured unstalled` | See below. |
| `another row in this graph fell back to events; every row remeasured in events mode` | One row on the graph fell back to `events` mode, so every row on that graph was remeasured in `events` mode. |

The stall gate has a watchdog. If the gate cannot arm, cannot be created, or
the watchdog releases it before the host finished the enqueue, that sample
includes host gaps and is invalid. The tool then discards every row of the
graph and measures the whole graph again in `events` mode. A warning on
stderr gives the cause.

The same graph-wide rerun applies when a per-row fallback (for example,
`host sync in enqueue: ...` on a PyTorch reference row or a PyTorch/rocKE
engine row) leaves a graph with both `staged` and `events` rows: the tool
discards the partial graph and remeasures every row in `events` mode so all
rows share one timing mode. Rows that did not fail their own probe record
`another row in this graph fell back to events; every row remeasured in events mode`.

These cases can still produce different modes:

- With `--oracle`, the tuned child can use a different mode than the OOTB
  plan. The oracle verdict is then `mixed-timing` (see `docs/usage.md`).
- `--timing-block N` (N > 1) times every row of the run in `block` mode.

Read `timing.mode` and `timing.launch_mode` of each row before you compare
two rows.

In `events` mode each iteration records the start event, calls `enqueue()`,
records the stop event and waits. The device records the start event before
the host finishes the enqueue. Thus the span includes the host submit time and
any host launch gap inside the graph. The two modes agree only when the
device time is much longer than the host submit time. For a short kernel or
a graph with many short kernels, `events` reads higher.

## Submit time

`host` (table column `submit`) is the host time of the `enqueue()` call
only. In `staged` mode the device is stopped at the gate during the call, so
the value is the pure host submit cost. It does not include device time. A
`submit` value smaller than `kernel_med` is normal.

## Priming and `first_call_ms`

Before the warmup, `measure()` always primes the engine:

1. It runs the first `enqueue()` and a full device sync, and records the wall
   time as `first_call_ms`. This value includes one-time costs such as kernel
   compile, MIOpen find and lazy allocation. The console shows it as `setup`
   and `first call`.
2. For the PyTorch runtime, it runs one more `enqueue()` with
   `torch.cuda.set_sync_debug_mode("error")`, then a device sync. If this
   call synchronizes with the host, the loop uses `events` mode.

Priming launches are not timed and never flushed.

### PyTorch SDPA setup

The PyTorch runtime builds SDPA setup that derives only from the inputs in the
first `enqueue()`, so `first_call_ms` includes it and `kernel_med` does not:

- GQA and MQA graphs with equal K and V head counts call
  `F.scaled_dot_product_attention(..., enable_gqa=True)` on K and V at their
  real head count, with the default backend selection and with
  `--pytorch-sdpa-backend flash`. This needs PyTorch 2.5 or later.
- In all other cases the tool expands K and V to the query head count once,
  before the timed loop: when K and V have different head counts, and with
  `--pytorch-sdpa-backend math`, `efficient`, `cudnn` or `overrideable`. Those
  strict backends can reject `enable_gqa`, and a row that does not use it
  still runs.
- Sliding-window and bottom-right masks, where a graph needs one, are built
  once. The dense per-sequence K and V of paged graphs are also built once. A
  timed paged iteration does one SDPA call per sequence. A packed Q (one
  batch row for all sequences) then gets one `torch.cat` along the query
  axis. A Q with one batch row per sequence, padded or ragged, gets a scatter
  of each sequence's rows into a zeroed per-batch output instead (RFC 0014
  limitation 9 covers ragged Q/O over a non-ragged paged K/V cache).
- A variable-length forward graph (`seq_len`, `cu_seq_len` or ragged
  tensors, without a page table) does one SDPA call per batch. The call
  reads the first `lq` query positions of the batch. It reads the first
  `lkv` key positions of the batch. Every mask, including the bottom-right
  causal alignment, uses these live lengths.
- A row with no visible key (a dead row) keeps `O = 0`. Its `stats` is
  `-inf`, or the sink logit when the graph has sink tokens.
  Dead rows are the leading rows of a bottom-right mask when `lq > lkv`.
  They are also the trailing rows of a top-left window when `lq >= lkv + w`,
  where `w` is the window size. A row past `lq` keeps `stats = -inf`, also
  with sink tokens.
- The tool reads the lengths once, before the timed loop. The timed region
  does the slices, the SDPA calls and the copy into the output. A
  variable-length backward graph runs per batch too. `dQ`, `dK` and `dV` are
  0 past each length and on dead rows.
- The `stats` output (log-sum-exp) comes from the fused kernel, as in a
  PyTorch training step. Q, K and V go into the call with `requires_grad`, and
  the tool reads the log-sum-exp that the kernel saves for the backward pass.
  The math backend saves no log-sum-exp. With math, the tool computes it with
  a separate score matmul, which is timed. Its fp32 operands are built once.
  If the kernel saves none under another selection, the tool does the same
  and the row gets a warning, because the timed region then includes that
  matmul.
- Attention sinks have no PyTorch SDPA input. These graphs run attention as
  separate fp32 ops (score matmul, softmax with the sinks, output matmul).
  The fp32 copies of Q, K and V, the K/V expansion, the mask and the sink
  logits are built once. The softmax denominator is also the `stats` output.

With the default backend selection, a GQA call that no fused kernel (Flash
or memory-efficient) accepts would use the math backend. That backend copies
K and V to the query head count inside the call. The tool skips these rows
as unsupported and does not time them. On ROCm, Flash rejects any
`attn_mask` and memory-efficient rejects GQA, so the skipped rows are the
GQA graphs that need a mask tensor:

- a sliding window (any `left_bound`);
- bottom-right causal with a query length above 1 that differs from the KV
  length;
- an explicit `attn_mask` input;
- causal (top-left) with query and KV lengths that differ.

Bounds that equal a native form need no mask and are timed: bottom-right
causal with one query (decode) or with equal query and KV lengths. On MI210
(gfx90a), fp32 GQA graphs are skipped too. Other architectures can differ,
because the result depends on the torch build.

Paged and variable-length graphs run one SDPA call per sequence. When such a
GQA call would use the math backend, the tool tries the call with K and V
expanded to the query head count. It builds the expansion once, before the
timed loop. When a fused kernel accepts the expanded call, the row is timed
with that call. The row is skipped only when the expanded call also needs
the math backend. On MI308X (gfx942), memory-efficient attention accepts the
expanded top-left causal call with unequal query and KV lengths.

The setup is built once on the assumption that the inputs do not change
between iterations. When a graph node computes an SDPA input, the tool skips
the PyTorch row instead of timing the setup.

Some SDPA features have no native PyTorch form. For these rows the timing
includes a workaround, and the row gets a reference warning:

- Paged KV: one SDPA call per sequence, then a `torch.cat` or a scatter into
  a zeroed output.
- Variable-length lengths (`seq_len`, `cu_seq_len` or ragged tensors): one
  SDPA call per batch.
- Sliding window or bottom-right causal with a query length above 1 that
  differs from the KV length: the kernel reads an additive mask tensor.
- Attention sinks: separate fp32 ops instead of an SDPA kernel.
- Explicit dropout mask (`dropout_mask` and `dropout_scale`): separate fp32
  ops instead of an SDPA kernel, because PyTorch SDPA takes no dropout mask.
  The tool builds the setup once, as for attention sinks.
- Different K and V head counts: the kernel reads K and V expanded to the
  query head count.
- ALiBi (`alibi_mask`, top-left causal masking or a top-left sliding
  window): the tool builds the bias `(j - i) * slope(h)` once in setup, for
  query position `i`, key position `j` and query head `h`, and reuses it
  every iteration; hipDNN computes the same bias inside the kernel instead,
  so the two differ in where that cost falls. The slopes follow the cuDNN
  schedule and use the query head, so GQA and MQA graphs use the query head
  count. The timed kernel reads the bias as part of one additive mask
  tensor in the graph data type, so fp16 and bf16 round the bias. The
  timed row keeps the log-sum-exp (`stats`) of the kernel, as a training
  step does. The reference computes `stats` again from the float32 bias.
- Explicit `attn_mask` bias with a causal mask, a sliding window or
  bottom-right alignment: the tool adds the bias first and applies the band
  second. The kernel reads the sum as one additive mask tensor.

### PyTorch SDPA backward

A timed SDPA backward iteration calls the fused backward kernel that a
PyTorch training step uses: `aten::_scaled_dot_product_flash_attention_backward`
or `aten::_scaled_dot_product_efficient_attention_backward`. On ROCm these run
AOTriton or CK. The call reads the graph's O and `stats` inputs, as the hipDNN
kernel does. PyTorch autograd cannot do this, because it keeps its own saved
output and log-sum-exp.

The setup runs once, before the timed loop:

1. Run the forward once with the current `--pytorch-sdpa-backend` selection,
   with `requires_grad` inputs and the graph's mask, bounds and scale. GQA and
   MQA graphs use `enable_gqa` on the real-size K and V, as a training step
   does. If that forward does not run Flash, the tool expands K and V to the
   query head count and runs the forward again.
2. Read from the autograd node which fused kernel ran, and its saved scalars.
3. Copy `stats` into the log-sum-exp layout of that kernel.
4. Copy O into the memory layout of the forward's output. For an O in the
   `[B, H, S, D]` memory order, the memory-efficient backward (AOTriton) of
   PyTorch 2.9.1 and of the 2.14 nightly on gfx90a gives wrong dK and dV
   (ALMIOPEN-3256). A training step also passes the forward's output in this
   layout.

The setup thus reads Q, K, V, O, `stats` and the mask once. If an earlier
node of the graph computes one of these inputs, the row does not run: its
setup cannot be outside the timed region.

A timed iteration does one backward kernel call. With `enable_gqa`, the Flash
backward writes dK and dV at the K/V head count, so the iteration does no
other work. With expanded K and V, the iteration also sums dK and dV over
each head group.

- With the default selection, the backward kernel is the one the forward
  dispatches to. If default dispatch chooses neither Flash nor
  memory-efficient attention (math, or cuDNN on CUDA), the row runs a manual
  fp32 backward formula and gets a warning.
- `--pytorch-sdpa-backend flash` and `efficient` select that kernel. Flash
  takes no mask tensor, so it fails sliding-window and `attn_mask` graphs, and
  bottom-right causal graphs with `Sq > 1` and `Sq != Skv`. It also rejects
  top-left causal graphs with `Sq != Skv`. `--pytorch-rocm-fa-library` applies
  to the timed backward call too. The tool binds no backward for `math`,
  `cudnn` and `overrideable`, so the row is an error.
- The CPU reference (`--validate pytorch` on the host) uses the manual fp32
  formula of hipDNN's CPU reference.
- The timed backward gets the same mask as the forward call of step 1. A
  sliding window, a bottom-right causal band with `Sq != Skv` and a bias
  reach the kernel as one additive mask tensor, as in the forward (see
  [PyTorch SDPA setup](#pytorch-sdpa-setup)). ALiBi graphs use the manual
  fp32 formula. The backward has no static warning for the additive mask;
  only the fallback to the manual fp32 formula is warned at run time.
- A variable-length backward graph (`seq_len` or ragged tensors, without a
  page table) uses the manual fp32 formula, one batch at a time, on the
  first `lq` query rows and the first `lkv` keys of the batch. The fused
  kernels take no lengths. With `flash` or `efficient` selected, the row is
  an error. The reference does not support a variable-length backward with
  a bottom-right causal band or a sliding window.

A graph with a bias gradient (`dbias_tensor_uid`) needs an `attn_mask` bias.
dBias is the unscaled `dS = P * (dP - D)`, summed over each dimension that
the bias broadcasts. It has the dims and the data type of the bias. This is
the definition of hipDNN's CPU reference (`CpuFpReferenceSdpa::backward`)
from rocm-libraries `4630e167ce1` (#14358), which the pinned `28f91cf`
includes.

- The timed row calls the memory-efficient backward kernel with its bias
  gradient enabled, and then sums the score-shaped gradient to the bias
  dims. A training step with a bias that requires a gradient does the same
  two operations. Flash takes no bias, so it cannot compute dBias.
- ALiBi graphs and rows that fall back to math run the manual fp32 formula,
  which also gives dBias.
- The reference accumulates dBias in fp32 and rounds it to the bias data
  type once, as hipDNN's CPU reference does. The timed kernel rounds each
  score-shaped element first. On MI210 the two differ by at most 0.002 on
  the fp16 `swin-base` graph (values up to 6.1) and the bf16 `alphafold2`
  graph (values up to 0.55) of the SDPA backward collection. The SDPA
  backward tolerances apply to dBias unchanged.

### PyTorch convolution backward

A timed dgrad (`ConvolutionBwdAttributes`) or wgrad
(`ConvolutionWrwAttributes`) iteration calls `aten::convolution_backward`
once. That call needs the operand that it does not differentiate only for
the shape and the memory layout. The tool gives it an uninitialized tensor
that is dense in the graph layout of the output. Thus the call does not copy
the operand.

`convolution_backward` has no asymmetric padding (`pre_padding` different
from `post_padding`). For these graphs:

- dgrad computes the gradient of the padded input. The output is a view of
  the unpadded part. The view adds no kernel.
- wgrad pads x with `F.pad` inside the timed region, as the forward conv
  does.

## Warmup

`--warmup N` (default 10) is a launch count, not a time budget. The priming
launches are part of the count. The remaining `N - 1` launches (`N - 2` for
the PyTorch runtime) go through the timed-iteration path: the same mode
(`staged` or `events`), the same per-iteration sync and, in `cold` mode, the
same flush. The tool discards their samples.

Thus the GPU clocks and the caches reach the state of the timed loop before
the first sample. Back-to-back warmup launches do not do this: the timed loop
syncs after each launch, so the clocks can change at the start of the loop.
On MI210 (`sample_conv_fwd`, 100 iterations), back-to-back warmup gave a
kernel CV of 9.7 % and a maximum of 50.6 us (2 x median). Warmup through the
timed path gave a CV of 0.95 % and a maximum of 26.2 us. The median did not
change (25.4 to 25.6 us).

`timing.warmup_iters` is the number of untimed launches that actually ran
inside `measure()`. Outside block mode it is `max(priming launches, --warmup)`:
priming is 1 launch for hipDNN, 2 for torch-driven runtimes (PyTorch, rocKE
and AITER: first call plus host-sync probe), and 3 when the probe finds a host
sync and runs the launch again. `timing.requested_warmup_iters` records the
requested `--warmup` separately. In block mode `warmup_iters` is the priming
launches plus `--warmup` for every sample, the discarded first sample included.

## Event-span overhead

`kernel_med` is a **device event span** (stop event minus start event), not a
profiler dispatch sum. On ROCm, even an empty or busy-wait staged span includes
a fixed event/dispatch boundary floor (measured at ~7.8 µs on MI210, ~3.1–3.7 µs
on MI325X, and ~3.6–4.1 µs on MI355X). For short kernels (<50 µs), this common
offset compresses cross-runtime latency ratios toward 1.0x. Do not subtract a
universal constant across workloads; use dispatch sums from a profiler trace
when isolating sub-20 µs device busy time.

## Cache modes

| Mode | Behavior |
|---|---|
| `warm` (default) | No flush. Inputs, outputs and workspace stay in L2 and MALL between iterations if they fit. |
| `cold` | Before each warmup and timed iteration: run a coalesced, read-only capacity sweep over unrelated device memory on the benchmark stream, then drain the device. Calibrated defaults: 24 MiB on `gfx90a`, 640 MiB on `gfx942`/`gfx950`, and 512 MiB fallback on unlisted or unknown architectures. |
| `rotate` | Allocate `N` independent copies of the graph I/O buffers (`--rotate-buffers`, default auto-sized from the active GPU's cache eviction target and capped at 60% of free VRAM) and advance to the next copy on every launch with zero extra dispatches between launches. Works across all four runtimes and with both per-launch and `--timing-block N` timing. |

Notes for `cold` and `rotate`:

- The `cold` sweep and drain run outside the timed span. Priming launches are
  never flushed. `timing.flush_method` records `read_sweep` (or `zero_fill` if
  a runtime fallback is used; never relabeled) and `timing.flush_bytes` records
  the exact byte size.
- Unlike the legacy zero-fill flush, the read-only sweep leaves no dirty lines
  in L2/MALL for the timed kernel to write back.
- The flush buffer is allocated once per process and freed before `--oracle`
  child processes start.
- Neither `cold` nor `rotate` resets the instruction cache or guarantees an
  empty hardware cache; they empirically evict benchmark data.
- `--cache-mode cold` requires `--timing-block 1`; use `--cache-mode rotate`
  for cold block timing.

`dnn-benchmark compare` refuses to compare files with different `cache_mode`,
`timing_block`, or `input_recipe` values unless you give `--allow-mismatch`.

## Input-value recipes

Input distributions affect power draw, clock frequency, and throughput even
when the dispatched kernel is unchanged. `--input-recipe` selects a named,
seeded, versioned recipe (`run.config.input_recipe` and
`graph.input_fingerprint`):

- `centered` (default): seeded `N(0, 1)` for unconstrained floating-point
  operands (GEMM, activations), `N(0, d^-0.25)` for SDPA Q/K (keeping `Q @ K^T`
  unit-variance even when `attn_scale_value` is omitted and hipDNN defaults to
  `scale = 1.0`), `U[0, 1)` for SDPA V (keeping `O = softmax(QK^T)V` centered
  near 0.5 instead of cancelling near zero under bf16 tolerance), signed
  `U[-1, 1)` for fp16-output convolutions, `U[0, 1)` for fp32/fp64-output
  convolutions (avoiding near-zero cancellation against the float64 reference's
  fixed `atol`), and strictly positive `U[0.5, 1.5)` for normalization
  variances, inverse variances, inverse RMS, scales, and epsilons.
- `llm`: role-aware heavy-tailed distributions fitted to measured LLM
  checkpoints (Student-t weights `std ~ 0.03`, heavy-tailed activations and
  Q/K, Laplace V, positive norm gains centered near `1.0`).
- `heavy_tail`: 99% `N(0, 0.8^2)` + 1% `N(0, 6^2)` clipped to `±16`.
- `legacy`: historical `U[0, 1)` generator (with `U[-1, 1)` for fp16 conv).
- `zeros`: diagnostic all-zero float fill (positive norm variances/scales kept
  at `1.0`).

## Stop rule

The loop stops when both conditions are true:

- The number of samples is `--iters` or more (default 100).
- The sum of the kernel times is `--min-time-ms` or more (default 0).

With `--min-time-ms 0`, the loop runs exactly `--iters` iterations. A hard cap
of `max(10000, --iters)` samples (`TimingPolicy.max_iters`) stops the loop in all cases. If
the cap stops the loop before the time budget is reached,
`timing.capped` is `true` and the row gets the warning `capped at max_iters`.

Use `--min-time-ms` for short kernels. For example, a 10 us kernel with
`--iters 100` gives 1 ms of samples. With `--min-time-ms 100` it gives about
10000 samples.

## Block timing

`--timing-block N` (N > 1) uses the rocKE block protocol (`time_launches`,
Solera `measure()`), in `timing.mode = block`:

1. Run the priming launch (timed as `first_call_ms`).
2. For each sample: run `--warmup` untimed launches, then drain the device.
3. Time `N` back-to-back launches in one event pair. Record `elapsed / N`.
4. Discard the first sample.

The stall gate is not used: `N` gated launches can fill the HIP queue and
block the host before the gate opens. Block timing hides the launch gap
between back-to-back kernels, so it reads lower than the per-launch modes
for short kernels. On the MI210 conv sample, `--timing-block 50` read
18.7 us against 25.6 us staged. Compare runs only with the same
`timing_block`. `--cache-mode cold` requires `--timing-block 1`, because a
flush before a block leaves only its first launch cold.

A PyTorch graph that reads a device value on the host inside `enqueue()`
sets the same `host sync in enqueue: ...` reason in `timing.fallback_reason`
in block mode. The row warning is then `block timing: <reason>`.

## Statistics

For `kernel` and `host` the result file stores `p25_ms`, `median_ms` and
`p75_ms`; the sample count `n` is the plan's `timing.samples`.
The interquartile range (IQR) is `p75_ms - p25_ms`. `median_ms` is
the upper median `sorted(t)[n // 2]`, the rocKE / Solera definition, so it is
always an observed sample. The quartiles use linear interpolation. The
verbose console also shows the mean, std (ddof 1), min, p95 (below 20
samples it shows `-`) and max.

The tool does not remove outliers. It adds warnings to the row:

| Warning | Condition |
|---|---|
| `noisy: IQR x% of median` | `iqr_ms / median_ms` more than 5 % and 10 or more samples. The table marks `kernel_med` with `*`. |
| `outlier: max Nx median` | Maximum more than 2 x median. The table marks `kernel_med` with `*` and shows the warning in `note`. |
| `timer resolution: zero or nonpositive sample(s)` | One or more timed samples are `<= 0` ms. |
| `host-bound: submit is x% of kernel_med in <mode> timing; ...` | Block or events timing only. The median host submit time is 90 % or more of the median span. The span likely measures host launch overhead rather than pure device time. |

The noise flag uses IQR/median, not CV (the console verbose block reports
mean, standard deviation, min, p95 and max; CV is derivable as `std / mean`
but is not printed directly). The `outlier` flag reports slow spikes.
The median is the headline because one slow iteration moves the mean but not
the median. In the MI210 runs below, one 37.6 us sample in a 29 us matmul run
moved the mean and the standard deviation, but not the median.

## Input data for variable-length SDPA

An SDPA node is variable-length when it has a length source on a side:
`seq_len_q_tensor_uid`, `seq_len_kv_tensor_uid`, `cu_seq_len_q_tensor_uid` or
`cu_seq_len_kv_tensor_uid`. A tensor is ragged when it sets
`ragged_offset_tensor_uid` (with the optional `ragged_offset_multiplier`,
default 1). The tool draws the values of these integer tensors from the seed.
They are not uniform floats:

- Lengths are in `1..S_max`, where `S_max` is the sequence dim of the
  operands. A group of lengths varies between batches. With two or more
  batches, one batch is padded (shorter than `S_max`) and one batch is full.
- For a paged graph, the KV lengths stay within the page allocation:
  `page_table` width times page size, capped by `max_seq_len_kv` when the
  graph sets it. The width is the product of the table dims after the
  first: hipDNN's tables are `[B, 1, max_blocks_per_seq, 1]`.
- An offset table has `B + 1` entries and starts with 0. Entry `b + 1` is
  the start of batch `b + 1`. That is the start of batch `b` plus its length
  and its gap, in the unit of the table. Only a group of tensors that has a
  `seq_len` gets gaps. A gap is padding tokens after the live tokens of a
  batch, and the length excludes them. With two or more batches, some batch
  starts after a gap. The exception is a graph where no batch before the last
  has room for one. Then every length is at the aligned sequence length, or
  the token capacity holds just one aligned block for each batch. Each entry
  is stored in the unit of the table. That is the token start times the
  sequence stride divided by `ragged_offset_multiplier`.
- A head-major ragged tensor (for example `[H, T, D]`) stores the token
  range in the head stride. The capacity of that range limits the lengths
  and the gaps. The tool reduces them to fit, and keeps one gap before a
  batch other than the last when the capacity has room for it. A graph that
  cannot fit one aligned block of tokens for each batch is not served.
- `cu_seq_len` tensors hold the running totals of the lengths: `0`, `l0`,
  `l0 + l1`, and so on, `B + 1` entries, in the dtype of the tensor. A side
  uses `seq_len` or `cu_seq_len`, not both. The two sides can use different
  sources. The key names follow rocm-libraries PR #14560 (forward
  `SdpaAttributes` only; ALMIOPEN-2935). hipDNN has not decided whether an
  engine reads `cu_seq_len` as the batch starts (as FlashAttention does) or
  as lengths only. Thus a group whose only length source is `cu_seq_len` gets
  no gaps: its batch starts, in tokens, equal `cu_seq_len`, and both readings
  are correct.
- Tensors that share one length or offset tensor get one set of values. For
  example, self-attention can use one offset table for Q and K.
- The lengths and the offsets come from a separate random stream that the
  seed also sets. The float inputs of a graph do not change when the graph
  has these tensors.

Every tensor of the graph stays logical (shaped by its dims) in the
reference. The graph layout (offsets, strides) applies only when the tool
writes inputs to device storage and reads outputs from it. A ragged tensor
stores each batch at its offset. Its padding (past each length and between
batches) holds generated values, as the padding of a non-ragged tensor (for
example the BSHD view of aiter graphs) and the golden data of hipDNN do. An
engine must ignore these values.

## TFLOP/s and GB/s

`tflops` and `gbps` are **graph-normalized rates**, not physical hardware
compute utilization or measured HBM bandwidth:

- `tflops = flops / (kernel.median_ms * 1e9)` (decimal 10^12 FLOP/s).
- `gbps = io_bytes / (kernel.median_ms * 1e6)` (decimal 10^9 bytes/s).
- `flops` is an analytical count from the graph JSON (`FMA = 2 FLOPs`; version
  recorded in `metrics.flops_formula_version`). Convolution counts use the
  standard dense-equivalent convention (every tap at every output position,
  including padded boundary positions). Normalization, reduction, pointwise,
  and pooling ops use fixed per-element proxy multipliers rather than
  instruction counts. SDPA counts only the QK and PV contractions (excluding
  softmax). When any node has no formula or lacks required dims — including
  packed paged SDPA with a diagonal mask and no sequence lengths — `flops` and
  `tflops` are `null`.
- Every runtime row on a graph shares the same graph numerator, even if one
  runtime executes extra padding, fill, or copy kernels inside its timed call
  (noted in `timing.timed_work_notes` where known).
- `io_bytes` is the **graph tensor footprint** across non-virtual tensors, not
  a lower bound on HBM traffic: strided spans include unreferenced holes,
  non-ragged paged K, V, Q, and O containers count their full storage size, and
  non-ragged padded outputs may still be written to memory. Variable-length
  tensors with valid lengths count only their valid elements.
- In `warm` mode, working sets that fit in L2/MALL can report `gbps` above HBM
  bandwidth. Use `--cache-mode cold` or `--cache-mode rotate` to evict
  benchmark data from caches.

## GPU telemetry during and after the timed loop

With `--metrics` (default) and amdsmi available on the mapped HIP PCI device,
the runner samples GPU clocks, power, hotspot temperature and throttle status
during the timed loop and once immediately after it (`metrics.clocks_after`).
The row gets the warning `throttled` when either the post-loop
`throttle_status` is nonzero or `throttle_any` is `true`. The tool does not
set or lock clocks.

## PyTorch dispatch record

With `--pytorch-dispatch`, a PyTorch row records which SDPA backend and which
GPU kernels its timed path ran, in the plan's `dispatch` object (see
[results-schema.md](results-schema.md#dispatch)). The option is off by
default, and `dispatch` is then `null`. After the timed loop, the tool runs
the graph one more time under `torch.profiler`, with CPU and GPU activities:

1. The tool synchronizes the device. No input-fill, warmup or cache-flush
   kernel is then in the trace.
2. It runs one launch with the inputs that the timed loop primed. The setup
   of the first call (fused-kernel probes, layout copies, host reads) does
   not run again, so its kernels are not in the trace.
3. It synchronizes the stream and stops the profiler.

The launch is not timed and gives no sample. `elapsed_s` of the row includes
it. The first profiler start in a process can take about one second.

- `kernels` holds the `kernel` events of the trace. Copies and fills are not
  kernels.
- `sdpa_backend` comes from the ATen SDPA op that the launch called, for
  example `aten::_scaled_dot_product_flash_attention`.
- On a reference row, the launch uses the timed backend selection, not the
  MATH selection of the reference output pass.
- With `--oracle`, the tuned child process records its own launch, so
  `oracle.dispatch` shows the kernels that tuning selected.
- The tool sets `KINETO_LOG_LEVEL=6` when it is not set. The profiler then
  writes no lines on stderr. A failed capture gives a row warning.
- When a `torch.profiler` session is already active in the process (for
  example, a caller that profiles the run), the tool does not start a second
  one. Kineto runs one session at a time, and a second one stops the first.
  The row then has no `dispatch` and gets a warning. The check finds a
  session of the same thread. It finds a Python `torch.profiler` session of
  another thread only through a private PyTorch flag
  (`torch.autograd.profiler._is_profiler_enabled`). PyTorch has no public
  check for this case, so the capture is opt-in.
## Profiling child process

`--pmc`, `--trace`, `--perf` and `--roofline` do not alter the parent's buffer
backend or overwrite timed samples. After all OOTB rows of the graph complete
(and GPU reference tensors are released), the tool starts one child process for
each pass under the profiler:
```text
python -m dnn_benchmarking --internal-profiling-run --graph G --engine E \
    --warmup W --iters 5 --seed S [--plugin-path P] [--input-recipe R] [--cache-mode rotate --rotate-buffers N]
```

The child builds one engine, fills the inputs with the same seed and recipe as
the timed run, and runs `W + 5` plain graph executes (rotating the same `N`
buffer sets when the timed run used `--cache-mode rotate`, so L2/MALL counters
match rotated timings without adding extra flush kernels): the same submissions
as the timed loop, with no workspace reset and no sync between them. Then it
drains the
device once. It does not use the timed loop, for two reasons:

- `rocprofv3 --pmc` serializes dispatches. A stalled stream then never
  starts, and the stall gate would deadlock.
- A profiler scopes its counters to a process. A new process gives a clean
  process tree and address space.

Before each child starts, the parent frees the engine buffers. The child thus
has the same free VRAM as the timed run. Counter values come from this child,
not from the timed loop. The child also launches input-fill kernels, and a
graph can launch more than one kernel. For an SDPA backward graph, the input
fill includes the PyTorch attention forward that computes O and `stats` (see
[SDPA backward inputs](#sdpa-backward-inputs)). Its matmul, `exp` and
`logsumexp` kernels run on the GPU before the graph executes, and the
profiler records them. Use `pmc.per_kernel[...].dispatches` and the kernel
name to find the engine kernel.

## Fast mode

With `--fast`, every graph of the run uses the same hipDNN handle, the same
loaded plugin and the kept OOTB selection state of the work directory (the
tuned state with `--autotune`; see
[usage.md](usage.md#fast-mode---fast)). This changes what some numbers of a
graph include:

- `build_ms` is not the same build in the two modes. The prime builds the
  plan of the graph itself, untimed. A normal run primes before every graph,
  so each `build_ms` is the second build of that graph. A `--fast` run primes
  before the first graph only: its `build_ms` is a second build, and the
  `build_ms` of every later graph is the first build of that problem on a
  warm engine. It can include per-problem first-build costs, for example
  reading that problem's kernel files.
- `first_call_ms` (`setup`) can be smaller than in a normal run. The MIOpen
  kernel cache of the handle and the find records of earlier graphs and
  earlier commands carry over, so a graph with a known convolution problem
  skips the compile and the find.
- `kernel_med` measures the same kernel as a normal run when the selection is
  the same. With `--autotune`, a problem that an earlier graph or command
  searched reuses that search result.

The order of the graphs can thus change `build_ms` and `first_call_ms`. Do
not compare them between a `--fast` run and a run without it, or between the
first graph of a `--fast` run and the later ones.

## Kept selection state

MIOpen stores its find results in the user database. A convolution's first
call runs the find when the database has no record for the problem, and the
find can take minutes per problem: on one gfx1151 host, 7 convolution graphs
took 475 s on an empty database and 46 s on the database of an earlier run.

The work directory keeps two selection states (see
[usage.md](usage.md#work-directory---benchmark-work-dir)).
`--autotune` runs, runs with a true `HIPDNN_FORCE_BENCHMARKING` in the
environment, and the tuned runs of `--oracle` read and extend
`<work>/tuned`. Every other run, `--fast` included, reads and extends
`<work>/ootb`. No tuned run writes to `<work>/ootb`. The find results and
rankings of earlier commands remove that setup work. In such a warm run:

- `first_call_ms` and the run's wall time are smaller; do not compare them
  with a run on an empty state.
- `kernel_med` measures the kernel that the stored result selects. It
  compares with a cold run when both select the same kernel.

Use `--fresh-state` for cold first-run numbers: the command then runs on a
new, empty state that the tool removes when the command ends.

## Dropout validation

`--validate pytorch` follows the hipDNN CPU reference for SDPA dropout. In
the formulas below, `*` is the element-wise product and `@` is the matrix
product. `P` is the softmax of the scores before dropout, `M` is the
dropout mask and `s` is the dropout scale.

- Forward with an explicit mask: `O = (P * M * s) @ V`. The tool multiplies
  the softmax by the mask and the scale, then by V. `M` is the
  `dropout_mask` input, with its values as given. A mask dimension of size
  1 applies to all batches or heads. The head index of `M` is the query
  head. The `stats` output (log-sum-exp) comes from the softmax before
  dropout. With attention sinks, the softmax includes the sink before the
  mask applies.
- Forward with Philox: the same formula, with the `rng_dump` of the engine
  row as `M`. The scale is `s = 1 / (1 - p)` in float32, where `p` is
  `dropout_probability`. At `p = 0` the dump applies as is, with `s = 1`.
  The tool does not read the values of `seed` and `offset`.
- Backward with an explicit mask: `inv` is the `dropout_scale_inv` input, or
  1 when the graph has none. `O` is the forward output that the graph gives.
  - `D = rowsum(dO * O) * inv`: the row sums of `dO * O`, times `inv`.
  - `dP = (dO @ V^T) * M`: the mask applies, the scale does not.
  - `dS = s * P * (dP - D)`.
  - `dV = (P * M * s)^T @ dO`.
  - `dQ` and `dK` come from `dS`, as without dropout. `inv` changes only
    `D`, so only `dQ`, `dK` and dBias show a wrong `inv`.
  - With a bias gradient, dBias is the `dS` above: `s * P * (dP - D)`.
    `dP` and `D` already carry `M` and `inv`. The tool takes dBias
    before the attention scale. It sums dBias over the bias's broadcast
    dims, as hipDNN's CPU reference does.

Dropout combines with causal masks, sliding windows, an additive bias, ALiBi
and attention sinks, where the reference supports that mask. Like the hipDNN
references, the reference declines these combinations:

- a mask or a scale with `dropout_probability`, `seed`, `offset` or
  `rng_dump`;
- a mask without a scale, or a scale without a mask;
- a mask that is not FLOAT with the dimensions `[B or 1, Hq or 1, Sq, Skv]`,
  or a scale that is not one FLOAT element;
- a `dropout_scale_inv` without a mask, or one that is not one FLOAT
  element;
- `seed`, `offset` or `rng_dump` without a `dropout_probability`;
- an `rng_dump` with a `dropout_probability` but without `seed` or `offset`;
- a `dropout_probability` that is NaN, less than 0, or 1 or more;
- an `rng_dump` that is not FLOAT, or that does not have the dimensions
  `[B, Hq, Sq, Skv]`;
- dropout with fp8 scaling.

For a backward node, the tool takes a `dropout_probability` of 0 without a
dropout mask, `dropout_scale` or `dropout_scale_inv` as no dropout, with or
without `seed` or `offset`, as hipDNN's backward references do
(rocm-libraries #15643). A probability of 0 drops no element, so the
gradients are those without dropout.

The hipDNN backward references still decline a `dropout_probability` above
0: the backward graph API has no `rng_dump`, so no reference can replay the
Philox mask (rocm-libraries #14358). The FlyDSL backward engines
(rocm-libraries #15643) serve Philox dropout in the backward, but no
reference validates them.

The engine makes a Philox mask, so the mask differs from row to row. A
reference built from the dump of one row does not apply to another row.
The tool thus builds the reference of each engine row from the dump of that
row, and the graph has no `reference` row. It compares every output except
the dump.

Before it builds that reference, the tool checks the dump against the
same statistical bound the hipDNN integration tests use, applied to the
count of ones instead of their rate. In this check, `n` is the number of
elements of the dump and `ones` is the count of ones. `keep` is `1 - p`
in double precision, from the float32 `p`:

1. Each value must be 0 or 1. -0.0 counts as 0, and NaN fails. The message
   names the first bad index.
2. The count of ones must be in the bound
   `|ones - keep*n| <= 6*sqrt(keep*(1-keep)*n) + 1`. The first term is 6
   standard deviations of the count. The second term allows one element
   off the keep rate. An empty dump passes.

The two forms are algebraically identical in real arithmetic: dividing
this bound by `n` gives the hipDNN harness's rate form. They diverge at
floating-point rounding, so hipDNN's rate form can still reject a dump
that this count form accepts, for example one off-keep element in an
all-kept dump at `dropout_probability = 0`.

The tool fills the dump with NaN before the validation run. An engine that
does not write the dump thus fails step 1, for all `n` and `p`. Without the
fill, a dump of zeros can pass step 2 when `n` is small and `p` is high. An
example is `p = 0.9` and `n = 256`.

The tool does not make the Philox stream again from `seed` and `offset`.
The hipDNN references do not make it either: they decline a Philox graph
without `rng_dump`. The check thus does not show that the dump is the
stream of that `seed` and `offset`. It shows two things:

- The dump is a 0/1 mask at the expected keep rate.
- The outputs match a reference built from that dump, within the tolerance
  of the output data type.

The tolerance limits the second result. With bf16 outputs and thousands of
keys, the tolerance cannot tell the dumped mask from another mask. It also
cannot find an engine that normalises the softmax again after the mask.
Another engine can write a correct dump but not apply the scale `1 / (1 - p)`.
Its outputs are then about `p` too small, relative to the reference. Such an
engine passes when `p` is below about the relative tolerance of the output
data type. That tolerance is 0.03 for bf16 and 0.001 for fp16.

## Parity with other harnesses

This table compares the tool with the rocKE runtime timer
(`rocke/runtime/launcher.py:time_launches`, `rocke/benchmark/summary.py`) and
with PyTorch (`torch.utils.benchmark.Timer.blocked_autorange`,
`triton.testing.do_bench`).

`--runtime rocke` does not use `time_launches`. It times rocKE launches with
`timing.measure`, so rocKE rows and hipDNN rows use the same protocol. With
`--timing-block N`, the protocol is the block protocol in the table.

| Feature | dnn-benchmarking | rocKE | PyTorch / Triton |
|---|---|---|---|
| Warmup | Fixed count (default 10) through the timed path, first call timed separately; per sample in block mode | Fixed count (default 5) per sample | do_bench: 25 ms time budget. Timer: block-size estimate. |
| Iterations | `--iters` floor plus optional `--min-time-ms` budget, cap `max(10000, --iters)` | Fixed (default 100) | do_bench: 100 ms budget. `blocked_autorange`: 0.2 s minimum. |
| Event granularity | One event pair per launch, stall-gated; `--timing-block N`: one pair around N launches | One event pair around N launches | do_bench: one pair per launch. Graph variant: one pair per replay. |
| Sync | After each launch; block mode: after each block | One at the end | do_bench: one at the end. Timer: one per block. |
| Host launch gaps removed | Yes (stall gate) | Amortized over N launches | Amortized (do_bench) |
| Host submit time | Yes (`host`) | No | Timer: wall time only |
| Cache flush | `--cache-mode cold` (per-arch read-only sweep) | Documented only | do_bench: 256 MB zero before each launch |
| Rotating buffers | `--cache-mode rotate` (`--rotate-buffers`) | Documented only | No |
| Graph replay | Raw launches (`launch_mode = raw` recorded) | Yes | `do_bench_cudagraph` |
| Outliers | Flagged (IQR/median noise flag, max/median outlier flag), never removed; block mode discards the first sample | Discard first run | Timer: IQR warnings |
| Statistics | n, mean, std, min, p25, median, p75, p95, max, IQR | median, min, max, mean, stdev, spread | do_bench: mean or quantiles. Timer: median, IQR. |
| Headline | Median | Median over attempts | do_bench: mean (default). Timer: median. |
| TFLOP/s basis | Median | Median | User calculates |
| Cross-process repeats | No | Yes (attempts, new process) | No |
| Clock capture | Sampled during and after the timed loop | Advice: lock clocks | None |
### Measured numbers (MI210, gfx90a)

These runs used the default warmup (10) and `--iters 100`. Values are kernel
medians in us.

Run-to-run stability, three runs each:

| Graph / engine | Run 1 | Run 2 | Run 3 | Spread | Min to max in one run |
|---|---|---|---|---|---|
| conv_fwd / MIOPEN_ENGINE | 25.44 | 25.60 | 25.44 | 0.16 (0.63 %) | 25.28 to 26.24 |
| conv_fwd / MIOPEN_ENGINE_DETERMINISTIC | 25.60 | 25.60 | 25.60 | 0.00 | 25.12 to 26.40 |
| matmul / HIPBLASLT_ENGINE | 28.96 | 28.96 | 28.96 | 0.00 | 28.48 to 37.60 |

The medians fall on a 0.16 us grid. This is the HIP event resolution, so the
run-to-run variation is at timer resolution.

Other methods on the same shapes:

| Method | conv_fwd | matmul |
|---|---|---|
| This tool, hipDNN, staged | 25.44 to 25.60 | 28.96 |
| This tool, `--runtime pytorch`, staged | 26 | 29 |
| Stall-gated timer on the raw PyTorch op | 25.76 to 25.92 | 28.96 |
| `triton.testing.do_bench` median (L2 flush, event pair per launch) | 25.76 | 29.12 |
| Event pair per launch after a 256 MB L2 flush | 25.44 to 25.76 | 28.96 to 29.12 |
| Batched: 100 launches per event pair | 24.98 to 25.16 | 24.77 to 25.21 |
| `blocked_autorange` median | 24.53 to 24.77 | 24.42 to 24.94 |
| Event pair per launch, no gate (includes launch gap, not valid) | 51.4 to 53.1 | 50.7 to 52.3 |

What the numbers show:

- The tool agrees with `do_bench` to within 1.3 % (25.44 against 25.76 us
  on conv_fwd). The stall gate removes the host launch gap. Without the gate,
  an event pair per launch reads about two times too high.
- For kernels of 25 us or less, the tool reads about 1 % to 19 % above the batched
  and `blocked_autorange` values. Those methods launch back to back, so each
  kernel starts while the previous kernel ends. The tool measures one
  isolated launch.
- For a 4096 x 4096 matmul the methods agree: 4183.7 us (staged),
  4234.3 us (batched), 4193.5 us (flushed).
- L2 state did not cause the difference for these shapes: 28.80 us warm and
  28.64 us cold.

hipDNN and `--runtime pytorch` on the same graphs, kernel median in ms:

| Graph | hipDNN | PyTorch |
|---|---|---|
| add | 0.078 | 0.071 |
| batchnorm inference (32 x 64 x 28 x 28) | 0.024 | 0.077 |
| conv_dgrad | 0.026 | 0.049 |
| conv_fwd | 0.026 | 0.026 |
| matmul | 0.029 | 0.029 |
| relu | 0.155 | 0.156 |
| rmsnorm | 0.010 | 0.032 |

These runs used a back-to-back warmup, so the minimum-to-maximum ranges can
include clock-ramp outliers.

Bimodal kernels (CV 0.1 to 0.3), for example `conv_dgrad`, have a median
that can move between the two modes from one measurement to the next. Their
mean moves less. Compare the mean and the CV of such a row before you report
a change.

## Paged SDPA inputs

A paged SDPA graph gives K and V as a block container `[num_blocks, H_kv,
page_size, D]`, and the page tables as int32 `[B, 1, max_blocks_per_seq,
1]`, as hipDNN and cuDNN require. The tool also reads rank-2 `[B,
max_blocks_per_seq]` tables. It fills the page tables with block ids
0, 1, 2, ... in sequence order, so no two sequences share a block when the
container has at least `B * max_blocks_per_seq` blocks. A smaller container
wraps the ids around its block count. The sequence lengths follow
[Input data for variable-length SDPA](#input-data-for-variable-length-sdpa).

## SDPA backward inputs

An SDPA backward node reads Q, K, V, the forward output O, the incoming
gradient dO and the forward log-sum-exp `stats`. The tool draws Q, K, V and
dO from U[0, 1) like other inputs, and then scales dO (see below). It
computes O and `stats` as the
exact float32 forward of Q, K and V with the graph's scale, mask, bounds,
additive bias, ALiBi and GQA layout. The forward runs in PyTorch, on the GPU
when one is visible;
without PyTorch a NumPy fallback gives the same values. O is rounded to its
graph type. Both runtimes and the reference read these values.

Independent random O and `stats` do not describe a forward pass. On gfx942
(ALMIOPEN-2976), random values gave gradients of 1e4 to 1e5 and fp16
overflow, and the ASM SDPA engine failed validation on every graph. An upper
bound for `stats` without a forward kept the gradients finite but between
4e-6 and 0.13, so the tolerances could not detect an error. With the forward
values, the gradients were 1 to 7 and the engine passed on every bf16 graph.
The kernel times were the same within 1% for all three input forms.

With many keys per query, as in decode shapes, dV and dK elements can all
be below 1e-3 even with forward values (one query and 4,096 keys without
GQA: dK up to 2e-4, dV up to 5e-4). The fixed `atol` of 1e-3 would then
accept an engine that never writes dK or dV. Every gradient is linear in
dO, so the tool scales dO like AMP loss scaling: when the largest element
of `dV = P^T dO` is below 0.5, dO is multiplied by the power of two that
puts it in [0.5, 1). The scale is at most 2^14, so fp16 dO stays finite.
A power of two keeps dO exact at its graph type. The other gradients scale
by the same factor. A square shape with Sq = Skv = 1,024 gets no scale,
because dV is already near 1; its dQ stays below 1e-2.

fp16 SDPA backward outputs use `rtol` 4e-3 (about 4 ULP) instead of the fp16
default of 1e-3. dQ, dK and dV sum over every key or query row. The value
was measured on gfx942 against the earlier AOTriton reference: the ASM SDPA
engine and AOTriton differed by 2 ULP on 1 in 10,000 to 1 in 50,000
elements (`max_rel_diff` 2^-9), so 1e-3 failed two correct kernels. It is
not yet measured again against the fp32 reference. The source is
`OP_TOLERANCES` in `src/dnn_benchmarking/execution/correctness.py`.

With `--validate pytorch`, the SDPA backward reference is the manual fp32
formula, on a GPU and on the host. It reads the graph's O and `stats`. The
reference pass does not call the fused PyTorch backward kernel: only the
timed PyTorch row does (see [PyTorch SDPA backward](#pytorch-sdpa-backward)).

A graph with sequence lengths (`seq_len_q` and `seq_len_kv`, or ragged
tensors) gets the forward of each batch on its first `lq` query rows over
its first `lkv` keys, with the generated lengths (see
[Input data for variable-length SDPA](#input-data-for-variable-length-sdpa)).
The engines and the reference read these rows. A bottom-right band follows
the live lengths. The rows past `lq` keep the forward of the full shape. The
dO scale uses the largest dV element of the per-batch forwards.

The forward has no dropout, so O of a dropout graph is the forward without
dropout. The forward adds host time to input generation
before the timed loop. It is proportional to the attention FLOPs of the
graph. Forward-consistent SDPA backward inputs and the scaled dO start at
tool version 0.4.0. Do not compare SDPA backward results with results from
earlier versions.

## Not yet

The tool does not do these things yet:

- Float64 reference for all operations. `--validate pytorch` computes the
  reference of convolution graphs (convolution and pointwise nodes) in
  float64. See [Reference precision](usage.md#reference-precision). Other
  graphs (matmul, normalization, SDPA) still use the graph data types, so
  for fp16 and bf16 the reference has the same precision as the result under
  test, and the tolerances must be loose.
- SDPA reference for every mask. The PyTorch reference supports top-left and
  bottom-right causal bands, bounded left windows with `right_bound = 0`,
  attention-sink logits, and ALiBi with top-left causal masking or a
  top-left sliding window. An explicit `attn_mask` bias can combine with a
  causal band or a sliding window. It resolves the mask as hipDNN does: a
  deprecated causal boolean acts as `right_bound = 0`, and the stored
  `left_bound` and diagonal alignment still apply. As in cuDNN and in hipDNN
  since rocm-libraries #14483, `left_bound = L` keeps `L` keys up to and
  including the diagonal, so a window of `L` keys has `left_bound = L`. The
  same rule sets the FLOP count and the SDPA backward inputs. Before 0.4.0,
  `left_bound = L` kept `L + 1` keys, so a window graph without a causal
  boolean has a band one key narrower at 0.4.0, with a different FLOP count,
  reference and backward inputs. Do not compare window rows across 0.3.0
  and 0.4.0. Tool versions before 0.4.0 also let the deprecated boolean
  override the bounds and the alignment. For a graph that sets a causal
  boolean with a bound or with `BOTTOM_RIGHT`, the FLOP count and the
  PyTorch and rocKE masks thus changed at 0.4.0. The reference declines
  these graphs as unsupported:
  `left_bound = 0`, which hipDNN rejects; ALiBi with bottom-right alignment
  or with neither a causal mask nor a sliding window; padding masks; paged
  or variable-length ALiBi graphs. For dropout, the reference supports an
  explicit dropout mask (forward and backward), a `dropout_probability` of
  0 in the backward (no dropout, with or without `seed` or `offset`), and
  Philox forward graphs with an `rng_dump` output (see
  [Dropout validation](#dropout-validation)). It declines Philox forward
  graphs without `rng_dump`, Philox backward graphs (`dropout_probability`
  above 0, with `seed` or `offset`), and dropout with paged, ragged or
  variable-length K/V. The PyTorch reference does not
  implement that last combination; it is a limit of this tool, not of
  hipDNN, which densifies paged and ragged dropout in its own reference.
  With `--validate`, the engine rows of these graphs fail. Some get
  `unchecked` instead when a CPU reference does
  not fit in host memory (see [SDPA dropout](usage.md#sdpa-dropout)). Other
  masks also make the reference decline the graph.
- Per-document masking inside one sequence (`varlen_docmask`).
  Variable-length SDPA with `seq_len`, `cu_seq_len` and ragged tensors is
  supported; see
  [Input data for variable-length SDPA](#input-data-for-variable-length-sdpa).
- One buffer manager for both runtimes. The hipDNN path (`BufferManager`) and
  the PyTorch path (`PyTorchCudaBufferManager`) still allocate and fill
  buffers with separate code. Both use the same timed loop.
- Graph-replay timing (HIP or CUDA graph capture). Back-to-back timing is
  available with `--timing-block`.
- Validation of large fp16 SDPA backward gradients. On the six fp16
  `aotriton_test_perf` graphs (B4, H48, D128, S 1024 to 4096), dK reaches
  about 440. Against the earlier AOTriton reference, the ASM SDPA engine
  differed by up to 1.0 on 18 to 194 elements of up to 3e8, where
  cancellation leaves small values. AOTriton itself differs from an fp32
  reference by up to 0.13. A norm-based tolerance would decide these rows.
- `--validate` allocator difference. With a device reference, the timed
  hipDNN buffers come from the PyTorch caching allocator, not from
  `hipMalloc`. The same engine can thus give slightly different numbers with
  and without `--validate`. `run.config.validate` records the setting, so do
  not compare runs with different values.
- Repeats across processes (rocKE attempts). Each engine is timed in one loop
  in one process.
- Validation of the page table. `graphs/sample_sdpa_paged.json` has one KV
  page: K and V are `[1, 4, 64, 128]` and the page table is `[[0], [0]]`.
  Every sequence reads page 0, so `--validate` cannot catch a kernel that
  ignores the page table. hipDNN rejects a K or V batch size (the page count)
  that differs from the Q batch size (`SdpaFwdNode::pre_validate_node`:
  "batch size mismatch between Q and K"). The Q batch of the sample is 1, so
  the sample can have only one page. In a real paged KV cache the page count
  does not depend on the batch size (hipDNN issue ALMIOPEN-3110). Unit tests
  check multi-page page tables in the input generator.

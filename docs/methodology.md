# Measurement methodology

This page tells you what each number in a result means and how the tool
measures it. The code is `src/dnn_benchmarking/execution/timing.py`
(`measure`), `reporting/statistics.py` and `execution/suite_runner.py`.

## Summary

- The headline number is `kernel_med`: the median device time of one launch
  of the graph, in the table and in `kernel.median_ms`.
- One loop implementation (`timing.measure`) times hipDNN engines and the
  PyTorch backend. The two backends get the same warmup, stop rule, cache
  mode and statistics.
- Each row records how it was measured in `timing`: mode, event backend,
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

The `staged` mode needs the HIP backend (`hipdnn_frontend` with the
`HipStallGate` binding) and a device with stream wait-value support. When
this is not available, the tool uses `events` mode and records the reason in
`timing.fallback_reason`:

| Reason | Cause |
|---|---|
| `staged timing requires the hip backend` | PyTorch backend on CUDA, or on ROCm without `hipdnn_frontend`. |
| `hipdnn_frontend is missing staging bindings: ...` | An old `hipdnn_frontend`. |
| `device does not support hipStreamWaitValue32` | The device or driver has no stream wait-value support. |
| `host sync in enqueue: ...` | The PyTorch graph reads a device value on the host. A stalled stream would then never complete. |

In `events` mode each iteration records the start event, calls `enqueue()`,
records the stop event and waits. The span then includes any host launch gap
inside the graph. For a single short kernel the two modes agree. For a graph
with many short kernels, `events` reads higher.

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
2. For the PyTorch backend, it runs one more `enqueue()` with
   `torch.cuda.set_sync_debug_mode("error")`, then a device sync. If this
   call synchronizes with the host, the loop uses `events` mode.

Priming launches are not timed and never flushed.

## Warmup

`--warmup N` (default 10) is a launch count, not a time budget. The priming
launches are part of the count. The remaining `N - 1` launches (`N - 2` for
the PyTorch backend) go through the timed-iteration path: the same mode
(`staged` or `events`), the same per-iteration sync and, in `cold` mode, the
same flush. The tool discards their samples.

Thus the GPU clocks and the caches reach the state of the timed loop before
the first sample. Back-to-back warmup launches do not do this: the timed loop
syncs after each launch, so the clocks can change at the start of the loop.
On MI210 (`sample_conv_fwd`, 100 iterations), back-to-back warmup gave a
kernel CV of 9.7 % and a maximum of 50.6 us (2 x median). Warmup through the
timed path gave a CV of 0.95 % and a maximum of 26.2 us. The median did not
change (25.4 to 25.6 us).

`timing.warmup_iters` is the number of untimed launches that actually ran.
Outside block mode it is `max(priming launches, --warmup)`: priming is 1
launch for hipDNN, 2 for the PyTorch backend (the host-sync probe), and 3
when the probe finds a host sync and runs the launch again. So it is 1 or
more, also when `--warmup 0` is given. In block mode it is the priming
launches plus `--warmup` for every sample, the discarded first sample
included.

## Cache modes

| Mode | Behavior |
|---|---|
| `warm` (default) | No flush. Inputs, outputs and workspace stay in L2 and MALL between iterations if they fit. This is the steady state of a layer that runs many times. |
| `cold` | Before each warmup and timed iteration: write zeros to a 512 MiB device buffer, then do a full device sync. Then the iteration starts. |

Notes for `cold`:

- The flush and the sync are outside the timed span. Priming launches are
  never flushed.
- 512 MiB is two times or more the largest last-level cache that the tool
  targets (MI300X MALL, 256 MiB). MI210 L2 is 8 MiB.
- The tool allocates the buffer one time per process, at the first cold
  iteration. The hipDNN path uses `hipdnn_frontend.DeviceBuffer`. The CUDA
  path uses a PyTorch tensor. If the allocation fails, the row fails with an
  error that tells you to use `--cache-mode warm`.
- The zero writes leave dirty lines in the cache. The first kernel in the
  span can pay the write-back of these lines. `triton.testing.do_bench` has
  the same effect.
- The flush does not change the instruction cache or TLB state.
- Each cold iteration costs more wall time than a warm iteration. The kernel
  time budget (`--min-time-ms`) does not count the flush.

`dnn-benchmark compare` refuses to compare files with different
`cache_mode` or `timing_block` values unless you give `--allow-mismatch`.

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

For `kernel` and `host` the tool reports `n`, `mean_ms`, `std_ms` (ddof 1),
`cv` (`std/mean`), `min_ms`, `p25_ms`, `median_ms`, `p75_ms`, `p95_ms`,
`max_ms` and `iqr_ms`. `median_ms` is the upper median `sorted(t)[n // 2]`,
the rocKE / Solera definition, so it is always an observed sample. The other
percentiles use linear interpolation. `p95_ms` is
`null` below 20 samples, because it is then close to the maximum.

The tool does not remove outliers. It adds warnings to the row:

| Warning | Condition |
|---|---|
| `noisy: IQR x% of median` | `iqr_ms / median_ms` more than 5 % and 10 or more samples. The table marks `kernel_med` with `*`. |
| `outlier: max Nx median` | Maximum more than 2 x median. The table marks `kernel_med` with `*` and shows the warning in `note`. |

The noise flag uses IQR/median, not CV. A few slow samples raise the CV a
lot but do not change the IQR. The `outlier` flag reports those samples.

The median is the headline because one slow iteration moves the mean but not
the median. In the MI210 runs below, one 37.6 us sample in a 29 us matmul run
moved the mean and the standard deviation, but not the median.

## TFLOP/s and GB/s

`tflops = flops / median` and `gbps = io_bytes / median`:

- `flops` is an analytical count from the graph JSON (FMA = 2 FLOPs). The
  dispatch table is in `metrics/analytical/__init__.py`; the per-op formulas
  are in its sibling modules (`conv.py`, `matmul.py`, `elementwise.py`,
  `normalization.py`, `reduction.py`, `sdpa.py`). When the graph has a node
  type with no formula, `flops_partial` is `true` and the table shows `~`.
- `io_bytes` is the sum of the sizes of all non-virtual tensors. It is a
  lower bound of the real memory traffic.
- In `warm` mode, a small problem can stay in L2. GB/s can then be more than
  the HBM bandwidth. Use `--cache-mode cold` to measure from memory.

## Clocks before and after

With `--metrics-tier basic` (default) and amdsmi available, the runner reads
the GPU clocks before priming and warmup, and right after the timed loop:
`sclk_mhz`, `mclk_mhz`, `power_w`, `temp_hotspot_c` and `throttle_status`.
The row gets the warning `throttled` when `throttle_status` after the loop is
not 0. The pair does not measure clock drift during the timed loop:
`clocks_before` is often the idle clock, because warmup has not run yet. The
tool does not set or lock clocks.

## Profiling child process

`--pmc`, `--emit-trace`, `--perf` and `--roofline` do not change the timed
numbers. After the timed row completes, the tool starts one child process for
each pass under the profiler:

```text
python -m dnn_benchmarking --internal-profiling-run --graph G --engine E \
    --warmup W --iters 5 --seed S [--plugin-path P]
```

The child builds one engine, fills the inputs with the same seed as the timed
run, and runs `W + 5` plain graph executes: the same submissions as the timed
loop, with no workspace reset and no sync between them. Then it drains the
device once. It does not use the timed loop, for two reasons:

- `rocprofv3 --pmc` serializes dispatches. A stalled stream then never
  starts, and the stall gate would deadlock.
- A profiler scopes its counters to a process. A new process gives a clean
  process tree and address space.

Before each child starts, the parent frees the engine buffers. The child thus
has the same free VRAM as the timed run. Counter values come from this child,
not from the timed loop. The child also launches input-fill kernels, and a
graph can launch more than one kernel. Use `pmc.per_kernel[...].dispatches`
and the kernel name to find the engine kernel.

## Parity with other harnesses

This table compares the tool with the rocKE runtime timer
(`rocke/runtime/launcher.py:time_launches`, `rocke/benchmark/summary.py`) and
with PyTorch (`torch.utils.benchmark.Timer.blocked_autorange`,
`triton.testing.do_bench`).

| Feature | dnn-benchmarking | rocKE | PyTorch / Triton |
|---|---|---|---|
| Warmup | Fixed count (default 10) through the timed path, first call timed separately; per sample in block mode | Fixed count (default 5) per sample | do_bench: 25 ms time budget. Timer: block-size estimate. |
| Iterations | `--iters` floor plus optional `--min-time-ms` budget, cap `max(10000, --iters)` | Fixed (default 100) | do_bench: 100 ms budget. `blocked_autorange`: 0.2 s minimum. |
| Event granularity | One event pair per launch, stall-gated; `--timing-block N`: one pair around N launches | One event pair around N launches | do_bench: one pair per launch. Graph variant: one pair per replay. |
| Sync | After each launch; block mode: after each block | One at the end | do_bench: one at the end. Timer: one per block. |
| Host launch gaps removed | Yes (stall gate) | Amortized over N launches | Amortized (do_bench) |
| Host submit time | Yes (`host`) | No | Timer: wall time only |
| Cache flush | `--cache-mode cold`, 512 MiB | Documented only | do_bench: 256 MB zero before each launch |
| Rotating buffers | No | Documented only | No |
| Graph replay | No | Yes | `do_bench_cudagraph` |
| Outliers | Flagged (IQR/median noise flag, max/median outlier flag), never removed; block mode discards the first sample | Discard first run | Timer: IQR warnings |
| Statistics | n, mean, std, CV, min, p25, median, p75, p95, max, IQR | median, min, max, mean, stdev, spread | do_bench: mean or quantiles. Timer: median, IQR. |
| Headline | Median | Median over attempts | do_bench: mean (default). Timer: median. |
| TFLOP/s basis | Median | Median | User calculates |
| Cross-process repeats | No | Yes (attempts, new process) | No |
| Clock capture | Before warmup and after the loop | Advice: lock clocks | None |

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
| This tool, `--backend pytorch`, staged | 26 | 29 |
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

hipDNN and `--backend pytorch` on the same graphs, kernel median in ms:

| Graph | hipDNN | PyTorch |
|---|---|---|
| add | 0.078 | 0.071 |
| batchnorm inference (32 x 64 x 28 x 28) | 0.024 | 0.077 |
| conv_dgrad | 0.026 | 0.049 |
| conv_fwd | 0.026 | 0.026 |
| matmul | 0.029 | 0.029 |
| relu | 0.155 | 0.156 |
| rmsnorm | 0.010 | 0.032 |

The tables above come from the review before this overhaul. The timed loop
was already stall-gated then, but the warmup ran back to back, so the
minimum-to-maximum ranges can include clock-ramp outliers.

### Change against the previous loop (MI210, gfx90a)

Every `graphs/*.json` ran on both backends with `--warmup 10 --iters 100`,
once with the previous loop (`main` at 2af0454) and once with this loop,
back to back, in two repeats (92 median pairs):

- 85 of 92 medians agree within 2 %. On kernels near 14 us, 1 % is one
  0.16 us step of the event timer.
- The other pairs are bimodal kernels (CV 0.1 to 0.3), where the median moves
  between the two modes. `conv_dgrad` on the PyTorch backend read 26 % and
  36 % lower; its mean moved less (43.1 to 43.0 us, 44.3 to 39.7 us).
  `conv_dgrad` on `MIOPEN_ENGINE_DETERMINISTIC` read 5.4 % lower in both
  repeats. Two other rows moved in one repeat only.
- CV fell where the previous warmup left clock-ramp samples in the loop, for
  example `conv_wgrad` 0.16 to 0.01 and `rmsnorm` on PyTorch 0.14 to 0.02.

A time series that crosses this change can show a step on bimodal rows.
Measure again after a timing change.

## Not yet

The tool does not do these things yet:

- Independent fp32 reference. `--validate pytorch` computes the reference in
  the dtype of the graph. For fp16 and bf16 graphs the reference then has the
  same precision as the result under test, and the tolerances must be loose.
  An fp32 reference needs new tolerances for all workloads.
- SDPA reference for every mask. The PyTorch reference follows hipDNN mask
  semantics: the deprecated `causal_mask` wins over left and right bounds
  (top-left causal). A bounded left window is supported only with
  `right_bound = 0`. Other masks make the reference decline the graph as
  unsupported.
- One buffer manager for both backends. The hipDNN path (`BufferManager`) and
  the PyTorch path (`PyTorchCudaBufferManager`) still allocate and fill
  buffers with separate code. Both use the same timed loop.
- Graph-replay timing (HIP or CUDA graph capture). Back-to-back timing is
  available with `--timing-block`.
- Rotating input buffers. `--cache-mode cold` controls the cache state, but
  the rocKE runbook also advises rotating buffers for bandwidth work.
- Signed input distribution. Floating-point inputs are uniform in [0, 1),
  because BatchNorm variance inputs must not be negative. Signed data needs a
  role for each tensor first.
- `--validate` allocator difference. With a device reference, the timed
  hipDNN buffers come from the PyTorch caching allocator, not from
  `hipMalloc`. The same engine can thus give slightly different numbers with
  and without `--validate`. `run.config.validate` records the setting, so do
  not compare runs with different values.
- Repeats across processes (rocKE attempts). Each engine is timed in one loop
  in one process.

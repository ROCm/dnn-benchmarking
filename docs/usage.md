# Benchmark Modes

Reference for the modes the [README](../README.md#usage) introduces. Run
`dnn-benchmark --help` for the full option list and defaults.

## PyTorch backend (`--backend pytorch`)

The PyTorch executor runs each graph instead of the hipDNN engine plugins and
produces one `provider="pytorch"` row for each graph. It uses the same suite
path as the hipDNN backend, so single-graph, glob, and tarball inputs and
`--output` JSON all work the same way.

```bash
dnn-benchmark --graph 'graphs/*.json' --backend pytorch -o pytorch_results.json

# Select the strict Flash category and prefer AOTriton within it on ROCm
dnn-benchmark --graph ./graphs/sample_sdpa.json --backend pytorch \
  --pytorch-sdpa-backend flash --pytorch-rocm-fa-library aotriton
```

`--pytorch-sdpa-backend default` keeps the normal PyTorch SDPA dispatch.
`flash`, `math`, `efficient`, `cudnn`, and `overrideable` are strict: the graph
must run native forward SDPA through that category, or the benchmark reports an
error. A non-default selection never falls back to normal dispatch or to a CPU
reference.

`--pytorch-rocm-fa-library LIBRARY` is ROCm-only and requires
`--pytorch-sdpa-backend flash`. It passes `LIBRARY` unchanged to PyTorch's
`preferred_rocm_fa_library`. PyTorch rejects an unknown library, and it can use
a different Flash implementation when the preferred one cannot serve an input.

`--backend pytorch` rejects these hipDNN-only options:

- `--engine` and `--plugin-path`: the backend loads no engine plugins.
- `--validate pytorch`: the backend would validate against itself.
- `--pmc`, `--emit-trace`, `--perf`, and `--roofline`: these are rocprofv3
  passes.
- `--oracle-mode`: auto-tuning is a hipDNN engine feature.

## Cross-machine comparison (ROCm and CUDA)

Both backends write the same `SuiteResult` JSON, so you can run the same graphs
on an AMD host and an NVIDIA host and compare the files:

```bash
# AMD host, hipDNN engines
dnn-benchmark --graph 'graphs/*.json' -o rocm_results.json
# AMD host, PyTorch executor
dnn-benchmark --graph 'graphs/*.json' --backend pytorch -o rocm_pytorch_results.json
# NVIDIA host, PyTorch executor
dnn-benchmark --graph 'graphs/*.json' --backend pytorch -o cuda_pytorch_results.json
```

Graphs match across files by `graph_name`. On CUDA, `rocm_version` and the
amdsmi snapshot are `None`, and `gpu_arch` is `"unknown"`. The repository has no
comparison tool yet.

## Kernel selection (`--autotune`, `--cache-dir`)

An engine has two kernel-selection paths, and each answers a different
question. By default, the engine serves the rank-0 pick of its cold heuristic,
so the table measures the *heuristic*. `--autotune` sets
`HIPDNN_FORCE_BENCHMARKING=1`: the first execute of each plan samples every
knob-filtered candidate and caches the winner. That measures what the shipped
*kernel set* can deliver. Use it for any best-against-best comparison. Without
it, an engine that adds good variants can measure slower when the heuristic
tie-break changes.

The tool always prints the selected path, so each table states which question
it answers.

`--cache-dir` sets `HIPDNN_CACHE_DIR` for the run. The winner cache is on disk
and is keyed by graph content and device, not by checkout, engine, or session.
Reads are not gated on benchmarking, but writes are. Two runs over the same
graphs therefore share rankings, and an untuned run can report a ranking that a
different run tuned. Give each phase its own empty directory:

```bash
dnn-benchmark --graph 'graphs/*.json' --cache-dir /tmp/cache-cold
dnn-benchmark --graph 'graphs/*.json' --autotune --cache-dir /tmp/cache-tuned
```

## Oracle comparison (`--oracle-mode`)

`--oracle-mode` compares the out-of-the-box (OOTB) plan with hipDNN's tuned
plan for each engine:

| Mode | Behavior |
|---|---|
| `off` | Run only the OOTB plan. This is the default. |
| `plan` | Benchmark every backend-generated plan for the engine. |
| `exhaustive` | Run `plan` mode and enable provider-managed kernel selection where supported. |

Exhaustive mode keeps every plan that the backend returns. It does not make a
Cartesian product of public knob values. The kernel ingestor and MIOpen support
provider-level selection. Other engines, including hipBLASLt, stay at
plan-level tuning. Providers can reuse cached selections, so exhaustive mode
does not prove that the current run measured every variant.

Both modes are slower than a normal run. Use `--engine` to limit the work.
`exhaustive` requires `--warmup >= 1`.

The summary table shows:

- `ootb_kernel_mean_ms`: the first OOTB measurement.
- `warm_ootb_kernel_mean_ms`: the same OOTB plan, measured again after tuning.
- `oracle_kernel_mean_ms`: the selected plan, measured after tuning.
- `oracle_speedup`: `warm_ootb_kernel_mean_ms / oracle_kernel_mean_ms`.

The warm OOTB measurement is the baseline because it has comparable device
warmup. The oracle can be slower: `0.99x` is a valid result. A single-plan row
with no search prints `no-search` and is not part of the suite geometric mean.

The JSON records the candidate counts and the provider-level state:

| Field | Meaning |
|---|---|
| `compiled_plans_benchmarked` | Compiled plans measured successfully. |
| `compiled_plans_total` | Eligible compiled plans, including failures. |
| `exhaustive_requested` | The run requested provider-level tuning. |
| `exhaustive_supported` | The engine advertises `global.benchmarking`. |
| `exhaustive_enabled` | Provider-level tuning was requested and supported. |
| `tuning_available` | Multiple plans competed or provider-level tuning was enabled. |

These counts do not include provider-internal kernel variants, because hipDNN
exposes no count for them. An empty `knob_settings` list means that no public
plan knob was set explicitly.

With `--validate`, the tool validates the OOTB plan and the tuned plan
separately. It reports no speedup if either plan fails.

Cache state can change the selection. Set `HIPDNN_DISABLE_EXACT_ENGINE_CACHE=1`
for a cold heuristic baseline. Exhaustive mode disables hipDNN's provider caches
during its oracle pass, but MIOpen FindDb and performance-database entries can
still supply tuned selections. The output records the cache and MIOpen database
paths in `metadata.hipdnn_selection_env`.

```bash
HIPDNN_DISABLE_EXACT_ENGINE_CACHE=1 dnn-benchmark \
  --graph ./graphs/sample_conv_fwd.json \
  --oracle-mode exhaustive -v -o oracle.json
```

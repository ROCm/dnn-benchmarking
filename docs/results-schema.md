# Result file schema (version 2)

`dnn-benchmark -o PATH` writes one result file. The file is JSON, or CSV when
`PATH` ends in `.csv`. This page describes both formats.

The source of truth is `src/dnn_benchmarking/reporting/suite_results.py`. The
key tuples in that module (`RUN_CONFIG_KEYS`, `PROFILING_KEYS`,
`ENVIRONMENT_KEYS`, `END_OF_RUN_KEYS`, `ROW_COLUMNS`) list the keys below.

## General rules

- Every key in this page is always present. A key that does not apply is
  `null`. Do not test for key presence.
- The file is strict JSON. The writer maps `NaN` and infinity to `null`.
- Times are in milliseconds (`_ms`) unless the key name gives another unit.
- `MB` and `GB` in environment key names are binary units: MiB (2^20 bytes)
  and GiB (2^30 bytes). `gbps` in `metrics` is decimal: 10^9 bytes per second.
- The writer replaces the file atomically. A reader never sees a
  half-written file.
- `SuiteResult.load(path)` reads a file. It rejects any `schema_version`
  other than 2. It prints a warning on stderr when `run.complete` is `false`.

## Top level

| Key | Type | Meaning |
|---|---|---|
| `schema_version` | int | Always `2`. |
| `tool` | object | `name` (`"dnn-benchmarking"`) and `version` (package version). |
| `run` | object | How and when the suite ran. See [run](#run). |
| `environment` | object | Machine and software snapshot. See [environment](#environment). |
| `summary` | object | Counts over all graphs. See [summary](#summary). |
| `graphs` | array | One object per graph. See [graph](#graph). |

## run

| Key | Type | Meaning |
|---|---|---|
| `started_at` | string | UTC ISO 8601 time of suite start. |
| `finished_at` | string or null | UTC ISO 8601 time of suite end. `null` in a partial file. |
| `complete` | bool | `true` only when every graph ran. See [Partial files](#partial-files). |
| `argv` | array of string | The command line. |
| `config` | object | The effective configuration. See [run.config](#runconfig). |

### run.config

| Key | Type | Meaning |
|---|---|---|
| `backend` | string | `hipdnn` or `pytorch`. |
| `engine_filter` | array of string or null | `--engine` selections as hex engine IDs (same format as `engine.id`). `null` means all discovered engines. |
| `plugin_paths` | array of string or null | Plugin directories. `null` for `--backend pytorch`. |
| `warmup_iters` | int | `--warmup`. |
| `iters` | int | `--iters` (minimum timed iterations). |
| `min_time_ms` | float | `--min-time-ms`. `0` means exactly `iters` samples. |
| `cache_mode` | string | `warm` or `cold`. |
| `seed` | int | Input data seed. |
| `validate` | string or null | Reference provider (`pytorch`), or `null` when validation is off. |
| `rtol` | float or null | `--rtol`. `null` means dtype-aware defaults. |
| `atol` | float or null | `--atol`. `null` means dtype-aware defaults. |
| `oracle_mode` | string | `off`, `plan` or `exhaustive`. |
| `autotune` | bool | `--autotune`. |
| `cache_dir` | string or null | `--cache-dir`. |
| `pytorch_sdpa_backend` | string or null | `--pytorch-sdpa-backend`. `null` unless `--backend pytorch` or `--validate pytorch`. |
| `pytorch_rocm_fa_library` | string or null | `--pytorch-rocm-fa-library`. `null` unless PyTorch is selected. |
| `metrics_tier` | string | `basic` or `off`. |
| `profiling` | object | Requested profiling passes: `pmc` (set name or null), `emit_trace` (`pftrace` or null), `perf` (bool), `roofline` (bool). |

## environment

The CLI collects these values one time, at suite start. Only `end_of_run` is
collected at suite end.

| Key | Type | Meaning |
|---|---|---|
| `hostname` | string | Host name. |
| `cpu_model` | string | First `model name` in `/proc/cpuinfo`. |
| `cpu_count` | int | Logical CPUs. |
| `numa_nodes` | int | NUMA nodes in `/sys/devices/system/node`. |
| `total_ram_gb` | float | Host RAM, GiB. |
| `kernel_version` | string | Linux kernel release. |
| `gpu_model` | string | GPU name from PyTorch device properties. |
| `gpu_arch` | string | gfx target (for example `gfx90a`). `"unknown"` when no AMD GPU is found, for example on a CUDA host. |
| `gpu_compute_units` | int | Compute units. |
| `gpu_hbm_gb` | float | Device memory, GiB. |
| `gpu_pcie_link` | string | PCIe link, for example `16 GT/s x16`. |
| `amdgpu_driver_version` | string | amdgpu driver version. |
| `gpu_power_cap_w` | float | Power cap, W. |
| `gpu_max_sclk_mhz` | float | Maximum shader clock, MHz. |
| `gpu_compute_partition` | string | Compute partition mode. |
| `rocm_version` | string | `torch.version.hip`. `null` without a ROCm PyTorch. |
| `cuda_version` | string | `torch.version.cuda` on a CUDA PyTorch. |
| `cudnn_version` | string | cuDNN version on a CUDA PyTorch. |
| `hipdnn_version` | string | `hipdnn_frontend.__version__`. |
| `python_version` | string | Python version. |
| `torch_version` | string | PyTorch version. |
| `amdsmi_available` | bool | `true` when amdsmi loads. Without amdsmi, the GPU fields above and all clock fields are `null`. |
| `selection_env` | object or null | Kernel-selection environment variables at start: `HIPDNN_DISABLE_EXACT_ENGINE_CACHE`, `HIPDNN_CACHE_DIR`, `HIPDNN_DISABLE_CACHE`, `HIPDNN_FORCE_BENCHMARKING`, `MIOPEN_USER_DB_PATH`, `MIOPEN_CUSTOM_CACHE_DIR`. `null` unless `--oracle-mode` or `--autotune` is set. |
| `end_of_run` | object | Snapshot at suite end: `host_rss_mb` (process RSS, MiB), `host_ram_available_mb` (MiB), `vram_used_mb` (MiB), `vram_total_mb` (MiB). |

## summary

The writer recalculates the summary from `graphs`. The row counts include
`role == "engine"` rows only. The row counts add up to `rows`.

| Key | Meaning |
|---|---|
| `graphs` | Number of graphs. |
| `rows` | Number of engine rows. |
| `passed` | Rows with verdict `passed`. |
| `unchecked` | Rows with verdict `unchecked`. |
| `failed` | Rows with verdict `failed`. |
| `skipped` | Rows with verdict `skipped`. |
| `errors` | Rows with verdict `error`. |
| `graph_errors` | Graphs with status `error`. |
| `no_engine_graphs` | Graphs with status `no_engines`. |

## graph

| Key | Type | Meaning |
|---|---|---|
| `graph_id` | string or null | Join key. See [graph_id](#graph_id). `null` when the file is not valid JSON. |
| `graph_name` | string | `name` field of the graph JSON. The file stem when the graph has no `name` or does not load. |
| `graph_path` | string | Path of the graph file. |
| `status` | string | `ok`, `no_engines` (no engine applies to the graph) or `error` (graph-level failure). |
| `error` | string or null | Graph-level failure, as `ExceptionType: message`. |
| `results` | array | One [row](#row) per engine or provider. Empty when `status` is `no_engines` or `error`. |

### graph_id

`graph_id` is the first 12 hex characters of the SHA-256 of the canonical
graph JSON:

```python
json.dumps(graph, sort_keys=True, separators=(",", ":"))
```

The graph content decides the ID. The file name and the file path do not. Use
`graph_id` to join the same graph across result files.

## row

| Key | Type | Meaning |
|---|---|---|
| `provider` | string | `hipdnn` or `pytorch`. |
| `role` | string | `engine` for a benchmarked engine. `reference` for the timed reference row that `--validate pytorch` adds. |
| `engine` | object | See [engine](#engine). |
| `status` | string | `success`, `error` or `skipped`. |
| `verdict` | string | See [Verdicts](#verdicts). |
| `message` | string or null | Error message or skip reason. |
| `started_at` | string | UTC ISO 8601 time when the row started. |
| `elapsed_s` | float | Wall time of the whole row, seconds: build, priming, timing, validation, oracle and profiling. |
| `build_ms` | float or null | CPU time to build the plan. |
| `timing` | object or null | How the samples were measured. See [timing](#timing). |
| `kernel` | object or null | Device time per launch. See [stats](#stats). |
| `host` | object or null | Host submit time per launch (enqueue call only). See [stats](#stats). |
| `metrics` | object | See [metrics](#metrics). |
| `correctness` | object or null | See [correctness](#correctness). `null` when validation did not run. |
| `warnings` | array of string | Non-fatal notes. See [Row warnings](#row-warnings). |
| `oracle` | object or null | See [oracle](#oracle). `null` unless `--oracle-mode` ran for the row. |
| `extra_metrics` | object or null | Profiling results. See [extra_metrics](#extra_metrics). |

### engine

| Key | Type | Meaning |
|---|---|---|
| `id` | string or null | Engine ID as `0x` plus 16 upper-case hex digits. `null` for PyTorch rows. |
| `name` | string or null | Engine name, for example `MIOPEN_ENGINE`. |
| `version` | string | Plugin version, or `"unavailable"`. |
| `plugin_path` | string or null | Plugin the engine was loaded from. |

hipDNN engine IDs are signed 64-bit integers. JSON readers that use
floating-point numbers lose precision above 2^53. For this reason the file
stores the ID as the hex form of its unsigned 64-bit value
(`"0x%016X" % (id & 0xFFFFFFFFFFFFFFFF)`). `--engine` accepts this string.

### timing

| Key | Type | Meaning |
|---|---|---|
| `mode` | string | `staged` (stall-gated device span) or `events` (event pair around each launch). |
| `backend` | string | Event backend: `hip` or `torch`. |
| `cache_mode` | string | `warm` or `cold`. |
| `warmup_iters` | int | Untimed launches that ran before the timed loop. Always 1 or more. |
| `first_call_ms` | float | Wall time of the first launch plus a device sync. It includes one-time costs such as kernel compile and MIOpen find. |
| `capped` | bool | `true` when the `max_iters` cap (10000) stopped the loop before `--min-time-ms` was reached. |
| `fallback_reason` | string or null | Why `staged` mode was not used. |

See [methodology.md](methodology.md) for the meaning of each mode.

### stats

`kernel` and `host` use the same object. All values are milliseconds, except
`n` and `cv`.

| Key | Meaning |
|---|---|
| `n` | Number of timed samples. |
| `mean_ms` | Arithmetic mean. |
| `std_ms` | Sample standard deviation (ddof = 1). `0` when `n` is 1. |
| `cv` | Coefficient of variation, `std_ms / mean_ms` (fraction, not percent). |
| `min_ms` | Minimum. |
| `p25_ms` | 25th percentile. |
| `median_ms` | Median. This is the headline value. |
| `p75_ms` | 75th percentile. |
| `p95_ms` | 95th percentile. `null` when `n` is less than 20. |
| `max_ms` | Maximum. |
| `iqr_ms` | `p75_ms - p25_ms`. |

Percentiles use linear interpolation (`numpy.percentile`). The tool does not
remove outliers.

### metrics

| Key | Type | Meaning |
|---|---|---|
| `flops` | int or null | Analytical FLOPs for one launch of the graph. `null` when no node type is recognised. |
| `flops_partial` | bool | `true` when the graph has a node type with no FLOP formula. `flops` then counts only the recognised nodes. The console shows `~` before TFLOP/s. |
| `io_bytes` | int or null | Sum of the sizes of all non-virtual tensors, bytes. |
| `tflops` | float or null | `flops / kernel.median_ms`, in 10^12 FLOP/s. |
| `gbps` | float or null | `io_bytes / kernel.median_ms`, in 10^9 bytes/s. |
| `workspace_bytes` | int or null | Workspace that hipDNN reserved for the plan. |
| `vram_mb` | float or null | Process VRAM in use after the timed loop, MiB. It can include cached allocations from earlier engines on the same graph. |
| `clocks_before` | object or null | GPU clocks right before the timed loop. See [clocks](#clocks). |
| `clocks_after` | object or null | GPU clocks right after the timed loop. |

`--metrics-tier off` sets the analytical values, `vram_mb` and both clock
objects to `null`.

#### clocks

amdsmi supplies these values. The object is `null` without amdsmi. A value is
`null` when the platform does not report it.

| Key | Unit |
|---|---|
| `sclk_mhz` | Shader clock, MHz. |
| `mclk_mhz` | Memory clock, MHz. |
| `power_w` | Socket power, W. |
| `temp_hotspot_c` | Hotspot temperature, degrees C. |
| `throttle_status` | Throttle status bit field. `0` means no throttle. |

### correctness

| Key | Type | Meaning |
|---|---|---|
| `match` | bool or null | `true` when every output is within tolerance. `false` on a mismatch. `null` when no comparison ran. |
| `rtol` | float | Relative tolerance used. |
| `atol` | float | Absolute tolerance used. |
| `max_abs_diff` | float or null | Largest absolute difference over all outputs. |
| `max_rel_diff` | float or null | Largest relative difference over all outputs. |
| `n_mismatch` | int or null | Elements outside tolerance, summed over outputs. |
| `n_total` | int or null | Elements compared, summed over outputs. |
| `worst_output_uid` | int or null | Tensor UID of the output with the largest difference. |
| `message` | string or null | Why `match` is `false` or `null`. |

### Row warnings

`warnings` holds short notes. The console table shows the first note in the
`note` column. The runner adds these notes after the timed loop:

| Note | Condition |
|---|---|
| `noisy: CV x%` | `kernel.cv` is more than 0.05 and `kernel.n` is 10 or more. |
| `outlier: max Nx median` | `kernel.max_ms` is more than 2 x `kernel.median_ms`. |
| `capped at max_iters` | `timing.capped` is `true`. |
| `events timing: <reason>` | `timing.fallback_reason` is set. stderr also shows the reason one time per process. |
| `throttled` | `clocks_after.throttle_status` is not 0, or `clocks_after.sclk_mhz` is less than 0.9 x `clocks_before.sclk_mhz`. |

The tool never removes samples because of a warning.

### oracle

`oracle` is `null` unless `--oracle-mode plan` or `--oracle-mode exhaustive`
ran for the row.

| Key | Type | Meaning |
|---|---|---|
| `status` | string | `ok`, or `error` when tuning produced no result. |
| `error` | string or null | Why tuning failed. When `status` is `error`, all other keys except `delta` are `null`. |
| `plan_name` | string | Name of the selected plan. |
| `compiled_plan_index` | int | Index of the selected compiled plan. |
| `rank` | int | Rank of the selected plan in the sweep. |
| `sweep_min_time_ms` | float | Fastest single iteration in the selection sweep. |
| `compiled_plans_benchmarked` | int | Compiled plans that the sweep measured. |
| `compiled_plans_total` | int | Eligible compiled plans, failures included. |
| `compiled_plans_failed` | int | Compiled plans that failed. |
| `tuning_available` | bool | More than one plan competed, or provider-level tuning was on. |
| `knob_settings` | array | Explicit plan knob settings. Empty means no knob was set. |
| `exhaustive_requested` | bool | The run requested provider-level tuning. |
| `exhaustive_enabled` | bool | Provider-level tuning was requested and supported. |
| `exhaustive_supported` | bool | The engine advertises `global.benchmarking`. |
| `cpu_build_time_ms` | float or null | CPU time to build the tuned plan. |
| `kernel` | stats or null | Tuned plan, device time. |
| `host` | stats or null | Tuned plan, host submit time. |
| `baseline_kernel` | stats or null | Default (OOTB) plan timed again after the sweep, device time. |
| `baseline_host` | stats or null | Default plan timed again after the sweep, host submit time. |
| `correctness` | object or null | Correctness of the tuned plan. The row `correctness` is for the default plan. |
| `delta` | object or null | Comparison by median. See below. |

`delta` compares `baseline_kernel` with `kernel` (basis `kernel`). When either
side has no kernel statistics, it compares the host statistics (basis
`host`).

| Key | Meaning |
|---|---|
| `basis` | `kernel` or `host`. |
| `baseline_median_ms` | Median of the default plan, timed after the sweep. |
| `oracle_median_ms` | Median of the tuned plan. |
| `delta_ms` | `baseline_median_ms - oracle_median_ms`. A positive value means the tuned plan is faster. |
| `speedup` | `baseline_median_ms / oracle_median_ms`. |

The baseline is the default plan timed again after the sweep, not the row
`kernel` value. Both sides then have the same device warmth.

### extra_metrics

`extra_metrics` is `null` unless a profiling flag is set. It has one key for
each requested pass: `pmc`, `trace`, `perf` and `roofline`. Each pass runs
in a separate child process. See [usage.md](usage.md#profiling) and
[troubleshooting.md](troubleshooting.md#viewing-profiling-artefacts).

A pass that cannot complete does not stop the run. Its object then has one or
more of these keys: `skipped` (reason), `returncode`, `error_tail` (last 40
lines of tool output), `warnings`, `unexpected_error`.

| Pass | Keys on success |
|---|---|
| `pmc` | `set`, `arch`, `counters_requested`, `db_path`, `per_kernel`. `arch_narrowed_to_fallback` is `true` when `--pmc all` used the fallback set. |
| `trace` | `format` (`pftrace`), `path`. |
| `perf` | `scope` (always `process_total`), `cycles_user`, `instructions_user`, `ipc_user`, `cycles_kernel`, `instructions_kernel`, `task_clock_ms`, `context_switches`, `page_faults`, `kernel_perf_paranoid`, `binary`, `csv_path`. Optional: `binary_substituted`, `kernel_events_skipped_reason`. |
| `roofline` | `roofline_csv`, `workload_path`, `sysinfo_csv`. |

`pmc.per_kernel` maps each kernel name to one object:

```json
{"dispatches": 15, "counters": {"SQ_WAVES": 1024.0}, "l2_hit_rate": 0.91}
```

`counters` holds the mean value per dispatch. `l2_hit_rate` is present only
when the set has `TCC_HIT` and `TCC_MISS` counters. The child process also
launches input-fill and warmup kernels. Use `dispatches` and the kernel name
to find the engine kernel.

`perf` counts cover the whole child process: interpreter start, imports,
plugin load, graph build, warmup and the loop. Do not compare them with
per-launch times.

## Verdicts

`verdict` is one label per row. The console, the summary and `compare` use
it.

| Verdict | Condition |
|---|---|
| `error` | `status` is `error`. |
| `skipped` | `status` is `skipped`. |
| `reference` | `role` is `reference` and the row ran. |
| `failed` | Validation ran and found a mismatch, or the run did not complete. |
| `passed` | Validation ran and every output is within tolerance. |
| `unchecked` | The row ran, but no validation result exists. This is not a pass. |

The checks apply in the order of the table. The CLI exit code follows from the
verdicts. See [usage.md](usage.md#exit-codes).

## Partial files

With `-o`, the CLI writes the file while the suite runs:

1. It checks that the output path is writable before the first graph. If it
   is not, the CLI stops with exit code 2.
2. After a graph completes, it writes the file again when 10 s or more have
   passed since the last write.
3. It writes the file one last time when the run ends, also on an error or an
   interrupt.

A file from steps 2 and 3 of an unfinished run has `run.complete = false` and
`run.finished_at = null`. It holds every graph that completed. After SIGINT
(Ctrl-C) the exit code is 130. After SIGTERM the exit code is 143.

A CSV file uses the same schedule, but it has no `complete` column. Use JSON
when you must detect a partial run.

## CSV format

A `.csv` path writes one line per row with these columns (`ROW_COLUMNS`):

| Column | Source |
|---|---|
| `gpu_arch` | `environment.gpu_arch` |
| `graph_name` | `graph.graph_name` |
| `graph_id` | `graph.graph_id` |
| `provider` | `row.provider` |
| `role` | `row.role` |
| `engine_id` | `row.engine.id` |
| `engine_name` | `row.engine.name` |
| `status` | `row.status` |
| `verdict` | `row.verdict` |
| `kernel_median_ms` | `row.kernel.median_ms` |
| `kernel_cv` | `row.kernel.cv` |
| `host_median_ms` | `row.host.median_ms` |
| `tflops` | `row.metrics.tflops` |
| `gbps` | `row.metrics.gbps` |
| `workspace_bytes` | `row.metrics.workspace_bytes` |
| `max_abs_diff` | `row.correctness.max_abs_diff` |
| `message` | `row.message` |

A graph with no rows (status `error` or `no_engines`) gives one line. That
line has the graph status in `status` and the graph error in `message`.
`dnn-benchmark compare` reads JSON only.

# Result file schema (version 2)

`dnn-benchmark -o PATH` writes one result file. The file is JSON, or CSV when
`PATH` ends in `.csv`. This page describes both formats.

The source of truth is `src/dnn_benchmarking/reporting/suite_results.py`. The
key tuples in that module (`RUN_CONFIG_KEYS`, `PROFILING_KEYS`,
`ENVIRONMENT_KEYS`, `END_OF_RUN_KEYS`, `ROW_COLUMNS`) list the keys below.

## General rules

- Every key in this page is always present. A key that does not apply is
  `null`. Do not test for key presence. The exception is the inside of
  `extra_metrics` objects, where the [extra_metrics](#extra_metrics) table
  marks the keys that can be absent.
- The file is strict JSON. The writer maps `NaN` and infinity to `null`.
- The writer rounds every float to 6 significant digits.
- The file is indented by one space. `--compact-json` writes it without
  whitespace.
- Times are in milliseconds (`_ms`) unless the key name gives another unit.
- `MB` and `GB` in environment key names are binary units: MiB (2^20 bytes)
  and GiB (2^30 bytes). `gbps` in `metrics` is decimal: 10^9 bytes per second.
- The writer replaces the file atomically. A reader never sees a
  half-written file. The file mode follows the umask, as for a file that
  `open()` creates.
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
| `runtime` | string | `hipdnn` or `pytorch`. |
| `engine_filter` | array of string or null | `--engine` selections as hex engine IDs (same format as `engine.id`). `null` means all discovered engines. |
| `plugin_paths` | array of string or null | Plugin directories. `null` for `--runtime pytorch`. |
| `warmup_iters` | int | `--warmup`. |
| `iters` | int | `--iters` (minimum timed iterations). |
| `min_time_ms` | float | `--min-time-ms`. `0` means exactly `iters` samples. |
| `cache_mode` | string | `warm` or `cold`. |
| `timing_block` | int | `--timing-block`. `1` = one launch per sample; `N > 1` = rocKE block timing. |
| `seed` | int | Input data seed. |
| `validate` | string or null | Reference runtime (`pytorch`), or `null` when validation is off. |
| `rtol` | float or null | `--rtol` as given. When both `rtol` and `atol` are `null`, validation uses dtype-aware defaults. When only one is given, it also sets the other, which stays `null` here; each row's `correctness.rtol` and `correctness.atol` hold the values a comparison applied (see [correctness](#correctness) for rows where none ran). |
| `atol` | float or null | `--atol` as given. `null` follows the same rule as `rtol`. |
| `oracle_mode` | string | `off`, `plan` or `exhaustive`. |
| `autotune` | bool | `--autotune`. |
| `hipdnn_cache_dir` | string or null | `--hipdnn-cache-dir`. |
| `pytorch_sdpa_backend` | string or null | `--pytorch-sdpa-backend`. `null` unless `--runtime pytorch` or `--validate pytorch`. |
| `pytorch_rocm_fa_library` | string or null | `--pytorch-rocm-fa-library`. `null` unless PyTorch is selected. |
| `metrics` | bool | `--metrics`: the always-on probes ran. |
| `profiling` | object | Requested profiling passes: `pmc` (set name or null), `trace` (bool), `perf` (bool), `roofline` (bool). |

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
| `gpu_model` | string | GPU name: from PyTorch when the process has already imported it (PyTorch runtime), else amdsmi `market_name`, else the `Marketing Name` from `rocminfo`. |
| `gpu_arch` | string | gfx target (for example `gfx90a`). `"unknown"` when no AMD GPU is found, for example on a CUDA host. |
| `gpu_compute_units` | int | Compute units, from amdsmi, or from PyTorch when the process has already imported it. |
| `gpu_hbm_gb` | float | Device memory, GiB. |
| `gpu_pcie_link` | string | PCIe link, for example `16 GT/s x16`. |
| `amdgpu_driver_version` | string | amdgpu driver version. |
| `gpu_power_cap_w` | float | Power cap, W. |
| `gpu_max_sclk_mhz` | float | Maximum shader clock, MHz. |
| `gpu_compute_partition` | string | Compute partition mode. |
| `rocm_version` | string | The `hip` value of the installed `torch/version.py`. The tool reads the file without an import of PyTorch. `null` without a ROCm PyTorch. |
| `cuda_version` | string | The `cuda` value of `torch/version.py` for a CUDA PyTorch. `null` for a ROCm PyTorch. |
| `cudnn_version` | string | cuDNN version. Set only for a CUDA PyTorch that the process has already imported. |
| `hipdnn_version` | string | `hipdnn_frontend.__version__`. |
| `python_version` | string | Python version. |
| `torch_version` | string | `__version__` of `torch/version.py`. |
| `amdsmi_available` | bool | `true` when amdsmi loads. Without amdsmi, `gpu_hbm_gb`, `gpu_pcie_link`, `amdgpu_driver_version`, `gpu_power_cap_w`, `gpu_max_sclk_mhz`, `gpu_compute_partition`, the VRAM values and all clock values are `null`. |
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
| `error` | string or null | Graph-level failure, as `ExceptionType: message`. A failure before any row runs has the prefix `Engine discovery failed: ` or `Input data generation failed: `. |
| `message` | string or null | Why no engine applies, when `status` is `no_engines`: hipDNN's reason, or the unsupported tensor data type. The console shows `no engines applicable: <message>`. |
| `results` | array | One [row](#row) per engine or reference. Empty when `status` is `error`. When `status` is `no_engines` it has no engine rows, but it holds the `reference` row when `--validate pytorch` ran and PyTorch supports the graph. |

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
| `runtime` | string | Runtime that produced the row: `hipdnn` or `pytorch`. |
| `role` | string | `engine` for a benchmarked engine. `reference` for the timed reference row that `--validate pytorch` adds. |
| `engine` | object | See [engine](#engine). |
| `status` | string | `success`, `error` or `skipped`. |
| `verdict` | string | See [Verdicts](#verdicts). |
| `message` | string or null | Error message or skip reason. |
| `started_at` | string | UTC ISO 8601 time when the row started. |
| `elapsed_s` | float | Wall time of the whole row, seconds: build, priming, timing, validation, oracle and profiling. |
| `metrics` | object | Graph values and GPU state. See [metrics](#metrics). |
| `ootb` | object or null | The default (out-of-the-box) plan. See [plan](#plan). `null` when `status` is `error` or `skipped`. |
| `oracle` | object or null | The tuned plan. See [oracle](#oracle). `null` unless `--oracle-mode` ran for the row and tuning produced a result. |
| `oracle_error` | string or null | Why tuning produced no result. `oracle` is then `null`. |
| `warnings` | array of string | Non-fatal notes. See [Row warnings](#row-warnings). |
| `extra_metrics` | object or null | Profiling results. See [extra_metrics](#extra_metrics). |

### engine

| Key | Type | Meaning |
|---|---|---|
| `id` | string or null | Engine ID as `0x` plus 16 upper-case hex digits. `null` for PyTorch rows. |
| `name` | string or null | Engine name, for example `MIOPEN_ENGINE`. `pytorch` on PyTorch rows. |
| `version` | string | Plugin version, or `"unavailable"`. On PyTorch rows, the torch version. |

hipDNN engine IDs are signed 64-bit integers. JSON readers that use
floating-point numbers lose precision above 2^53. For this reason the file
stores the ID as the hex form of its unsigned 64-bit value
(`"0x%016X" % (id & 0xFFFFFFFFFFFFFFFF)`). `--engine` accepts this string.

### plan

`ootb` and `oracle` use the same object, so the default plan and the tuned
plan compare key by key. `oracle` adds the keys in [oracle](#oracle).

| Key | Type | Meaning |
|---|---|---|
| `build_ms` | float or null | CPU time to build the plan. |
| `timing` | object | How the samples were measured. See [timing](#timing). |
| `kernel` | object or null | Device time per launch. See [stats](#stats). |
| `host` | object or null | Host submit time per launch (enqueue call only). See [stats](#stats). |
| `workspace_bytes` | int or null | Workspace that hipDNN reserved for the plan. `null` for PyTorch. |
| `tflops` | float or null | `metrics.flops / kernel.median_ms`, in 10^12 FLOP/s. |
| `gbps` | float or null | `metrics.io_bytes / kernel.median_ms`, in 10^9 bytes/s. |
| `correctness` | object or null | See [correctness](#correctness). `null` when validation did not run. |

### timing

| Key | Type | Meaning |
|---|---|---|
| `mode` | string | `staged` (stall-gated device span), `events` (event pair around each launch), or `block` (event pair around `timing_block` back-to-back launches; samples are `elapsed / timing_block`). |
| `timer` | string | Event timer: `hip` (HIP events) or `torch` (`torch.cuda.Event`). |
| `warmup_iters` | int | Untimed launches that actually ran: priming (and, for PyTorch, the host-sync probe and its rerun) plus the discarded warmups. Always 1 or more. In `block` mode, also the `--warmup` launches before every sample, the discarded first sample included. See [methodology.md](methodology.md#warmup). |
| `first_call_ms` | float | Wall time of the first launch plus a device sync. It includes one-time costs such as kernel compile and MIOpen find. |
| `capped` | bool | `true` when the `max_iters` cap (`max(10000, --iters)`) stopped the loop before `--min-time-ms` was reached. |
| `fallback_reason` | string or null | Why `staged` mode was not used. In `block` mode, set when the PyTorch `enqueue()` syncs with the host. |

The cache mode and the launches per sample apply to the whole run, so they
are only in `run.config` (`cache_mode`, `timing_block`). See
[methodology.md](methodology.md) for the meaning of each mode.

### stats

`kernel` and `host` use the same object. All values are milliseconds, except
`n`.

| Key | Meaning |
|---|---|
| `n` | Number of timed samples. |
| `mean_ms` | Arithmetic mean. |
| `std_ms` | Sample standard deviation (ddof = 1). `0` when `n` is 1. |
| `min_ms` | Minimum. |
| `p25_ms` | 25th percentile. |
| `median_ms` | Upper median, `sorted(samples)[n // 2]` (the rocKE / Solera definition; always an observed sample). This is the headline value. |
| `p75_ms` | 75th percentile. |
| `p95_ms` | 95th percentile. `null` when `n` is less than 20. |
| `max_ms` | Maximum. |

Percentiles use linear interpolation (`numpy.percentile`). The tool does not
remove outliers. The file does not store values that a reader can derive:
the coefficient of variation is `std_ms / mean_ms`, and the interquartile
range (IQR) is `p75_ms - p25_ms`.

### metrics

| Key | Type | Meaning |
|---|---|---|
| `flops` | int or null | Analytical FLOPs for one launch of the graph. `null` when no node type is recognised. |
| `flops_partial` | bool | `true` when the graph has a node type with no FLOP formula. `flops` then counts only the recognised nodes. The console shows `~` before TFLOP/s. |
| `io_bytes` | int or null | Sum of the sizes of all non-virtual tensors, bytes. |
| `vram_mb` | float or null | Device VRAM in use after the timed loop, MiB, while the row's buffers are still allocated. amdsmi `vram_used` counts every process on the GPU, not only this one. It can include cached allocations from earlier engines on the same graph. |
| `clocks_before` | object or null | GPU clocks before priming and warmup of the default plan, so often the idle clocks. See [clocks](#clocks). |
| `clocks_after` | object or null | GPU clocks right after the default plan's timed loop. |

`--no-metrics` sets the analytical values, `vram_mb`, both clock
objects, and the plan `workspace_bytes`, `tflops` and `gbps` to `null`, and
`flops_partial` to `false`.

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
| `match` | bool or null | `true` when every output is within tolerance. `false` on a mismatch, or when validation was requested but no usable reference exists (`message` gives the reason). `null` when no comparison ran. |
| `rtol` | float | Relative tolerance applied, the largest over outputs. For `fp8_e8m0` outputs it applies to log2 values. When no comparison ran (`match` is `false`, `message` is set and `n_total` is `null`), this is `--rtol` when given, else the fp32 default 1e-5, not the tolerance of the output dtype. |
| `atol` | float | Absolute tolerance applied, with the same rules as `rtol`. When no comparison ran, this is `--atol` when given, else 1e-6. |
| `max_abs_diff` | float or null | Largest absolute difference over all outputs. |
| `max_rel_diff` | float or null | Largest `\|actual - expected\| / \|expected\|` over all outputs, taken only over elements with `\|expected\| > atol`. `0.0` when no element qualifies, so a mismatch against a near-zero reference can show `0.0` with `n_mismatch > 0`. Use `match` and `n_mismatch` for pass or fail. |
| `n_mismatch` | int or null | Elements outside tolerance, summed over outputs. |
| `n_total` | int or null | Elements compared, summed over outputs. |
| `worst_output_uid` | int or null | Tensor UID of the output with the largest difference. |
| `message` | string or null | Why `match` is `false` or `null`. |

### Row warnings

`warnings` holds short notes. The console table shows the first note in the
`note` column. The runner adds these notes after the default plan's timed
loop. Paths are relative to the row:

| Note | Condition |
|---|---|
| `noisy: IQR x% of median` | `(ootb.kernel.p75_ms - ootb.kernel.p25_ms) / ootb.kernel.median_ms` is more than 0.05 and `ootb.kernel.n` is 10 or more. |
| `outlier: max Nx median` | `ootb.kernel.max_ms` is more than 2 x `ootb.kernel.median_ms`. |
| `capped at max_iters` | `ootb.timing.capped` is `true`. |
| `<mode> timing: <reason>` | `ootb.timing.fallback_reason` is set. `<mode>` is `events` or `block` (`ootb.timing.mode`). stderr also shows the same text one time per process. |
| `throttled` | `metrics.clocks_after.throttle_status` is not 0. |
| `profiling failed: <error>` | A profiling pass raised an exception. The timed values stay in the row. |

The tool never removes samples because of a warning.

### oracle

`oracle` is `null` unless `--oracle-mode plan` or `--oracle-mode exhaustive`
ran for the row and tuning produced a result. When tuning fails,
`oracle_error` gives the reason and `oracle` is `null`.

`oracle` is a [plan](#plan) object for the tuned plan, with these keys added:

| Key | Type | Meaning |
|---|---|---|
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
| `baseline_kernel` | stats or null | Default (OOTB) plan timed again after the sweep, device time. |
| `baseline_host` | stats or null | Default plan timed again after the sweep, host submit time. |
| `baseline_tflops` | float or null | Default plan TFLOP/s: row `metrics.flops` / `baseline_kernel.median_ms`. |
| `delta` | object or null | Comparison by kernel median. See below. |

`delta` compares `baseline_kernel` with `kernel`. It is `null` when either
side has no kernel statistics, when either median is 0 or less, or when the
default or the tuned plan failed validation.

| Key | Meaning |
|---|---|
| `basis` | Always `kernel`. |
| `baseline_median_ms` | Median of the default plan, timed after the sweep. |
| `oracle_median_ms` | Median of the tuned plan. |
| `delta_ms` | `baseline_median_ms - oracle_median_ms`. A positive value means the tuned plan is faster. |
| `speedup` | `baseline_median_ms / oracle_median_ms`. |

The baseline is the default plan timed again after the sweep, not
`ootb.kernel`. Both sides then have the same device warmth.

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
| `perf` | `scope` (always `process_total`), `cycles_user`, `instructions_user`, `ipc_user`, `cycles_kernel`, `instructions_kernel`, `task_clock_ms`, `context_switches`, `page_faults`, `kernel_perf_paranoid`, `binary`. Optional: `csv_path` (only when perf wrote its CSV), `binary_substituted`, `kernel_events_skipped_reason`. |
| `roofline` | `roofline_csv` and `workload_path` only when `roofline.csv` was produced; `sysinfo_csv` only when `sysinfo.csv` was produced. At least one of the two is present. |

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
| `failed` | Validation found a mismatch, or validation was requested but no usable reference exists. |
| `passed` | Validation ran and every output is within tolerance. |
| `unchecked` | The row ran, but no validation result exists. This is not a pass. |

The checks apply in the order of the table. The CLI exit code follows from the
verdicts of the `engine` rows; `reference` rows do not count. See
[usage.md](usage.md#exit-codes).

## Partial files

With `-o`, the CLI writes the file while the suite runs:

1. It checks that the output path is writable before the first graph. If it
   is not, the CLI stops with exit code 2.
2. After a graph completes, it writes the file again when 10 s or more have
   passed since the last write.
3. It writes the file one last time when the run ends, also on an error or an
   interrupt.

A failed write in step 2 prints an `ERROR` line on stderr, and the run goes
on. When the last write succeeds, the exit code still follows the verdicts.
A failed last write gives exit code 1, unless an engine row failed
validation, which gives 3 (see [usage.md](usage.md#exit-codes)).

A file from steps 2 and 3 of an unfinished run has `run.complete = false` and
`run.finished_at = null`. It holds every graph that completed. After SIGINT
(Ctrl-C) the exit code is 130. After SIGTERM the exit code is 143. A second
Ctrl-C or SIGTERM during the last write is ignored, so the file is not cut
short.

A CSV file uses the same schedule, but it has no `complete` column. Use JSON
when you must detect a partial run.

## CSV format

A `.csv` path writes one line per row with these columns (`ROW_COLUMNS`):

| Column | Source |
|---|---|
| `gpu_arch` | `environment.gpu_arch` |
| `graph_name` | `graph.graph_name` |
| `graph_id` | `graph.graph_id` |
| `runtime` | `row.runtime` |
| `role` | `row.role` |
| `engine_id` | `row.engine.id` |
| `engine_name` | `row.engine.name` |
| `status` | `row.status` |
| `verdict` | `row.verdict` |
| `kernel_median_ms` | `row.ootb.kernel.median_ms` |
| `kernel_cv` | `row.ootb.kernel.std_ms / row.ootb.kernel.mean_ms` |
| `host_median_ms` | `row.ootb.host.median_ms` |
| `n` | `row.ootb.kernel.n` |
| `timing_mode` | `row.ootb.timing.mode` |
| `cache_mode` | `run.config.cache_mode` |
| `timing_block` | `run.config.timing_block` |
| `seed` | `run.config.seed` |
| `tflops` | `row.ootb.tflops` |
| `gbps` | `row.ootb.gbps` |
| `workspace_bytes` | `row.ootb.workspace_bytes` |
| `max_abs_diff` | `row.ootb.correctness.max_abs_diff` |
| `message` | `row.message` |

A graph with no rows (status `error`, or `no_engines` without `--validate`)
gives one line. That line has the graph status in `status`, and the graph
`error` or `message` in `message`. A `no_engines` graph with `--validate
pytorch` gives only its `reference` row when PyTorch supports the graph; the
graph status is then not in the CSV. When PyTorch does not support it, the
graph has no rows and gives the one status line.

`dnn-benchmark compare` reads JSON only. It stops with exit code 2 on a CSV
file.

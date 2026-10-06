# Usage

This page is the reference for the `dnn-benchmark` command. `dnn-benchmark
--help` gives the same options with their defaults. `python -m
dnn_benchmarking` is the same command.

```text
dnn-benchmark -g PATH [PATH ...] [options]
dnn-benchmark --config FILE [options]
dnn-benchmark compare A.json B.json [options]
```

## Options

The groups below are the groups in `--help`. A flag with a `--no-` form
(for example `--no-autotune`) sets the value to false. Use this form to
override a config file.

### Input

| Option | Default | Description |
|---|---|---|
| `-g`, `--graph PATH [PATH ...]` | required | Graph JSON files, directories, globs or tarballs (`.tar`, `.tar.gz`, `.tgz`, `.tar.bz2`, `.tar.xz`). Quote globs so the shell does not expand them. |
| `--config PATH` | none | TOML recipe. Explicit CLI flags override its values. See [Config files](#config-files). |

Graphs run in command-line order. The files that one argument expands to
(a glob, a directory or a tarball) run in sorted order. A file that you name
two times runs one time, at its first position. The tool extracts a tarball
to a temporary directory and deletes the directory at the end of the run.
All graphs share one execution path: one graph, a glob and a tarball give
the same output format.

### Run

| Option | Default | Description |
|---|---|---|
| `-w`, `--warmup N` | 10 | Untimed launches per engine before the timed loop. The first launch is timed separately as `first_call_ms`. The other launches use the timed-iteration path (same mode and cache flush) and are discarded. At least one launch always runs. |
| `-i`, `--iters N` | 100 | Minimum timed iterations per engine. |
| `--min-time-ms MS` | 0 | Continue until the summed kernel time is `MS` or more. `0` gives exactly `--iters` samples. A cap of `max(10000, --iters)` samples applies. |
| `--cache-mode {warm,cold}` | warm | `cold` flushes L2 and MALL with a 512 MiB write before each warmup and timed iteration. |
| `--timing-block N` | 1 | Launches per timed sample. `1` times each launch on its own. `N > 1` uses rocKE block timing: before each sample, `--warmup` untimed launches and a drain; then `N` back-to-back launches in one event pair, recorded as `elapsed / N`. The first sample is discarded. Requires `--cache-mode warm`. |
| `-s`, `--seed SEED` | 0 | Random seed for the input data. |

[methodology.md](methodology.md) explains each value.

### Backend/Selection

| Option | Default | Description |
|---|---|---|
| `-b`, `--backend {hipdnn,pytorch}` | hipdnn | `hipdnn` runs the engine plugins. `pytorch` runs the graph through PyTorch (ROCm or CUDA). |
| `-e`, `--engine ENGINES` | all discovered | Comma list of engines, run in the given order. See [Engine selection](#engine-selection). |
| `--plugin-path PATHS` | see below | Plugin directory, or a comma list with one directory for each `--engine` entry. |
| `--autotune`, `--no-autotune` | off | Sets `HIPDNN_FORCE_BENCHMARKING=1`: each plan benchmarks its candidate kernels on the first execute and caches the winner. Use with `--cache-dir`. |
| `--cache-dir PATH` | none | Sets `HIPDNN_CACHE_DIR` for this run. |
| `--pytorch-sdpa-backend {default,flash,math,efficient,cudnn,overrideable}` | default | PyTorch SDPA category. A non-default category is strict: the graph fails if PyTorch cannot use it. |
| `--pytorch-rocm-fa-library LIBRARY` | none | ROCm Flash Attention implementation for PyTorch, for example `aotriton`. Requires `--pytorch-sdpa-backend flash`. |

When `--plugin-path` is not given, the tool uses
`$ROCM_PATH/lib/hipdnn_plugins/engines`. When `ROCM_PATH` is not set, it uses
the plugin directory of the pip ROCm SDK. The `setup_env.py` activation script
sets `ROCM_PATH`.

### Validation

| Option | Default | Description |
|---|---|---|
| `--validate {none,pytorch}` | none | Compare each engine output with a PyTorch reference. `pytorch` also adds a timed `reference` row. |
| `--rtol TOL` | dtype-aware | Relative tolerance. If you give only `--rtol` or only `--atol`, the value sets both. |
| `--atol TOL` | dtype-aware | Absolute tolerance. |

### Comparison

| Option | Default | Description |
|---|---|---|
| `--oracle-mode {off,plan,exhaustive}` | off | Also time the auto-tuned plan of each engine. See [Oracle mode](#oracle-mode). |

### Output

| Option | Default | Description |
|---|---|---|
| `-o`, `--output PATH` | none | Write the results to `PATH`: CSV when `PATH` ends in `.csv`, else JSON. See [results-schema.md](results-schema.md). |
| `-v`, `--verbose`, `--no-verbose` | off | Add a detail block for each engine below each table. |
| `-q`, `--quiet`, `--no-quiet` | off | Do not show progress lines and info messages. Tables, the summary, warnings and errors still show. |
| `--metrics-tier {basic,off}` | basic | `basic` adds FLOPs, I/O bytes, TFLOP/s, GB/s, workspace, VRAM and GPU clocks. They do not change the timed numbers. `off` disables them. |

### Profiling

Each profiling flag runs the workload again, in a child process under a
profiler, after the timed row completes. The timed numbers do not change.

| Option | Default | Description |
|---|---|---|
| `--pmc {basic,memory,flops,all}` | off | Collect hardware counters with `rocprofv3` (about 30 % extra wall time). `all` requires `--pmc-allow-multipass`. |
| `--pmc-allow-multipass`, `--no-pmc-allow-multipass` | off | Allow `--pmc all`. Multi-pass replay can hang for minutes. |
| `--emit-trace {pftrace}` | off | Write a Perfetto kernel and memcpy trace with `rocprofv3`. |
| `--perf`, `--no-perf` | off | Collect CPU cycles, instructions and IPC with `perf stat`. |
| `--roofline`, `--no-roofline` | off | Collect HBM and compute ceilings with `rocprof-compute --roof-only` (about 3 extra runs). |
| `--profiling-output-dir DIR` | `./profiling-output/<utc-timestamp>/` | Root directory for profiler artefacts. With a profiling flag, the tool checks at startup that it can create files there. Without one, the option has no effect and the tool shows a warning. |
| `--profiling-timeout SECONDS` | 600 | Wall-clock limit for each profiler process. `0` disables the limit. |

See [Profiling](#profiling) below.

## Engine selection

`--engine` accepts three forms. You can mix them in one list:

| Form | Example | Meaning |
|---|---|---|
| Name | `MIOPEN_ENGINE_DETERMINISTIC` | Case-sensitive engine name. The tool calculates the ID with 64-bit FNV-1a, as hipDNN does. |
| Hex ID | `0xA258541A6DAA1DE3` | The `engine.id` format of the result file. Values above `0x7FFF...` wrap to a negative signed ID. |
| Decimal ID | `-6748551569128940061` | Signed 64-bit ID. |

The three examples select the same engine.

The tool keeps the list order and duplicate entries. Use duplicates to time
the same engine from two plugin builds:

```bash
dnn-benchmark -g graphs/sample_conv_fwd.json \
  -e MIOPEN_ENGINE,MIOPEN_ENGINE \
  --plugin-path /path/to/build_a/engines,/path/to/build_b/engines
```

If no loaded plugin provides an engine in `--engine`, the run stops with exit
code 2 and names the engine. To list the engine names and IDs of a plugin
directory, run `hipdnn_list_engines --plugin-dir <dir>`.

The tables show the engine name. `-v` shows `NAME (0xHEX)`.

## PyTorch backend

`--backend pytorch` runs each graph through PyTorch and gives one
`provider = "pytorch"` row per graph. It uses the same timed loop, output
format and result schema as the hipDNN backend. It runs on ROCm and on CUDA.

These options are hipDNN-only. With `--backend pytorch` they stop the run
with exit code 2: `--engine`, `--plugin-path`, `--validate pytorch`, `--pmc`,
`--emit-trace`, `--perf`, `--roofline`, `--oracle-mode` (other than `off`),
`--autotune` and `--cache-dir`.

The SDPA options (`--pytorch-sdpa-backend`, `--pytorch-rocm-fa-library`) have
an effect only with `--backend pytorch` or `--validate pytorch`. Otherwise
the tool shows a warning.

```bash
dnn-benchmark -g graphs/sample_sdpa.json --backend pytorch \
  --pytorch-sdpa-backend flash --pytorch-rocm-fa-library aotriton
```

`--pytorch-rocm-fa-library` passes the value unchanged to PyTorch
(`preferred_rocm_fa_library`). PyTorch rejects an unknown library. PyTorch
can use a different Flash implementation when the preferred one does not
support an input.

## Kernel selection (`--autotune`, `--cache-dir`)

A hipDNN engine selects a kernel in one of two ways:

- Default (heuristic): the engine uses the rank-0 kernel of its heuristic.
  The table then measures the heuristic.
- `--autotune`: each plan benchmarks every candidate kernel on its first
  execute and caches the winner. The table then measures the best kernel in
  the kernel set.

With the hipDNN backend the tool states the path on stderr, for example
`kernel selection: heuristic; cache: shared per-user (~/.cache/hipdnn)`.
This is an info line, so `-q` hides it.

The winner cache is on disk. Its key is the graph content and the device, not
the checkout or the session. Thus a run can report a ranking that an earlier
or a concurrent run tuned. Give each phase its own empty `--cache-dir`:

```bash
dnn-benchmark -g 'graphs/*.json' --cache-dir /tmp/cache-heuristic
dnn-benchmark -g 'graphs/*.json' --autotune --cache-dir /tmp/cache-tuned
```

The tool shows a warning when `HIPDNN_FORCE_BENCHMARKING` is set to a true
value in the environment without `--autotune`, and when `--autotune` has no
`--cache-dir`.

## Oracle mode

`--oracle-mode` compares the default (out-of-the-box, OOTB) plan of each
engine with the plan that hipDNN tuning selects:

| Mode | Behavior |
|---|---|
| `off` | Time the default plan only. |
| `plan` | Also benchmark every plan that the backend generates, and time the fastest. |
| `exhaustive` | `plan`, plus provider-level kernel search where the engine supports it. Much slower. |

After the search, the tool times the default plan again and then the tuned
plan, back to back. The `oracle` column shows `baseline median / tuned
median`. A value less than 1.00x is a valid result. The column shows a label
instead of a speedup when no speedup applies:

| Label | Meaning |
|---|---|
| `no-search` | Only one plan was available. |
| `invalid` | The default or the tuned plan failed validation. |
| `failed` | Tuning did not produce a result. |
| `n/a` | No comparison is available. |

The summary shows the geometric mean of the valid speedups. The JSON `oracle`
object holds the plan counts and both measurements. See
[results-schema.md](results-schema.md#oracle).

Cache state changes the selection. For a cold heuristic baseline, set
`HIPDNN_DISABLE_EXACT_ENGINE_CACHE=1` and `HIPDNN_DISABLE_CACHE=1`. The tool
shows a warning when `HIPDNN_DISABLE_EXACT_ENGINE_CACHE` is not set, and,
once that is set, when `HIPDNN_DISABLE_CACHE` is not set. MIOpen FindDb and
performance database entries can still supply tuned selections.
`environment.selection_env` records the relevant variables.

```bash
HIPDNN_DISABLE_EXACT_ENGINE_CACHE=1 HIPDNN_DISABLE_CACHE=1 dnn-benchmark \
  -g graphs/sample_conv_fwd.json --oracle-mode exhaustive -v -o oracle.json
```

## Config files

`--config FILE` reads a TOML recipe. The precedence is:
defaults, then the config file, then explicit CLI flags. A relative path in
the file is relative to the directory of the file.

```bash
dnn-benchmark --config sample_configs/basic.toml.example -g graphs/sample_conv_fwd.json
dnn-benchmark --config sample_configs/config.toml.example --iters 500
```

The keys are the long option names with `_` in place of `-`, with these
differences:

| Config key | CLI option |
|---|---|
| `version` | none. Must be `1` when present. |
| `graphs` (list of paths) | `--graph` |
| `[[engines]]` tables with `id` and optional `plugin_path` | `--engine` and `--plugin-path` |

`sample_configs/config.toml.example` lists every key. An unknown key, a wrong
type or a value outside the choices stops the run with exit code 2.

Rules for `[[engines]]`:

- `id` is an engine name, a decimal ID or a `"0x..."` hex ID.
- If one entry sets `plugin_path`, every entry must set it, and the top-level
  `plugin_path` must be unset.
- `--engine` or `--plugin-path` on the command line replaces the whole engine
  list. A top-level `plugin_path` stays when you give only `--engine`.

## Output

### Console

The tool writes results and progress to different streams:

| Stream | Content |
|---|---|
| stdout | Suite header (host, GPU, ROCm or CUDA version, timing settings), one table for each graph, the legend (one time), `-v` detail blocks, the summary line, the results path, the oracle summary. |
| stderr | Graph progress (`[i/n] name`), engine progress lines, info lines (kernel selection, tarball extraction, profiling passes), `WARNING:` and `ERROR:` lines, and one-time `[metrics:<source>]` diagnostics. |

Thus `dnn-benchmark ... > results.txt` keeps only the results. `-q` removes
the progress and info lines.

On a terminal, an engine progress line shows `  NAME ...` while the engine
runs and completes in place. When stderr is not a terminal, the tool writes one
complete line after each engine. A warning never breaks a progress line.

Table columns:

| Column | Meaning |
|---|---|
| `engine` | Engine name (hipDNN) or `pytorch`. |
| `verdict` | Row verdict: `passed`, `unchecked`, `failed`, `reference`, `skipped` or `error`. |
| `kernel_med` | Median device time per launch, in µs, ms or s. `*` marks a noisy row or an outlier. |
| `iqr%` | Interquartile range of the kernel samples, as a percentage of the median. |
| `submit` | Median host time of the enqueue call. |
| `tflops` | TFLOP/s from the median. `~` means the FLOP count is partial. |
| `gbps` | GB/s (10^9 bytes/s) from the median. |
| `vs_best` | Best median of the graph divided by the row median. `1.00x` is the fastest row. `ref` marks the reference row. |
| `oracle` | Only with `--oracle-mode`. See [Oracle mode](#oracle-mode). |
| `note` | The skip or error reason, or the first row warning other than `noisy:`. `(+N)` shows the number of more warnings. |

Example (MI210, `-g graphs/sample_conv_fwd.json graphs/sample_layernorm.json`):

```text
sample_conv_fwd (sample_conv_fwd_16x16x16x16_k16_3x3)  [e50543fd1d83]
  engine                       verdict    kernel_med  iqr%    submit  tflops  gbps  vs_best
  MIOPEN_ENGINE                unchecked   25.44 µs    1.9  11.55 µs    0.74  21.0    1.00x
  MIOPEN_ENGINE_DETERMINISTIC  unchecked   25.60 µs    1.3  11.47 µs    0.74  20.8    0.99x
  kernel_med = median device time per launch (staged stall-gate; cache warm); submit = host enqueue time; vs_best = best
  median / row median; * = noisy

sample_layernorm (sample_layernorm_2x3x4)  [ad562fbb5a14]
  no engines applicable: Failed to get ranked engine ids: No engine configurations available for the graph.

Summary: 2 graph(s), 2 row(s): 0 passed, 2 unchecked, 0 failed, 0 skipped, 0 error(s); 1 graph(s) without engines
```

The graph title is the file stem, then the `name` of the graph JSON in
parentheses when it is different. The value in brackets is the `graph_id`.
A graph with no applicable engine shows the reason from hipDNN.
`unchecked` means that the row ran without validation. It is not a pass.

`-v` adds a block for each row with the plugin path, costs (build, first
call, row total), timing mode, a statistics table for kernel and submit time,
clocks before and after the loop, FLOPs, I/O and VRAM, correctness, oracle and
profiling results, and all warnings.

### Result files

`-o results.json` writes the full result (schema version 2). `-o results.csv`
writes one line per row with the main columns. See
[results-schema.md](results-schema.md).

The tool checks the output path before the first graph runs. During the run it
writes the file again at most every 10 seconds, and one last time at the end.
After Ctrl-C or SIGTERM the file holds every completed graph and has
`run.complete = false`.

## Exit codes

| Code | Meaning |
|---|---|
| 0 | All engine rows ran. No engine row failed validation. A run where every row is skipped also exits 0. |
| 1 | An engine row error, a graph error, a result write failure, no graph files found, a tarball that cannot be read, or a backend that is not available at startup (hipDNN, PyTorch or the reference provider). |
| 2 | Usage error: a bad flag, a bad config file, an option that the backend does not support, an unknown `--engine`, an output path or a `--profiling-output-dir` that cannot be written, or a missing profiler tool. |
| 3 | At least one engine row failed validation. |
| 130 | Interrupted by SIGINT (Ctrl-C). |
| 143 | Interrupted by SIGTERM. |

When more than one condition applies, 3 wins over 1, and 1 wins over 0.
Only `engine` rows count: the `reference` row that `--validate pytorch` adds
never changes the exit code.

## compare

`dnn-benchmark compare A.json B.json` compares two JSON result files. It
reads JSON only, not CSV.

| Option | Default | Description |
|---|---|---|
| `--by {best,ref,engine}` | best | `best`: the fastest usable engine row of each graph. `ref`: the reference rows. `engine`: the same engine in both files. |
| `--metric {kernel,host}` | kernel | Compare kernel medians or submit medians. |
| `--threshold PCT` | 5 | Regression threshold in percent. |
| `--allow-mismatch` | off | Compare also when `run.config.cache_mode` is different. |
| `--csv` | off | Write CSV to stdout. Not with `--json`. |
| `--json` | off | Write JSON to stdout. Not with `--csv`. |

Rules:

- `speedup = A_median / B_median`. A value more than 1 means B is faster.
- Graphs join on `graph_id` only. A graph that did not load has no
  `graph_id`, so it shows as `graph only in A` or `graph only in B`.
- In `--by engine` mode, rows join on role, provider, engine ID and engine
  name. When a file has more than one row with the same key, the rows pair
  in order of appearance.
- `best` and `ref` select only rows with the verdict `passed`, `unchecked`
  or `reference`. In `engine` mode, a `failed` row gets a speedup with the
  label `A failed` or `B failed`, but it is not in the geometric mean. Rows
  with `error` or `skipped` get no speedup.
- A pair is `within noise` when the relative change is not more than
  `max(threshold, 2 * sqrt(r_A^2 + r_B^2))`, where `r` is `iqr_ms / median_ms`
  of the compared metric in each file (`a_rel_iqr`, `b_rel_iqr` in the
  `--json` and `--csv` output). A slower B beyond that limit is a
  `REGRESSION`. A faster B is `faster`.
- A different `run.config` value gives a warning on stderr. A different
  `cache_mode` stops the comparison, unless you give `--allow-mismatch`.

Exit codes: 0 no regression, 1 one or more regressions, 2 usage error or a
file that cannot be read or has a different schema version.

```bash
dnn-benchmark -g 'graphs/*.json' -o base.json
dnn-benchmark -g 'graphs/*.json' -o new.json
dnn-benchmark compare base.json new.json
```

To compare a ROCm host with a CUDA host, run `--backend pytorch -o` on each
host, then compare the two files on one host.

## Profiling

Each requested pass runs this child process under its profiler:

```text
python -m dnn_benchmarking --internal-profiling-run --graph G --engine E \
    --warmup W --iters 5 --seed S [--plugin-path P]
```

The child does not recurse and does not print on success.
[methodology.md](methodology.md#profiling-child-process) tells why the
child does not use the timed loop.

Before the first graph, the tool checks that each requested profiler exists
and that it can create files in `--profiling-output-dir`. If a check fails,
the run stops with exit code 2. A pass that starts but fails does not stop
the run: the row records `skipped`, `returncode` or `error_tail` in
`extra_metrics`. If a pass raises an exception, the row keeps its timed
values and gets the warning `profiling failed: <error>`.

Artefacts go to
`<profiling-output-dir>/<graph-stem>-<hash6>/<engine-name>/<pass>/`. The file
`<graph-stem>-<hash6>/.source` holds the path of the graph. The row stores the
parsed values in `extra_metrics.pmc`, `.trace`, `.perf` and `.roofline`. See
[results-schema.md](results-schema.md#extra_metrics) and
[troubleshooting.md](troubleshooting.md) for how to view the artefacts.

## Check graph files

`tools/check_deserialize.py` checks that graph JSON files load, without a
plan build or a kernel launch:

```bash
# Pure-Python loader only (no hipDNN build required)
python tools/check_deserialize.py --level json --src src 'Workloads/**/*.json'

# Full deserialize and operation graph build (needs a built hipDNN)
python tools/check_deserialize.py --level opgraph 'Workloads/**/*.json'
```

The script accepts globs, directories and extracted tarball trees. It exits
with a non-zero code on a failure and prints the first failures.

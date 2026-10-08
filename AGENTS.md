# AGENTS.md

Guidance for coding agents (Claude Code, Codex and others) that work in this
repository. `CLAUDE.md` imports this file. Keep one copy of each rule here.

## Project overview

dnn-benchmarking is a benchmarking and validation tool for hipDNN graphs. It
loads JSON-serialized hipDNN graphs, runs them through hipDNN engine plugins,
measures each engine, and optionally compares the outputs with a PyTorch
reference.

`--runtime pytorch` runs the same graphs through PyTorch on ROCm or CUDA. Both
runtimes share the suite path, the timed loop (`execution/timing.measure`)
and the result schema. A CUDA host without hipDNN can thus produce a result
file that `dnn-benchmark compare` can compare with a ROCm result. The package
stays importable without `hipdnn_frontend`: every hipDNN import is lazy. The
hipDNN backend does not import `torch` at startup.

User documentation:

- `README.md`: quick start.
- `docs/setup.md`: `setup_env.py`.
- `docs/usage.md`: CLI reference, console output, exit codes, `compare`.
- `docs/methodology.md`: timing method and known limits.
- `docs/results-schema.md`: result file schema version 2.
- `docs/troubleshooting.md`: ROCm paths and profiling.

Update the matching document in the same change when you change a CLI
option, a result key, the timing method or an exit code.

## Setup

```bash
python3 setup_env.py --workspace .workspace     # ROCm host: venv, ROCm torch, hipDNN, plugins
python3 setup_env.py --torch-mode cuda          # CUDA host: PyTorch runtime only
pip install -e ".[test]"                         # package only, into an existing env
source .workspace/.venv/bin/activate             # sets ROCM_PATH and LD_LIBRARY_PATH
```

Each worktree keeps its own `.workspace/` (or `build/`) and `.venv`.

## Running the tool

```bash
dnn-benchmark -g graphs/sample_conv_fwd.json                  # all engines, summary table
dnn-benchmark -g graphs/sample_conv_fwd.json -e MIOPEN_ENGINE -v
dnn-benchmark -g 'graphs/*.json' --validate pytorch -o results.json
dnn-benchmark -g graphs/sample_conv_fwd.json --runtime pytorch
dnn-benchmark compare base.json new.json
```

`python -m dnn_benchmarking` is the same command. Without an installed
package, run `PYTHONPATH=src python -m dnn_benchmarking ...`.

## Exit codes

| Code | Meaning |
|---|---|
| 0 | Success. An all-skipped run is also 0. |
| 1 | Engine row error, graph error, failed final result write (not a periodic one), no graph files, or runtime not available at startup. |
| 2 | Usage error: argparse, config file, runtime-incompatible option, unknown `--engine`, unwritable `-o` or `--profiling-output-dir`, missing profiler tool. |
| 3 | At least one engine row failed validation. 3 wins over 1, and 1 wins over 0. Only `role == "engine"` rows count, as in `summary()`. |
| 130 / 143 | SIGINT / SIGTERM. The result file is partial (`run.complete = false`). |

`dnn-benchmark compare` uses 0 (no regression), 1 (regression) and 2 (usage
or unreadable input).

## Architecture

```
src/dnn_benchmarking/
├── cli/
│   ├── main.py               # entry point; routes `compare`, resolves --graph
│   ├── parser.py             # CLI_OPTIONS: one table for flags, defaults, config keys
│   ├── config_file.py        # TOML recipe merge (defaults < config < CLI)
│   ├── suite_runner_cli.py   # startup checks, per-graph loop, result writes, exit codes
│   ├── runtimes.py           # runtime startup; returns the per-graph runner
│   └── internal_profiling.py # hidden --internal-profiling-run child
├── common/                   # exceptions, dtypes registry, torch/ROCm runtime helpers
├── config/benchmark_config.py# SuiteConfig, TimingPolicy, MetricsConfig, ValidationConfig
├── execution/
│   ├── timing.py             # measure(): the one timed loop (staged / events)
│   ├── executor.py           # hipDNN engine executor
│   ├── pytorch_executor.py   # PyTorch executor
│   ├── buffer_manager.py     # hipDNN device buffers and input generation
│   ├── pytorch_buffer_manager.py
│   ├── pytorch_ops/          # graph node -> PyTorch op handlers
│   ├── suite_runner.py       # one graph: engines, rows, clocks, warnings
│   ├── oracle.py             # --oracle-mode pass
│   └── correctness.py        # output comparison for a row
├── graph/                    # loader (load + validate), tensor_info, resolver (globs, tarballs)
├── metrics/
│   ├── analytical/           # FLOP and I/O byte formulas
│   ├── gpu_smi.py            # amdsmi clocks, VRAM, static GPU info
│   ├── machine_info.py       # environment block, collected once per suite
│   ├── host.py               # host memory snapshot
│   ├── profiling_orchestrator.py  # re-runs the child under each profiler
│   ├── rocprof_pmc.py, rocprof_trace.py, perf.py, roofline.py
│   └── _diagnostic.py        # warn_once
├── reporting/
│   ├── statistics.py         # BenchmarkStats, noise_warnings, TimingInfo
│   ├── suite_results.py      # result data model, schema v2, JSON/CSV write, load
│   ├── reporter.py           # console: results on stdout, progress on stderr
│   └── compare.py            # `dnn-benchmark compare`
└── validation/               # reference providers and tolerance comparison
```

Data flow (hipDNN): CLI -> `SuiteConfig` -> `runtimes.start_runtime` -> for
each graph: `GraphLoader` -> `suite_runner.run_graph_all_providers` ->
`Executor` + `BufferManager` -> `timing.measure` -> correctness, oracle,
profiling -> `GraphResult` -> `Reporter` and `SuiteResult.write`.

Data flow (PyTorch): the same, with `run_graph_pytorch`,
`PyTorchCudaExecutor` and `PyTorchCudaBufferManager`. One
`runtime = "pytorch"` row per graph. No engine discovery or plugins.

Rules that keep the design intact:

- Time every row with `timing.measure`. Do not add a second timed loop.
- Keep results on stdout and progress, warnings and errors on stderr. Use
  the `Reporter` methods; do not `print` from other modules. Use
  `metrics._diagnostic.warn_once` for one-time diagnostics.
- Every result key is always present. Add a new key to the key tuples in
  `suite_results.py`, and to `docs/results-schema.md`.
- Add a CLI option only in `cli/parser.py:CLI_OPTIONS`. The config file and
  the defaults derive from that table.
- Keep `hipdnn_frontend` and `torch` imports lazy. Importing `torch` takes
  seconds, so the hipDNN path must not import it at startup. Code that only
  reads torch facts (environment, GPU identity) uses `sys.modules["torch"]`
  when torch is already loaded, and does not import it.
- Build every hipDNN plan through `Executor.prepare`, which times only
  `create_execution_plan_ext(engine_id, knobs)` -> `check_support()` ->
  `build_plans()`. Prime each engine with `Executor.prime` before its timed
  OOTB build, and build the oracle's `global.benchmarking=1` plan right after
  the OOTB build, so the two build times start from the same process state.
- Run tuned PyTorch only in the `--internal-pytorch-tuned` child process.
  PyTorch's conv algorithm cache and MIOpen's user database would otherwise
  carry tuning into the OOTB measurement.

## Tests

| Tier | Location / marker | Needs |
|---|---|---|
| Unit | `tests/unit/`, no marker | Any host. Fake torch. No GPU. |
| GPU-generic | `gpu` | Any live GPU (ROCm or CUDA). `expected_timer()` adapts the assertions. ROCm-only tests skip through the `hipdnn` and `plugin_paths` fixtures. |
| CUDA-only | `gpu` + `cuda` | A CUDA PyTorch build and GPU. |
| Profiling | `rocprofv3`, `perf`, `rocprof_compute` | The profiler binary. See `docs/troubleshooting.md`. |
| Strict profiling | `profiling_strict` | Real profiler artefacts. Runs only with `--profiling-strict`. |

```bash
pytest -m "not gpu"                                  # unit tests, any host
pytest                                               # GPU host (ROCm or CUDA): unit + GPU
pytest tests/unit/execution/test_timing.py           # one file
pytest -m gpu --dnn-plugin-paths /path/to/engines    # custom plugin builds
pytest --profiling-strict -m profiling_strict        # known-good profiling host
```

CI runs `pytest tests/unit -m "not gpu"` (unit-tests.yml, Linux and
Windows), `pytest -m "not gpu"` after the CUDA-path install (setup.yml), and
`pytest` on a gfx950 runner (setup.yml `gpu-test`).

Every GPU test skips itself on the wrong platform, so a bare `pytest` is safe
on any host. The `-ra` in pyproject.toml prints the reason of every skip;
read them on a GPU host. GPU tests need the ROCm libraries on
`LD_LIBRARY_PATH`; the `setup_env.py` activation script sets it. The
integration fixture `hipdnn` needs only `hipdnn_frontend` and a HIP device;
request `torch_gpu` too only in tests that use PyTorch (`--runtime pytorch`,
`--validate pytorch`).

Test rules:

- A test defends an observable contract. Do not pin message wording,
  defaults, `hasattr` checks or mock wiring.
- Keep unit tests free of GPU and network access.

## Workflow

1. Plan first for non-trivial tasks.
2. Verify before you report done: run the tests of the changed files, and
   run the tool on a sample graph when you change the timed path or the
   console.
3. Use absolute paths when you work in project clones and worktrees.
4. Do not commit unless the user asks.

## Writing standard

Use ASD-STE100 Simplified Technical English for docs, comments, commit
messages and reports:

- Use short sentences (25 words or fewer) with one main action.
- Use the active voice and the imperative mood for instructions.
- Use one term for each concept. Define each abbreviation at first use.
- Use numbered steps for procedures.
- Do not use marketing words or emojis.
- Keep command names, paths and identifiers unchanged.

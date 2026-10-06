# dnn-benchmarking

Benchmarking and validation tool for hipDNN graphs.

> **Caution**: This tool is in early development and can change. Do not use
> it in build workflows or CI gating.

dnn-benchmarking loads serialized hipDNN graphs (JSON). It runs each graph
through the installed hipDNN engine plugins, or through PyTorch with
`--backend pytorch`. For each engine it measures the device time of one
launch and, if you ask, compares the output with a PyTorch reference.

- One timed loop for all backends. The headline number is the median device
  time per launch (`kernel_med`), measured with a stall-gated event span.
- Each result records how it was measured: timing mode, cache mode, warmup,
  sample count, seed and first-call cost.
- Results go to stdout as tables, progress goes to stderr, and `-o` writes a
  versioned JSON or CSV file.
- `dnn-benchmark compare` compares two result files, for example two builds
  or a ROCm host and a CUDA host.

## Requirements

- Python 3.12 or newer.
- hipDNN backend: an AMD GPU with ROCm, the hipDNN Python bindings
  (`hipdnn_frontend`) and the engine plugins. `setup_env.py` builds them.
- PyTorch backend: a ROCm or CUDA build of PyTorch. hipDNN is not necessary.
- `--validate pytorch`: any PyTorch build. A CPU build is sufficient.
- Optional: amdsmi for GPU clocks and throttle status; `rocprofv3`, `perf`
  and `rocprof-compute` for the profiling flags.

## Quick start

1. Set up the environment (ROCm host). See [docs/setup.md](docs/setup.md)
   for CUDA hosts, CPU-only PyTorch and Docker.

   ```bash
   python3 setup_env.py --workspace .workspace
   source .workspace/.venv/bin/activate
   ```

2. Benchmark one graph with all engines that support it:

   ```bash
   dnn-benchmark -g graphs/sample_conv_fwd.json
   ```

   ```text
   sample_conv_fwd (sample_conv_fwd_16x16x16x16_k16_3x3)  [e50543fd1d83]
     engine                       verdict    kernel_med  iqr%    submit  tflops  gbps  vs_best
     MIOPEN_ENGINE                unchecked   25.44 µs    1.9  11.55 µs    0.74  21.0    1.00x
     MIOPEN_ENGINE_DETERMINISTIC  unchecked   25.60 µs    1.3  11.47 µs    0.74  20.8    0.99x
   ```

3. Run the same graph through PyTorch:

   ```bash
   dnn-benchmark -g graphs/sample_conv_fwd.json --backend pytorch
   ```

4. Benchmark a set of graphs, validate the outputs and write the results:

   ```bash
   dnn-benchmark -g 'graphs/*.json' --validate pytorch -o results.json
   dnn-benchmark -g 'graphs/*.json' -o results.csv
   ```

5. Compare two result files. `speedup = A / B`; more than 1 means B is
   faster:

   ```bash
   dnn-benchmark compare base.json results.json
   ```

More examples:

```bash
# Select engines by name, hex ID or decimal ID, in order
dnn-benchmark -g graphs/sample_conv_fwd.json -e MIOPEN_ENGINE -v

# Cold cache, at least 200 iterations and 100 ms of kernel time
dnn-benchmark -g graphs/sample_conv_fwd.json -i 200 --min-time-ms 100 --cache-mode cold

# rocKE block timing: each of 100 samples times 50 back-to-back launches
dnn-benchmark -g 'graphs/*.json' --iters 100 --timing-block 50

# Run every graph in a tarball (fetch it first, see Workload files)
dnn-benchmark -g Workloads/headline/conv.tar.gz -o conv.json

# Repeatable recipe; CLI flags override the file
dnn-benchmark --config sample_configs/basic.toml.example -g graphs/sample_conv_fwd.json
```

## Exit codes

| Code | Meaning |
|---|---|
| 0 | Success. |
| 1 | Engine row error, graph error, write failure, no graph files, or backend not available. |
| 2 | Usage or config error. |
| 3 | An engine row failed validation. |
| 130 / 143 | Interrupted by SIGINT / SIGTERM. A partial result file has `run.complete = false`. |

## Documentation

| Document | Content |
|---|---|
| [docs/setup.md](docs/setup.md) | `setup_env.py`: torch modes, CMake defines, rocm-libraries source, CUDA hosts, Docker. |
| [docs/usage.md](docs/usage.md) | All CLI options, engine selection, config files, console output, exit codes, `compare`, profiling. |
| [docs/methodology.md](docs/methodology.md) | What `kernel_med` measures, warmup, cache modes, stop rule, statistics, parity with rocKE and PyTorch, known limits. |
| [docs/results-schema.md](docs/results-schema.md) | Result file schema version 2 (JSON and CSV). |
| [docs/troubleshooting.md](docs/troubleshooting.md) | ROCm library paths, profiler requirements, profiling artefacts. |
| [AGENTS.md](AGENTS.md) | Architecture and development rules. |

`dnn-benchmark --help` is the authoritative option list.

## Workload files

The `Workloads/` directory holds benchmark workload tarballs tracked with
[DVC](https://dvc.org/). The public DVC remote in `.dvc/config` allows
anonymous reads.

- `Workloads/headline/`: the curated set for regression monitoring
  (`conv`, `bnorm`, `attn`, `hipblaslt`, `moe`, `norm`). No ROCm engine
  implements RMSNorm or LayerNorm yet, so `norm.tar.gz` has no applicable
  engines today.
- `Workloads/microbench/`: larger shape sweeps and per-source collections.
- `Workloads/models/`: workloads for each model.

```bash
python -m pip install "dvc[s3]"
dvc pull                                   # every workload
dvc pull Workloads/headline/conv.tar.gz.dvc  # one workload
```

Keep credentials in `.dvc/config.local`, which git ignores.

## Tests

```bash
pytest -m "not gpu"        # unit tests, any host
pytest -m "not cuda"       # ROCm host: unit + GPU tests
pytest -m "not rocm"       # CUDA host: unit + GPU tests
```

See [AGENTS.md](AGENTS.md#tests) for the test tiers and markers.

## Related tools

For the MIOpen shape conversion tool, see
[`dnn-convert-shapes`](https://github.com/ROCm/rocm-libraries/tree/develop/projects/hipdnn/tools/dnn-convert-shapes).

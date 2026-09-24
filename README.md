# dnn-benchmarking

Benchmarks and validates serialized hipDNN graphs. The tool runs each graph
through the installed hipDNN engine plugins, or through PyTorch with
`--backend pytorch`, and reports kernel and end-to-end timing.

> **Caution**: This tool is in early development and subject to change.
> Do not use it in build workflows or CI pipelines.

## Installation

The tool needs Python 3.12 or newer.

### Released wheels (AMD GPUs)

This path needs no compiler, CMake, or system ROCm. Wheels exist for `gfx90a`,
`gfx942`, `gfx950`, `gfx1100`, and `gfx1151`.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r https://github.com/ROCm/dnn-benchmarking/releases/download/<tag>/requirements-gfx942.txt
```

Take `<tag>` from the [releases page](https://github.com/ROCm/dnn-benchmarking/releases).
Replace `gfx942` with the architecture that `rocm_agent_enumerator` reports.
The requirements file pins torch and the ROCm SDK to one nightly. It also
installs hipDNN and the engine plugins, built from the `rocm-libraries` commit
that this repository pins. If `ROCM_PATH` is set, it overrides the bundled
plugins.

### Source build

Use the source build to change hipDNN or the providers, or when no wheel exists
for your GPU:

```bash
python3 setup_env.py --workspace .workspace
source .workspace/.venv/bin/activate
```

`setup_env.py` installs ROCm PyTorch for the detected GPU. It then builds hipDNN,
the MIOpen, hipBLASLt, and hip-kernel providers, and the Python bindings from
the `rocm-libraries` submodule. See [docs/source-build.md](docs/source-build.md)
for CPU-only, CUDA, and existing-torch modes, extra CMake defines, and the
workspace location.

## Usage

`--graph` accepts a JSON file, a glob, or a tarball (`.tar`, `.tar.gz`, `.tgz`,
`.tar.bz2`, `.tar.xz`). All three use the same suite path.

```bash
dnn-benchmark --graph ./graphs/sample_conv_fwd.json --warmup 10 --iters 100
dnn-benchmark --graph 'graphs/*.json' -o results.json      # full JSON results
dnn-benchmark --graph ./Workloads/headline/conv.tar.gz -v  # per-engine detail
dnn-benchmark --graph ./graphs/sample_conv_fwd.json --engine 1,2 --seed 42
```

The console shows one summary line for each graph and a suite summary. `-v`
shows a block for each engine. `--output` always writes the full `SuiteResult`
JSON.

The tool finds engine plugins from the installed `hipdnn_runtime` wheel or from
`$ROCM_PATH/lib/hipdnn_plugins/engines`. `--plugin-path` takes one directory or
a comma-separated list in `--engine` order.

Other modes, described in [docs/usage.md](docs/usage.md):

- `--backend pytorch`: run the graphs through PyTorch, on ROCm or CUDA, with
  the same JSON schema for cross-machine comparison.
- `--autotune` and `--cache-dir`: measure the full kernel set instead of the
  heuristic pick, with an isolated winner cache.
- `--oracle-mode plan|exhaustive`: compare the out-of-the-box plan with the
  tuned plan.

`--config <file.toml>` loads a repeatable recipe; CLI flags override its values.
See `sample_configs/`. Run `dnn-benchmark --help` for all options.

## Workload Files

`Workloads/` holds benchmark tarballs tracked with [DVC](https://dvc.org/). The
committed remote in `.dvc/config` allows anonymous reads.

- `Workloads/headline/`: the curated regression set (`conv`, `bnorm`, `attn`,
  `hipblaslt`, `moe`, `norm`). `norm.tar.gz` has 0% applicability until a ROCm
  engine implements RMSNorm or LayerNorm.
- `Workloads/microbench/`: broader shape sweeps, outside the headline signal.
- `Workloads/models/`: workload collections for each model.

```bash
python -m pip install "dvc[s3]"
dvc pull                                        # every workload
dvc pull Workloads/headline/conv.tar.gz.dvc     # one workload
```

Keep credentials in `.dvc/config.local`, which git ignores.

To check that new graphs deserialize without running a kernel:

```bash
# Pure-Python loader only, no hipDNN build needed
python tools/check_deserialize.py --level json --src src 'Workloads/**/*.json'
# Deserialize and finalize the backend operation graph, needs hipDNN
python tools/check_deserialize.py --level opgraph 'Workloads/**/*.json'
```

For MIOpen shape conversion, see
[`dnn-convert-shapes`](https://github.com/ROCm/rocm-libraries/tree/develop/projects/hipdnn/tools/dnn-convert-shapes).

## Running Tests

```bash
pytest -m "not gpu"   # no GPU or hipDNN needed
pytest                # everything; GPU tests skip on the wrong platform
pytest -m "not cuda"  # ROCm host: drop CUDA-only tests
pytest -m "not rocm"  # CUDA host: drop ROCm-only tests
pytest -m gpu --dnn-plugin-paths /path/to/hipdnn_plugins/engines
```

GPU tests find plugins in provider build trees, the active venv's ROCm SDK, and
`/opt/rocm`. Markers: `gpu` runs on any GPU, `rocm` only on AMD, `cuda` only on
NVIDIA.

Strict profiling tests need real profiler artifacts and skip by default:

```bash
LD_LIBRARY_PATH=$HIPDNN_PREFIX/lib:$LD_LIBRARY_PATH pytest --profiling-strict -m profiling_strict
```

See [docs/troubleshooting.md](docs/troubleshooting.md) for profiling setup and
library-path problems.

## Releasing Wheels

`tools/build_release_wheels.py` builds the per-architecture `hipdnn_runtime`
wheel, the `hipdnn_frontend` bindings, this package, and a requirements file for
each architecture. It uses the pinned `rocm-libraries` commit. The ROCm SDK
wheels bundle an older hipDNN that lacks entry points the bindings need, so the
runtime wheel ships its own. The tool loads that copy first.

The `release-wheels` workflow builds and publishes a release. Start it manually
from the Actions tab: CI cross-compiles the wheels and cannot run them on the
target GPUs. To publish by hand:

```bash
TAG=v0.1.0
python3 tools/build_release_wheels.py \
    --base-url https://github.com/ROCm/dnn-benchmarking/releases/download/$TAG
gh release create $TAG dist/*.whl dist/requirements-*.txt
```

Do not publish to `rocm.nightlies.amd.com`. That index belongs to TheRock.

## Limitations

- Validation-provider rows are timing baselines. They are not counted as hipDNN
  engine pass or fail results. Use `--validate` to check output correctness.

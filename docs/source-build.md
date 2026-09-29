# Source Build (`setup_env.py`)

Use the source build when you change hipDNN or the providers, or when no
released wheel exists for your GPU architecture. Otherwise, use the released
wheels in the [README](../README.md#installation).

```bash
python3 setup_env.py --workspace .workspace
source .workspace/.venv/bin/activate
```

## What setup does

The default `--torch-mode rocm` flow assumes no system ROCm installation:

1. Creates a virtual environment under the selected workspace.
2. Detects the GPU architecture and installs the matching ROCm PyTorch nightly
   wheel.
3. Discovers ROCm libraries from the torch wheel's bundled ROCm SDK.
4. Builds hipDNN when its CMake configs are absent from the selected prefix.
5. Builds the MIOpen, hipBLASLt, and hip-kernel providers when their installed
   artifacts are missing. It uses the ROCm SDK devel wheel for the compiler.
6. Installs the hipDNN Python bindings against the selected ROCm SDK libraries.

Setup prints the prefix as `Using hipDNN/ROCm prefix: ...`. Activation sets
`ROCM_PATH` to that prefix and prepends its `lib` directory to
`LD_LIBRARY_PATH`. The tool finds plugins in
`$ROCM_PATH/lib/hipdnn_plugins/engines`.

If GPU detection is unavailable on the setup host, pass `--gpu-arch`, for
example `--gpu-arch gfx942`. Pass `--reuse-artifacts` to skip the build and use
the artifacts that are already installed.

## Workspace location

The workspace holds the virtual environment, the Python bytecode cache, and the
runtime benchmark caches. Setup selects the first of these:

1. `--workspace <path>`
2. `$DNN_BENCH_WORKSPACE`
3. `/workspace`, when it exists and is writable
4. `.workspace` under the `dnn-benchmarking` directory

## rocm-libraries checkout

`rocm-libraries` is a git submodule that tracks `develop`. On first run,
`setup_env.py` fetches it as a sparse, blobless clone of the two subtrees this
tool builds (`projects/hipdnn`, `dnn-providers`), not the full ~9 GB
monorepo. Setup leaves a checkout from `git submodule update --init` as it is.

To build against a different ref, check it out directly:

```bash
git -C rocm-libraries fetch --depth 1 origin <ref>
git -C rocm-libraries checkout FETCH_HEAD
```

## Extra CMake defines (`--cmake-arg`)

`--cmake-arg NAME=VALUE` appends a define to the hipDNN and provider configure.
It is repeatable. Setup appends the extra defines after its defaults, so an
extra define overrides a default.

Both `NAME=VALUE` and `-DNAME=VALUE` work. Write the `-D` form with `=`
(`--cmake-arg=-DFOO=ON`): with a space, argparse reads `-DFOO=ON` as an option.

```bash
# Build the descriptor-backed kernel-ingestor engine, which is gated OFF
python3 setup_env.py --workspace .workspace \
  --cmake-arg HIPDNN_ENABLE_KERNEL_INGESTOR=ON

python3 setup_env.py --cmake-arg HIPDNN_ENABLE_SDPA=OFF --cmake-arg=-DCMAKE_BUILD_TYPE=Debug
```

Any engine gated behind a non-default CMake option needs this. Without the
option, the engine compiles into no plugin. The plugin `.so` still installs, so
`--plugin-path` looks correct and every graph reports `no engines applicable`.

`HIPDNN_ENABLE_KERNEL_INGESTOR=ON` also needs the `rocm-kpack` CMake package
and the `rocm_kpack` Python package with `zstandard` and `msgpack`. The default
`--torch-mode rocm` path provides neither: the bundled ROCm SDK ships
`librocm_kpack.so` but no CMake config, so configure fails. Use a ROCm install
that ships the package, and supply the Python half:

```bash
python3 setup_env.py --torch-mode existing --rocm-prefix /opt/rocm \
  --cmake-arg HIPDNN_ENABLE_KERNEL_INGESTOR=ON \
  --cmake-arg HIPKERNELPROVIDER_KPACK_ALLOW_FETCH=ON
```

`HIPKERNELPROVIDER_KPACK_ALLOW_FETCH=ON` clones the pinned `rocm_kpack` source.
To use a local copy, pass `--cmake-arg HIPKERNELPROVIDER_KPACK_PYTHON_DIR=<dir>`
instead. The hipDNN dev container stages both at `/opt/rocm-kpack/python` and
needs neither flag.

## Other torch modes

| Mode | Use | Result |
| --- | --- | --- |
| `rocm` (default) | AMD GPU, no system ROCm | Everything above. |
| `cpu` | CI with ROCm and hipDNN already installed | CPU-only torch for `--validate pytorch`. The `--backend pytorch` executor stays unavailable. |
| `existing` | Reuse torch in the target venv | ROCm torch uses its bundled SDK. CUDA torch skips all ROCm setup. CPU torch builds the bindings against `--rocm-prefix`, `$ROCM_PATH`, or `/opt/rocm`. |
| `cuda` | NVIDIA GPU | Torch from PyPI (`--torch-index-url` overrides). Skips all ROCm setup, so only `--backend pytorch` works. |

```bash
python3 setup_env.py --torch-mode cpu --rocm-prefix /opt/rocm
python3 setup_env.py --torch-mode cuda --workspace .workspace
```

On CUDA, `torch.cuda` events time the kernels. The ROCm metadata fields
(`rocm_version`, the amdsmi GPU snapshot) are `None` in the JSON, and
`gpu_arch` is `"unknown"`.

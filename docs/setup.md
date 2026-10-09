# Setup

`setup_env.py` creates a virtual environment, installs PyTorch, builds hipDNN
and the provider plugins, and installs dnn-benchmarking. It requires Python
3.12 or newer. `python3 setup_env.py --help` lists every option.

## Quick setup (ROCm)

```bash
python3 setup_env.py --workspace .workspace
source .workspace/.venv/bin/activate
dnn-benchmark -g graphs/sample_conv_fwd.json
```

The default `--torch-mode rocm` does not need a system ROCm installation.
Setup prints each stage as `==> [k/n] <name>`. With `--torch-mode rocm` the
stages are:

1. `Virtual environment`: create `<workspace>/.venv`, or reuse it. See
   [Reusing a venv](#reusing-a-venv).
2. `PyTorch (--torch-mode rocm)`: detect the GPU architecture and install the
   matching ROCm PyTorch nightly (`torch[device-<arch>]`). Its ROCm SDK
   libraries and toolchain are used for the build.
3. `dnn-benchmarking package`: install dnn-benchmarking (editable unless
   `--no-editable`).
4. `rocm-libraries sources`: fetch `rocm-libraries` if it is absent (see
   [rocm-libraries source](#rocm-libraries-source)).
5. `Build dependencies`: pip packages the source build needs. Skipped with
   `--reuse-artifacts`.
6. `hipDNN and provider plugins`: build hipDNN and the MIOpen, hipBLASLt and
   hip-kernel providers, and install them into the ROCm SDK prefix. With
   `--reuse-artifacts`, only check that hipDNN is in the prefix.
7. `hipDNN Python bindings`: build `hipdnn_frontend` against that prefix.
   With `--reuse-artifacts`, build nothing and instead check that the venv
   has the `hipdnn_frontend` package, without importing it (its native
   libraries may need a GPU this host has not got). Setup fails when the
   package is absent.
8. `amdsmi and rocprofiler libraries`: install the amdsmi Python bindings
   and link the wheel rocprofiler libraries.
9. `Verify installation`.

Then setup prints the `Profiling sources:` block. See
[troubleshooting.md](troubleshooting.md#what-each-profiling-source-needs).
With `--torch-mode cuda`, or `--torch-mode existing` on a venv that holds
CUDA PyTorch, only stages 1, 2, 3 and `Verify installation` run, and there is
no `Profiling sources:` block.

The setup prints the prefix as `Using hipDNN/ROCm prefix: ...`.

## Options

| Option | Description |
|---|---|
| `--torch-mode {rocm,cuda,cpu,existing,none}` | How to provide PyTorch. Default `rocm`. See [Torch modes](#torch-modes). |
| `--workspace WORKSPACE` | Root for the venv, the Python bytecode cache and the runtime caches. See [Workspace](#workspace). |
| `--no-editable` | Install dnn-benchmarking into the venv. The default links to the source tree. |
| `--torch-index-url URL` | pip index URL for PyTorch. |
| `--gpu-arch GPU_ARCH` | GPU architecture for the ROCm PyTorch nightly and the build. See [GPU architecture](#gpu-architecture). |
| `--rocm-prefix ROCM_PREFIX` | ROCm and hipDNN prefix for the binding and provider builds. This prefix has priority over the venv ROCm SDK. |
| `--reuse-artifacts` | Do not build hipDNN, the providers or the `hipdnn_frontend` bindings. Use what is installed in the selected prefix and venv. Setup fails if hipDNN is absent from the prefix, or if the venv has no `hipdnn_frontend` package. |
| `--clean` | Delete and create again the venv, and delete the hipDNN, provider and binding build directories. Not allowed with `--torch-mode existing`. |
| `--rocm-libraries-ref SHA` | Full 40-character `rocm-libraries` commit to fetch when `rocm-libraries/` is absent. Abbreviated SHAs and branch names are rejected. |
| `--cmake-arg NAME=VALUE` | Extra CMake define for the hipDNN and provider configure. Repeatable. See [Extra CMake defines](#extra-cmake-defines). |
| `-y`, `--yes` | Answer yes to every prompt. Required when stdin is not a terminal and a prompt would show. |

Setup asks for confirmation before a source build, before `--clean` deletes
a venv, and before it replaces a non-git `rocm-libraries/` directory.

## Workspace

Setup selects the workspace in this order:

1. `--workspace <path>`
2. `$DNN_BENCH_WORKSPACE`
3. `/workspace`, when it exists and is writable (Linux only)
4. `.workspace` in the dnn-benchmarking directory

```bash
python3 setup_env.py --workspace /tmp/dnn-bench
source /tmp/dnn-bench/.venv/bin/activate
```

On Linux the venv activation script exports these variables:

| Variable | Value |
|---|---|
| `ROCM_PATH` | The selected ROCm prefix. |
| `LD_LIBRARY_PATH` | `$ROCM_PATH/lib`, then the ROCm toolchain `lib` directory (rocm-sdk-devel in `rocm` torch mode), added at the start. |
| `DNN_BENCH_WORKSPACE` | The workspace. dnn-benchmarking puts MIOpen, comgr and XDG caches under it when those variables are not set. |
| `PYTHONPYCACHEPREFIX` | `<workspace>/pycache`. |

dnn-benchmarking finds the engine plugins in
`$ROCM_PATH/lib/hipdnn_plugins/engines/`. Give this directory to
`--plugin-path` only when you use a different plugin build.

## Torch modes

| Mode | PyTorch | hipDNN and providers | Use |
|---|---|---|---|
| `rocm` (default) | ROCm nightly for the detected architecture | Built against the ROCm SDK from the PyTorch wheels | Benchmarks on AMD GPUs. |
| `cuda` | CUDA PyTorch from PyPI, or `--torch-index-url` | Not built. No bindings, no amdsmi, no `ROCM_PATH` | `--runtime pytorch` on NVIDIA GPUs. |
| `cpu` | CPU-only PyTorch | Built from source and installed into `--rocm-prefix`, `$ROCM_PATH` or `/opt/rocm`. With `--reuse-artifacts`, uses the hipDNN installed there. | CI and `--validate pytorch` with a system ROCm. |
| `existing` | Keep the PyTorch in the venv | ROCm PyTorch: its SDK libraries. CUDA PyTorch: the `cuda` path. CPU PyTorch: the installed ROCm. | Reuse a venv. |
| `none` | Not installed | Same as `cpu` | hipDNN backend without PyTorch. |

A CPU-only PyTorch never enables `--runtime pytorch`. It is only for
`--validate pytorch`. `--runtime pytorch` needs a ROCm or CUDA build.

The modes `cpu`, `existing` (with CPU PyTorch) and `none` use
`--rocm-prefix`, `$ROCM_PATH` or `/opt/rocm` as the prefix. Without
`--reuse-artifacts` they build hipDNN and the providers from source and
install them into that prefix, over any hipDNN it already holds. To use the
hipDNN of a system ROCm, pass `--reuse-artifacts`:

```bash
python3 setup_env.py --torch-mode cpu --rocm-prefix /opt/rocm --reuse-artifacts
```

## CUDA hosts

```bash
python3 setup_env.py --torch-mode cuda --workspace .workspace
source .workspace/.venv/bin/activate
dnn-benchmark -g 'graphs/*.json' --runtime pytorch -o cuda_results.json
```

Only `--runtime pytorch` works on a CUDA host. Kernel timing uses
`torch.cuda` events (`timing.mode = "events"`). In the result file,
`gpu_arch` is `"unknown"`, `rocm_version` is `null`, and `cuda_version` and
`cudnn_version` are set. Use `dnn-benchmark compare` to compare the file
with a ROCm result.

## GPU architecture

On Linux, setup detects the architecture with `rocm_agent_enumerator` or
`rocminfo`. If detection fails, or the host has more than one GPU
architecture, give `--gpu-arch`:

| GPU | `--gpu-arch` |
|---|---|
| MI200, MI210, MI250 | `gfx90a` |
| MI300X, MI300A | `gfx942` |
| MI350 | `gfx950` |

On Windows the default is `gfx1151`.

## rocm-libraries source

`rocm-libraries/` is a git submodule. It holds the hipDNN and provider
sources.

- When `rocm-libraries/` is absent, setup does a sparse, blobless clone of
  the pinned submodule commit (`git rev-parse HEAD:rocm-libraries`). The clone
  contains only `cmake`, `shared/ctest`, `projects/hipdnn` and
  `dnn-providers`, not the full monorepo. `shared/ctest` carries the CMake
  file that declares hipDNN's test categories; the rest of `shared/` (Tensile
  and friends) stays out.
- Where no git metadata exists (Docker builds, source tarballs), give
  `--rocm-libraries-ref SHA`. Without it, setup uses the moving
  `.gitmodules` branch and shows a warning.
- Setup uses an existing checkout as it is. A checkout from
  `git submodule update --init` also works. Setup shows a warning when the
  checkout is not at the pinned commit. When that checkout is sparse and was
  made by an older setup, the root directories added since are added to its
  sparse set; a full checkout is left alone.

To build against a different commit, check it out directly:

```bash
git -C rocm-libraries fetch --depth 1 origin <ref>
git -C rocm-libraries checkout FETCH_HEAD
```

Setup builds only from `rocm-libraries/` in this directory. To benchmark a
revision you already have in another checkout, add it there as a worktree
(`rocm-libraries/` must be absent or empty):

```bash
git -C <your-checkout> worktree add <dnn-benchmarking>/rocm-libraries <rev>
```

Each run replaces the build and install in this checkout, so comparing two
revisions needs two `dnn-benchmarking` checkouts.

## Extra CMake defines

Setup configures hipDNN and the providers with fixed defaults.
`--cmake-arg NAME=VALUE` adds a define after the defaults, so it can override
a default. The option is repeatable.

Setup accepts `NAME=VALUE` and `-DNAME=VALUE`. It adds `-D` when absent.
Write the `-D` form with `=` (`--cmake-arg=-DFOO=ON`). With a space, argparse
reads `-DFOO=ON` as an option.

```bash
# Build without rocKE and its descriptor-backed engines
python3 setup_env.py \
  --cmake-arg HIPKERNELPROVIDER_ENABLE_ROCKE=OFF \
  --cmake-arg HIPDNN_ENABLE_KERNEL_INGESTOR=OFF

# The -D form needs '='
python3 setup_env.py --cmake-arg HIPDNN_ENABLE_SDPA=OFF --cmake-arg=-DCMAKE_BUILD_TYPE=Debug
```

On Linux, rocKE and its descriptor-backed engines are on by default
(`hipkernel:Gfx950AttentionDense` on gfx950). On Windows they are off. When
you turn an engine option off, setup still installs the plugin `.so`. Graphs
that only that engine supports then report `no engines applicable`.

## Rebuild from zero

```bash
python3 setup_env.py --clean -y
```

`--clean` deletes the venv (and its PyTorch) and the CMake build directories.

### Reusing a venv

Without `--clean`, setup keeps the venv and its PyTorch and builds
incrementally. When setup installs PyTorch it records the index URL and GPU
architecture in `<workspace>/.venv/dnn-bench-torch.json`. On a later run, an
explicit `--gpu-arch` or `--torch-index-url` that differs from the record
stops setup with an error that asks for `--clean`. Without them, a
`--torch-mode rocm` run builds for the recorded GPU architecture and installs
a missing rocm-sdk-devel from the recorded index. A venv without the record
(PyTorch installed another way) cannot be checked; setup shows a warning
instead. A different `--torch-mode` than the one in the venv also asks for
`--clean`, including `--torch-mode none` on a venv that has PyTorch.

The CMake build directories live in the checkout (`rocm-libraries/build`,
`rocm-libraries/projects/hipdnn/python/build`), so every workspace shares
them. Each records its full CMake configure argument list in
`dnn-bench-prefixes.json`: the install and toolchain prefixes, the GPU
architecture, the defaults setup passes and any `--cmake-arg` defines. A run
whose configure arguments differ (another `--workspace`, `--rocm-prefix` or
GPU architecture, other defines, or a setup_env.py with other defaults) wipes
that directory and rebuilds it from zero. So one workspace never builds with
another workspace's compiler or ROCm SDK, and a define that drops off the
configure line does not stay in the CMake cache.

## Manual install

To install the package in an existing environment:

```bash
pip install -e ".[test]"
```

This installs `numpy`, `psutil` and the test tools. It does not install
PyTorch, hipDNN or the plugins. PyTorch is not in `pyproject.toml`, because
the correct wheel depends on the target (ROCm, CUDA or CPU).

## Docker

`dockerfiles/build-dnn-benchmark-performance-image.sh` builds a runtime image
from the local checkout:

```bash
./dockerfiles/build-dnn-benchmark-performance-image.sh --device-target gfx942
./dockerfiles/build-dnn-benchmark-performance-image.sh --container-command podman --device-target gfx90a
```

| Option | Default | Description |
|---|---|---|
| `--container-command CMD` | `docker` | Container command. |
| `--device-target TARGET` | `gfx942` | ROCm PyTorch architecture: `gfx90a`, `gfx942`, `gfx950`, `gfx1151` or `gfx1152`. |
| `--tag TAG` | `dnn-benchmark-performance:<device-target>` | Image tag. |
| `-- ARGS...` | none | More arguments for the container build command. |

The script passes the pinned `rocm-libraries` commit to the build, because
the build context has no `.git`. The image runs `setup_env.py --torch-mode
rocm --no-editable -y` with the venv in `/opt/dnn-benchmark/.venv`. The
entry point is `dnn-benchmark`, so arguments go straight to the tool:

```bash
docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video \
  --group-add "$(getent group render | cut -d: -f3)" \
  dnn-benchmark-performance:gfx942 -g graphs/sample_conv_fwd.json
```

The image contains `rocprofv3` from the ROCm wheels and `perf`. It does not
contain `rocprof-compute`, so `--roofline` does not work in the image. See
[troubleshooting.md](troubleshooting.md#what-each-profiling-source-needs).

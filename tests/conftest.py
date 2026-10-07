# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Pytest fixtures for dnn-benchmarking tests."""

from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--profiling-strict",
        action="store_true",
        default=False,
        help=(
            "run profiling_strict tests that require profiler subprocesses "
            "to produce real artifacts, not just error/skip diagnostics"
        ),
    )
    parser.addoption(
        "--dnn-plugin-paths",
        action="store",
        default=None,
        help=(
            "Comma-separated hipDNN engine plugin directories for GPU tests. "
            "Each directory must exist and contain at least one .so file."
        ),
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--profiling-strict"):
        return

    skip_strict = pytest.mark.skip(
        reason="profiling_strict tests require --profiling-strict"
    )
    for item in items:
        if "profiling_strict" in item.keywords:
            item.add_marker(skip_strict)


def expected_timing_backend() -> str:
    """Return the GPU timing backend the executor selects on this host.

    Mirrors PyTorchCudaExecutor: ROCm torch with a usable hipdnn_frontend
    uses direct HIP events ("hip"); anything else (CUDA, or ROCm without the
    HIP bindings) uses torch.cuda events ("torch"). Lets GPU-generic tests
    assert the right backend without pinning to one platform.
    """
    from dnn_benchmarking.common import torch_support
    from dnn_benchmarking.execution.timing import is_hip_available

    rocm_hip = torch_support.is_rocm_build() and is_hip_available()
    return "hip" if rocm_hip else "torch"


def skip_if_no_gpu_torch() -> None:
    """Skip unless torch can drive a GPU (ROCm or CUDA), for generic tests."""
    try:
        import torch
    except ImportError:
        pytest.skip("PyTorch not available")

    if not torch.cuda.is_available():
        pytest.skip("PyTorch GPU not available")


def skip_if_no_cuda_torch() -> None:
    """Skip unless this is a CUDA (non-ROCm) torch build with a usable GPU."""
    try:
        import torch
    except ImportError:
        pytest.skip("PyTorch not available")

    if torch.version.cuda is None or torch.version.hip is not None:
        pytest.skip("CUDA PyTorch build required")

    if not torch.cuda.is_available():
        pytest.skip("PyTorch GPU not available")


@pytest.fixture
def sample_conv_fwd_json() -> Dict[str, Any]:
    """Minimal Conv Fwd JSON for testing (matches hipDNN serialization format)."""
    return {
        "name": "sample_conv_fwd_16x16x16x16_k16_3x3",
        "compute_data_type": "float",
        "io_data_type": "float",
        "intermediate_data_type": "float",
        "tensors": [
            {
                "uid": 0,
                "name": "output_y",
                "dims": [16, 16, 16, 16],
                "strides": [4096, 256, 16, 1],
                "data_type": "float",
                "virtual": False,
            },
            {
                "uid": 1,
                "name": "input_x",
                "dims": [16, 16, 16, 16],
                "strides": [4096, 256, 16, 1],
                "data_type": "float",
                "virtual": False,
            },
            {
                "uid": 2,
                "name": "weight",
                "dims": [16, 16, 3, 3],
                "strides": [144, 9, 3, 1],
                "data_type": "float",
                "virtual": False,
            },
        ],
        "nodes": [
            {
                "name": "conv_fprop_node",
                "type": "ConvolutionFwdAttributes",
                "compute_data_type": "unset",
                "inputs": {
                    "x_tensor_uid": 1,
                    "w_tensor_uid": 2,
                },
                "outputs": {"y_tensor_uid": 0},
                "parameters": {
                    "conv_mode": "CROSS_CORRELATION",
                    "pre_padding": [1, 1],
                    "post_padding": [1, 1],
                    "stride": [1, 1],
                    "dilation": [1, 1],
                },
            }
        ],
    }


def _valid_plugin_dir(path: Path) -> bool:
    return path.is_dir() and any(path.glob("*.so"))


def _parse_plugin_paths(raw_paths: str) -> List[Path]:
    paths = [Path(path.strip()) for path in raw_paths.split(",") if path.strip()]
    if not paths:
        raise pytest.UsageError("--dnn-plugin-paths requires at least one path")

    invalid_paths = [path for path in paths if not _valid_plugin_dir(path)]
    if invalid_paths:
        formatted_paths = ", ".join(str(path) for path in invalid_paths)
        raise pytest.UsageError(
            "--dnn-plugin-paths entries must be directories containing at "
            f"least one .so file: {formatted_paths}"
        )

    return paths


def _venv_rocm_sdk_plugin_dirs() -> list[Path]:
    """Return ROCm SDK wheel plugin directories from the active venv only."""
    import sys
    import sysconfig

    venv_root = Path(sys.prefix).resolve()
    dirs: list[Path] = []
    for key in ("purelib", "platlib"):
        value = sysconfig.get_path(key)
        if not value:
            continue
        site_dir = Path(value).resolve()
        if site_dir != venv_root and venv_root not in site_dir.parents:
            continue
        if not site_dir.is_dir():
            continue
        dirs.extend(
            child / "lib" / "hipdnn_plugins" / "engines"
            for child in site_dir.iterdir()
            # Multi-arch indexes ship a single arch-agnostic
            # "_rocm_sdk_libraries"; older per-arch indexes name it
            # "_rocm_sdk_libraries_<arch>". Match both, as setup_env does —
            # the suffixed-only check silently skipped every plugin-gated
            # integration test on a current wheel install.
            if child.is_dir()
            and (
                child.name == "_rocm_sdk_libraries"
                or child.name.startswith("_rocm_sdk_libraries_")
            )
        )
    return dirs


def _find_plugin_paths(pytestconfig) -> Optional[List[str]]:
    """Find hipDNN engine plugin directories.

    Returns explicitly configured plugin paths, then falls back to known build
    and system install locations. Returns None if no plugin directory is found.
    """
    configured_paths = pytestconfig.getoption("--dnn-plugin-paths", default=None)
    if configured_paths:
        return [str(path) for path in _parse_plugin_paths(configured_paths)]

    project_root = Path(__file__).parent.parent
    candidates = [
        # Worktree/superbuild: relative to dnn-benchmarking tool
        project_root.parent.parent.parent.parent
        / "dnn-providers"
        / "miopen-provider"
        / "build"
        / "lib"
        / "hipdnn_plugins"
        / "engines",
        *_venv_rocm_sdk_plugin_dirs(),
        # System install
        Path("/opt/rocm/lib/hipdnn_plugins/engines"),
    ]
    for path in candidates:
        if _valid_plugin_dir(path):
            return [str(path)]
    return None


@pytest.fixture
def plugin_paths(pytestconfig):
    """Get hipDNN engine plugin paths, or skip if none are found."""
    paths = _find_plugin_paths(pytestconfig)
    if paths is None:
        pytest.skip("No hipDNN engine plugin found")
    return paths


@pytest.fixture
def plugin_path_cli_args(plugin_paths):
    """Return CLI args for --plugin-path using the first resolved plugin path."""
    return ["--plugin-path", plugin_paths[0]]

# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""ROCm runtime discovery for pip-installed ROCm SDK/PyTorch wheels."""

from __future__ import annotations

import os
from pathlib import Path
from types import ModuleType
from typing import Optional


# Mirrors the preload order generated into ROCm PyTorch's torch/_rocm_init.py.
# Missing libraries are skipped so smaller ROCm SDK wheel selections can still
# run until they touch an unavailable provider/runtime component.
_ROCM_PRELOAD_ORDER = (
    "amd_comgr",
    "amdhip64",
    "rocprofiler-sdk",
    "rocprofiler-sdk-roctx",
    "roctracer64",
    "roctx64",
    "hiprtc",
    "hipblas",
    "hipfft",
    "hiprand",
    "hipsparse",
    "hipsparselt",
    "hipsolver",
    "rccl",
    "hipblaslt",
    "miopen",
    "hipdnn",
    "rocm_sysdeps_liblzma",
    "rocm-openblas",
    "rocm_smi64",
)

_INITIALIZED_PIP_ROCM = False


def _import_rocm_sdk() -> Optional[ModuleType]:
    try:
        import rocm_sdk  # type: ignore[import-not-found]
    except ImportError:
        return None
    return rocm_sdk


def _plugin_path_from_prefix(prefix: Path) -> Path:
    return prefix / "lib" / "hipdnn_plugins" / "engines"


def _hipdnn_library_path(rocm_sdk: ModuleType) -> Optional[Path]:
    try:
        paths = rocm_sdk.find_libraries("hipdnn")
    except (ModuleNotFoundError, FileNotFoundError):
        return None
    if not paths:
        return None
    return Path(paths[0])


def pip_rocm_plugin_path() -> Optional[Path]:
    """Return the hipDNN plugin directory from pip ROCm SDK wheels, if present."""
    rocm_sdk = _import_rocm_sdk()
    if rocm_sdk is None:
        return None

    hipdnn_library = _hipdnn_library_path(rocm_sdk)
    if hipdnn_library is None:
        return None

    plugin_path = hipdnn_library.parent / "hipdnn_plugins" / "engines"
    if not plugin_path.is_dir():
        return None
    return plugin_path


def default_hipdnn_plugin_paths() -> Optional[list[Path]]:
    """Return default hipDNN plugin paths.

    ``ROCM_PATH`` wins so users can point at an alternate ROCm/hipDNN install.
    Without ``ROCM_PATH``, fall back to the pip-installed ROCm SDK runtime used
    by ROCm PyTorch wheels.
    """
    rocm_path = os.environ.get("ROCM_PATH")
    if rocm_path:
        return [_plugin_path_from_prefix(Path(rocm_path))]

    plugin_path = pip_rocm_plugin_path()
    if plugin_path is None:
        return None
    return [plugin_path]


def _available_preload_shortnames(rocm_sdk: ModuleType) -> list[str]:
    available: list[str] = []
    for shortname in _ROCM_PRELOAD_ORDER:
        try:
            rocm_sdk.find_libraries(shortname)
        except (ModuleNotFoundError, FileNotFoundError):
            continue
        available.append(shortname)
    return available


def initialize_pip_rocm_runtime() -> bool:
    """Preload pip-installed ROCm SDK libraries when no ``ROCM_PATH`` is set.

    Returns True when initialization was attempted. ``ROCM_PATH`` deliberately
    disables this path: an explicit external ROCm install should control its own
    linker environment.
    """
    global _INITIALIZED_PIP_ROCM

    if os.environ.get("ROCM_PATH"):
        return False
    if _INITIALIZED_PIP_ROCM:
        return True

    rocm_sdk = _import_rocm_sdk()
    if rocm_sdk is None:
        return False

    preload_shortnames = _available_preload_shortnames(rocm_sdk)
    if not preload_shortnames:
        return False

    try:
        rocm_sdk.initialize_process(
            preload_shortnames=preload_shortnames,
            env_override=True,
        )
    except Exception as e:  # pragma: no cover - platform-specific ctypes failures.
        raise RuntimeError(
            f"Failed to initialize pip-installed ROCm runtime: {e}"
        ) from e

    _INITIALIZED_PIP_ROCM = True
    return True

# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""ROCm runtime discovery for pip-installed ROCm SDK/PyTorch wheels."""

from __future__ import annotations

import ctypes
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


def wheel_hipdnn_plugin_path() -> Optional[Path]:
    """Return the plugin directory from an installed hipdnn-runtime-<arch> wheel."""
    try:
        import hipdnn_runtime  # type: ignore[import-not-found]
    except ImportError:
        return None

    plugin_path = hipdnn_runtime.plugin_path()
    if not plugin_path.is_dir():
        return None
    return plugin_path


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

    ``HIPDNN_SDK`` selects a separately installed hipDNN while ROCm dependencies
    can remain in wheels or ``ROCM_PATH``. Otherwise prefer ``ROCM_PATH``, then
    a released runtime wheel, then the backend bundled in the ROCm SDK.
    """
    prefix = os.environ.get("HIPDNN_SDK") or os.environ.get("ROCM_PATH")
    if prefix:
        return [_plugin_path_from_prefix(Path(prefix))]

    plugin_path = wheel_hipdnn_plugin_path() or pip_rocm_plugin_path()
    if plugin_path is None:
        return None
    return [plugin_path]


def _preload_wheel_hipdnn() -> bool:
    """Load the hipdnn-runtime wheel's backend so it claims the SONAME.

    The ROCm SDK libraries wheel bundles its own ``libhipdnn_backend.so``. Both
    copies share a SONAME, so whichever is loaded first satisfies every later
    request -- and the SDK's copy tracks the nightly, not this release, which
    surfaces as an ``undefined symbol`` when the bindings look for a newer
    entry point. Loading ours first, globally, makes the released wheel win.
    """
    try:
        import hipdnn_runtime  # type: ignore[import-not-found]
    except ImportError:
        return False

    backend = hipdnn_runtime.backend_library()
    if not backend.is_file():
        return False
    ctypes.CDLL(str(backend), mode=ctypes.RTLD_GLOBAL)
    return True


def _available_preload_shortnames(rocm_sdk: ModuleType, skip: set[str]) -> list[str]:
    available: list[str] = []
    for shortname in _ROCM_PRELOAD_ORDER:
        if shortname in skip:
            continue
        try:
            rocm_sdk.find_libraries(shortname)
        except (ModuleNotFoundError, FileNotFoundError):
            continue
        available.append(shortname)
    return available


def initialize_pip_rocm_runtime() -> bool:
    """Initialize the selected hipDNN backend and its ROCm dependencies.

    ``HIPDNN_SDK`` owns the backend when set; SDK initialization must not load
    its competing hipDNN copy. Without that override, ``ROCM_PATH`` preserves
    the external installation's linker environment.
    """
    global _INITIALIZED_PIP_ROCM

    hipdnn_prefix = os.environ.get("HIPDNN_SDK")
    if not hipdnn_prefix and os.environ.get("ROCM_PATH"):
        return False
    if _INITIALIZED_PIP_ROCM:
        return True

    if hipdnn_prefix:
        prefix = Path(hipdnn_prefix)
        backend = (
            prefix / "bin" / "hipdnn_backend.dll"
            if os.name == "nt"
            else prefix / "lib" / "libhipdnn_backend.so"
        )
        if not backend.is_file():
            raise RuntimeError(f"HIPDNN_SDK contains no hipDNN backend: {backend}")
        rocm_sdk = _import_rocm_sdk()
        try:
            if rocm_sdk is not None:
                rocm_sdk.initialize_process(
                    preload_shortnames=_available_preload_shortnames(
                        rocm_sdk, {"hipdnn"}
                    ),
                    env_override=False,
                )
            ctypes.CDLL(str(backend), mode=ctypes.RTLD_GLOBAL)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load HIPDNN_SDK backend {backend}: {e}"
            ) from e
        _INITIALIZED_PIP_ROCM = True
        return True

    rocm_sdk = _import_rocm_sdk()
    if rocm_sdk is None:
        return False

    # A released hipdnn-runtime wheel owns hipDNN; keep the SDK's copy out of
    # the preload entirely so it cannot take the SONAME back.
    skip = {"hipdnn"} if _preload_wheel_hipdnn() else set()
    preload_shortnames = _available_preload_shortnames(rocm_sdk, skip)
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

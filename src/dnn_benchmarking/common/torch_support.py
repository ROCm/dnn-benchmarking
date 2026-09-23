# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""PyTorch capability checks.

These helpers only describe PyTorch's availability. They do not answer whether
HIP, hipDNN, or the host has a usable GPU through any non-PyTorch runtime.
"""

import functools


def module_available() -> bool:
    """Return True when the torch Python module can be imported.

    Broken installs commonly fail import with OSError/RuntimeError from
    missing shared libraries, not ImportError, so any import-time failure
    means "not available".
    """
    try:
        import torch  # noqa: F401
    except Exception:
        return False
    return True


def gpu_available() -> bool:
    """Return True when PyTorch's CUDA/ROCm facade can use a GPU."""
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


@functools.lru_cache(maxsize=1)
def gpu_usable() -> bool:
    """Return True when torch can allocate and run a kernel on the current GPU.

    ``torch.cuda.is_available()`` stays True on hosts where every kernel fails
    (for example ``hipErrorInvalidImage`` from a mismatched torch build), so
    this runs one tiny kernel and checks the result.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        return (torch.zeros(1, device="cuda") + 1).item() == 1.0
    except Exception:
        return False


def is_rocm_build() -> bool:
    """Return True when the installed torch is a ROCm/HIP build."""
    try:
        import torch

        return bool(getattr(torch.version, "hip", None))
    except Exception:
        return False


def is_cuda_build() -> bool:
    """Return True when the installed torch is a CUDA (non-ROCm) build."""
    try:
        import torch

        return bool(getattr(torch.version, "cuda", None)) and not bool(
            getattr(torch.version, "hip", None)
        )
    except Exception:
        return False

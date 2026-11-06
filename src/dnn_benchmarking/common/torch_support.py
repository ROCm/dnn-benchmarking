# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""PyTorch capability checks.

These helpers only describe PyTorch's availability. They do not answer whether
HIP, hipDNN, or the host has a usable GPU through any non-PyTorch runtime.
"""


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


def device_arch(device) -> str:
    """GPU architecture name of a torch device, without feature flags (``gfx950``)."""
    import torch

    return str(torch.cuda.get_device_properties(device).gcnArchName).split(":")[0]


def device_num_cus(device) -> int:
    """Compute-unit count of a torch device; kernel dispatchers size grids by it."""
    import torch

    return int(torch.cuda.get_device_properties(device).multi_processor_count)

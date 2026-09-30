# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Static machine and software metadata for the result ``environment`` block.

:func:`collect_environment_info` is called once at suite start, before
any progress output, so its single amdsmi notice cannot interleave with
a progress line. Every key is always present; unknown values are None,
except ``gpu_arch``, which is ``"unknown"``. torch is queried only when the
process has already imported it: importing it takes seconds.
"""

import importlib.util
import os
import platform
import re
import runpy
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from ..common import torch_support
from ._diagnostic import warn_once
from .arch import detect_gpu
from .gpu_smi import GpuSmiProbe, is_amdsmi_available


def _read_cpu_model() -> Optional[str]:
    """Parse the first ``model name`` line from ``/proc/cpuinfo``."""
    try:
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.partition(":")[2].strip() or None
    except OSError:
        pass
    return None


def _read_numa_nodes() -> Optional[int]:
    """Count ``node%d`` entries under ``/sys/devices/system/node/``."""
    try:
        return sum(
            1
            for child in Path("/sys/devices/system/node").iterdir()
            if re.fullmatch(r"node\d+", child.name)
        )
    except OSError:
        return None


def _read_total_ram_gb() -> Optional[float]:
    try:
        import psutil

        return round(psutil.virtual_memory().total / (1024.0**3), 2)
    except Exception:
        pass
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return round(int(line.split()[1]) / (1024.0**2), 2)
    except (OSError, ValueError, IndexError):
        pass
    return None


def _format_cudnn_version(raw: Optional[int]) -> Optional[str]:
    """Decode the packed int from ``torch.backends.cudnn.version()``.

    cuDNN 9+ packs ``major*10000 + minor*100 + patch``; earlier releases
    ``major*1000 + minor*100 + patch``. None for a missing/zero version.
    """
    if not raw:
        return None
    if raw >= 90000:
        major, minor, patch = raw // 10000, (raw % 10000) // 100, raw % 100
    else:
        major, minor, patch = raw // 1000, (raw % 1000) // 100, raw % 100
    return f"{major}.{minor}.{patch}"


def _torch_build_versions() -> Dict[str, Any]:
    """Globals of ``torch/version.py``, run on its own without importing torch."""
    try:
        spec = importlib.util.find_spec("torch")
        if spec is None or not spec.submodule_search_locations:
            return {}
        package_dir = Path(next(iter(spec.submodule_search_locations)))
        return runpy.run_path(str(package_dir / "version.py"))
    except Exception as e:
        warn_once("machine_info", f"torch version probe failed: {e}")
        return {}


def _torch_info() -> Dict[str, Any]:
    versions = _torch_build_versions()
    hip, cuda = versions.get("hip"), versions.get("cuda")
    info: Dict[str, Any] = {
        "torch_version": versions.get("__version__"),
        "rocm_version": hip,
        "cuda_version": None if hip else cuda,
        "cudnn_version": None,
    }
    torch = sys.modules.get("torch")
    if torch is None:
        return info
    try:
        if info["cuda_version"]:
            info["cudnn_version"] = _format_cudnn_version(
                torch.backends.cudnn.version()
            )
        if torch_support.gpu_available():
            props = torch.cuda.get_device_properties(torch.cuda.current_device())
            info["gpu_compute_units"] = props.multi_processor_count
    except Exception as e:
        warn_once("machine_info", f"torch probe failed: {e}")
    return info


def _hipdnn_version() -> Optional[str]:
    try:
        import hipdnn_frontend
    except ImportError:
        return None
    return getattr(hipdnn_frontend, "__version__", None)


def collect_environment_info() -> Dict[str, Any]:
    """Host, GPU and software versions for the result ``environment`` block.

    Never raises. ``end_of_run`` and ``selection_env`` are filled by the
    suite runner, not here.
    """
    amdsmi_available = is_amdsmi_available()
    if not amdsmi_available:
        warn_once(
            "amdsmi",
            "amdsmi not available; GPU clocks, power and throttle status "
            "will not be recorded",
        )
    gpu_arch, gpu_model = detect_gpu()
    info: Dict[str, Any] = {
        "hostname": platform.node() or None,
        "cpu_model": _read_cpu_model(),
        "cpu_count": os.cpu_count(),
        "numa_nodes": _read_numa_nodes(),
        "total_ram_gb": _read_total_ram_gb(),
        "kernel_version": platform.release() or None,
        "gpu_model": gpu_model,
        "gpu_arch": gpu_arch,
        "hipdnn_version": _hipdnn_version(),
        "python_version": platform.python_version(),
        "amdsmi_available": amdsmi_available,
    }
    info.update(_torch_info())
    for key, value in GpuSmiProbe().static_info().items():
        # amdsmi wins where it reports; torch fills e.g. CU count, which
        # some amdsmi builds omit from asic_info.
        if value is not None or key not in info:
            info[key] = value
    return info

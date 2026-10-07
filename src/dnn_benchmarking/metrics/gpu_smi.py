# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""GPU telemetry via the AMD SMI Python library.

amdsmi ships with system ROCm installs and ROCm SDK wheels under
``share/amd_smi/`` but is not a hard dependency. Availability is
resolved once per process and silently: every query returns ``None``
when amdsmi is missing. The one user-facing notice is emitted by
:func:`machine_info.collect_environment_info` at suite start, which also
records ``amdsmi_available`` in the environment block.

``GpuSmiProbe`` targets the GPU the workload runs on: device indices are
HIP (torch) logical indices, mapped to the amdsmi handle by PCI bus
address. When that address is unknown (torch missing or seeing no GPU) or
matches no amdsmi device, the HIP index is used as the amdsmi index; under
``HIP_VISIBLE_DEVICES`` / ``ROCR_VISIBLE_DEVICES`` remapping that can be
another physical GPU.
"""

import functools
import os
import sys
from typing import Any, Callable, Dict, Optional, Tuple

from ..common import torch_support

# Set when the HIP device list is a remapped subset of the physical GPUs.
_VISIBILITY_ENV = (
    "HIP_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
    "CUDA_VISIBLE_DEVICES",
    "GPU_DEVICE_ORDINAL",
)

_UNAVAILABLE_VALUES = {"", "N/A", "NA", "NONE", "NULL", "UNSUPPORTED"}


def _is_unavailable(value: Any) -> bool:
    if value is None:
        return True
    return isinstance(value, str) and value.strip().upper() in _UNAVAILABLE_VALUES


def _query(fn: Callable[[], Any], convert: Callable[[Any], Any] = float) -> Any:
    """``convert(fn())``, or None when the query fails or reports N/A.

    Platforms expose different SMI subsets (NOT_SUPPORTED errors, "N/A"
    payloads); a missing reading is expected, not a diagnostic.
    """
    try:
        value = fn()
        return None if _is_unavailable(value) else convert(value)
    except Exception:
        return None


@functools.lru_cache(maxsize=None)
def _amdsmi() -> Optional[Any]:
    """Imported and initialised ``amdsmi`` module, or None. Resolved once."""
    try:
        import amdsmi

        amdsmi.amdsmi_init()
    except Exception:
        return None
    return amdsmi


def is_amdsmi_available() -> bool:
    """True when amdsmi imports and initialises."""
    return _amdsmi() is not None


def _hip_device_bdf(device_index: Optional[int]) -> Optional[str]:
    """``dddd:bb:dd`` PCI address of a HIP device (None = current), via torch.

    Importing torch takes seconds, so it is imported only when device
    visibility is remapped. Otherwise the caller maps the HIP index directly.
    """
    # ponytail: assumes HIP and amdsmi both list unmasked GPUs in PCI order
    # (true on MI210/MI300 nodes); query the HIP runtime if that ever breaks.
    if "torch" not in sys.modules and not any(map(os.environ.get, _VISIBILITY_ENV)):
        return None
    if not torch_support.module_available() or not torch_support.gpu_available():
        return None
    try:
        import torch

        index = torch.cuda.current_device() if device_index is None else device_index
        p = torch.cuda.get_device_properties(index)
        return f"{p.pci_domain_id:04x}:{p.pci_bus_id:02x}:{p.pci_device_id:02x}"
    except Exception:
        return None


@functools.lru_cache(maxsize=None)
def _handle_for(device_index: Optional[int]) -> Optional[Any]:
    """amdsmi processor handle for a HIP device index (None = current)."""
    amdsmi = _amdsmi()
    if amdsmi is None:
        return None
    try:
        handles = amdsmi.amdsmi_get_processor_handles()
    except Exception:
        return None
    if not handles:
        return None
    bdf = _hip_device_bdf(device_index)
    if bdf is not None:
        for handle in handles:
            # amdsmi reports "dddd:bb:dd.f"; torch exposes no function number.
            smi_bdf = _query(lambda: amdsmi.amdsmi_get_gpu_device_bdf(handle), str)
            if smi_bdf is not None and smi_bdf.lower().split(".")[0] == bdf:
                return handle
    index = device_index or 0
    return handles[index] if index < len(handles) else None


class GpuSmiProbe:
    """amdsmi queries for one GPU (HIP device index; None = current device)."""

    def __init__(self, device_index: Optional[int] = None) -> None:
        self._amdsmi = _amdsmi()
        self._handle = _handle_for(device_index) if self._amdsmi is not None else None

    def clocks(self) -> Optional[Dict[str, Any]]:
        """Current clocks, power, hotspot temperature and throttle status.

        Keys: ``sclk_mhz``, ``mclk_mhz``, ``power_w``, ``temp_hotspot_c``,
        ``throttle_status``; a value is None when the platform does not
        report it. Returns None when amdsmi or the device is unavailable.
        """
        if self._handle is None:
            return None
        smi, h = self._amdsmi, self._handle

        def power() -> Any:
            info = smi.amdsmi_get_power_info(h)
            avg = info.get("average_socket_power")
            return info.get("current_socket_power") if _is_unavailable(avg) else avg

        return {
            "sclk_mhz": _query(
                lambda: smi.amdsmi_get_clock_info(h, smi.AmdSmiClkType.GFX)["clk"]
            ),
            "mclk_mhz": _query(
                lambda: smi.amdsmi_get_clock_info(h, smi.AmdSmiClkType.MEM)["clk"]
            ),
            "power_w": _query(power),
            "temp_hotspot_c": _query(
                lambda: smi.amdsmi_get_temp_metric(
                    h,
                    smi.AmdSmiTemperatureType.HOTSPOT,
                    smi.AmdSmiTemperatureMetric.CURRENT,
                )
            ),
            "throttle_status": _query(
                lambda: smi.amdsmi_get_gpu_metrics_info(h)["throttle_status"], int
            ),
        }

    def snapshot(self) -> Dict[str, Optional[float]]:
        """VRAM usage in MB: ``vram_used_mb``, ``vram_total_mb`` (None if unknown)."""
        snap: Dict[str, Optional[float]] = {"vram_used_mb": None, "vram_total_mb": None}
        if self._handle is None:
            return snap
        vram = _query(
            lambda: self._amdsmi.amdsmi_get_gpu_vram_usage(self._handle), dict
        )
        if vram:
            snap["vram_used_mb"] = _query(lambda: vram["vram_used"])
            snap["vram_total_mb"] = _query(lambda: vram["vram_total"])
        return snap

    def identity(self) -> Tuple[Optional[str], Optional[str]]:
        """``(target_graphics_version, market_name)``; None where unreported.

        Older amdsmi builds have no ``target_graphics_version``.
        """
        if self._handle is None:
            return None, None
        asic = _query(lambda: self._amdsmi.amdsmi_get_gpu_asic_info(self._handle), dict)
        if not asic:
            return None, None
        return (
            _query(lambda: asic["target_graphics_version"], str),
            _query(lambda: asic["market_name"], str),
        )

    def static_info(self) -> Dict[str, Any]:
        """One-time static GPU facts for the environment block.

        Keys: ``gpu_compute_units``, ``gpu_hbm_gb``, ``gpu_pcie_link``,
        ``amdgpu_driver_version``, ``gpu_power_cap_w``,
        ``gpu_max_sclk_mhz``, ``gpu_compute_partition``; None when unknown.
        """
        info: Dict[str, Any] = {
            "gpu_compute_units": None,
            "gpu_hbm_gb": None,
            "gpu_pcie_link": None,
            "amdgpu_driver_version": None,
            "gpu_power_cap_w": None,
            "gpu_max_sclk_mhz": None,
            "gpu_compute_partition": None,
        }
        if self._handle is None:
            return info
        smi, h = self._amdsmi, self._handle

        def pcie_link() -> Any:
            # pcie_speed is the per-lane rate in MT/s (16000 = PCIe gen4).
            metric = smi.amdsmi_get_pcie_info(h)["pcie_metric"]
            speed, width = metric.get("pcie_speed"), metric.get("pcie_width")
            if _is_unavailable(speed) or _is_unavailable(width):
                return None
            return f"{float(speed) / 1000:g} GT/s x{width}"

        def compute_units() -> Any:
            asic = smi.amdsmi_get_gpu_asic_info(h)
            return asic.get("num_of_compute_units") or asic.get("num_compute_units")

        def hbm_mb() -> Any:
            vram = smi.amdsmi_get_gpu_vram_info(h)
            return vram.get("vram_size") or vram.get("vram_size_mb")

        def driver() -> Any:
            d = smi.amdsmi_get_gpu_driver_info(h)
            return d.get("driver_version") or d.get("driver_name")

        info["gpu_compute_units"] = _query(compute_units, int)
        info["gpu_hbm_gb"] = _query(hbm_mb, lambda mb: round(float(mb) / 1024.0, 2))
        info["gpu_pcie_link"] = _query(pcie_link, str)
        info["amdgpu_driver_version"] = _query(driver, str)
        # amdsmi reports the power cap in microwatts.
        info["gpu_power_cap_w"] = _query(
            lambda: smi.amdsmi_get_power_cap_info(h)["power_cap"],
            lambda uw: float(uw) / 1e6,
        )
        info["gpu_max_sclk_mhz"] = _query(
            lambda: smi.amdsmi_get_clock_info(h, smi.AmdSmiClkType.GFX)["max_clk"]
        )
        info["gpu_compute_partition"] = _query(
            lambda: smi.amdsmi_get_gpu_compute_partition(h), str
        )
        return info

# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""GPU telemetry probe using the AMD SMI Python library.

The amdsmi library ships with system ROCm installs and ROCm SDK wheels under
``share/amd_smi/``. ``setup_env.py`` installs the bindings when it finds a usable
source tree, but they are not a hard dependency — ``GpuSmiProbe.snapshot()``
returns a stable-shape dict whose values are ``None`` whenever the library,
init, or a per-metric query fails.

Two collaborators:

* :class:`AmdsmiSession` owns the amdsmi import, the one-time
  ``amdsmi_init()`` call, and processor-handle resolution. A
  module-level singleton (:func:`default_session`) is used by default
  so the global init only fires once per process.
* :class:`GpuSmiProbe` issues telemetry queries through a session.
  Tests can pass a fake session to bypass the global state entirely.

The split keeps the per-call telemetry methods on the probe focused on
"call this amdsmi function and stash the result", and concentrates the
fragile import / init / handle-lookup logic in one testable place.
"""

from typing import Any, Dict, Optional

from ._diagnostic import warn_once

_SNAPSHOT_KEYS = (
    "vram_used_mb",
    "vram_total_mb",
    "power_w",
    "sclk_mhz",
    "mclk_mhz",
    "temp_edge_c",
    "temp_hotspot_c",
    "gpu_utilization_pct",
    "memory_utilization_pct",
    "throttle_status",
)


def _empty_snapshot() -> Dict[str, Optional[Any]]:
    return {k: None for k in _SNAPSHOT_KEYS}


_UNAVAILABLE_VALUES = {"", "N/A", "NA", "NONE", "NULL", "UNSUPPORTED"}


def _is_unavailable(value: Any) -> bool:
    if value is None:
        return True
    return isinstance(value, str) and value.strip().upper() in _UNAVAILABLE_VALUES


def _first_available(*values: Any) -> Any:
    for value in values:
        if not _is_unavailable(value):
            return value
    return None


def _optional_float(value: Any) -> Optional[float]:
    if _is_unavailable(value):
        return None
    return float(value)


def _optional_int(value: Any) -> Optional[int]:
    if _is_unavailable(value):
        return None
    return int(value)


def _is_not_supported_error(e: Exception) -> bool:
    text = str(e)
    return "AMDSMI_STATUS_NOT_SUPPORTED" in text or "Feature not supported" in text


def _warn_optional_query_failure(
    amdsmi: Any,
    reason: str,
    e: Exception,
) -> None:
    # Some platforms expose a subset of SMI counters and report unsupported
    # values either as "N/A" payloads or AMDSMI_STATUS_NOT_SUPPORTED. These are
    # expected missing telemetry, not actionable benchmark diagnostics.
    if isinstance(e, amdsmi.AmdSmiException) and _is_not_supported_error(e):
        return
    warn_once("amdsmi", f"{reason} failed: {e}")


def is_amdsmi_available() -> bool:
    """Return True if amdsmi can be imported."""
    try:
        import amdsmi  # noqa: F401

        return True
    except ImportError:
        return False


class AmdsmiSession:
    """Process-wide amdsmi lifecycle: import, init, processor handles.

    Designed as a thin singleton-style collaborator for
    :class:`GpuSmiProbe`. The default instance is reached via
    :func:`default_session`; tests construct their own to exercise
    init / handle-resolution failures without monkeypatching globals.

    All public methods degrade gracefully — ``module()`` returns
    ``None`` when amdsmi can't be imported, and ``handle()`` returns
    ``None`` for any failure in init or processor-handle lookup. A
    one-shot diagnostic warning surfaces each failure mode.
    """

    def __init__(self) -> None:
        self._initialised = False
        # Cache resolved handles by device index so repeated probes
        # reuse the same object rather than re-querying amdsmi.
        self._handles: Dict[int, Any] = {}

    def module(self) -> Optional[Any]:
        """Return the imported ``amdsmi`` module, or ``None``."""
        try:
            import amdsmi
        except ImportError:
            warn_once("amdsmi", "module not installed; GPU snapshot disabled")
            return None
        return amdsmi

    def handle(self, device_index: int) -> Optional[Any]:
        """Return the amdsmi processor handle for ``device_index``.

        Lazily runs ``amdsmi_init`` on first use. Returns ``None`` if
        amdsmi is missing, init fails, the handle list is unavailable,
        or the device index is out of range.
        """
        if device_index in self._handles:
            return self._handles[device_index]

        amdsmi = self.module()
        if amdsmi is None:
            return None

        if not self._initialised:
            try:
                amdsmi.amdsmi_init()
                self._initialised = True
            except amdsmi.AmdSmiException as e:
                warn_once("amdsmi", f"init failed: {e}")
                return None

        try:
            handles = amdsmi.amdsmi_get_processor_handles()
        except amdsmi.AmdSmiException as e:
            warn_once("amdsmi", f"get_processor_handles failed: {e}")
            return None

        if not handles or device_index >= len(handles):
            warn_once(
                "amdsmi",
                f"device index {device_index} out of range "
                f"({len(handles)} handles)",
            )
            return None

        self._handles[device_index] = handles[device_index]
        return self._handles[device_index]


_default_session: Optional[AmdsmiSession] = None


def default_session() -> AmdsmiSession:
    """Return the process-wide default :class:`AmdsmiSession` instance."""
    global _default_session
    if _default_session is None:
        _default_session = AmdsmiSession()
    return _default_session


def _reset_default_session_for_tests() -> None:
    """Test-only: drop the module singleton so init / handle state is fresh."""
    global _default_session
    _default_session = None


class GpuSmiProbe:
    """Stateful amdsmi probe targeting a single GPU.

    The amdsmi lifecycle (import, ``amdsmi_init``, processor handles)
    lives in :class:`AmdsmiSession`. Pass a custom session to inject a
    fake for tests; otherwise the module singleton is used and the
    global ``amdsmi_init`` fires at most once per process.
    """

    def __init__(
        self,
        device_index: int = 0,
        session: Optional[AmdsmiSession] = None,
    ) -> None:
        self._device_index = device_index
        self._session = session if session is not None else default_session()

    def _amdsmi_handle(self) -> Optional[Any]:
        """Resolve (and cache) the amdsmi handle for this probe's device."""
        return self._session.handle(self._device_index)

    def snapshot(self) -> Dict[str, Optional[Any]]:
        """Return a single-shot snapshot of GPU telemetry.

        Every key in :data:`_SNAPSHOT_KEYS` is present in the returned
        dict; values are ``None`` when the underlying query fails or
        amdsmi is unavailable. Failures emit a deduplicated warning via
        :func:`warn_once`.
        """
        snap = _empty_snapshot()
        handle = self._amdsmi_handle()
        if handle is None:
            return snap
        amdsmi = self._session.module()
        if amdsmi is None:
            return snap

        # VRAM usage
        try:
            vram = amdsmi.amdsmi_get_gpu_vram_usage(handle)
            # amdsmi reports MB already
            snap["vram_used_mb"] = _optional_float(vram.get("vram_used"))
            snap["vram_total_mb"] = _optional_float(vram.get("vram_total"))
        except (amdsmi.AmdSmiException, KeyError, TypeError, ValueError) as e:
            _warn_optional_query_failure(amdsmi, "vram_usage", e)

        # Power
        try:
            power = amdsmi.amdsmi_get_power_info(handle)
            socket_w = _first_available(
                power.get("average_socket_power"),
                power.get("current_socket_power"),
            )
            snap["power_w"] = _optional_float(socket_w)
        except (amdsmi.AmdSmiException, KeyError, TypeError, ValueError) as e:
            _warn_optional_query_failure(amdsmi, "power_info", e)

        # Clocks (GFX = sclk, MEM = mclk)
        try:
            sclk = amdsmi.amdsmi_get_clock_info(handle, amdsmi.AmdSmiClkType.GFX)
            snap["sclk_mhz"] = _optional_float(sclk.get("clk"))
        except (amdsmi.AmdSmiException, KeyError, TypeError, ValueError) as e:
            _warn_optional_query_failure(amdsmi, "clock_info GFX", e)

        try:
            mclk = amdsmi.amdsmi_get_clock_info(handle, amdsmi.AmdSmiClkType.MEM)
            snap["mclk_mhz"] = _optional_float(mclk.get("clk"))
        except (amdsmi.AmdSmiException, KeyError, TypeError, ValueError) as e:
            _warn_optional_query_failure(amdsmi, "clock_info MEM", e)

        # Temperatures
        try:
            edge = amdsmi.amdsmi_get_temp_metric(
                handle,
                amdsmi.AmdSmiTemperatureType.EDGE,
                amdsmi.AmdSmiTemperatureMetric.CURRENT,
            )
            snap["temp_edge_c"] = _optional_float(edge)
        except (amdsmi.AmdSmiException, KeyError, TypeError, ValueError) as e:
            _warn_optional_query_failure(amdsmi, "temp EDGE", e)

        try:
            hot = amdsmi.amdsmi_get_temp_metric(
                handle,
                amdsmi.AmdSmiTemperatureType.HOTSPOT,
                amdsmi.AmdSmiTemperatureMetric.CURRENT,
            )
            snap["temp_hotspot_c"] = _optional_float(hot)
        except (amdsmi.AmdSmiException, KeyError, TypeError, ValueError) as e:
            _warn_optional_query_failure(amdsmi, "temp HOTSPOT", e)

        # Utilisation + throttle status from gpu_metrics
        try:
            metrics = amdsmi.amdsmi_get_gpu_metrics_info(handle)
            gpu_util = metrics.get("average_gfx_activity")
            mem_util = metrics.get("average_umc_activity")
            throttle = metrics.get("throttle_status")
            snap["gpu_utilization_pct"] = _optional_float(gpu_util)
            snap["memory_utilization_pct"] = _optional_float(mem_util)
            snap["throttle_status"] = _optional_int(throttle)
        except (amdsmi.AmdSmiException, KeyError, TypeError, ValueError) as e:
            _warn_optional_query_failure(amdsmi, "gpu_metrics_info", e)

        return snap

    def static_info(self) -> Dict[str, Optional[Any]]:
        """Return one-time static info: CUs, HBM size, PCIe link, driver.

        Used by :func:`machine_info.collect_machine_info`. Stable-shape
        dict; missing values are ``None``.
        """
        info: Dict[str, Optional[Any]] = {
            "gpu_compute_units": None,
            "gpu_hbm_gb": None,
            "gpu_pcie_link": None,
            "amdgpu_driver_version": None,
        }
        handle = self._amdsmi_handle()
        if handle is None:
            return info
        amdsmi = self._session.module()
        if amdsmi is None:
            return info

        try:
            asic = amdsmi.amdsmi_get_gpu_asic_info(handle)
            cus = asic.get("num_of_compute_units") or asic.get("num_compute_units")
            if cus is not None:
                info["gpu_compute_units"] = int(cus)
        except (amdsmi.AmdSmiException, KeyError, TypeError, ValueError) as e:
            warn_once("amdsmi", f"asic_info failed: {e}")

        try:
            vram = amdsmi.amdsmi_get_gpu_vram_info(handle)
            size_mb = vram.get("vram_size") or vram.get("vram_size_mb")
            if size_mb is not None:
                info["gpu_hbm_gb"] = round(float(size_mb) / 1024.0, 2)
        except (amdsmi.AmdSmiException, KeyError, TypeError, ValueError) as e:
            warn_once("amdsmi", f"vram_info failed: {e}")

        try:
            pcie = amdsmi.amdsmi_get_pcie_info(handle)
            metric = pcie.get("pcie_metric") or {}
            gen = metric.get("pcie_speed") or pcie.get("pcie_speed")
            width = metric.get("pcie_width") or pcie.get("pcie_lanes")
            if gen is not None and width is not None:
                info["gpu_pcie_link"] = f"gen{gen} x{width}"
        except (amdsmi.AmdSmiException, KeyError, TypeError, ValueError) as e:
            warn_once("amdsmi", f"pcie_info failed: {e}")

        try:
            driver = amdsmi.amdsmi_get_gpu_driver_info(handle)
            ver = driver.get("driver_version") or driver.get("driver_name")
            if ver:
                info["amdgpu_driver_version"] = str(ver)
        except (
            AttributeError,
            amdsmi.AmdSmiException,
            KeyError,
            TypeError,
            ValueError,
        ) as e:
            # AttributeError caught because amdsmi_get_gpu_driver_info may
            # not exist in older amdsmi versions.
            warn_once("amdsmi", f"driver_info failed: {e}")

        return info

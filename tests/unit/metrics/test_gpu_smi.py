# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for metrics.gpu_smi.

amdsmi is optional, so a fake module stands in for it. The module-level
resolution caches are cleared around every test.
"""

import types

import pytest

from dnn_benchmarking.metrics import gpu_smi


@pytest.fixture(autouse=True)
def _fresh_caches():
    gpu_smi._amdsmi.cache_clear()
    gpu_smi._handle_for.cache_clear()
    yield
    gpu_smi._amdsmi.cache_clear()
    gpu_smi._handle_for.cache_clear()


class _NotSupported(Exception):
    pass


def _fake_amdsmi(handles=("h0",), bdfs=None, fail=()):
    """Stand-in amdsmi. ``fail`` names queries that raise NOT_SUPPORTED."""
    bdfs = bdfs or {}
    mod = types.SimpleNamespace()
    mod.AmdSmiClkType = types.SimpleNamespace(GFX="gfx", MEM="mem")
    mod.AmdSmiTemperatureType = types.SimpleNamespace(HOTSPOT="hotspot")
    mod.AmdSmiTemperatureMetric = types.SimpleNamespace(CURRENT="cur")

    def q(name, value):
        def fn(*_a):
            if name in fail:
                raise _NotSupported("AMDSMI_STATUS_NOT_SUPPORTED")
            return value(*_a) if callable(value) else value

        setattr(mod, name, fn)

    q("amdsmi_init", None)
    q("amdsmi_get_processor_handles", list(handles))
    q("amdsmi_get_gpu_device_bdf", lambda h: bdfs[h])
    q(
        "amdsmi_get_clock_info",
        lambda h, t: {"clk": 1700 if t == "gfx" else 1600, "max_clk": 1700},
    )
    # N/A average must fall back to the current reading.
    q(
        "amdsmi_get_power_info",
        {"average_socket_power": "N/A", "current_socket_power": 250},
    )
    q("amdsmi_get_temp_metric", 65)
    q("amdsmi_get_gpu_metrics_info", {"throttle_status": False})
    q("amdsmi_get_gpu_vram_usage", {"vram_used": 1024, "vram_total": 65536})
    q("amdsmi_get_gpu_asic_info", {"num_of_compute_units": 104})
    q("amdsmi_get_gpu_vram_info", {"vram_size": 65536})
    q("amdsmi_get_pcie_info", {"pcie_metric": {"pcie_speed": 16000, "pcie_width": 16}})
    q("amdsmi_get_gpu_driver_info", {"driver_version": "6.8.5"})
    q("amdsmi_get_power_cap_info", {"power_cap": 300_000_000})
    q("amdsmi_get_gpu_compute_partition", "SPX")
    return mod


def _install(monkeypatch, fake, hip_bdf=None):
    monkeypatch.setattr(gpu_smi, "_amdsmi", _cached(lambda: fake))
    monkeypatch.setattr(gpu_smi, "_hip_device_bdf", lambda _i: hip_bdf)


def _cached(fn):
    import functools

    return functools.lru_cache(maxsize=None)(fn)


class TestClocks:
    def test_reports_clock_power_temp_throttle(self, monkeypatch):
        _install(monkeypatch, _fake_amdsmi())
        assert gpu_smi.GpuSmiProbe().clocks() == {
            "sclk_mhz": 1700.0,
            "mclk_mhz": 1600.0,
            "power_w": 250.0,
            "temp_hotspot_c": 65.0,
            "throttle_status": 0,
        }

    @pytest.mark.parametrize("raw, expected", [(4, 4), ("N/A", None)])
    def test_throttle_status_value(self, monkeypatch, raw, expected):
        """Nonzero throttle_status puts the 'throttled' warning on the row."""
        fake = _fake_amdsmi()
        fake.amdsmi_get_gpu_metrics_info = lambda h: {"throttle_status": raw}
        _install(monkeypatch, fake)
        assert gpu_smi.GpuSmiProbe().clocks()["throttle_status"] == expected

    def test_unsupported_readings_are_none_not_errors(self, monkeypatch, capsys):
        _install(monkeypatch, _fake_amdsmi(fail={"amdsmi_get_temp_metric"}))
        clocks = gpu_smi.GpuSmiProbe().clocks()
        assert clocks["temp_hotspot_c"] is None
        assert clocks["sclk_mhz"] == 1700.0
        assert capsys.readouterr().err == ""

    def test_none_without_amdsmi_and_silent(self, monkeypatch, capsys):
        monkeypatch.setattr(gpu_smi, "_amdsmi", _cached(lambda: None))
        probe = gpu_smi.GpuSmiProbe()
        assert probe.clocks() is None
        assert probe.snapshot() == {"vram_used_mb": None, "vram_total_mb": None}
        # Missing amdsmi is reported once at startup, never per probe.
        assert capsys.readouterr().err == ""


class TestDeviceMapping:
    def test_current_hip_device_maps_by_pci_address(self, monkeypatch):
        """HIP_VISIBLE_DEVICES remaps HIP index 0 to any physical GPU; the
        probe must follow the PCI address, not amdsmi's handle order."""
        fake = _fake_amdsmi(
            handles=("h0", "h1", "h2"),
            bdfs={"h0": "0000:03:00.0", "h1": "0000:09:00.0", "h2": "0000:0c:00.0"},
        )
        _install(monkeypatch, fake, hip_bdf="0000:09:00")
        assert gpu_smi.GpuSmiProbe()._handle == "h1"

    def test_falls_back_to_index_when_address_unknown(self, monkeypatch):
        _install(monkeypatch, _fake_amdsmi(handles=("h0", "h1")))
        assert gpu_smi.GpuSmiProbe()._handle == "h0"
        assert gpu_smi.GpuSmiProbe(1)._handle == "h1"
        assert gpu_smi.GpuSmiProbe(5).clocks() is None


def test_snapshot_reports_vram_only(monkeypatch):
    _install(monkeypatch, _fake_amdsmi())
    assert gpu_smi.GpuSmiProbe().snapshot() == {
        "vram_used_mb": 1024.0,
        "vram_total_mb": 65536.0,
    }


def test_static_info_units(monkeypatch):
    _install(monkeypatch, _fake_amdsmi())
    assert gpu_smi.GpuSmiProbe().static_info() == {
        "gpu_compute_units": 104,
        "gpu_hbm_gb": 64.0,
        "gpu_pcie_link": "16 GT/s x16",
        "amdgpu_driver_version": "6.8.5",
        "gpu_power_cap_w": 300.0,  # amdsmi reports microwatts
        "gpu_max_sclk_mhz": 1700.0,
        "gpu_compute_partition": "SPX",
    }


def test_amdsmi_init_failure_means_unavailable(monkeypatch):
    import sys

    broken = types.ModuleType("amdsmi")

    def boom():
        raise RuntimeError("init failed")

    broken.amdsmi_init = boom
    monkeypatch.setitem(sys.modules, "amdsmi", broken)
    assert gpu_smi.is_amdsmi_available() is False

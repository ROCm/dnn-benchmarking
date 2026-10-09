# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for metrics.machine_info (the result ``environment`` block)."""

import importlib.util
import sys
from unittest.mock import MagicMock, mock_open, patch

import pytest

from dnn_benchmarking.metrics import arch, gpu_smi, machine_info
from dnn_benchmarking.metrics._diagnostic import reset as _reset_warns

# Contract: result JSON v2 environment keys (minus selection_env,
# which the suite runner fills).
_ENVIRONMENT_KEYS = {
    "hostname",
    "cpu_model",
    "cpu_count",
    "numa_nodes",
    "total_ram_gb",
    "kernel_version",
    "gpu_model",
    "gpu_arch",
    "gpu_compute_units",
    "gpu_hbm_gb",
    "gpu_pcie_link",
    "amdgpu_driver_version",
    "gpu_power_cap_w",
    "gpu_max_sclk_mhz",
    "gpu_compute_partition",
    "rocm_version",
    "cuda_version",
    "cudnn_version",
    "hipdnn_version",
    "python_version",
    "torch_version",
    "amdsmi_available",
}


@pytest.fixture(autouse=True)
def _reset_warn_state():
    _reset_warns()
    yield
    _reset_warns()


class TestReadCpuModel:
    def test_returns_first_model_name(self):
        cpuinfo = (
            "processor\t: 0\n"
            "vendor_id\t: AuthenticAMD\n"
            "model name\t: AMD EPYC 9654 96-Core Processor\n"
            "cache size\t: 1024 KB\n"
        )
        with patch("builtins.open", mock_open(read_data=cpuinfo)):
            assert machine_info._read_cpu_model() == "AMD EPYC 9654 96-Core Processor"

    def test_returns_none_when_file_missing(self):
        with patch("builtins.open", side_effect=OSError("no such file")):
            assert machine_info._read_cpu_model() is None


class TestReadTotalRam:
    def test_psutil_bytes_to_gib(self, monkeypatch):
        import psutil

        total = type("VM", (), {"total": 48 * 1024**3})
        monkeypatch.setattr(psutil, "virtual_memory", lambda: total)
        assert machine_info._read_total_ram_gb() == 48.0

    def test_meminfo_fallback_kib_to_gib(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "psutil", None)  # import fails
        meminfo = "MemTotal:       50331648 kB\nMemFree:        1024 kB\n"
        with patch("builtins.open", mock_open(read_data=meminfo)):
            assert machine_info._read_total_ram_gb() == 48.0


@pytest.mark.parametrize(
    "raw, expected",
    [(None, None), (0, None), (8902, "8.9.2"), (90100, "9.1.0")],
)
def test_cudnn_version_decoding_across_packing_schemes(raw, expected):
    assert machine_info._format_cudnn_version(raw) == expected


@pytest.mark.parametrize(
    "hip, cuda, rocm_version, cuda_version",
    [("7.0.1", None, "7.0.1", None), (None, "12.4", None, "12.4")],
)
def test_torch_versions_come_from_version_py_without_importing_torch(
    tmp_path, monkeypatch, hip, cuda, rocm_version, cuda_version
):
    package = tmp_path / "torch"
    package.mkdir()
    (package / "__init__.py").write_text("raise ImportError('torch was imported')\n")
    (package / "version.py").write_text(
        f"__version__ = '2.9.0'\nhip = {hip!r}\ncuda = {cuda!r}\n"
    )
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    monkeypatch.syspath_prepend(str(tmp_path))

    info = machine_info._torch_info()

    assert info["torch_version"] == "2.9.0"
    assert info["rocm_version"] == rocm_version
    assert info["cuda_version"] == cuda_version


class TestCollectEnvironmentInfo:
    def test_has_exactly_the_contract_keys(self):
        with patch.object(machine_info, "is_amdsmi_available", return_value=False):
            info = machine_info.collect_environment_info()
        assert set(info) == _ENVIRONMENT_KEYS

    def test_gpu_arch_and_model_come_from_detect_gpu(self):
        with (
            patch.object(machine_info, "is_amdsmi_available", return_value=False),
            patch.object(machine_info, "detect_gpu", return_value=("gfx942", "X")),
        ):
            info = machine_info.collect_environment_info()
        assert (info["gpu_arch"], info["gpu_model"]) == ("gfx942", "X")

    def test_unloaded_hipdnn_is_not_imported(self, tmp_path, monkeypatch, capsys):
        package = tmp_path / "hipdnn_frontend"
        package.mkdir()
        (package / "__init__.py").write_text("raise RuntimeError('no HIP device')\n")
        monkeypatch.delitem(sys.modules, "hipdnn_frontend", raising=False)
        monkeypatch.syspath_prepend(str(tmp_path))
        with patch.object(machine_info, "is_amdsmi_available", return_value=False):
            info = machine_info.collect_environment_info()
        assert info["hipdnn_version"] is None
        assert "hipdnn_frontend" not in sys.modules
        assert "no HIP device" not in capsys.readouterr().err

    def test_loaded_hipdnn_version_is_reported(self, monkeypatch):
        module = type(sys)("hipdnn_frontend")
        module.__version__ = "test-version"
        monkeypatch.setitem(sys.modules, "hipdnn_frontend", module)
        with patch.object(machine_info, "is_amdsmi_available", return_value=False):
            info = machine_info.collect_environment_info()
        assert info["hipdnn_version"] == "test-version"

    def test_missing_amdsmi_is_recorded_and_warned_once(self, capsys):
        with patch.object(machine_info, "is_amdsmi_available", return_value=False):
            first = machine_info.collect_environment_info()
            machine_info.collect_environment_info()
        assert first["amdsmi_available"] is False
        assert capsys.readouterr().err.count("amdsmi not available") == 1

    def test_does_not_import_torch(self, monkeypatch):
        """Importing torch costs seconds at startup on the hipDNN backend,
        also when device visibility is remapped and amdsmi sees a GPU."""
        attempts = []

        class _RecordTorchImport:
            """Finds torch (version lookups may), records and fails loading it."""

            def find_spec(self, name, path=None, target=None):
                return (
                    importlib.util.spec_from_loader(name, self)
                    if name == "torch"
                    else None
                )

            def create_module(self, spec):
                return None

            def exec_module(self, module):
                attempts.append(module.__name__)
                raise ImportError("torch import attempted")

        monkeypatch.delitem(sys.modules, "torch", raising=False)
        monkeypatch.setattr(sys, "meta_path", [_RecordTorchImport(), *sys.meta_path])
        monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0")
        smi = MagicMock()
        smi.amdsmi_get_processor_handles.return_value = ["gpu0"]
        arch.detect_gpu.cache_clear()
        gpu_smi._handle_for.cache_clear()
        try:
            with (
                patch.object(arch, "resolve_rocm_tool", return_value=None),
                patch.object(gpu_smi, "_amdsmi", return_value=smi),
            ):
                machine_info.collect_environment_info()
        finally:
            arch.detect_gpu.cache_clear()
            gpu_smi._handle_for.cache_clear()
        smi.amdsmi_get_processor_handles.assert_called()
        assert attempts == []

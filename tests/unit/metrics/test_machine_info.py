# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for metrics.machine_info (the result ``environment`` block)."""

from unittest.mock import mock_open, patch

import pytest

from dnn_benchmarking.metrics import machine_info
from dnn_benchmarking.metrics._diagnostic import reset as _reset_warns

# Contract: result JSON v2 environment keys (minus end_of_run / selection_env,
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


@pytest.mark.parametrize(
    "raw, expected",
    [(None, None), (0, None), (8902, "8.9.2"), (90100, "9.1.0")],
)
def test_cudnn_version_decoding_across_packing_schemes(raw, expected):
    assert machine_info._format_cudnn_version(raw) == expected


class TestCollectEnvironmentInfo:
    def test_has_exactly_the_contract_keys(self):
        with patch.object(machine_info, "is_amdsmi_available", return_value=False):
            info = machine_info.collect_environment_info()
        assert set(info) == _ENVIRONMENT_KEYS

    def test_missing_amdsmi_is_recorded_and_warned_once(self, capsys):
        with patch.object(machine_info, "is_amdsmi_available", return_value=False):
            first = machine_info.collect_environment_info()
            machine_info.collect_environment_info()
        assert first["amdsmi_available"] is False
        assert capsys.readouterr().err.count("amdsmi not available") == 1

# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for the GPU identity detection chain."""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from dnn_benchmarking.metrics import arch as _arch
from dnn_benchmarking.metrics.arch import detect_arch, detect_gpu

_ROCMINFO = """\
  Name:                    AMD EPYC 7513 32-Core Processor
  Marketing Name:          AMD EPYC 7513 32-Core Processor
  Device Type:             CPU
  Name:                    gfx90a
  Marketing Name:          AMD Instinct MI210
  Device Type:             GPU
      Name:                    amdgcn-amd-amdhsa--gfx90a:sramecc+:xnack-
"""


@pytest.fixture(autouse=True)
def _uncached():
    detect_gpu.cache_clear()
    yield
    detect_gpu.cache_clear()


def _sources(torch=(None, None), amdsmi=(None, None), rocminfo=(None, None)):
    return (
        patch.object(_arch, "_detect_via_torch", return_value=torch),
        patch.object(_arch, "_detect_via_amdsmi", return_value=amdsmi),
        patch.object(_arch, "_detect_via_rocminfo", return_value=rocminfo),
    )


def test_each_field_comes_from_the_first_source_reporting_it():
    torch, amdsmi, rocminfo = _sources(
        amdsmi=(None, "Aldebaran/MI200 [Instinct MI210]"),
        rocminfo=("gfx90a", "AMD Instinct MI210"),
    )
    with torch, amdsmi, rocminfo:
        assert detect_gpu() == ("gfx90a", "Aldebaran/MI200 [Instinct MI210]")


def test_later_sources_skipped_once_both_fields_are_known():
    torch, amdsmi, rocminfo = _sources(torch=("gfx942", "AMD Instinct MI300X"))
    with torch, amdsmi as smi, rocminfo as info:
        assert detect_arch() == "gfx942"
    smi.assert_not_called()
    info.assert_not_called()


def test_nothing_detected_is_unknown_arch_and_no_model():
    torch, amdsmi, rocminfo = _sources()
    with torch, amdsmi, rocminfo:
        assert detect_gpu() == ("unknown", None)


def test_rocminfo_reports_the_first_gpu_agent_not_the_cpu():
    proc = MagicMock(returncode=0, stdout=_ROCMINFO, stderr="")
    # resolve_rocm_tool, not shutil.which: production finds rocminfo under
    # $ROCM_PATH/bin even when it is not on PATH.
    with (
        patch.object(_arch, "resolve_rocm_tool", return_value="/opt/rocm/bin/rocminfo"),
        patch("subprocess.run", return_value=proc),
    ):
        assert _arch._detect_via_rocminfo() == ("gfx90a", "AMD Instinct MI210")


def test_rocminfo_missing_reports_nothing():
    with patch.object(_arch, "resolve_rocm_tool", return_value=None):
        assert _arch._detect_via_rocminfo() == (None, None)


@pytest.mark.parametrize(
    "props, expected",
    [
        (
            SimpleNamespace(gcnArchName="gfx942:sramecc+:xnack-", name="MI300X"),
            ("gfx942", "MI300X"),
        ),
        (SimpleNamespace(name="MI300X"), (None, "MI300X")),  # CUDA: no gcnArchName
    ],
    ids=["rocm", "cuda"],
)
def test_torch_source_reads_device_properties(monkeypatch, props, expected):
    cuda = SimpleNamespace(
        current_device=lambda: 0, get_device_properties=lambda i: props
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))
    monkeypatch.setattr(_arch.torch_support, "gpu_available", lambda: True)
    assert _arch._detect_via_torch() == expected

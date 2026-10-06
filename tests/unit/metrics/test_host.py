# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for metrics.host (host memory snapshot)."""

import sys
import types
from unittest.mock import patch

import pytest

from dnn_benchmarking.metrics import host
from dnn_benchmarking.metrics._diagnostic import reset as _reset_warns


@pytest.fixture(autouse=True)
def _reset_warn_state():
    _reset_warns()
    yield
    _reset_warns()


class TestHostMemorySnapshot:
    def test_returns_stable_keys_when_psutil_missing(self):
        # Simulate ImportError by removing psutil from sys.modules and
        # blocking re-import via meta_path.
        with patch.dict(sys.modules, {"psutil": None}):
            snap = host.host_memory_snapshot()
        assert set(snap.keys()) == {"host_rss_mb", "host_ram_available_mb"}
        assert snap["host_rss_mb"] is None
        assert snap["host_ram_available_mb"] is None

    def test_psutil_path_returns_floats(self):
        fake_psutil = types.ModuleType("psutil")

        class _Mem:
            rss = 256 * 1024 * 1024  # 256 MiB

        class _Proc:
            def memory_info(self):
                return _Mem()

        class _VMem:
            available = 16 * 1024 * 1024 * 1024  # 16 GiB

        fake_psutil.Process = lambda: _Proc()
        fake_psutil.virtual_memory = lambda: _VMem()
        # psutil.Error is referenced inside the except handler — need it
        # to exist as a class so the import-level reference resolves.
        fake_psutil.Error = Exception

        with patch.dict(sys.modules, {"psutil": fake_psutil}):
            snap = host.host_memory_snapshot()
        assert snap["host_rss_mb"] == pytest.approx(256.0)
        assert snap["host_ram_available_mb"] == pytest.approx(16384.0)

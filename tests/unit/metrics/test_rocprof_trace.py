# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for rocprofv3 trace export."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from dnn_benchmarking.metrics import _subprocess, rocprof_trace
from dnn_benchmarking.metrics._diagnostic import reset as reset_warn_once


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    reset_warn_once()
    monkeypatch.setattr(rocprof_trace, "resolve_rocm_tool", lambda name: "rocprofv3")


def _run(out_dir):
    return rocprof_trace.run(
        inner_argv=["python"], out_dir=out_dir, timeout_s=60, context="g/E"
    )


def test_argv_requests_kernel_and_memcpy_pftrace(tmp_path):
    args = rocprof_trace._build_argv(tmp_path, ["python", "-m", "dnn_benchmarking"])
    assert "--kernel-trace" in args and "--memory-copy-trace" in args
    assert args[args.index("--output-format") + 1] == "pftrace"
    assert args[args.index("--") + 1 :] == ["python", "-m", "dnn_benchmarking"]


def test_records_hoisted_pftrace_path(tmp_path):
    out_dir = tmp_path / "trace_out"

    def fake_run(argv, timeout_s=None):
        host_dir = Path(argv[argv.index("-d") + 1]) / "host"
        host_dir.mkdir(parents=True, exist_ok=True)
        (host_dir / "results_results.pftrace").write_bytes(b"fake-pftrace")
        return MagicMock(returncode=0, stdout="", stderr="")

    with patch.object(_subprocess, "run_capped", side_effect=fake_run):
        trace = _run(out_dir)["trace"]
    assert trace == {"format": "pftrace", "path": str(out_dir / "results.pftrace")}


def test_nonzero_returncode_records_error_tail(tmp_path):
    proc = MagicMock(returncode=2, stdout="", stderr="rocprofv3: failed for reasons\n")
    with patch.object(_subprocess, "run_capped", return_value=proc):
        trace = _run(tmp_path)["trace"]
    assert trace["returncode"] == 2
    assert "failed" in trace["error_tail"]
    assert "path" not in trace

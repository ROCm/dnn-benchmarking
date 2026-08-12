# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for rocprofv3 trace export."""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from dnn_benchmarking.metrics import rocprof_trace
from dnn_benchmarking.metrics._diagnostic import reset as reset_warn_once


@pytest.fixture(autouse=True)
def _reset():
    reset_warn_once()


@pytest.fixture
def _force_rocprofv3_present(monkeypatch):
    """Pretend `/opt/rocm/bin/rocprofv3` exists for tests that don't care
    about binary resolution. Stops tests from coupling to the host's ROCm
    install layout."""
    monkeypatch.setattr(
        rocprof_trace, "resolve_rocm_tool", lambda name: "/opt/rocm/bin/rocprofv3"
    )


class TestArgvBuild:
    def test_includes_kernel_and_memcpy_traces(self, tmp_path):
        argv = rocprof_trace._build_argv(
            tmp_path,
            ["python", "-m", "dnn_benchmarking"],
            "/opt/rocm/bin/rocprofv3",
        )
        # Absolute binary path is preserved — the orchestrator must not
        # silently rewrite to a bare command name (PATH resolution in the
        # spawned process would otherwise pick up the venv shim).
        assert argv[0] == "/opt/rocm/bin/rocprofv3"
        assert "--kernel-trace" in argv
        assert "--memory-copy-trace" in argv
        assert "--output-format" in argv
        # Format follows --output-format
        assert argv[argv.index("--output-format") + 1] == "pftrace"
        # `-o results` strips the `<pid>_` prefix from rocprofv3's
        # default filename so the artifact path stays predictable.
        assert "-o" in argv
        assert argv[argv.index("-o") + 1] == "results"


class TestPftracePath:
    def test_happy_path_records_path(self, tmp_path, _force_rocprofv3_present):
        out_dir = tmp_path / "trace_out"

        def fake_run(argv, timeout_s=None, **kwargs):
            host_dir = Path(argv[argv.index("-d") + 1])
            host_dir.mkdir(parents=True, exist_ok=True)
            (host_dir / "results.pftrace").write_bytes(b"fake-pftrace")
            return MagicMock(returncode=0, stdout="", stderr="")

        with patch.object(rocprof_trace, "run_capped", side_effect=fake_run):
            extra = rocprof_trace.run(inner_argv=["python"], out_dir=out_dir)
        assert extra["trace"]["format"] == "pftrace"
        assert extra["trace"]["path"].endswith(".pftrace")

    def test_nonzero_returncode_records_error_tail(
        self, tmp_path, _force_rocprofv3_present
    ):
        out_dir = tmp_path / "trace_out"
        proc = MagicMock(
            returncode=2, stdout="", stderr="rocprofv3: failed for reasons\n"
        )
        with patch.object(rocprof_trace, "run_capped", return_value=proc):
            extra = rocprof_trace.run(inner_argv=["python"], out_dir=out_dir)
        assert extra["trace"]["returncode"] == 2
        assert "failed" in extra["trace"]["error_tail"]

    def test_rocprofv3_binary_missing_returns_skipped(self, tmp_path, monkeypatch):
        monkeypatch.setattr(rocprof_trace, "resolve_rocm_tool", lambda name: None)
        extra = rocprof_trace.run(inner_argv=["python"], out_dir=tmp_path)
        assert extra["trace"]["skipped"] == "rocprofv3 binary not found"

    def test_timeout_returns_skipped(self, tmp_path, _force_rocprofv3_present):
        """A wedged rocprofv3 must surface as a `skipped` slice. rocprofv3
        execs the workload as a grandchild, so the timeout only fires
        because run_capped kills the group — see metrics/_subprocess.py."""
        with patch.object(
            rocprof_trace,
            "run_capped",
            side_effect=subprocess.TimeoutExpired(cmd="rocprofv3", timeout=600),
        ):
            extra = rocprof_trace.run(inner_argv=["python"], out_dir=tmp_path)
        assert "timed out" in extra["trace"]["skipped"]

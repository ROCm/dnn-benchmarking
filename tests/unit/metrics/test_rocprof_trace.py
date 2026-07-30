# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for rocprofv3 trace export."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from dnn_benchmarking.metrics import rocprof_trace
from dnn_benchmarking.metrics._diagnostic import reset as reset_warn_once


@pytest.fixture(autouse=True)
def _reset():
    reset_warn_once()
    # The chrome-capability probe is cached for the process; a stale
    # entry would make these tests depend on execution order.
    rocprof_trace._rocpd_can_emit_chrome.cache_clear()


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
            "pftrace",
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
            extra = rocprof_trace.run(
                inner_argv=["python"], out_dir=out_dir, fmt="pftrace"
            )
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
            extra = rocprof_trace.run(
                inner_argv=["python"], out_dir=out_dir, fmt="pftrace"
            )
        assert extra["trace"]["returncode"] == 2
        assert "failed" in extra["trace"]["error_tail"]

    def test_rocprofv3_binary_missing_returns_skipped(self, tmp_path, monkeypatch):
        monkeypatch.setattr(rocprof_trace, "resolve_rocm_tool", lambda name: None)
        extra = rocprof_trace.run(
            inner_argv=["python"], out_dir=tmp_path, fmt="pftrace"
        )
        assert extra["trace"]["skipped"] == "rocprofv3 binary not found"


class TestKineto:
    def test_kineto_records_pftrace_directly_when_rocpd_missing(
        self, tmp_path, monkeypatch, _force_rocprofv3_present
    ):
        """When rocpd isn't importable, kineto downgrades to pftrace on
        the *first* (and only) rocprofv3 invocation. The earlier
        implementation ran rocprofv3 twice — once for the rocpd db,
        once for the pftrace fallback. That doubled profiling time
        silently; the upfront probe collapses it to a single run."""
        out_dir = tmp_path / "trace_out"

        # Hide the rocpd module from the upfront probe.
        monkeypatch.setitem(sys.modules, "rocpd", None)

        call_count = {"n": 0}
        recorded_fmts: list[str] = []

        def fake_run(argv, timeout_s=None, **kwargs):
            host_dir = Path(argv[argv.index("-d") + 1])
            host_dir.mkdir(parents=True, exist_ok=True)
            call_count["n"] += 1
            recorded_fmts.append(argv[argv.index("--output-format") + 1])
            (host_dir / "results.pftrace").write_bytes(b"fake-pftrace")
            return MagicMock(returncode=0, stdout="", stderr="")

        with patch.object(rocprof_trace, "run_capped", side_effect=fake_run):
            extra = rocprof_trace.run(
                inner_argv=["python"], out_dir=out_dir, fmt="kineto"
            )
        trace = extra["trace"]
        assert call_count["n"] == 1, "rocprofv3 must run exactly once"
        assert recorded_fmts == ["pftrace"]
        # Caller-requested format echoed unchanged so downstream knows
        # what was asked for; recorded_format reveals what we actually
        # captured; kineto_unavailable explains the downgrade.
        assert trace["format"] == "kineto"
        assert trace["recorded_format"] == "pftrace"
        assert "kineto_unavailable" in trace
        assert trace["path"].endswith(".pftrace")

    def test_kineto_downgrades_when_rocpd_cannot_emit_chrome(
        self, tmp_path, monkeypatch, _force_rocprofv3_present
    ):
        """ROCm 7.15's rocpd narrowed convert to csv/pftrace/otf2. An
        importable rocpd is therefore not enough — asking it for chrome is
        a hard argparse error, so the probe must read the offered formats
        and downgrade before the workload runs, not after."""
        out_dir = tmp_path / "trace_out"
        monkeypatch.setitem(sys.modules, "rocpd", MagicMock())
        monkeypatch.setitem(sys.modules, "otf2", MagicMock())

        recorded_fmts: list[str] = []

        def fake_run(argv, timeout_s=None, **kwargs):
            if "--help" in argv:
                return MagicMock(
                    returncode=0,
                    stdout="-f {csv,pftrace,otf2} [{csv,pftrace,otf2} ...]",
                    stderr="",
                )
            host_dir = Path(argv[argv.index("-d") + 1])
            host_dir.mkdir(parents=True, exist_ok=True)
            recorded_fmts.append(argv[argv.index("--output-format") + 1])
            (host_dir / "results.pftrace").write_bytes(b"fake-pftrace")
            return MagicMock(returncode=0, stdout="", stderr="")

        with patch.object(rocprof_trace, "run_capped", side_effect=fake_run):
            extra = rocprof_trace.run(
                inner_argv=["python"], out_dir=out_dir, fmt="kineto"
            )

        assert recorded_fmts == ["pftrace"], "the workload must run exactly once"
        assert extra["trace"]["recorded_format"] == "pftrace"
        assert "chrome" in extra["trace"]["kineto_unavailable"]

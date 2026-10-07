# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for the rocprof-compute roofline wrapper."""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from dnn_benchmarking.metrics import _subprocess
from dnn_benchmarking.metrics import roofline as roofline_mod
from dnn_benchmarking.metrics._diagnostic import reset as reset_warn_once


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    reset_warn_once()
    monkeypatch.setattr(
        roofline_mod, "resolve_rocm_tool", lambda name: "rocprof-compute"
    )


def _run(out_dir, returncode=0, stderr="", side_effect=None):
    captured = {}

    def fake_run(argv, timeout_s=None):
        captured["argv"] = argv
        if side_effect is not None:
            side_effect()
        return MagicMock(returncode=returncode, stdout="", stderr=stderr)

    with patch.object(_subprocess, "run_capped", side_effect=fake_run):
        extra = roofline_mod.run(
            inner_argv=["python"], out_dir=out_dir, timeout_s=60, context="g/E"
        )
    return extra["roofline"], captured["argv"]


def test_profile_argv_has_no_data_type_flag(tmp_path):
    """``--roofline-data-type`` exists only under ``analyze``; passing it
    to ``profile`` errors."""
    args = roofline_mod._build_argv(tmp_path / "workload", ["python", "-m", "x"])
    assert args[:2] == ["profile", "--roof-only"]
    assert "--roofline-data-type" not in args
    assert args[args.index("--") + 1 :] == ["python", "-m", "x"]


def test_workload_root_is_the_private_source_dir(tmp_path):
    """rocprof-compute clears the whole ``-p`` root (verified on 3.3.0), so
    ``-p`` must be the per-source roofline leaf, never a directory shared
    with the pmc db or the trace."""
    out_dir = tmp_path / "graph" / "MIOPEN_ENGINE" / "roofline"
    _, argv = _run(out_dir)
    assert Path(argv[argv.index("-p") + 1]) == out_dir


def test_records_csv_and_workload_paths(tmp_path):
    def write_outputs():
        inner = tmp_path / "workload" / "gfx90a"
        inner.mkdir(parents=True, exist_ok=True)
        (inner / "roofline.csv").write_text("Empirical_HBM,123\n")
        (inner / "sysinfo.csv").write_text("gpu,gfx90a\n")

    rl, _ = _run(tmp_path, side_effect=write_outputs)
    assert rl["roofline_csv"].endswith("roofline.csv")
    assert rl["sysinfo_csv"].endswith("sysinfo.csv")
    # What `rocprof-compute analyze --path` expects.
    assert Path(rl["workload_path"]).parts[-2:] == ("workload", "gfx90a")


def test_nonzero_exit_records_error_tail(tmp_path):
    rl, _ = _run(tmp_path, returncode=1, stderr="rocprof-compute: workload failed\n")
    assert rl["returncode"] == 1
    assert "failed" in rl["error_tail"]


def test_success_with_no_csv_at_all_points_at_the_tool(tmp_path):
    """Exit 0 with no CSV anywhere is a tool/version mismatch where
    ``profile --roof-only`` is a silent no-op."""
    rl, _ = _run(tmp_path)
    assert len(rl["warnings"]) == 1
    assert "no CSV output" in rl["warnings"][0]


def test_success_with_other_csv_but_no_named_files(tmp_path):
    def write_other():
        (tmp_path / "results_pmc_perf_0.csv").write_text("counter,value\n")

    rl, _ = _run(tmp_path, side_effect=write_other)
    assert rl["warnings"] == ["no roofline.csv or sysinfo.csv produced"]


def test_missing_binary_returns_skipped(tmp_path, monkeypatch):
    monkeypatch.setattr(roofline_mod, "resolve_rocm_tool", lambda name: None)
    extra = roofline_mod.run(
        inner_argv=["python"], out_dir=tmp_path, timeout_s=60, context="g/E"
    )
    assert extra == {"roofline": {"skipped": "profiling tool not found"}}


def test_timeout_returns_skipped(tmp_path):
    """rocprof-compute replays the workload several times, so it is the
    pass most likely to wedge; --profiling-timeout must reach the cap."""
    seen = []

    def wedge(argv, timeout_s):
        seen.append(timeout_s)
        raise subprocess.TimeoutExpired(argv, timeout_s)

    with patch.object(_subprocess, "run_capped", side_effect=wedge):
        rl = roofline_mod.run(
            inner_argv=["python"], out_dir=tmp_path, timeout_s=123, context="g/E"
        )["roofline"]
    assert seen == [123]
    assert "timed out after 123s" in rl["skipped"]

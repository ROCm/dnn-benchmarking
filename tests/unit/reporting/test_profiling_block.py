# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Opt-in profiling slice of the verbose detail block.

Each source contributes its own lines; failures show the reason and the
last lines of the tool's stderr so a failed pass is visible without -o.
"""

import io
from pathlib import Path

import pytest

from dnn_benchmarking.reporting.reporter import Reporter
from dnn_benchmarking.reporting.suite_results import GraphResult, ProviderEngineResult


def _render(extra_metrics) -> str:
    pe = ProviderEngineResult(
        provider="hipdnn",
        engine_id=1,
        engine_name="MIOPEN_ENGINE",
        status="success",
        extra_metrics=extra_metrics,
    )
    out = io.StringIO()
    Reporter(out, io.StringIO()).print_graph_verbose(
        GraphResult("g", "/tmp/g.json", [pe], engine_ids=[1])
    )
    return out.getvalue()


@pytest.mark.parametrize("extra", [None, {}])
def test_no_profiling_data_renders_no_profiling_lines(extra) -> None:
    assert "profiling" not in _render(extra)


def test_pmc_shows_busiest_kernels_first_and_folds_the_rest() -> None:
    per_kernel = {
        f"kernel_{i}": {"dispatches": i, "counters": {"SQ_WAVES": 64.0 * i}}
        for i in range(1, 6)
    }
    per_kernel["kernel_5"]["counters"].update({"A": 1.0, "B": 2.0, "C": 3.0})
    per_kernel["kernel_5"]["l2_hit_rate"] = 0.85
    text = _render(
        {"pmc": {"set": "basic", "arch": "gfx90a", "per_kernel": per_kernel}}
    )

    pmc_lines = [line for line in text.splitlines() if "pmc (basic, gfx90a)" in line]
    assert "kernel_5 x5" in pmc_lines[0]
    assert "SQ_WAVES=320" in pmc_lines[0] and "[+1]" in pmc_lines[0]
    assert "l2_hit 85.0%" in pmc_lines[0]
    assert "kernel_1 x1" not in text
    assert "[2 more kernel(s), see JSON]" in text


def test_pmc_db_path_renders_with_analyze_hint() -> None:
    db = "/tmp/prof/sample/MIOPEN_ENGINE/pmc_basic/results.db"
    text = _render(
        {"pmc": {"set": "basic", "arch": "gfx90a", "per_kernel": {}, "db_path": db}}
    )
    assert db in text
    assert f"rocprof-compute analyze --path {Path(db).parent}" in text


@pytest.mark.parametrize("source", ["pmc", "trace", "perf", "roofline"])
def test_failed_pass_shows_return_code_and_last_three_stderr_lines(source) -> None:
    tail = "\n".join(f"stderr line {i}" for i in range(1, 11))
    text = _render({source: {"returncode": -6, "error_tail": tail}})
    assert "failed (rc=-6)" in text
    assert "stderr line 7" not in text
    for i in (8, 9, 10):
        assert f"| stderr line {i}" in text


def test_silent_failure_shows_return_code_without_tail() -> None:
    text = _render({"perf": {"returncode": 1}})
    assert "perf: failed (rc=1)" in text
    assert "|" not in text


def test_timeout_shows_reason_and_stderr_tail() -> None:
    text = _render(
        {"perf": {"skipped": "timed out after 600 s", "error_tail": "last words"}}
    )
    assert "perf: skipped — timed out after 600 s" in text
    assert "| last words" in text


def test_trace_path_renders_with_perfetto_hint() -> None:
    text = _render({"trace": {"format": "pftrace", "path": "/tmp/out/results.pftrace"}})
    assert "trace (pftrace): /tmp/out/results.pftrace" in text
    assert "ui.perfetto.dev" in text


def test_perf_counters_render_with_scope() -> None:
    text = _render(
        {"perf": {"ipc_user": 0.795, "task_clock_ms": 123.4, "scope": "process_total"}}
    )
    assert "IPC=0.80" in text and "task_clock=123.4ms" in text
    assert "process_total" in text


def test_roofline_csv_and_analyze_hint_render() -> None:
    text = _render(
        {
            "roofline": {
                "roofline_csv": "/tmp/r/workload/gfx90a/roofline.csv",
                "workload_path": "/tmp/r/workload/gfx90a",
            }
        }
    )
    assert "/tmp/r/workload/gfx90a/roofline.csv" in text
    assert "rocprof-compute analyze --path /tmp/r/workload/gfx90a --block 4" in text

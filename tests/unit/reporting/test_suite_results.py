# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for verdicts, summary counts, oracle delta and result file I/O."""

import csv
import json
import os

import pytest

from dnn_benchmarking.reporting.statistics import BenchmarkStats, TimingInfo
from dnn_benchmarking.reporting.suite_results import (
    ROW_COLUMNS,
    CorrectnessResult,
    GraphResult,
    OracleResult,
    ProviderEngineResult,
    RunInfo,
    SuiteResult,
    build_oracle_delta,
    graph_id_for,
)


def _correct(match) -> CorrectnessResult:
    return CorrectnessResult(match, rtol=1e-3, atol=1e-5)


def _row(status="success", role="engine", correctness=None) -> ProviderEngineResult:
    return ProviderEngineResult(
        "hipdnn", 1, status, role=role, correctness=correctness
    )


def _suite(graphs, complete=True) -> SuiteResult:
    return SuiteResult(
        run=RunInfo(
            started_at="t", argv=["x"], config={"seed": 7}, complete=complete
        ),
        environment={"gpu_arch": "gfx90a"},
        graphs=graphs,
    )


class TestVerdict:
    @pytest.mark.parametrize(
        "row, expected",
        [
            (_row(correctness=_correct(True)), "passed"),
            (_row(correctness=_correct(False)), "failed"),
            (_row(correctness=_correct(None)), "unchecked"),
            (_row(correctness=None), "unchecked"),
            (_row(role="reference", correctness=_correct(False)), "reference"),
            (_row(status="error", correctness=_correct(True)), "error"),
            (_row(status="error", role="reference"), "error"),
            (_row(status="skipped"), "skipped"),
        ],
    )
    def test_verdict_matrix(self, row, expected) -> None:
        assert row.verdict == expected

    def test_row_factories_carry_reason_without_correctness(self) -> None:
        err = ProviderEngineResult.error_row("hipdnn", 5, "boom", engine_name="E")
        skip = ProviderEngineResult.skipped_row("pytorch", None, "unsupported")
        assert (err.verdict, err.to_dict()["message"]) == ("error", "boom")
        assert (skip.verdict, skip.to_dict()["message"]) == ("skipped", "unsupported")
        assert err.correctness is None and skip.correctness is None


class TestGraphStatusAndSummary:
    def test_graph_status(self) -> None:
        assert GraphResult("g", "p", [], engine_ids=[1]).status == "ok"
        assert GraphResult("g", "p", []).status == "no_engines"
        assert GraphResult("g", "p", [], engine_ids=[1], error="x").status == "error"

    def test_summary_counts_engine_rows_only(self) -> None:
        rows = [
            _row(correctness=_correct(True)),
            _row(correctness=_correct(False)),
            _row(),
            _row(status="skipped"),
            _row(status="error"),
            _row(role="reference"),
            _row(status="error", role="reference"),
        ]
        suite = _suite(
            [
                GraphResult("a", "a.json", rows, engine_ids=[1]),
                GraphResult("b", "b.json", [], error="load failed"),
                GraphResult("c", "c.json", []),
            ]
        )
        assert suite.summary() == {
            "graphs": 3,
            "rows": 5,
            "passed": 1,
            "unchecked": 1,
            "failed": 1,
            "skipped": 1,
            "errors": 1,
            "graph_errors": 1,
            "no_engine_graphs": 1,
        }

    def test_summary_recomputed_after_mutation(self) -> None:
        suite = _suite([GraphResult("a", "a.json", [], engine_ids=[1])])
        suite.graphs[0].results.append(_row())
        assert suite.summary()["unchecked"] == 1
        assert suite.to_dict()["summary"]["rows"] == 1


def _oracle(**stats) -> OracleResult:
    return OracleResult(
        plan_name="p",
        compiled_plan_index=0,
        rank=0,
        sweep_min_time_ms=1.0,
        compiled_plans_benchmarked=1,
        compiled_plans_total=2,
        compiled_plans_failed=0,
        knob_settings=[],
        **stats,
    )


class TestBuildOracleDelta:
    def test_uses_median_not_mean(self) -> None:
        # Skewed samples: means differ from medians.
        baseline = BenchmarkStats.from_timings([2.0, 2.0, 2.0, 20.0])
        tuned = BenchmarkStats.from_timings([1.0, 1.0, 1.0, 1.0])
        delta = build_oracle_delta(
            _oracle(gpu_kernel_stats=tuned, warm_baseline_gpu_kernel_stats=baseline)
        )
        assert delta.basis == "kernel"
        assert delta.baseline_median_ms == 2.0
        assert delta.oracle_median_ms == 1.0
        assert delta.delta_ms == 1.0
        assert delta.speedup == 2.0

    def test_no_warm_baseline_means_no_delta(self) -> None:
        # The row's own OOTB timing must never stand in for the baseline.
        tuned = BenchmarkStats.from_timings([1.0])
        assert build_oracle_delta(_oracle(gpu_kernel_stats=tuned)) is None

    def test_non_positive_median_means_no_delta(self) -> None:
        delta = build_oracle_delta(
            _oracle(
                gpu_kernel_stats=BenchmarkStats.from_timings([0.0]),
                warm_baseline_gpu_kernel_stats=BenchmarkStats.from_timings([2.0]),
            )
        )
        assert delta is None


def test_graph_id_is_order_independent_and_content_sensitive() -> None:
    a = graph_id_for({"nodes": [1, 2], "name": "g"})
    assert a == graph_id_for({"name": "g", "nodes": [1, 2]})
    assert a != graph_id_for({"name": "g", "nodes": [2, 1]})
    assert len(a) == 12


def _sample_suite(complete=True) -> SuiteResult:
    row = ProviderEngineResult(
        "hipdnn",
        -1,
        "success",
        engine_name="MIOPEN_ENGINE",
        gpu_kernel_stats=BenchmarkStats.from_timings([0.5] * 30),
        host_stats=BenchmarkStats.from_timings([0.01] * 30),
        correctness=CorrectnessResult(True, 1e-3, 1e-5, max_abs_diff=2e-6),
        timing=TimingInfo("staged", "hip", "cold", 10, 3.0),
        derived_tflops_per_s=1.5,
    )
    return _suite(
        [
            GraphResult("g", "g.json", [row], engine_ids=[-1], graph_id="0123456789ab"),
            GraphResult("bad", "bad.json", [], error="parse failed"),
            GraphResult("none", "none.json", [], message="no engine configs"),
        ],
        complete=complete,
    )


class TestWriteLoad:
    def test_round_trip(self, tmp_path) -> None:
        suite = _sample_suite()
        path = tmp_path / "out" / "r.json"
        suite.write(path)
        assert SuiteResult.load(path) == json.loads(suite.to_json())
        assert [p.name for p in path.parent.iterdir()] == ["r.json"]

    def test_written_file_follows_umask(self, tmp_path) -> None:
        path = tmp_path / "r.json"
        old = os.umask(0o022)
        try:
            _sample_suite().write(path)
        finally:
            os.umask(old)
        assert path.stat().st_mode & 0o777 == 0o644

    def test_load_rejects_non_json_naming_the_file(self, tmp_path) -> None:
        path = tmp_path / "r.csv"
        _sample_suite().write(path)
        with pytest.raises(ValueError, match="r.csv"):
            SuiteResult.load(path)

    def test_failed_serialization_leaves_previous_file(self, tmp_path) -> None:
        path = tmp_path / "r.json"
        _sample_suite().write(path)
        before = path.read_text()
        broken = _sample_suite()
        broken.graphs[0].results[0].extra_metrics = {"bad": object()}
        with pytest.raises(TypeError):
            broken.write(path)
        assert path.read_text() == before
        assert [p.name for p in tmp_path.iterdir()] == ["r.json"]

    def test_load_refuses_other_schema_versions(self, tmp_path) -> None:
        path = tmp_path / "v1.json"
        path.write_text(json.dumps({"metadata": {}, "graphs": []}))
        with pytest.raises(ValueError, match="schema_version"):
            SuiteResult.load(path)

    def test_load_warns_on_partial_results(self, tmp_path, capsys) -> None:
        path = tmp_path / "partial.json"
        _sample_suite(complete=False).write(path)
        SuiteResult.load(path)
        assert "partial" in capsys.readouterr().err

    def test_csv_rows(self, tmp_path) -> None:
        path = tmp_path / "r.csv"
        _sample_suite().write(path)
        with open(path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert tuple(reader.fieldnames) == ROW_COLUMNS
        row, graph_error, no_engines = rows
        assert row["gpu_arch"] == "gfx90a"
        assert row["graph_id"] == "0123456789ab"
        assert row["engine_id"] == "0xFFFFFFFFFFFFFFFF"
        assert row["verdict"] == "passed"
        assert float(row["kernel_median_ms"]) == 0.5
        assert float(row["host_median_ms"]) == 0.01
        assert float(row["max_abs_diff"]) == 2e-6
        assert (row["n"], row["timing_mode"], row["cache_mode"]) == ("30", "staged", "cold")
        assert row["seed"] == no_engines["seed"] == "7"
        assert (graph_error["graph_name"], graph_error["status"]) == ("bad", "error")
        assert graph_error["message"] == "parse failed"
        assert no_engines["status"] == "no_engines"
        assert no_engines["message"] == "no engine configs"

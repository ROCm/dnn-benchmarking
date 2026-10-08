# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for verdicts, summary counts, oracle speedup and result file I/O."""

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
    PlanResult,
    ProviderEngineResult,
    RunInfo,
    SuiteResult,
    graph_id_for,
    oracle_speedup,
)


def _correct(match) -> CorrectnessResult:
    return CorrectnessResult(match, rtol=1e-3, atol=1e-5)


def _row(status="success", role="engine", correctness=None) -> ProviderEngineResult:
    return ProviderEngineResult(
        "hipdnn", 1, status, role=role, ootb=PlanResult(correctness=correctness)
    )


def _suite(graphs, complete=True) -> SuiteResult:
    return SuiteResult(
        run=RunInfo(
            started_at="t",
            argv=["x"],
            config={"seed": 7, "cache_mode": "cold", "timing_block": 4},
            complete=complete,
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
        assert err.ootb.correctness is None and skip.ootb.correctness is None


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


def _kernel(*timings) -> BenchmarkStats:
    return BenchmarkStats.from_timings(list(timings))


def _tuned_row(
    ootb=(2.0, 2.0, 2.0, 20.0),
    tuned=(1.0, 1.0, 1.0, 9.0),
    ootb_correctness=None,
    tuned_correctness=None,
) -> ProviderEngineResult:
    # Default samples are skewed, so means (6.5, 3.0) differ from medians (2, 1).
    row = _row(correctness=ootb_correctness)
    row.ootb.gpu_kernel_stats = _kernel(*ootb) if ootb is not None else None
    row.oracle = OracleResult(
        tuning_available=True,
        gpu_kernel_stats=_kernel(*tuned) if tuned is not None else None,
        correctness=tuned_correctness,
    )
    return row


class TestOracleSpeedup:
    def test_is_ootb_median_over_tuned_median(self) -> None:
        assert oracle_speedup(_tuned_row()) == 2.0

    def test_no_tuned_run_means_no_speedup(self) -> None:
        row = _row()
        row.ootb.gpu_kernel_stats = _kernel(1.0)
        assert oracle_speedup(row) is None

    @pytest.mark.parametrize(
        "ootb_match, tuned_match, expected",
        [
            (False, True, None),
            (True, False, None),
            (None, None, 2.0),  # unchecked is not a failure
            (True, True, 2.0),
        ],
    )
    def test_explicit_validation_failure_suppresses_speedup(
        self, ootb_match, tuned_match, expected
    ) -> None:
        row = _tuned_row(
            ootb_correctness=_correct(ootb_match),
            tuned_correctness=_correct(tuned_match),
        )
        assert oracle_speedup(row) == expected

    @pytest.mark.parametrize(
        "ootb, tuned",
        [((2.0,), (0.0,)), ((0.0,), (2.0,)), (None, (1.0,)), ((1.0,), None)],
    )
    def test_missing_or_non_positive_median_means_no_speedup(self, ootb, tuned) -> None:
        assert oracle_speedup(_tuned_row(ootb=ootb, tuned=tuned)) is None


def test_graph_id_is_order_independent_and_content_sensitive() -> None:
    a = graph_id_for({"nodes": [1, 2], "name": "g"})
    assert a == graph_id_for({"name": "g", "nodes": [1, 2]})
    assert a != graph_id_for({"name": "g", "nodes": [2, 1]})
    # Golden value: sha256 of the compact sorted JSON '{"name":"g","nodes":[1,2]}'.
    # It must not change, or old and new result files stop joining in compare.
    assert a == "b01a1b729a95"


def _sample_suite(complete=True) -> SuiteResult:
    # Skewed samples so mean != median; every value-carrying field distinct.
    row = ProviderEngineResult(
        "hipdnn",
        -1,
        "success",
        engine_name="MIOPEN_ENGINE",
        elapsed_time_ms=2500.0,
        ootb=PlanResult(
            gpu_kernel_stats=BenchmarkStats.from_timings([0.5] * 29 + [5.0]),
            host_stats=BenchmarkStats.from_timings([0.01] * 29 + [1.0]),
            correctness=CorrectnessResult(True, 1e-3, 1e-5, max_abs_diff=2e-6),
            timing=TimingInfo("staged", "hip", 10, 3.0),
            cpu_build_time_ms=12.0,
            workspace_bytes=4096,
            derived_tflops_per_s=1.5,
            derived_gbytes_per_s=20.0,
        ),
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

    @pytest.mark.skipif(
        os.name == "nt", reason="POSIX permission bits; Windows only has read-only"
    )
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

    def test_failed_rename_leaves_previous_file_and_no_temp(
        self, tmp_path, monkeypatch
    ) -> None:
        path = tmp_path / "r.json"
        _sample_suite().write(path)
        before = path.read_text()
        changed = _sample_suite()
        changed.graphs.clear()

        def fail(src, dst):
            raise OSError("disk full")

        monkeypatch.setattr(os, "replace", fail)
        with pytest.raises(OSError, match="disk full"):
            changed.write(path)
        assert path.read_text() == before
        assert [p.name for p in tmp_path.iterdir()] == ["r.json"]

    @pytest.mark.parametrize(
        "version", [{}, {"schema_version": 1}, {"schema_version": 3}]
    )
    def test_load_refuses_other_schema_versions(self, tmp_path, version) -> None:
        path = tmp_path / "other.json"
        path.write_text(json.dumps({**version, "metadata": {}, "graphs": []}))
        with pytest.raises(ValueError, match="schema_version"):
            SuiteResult.load(path)

    def test_load_warns_on_partial_results(self, tmp_path, capsys) -> None:
        path = tmp_path / "partial.json"
        _sample_suite(complete=False).write(path)
        SuiteResult.load(path)
        assert "partial" in capsys.readouterr().err

    def test_floats_are_rounded_in_both_layouts(self, tmp_path) -> None:
        suite = _sample_suite()
        suite.graphs[0].results[0].ootb.derived_tflops_per_s = 1 / 3
        indented, compact = tmp_path / "i.json", tmp_path / "c.json"
        suite.write(indented)
        suite.write(compact, compact=True)
        text = compact.read_text()
        assert "\n" not in text and ", " not in text and ": " not in text
        assert indented.read_text().startswith('{\n "schema_version": 2,')
        assert SuiteResult.load(compact) == SuiteResult.load(indented)
        row = SuiteResult.load(compact)["graphs"][0]["results"][0]
        assert row["ootb"]["tflops"] == 0.333333

    def test_json_row_values(self, tmp_path) -> None:
        path = tmp_path / "r.json"
        _sample_suite().write(path)
        row = SuiteResult.load(path)["graphs"][0]["results"][0]
        assert row["elapsed_s"] == 2.5
        ootb = row["ootb"]
        assert ootb["build_ms"] == 12.0
        assert ootb["kernel"]["median_ms"] == 0.5
        assert ootb["host"]["median_ms"] == 0.01
        assert (ootb["tflops"], ootb["gbps"], ootb["workspace_bytes"]) == (
            1.5,
            20.0,
            4096,
        )

    def test_csv_rows(self, tmp_path) -> None:
        path = tmp_path / "r.csv"
        _sample_suite().write(path)
        with open(path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert tuple(reader.fieldnames) == ROW_COLUMNS
        row, graph_error, no_engines = rows
        # IQR over the median, in percent.
        kernel = _sample_suite().graphs[0].results[0].ootb.gpu_kernel_stats
        expected = kernel.iqr_ms / kernel.median_ms * 100
        assert float(row.pop("kernel_iqr_pct")) == pytest.approx(expected, rel=1e-5)
        assert row == {
            "gpu_arch": "gfx90a",
            "graph_name": "g",
            "graph_id": "0123456789ab",
            "runtime": "hipdnn",
            "role": "engine",
            "engine_id": "0xFFFFFFFFFFFFFFFF",
            "engine_name": "MIOPEN_ENGINE",
            "status": "success",
            "verdict": "passed",
            "kernel_median_ms": "0.5",
            "host_median_ms": "0.01",
            "n": "30",
            "timing_mode": "staged",
            "cache_mode": "cold",
            "timing_block": "4",
            "seed": "7",
            "tflops": "1.5",
            "gbps": "20.0",
            "workspace_bytes": "4096",
            "max_abs_diff": "2e-06",
            "message": "",
        }
        assert no_engines["seed"] == "7"
        assert (graph_error["graph_name"], graph_error["status"]) == ("bad", "error")
        assert graph_error["message"] == "parse failed"
        assert no_engines["status"] == "no_engines"
        assert no_engines["message"] == "no engine configs"

    def test_csv_writes_non_finite_statistics_as_empty_cells(self, tmp_path) -> None:
        suite = _sample_suite()
        suite.graphs[0].results[0].ootb.gpu_kernel_stats = BenchmarkStats.from_timings(
            [float("nan")] * 3
        )
        path = tmp_path / "r.csv"
        suite.write(path)
        with open(path) as f:
            row = next(csv.DictReader(f))
        assert row["kernel_median_ms"] == ""

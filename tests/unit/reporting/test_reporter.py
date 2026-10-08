# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Reporter streams, progress lines, suite header and summaries."""

import io

import pytest

from dnn_benchmarking.metrics import _diagnostic
from dnn_benchmarking.reporting.reporter import Reporter
from dnn_benchmarking.reporting.statistics import BenchmarkStats, TimingInfo
from dnn_benchmarking.reporting.suite_results import (
    CorrectnessResult,
    GraphResult,
    PlanResult,
    ProviderEngineResult,
    RunInfo,
    SuiteResult,
)


class _Tty(io.StringIO):
    def isatty(self) -> bool:
        return True


def _passed_row(name: str = "MIOPEN_ENGINE") -> ProviderEngineResult:
    return ProviderEngineResult(
        runtime="hipdnn",
        engine_id=1,
        engine_name=name,
        status="success",
        elapsed_time_ms=7900.0,
        ootb=PlanResult(
            gpu_kernel_stats=BenchmarkStats.from_timings([0.0256] * 100),
            host_stats=BenchmarkStats.from_timings([0.013] * 100),
            timing=TimingInfo("staged", "hip", 10, 7800.0),
            correctness=CorrectnessResult(True, 1e-5, 1e-6),
        ),
    )


def _graph(*rows: ProviderEngineResult, **kwargs) -> GraphResult:
    return GraphResult("g", "/tmp/g.json", list(rows), engine_ids=[1], **kwargs)


@pytest.fixture(autouse=True)
def _clean_diagnostics(monkeypatch):
    monkeypatch.setenv("COLUMNS", "120")
    _diagnostic.reset()
    _diagnostic.set_sink(None)
    yield
    _diagnostic.reset()
    _diagnostic.set_sink(None)


class TestStreams:
    def test_results_on_output_progress_and_diagnostics_on_err(self) -> None:
        out, err = io.StringIO(), io.StringIO()
        reporter = Reporter(out, err)
        reporter.graph_start(1, 1, "g")
        reporter.engine_start("MIOPEN_ENGINE")
        reporter.engine_done(_passed_row())
        reporter.info("Initializing hipDNN")
        reporter.warning("cache shared")
        reporter.error("plugin missing")
        reporter.print_graph_table(_graph(_passed_row()))

        assert "kernel_med" in out.getvalue()
        for text in (
            "[1/1] g",
            "MIOPEN_ENGINE ...",
            "Initializing hipDNN",
            "WARNING: cache shared",
            "ERROR: plugin missing",
        ):
            assert text in err.getvalue()
            assert text not in out.getvalue()
        assert "kernel_med" not in err.getvalue()

    def test_err_defaults_to_explicit_output_else_stderr(self, capsys) -> None:
        out = io.StringIO()
        Reporter(out).warning("same stream")
        assert "WARNING: same stream" in out.getvalue()

        Reporter().warning("default stream")
        captured = capsys.readouterr()
        assert "WARNING: default stream" in captured.err
        assert captured.out == ""

    def test_quiet_suppresses_progress_and_info_only(self) -> None:
        out, err = io.StringIO(), _Tty()
        reporter = Reporter(out, err, quiet=True)
        reporter.graph_start(1, 1, "g")
        reporter.engine_start("MIOPEN_ENGINE")
        reporter.engine_done(_passed_row())
        reporter.info("progress note")
        reporter.warning("still shown")
        reporter.error("also shown")
        reporter.print_graph_table(_graph(_passed_row()))

        assert err.getvalue() == "WARNING: still shown\nERROR: also shown\n"
        assert "MIOPEN_ENGINE" in out.getvalue()


class TestProgress:
    def test_non_tty_prints_one_complete_line_after_the_engine_finishes(self) -> None:
        err = io.StringIO()
        reporter = Reporter(io.StringIO(), err)
        reporter.engine_start("MIOPEN_ENGINE")
        assert err.getvalue() == ""

        reporter.engine_done(_passed_row())
        lines = err.getvalue().splitlines()
        assert len(lines) == 1
        assert lines[0].startswith("  MIOPEN_ENGINE ... passed")
        assert "25.60 µs" in lines[0]
        assert "setup 7.8 s" in lines[0]

    def test_sub_10ms_setup_keeps_two_significant_digits(self) -> None:
        err = io.StringIO()
        row = _passed_row()
        row.ootb.timing.first_call_ms = 0.123
        Reporter(io.StringIO(), err).engine_done(row)
        assert "setup 0.12 ms" in err.getvalue()

    def test_tty_line_is_pending_then_completed_in_place(self) -> None:
        err = _Tty()
        reporter = Reporter(io.StringIO(), err)
        reporter.engine_start("MIOPEN_ENGINE")
        assert err.getvalue() == "  MIOPEN_ENGINE ..."

        reporter.engine_done(_passed_row())
        assert err.getvalue().count("\n") == 1
        assert err.getvalue().startswith("  MIOPEN_ENGINE ... passed  25.60 µs")

    def test_warn_once_terminates_pending_line_before_writing(self, capsys) -> None:
        err = _Tty()
        reporter = Reporter(io.StringIO(), err)
        reporter.engine_start("MIOPEN_ENGINE")
        _diagnostic.warn_once("amdsmi", "module not installed")
        reporter.engine_done(_passed_row())

        lines = err.getvalue().splitlines()
        assert lines[0] == "  MIOPEN_ENGINE ..."
        assert lines[1] == "[metrics:amdsmi] module not installed"
        assert lines[2].startswith("  MIOPEN_ENGINE ... passed")

        # Once no line is pending, warnings go straight to stderr again.
        _diagnostic.warn_once("psutil", "missing")
        assert "[metrics:psutil] missing" in capsys.readouterr().err

    def test_results_written_mid_line_start_on_a_fresh_line(self) -> None:
        stream = _Tty()
        reporter = Reporter(stream, stream)
        reporter.engine_start("MIOPEN_ENGINE")
        reporter.print_graph_table(_graph(_passed_row()))
        assert stream.getvalue().startswith("  MIOPEN_ENGINE ...\ng\n")

    @pytest.mark.parametrize("tty", [False, True])
    def test_skip_and_error_lines_show_the_real_reason(self, tty) -> None:
        err = _Tty() if tty else io.StringIO()
        reporter = Reporter(io.StringIO(), err)
        reporter.engine_start("A")
        reporter.engine_done(
            ProviderEngineResult.skipped_row(
                "hipdnn", 1, "No engine configurations available"
            )
        )
        reporter.engine_start("B")
        reporter.engine_done(
            ProviderEngineResult.error_row(
                "hipdnn", 2, "Graph validation failed:\nbatch mismatch"
            )
        )
        lines = err.getvalue().splitlines()
        assert lines == [
            "  A ... skipped: No engine configurations available",
            "  B ... error: Graph validation failed: batch mismatch",
        ]

    def test_long_reason_is_truncated_to_terminal_width(self, monkeypatch) -> None:
        monkeypatch.setenv("COLUMNS", "80")
        err = io.StringIO()
        reporter = Reporter(io.StringIO(), err)
        reporter.engine_start("MIOPEN_ENGINE")
        reporter.engine_done(ProviderEngineResult.error_row("hipdnn", 1, "x" * 500))
        assert len(err.getvalue().rstrip("\n")) <= 80


class TestSuiteHeader:
    RUN_CONFIG = {
        "warmup_iters": 10,
        "iters": 100,
        "min_time_ms": 0.0,
        "cache_mode": "cold",
        "seed": 0,
        "runtime": "hipdnn",
    }

    def test_header_states_machine_and_methodology(self) -> None:
        out = io.StringIO()
        env = {
            "cpu_model": "AMD EPYC 7513",
            "gpu_model": "AMD Instinct MI210",
            "gpu_arch": "gfx90a",
            "gpu_compute_units": 104,
            "gpu_hbm_gb": 64.0,
            "rocm_version": "7.0.2",
        }
        Reporter(out, io.StringIO()).print_suite_header(env, self.RUN_CONFIG, 3)
        text = out.getvalue()
        assert "AMD EPYC 7513" in text
        assert "AMD Instinct MI210 (gfx90a, 104 CUs, 64 GB HBM)" in text
        assert "ROCm:    7.0.2" in text
        assert (
            "warmup 10, iters 100 (min-time 0 ms), cache cold, seed 0, runtime hipdnn"
            in text
        )

    def test_cuda_host_shows_cuda_not_rocm(self) -> None:
        out = io.StringIO()
        env = {"cuda_version": "12.8", "cudnn_version": "9.10.2"}
        Reporter(out, io.StringIO()).print_suite_header(env, self.RUN_CONFIG, 1)
        assert "CUDA:    12.8, cuDNN 9.10.2" in out.getvalue()
        assert "ROCm" not in out.getvalue()


class TestSummaries:
    def test_summary_counts_unchecked_and_graph_errors_and_names_the_file(self) -> None:
        unchecked = _passed_row("U")
        unchecked.ootb.correctness = None
        failed = _passed_row("F")
        failed.ootb.correctness = CorrectnessResult(False, 1e-5, 1e-6)
        suite = SuiteResult(
            run=RunInfo(started_at="t", argv=[], config={}),
            environment={},
            graphs=[
                _graph(_passed_row(), unchecked, failed),
                _graph(ProviderEngineResult.skipped_row("hipdnn", 3, "no config")),
                GraphResult("bad", "/tmp/bad.json", [], error="Invalid JSON"),
            ],
        )
        out = io.StringIO()
        Reporter(out, io.StringIO()).print_summary(suite, "out.json")
        text = out.getvalue()
        assert (
            "3 graph(s), 4 row(s): 1 passed, 1 unchecked, 1 failed, 1 skipped, 0 error(s)"
            in text
        )
        assert "1 graph error(s)" in text
        assert "Results: out.json" in text

    def test_graph_table_verdict_column_shows_verdicts(self) -> None:
        failed = _passed_row("F")
        failed.ootb.correctness = CorrectnessResult(False, 1e-5, 1e-6)
        reference = _passed_row("R")
        reference.role = "reference"
        out = io.StringIO()
        Reporter(out, io.StringIO()).print_graph_table(
            _graph(_passed_row("P"), failed, reference)
        )
        lines = out.getvalue().splitlines()
        header = next(i for i, line in enumerate(lines) if "verdict" in line)
        cells = [line.split()[1] for line in lines[header + 1 : header + 4]]
        assert cells == ["passed", "failed", "reference"]

    def test_oracle_summary_geomean_excludes_rows_without_a_real_search(self) -> None:
        from dnn_benchmarking.reporting.suite_results import OracleDelta, OracleResult

        def tuned(speedup: float, plans: int) -> ProviderEngineResult:
            pe = _passed_row()
            pe.oracle = OracleResult("p", 0, 0, 0.1, plans, plans, 0, [])
            pe.oracle_delta = OracleDelta(
                "gpu_kernel", speedup, 1.0, speedup - 1.0, speedup
            )
            return pe

        out = io.StringIO()
        Reporter(out, io.StringIO()).print_oracle_summary(
            [_graph(tuned(2.0, 5), tuned(8.0, 5), tuned(100.0, 1))]
        )
        text = out.getvalue()
        assert "2 tuned row(s), geomean speedup 4.00x" in text
        assert "1 row(s) excluded" in text

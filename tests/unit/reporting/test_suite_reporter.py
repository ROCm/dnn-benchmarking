# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Per-graph table and verbose detail block rendering."""

import io
from typing import List, Optional

import pytest

from dnn_benchmarking.reporting.reporter import Reporter
from dnn_benchmarking.reporting.statistics import BenchmarkStats, TimingInfo
from dnn_benchmarking.reporting.suite_results import (
    CorrectnessResult,
    GraphResult,
    OracleResult,
    ProviderEngineResult,
    build_oracle_delta,
)


def _stats(median_ms: float, n: int = 100, jitter: float = 0.0) -> BenchmarkStats:
    """Samples around ``median_ms``; ``jitter`` > 0.05 makes them noisy."""
    timings = [median_ms * (1 + jitter * (-1) ** i) for i in range(n - 1)]
    return BenchmarkStats.from_timings(timings + [median_ms])


def _row(
    name: str = "MIOPEN_ENGINE",
    median_ms: Optional[float] = 0.0256,
    *,
    role: str = "engine",
    correctness: Optional[CorrectnessResult] = None,
    **kwargs,
) -> ProviderEngineResult:
    stats = _stats(median_ms) if median_ms is not None else None
    return ProviderEngineResult(
        provider="pytorch" if role == "reference" else "hipdnn",
        engine_id=None if role == "reference" else 0x15B46865C717A122,
        engine_name=None if role == "reference" else name,
        status="success",
        role=role,
        gpu_kernel_stats=stats,
        host_stats=_stats(0.013) if stats is not None else None,
        correctness=correctness,
        **kwargs,
    )


def _verdict(match: Optional[bool]) -> CorrectnessResult:
    return CorrectnessResult(
        tolerance_match=match,
        rtol=1e-5,
        atol=1e-6,
        error_message=None if match else "output mismatch",
    )


def _graph(*rows: ProviderEngineResult, **kwargs) -> GraphResult:
    return GraphResult(
        graph_name="g",
        graph_path="/tmp/g.json",
        results=list(rows),
        engine_ids=[1],
        **kwargs,
    )


def _table(*rows: ProviderEngineResult, verbose: bool = False) -> str:
    out = io.StringIO()
    Reporter(out, io.StringIO(), verbose=verbose).print_graph_table(_graph(*rows))
    return out.getvalue()


def _cells(text: str, engine: str) -> List[str]:
    """Whitespace-split cells of the table row whose engine cell is ``engine``."""
    for line in text.splitlines():
        if line.startswith(f"  {engine} "):
            return line.split()
    raise AssertionError(f"no row for {engine!r} in:\n{text}")


@pytest.fixture(autouse=True)
def _columns(monkeypatch):
    monkeypatch.setenv("COLUMNS", "120")


class TestTableLayout:
    @pytest.mark.parametrize("columns", [100, 160])
    def test_long_name_warning_and_plugin_path_stay_within_terminal_width(
        self, monkeypatch, columns
    ) -> None:
        monkeypatch.setenv("COLUMNS", str(columns))
        long_name = "HIPBLASLT_ENGINE_WITH_AN_EXTREMELY_LONG_DESCRIPTIVE_NAME_XYZ"
        warning = (
            "resample_avgpool_node: ResampleFwdAttributes AVGPOOL_EXCLUDE_PADDING with "
            "asymmetric padding uses manual valid-count correction " * 2
        )
        plugin = "/very/long/" + "nested/" * 25 + "libhipdnn_plugin.so"
        text = _table(_row(long_name, warnings=[warning], plugin_path=plugin), _row())

        lines = text.splitlines()
        assert max(len(line) for line in lines) <= columns
        assert plugin not in text
        assert "HIPBLASLT_ENGINE" in text  # truncated, not dropped

    def test_vs_best_is_best_successful_engine_median_over_row_median(self) -> None:
        text = _table(
            _row("FAST", 0.020),
            _row("MID", 0.025),
            _row("SLOW", 0.040),
            # A wrong answer is not a "best" and the reference is not an engine.
            _row("WRONG", 0.005, correctness=_verdict(False)),
            _row("pytorch", 0.010, role="reference"),
        )
        assert _cells(text, "FAST")[-1] == "1.00x"
        assert _cells(text, "MID")[-1] == "0.80x"
        assert _cells(text, "SLOW")[-1] == "0.50x"
        assert _cells(text, "WRONG")[-1] == "4.00x"
        assert _cells(text, "pytorch")[-1] == "ref"

    def test_skipped_and_error_rows_show_their_reason(self) -> None:
        skipped = ProviderEngineResult.skipped_row(
            "hipdnn",
            7,
            "No engine configurations available for the graph.",
            engine_name="SKIPPER",
        )
        errored = ProviderEngineResult.error_row(
            "hipdnn", 8, "HIP error:\n  invalid device function", engine_name="BROKEN"
        )
        text = _table(_row(), skipped, errored)
        assert "No engine configurations available" in text
        assert "HIP error: invalid device function" in text
        assert _cells(text, "SKIPPER")[1] == "skipped"
        assert _cells(text, "BROKEN")[1] == "error"

    def test_noisy_row_is_marked_and_partial_flops_are_approximate(self) -> None:
        noisy = _row("NOISY")
        noisy.gpu_kernel_stats = _stats(0.0256, jitter=0.2)
        steady = _row("STEADY", derived_tflops_per_s=1.5, analytical_flops_partial=True)
        text = _table(noisy, steady)
        assert "µs*" in _cells(text, "NOISY")
        assert "µs*" not in _cells(text, "STEADY")
        assert "~1.50" in _cells(text, "STEADY")

    def test_kernel_median_uses_readable_units(self) -> None:
        text = _table(_row("SMALL", 0.0256), _row("BIG", 4.1837))
        assert "25.60 µs" in text
        assert "4.184 ms" in text

    def test_legend_printed_once_after_first_table(self) -> None:
        out = io.StringIO()
        reporter = Reporter(out, io.StringIO())
        reporter.print_graph_table(_graph(_row()))
        first = out.getvalue()
        reporter.print_graph_table(_graph(_row()))
        second = out.getvalue()[len(first) :]
        assert len(second.splitlines()) < len(first.splitlines())

    def test_iqr_column_is_iqr_over_median(self) -> None:
        pe = _row()
        pe.gpu_kernel_stats = BenchmarkStats.from_timings(
            [0.9] * 10 + [1.0] + [1.1] * 10
        )
        text = _table(pe)
        header = text.splitlines()[1].split()
        assert header[1:4] == ["verdict", "kernel_med", "iqr%"]
        assert _cells(text, "MIOPEN_ENGINE")[4] == "20.0"

    def test_note_skips_noise_but_keeps_other_warnings(self) -> None:
        text = _table(_row("NOISY", warnings=["noisy: IQR 9% of median", "throttled"]))
        assert "throttled" in text
        assert "noisy:" not in text

    def test_engine_name_wins_over_the_note(self, monkeypatch) -> None:
        monkeypatch.setenv("COLUMNS", "100")
        name = "MIOPEN_ENGINE_DETERMINISTIC"
        text = _table(_row(name, warnings=["throttled " * 20]), _row())
        assert f"  {name} " in text
        assert max(len(line) for line in text.splitlines()) <= 100

    def test_title_links_file_stem_to_graph_name(self) -> None:
        out = io.StringIO()
        Reporter(out, io.StringIO()).print_graph_table(
            GraphResult("pointwise_add_1x2", "/g/sample_add.json", [], graph_id="abc")
        )
        assert out.getvalue().splitlines()[0] == "sample_add (pointwise_add_1x2)  [abc]"

    def test_no_engine_graph_shows_why(self) -> None:
        out = io.StringIO()
        Reporter(out, io.StringIO()).print_graph_table(
            GraphResult(
                "g", "/tmp/g.json", [], message="No engine configurations available"
            )
        )
        assert "No engine configurations available" in out.getvalue()

    def test_no_engine_graph_with_reference_row_still_shows_why(self) -> None:
        out = io.StringIO()
        Reporter(out, io.StringIO()).print_graph_table(
            GraphResult(
                "g",
                "/tmp/g.json",
                [_row(role="reference")],
                engine_ids=[],
                message="No engine configurations available",
            )
        )
        assert "no engines applicable: No engine configurations available" in (
            out.getvalue()
        )

    def test_graph_error_is_shown_instead_of_rows(self) -> None:
        out = io.StringIO()
        Reporter(out, io.StringIO()).print_graph_table(
            GraphResult("bad", "/tmp/bad.json", [], error="Invalid JSON in graph file")
        )
        assert "Invalid JSON in graph file" in out.getvalue()


def _oracle(**overrides) -> OracleResult:
    kwargs = dict(
        plan_name="tuned_plan_7",
        compiled_plan_index=2,
        rank=0,
        sweep_min_time_ms=0.210,
        compiled_plans_benchmarked=5,
        compiled_plans_total=5,
        compiled_plans_failed=0,
        knob_settings=[],
        # Warm heuristic twice as slow as the tuned run: speedup 2.00x.
        gpu_kernel_stats=_stats(0.250),
        warm_baseline_gpu_kernel_stats=_stats(0.500),
    )
    kwargs.update(overrides)
    return OracleResult(**kwargs)


def _oracle_row(baseline: Optional[CorrectnessResult] = None, **oracle_overrides):
    pe = _row(correctness=baseline)
    pe.oracle = _oracle(**oracle_overrides)
    pe.oracle_delta = build_oracle_delta(pe.oracle)
    return pe


class TestOracleColumn:
    def test_no_oracle_column_without_oracle_data(self) -> None:
        assert "oracle" not in _table(_row()).splitlines()[1]

    @pytest.mark.parametrize(
        "pe, expected",
        [
            (_oracle_row(), "2.00x"),
            # Unchecked is not failed and keeps the ratio.
            (_oracle_row(_verdict(None)), "2.00x"),
            # A wrong baseline or tuned plan cannot measure a gain.
            (_oracle_row(_verdict(False)), "invalid"),
            (_oracle_row(correctness=_verdict(False)), "invalid"),
            # One compiled plan and no provider search: the ratio is noise.
            (
                _oracle_row(compiled_plans_benchmarked=1, compiled_plans_total=1),
                "no-search",
            ),
            (
                _oracle_row(
                    compiled_plans_benchmarked=1,
                    compiled_plans_total=1,
                    exhaustive_requested=True,
                    exhaustive_supported=True,
                ),
                "2.00x",
            ),
            (_row(oracle_error="sweep exploded"), "failed"),
        ],
    )
    def test_oracle_cell(self, pe, expected) -> None:
        text = _table(pe)
        assert "oracle" in text.splitlines()[1]
        assert expected in _cells(text, "MIOPEN_ENGINE")


class TestVerboseBlock:
    def _verbose_row(self) -> ProviderEngineResult:
        pe = _row(
            plugin_path="/opt/plugins/libmiopen_plugin.so",
            cpu_build_time_ms=4.5,
            correctness=CorrectnessResult(
                tolerance_match=False,
                rtol=1e-5,
                atol=1e-6,
                max_abs_diff=6.2e-3,
                n_mismatch=3,
                n_total=1024,
            ),
            clocks_before={"sclk_mhz": 1700.0, "throttle_status": 0},
            clocks_after={"sclk_mhz": 1500.0, "throttle_status": 0},
            timing=TimingInfo("staged", "hip", "warm", 10, 7800.0),
        )
        pe.host_stats = _stats(0.013, n=10)
        return pe

    def test_table_always_printed_and_verbose_adds_detail(self) -> None:
        plain = _table(self._verbose_row())
        verbose = _table(self._verbose_row(), verbose=True)
        assert verbose.startswith(plain)
        assert "/opt/plugins/libmiopen_plugin.so" not in plain
        assert "/opt/plugins/libmiopen_plugin.so" in verbose

    def test_detail_block_renders_identity_costs_stats_and_correctness(self) -> None:
        out = io.StringIO()
        Reporter(out, io.StringIO()).print_graph_verbose(_graph(self._verbose_row()))
        text = out.getvalue()

        assert "MIOPEN_ENGINE (0x15B46865C717A122)" in text
        assert "build 4.5 ms" in text and "first call 7.8 s" in text
        kernel = next(
            line.split() for line in text.splitlines() if line.split()[:1] == ["kernel"]
        )
        # n, mean, median, std, min, p95, max, unit
        assert kernel[1] == "100" and kernel[3] == "25.600" and kernel[-1] == "µs"
        submit = next(
            line.split() for line in text.splitlines() if line.split()[:1] == ["submit"]
        )
        assert submit[1] == "10" and submit[6] == "-"  # no p95 below 20 samples
        assert "sclk 1700->1500 MHz" in text
        assert "FAILED" in text and "max_abs_diff 6.20e-03" in text
        assert "n_mismatch 3/1024" in text

    def test_oracle_detail_names_plan_knobs_and_speedup(self) -> None:
        pe = _oracle_row(knob_settings=[{"knob_id": "SPLIT_K", "value": 4}])
        out = io.StringIO()
        Reporter(out, io.StringIO()).print_graph_verbose(_graph(pe))
        text = out.getvalue()
        assert "tuned_plan_7" in text and "SPLIT_K=4" in text
        assert "500.00 µs" in text and "250.00 µs" in text and "2.00x" in text

    def test_failed_tuned_validation_is_explained(self) -> None:
        pe = _oracle_row(correctness=_verdict(False))
        out = io.StringIO()
        Reporter(out, io.StringIO()).print_graph_verbose(_graph(pe))
        assert "output mismatch" in out.getvalue()

    @pytest.mark.parametrize(
        "pe, expected",
        [
            (
                _row(oracle_error="sweep exploded"),
                "oracle      unavailable: sweep exploded",
            ),
            (
                _oracle_row(exhaustive_requested=True, exhaustive_supported=True),
                "exhaustive search enabled (a cached selection may be reused)",
            ),
            (
                _oracle_row(exhaustive_requested=True),
                "exhaustive unsupported by this engine; plan-level tuning only",
            ),
            (
                _oracle_row(compiled_plans_benchmarked=1, compiled_plans_total=1),
                "no tuning alternative: re-measured the heuristic plan; delta is noise",
            ),
            (_row(correctness=_verdict(True)), "passed (rtol 1e-05, atol 1e-06)"),
            (
                _row(correctness=CorrectnessResult(None, 1e-5, 1e-6)),
                "unchecked (no comparison performed)",
            ),
            (_row(role="reference", warnings=["math SDPA"]), "warning     math SDPA"),
            (
                ProviderEngineResult.error_row(
                    "hipdnn", 8, "HIP error: bad", engine_name="BROKEN"
                ),
                "error       HIP error: bad",
            ),
            (
                ProviderEngineResult.skipped_row(
                    "hipdnn", 7, "no configs", engine_name="SKIPPER"
                ),
                "skipped     no configs",
            ),
            (
                _row(analytical_flops_partial=True),
                "metrics     flops n/a (no analytical model)",
            ),
            (_oracle_row(), "rank 0); knobs engine defaults"),
            (
                _oracle_row(compiled_plans_benchmarked=4, compiled_plans_failed=1),
                "4/5 compiled plans benchmarked (1 failed); sweep min 210.00 µs",
            ),
            (
                _oracle_row(),
                "500.00 µs warm heuristic -> 250.00 µs tuned = 2.00x (basis kernel)",
            ),
            (
                _oracle_row(
                    derived_tflops_per_s=1.5, warm_baseline_derived_tflops_per_s=0.75
                ),
                "throughput 1.500 TFLOP/s tuned, 0.750 warm heuristic",
            ),
        ],
        ids=[
            "oracle-error",
            "exhaustive",
            "exhaustive-unsupported",
            "single-plan",
            "passed",
            "unchecked",
            "reference-warning",
            "error-row",
            "skipped-row",
            "flops-no-model",
            "default-knobs",
            "plan-counts",
            "delta",
            "throughput",
        ],
    )
    def test_detail_line(self, pe, expected) -> None:
        out = io.StringIO()
        Reporter(out, io.StringIO()).print_graph_verbose(_graph(pe))
        assert expected in out.getvalue()

    def test_reference_row_has_no_correctness_line(self) -> None:
        out = io.StringIO()
        pe = _row(role="reference", correctness=_verdict(False))
        Reporter(out, io.StringIO()).print_graph_verbose(_graph(pe))
        assert "[reference]" in out.getvalue()
        assert "correctness" not in out.getvalue()

    def test_oracle_summary_without_reportable_speedup(self) -> None:
        out = io.StringIO()
        pe = _oracle_row(compiled_plans_benchmarked=1, compiled_plans_total=1)
        Reporter(out, io.StringIO()).print_oracle_summary([_graph(pe)])
        assert out.getvalue().startswith(
            "Oracle: no reportable speedup on any of 1 tuned row(s)"
        )

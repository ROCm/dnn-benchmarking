# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Golden key-set tests for result JSON schema v2.

Consumers rely on every key being present regardless of row status, so the
full and the minimal document must have identical key sets at every level.
"""

import json

from dnn_benchmarking.reporting.statistics import BenchmarkStats, TimingInfo
from dnn_benchmarking.reporting.suite_results import (
    CorrectnessResult,
    GraphResult,
    OracleResult,
    ProviderEngineResult,
    RunInfo,
    SuiteResult,
    build_oracle_delta,
)

TOP_KEYS = {"schema_version", "tool", "run", "environment", "summary", "graphs"}
RUN_KEYS = {"started_at", "finished_at", "complete", "argv", "config"}
CONFIG_KEYS = {
    "backend", "engine_filter", "plugin_paths", "warmup_iters", "iters",
    "min_time_ms", "cache_mode", "seed", "validate", "rtol", "atol",
    "oracle_mode", "autotune", "cache_dir", "pytorch_sdpa_backend",
    "pytorch_rocm_fa_library", "metrics_tier", "profiling",
}  # fmt: skip
PROFILING_KEYS = {"pmc", "emit_trace", "perf", "roofline"}
ENV_KEYS = {
    "hostname", "cpu_model", "cpu_count", "numa_nodes", "total_ram_gb",
    "kernel_version", "gpu_model", "gpu_arch", "gpu_compute_units", "gpu_hbm_gb",
    "gpu_pcie_link", "amdgpu_driver_version", "gpu_power_cap_w",
    "gpu_max_sclk_mhz", "gpu_compute_partition", "rocm_version", "cuda_version",
    "cudnn_version", "hipdnn_version", "python_version", "torch_version",
    "amdsmi_available", "selection_env", "end_of_run",
}  # fmt: skip
END_OF_RUN_KEYS = {"host_rss_mb", "host_ram_available_mb", "vram_used_mb", "vram_total_mb"}
SUMMARY_KEYS = {
    "graphs", "rows", "passed", "unchecked", "failed", "skipped", "errors",
    "graph_errors", "no_engine_graphs",
}  # fmt: skip
GRAPH_KEYS = {"graph_id", "graph_name", "graph_path", "status", "error", "results"}
ROW_KEYS = {
    "provider", "role", "engine", "status", "verdict", "message", "started_at",
    "elapsed_s", "build_ms", "timing", "kernel", "host", "metrics", "correctness",
    "warnings", "oracle", "extra_metrics",
}  # fmt: skip
ENGINE_KEYS = {"id", "name", "version", "plugin_path"}
METRICS_KEYS = {
    "flops", "flops_partial", "io_bytes", "tflops", "gbps", "workspace_bytes",
    "vram_mb", "clocks_before", "clocks_after",
}  # fmt: skip
STATS_KEYS = {
    "n", "mean_ms", "std_ms", "cv", "min_ms", "p25_ms", "median_ms", "p75_ms",
    "p95_ms", "max_ms", "iqr_ms",
}  # fmt: skip
TIMING_KEYS = {
    "mode", "backend", "cache_mode", "warmup_iters", "first_call_ms", "capped",
    "fallback_reason",
}  # fmt: skip
CORRECTNESS_KEYS = {
    "match", "rtol", "atol", "max_abs_diff", "max_rel_diff", "n_mismatch",
    "n_total", "worst_output_uid", "message",
}  # fmt: skip
ORACLE_KEYS = {
    "status", "error", "plan_name", "compiled_plan_index", "rank",
    "sweep_min_time_ms", "compiled_plans_benchmarked", "compiled_plans_total",
    "compiled_plans_failed", "tuning_available", "knob_settings",
    "exhaustive_requested", "exhaustive_enabled", "exhaustive_supported",
    "cpu_build_time_ms", "kernel", "host", "baseline_kernel", "baseline_host",
    "correctness", "delta",
}  # fmt: skip
DELTA_KEYS = {"basis", "baseline_median_ms", "oracle_median_ms", "delta_ms", "speedup"}


def _stats(median: float = 1.0) -> BenchmarkStats:
    return BenchmarkStats.from_timings([median] * 25)


def _oracle() -> OracleResult:
    return OracleResult(
        plan_name="plan",
        compiled_plan_index=1,
        rank=0,
        sweep_min_time_ms=0.5,
        compiled_plans_benchmarked=2,
        compiled_plans_total=2,
        compiled_plans_failed=0,
        knob_settings=[],
        gpu_kernel_stats=_stats(1.0),
        warm_baseline_gpu_kernel_stats=_stats(2.0),
    )


def full_suite() -> SuiteResult:
    """Every optional field populated, one row per status."""
    oracle = _oracle()
    row = ProviderEngineResult(
        provider="hipdnn",
        engine_id=0x15B46865C717A122,
        engine_name="MIOPEN_ENGINE",
        status="success",
        engine_version="1.0",
        plugin_path="/opt/lib/plugin.so",
        cpu_build_time_ms=3.0,
        gpu_kernel_stats=_stats(0.5),
        host_stats=_stats(0.01),
        elapsed_time_ms=1500.0,
        correctness=CorrectnessResult(
            execution_success=True,
            tolerance_match=True,
            rtol=1e-3,
            atol=1e-5,
            max_abs_diff=1e-6,
            max_rel_diff=1e-4,
            n_mismatch=0,
            n_total=4096,
            worst_output_uid=7,
        ),
        warnings=["noisy: CV 6.0%"],
        workspace_bytes=1024,
        analytical_flops=10**9,
        analytical_flops_partial=True,
        analytical_io_bytes=10**6,
        derived_tflops_per_s=2.0,
        derived_gbytes_per_s=2000.0,
        vram_used_mb=512.0,
        extra_metrics={"pmc": {"SQ_WAVES": 64}, "perf": None, "roofline": None, "trace": None},
        oracle=oracle,
        oracle_delta=build_oracle_delta(oracle),
        timing=TimingInfo("staged", "hip", "warm", 10, 12.5),
        clocks_before={"sclk_mhz": 1700},
        clocks_after={"sclk_mhz": 1700},
    )
    env = {k: "x" for k in ENV_KEYS}
    env["end_of_run"] = {k: 1.0 for k in END_OF_RUN_KEYS}
    config = {k: "x" for k in CONFIG_KEYS}
    config["profiling"] = {k: False for k in PROFILING_KEYS}
    return SuiteResult(
        run=RunInfo(
            started_at="2026-01-01T00:00:00+00:00",
            argv=["dnn-benchmark", "g.json"],
            config=config,
            finished_at="2026-01-01T00:01:00+00:00",
            complete=True,
        ),
        environment=env,
        graphs=[
            GraphResult(
                graph_name="g",
                graph_path="graphs/g.json",
                results=[row],
                engine_ids=[row.engine_id],
                graph_id="abcdef012345",
            )
        ],
    )


def minimal_suite() -> SuiteResult:
    """Nothing optional populated: an error row with an oracle error."""
    row = ProviderEngineResult.error_row("hipdnn", None, "boom")
    row.oracle_error = "tuning failed"
    return SuiteResult(
        run=RunInfo(started_at="t", argv=[], config={}),
        environment={},
        graphs=[GraphResult("g", "g.json", [row], graph_id=None, error=None)],
    )


def _key_sets(doc: dict) -> dict:
    graph = doc["graphs"][0]
    row = graph["results"][0]
    return {
        "top": set(doc),
        "tool": set(doc["tool"]),
        "run": set(doc["run"]),
        "config": set(doc["run"]["config"]),
        "profiling": set(doc["run"]["config"]["profiling"]),
        "environment": set(doc["environment"]),
        "end_of_run": set(doc["environment"]["end_of_run"]),
        "summary": set(doc["summary"]),
        "graph": set(graph),
        "row": set(row),
        "engine": set(row["engine"]),
        "metrics": set(row["metrics"]),
        "oracle": set(row["oracle"]),
    }


def test_full_document_has_exact_key_sets() -> None:
    doc = json.loads(full_suite().to_json())
    keys = _key_sets(doc)
    assert keys["top"] == TOP_KEYS
    assert keys["tool"] == {"name", "version"}
    assert keys["run"] == RUN_KEYS
    assert keys["config"] == CONFIG_KEYS
    assert keys["profiling"] == PROFILING_KEYS
    assert keys["environment"] == ENV_KEYS
    assert keys["end_of_run"] == END_OF_RUN_KEYS
    assert keys["summary"] == SUMMARY_KEYS
    assert keys["graph"] == GRAPH_KEYS
    assert keys["row"] == ROW_KEYS
    assert keys["engine"] == ENGINE_KEYS
    assert keys["metrics"] == METRICS_KEYS
    row = doc["graphs"][0]["results"][0]
    assert set(row["kernel"]) == STATS_KEYS
    assert set(row["host"]) == STATS_KEYS
    assert set(row["timing"]) == TIMING_KEYS
    assert set(row["correctness"]) == CORRECTNESS_KEYS
    assert keys["oracle"] == ORACLE_KEYS
    assert set(row["oracle"]["kernel"]) == STATS_KEYS
    assert set(row["oracle"]["baseline_kernel"]) == STATS_KEYS
    assert set(row["oracle"]["delta"]) == DELTA_KEYS
    assert row["oracle"]["delta"]["basis"] == "kernel"
    assert doc["schema_version"] == 2


def test_minimal_document_has_the_same_key_sets() -> None:
    full = _key_sets(json.loads(full_suite().to_json()))
    minimal = _key_sets(json.loads(minimal_suite().to_json()))
    assert minimal == full


def test_minimal_row_nulls() -> None:
    row = json.loads(minimal_suite().to_json())["graphs"][0]["results"][0]
    assert row["engine"]["id"] is None
    assert row["kernel"] is None and row["host"] is None and row["timing"] is None
    assert row["correctness"] is None
    assert row["warnings"] == []
    assert row["message"] == "boom"
    assert row["verdict"] == "error"
    assert row["oracle"]["status"] == "error"
    assert row["oracle"]["error"] == "tuning failed"
    assert row["oracle"]["plan_name"] is None
    assert row["oracle"]["delta"] is None


def test_oracle_is_null_when_not_requested() -> None:
    row = ProviderEngineResult("hipdnn", 1, "success")
    assert row.to_dict()["oracle"] is None


def test_engine_id_is_unsigned_hex() -> None:
    doc = json.loads(full_suite().to_json())
    assert doc["graphs"][0]["results"][0]["engine"]["id"] == "0x15B46865C717A122"
    negative = ProviderEngineResult("hipdnn", -2478880896520848391, "success")
    assert negative.to_dict()["engine"]["id"] == "0xDD993EF5525F7BF9"


def test_non_finite_floats_serialize_as_null() -> None:
    suite = full_suite()
    row = suite.graphs[0].results[0]
    row.correctness.max_rel_diff = float("inf")
    row.derived_tflops_per_s = float("nan")
    doc = json.loads(suite.to_json())  # strict: would raise on NaN tokens
    out = doc["graphs"][0]["results"][0]
    assert out["correctness"]["max_rel_diff"] is None
    assert out["metrics"]["tflops"] is None


def test_extra_metrics_serialize_on_non_success_rows() -> None:
    row = ProviderEngineResult.error_row("hipdnn", 1, "profiling crashed")
    row.extra_metrics = {"pmc": None}
    assert row.to_dict()["extra_metrics"] == {"pmc": None}

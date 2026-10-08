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
    PlanResult,
    ProviderEngineResult,
    RunInfo,
    SuiteResult,
)

TOP_KEYS = {"schema_version", "tool", "run", "environment", "summary", "graphs"}
RUN_KEYS = {"started_at", "finished_at", "complete", "argv", "config"}
CONFIG_KEYS = {
    "runtime", "engine_filter", "plugin_paths", "warmup_iters", "iters",
    "min_time_ms", "cache_mode", "timing_block", "seed", "validate", "rtol", "atol",
    "oracle", "autotune", "hipdnn_cache_dir", "pytorch_sdpa_backend",
    "pytorch_rocm_fa_library", "metrics", "profiling",
}  # fmt: skip
PROFILING_KEYS = {"pmc", "trace", "perf", "roofline"}
ENV_KEYS = {
    "hostname", "cpu_model", "cpu_count", "numa_nodes", "total_ram_gb",
    "kernel_version", "gpu_model", "gpu_arch", "gpu_compute_units", "gpu_hbm_gb",
    "gpu_pcie_link", "amdgpu_driver_version", "gpu_power_cap_w",
    "gpu_max_sclk_mhz", "gpu_compute_partition", "rocm_version", "cuda_version",
    "cudnn_version", "hipdnn_version", "python_version", "torch_version",
    "amdsmi_available", "selection_env",
}  # fmt: skip
SUMMARY_KEYS = {
    "graphs", "rows", "passed", "unchecked", "failed", "skipped", "errors",
    "graph_errors", "no_engine_graphs",
}  # fmt: skip
GRAPH_KEYS = {
    "graph_id",
    "graph_name",
    "graph_path",
    "status",
    "error",
    "message",
    "results",
}
ROW_KEYS = {
    "runtime", "role", "engine", "status", "verdict", "message", "started_at",
    "elapsed_s", "metrics", "ootb", "oracle", "oracle_error", "warnings",
    "extra_metrics",
}  # fmt: skip
ENGINE_KEYS = {"id", "name", "version"}
METRICS_KEYS = {"flops", "io_bytes", "clocks_after"}
PLAN_KEYS = {
    "build_ms", "timing", "kernel", "host", "workspace_bytes", "tflops", "gbps",
    "correctness",
}  # fmt: skip
STATS_KEYS = {"n", "p25_ms", "median_ms", "p75_ms"}
TIMING_KEYS = {
    "mode", "timer", "warmup_iters", "first_call_ms", "capped", "fallback_reason",
}  # fmt: skip
CORRECTNESS_KEYS = {
    "match", "rtol", "atol", "max_abs_diff", "max_rel_diff", "n_mismatch",
    "n_total", "worst_output_uid", "message",
}  # fmt: skip
ORACLE_KEYS = PLAN_KEYS | {"tuning_available"}


def _stats(median: float = 1.0) -> BenchmarkStats:
    return BenchmarkStats.from_timings([median] * 25)


def _oracle() -> OracleResult:
    # Distinct values, so a field swap in to_dict changes the output.
    return OracleResult(
        tuning_available=False,
        cpu_build_time_ms=7.0,
        timing=TimingInfo("events", "hip", 11, 13.5),
        gpu_kernel_stats=_stats(1.0),
        host_stats=_stats(3.0),
        correctness=CorrectnessResult(False, 2e-3, 3e-5, 0.25, 0.75, "off", 5, 64, 9),
        derived_tflops_per_s=4.0,
        derived_gbytes_per_s=3000.0,
        workspace_bytes=2048,
    )


def full_suite() -> SuiteResult:
    """Every optional field populated, one row per status."""
    row = ProviderEngineResult(
        runtime="hipdnn",
        engine_id=0x15B46865C717A122,
        engine_name="MIOPEN_ENGINE",
        status="success",
        engine_version="1.0",
        plugin_path="/opt/lib/plugin.so",
        elapsed_time_ms=1500.0,
        warnings=["noisy: CV 6.0%"],
        analytical_flops=10**9,
        analytical_io_bytes=10**6,
        extra_metrics={
            "pmc": {"SQ_WAVES": 64},
            "perf": None,
            "roofline": None,
            "trace": None,
        },
        oracle=_oracle(),
        clocks_after={"sclk_mhz": 1650},
        ootb=PlanResult(
            cpu_build_time_ms=3.0,
            gpu_kernel_stats=_stats(0.5),
            host_stats=_stats(0.01),
            correctness=CorrectnessResult(
                tolerance_match=True,
                rtol=1e-3,
                atol=1e-5,
                max_abs_diff=1e-6,
                max_rel_diff=1e-4,
                error_message="note",
                n_mismatch=0,
                n_total=4096,
                worst_output_uid=7,
            ),
            workspace_bytes=1024,
            derived_tflops_per_s=2.0,
            derived_gbytes_per_s=2000.0,
            timing=TimingInfo("staged", "hip", 10, 12.5),
        ),
    )
    # Distinct values, so a dropped or swapped value changes the output.
    env = {k: f"env_{k}" for k in ENV_KEYS}
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
    """Nothing optional populated: an error row that also records an oracle error."""
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
        "summary": set(doc["summary"]),
        "graph": set(graph),
        "row": set(row),
        "engine": set(row["engine"]),
        "metrics": set(row["metrics"]),
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
    assert keys["summary"] == SUMMARY_KEYS
    assert keys["graph"] == GRAPH_KEYS
    assert keys["row"] == ROW_KEYS
    assert keys["engine"] == ENGINE_KEYS
    assert keys["metrics"] == METRICS_KEYS
    row = doc["graphs"][0]["results"][0]
    # The OOTB plan and the tuned plan are the same object type.
    assert set(row["ootb"]) == PLAN_KEYS
    assert set(row["oracle"]) == ORACLE_KEYS
    for plan in (row["ootb"], row["oracle"]):
        assert set(plan["kernel"]) == STATS_KEYS
        assert set(plan["host"]) == STATS_KEYS
        assert set(plan["timing"]) == TIMING_KEYS
        assert set(plan["correctness"]) == CORRECTNESS_KEYS
    assert doc["schema_version"] == 2
    assert doc["environment"] == full_suite().environment  # values unchanged


def test_full_row_values() -> None:
    row = json.loads(full_suite().to_json())["graphs"][0]["results"][0]
    assert row["engine"] == {
        "id": "0x15B46865C717A122",
        "name": "MIOPEN_ENGINE",
        "version": "1.0",
    }
    assert row["metrics"] == {
        "flops": 10**9,
        "io_bytes": 10**6,
        "clocks_after": {"sclk_mhz": 1650},
    }
    ootb = row["ootb"]
    assert {k: ootb[k] for k in ("build_ms", "workspace_bytes", "tflops", "gbps")} == {
        "build_ms": 3.0,
        "workspace_bytes": 1024,
        "tflops": 2.0,
        "gbps": 2000.0,
    }
    assert (ootb["kernel"]["median_ms"], ootb["host"]["median_ms"]) == (0.5, 0.01)
    assert ootb["timing"]["mode"] == "staged"
    assert ootb["correctness"] == {
        "match": True,
        "rtol": 1e-3,
        "atol": 1e-5,
        "max_abs_diff": 1e-6,
        "max_rel_diff": 1e-4,
        "n_mismatch": 0,
        "n_total": 4096,
        "worst_output_uid": 7,
        "message": "note",
    }
    stats_keys = {"kernel", "host"}
    assert {k: v for k, v in row["oracle"].items() if k not in stats_keys} == {
        "build_ms": 7.0,
        "timing": {
            "mode": "events",
            "timer": "hip",
            "warmup_iters": 11,
            "first_call_ms": 13.5,
            "capped": False,
            "fallback_reason": None,
        },
        "workspace_bytes": 2048,
        "tflops": 4.0,
        "gbps": 3000.0,
        "tuning_available": False,
        "correctness": {
            "match": False,
            "rtol": 2e-3,
            "atol": 3e-5,
            "max_abs_diff": 0.25,
            "max_rel_diff": 0.75,
            "n_mismatch": 5,
            "n_total": 64,
            "worst_output_uid": 9,
            "message": "off",
        },
    }
    medians = {k: row["oracle"][k]["median_ms"] for k in stats_keys}
    assert medians == {"kernel": 1.0, "host": 3.0}
    assert row["oracle_error"] is None
    assert row["warnings"] == ["noisy: CV 6.0%"]
    assert row["elapsed_s"] == 1.5


def test_minimal_document_has_the_same_key_sets() -> None:
    full = _key_sets(json.loads(full_suite().to_json()))
    minimal = _key_sets(json.loads(minimal_suite().to_json()))
    assert minimal == full


def test_minimal_row_nulls() -> None:
    row = json.loads(minimal_suite().to_json())["graphs"][0]["results"][0]
    assert row["engine"]["id"] is None
    assert row["ootb"] is None
    assert row["warnings"] == []
    assert row["message"] == "boom"
    assert row["verdict"] == "error"
    assert row["oracle"] is None
    assert row["oracle_error"] == "tuning failed"


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
    row.ootb.correctness.max_rel_diff = float("inf")
    row.ootb.derived_tflops_per_s = float("nan")
    doc = json.loads(suite.to_json())  # strict: would raise on NaN tokens
    out = doc["graphs"][0]["results"][0]
    assert out["ootb"]["correctness"]["max_rel_diff"] is None
    assert out["ootb"]["tflops"] is None


def test_extra_metrics_serialize_on_non_success_rows() -> None:
    row = ProviderEngineResult.error_row("hipdnn", 1, "profiling crashed")
    row.extra_metrics = {"pmc": None}
    assert row.to_dict()["extra_metrics"] == {"pmc": None}

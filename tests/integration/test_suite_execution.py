# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Suite runner on a real GPU: rows, metrics, engine selection, oracle, PyTorch."""

import io
import json
import re

import pytest

from dnn_benchmarking.config import (
    RuntimeName,
    MetricsConfig,
    OracleMode,
    PyTorchSdpaBackendName,
    SuiteConfig,
)
from dnn_benchmarking.execution import Executor
from dnn_benchmarking.execution.suite_runner import (
    run_graph_all_providers,
    run_graph_pytorch,
)
from dnn_benchmarking.reporting.reporter import Reporter
from tests.conftest import expected_timer
from tests.integration.conftest import load_graph

pytestmark = pytest.mark.gpu

HEX_ID = re.compile(r"0x[0-9A-F]{16}")


def _run_conv(hipdnn, **config):
    path, graph_json, tensor_infos = load_graph("sample_conv_fwd.json")
    result = run_graph_all_providers(
        path,
        graph_json,
        tensor_infos,
        SuiteConfig(warmup_iters=1, benchmark_iters=3, **config),
        hipdnn.Handle(),
        Reporter(output=io.StringIO()),
    )
    assert result.error is None, result.error
    successes = [r for r in result.results if r.status == "success"]
    assert successes, [r.skip_reason or r.error_message for r in result.results]
    return result, successes


def test_rows_carry_timing_metrics_and_v2_schema(hipdnn) -> None:
    """Default metrics on: timing, derived metrics, and a strict-JSON v2 row."""
    result, successes = _run_conv(hipdnn)
    assert result.status == "ok"
    assert re.fullmatch(r"[0-9a-f]{12}", result.graph_id)

    for r in successes:
        assert r.runtime == "hipdnn"
        assert r.engine_name and not r.engine_name.startswith("0x")
        assert r.cpu_build_time_ms > 0
        assert r.gpu_kernel_stats.n == r.host_stats.n == 3
        assert r.timing.timer == "hip" and r.timing.warmup_iters == 1
        assert r.workspace_bytes >= 0
        assert r.analytical_flops > 0 and r.analytical_io_bytes > 0
        assert r.derived_tflops_per_s > 0 and r.derived_gbytes_per_s > 0

        row = json.loads(json.dumps(r.to_dict(), allow_nan=False))
        assert HEX_ID.fullmatch(row["engine"]["id"])
        assert row["engine"]["name"] == r.engine_name
        assert row["verdict"] == "unchecked"
        ootb = row["ootb"]
        assert ootb["kernel"]["median_ms"] > 0
        assert ootb["host"]["median_ms"] > 0
        # p95 needs n >= 20.
        assert ootb["kernel"]["p95_ms"] is None
        assert ootb["correctness"] is None
        assert row["oracle"] is None


def test_no_metrics_suppresses_basic_fields(hipdnn) -> None:
    """``--no-metrics`` skips the always-on probes; timing still runs."""
    _, successes = _run_conv(hipdnn, metrics=MetricsConfig(basic=False))
    for r in successes:
        assert r.gpu_kernel_stats is not None
        assert r.workspace_bytes is None
        assert r.analytical_flops is None
        assert r.analytical_io_bytes is None
        assert r.derived_tflops_per_s is None
        assert r.vram_used_mb is None


def test_engine_selection_runs_in_caller_order(hipdnn) -> None:
    """--engine is a selection kept in the caller's order, not a filter."""
    path, graph_json, _ = load_graph("sample_conv_fwd.json")
    ranked = Executor(json.dumps(graph_json), SuiteConfig().timing_policy)
    ids = ranked.discover_engines(hipdnn.Handle())
    if len(ids) < 2:
        pytest.skip(f"need two engines for the conv graph, found {len(ids)}")

    selection = list(reversed(ids))
    result, _ = _run_conv(hipdnn, engine_filter=selection)
    assert [r.engine_id for r in result.results] == selection


def test_oracle_plan_records_tuned_payload(hipdnn) -> None:
    """--oracle-mode plan attaches the tuned plan and a median-based delta."""
    result, successes = _run_conv(hipdnn, oracle_mode=OracleMode.PLAN)
    tuned = [r for r in successes if r.oracle is not None]
    assert tuned, [(r.engine_name, r.oracle_error) for r in successes]

    r = tuned[0]
    # The active plan resolves to the row's own registered engine name,
    # never the "0x..." fallback an unregistered engine ID produces.
    assert r.oracle.plan_name == r.engine_name
    assert r.oracle.rank == 0
    assert r.oracle.compiled_plan_index >= 0
    assert r.oracle.compiled_plans_benchmarked >= 1

    row = json.loads(json.dumps(r.to_dict(), allow_nan=False))
    oracle = row["oracle"]
    assert row["oracle_error"] is None
    # The tuned plan is the same object as the OOTB plan, plus tuning keys.
    assert set(row["ootb"]) <= set(oracle)
    assert set(oracle["delta"]) == {
        "basis",
        "baseline_median_ms",
        "oracle_median_ms",
        "delta_ms",
        "speedup",
    }
    # The baseline is the post-sweep re-timing, never the row's own
    # pre-sweep OOTB number.
    assert oracle["delta"]["baseline_median_ms"] == (
        oracle["baseline_kernel"]["median_ms"]
    )


@pytest.mark.parametrize("graph_name", ["sample_conv_fwd.json", "sample_relu.json"])
def test_pytorch_runtime_times_graph(torch_gpu, graph_name: str) -> None:
    """--runtime pytorch: one timed pytorch row per graph, no hipDNN."""
    path, graph_json, tensor_infos = load_graph(graph_name)
    config = SuiteConfig(warmup_iters=1, benchmark_iters=2, runtime=RuntimeName.PYTORCH)
    result = run_graph_pytorch(
        path, graph_json, tensor_infos, config, Reporter(output=io.StringIO())
    )

    [row] = result.results
    assert (row.runtime, row.status, row.verdict) == (
        "pytorch",
        "success",
        "unchecked",
    ), row.error_message
    assert row.gpu_kernel_stats.n == row.host_stats.n == 2
    assert row.timing.timer == expected_timer()


def test_nondefault_sdpa_backend_errors_without_native_sdpa(torch_gpu) -> None:
    """A strict SDPA selection rejects a graph that never reaches native SDPA."""
    path, graph_json, tensor_infos = load_graph("sample_conv_fwd.json")
    config = SuiteConfig(
        warmup_iters=1,
        benchmark_iters=2,
        runtime=RuntimeName.PYTORCH,
        pytorch_sdpa_backend=PyTorchSdpaBackendName.MATH,
    )
    result = run_graph_pytorch(
        path, graph_json, tensor_infos, config, Reporter(output=io.StringIO())
    )

    [row] = result.results
    assert row.verdict == "error"
    assert "native forward SDPA call" in row.error_message
    assert row.gpu_kernel_stats is None and row.host_stats is None

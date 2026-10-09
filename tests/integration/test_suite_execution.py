# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Suite runner on a real GPU: rows, metrics, engine selection, oracle, PyTorch."""

import io
import json
import re
import sys

import pytest

from dnn_benchmarking.config import (
    RuntimeName,
    MetricsConfig,
    PyTorchSdpaBackendName,
    SuiteConfig,
)
from dnn_benchmarking.execution import Executor
from dnn_benchmarking.execution.suite_runner import (
    run_graph_all_providers,
    run_graph_pytorch,
)
from dnn_benchmarking.metrics._subprocess import run_capped
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
        assert r.ootb.cpu_build_time_ms > 0
        assert r.ootb.gpu_kernel_stats.n == r.ootb.host_stats.n == 3
        assert r.ootb.timing.timer == "hip" and r.ootb.timing.warmup_iters == 1
        assert r.ootb.workspace_bytes >= 0
        assert r.analytical_flops > 0 and r.analytical_io_bytes > 0
        assert r.ootb.derived_tflops_per_s > 0 and r.ootb.derived_gbytes_per_s > 0

        row = json.loads(json.dumps(r.to_dict(), allow_nan=False))
        assert HEX_ID.fullmatch(row["engine"]["id"])
        assert row["engine"]["name"] == r.engine_name
        assert row["verdict"] == "unchecked"
        ootb = row["ootb"]
        assert ootb["kernel"]["median_ms"] > 0
        assert ootb["host"]["median_ms"] > 0
        assert set(ootb["kernel"]) == {"p25_ms", "median_ms", "p75_ms"}
        assert ootb["correctness"] is None
        assert row["oracle"] is None


def test_no_metrics_suppresses_basic_fields(hipdnn) -> None:
    """``--no-metrics`` skips the always-on probes; timing still runs."""
    _, successes = _run_conv(hipdnn, metrics=MetricsConfig(basic=False))
    for r in successes:
        assert r.ootb.gpu_kernel_stats is not None
        assert r.ootb.workspace_bytes is None
        assert r.analytical_flops is None
        assert r.analytical_io_bytes is None
        assert r.ootb.derived_tflops_per_s is None
        assert r.clocks_after is None


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


def test_oracle_times_a_knob_built_plan(hipdnn, plugin_path_cli_args, tmp_path) -> None:
    """--oracle builds and times a global.benchmarking plan.

    Runs the CLI in a child process: hipDNN keeps a tuned winner in memory
    for the life of the process, so an in-process run would leave later
    tests unable to time the OOTB plan of this graph.
    """
    out = tmp_path / "oracle.json"
    proc = run_capped(
        [
            sys.executable,
            "-m",
            "dnn_benchmarking",
            "--graph",
            str(load_graph("sample_conv_fwd.json")[0]),
            "--warmup",
            "1",
            "--iters",
            "3",
            "--oracle",
            "-o",
            str(out),
            *plugin_path_cli_args,
        ],
        600,
    )
    assert out.is_file(), proc.stderr
    rows = [r for g in json.loads(out.read_text())["graphs"] for r in g["results"]]
    tuned = [r for r in rows if r["status"] == "success" and r["oracle"] is not None]
    assert tuned, [(r["engine"]["name"], r["oracle_error"]) for r in rows]

    ootb, oracle = tuned[0]["ootb"], tuned[0]["oracle"]
    assert tuned[0]["oracle_error"] is None
    # The tuned plan is the same object as the OOTB plan, plus one key.
    assert set(oracle) - set(ootb) == {"tuning_available"}
    # Both plans report their own plan build through the same timed path.
    assert ootb["build_ms"] > 0 and oracle["build_ms"] > 0
    assert oracle["kernel"]["median_ms"] > 0


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
    assert row.ootb.gpu_kernel_stats.n == row.ootb.host_stats.n == 2
    assert row.ootb.timing.timer == expected_timer()


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
    assert row.ootb.gpu_kernel_stats is None and row.ootb.host_stats is None

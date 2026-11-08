# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""AITER runtime on a live GPU: real AITER launches, timed and validated."""

import io
from pathlib import Path

import pytest

from dnn_benchmarking.config.benchmark_config import SuiteConfig, ValidationConfig
from dnn_benchmarking.graph.loader import GraphLoader
from dnn_benchmarking.reporting.reporter import Reporter
from tests.rocke_graphs import sdpa_graph

pytestmark = pytest.mark.gpu


@pytest.fixture
def aiter_session(torch_gpu):
    """The AITER session on a GPU that --runtime aiter supports, or skip."""
    from dnn_benchmarking.common.torch_support import device_arch
    from dnn_benchmarking.execution.aiter_executor import prepare_aiter_environment
    from dnn_benchmarking.execution.aiter_sdpa import AITER_ARCHES
    from dnn_benchmarking.execution.suite_runner import aiter_session

    prepare_aiter_environment()  # import aiter compiles its core module
    pytest.importorskip("aiter.ops.mha")

    arch = device_arch(torch_gpu.device("cuda"))
    if arch not in AITER_ARCHES:
        pytest.skip(f"--runtime aiter does not run on {arch}")
    return aiter_session()


_BF16_BOTTOM_RIGHT = dict(
    dtype="bfloat16", causal_mask=False, causal_mask_bottom_right=True
)


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param(
            dict(b=2, hq=16, hkv=4, s=512, d=128, dtype="bfloat16"),
            id="bf16-gqa-causal-d128",
        ),
        pytest.param(
            dict(b=2, hq=16, hkv=2, s=256, s_kv=1024, d=192, d_v=128),
            id="bf16-bottom-right-d192-v128",
        ),
        pytest.param(
            dict(b=1, hq=8, hkv=8, s=128, s_kv=4096, d=64),
            id="bf16-bottom-right-split-kv-d64",
        ),
    ],
)
def test_every_planned_aiter_row_runs_and_matches_pytorch(aiter_session, shape):
    """Each backend that the selection plans runs the graph and matches the
    reference: a backend check that admits a shape the kernel cannot run
    fails here."""
    from dnn_benchmarking.execution.aiter_sdpa import aiter_problem, backend_rows
    from dnn_benchmarking.execution.suite_runner import aiter_planner, run_graph

    graph = sdpa_graph(**{**_BF16_BOTTOM_RIGHT, **shape})
    infos = GraphLoader().extract_tensor_info(graph)
    config = SuiteConfig(
        runtime="aiter",
        warmup_iters=1,
        benchmark_iters=3,
        validation=ValidationConfig(provider="pytorch"),
    )
    reporter = Reporter(output=io.StringIO())
    planned = [
        b.row_name
        for b in backend_rows(aiter_problem(graph, infos), aiter_session.device)
    ]

    result = run_graph(
        Path("aiter.json"),
        graph,
        infos,
        config,
        reporter,
        [aiter_planner(aiter_session)],
    )

    rows = [r for r in result.results if r.role == "engine"]
    assert [r.engine_name for r in rows] == planned
    assert {"aiter:auto", "aiter:ck"} <= set(planned)
    for row in rows:
        assert row.verdict == "passed", (
            row.engine_name,
            row.error_message,
            row.ootb.correctness,
        )
        assert row.ootb.gpu_kernel_stats.median_ms > 0

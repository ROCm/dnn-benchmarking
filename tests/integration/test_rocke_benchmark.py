# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""rocKE runtime on a live gfx950: real rocKE launches, timed and validated."""

import io
from pathlib import Path

import pytest

from dnn_benchmarking.config.benchmark_config import SuiteConfig, ValidationConfig
from dnn_benchmarking.graph.loader import GraphLoader
from dnn_benchmarking.reporting.reporter import Reporter
from tests.rocke_graphs import sdpa_graph

pytestmark = pytest.mark.gpu


@pytest.fixture
def rocke_gfx950(torch_gpu, plugin_paths):
    """Plugin paths on a gfx950 host with the rocKE packages, or skip."""
    pytest.importorskip("kernels.gfx950.attention_dense")
    pytest.importorskip("dispatch.attention")
    from dnn_benchmarking.execution.rocke_executor import device_arch

    arch = device_arch(torch_gpu.device("cuda"))
    if arch != "gfx950":
        pytest.skip(f"hipDNN ships rocKE attention kernels for gfx950 only, not {arch}")
    return plugin_paths


def test_rocke_rows_rebuild_hipdnn_kernels_and_match_pytorch(rocke_gfx950) -> None:
    from dnn_benchmarking.execution.rocke_sdpa import descriptor_files
    from dnn_benchmarking.execution.suite_runner import rocke_session, run_graph_rocke

    graph = sdpa_graph(b=1, hq=32, hkv=8, s=512, d=128)
    infos = GraphLoader().extract_tensor_info(graph)
    config = SuiteConfig(
        runtime="rocke",
        plugin_paths=rocke_gfx950,
        warmup_iters=1,
        benchmark_iters=3,
        validation=ValidationConfig(provider="pytorch"),
    )
    reporter = Reporter(output=io.StringIO())
    session = rocke_session(config, reporter)
    if not descriptor_files([Path(p) for p in rocke_gfx950], "gfx950"):
        pytest.skip("the plugin build ships no rocKE descriptors")
    # Descriptors present but no usable variant: the authored join broke.
    assert session.catalog.kernels, session.catalog

    result = run_graph_rocke(
        Path("rocke.json"), graph, infos, config, reporter, session
    )

    rows = [r for r in result.results if r.role == "engine"]
    names = [r.engine_name for r in rows]
    assert any(n.startswith("rocke:attention_dense.") for n in names), names
    assert any(n.startswith("rocke:auto:") for n in names), names
    for row in rows:
        assert row.verdict == "passed", (row.engine_name, row.error_message)
        assert row.ootb.gpu_kernel_stats.median_ms > 0
        # The plugin and the rocKE packages come from one checkout, so each
        # variant row must launch hipDNN's own code object.
        assert not any("code object differs" in w for w in row.warnings or []), (
            row.engine_name,
            row.warnings,
        )


def test_hipdnn_and_rocke_rows_share_one_graph_run(hipdnn, rocke_gfx950) -> None:
    """``--runtime hipdnn,rocke``: both row sets in one process and one result.

    ``hipdnn`` comes first in the signature so hipDNN binds HIP before torch
    loads, as ``start_runtime`` does.
    """
    from dnn_benchmarking.cli.runtimes import start_runtime

    graph = sdpa_graph(b=1, hq=32, hkv=8, s=512, d=128)
    infos = GraphLoader().extract_tensor_info(graph)
    config = SuiteConfig(
        runtime="hipdnn,rocke",
        plugin_paths=rocke_gfx950,
        warmup_iters=1,
        benchmark_iters=3,
        validation=ValidationConfig(provider="pytorch"),
    )

    result = start_runtime(config, Reporter(output=io.StringIO()))(
        Path("both.json"), graph, infos
    )

    assert result.status == "ok", (result.error, result.message)
    rows = [(r.runtime, r.role, r.engine_name, r.verdict) for r in result.results]
    assert [r[:2] for r in rows].count(("pytorch", "reference")) == 1, rows
    assert ("hipdnn", "engine", "hipkernel:Gfx950AttentionDense", "passed") in rows
    rocke = [r for r in rows if r[0] == "rocke"]
    assert rocke and all(r[3] == "passed" for r in rocke), rocke

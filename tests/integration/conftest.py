# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Shared fixtures for GPU integration tests."""

from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

from dnn_benchmarking.graph import GraphLoader, TensorInfo
from tests.conftest import skip_if_no_gpu_torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GRAPHS_DIR = PROJECT_ROOT / "graphs"


def load_graph(name: str) -> Tuple[Path, Dict[str, Any], List[TensorInfo]]:
    """(path, graph JSON, tensor infos) of a shipped sample graph."""
    path = GRAPHS_DIR / name
    loader = GraphLoader()
    graph_json = loader.load_json(path)
    return path, graph_json, loader.extract_tensor_info(graph_json)


@pytest.fixture
def torch_gpu():
    """torch, or skip unless it can run a kernel on the GPU."""
    skip_if_no_gpu_torch()
    import torch

    try:
        ok = (torch.ones(4, device="cuda") * 2).sum().item() == 8.0
    except RuntimeError as e:
        pytest.skip(f"PyTorch GPU kernels do not run on this host: {e}")
    if not ok:
        pytest.skip("PyTorch GPU kernels return wrong results on this host")
    return torch


@pytest.fixture
def hipdnn(plugin_paths: List[str], torch_gpu):
    """hipdnn_frontend with the plugin paths loaded, or skip."""
    try:
        import hipdnn_frontend

        hipdnn_frontend.set_engine_plugin_paths(
            plugin_paths, hipdnn_frontend.PluginLoadingMode.ABSOLUTE
        )
        hipdnn_frontend.Handle()
    except Exception as e:
        pytest.skip(f"hipdnn_frontend not available or no GPU: {e}")
    return hipdnn_frontend

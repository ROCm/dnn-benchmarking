# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Loading the shipped sample graphs and GraphLoader error paths."""

import math
from pathlib import Path

import pytest

from dnn_benchmarking.common.exceptions import GraphLoadError
from dnn_benchmarking.graph import GraphLoader
from tests.integration.conftest import GRAPHS_DIR, load_graph


@pytest.mark.parametrize(
    "graph_path", sorted(GRAPHS_DIR.glob("*.json")), ids=lambda p: p.name
)
def test_every_sample_graph_loads(graph_path: Path) -> None:
    """Every shipped graph validates and sizes each tensor from its dtype."""
    loader = GraphLoader()
    graph_json = loader.load_json(graph_path)
    loader.validate(graph_json)
    tensor_infos = loader.extract_tensor_info(graph_json)

    assert any(ti.is_output for ti in tensor_infos)
    for ti in tensor_infos:
        if not ti.is_pass_by_value:
            assert ti.num_elements == math.prod(ti.dims)
            assert ti.size_bytes >= ti.num_elements * ti.dtype.size


@pytest.mark.parametrize(
    "graph_name,name,node_type,n_tensors,output_uid,output_dims",
    [
        (
            "sample_conv_fwd.json",
            "sample_conv_fwd_16x16x16x16_k16_3x3",
            "ConvolutionFwdAttributes",
            3,
            0,
            [16, 16, 16, 16],
        ),
        (
            "sample_matmul.json",
            "sample_matmul_256x512x1024",
            "MatmulAttributes",
            3,
            3,
            [256, 1024],
        ),
        (
            "sample_relu.json",
            "sample_relu_activation_64x128x56x56",
            "PointwiseAttributes",
            2,
            2,
            [64, 128, 56, 56],
        ),
        (
            "sample_add.json",
            "sample_pointwise_add_128x256x14x14",
            "PointwiseAttributes",
            3,
            3,
            [128, 256, 14, 14],
        ),
        (
            "sample_batchnorm.json",
            "sample_batchnorm_inference_32x64x28x28",
            "BatchnormInferenceAttributes",
            6,
            6,
            [32, 64, 28, 28],
        ),
    ],
)
def test_sample_graph_tensor_info(
    graph_name, name, node_type, n_tensors, output_uid, output_dims
) -> None:
    """Known sample graphs expose the expected name, node and output tensor."""
    _, graph_json, tensor_infos = load_graph(graph_name)

    assert graph_json["name"] == name
    assert [n["type"] for n in graph_json["nodes"]] == [node_type]
    assert len(tensor_infos) == n_tensors
    [output] = [ti for ti in tensor_infos if ti.is_output]
    assert (output.uid, output.dims) == (output_uid, output_dims)
    assert output.size_bytes == math.prod(output_dims) * 4  # float32


@pytest.mark.parametrize("graph_name", ["sample_relu.json", "sample_add.json"])
def test_sample_pointwise_graphs_include_backend_optional_keys(graph_name) -> None:
    """Pointwise JSON must include nullable fields required by hipDNN's parser."""
    _, graph_json, _ = load_graph(graph_name)
    inputs = graph_json["nodes"][0]["inputs"]
    for key in (
        "relu_lower_clip",
        "relu_upper_clip",
        "relu_lower_clip_slope",
        "axis_tensor_uid",
        "in_2_tensor_uid",
        "swish_beta",
        "elu_alpha",
        "softplus_beta",
    ):
        assert key in inputs


def test_load_nonexistent_file_raises() -> None:
    with pytest.raises(GraphLoadError, match="not found"):
        GraphLoader().load_json(Path("/nonexistent/path/graph.json"))


def test_load_invalid_json_raises(tmp_path: Path) -> None:
    invalid_json = tmp_path / "invalid.json"
    invalid_json.write_text("{ invalid json }")
    with pytest.raises(GraphLoadError, match="Invalid JSON"):
        GraphLoader().load_json(invalid_json)

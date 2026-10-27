# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Sign of generated inputs: convolution operands are signed, others are not."""

import numpy as np
import pytest

from dnn_benchmarking.execution.buffer_manager import generate_input_data
from dnn_benchmarking.graph.tensor_info import TensorInfo


def _tensor(uid: int, dims, data_type: str = "half") -> TensorInfo:
    strides = [int(np.prod(dims[i + 1 :])) for i in range(len(dims))]
    return TensorInfo(uid, f"t{uid}", list(dims), strides, data_type, is_virtual=False)


def _wrw_graph() -> dict:
    return {
        "tensors": [],
        "nodes": [
            {
                "type": "ConvolutionWrwAttributes",
                "inputs": {"x_tensor_uid": 2, "dy_tensor_uid": 1},
                "outputs": {"dw_tensor_uid": 0},
            }
        ],
    }


def test_fp16_backward_weights_sum_stays_finite() -> None:
    """A backward-weights value sums N*Ho*Wo products. With U[0, 1) data a
    sum of 300,000 terms is about 75,000, above the fp16 maximum (65504)."""
    terms = 300_000
    x, dy = _tensor(2, [terms]), _tensor(1, [terms])

    data = generate_input_data([x, dy], seed=0, graph_json=_wrw_graph())
    dw = np.float16(np.dot(data[2].astype(np.float32), data[1].astype(np.float32)))

    assert np.isfinite(dw)
    assert data[2].min() < 0 <= data[2].max() <= 1
    assert data[1].min() < 0 <= data[1].max() <= 1


@pytest.mark.parametrize(
    "node_type, inputs, output",
    [
        (
            "ConvolutionFwdAttributes",
            {"x_tensor_uid": 1, "w_tensor_uid": 2},
            "y_tensor_uid",
        ),
        (
            "ConvolutionBwdAttributes",
            {"dy_tensor_uid": 1, "w_tensor_uid": 2},
            "dx_tensor_uid",
        ),
        (
            "ConvolutionWrwAttributes",
            {"x_tensor_uid": 1, "dy_tensor_uid": 2},
            "dw_tensor_uid",
        ),
    ],
    ids=["fwd", "dgrad", "wgrad"],
)
def test_both_operands_of_each_convolution_direction_are_signed(
    node_type: str, inputs: dict, output: str
) -> None:
    graph = {
        "tensors": [],
        "nodes": [{"type": node_type, "inputs": inputs, "outputs": {output: 3}}],
    }
    a, b = _tensor(1, [4096]), _tensor(2, [4096])

    data = generate_input_data([a, b], seed=0, graph_json=graph)

    assert data[1].min() < 0 and data[2].min() < 0


def test_tensor_shared_with_a_non_convolution_node_stays_non_negative() -> None:
    """A batch-norm variance must not be negative, even in a graph that also
    has a convolution. A tensor that feeds both kinds of node keeps U[0, 1)."""
    graph = {
        "tensors": [],
        "nodes": [
            {
                "type": "ConvolutionFwdAttributes",
                "inputs": {"x_tensor_uid": 1, "w_tensor_uid": 2},
                "outputs": {"y_tensor_uid": 3},
            },
            {
                "type": "BatchnormInferenceAttributes",
                "inputs": {
                    "x_tensor_uid": 3,
                    "mean_tensor_uid": 2,
                    "inv_variance_tensor_uid": 4,
                },
                "outputs": {"y_tensor_uid": 5},
            },
        ],
    }
    x, w, variance = _tensor(1, [4096]), _tensor(2, [4096]), _tensor(4, [4096])

    data = generate_input_data([x, w, variance], seed=0, graph_json=graph)

    assert data[1].min() < 0
    assert data[2].min() >= 0
    assert data[4].min() >= 0


def test_unsigned_inputs_do_not_change_when_the_graph_has_a_convolution() -> None:
    """Signed and unsigned tensors use the same random draw, so a tensor that
    is not a convolution operand keeps the values it had before."""
    x, dy, other = _tensor(2, [64]), _tensor(1, [64]), _tensor(7, [64])

    with_graph = generate_input_data([x, dy, other], seed=3, graph_json=_wrw_graph())
    without = generate_input_data([x, dy, other], seed=3)

    np.testing.assert_array_equal(with_graph[7], without[7])


def test_pointwise_attributes_in_the_inputs_map_are_not_tensors() -> None:
    """A pointwise node keeps attributes such as ``operation`` and clip values
    in its ``inputs`` map next to the tensor UIDs."""
    graph = {
        "tensors": [],
        "nodes": [
            {
                "type": "PointwiseAttributes",
                "inputs": {
                    "in_0_tensor_uid": 1,
                    "in_1_tensor_uid": 2,
                    "operation": "relu_fwd",
                    "relu_upper_clip": 6.0,
                    "elu_alpha": None,
                },
                "outputs": {"out_0_tensor_uid": 3},
            }
        ],
    }
    a, b = _tensor(1, [64]), _tensor(2, [64])

    data = generate_input_data([a, b], seed=0, graph_json=graph)

    assert data[1].min() >= 0 and data[2].min() >= 0

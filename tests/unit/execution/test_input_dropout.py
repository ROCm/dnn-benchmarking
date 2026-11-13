# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Generated SDPA dropout inputs: a 0/1 mask at a keep rate, a scale and a
scale_inv that agree with that rate, and nonzero seed and offset values."""

import math

import numpy as np
import pytest

from dnn_benchmarking.execution.buffer_manager import generate_input_data
from dnn_benchmarking.graph.tensor_info import TensorInfo

_FORWARD = "SdpaAttributes"
_BACKWARD = "SdpaBackwardAttributes"

# Mask elements for the keep-rate checks. With n = 2**20 the 6-sigma bound at
# keep 0.9 is about 0.0018, so a fill at a different rate cannot pass.
_MASK_DIMS = [1, 1, 1024, 1024]


def _tensor(uid: int, dims, data_type: str = "float", value=None) -> TensorInfo:
    strides = [int(np.prod(dims[i + 1 :])) for i in range(len(dims))]
    return TensorInfo(
        uid, f"t{uid}", list(dims), strides, data_type, is_virtual=False, value=value
    )


def _graph(node_type: str, **dropout_uids: int) -> dict:
    inputs = {"q_tensor_uid": 1, "k_tensor_uid": 2, "v_tensor_uid": 3}
    inputs.update(dropout_uids)
    return {
        "tensors": [],
        "nodes": [{"type": node_type, "inputs": inputs, "outputs": {}}],
    }


def _assert_keep_rate(mask: np.ndarray, keep: float) -> None:
    n = mask.size
    bound = 6 * math.sqrt(keep * (1 - keep) / n) + 1 / n
    rate = float(np.count_nonzero(mask)) / n
    assert abs(rate - keep) <= bound, (rate, keep, bound)


@pytest.mark.parametrize("node_type", [_FORWARD, _BACKWARD])
@pytest.mark.parametrize("mask_type", ["float", "boolean"])
def test_mask_is_exactly_zero_or_one_at_the_keep_rate_and_repeats_for_a_seed(
    node_type: str, mask_type: str
) -> None:
    graph = _graph(node_type, dropout_mask_tensor_uid=10, dropout_scale_tensor_uid=11)
    infos = [_tensor(10, _MASK_DIMS, mask_type), _tensor(11, [1])]

    mask = generate_input_data(infos, seed=7, graph_json=graph)[10]
    again = generate_input_data(infos, seed=7, graph_json=graph)[10]
    other = generate_input_data(infos, seed=8, graph_json=graph)[10]

    if mask_type == "boolean":
        assert mask.dtype == np.bool_
    else:
        assert np.isin(mask, [0.0, 1.0]).all()
    _assert_keep_rate(mask, 0.9)
    np.testing.assert_array_equal(mask, again)
    assert not np.array_equal(mask, other)


def test_scale_and_scale_inv_follow_the_keep_rate() -> None:
    graph = _graph(
        _BACKWARD,
        dropout_mask_tensor_uid=10,
        dropout_scale_tensor_uid=11,
        dropout_scale_inv_tensor_uid=12,
    )
    mask = _tensor(10, _MASK_DIMS)

    default = generate_input_data(
        [mask, _tensor(11, [1]), _tensor(12, [1])], seed=1, graph_json=graph
    )
    np.testing.assert_allclose(default[11], [1 / 0.9], rtol=1e-6)
    np.testing.assert_allclose(default[12], [0.9], rtol=1e-6)

    given_scale = generate_input_data(
        [mask, _tensor(11, [1], value=2.0), _tensor(12, [1])], seed=1, graph_json=graph
    )
    _assert_keep_rate(given_scale[10], 0.5)
    np.testing.assert_array_equal(given_scale[12], [0.5])

    given_scale_inv = generate_input_data(
        [mask, _tensor(11, [1]), _tensor(12, [1], value=0.75)], seed=1, graph_json=graph
    )
    np.testing.assert_array_equal(given_scale_inv[12], [0.75])


@pytest.mark.parametrize("scale", [0.5, 0.0])
def test_scale_with_keep_rate_outside_0_1_is_refused(scale: float) -> None:
    graph = _graph(_FORWARD, dropout_mask_tensor_uid=10, dropout_scale_tensor_uid=11)
    infos = [_tensor(10, [1, 1, 4, 4]), _tensor(11, [1], value=scale)]

    with pytest.raises(ValueError, match="dropout_scale"):
        generate_input_data(infos, seed=0, graph_json=graph)


def test_scale_of_one_keeps_every_element() -> None:
    """A dropout_scale of 1 is a keep rate of exactly 1, the top of (0, 1]."""
    graph = _graph(_FORWARD, dropout_mask_tensor_uid=10, dropout_scale_tensor_uid=11)
    infos = [_tensor(10, [1, 1, 4, 4]), _tensor(11, [1], value=1.0)]

    mask = generate_input_data(infos, seed=0, graph_json=graph)[10]

    np.testing.assert_array_equal(mask, np.ones([1, 1, 4, 4]))


@pytest.mark.parametrize("data_type", ["int32", "int64"])
def test_seed_and_offset_are_nonzero_and_deterministic(data_type: str) -> None:
    graph = _graph(_FORWARD, seed_tensor_uid=20, offset_tensor_uid=21)
    graph["nodes"][0]["parameters"] = {"dropout_probability": 0.1}
    infos = [_tensor(20, [1], data_type), _tensor(21, [1], data_type)]

    data = generate_input_data(infos, seed=4, graph_json=graph)
    again = generate_input_data(infos, seed=4, graph_json=graph)

    for uid in (20, 21):
        assert data[uid].dtype == np.dtype(data_type)
        assert data[uid].min() >= 1
        np.testing.assert_array_equal(data[uid], again[uid])


def test_non_dropout_inputs_do_not_change_with_dropout_fill() -> None:
    """The dropout tensors sit between Q, K and V in the fill order. They draw
    from their own generator, so Q, K and V keep their values."""
    dims = [1, 2, 8, 4]
    q, k, v = (_tensor(uid, dims) for uid in (1, 2, 3))
    mask, scale = _tensor(10, [1, 2, 8, 8]), _tensor(11, [1])
    dropout_graph = _graph(
        _FORWARD, dropout_mask_tensor_uid=10, dropout_scale_tensor_uid=11
    )

    with_dropout = generate_input_data(
        [q, mask, k, scale, v], seed=9, graph_json=dropout_graph
    )
    without = generate_input_data([q, k, v], seed=9, graph_json=_graph(_FORWARD))

    for uid in (1, 2, 3):
        np.testing.assert_array_equal(with_dropout[uid], without[uid])

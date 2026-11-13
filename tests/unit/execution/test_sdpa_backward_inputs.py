# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""SDPA backward inputs: O and stats are the exact forward of Q, K and V,
and dO makes the gradients large enough to validate."""

import numpy as np
import pytest

from dnn_benchmarking.execution import sdpa_backward_inputs
from dnn_benchmarking.execution.buffer_manager import generate_input_data
from dnn_benchmarking.execution.correctness import OP_TOLERANCES, TOLERANCES
from dnn_benchmarking.graph.tensor_info import TensorInfo
from dnn_benchmarking.validation import compare


@pytest.fixture(autouse=True, params=["torch", "numpy"])
def implementation(request, monkeypatch) -> None:
    """Run every test with the PyTorch forward (on the host, as unit tests use
    no GPU) and with the NumPy fallback."""
    if request.param == "torch":
        torch = pytest.importorskip("torch")
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    else:
        monkeypatch.setattr(sdpa_backward_inputs, "_torch", lambda: None)


def _info(uid: int, dims, data_type: str) -> TensorInfo:
    strides = [int(np.prod(dims[i + 1 :])) for i in range(len(dims))]
    return TensorInfo(uid, f"t{uid}", list(dims), strides, data_type, is_virtual=False)


def _graph(parameters: dict, mask: bool = False) -> dict:
    inputs = {
        "q_tensor_uid": 1,
        "k_tensor_uid": 2,
        "v_tensor_uid": 3,
        "o_tensor_uid": 4,
        "do_tensor_uid": 5,
        "stats_tensor_uid": 6,
        "attn_mask_tensor_uid": 10 if mask else None,
    }
    return {
        "nodes": [
            {
                "type": "SdpaBackwardAttributes",
                "inputs": inputs,
                "outputs": {"dq_tensor_uid": 7, "dk_tensor_uid": 8, "dv_tensor_uid": 9},
                "parameters": parameters,
            }
        ]
    }


def _expected(data, scale, q_len, kv_len, keep, rep, mask=None):
    """Brute-force float64 forward: (O, stats) per query head."""
    q, k, v = (data[uid].astype(np.float64) for uid in (1, 2, 3))
    k, v = np.repeat(k, rep, axis=1), np.repeat(v, rep, axis=1)
    scores = q @ k.swapaxes(-1, -2) * scale
    if mask is not None:
        scores = scores + mask
    visible = np.array([[keep(i, j) for j in range(kv_len)] for i in range(q_len)])
    scores = np.where(visible, scores, -np.inf)
    stats = np.log(np.exp(scores).sum(-1, keepdims=True))
    return np.exp(scores - stats) @ v, stats


@pytest.mark.parametrize(
    "parameters, q_len, kv_len, keep",
    [
        ({"attn_scale_value": 0.3, "causal_mask": True}, 5, 5, lambda i, j: j <= i),
        (
            {"attn_scale_value": 0.3, "causal_mask_bottom_right": True},
            3,
            6,
            lambda i, j: j <= i + 3,
        ),
        (
            {"attn_scale_value": 0.3, "left_bound": 2, "right_bound": 0},
            6,
            6,
            lambda i, j: i - 1 <= j <= i,
        ),
        # hipDNN composes a deprecated causal flag with the stored alignment
        # and bounds (resolveDiagonalBand): the flag only sets right_bound 0.
        (
            {
                "attn_scale_value": 0.3,
                "causal_mask": True,
                "diagonal_alignment": "BOTTOM_RIGHT",
            },
            3,
            6,
            lambda i, j: j <= i + 3,
        ),
        (
            {"attn_scale_value": 0.3, "causal_mask": True, "left_bound": 2},
            6,
            6,
            lambda i, j: i - 1 <= j <= i,
        ),
        # An absent scale is 1.0 in hipDNN, not 1/sqrt(d).
        ({}, 4, 7, lambda i, j: True),
    ],
    ids=[
        "causal",
        "bottom-right",
        "window",
        "causal-bottom-right-alignment",
        "causal-left-bound",
        "default-scale",
    ],
)
def test_o_and_stats_are_the_forward_of_qkv(parameters, q_len, kv_len, keep) -> None:
    """GQA (4 query heads over 2 K/V heads), with each mask form."""
    infos = [
        _info(1, [2, 4, q_len, 8], "float"),
        _info(2, [2, 2, kv_len, 8], "float"),
        _info(3, [2, 2, kv_len, 8], "float"),
        _info(4, [2, 4, q_len, 8], "float"),
        _info(5, [2, 4, q_len, 8], "float"),
        _info(6, [2, 4, q_len, 1], "float"),
    ]

    data = generate_input_data(infos, seed=3, graph_json=_graph(parameters))

    scale = parameters.get("attn_scale_value", 1.0)
    o, stats = _expected(data, scale, q_len, kv_len, keep, rep=2)
    np.testing.assert_allclose(data[4], o, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(data[6], stats, rtol=1e-5, atol=1e-6)


def test_additive_mask_enters_the_forward() -> None:
    infos = [
        _info(1, [1, 2, 4, 8], "float"),
        _info(2, [1, 2, 6, 8], "float"),
        _info(3, [1, 2, 6, 8], "float"),
        _info(4, [1, 2, 4, 8], "float"),
        _info(5, [1, 2, 4, 8], "float"),
        _info(6, [1, 2, 4, 1], "float"),
        _info(10, [1, 1, 4, 6], "float"),
    ]

    data = generate_input_data(
        infos, seed=5, graph_json=_graph({"attn_scale_value": 0.5}, mask=True)
    )

    o, stats = _expected(data, 0.5, 4, 6, lambda i, j: True, rep=1, mask=data[10])
    np.testing.assert_allclose(data[4], o, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(data[6], stats, rtol=1e-5, atol=1e-6)


def test_a_row_with_no_visible_key_gets_zero_o_and_stats() -> None:
    """Bottom-right causal with Sq > Skv leaves the first Sq - Skv rows
    empty. O = 0 and stats = 0 there give P = exp(-inf - 0) = 0 in the
    backward; stats = -inf would give NaN gradients."""
    infos = [
        _info(1, [1, 2, 4, 8], "float"),
        _info(2, [1, 2, 2, 8], "float"),
        _info(3, [1, 2, 2, 8], "float"),
        _info(4, [1, 2, 4, 8], "float"),
        _info(5, [1, 2, 4, 8], "float"),
        _info(6, [1, 2, 4, 1], "float"),
    ]
    parameters = {"attn_scale_value": 0.5, "causal_mask_bottom_right": True}

    data = generate_input_data(infos, seed=7, graph_json=_graph(parameters))

    assert not data[4][:, :, :2].any() and not data[6][:, :, :2].any()
    with np.errstate(divide="ignore", invalid="ignore"):  # the empty rows
        o, stats = _expected(data, 0.5, 4, 2, lambda i, j: j <= i - 2, rep=1)
    np.testing.assert_allclose(data[4][:, :, 2:], o[:, :, 2:], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(data[6][:, :, 2:], stats[:, :, 2:], rtol=1e-5, atol=1e-6)


def test_alibi_enters_the_forward() -> None:
    """ALiBi adds slope * (key - query) per query head. With 3 heads, cuDNN's
    schedule gives 2^-4 and 2^-8 for the power-of-two part and 2^-2 for the
    extra head."""
    infos = [
        _info(1, [1, 3, 5, 8], "float"),
        _info(2, [1, 3, 5, 8], "float"),
        _info(3, [1, 3, 5, 8], "float"),
        _info(4, [1, 3, 5, 8], "float"),
        _info(5, [1, 3, 5, 8], "float"),
        _info(6, [1, 3, 5, 1], "float"),
    ]
    parameters = {"attn_scale_value": 0.5, "causal_mask": True, "alibi_mask": True}

    data = generate_input_data(infos, seed=2, graph_json=_graph(parameters))

    slopes = np.array([2.0**-4, 2.0**-8, 2.0**-2]).reshape(1, 3, 1, 1)
    distance = np.arange(5)[None, :] - np.arange(5)[:, None]
    o, stats = _expected(
        data, 0.5, 5, 5, lambda i, j: j <= i, rep=1, mask=slopes * distance
    )
    np.testing.assert_allclose(data[4], o, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(data[6], stats, rtol=1e-5, atol=1e-6)


def test_bf16_output_is_rounded_and_blocks_cover_every_row(monkeypatch) -> None:
    """O is stored at the graph type. Score blocks smaller than the query
    length must still give the full-size result."""
    monkeypatch.setattr(sdpa_backward_inputs, "_SCORE_BLOCK_ELEMENTS", 2 * 6)
    infos = [
        _info(1, [1, 1, 5, 8], "bfloat16"),
        _info(2, [1, 1, 6, 8], "bfloat16"),
        _info(3, [1, 1, 6, 8], "bfloat16"),
        _info(4, [1, 1, 5, 8], "bfloat16"),
        _info(5, [1, 1, 5, 8], "bfloat16"),
        _info(6, [1, 1, 5, 1], "float"),
    ]

    data = generate_input_data(
        infos, seed=1, graph_json=_graph({"attn_scale_value": 0.4, "causal_mask": True})
    )

    o, stats = _expected(data, 0.4, 5, 6, lambda i, j: j <= i, rep=1)
    bits = data[4].astype(np.float32).view(np.uint32)
    assert not (bits & 0xFFFF).any()  # exactly representable in bfloat16
    np.testing.assert_allclose(data[4], o, rtol=2**-8, atol=1e-6)
    np.testing.assert_allclose(data[6], stats, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("data_type", ["half", "bfloat16"])
def test_unwritten_decode_gradients_fail_validation(data_type) -> None:
    """With one query over 2,048 keys, U[0, 1) inputs put every dK and dV
    element below the fixed atol. The scaled dO must make all-zero dK and dV
    fail at the default SDPA backward tolerance, and keep dO finite."""
    q_len, kv_len, dim, scale = 1, 2048, 64, 0.125
    infos = [
        _info(1, [1, 2, q_len, dim], data_type),
        _info(2, [1, 2, kv_len, dim], data_type),
        _info(3, [1, 2, kv_len, dim], data_type),
        _info(4, [1, 2, q_len, dim], data_type),
        _info(5, [1, 2, q_len, dim], data_type),
        _info(6, [1, 2, q_len, 1], "float"),
    ]

    data = generate_input_data(
        infos, seed=3, graph_json=_graph({"attn_scale_value": scale})
    )

    q, k, v, d_out = (data[uid].astype(np.float64) for uid in (1, 2, 3, 5))
    assert np.isfinite(d_out).all()
    scores = q @ k.swapaxes(-1, -2) * scale
    probs = np.exp(scores - scores.max(-1, keepdims=True))
    probs /= probs.sum(-1, keepdims=True)
    d_v = probs.swapaxes(-1, -2) @ d_out
    d_p = d_out @ v.swapaxes(-1, -2)
    d_s = probs * (d_p - (d_p * probs).sum(-1, keepdims=True))
    d_k = d_s.swapaxes(-1, -2) @ q * scale
    rtol, atol = OP_TOLERANCES.get(
        ("SdpaBackwardAttributes", data_type), TOLERANCES[data_type]
    )
    for gradient in (d_k, d_v):
        result = compare(np.zeros_like(gradient), gradient, rtol=rtol, atol=atol)
        assert not result.passed

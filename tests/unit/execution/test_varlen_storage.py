# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Ragged and padded varlen tensors through the hipDNN storage codec."""

import dataclasses

import numpy as np
import pytest

from dnn_benchmarking.execution.buffer_manager import (
    BufferManager,
    _decode,
    _decode_storage,
    _encode_to_storage,
)
from dnn_benchmarking.graph.tensor_info import TensorInfo, VarlenLayout

B, H, D = 2, 2, 3
LENGTHS = (3, 2)
# Q-like operands in each storage width; ragged storage is addressed per element.
DTYPES = ["float", "half", "bfloat16"]


def _info(dims, strides, varlen, ragged=True, data_type="float"):
    info = TensorInfo(
        uid=1,
        name="q",
        dims=list(dims),
        strides=list(strides),
        data_type=data_type,
        is_virtual=False,
        ragged_offset_tensor_uid=9 if ragged else None,
    )
    return dataclasses.replace(info, varlen=varlen)


def _data(dims):
    return np.arange(1, int(np.prod(dims)) + 1, dtype=np.float32).reshape(dims)


def _layouts():
    """(dims, strides, starts, end) of the ragged layouts hipDNN and AOTriton use."""
    s = 4  # sequence capacity of one batch
    # Token-major: batch 1 starts after one padding token of batch 0.
    token_major = ([B, H, s, D], [0, D, H * D, 1], (0, 4 * H * D), 6 * H * D)
    # Head-major [H, T, D]: the token count sits in the head stride.
    tokens = 6
    head_major = (
        [B, H, s, D],
        [0, tokens * D, D, 1],
        (0, 4 * D),
        tokens * D,
    )
    # Q chunk of a fused QKV tensor: the token stride spans Q, K and V.
    fused = ([B, H, s, D], [0, D, 3 * H * D, 1], (0, 4 * 3 * H * D), 6 * 3 * H * D)
    return {"token_major": token_major, "head_major": head_major, "fused_qkv": fused}


def _valid_positions(info):
    """``(storage index, logical index)`` of every valid position, by the
    address formula start + h*s1 + t*s2 + d*s3."""
    s1, s2, s3 = info.strides[1:]
    for b, (start, length) in enumerate(zip(info.varlen.starts, info.varlen.lengths)):
        for h in range(info.dims[1]):
            for t in range(length):
                for d in range(info.dims[3]):
                    yield start + h * s1 + t * s2 + d * s3, (b, h, t, d)


@pytest.mark.parametrize("data_type", DTYPES)
@pytest.mark.parametrize("layout", list(_layouts()))
def test_ragged_encode_places_valid_positions_by_the_address_formula(layout, data_type):
    dims, strides, starts, end = _layouts()[layout]
    info = _info(dims, strides, VarlenLayout(LENGTHS, starts, end), data_type=data_type)
    data = _data(dims)  # small integers: exact in every DTYPES entry

    words = _encode_to_storage(data, info)

    assert words.size == info.storage_elements
    assert info.size_bytes == words.nbytes
    values = _decode(words, info.dtype)
    addressed = np.zeros(words.size, dtype=bool)
    for index, position in _valid_positions(info):
        assert values[index] == data[position], (index, position)
        addressed[index] = True
    # Padding and gaps hold generated values, as in a padded tensor, not zeros.
    assert not addressed.all(), "the layout has no padding; the case checks nothing"
    assert np.all(values[~addressed] != 0)


@pytest.mark.parametrize("data_type", DTYPES)
@pytest.mark.parametrize("layout", list(_layouts()))
def test_ragged_decode_returns_valid_positions_and_zeros_elsewhere(layout, data_type):
    dims, strides, starts, end = _layouts()[layout]
    info = _info(dims, strides, VarlenLayout(LENGTHS, starts, end), data_type=data_type)
    data = _data(dims)

    out = _decode_storage(_encode_to_storage(data, info), info)

    expected = np.zeros(dims, dtype=np.float32)
    for b, length in enumerate(LENGTHS):
        expected[b, :, :length, :] = data[b, :, :length, :]
    assert out.shape == tuple(dims)
    np.testing.assert_array_equal(out, expected)


@pytest.mark.parametrize("layout", ["token_major", "fused_qkv"])
def test_ragged_storage_reaches_a_trailing_gap(layout):
    """The last batch ends two tokens before the table's end entry: the
    buffer must reach the end entry, not stop at the last valid element."""
    dims, strides, starts, _ = _layouts()[layout]
    end = starts[1] + (LENGTHS[1] + 2) * strides[2]
    info = _info(dims, strides, VarlenLayout(LENGTHS, starts, end))

    assert info.storage_elements == end
    assert _encode_to_storage(_data(dims), info).size == end


def test_ragged_storage_ignores_batch_stride():
    dims, strides, starts, end = _layouts()["token_major"]
    layout = VarlenLayout(LENGTHS, starts, end)
    data = _data(dims)
    first = _encode_to_storage(data, _info(dims, strides, layout))
    strides[0] = 12345
    second = _encode_to_storage(data, _info(dims, strides, layout))
    np.testing.assert_array_equal(first, second)


def test_padded_varlen_tensor_keeps_the_dense_codec():
    dims = [B, H, 4, D]
    strides = [H * 4 * D, 4 * D, D, 1]
    info = _info(dims, strides, VarlenLayout(LENGTHS), ragged=False)
    data = _data(dims)

    storage = _encode_to_storage(data, info)

    # Padding positions hold the generated values; engines must ignore them.
    np.testing.assert_array_equal(storage, data.reshape(-1))
    np.testing.assert_array_equal(_decode_storage(storage, info), data)


def test_ragged_output_has_no_torch_view():
    dims, strides, starts, end = _layouts()["token_major"]
    info = _info(dims, strides, VarlenLayout(LENGTHS, starts, end))
    manager = BufferManager([info], device="cpu")
    manager._buffers[info.uid] = object()  # never read: the lookup refuses first
    assert manager.get_output_tensor(info.uid) is None

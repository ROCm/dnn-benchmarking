# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Ragged and padded varlen tensors through the hipDNN storage codec."""

import dataclasses

import numpy as np
import pytest

from dnn_benchmarking.execution.buffer_manager import (
    BufferManager,
    _decode_storage,
    _encode_to_storage,
)
from dnn_benchmarking.graph.tensor_info import TensorInfo, VarlenLayout

B, H, D = 2, 2, 3
LENGTHS = (3, 2)


def _info(dims, strides, varlen, ragged=True):
    info = TensorInfo(
        uid=1,
        name="q",
        dims=list(dims),
        strides=list(strides),
        data_type="float",
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


def _expected(info, data):
    """Storage built with the address formula start + h*s1 + t*s2 + d*s3."""
    storage = np.zeros(info.storage_elements, dtype=np.float32)
    s1, s2, s3 = info.strides[1:]
    for b, (start, length) in enumerate(zip(info.varlen.starts, info.varlen.lengths)):
        for h in range(info.dims[1]):
            for t in range(length):
                for d in range(info.dims[3]):
                    storage[start + h * s1 + t * s2 + d * s3] = data[b, h, t, d]
    return storage


@pytest.mark.parametrize("layout", list(_layouts()))
def test_ragged_encode_places_valid_positions_by_the_address_formula(layout):
    dims, strides, starts, end = _layouts()[layout]
    info = _info(dims, strides, VarlenLayout(LENGTHS, starts, end))
    data = _data(dims)

    storage = _encode_to_storage(data, info)

    # Equality with a zero-initialised expectation also proves that nothing
    # outside the valid tokens is written.
    np.testing.assert_array_equal(storage, _expected(info, data))
    assert storage.size == info.storage_elements
    assert info.size_bytes == storage.size * 4
    assert info.size_bytes >= (max(np.flatnonzero(storage)) + 1) * 4


@pytest.mark.parametrize("layout", list(_layouts()))
def test_ragged_decode_returns_valid_positions_and_zeros_elsewhere(layout):
    dims, strides, starts, end = _layouts()[layout]
    info = _info(dims, strides, VarlenLayout(LENGTHS, starts, end))
    data = _data(dims)

    out = _decode_storage(_encode_to_storage(data, info), info)

    expected = np.zeros(dims, dtype=np.float32)
    for b, length in enumerate(LENGTHS):
        expected[b, :, :length, :] = data[b, :, :length, :]
    assert out.shape == tuple(dims)
    np.testing.assert_array_equal(out, expected)


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

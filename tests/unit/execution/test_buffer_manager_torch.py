# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for the torch storage backend of BufferManager (CPU torch is enough)."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from dnn_benchmarking.execution.buffer_manager import (
    BufferManager,
    _encode_bfloat16_dense_to_storage_bytes,
)
from dnn_benchmarking.graph.tensor_info import TensorInfo

torch = pytest.importorskip("torch")


def _strided(uid: int, data_type: str, is_output: bool) -> TensorInfo:
    # Row stride 4 over 3 columns leaves one padding element per row.
    return TensorInfo(
        uid=uid,
        name=f"t{uid}",
        dims=[2, 3],
        strides=[4, 1],
        data_type=data_type,
        is_virtual=False,
        is_output=is_output,
    )


class TestTorchBackend:
    def test_typed_views_match_host_decoding(self) -> None:
        x = _strided(1, "float", is_output=False)
        y = _strided(2, "bfloat16", is_output=True)
        bm = BufferManager([x, y], device="cpu")
        bm.allocate_all()
        bm.load_input_data({x.uid: np.arange(6, dtype=np.float32).reshape(2, 3)})
        out = np.array([[1.0, -2.5, 3.25], [0.5, 4.0, -6.0]], dtype=np.float32)
        bm._write_bytes(
            bm._buffers[y.uid], _encode_bfloat16_dense_to_storage_bytes(out, y)
        )

        x_view = bm.get_output_tensor(x.uid)
        y_view = bm.get_output_tensor(y.uid)

        assert x_view.dtype == torch.float32
        assert y_view.dtype == torch.bfloat16
        assert torch.equal(x_view, torch.from_numpy(bm.get_input_data(x.uid)))
        assert torch.equal(y_view.float(), torch.from_numpy(bm.get_output_data(y.uid)))
        assert torch.equal(y_view.float(), torch.from_numpy(out))

    def test_variant_pack_holds_torch_storage(self) -> None:
        x = _strided(1, "float", is_output=False)
        bm = BufferManager([x], device="cpu")
        bm.allocate_all()

        pack = bm.create_variant_pack()

        assert list(pack) == [x.uid]
        assert pack[x.uid] is bm._buffers[x.uid]

    def test_zero_outputs_clears_torch_storage(self) -> None:
        y = _strided(2, "float", is_output=True)
        bm = BufferManager([y], device="cpu")
        bm.allocate_all()
        bm._buffers[y.uid].fill_(7)

        bm.zero_outputs()

        assert not bm._buffers[y.uid].any()


class TestDeviceBufferBackend:
    def test_get_output_tensor_is_none(self) -> None:
        y = _strided(2, "float", is_output=True)
        bm = BufferManager([y])
        buffer = MagicMock()
        buffer.ptr.return_value = 1234
        bm._buffers[y.uid] = buffer

        assert bm.get_output_tensor(y.uid) is None
        assert bm.create_variant_pack() == {y.uid: 1234}

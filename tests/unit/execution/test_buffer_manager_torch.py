# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for the torch storage backend of BufferManager (CPU torch is enough)."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from dnn_benchmarking.common.exceptions import ExecutionError
from dnn_benchmarking.execution.buffer_manager import (
    BufferManager,
    _encode_to_storage,
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


def _dump(uid: int, data_type: str = "float") -> TensorInfo:
    # A packed [B, Hq, Sq, Skv] rng_dump output.
    return TensorInfo(
        uid=uid,
        name=f"t{uid}",
        dims=[1, 2, 2, 3],
        strides=[12, 6, 3, 1],
        data_type=data_type,
        is_virtual=False,
        is_output=True,
    )


class TestTorchBackend:
    def test_typed_views_match_host_decoding(self) -> None:
        x = _strided(1, "float", is_output=False)
        y = _strided(2, "bfloat16", is_output=True)
        bm = BufferManager([x, y], device="cpu")
        bm.allocate_all()
        x_data = np.arange(6, dtype=np.float32).reshape(2, 3)
        bm.load_input_data({x.uid: x_data})
        out = np.array([[1.0, -2.5, 3.25], [0.5, 4.0, -6.0]], dtype=np.float32)
        bm._write_storage(bm._buffers[y.uid], _encode_to_storage(out, y))

        x_view = bm.get_output_tensor(x.uid)
        y_view = bm.get_output_tensor(y.uid)

        assert x_view.dtype == torch.float32
        assert y_view.dtype == torch.bfloat16
        assert torch.equal(x_view, torch.from_numpy(x_data))
        assert torch.equal(y_view.float(), torch.from_numpy(bm.get_output_data(y.uid)))
        assert torch.equal(y_view.float(), torch.from_numpy(out))

    def test_variant_pack_holds_torch_pointers(self) -> None:
        x = _strided(1, "float", is_output=False)
        bm = BufferManager([x], device="cpu")
        bm.allocate_all()

        pack = bm.create_variant_pack()

        assert pack == {x.uid: bm._buffers[x.uid].data_ptr()}

    def test_zero_outputs_clears_torch_storage(self) -> None:
        y = _strided(2, "float", is_output=True)
        bm = BufferManager([y], device="cpu")
        bm.allocate_all()
        bm._buffers[y.uid].fill_(7)

        bm.zero_outputs()

        assert not bm._buffers[y.uid].any()

    def test_zero_outputs_syncs_torch_before_hipdnn_runs(self, monkeypatch) -> None:
        """zero_() runs on torch's stream and hipDNN on the handle stream; the
        device sync is the only ordering between them."""
        y = _strided(2, "float", is_output=True)
        bm = BufferManager([y], device="cuda")
        calls = []
        bm._buffers[y.uid] = MagicMock(zero_=lambda: calls.append("zero_"))
        monkeypatch.setattr(torch.cuda, "synchronize", lambda: calls.append("sync"))

        bm.zero_outputs()

        assert calls == ["zero_", "sync"]

    def test_zero_outputs_fills_listed_outputs_with_nan(self) -> None:
        x = _strided(1, "float", is_output=False)
        dump = _dump(2)
        y = _strided(3, "float", is_output=True)
        bm = BufferManager([x, dump, y], device="cpu")
        bm.allocate_all()
        x_data = np.arange(6, dtype=np.float32).reshape(2, 3)
        bm.load_input_data({x.uid: x_data})
        bm._buffers[dump.uid].fill_(7)
        bm._buffers[y.uid].fill_(7)

        bm.zero_outputs(nan_uids={dump.uid})

        assert np.isnan(bm.get_output_data(dump.uid)).all()
        assert not bm.get_output_data(y.uid).any()
        np.testing.assert_array_equal(bm.get_output_data(x.uid), x_data)


class TestDeviceBufferBackend:
    def test_get_output_tensor_is_none(self) -> None:
        y = _strided(2, "float", is_output=True)
        bm = BufferManager([y])
        buffer = MagicMock()
        buffer.ptr.return_value = 1234
        bm._buffers[y.uid] = buffer

        assert bm.get_output_tensor(y.uid) is None
        assert bm.create_variant_pack() == {y.uid: 1234}

    @pytest.mark.parametrize(
        "call",
        [
            lambda bm: bm.create_variant_pack(),
            lambda bm: bm.load_input_data({}),
            lambda bm: bm.zero_outputs(),
        ],
        ids=["create_variant_pack", "load_input_data", "zero_outputs"],
    )
    def test_use_before_allocate_all_raises(self, call) -> None:
        bm = BufferManager([_strided(1, "float", is_output=True)])

        with pytest.raises(ExecutionError, match="Buffers not allocated"):
            call(bm)

    def test_zero_outputs_sends_one_nan_bytes_to_each_listed_buffer(self) -> None:
        """DeviceBuffer takes only ``bytes``: the listed buffer gets one NaN
        pattern of its full size, and every other output is zeroed."""
        dump = _dump(2)
        y = _dump(3)
        bm = BufferManager([dump, y])
        dump_buffer, y_buffer = MagicMock(), MagicMock()
        bm._buffers.update({dump.uid: dump_buffer, y.uid: y_buffer})

        bm.zero_outputs(nan_uids={dump.uid})

        dump_buffer.copy_from_host.assert_called_once()
        (data,), _ = dump_buffer.copy_from_host.call_args
        assert isinstance(data, bytes) and len(data) == dump.size_bytes
        assert np.isnan(np.frombuffer(data, dtype="<f4")).all()
        y_buffer.copy_from_host.assert_not_called()
        y_buffer.zeros.assert_called_once()


@pytest.mark.parametrize("device", [None, "cpu"], ids=["device_buffer", "torch"])
def test_zero_outputs_nan_fill_covers_strided_padding(device) -> None:
    """A dump with non-packed strides is filled over its whole storage
    (7 words: 6 elements and 1 padding word), not over its 6 elements."""
    dump = _strided(2, "float", is_output=True)
    bm = BufferManager([dump], device=device)
    written = []
    if device is None:
        bm._buffers[dump.uid] = MagicMock(copy_from_host=written.append)
    else:
        bm.allocate_all()
        bm._buffers[dump.uid].fill_(7)

    bm.zero_outputs(nan_uids={dump.uid})

    if device is None:
        assert len(written) == 1
        words = np.frombuffer(written[0], dtype="<f4")
    else:
        words = bm._buffers[dump.uid].view(torch.float32).numpy()
    assert words.size * 4 == dump.size_bytes
    assert np.isnan(words).all()


def test_zero_outputs_rejects_a_listed_output_that_is_not_float() -> None:
    dump = _dump(2, "bfloat16")
    bm = BufferManager([dump], device="cpu")
    bm.allocate_all()

    with pytest.raises(ValueError):
        bm.zero_outputs(nan_uids={dump.uid})

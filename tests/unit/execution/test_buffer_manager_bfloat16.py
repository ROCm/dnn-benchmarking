# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for buffer_manager storage encoding and input generation.

The bf16 encoder is round-to-nearest, ties-to-even (RNE), matching
``torch.Tensor.bfloat16()``; NaN inputs stay NaN with the bf16 quiet bit
forced on.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest

from dnn_benchmarking.common.dtypes import get_dtype
from dnn_benchmarking.common.exceptions import ExecutionError
from dnn_benchmarking.execution.buffer_manager import (
    BufferManager,
    _decode,
    _decode_storage_bytes,
    _encode_to_storage_bytes,
    _f32_to_bf16,
    generate_input_data,
)
from dnn_benchmarking.graph.tensor_info import TensorInfo

BF16 = get_dtype("bfloat16")


def _bf16_roundtrip(x: np.ndarray) -> np.ndarray:
    return _decode(_f32_to_bf16(x), BF16)


def _tensor(uid, data_type="float", dims=(2, 3), strides=(), **extra) -> TensorInfo:
    return TensorInfo(
        uid=uid,
        name=f"t{uid}",
        dims=list(dims),
        strides=list(strides),
        data_type=data_type,
        is_virtual=False,
        **extra,
    )


class TestF32ToBf16RoundtripRNE:
    """Roundtrip and exactness tests for the RNE conversion."""

    def test_values_with_zero_low_bits_roundtrip_exactly(self) -> None:
        x = np.array([1.0, -1.0, 0.0, 0.5, 2.0, 4.0, 100.0, -7.5], dtype=np.float32)
        np.testing.assert_array_equal(_bf16_roundtrip(x), x)

    @pytest.mark.parametrize(
        "f32_bits, bf16_word",
        [
            (0x3F8080FF, 0x3F81),  # low half-word > 0x8000: round up
            (0x3F807FFF, 0x3F80),  # low half-word < 0x8000: round down
            (0x3F808000, 0x3F80),  # exact tie, even LSB: stays
            (0x3F818000, 0x3F82),  # exact tie, odd LSB: rounds up to even
        ],
    )
    def test_rounding(self, f32_bits: int, bf16_word: int) -> None:
        x = np.array([f32_bits], dtype=np.uint32).view(np.float32)
        assert _f32_to_bf16(x)[0] == bf16_word

    def test_zero_and_negative_zero_preserved(self) -> None:
        result = _bf16_roundtrip(np.array([0.0, -0.0], dtype=np.float32))
        assert result[0] == 0.0 and result[1] == 0.0
        assert not np.signbit(result[0])
        assert np.signbit(result[1])

    def test_smallest_f32_subnormal_rounds_to_zero(self) -> None:
        x = np.array([0x00000001], dtype=np.uint32).view(np.float32)
        assert _bf16_roundtrip(x)[0] == 0.0


class TestBf16SpecialValues:
    @pytest.mark.parametrize(
        "word, check",
        [
            (0x7F80, lambda v: np.isinf(v) and v > 0),
            (0xFF80, lambda v: np.isinf(v) and v < 0),
            (0x7FC0, np.isnan),
        ],
    )
    def test_decode(self, word: int, check) -> None:
        assert check(_decode(np.array([word], dtype=np.uint16), BF16)[0])

    @pytest.mark.parametrize("f32_bits", [0x7FC00000, 0x7F800001])
    def test_nan_input_encodes_to_nan(self, f32_bits: int) -> None:
        """A signalling NaN whose payload is only in the low 16 bits would
        truncate to inf without the forced quiet bit."""
        x = np.array([f32_bits], dtype=np.uint32).view(np.float32)
        assert np.isnan(_bf16_roundtrip(x)[0])


class TestTorchParity:
    def test_rounding_cases_match_torch_bfloat16(self) -> None:
        torch = pytest.importorskip("torch")
        bits = np.array(
            [0x3F8080FF, 0x3F807FFF, 0x3F808000, 0x3F818000, 0x3F800000],
            dtype=np.uint32,
        )
        x = bits.view(np.float32)

        torch_words = torch.from_numpy(x.copy()).bfloat16().view(torch.int16)

        np.testing.assert_array_equal(
            _f32_to_bf16(x), torch_words.numpy().view(np.uint16)
        )


class TestStridedStorage:
    def test_size_bytes_uses_last_addressed_element(self) -> None:
        tensor = _tensor(11, strides=(4, 1))
        # Addresses touched: (0,0)..(1,2) => max offset 1*4 + 2*1 = 6.
        assert tensor.storage_elements == 7
        assert tensor.size_bytes == 28

    def test_load_input_data_copies_strided_storage(self) -> None:
        tensor = _tensor(24, strides=(4, 1))
        data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        buffer_manager = BufferManager([tensor])
        mock_buffer = MagicMock()
        buffer_manager._buffers[tensor.uid] = mock_buffer

        buffer_manager.load_input_data({tensor.uid: data})

        raw = mock_buffer.copy_from_host.call_args[0][0]
        storage = np.frombuffer(raw, dtype=np.float32)
        assert len(raw) == tensor.size_bytes
        np.testing.assert_array_equal(storage[0:3], data[0])
        np.testing.assert_array_equal(storage[4:7], data[1])

    def test_get_output_data_returns_contiguous_dense_array(self) -> None:
        tensor = _tensor(15, strides=(4, 1), is_output=True)
        storage = np.array([1.0, 2.0, 3.0, -99.0, 4.0, 5.0, 6.0], dtype=np.float32)
        mock_buffer = MagicMock()
        mock_buffer.copy_to_host.return_value = storage.tobytes()
        buffer_manager = BufferManager([tensor])
        buffer_manager._buffers[tensor.uid] = mock_buffer

        output = buffer_manager.get_output_data(tensor.uid)

        assert output.flags.c_contiguous
        np.testing.assert_array_equal(
            output, np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        )

    def test_bfloat16_strided_storage_roundtrips_with_padding(self) -> None:
        tensor = _tensor(14, "bfloat16", strides=(4, 1))
        data = np.array(
            [[1.00390625, 1.0078125, -2.25], [0.33398438, -0.0, 4.5]],
            dtype=np.float32,
        )

        raw = _encode_to_storage_bytes(data, tensor)

        storage = np.frombuffer(raw, dtype=np.uint16)
        words = _f32_to_bf16(data)
        assert len(raw) == tensor.size_bytes
        np.testing.assert_array_equal(storage[0:3], words[0])
        assert storage[3] == 0
        np.testing.assert_array_equal(storage[4:7], words[1])
        np.testing.assert_array_equal(
            _decode_storage_bytes(raw, tensor), _decode(words, BF16)
        )

    def test_bfloat16_get_output_data_decodes_device_bytes(self) -> None:
        tensor = _tensor(9, "bfloat16", is_output=True)
        words = np.array(
            [0x3F80, 0x4000, 0xBF80, 0x0000, 0x8000, 0x7F80], dtype=np.uint16
        )
        mock_buffer = MagicMock()
        mock_buffer.copy_to_host.return_value = words.tobytes()
        buffer_manager = BufferManager([tensor])
        buffer_manager._buffers[tensor.uid] = mock_buffer

        output = buffer_manager.get_output_data(tensor.uid)

        assert output.dtype == np.float32
        np.testing.assert_array_equal(
            output,
            np.array([[1.0, 2.0, -1.0], [0.0, -0.0, np.inf]], dtype=np.float32),
        )


class TestLoadInputData:
    def test_device_bytes_decode_to_the_generated_values(self) -> None:
        tensors = [_tensor(1, "bfloat16"), _tensor(2, "half"), _tensor(3, "fp8_e4m3")]
        input_data = generate_input_data(tensors, seed=101)
        buffer_manager = BufferManager(tensors)
        buffers = {t.uid: MagicMock() for t in tensors}
        buffer_manager._buffers.update(buffers)

        buffer_manager.load_input_data(input_data)

        for t in tensors:
            raw = buffers[t.uid].copy_from_host.call_args[0][0]
            np.testing.assert_array_equal(
                _decode_storage_bytes(raw, t), input_data[t.uid]
            )

    def test_pass_by_value_scalar_has_no_device_buffer(self) -> None:
        scalar = _tensor(23, "double", dims=(1,), strides=(1,), value=1e-5)
        other = _tensor(24)
        input_data = generate_input_data([scalar, other], seed=1)
        buffer_manager = BufferManager([scalar, other])
        buffer_manager._buffers[other.uid] = MagicMock()

        buffer_manager.load_input_data(input_data)

        np.testing.assert_array_equal(
            input_data[scalar.uid], np.asarray([1e-5], dtype=np.float64)
        )
        assert scalar.uid not in buffer_manager._buffers

    def test_missing_input_raises(self) -> None:
        tensor = _tensor(5)
        buffer_manager = BufferManager([tensor])
        buffer_manager._buffers[tensor.uid] = MagicMock()

        with pytest.raises(ExecutionError, match="UID 5"):
            buffer_manager.load_input_data({})


class TestGenerateInputData:
    def test_same_seed_same_data_and_different_seed_differs(self) -> None:
        tensor = _tensor(21, "bfloat16", dims=(8, 8))

        first = generate_input_data([tensor], seed=123)[tensor.uid]
        second = generate_input_data([tensor], seed=123)[tensor.uid]
        other = generate_input_data([tensor], seed=124)[tensor.uid]

        np.testing.assert_array_equal(first, second)
        assert not np.array_equal(first, other)

    @pytest.mark.parametrize(
        "data_type", ["float", "half", "bfloat16", "double", "fp8_e4m3", "fp8_e5m2"]
    )
    def test_float_inputs_are_unit_interval_in_logical_dtype(self, data_type) -> None:
        tensor = _tensor(1, data_type, dims=(64, 64))

        data = generate_input_data([tensor], seed=0)[tensor.uid]

        assert data.dtype == tensor.dtype.numpy
        assert data.shape == (64, 64)
        # Non-negative (batch-norm variances stay valid), rounded values <= 1.
        assert data.min() >= 0.0 and data.max() <= 1.0
        assert len(np.unique(data)) > 1

    @pytest.mark.parametrize(
        "data_type", ["int8", "uint8", "int32", "int64", "boolean"]
    )
    def test_integer_inputs_are_zero_so_indices_stay_in_bounds(self, data_type) -> None:
        tensor = _tensor(1, data_type, dims=(32, 32))

        data = generate_input_data([tensor], seed=0)[tensor.uid]

        assert data.dtype == tensor.dtype.numpy
        assert not data.any(), "non-zero offsets/indices can read out of bounds"

    def test_outputs_and_virtuals_get_no_input(self) -> None:
        out = _tensor(1, is_output=True)
        virtual = TensorInfo(2, "v", [2], [], "float", is_virtual=True)

        assert generate_input_data([out, virtual], seed=0) == {}


class TestPyTorchHostNumpy:
    def test_strided_and_bfloat16_tensors_become_dense_numpy(self) -> None:
        torch = pytest.importorskip("torch")
        from dnn_benchmarking.execution.pytorch_buffer_manager import host_numpy

        tensor = torch.empty_strided((2, 3), (4, 1), dtype=torch.bfloat16)
        tensor.copy_(torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))

        output = host_numpy(tensor)

        assert output.flags.c_contiguous
        assert output.dtype == np.float32
        np.testing.assert_array_equal(
            output, np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        )


class TestPagedInputGeneration:
    """Page tables and sequence lengths are integer inputs with invariants
    random data cannot satisfy: a page table must address distinct pages in
    the cache, and a length must be positive and fit its page allocation."""

    PAGE_SIZE = 64
    NUM_PAGES = 128
    NUM_SEQS = 2
    BLOCKS_PER_SEQ = 32

    def _graph(self):
        return {
            "tensors": [
                {
                    "uid": 2,
                    "name": "K",
                    "dims": [self.NUM_PAGES, 4, self.PAGE_SIZE, 128],
                },
            ],
            "nodes": [
                {
                    "type": "SdpaAttributes",
                    "inputs": {
                        "q_tensor_uid": 1,
                        "k_tensor_uid": 2,
                        "v_tensor_uid": 3,
                        "page_table_k_tensor_uid": 5,
                        "page_table_v_tensor_uid": 6,
                        "seq_len_q_tensor_uid": 7,
                        "seq_len_kv_tensor_uid": 8,
                    },
                    "outputs": {"o_tensor_uid": 4},
                    "attributes": {},
                }
            ],
        }

    def _int_tensor(self, uid, dims):
        return _tensor(uid, "int32", dims=dims, strides=[1] * len(dims))

    def test_page_table_spans_multiple_pages_within_the_cache(self) -> None:
        table = self._int_tensor(5, [self.NUM_SEQS, self.BLOCKS_PER_SEQ])
        data = generate_input_data([table], seed=0, graph_json=self._graph())[5]
        assert data.dtype == np.int32
        assert data.max() > 0, "every sequence would read page 0"
        assert data.max() < self.NUM_PAGES, "page id addresses beyond the cache"

    def test_lengths_are_positive_and_fit_the_page_allocation(self) -> None:
        lengths = self._int_tensor(8, [self.NUM_SEQS])
        data = generate_input_data([lengths], seed=0, graph_json=self._graph())[8]
        assert data.min() > 0, "a zero length is an empty sequence"
        assert data.max() <= self.BLOCKS_PER_SEQ * self.PAGE_SIZE

    def test_dense_graph_inputs_are_untouched(self) -> None:
        """A graph with no page table generates exactly what it would without
        the graph."""
        tensor = _tensor(31, "bfloat16")
        dense_graph = {
            "tensors": [],
            "nodes": [
                {
                    "type": "SdpaAttributes",
                    "inputs": {"q_tensor_uid": 1, "page_table_k_tensor_uid": None},
                    "outputs": {"o_tensor_uid": 4},
                }
            ],
        }
        with_graph = generate_input_data([tensor], seed=7, graph_json=dense_graph)
        without = generate_input_data([tensor], seed=7)
        np.testing.assert_array_equal(with_graph[tensor.uid], without[tensor.uid])

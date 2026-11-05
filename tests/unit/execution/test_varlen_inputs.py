# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Generated lengths, cumulative lengths and ragged offset tables of
variable-length SDPA graphs."""

import numpy as np
import pytest

from dnn_benchmarking.common.exceptions import UnsupportedGraphError
from dnn_benchmarking.execution.buffer_manager import generate_input_data
from dnn_benchmarking.execution.varlen_inputs import generate_varlen_inputs
from dnn_benchmarking.graph.tensor_info import TensorInfo
from dnn_benchmarking.graph.varlen import resolve_varlen_layouts

SEEDS = range(12)


def _tensor_json(uid, name, dims, strides=None, data_type="bfloat16", **extra):
    if strides is None:
        strides = [int(np.prod(dims[i + 1 :])) for i in range(len(dims))]
    return {
        "uid": uid,
        "name": name,
        "dims": list(dims),
        "strides": list(strides),
        "data_type": data_type,
        "virtual": False,
        **extra,
    }


def _info(uid, name, dims, data_type="int32"):
    strides = [int(np.prod(dims[i + 1 :])) for i in range(len(dims))]
    return TensorInfo(uid, name, list(dims), strides, data_type, is_virtual=False)


def _token_major(batch, heads, seq, head_dim):
    """Strides of a token-major ``[B, H, S, D]`` view (strides[0] unused)."""
    return [seq * heads * head_dim, head_dim, heads * head_dim, 1]


def _node(inputs, outputs=None, attributes=None):
    return {
        "type": "SdpaAttributes",
        "inputs": inputs,
        "outputs": outputs or {"o_tensor_uid": 4},
        "attributes": attributes or {},
    }


def _graph(tensors, *nodes):
    return {"tensors": list(tensors), "nodes": list(nodes)}


def _dense_varlen(batch=4, seq=16, heads=2, head_dim=8):
    """Dense padded Q/K/V/O with seq_len_q (uid 7) and seq_len_kv (uid 8)."""
    dims = [batch, heads, seq, head_dim]
    graph = _graph(
        [
            _tensor_json(1, "Q", dims),
            _tensor_json(2, "K", dims),
            _tensor_json(3, "V", dims),
            _tensor_json(4, "O", dims),
            _tensor_json(7, "SEQ_LEN_Q", [batch], data_type="int32"),
            _tensor_json(8, "SEQ_LEN_KV", [batch], data_type="int32"),
        ],
        _node(
            {
                "q_tensor_uid": 1,
                "k_tensor_uid": 2,
                "v_tensor_uid": 3,
                "seq_len_q_tensor_uid": 7,
                "seq_len_kv_tensor_uid": 8,
            }
        ),
    )
    infos = [_info(7, "SEQ_LEN_Q", [batch]), _info(8, "SEQ_LEN_KV", [batch])]
    return graph, infos


def _ragged_graph(
    batch=4,
    seq=16,
    heads=2,
    head_dim=8,
    with_seq_len=True,
    multiplier=None,
    strides=None,
):
    """Token-major ragged Q/K/V/O sharing one offset table (uid 9)."""
    dims = [batch, heads, seq, head_dim]
    strides = strides or _token_major(batch, heads, seq, head_dim)
    ragged = {
        "ragged_offset_tensor_uid": 9,
        "ragged_offset_multiplier": multiplier or heads * head_dim,
    }
    tensors = [
        _tensor_json(uid, name, dims, strides, **ragged)
        for uid, name in ((1, "Q"), (2, "K"), (3, "V"), (4, "O"))
    ]
    tensors.append(_tensor_json(9, "OFFSETS", [batch + 1, 1, 1, 1], data_type="int32"))
    inputs = {"q_tensor_uid": 1, "k_tensor_uid": 2, "v_tensor_uid": 3}
    infos = [_info(9, "OFFSETS", [batch + 1, 1, 1, 1])]
    if with_seq_len:
        tensors.append(_tensor_json(7, "SEQ_LEN_Q", [batch], data_type="int32"))
        tensors.append(_tensor_json(8, "SEQ_LEN_KV", [batch], data_type="int32"))
        inputs["seq_len_q_tensor_uid"] = 7
        inputs["seq_len_kv_tensor_uid"] = 8
        infos += [_info(7, "SEQ_LEN_Q", [batch]), _info(8, "SEQ_LEN_KV", [batch])]
    return _graph(tensors, _node(inputs)), infos


class TestLengths:
    @pytest.mark.parametrize("batch", [2, 4, 7])
    def test_lengths_fit_the_operand_vary_and_include_padding(self, batch) -> None:
        for seed in SEEDS:
            graph, infos = _dense_varlen(batch=batch, seq=16)
            lengths = generate_varlen_inputs(graph, infos, seed)[7]
            assert lengths.dtype == np.int32
            assert lengths.shape == (batch,)
            assert lengths.min() >= 1 and lengths.max() <= 16
            assert lengths.min() < 16, "no padded batch"
            assert lengths.max() == 16, "no full batch"

    def test_single_batch_is_padded(self) -> None:
        # S_max = 2: a padded draw is always 1, so drawing 2 would fail.
        for seed in SEEDS:
            graph, infos = _dense_varlen(batch=1, seq=2)
            out = generate_varlen_inputs(graph, infos, seed)
            assert out[7][0] == 1 and out[8][0] == 1

    def test_one_token_sequences_have_length_one(self) -> None:
        graph, infos = _dense_varlen(batch=3, seq=1)
        lengths = generate_varlen_inputs(graph, infos, 0)[7]
        np.testing.assert_array_equal(lengths, [1, 1, 1])

    def test_lengths_differ_between_seeds_and_repeat_for_one_seed(self) -> None:
        graph, infos = _dense_varlen(batch=6, seq=64)
        runs = [generate_varlen_inputs(graph, infos, seed)[7] for seed in SEEDS]
        again = generate_varlen_inputs(graph, infos, 3)[7]
        np.testing.assert_array_equal(runs[3], again)
        assert len({tuple(run) for run in runs}) > 1

    def test_query_and_key_lengths_are_independent_tensors(self) -> None:
        graph, infos = _dense_varlen(batch=6, seq=64)
        out = generate_varlen_inputs(graph, infos, 1)
        assert not np.array_equal(out[7], out[8])

    def test_graph_without_varlen_tensors_generates_nothing(self) -> None:
        dims = [1, 2, 4, 8]
        graph = _graph(
            [_tensor_json(1, "Q", dims)],
            _node({"q_tensor_uid": 1, "k_tensor_uid": 1, "v_tensor_uid": 1}),
        )
        assert generate_varlen_inputs(graph, [], 0) == {}
        assert generate_varlen_inputs(None, [], 0) == {}

    def test_two_nodes_sharing_a_seq_len_share_its_values(self) -> None:
        dims_a, dims_b = [4, 2, 16, 8], [4, 2, 8, 8]
        tensors = [
            _tensor_json(1, "QA", dims_a),
            _tensor_json(2, "KA", dims_a),
            _tensor_json(5, "QB", dims_b),
            _tensor_json(6, "KB", dims_b),
            _tensor_json(7, "SEQ_LEN_Q", [4], data_type="int32"),
            _tensor_json(8, "SEQ_LEN_KV_A", [4], data_type="int32"),
            _tensor_json(9, "SEQ_LEN_KV_B", [4], data_type="int32"),
        ]
        graph = _graph(
            tensors,
            _node(
                {
                    "q_tensor_uid": 1,
                    "k_tensor_uid": 2,
                    "seq_len_q_tensor_uid": 7,
                    "seq_len_kv_tensor_uid": 8,
                }
            ),
            _node(
                {
                    "q_tensor_uid": 5,
                    "k_tensor_uid": 6,
                    "seq_len_q_tensor_uid": 7,
                    "seq_len_kv_tensor_uid": 9,
                }
            ),
        )
        infos = [
            _info(7, "SEQ_LEN_Q", [4]),
            _info(8, "SEQ_LEN_KV_A", [4]),
            _info(9, "SEQ_LEN_KV_B", [4]),
        ]
        for seed in SEEDS:
            out = generate_varlen_inputs(graph, infos, seed)
            assert out[7].max() <= 8, "shared lengths exceed the smaller operand"
            assert out[8].max() <= 16 and out[9].max() <= 8

    def test_lengths_under_any_node_section_are_found_like_the_resolver_does(
        self,
    ) -> None:
        graph, infos = _dense_varlen(batch=4, seq=16)
        node = graph["nodes"][0]
        for key in ("seq_len_q_tensor_uid", "seq_len_kv_tensor_uid"):
            node["attributes"][key] = node["inputs"].pop(key)
        infos = [TensorInfo.from_json(t) for t in graph["tensors"]]
        for seed in SEEDS:
            data = generate_input_data(infos, seed, graph)
            assert data[7].min() >= 1 and data[8].min() >= 1
            resolved = resolve_varlen_layouts(infos, graph, data)
            q = next(ti for ti in resolved if ti.uid == 1)
            assert q.varlen.lengths == tuple(int(n) for n in data[7])


class TestInputDataIntegration:
    def test_float_values_do_not_depend_on_the_graph(self) -> None:
        graph, infos = _dense_varlen()
        floats = [
            TensorInfo(
                uid,
                name,
                [4, 2, 16, 8],
                [256, 128, 8, 1],
                "bfloat16",
                is_virtual=False,
            )
            for uid, name in ((1, "Q"), (2, "K"), (3, "V"))
        ]
        tensor_infos = floats + infos
        with_graph = generate_input_data(tensor_infos, seed=5, graph_json=graph)
        without = generate_input_data(tensor_infos, seed=5)
        for uid in (1, 2, 3):
            np.testing.assert_array_equal(with_graph[uid], without[uid])

    def test_float_values_do_not_depend_on_the_graph_for_ragged(self) -> None:
        graph, infos = _ragged_graph()
        floats = [
            TensorInfo(1, "Q", [4, 2, 16, 8], [256, 128, 8, 1], "bfloat16", False)
        ]
        with_graph = generate_input_data(floats + infos, seed=5, graph_json=graph)
        without = generate_input_data(floats + infos, seed=5)
        np.testing.assert_array_equal(with_graph[1], without[1])

    def test_generate_input_data_uses_the_varlen_values(self) -> None:
        graph, infos = _dense_varlen()
        data = generate_input_data(infos, seed=2, graph_json=graph)
        expected = generate_varlen_inputs(graph, infos, 2)
        for uid in (7, 8):
            np.testing.assert_array_equal(data[uid], expected[uid])
        assert data[7].min() >= 1

    def test_integer_tensor_without_an_sdpa_role_stays_zero(self) -> None:
        graph, infos = _dense_varlen()
        other = _info(50, "indices", [6])
        data = generate_input_data(infos + [other], seed=2, graph_json=graph)
        np.testing.assert_array_equal(data[50], np.zeros(6, dtype=np.int32))


class TestOffsetTables:
    def test_table_has_b_plus_one_monotone_entries_starting_at_zero(self) -> None:
        for seed in SEEDS:
            graph, infos = _ragged_graph(batch=4, seq=16)
            table = generate_varlen_inputs(graph, infos, seed)[9]
            assert table.shape == (5, 1, 1, 1)
            flat = table.reshape(-1)
            assert flat[0] == 0
            assert np.all(np.diff(flat) >= 1)

    def test_token_major_table_covers_the_lengths_within_the_sequence(self) -> None:
        # Token-major: stride[2] == multiplier, so the table counts tokens.
        gaps_seen = False
        for seed in SEEDS:
            graph, infos = _ragged_graph(batch=4, seq=16)
            out = generate_varlen_inputs(graph, infos, seed)
            ranges = np.diff(out[9].reshape(-1))
            lengths = out[7]
            assert np.all(ranges >= lengths)
            assert np.all(ranges <= 16)
            gaps_seen |= bool(np.any(ranges > lengths))
        assert gaps_seen, "no batch ever started after a gap"

    def test_query_and_key_lengths_share_a_table(self) -> None:
        for seed in SEEDS:
            graph, infos = _ragged_graph(batch=4, seq=16)
            out = generate_varlen_inputs(graph, infos, seed)
            ranges = np.diff(out[9].reshape(-1))
            assert np.all(ranges >= out[7]) and np.all(ranges >= out[8])

    def test_table_without_a_length_source_has_no_gaps_to_hide(self) -> None:
        for seed in SEEDS:
            graph, infos = _ragged_graph(batch=4, seq=16, with_seq_len=False)
            table = generate_varlen_inputs(graph, infos, seed)[9]
            ranges = np.diff(table.reshape(-1))
            assert ranges.min() >= 1 and ranges.max() <= 16
            assert ranges.min() < 16, "no padded batch"
            assert ranges.max() == 16, "no full batch"

    def test_multiplier_scales_the_stored_offsets(self) -> None:
        # Element units: multiplier 1, so the table holds token * strides[2].
        for seed in SEEDS:
            graph, infos = _ragged_graph(batch=4, seq=16, multiplier=1)
            out = generate_varlen_inputs(graph, infos, seed)
            table = out[9].reshape(-1)
            stride = 2 * 8
            assert np.all(table % stride == 0)
            assert np.all(np.diff(table) // stride >= out[7])

    def test_tensors_with_different_head_widths_share_one_token_table(self) -> None:
        """Q and O use token multiplier H*D with different H*D; the shared
        table must hold token counts that both read the same way."""
        dims_q, dims_o = [3, 4, 8, 8], [3, 2, 8, 8]
        ragged = {"ragged_offset_tensor_uid": 9}
        tensors = [
            _tensor_json(
                1,
                "Q",
                dims_q,
                _token_major(3, 4, 8, 8),
                ragged_offset_multiplier=32,
                **ragged,
            ),
            _tensor_json(
                4,
                "O",
                dims_o,
                _token_major(3, 2, 8, 8),
                ragged_offset_multiplier=16,
                **ragged,
            ),
            _tensor_json(9, "OFFSETS", [4, 1, 1, 1], data_type="int32"),
            _tensor_json(7, "SEQ_LEN_Q", [3], data_type="int32"),
        ]
        graph = _graph(tensors, _node({"q_tensor_uid": 1, "seq_len_q_tensor_uid": 7}))
        infos = [_info(9, "OFFSETS", [4, 1, 1, 1]), _info(7, "SEQ_LEN_Q", [3])]
        for seed in SEEDS:
            out = generate_varlen_inputs(graph, infos, seed)
            assert np.all(np.diff(out[9].reshape(-1)) >= out[7])

    def test_tensors_that_read_a_table_differently_are_refused(self) -> None:
        graph, infos = _ragged_graph(batch=4, seq=16, multiplier=1)
        graph["tensors"][1]["strides"] = [0, 8, 32, 1]
        with pytest.raises(UnsupportedGraphError, match="OFFSETS"):
            generate_varlen_inputs(graph, infos, 0)

    def test_a_fractional_offset_is_refused_with_the_tensor_name(self) -> None:
        graph, infos = _ragged_graph(batch=4, seq=16, multiplier=10**9)
        with pytest.raises(UnsupportedGraphError, match="'Q'"):
            generate_varlen_inputs(graph, infos, 0)

    def test_head_major_table_fits_the_token_capacity(self) -> None:
        batch, heads, seq, head_dim = 4, 2, 16, 8
        capacity = batch * seq - 5
        strides = [heads * capacity * head_dim, capacity * head_dim, head_dim, 1]
        for seed in SEEDS:
            graph, infos = _ragged_graph(
                batch=batch, seq=seq, heads=heads, strides=strides, multiplier=1
            )
            out = generate_varlen_inputs(graph, infos, seed)
            table = out[9].reshape(-1)
            assert table[-1] // head_dim <= capacity
            assert np.all(np.diff(table) // head_dim >= out[7])
            assert out[7].min() >= 1

    def test_too_small_a_capacity_is_refused(self) -> None:
        batch, heads, seq, head_dim = 4, 2, 16, 8
        strides = [heads * 3 * head_dim, 3 * head_dim, head_dim, 1]  # 3 tokens
        graph, infos = _ragged_graph(
            batch=batch, seq=seq, heads=heads, strides=strides, multiplier=1
        )
        with pytest.raises(UnsupportedGraphError, match="Q"):
            generate_varlen_inputs(graph, infos, 0)

    def test_single_token_head_major_tensor_is_accepted(self) -> None:
        # strides[1] == strides[2]: the head axis is outer, one token fits.
        graph, infos = _ragged_graph(
            batch=1, seq=1, heads=4, strides=[32, 8, 8, 1], multiplier=8
        )
        out = generate_varlen_inputs(graph, infos, 0)
        np.testing.assert_array_equal(out[7], [1])
        np.testing.assert_array_equal(out[9].reshape(-1), [0, 1])

    def test_two_batches_do_not_fit_a_single_token_head_major_tensor(self) -> None:
        graph, infos = _ragged_graph(
            batch=2, seq=1, heads=4, strides=[32, 8, 8, 1], multiplier=8
        )
        with pytest.raises(UnsupportedGraphError, match="'Q'"):
            generate_varlen_inputs(graph, infos, 0)

    @pytest.mark.parametrize("batch", [2, 4])
    def test_a_length_source_always_leaves_a_gap_before_some_batch(self, batch) -> None:
        for seed in range(40):
            graph, infos = _ragged_graph(batch=batch, seq=16)
            out = generate_varlen_inputs(graph, infos, seed)
            ranges = np.diff(out[9].reshape(-1))
            assert np.any(ranges > out[7]), f"seed {seed}: no gap"

    @pytest.mark.parametrize("with_seq_len", [True, False])
    def test_a_multiplier_above_the_token_stride_succeeds_for_every_seed(
        self, with_seq_len
    ) -> None:
        # Token stride 16, multiplier 32: an offset is a whole number only for
        # an even token start.
        for seed in range(30):
            graph, infos = _ragged_graph(
                batch=4, seq=16, multiplier=32, with_seq_len=with_seq_len
            )
            data = generate_input_data(infos, seed, graph)
            table = data[9].reshape(-1)
            assert table.dtype == np.int32
            resolved = {
                ti.uid: ti
                for ti in resolve_varlen_layouts(
                    [TensorInfo.from_json(t) for t in graph["tensors"]], graph, data
                )
            }
            layout = resolved[1].varlen
            assert layout.starts == tuple(int(off) * 32 for off in table[:-1])
            assert layout.end == int(table[-1]) * 32
            ranges = np.diff(table) * 2
            assert np.all(ranges <= 16)
            if with_seq_len:
                assert layout.lengths == tuple(int(n) for n in data[7])
            else:
                assert layout.lengths == tuple(int(n) for n in ranges)
                assert all(n >= 2 and n % 2 == 0 for n in layout.lengths)

    def test_an_alignment_longer_than_a_sequence_is_refused_for_every_seed(
        self,
    ) -> None:
        for seed in SEEDS:
            graph, infos = _ragged_graph(batch=4, seq=16, multiplier=16 * 32)
            with pytest.raises(UnsupportedGraphError, match="'Q'"):
                generate_varlen_inputs(graph, infos, seed)

    def test_an_offset_table_that_overflows_its_dtype_is_refused(self) -> None:
        strides = [2**33, 8, 2**31, 1]
        graph, infos = _ragged_graph(batch=4, seq=16, strides=strides, multiplier=1)
        with pytest.raises(UnsupportedGraphError, match="OFFSETS"):
            generate_varlen_inputs(graph, infos, 0)

    def test_a_zero_sequence_stride_is_refused_naming_the_tensor(self) -> None:
        graph, infos = _ragged_graph(batch=4, seq=16, strides=[0, 8, 0, 1])
        with pytest.raises(UnsupportedGraphError, match="'Q'"):
            generate_varlen_inputs(graph, infos, 0)


class TestCumulativeLengths:
    def _graph(self, batch=4, seq=16, mixed=False):
        dims = [batch, 2, seq, 8]
        inputs = {
            "q_tensor_uid": 1,
            "k_tensor_uid": 2,
            "cu_seq_len_q_tensor_uid": 7,
        }
        tensors = [
            _tensor_json(1, "Q", dims),
            _tensor_json(2, "K", dims),
            _tensor_json(7, "CU_Q", [batch + 1, 1, 1, 1], data_type="int32"),
        ]
        infos = [_info(7, "CU_Q", [batch + 1, 1, 1, 1])]
        if mixed:
            inputs["seq_len_kv_tensor_uid"] = 8
            tensors.append(_tensor_json(8, "SEQ_LEN_KV", [batch], data_type="int32"))
            infos.append(_info(8, "SEQ_LEN_KV", [batch]))
        else:
            inputs["cu_seq_len_kv_tensor_uid"] = 8
            tensors.append(
                _tensor_json(8, "CU_KV", [batch + 1, 1, 1, 1], data_type="int32")
            )
            infos.append(_info(8, "CU_KV", [batch + 1, 1, 1, 1]))
        return _graph(tensors, _node(inputs)), infos

    def test_values_are_running_totals_of_valid_lengths(self) -> None:
        for seed in SEEDS:
            graph, infos = self._graph()
            out = generate_varlen_inputs(graph, infos, seed)
            for uid in (7, 8):
                cu = out[uid]
                assert cu.shape == (5, 1, 1, 1) and cu.dtype == np.int32
                flat = cu.reshape(-1)
                assert flat[0] == 0
                lengths = np.diff(flat)
                assert lengths.min() >= 1 and lengths.max() <= 16
                assert lengths.min() < 16 and lengths.max() == 16

    def test_sides_may_mix_cumulative_and_plain_lengths(self) -> None:
        graph, infos = self._graph(mixed=True)
        out = generate_varlen_inputs(graph, infos, 0)
        assert out[7].reshape(-1)[0] == 0
        assert out[8].shape == (4,) and out[8].min() >= 1

    def test_one_dimensional_table_is_accepted(self) -> None:
        graph, infos = self._graph()
        graph["tensors"][2]["dims"] = [5]
        infos[0] = _info(7, "CU_Q", [5])
        out = generate_varlen_inputs(graph, infos, 0)
        assert out[7].shape == (5,) and out[7][0] == 0

    def test_cumulative_lengths_of_a_ragged_group_match_the_table(self) -> None:
        graph, infos = _ragged_graph(with_seq_len=False)
        node = graph["nodes"][0]
        node["inputs"]["cu_seq_len_q_tensor_uid"] = 7
        graph["tensors"].append(
            _tensor_json(7, "CU_Q", [5, 1, 1, 1], data_type="int32")
        )
        infos.append(_info(7, "CU_Q", [5, 1, 1, 1]))
        for seed in SEEDS:
            out = generate_varlen_inputs(graph, infos, seed)
            lengths = np.diff(out[7].reshape(-1))
            assert np.all(np.diff(out[9].reshape(-1)) >= lengths)


class TestPagedLengths:
    NUM_PAGES = 32
    PAGE_SIZE = 64
    WIDTH = 4

    def _graph(self, total_q=128, max_seq_len_kv=None, batch=3):
        tensors = [
            _tensor_json(1, "Q", [1, 16, total_q, 128]),
            _tensor_json(2, "K", [self.NUM_PAGES, 4, self.PAGE_SIZE, 128]),
            _tensor_json(3, "V", [self.NUM_PAGES, 4, self.PAGE_SIZE, 128]),
            _tensor_json(5, "PAGE_TABLE_K", [batch, self.WIDTH], data_type="int32"),
            _tensor_json(6, "PAGE_TABLE_V", [batch, self.WIDTH], data_type="int32"),
            _tensor_json(7, "SEQ_LEN_Q", [batch], data_type="int32"),
            _tensor_json(8, "SEQ_LEN_KV", [batch], data_type="int32"),
        ]
        attributes = {}
        if max_seq_len_kv is not None:
            attributes["max_seq_len_kv"] = max_seq_len_kv
        graph = _graph(
            tensors,
            _node(
                {
                    "q_tensor_uid": 1,
                    "k_tensor_uid": 2,
                    "v_tensor_uid": 3,
                    "page_table_k_tensor_uid": 5,
                    "page_table_v_tensor_uid": 6,
                    "seq_len_q_tensor_uid": 7,
                    "seq_len_kv_tensor_uid": 8,
                },
                attributes=attributes,
            ),
        )
        infos = [
            _info(5, "PAGE_TABLE_K", [batch, self.WIDTH]),
            _info(6, "PAGE_TABLE_V", [batch, self.WIDTH]),
            _info(7, "SEQ_LEN_Q", [batch]),
            _info(8, "SEQ_LEN_KV", [batch]),
        ]
        return graph, infos

    def test_key_lengths_fit_the_page_table_allocation(self) -> None:
        for seed in SEEDS:
            graph, infos = self._graph()
            kv = generate_input_data(infos, seed, graph_json=graph)[8]
            assert kv.min() >= 1
            assert kv.max() <= self.WIDTH * self.PAGE_SIZE
            assert kv.min() < self.WIDTH * self.PAGE_SIZE

    def test_key_lengths_are_capped_by_max_seq_len_kv(self) -> None:
        for seed in SEEDS:
            graph, infos = self._graph(max_seq_len_kv=100)
            kv = generate_input_data(infos, seed, graph_json=graph)[8]
            assert 1 <= kv.min() and kv.max() <= 100

    def test_packed_query_lengths_partition_the_token_count(self) -> None:
        for seed in SEEDS:
            graph, infos = self._graph(total_q=128, batch=3)
            q = generate_input_data(infos, seed, graph_json=graph)[7]
            assert q.min() >= 1
            assert int(q.sum()) == 128

    def test_page_table_invariants_are_kept(self) -> None:
        graph, infos = self._graph(batch=3)
        data = generate_input_data(infos, 0, graph_json=graph)
        for uid in (5, 6):
            table = data[uid]
            assert table.shape == (3, self.WIDTH)
            assert table.max() > 0
            assert table.max() < self.NUM_PAGES
            assert len(set(table.reshape(-1).tolist())) == table.size

    def test_unpacked_query_lengths_fit_the_query(self) -> None:
        graph, infos = self._graph(batch=3)
        graph["tensors"][0]["dims"] = [3, 16, 32, 128]
        for seed in SEEDS:
            q = generate_varlen_inputs(graph, infos, seed)[7]
            assert 1 <= q.min() and q.max() <= 32

    def test_more_sequences_than_packed_tokens_names_the_query(self) -> None:
        graph, infos = self._graph(total_q=2, batch=3)
        with pytest.raises(UnsupportedGraphError, match=r"'Q' \(uid 1\)"):
            generate_varlen_inputs(graph, infos, 0)


class TestBatchCount:
    def test_group_without_a_batch_count_names_its_tensors(self) -> None:
        graph, infos = _dense_varlen()
        for tensor in graph["tensors"]:
            tensor["dims"] = []
        infos = [_info(7, "SEQ_LEN_Q", []), _info(8, "SEQ_LEN_KV", [])]
        with pytest.raises(UnsupportedGraphError, match=r"no batch count.*SEQ_LEN_Q"):
            generate_varlen_inputs(graph, infos, 0)


class TestOffsetDeterminism:
    @staticmethod
    def _offsets(seed):
        graph, infos = _ragged_graph(batch=6, seq=32)
        return generate_varlen_inputs(graph, infos, seed)[9]

    def test_one_seed_gives_one_offset_table(self) -> None:
        for seed in SEEDS:
            np.testing.assert_array_equal(self._offsets(seed), self._offsets(seed))

    def test_seeds_give_different_offset_tables(self) -> None:
        tables = {tuple(self._offsets(seed).ravel().tolist()) for seed in SEEDS}
        assert len(tables) > 1

# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for varlen/ragged SDPA layout resolution, storage and refusals."""

import dataclasses

import numpy as np
import pytest

from dnn_benchmarking.common.exceptions import UnsupportedGraphError
from dnn_benchmarking.graph import TensorInfo
from dnn_benchmarking.graph.varlen import (
    live_rows,
    resolve_varlen_layouts,
    sdpa_lengths,
    sdpa_operands,
)

# B=2, H=2, S=4, D=3. Token-major: element = off*6 + h*3 + t*6 + d.
TOKEN_MAJOR = [24, 3, 6, 1]
# Head-major over 6 packed tokens: element = off*3 + h*18 + t*3 + d.
HEAD_MAJOR = [36, 18, 3, 1]
DIMS = [2, 2, 4, 3]


def _tensor(uid, name, dims, strides=None, dtype="half", **extra):
    entry = {"uid": uid, "name": name, "dims": dims, "data_type": dtype, **extra}
    if strides is not None:
        entry["strides"] = strides
    return entry


def _scenario(
    strides=TOKEN_MAJOR,
    mult=6,
    off=None,
    seq=None,
    cu=None,
    attrs=None,
    extra_inputs=None,
    ragged=("q", "k", "v", "o"),
    extra_tensors=(),
    extra_nodes=(),
):
    """A forward SDPA graph plus its input data; tables are uids 10 (offsets),
    11 (seq_len) and 12 (cu_seq_len)."""
    ragged_keys = {}
    if off is not None:
        ragged_keys = {"ragged_offset_tensor_uid": 10, "ragged_offset_multiplier": mult}
    tensors = [
        _tensor(uid, name, DIMS, strides, **(ragged_keys if name in ragged else {}))
        for uid, name in ((1, "q"), (2, "k"), (3, "v"), (4, "o"))
    ]
    data = {}
    inputs = {"q_tensor_uid": 1, "k_tensor_uid": 2, "v_tensor_uid": 3}
    for uid, key, values in (
        (10, None, off),
        (11, "seq_len_q_tensor_uid", seq),
        (12, "cu_seq_len_q_tensor_uid", cu),
    ):
        if values is None:
            continue
        tensors.append(
            _tensor(uid, f"table{uid}", [len(values), 1, 1, 1], dtype="int32")
        )
        data[uid] = np.array(values, dtype=np.int32).reshape(-1, 1, 1, 1)
        if key is not None:
            inputs[key] = uid
            inputs[key.replace("_q_", "_kv_")] = uid
    inputs.update(extra_inputs or {})
    node = {
        "type": "SdpaAttributes",
        "inputs": inputs,
        "outputs": {"o_tensor_uid": 4},
        "attributes": attrs or {},
    }
    graph = {
        "nodes": [node, *extra_nodes],
        "tensors": tensors + list(extra_tensors),
    }
    return graph, data


def _resolve(graph, data):
    infos = [TensorInfo.from_json(t) for t in graph["tensors"] if not t.get("virtual")]
    return {ti.uid: ti for ti in resolve_varlen_layouts(infos, graph, data)}


def test_seq_len_gives_lengths_to_every_operand_without_starts() -> None:
    graph, data = _scenario(seq=[2, 3])

    infos = _resolve(graph, data)

    for uid in (1, 2, 3, 4):
        assert infos[uid].varlen.lengths == (2, 3)
        assert infos[uid].varlen.starts is None


def test_cu_seq_len_differences_give_lengths() -> None:
    graph, data = _scenario(cu=[0, 2, 5])

    assert _resolve(graph, data)[1].varlen.lengths == (2, 3)


def test_a_side_may_use_seq_len_while_the_other_uses_cu_seq_len() -> None:
    graph, data = _scenario(
        seq=[2, 3],
        extra_inputs={"seq_len_kv_tensor_uid": None, "cu_seq_len_kv_tensor_uid": 12},
    )
    graph["tensors"].append(_tensor(12, "cu", [3, 1, 1, 1], dtype="int32"))
    data[12] = np.array([0, 4, 8], dtype=np.int32)

    infos = _resolve(graph, data)

    assert infos[1].varlen.lengths == (2, 3)
    assert infos[2].varlen.lengths == (4, 4)


def test_offset_differences_give_lengths_starts_and_end_from_multiplier() -> None:
    graph, data = _scenario(off=[0, 2, 5])

    q = _resolve(graph, data)[1]

    assert q.varlen.lengths == (2, 3)
    assert q.varlen.starts == (0, 12)
    assert q.varlen.end == 30


def test_seq_len_with_gapped_offsets_keeps_the_shorter_length() -> None:
    graph, data = _scenario(off=[0, 3, 6], seq=[2, 3])

    q = _resolve(graph, data)[1]

    assert q.varlen.lengths == (2, 3)
    assert q.varlen.starts == (0, 18)
    assert q.varlen.end == 36


def test_token_major_storage_reaches_the_end_of_the_offset_table() -> None:
    graph, data = _scenario(off=[0, 3, 6], seq=[2, 3])

    q = _resolve(graph, data)[1]

    # batch 1 ends at 18 + 2*6 + (1*3 + 2*1) + 1 = 36, equal to off[B]*mult.
    assert q.storage_elements == 36
    assert q.size_bytes == 72


def test_storage_extends_past_the_table_end_when_the_last_batch_reaches_further() -> (
    None
):
    graph, data = _scenario(off=[0, 2, 5], seq=[2, 3])
    q = _resolve(graph, data)[1]
    shorter = dataclasses.replace(q, varlen=dataclasses.replace(q.varlen, end=1))

    # batch 1 last element: 12 + 2*6 + 5 = 29 -> 30 elements.
    assert shorter.storage_elements == 30


def test_head_major_storage_counts_the_head_stride() -> None:
    graph, data = _scenario(strides=HEAD_MAJOR, mult=3, off=[0, 2, 5])

    q = _resolve(graph, data)[1]

    assert q.varlen.starts == (0, 6)
    assert q.varlen.end == 15
    # batch 1, h=1, t=2, d=2: 6 + 18 + 6 + 2 = 32 -> 33 elements.
    assert q.storage_elements == 33


def test_logical_strides_pack_dims_in_graph_stride_order() -> None:
    token_major, data = _scenario(off=[0, 2, 5])
    head_major, head_data = _scenario(strides=HEAD_MAJOR, mult=3, off=[0, 2, 5])

    assert _resolve(token_major, data)[1].logical_strides == [24, 3, 6, 1]
    assert _resolve(head_major, head_data)[1].logical_strides == [24, 12, 3, 1]


def test_logical_strides_of_a_dense_tensor_are_its_graph_strides() -> None:
    info = TensorInfo.from_json(_tensor(1, "x", DIMS, [100, 3, 6, 1]))

    assert info.logical_strides == [100, 3, 6, 1]


def test_valid_elements_count_only_live_positions() -> None:
    graph, data = _scenario(seq=[2, 3])

    q = _resolve(graph, data)[1]

    assert q.num_elements == 48
    assert q.valid_elements == 30
    assert TensorInfo.from_json(_tensor(1, "x", DIMS, TOKEN_MAJOR)).valid_elements == 48


def test_ragged_tensor_without_a_layout_has_no_storage_size() -> None:
    info = TensorInfo.from_json(
        _tensor(1, "q", DIMS, TOKEN_MAJOR, ragged_offset_tensor_uid=10)
    )

    with pytest.raises(UnsupportedGraphError, match="'q'"):
        info.storage_elements


def test_graph_without_varlen_returns_the_input_list() -> None:
    graph, data = _scenario()
    infos = [TensorInfo.from_json(t) for t in graph["tensors"]]

    assert resolve_varlen_layouts(infos, graph, data) is infos


def test_sdpa_lengths_is_none_for_a_dense_node() -> None:
    graph, _ = _scenario()
    tensors = {t["uid"]: t for t in graph["tensors"]}

    assert sdpa_lengths(graph["nodes"][0], tensors, lambda uid: []) is None


def test_paged_lengths_split_total_q_when_seq_len_q_is_absent() -> None:
    node = {
        "type": "SdpaAttributes",
        "inputs": {
            "q_tensor_uid": 1,
            "k_tensor_uid": 2,
            "page_table_k_tensor_uid": 5,
            "page_table_v_tensor_uid": 6,
            "seq_len_kv_tensor_uid": 7,
        },
    }
    tensors = {1: {"uid": 1, "dims": [1, 2, 6, 3]}}

    assert sdpa_operands(node).is_paged
    assert sdpa_lengths(node, tensors, lambda uid: [4, 9, 2]) == ([2, 2, 2], [4, 9, 2])


def _rank3():
    graph, data = _scenario(off=[0, 2, 5])
    graph["tensors"][0]["dims"] = [2, 4, 3]
    graph["tensors"][0]["strides"] = [24, 6, 1]
    return graph, data


def _drop_data(scenario, uid):
    graph, data = scenario
    del data[uid]
    return graph, data


def _own_table_for_o():
    """O reads its own table, whose ranges (3, 3) differ from Q's lengths."""
    graph, data = _scenario(off=[0, 2, 5], ragged=("q", "k", "v"))
    graph["tensors"][3].update(ragged_offset_tensor_uid=13, ragged_offset_multiplier=6)
    graph["tensors"].append(_tensor(13, "table13", [3, 1, 1, 1], dtype="int32"))
    data[13] = np.array([0, 3, 6], dtype=np.int32).reshape(-1, 1, 1, 1)
    return graph, data


def _wide_k():
    graph, data = _scenario(seq=[2, 3])
    graph["tensors"][1]["dims"] = [3, 2, 4, 3]
    return graph, data


def _second_node():
    """A second node reads the same Q/K/V/O with other lengths."""
    graph, data = _scenario(seq=[2, 3])
    graph["tensors"].append(_tensor(14, "table14", [2, 1, 1, 1], dtype="int32"))
    data[14] = np.array([1, 1], dtype=np.int32).reshape(-1, 1, 1, 1)
    graph["nodes"].append(
        {
            "type": "SdpaAttributes",
            "inputs": {
                "q_tensor_uid": 1,
                "k_tensor_uid": 2,
                "v_tensor_uid": 3,
                "seq_len_q_tensor_uid": 14,
                "seq_len_kv_tensor_uid": 14,
            },
            "outputs": {"o_tensor_uid": 4},
            "attributes": {},
        }
    )
    return graph, data


REFUSALS = {
    "rank": _rank3,
    "table size": lambda: _scenario(off=[0, 2]),
    "off0": lambda: _scenario(off=[1, 3, 5]),
    "decreasing": lambda: _scenario(off=[0, 3, 2]),
    "length above dim": lambda: _scenario(seq=[2, 5]),
    "negative length": lambda: _scenario(seq=[-1, 2]),
    "seq_len above packed range": lambda: _scenario(off=[0, 1, 4], seq=[2, 3]),
    "unaligned range": lambda: _scenario(off=[0, 2, 5], mult=4),
    "unaligned range with seq_len": lambda: _scenario(
        off=[0, 2, 5], mult=4, seq=[1, 1]
    ),
    "padding_mask without sources": lambda: _scenario(attrs={"padding_mask": True}),
    "seq_len on one side only": lambda: _scenario(
        seq=[2, 3], extra_inputs={"seq_len_kv_tensor_uid": None}
    ),
    "both sources on a side": lambda: _scenario(
        seq=[2, 3], extra_inputs={"cu_seq_len_q_tensor_uid": 12}
    ),
    "batch count": lambda: _scenario(seq=[2, 3, 1]),
    "cu_seq_len batch count": lambda: _scenario(cu=[0, 2]),
    "paged ragged": lambda: _scenario(
        off=[0, 2, 5],
        extra_inputs={"page_table_k_tensor_uid": 12, "page_table_v_tensor_uid": 12},
    ),
    "paged cu_seq_len": lambda: _scenario(
        cu=[0, 2, 5],
        extra_inputs={"page_table_k_tensor_uid": 13, "page_table_v_tensor_uid": 13},
    ),
    "overlapping tokens": lambda: _scenario(
        off=[0, 1, 3], mult=3, strides=[24, 6, 3, 1]
    ),
    "computed table": lambda: _scenario(
        seq=[2, 3],
        extra_nodes=[{"type": "Other", "outputs": {"y_tensor_uid": 11}}],
    ),
    "missing table data": lambda: _drop_data(_scenario(seq=[2, 3]), 11),
    # With seq_len the lengths do not come from the table, so only the
    # resolver's stride check prevents a division by zero.
    "zero sequence stride": lambda: _scenario(
        off=[0, 2, 5], seq=[2, 3], strides=[24, 3, 0, 1]
    ),
    "zero sequence stride, lengths from the table": lambda: _scenario(
        off=[0, 2, 5], strides=[24, 3, 0, 1]
    ),
    "operand table disagrees with the side lengths": _own_table_for_o,
    "operand batch count": _wide_k,
    "layouts differ between nodes": _second_node,
}


@pytest.mark.parametrize("case", sorted(REFUSALS))
def test_unsupported_layout_is_refused_naming_the_tensor(case: str) -> None:
    graph, data = REFUSALS[case]()

    with pytest.raises(UnsupportedGraphError, match=r"'[qkvo]'|table11"):
        _resolve(graph, data)


def test_ragged_tensor_outside_a_non_paged_sdpa_node_is_refused() -> None:
    graph, data = _scenario(off=[0, 2, 5])
    graph["tensors"].append(
        _tensor(9, "stray", DIMS, TOKEN_MAJOR, ragged_offset_tensor_uid=10)
    )

    with pytest.raises(UnsupportedGraphError, match="'stray'"):
        _resolve(graph, data)


def _head_major_tokens(tokens):
    """Q/K/V/O ``[1, 4, tokens, 8]`` over strides ``[32, 8, 8, 1]``, so each
    head holds one token; the offset table is ``[0, tokens]``."""
    graph, data = _scenario(off=[0, tokens], mult=8)
    for tensor in graph["tensors"][:4]:
        tensor["dims"] = [1, 4, tokens, 8]
        tensor["strides"] = [32, 8, 8, 1]
    return graph, data


def test_single_token_head_major_tensor_is_accepted() -> None:
    graph, data = _head_major_tokens(1)

    assert _resolve(graph, data)[1].varlen.lengths == (1,)


def test_two_tokens_in_a_one_token_head_stride_are_refused() -> None:
    graph, data = _head_major_tokens(2)

    with pytest.raises(UnsupportedGraphError, match="'q'"):
        _resolve(graph, data)


@pytest.mark.parametrize(
    "lq, lkv, causal, window, bottom_right, expected",
    [
        # No mask: every row is live.
        (5, 3, False, None, False, (0, 5)),
        # Top-left causal never has a dead row.
        (3, 5, True, None, False, (0, 3)),
        (5, 3, True, None, False, (0, 5)),
        # Bottom-right: leading rows are dead only when lq > lkv.
        (5, 3, False, None, True, (2, 5)),
        (3, 5, False, None, True, (0, 3)),
        (4, 4, False, None, True, (0, 4)),
        (5, 1, False, None, True, (4, 5)),
        (5, 2, False, 2, True, (3, 5)),
        (6, 2, False, 1, True, (4, 6)),
        # Top-left window: rows with i - w >= lkv - 1 are dead once lq >= lkv + w.
        (4, 3, False, 2, False, (0, 4)),
        (5, 3, False, 2, False, (0, 4)),
        (4, 2, False, 1, False, (0, 2)),
        # Empty operands.
        (2, 0, True, None, False, (0, 0)),
        (0, 3, False, None, True, (0, 0)),
    ],
)
def test_live_rows_is_the_half_open_range_of_rows_with_a_key(
    lq, lkv, causal, window, bottom_right, expected
) -> None:
    assert live_rows(lq, lkv, causal, window, bottom_right) == expected

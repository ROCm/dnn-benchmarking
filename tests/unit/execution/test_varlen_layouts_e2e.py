# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Ragged and variable-length SDPA layouts, from generated inputs to the reference.

Each case builds a graph, generates its inputs, resolves the layouts, writes the
inputs to hipDNN storage and reads every batch back with an address formula that
lives in this file. A dense per-batch loop over ``F.scaled_dot_product_attention``
is the oracle. It is compared with the PyTorch reference output after that output
has been written to storage too, on valid positions only.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from dnn_benchmarking.execution.buffer_manager import (  # noqa: E402
    _encode_to_storage,
    generate_input_data,
)
from dnn_benchmarking.graph.loader import GraphLoader  # noqa: E402
from dnn_benchmarking.graph.varlen import resolve_varlen_layouts  # noqa: E402
from dnn_benchmarking.validation.providers.pytorch_provider import (  # noqa: E402
    PyTorchReferenceProvider,
)

B, D = 3, 4
SCALE = 0.5
SEED = 7

Q, K, V, O = 1, 2, 3, 4
FWD_STATS = 5
DO, BWD_STATS, DQ, DK, DV = 5, 6, 7, 8, 9
Q_TABLE, KV_TABLE, STATS_TABLE = 20, 21, 22
SEQ_LEN_Q, SEQ_LEN_KV, CU_Q, CU_KV = 30, 31, 32, 33

RAGGED_KINDS = ("token_major", "element", "head_major", "fused")


def _layout(kind, heads, seq, tokens, stats):
    """``(strides, multiplier)`` of a ``[B, heads, seq, D]`` tensor in ``kind``.

    ``stats`` tensors have a last dim of 1: LSE is ``[T, H]`` token-major and
    ``[H, T]`` head-major. ``tokens`` is the token capacity head-major stores
    in its head stride. The multiplier is one token for a token-unit table and
    one element for ``element``.
    """
    if kind == "aiter":  # token-major BSHD view of padded batches
        if stats:
            return [seq * heads, 1, heads, 1], 1
        return [seq * heads * D, D, heads * D, 1], 1
    if kind == "aotriton":  # contiguous BHSD, padded
        if stats:
            return [heads * seq, seq, 1, 1], 1
        return [heads * seq * D, seq * D, D, 1], 1
    if kind == "head_major":
        strides = [0, tokens, 1, 1] if stats else [0, tokens * D, D, 1]
    elif stats:
        strides = [0, 1, heads, 1]
    else:  # a fused QKV chunk spans Q, K and V in each token
        strides = [0, D, heads * D * (3 if kind == "fused" else 1), 1]
    return strides, 1 if kind == "element" else strides[2]


def _graph(kind, source, shape, attributes, hq=2, hkv=1, shared=False, backward=False):
    sq, skv = shape
    if kind == "fused":
        hkv = hq
    ragged = kind in RAGGED_KINDS
    tensors = []

    def add(uid, name, heads, seq, stats, table):
        strides, mult = _layout(kind, heads, seq, B * seq, stats)
        tensor = {
            "uid": uid,
            "name": name,
            "dims": [B, heads, seq, 1 if stats else D],
            "strides": strides,
            "data_type": "float",
            "virtual": False,
        }
        if ragged:
            tensor["ragged_offset_tensor_uid"] = table
            tensor["ragged_offset_multiplier"] = mult
        tensors.append(tensor)

    q_table = Q_TABLE
    kv_table = Q_TABLE if shared else KV_TABLE
    add(Q, "q", hq, sq, False, q_table)
    add(K, "k", hkv, skv, False, kv_table)
    add(V, "v", hkv, skv, False, kv_table)
    stats_uid = BWD_STATS if backward else FWD_STATS
    # Element-unit offsets store each tensor's own token stride, and LSE has a
    # different one from Q and O, so it cannot share their table.
    stats_table = STATS_TABLE if kind == "element" else q_table
    add(O, "o", hq, sq, False, q_table)
    add(stats_uid, "stats", hq, sq, True, stats_table)
    if backward:
        add(DO, "do", hq, sq, False, q_table)
        add(DQ, "dq", hq, sq, False, q_table)
        add(DK, "dk", hkv, skv, False, kv_table)
        add(DV, "dv", hkv, skv, False, kv_table)

    def ints(uid, name, dims):
        tensors.append(
            {
                "uid": uid,
                "name": name,
                "dims": dims,
                "strides": [1] * len(dims),
                "data_type": "int32",
                "virtual": False,
            }
        )

    if ragged:
        ints(Q_TABLE, "q_offsets", [B + 1, 1, 1, 1])
        if not shared:
            ints(KV_TABLE, "kv_offsets", [B + 1, 1, 1, 1])
        if stats_table != q_table:
            ints(STATS_TABLE, "stats_offsets", [B + 1, 1, 1, 1])
    inputs = {"q_tensor_uid": Q, "k_tensor_uid": K, "v_tensor_uid": V}
    if source == "seq_len":
        ints(SEQ_LEN_Q, "seq_len_q", [B])
        ints(SEQ_LEN_KV, "seq_len_kv", [B])
        inputs["seq_len_q_tensor_uid"] = SEQ_LEN_Q
        inputs["seq_len_kv_tensor_uid"] = SEQ_LEN_KV
    elif source == "cu_seq_len":
        ints(CU_Q, "cu_seq_len_q", [B + 1, 1, 1, 1])
        ints(CU_KV, "cu_seq_len_kv", [B + 1, 1, 1, 1])
        inputs["cu_seq_len_q_tensor_uid"] = CU_Q
        inputs["cu_seq_len_kv_tensor_uid"] = CU_KV
    if backward:
        node = {
            "name": "sdpa_bwd",
            "type": "SdpaBackwardAttributes",
            "inputs": {
                **inputs,
                "o_tensor_uid": O,
                "do_tensor_uid": DO,
                "stats_tensor_uid": BWD_STATS,
            },
            "outputs": {
                "dq_tensor_uid": DQ,
                "dk_tensor_uid": DK,
                "dv_tensor_uid": DV,
            },
            "attributes": {"attn_scale_value": SCALE, **attributes},
        }
    else:
        node = {
            "name": "sdpa",
            "type": "SdpaAttributes",
            "inputs": inputs,
            "outputs": {"o_tensor_uid": O, "stats_tensor_uid": FWD_STATS},
            "attributes": {"attn_scale_value": SCALE, **attributes},
        }
    return {"name": "varlen", "tensors": tensors, "nodes": [node]}


def _prepare(graph_json):
    infos = GraphLoader().extract_tensor_info(graph_json)
    input_data = generate_input_data(infos, SEED, graph_json)
    infos = resolve_varlen_layouts(infos, graph_json, input_data)
    return {info.uid: info for info in infos}, input_data


def _gather(storage, info, batch, input_data):
    """``[heads, length, last]`` valid positions of one batch, by
    ``start + h * s1 + t * s2 + d * s3``; the start is ``off[b] * multiplier``
    for a ragged tensor, from the generated table and the graph's multiplier,
    and ``b * strides[0]`` for a padded one."""
    layout = info.varlen
    if info.ragged_offset_tensor_uid is None:
        start = batch * info.strides[0]
    else:
        table = input_data[info.ragged_offset_tensor_uid].reshape(-1)
        start = int(table[batch]) * info.ragged_offset_multiplier

    s1, s2, s3 = info.strides[1:]
    heads, last = info.dims[1], info.dims[3]
    index = (
        start
        + np.arange(heads)[:, None, None] * s1
        + np.arange(layout.lengths[batch])[None, :, None] * s2
        + np.arange(last)[None, None, :] * s3
    )
    return storage[index]


def _gather_all(infos, input_data, uid, array):
    info = infos[uid]
    storage = _encode_to_storage(array, info)
    return [_gather(storage, info, b, input_data) for b in range(B)]


def _has_gap(info):
    """Whether some ragged batch starts after unused tokens of the one before."""
    token = info.strides[2]
    return any(
        nxt - start > length * token
        for start, nxt, length in zip(
            info.varlen.starts, info.varlen.starts[1:], info.varlen.lengths
        )
    )


def _has_padding(info):
    """Whether some batch of a padded tensor is shorter than the capacity."""
    return any(length < info.dims[2] for length in info.varlen.lengths)


def _keep(mode, lq, lkv, dead):
    if mode == "causal":
        return torch.ones(lq, lkv, dtype=torch.bool).tril()[dead:]
    if mode == "bottom_right":
        return torch.ones(lq, lkv, dtype=torch.bool).tril(lkv - lq)[dead:]
    return None


def _oracle_forward(q, k, v, mode):
    """One batch, ``q`` ``[Hq, lq, D]`` and ``k``/``v`` ``[Hkv, lkv, D]``:
    ``(O, LSE)`` from ``F.scaled_dot_product_attention`` on the live rows. Rows
    with no key are 0 and -inf."""
    hq, lq = q.shape[:2]
    lkv = k.shape[1]
    out = np.zeros((hq, lq, v.shape[-1]), dtype=np.float32)
    lse = np.full((hq, lq, 1), -np.inf, dtype=np.float32)
    dead = max(0, lq - lkv) if mode == "bottom_right" else 0
    if lkv == 0 or lq <= dead:
        return out, lse
    rep = hq // k.shape[0]
    q_t = torch.from_numpy(np.ascontiguousarray(q[:, dead:]))[None]
    k_t = torch.from_numpy(k).repeat_interleave(rep, dim=0)[None]
    v_t = torch.from_numpy(v).repeat_interleave(rep, dim=0)[None]
    keep = _keep(mode, lq, lkv, dead)
    scores = (q_t @ k_t.transpose(-2, -1)) * SCALE
    if keep is not None:
        scores = scores.masked_fill(~keep, float("-inf"))
    out[:, dead:] = torch.nn.functional.scaled_dot_product_attention(
        q_t, k_t, v_t, attn_mask=keep, scale=SCALE
    )[0].numpy()
    lse[:, dead:] = torch.logsumexp(scores, dim=-1, keepdim=True)[0].numpy()
    return out, lse


CASES = {
    # Lengths from offset-table differences; token-unit offsets.
    "offset_diff": ("token_major", "offsets", False),
    # Offsets in elements instead of tokens.
    "element_offsets": ("element", "offsets", False),
    # One token-unit table shared by Q, K, V, O and LSE.
    "shared_table": ("token_major", "offsets", True),
    # AOTriton [H, T, D] with an [H, T] LSE.
    "head_major": ("head_major", "offsets", False),
    # Q, K and V as chunks of a fused QKV tensor.
    "fused_qkv": ("fused", "seq_len", False),
    # seq_len bounds batches that may start after padding tokens.
    "gapped_seq_len": ("token_major", "seq_len", False),
    "head_major_seq_len": ("head_major", "seq_len", False),
    "cu_seq_len": ("token_major", "cu_seq_len", False),
    "cu_seq_len_padded": ("aiter", "cu_seq_len", False),
    # aiter token-major BSHD view and aotriton contiguous BHSD, both padded.
    "aiter_bshd": ("aiter", "seq_len", False),
    "aotriton_bhsd": ("aotriton", "seq_len", False),
}

MODES = {
    "full": {},
    "causal": {"causal_mask": True},
    "bottom_right": {"causal_mask_bottom_right": True},
}

GAPPED_CASES = {"gapped_seq_len"}
PADDED_CASES = {"cu_seq_len_padded", "aiter_bshd", "aotriton_bhsd"}


def _shape(mode, shared):
    """``(q_len, kv_len)`` capacities. Bottom-right gets more query rows than
    keys, so a full-length query batch has dead rows."""
    if shared:
        return 5, 5
    return (6, 3) if mode == "bottom_right" else (5, 6)


@pytest.mark.parametrize("mode", list(MODES))
@pytest.mark.parametrize("case", list(CASES))
def test_reference_matches_the_dense_loop_on_valid_positions(case, mode):
    kind, source, shared = CASES[case]
    graph_json = _graph(kind, source, _shape(mode, shared), MODES[mode], shared=shared)
    infos, input_data = _prepare(graph_json)

    q = _gather_all(infos, input_data, Q, input_data[Q])
    k = _gather_all(infos, input_data, K, input_data[K])
    v = _gather_all(infos, input_data, V, input_data[V])
    reference = PyTorchReferenceProvider().compute_reference(graph_json, input_data)
    got_o = _gather_all(infos, input_data, O, reference[O].data)
    got_lse = _gather_all(infos, input_data, FWD_STATS, reference[FWD_STATS].data)

    if source == "seq_len":
        assert list(infos[Q].varlen.lengths) == list(input_data[SEQ_LEN_Q])
        assert list(infos[K].varlen.lengths) == list(input_data[SEQ_LEN_KV])
    dead_seen = False
    for b in range(B):
        lq, lkv = infos[Q].varlen.lengths[b], infos[K].varlen.lengths[b]
        assert q[b].shape[1] == lq and k[b].shape[1] == lkv
        want_o, want_lse = _oracle_forward(q[b], k[b], v[b], mode)
        np.testing.assert_allclose(got_o[b], want_o, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(got_lse[b], want_lse, rtol=1e-5, atol=1e-5)
        dead_seen |= mode == "bottom_right" and lq > lkv
    if mode == "bottom_right" and not shared:
        assert dead_seen, "no batch has dead rows; the case checks nothing"
    if case in GAPPED_CASES:
        gap_seen = _has_gap(infos[Q]) or _has_gap(infos[K])
        assert gap_seen, "no batch starts after a gap; the case checks nothing"
    if case in PADDED_CASES:
        padded_seen = _has_padding(infos[Q]) or _has_padding(infos[K])
        assert padded_seen, "no batch is padded; the case checks nothing"


@pytest.mark.parametrize("mode", ["full", "causal"])
@pytest.mark.parametrize(
    "case",
    [
        "offset_diff",
        "element_offsets",
        "shared_table",
        "head_major",
        "head_major_seq_len",
        "fused_qkv",
        "gapped_seq_len",
        "aiter_bshd",
        "aotriton_bhsd",
    ],
)
def test_backward_reference_matches_per_batch_autograd(case, mode):
    kind, source, shared = CASES[case]
    graph_json = _graph(
        kind, source, _shape(mode, shared), MODES[mode], shared=shared, backward=True
    )
    infos, input_data = _prepare(graph_json)
    hq = infos[Q].dims[1]
    rep = hq // infos[K].dims[1]

    q = _gather_all(infos, input_data, Q, input_data[Q])
    k = _gather_all(infos, input_data, K, input_data[K])
    v = _gather_all(infos, input_data, V, input_data[V])
    do = _gather_all(infos, input_data, DO, input_data[DO])

    # The handler consumes external O and LSE, so they must come from a
    # consistent forward: the per-batch dense loop.
    o_logical = np.zeros(infos[O].dims, dtype=np.float32)
    lse_logical = np.zeros(infos[BWD_STATS].dims, dtype=np.float32)
    want = {uid: [] for uid in (DQ, DK, DV)}
    for b in range(B):
        lq, lkv = q[b].shape[1], k[b].shape[1]
        o_b, lse_b = _oracle_forward(q[b], k[b], v[b], mode)
        o_logical[b, :, :lq] = o_b
        lse_logical[b, :, :lq] = lse_b

        q_g = torch.from_numpy(q[b]).clone().requires_grad_()
        k_g = torch.from_numpy(k[b]).clone().requires_grad_()
        v_g = torch.from_numpy(v[b]).clone().requires_grad_()
        out = torch.nn.functional.scaled_dot_product_attention(
            q_g[None],
            k_g.repeat_interleave(rep, dim=0)[None],
            v_g.repeat_interleave(rep, dim=0)[None],
            attn_mask=_keep(mode, lq, lkv, 0),
            scale=SCALE,
        )
        out.backward(torch.from_numpy(do[b])[None])
        want[DQ].append(q_g.grad.numpy())
        want[DK].append(k_g.grad.numpy())
        want[DV].append(v_g.grad.numpy())
    input_data[O] = o_logical
    input_data[BWD_STATS] = lse_logical

    reference = PyTorchReferenceProvider().compute_reference(graph_json, input_data)

    for uid, lengths_of in ((DQ, Q), (DK, K), (DV, V)):
        logical = reference[uid].data
        got = _gather_all(infos, input_data, uid, logical)
        for b in range(B):
            np.testing.assert_allclose(got[b], want[uid][b], rtol=1e-4, atol=1e-4)
            length = infos[lengths_of].varlen.lengths[b]
            assert not logical[b, :, length:].any()

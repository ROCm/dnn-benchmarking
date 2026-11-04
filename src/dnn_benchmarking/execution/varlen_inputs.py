# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Input generation for variable-length and ragged SDPA graphs.

Per-batch sequence lengths, cumulative lengths and ragged offset tables have
invariants random data cannot satisfy: a length must be positive and fit the
operand, cumulative lengths must be running totals, and offset tables must
agree with the lengths, the strides and the offset multiplier of every tensor
that uses them.

The values come from their own random stream, so the floating-point inputs of
a graph are the same with and without these tensors.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..common.exceptions import UnsupportedGraphError
from ..graph.tensor_info import TensorInfo

# Mixed into the seed so this stream is separate from the float draws.
_STREAM = 0x3139

_QUERY_KEYS = (
    "q_tensor_uid",
    "o_tensor_uid",
    "stats_tensor_uid",
    "do_tensor_uid",
    "dq_tensor_uid",
)
_KEY_KEYS = ("k_tensor_uid", "v_tensor_uid", "dk_tensor_uid", "dv_tensor_uid")


@dataclass
class _Side:
    """One side (query or key) of one SDPA node."""

    seq_uid: Optional[int] = None
    cu_uid: Optional[int] = None
    operands: List[Dict[str, Any]] = field(default_factory=list)
    # Largest valid length of a batch; None when no operand bounds it.
    max_len: Optional[int] = None
    # "bounded": lengths in 1..max_len. "partition": lengths split total_q.
    # "fixed": every length is max_len.
    mode: str = "bounded"
    batch: Optional[int] = None
    total_q: int = 0
    q_label: str = ""

    def tokens(self) -> List[Tuple[str, int]]:
        tokens: List[Tuple[str, int]] = []
        if self.seq_uid is not None:
            tokens.append(("seq", self.seq_uid))
        if self.cu_uid is not None:
            tokens.append(("cu", self.cu_uid))
        for operand in self.operands:
            offset_uid = _offset_uid(operand)
            if offset_uid is not None:
                tokens.append(("off", offset_uid))
        return tokens


def _uid(value: Any) -> Optional[int]:
    return None if value is None else int(value)


def _offset_uid(tensor: Dict[str, Any]) -> Optional[int]:
    return _uid(tensor.get("ragged_offset_tensor_uid"))


def _multiplier(tensor: Dict[str, Any]) -> int:
    value = tensor.get("ragged_offset_multiplier")
    return 1 if value is None else int(value)


def _describe(tensor: Dict[str, Any]) -> str:
    return f"tensor '{tensor.get('name', '')}' (uid {tensor.get('uid')})"


def _numel(dims: Sequence[int]) -> int:
    result = 1
    for dim in dims:
        result *= int(dim)
    return result


class _Graph:
    """Tensor lookup over the graph JSON, falling back to the tensor infos."""

    def __init__(
        self, graph_json: Dict[str, Any], tensor_infos: List[TensorInfo]
    ) -> None:
        self.json = {
            int(t["uid"]): t for t in graph_json.get("tensors", []) or [] if "uid" in t
        }
        # Only real, non-output inputs are generated here.
        self.infos = {
            t.uid: t
            for t in tensor_infos
            if not (t.is_output or t.is_virtual or t.is_pass_by_value)
        }

    def dims(self, uid: Optional[int]) -> Optional[List[int]]:
        if uid is None:
            return None
        if uid in self.json:
            return [int(d) for d in self.json[uid].get("dims", [])]
        if uid in self.infos:
            return list(self.infos[uid].dims)
        return None


def _node_side(
    node: Dict[str, Any], keys: Tuple[str, ...], graph: _Graph
) -> List[Dict[str, Any]]:
    operands: List[Dict[str, Any]] = []
    seen = set()
    for section in ("inputs", "outputs"):
        values = node.get(section) or {}
        for key in keys:
            uid = _uid(values.get(key))
            if uid is not None and uid in graph.json and uid not in seen:
                seen.add(uid)
                operands.append(graph.json[uid])
    return operands


def _collect_sides(graph_json: Dict[str, Any], graph: _Graph) -> List[_Side]:
    sides: List[_Side] = []
    for node in graph_json.get("nodes", []) or []:
        if not str(node.get("type", "")).startswith("Sdpa"):
            continue
        inputs = node.get("inputs") or {}
        attributes = node.get("attributes") or {}
        seq_q = _uid(inputs.get("seq_len_q_tensor_uid"))
        seq_kv = _uid(inputs.get("seq_len_kv_tensor_uid"))
        cu_q = _uid(inputs.get("cu_seq_len_q_tensor_uid"))
        cu_kv = _uid(inputs.get("cu_seq_len_kv_tensor_uid"))
        table_uid = _uid(inputs.get("page_table_k_tensor_uid"))

        query = _Side(
            seq_uid=seq_q, cu_uid=cu_q, operands=_node_side(node, _QUERY_KEYS, graph)
        )
        key = _Side(
            seq_uid=seq_kv, cu_uid=cu_kv, operands=_node_side(node, _KEY_KEYS, graph)
        )
        if table_uid is not None:
            _bound_paged(node, graph, query, key, table_uid, attributes)
        else:
            for side in (query, key):
                _bound_dense(side)
        for side in (query, key):
            if side.tokens():
                sides.append(side)
    return sides


def _bound_dense(side: _Side) -> None:
    bounds = [int(t["dims"][-2]) for t in side.operands if len(t.get("dims", [])) >= 2]
    side.max_len = min(bounds) if bounds else None
    for tensor in side.operands:
        if tensor.get("dims"):
            side.batch = int(tensor["dims"][0])
            break


def _bound_paged(
    node: Dict[str, Any],
    graph: _Graph,
    query: _Side,
    key: _Side,
    table_uid: int,
    attributes: Dict[str, Any],
) -> None:
    """Bound a paged node: key lengths by the page allocation, query lengths
    by the query tensor (a packed query is split over the sequences)."""
    inputs = node.get("inputs") or {}
    table_dims = graph.dims(table_uid)
    key_tensor = graph.json.get(_uid(inputs.get("k_tensor_uid")) or -1)
    page_size = 1
    if key_tensor and len(key_tensor.get("dims", [])) == 4:
        page_size = max(int(key_tensor["dims"][2]), 1)
    width = int(table_dims[1]) if table_dims and len(table_dims) > 1 else 1
    allocation = width * page_size if table_dims else page_size
    cap = attributes.get("max_seq_len_kv")
    if cap is not None and int(cap) > 0:
        allocation = min(allocation, int(cap))
    batch = int(table_dims[0]) if table_dims else None

    # The key containers are page pools, not per-batch tensors.
    key.operands = []
    key.max_len = allocation
    key.batch = batch
    query.batch = batch

    q_tensors = query.operands
    q_main = graph.json.get(_uid(inputs.get("q_tensor_uid")) or -1)
    if q_main is None or len(q_main.get("dims", [])) != 4:
        # No query tensor to bound the lengths: take the page size.
        query.operands = []
        query.mode = "fixed"
        query.max_len = page_size
        return
    dims = [int(d) for d in q_main["dims"]]
    if dims[0] == 1 and (batch is None or batch > 1):
        query.mode = "partition"
        query.total_q = dims[2]
        query.q_label = _describe(q_main)
        query.max_len = dims[2]
    else:
        query.max_len = dims[2]
        query.batch = query.batch or dims[0]
    query.operands = [t for t in q_tensors if _offset_uid(t) is not None]


def _groups(sides: List[_Side]) -> List[List[_Side]]:
    """Sides that share a seq_len, cu_seq_len or offset table form one group."""
    parent = list(range(len(sides)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    first: Dict[Tuple[str, int], int] = {}
    for index, side in enumerate(sides):
        for token in side.tokens():
            if token in first:
                parent[find(index)] = find(first[token])
            else:
                first[token] = index
    grouped: Dict[int, List[_Side]] = {}
    for index, side in enumerate(sides):
        grouped.setdefault(find(index), []).append(side)
    return list(grouped.values())


def _draw_bounded(batch: int, max_len: int, rng: np.random.Generator) -> np.ndarray:
    """Lengths in 1..max_len, varied."""
    if max_len <= 1:
        return np.ones(batch, dtype=np.int64)
    if batch == 1:
        return rng.integers(1, max_len, size=1)
    lengths = rng.integers(1, max_len + 1, size=batch)
    full, padded = rng.permutation(batch)[:2]
    lengths[full] = max_len
    lengths[padded] = rng.integers(1, max_len)
    return lengths


def _draw_partition(
    batch: int, total: int, rng: np.random.Generator, label: str
) -> np.ndarray:
    """Split ``total`` into ``batch`` positive parts."""
    if batch > total:
        raise UnsupportedGraphError(
            f"a packed query of {total} tokens ({label}) cannot hold "
            f"{batch} sequences"
        )
    if batch == 1:
        return np.asarray([total], dtype=np.int64)
    cuts = np.sort(rng.choice(total - 1, size=batch - 1, replace=False) + 1)
    return np.diff(np.concatenate(([0], cuts, [total])))


def _token_capacity(tensor: Dict[str, Any]) -> Optional[int]:
    """Tokens a ragged tensor can hold in total. A head-major tensor stores
    its token axis inside the head stride; a token-major tensor has no bound.
    Axis 1 is head-major when its stride is at least the sequence stride."""
    dims, strides = tensor["dims"], tensor["strides"]
    seq_stride = int(strides[2])
    capacities = [
        int(strides[i]) // seq_stride
        for i in (1, 3)
        if int(dims[i]) > 1
        and (int(strides[i]) >= seq_stride if i == 1 else int(strides[i]) > seq_stride)
    ]
    return min(capacities) if capacities else None


def _fit(
    lengths: np.ndarray,
    gaps: np.ndarray,
    capacity: int,
    limiter: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray]:
    """Shrink gaps, then lengths, until the batches fit ``capacity`` tokens."""
    if len(lengths) > capacity:
        raise UnsupportedGraphError(
            f"{_describe(limiter)} holds {capacity} tokens, fewer than the "
            f"{len(lengths)} batches that each need at least one"
        )
    if int(lengths.sum() + gaps.sum()) <= capacity:
        return lengths, gaps
    gaps = np.zeros_like(gaps)
    total = int(lengths.sum())
    if total <= capacity:
        return lengths, gaps
    lengths = np.maximum(1, lengths * capacity // total)
    while int(lengths.sum()) > capacity:
        lengths[int(np.argmax(lengths))] -= 1
    return lengths, gaps


def _fill(info: TensorInfo, values: Sequence[int], what: str) -> np.ndarray:
    if info.num_elements != len(values):
        raise UnsupportedGraphError(
            f"tensor '{info.name}' (uid {info.uid}) has {info.num_elements} "
            f"elements, but {what} needs {len(values)}"
        )
    return np.asarray(values, dtype=info.dtype.numpy).reshape(info.dims)


def _ordered(values: List[Optional[int]]) -> List[int]:
    return list(dict.fromkeys(v for v in values if v is not None))


def _batch_count(group: List[_Side], graph: _Graph) -> int:
    for side in group:
        if side.batch is not None:
            return side.batch
    for side in group:
        for token_kind, uid in side.tokens():
            dims = graph.dims(uid)
            if not dims:
                continue
            if token_kind == "seq":
                return _numel(dims)
            return _numel(dims) - 1
    names = ", ".join(
        (
            _describe(graph.json[uid])
            if uid in graph.json
            else (
                f"tensor '{graph.infos[uid].name}' (uid {uid})"
                if uid in graph.infos
                else f"tensor (uid {uid})"
            )
        )
        for side in group
        for _, uid in side.tokens()
    )
    raise UnsupportedGraphError(
        f"a variable-length SDPA group has no batch count; its tensors are "
        f"{names or 'none'}"
    )


def _generate_group(
    group: List[_Side], graph: _Graph, rng: np.random.Generator, out: Dict[int, Any]
) -> None:
    seq_uids = _ordered([s.seq_uid for s in group])
    cu_uids = _ordered([s.cu_uid for s in group])
    tables: Dict[int, List[Dict[str, Any]]] = {}
    for side in group:
        for tensor in side.operands:
            uid = _offset_uid(tensor)
            if uid is not None and len(tensor.get("dims", [])) != 4:
                raise UnsupportedGraphError(
                    f"{_describe(tensor)} is ragged and must have 4 dims, "
                    f"has {len(tensor.get('dims', []))}"
                )
            if uid is not None and tensor not in tables.setdefault(uid, []):
                tables[uid].append(tensor)

    targets = [u for u in seq_uids + cu_uids + list(tables) if u in graph.infos]
    if not targets:
        return

    batch = _batch_count(group, graph)
    lengths = _draw_lengths(group, batch, rng)

    gaps = np.zeros(batch, dtype=np.int64)
    if tables:
        has_source = bool(seq_uids or cu_uids)
        bounds = [s.max_len for s in group if s.max_len is not None]
        max_len = min(bounds) if bounds else 1
        if has_source:
            room = np.maximum(max_len - lengths, 0)
            wanted = rng.random(batch) < 0.5
            drawn = rng.integers(1, np.maximum(room, 1) + 1)
            gaps = np.where(wanted & (room > 0), drawn, 0).astype(np.int64)
        for tensors in tables.values():
            for tensor in tensors:
                capacity = _token_capacity(tensor)
                if capacity is not None:
                    lengths, gaps = _fit(lengths, gaps, capacity, tensor)

    for uid in seq_uids:
        if uid in graph.infos:
            out[uid] = _fill(graph.infos[uid], lengths.tolist(), "seq_len")
    if cu_uids:
        totals = np.concatenate(([0], np.cumsum(lengths))).tolist()
        for uid in cu_uids:
            if uid in graph.infos:
                out[uid] = _fill(graph.infos[uid], totals, "cu_seq_len")

    starts = np.concatenate(([0], np.cumsum(lengths + gaps)))
    for uid, tensors in tables.items():
        if uid in graph.infos:
            out[uid] = _offset_table(graph.infos[uid], tensors, starts)


def _draw_lengths(
    group: List[_Side], batch: int, rng: np.random.Generator
) -> np.ndarray:
    for side in group:
        if side.mode == "partition":
            return _draw_partition(batch, side.total_q, rng, side.q_label)
    for side in group:
        if side.mode == "fixed" and side.max_len is not None:
            return np.full(batch, side.max_len, dtype=np.int64)
    bounds = [s.max_len for s in group if s.max_len is not None]
    return _draw_bounded(batch, min(bounds) if bounds else 1, rng)


def _offset_table(
    info: TensorInfo, tensors: List[Dict[str, Any]], starts: np.ndarray
) -> np.ndarray:
    """Offsets ``t_b * strides[2] / multiplier`` that every user of the table
    must read the same way."""
    values: Optional[np.ndarray] = None
    for tensor in tensors:
        scaled = starts * int(tensor["strides"][2])
        multiplier = _multiplier(tensor)
        if multiplier <= 0 or np.any(scaled % multiplier):
            raise UnsupportedGraphError(
                f"{_describe(tensor)}: ragged_offset_multiplier {multiplier} does "
                f"not divide the token starts (token stride {tensor['strides'][2]}) "
                f"stored in offset table '{info.name}' (uid {info.uid})"
            )
        scaled = scaled // multiplier
        if values is not None and not np.array_equal(values, scaled):
            raise UnsupportedGraphError(
                f"{_describe(tensor)} reads offset table '{info.name}' "
                f"(uid {info.uid}) as different values than another tensor "
                "that shares it"
            )
        values = scaled
    assert values is not None
    return _fill(info, values.tolist(), "the offset table")


def generate_varlen_inputs(
    graph_json: Optional[Dict[str, Any]],
    tensor_infos: List[TensorInfo],
    seed: int,
) -> Dict[int, np.ndarray]:
    """Values of every ``seq_len_*``, ``cu_seq_len_*`` and ragged offset table.

    Each array has the dims and dtype of its tensor. Sides that share a
    ``seq_len``, ``cu_seq_len`` or offset table tensor share one set of lengths.

    Lengths are in ``1..S_max`` and vary across batches; with several batches
    one is padded and one is full, and a single batch is padded. A paged key
    side is bounded by ``page_table.dims[1] * page_size`` (and
    ``max_seq_len_kv``); a packed paged query splits its token count over the
    sequences. Offset tables start at 0, store the token start of every batch
    scaled by the tensor's token stride and multiplier, and have a trailing
    entry for the end. Batches start after gaps of padding tokens only when
    the group has a ``seq_len`` or ``cu_seq_len`` that excludes the padding.

    Raises :class:`UnsupportedGraphError` when tensors that share an offset
    table read it differently, an offset is not a whole number, or the
    tokens do not fit the tensor.
    """
    if not graph_json:
        return {}
    graph = _Graph(graph_json, tensor_infos)
    sides = _collect_sides(graph_json, graph)
    if not sides:
        return {}
    rng = np.random.default_rng([seed, _STREAM])
    out: Dict[int, np.ndarray] = {}
    for group in _groups(sides):
        _generate_group(group, graph, rng, out)
    return out

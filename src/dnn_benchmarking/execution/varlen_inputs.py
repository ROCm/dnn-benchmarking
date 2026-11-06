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

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..common.exceptions import UnsupportedGraphError
from ..graph.loader import tensors_by_uid
from ..graph.tensor_info import TensorInfo, ragged_fields, tensor_label
from ..graph.varlen import (
    SdpaOperands,
    check_ragged_shape,
    ragged_token_capacity,
    sdpa_operands,
)

# Mixed into the seed so this stream is separate from the float draws.
_STREAM = 0x3139


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
            offset_uid = ragged_fields(operand)[0]
            if offset_uid is not None:
                tokens.append(("off", offset_uid))
        return tokens


def _label(tensor: Dict[str, Any]) -> str:
    return tensor_label(tensor["uid"], tensor.get("name"))


def _shape(tensor: Dict[str, Any]) -> Tuple[List[int], List[int]]:
    """Integer dims and strides of a tensor's JSON."""
    return (
        [int(d) for d in tensor.get("dims") or []],
        [int(s) for s in tensor.get("strides") or []],
    )


class _Graph:
    """Tensor lookup over the graph JSON, falling back to the tensor infos."""

    def __init__(
        self, graph_json: Dict[str, Any], tensor_infos: List[TensorInfo]
    ) -> None:
        self.json = tensors_by_uid(graph_json)
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


def _tensors(uids: Tuple[int, ...], graph: _Graph) -> List[Dict[str, Any]]:
    """JSON of the listed tensors that the graph defines, without repeats."""
    return [graph.json[uid] for uid in dict.fromkeys(uids) if uid in graph.json]


def _collect_sides(graph_json: Dict[str, Any], graph: _Graph) -> List[_Side]:
    sides: List[_Side] = []
    for node in graph_json.get("nodes", []) or []:
        if not str(node.get("type", "")).startswith("Sdpa"):
            continue
        operands = sdpa_operands(node)
        query = _Side(
            seq_uid=operands.seq_len_q_tensor_uid,
            cu_uid=operands.cu_seq_len_q_tensor_uid,
            operands=_tensors(operands.q_side, graph),
        )
        key = _Side(
            seq_uid=operands.seq_len_kv_tensor_uid,
            cu_uid=operands.cu_seq_len_kv_tensor_uid,
            operands=_tensors(operands.kv_side, graph),
        )
        if operands.is_paged:
            table_uid = operands.page_table_k_tensor_uid
            if table_uid is None:
                table_uid = operands.page_table_v_tensor_uid
            _bound_paged(node, operands, graph, query, key, table_uid)
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
    operands: SdpaOperands,
    graph: _Graph,
    query: _Side,
    key: _Side,
    table_uid: int,
) -> None:
    """Bound a paged node: key lengths by the page allocation, query lengths
    by the query tensor (a packed query is split over the sequences)."""
    attributes = node.get("attributes") or {}
    table_dims = graph.dims(table_uid)
    key_tensor = graph.json.get(operands.k)
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
    q_main = graph.json.get(operands.q)
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
        query.q_label = _label(q_main)
        query.max_len = dims[2]
    else:
        query.max_len = dims[2]
        query.batch = query.batch or dims[0]
    query.operands = [t for t in q_tensors if ragged_fields(t)[0] is not None]


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


def _ranges(lengths: np.ndarray, gaps: np.ndarray, align: int) -> np.ndarray:
    """Token range of each batch: its length and gap, rounded up to ``align``."""
    return -(-(lengths + gaps) // align) * align


def _token_alignment(tables: Dict[int, List[Dict[str, Any]]], max_len: int) -> int:
    """Token multiple that every start must be, for every tensor to read its
    offset as a whole number: ``start * strides[2]`` must divide by the
    multiplier."""
    align = 1
    for uid, tensors in tables.items():
        for tensor in tensors:
            multiplier = ragged_fields(tensor)[1]
            if multiplier <= 0:
                raise UnsupportedGraphError(
                    f"{_label(tensor)} has ragged_offset_multiplier "
                    f"{multiplier}, which must be positive"
                )
            stride = int(tensor["strides"][2])
            align = math.lcm(align, multiplier // math.gcd(multiplier, stride))
            if align > max_len:
                raise UnsupportedGraphError(
                    f"{_label(tensor)}: ragged_offset_multiplier {multiplier} "
                    f"with token stride {stride} needs every start of offset "
                    f"table (uid {uid}) to be a multiple of {align} tokens, but "
                    f"a sequence holds {max_len}"
                )
    return align


def _interior_gap(
    lengths: np.ndarray,
    gaps: np.ndarray,
    align: int,
    limit: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """``gaps`` with one aligned block before a batch other than the last,
    unless such a batch already starts after unused tokens or none has room."""
    if len(lengths) < 2:
        return gaps
    ranges = _ranges(lengths, gaps, align)
    if np.any(ranges[:-1] > lengths[:-1]):
        return gaps
    candidates = np.flatnonzero(ranges[:-1] + align <= limit)
    if not candidates.size:
        return gaps
    gaps = gaps.copy()
    gaps[int(rng.choice(candidates))] += align
    return gaps


def _fit(
    lengths: np.ndarray,
    gaps: np.ndarray,
    capacity: int,
    limiter: Dict[str, Any],
    align: int,
    limit: int,
    keep_gap: bool,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Shrink gaps, then lengths, until the batches fit ``capacity`` tokens.

    With ``keep_gap``, one aligned gap before a batch other than the last
    survives when it fits, so some batch still starts after unused tokens.
    """
    blocks_capacity = capacity // align
    if len(lengths) > blocks_capacity:
        raise UnsupportedGraphError(
            f"{_label(limiter)} holds {capacity} tokens, fewer than the "
            f"{len(lengths)} batches that each need {align}"
        )
    if int(_ranges(lengths, gaps, align).sum()) <= capacity:
        return lengths, gaps
    gaps = np.zeros_like(gaps)
    if keep_gap:
        kept = _interior_gap(lengths, gaps, align, limit, rng)
        if int(_ranges(lengths, kept, align).sum()) <= capacity:
            return lengths, kept
    elif int(_ranges(lengths, gaps, align).sum()) <= capacity:
        return lengths, gaps
    # Scale the lengths to one block less than the capacity when a gap must
    # stay, and give that block to the gap.
    reserve = 1 if keep_gap and blocks_capacity > len(lengths) else 0
    target = blocks_capacity - reserve
    blocks = -(-lengths // align)
    blocks = np.maximum(1, blocks * target // int(blocks.sum()))
    while int(blocks.sum()) > target:
        blocks[int(np.argmax(blocks))] -= 1
    lengths = np.minimum(lengths, blocks * align)
    if reserve:
        gaps = _interior_gap(lengths, gaps, align, limit, rng)
    return lengths, gaps


def _fill(info: TensorInfo, values: Sequence[int], what: str) -> np.ndarray:
    if info.num_elements != len(values):
        raise UnsupportedGraphError(
            f"{info.label} has {info.num_elements} elements, but {what} needs "
            f"{len(values)}"
        )
    dtype = np.dtype(info.dtype.numpy)
    if dtype.kind in "iu" and len(values):
        limits = np.iinfo(dtype)
        if min(values) < limits.min or max(values) > limits.max:
            raise UnsupportedGraphError(
                f"{info.label} is {dtype.name}, which holds "
                f"{limits.min}..{limits.max}, but {what} needs "
                f"{min(values)}..{max(values)}"
            )
    return np.asarray(values, dtype=dtype).reshape(info.dims)


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
                return math.prod(dims)
            return math.prod(dims) - 1

    def label(uid: int) -> str:
        if uid in graph.json:
            return _label(graph.json[uid])
        if uid in graph.infos:
            return graph.infos[uid].label
        return f"tensor (uid {uid})"

    names = ", ".join(label(uid) for side in group for _, uid in side.tokens())
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
            uid = ragged_fields(tensor)[0]
            if uid is None:
                continue
            check_ragged_shape(*_shape(tensor), _label(tensor))
            if tensor not in tables.setdefault(uid, []):
                tables[uid].append(tensor)

    targets = [u for u in seq_uids + cu_uids + list(tables) if u in graph.infos]
    if not targets:
        return

    batch = _batch_count(group, graph)
    bounds = [s.max_len for s in group if s.max_len is not None]
    max_len = min(bounds) if bounds else 1
    align = _token_alignment(tables, max_len)
    lengths = _draw_lengths(group, batch, rng)

    gaps = np.zeros(batch, dtype=np.int64)
    if tables:
        # Every start is a multiple of `align`, so a range never passes `limit`.
        limit = max_len // align * align
        lengths = np.minimum(lengths, limit)
        if seq_uids or cu_uids:
            room = limit - lengths
            wanted = rng.random(batch) < 0.5
            drawn = rng.integers(1, np.maximum(room, 1) + 1)
            gaps = np.where(wanted & (room > 0), drawn, 0).astype(np.int64)
            # As a padded batch is forced, so is a gap before a later batch. A
            # gap after the last batch moves only the table's end entry, so no
            # batch would start after unused tokens.
            ranges = _ranges(lengths, gaps, align)
            if batch > 1 and not np.any(ranges[:-1] > lengths[:-1]):
                if not np.any(room[:-1] > 0) and room[-1] > 0:
                    # Every earlier batch is full: move the padded last one first.
                    for values in (lengths, room, drawn, gaps):
                        values[[0, -1]] = values[[-1, 0]]
                candidates = np.flatnonzero(room[:-1] > 0)
                if candidates.size:
                    chosen = int(rng.choice(candidates))
                    gaps[chosen] = drawn[chosen]
        else:
            # The lengths are the ranges, so they must be aligned themselves.
            lengths = np.maximum(align, lengths // align * align)
        keep_gap = batch > 1 and bool(seq_uids or cu_uids)
        for tensors in tables.values():
            for tensor in tensors:
                capacity = ragged_token_capacity(*_shape(tensor))
                if capacity is not None:
                    lengths, gaps = _fit(
                        lengths, gaps, capacity, tensor, align, limit, keep_gap, rng
                    )

    for uid in seq_uids:
        if uid in graph.infos:
            out[uid] = _fill(graph.infos[uid], lengths.tolist(), "seq_len")
    if cu_uids:
        totals = np.concatenate(([0], np.cumsum(lengths))).tolist()
        for uid in cu_uids:
            if uid in graph.infos:
                out[uid] = _fill(graph.infos[uid], totals, "cu_seq_len")

    starts = np.concatenate(([0], np.cumsum(_ranges(lengths, gaps, align))))
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
    must read the same way. The starts are multiples of the token alignment,
    so the division is exact."""
    values: Optional[np.ndarray] = None
    for tensor in tensors:
        scaled = starts * int(tensor["strides"][2]) // ragged_fields(tensor)[1]
        if values is not None and not np.array_equal(values, scaled):
            raise UnsupportedGraphError(
                f"{_label(tensor)} reads offset table '{info.name}' "
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
    the group has a ``seq_len`` or ``cu_seq_len`` that excludes the padding;
    then, with several batches, some batch other than the first does when a
    batch before the last has room for a gap.

    Raises :class:`UnsupportedGraphError` when tensors that share an offset
    table read it differently, the offset alignment exceeds a sequence, or the
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

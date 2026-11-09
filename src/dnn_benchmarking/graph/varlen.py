# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Variable-length and ragged SDPA: operands, lengths and layout resolution.

Shared by input generation, the PyTorch reference, comparison and metrics, so
it reads node JSON and numpy values only and never imports torch.
"""

from dataclasses import dataclass, replace
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from ..common.exceptions import UnsupportedGraphError
from .loader import output_uids, tensors_by_uid
from .tensor_info import TensorInfo, VarlenLayout, ragged_fields, tensor_label

_Node = Dict[str, Any]
ReadInts = Callable[[int], List[int]]
_Error = Callable[[str], Exception]


def _section(node: _Node, name: str) -> Dict[str, Any]:
    value = node.get(name)
    return value if isinstance(value, dict) else {}


def node_value(node: _Node, key: str, sections: Tuple[str, ...]) -> Any:
    """Value of ``key`` in the first of ``sections`` that has it, else at the
    node's top level. An explicit null counts as absent, as in the executor's
    ``_node_param``."""
    for name in sections:
        value = _section(node, name).get(key)
        if value is not None:
            return value
    return node.get(key)


def _optional_uid(node: _Node, key: str) -> Optional[int]:
    value = node_value(node, key, ("inputs", "outputs", "attributes", "parameters"))
    return None if value is None else int(value)


@dataclass(frozen=True)
class SdpaOperands:
    """Tensor UIDs of one SDPA node (forward or backward); None when absent.

    The query side is ``q, o, stats, do, dq``; the key side is ``k, v, dk, dv``.
    """

    q: Optional[int] = None
    o: Optional[int] = None
    stats: Optional[int] = None
    do: Optional[int] = None
    dq: Optional[int] = None
    k: Optional[int] = None
    v: Optional[int] = None
    dk: Optional[int] = None
    dv: Optional[int] = None
    seq_len_q_tensor_uid: Optional[int] = None
    seq_len_kv_tensor_uid: Optional[int] = None
    cu_seq_len_q_tensor_uid: Optional[int] = None
    cu_seq_len_kv_tensor_uid: Optional[int] = None
    page_table_k_tensor_uid: Optional[int] = None
    page_table_v_tensor_uid: Optional[int] = None

    @property
    def is_paged(self) -> bool:
        return (
            self.page_table_k_tensor_uid is not None
            or self.page_table_v_tensor_uid is not None
        )

    @property
    def q_side(self) -> Tuple[int, ...]:
        """UIDs of the operands whose sequence axis is the query length."""
        uids = (self.q, self.o, self.stats, self.do, self.dq)
        return tuple(uid for uid in uids if uid is not None)

    @property
    def kv_side(self) -> Tuple[int, ...]:
        """UIDs of the operands whose sequence axis is the key/value length."""
        uids = (self.k, self.v, self.dk, self.dv)
        return tuple(uid for uid in uids if uid is not None)

    def seq_len_uid(self, side: str) -> Optional[int]:
        return self.seq_len_q_tensor_uid if side == "q" else self.seq_len_kv_tensor_uid

    def cu_seq_len_uid(self, side: str) -> Optional[int]:
        return (
            self.cu_seq_len_q_tensor_uid
            if side == "q"
            else self.cu_seq_len_kv_tensor_uid
        )

    def side_uids(self, side: str) -> Tuple[int, ...]:
        return self.q_side if side == "q" else self.kv_side


def sdpa_operands(node: _Node) -> SdpaOperands:
    """Read the operand and length-source UIDs of an SDPA node."""
    return SdpaOperands(
        q=_optional_uid(node, "q_tensor_uid"),
        o=_optional_uid(node, "o_tensor_uid"),
        stats=_optional_uid(node, "stats_tensor_uid"),
        do=_optional_uid(node, "do_tensor_uid"),
        dq=_optional_uid(node, "dq_tensor_uid"),
        k=_optional_uid(node, "k_tensor_uid"),
        v=_optional_uid(node, "v_tensor_uid"),
        dk=_optional_uid(node, "dk_tensor_uid"),
        dv=_optional_uid(node, "dv_tensor_uid"),
        seq_len_q_tensor_uid=_optional_uid(node, "seq_len_q_tensor_uid"),
        seq_len_kv_tensor_uid=_optional_uid(node, "seq_len_kv_tensor_uid"),
        cu_seq_len_q_tensor_uid=_optional_uid(node, "cu_seq_len_q_tensor_uid"),
        cu_seq_len_kv_tensor_uid=_optional_uid(node, "cu_seq_len_kv_tensor_uid"),
        page_table_k_tensor_uid=_optional_uid(node, "page_table_k_tensor_uid"),
        page_table_v_tensor_uid=_optional_uid(node, "page_table_v_tensor_uid"),
    )


def has_ragged_operand(
    operands: SdpaOperands, tensors_by_uid: Dict[int, Dict[str, Any]]
) -> bool:
    """Whether a query-side or key-side operand is ragged (has an offset table)."""
    return any(
        ragged_fields(tensors_by_uid[uid])[0] is not None
        for uid in operands.q_side + operands.kv_side
        if uid in tensors_by_uid
    )


def has_varlen_source(node: _Node, tensors_by_uid: Dict[int, Dict[str, Any]]) -> bool:
    """Whether an SDPA node has a length source or a ragged operand.

    A length source is a ``seq_len`` or ``cu_seq_len`` tensor on either side.
    Paging is not tested here: a paged node always has ``seq_len_kv``.
    """
    operands = sdpa_operands(node)
    sources = (
        operands.seq_len_q_tensor_uid,
        operands.seq_len_kv_tensor_uid,
        operands.cu_seq_len_q_tensor_uid,
        operands.cu_seq_len_kv_tensor_uid,
    )
    return any(uid is not None for uid in sources) or has_ragged_operand(
        operands, tensors_by_uid
    )


def _is_backward(node: _Node, operands: SdpaOperands) -> bool:
    return "Backward" in str(node.get("type", "")) or operands.do is not None


def padding_rule(node: _Node, operands: SdpaOperands) -> None:
    """Check the ``padding_mask`` and length-source rules; raise ValueError.

    Each side takes ``seq_len`` or ``cu_seq_len``, never both, and the sides
    may mix. A non-paged node has a length source on both sides or on neither,
    and ``padding_mask`` needs both. ``cu_seq_len`` exists on forward only.
    """
    has_source = {}
    for side in ("q", "kv"):
        seq, cu = operands.seq_len_uid(side), operands.cu_seq_len_uid(side)
        if seq is not None and cu is not None:
            raise ValueError(
                f"SDPA {side} side has both seq_len and cu_seq_len; use one"
            )
        has_source[side] = seq is not None or cu is not None
    if _is_backward(node, operands) and (
        operands.cu_seq_len_q_tensor_uid is not None
        or operands.cu_seq_len_kv_tensor_uid is not None
    ):
        raise ValueError("cu_seq_len tensors exist on SDPA forward only")
    if operands.is_paged:
        for uid, name in (
            (operands.cu_seq_len_q_tensor_uid, "cu_seq_len_q"),
            (operands.cu_seq_len_kv_tensor_uid, "cu_seq_len_kv"),
        ):
            if uid is not None:
                raise ValueError(
                    f"Paged SDPA does not take {name}: tensor uid {uid} "
                    "must be seq_len_q/seq_len_kv instead"
                )
        return
    if has_source["q"] != has_source["kv"]:
        raise ValueError(
            "SDPA needs a length source (seq_len or cu_seq_len) on both the query "
            "and key side, or on neither"
        )
    if node_value(
        node, "padding_mask", ("parameters", "attributes", "inputs")
    ) and not (has_source["q"] and has_source["kv"]):
        raise ValueError(
            "SDPA padding_mask needs seq_len or cu_seq_len on both the query and "
            "key side"
        )


def check_ragged_shape(
    dims: Sequence[int],
    strides: Sequence[int],
    label: str,
    error: _Error = UnsupportedGraphError,
) -> None:
    """Refuse a ragged tensor that is not rank 4 with a positive token stride.

    Axes 1 and 3 need a positive stride when their size is above 1: a zero
    stride broadcasts values the reference keeps apart, and a negative one
    addresses storage before the batch start. ``strides[0]`` is not used.
    ``label`` names the tensor in the message; ``error`` is the type raised.
    """
    if len(dims) != 4 or len(strides) != 4:
        raise error(
            f"Ragged {label} must be rank 4, got dims {list(dims)} and strides "
            f"{list(strides)}"
        )
    if strides[2] <= 0:
        raise error(
            f"Ragged {label} needs a positive sequence stride, got strides "
            f"{list(strides)}"
        )
    if any(strides[i] < 0 or (strides[i] == 0 and dims[i] > 1) for i in (1, 3)):
        raise error(
            f"Ragged {label} needs a positive stride on each axis of size above "
            f"1 and no negative stride, got dims {list(dims)} and strides "
            f"{list(strides)}"
        )


def _outer_axes(dims: Sequence[int], strides: Sequence[int]) -> List[int]:
    """Axes 1 and 3, of size above 1, whose stride steps past a whole token.

    Axis 1 (heads) is outer when its stride is at least the sequence stride,
    axis 3 when its stride is above it.
    """
    seq_stride = strides[2]
    return [
        i
        for i in (1, 3)
        if dims[i] > 1
        and (strides[i] >= seq_stride if i == 1 else strides[i] > seq_stride)
    ]


def ragged_token_capacity(dims: Sequence[int], strides: Sequence[int]) -> Optional[int]:
    """Tokens a ragged rank-4 tensor can hold in total, None when unbounded.

    A head-major tensor stores its token axis inside the head stride, so it
    holds ``strides[1] // strides[2]`` tokens. A token-major tensor has no
    bound.
    """
    capacities = [strides[i] // strides[2] for i in _outer_axes(dims, strides)]
    return min(capacities) if capacities else None


def offset_spans(
    offsets: Sequence[int],
    multiplier: int,
    seq_stride: int,
    label: str,
    error: _Error = UnsupportedGraphError,
) -> List[int]:
    """Tokens in each batch's offset range, ``(off[b+1] - off[b]) * multiplier
    / seq_stride``.

    Raises ``error``, naming the tensor ``label``, for a multiplier below 1 or
    a range that is not a whole number of sequence strides.
    """
    if multiplier < 1:
        raise error(
            f"{label} has ragged_offset_multiplier {multiplier}, which must be "
            "positive"
        )
    spans = []
    for b, (lo, hi) in enumerate(zip(offsets, offsets[1:])):
        span = (hi - lo) * multiplier
        if span % seq_stride:
            raise error(
                f"Ragged {label} batch {b} spans {span} elements, not a whole "
                f"number of sequence strides ({seq_stride})"
            )
        spans.append(span // seq_stride)
    return spans


def _offset_lengths(
    tensor: Dict[str, Any], uid: int, read_ints: ReadInts
) -> Optional[List[int]]:
    """Per-batch lengths from a ragged tensor's offset table, or None."""
    offset_uid, multiplier = ragged_fields(tensor)
    if offset_uid is None:
        return None
    label = tensor_label(uid, tensor.get("name"))
    dims = [int(d) for d in tensor.get("dims") or []]
    strides = [int(s) for s in tensor.get("strides") or []]
    check_ragged_shape(dims, strides, label, ValueError)
    offsets = [int(x) for x in read_ints(offset_uid)]
    return offset_spans(offsets, multiplier, strides[2], label, ValueError)


def _side_lengths(
    operands: SdpaOperands,
    side: str,
    tensors_by_uid: Dict[int, Dict[str, Any]],
    read_ints: ReadInts,
) -> Optional[List[int]]:
    seq, cu = operands.seq_len_uid(side), operands.cu_seq_len_uid(side)
    if seq is not None and cu is not None:
        raise ValueError(f"SDPA {side} side has both seq_len and cu_seq_len; use one")
    if seq is not None:
        return [int(x) for x in read_ints(seq)]
    if cu is not None:
        totals = [int(x) for x in read_ints(cu)]
        return [hi - lo for lo, hi in zip(totals, totals[1:])]
    for uid in operands.side_uids(side):
        tensor = tensors_by_uid.get(uid)
        lengths = None if tensor is None else _offset_lengths(tensor, uid, read_ints)
        if lengths is not None:
            return lengths
    return None


def _full_length(
    operands: SdpaOperands, side: str, tensors_by_uid: Dict[int, Dict[str, Any]]
) -> int:
    for uid in operands.side_uids(side):
        tensor = tensors_by_uid.get(uid)
        if tensor and tensor.get("dims"):
            return int(tensor["dims"][-2])
    raise ValueError(f"SDPA node has no {side}-side tensor to take a length from")


def sdpa_lengths(
    node: _Node,
    tensors_by_uid: Dict[int, Dict[str, Any]],
    read_ints: ReadInts,
) -> Optional[Tuple[List[int], List[int]]]:
    """Per-batch ``(q_lengths, kv_lengths)`` of an SDPA node, None when dense.

    Each side takes ``seq_len``, else ``cu_seq_len`` differences, else the
    offset-table differences of its first ragged operand. A side without a
    source gets its full sequence length. Paged: kv lengths come from
    ``seq_len_kv``; q lengths from ``seq_len_q`` or ``total_q / num_seqs``.
    Raises ValueError for a malformed source.
    """
    operands = sdpa_operands(node)
    if operands.is_paged:
        if operands.seq_len_kv_tensor_uid is None:
            raise ValueError("Paged SDPA needs seq_len_kv to bound each sequence")
        kv = [int(x) for x in read_ints(operands.seq_len_kv_tensor_uid)]
        if operands.seq_len_q_tensor_uid is not None:
            q = [int(x) for x in read_ints(operands.seq_len_q_tensor_uid)]
            if len(q) != len(kv):
                raise ValueError("Paged SDPA seq_len_q/seq_len_kv disagree on num_seqs")
            return q, kv
        total_q = _full_length(operands, "q", tensors_by_uid)
        if not kv or total_q % len(kv):
            raise ValueError(
                f"Paged SDPA cannot split {total_q} queries across {len(kv)} "
                "sequences without seq_len_q"
            )
        return [total_q // len(kv)] * len(kv), kv

    q = _side_lengths(operands, "q", tensors_by_uid, read_ints)
    kv = _side_lengths(operands, "kv", tensors_by_uid, read_ints)
    if q is None and kv is None:
        return None
    if q is None:
        q = [_full_length(operands, "q", tensors_by_uid)] * len(kv)
    if kv is None:
        kv = [_full_length(operands, "kv", tensors_by_uid)] * len(q)
    return q, kv


def live_rows(
    lq: int,
    lkv: int,
    is_causal: bool,
    window: Optional[int],
    bottom_right: bool,
) -> Tuple[int, int]:
    """Return the half-open range ``[start, stop)`` of query rows with a key.

    Arguments follow ``_sdpa_derive_mask``: row ``i`` keeps keys ``k`` with
    ``k <= i + shift`` and, with a window width ``w``, ``k > i + shift - w``,
    where ``shift`` is ``lkv - lq`` for bottom-right alignment and 0 otherwise.
    Leading rows with ``i + shift < 0`` have no key. With a window, trailing
    rows with ``i + shift - w >= lkv - 1`` have none either. ``start == stop``
    when no row has a key.
    """
    if lq <= 0 or lkv <= 0:
        return 0, 0
    if not (is_causal or window is not None or bottom_right):
        return 0, lq
    if window is not None and window <= 0:
        return 0, 0
    shift = lkv - lq if bottom_right else 0
    start = min(lq, max(0, -shift))
    stop = lq
    if window is not None:
        stop = min(lq, lkv - 1 + window - shift)
    return (start, stop) if start < stop else (0, 0)


_refuse = UnsupportedGraphError


def read_input_ints(input_data: Dict[int, Any], uid: int) -> List[int]:
    """The values of input ``uid`` as a flat list of Python ints."""
    return [int(x) for x in np.asarray(input_data[uid]).reshape(-1)]


def _check_source_inputs(
    node_label: str,
    uids: Sequence[int],
    computed: Set[int],
    input_data: Dict[int, Any],
    tensors_json: Dict[int, Dict[str, Any]],
) -> None:
    for uid in uids:
        label = tensor_label(uid, tensors_json.get(uid, {}).get("name"))
        if uid in computed:
            raise _refuse(
                f"{node_label}: length or offset {label} is computed in the "
                "graph; it must be an input"
            )
        if uid not in input_data:
            raise _refuse(f"{node_label}: length or offset {label} has no input data")


def _ragged_offsets(info: TensorInfo, batches: int, read_ints: ReadInts) -> List[int]:
    check_ragged_shape(info.dims, info.strides, info.label)
    offsets = read_ints(info.ragged_offset_tensor_uid)
    if len(offsets) != batches + 1:
        raise _refuse(
            f"Ragged {info.label} offset table (uid {info.ragged_offset_tensor_uid}) "
            f"has {len(offsets)} entries; {batches + 1} are needed for {batches} batches"
        )
    if offsets[0] != 0 or any(hi < lo for lo, hi in zip(offsets, offsets[1:])):
        raise _refuse(
            f"Ragged {info.label} offset table (uid {info.ragged_offset_tensor_uid}) "
            f"must start at 0 and not decrease, got {offsets}"
        )
    return offsets


def _check_overlap(info: TensorInfo, end: int) -> None:
    """A batch's tokens must not overlap the next token or the next slab.

    Only axis 1 may be outer (head-major), and the tokens up to element
    ``end`` must fit its ``ragged_token_capacity``.
    """
    seq_stride = info.strides[2]
    outer = _outer_axes(info.dims, info.strides)
    small_span = 1 + sum(
        (info.dims[i] - 1) * info.strides[i] for i in (1, 3) if i not in outer
    )
    if small_span > seq_stride:
        raise _refuse(
            f"Ragged {info.label} has inner dims {info.dims} with strides "
            f"{info.strides} that overlap the next token"
        )
    capacity = ragged_token_capacity(info.dims, info.strides)
    for axis in outer:
        if axis != 1 or -(-end // seq_stride) > capacity:
            raise _refuse(
                f"Ragged {info.label} has dims {info.dims} with strides "
                f"{info.strides}; axis {axis} overlaps the packed tokens"
            )


def _resolve_node(
    node: _Node,
    operands: SdpaOperands,
    infos: Dict[int, TensorInfo],
    tensors_json: Dict[int, Dict[str, Any]],
    computed: Set[int],
    input_data: Dict[int, Any],
    layouts: Dict[int, VarlenLayout],
) -> None:
    node_label = f"SDPA node ({node.get('name') or node.get('type')})"
    q_info = infos.get(operands.q) if operands.q is not None else None
    anchor = q_info or next(
        (infos[u] for u in operands.q_side + operands.kv_side if u in infos), None
    )
    if anchor is None:
        return
    try:
        padding_rule(node, operands)
    except ValueError as error:
        raise _refuse(f"{anchor.label}: {error}") from error

    sides = {"q": operands.q_side, "kv": operands.kv_side}
    side_infos = {
        side: [infos[uid] for uid in uids if uid in infos]
        for side, uids in sides.items()
    }
    ragged = [ti for tis in side_infos.values() for ti in tis if ti.is_ragged]
    sources = [
        uid
        for side in ("q", "kv")
        for uid in (operands.seq_len_uid(side), operands.cu_seq_len_uid(side))
        if uid is not None
    ] + [ti.ragged_offset_tensor_uid for ti in ragged]
    if not sources:
        return
    _check_source_inputs(node_label, sources, computed, input_data, tensors_json)
    read = partial(read_input_ints, input_data)
    names = {uid: t.get("name", uid) for uid, t in tensors_json.items()}

    batches = anchor.dims[0]
    for side in ("q", "kv"):
        uid = operands.seq_len_uid(side)
        if uid is not None and len(read(uid)) != batches:
            raise _refuse(
                f"{anchor.label}: seq_len '{names.get(uid, uid)}' (uid {uid}) has "
                f"{len(read(uid))} entries for {batches} batches"
            )
    for side in ("q", "kv"):
        uid = operands.cu_seq_len_uid(side)
        if uid is not None and len(read(uid)) != batches + 1:
            raise _refuse(
                f"{anchor.label}: cu_seq_len '{names.get(uid, uid)}' (uid {uid}) has "
                f"{len(read(uid))} entries; {batches + 1} are needed for {batches} "
                "batches"
            )
    offsets = {ti.uid: _ragged_offsets(ti, batches, read) for ti in ragged}

    try:
        resolved = sdpa_lengths(node, tensors_json, read)
    except ValueError as error:
        raise _refuse(f"{anchor.label}: {error}") from error
    if resolved is None:
        return

    for side, lengths in zip(("q", "kv"), resolved):
        explicit = (
            operands.seq_len_uid(side) is not None
            or operands.cu_seq_len_uid(side) is not None
        )
        if len(lengths) != batches:
            raise _refuse(
                f"{anchor.label}: the {side} side has {len(lengths)} lengths for "
                f"{batches} batches"
            )
        for info in side_infos[side]:
            if info.dims[0] != batches:
                raise _refuse(
                    f"{info.label} has {info.dims[0]} batches; the node has {batches}"
                )
            bad = [n for n in lengths if n < 0 or n > info.dims[-2]]
            if bad:
                raise _refuse(
                    f"{info.label} has lengths {bad} outside 0..{info.dims[-2]}"
                )
            if not info.is_ragged:
                continue
            table = offsets[info.uid]
            mult = info.ragged_offset_multiplier
            spans = offset_spans(table, mult, info.strides[2], info.label)
            for b, packed in enumerate(spans):
                if (lengths[b] > packed) if explicit else (lengths[b] != packed):
                    raise _refuse(
                        f"{info.label} batch {b}: length {lengths[b]} "
                        + (
                            f"exceeds its packed range of {packed} tokens"
                            if explicit
                            else f"disagrees with its offset range of {packed} tokens"
                        )
                    )
            _check_overlap(info, table[-1] * mult)

    for side, lengths in zip(("q", "kv"), resolved):
        for info in side_infos[side]:
            if info.is_ragged:
                table = offsets[info.uid]
                mult = info.ragged_offset_multiplier
                layout = VarlenLayout(
                    lengths=tuple(lengths),
                    starts=tuple(off * mult for off in table[:-1]),
                    end=table[-1] * mult,
                )
            else:
                layout = VarlenLayout(lengths=tuple(lengths))
            if layouts.setdefault(info.uid, layout) != layout:
                raise _refuse(
                    f"{info.label} gets different varlen layouts from different "
                    "SDPA nodes"
                )


def resolve_varlen_layouts(
    tensor_infos: List[TensorInfo],
    graph_json: Dict[str, Any],
    input_data: Dict[int, Any],
) -> List[TensorInfo]:
    """Set the comparison and layout fields that the graph decides per tensor.

    ``varlen`` goes on every operand of each non-paged varlen SDPA node, and
    ``matching_infinities`` on the ``stats`` output of each SDPA forward node:
    rows with no live key hold a log-sum-exp of -inf in both the engine and
    the reference.

    Returns copies for the tensors that gain a field and the input list itself
    when none does. Raises UnsupportedGraphError, naming the tensor, for any
    layout the benchmark cannot generate, store or reference faithfully.
    """
    infos = {ti.uid: ti for ti in tensor_infos}
    tensors_json = tensors_by_uid(graph_json)
    computed = output_uids(graph_json) | {
        uid for uid, t in tensors_json.items() if t.get("virtual")
    }
    layouts: Dict[int, VarlenLayout] = {}
    stats: Set[int] = set()
    claimed: Set[int] = set()
    for node in graph_json.get("nodes") or []:
        if not str(node.get("type", "")).startswith("Sdpa"):
            continue
        operands = sdpa_operands(node)
        if node.get("type") == "SdpaAttributes" and operands.stats in infos:
            stats.add(operands.stats)
        operand_infos = [
            infos[uid] for uid in operands.q_side + operands.kv_side if uid in infos
        ]
        if operands.is_paged:
            for info in operand_infos:
                if info.is_ragged:
                    raise _refuse(
                        f"Ragged {info.label} is an operand of a paged SDPA node, "
                        "which is not supported"
                    )
            try:
                padding_rule(node, operands)
            except ValueError as error:
                where = operand_infos[0].label if operand_infos else "paged SDPA node"
                raise _refuse(f"{where}: {error}") from error
            continue
        claimed.update(info.uid for info in operand_infos)
        _resolve_node(
            node, operands, infos, tensors_json, computed, input_data, layouts
        )
    for info in tensor_infos:
        if info.is_ragged and info.uid not in claimed:
            raise _refuse(
                f"Ragged {info.label} is not an operand of a non-paged SDPA node"
            )
    if not layouts and not stats:
        return tensor_infos
    return [
        (
            replace(
                info,
                varlen=layouts.get(info.uid, info.varlen),
                matching_infinities=info.uid in stats or info.matching_infinities,
            )
            if info.uid in layouts or info.uid in stats
            else info
        )
        for info in tensor_infos
    ]

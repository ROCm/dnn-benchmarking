# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Variable-length and ragged SDPA: operands, lengths and layout resolution.

Shared by input generation, the PyTorch reference, comparison and metrics, so
it reads node JSON and numpy values only and never imports torch.
"""

from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from ..common.exceptions import UnsupportedGraphError
from .loader import output_uids
from .tensor_info import TensorInfo, VarlenLayout

_Node = Dict[str, Any]
_ReadInts = Callable[[int], Sequence[int]]


def _section(node: _Node, name: str) -> Dict[str, Any]:
    value = node.get(name)
    return value if isinstance(value, dict) else {}


def _lookup(node: _Node, key: str, sections: Tuple[str, ...]) -> Any:
    # An explicit null counts as absent, as in the executor's ``_node_param``.
    for name in sections:
        value = _section(node, name).get(key)
        if value is not None:
            return value
    return node.get(key)


def _optional_uid(node: _Node, key: str) -> Optional[int]:
    value = _lookup(node, key, ("inputs", "outputs", "attributes", "parameters"))
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
    if _lookup(node, "padding_mask", ("parameters", "attributes", "inputs")) and not (
        has_source["q"] and has_source["kv"]
    ):
        raise ValueError(
            "SDPA padding_mask needs seq_len or cu_seq_len on both the query and "
            "key side"
        )


def _tensor_label(tensor: Dict[str, Any], uid: int) -> str:
    return f"tensor '{tensor.get('name', f'tensor_{uid}')}' (uid {uid})"


def _offset_lengths(
    tensor: Dict[str, Any], uid: int, read_ints: _ReadInts
) -> Optional[List[int]]:
    """Per-batch lengths from a ragged tensor's offset table, or None."""
    offset_uid = tensor.get("ragged_offset_tensor_uid")
    if offset_uid is None:
        return None
    multiplier = int(tensor.get("ragged_offset_multiplier") or 1)
    strides = tensor.get("strides") or []
    seq_stride = int(strides[-2]) if len(strides) >= 2 else 0
    offsets = [int(x) for x in read_ints(int(offset_uid))]
    if seq_stride <= 0:
        raise ValueError(
            f"Ragged {_tensor_label(tensor, uid)} needs a positive sequence stride"
        )
    lengths = []
    for lo, hi in zip(offsets, offsets[1:]):
        span = (hi - lo) * multiplier
        if span % seq_stride:
            raise ValueError(
                f"Ragged {_tensor_label(tensor, uid)} has a range of {span} elements, "
                f"not a whole number of sequence strides ({seq_stride})"
            )
        lengths.append(span // seq_stride)
    return lengths


def _side_lengths(
    operands: SdpaOperands,
    side: str,
    tensors_by_uid: Dict[int, Dict[str, Any]],
    read_ints: _ReadInts,
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
    read_ints: _ReadInts,
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


def _label(info: TensorInfo) -> str:
    return f"tensor '{info.name}' (uid {info.uid})"


_refuse = UnsupportedGraphError


def _read_input_ints(input_data: Dict[int, Any]) -> _ReadInts:
    def read(uid: int) -> List[int]:
        return [int(x) for x in np.asarray(input_data[uid]).reshape(-1)]

    return read


def _check_source_inputs(
    node_label: str,
    uids: Sequence[int],
    computed: Set[int],
    input_data: Dict[int, Any],
    names: Dict[int, str],
) -> None:
    for uid in uids:
        if uid in computed:
            raise _refuse(
                f"{node_label}: length or offset tensor '{names.get(uid, uid)}' "
                f"(uid {uid}) is computed in the graph; it must be an input"
            )
        if uid not in input_data:
            raise _refuse(
                f"{node_label}: length or offset tensor '{names.get(uid, uid)}' "
                f"(uid {uid}) has no input data"
            )


def _ragged_offsets(info: TensorInfo, batches: int, read_ints: _ReadInts) -> List[int]:
    if len(info.dims) != 4 or len(info.strides) != 4:
        raise _refuse(f"Ragged {_label(info)} must be rank 4, got dims {info.dims}")
    if info.strides[2] <= 0:
        raise _refuse(
            f"Ragged {_label(info)} needs a positive sequence stride, got "
            f"strides {info.strides}"
        )
    offsets = read_ints(info.ragged_offset_tensor_uid)
    if len(offsets) != batches + 1:
        raise _refuse(
            f"Ragged {_label(info)} offset table (uid {info.ragged_offset_tensor_uid}) "
            f"has {len(offsets)} entries; {batches + 1} are needed for {batches} batches"
        )
    if offsets[0] != 0 or any(hi < lo for lo, hi in zip(offsets, offsets[1:])):
        raise _refuse(
            f"Ragged {_label(info)} offset table (uid {info.ragged_offset_tensor_uid}) "
            f"must start at 0 and not decrease, got {offsets}"
        )
    return offsets


def _check_overlap(info: TensorInfo, end: int) -> None:
    """A batch's tokens must not overlap the next token or the next slab.

    Axis 1 is the outer (head) axis when its stride is at least the sequence
    stride. It then holds ``strides[1] // strides[2]`` tokens per head.
    """
    seq_stride = info.strides[2]
    inner = [i for i in (1, 3) if info.dims[i] > 1]

    def is_inner(i: int) -> bool:
        if i == 1:
            return info.strides[i] < seq_stride
        return info.strides[i] <= seq_stride

    small_span = 1 + sum(
        (info.dims[i] - 1) * info.strides[i] for i in inner if is_inner(i)
    )
    if small_span > seq_stride:
        raise _refuse(
            f"Ragged {_label(info)} has inner dims {info.dims} with strides "
            f"{info.strides} that overlap the next token"
        )
    for i in inner:
        if is_inner(i):
            continue
        capacity = info.strides[i] // seq_stride
        if i != 1 or -(-end // seq_stride) > capacity:
            raise _refuse(
                f"Ragged {_label(info)} has dims {info.dims} with strides "
                f"{info.strides}; axis {i} overlaps the packed tokens"
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
        raise _refuse(f"{_label(anchor)}: {error}") from error

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
    names = {uid: t.get("name", uid) for uid, t in tensors_json.items()}
    _check_source_inputs(node_label, sources, computed, input_data, names)
    read = _read_input_ints(input_data)

    batches = anchor.dims[0]
    for side in ("q", "kv"):
        uid = operands.seq_len_uid(side)
        if uid is not None and len(read(uid)) != batches:
            raise _refuse(
                f"{_label(anchor)}: seq_len '{names.get(uid, uid)}' (uid {uid}) has "
                f"{len(read(uid))} entries for {batches} batches"
            )
    for side in ("q", "kv"):
        uid = operands.cu_seq_len_uid(side)
        if uid is not None and len(read(uid)) != batches + 1:
            raise _refuse(
                f"{_label(anchor)}: cu_seq_len '{names.get(uid, uid)}' (uid {uid}) has "
                f"{len(read(uid))} entries; {batches + 1} are needed for {batches} "
                "batches"
            )
    offsets = {ti.uid: _ragged_offsets(ti, batches, read) for ti in ragged}

    try:
        resolved = sdpa_lengths(node, tensors_json, read)
    except ValueError as error:
        raise _refuse(f"{_label(anchor)}: {error}") from error
    if resolved is None:
        return

    for side, lengths in zip(("q", "kv"), resolved):
        explicit = (
            operands.seq_len_uid(side) is not None
            or operands.cu_seq_len_uid(side) is not None
        )
        if len(lengths) != batches:
            raise _refuse(
                f"{_label(anchor)}: the {side} side has {len(lengths)} lengths for "
                f"{batches} batches"
            )
        for info in side_infos[side]:
            if info.dims[0] != batches:
                raise _refuse(
                    f"{_label(info)} has {info.dims[0]} batches; the node has {batches}"
                )
            bad = [n for n in lengths if n < 0 or n > info.dims[-2]]
            if bad:
                raise _refuse(
                    f"{_label(info)} has lengths {bad} outside 0..{info.dims[-2]}"
                )
            if not info.is_ragged:
                continue
            table = offsets[info.uid]
            seq_stride = info.strides[2]
            for b, (lo, hi) in enumerate(zip(table, table[1:])):
                span = (hi - lo) * info.ragged_offset_multiplier
                if span % seq_stride:
                    raise _refuse(
                        f"Ragged {_label(info)} batch {b} spans {span} elements, not a "
                        f"whole number of sequence strides ({seq_stride})"
                    )
                packed = span // seq_stride
                if (lengths[b] > packed) if explicit else (lengths[b] != packed):
                    raise _refuse(
                        f"{_label(info)} batch {b}: length {lengths[b]} "
                        + (
                            f"exceeds its packed range of {packed} tokens"
                            if explicit
                            else f"disagrees with its offset range of {packed} tokens"
                        )
                    )
            _check_overlap(info, table[-1] * info.ragged_offset_multiplier)

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
                    f"{_label(info)} gets different varlen layouts from different "
                    "SDPA nodes"
                )


def resolve_varlen_layouts(
    tensor_infos: List[TensorInfo],
    graph_json: Dict[str, Any],
    input_data: Dict[int, Any],
) -> List[TensorInfo]:
    """Set ``varlen`` on every operand of each non-paged varlen SDPA node.

    Returns copies for the tensors that gain a layout and the input list itself
    when none does. Raises UnsupportedGraphError, naming the tensor, for any
    layout the benchmark cannot generate, store or reference faithfully.
    """
    infos = {ti.uid: ti for ti in tensor_infos}
    tensors_json = {
        int(t["uid"]): t for t in graph_json.get("tensors") or [] if "uid" in t
    }
    computed = output_uids(graph_json) | {
        uid for uid, t in tensors_json.items() if t.get("virtual")
    }
    layouts: Dict[int, VarlenLayout] = {}
    claimed: Set[int] = set()
    for node in graph_json.get("nodes") or []:
        if not str(node.get("type", "")).startswith("Sdpa"):
            continue
        operands = sdpa_operands(node)
        operand_infos = [
            infos[uid] for uid in operands.q_side + operands.kv_side if uid in infos
        ]
        if operands.is_paged:
            for info in operand_infos:
                if info.is_ragged:
                    raise _refuse(
                        f"Ragged {_label(info)} is an operand of a paged SDPA node, "
                        "which is not supported"
                    )
            try:
                padding_rule(node, operands)
            except ValueError as error:
                where = _label(operand_infos[0]) if operand_infos else "paged SDPA node"
                raise _refuse(f"{where}: {error}") from error
            continue
        claimed.update(info.uid for info in operand_infos)
        _resolve_node(
            node, operands, infos, tensors_json, computed, input_data, layouts
        )
    for info in tensor_infos:
        if info.is_ragged and info.uid not in claimed:
            raise _refuse(
                f"Ragged {_label(info)} is not an operand of a non-paged SDPA node"
            )
    if not layouts:
        return tensor_infos
    return [
        replace(info, varlen=layouts[info.uid]) if info.uid in layouts else info
        for info in tensor_infos
    ]

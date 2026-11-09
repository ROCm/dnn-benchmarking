# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Scaled-dot-product-attention forward/backward reference handlers."""

from typing import Any, Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple

import torch

from ....common.exceptions import UnsupportedGraphError
from ....config.benchmark_config import PyTorchSdpaBackendName
from ....graph.loader import tensors_by_uid
from ....graph.tensor_info import ragged_fields
from ....graph.varlen import (
    has_varlen_source,
    live_rows,
    padding_rule,
    sdpa_lengths,
    sdpa_operands,
)
from .._common import *  # noqa: F401,F403
from .._registry import CompiledOp, register_handler
from .._sdpa_backend import (
    default_sdpa_avoids_math,
    execute_selected_sdpa,
    fused_logsumexp,
    strict_sdpa_selection,
)

_STATS_FALLBACK_WARNING = (
    "PyTorch SDPA: the kernel saved no log-sum-exp, so the timed region "
    "includes a separate fp32 score matmul and logsumexp for the stats output"
)


class _PagedSegment(NamedTuple):
    """One paged sequence, prepared untimed.

    ``k`` and ``v`` are the dense K and V for the SDPA call. With sinks they
    are the fp32 K^T and V of the manual attention instead. ``batch`` is the
    row of a per-batch Q/O to read and write; ``None`` for a packed Q/O,
    whose segments are concatenated along the query axis instead.
    """

    q_start: int
    q_len: int
    k: torch.Tensor
    v: torch.Tensor
    mask: Optional[torch.Tensor]
    is_causal: bool
    enable_gqa: bool
    batch: Optional[int] = None


class _PagedSdpa(NamedTuple):
    """Setup of a paged forward, built once. ``per_batch``: Q and O have one
    batch row per sequence (see ``_plan_paged_sdpa``)."""

    segments: List[_PagedSegment]
    scale: float
    q_float: Optional[torch.Tensor]
    sink: Optional[torch.Tensor]
    per_batch: bool


class _DenseSdpa(NamedTuple):
    """Setup of a dense forward that calls the SDPA kernel, built once."""

    attn_mask: Optional[torch.Tensor]
    is_causal: bool
    scale: float
    rep_k: int
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    enable_gqa: bool


class _SinkSdpa(NamedTuple):
    """Setup of a dense forward with sinks (manual fp32 attention)."""

    scale: float
    q_float: torch.Tensor
    k_t: torch.Tensor
    v_float: torch.Tensor
    mask: Optional[torch.Tensor]
    sink: torch.Tensor


class _VarlenBatch(NamedTuple):
    """One batch of a varlen forward, prepared untimed.

    The kernel path holds the SDPA operands. The sink path holds the fp32 Q,
    K^T and V of the manual attention instead. ``o_rows`` and ``stats_rows``
    are views of the preallocated outputs for the batch's live rows.
    """

    index: int
    rows: int
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    k_raw: Optional[torch.Tensor]
    mask: Optional[torch.Tensor]
    is_causal: bool
    o_rows: torch.Tensor
    stats_rows: Optional[torch.Tensor]
    enable_gqa: bool


class _VarlenSdpa(NamedTuple):
    """Setup of a non-paged varlen forward, built once."""

    batches: List[_VarlenBatch]
    scale: float
    rep_k: int
    sink: Optional[torch.Tensor]
    o: torch.Tensor
    stats: Optional[torch.Tensor]


def _sdpa_bool(node: Dict[str, Any], key: str, default: bool = False) -> bool:
    return bool(_node_param(node, key, default))


def _unsupported_optional(key: str) -> ValueError:
    return ValueError(f"Unsupported SDPA optional tensor '{key}' in PyTorch reference")


def _sdpa_unsupported_if_present(node: Dict[str, Any], keys: Sequence[str]) -> None:
    for key in keys:
        if _optional_uid(node, key) is not None:
            raise _unsupported_optional(key)


def _sdpa_resolve_scale(
    node: Dict[str, Any],
    tensors: Dict[int, torch.Tensor],
    scale_uid: Optional[int],
    attn_scale_value: Any,
) -> float:
    # hipDNN reads an absent attn_scale as 1.0 (cuDNN semantics), not 1/sqrt(d).
    if scale_uid is not None:
        return _scalar_value(tensors, scale_uid, node)
    return 1.0 if attn_scale_value is None else float(attn_scale_value)


def _sdpa_head_repeat(q_heads: int, kv_heads: int, label: str) -> int:
    """Validate and return the per-query-head repeat factor for K or V.

    hipDNN allows independent K and V head counts; each must divide the query
    head count (frontend SdpaBwdNode validation), and the CPU reference maps K
    and V with separate ratios.
    """
    if kv_heads <= 0 or q_heads % kv_heads != 0:
        raise ValueError(
            f"Unsupported SDPA {label} head count: q_heads={q_heads}, "
            f"{label.lower()}_heads={kv_heads}"
        )
    return q_heads // kv_heads


class _SdpaPlan(NamedTuple):
    mask_uid: Optional[int]
    dropout_p: float
    is_causal: bool
    scale_uid: Optional[int]
    attn_scale_value: Any
    window: Optional[int]
    bottom_right: bool
    sink_uid: Optional[int]


def _plan_sdpa_common(
    node: Dict[str, Any],
    allow_paged: bool = False,
    allow_sink: bool = False,
) -> _SdpaPlan:
    unsupported = [
        "seed_tensor_uid",
        "offset_tensor_uid",
        "dropout_mask_tensor_uid",
        "dropout_scale_tensor_uid",
        "block_mask_tensor_uid",
        "descale_q_tensor_uid",
        "descale_k_tensor_uid",
        "descale_v_tensor_uid",
        "descale_s_tensor_uid",
        "scale_s_tensor_uid",
        "scale_o_tensor_uid",
    ]
    if not allow_sink:
        unsupported.append("sink_token_tensor_uid")
    _sdpa_unsupported_if_present(node, unsupported)
    operands = sdpa_operands(node)
    if not allow_paged:
        # Page tables and the cumulative lengths are served by the forward
        # handler only: it gathers the page table into dense per-sequence K/V
        # before calling PyTorch, which has no paged API. Backward has no such
        # path, so for it these remain hard rejections rather than a silently
        # dense gradient.
        for key in (
            "cu_seq_len_q_tensor_uid",
            "cu_seq_len_kv_tensor_uid",
            "page_table_k_tensor_uid",
            "page_table_v_tensor_uid",
        ):
            if getattr(operands, key) is not None:
                raise _unsupported_optional(key)

    if _sdpa_bool(node, "alibi_mask"):
        raise ValueError("SDPA alibi masks are not supported by the PyTorch reference")
    try:
        padding_rule(node, operands)
    except (ValueError, UnsupportedGraphError) as e:
        raise ValueError(f"SDPA padding_mask rule: {e}") from e

    dropout_probability = _node_param(node, "dropout_probability", 0.0)
    dropout_p = 0.0 if dropout_probability is None else float(dropout_probability)
    if dropout_p != 0.0:
        raise ValueError(
            "Nonzero SDPA dropout cannot be exactly validated against PyTorch"
        )

    mask_uid = _optional_uid(node, "attn_mask_tensor_uid")
    is_causal, window, bottom_right = _sdpa_derive_mask(node)
    if mask_uid is not None and (is_causal or window is not None or bottom_right):
        raise ValueError(
            "PyTorch SDPA reference does not support both attn_mask and causal_mask"
        )

    scale_uid = _optional_uid(node, "scale_tensor_uid")
    attn_scale_value = _node_param(node, "attn_scale_value", None)
    sink_uid = _optional_uid(node, "sink_token_tensor_uid")
    return _SdpaPlan(
        mask_uid=mask_uid,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale_uid=scale_uid,
        attn_scale_value=attn_scale_value,
        window=window,
        bottom_right=bottom_right,
        sink_uid=sink_uid,
    )


def _sdpa_derive_mask(
    node: Dict[str, Any],
) -> Tuple[bool, Optional[int], bool]:
    """Resolve ``(is_causal, window_width, bottom_right)`` as hipDNN does.

    The deprecated causal booleans take precedence over the modern bounds and
    alignment. Otherwise, bounds describe a diagonal band, with ``-1`` meaning
    unbounded. ``left_bound = L`` keeps ``L`` keys up to and including the
    diagonal, as in hipDNN and cuDNN, so the window width is ``L``; hipDNN
    rejects ``left_bound = 0``. ``is_causal`` is reserved for PyTorch's native
    top-left mask; bottom-right masks are materialized as additive masks.
    """
    causal = _sdpa_bool(node, "causal_mask")
    bottom_right_causal = _sdpa_bool(node, "causal_mask_bottom_right")
    if causal and bottom_right_causal:
        raise ValueError(
            "SDPA causal_mask and causal_mask_bottom_right are mutually exclusive"
        )
    if causal:
        return True, None, False
    if bottom_right_causal:
        return False, None, True

    alignment = _node_param(node, "diagonal_alignment", "TOP_LEFT")
    if alignment not in ("TOP_LEFT", "BOTTOM_RIGHT", 0, 1, None):
        raise ValueError(f"Unsupported SDPA diagonal_alignment {alignment!r}")
    bottom_right = alignment in ("BOTTOM_RIGHT", 1)

    unbounded = -1
    left = _node_param(node, "left_bound", unbounded)
    right = _node_param(node, "right_bound", unbounded)
    left = unbounded if left is None else int(left)
    right = unbounded if right is None else int(right)

    if left < unbounded or left == 0:
        raise ValueError(
            f"SDPA left_bound {left} is neither unbounded (-1) nor a positive width"
        )
    if right != 0:
        if right == unbounded and left == unbounded:
            return False, None, False
        raise ValueError(
            f"SDPA right_bound {right} is a forward-looking band the reference "
            "cannot express"
        )
    if left == unbounded:
        return (False, None, True) if bottom_right else (True, None, False)
    return False, left, bottom_right


def _sdpa_resolve(
    node: Dict[str, Any],
    tensors: Dict[int, torch.Tensor],
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask_uid: Optional[int],
    scale_uid: Optional[int],
    attn_scale_value: Any,
) -> Tuple[Optional[torch.Tensor], float, int, int]:
    attn_mask = _tensor(tensors, mask_uid, node) if mask_uid is not None else None
    scale = _sdpa_resolve_scale(node, tensors, scale_uid, attn_scale_value)
    if q.ndim < 3 or k.ndim < 3 or v.ndim < 3:
        raise ValueError("SDPA expects q/k/v tensors with head and matrix dimensions")
    q_heads = int(q.shape[-3])
    rep_k = _sdpa_head_repeat(q_heads, int(k.shape[-3]), "K")
    rep_v = _sdpa_head_repeat(q_heads, int(v.shape[-3]), "V")
    return attn_mask, scale, rep_k, rep_v


def _diagonal_band_mask(
    q_len: int,
    kv_len: int,
    width: Optional[int],
    *,
    bottom_right: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build the additive diagonal-band mask used by hipDNN SDPA.

    Bottom-right alignment shifts the diagonal by ``kv_len - q_len``. A
    ``None`` width leaves the lower edge unbounded, producing a causal mask.
    """
    q_idx = torch.arange(q_len, device=device).unsqueeze(-1)
    k_idx = torch.arange(kv_len, device=device).unsqueeze(0)
    diagonal = q_idx + (kv_len - q_len if bottom_right else 0)
    keep = k_idx <= diagonal
    if width is not None:
        keep &= k_idx > diagonal - width
    mask = torch.zeros((q_len, kv_len), device=device, dtype=dtype)
    return mask.masked_fill(~keep, float("-inf"))


def _attention_bounds(
    q_len: int,
    kv_len: int,
    window: Optional[int],
    bottom_right: bool,
    is_causal: bool,
    like: torch.Tensor,
) -> Tuple[Optional[torch.Tensor], bool]:
    """The ``(attn_mask, is_causal)`` pair that expresses a call's bounds.

    torch's SDPA spells only top-left causal natively (``is_causal``). A window
    or a bottom-right alignment needs an additive band mask, unless the bounds
    equal something native: bottom-right causal with ``q_len == kv_len`` is
    top-left causal, and with ``q_len == 1`` (decode) every key is visible.
    Passing those natively keeps the fused kernels eligible, since Flash
    rejects any ``attn_mask``.
    """
    if window is None and not bottom_right:
        return None, is_causal
    if window is None:
        if q_len == kv_len:
            return None, True
        if q_len == 1:
            return None, False
    mask = _diagonal_band_mask(
        q_len,
        kv_len,
        window,
        bottom_right=bottom_right,
        device=like.device,
        dtype=like.dtype,
    )
    return mask, False


def _sink_logits(sink: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """fp32 sink logits shaped to broadcast against scores shaped like ``like``
    (``[..., heads, q_len, kv_len]``)."""
    heads = int(like.shape[-3])
    if sink.numel() != heads:
        raise ValueError(
            f"SDPA sink tensor has {sink.numel()} values for {heads} query heads"
        )
    shape = (1,) * (like.ndim - 3) + (heads, 1, 1)
    return sink.to(dtype=torch.float32).reshape(shape)


def _sdpa_kv_operands(
    k: torch.Tensor, v: torch.Tensor, rep_k: int, rep_v: int
) -> Tuple[torch.Tensor, torch.Tensor, bool]:
    """K and V as the SDPA call takes them, plus its ``enable_gqa`` flag.

    Equal K/V head counts below the query count use PyTorch's own
    ``enable_gqa`` path on the real-size K/V, under default dispatch and a
    strict Flash selection. Every other case gets an explicit expansion here:
    ``enable_gqa`` cannot express Hk != Hv, strict math would expand K/V inside
    the call, and the strict efficient, cuDNN and overrideable backends may
    reject ``enable_gqa``, which would fail rows that run with expanded K/V.
    Callers on the benchmark path build this once, so the expansion is never
    timed.
    """
    if rep_k == rep_v == 1:
        return k, v, False
    if rep_k == rep_v and strict_sdpa_selection() in (
        None,
        PyTorchSdpaBackendName.FLASH,
    ):
        return k, v, True
    return (
        k.repeat_interleave(rep_k, dim=-3),
        v.repeat_interleave(rep_v, dim=-3),
        False,
    )


def _gqa_call_is_timed_on_gpu(
    tensors: Dict[int, torch.Tensor], q: torch.Tensor, enable_gqa: bool
) -> bool:
    """Whether a GQA call is on the GPU benchmark path under default dispatch.

    Only that path (``ReplayTensors``) is checked: the one-shot reference path
    computes outputs, not timings, and math is a correct answer there.
    """
    return (
        enable_gqa
        and isinstance(tensors, ReplayTensors)
        and q.is_cuda
        and strict_sdpa_selection() is None
    )


_MATH_BACKEND_MESSAGE = (
    "PyTorch default SDPA dispatch would use the math backend for this "
    "GQA graph, which expands K/V inside the timed call"
)


def _require_fused_default_sdpa(
    tensors: Dict[int, torch.Tensor],
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: Optional[torch.Tensor],
    dropout_p: float,
    is_causal: bool,
    scale: Optional[float],
    enable_gqa: bool,
) -> None:
    """Refuse to time a GQA call that default dispatch sends to math.

    The math backend expands K/V to the query head count inside the call, the
    copy this handler keeps out of the timed region.
    """
    if not _gqa_call_is_timed_on_gpu(tensors, q, enable_gqa):
        return
    if not default_sdpa_avoids_math(
        q,
        k,
        v,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
        enable_gqa=enable_gqa,
    ):
        raise UnsupportedGraphError(_MATH_BACKEND_MESSAGE)


def _expand_kv(
    k: torch.Tensor, v: torch.Tensor, rep_k: int, rep_v: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """K and V at the query head count. Autograd recording is kept as it was."""
    return tuple(  # type: ignore[return-value]
        t.detach().repeat_interleave(rep, dim=-3).requires_grad_(t.requires_grad)
        for t, rep in ((k, rep_k), (v, rep_v))
    )


def _fused_kv_for_call(
    tensors: Dict[int, torch.Tensor],
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: Optional[torch.Tensor],
    dropout_p: float,
    is_causal: bool,
    scale: Optional[float],
    enable_gqa: bool,
    rep_k: int,
    rep_v: int,
) -> Tuple[torch.Tensor, torch.Tensor, bool]:
    """K, V and ``enable_gqa`` for one call that default dispatch keeps fused.

    Operands come back unchanged when the call is off the GPU benchmark path or
    its default dispatch avoids math. Otherwise some shapes reject
    ``enable_gqa`` in the fused kernels (top-left causal with Lq != Lkv): the
    call then takes K/V expanded here, in setup, with ``enable_gqa=False``.
    A call that math would still serve after that is refused.
    """
    if not _gqa_call_is_timed_on_gpu(tensors, q, enable_gqa):
        return k, v, enable_gqa
    if default_sdpa_avoids_math(
        q,
        k,
        v,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
        enable_gqa=True,
    ):
        return k, v, True
    k_exp, v_exp = _expand_kv(k, v, rep_k, rep_v)
    if not default_sdpa_avoids_math(
        q,
        k_exp,
        v_exp,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
        enable_gqa=False,
    ):
        raise UnsupportedGraphError(_MATH_BACKEND_MESSAGE)
    return k_exp, v_exp, False


_CallForm = Tuple[int, int, bool, bool]


def _kv_for_call(
    decided: Dict[_CallForm, bool],
    key: _CallForm,
    tensors: Dict[int, torch.Tensor],
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: Optional[torch.Tensor],
    dropout_p: float,
    is_causal: bool,
    scale: float,
    enable_gqa: bool,
    rep_k: int,
    rep_v: int,
) -> Tuple[torch.Tensor, torch.Tensor, bool]:
    """``_fused_kv_for_call``, decided once per call form and then replayed.

    Dispatch depends on shapes and the call form, not on values. ``key`` is
    ``(q_len, kv_len, attn_mask is None, is_causal)``; ``decided`` records for
    each key whether K/V are expanded because native GQA would hit math.
    """
    if key not in decided:
        k_call, v_call, gqa_call = _fused_kv_for_call(
            tensors,
            q,
            k,
            v,
            attn_mask,
            dropout_p,
            is_causal,
            scale,
            enable_gqa,
            rep_k,
            rep_v,
        )
        decided[key] = enable_gqa and not gqa_call
        return k_call, v_call, gqa_call
    if decided[key]:
        k_exp, v_exp = _expand_kv(k, v, rep_k, rep_v)
        return k_exp, v_exp, False
    return k, v, enable_gqa


def _plan_paged_sdpa(
    node: Dict[str, Any],
    tensors: Dict[int, torch.Tensor],
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    page_table_k_uid: int,
    page_table_v_uid: int,
    tensors_json: Dict[int, Dict[str, Any]],
    attn_mask: Optional[torch.Tensor],
    dropout_p: float,
    is_causal: bool,
    scale: float,
    rep_k: int,
    rep_v: int,
    window: Optional[int],
    bottom_right: bool,
    sink: Optional[torch.Tensor],
) -> List[_PagedSegment]:
    """Gather a paged KV cache into dense per-sequence K/V and masks.

    Layouts, taken from the committed paged bundles and hipDNN's
    ``SdpaFwdNode::validatePagedKv`` rather than assumed:

      * paged K/V are ``[num_blocks, num_kv_heads, page_size, head_size]``
      * the page table holds int32 block ids, ``[num_seqs, 1,
        max_blocks_per_seq, 1]`` in hipDNN and ``[num_seqs,
        max_blocks_per_seq]`` in older graphs; it is read as ``[num_seqs,
        product of the other dims]``
      * ``seq_len_kv`` is ``[num_seqs]`` LENGTHS (not offsets -- the page table
        is itself the per-sequence indirection, so paged K/V carry no ragged
        offsets)
      * Q and O are per batch, ``[num_seqs, num_query_heads, S_max,
        head_size]``, padded or ragged, when Q has one batch row per
        sequence. Sequence ``b`` reads the first ``q_len`` rows of Q row
        ``b``. Otherwise Q is ``[1, num_query_heads, total_q, head_size]``,
        with the sequences packed along ``total_q``.

    Each sequence's live KV is ``ceil(len / page_size)`` blocks gathered in page
    order and then trimmed to the exact length, so trailing slots in the last
    page never contribute. The gather emulates a layout PyTorch has no API for,
    so it is setup: the benchmark path builds it once, untimed.
    """
    if attn_mask is not None:
        raise ValueError("Paged SDPA with an explicit attn_mask is not supported")

    # Host-resolved once per replay map: a .tolist() here would sync the
    # stream inside the staged timer's gated region and deadlock.
    q_lengths, kv_lengths = sdpa_lengths(
        node, tensors_json, lambda uid: _host_ints(tensors, uid, node)
    )

    if k.ndim != 4 or v.ndim != 4:
        raise ValueError("Paged SDPA expects rank-4 paged K/V containers")
    page_size = int(k.shape[-2])
    num_seqs = len(kv_lengths)
    page_table_k = _tensor(tensors, page_table_k_uid, node)
    page_table_v = _tensor(tensors, page_table_v_uid, node)
    if int(page_table_k.shape[0]) != num_seqs:
        raise ValueError(
            f"Paged SDPA page table has {int(page_table_k.shape[0])} rows for "
            f"{num_seqs} sequences"
        )
    page_table_k = page_table_k.reshape(num_seqs, -1)
    page_table_v = page_table_v.reshape(int(page_table_v.shape[0]), -1)
    per_batch = q.ndim == 4 and int(q.shape[0]) == num_seqs
    if q.ndim == 4 and not per_batch and int(q.shape[0]) != 1:
        raise ValueError(
            f"Paged SDPA Q has {int(q.shape[0])} batches for {num_seqs} sequences"
        )
    segments: List[_PagedSegment] = []
    decided: Dict[_CallForm, bool] = {}
    q_start = 0
    for seq, (q_len, kv_len) in enumerate(zip(q_lengths, kv_lengths)):
        if kv_len <= 0:
            raise ValueError(f"Paged SDPA sequence {seq} has non-positive KV length")
        blocks_needed = (kv_len + page_size - 1) // page_size
        if blocks_needed > int(page_table_k.shape[1]):
            raise ValueError(
                f"Paged SDPA sequence {seq} needs {blocks_needed} blocks but the "
                f"page table holds {int(page_table_k.shape[1])}"
            )
        ids_k = page_table_k[seq, :blocks_needed].to(torch.long)
        ids_v = page_table_v[seq, :blocks_needed].to(torch.long)

        # [blocks, H, page, D] -> [H, blocks*page, D] -> trimmed to kv_len.
        k_seq = (
            k[ids_k]
            .permute(1, 0, 2, 3)
            .reshape(int(k.shape[1]), blocks_needed * page_size, int(k.shape[-1]))[
                :, :kv_len, :
            ]
        )
        v_seq = (
            v[ids_v]
            .permute(1, 0, 2, 3)
            .reshape(int(v.shape[1]), blocks_needed * page_size, int(v.shape[-1]))[
                :, :kv_len, :
            ]
        )

        if per_batch:
            q_seq = q[seq : seq + 1, :, :q_len, :]
        else:
            q_seq = q[..., q_start : q_start + q_len, :]
        if q_seq.ndim == 4:
            k_seq = k_seq.unsqueeze(0)
            v_seq = v_seq.unsqueeze(0)
        seq_q_len = int(q_seq.shape[-2])
        seq_mask, seq_causal = _attention_bounds(
            seq_q_len, kv_len, window, bottom_right, is_causal, q
        )
        if sink is not None:
            # Sinks run the manual fp32 attention: its operands replace K/V.
            k_t, seq_mask = _score_operands(
                k_seq, seq_mask, seq_causal, seq_q_len, rep_k
            )
            segments.append(
                _PagedSegment(
                    q_start,
                    q_len,
                    k_t,
                    _fp32_at_query_heads(v_seq, rep_v),
                    seq_mask,
                    seq_causal,
                    False,
                    seq if per_batch else None,
                )
            )
            q_start += q_len
            continue

        k_seq, v_seq, enable_gqa = _sdpa_kv_operands(
            k_seq.contiguous(), v_seq.contiguous(), rep_k, rep_v
        )
        k_call, v_call, gqa_call = _kv_for_call(
            decided,
            (seq_q_len, kv_len, seq_mask is None, seq_causal),
            tensors,
            q_seq,
            k_seq,
            v_seq,
            seq_mask,
            dropout_p,
            seq_causal,
            scale,
            enable_gqa,
            rep_k,
            rep_v,
        )
        segments.append(
            _PagedSegment(
                q_start,
                q_len,
                k_call,
                v_call,
                seq_mask,
                seq_causal,
                gqa_call,
                seq if per_batch else None,
            )
        )
        q_start += q_len
    return segments


def _fp32_at_query_heads(t: torch.Tensor, rep: int) -> torch.Tensor:
    t = t.to(dtype=torch.float32)
    return t.repeat_interleave(rep, dim=-3) if rep > 1 else t


#: Bytes of one float32 score block in the explicit (non-fused) SDPA paths on
#: the host. A block takes every query row that fits, so a small graph runs as
#: one block, and a large one keeps a few score-sized temporaries instead of
#: ``q_len * kv_len`` ones. On an MI210 host, 2 GiB blocks ran a B1 H32 S16384
#: causal backward in 28 s with 5 GiB; 64-row blocks took 113 s with 2 GiB, and
#: one full-size block took 32 s with 162 GiB.
_SCORE_BLOCK_BYTES = 2 << 30


def _query_blocks(q: torch.Tensor, kv_len: int) -> Sequence[Tuple[int, int]]:
    """``(start, end)`` query row ranges; one empty range when ``q_len`` is 0.

    Only host tensors are split. On a GPU the timed PyTorch row and the GPU
    reference keep one block: smaller launches would slow the timed row, and
    a GPU out-of-memory error already falls back to the CPU reference.
    """
    q_len = int(q.shape[-2])
    step = max(q_len, 1)
    if q.device.type == "cpu":
        row_bytes = q.shape[:-2].numel() * kv_len * 4  # one query row of scores
        step = min(step, max(1, _SCORE_BLOCK_BYTES // max(row_bytes, 1)))
    return [
        (start, min(start + step, q_len)) for start in range(0, max(q_len, 1), step)
    ]


def _rows(
    tensor: Optional[torch.Tensor], start: int, end: int
) -> Optional[torch.Tensor]:
    """Query rows ``start:end`` of a per-row tensor; a broadcast row stays whole."""
    if tensor is None or tensor.ndim < 2 or tensor.shape[-2] == 1:
        return tensor
    return tensor[..., start:end, :]


def _block_scores(
    q_block: torch.Tensor,
    k_f: torch.Tensor,
    mask_block: Optional[torch.Tensor],
    is_causal: bool,
    start: int,
    scale: float,
) -> torch.Tensor:
    """float32 scores of query rows ``start:start+len`` against every key.

    The causal mask uses absolute query indices, so key ``j`` is visible to
    query ``i`` only when ``j <= i`` (top-left alignment).
    """
    scores = torch.matmul(q_block.to(dtype=torch.float32), k_f.transpose(-2, -1))
    scores *= scale
    if mask_block is not None:
        scores += mask_block.to(dtype=torch.float32)
    if is_causal:
        rows = torch.arange(
            start, start + int(scores.shape[-2]), device=scores.device
        ).unsqueeze(-1)
        cols = torch.arange(int(scores.shape[-1]), device=scores.device).unsqueeze(0)
        scores.masked_fill_(cols > rows, float("-inf"))
    return scores


def _score_operands(
    k: torch.Tensor,
    attn_mask: Optional[torch.Tensor],
    is_causal: bool,
    q_len: int,
    rep_k: int,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """fp32 K^T at the query head count and the fp32 additive mask.

    For the score matmul of the manual paths (stats under math, sinks). These
    depend only on the inputs, so the benchmark path builds them once. A
    causal mask becomes an additive ``-inf`` mask; ``attn_mask`` and
    ``is_causal`` never come together (``_plan_sdpa_common``).
    """
    mask = None if attn_mask is None else attn_mask.to(dtype=torch.float32)
    if is_causal:
        mask = _diagonal_band_mask(
            q_len,
            int(k.shape[-2]),
            None,
            bottom_right=False,
            device=k.device,
            dtype=torch.float32,
        )
    return _fp32_at_query_heads(k, rep_k).transpose(-2, -1), mask


def _scores(
    q_float: torch.Tensor,
    k_float_t: torch.Tensor,
    mask: Optional[torch.Tensor],
    scale: float,
) -> torch.Tensor:
    scores = torch.matmul(q_float, k_float_t) * scale
    return scores if mask is None else scores + mask


def _sdpa_stats(
    q_float: torch.Tensor,
    k_float_t: torch.Tensor,
    mask: Optional[torch.Tensor],
    scale: float,
) -> torch.Tensor:
    """Log-sum-exp of the masked, scaled scores (``_score_operands``)."""
    blocks = _query_blocks(q_float, int(k_float_t.shape[-1]))
    if len(blocks) == 1:
        return torch.logsumexp(
            _scores(q_float, k_float_t, mask, scale), -1, keepdim=True
        )
    return torch.cat(
        [
            torch.logsumexp(
                _scores(
                    q_float[..., start:end, :],
                    k_float_t,
                    _rows(mask, start, end),
                    scale,
                ),
                -1,
                keepdim=True,
            )
            for start, end in blocks
        ],
        dim=-2,
    )


def _note_stats_fallback(tensors: Dict[int, torch.Tensor]) -> None:
    """Note on the row that the stats fallback runs inside the timed region.

    Reached when the kernel saved no log-sum-exp. Math saves none, and a math
    user computes it. Under any other selection the fallback adds a score
    matmul and a logsumexp to the timed region, so say so on the row.
    """
    if (
        isinstance(tensors, ReplayTensors)
        and strict_sdpa_selection() is not PyTorchSdpaBackendName.MATH
    ):
        tensors.note(_STATS_FALLBACK_WARNING)


def _sink_attention(
    q_float: torch.Tensor,
    k_float_t: torch.Tensor,
    v_float: torch.Tensor,
    mask: Optional[torch.Tensor],
    sink_logits: torch.Tensor,
    scale: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """fp32 attention with sink logits in the softmax denominator.

    PyTorch's SDPA has no sink input, so this is computed by hand. Returns
    the fp32 output and the log-sum-exp including the sinks (the stats).
    Host tensors run one query block at a time (``_query_blocks``), and the
    probabilities overwrite the block's scores.
    """
    blocks = _query_blocks(q_float, int(k_float_t.shape[-1]))
    if len(blocks) == 1:
        scores = _scores(q_float, k_float_t, mask, scale)
        log_denom = torch.logaddexp(
            torch.logsumexp(scores, dim=-1, keepdim=True), sink_logits
        )
        return torch.matmul(torch.exp(scores - log_denom), v_float), log_denom
    outputs, stats = [], []
    for start, end in blocks:
        scores = _scores(
            q_float[..., start:end, :], k_float_t, _rows(mask, start, end), scale
        )
        log_denom = torch.logaddexp(
            torch.logsumexp(scores, dim=-1, keepdim=True), sink_logits
        )
        outputs.append(torch.matmul(scores.sub_(log_denom).exp_(), v_float))
        stats.append(log_denom)
    return torch.cat(outputs, dim=-2), torch.cat(stats, dim=-2)


def _batch_attn_mask(
    mask: Optional[torch.Tensor], batch: int, start: int, end: int, kv_len: int
) -> Optional[torch.Tensor]:
    """Batch ``batch``, query rows ``start:end`` and keys ``:kv_len`` of an
    explicit attention mask. Size-1 broadcast dims stay whole."""
    if mask is None:
        return None
    if mask.ndim == 4 and mask.shape[0] > 1:
        mask = mask[batch : batch + 1]
    mask = _rows(mask, start, end)
    if mask.ndim >= 2 and mask.shape[-1] > 1:
        mask = mask[..., :kv_len]
    return mask


def _sdpa_varlen_lengths(
    node: Dict[str, Any],
    tensors_json: Dict[int, Dict[str, Any]],
    tensors: Dict[int, torch.Tensor],
    q: torch.Tensor,
    k: torch.Tensor,
) -> Tuple[List[int], List[int]]:
    """Per-batch ``(q_lengths, kv_lengths)`` of a varlen node, read from the host.

    The tensors the handlers see are logical: batch ``b`` is at index ``b``.
    Offsets are only a length source.
    """
    lengths = sdpa_lengths(
        node, tensors_json, lambda uid: _host_ints(tensors, uid, node)
    )
    if lengths is None:
        raise ValueError("SDPA varlen graph has no sequence length source")
    q_lens, kv_lens = lengths
    batches = int(q.shape[0])
    if len(q_lens) != batches or len(kv_lens) != batches:
        raise ValueError(
            f"SDPA lengths cover {len(q_lens)} q and {len(kv_lens)} kv batches "
            f"for {batches} batches"
        )
    for b, (lq, lkv) in enumerate(zip(q_lens, kv_lens)):
        if not 0 <= lq <= int(q.shape[-2]) or not 0 <= lkv <= int(k.shape[-2]):
            raise ValueError(
                f"SDPA batch {b} lengths (q={lq}, kv={lkv}) exceed the tensor "
                f"sequence dims (q={int(q.shape[-2])}, kv={int(k.shape[-2])})"
            )
    return list(q_lens), list(kv_lens)


def _plan_varlen_sdpa(
    node: Dict[str, Any],
    tensors_json: Dict[int, Dict[str, Any]],
    tensors: Dict[int, torch.Tensor],
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    plan: _SdpaPlan,
    attn_mask: Optional[torch.Tensor],
    scale: float,
    rep_k: int,
    rep_v: int,
    sink: Optional[torch.Tensor],
    want_stats: bool,
) -> _VarlenSdpa:
    """Slice a non-paged varlen forward into per-batch SDPA calls.

    Each batch runs on its live rows ``start:stop`` and keys ``:lkv`` only,
    with bounds built for those live lengths (bottom-right alignment follows
    them). Rows at or past ``lq`` keep O = 0 and stats = -inf, also with
    sinks. Dead rows below ``lq`` keep O = 0 and stats = -inf, or the sink
    logit with sinks.
    """
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("Varlen SDPA expects rank-4 q/k/v tensors [B, H, S, D]")
    q_lens, kv_lens = _sdpa_varlen_lengths(node, tensors_json, tensors, q, k)

    q_op, k_op, v_op, enable_gqa = q, k, v, False
    if sink is None:
        k_op, v_op, enable_gqa = _sdpa_kv_operands(k, v, rep_k, rep_v)
        if want_stats:
            # Recording autograd makes the fused kernel emit the log-sum-exp.
            q_op, k_op, v_op = (t.detach().requires_grad_() for t in (q, k_op, v_op))

    o = torch.zeros((*q.shape[:-1], int(v.shape[-1])), dtype=q.dtype, device=q.device)
    stats = None
    if want_stats:
        stats = torch.full(
            (*q.shape[:-1], 1), float("-inf"), dtype=torch.float32, device=q.device
        )
        if sink is not None:
            for b, lq in enumerate(q_lens):
                stats[b : b + 1, :, :lq] = sink

    batches: List[_VarlenBatch] = []
    decided: Dict[_CallForm, bool] = {}
    for b, (lq, lkv) in enumerate(zip(q_lens, kv_lens)):
        if lkv == 0:
            continue
        start, stop = live_rows(lq, lkv, plan.is_causal, plan.window, plan.bottom_right)
        rows = stop - start
        if rows <= 0:
            continue
        if start and plan.window is None and plan.bottom_right:
            # The trimmed block is top-left causal: keep the call native.
            band, causal = _attention_bounds(rows, lkv, None, True, plan.is_causal, q)
            mask = band
        else:
            band, causal = _attention_bounds(
                lq, lkv, plan.window, plan.bottom_right, plan.is_causal, q
            )
            mask = None
            if band is not None:
                mask = band[start:stop] if (start or stop < lq) else band
        if mask is None:
            mask = _batch_attn_mask(attn_mask, b, start, stop, lkv)
        q_rows = slice(start, stop)
        o_rows = o[b : b + 1, :, q_rows]
        stats_rows = None if stats is None else stats[b : b + 1, :, q_rows]
        if sink is not None:
            k_t, mask = _score_operands(
                k[b : b + 1, :, :lkv], mask, causal, rows, rep_k
            )
            batches.append(
                _VarlenBatch(
                    b,
                    rows,
                    q[b : b + 1, :, q_rows].to(dtype=torch.float32),
                    k_t,
                    _fp32_at_query_heads(v[b : b + 1, :, :lkv], rep_v),
                    None,
                    mask,
                    causal,
                    o_rows,
                    stats_rows,
                    False,
                )
            )
            continue
        q_b = q_op[b : b + 1, :, q_rows]
        k_b, v_b, gqa_b = _kv_for_call(
            decided,
            (rows, lkv, mask is None, causal),
            tensors,
            q_b,
            k_op[b : b + 1, :, :lkv],
            v_op[b : b + 1, :, :lkv],
            mask,
            plan.dropout_p,
            causal,
            scale,
            enable_gqa,
            rep_k,
            rep_v,
        )
        batches.append(
            _VarlenBatch(
                b,
                rows,
                q_b,
                k_b,
                v_b,
                k[b : b + 1, :, :lkv],
                mask,
                causal,
                o_rows,
                stats_rows,
                gqa_b,
            )
        )
    return _VarlenSdpa(batches, scale, rep_k, sink, o, stats)


def _is_computed(graph_json: Dict[str, Any], uid: int) -> bool:
    """Whether some node of the graph writes tensor ``uid``."""
    for node in graph_json.get("nodes", []):
        outputs = node.get("outputs")
        if not isinstance(outputs, dict):
            continue
        for value in outputs.values():
            if value == uid or (isinstance(value, list) and uid in value):
                return True
    return False


# -----------------------------------------------------------------------------
# Operation Handlers
# -----------------------------------------------------------------------------


@register_handler("SdpaAttributes")
def compile_sdpa(
    node: Dict[str, Any],
    graph_json: Dict[str, Any],
) -> CompiledOp:
    """Plan scaled dot-product attention forward.

    Serves paged (KV-cache) graphs as well as dense ones. PyTorch has no paged
    SDPA API -- ``F.scaled_dot_product_attention`` takes no page table -- so a
    paged graph is gathered through its page table into dense per-sequence K/V
    and then run one sequence at a time. ``--validate pytorch`` walks these very
    handlers with CPU tensors, so there is no reference-side shortcut. On the
    benchmark path (``ReplayTensors``) the gather, K/V head expansion and masks
    are built once during priming, so the timed region holds only the op work.

    A non-paged varlen or ragged graph runs one SDPA call per batch on the
    batch's live rows and keys (``_plan_varlen_sdpa``). A paged Q/O with one
    batch row per sequence, padded or ragged (RFC 0014 limitation 9: ragged
    Q/O over a non-ragged paged K/V cache), runs the same per-sequence loop
    as a packed paged Q, addressed by batch instead of a running offset, and
    its output is scattered by batch into zeros instead of concatenated
    (``_plan_paged_sdpa``). Paged backward and paged stats stay unsupported
    either way.

    The ``stats`` output (log-sum-exp) comes from the fused kernel, as in a
    training forward: Q/K/V go in with ``requires_grad`` and the kernel's saved
    log-sum-exp is read back (``fused_logsumexp``). Only a backend that saves
    none (math) gets a separate score matmul.

    PyTorch's SDPA has no attention-sink input, so a graph with sinks runs a
    manual fp32 attention (``_sink_attention``); its fp32 operands are built
    once too, and its softmax denominator is the stats output.
    """
    _sdpa_unsupported_if_present(
        node,
        [
            "max_tensor_uid",
            "sum_exp_tensor_uid",
            "rng_dump_tensor_uid",
            "amax_s_tensor_uid",
            "amax_o_tensor_uid",
        ],
    )
    q_uid = _required_input_uid(node, "q_tensor_uid")
    k_uid = _required_input_uid(node, "k_tensor_uid")
    v_uid = _required_input_uid(node, "v_tensor_uid")
    o_uid = _required_output_uid(node, "o_tensor_uid")
    plan = _plan_sdpa_common(node, allow_paged=True, allow_sink=True)
    dropout_p, is_causal = plan.dropout_p, plan.is_causal
    operands = sdpa_operands(node)
    stats_uid = operands.stats
    tensors_json = tensors_by_uid(graph_json)

    is_paged = operands.is_paged
    if is_paged:
        if (
            operands.page_table_k_tensor_uid is None
            or operands.page_table_v_tensor_uid is None
        ):
            raise ValueError("Paged SDPA needs both K and V page tables")
        if operands.seq_len_kv_tensor_uid is None:
            raise ValueError("Paged SDPA needs seq_len_kv to bound each sequence")
        if stats_uid is not None:
            raise ValueError("Paged SDPA stats output is not supported")
        if any(
            ragged_fields(tensors_json.get(uid, {}))[0] is not None
            for uid in operands.kv_side
        ):
            raise ValueError("Paged SDPA K/V cache must stay non-ragged")
    is_varlen = not is_paged and has_varlen_source(node, tensors_json)

    # Setup is cached on the assumption that its inputs repeat unchanged every
    # replay. A tensor written by an earlier node breaks that assumption,
    # whether or not the graph marks it virtual.
    computed_inputs = [
        uid
        for uid in (
            q_uid,
            k_uid,
            v_uid,
            plan.mask_uid,
            plan.scale_uid,
            plan.sink_uid,
            operands.page_table_k_tensor_uid,
            operands.page_table_v_tensor_uid,
            operands.seq_len_q_tensor_uid,
            operands.seq_len_kv_tensor_uid,
            operands.cu_seq_len_q_tensor_uid,
            operands.cu_seq_len_kv_tensor_uid,
        )
        if uid is not None and _is_computed(graph_json, uid)
    ]

    def setup(
        tensors: Dict[int, torch.Tensor], part: str, build: Callable[[], Any]
    ) -> Any:
        if computed_inputs and isinstance(tensors, ReplayTensors):
            raise UnsupportedGraphError(
                f"SDPA inputs {computed_inputs} are computed inside the graph, so "
                "the PyTorch baseline cannot build their setup outside the timed "
                "region"
            )
        return _replay_constant(tensors, ("sdpa", o_uid, part), build)

    def resolve(tensors, q, k, v):
        attn_mask, scale, rep_k, rep_v = _sdpa_resolve(
            node,
            tensors,
            q,
            k,
            v,
            plan.mask_uid,
            plan.scale_uid,
            plan.attn_scale_value,
        )
        sink = None
        if plan.sink_uid is not None:
            sink = _sink_logits(_tensor(tensors, plan.sink_uid, node), q)
        return attn_mask, scale, rep_k, rep_v, sink

    def bounds(q, k, attn_mask):
        band, causal = _attention_bounds(
            int(q.shape[-2]),
            int(k.shape[-2]),
            plan.window,
            plan.bottom_right,
            is_causal,
            q,
        )
        return (attn_mask if band is None else band), causal

    def prepare_sink(
        tensors: Dict[int, torch.Tensor],
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> _SinkSdpa:
        attn_mask, scale, rep_k, rep_v, sink = resolve(tensors, q, k, v)
        attn_mask, causal = bounds(q, k, attn_mask)
        k_t, mask = _score_operands(k, attn_mask, causal, int(q.shape[-2]), rep_k)
        return _SinkSdpa(
            scale,
            q.to(dtype=torch.float32),
            k_t,
            _fp32_at_query_heads(v, rep_v),
            mask,
            sink,
        )

    def prepare_dense(
        tensors: Dict[int, torch.Tensor],
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> _DenseSdpa:
        attn_mask, scale, rep_k, rep_v, _ = resolve(tensors, q, k, v)
        attn_mask, causal = bounds(q, k, attn_mask)
        k_op, v_op, enable_gqa = _sdpa_kv_operands(k, v, rep_k, rep_v)
        q_op = q
        if stats_uid is not None:
            # A forward that autograd records makes the fused kernel emit the
            # log-sum-exp, the way a training step gets it.
            q_op, k_op, v_op = (t.detach().requires_grad_() for t in (q_op, k_op, v_op))
        _require_fused_default_sdpa(
            tensors,
            q_op,
            k_op,
            v_op,
            attn_mask,
            dropout_p,
            causal,
            scale,
            enable_gqa,
        )
        return _DenseSdpa(attn_mask, causal, scale, rep_k, q_op, k_op, v_op, enable_gqa)

    def prepare_paged(
        tensors: Dict[int, torch.Tensor],
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> _PagedSdpa:
        attn_mask, scale, rep_k, rep_v, sink = resolve(tensors, q, k, v)
        segments = _plan_paged_sdpa(
            node,
            tensors,
            q,
            k,
            v,
            operands.page_table_k_tensor_uid,
            operands.page_table_v_tensor_uid,
            tensors_json,
            attn_mask,
            dropout_p,
            is_causal,
            scale,
            rep_k,
            rep_v,
            plan.window,
            plan.bottom_right,
            sink,
        )
        q_float = None if sink is None else q.to(dtype=torch.float32)
        per_batch = any(s.batch is not None for s in segments)
        return _PagedSdpa(segments, scale, q_float, sink, per_batch)

    def prepare_varlen(
        tensors: Dict[int, torch.Tensor],
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> _VarlenSdpa:
        attn_mask, scale, rep_k, rep_v, sink = resolve(tensors, q, k, v)
        return _plan_varlen_sdpa(
            node,
            tensors_json,
            tensors,
            q,
            k,
            v,
            plan,
            attn_mask,
            scale,
            rep_k,
            rep_v,
            sink,
            stats_uid is not None,
        )

    def varlen_stats_operands(
        tensors: Dict[int, torch.Tensor], s: _VarlenSdpa, batch: _VarlenBatch
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        _note_stats_fallback(tensors)
        return (
            batch.q.detach().to(dtype=torch.float32),
            *_score_operands(
                batch.k_raw.detach(), batch.mask, batch.is_causal, batch.rows, s.rep_k
            ),
        )

    def run_varlen(
        tensors: Dict[int, torch.Tensor],
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> None:
        s = setup(tensors, "varlen", lambda: prepare_varlen(tensors, q, k, v))
        for batch in s.batches:
            if s.sink is not None:
                o, stats = _sink_attention(
                    batch.q, batch.k, batch.v, batch.mask, s.sink, s.scale
                )
            else:
                with torch.enable_grad():
                    o = execute_selected_sdpa(
                        batch.q,
                        batch.k,
                        batch.v,
                        attn_mask=batch.mask,
                        dropout_p=dropout_p,
                        is_causal=batch.is_causal,
                        scale=s.scale,
                        enable_gqa=batch.enable_gqa,
                    )
                stats = None
                if stats_uid is not None:
                    stats = fused_logsumexp(o, batch.rows)
                    if stats is None:
                        stats = _sdpa_stats(
                            *setup(
                                tensors,
                                ("varlen_stats", batch.index),
                                lambda: varlen_stats_operands(tensors, s, batch),
                            ),
                            s.scale,
                        )
            with torch.no_grad():
                batch.o_rows.copy_(o)
                if stats_uid is not None:
                    batch.stats_rows.copy_(stats)
        _store_tensor(tensors, o_uid, s.o)
        if stats_uid is not None:
            _store_tensor(tensors, stats_uid, s.stats)

    def stats_operands(
        tensors: Dict[int, torch.Tensor],
        q: torch.Tensor,
        k: torch.Tensor,
        d: _DenseSdpa,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        _note_stats_fallback(tensors)
        return (
            q.to(dtype=torch.float32),
            *_score_operands(k, d.attn_mask, d.is_causal, int(q.shape[-2]), d.rep_k),
        )

    # Everything derived from the constant inputs is built once per replay
    # map, during priming. Timed iterations run only the SDPA call(s) and, for
    # paged graphs, the per-sequence loop and re-pack.
    def run(tensors: Dict[int, torch.Tensor]) -> None:
        q = _tensor(tensors, q_uid, node)
        k = _tensor(tensors, k_uid, node)
        v = _tensor(tensors, v_uid, node)
        if is_varlen:
            run_varlen(tensors, q, k, v)
            return

        if is_paged:
            paged = setup(tensors, "paged", lambda: prepare_paged(tensors, q, k, v))

            def q_slice(t: torch.Tensor, s: _PagedSegment) -> torch.Tensor:
                if s.batch is not None:
                    return t[s.batch : s.batch + 1, :, : s.q_len, :]
                return t[..., s.q_start : s.q_start + s.q_len, :]

            if paged.sink is not None:
                outputs = [
                    _sink_attention(
                        q_slice(paged.q_float, s),
                        s.k,
                        s.v,
                        s.mask,
                        paged.sink,
                        paged.scale,
                    )[0]
                    for s in paged.segments
                ]
            else:
                outputs = [
                    execute_selected_sdpa(
                        q_slice(q, s),
                        s.k,
                        s.v,
                        attn_mask=s.mask,
                        dropout_p=dropout_p,
                        is_causal=s.is_causal,
                        scale=paged.scale,
                        enable_gqa=s.enable_gqa,
                    )
                    for s in paged.segments
                ]
            if paged.per_batch:
                # A per-batch Q/O, padded or ragged: scatter each sequence's
                # output into its batch row and leave the rows past its
                # q_len at 0, as hipDNN defines them.
                o = torch.zeros(
                    (*q.shape[:-1], int(v.shape[-1])), dtype=q.dtype, device=q.device
                )
                for s, out in zip(paged.segments, outputs):
                    o[s.batch : s.batch + 1, :, : s.q_len, :] = out.to(dtype=o.dtype)
            else:
                # Re-packed along the query axis in the order the sequences
                # were laid out.
                o = torch.cat(outputs, dim=-2)
                if paged.sink is not None:
                    o = o.to(dtype=q.dtype)
            _store_tensor(tensors, o_uid, o)
            return

        if plan.sink_uid is not None:
            s = setup(tensors, "sink", lambda: prepare_sink(tensors, q, k, v))
            o, stats = _sink_attention(
                s.q_float, s.k_t, s.v_float, s.mask, s.sink, s.scale
            )
            _store_tensor(tensors, o_uid, o.to(dtype=q.dtype))
            if stats_uid is not None:
                _store_tensor(tensors, stats_uid, stats)
            return

        d = setup(tensors, "dense", lambda: prepare_dense(tensors, q, k, v))
        with torch.enable_grad():
            o = execute_selected_sdpa(
                d.q,
                d.k,
                d.v,
                attn_mask=d.attn_mask,
                dropout_p=dropout_p,
                is_causal=d.is_causal,
                scale=d.scale,
                enable_gqa=d.enable_gqa,
            )
        if stats_uid is None:
            _store_tensor(tensors, o_uid, o)
            return

        stats = fused_logsumexp(o, int(q.shape[-2]))
        _store_tensor(tensors, o_uid, o.detach())
        if stats is None:
            stats = _sdpa_stats(
                *setup(tensors, "stats", lambda: stats_operands(tensors, q, k, d)),
                d.scale,
            )
        _store_tensor(tensors, stats_uid, stats)

    return run


def _sdpa_gradients(
    q_f: torch.Tensor,
    k_f: torch.Tensor,
    v_f: torch.Tensor,
    o_f: torch.Tensor,
    do_f: torch.Tensor,
    stats_f: torch.Tensor,
    attn_mask: Optional[torch.Tensor],
    is_causal: bool,
    scale: float,
    rep_k: int,
    rep_v: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """fp32 ``(dQ, dK, dV)`` of one dense backward, dK and dV at the K and V heads.

    Consumes the saved ``stats`` directly (see ``compile_sdpa_backward``).
    """
    head_dim = int(q_f.shape[-1])
    k_heads = int(k_f.shape[1])
    v_heads = int(v_f.shape[1])
    if rep_k > 1:
        k_f = k_f.repeat_interleave(rep_k, dim=1)
    if rep_v > 1:
        v_f = v_f.repeat_interleave(rep_v, dim=1)

    # One query block at a time: a block's probs and d_scores are at most
    # _SCORE_BLOCK_BYTES each, so memory does not grow with q_len * kv_len.
    # dQ rows are independent; dK and dV sum over every query row.
    dq = torch.empty_like(q_f)
    dk_full = torch.zeros_like(k_f)
    dv_full = torch.zeros_like(v_f)
    for start, end in _query_blocks(q_f, int(k_f.shape[-2])):
        q_block = q_f[..., start:end, :]
        do_block = do_f[..., start:end, :]
        scores = _block_scores(
            q_block, k_f, _rows(attn_mask, start, end), is_causal, start, scale
        )
        probs = scores.sub_(_rows(stats_f, start, end)).exp_()
        row_dot = (do_block * o_f[..., start:end, :]).sum(dim=-1, keepdim=True)
        # d_scores = probs * (do @ v^T - row_dot) * scale
        d_scores = torch.matmul(do_block, v_f.transpose(-2, -1))
        d_scores.sub_(row_dot).mul_(probs).mul_(scale)
        dq[..., start:end, :] = torch.matmul(d_scores, k_f)
        dk_full += torch.matmul(d_scores.transpose(-2, -1), q_block)
        dv_full += torch.matmul(probs.transpose(-2, -1), do_block)
        del scores, probs, d_scores

    batch, seq_kv = dk_full.shape[0], dk_full.shape[2]
    if rep_k > 1:
        dk_f = dk_full.view(batch, k_heads, rep_k, seq_kv, head_dim).sum(dim=2)
    else:
        dk_f = dk_full
    if rep_v > 1:
        dv_f = dv_full.view(batch, v_heads, rep_v, seq_kv, int(v_f.shape[-1])).sum(
            dim=2
        )
    else:
        dv_f = dv_full
    return dq, dk_f, dv_f


@register_handler("SdpaBackwardAttributes")
def compile_sdpa_backward(
    node: Dict[str, Any],
    graph_json: Dict[str, Any],
) -> CompiledOp:
    """Plan scaled dot-product attention backward.

    Mirrors hipDNN's CPU reference (CpuFpReferenceSdpa::backward): the saved
    softmax statistics ``stats`` (forward log-sum-exp) are consumed directly to
    recompute probabilities as ``P = exp(scores - stats)`` without
    renormalization.  PyTorch's built-in SDPA autograd cannot consume an
    external ``stats`` tensor and always renormalizes its own softmax, so it
    would diverge from hipDNN whenever ``stats`` is not the exact, consistent
    forward LSE.  This handler therefore implements the gradient manually.
    """
    _sdpa_unsupported_if_present(node, ["dropout_scale_inv_tensor_uid"])
    if _optional_uid(node, "dbias_tensor_uid") is not None:
        raise ValueError(
            "SDPA backward dBias gradient is not supported by the PyTorch reference"
        )

    q_uid = _required_input_uid(node, "q_tensor_uid")
    k_uid = _required_input_uid(node, "k_tensor_uid")
    v_uid = _required_input_uid(node, "v_tensor_uid")
    o_uid = _required_input_uid(node, "o_tensor_uid")
    do_uid = _required_input_uid(node, "do_tensor_uid")
    stats_uid = _required_input_uid(node, "stats_tensor_uid")
    dq_uid = _required_output_uid(node, "dq_tensor_uid")
    dk_uid = _required_output_uid(node, "dk_tensor_uid")
    dv_uid = _required_output_uid(node, "dv_tensor_uid")
    plan = _plan_sdpa_common(node)
    tensors_json = tensors_by_uid(graph_json)
    is_varlen = not sdpa_operands(node).is_paged and has_varlen_source(
        node, tensors_json
    )
    is_causal = plan.is_causal
    if plan.window is not None or plan.bottom_right:
        raise ValueError(
            "SDPA bottom-right/sliding-window bounds are not supported by the "
            "PyTorch reference backward"
        )

    def run(tensors: Dict[int, torch.Tensor]) -> None:
        q = _tensor(tensors, q_uid, node)
        k = _tensor(tensors, k_uid, node)
        v = _tensor(tensors, v_uid, node)
        o = _tensor(tensors, o_uid, node)
        do = _tensor(tensors, do_uid, node)
        stats = _tensor(tensors, stats_uid, node)
        attn_mask, scale, rep_k, rep_v = _sdpa_resolve(
            node,
            tensors,
            q,
            k,
            v,
            plan.mask_uid,
            plan.scale_uid,
            plan.attn_scale_value,
        )

        if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
            raise ValueError("SDPA backward expects rank-4 q/k/v tensors [B, H, S, D]")

        q_f = q.to(dtype=torch.float32)
        k_f = k.to(dtype=torch.float32)
        v_f = v.to(dtype=torch.float32)
        o_f = o.to(dtype=torch.float32)
        do_f = do.to(dtype=torch.float32)
        stats_f = _require_fp32_stat(stats, "SDPA stats (log-sum-exp)")
        scale_value = float(scale)

        if not is_varlen:
            dq, dk_f, dv_f = _sdpa_gradients(
                q_f,
                k_f,
                v_f,
                o_f,
                do_f,
                stats_f,
                attn_mask,
                is_causal,
                scale_value,
                rep_k,
                rep_v,
            )
        else:
            # Per batch on the valid slices. Gradients past a length stay 0.
            # Backward refuses bottom-right and window bounds, so no live row
            # is dead.
            q_lens, kv_lens = _sdpa_varlen_lengths(node, tensors_json, tensors, q, k)
            dq = torch.zeros_like(q_f)
            dk_f = torch.zeros_like(k_f)
            dv_f = torch.zeros_like(v_f)
            for b, (lq, lkv) in enumerate(zip(q_lens, kv_lens)):
                if lq == 0 or lkv == 0:
                    continue
                one = slice(b, b + 1)
                dq_b, dk_b, dv_b = _sdpa_gradients(
                    q_f[one, :, :lq],
                    k_f[one, :, :lkv],
                    v_f[one, :, :lkv],
                    o_f[one, :, :lq],
                    do_f[one, :, :lq],
                    stats_f[one, :, :lq],
                    _batch_attn_mask(attn_mask, b, 0, lq, lkv),
                    is_causal,
                    scale_value,
                    rep_k,
                    rep_v,
                )
                dq[one, :, :lq] = dq_b
                dk_f[one, :, :lkv] = dk_b
                dv_f[one, :, :lkv] = dv_b

        _store_tensor(tensors, dq_uid, dq.to(dtype=q.dtype))
        _store_tensor(tensors, dk_uid, dk_f.to(dtype=k.dtype))
        _store_tensor(tensors, dv_uid, dv_f.to(dtype=v.dtype))

    return run

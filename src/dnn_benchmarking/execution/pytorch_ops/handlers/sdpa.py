# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Scaled-dot-product-attention forward/backward reference handlers."""

from typing import Any, Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple

import torch

from ....common.exceptions import UnsupportedGraphError
from ....config.benchmark_config import PyTorchSdpaBackendName
from ....graph.loader import tensor_data_type
from ...sdpa_dropout import SdpaDropout, read_dropout
from .._common import *  # noqa: F401,F403
from .._registry import CompiledOp, register_handler
from .._sdpa_backend import (
    current_philox_dumps,
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
    are the fp32 K^T and V of the manual attention instead.
    """

    q_start: int
    q_len: int
    k: torch.Tensor
    v: torch.Tensor
    mask: Optional[torch.Tensor]
    is_causal: bool


class _PagedSdpa(NamedTuple):
    """Setup of a paged forward, built once."""

    segments: List[_PagedSegment]
    scale: float
    enable_gqa: bool
    q_float: Optional[torch.Tensor]
    sink: Optional[torch.Tensor]


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


class _ManualSdpa(NamedTuple):
    """Setup of a dense forward that runs a manual fp32 attention (sinks or
    dropout)."""

    scale: float
    q_float: torch.Tensor
    k_t: torch.Tensor
    v_float: torch.Tensor
    mask: Optional[torch.Tensor]
    sink: Optional[torch.Tensor]


class _ExplicitDropoutSdpa(NamedTuple):
    """Setup of a dense forward with an explicit dropout mask."""

    attention: _ManualSdpa
    keep: torch.Tensor  # float32 dropout mask
    keep_scale: float  # dropout_scale


def _sdpa_bool(node: Dict[str, Any], key: str, default: bool = False) -> bool:
    return bool(_node_param(node, key, default))


def _sdpa_unsupported_if_present(node: Dict[str, Any], keys: Sequence[str]) -> None:
    for key in keys:
        if _optional_uid(node, key) is not None:
            raise ValueError(
                f"Unsupported SDPA optional tensor '{key}' in PyTorch reference"
            )


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


#: Dropout forms of an SDPA node (``_SdpaPlan.dropout_form``). None is no
#: dropout.
_EXPLICIT_DROPOUT = "explicit"
_PHILOX_DROPOUT = "philox"


class _SdpaPlan(NamedTuple):
    mask_uid: Optional[int]
    is_causal: bool
    scale_uid: Optional[int]
    attn_scale_value: Any
    window: Optional[int]
    bottom_right: bool
    sink_uid: Optional[int]
    alibi: bool
    dropout: SdpaDropout
    dropout_form: Optional[str]


def _forward_dropout_form(dropout: SdpaDropout) -> Optional[str]:
    """The dropout form of a forward node, declined as hipDNN declines it.

    An explicit mask comes with a scale and with no probability, seed, offset
    or ``rng_dump`` (PlanUtils.hpp, rocm-libraries #14358 and #15066; the
    submodule pin predates both). Seed, offset and ``rng_dump`` need a
    non-null probability. A probability of 0 without ``rng_dump`` is no
    dropout, and seed and offset are not read. Philox dropout is validated
    only through the engine's ``rng_dump``: it needs a probability in
    [0, 1), a seed and an offset.
    """
    explicit = {
        "dropout_mask_tensor_uid": dropout.mask_uid,
        "dropout_scale_tensor_uid": dropout.scale_uid,
    }
    given = [key for key, uid in explicit.items() if uid is not None]
    if given:
        if (
            dropout.probability is not None
            or dropout.seed_uid is not None
            or dropout.offset_uid is not None
        ):
            raise ValueError(
                f"SDPA {' and '.join(given)} cannot be combined with "
                "dropout_probability, seed_tensor_uid or offset_tensor_uid"
            )
        if dropout.rng_dump_uid is not None:
            raise ValueError(
                f"SDPA {' and '.join(given)} cannot be combined with "
                "rng_dump_tensor_uid"
            )
        if len(given) == 1:
            missing = next(key for key in explicit if key not in given)
            raise ValueError(f"SDPA {given[0]} needs {missing}")
        return _EXPLICIT_DROPOUT

    philox = {
        "seed_tensor_uid": dropout.seed_uid,
        "offset_tensor_uid": dropout.offset_uid,
        "rng_dump_tensor_uid": dropout.rng_dump_uid,
    }
    p = dropout.probability
    if p is None:
        given = [key for key, uid in philox.items() if uid is not None]
        if given:
            raise ValueError(
                f"SDPA {' and '.join(given)} needs a non-null dropout_probability"
            )
        return None
    if dropout.rng_dump_uid is None:
        if p == 0.0:
            return None
        raise ValueError(
            f"SDPA Philox dropout (dropout_probability={p}) can be validated "
            "only against the engine's rng_dump output (rng_dump_tensor_uid), "
            "and this graph has none"
        )
    if not 0.0 <= p < 1.0:
        raise ValueError(f"SDPA Philox dropout_probability {p} is not in [0, 1)")
    missing = [key for key, uid in philox.items() if uid is None]
    if missing:
        raise ValueError(
            f"SDPA Philox dropout with rng_dump_tensor_uid needs "
            f"{' and '.join(missing)}"
        )
    return _PHILOX_DROPOUT


def _plan_sdpa_common(
    node: Dict[str, Any],
    allow_paged: bool = False,
    allow_sink: bool = False,
) -> _SdpaPlan:
    unsupported = [
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
    if not allow_paged:
        # The paged/varlen inputs are served by the forward handler only: it
        # gathers the page table into dense per-sequence K/V before calling
        # PyTorch, which has no paged API. Backward has no such path, so for it
        # these remain hard rejections rather than a silently dense gradient.
        unsupported += [
            "seq_len_q_tensor_uid",
            "seq_len_kv_tensor_uid",
            "page_table_k_tensor_uid",
            "page_table_v_tensor_uid",
        ]
    _sdpa_unsupported_if_present(node, unsupported)

    if _sdpa_bool(node, "padding_mask"):
        raise ValueError(
            "SDPA padding masks are not supported by the PyTorch reference"
        )

    dropout = read_dropout(node)
    dropout_form = _forward_dropout_form(dropout)

    mask_uid = _optional_uid(node, "attn_mask_tensor_uid")
    is_causal, window, bottom_right = _sdpa_derive_mask(node)
    alibi = _sdpa_bool(node, "alibi_mask")
    if alibi:
        # hipDNN's reference (isSupportedSdpaAlibi) requires top-left
        # alignment with right_bound == 0, and allows any left_bound; cuDNN
        # itself requires only right_bound == 0. Bottom-right alignment and
        # an unbounded non-causal call are declined; a top-left window is not.
        if bottom_right:
            raise ValueError(
                "SDPA ALiBi does not support bottom-right causal alignment; "
                "hipDNN's ALiBi form requires top-left alignment"
            )
        if not is_causal and window is None:
            raise ValueError(
                "SDPA ALiBi requires top-left causal masking or a top-left "
                "sliding window"
            )

    scale_uid = _optional_uid(node, "scale_tensor_uid")
    attn_scale_value = _node_param(node, "attn_scale_value", None)
    sink_uid = _optional_uid(node, "sink_token_tensor_uid")
    return _SdpaPlan(
        mask_uid=mask_uid,
        is_causal=is_causal,
        scale_uid=scale_uid,
        attn_scale_value=attn_scale_value,
        window=window,
        bottom_right=bottom_right,
        sink_uid=sink_uid,
        alibi=alibi,
        dropout=dropout,
        dropout_form=dropout_form,
    )


def _sdpa_derive_mask(
    node: Dict[str, Any],
) -> Tuple[bool, Optional[int], bool]:
    """Resolve ``(is_causal, window_width, bottom_right)`` as hipDNN does.

    The deprecated causal booleans take precedence over the modern bounds and
    alignment. Otherwise, bounds describe a diagonal band, with ``-1`` meaning
    unbounded. ``is_causal`` is reserved for PyTorch's native top-left mask;
    bottom-right masks are materialized as additive masks.
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

    if left < unbounded:
        raise ValueError(f"SDPA left_bound {left} is neither unbounded nor a width")
    if right != 0:
        if right == unbounded and left == unbounded:
            return False, None, False
        raise ValueError(
            f"SDPA right_bound {right} is a forward-looking band the reference "
            "cannot express"
        )
    if left == unbounded:
        return (False, None, True) if bottom_right else (True, None, False)
    return False, left + 1, bottom_right


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
    rows: Optional[Tuple[int, int]] = None,
) -> torch.Tensor:
    """Build the additive diagonal-band mask used by hipDNN SDPA.

    Bottom-right alignment shifts the diagonal by ``kv_len - q_len``. A
    ``None`` width leaves the lower edge unbounded, producing a causal mask.
    ``rows`` limits the mask to query rows ``[start, end)``.
    """
    start, end = (0, q_len) if rows is None else rows
    q_idx = torch.arange(start, end, device=device).unsqueeze(-1)
    k_idx = torch.arange(kv_len, device=device).unsqueeze(0)
    diagonal = q_idx + (kv_len - q_len if bottom_right else 0)
    keep = k_idx <= diagonal
    if width is not None:
        keep &= k_idx > diagonal - width
    mask = torch.zeros((end - start, kv_len), device=device, dtype=dtype)
    return mask.masked_fill(~keep, float("-inf"))


def _alibi_slopes(
    num_heads: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """cuDNN's per-query-head ALiBi slopes (``get_alibi_slope``).

    The geometric schedule covers the closest power of two below or at
    ``num_heads``. Extra heads use the next power of two's odd-indexed
    slopes, as cuDNN and common open-source ALiBi implementations do.
    """
    base = 1 << (num_heads.bit_length() - 1)
    head = torch.arange(base, device=device, dtype=torch.float64)
    exponents = -8.0 * (head + 1.0) / base
    extra = num_heads - base
    if extra > 0:
        odd = torch.arange(extra, device=device, dtype=torch.float64)
        exponents = torch.cat([exponents, -4.0 * (2.0 * odd + 1.0) / base])
    return torch.pow(2.0, exponents).to(dtype=dtype)


def _alibi_bias_block(
    start: int,
    end: int,
    kv_len: int,
    slopes: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """fp32 ALiBi bias ``(j - i) * slope(h)`` for query rows ``[start, end)``.

    ``slopes`` is cuDNN's per-query-head schedule (``_alibi_slopes``), in
    float64. Shaped ``[H, end - start, kv_len]``, broadcasting against scores
    shaped ``[..., H, end - start, kv_len]``. Generated one query block at a
    time so a blocked caller (``_block_scores``) never holds the full
    ``[H, q_len, kv_len]`` bias; stays in float32 past this point so a low
    graph precision never rounds the bias before it reaches the scores.
    """
    q_idx = torch.arange(start, end, device=device, dtype=torch.float64).unsqueeze(-1)
    k_idx = torch.arange(kv_len, device=device, dtype=torch.float64).unsqueeze(0)
    return ((k_idx - q_idx) * slopes.reshape(-1, 1, 1)).to(dtype=torch.float32)


def _alibi_mask_rows(
    start: int,
    end: int,
    q_len: int,
    kv_len: int,
    slopes: torch.Tensor,
    attn_mask: Optional[torch.Tensor],
    window: Optional[int],
) -> torch.Tensor:
    """float32 additive mask of an ALiBi call for query rows ``[start, end)``.

    The ALiBi bias and the explicit ``attn_mask`` go first, the band second
    (SEM-006). ``_plan_sdpa_common`` leaves ALiBi only top-left causal or
    top-left windowed bounds, so the band is always part of the mask.
    """
    mask = _alibi_bias_block(start, end, kv_len, slopes, slopes.device)
    if attn_mask is not None:
        mask = mask + _rows(attn_mask, start, end).to(dtype=torch.float32)
    return mask.add_(
        _diagonal_band_mask(
            q_len,
            kv_len,
            window,
            bottom_right=False,
            device=mask.device,
            dtype=torch.float32,
            rows=(start, end),
        )
    )


def _alibi_mask(
    q_len: int,
    kv_len: int,
    slopes: torch.Tensor,
    attn_mask: Optional[torch.Tensor],
    window: Optional[int],
    dtype: torch.dtype,
) -> torch.Tensor:
    """The whole additive mask of an ALiBi call (``_alibi_mask_rows``).

    Shaped ``[H, Sq, Skv]``, or broadcast with ``attn_mask``. For a call that
    takes the mask in one tensor: the SDPA kernel, in the graph dtype, or the
    manual fp32 paths. It is filled one query block at a time, so each value
    is rounded to ``dtype`` once and no full-size float32 copy exists.
    """
    shape: Sequence[int] = (int(slopes.numel()), q_len, kv_len)
    if attn_mask is not None:
        shape = torch.broadcast_shapes(shape, attn_mask.shape)
    mask = torch.empty(shape, device=slopes.device, dtype=dtype)
    # A block holds the float64 bias, its float32 copy and the float32 sum
    # with ``attn_mask``: 16 bytes per mask element.
    row_bytes = torch.Size(shape[:-2]).numel() * kv_len * 16
    for start, end in _row_blocks(q_len, row_bytes):
        mask[..., start:end, :] = _alibi_mask_rows(
            start, end, q_len, kv_len, slopes, attn_mask, window
        )
    return mask


def _alibi_stats(
    q: torch.Tensor,
    k: torch.Tensor,
    rep_k: int,
    scale: float,
    attn_mask: Optional[torch.Tensor],
    window: Optional[int],
) -> torch.Tensor:
    """float32 log-sum-exp of an ALiBi forward, from the unrounded bias.

    For the one-shot reference, whose SDPA kernel read the bias rounded to
    the graph dtype. One query block at a time on every device, with each
    block's mask built on the fly (``_alibi_mask_rows``), so no full-size
    score or mask tensor exists.
    """
    q_len, kv_len = int(q.shape[-2]), int(k.shape[-2])
    slopes = _alibi_slopes(int(q.shape[-3]), q.device, torch.float64)
    q_float = q.to(dtype=torch.float32)
    k_float_t = _fp32_at_query_heads(k, rep_k).transpose(-2, -1)
    row_bytes = q.shape[:-2].numel() * kv_len * 4
    return torch.cat(
        [
            torch.logsumexp(
                _scores(
                    q_float[..., start:end, :],
                    k_float_t,
                    _alibi_mask_rows(
                        start, end, q_len, kv_len, slopes, attn_mask, window
                    ),
                    scale,
                ),
                -1,
                keepdim=True,
            )
            for start, end in _row_blocks(q_len, row_bytes)
        ],
        dim=-2,
    )


def _attention_bounds(
    q_len: int,
    kv_len: int,
    window: Optional[int],
    bottom_right: bool,
    extra_bias: Optional[torch.Tensor],
    is_causal: bool,
    like: torch.Tensor,
) -> Tuple[Optional[torch.Tensor], bool]:
    """The ``(attn_mask, is_causal)`` pair that expresses a call's bounds.

    ``extra_bias`` is the explicit ``attn_mask``. The bias goes onto the
    scores first and the band second (SEM-006), so the band is added to the
    bias. ALiBi calls build their mask with ``_alibi_mask`` instead.

    torch's SDPA spells only top-left causal natively (``is_causal``). A window
    or a bottom-right alignment needs an additive band mask, unless the bounds
    equal something native: bottom-right causal with ``q_len == kv_len`` is
    top-left causal, and with ``q_len == 1`` (decode) every key is visible.
    Passing those natively keeps the fused kernels eligible, since Flash
    rejects any ``attn_mask``. A bias always needs the band as a mask tensor.
    """
    if extra_bias is None:
        if window is None and not bottom_right:
            return None, is_causal
        if window is None:
            if q_len == kv_len:
                return None, True
            if q_len == 1:
                return None, False
    elif window is None and not bottom_right and not is_causal:
        return extra_bias, False
    band = _diagonal_band_mask(
        q_len,
        kv_len,
        window,
        bottom_right=bottom_right,
        device=like.device,
        dtype=like.dtype,
    )
    return (band if extra_bias is None else band + extra_bias), False


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


def _require_fused_default_sdpa(
    tensors: Dict[int, torch.Tensor],
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: Optional[torch.Tensor],
    is_causal: bool,
    scale: Optional[float],
    enable_gqa: bool,
) -> None:
    """Refuse to time a GQA call that default dispatch sends to math.

    The math backend expands K/V to the query head count inside the call, the
    copy this handler keeps out of the timed region. Only the GPU benchmark
    path (``ReplayTensors``) is checked: the one-shot reference path computes
    outputs, not timings, and math is a correct answer there.
    """
    if not (
        enable_gqa
        and isinstance(tensors, ReplayTensors)
        and q.is_cuda
        and strict_sdpa_selection() is None
    ):
        return
    if not default_sdpa_avoids_math(
        q,
        k,
        v,
        attn_mask=attn_mask,
        dropout_p=0.0,
        is_causal=is_causal,
        scale=scale,
        enable_gqa=enable_gqa,
    ):
        raise UnsupportedGraphError(
            "PyTorch default SDPA dispatch would use the math backend for this "
            "GQA graph, which expands K/V inside the timed call"
        )


def _plan_paged_sdpa(
    node: Dict[str, Any],
    tensors: Dict[int, torch.Tensor],
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    page_table_k_uid: int,
    page_table_v_uid: int,
    seq_len_q_uid: Optional[int],
    seq_len_kv_uid: int,
    attn_mask: Optional[torch.Tensor],
    is_causal: bool,
    scale: float,
    rep_k: int,
    rep_v: int,
    window: Optional[int],
    bottom_right: bool,
    sink: Optional[torch.Tensor],
) -> Tuple[List[_PagedSegment], bool]:
    """Gather a paged KV cache into dense per-sequence K/V and masks.

    Layouts, taken from the committed paged bundles rather than assumed:

      * paged K/V are ``[num_blocks, num_kv_heads, page_size, head_size]``
      * the page table is ``[num_seqs, max_blocks_per_seq]`` of int32 block ids
      * ``seq_len_kv`` is ``[num_seqs]`` LENGTHS (not offsets -- the page table
        is itself the per-sequence indirection, so paged K/V carry no ragged
        offsets)
      * Q is ``[1, num_query_heads, total_q, head_size]``, with the sequences
        packed along ``total_q``

    Each sequence's live KV is ``ceil(len / page_size)`` blocks gathered in page
    order and then trimmed to the exact length, so trailing slots in the last
    page never contribute. The gather emulates a layout PyTorch has no API for,
    so it is setup: the benchmark path builds it once, untimed.
    """
    if attn_mask is not None:
        raise ValueError("Paged SDPA with an explicit attn_mask is not supported")

    page_table_k = _tensor(tensors, page_table_k_uid, node)
    page_table_v = _tensor(tensors, page_table_v_uid, node)
    kv_lengths = _host_ints(tensors, seq_len_kv_uid, node)

    if k.ndim != 4 or v.ndim != 4:
        raise ValueError("Paged SDPA expects rank-4 paged K/V containers")
    page_size = int(k.shape[-2])
    num_seqs = len(kv_lengths)
    if int(page_table_k.shape[0]) != num_seqs:
        raise ValueError(
            f"Paged SDPA page table has {int(page_table_k.shape[0])} rows for "
            f"{num_seqs} sequences"
        )

    if seq_len_q_uid is not None:
        # Host-resolved once per replay map: a .tolist() here would sync the
        # stream inside the staged timer's gated region and deadlock.
        q_lengths = _host_ints(tensors, seq_len_q_uid, node)
        if len(q_lengths) != num_seqs:
            raise ValueError("Paged SDPA seq_len_q/seq_len_kv disagree on num_seqs")
    else:
        # No per-sequence Q lengths: the packed Q must divide evenly.
        total_q = int(q.shape[-2])
        if total_q % num_seqs != 0:
            raise ValueError(
                f"Paged SDPA cannot split {total_q} queries across {num_seqs} "
                "sequences without seq_len_q"
            )
        q_lengths = [total_q // num_seqs] * num_seqs

    segments: List[_PagedSegment] = []
    enable_gqa = False
    probed = set()
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

        q_seq = q[..., q_start : q_start + q_len, :]
        if q_seq.ndim == 4:
            k_seq = k_seq.unsqueeze(0)
            v_seq = v_seq.unsqueeze(0)
        seq_q_len = int(q_seq.shape[-2])
        seq_mask, seq_causal = _attention_bounds(
            seq_q_len, kv_len, window, bottom_right, None, is_causal, q
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
                )
            )
            q_start += q_len
            continue

        k_seq, v_seq, enable_gqa = _sdpa_kv_operands(
            k_seq.contiguous(), v_seq.contiguous(), rep_k, rep_v
        )
        # Dispatch depends on shapes, not values: probe each shape once.
        if (seq_q_len, kv_len) not in probed:
            probed.add((seq_q_len, kv_len))
            _require_fused_default_sdpa(
                tensors,
                q_seq,
                k_seq,
                v_seq,
                seq_mask,
                seq_causal,
                scale,
                enable_gqa,
            )
        segments.append(
            _PagedSegment(q_start, q_len, k_seq, v_seq, seq_mask, seq_causal)
        )
        q_start += q_len
    return segments, enable_gqa


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


def _block_rows(q_len: int, row_bytes: int) -> int:
    """Query rows per block when one row takes ``row_bytes``."""
    return min(max(q_len, 1), max(1, _SCORE_BLOCK_BYTES // max(row_bytes, 1)))


def _row_blocks(q_len: int, row_bytes: int) -> Sequence[Tuple[int, int]]:
    """``(start, end)`` query row ranges of at most ``_SCORE_BLOCK_BYTES``
    each; one empty range when ``q_len`` is 0."""
    step = _block_rows(q_len, row_bytes)
    return [
        (start, min(start + step, q_len)) for start in range(0, max(q_len, 1), step)
    ]


def _query_blocks(q: torch.Tensor, kv_len: int) -> Sequence[Tuple[int, int]]:
    """``(start, end)`` query row ranges of the score matrix (``_row_blocks``).

    Only host tensors are split. On a GPU the timed PyTorch row and the GPU
    reference keep one block: smaller launches would slow the timed row, and
    a GPU out-of-memory error already falls back to the CPU reference.
    """
    q_len = int(q.shape[-2])
    if q.device.type != "cpu":
        return [(0, q_len)]
    return _row_blocks(q_len, q.shape[:-2].numel() * kv_len * 4)


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
    alibi_slopes: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """float32 scores of query rows ``start:start+len`` against every key.

    The causal mask uses absolute query indices, so key ``j`` is visible to
    query ``i`` only when ``j <= i`` (top-left alignment). ``alibi_slopes``,
    when given, adds this block's ALiBi bias only (``_alibi_bias_block``), so
    memory never grows with ``q_len * kv_len``.
    """
    scores = torch.matmul(q_block.to(dtype=torch.float32), k_f.transpose(-2, -1))
    scores *= scale
    if mask_block is not None:
        scores += mask_block.to(dtype=torch.float32)
    if alibi_slopes is not None:
        scores += _alibi_bias_block(
            start,
            start + int(scores.shape[-2]),
            int(scores.shape[-1]),
            alibi_slopes,
            scores.device,
        )
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
    causal mask becomes an additive ``-inf`` band that is added to any
    ``attn_mask`` bias.
    """
    mask = None if attn_mask is None else attn_mask.to(dtype=torch.float32)
    if is_causal:
        band = _diagonal_band_mask(
            q_len,
            int(k.shape[-2]),
            None,
            bottom_right=False,
            device=k.device,
            dtype=torch.float32,
        )
        mask = band if mask is None else band + mask
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


def _dropout_attention(
    q_float: torch.Tensor,
    k_float_t: torch.Tensor,
    v_float: torch.Tensor,
    mask_rows: Callable[[int, int], Optional[torch.Tensor]],
    sink_logits: Optional[torch.Tensor],
    scale: float,
    keep: torch.Tensor,
    keep_scale: float,
    blocks: Sequence[Tuple[int, int]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """fp32 attention with dropout: ``O = (P * keep * keep_scale) V``.

    ``P`` is the softmax before dropout, normalised with the sink logits when
    there are sinks (CpuFpReferenceSdpa.hpp, rocm-libraries #14358; the
    submodule pin predates it). ``keep`` is the dropout mask
    (``[B|1, Hq|1, Sq, Skv]``), not renormalised. ``mask_rows(start, end)``
    gives the additive mask of query rows ``[start, end)``. Returns the fp32
    output and the log-sum-exp before dropout (the stats). A query row that
    sees no key gets ``P = 0`` and stats of -inf, as in CpuFpReferenceSdpa.
    One query block of ``blocks`` at a time; the probabilities overwrite the
    block's scores.
    """
    finite_min = torch.finfo(torch.float32).min
    outputs, stats = [], []
    for start, end in blocks:
        scores = _scores(
            q_float[..., start:end, :], k_float_t, mask_rows(start, end), scale
        )
        log_denom = torch.logsumexp(scores, dim=-1, keepdim=True)
        if sink_logits is not None:
            log_denom = torch.logaddexp(log_denom, sink_logits)
        # A -inf denominator (no visible key) would give -inf - -inf = NaN. A
        # finite one keeps exp(-inf - finite) = 0 for those rows.
        probs = scores.sub_(log_denom.clamp(min=finite_min)).exp_()
        probs.mul_(_rows(keep, start, end)).mul_(keep_scale)
        outputs.append(torch.matmul(probs, v_float))
        stats.append(log_denom)
    if len(outputs) == 1:
        return outputs[0], stats[0]
    return torch.cat(outputs, dim=-2), torch.cat(stats, dim=-2)


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


def _graph_tensor(
    graph_json: Dict[str, Any], uid: int, key: str
) -> Tuple[Tuple[int, ...], Optional[str]]:
    """``(dims, data_type)`` of tensor ``uid`` from the graph's tensor list."""
    for tensor_json in graph_json.get("tensors", []):
        if tensor_json.get("uid") == uid:
            dims = tuple(int(dim) for dim in tensor_json.get("dims", []))
            return dims, tensor_data_type(tensor_json, graph_json)
    raise ValueError(f"SDPA {key} {uid} is not in the graph's tensor list")


def _attention_dims(
    node: Dict[str, Any], graph_json: Dict[str, Any]
) -> Tuple[int, int, int, int]:
    """``(B, Hq, Sq, Skv)`` of an SDPA node from its Q and K tensors."""
    q, _ = _graph_tensor(
        graph_json, _required_input_uid(node, "q_tensor_uid"), "q_tensor_uid"
    )
    k, _ = _graph_tensor(
        graph_json, _required_input_uid(node, "k_tensor_uid"), "k_tensor_uid"
    )
    if len(q) != 4 or len(k) != 4:
        raise ValueError(
            f"SDPA dropout needs rank-4 Q and K, not {list(q)} and {list(k)}"
        )
    return q[0], q[1], q[2], k[2]


def _check_explicit_dropout(
    node: Dict[str, Any], graph_json: Dict[str, Any], dropout: SdpaDropout
) -> None:
    """Decline the dropout mask and scale tensors that hipDNN declines.

    The mask is FLOAT ``[B|1, Hq|1, Sq, Skv]``, indexed by the query head;
    the scale is one FLOAT element (PlanUtils.hpp, rocm-libraries #14358;
    the submodule pin predates it).
    """
    b, hq, sq, skv = _attention_dims(node, graph_json)
    dims, data_type = _graph_tensor(
        graph_json, dropout.mask_uid, "dropout_mask_tensor_uid"
    )
    if data_type != "float":
        raise ValueError(f"SDPA dropout mask must be FLOAT, not {data_type!r}")
    if (
        len(dims) != 4
        or dims[0] not in (b, 1)
        or dims[1] not in (hq, 1)
        or dims[2:] != (sq, skv)
    ):
        raise ValueError(
            f"SDPA dropout mask dims {list(dims)} are not [B|1, Hq|1, Sq, Skv] "
            f"= [{b}|1, {hq}|1, {sq}, {skv}]"
        )
    dims, data_type = _graph_tensor(
        graph_json, dropout.scale_uid, "dropout_scale_tensor_uid"
    )
    if data_type != "float" or _numel(dims) != 1:
        raise ValueError(
            f"SDPA dropout scale must be one FLOAT element, not {data_type!r} "
            f"with dims {list(dims)}"
        )


def _philox_dump(
    node: Dict[str, Any], graph_json: Dict[str, Any], dropout: SdpaDropout
) -> torch.Tensor:
    """The engine's ``rng_dump`` of a Philox forward, as its float32 mask.

    The graph's ``rng_dump`` is FLOAT ``[B, Hq, Sq, Skv]`` (PlanUtils.hpp,
    rocm-libraries #14358; the submodule pin predates it). The
    dump comes from the ``use_philox_dumps`` scope, never from the tensor map:
    a buffer manager puts a zeroed ``rng_dump`` buffer there. Outside a scope
    the graph is declined, so PyTorch rows of Philox graphs stay declined.
    """
    b, hq, sq, skv = _attention_dims(node, graph_json)
    uid = dropout.rng_dump_uid
    dims, data_type = _graph_tensor(graph_json, uid, "rng_dump_tensor_uid")
    if data_type != "float":
        raise ValueError(f"SDPA rng_dump must be FLOAT, not {data_type!r}")
    if dims != (b, hq, sq, skv):
        raise ValueError(
            f"SDPA rng_dump dims {list(dims)} are not [B, Hq, Sq, Skv] "
            f"= [{b}, {hq}, {sq}, {skv}]"
        )
    dump = current_philox_dumps().get(uid)
    if dump is None:
        raise ValueError(
            "SDPA Philox dropout can be validated only against the engine's "
            f"rng_dump (rng_dump_tensor_uid {uid}), and this PyTorch run has none"
        )
    if tuple(dump.shape) != dims:
        # A caller error, not an unsupported graph: fail, do not decline.
        raise RuntimeError(
            f"The rng_dump given for tensor {uid} has shape "
            f"{list(dump.shape)}, not the graph's {list(dims)}"
        )
    return dump.to(dtype=torch.float32)


def _philox_scale(probability: float) -> float:
    """``1 / (1 - p)`` evaluated in float32, as hipDNN evaluates it."""
    one = torch.tensor(1.0, dtype=torch.float32)
    return float(one / (one - torch.tensor(probability, dtype=torch.float32)))


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

    The ``stats`` output (log-sum-exp) comes from the fused kernel, as in a
    training forward: Q/K/V go in with ``requires_grad`` and the kernel's saved
    log-sum-exp is read back (``fused_logsumexp``). Only a backend that saves
    none (math) gets a separate score matmul. The kernel reads an ALiBi bias
    rounded to a bf16 or fp16 graph dtype, so the one-shot reference of such
    a graph computes the log-sum-exp from the float32 bias (``_alibi_stats``).

    PyTorch's SDPA has no attention-sink input, so a graph with sinks runs a
    manual fp32 attention (``_sink_attention``); its fp32 operands are built
    once too, and its softmax denominator is the stats output.

    PyTorch's SDPA takes no dropout mask either. A graph with dropout runs a
    manual fp32 attention (``_dropout_attention``) that applies the dropout
    mask and scale to the softmax; the stats are the log-sum-exp before
    dropout. An explicit mask builds its operands once. Philox dropout uses
    the engine's ``rng_dump`` as its mask (``_philox_dump``) with the scale
    ``1 / (1 - p)``, one query block at a time on every device. Dropout with
    paged or ragged K/V is declined.
    """
    _sdpa_unsupported_if_present(
        node,
        [
            "max_tensor_uid",
            "sum_exp_tensor_uid",
            "amax_s_tensor_uid",
            "amax_o_tensor_uid",
        ],
    )
    q_uid = _required_input_uid(node, "q_tensor_uid")
    k_uid = _required_input_uid(node, "k_tensor_uid")
    v_uid = _required_input_uid(node, "v_tensor_uid")
    o_uid = _required_output_uid(node, "o_tensor_uid")
    plan = _plan_sdpa_common(node, allow_paged=True, allow_sink=True)
    is_causal = plan.is_causal
    stats_uid = _optional_uid(node, "stats_tensor_uid")

    page_table_k_uid = _optional_uid(node, "page_table_k_tensor_uid")
    page_table_v_uid = _optional_uid(node, "page_table_v_tensor_uid")
    seq_len_q_uid = _optional_uid(node, "seq_len_q_tensor_uid")
    seq_len_kv_uid = _optional_uid(node, "seq_len_kv_tensor_uid")
    if plan.dropout_form is not None:
        layout = [
            key
            for key, uid in (
                ("page_table_k_tensor_uid", page_table_k_uid),
                ("page_table_v_tensor_uid", page_table_v_uid),
                ("seq_len_q_tensor_uid", seq_len_q_uid),
                ("seq_len_kv_tensor_uid", seq_len_kv_uid),
            )
            if uid is not None
        ]
        if layout:
            raise ValueError(
                "SDPA dropout is not supported with paged or ragged K/V "
                f"({', '.join(layout)})"
            )
    is_paged = page_table_k_uid is not None or page_table_v_uid is not None
    if is_paged and plan.alibi:
        raise UnsupportedGraphError("SDPA ALiBi is not supported with paged attention")
    if is_paged:
        if page_table_k_uid is None or page_table_v_uid is None:
            raise ValueError("Paged SDPA needs both K and V page tables")
        if seq_len_kv_uid is None:
            raise ValueError("Paged SDPA needs seq_len_kv to bound each sequence")
        if stats_uid is not None:
            raise ValueError("Paged SDPA stats output is not supported")
    if plan.dropout_form == _EXPLICIT_DROPOUT:
        _check_explicit_dropout(node, graph_json, plan.dropout)
    philox_dump = None
    if plan.dropout_form == _PHILOX_DROPOUT:
        philox_dump = _philox_dump(node, graph_json, plan.dropout)
        philox_scale = _philox_scale(plan.dropout.probability)

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
            plan.dropout.mask_uid,
            plan.dropout.scale_uid,
            page_table_k_uid,
            page_table_v_uid,
            seq_len_q_uid,
            seq_len_kv_uid,
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

    def bounds(q, k, attn_mask, alibi_dtype):
        """``(attn_mask, is_causal)`` of the call; an ALiBi mask in ``alibi_dtype``."""
        if plan.alibi:
            slopes = _alibi_slopes(int(q.shape[-3]), q.device, torch.float64)
            return (
                _alibi_mask(
                    int(q.shape[-2]),
                    int(k.shape[-2]),
                    slopes,
                    attn_mask,
                    plan.window,
                    alibi_dtype,
                ),
                False,
            )
        return _attention_bounds(
            int(q.shape[-2]),
            int(k.shape[-2]),
            plan.window,
            plan.bottom_right,
            attn_mask,
            is_causal,
            q,
        )

    def block_bounds(
        q: torch.Tensor, k: torch.Tensor, attn_mask: Optional[torch.Tensor]
    ) -> Callable[[int, int], Optional[torch.Tensor]]:
        """The float32 additive mask of query rows ``[start, end)``, built for
        one block at a time, so no full-size mask exists."""
        q_len, kv_len = int(q.shape[-2]), int(k.shape[-2])
        if plan.alibi:
            slopes = _alibi_slopes(int(q.shape[-3]), q.device, torch.float64)
            return lambda start, end: _alibi_mask_rows(
                start, end, q_len, kv_len, slopes, attn_mask, plan.window
            )
        banded = is_causal or plan.window is not None or plan.bottom_right

        def rows(start: int, end: int) -> Optional[torch.Tensor]:
            mask = _rows(attn_mask, start, end)
            if mask is not None:
                mask = mask.to(dtype=torch.float32)
            if not banded:
                return mask
            band = _diagonal_band_mask(
                q_len,
                kv_len,
                plan.window,
                bottom_right=plan.bottom_right,
                device=q.device,
                dtype=torch.float32,
                rows=(start, end),
            )
            return band if mask is None else mask + band

        return rows

    def rounds_alibi(d: _DenseSdpa) -> bool:
        """Whether the call mask holds the ALiBi bias rounded below float32."""
        return plan.alibi and torch.finfo(d.attn_mask.dtype).bits < 32

    def prepare_manual(
        tensors: Dict[int, torch.Tensor],
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> _ManualSdpa:
        attn_mask, scale, rep_k, rep_v, sink = resolve(tensors, q, k, v)
        attn_mask, causal = bounds(q, k, attn_mask, torch.float32)
        k_t, mask = _score_operands(k, attn_mask, causal, int(q.shape[-2]), rep_k)
        return _ManualSdpa(
            scale,
            q.to(dtype=torch.float32),
            k_t,
            _fp32_at_query_heads(v, rep_v),
            mask,
            sink,
        )

    def prepare_explicit_dropout(
        tensors: Dict[int, torch.Tensor],
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> _ExplicitDropoutSdpa:
        mask = _tensor(tensors, plan.dropout.mask_uid, node)
        return _ExplicitDropoutSdpa(
            prepare_manual(tensors, q, k, v),
            mask.to(dtype=torch.float32),
            _scalar_value(tensors, plan.dropout.scale_uid, node),
        )

    def prepare_dense(
        tensors: Dict[int, torch.Tensor],
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> _DenseSdpa:
        attn_mask, scale, rep_k, rep_v, _ = resolve(tensors, q, k, v)
        # The kernel reads the ALiBi bias in the graph dtype. It is rounded
        # here, once, and never in a timed iteration.
        attn_mask, causal = bounds(q, k, attn_mask, q.dtype)
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
        segments, enable_gqa = _plan_paged_sdpa(
            node,
            tensors,
            q,
            k,
            v,
            page_table_k_uid,
            page_table_v_uid,
            seq_len_q_uid,
            seq_len_kv_uid,
            attn_mask,
            is_causal,
            scale,
            rep_k,
            rep_v,
            plan.window,
            plan.bottom_right,
            sink,
        )
        q_float = None if sink is None else q.to(dtype=torch.float32)
        return _PagedSdpa(segments, scale, enable_gqa, q_float, sink)

    def stats_operands(
        tensors: Dict[int, torch.Tensor],
        q: torch.Tensor,
        k: torch.Tensor,
        d: _DenseSdpa,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        # Reached when the kernel saved no log-sum-exp. Math saves none, and a
        # math user computes it. Under any other selection the fallback adds a
        # score matmul and a logsumexp to the timed region, so say so on the row.
        if (
            isinstance(tensors, ReplayTensors)
            and strict_sdpa_selection() is not PyTorchSdpaBackendName.MATH
        ):
            tensors.note(_STATS_FALLBACK_WARNING)
        attn_mask = d.attn_mask
        if rounds_alibi(d):
            attn_mask = bounds(q, k, mask_input(tensors), torch.float32)[0]
        return (
            q.to(dtype=torch.float32),
            *_score_operands(k, attn_mask, d.is_causal, int(q.shape[-2]), d.rep_k),
        )

    def mask_input(tensors: Dict[int, torch.Tensor]) -> Optional[torch.Tensor]:
        if plan.mask_uid is None:
            return None
        return _tensor(tensors, plan.mask_uid, node)

    def store_manual(
        tensors: Dict[int, torch.Tensor],
        q: torch.Tensor,
        o: torch.Tensor,
        stats: torch.Tensor,
    ) -> None:
        """Store the fp32 output of a manual attention and its stats."""
        _store_tensor(tensors, o_uid, o.to(dtype=q.dtype))
        if stats_uid is not None:
            _store_tensor(tensors, stats_uid, stats)

    # Everything derived from the constant inputs is built once per replay
    # map, during priming. Timed iterations run only the SDPA call(s) and, for
    # paged graphs, the per-sequence loop and re-pack.
    def run(tensors: Dict[int, torch.Tensor]) -> None:
        q = _tensor(tensors, q_uid, node)
        k = _tensor(tensors, k_uid, node)
        v = _tensor(tensors, v_uid, node)
        if is_paged:
            paged = setup(tensors, "paged", lambda: prepare_paged(tensors, q, k, v))
            if paged.sink is not None:
                outputs = [
                    _sink_attention(
                        paged.q_float[..., s.q_start : s.q_start + s.q_len, :],
                        s.k,
                        s.v,
                        s.mask,
                        paged.sink,
                        paged.scale,
                    )[0]
                    for s in paged.segments
                ]
                o = torch.cat(outputs, dim=-2).to(dtype=q.dtype)
            else:
                outputs = [
                    execute_selected_sdpa(
                        q[..., s.q_start : s.q_start + s.q_len, :],
                        s.k,
                        s.v,
                        attn_mask=s.mask,
                        dropout_p=0.0,
                        is_causal=s.is_causal,
                        scale=paged.scale,
                        enable_gqa=paged.enable_gqa,
                    )
                    for s in paged.segments
                ]
                o = torch.cat(outputs, dim=-2)
            # Re-packed along the query axis in the order the sequences were laid out.
            _store_tensor(tensors, o_uid, o)
            return

        if philox_dump is not None:
            attn_mask, scale, rep_k, rep_v, sink = resolve(tensors, q, k, v)
            q_len, kv_len = int(q.shape[-2]), int(k.shape[-2])
            o, stats = _dropout_attention(
                q.to(dtype=torch.float32),
                _fp32_at_query_heads(k, rep_k).transpose(-2, -1),
                _fp32_at_query_heads(v, rep_v),
                block_bounds(q, k, attn_mask),
                sink,
                scale,
                philox_dump.to(device=q.device),
                philox_scale,
                _row_blocks(q_len, q.shape[:-2].numel() * kv_len * 4),
            )
            store_manual(tensors, q, o, stats)
            return

        if plan.dropout_form == _EXPLICIT_DROPOUT:
            e = setup(
                tensors,
                "dropout",
                lambda: prepare_explicit_dropout(tensors, q, k, v),
            )
            a = e.attention
            o, stats = _dropout_attention(
                a.q_float,
                a.k_t,
                a.v_float,
                lambda start, end: _rows(a.mask, start, end),
                a.sink,
                a.scale,
                e.keep,
                e.keep_scale,
                _query_blocks(a.q_float, int(a.k_t.shape[-1])),
            )
            store_manual(tensors, q, o, stats)
            return

        if plan.sink_uid is not None:
            s = setup(tensors, "sink", lambda: prepare_manual(tensors, q, k, v))
            o, stats = _sink_attention(
                s.q_float, s.k_t, s.v_float, s.mask, s.sink, s.scale
            )
            store_manual(tensors, q, o, stats)
            return

        d = setup(tensors, "dense", lambda: prepare_dense(tensors, q, k, v))
        with torch.enable_grad():
            o = execute_selected_sdpa(
                d.q,
                d.k,
                d.v,
                attn_mask=d.attn_mask,
                dropout_p=0.0,
                is_causal=d.is_causal,
                scale=d.scale,
                enable_gqa=d.enable_gqa,
            )
        if stats_uid is None:
            _store_tensor(tensors, o_uid, o)
            return

        _store_tensor(tensors, o_uid, o.detach())
        if rounds_alibi(d) and not isinstance(tensors, ReplayTensors):
            # The one-shot reference. The kernel's log-sum-exp saw the bias
            # rounded to the graph dtype, so take it from the float32 bias.
            # A timed row keeps the kernel's own, as a training step would.
            _store_tensor(
                tensors,
                stats_uid,
                _alibi_stats(q, k, d.rep_k, d.scale, mask_input(tensors), plan.window),
            )
            return
        stats = fused_logsumexp(o, int(q.shape[-2]))
        if stats is None:
            stats = _sdpa_stats(
                *setup(tensors, "stats", lambda: stats_operands(tensors, q, k, d)),
                d.scale,
            )
        _store_tensor(tensors, stats_uid, stats)

    return run


def _decline_backward_philox(dropout: SdpaDropout) -> None:
    """Decline Philox dropout in the backward, as hipDNN does (PlanUtils.hpp,
    rocm-libraries #14358; the submodule pin predates it).

    Runs before the forward dropout rules of ``_plan_sdpa_common``, whose
    messages name an ``rng_dump`` output that a backward node cannot have.
    """
    p = dropout.probability
    if p is not None and p != 0.0:
        raise ValueError(
            f"SDPA Philox backward dropout (dropout_probability={p}) is not "
            "supported: the hipDNN backward graph has no dropout mask output, "
            "so the reference has no mask to apply (see rocm-libraries #15643)"
        )
    if dropout.seed_uid is not None or dropout.offset_uid is not None:
        raise ValueError(
            "SDPA backward dropout with seed_tensor_uid or offset_tensor_uid is "
            "not supported"
        )


def _check_backward_dropout(
    node: Dict[str, Any], graph_json: Dict[str, Any], plan: _SdpaPlan
) -> None:
    """Decline the backward explicit-mask dropout that hipDNN declines.

    The backward takes an explicit mask with its scale and an optional
    one-FLOAT-element ``dropout_scale_inv`` (PlanUtils.hpp, rocm-libraries
    #14358; the submodule pin predates it).
    """
    dropout = plan.dropout
    if dropout.scale_inv_uid is not None:
        if plan.dropout_form != _EXPLICIT_DROPOUT:
            raise ValueError(
                "SDPA dropout_scale_inv_tensor_uid needs dropout_mask_tensor_uid"
            )
        dims, data_type = _graph_tensor(
            graph_json, dropout.scale_inv_uid, "dropout_scale_inv_tensor_uid"
        )
        if data_type != "float" or _numel(dims) != 1:
            raise ValueError(
                "SDPA dropout_scale_inv must be one FLOAT element, not "
                f"{data_type!r} with dims {list(dims)}"
            )
    if plan.dropout_form == _EXPLICIT_DROPOUT:
        _check_explicit_dropout(node, graph_json, dropout)


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

    With an explicit dropout mask ``M``, scale ``s`` and ``dropout_scale_inv``
    ``inv`` (1 when absent), ``P`` stays the softmax before dropout and
    ``D = rowsum(dO * O) * inv``, ``dP = (dO V^T) * M``,
    ``dS = s * P * (dP - D)``, ``dV = (P * M * s)^T dO``
    (CpuFpReferenceSdpa.hpp, rocm-libraries #14358; the submodule pin
    predates it).
    """
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
    _decline_backward_philox(read_dropout(node))
    plan = _plan_sdpa_common(node)
    _check_backward_dropout(node, graph_json, plan)
    dropout = plan.dropout if plan.dropout_form == _EXPLICIT_DROPOUT else None
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
        alibi_slopes = (
            _alibi_slopes(int(q.shape[1]), q.device, torch.float64)
            if plan.alibi
            else None
        )
        keep = None
        keep_scale = 1.0
        if dropout is not None:
            keep = _tensor(tensors, dropout.mask_uid, node).to(dtype=torch.float32)
            keep_scale = _scalar_value(tensors, dropout.scale_uid, node)
            scale_inv = 1.0
            if dropout.scale_inv_uid is not None:
                scale_inv = _scalar_value(tensors, dropout.scale_inv_uid, node)

        if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
            raise ValueError("SDPA backward expects rank-4 q/k/v tensors [B, H, S, D]")

        q_f = q.to(dtype=torch.float32)
        k_f = k.to(dtype=torch.float32)
        v_f = v.to(dtype=torch.float32)
        o_f = o.to(dtype=torch.float32)
        do_f = do.to(dtype=torch.float32)
        stats_f = _require_fp32_stat(stats, "SDPA stats (log-sum-exp)")

        head_dim = int(q.shape[-1])
        scale_value = float(scale)
        k_heads = int(k.shape[1])
        v_heads = int(v.shape[1])
        if rep_k > 1:
            k_f = k_f.repeat_interleave(rep_k, dim=1)
        if rep_v > 1:
            v_f = v_f.repeat_interleave(rep_v, dim=1)

        # One query block at a time: a block's probs, d_scores and ALiBi bias
        # are at most _SCORE_BLOCK_BYTES each, so memory does not grow with
        # q_len * kv_len. dQ rows are independent; dK and dV sum over every
        # query row.
        dq = torch.empty_like(q_f)
        dk_full = torch.zeros_like(k_f)
        dv_full = torch.zeros_like(v_f)
        for start, end in _query_blocks(q, int(k_f.shape[-2])):
            q_block = q_f[..., start:end, :]
            do_block = do_f[..., start:end, :]
            scores = _block_scores(
                q_block,
                k_f,
                _rows(attn_mask, start, end),
                is_causal,
                start,
                scale_value,
                alibi_slopes,
            )
            probs = scores.sub_(_rows(stats_f, start, end)).exp_()
            row_dot = (do_block * o_f[..., start:end, :]).sum(dim=-1, keepdim=True)
            # d_scores = probs * (do @ v^T - row_dot) * scale. With dropout,
            # dP = (do @ v^T) * keep, D = row_dot * scale_inv and
            # dS = keep_scale * probs * (dP - D).
            d_scores = torch.matmul(do_block, v_f.transpose(-2, -1))
            if keep is not None:
                keep_block = _rows(keep, start, end)
                d_scores.mul_(keep_block)
                row_dot.mul_(scale_inv)
            d_scores.sub_(row_dot).mul_(probs).mul_(keep_scale * scale_value)
            dq[..., start:end, :] = torch.matmul(d_scores, k_f)
            dk_full += torch.matmul(d_scores.transpose(-2, -1), q_block)
            if keep is not None:
                probs.mul_(keep_block).mul_(keep_scale)
            dv_full += torch.matmul(probs.transpose(-2, -1), do_block)
            del scores, probs, d_scores

        batch, seq_kv = dk_full.shape[0], dk_full.shape[2]
        if rep_k > 1:
            dk_f = dk_full.view(batch, k_heads, rep_k, seq_kv, head_dim).sum(dim=2)
        else:
            dk_f = dk_full
        if rep_v > 1:
            dv_f = dv_full.view(batch, v_heads, rep_v, seq_kv, int(v.shape[-1])).sum(
                dim=2
            )
        else:
            dv_f = dv_full

        _store_tensor(tensors, dq_uid, dq.to(dtype=q.dtype))
        _store_tensor(tensors, dk_uid, dk_f.to(dtype=k.dtype))
        _store_tensor(tensors, dv_uid, dv_f.to(dtype=v.dtype))

    return run

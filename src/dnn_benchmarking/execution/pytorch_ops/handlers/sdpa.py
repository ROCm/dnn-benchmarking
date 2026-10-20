# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Scaled-dot-product-attention forward/backward reference handlers."""

from typing import Any, Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple

import torch

from ....common.exceptions import UnsupportedGraphError
from ....config.benchmark_config import PyTorchSdpaBackendName
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


class _SinkSdpa(NamedTuple):
    """Setup of a dense forward with sinks (manual fp32 attention)."""

    scale: float
    q_float: torch.Tensor
    k_t: torch.Tensor
    v_float: torch.Tensor
    mask: Optional[torch.Tensor]
    sink: torch.Tensor


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

    if _sdpa_bool(node, "alibi_mask") or _sdpa_bool(node, "padding_mask"):
        raise ValueError(
            "SDPA alibi/padding masks are not supported by the PyTorch reference"
        )

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
        dropout_p=dropout_p,
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
    dropout_p: float,
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
                dropout_p,
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
    return torch.logsumexp(_scores(q_float, k_float_t, mask, scale), -1, keepdim=True)


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
    """
    scores = _scores(q_float, k_float_t, mask, scale)
    log_denom = torch.logaddexp(
        torch.logsumexp(scores, dim=-1, keepdim=True), sink_logits
    )
    return torch.matmul(torch.exp(scores - log_denom), v_float), log_denom


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
    stats_uid = _optional_uid(node, "stats_tensor_uid")

    page_table_k_uid = _optional_uid(node, "page_table_k_tensor_uid")
    page_table_v_uid = _optional_uid(node, "page_table_v_tensor_uid")
    seq_len_q_uid = _optional_uid(node, "seq_len_q_tensor_uid")
    seq_len_kv_uid = _optional_uid(node, "seq_len_kv_tensor_uid")
    is_paged = page_table_k_uid is not None or page_table_v_uid is not None
    if is_paged:
        if page_table_k_uid is None or page_table_v_uid is None:
            raise ValueError("Paged SDPA needs both K and V page tables")
        if seq_len_kv_uid is None:
            raise ValueError("Paged SDPA needs seq_len_kv to bound each sequence")
        if stats_uid is not None:
            raise ValueError("Paged SDPA stats output is not supported")

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
                        dropout_p=dropout_p,
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

        head_dim = int(q.shape[-1])
        scale_value = float(scale)
        k_heads = int(k.shape[1])
        v_heads = int(v.shape[1])
        if rep_k > 1:
            k_f = k_f.repeat_interleave(rep_k, dim=1)
        if rep_v > 1:
            v_f = v_f.repeat_interleave(rep_v, dim=1)

        scores = torch.matmul(q_f, k_f.transpose(-2, -1)) * scale_value
        if attn_mask is not None:
            scores = scores + attn_mask.to(dtype=torch.float32)
        if is_causal:
            causal = torch.ones(
                scores.shape[-2],
                scores.shape[-1],
                dtype=torch.bool,
                device=scores.device,
            ).tril()
            scores = scores.masked_fill(~causal, float("-inf"))

        probs = torch.exp(scores - stats_f)
        row_dot = (do_f * o_f).sum(dim=-1, keepdim=True)
        d_probs = torch.matmul(do_f, v_f.transpose(-2, -1))
        d_scores = probs * (d_probs - row_dot)
        d_scores_scaled = d_scores * scale_value

        dq = torch.matmul(d_scores_scaled, k_f)
        dk_full = torch.matmul(d_scores_scaled.transpose(-2, -1), q_f)
        dv_full = torch.matmul(probs.transpose(-2, -1), do_f)

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

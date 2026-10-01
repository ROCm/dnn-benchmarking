# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Scaled-dot-product-attention forward/backward reference handlers."""

from math import sqrt
from typing import Any, Dict, Optional, Sequence, Tuple

import torch

from .._common import *  # noqa: F401,F403
from .._registry import CompiledOp, register_handler
from .._sdpa_backend import execute_selected_sdpa


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
) -> Optional[float]:
    if scale_uid is not None:
        return _scalar_value(tensors, scale_uid, node)
    return None if attn_scale_value is None else float(attn_scale_value)


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


def _plan_sdpa_common(
    node: Dict[str, Any],
    allow_paged: bool = False,
) -> Tuple[Optional[int], float, bool, Optional[int], Any, Optional[int]]:
    unsupported = [
        "seed_tensor_uid",
        "offset_tensor_uid",
        "dropout_mask_tensor_uid",
        "dropout_scale_tensor_uid",
        "block_mask_tensor_uid",
        "sink_token_tensor_uid",
        "descale_q_tensor_uid",
        "descale_k_tensor_uid",
        "descale_v_tensor_uid",
        "descale_s_tensor_uid",
        "scale_s_tensor_uid",
        "scale_o_tensor_uid",
    ]
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
    if _sdpa_bool(node, "causal_mask_bottom_right"):
        raise ValueError(
            "SDPA bottom-right causal mask is not supported by the PyTorch reference"
        )
    diagonal_alignment = _node_param(node, "diagonal_alignment", "TOP_LEFT")
    if diagonal_alignment not in ("TOP_LEFT", 0, None):
        raise ValueError("Only TOP_LEFT SDPA diagonal alignment is supported")

    dropout_probability = _node_param(node, "dropout_probability", 0.0)
    dropout_p = 0.0 if dropout_probability is None else float(dropout_probability)
    if dropout_p != 0.0:
        raise ValueError(
            "Nonzero SDPA dropout cannot be exactly validated against PyTorch"
        )

    mask_uid = _optional_uid(node, "attn_mask_tensor_uid")
    is_causal, window = _sdpa_derive_mask(node)
    if mask_uid is not None and (is_causal or window is not None):
        raise ValueError(
            "PyTorch SDPA reference does not support both attn_mask and causal_mask"
        )

    scale_uid = _optional_uid(node, "scale_tensor_uid")
    attn_scale_value = _node_param(node, "attn_scale_value", None)
    return mask_uid, dropout_p, is_causal, scale_uid, attn_scale_value, window


def _sdpa_derive_mask(node: Dict[str, Any]) -> Tuple[bool, Optional[int]]:
    """Resolve (is_causal, sliding_window_width) the way the engine does.

    Mirrors ``Gfx950AttentionTiledNative.cpp::maskTypeFor``. Two things there are
    easy to get wrong and both produce a wrong answer rather than an error:

      * **A real bound wins over the deprecated booleans.** They can only say
        top-left vs bottom-right, so a graph that sets ``causal_mask`` *and*
        carries ``left_bound`` is asking for a window; reading the boolean first
        silently discards it.
      * **Both spellings occur in this repo.** The shipped ``quick/SdpaFwd``
        bundles leave the booleans false and express causality as
        ``left_bound=-1, right_bound=0``, while the model traces set
        ``causal_mask: true``. Reading only one convention passes one population
        and mis-serves the other.

    The window WIDTH is ``left_bound + 1``: hipDNN's left bound counts tokens
    strictly before the current one, the band includes it.
    """
    unbounded = -1
    left = _node_param(node, "left_bound", unbounded)
    right = _node_param(node, "right_bound", unbounded)
    left = unbounded if left is None else int(left)
    right = unbounded if right is None else int(right)

    if left != unbounded:
        if left < 0:
            raise ValueError(f"SDPA left_bound {left} is neither unbounded nor a width")
        return False, left + 1

    if _sdpa_bool(node, "causal_mask"):
        return True, None
    if right == unbounded:
        return False, None
    if right == 0:
        return True, None
    raise ValueError(
        f"SDPA right_bound {right} is a forward-looking band the reference "
        "cannot express"
    )


def _sdpa_resolve(
    node: Dict[str, Any],
    tensors: Dict[int, torch.Tensor],
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask_uid: Optional[int],
    scale_uid: Optional[int],
    attn_scale_value: Any,
) -> Tuple[Optional[torch.Tensor], Optional[float], int, int]:
    attn_mask = _tensor(tensors, mask_uid, node) if mask_uid is not None else None
    scale = _sdpa_resolve_scale(node, tensors, scale_uid, attn_scale_value)
    if q.ndim < 3 or k.ndim < 3 or v.ndim < 3:
        raise ValueError("SDPA expects q/k/v tensors with head and matrix dimensions")
    q_heads = int(q.shape[-3])
    rep_k = _sdpa_head_repeat(q_heads, int(k.shape[-3]), "K")
    rep_v = _sdpa_head_repeat(q_heads, int(v.shape[-3]), "V")
    return attn_mask, scale, rep_k, rep_v


def _sliding_window_mask(
    q_len: int,
    kv_len: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Additive mask for a causal band of ``width`` tokens INCLUDING the current
    one, i.e. the kernel's ``q - W + 1 <= k <= q``. Rows are aligned top-left,
    matching the only diagonal alignment this reference accepts."""
    q_idx = torch.arange(q_len, device=device).unsqueeze(-1)
    k_idx = torch.arange(kv_len, device=device).unsqueeze(0)
    keep = (k_idx <= q_idx) & (k_idx > q_idx - width)
    mask = torch.zeros((q_len, kv_len), device=device, dtype=dtype)
    return mask.masked_fill(~keep, float("-inf"))


def _call_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: Optional[torch.Tensor],
    dropout_p: float,
    is_causal: bool,
    scale: Optional[float],
    rep_k: int,
    rep_v: int,
    window: Optional[int] = None,
) -> torch.Tensor:
    # Expand K and V independently to the query head count. PyTorch's
    # enable_gqa only models equal K/V head counts, so explicit repeat is the
    # only correct path when Hk != Hv.
    if rep_k > 1:
        k = k.repeat_interleave(rep_k, dim=-3)
    if rep_v > 1:
        v = v.repeat_interleave(rep_v, dim=-3)
    if window is not None:
        # A sliding window has no boolean spelling in torch's SDPA, so it is
        # expressed as the additive mask it actually is. is_causal is already
        # False here: a bounded left edge wins over the deprecated booleans.
        attn_mask = _sliding_window_mask(
            int(q.shape[-2]), int(k.shape[-2]), window, q.device, q.dtype
        )
    return execute_selected_sdpa(
        q,
        k,
        v,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
    )


def _run_paged_sdpa(
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
    scale: Optional[float],
    rep_k: int,
    rep_v: int,
    window: Optional[int],
) -> torch.Tensor:
    """Gather a paged KV cache to dense and run attention per sequence.

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
    page never contribute.
    """
    if attn_mask is not None:
        raise ValueError("Paged SDPA with an explicit attn_mask is not supported")

    page_table_k = _tensor(tensors, page_table_k_uid, node)
    page_table_v = _tensor(tensors, page_table_v_uid, node)
    seq_len_kv = _tensor(tensors, seq_len_kv_uid, node)
    seq_len_q = (
        _tensor(tensors, seq_len_q_uid, node) if seq_len_q_uid is not None else None
    )

    if k.ndim != 4 or v.ndim != 4:
        raise ValueError("Paged SDPA expects rank-4 paged K/V containers")
    page_size = int(k.shape[-2])
    num_seqs = int(seq_len_kv.numel())
    if int(page_table_k.shape[0]) != num_seqs:
        raise ValueError(
            f"Paged SDPA page table has {int(page_table_k.shape[0])} rows for "
            f"{num_seqs} sequences"
        )

    kv_lengths = [int(x) for x in seq_len_kv.flatten().tolist()]
    if seq_len_q is not None:
        q_lengths = [int(x) for x in seq_len_q.flatten().tolist()]
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

    outputs = []
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
        q_start += q_len

        outputs.append(
            _call_sdpa(
                q_seq,
                k_seq,
                v_seq,
                None,
                dropout_p,
                is_causal,
                scale,
                rep_k,
                rep_v,
                window,
            )
        )

    # Re-pack along the query axis in the order the sequences were laid out.
    return torch.cat(outputs, dim=-2)


def _sdpa_stats(
    q: torch.Tensor,
    k: torch.Tensor,
    attn_mask: Optional[torch.Tensor],
    is_causal: bool,
    scale: Optional[float],
    rep_k: int,
) -> torch.Tensor:
    q_float = q.to(dtype=torch.float32)
    k_float = k.to(dtype=torch.float32)
    if rep_k > 1:
        k_float = k_float.repeat_interleave(rep_k, dim=-3)
    scale_value = (1.0 / sqrt(float(q.shape[-1]))) if scale is None else scale
    scores = torch.matmul(q_float, k_float.transpose(-2, -1)) * scale_value
    if attn_mask is not None:
        scores = scores + attn_mask.to(dtype=torch.float32)
    if is_causal:
        length_q = scores.shape[-2]
        length_k = scores.shape[-1]
        causal = torch.ones(
            length_q,
            length_k,
            dtype=torch.bool,
            device=scores.device,
        ).tril()
        scores = scores.masked_fill(~causal, float("-inf"))
    return torch.logsumexp(scores, dim=-1, keepdim=True)


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
    and then run one sequence at a time. That gather is unavoidable and it is the
    same on both paths: ``--validate pytorch`` walks these very handlers with CPU
    tensors, so there is no reference-side shortcut.
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
    (
        mask_uid,
        dropout_p,
        is_causal,
        scale_uid,
        attn_scale_value,
        window,
    ) = _plan_sdpa_common(node, allow_paged=True)
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

    def run(tensors: Dict[int, torch.Tensor]) -> None:
        q = _tensor(tensors, q_uid, node)
        k = _tensor(tensors, k_uid, node)
        v = _tensor(tensors, v_uid, node)
        attn_mask, scale, rep_k, rep_v = _sdpa_resolve(
            node, tensors, q, k, v, mask_uid, scale_uid, attn_scale_value
        )
        if is_paged:
            o = _run_paged_sdpa(
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
                window,
            )
            _store_tensor(tensors, o_uid, o)
            return

        o = _call_sdpa(
            q, k, v, attn_mask, dropout_p, is_causal, scale, rep_k, rep_v, window
        )
        _store_tensor(tensors, o_uid, o)

        if stats_uid is not None:
            _store_tensor(
                tensors,
                stats_uid,
                _sdpa_stats(q, k, attn_mask, is_causal, scale, rep_k),
            )

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
    (
        mask_uid,
        _dropout_p,
        is_causal,
        scale_uid,
        attn_scale_value,
        window,
    ) = _plan_sdpa_common(node)
    if window is not None:
        raise ValueError(
            "SDPA sliding-window bounds are not supported by the PyTorch reference "
            "backward"
        )

    def run(tensors: Dict[int, torch.Tensor]) -> None:
        q = _tensor(tensors, q_uid, node)
        k = _tensor(tensors, k_uid, node)
        v = _tensor(tensors, v_uid, node)
        o = _tensor(tensors, o_uid, node)
        do = _tensor(tensors, do_uid, node)
        stats = _tensor(tensors, stats_uid, node)
        attn_mask, scale, rep_k, rep_v = _sdpa_resolve(
            node, tensors, q, k, v, mask_uid, scale_uid, attn_scale_value
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
        scale_value = (1.0 / sqrt(float(head_dim))) if scale is None else float(scale)
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

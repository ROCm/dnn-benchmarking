# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""FLOP handlers for scaled-dot-product attention (SdpaAttributes,
SdpaBackwardAttributes).

Matmul-only convention: FMA = 2 FLOPs; softmax/exp/scaling/sinks/dropout
are ignored. Only query heads count (GQA repeats KV heads, which adds no
arithmetic; dK/dV head reduction is an accumulate). With ``P`` the count of
unmasked ``(q, kv)`` pairs per query head:

* forward, two matmuls (``S = QKᵀ``, ``O = P·V``)::

      2 * batch * num_q_heads * P * (head_dim_qk + head_dim_vo)

* backward, five matmuls (recompute ``S = QKᵀ``; ``dV = Pᵀ·dO``;
  ``dP = dO·Vᵀ``; ``dQ = dS·K``; ``dK = dSᵀ·Q``)::

      2 * batch * num_q_heads * P * (3 * head_dim_qk + 2 * head_dim_vo)

  For ``head_dim_qk == head_dim_vo`` this is 2.5x forward, the convention
  FlashAttention's benchmarks use.

``P`` is exact. The mask is resolved the way hipDNN's reference executor and
providers resolve it (``extractDiagonalBandParams`` / ``getMaskType``):

* deprecated ``causal_mask`` -> top-left causal (bounds/alignment ignored);
* deprecated ``causal_mask_bottom_right`` -> bottom-right causal;
* otherwise ``left_bound`` / ``right_bound`` / ``diagonal_alignment``, where
  row ``i`` keeps ``kv`` in ``[i + off - left, i + off + right]`` with
  ``off = 0`` (TOP_LEFT) or ``Skv - Sq`` (BOTTOM_RIGHT); an unset bound is
  unbounded.

A causal sliding window of ``W`` keys (``left = W - 1``, ``right = 0``,
BOTTOM_RIGHT) therefore counts ``sum_i min(i + off + 1, W)`` pairs — the same
windowed count the rocKE attention benchmarks report.
"""

from typing import Any, Dict, Optional, Tuple

from ._common import node_param, node_tensor


def _nonmasked_pairs(
    q_seqlen: int,
    kv_seqlen: int,
    left: Optional[int],
    right: Optional[int],
    bottom_right: bool,
) -> int:
    """Count unmasked (q, kv) pairs for a diagonal band mask."""
    if left is None and right is None:
        return q_seqlen * kv_seqlen
    offset = kv_seqlen - q_seqlen if bottom_right else 0
    total = 0
    for i in range(q_seqlen):
        lo = max(i + offset - left, 0) if left is not None else 0
        hi = min(i + offset + right + 1, kv_seqlen) if right is not None else kv_seqlen
        total += max(hi - lo, 0)
    return total


def _attention_terms(
    node: Dict[str, Any], tensors_by_uid: Dict[int, Dict[str, Any]]
) -> Optional[Tuple[int, int, int]]:
    """Return ``(batch * num_q_heads * P, head_dim_qk, head_dim_vo)``.

    Returns None when q/k/v tensor data is incomplete, or when the mask is
    one hipDNN rejects (a bound below -1, or both deprecated causal flags).
    """
    q = node_tensor(node, "q_tensor_uid", tensors_by_uid)
    k = node_tensor(node, "k_tensor_uid", tensors_by_uid)
    v = node_tensor(node, "v_tensor_uid", tensors_by_uid)
    if not q or not k or not v:
        return None
    q_dims = q.get("dims") or []
    k_dims = k.get("dims") or []
    v_dims = v.get("dims") or []
    if len(q_dims) < 3 or len(k_dims) < 3 or len(v_dims) < 3:
        return None

    q_seqlen = int(q_dims[-2])
    kv_seqlen = int(k_dims[-2])
    rows = 1  # batch * num_q_heads
    for d in q_dims[:-2]:
        rows *= int(d)

    # Validate and resolve the mask in the order of hipDNN's
    # extractDiagonalBandParams.
    left = int(node_param(node, "left_bound", -1))
    right = int(node_param(node, "right_bound", -1))
    if left < -1 or right < -1:
        return None
    causal_top_left = node_param(node, "causal_mask") is True
    causal_bottom_right = node_param(node, "causal_mask_bottom_right") is True
    if causal_top_left and causal_bottom_right:
        return None
    if causal_top_left or causal_bottom_right:
        left, right, bottom_right = -1, 0, causal_bottom_right
    else:
        bottom_right = node_param(node, "diagonal_alignment") in ("BOTTOM_RIGHT", 1)

    pairs = _nonmasked_pairs(
        q_seqlen,
        kv_seqlen,
        None if left == -1 else left,
        None if right == -1 else right,
        bottom_right,
    )
    return rows * pairs, int(q_dims[-1]), int(v_dims[-1])


def sdpa_fwd_flops(
    node: Dict[str, Any], tensors_by_uid: Dict[int, Dict[str, Any]]
) -> Optional[int]:
    """FLOPs for SdpaAttributes: ``2 * B * Hq * P * (Dqk + Dvo)``."""
    terms = _attention_terms(node, tensors_by_uid)
    if terms is None:
        return None
    pairs, head_dim_qk, head_dim_vo = terms
    return 2 * pairs * (head_dim_qk + head_dim_vo)


def sdpa_bwd_flops(
    node: Dict[str, Any], tensors_by_uid: Dict[int, Dict[str, Any]]
) -> Optional[int]:
    """FLOPs for SdpaBackwardAttributes: ``2 * B * Hq * P * (3*Dqk + 2*Dvo)``."""
    terms = _attention_terms(node, tensors_by_uid)
    if terms is None:
        return None
    pairs, head_dim_qk, head_dim_vo = terms
    return 2 * pairs * (3 * head_dim_qk + 2 * head_dim_vo)

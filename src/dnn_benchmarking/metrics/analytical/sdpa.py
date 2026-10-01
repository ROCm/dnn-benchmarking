# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""FLOP handler for scaled-dot-product attention forward (SdpaAttributes).

Attention forward is two batched matmuls (``QKᵀ`` then ``P·V``); FMA = 2
FLOPs. Only query heads count (GQA repeats KV heads, which adds no
arithmetic); softmax/exp/scaling/sinks are ignored (matmul-only convention).

    flops = 2 * batch * num_q_heads * num_nonmasked * (head_dim_qk + head_dim_vo)

where ``num_nonmasked`` is the exact count of unmasked ``(q, kv)`` pairs per
query head. The mask is resolved the way hipDNN's reference executor and
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

from typing import Any, Dict, Optional


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


def sdpa_fwd_flops(
    node: Dict[str, Any], tensors_by_uid: Dict[int, Dict[str, Any]]
) -> Optional[int]:
    """FLOPs for SdpaAttributes (forward attention).

    Returns None (marking the graph partial) when q/k/v tensor data is
    incomplete, or when the mask is one hipDNN rejects (a bound below -1,
    or both deprecated causal flags set).
    """
    inputs = node.get("inputs", {}) or {}
    q_uid = inputs.get("q_tensor_uid")
    k_uid = inputs.get("k_tensor_uid")
    v_uid = inputs.get("v_tensor_uid")
    if q_uid is None or k_uid is None or v_uid is None:
        return None
    q = tensors_by_uid.get(int(q_uid))
    k = tensors_by_uid.get(int(k_uid))
    v = tensors_by_uid.get(int(v_uid))
    if not q or not k or not v:
        return None

    q_dims = q.get("dims") or []
    k_dims = k.get("dims") or []
    v_dims = v.get("dims") or []
    if len(q_dims) < 3 or len(k_dims) < 3 or len(v_dims) < 3:
        return None

    q_heads = int(q_dims[-3])
    q_seqlen = int(q_dims[-2])
    head_dim_qk = int(q_dims[-1])
    kv_seqlen = int(k_dims[-2])
    head_dim_vo = int(v_dims[-1])

    batch = 1
    for d in q_dims[:-3]:
        batch *= int(d)

    # Validate and resolve the mask in the order of hipDNN's
    # extractDiagonalBandParams; graphs hipDNN rejects get no count (None).
    attributes = node.get("attributes", {}) or {}
    left = attributes.get("left_bound")
    right = attributes.get("right_bound")
    left = -1 if left is None else int(left)
    right = -1 if right is None else int(right)
    if left < -1 or right < -1:
        return None
    causal_top_left = attributes.get("causal_mask") is True
    causal_bottom_right = attributes.get("causal_mask_bottom_right") is True
    if causal_top_left and causal_bottom_right:
        return None

    if causal_top_left or causal_bottom_right:
        left, right, bottom_right = -1, 0, causal_bottom_right
    else:
        bottom_right = attributes.get("diagonal_alignment") in ("BOTTOM_RIGHT", 1)

    num_nonmasked = _nonmasked_pairs(
        q_seqlen,
        kv_seqlen,
        None if left == -1 else left,
        None if right == -1 else right,
        bottom_right,
    )
    return 2 * batch * q_heads * num_nonmasked * (head_dim_qk + head_dim_vo)

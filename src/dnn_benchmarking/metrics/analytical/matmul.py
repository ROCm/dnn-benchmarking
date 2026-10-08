# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""FLOP handlers for matrix multiplication (MatmulAttributes) and MoE
grouped matmul (MoeGroupedMatmulAttributes / MoeGroupedMatmulBwdAttributes)."""

from typing import Any, Dict, Optional

from ._common import node_param, node_tensor


def matmul_flops(
    node: Dict[str, Any], tensors_by_uid: Dict[int, Dict[str, Any]]
) -> Optional[int]:
    """FLOPs for MatmulAttributes: ``2 * batch * M * N * K``.

    Supports batched matmul by multiplying all leading dims of the
    output tensor.
    """
    inputs = node.get("inputs", {}) or {}
    outputs = node.get("outputs", {}) or {}
    a_uid = inputs.get("a_tensor_uid")
    b_uid = inputs.get("b_tensor_uid")
    c_uid = outputs.get("c_tensor_uid")
    if a_uid is None or b_uid is None or c_uid is None:
        return None
    a = tensors_by_uid.get(int(a_uid))
    b = tensors_by_uid.get(int(b_uid))
    c = tensors_by_uid.get(int(c_uid))
    if not a or not b or not c:
        return None

    a_dims = a.get("dims") or []
    b_dims = b.get("dims") or []
    c_dims = c.get("dims") or []
    if len(a_dims) < 2 or len(b_dims) < 2 or len(c_dims) < 2:
        return None

    m = int(c_dims[-2])
    n = int(c_dims[-1])
    k = int(a_dims[-1])

    batch = 1
    for d in c_dims[:-2]:
        batch *= int(d)

    return 2 * batch * m * n * k


def moe_grouped_matmul_flops(
    node: Dict[str, Any], tensors_by_uid: Dict[int, Dict[str, Any]]
) -> Optional[int]:
    """FLOPs for MoeGroupedMatmulAttributes: ``2 * routed_rows * K * N``.

    Weight is ``[experts, K, N]``. Each routed row is one ``[1, K] x [K, N]``
    product against its expert. Following hipDNN's CPU reference, the routed
    rows are the token rows (``token`` dim 1) in NONE and SCATTER mode, and the
    ``token_index`` rows (dim 1) in GATHER mode. The count assumes the
    ``first_token_offset`` table routes every row, which is runtime data.
    """
    weight = node_tensor(node, "weight_tensor_uid", tensors_by_uid)
    rows_key = (
        "token_index_tensor_uid"
        if str(node_param(node, "mode", "")).upper() == "GATHER"
        else "token_tensor_uid"
    )
    rows_tensor = node_tensor(node, rows_key, tensors_by_uid)
    if not weight or not rows_tensor:
        return None
    w_dims = weight.get("dims") or []
    r_dims = rows_tensor.get("dims") or []
    if len(w_dims) != 3 or len(r_dims) < 2:
        return None
    return 2 * int(r_dims[1]) * int(w_dims[1]) * int(w_dims[2])


def moe_grouped_matmul_bwd_flops(
    node: Dict[str, Any], tensors_by_uid: Dict[int, Dict[str, Any]]
) -> Optional[int]:
    """FLOPs for MoeGroupedMatmulBwdAttributes: ``2 * token_rows * K * N``.

    The node computes only the weight gradient: each expert's
    ``dWeight[e] = Token[rows_e]ᵀ · dOutput[rows_e]``, and the expert row
    ranges partition the token rows (``token`` dim 1). ``dWeight`` is
    ``[experts, K, N]``.
    """
    token = node_tensor(node, "token_tensor_uid", tensors_by_uid)
    dweight = node_tensor(node, "dweight_tensor_uid", tensors_by_uid)
    if not token or not dweight:
        return None
    t_dims = token.get("dims") or []
    w_dims = dweight.get("dims") or []
    if len(t_dims) < 2 or len(w_dims) != 3:
        return None
    return 2 * int(t_dims[1]) * int(w_dims[1]) * int(w_dims[2])

# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Read one dense SDPA forward graph for a runtime that launches kernels directly.

The rocKE and AITER runtimes launch attention kernels on the graph tensors
without hipDNN. Both accept the same graph form: one ``SdpaAttributes`` node,
fp16 or bf16 Q/K/V/O in packed BSHD memory, and a mask that is either absent
or causal. Each runtime adds its own limits on top of :func:`dense_sdpa`.

This module needs neither torch nor a kernel library at import time.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

from ..common.exceptions import UnsupportedGraphError
from ..graph.tensor_info import TensorInfo

#: hipDNN data types that the direct runtimes launch.
DENSE_DTYPES = ("half", "bfloat16")

_UNSUPPORTED_INPUTS = (
    "attn_mask_tensor_uid",
    "scale_tensor_uid",
    "seq_len_q_tensor_uid",
    "seq_len_kv_tensor_uid",
    "seed_tensor_uid",
    "offset_tensor_uid",
    "dropout_mask_tensor_uid",
    "dropout_scale_tensor_uid",
    "page_table_k_tensor_uid",
    "page_table_v_tensor_uid",
    "block_mask_tensor_uid",
    "sink_token_tensor_uid",
    "descale_q_tensor_uid",
    "descale_k_tensor_uid",
    "descale_v_tensor_uid",
    "descale_s_tensor_uid",
    "scale_s_tensor_uid",
    "scale_o_tensor_uid",
)
_UNSUPPORTED_OUTPUTS = (
    "stats_tensor_uid",
    "max_tensor_uid",
    "sum_exp_tensor_uid",
    "rng_dump_tensor_uid",
    "amax_s_tensor_uid",
    "amax_o_tensor_uid",
)


@dataclass(frozen=True)
class DenseSdpa:
    """One dense SDPA forward problem read from a graph.

    Q is ``[batch, num_query_heads, seqlen_q, head_size_qk]``, K is
    ``[batch, num_kv_heads, seqlen_kv, head_size_qk]``, V is
    ``[batch, num_kv_heads, seqlen_kv, head_size_v]`` and O is
    ``[batch, num_query_heads, seqlen_q, head_size_v]`` in graph order. Their
    memory layout is packed BSHD.

    ``causal`` is a top-left causal mask and ``bottom_right`` a bottom-right
    causal mask; at most one of them is set.
    """

    data_type: str
    batch: int
    num_query_heads: int
    num_kv_heads: int
    seqlen_q: int
    seqlen_kv: int
    head_size_qk: int
    head_size_v: int
    causal: bool
    bottom_right: bool
    scale: float
    q_uid: int
    k_uid: int
    v_uid: int
    o_uid: int


def is_packed_bshd(dims: Sequence[int], strides: Sequence[int]) -> bool:
    """Whether ``[B, H, S, D]`` (graph order) is packed BSHD in memory.

    The stride of a size-1 dimension never addresses memory, so it is free.
    """
    _, heads, seq, head_size = dims
    packed = (seq * heads * head_size, head_size, heads * head_size, 1)
    return all(n == 1 or s == p for n, s, p in zip(dims, strides, packed))


def dense_sdpa(
    graph_json: Dict[str, Any], tensor_infos: List[TensorInfo], runtime: str
) -> DenseSdpa:
    """Read the dense SDPA forward problem from a graph.

    Args:
        graph_json: The hipDNN graph.
        tensor_infos: The graph's tensors.
        runtime: Runtime label for the error messages, for example ``rocKE``.

    Raises:
        UnsupportedGraphError: The graph is not one dense SDPA forward node
            in the form described in the module docstring.
    """
    from .pytorch_ops._common import _node_param, _optional_uid
    from .pytorch_ops.handlers.sdpa import _sdpa_derive_mask

    nodes = graph_json.get("nodes") or []
    if len(nodes) != 1 or nodes[0].get("type") != "SdpaAttributes":
        raise UnsupportedGraphError(
            f"{runtime} runtime runs graphs with one SdpaAttributes node only"
        )
    node = nodes[0]

    present = [
        key
        for key in _UNSUPPORTED_INPUTS + _UNSUPPORTED_OUTPUTS
        if _optional_uid(node, key) is not None
    ]
    for flag in ("alibi_mask", "padding_mask"):
        if _node_param(node, flag, False):
            present.append(flag)
    if _node_param(node, "generate_stats", False):
        present.append("generate_stats")
    if float(_node_param(node, "dropout_probability", 0.0) or 0.0) != 0.0:
        present.append("dropout_probability")
    if present:
        raise UnsupportedGraphError(
            f"{runtime} runtime does not support SDPA features: {', '.join(present)}"
        )
    # The mask resolves as hipDNN resolves it: the deprecated causal booleans
    # win over the bounds and the alignment.
    try:
        causal, window, bottom_right = _sdpa_derive_mask(node)
    except ValueError as e:
        raise UnsupportedGraphError(str(e)) from e
    if window is not None:
        raise UnsupportedGraphError(
            f"{runtime} runtime does not support sliding windows"
        )

    infos = {info.uid: info for info in tensor_infos}
    uids = {}
    for role, key in (
        ("q", "q_tensor_uid"),
        ("k", "k_tensor_uid"),
        ("v", "v_tensor_uid"),
        ("o", "o_tensor_uid"),
    ):
        uid = _optional_uid(node, key)
        if uid is None or uid not in infos:
            raise UnsupportedGraphError(f"SDPA node has no {role} tensor")
        uids[role] = uid
    q, k, v, o = (infos[uids[r]] for r in ("q", "k", "v", "o"))

    dtypes = {t.data_type for t in (q, k, v, o)}
    if len(dtypes) != 1 or next(iter(dtypes)) not in DENSE_DTYPES:
        raise UnsupportedGraphError(
            f"{runtime} runtime needs one fp16 or bf16 dtype for Q/K/V/O, "
            f"got {sorted(dtypes)}"
        )
    if any(len(t.dims) != 4 for t in (q, k, v, o)):
        raise UnsupportedGraphError(f"{runtime} runtime needs 4-D Q/K/V/O tensors")
    batch, hq, sq, dqk = q.dims
    _, hkv, skv, dk = k.dims
    dv = v.dims[3]
    if (
        dk != dqk
        or list(v.dims[:3]) != list(k.dims[:3])
        or list(o.dims) != [batch, hq, sq, dv]
    ):
        raise UnsupportedGraphError(
            f"{runtime} runtime needs K with Q's head size, V shaped like K except "
            "the head size, and O shaped like Q with V's head size"
        )
    if k.dims[0] != batch or hkv <= 0 or hq % hkv != 0:
        raise UnsupportedGraphError(
            f"{runtime} runtime needs matching batch and Hq divisible by Hkv "
            f"(Hq={hq}, Hkv={hkv})"
        )
    for t in (q, k, v, o):
        if not is_packed_bshd(t.dims, t.strides):
            raise UnsupportedGraphError(
                f"{runtime} runtime needs packed BSHD tensors; '{t.name}' has "
                f"strides {list(t.strides)}"
            )

    scale = _node_param(node, "attn_scale_value", None)
    return DenseSdpa(
        data_type=q.data_type,
        batch=batch,
        num_query_heads=hq,
        num_kv_heads=hkv,
        seqlen_q=sq,
        seqlen_kv=skv,
        head_size_qk=dqk,
        head_size_v=dv,
        causal=causal,
        bottom_right=bottom_right,
        # hipDNN reads an absent attn_scale as 1.0 (cuDNN semantics).
        scale=1.0 if scale is None else float(scale),
        q_uid=uids["q"],
        k_uid=uids["k"],
        v_uid=uids["v"],
        o_uid=uids["o"],
    )

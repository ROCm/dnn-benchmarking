# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Static warnings describing manual (non-built-in) PyTorch reference paths."""

from typing import Any, Dict, List

import torch.nn.functional as F

from ._common import (
    _effective_compute_type,
    _is_float32_compute,
    _node_param,
    _node_uid,
    _tensor_shape,
)
from .handlers.norm import _rmsnorm_graph_can_use_builtin
from .handlers.reduction import _reduction_mode_name
from .handlers.resample import _resample_has_asymmetric_padding, _resample_mode_name
from .handlers.sdpa import _sdpa_derive_mask


def _sdpa_forward_workarounds(
    node: Dict[str, Any], graph_json: Dict[str, Any]
) -> List[str]:
    """SDPA forward features PyTorch's SDPA cannot express natively."""

    def uid(key: str) -> Any:
        return _node_uid(node, key, ("inputs",), required=False)

    reasons: List[str] = []
    if uid("sink_token_tensor_uid") is not None:
        reasons.append(
            "PyTorch SDPA has no attention-sink input, so attention runs as "
            "separate fp32 matmul, softmax and matmul ops instead of an SDPA kernel"
        )
    if uid("page_table_k_tensor_uid") is not None:
        reasons.append(
            "PyTorch has no paged SDPA, so K/V are gathered into dense "
            "per-sequence tensors once and each sequence runs as its own SDPA "
            "call, re-packed with torch.cat"
        )
    try:
        _, window, bottom_right = _sdpa_derive_mask(node)
    except ValueError:
        window, bottom_right = None, False
    if window is not None:
        reasons.append(
            "PyTorch SDPA has no sliding-window option, so the window is passed "
            "as an additive mask tensor that the kernel reads"
        )
    elif bottom_right:
        reasons.append(
            "the bottom-right causal mask is passed as an additive mask tensor "
            "that the kernel reads (is_causal is top-left only)"
        )
    k_shape = _tensor_shape(graph_json, uid("k_tensor_uid"))
    v_shape = _tensor_shape(graph_json, uid("v_tensor_uid"))
    if k_shape and v_shape and len(k_shape) >= 3 and len(v_shape) >= 3:
        if k_shape[-3] != v_shape[-3]:
            reasons.append(
                "K and V head counts differ, which enable_gqa cannot express, so "
                "K/V are expanded to the query head count once and the kernel "
                "reads the expanded tensors"
            )
    return reasons


def get_reference_warnings(graph_json: Dict[str, Any]) -> List[str]:
    """Describe manual/non-built-in portions of the PyTorch reference graph.

    The timed PyTorch reference row is useful as a baseline only when users can
    tell whether it is measuring a public PyTorch primitive or local reference
    glue.  This helper is intentionally static: it only inspects graph metadata
    and reports paths whose handler is not solely a built-in PyTorch operator.
    """

    warnings: List[str] = []
    for node in graph_json.get("nodes", []):
        op_type = str(node.get("type", ""))
        name = str(node.get("name") or op_type)

        if not _is_float32_compute(node, graph_json):
            cdt = _effective_compute_type(node, graph_json)
            warnings.append(
                f"{name}: graph compute_data_type={cdt!r} is not float32; the PyTorch "
                f"reference assumes float32 accumulation and does not honor other compute "
                f"types, so its results and timing may diverge from the graph's execution."
            )

        if op_type == "RMSNormAttributes":
            reasons: List[str] = []
            if not hasattr(F, "rms_norm"):
                reasons.append("torch.nn.functional.rms_norm is unavailable")
            if (
                _node_uid(node, "bias_tensor_uid", ("inputs",), required=False)
                is not None
            ):
                reasons.append("optional bias is applied manually")
            if (
                _node_uid(node, "inv_rms_tensor_uid", ("outputs",), required=False)
                is not None
            ):
                reasons.append("requested inv_rms output is computed manually")
            if not _rmsnorm_graph_can_use_builtin(node, graph_json):
                reasons.append("per-channel layout uses a manual RMSNorm formula")
            if reasons:
                warnings.append(
                    f"{name}: RMSNormAttributes includes manual reference work "
                    f"({'; '.join(dict.fromkeys(reasons))}); PyTorch reference timing "
                    "is not solely built-in PyTorch operator time."
                )

        elif op_type == "RMSNormBackwardAttributes":
            warnings.append(
                f"{name}: RMSNormBackwardAttributes uses a manual RMSNorm backward "
                "formula because PyTorch has no public operator matching hipDNN's "
                "saved-inv_rms backward node; PyTorch reference timing is not solely "
                "built-in PyTorch operator time."
            )

        elif op_type == "SdpaBackwardAttributes":
            warnings.append(
                f"{name}: SdpaBackwardAttributes uses a manual flash-attention "
                "backward formula that consumes hipDNN's saved stats (log-sum-exp); "
                "torch.nn.functional.scaled_dot_product_attention autograd cannot "
                "consume external stats, so PyTorch reference timing is not solely "
                "built-in PyTorch operator time."
            )

        elif op_type == "SdpaAttributes":
            reasons = _sdpa_forward_workarounds(node, graph_json)
            if reasons:
                warnings.append(
                    f"{name}: SdpaAttributes includes manual reference work "
                    f"({'; '.join(reasons)}); PyTorch reference timing is not "
                    "solely built-in PyTorch operator time."
                )

        elif op_type == "ReductionAttributes":
            if (
                _reduction_mode_name(_node_param(node, "mode", "NOT_SET"))
                == "MUL_NO_ZEROS"
            ):
                warnings.append(
                    f"{name}: ReductionAttributes mode MUL_NO_ZEROS uses a manual "
                    "masked product; PyTorch reference timing is not solely built-in "
                    "PyTorch operator time."
                )

        elif op_type == "ResampleFwdAttributes":
            mode = _resample_mode_name(_node_param(node, "resample_mode", "NOT_SET"))
            if mode == "AVGPOOL_EXCLUDE_PADDING" and _resample_has_asymmetric_padding(
                node, graph_json
            ):
                warnings.append(
                    f"{name}: ResampleFwdAttributes AVGPOOL_EXCLUDE_PADDING with "
                    "asymmetric padding uses manual valid-count correction around "
                    "torch.nn.functional.avg_pool; PyTorch reference timing is not "
                    "solely built-in PyTorch operator time."
                )

    return warnings

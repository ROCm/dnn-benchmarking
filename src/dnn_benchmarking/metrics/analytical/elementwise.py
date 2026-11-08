# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""FLOP handlers for element-wise ops driven by output element count.

Pointwise (relu, add, mul, …), Rng, and block-scale quantize/dequantize
all do O(num_output_elements) work and are dominated by memory traffic.
We don't distinguish unary vs binary pointwise because the FLOP
component is small relative to fused-graph totals.
"""

from typing import Any, Dict, Optional

from ._common import output_elements, tensor_dim_product


def pointwise_flops(
    node: Dict[str, Any], tensors_by_uid: Dict[int, Dict[str, Any]]
) -> Optional[int]:
    """PointwiseAttributes: 1 FLOP per output element.

    Element-wise operations (relu, add, mul, sub, div, abs, neg, exp,
    log, tanh, sigmoid, sqrt) all do O(num_elements) work.
    """
    outputs = node.get("outputs", {}) or {}
    out_uid = outputs.get("out_0_tensor_uid")
    if out_uid is None:
        return None
    out = tensors_by_uid.get(int(out_uid))
    if not out:
        return None
    return tensor_dim_product(out)


def rng_flops(
    node: Dict[str, Any], tensors_by_uid: Dict[int, Dict[str, Any]]
) -> Optional[int]:
    """Rng: 1 op per generated value (conservative — most PRNGs do more)."""
    outputs = node.get("outputs", {}) or {}
    out_uid = outputs.get("out_0_tensor_uid")
    if out_uid is None:
        out_uid = outputs.get("y_tensor_uid")
    if out_uid is None:
        return None
    tensor = tensors_by_uid.get(int(out_uid))
    if not tensor:
        return None
    return tensor_dim_product(tensor)


def block_scale_quantize_flops(
    node: Dict[str, Any], tensors_by_uid: Dict[int, Dict[str, Any]]
) -> Optional[int]:
    """BlockScaleQuantize: 2 ops per element.

    One for the per-block abs-max reduction and one to apply the scale.
    The per-block scale computation is one op per ``block_size`` elements
    and is ignored.
    """
    elems = output_elements(node, tensors_by_uid)
    return 2 * elems if elems is not None else None


def block_scale_dequantize_flops(
    node: Dict[str, Any], tensors_by_uid: Dict[int, Dict[str, Any]]
) -> Optional[int]:
    """BlockScaleDequantize: 1 op per element (multiply by the block scale)."""
    return output_elements(node, tensors_by_uid)

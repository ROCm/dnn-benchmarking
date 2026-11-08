# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""FLOP handlers for reductions: ReductionAttributes and windowed
resampling (ResampleFwdAttributes / ResampleBwdAttributes, i.e. pooling).

Plain reduction work scales with the input element count (the output is
typically a scalar or row), so it is input-driven instead of
output-driven. Pooling reduces one window per output element.
"""

from typing import Any, Dict, Optional

from ._common import node_param, node_tensor, output_elements, tensor_dim_product


def reduction_flops(
    node: Dict[str, Any], tensors_by_uid: Dict[int, Dict[str, Any]]
) -> Optional[int]:
    """Reduction: 1 op/elem of the *input* tensor."""
    tensor = node_tensor(node, "in_tensor_uid", tensors_by_uid)
    return tensor_dim_product(tensor) if tensor else None


def _window_size(node: Dict[str, Any]) -> Optional[int]:
    window = node_param(node, "window")
    if not window:
        return None
    size = 1
    for w in window:
        size *= int(w)
    return size


def resample_fwd_flops(
    node: Dict[str, Any], tensors_by_uid: Dict[int, Dict[str, Any]]
) -> Optional[int]:
    """ResampleFwd (max / avg pooling): ``window`` ops per output element.

    Each output reduces one window: a compare per element for max pooling,
    an add per element for average pooling (the final divide is ignored).
    """
    elems = output_elements(node, tensors_by_uid)
    window = _window_size(node)
    if elems is None or window is None:
        return None
    return elems * window


def resample_bwd_flops(
    node: Dict[str, Any], tensors_by_uid: Dict[int, Dict[str, Any]]
) -> Optional[int]:
    """ResampleBwd: per ``dy`` element, 1 op (max) or ``window`` ops (avg).

    Max pooling routes each ``dy`` to one argmax position (one accumulate).
    Average pooling spreads each ``dy`` over its whole window.
    """
    dy = node_tensor(node, "dy_tensor_uid", tensors_by_uid)
    if not dy:
        return None
    elems = tensor_dim_product(dy)
    if str(node_param(node, "resample_mode", "")).upper() == "MAXPOOL":
        return elems
    window = _window_size(node)
    return elems * window if window is not None else None

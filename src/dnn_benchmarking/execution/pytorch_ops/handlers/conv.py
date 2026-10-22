# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Convolution forward/backward/wgrad reference handlers."""

from typing import Any, Callable, Dict, Sequence, Tuple

import torch
import torch.nn.functional as F

from .._common import *  # noqa: F401,F403
from .._registry import CompiledOp, register_handler


def _validate_cross_correlation(node: Dict[str, Any]) -> None:
    conv_mode = _node_param(node, "conv_mode", "CROSS_CORRELATION")
    if conv_mode != "CROSS_CORRELATION":
        raise ValueError(
            f"Unsupported convolution mode {conv_mode!r}; PyTorch reference only supports CROSS_CORRELATION"
        )


def _conv_padding(node: Dict[str, Any]) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    pre = tuple(_as_tuple(_node_param(node, "pre_padding", [0, 0]), [0, 0]))
    post = tuple(_as_tuple(_node_param(node, "post_padding", pre), pre))
    if len(pre) != len(post):
        raise ValueError("Convolution pre/post padding must have equal rank")
    return pre, post


def _conv_stride_dilation(
    node: Dict[str, Any],
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    stride = tuple(_as_tuple(_node_param(node, "stride", [1, 1]), [1, 1]))
    dilation = tuple(_as_tuple(_node_param(node, "dilation", [1, 1]), [1, 1]))
    if len(stride) != len(dilation):
        raise ValueError("Convolution stride/dilation must have equal rank")
    return stride, dilation


def _conv_group_count(input_shape: Sequence[int], weight_shape: Sequence[int]) -> int:
    """Infer grouped convolution count from hipDNN tensor shapes."""
    if len(input_shape) < 2:
        raise ValueError(
            "Convolution input tensor must have at least 2 dimensions, "
            f"got {len(input_shape)}"
        )
    if len(weight_shape) < 2:
        raise ValueError(
            "Convolution weight tensor must have at least 2 dimensions, "
            f"got {len(weight_shape)}"
        )

    input_channels = int(input_shape[1])
    weight_channels_per_group = int(weight_shape[1])
    output_channels = int(weight_shape[0])
    if input_channels <= 0:
        raise ValueError(
            f"Convolution input channels must be positive, got {input_channels}"
        )
    if weight_channels_per_group <= 0:
        raise ValueError(
            "Convolution weight channels per group must be positive, "
            f"got {weight_channels_per_group}"
        )
    if output_channels <= 0:
        raise ValueError(
            f"Convolution weight output channels must be positive, got {output_channels}"
        )
    if input_channels % weight_channels_per_group != 0:
        raise ValueError(
            f"Convolution input channels ({input_channels}) must be evenly divisible "
            f"by weight channels per group ({weight_channels_per_group})"
        )

    groups = input_channels // weight_channels_per_group
    if output_channels % groups != 0:
        raise ValueError(
            f"Convolution weight output channels ({output_channels}) must be evenly "
            f"divisible by inferred group count ({groups})"
        )
    return groups


def _pad_conv_input(
    x: torch.Tensor, pre: Tuple[int, ...], post: Tuple[int, ...]
) -> torch.Tensor:
    if all(p == 0 for p in pre) and all(p == 0 for p in post):
        return x
    pad = []
    for i in range(len(pre) - 1, -1, -1):
        pad.extend((pre[i], post[i]))
    return F.pad(x, pad)


_CONV_FORWARD_FNS = {1: F.conv1d, 2: F.conv2d, 3: F.conv3d}


def _plan_conv_forward(
    node: Dict[str, Any],
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Parse a forward-conv node once and return a closure applying it to (x, w)."""
    _validate_cross_correlation(node)
    pre, post = _conv_padding(node)
    stride, dilation = _conv_stride_dilation(node)
    conv_fn = _CONV_FORWARD_FNS.get(len(stride))
    if conv_fn is None:
        raise ValueError(
            f"Unsupported convolution spatial rank {len(stride)}; "
            "PyTorch reference supports 1D/2D/3D"
        )
    symmetric = pre == post

    def apply(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        groups = _conv_group_count(x.shape, w.shape)
        if symmetric:
            # Symmetric padding folds into the conv, matching the engine's native
            # descriptor (no separate F.pad kernel in the timed window).
            return conv_fn(
                x, w, stride=stride, padding=pre, dilation=dilation, groups=groups
            )
        # Asymmetric padding can't be expressed via conv padding; pre-pad explicitly.
        padded_x = _pad_conv_input(x, pre, post)
        return conv_fn(padded_x, w, stride=stride, dilation=dilation, groups=groups)

    return apply


@register_handler("ConvolutionFwdAttributes")
def compile_conv_fwd(
    node: Dict[str, Any],
    graph_json: Dict[str, Any],
) -> CompiledOp:
    """Plan ConvolutionFwdAttributes (1D/2D/3D convolution forward pass)."""
    x_uid = _required_input_uid(node, "x_tensor_uid")
    w_uid = _required_input_uid(node, "w_tensor_uid")
    y_uid = _required_output_uid(node, "y_tensor_uid")
    apply_conv = _plan_conv_forward(node)

    def run(tensors: Dict[int, torch.Tensor]) -> None:
        y = apply_conv(_tensor(tensors, x_uid, node), _tensor(tensors, w_uid, node))
        _store_tensor(tensors, y_uid, y)

    return run


def _graph_dim_order(
    graph_json: Dict[str, Any], uid: int, rank: int
) -> Tuple[int, ...]:
    """Return the dims of graph tensor ``uid`` from outermost to innermost.

    The order comes from the graph strides. A tensor without strides uses the
    logical order.
    """
    for tensor_json in graph_json.get("tensors", []):
        if tensor_json.get("uid") == uid:
            strides = [int(s) for s in tensor_json.get("strides") or []]
            if len(strides) == rank:
                # Stable sort: dims with equal strides keep their logical order.
                return tuple(sorted(range(rank), key=lambda d: -strides[d]))
            break
    return tuple(range(rank))


def _layout_placeholder(
    size: Sequence[int], dim_order: Sequence[int], like: torch.Tensor
) -> torch.Tensor:
    """Allocate an uninitialized tensor of ``size`` that is dense in ``dim_order``.

    ``aten.convolution_backward`` reads only the shape and the memory layout of
    the operand that it does not differentiate. It calls ``.contiguous()`` on
    that operand. A dense placeholder passes that call without a kernel. The
    zero-stride placeholder of ``torch.nn.grad.conv*_input`` and
    ``conv*_weight`` does not: each call copies a tensor of the operand size.
    """
    physical = like.new_empty([size[d] for d in dim_order])
    logical_to_physical = [0] * len(dim_order)
    for position, dim in enumerate(dim_order):
        logical_to_physical[dim] = position
    return physical.permute(logical_to_physical)


def _conv_backward(
    grad_output: torch.Tensor,
    x: torch.Tensor,
    w: torch.Tensor,
    stride: Tuple[int, ...],
    padding: Tuple[int, ...],
    dilation: Tuple[int, ...],
    groups: int,
    output_mask: Tuple[bool, bool, bool],
) -> Tuple[torch.Tensor, ...]:
    return torch.ops.aten.convolution_backward(
        grad_output,
        x,
        w,
        None,
        stride,
        padding,
        dilation,
        False,
        [0] * len(stride),
        groups,
        output_mask,
    )


@register_handler("ConvolutionBwdAttributes")
def compile_conv_bwd(
    node: Dict[str, Any],
    graph_json: Dict[str, Any],
) -> CompiledOp:
    """Plan ConvolutionBwdAttributes (gradient with respect to input)."""
    _validate_cross_correlation(node)
    dy_uid = _required_input_uid(node, "dy_tensor_uid")
    w_uid = _required_input_uid(node, "w_tensor_uid")
    dx_uid = _required_output_uid(node, "dx_tensor_uid")

    input_size = _tensor_shape(graph_json, dx_uid)
    if input_size is None:
        raise ValueError(
            f"ConvolutionBwdAttributes missing dx tensor shape for UID {dx_uid}"
        )

    stride, dilation = _conv_stride_dilation(node)
    pre, post = _conv_padding(node)
    if pre == post:
        # Symmetric padding folds into the conv, as in the engine descriptor.
        x_size, padding, crop = input_size, pre, None
    else:
        # convolution_backward has no asymmetric padding. Compute the gradient
        # of the padded input, then keep a view of the unpadded part (no kernel).
        x_size = tuple(input_size[:2]) + tuple(
            dim + before + after
            for dim, before, after in zip(input_size[2:], pre, post)
        )
        padding = (0,) * len(stride)
        crop = (slice(None), slice(None)) + tuple(
            slice(before, before + dim) for dim, before in zip(input_size[2:], pre)
        )
    dim_order = _graph_dim_order(graph_json, dx_uid, len(input_size))

    def run(tensors: Dict[int, torch.Tensor]) -> None:
        dy = _tensor(tensors, dy_uid, node)
        w = _tensor(tensors, w_uid, node)
        groups = _conv_group_count(input_size, w.shape)
        x = _layout_placeholder(x_size, dim_order, dy)
        dx = _conv_backward(
            dy, x, w, stride, padding, dilation, groups, (True, False, False)
        )[0]
        if crop is not None:
            dx = dx[crop]
        _store_tensor(tensors, dx_uid, dx)

    return run


@register_handler("ConvolutionWrwAttributes")
def compile_conv_wrw(
    node: Dict[str, Any],
    graph_json: Dict[str, Any],
) -> CompiledOp:
    """Plan ConvolutionWrwAttributes (gradient with respect to weights)."""
    _validate_cross_correlation(node)
    x_uid = _required_input_uid(node, "x_tensor_uid")
    dy_uid = _required_input_uid(node, "dy_tensor_uid")
    dw_uid = _required_output_uid(node, "dw_tensor_uid")

    weight_size = _tensor_shape(graph_json, dw_uid)
    if weight_size is None:
        raise ValueError(
            f"ConvolutionWrwAttributes missing dw tensor shape for UID {dw_uid}"
        )

    stride, dilation = _conv_stride_dilation(node)
    pre, post = _conv_padding(node)
    symmetric = pre == post
    # Symmetric padding folds into the conv. convolution_backward has no
    # asymmetric padding, so pad x explicitly, as the forward handler does.
    padding = pre if symmetric else (0,) * len(stride)
    dim_order = _graph_dim_order(graph_json, dw_uid, len(weight_size))

    def run(tensors: Dict[int, torch.Tensor]) -> None:
        x = _tensor(tensors, x_uid, node)
        dy = _tensor(tensors, dy_uid, node)
        groups = _conv_group_count(x.shape, weight_size)
        if not symmetric:
            x = _pad_conv_input(x, pre, post)
        w = _layout_placeholder(weight_size, dim_order, x)
        dw = _conv_backward(
            dy, x, w, stride, padding, dilation, groups, (False, True, False)
        )[1]
        _store_tensor(tensors, dw_uid, dw)

    return run

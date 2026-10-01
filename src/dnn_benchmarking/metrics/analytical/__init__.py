# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Analytical FLOPs and I/O byte computation from graph JSON.

Every recognised op contributes a real arithmetic FLOP count, and the
caller always also receives ``analytical_io_bytes`` and
``derived_gbytes_per_s``. Reporting both signals lets the consumer
decide which is the dominant constraint for each op type — a low
TFLOPs/s number for a memory-bound kernel is informative when paired
with a high GB/s number, in the same way NVIDIA Nsight Compute exposes
Compute Throughput and Memory Throughput as independent percentages of
peak. We deliberately do *not* mirror MIOpen's ``bn_driver.hpp``
``flopCnt = 0`` choice (which then mislabels a bandwidth metric as
"GFLOPs"); the precedent we follow is Composable Kernel, whose
example/profiling code reports honest arithmetic FLOPs alongside GB/s
for the same kernel.

Per-op formulas (FMA = 2 FLOPs throughout):

* Conv 1D / 2D / 3D fwd / dgrad / wgrad:
  ``2 * N * (C_in / group) * R * S * K * H_out * W_out``
  — see :mod:`.conv`. Matches MIOpen's ``conv_driver.hpp`` and the
  rocKE conv benchmarks; ``C_in / group`` comes from the weight dims.
* SDPA fwd: ``2 * B * H_q * unmasked_pairs * (D_qk + D_vo)``; SDPA bwd:
  ``2 * B * H_q * unmasked_pairs * (3 * D_qk + 2 * D_vo)`` (2.5x fwd for
  equal head dims, FlashAttention's convention) — see :mod:`.sdpa`.
  Exact causal / sliding-window pair counts using hipDNN's mask semantics.
* GEMM: ``2 * batch * M * N * K`` — see :mod:`.matmul`. Standard
  textbook formula (e.g. NVIDIA's perf model docs). MoE grouped matmul
  fwd: ``2 * routed_rows * K * N``; bwd (weight gradient only):
  ``2 * token_rows * K * N``.
* Pointwise (relu, add, mul, …): ``num_output_elements`` (1 op/elem)
  — see :mod:`.elementwise`.
* Rng: ``num_output_elements`` (conservative; most PRNGs do more
  per draw) — see :mod:`.elementwise`.
* Block-scale quantize: ``2 * num_elements`` (block abs-max + scale);
  dequantize: ``num_elements`` (scale multiply) — see :mod:`.elementwise`.
* BatchNorm inference (including ``VarianceExt``): ``4 *
  num_output_elements`` (subtract mean, multiply by inv_var, multiply by
  scale, add bias). Training fwd: ``8 * num_output_elements`` (above
  plus mean / variance reductions). BatchNorm / LayerNorm / RMSNorm bwd:
  ``8 * num_dx_elements``. LayerNorm / RMSNorm fwd: ``8 * num_output_elements``.
  SoftMax fwd: ``4 * num_output_elements`` (max, exp, sum, divide).
  See :mod:`.normalization`. Multipliers follow Composable Kernel's
  norm benchmarks and PyTorch's profiler conventions.
* Reduction (sum/mean/etc.): ``num_input_elements`` (1 op/elem on
  the input — output is typically scalar/row). See :mod:`.reduction`.
* Resample (pooling) fwd: ``num_output_elements * window``; bwd:
  ``num_dy_elements`` for max pooling, ``num_dy_elements * window`` for
  average pooling. See :mod:`.reduction`.

Per-op FLOP / IO handlers live in this directory split by op family:
``conv.py``, ``matmul.py``, ``elementwise.py``, ``normalization.py``,
``reduction.py``. The dispatch table is the single source of truth
for which node types we recognise.

When a graph contains a node type this module does not recognise, the
``partial`` flag in :func:`compute_flops` is set so callers can label
the value as incomplete, and a one-shot warning surfaces the unknown
type via :mod:`.._diagnostic.warn_once`.
"""

from typing import Any, Dict, Iterable, List, Optional, Tuple

from ...graph.tensor_info import TensorInfo
from .._diagnostic import warn_once
from ._common import tensor_lookup
from .conv import conv_dgrad_flops, conv_fwd_flops, conv_wgrad_flops
from .elementwise import (
    block_scale_dequantize_flops,
    block_scale_quantize_flops,
    pointwise_flops,
    rng_flops,
)
from .matmul import matmul_flops, moe_grouped_matmul_bwd_flops, moe_grouped_matmul_flops
from .normalization import (
    batchnorm_inference_flops,
    batchnorm_training_flops,
    layernorm_flops,
    norm_backward_flops,
    softmax_flops,
)
from .reduction import reduction_flops, resample_bwd_flops, resample_fwd_flops
from .sdpa import sdpa_bwd_flops, sdpa_fwd_flops

# Dispatch table: node "type" -> handler returning int FLOPs (or None
# when tensor data is incomplete). Unrecognised types flip the
# ``partial`` flag in compute_flops.
_FLOP_HANDLERS = {
    # Convolution: hipDNN's actual node names are ConvolutionFwdAttributes,
    # ConvolutionBwdAttributes (dgrad), and ConvolutionWrwAttributes
    # (wgrad). The *DataAttributes / *FilterAttributes spellings are
    # kept as aliases for older graph snapshots.
    "ConvolutionFwdAttributes": conv_fwd_flops,
    "ConvolutionBwdAttributes": conv_dgrad_flops,
    "ConvolutionBwdDataAttributes": conv_dgrad_flops,
    "ConvolutionWrwAttributes": conv_wgrad_flops,
    "ConvolutionBwdFilterAttributes": conv_wgrad_flops,
    "MatmulAttributes": matmul_flops,
    "MoeGroupedMatmulAttributes": moe_grouped_matmul_flops,
    "MoeGroupedMatmulBwdAttributes": moe_grouped_matmul_bwd_flops,
    "PointwiseAttributes": pointwise_flops,
    # BatchNorm: hipDNN's BatchnormAttributes covers fwd training (with
    # next_running_* outputs) and BatchnormBackwardAttributes covers
    # bwd. BatchnormFwdAttributes / BatchnormBwdAttributes are kept as
    # aliases for older graph snapshots.
    "BatchnormInferenceAttributes": batchnorm_inference_flops,
    "BatchnormInferenceAttributesVarianceExt": batchnorm_inference_flops,
    "BatchnormAttributes": batchnorm_training_flops,
    "BatchnormFwdAttributes": batchnorm_training_flops,
    "BatchnormBackwardAttributes": norm_backward_flops,
    "BatchnormBwdAttributes": norm_backward_flops,
    "LayernormAttributes": layernorm_flops,
    "LayernormBackwardAttributes": norm_backward_flops,
    "RMSNormAttributes": layernorm_flops,
    "RMSNormBackwardAttributes": norm_backward_flops,
    "SoftmaxAttributes": softmax_flops,
    "SdpaAttributes": sdpa_fwd_flops,
    "SdpaBackwardAttributes": sdpa_bwd_flops,
    "ReductionAttributes": reduction_flops,
    "ResampleFwdAttributes": resample_fwd_flops,
    "ResampleBwdAttributes": resample_bwd_flops,
    "BlockScaleQuantizeAttributes": block_scale_quantize_flops,
    "BlockScaleDequantizeAttributes": block_scale_dequantize_flops,
    "RngAttributes": rng_flops,
}


def compute_flops(graph_json: Dict[str, Any]) -> Tuple[Optional[int], bool]:
    """Sum analytical FLOPs across a graph's nodes.

    Args:
        graph_json: Parsed hipDNN graph dictionary.

    Returns:
        ``(total_flops, partial)``. ``total_flops`` is ``None`` when the
        graph has no nodes at all, or when no node could be modelled
        analytically (every node was unrecognised or lacked tensor
        data) — a count of ``0`` in that case would be indistinguishable
        from a genuine zero-FLOP graph, so ``None`` ("unknown") is
        returned instead. ``partial`` is True when at least one node was
        unrecognised or had missing tensor data; the returned sum then
        reflects only the recognised nodes.

    Unrecognised node types also surface a one-shot warning via
    :func:`warn_once` so the user notices during a run; the structured
    ``partial`` flag stays as the machine-readable signal.
    """
    nodes = graph_json.get("nodes") or []
    if not nodes:
        return None, False

    tensors_by_uid = tensor_lookup(graph_json)

    total = 0
    partial = False
    modelled_any = False
    for node in nodes:
        node_type = node.get("type", "")
        handler = _FLOP_HANDLERS.get(node_type)
        if handler is None:
            partial = True
            warn_once(
                "analytical",
                f"unrecognised node type {node_type!r}; FLOPs marked partial",
            )
            continue
        flops = handler(node, tensors_by_uid)
        if flops is None:
            partial = True
            continue
        total += flops
        modelled_any = True

    if not modelled_any:
        # Nothing could be modelled analytically: report unknown (None)
        # rather than a misleading 0 that reads as a real FLOP count.
        return None, partial

    return total, partial


def compute_io_bytes(tensor_infos: Iterable[TensorInfo]) -> int:
    """Sum bytes of all non-virtual tensors (inputs + outputs + weights).

    Virtual tensors are intermediate buffers that hipDNN may allocate
    inside a fused kernel and never materialise to global memory, so
    they are excluded. Uses :attr:`TensorInfo.size_bytes` which already
    accounts for non-contiguous strides.
    """
    total = 0
    for ti in tensor_infos:
        if ti.is_virtual:
            continue
        total += ti.size_bytes
    return total


def derive_throughputs(
    flops: Optional[int],
    io_bytes: Optional[int],
    kernel_median_ms: Optional[float],
) -> Tuple[Optional[float], Optional[float]]:
    """Derive TFLOPs/s and GB/s from totals + median kernel time.

    The median is the denominator the rocKE benchmark pipeline
    (Solera -> Strata) uses for its TFLOP/s, so the two sources are
    comparable for the same shape. It is also robust to a single noisy
    iteration, unlike the mean.

    Args:
        flops: Total analytical FLOPs (or ``None``).
        io_bytes: Total non-virtual tensor bytes (or ``None``).
        kernel_median_ms: Median GPU kernel time in ms (or ``None``).

    Returns:
        ``(tflops_per_s, gbytes_per_s)`` — either component is ``None``
        when its inputs are missing or zero.
    """
    if not kernel_median_ms or kernel_median_ms <= 0:
        return None, None
    seconds = kernel_median_ms / 1000.0
    tflops = (flops / seconds / 1e12) if flops else None
    gbytes = (io_bytes / seconds / 1e9) if io_bytes else None
    return tflops, gbytes


# ---------------------------------------------------------------------------
# Convenience wrappers used by tests that want a single-call surface.
# ---------------------------------------------------------------------------


def list_unsupported_node_types(graph_json: Dict[str, Any]) -> List[str]:
    """Return node type strings present in the graph that have no handler.

    Useful for diagnostic output that explains *why* ``partial`` is True.
    """
    seen: List[str] = []
    seen_set: set = set()
    for node in graph_json.get("nodes") or []:
        nt = node.get("type", "")
        if not nt:
            continue
        if nt in _FLOP_HANDLERS:
            continue
        if nt not in seen_set:
            seen_set.add(nt)
            seen.append(nt)
    return seen


__all__ = [
    "compute_flops",
    "compute_io_bytes",
    "derive_throughputs",
    "list_unsupported_node_types",
]

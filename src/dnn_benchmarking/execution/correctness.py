# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Compare one engine's outputs against the cached reference outputs."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..common.exceptions import ExecutionError
from ..config.benchmark_config import SuiteConfig
from ..graph.tensor_info import TensorInfo
from ..reporting.suite_results import CorrectnessResult
from ..validation import (
    ComparisonResult,
    ReferenceOutput,
    compare,
    compare_normalized_rms,
)

DEFAULT_TOLERANCE = (1e-5, 1e-6)
# (rtol, atol) by output dtype; anything else uses DEFAULT_TOLERANCE.
# bf16 has a 7-bit mantissa: 1 ULP ~= 2^-7 = 0.78% relative. Backward
# convolutions (wgrad/dgrad) accumulate over large reductions, and the MIOpen
# kernels hipDNN and PyTorch select round 2-3 ULP apart even when they pick the
# same solver. A 1% (~1.3 ULP) rtol flags that legitimate bf16 drift as a
# failure; 3% (~3.8 ULP) keeps validation meaningful while tolerating it.
# fp8 outputs: hipDNN and the reference round slightly different fp32
# accumulators to the same fp8 grid, so one element rounding the other way is
# a full ULP apart. compare() passes |a - e| <= atol + rtol * |e|. rtol is one
# ULP at the bottom of |e|'s binade (2^-mantissa_bits: e4m3 2^-3, e5m2 2^-2),
# so one ULP of |e|'s binade always passes and two never do above the smallest
# normal, except stepping down across a power of two, where the lower binade's
# ULP is half (e4m3 1.0 -> 0.875). atol is the smallest subnormal (e4m3 2^-9,
# e5m2 2^-16; the fnuz bias is one higher, halving it), so a zero reference
# accepts only the neighbouring subnormal. e8m0 holds only powers of two and
# has no zero: a value bound loose enough for one step up (2x) accepts every
# value below the reference. e8m0 therefore compares log2 values, whose
# difference is the code distance, and its (rtol, atol) = (0, 1) allows one
# code step either way (2x or 0.5x): the same one-element-rounds-the-other-way
# allowance as the other formats. --rtol/--atol also apply in log2 for e8m0.
TOLERANCES = {
    "bfloat16": (3e-2, 1e-3),
    "half": (1e-3, 1e-3),
    "fp8_e4m3": (2**-3, 2**-9),
    "fp8_e4m3_fnuz": (2**-3, 2**-10),
    "fp8_e5m2": (2**-2, 2**-16),
    "fp8_e5m2_fnuz": (2**-2, 2**-17),
    "fp8_e8m0": (0.0, 1.0),
}

# MIOpen's convolution driver validates normalized RMS error instead of
# elementwise allclose. The base tolerances come from GetDefaultTolerance();
# backward-weights doubles the base because it performs more reductions.
_MIOPEN_CONVOLUTION_RMS_BASE = {
    "float": 1.5e-6,
    "half": 8.2e-3,
    "bfloat16": 6.56e-2,
    "fp8_e4m3_fnuz": 1e-3,
    "fp8_e5m2_fnuz": 2e-3,
}
CONVOLUTION_RMS_TOLERANCES = {
    (node_type, dtype): tolerance * (2 if node_type == "ConvolutionWrwAttributes" else 1)
    for node_type in (
        "ConvolutionFwdAttributes",
        "ConvolutionBwdAttributes",
        "ConvolutionWrwAttributes",
    )
    for dtype, tolerance in _MIOPEN_CONVOLUTION_RMS_BASE.items()
}


def _producer_type(
    graph_json: Optional[Dict[str, Any]], output_uid: int
) -> Optional[str]:
    """Return the node type that produces an output tensor."""
    if not graph_json:
        return None
    for node in graph_json.get("nodes", []) or []:
        outputs = (node.get("outputs") or {}).values()
        if output_uid in outputs:
            return str(node.get("type", ""))
    return None




def _compared(data: Any, tensor_info: TensorInfo) -> Any:
    """The values compare() sees: log2 for e8m0 (see TOLERANCES), else ``data``."""
    if tensor_info.dtype.name != "fp8_e8m0":
        return data
    if hasattr(data, "detach"):
        return data.detach().float().log2()
    return np.log2(np.asarray(data, dtype=np.float32))


def tolerance_for(
    config: SuiteConfig,
    tensor_info: TensorInfo,
    graph_json: Optional[Dict[str, Any]] = None,
) -> Tuple[float, float]:
    """Return the explicit override or the output's dtype-aware default."""
    del graph_json
    if config.validation.tolerance_override is not None:
        return config.validation.tolerance_override
    return TOLERANCES.get(tensor_info.dtype.name, DEFAULT_TOLERANCE)


def normalized_rms_tolerance_for(
    config: SuiteConfig,
    tensor_info: TensorInfo,
    graph_json: Optional[Dict[str, Any]],
) -> Optional[float]:
    """Return the MIOpen RMS tolerance for a convolution output."""
    if config.validation.tolerance_override is not None:
        return None
    producer = _producer_type(graph_json, tensor_info.uid)
    return CONVOLUTION_RMS_TOLERANCES.get((producer, tensor_info.dtype.name))


def mismatch(config: SuiteConfig, message: str) -> CorrectnessResult:
    """A negative verdict with no comparison (validation is a hard gate)."""
    rtol, atol = config.validation.tolerance_override or DEFAULT_TOLERANCE
    return CorrectnessResult(
        tolerance_match=False,
        rtol=rtol,
        atol=atol,
        error_message=message,
    )


def _compare_output(
    buffer_manager: Any,
    tensor_info: TensorInfo,
    ref: ReferenceOutput,
    rtol: float,
    atol: float,
    rms_tolerance: Optional[float],
) -> Optional[ComparisonResult]:
    """Compare on the GPU when both sides are device tensors, else on the host."""

    def run(actual: Any, expected: Any) -> ComparisonResult:
        if rms_tolerance is not None:
            return compare_normalized_rms(actual, expected, tolerance=rms_tolerance)
        return compare(
            _compared(actual, tensor_info),
            _compared(expected, tensor_info),
            rtol=rtol,
            atol=atol,
        )

    if ref.device_data is not None:
        actual = buffer_manager.get_output_tensor(tensor_info.uid)
        if actual is not None:
            import torch

            try:
                return run(actual, ref.device_data)
            except torch.cuda.OutOfMemoryError:
                pass  # Fall back to the host comparison.
    actual = buffer_manager.get_output_data(tensor_info.uid)
    if actual is None:
        return None
    return run(actual, ref.data)


def check_correctness(
    buffer_manager: Any,
    tensor_infos: List[TensorInfo],
    ref_outputs: Dict[int, ReferenceOutput],
    reference_provider_name: str,
    config: SuiteConfig,
    graph_json: Optional[Dict[str, Any]] = None,
) -> CorrectnessResult:
    """Compare every graph output against the precomputed reference outputs.

    Diffs and counts aggregate over outputs; ``worst_output_uid`` and the
    message come from the output with the largest absolute difference (a
    failing output wins over a passing one).
    """
    try:
        compared: List[
            Tuple[int, float, float, Optional[float], ComparisonResult]
        ] = []
        for ti in tensor_infos:
            if not ti.is_output:
                continue
            ref = ref_outputs.get(ti.uid)
            if ref is None:
                return mismatch(
                    config,
                    f"Reference provider '{reference_provider_name}' did not "
                    f"produce output tensor UID {ti.uid}",
                )
            rtol, atol = tolerance_for(config, ti, graph_json)
            rms_tolerance = normalized_rms_tolerance_for(config, ti, graph_json)
            result = _compare_output(
                buffer_manager, ti, ref, rtol, atol, rms_tolerance
            )
            if result is not None:
                compared.append((ti.uid, rtol, atol, rms_tolerance, result))
    except (ValueError, RuntimeError, ExecutionError) as e:
        return mismatch(config, str(e))

    if not compared:
        return mismatch(config, "No output tensors to compare")

    uid, _, _, _, worst = max(
        compared, key=lambda c: (not c[4].passed, c[4].max_abs_diff)
    )
    passed = all(c[4].passed for c in compared)
    rms_comparisons = [c for c in compared if c[3] is not None]
    elementwise_comparisons = [c for c in compared if c[3] is None]
    methods = {c[4].comparison_method for c in compared}
    return CorrectnessResult(
        tolerance_match=passed,
        rtol=max((c[1] for c in elementwise_comparisons), default=0.0),
        atol=max((c[2] for c in elementwise_comparisons), default=0.0),
        max_abs_diff=max(c[4].max_abs_diff for c in compared),
        max_rel_diff=max(
            (c[4].max_rel_diff for c in elementwise_comparisons), default=None
        ),
        n_mismatch=(
            sum(c[4].n_mismatch for c in compared) if not rms_comparisons else None
        ),
        n_total=sum(c[4].n_total for c in compared),
        worst_output_uid=uid,
        error_message=None if passed else f"output {uid}: {worst.message}",
        comparison_method=next(iter(methods)) if len(methods) == 1 else "mixed",
        normalized_rms_tolerance=max(
            (c[3] for c in rms_comparisons if c[3] is not None), default=None
        ),
        normalized_rms_error=max(
            (
                c[4].normalized_rms_error
                for c in rms_comparisons
                if c[4].normalized_rms_error is not None
            ),
            default=None,
        ),
    )

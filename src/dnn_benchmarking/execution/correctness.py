# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Compare one engine's outputs against the cached reference outputs."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..common.exceptions import ExecutionError
from ..config.benchmark_config import SuiteConfig
from ..graph.tensor_info import TensorInfo
from ..metrics.analytical import convolution_terms_per_output
from ..reporting.suite_results import CorrectnessResult
from ..validation import ComparisonResult, ReferenceOutput, compare

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
# fp32 convolution outputs: atol grows with the number of summed products K.
# Convolution operands are zero-mean in [-1, 1) (generate_input_data), so many
# outputs are close to zero and only atol applies to them. The fp32 rounding
# error of a K-term sum does not shrink with the result, so two correct
# kernels that sum in a different order can differ by more than the fixed
# 1e-6 on such elements. The per-term factor is a heuristic, not an error
# bound: PyTorch CPU fp32 against float64 needed at most 0.73 u (u = 2^-24)
# per term over fwd, dgrad and wgrad shapes (K = 9 to 131,072, with stride,
# 1x1, 7x7 and depthwise cases). 4 u leaves about 5x headroom for two
# kernels that both round. A defect that changes an element by less than
# 4 u * K (for example one wrong term in a large sum) passes.
FP32_CONVOLUTION_ATOL_PER_TERM = 4 * 2.0**-24


def _compared(data: Any, tensor_info: TensorInfo) -> Any:
    """The values compare() sees: log2 for e8m0 (see TOLERANCES), else ``data``."""
    if tensor_info.dtype.name != "fp8_e8m0":
        return data
    if hasattr(data, "detach"):
        return data.detach().float().log2()
    return np.log2(np.asarray(data, dtype=np.float32))


def tolerance_for(
    config: SuiteConfig, tensor_info: TensorInfo, convolution_terms: Dict[int, int]
) -> Tuple[float, float]:
    """``--rtol/--atol`` when given, else the dtype default for this output.

    An fp32 output of a convolution node gets an atol of at least
    ``FP32_CONVOLUTION_ATOL_PER_TERM`` times its terms per element
    (``convolution_terms``, from :func:`convolution_terms_per_output`).
    """
    if config.validation.tolerance_override:
        return config.validation.tolerance_override
    rtol, atol = TOLERANCES.get(tensor_info.dtype.name, DEFAULT_TOLERANCE)
    terms = convolution_terms.get(tensor_info.uid)
    if tensor_info.dtype.name == "float" and terms:
        atol = max(atol, terms * FP32_CONVOLUTION_ATOL_PER_TERM)
    return rtol, atol


def mismatch(config: SuiteConfig, message: str) -> CorrectnessResult:
    """A negative verdict with no comparison (validation is a hard gate)."""
    rtol, atol = config.validation.tolerance_override or DEFAULT_TOLERANCE
    return CorrectnessResult(
        tolerance_match=False,
        rtol=rtol,
        atol=atol,
        error_message=message,
    )


def unchecked(config: SuiteConfig, message: str) -> CorrectnessResult:
    """No verdict: validation was skipped on purpose, with ``message`` as reason."""
    rtol, atol = config.validation.tolerance_override or DEFAULT_TOLERANCE
    return CorrectnessResult(
        tolerance_match=None,
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
) -> Optional[ComparisonResult]:
    """Compare on the GPU when both sides are device tensors, else on the host."""
    if ref.device_data is not None:
        actual = buffer_manager.get_output_tensor(tensor_info.uid)
        if actual is not None:
            import torch

            try:
                return compare(
                    _compared(actual, tensor_info),
                    _compared(ref.device_data, tensor_info),
                    rtol=rtol,
                    atol=atol,
                )
            except torch.cuda.OutOfMemoryError:
                pass  # Fall back to the host comparison.
    actual = buffer_manager.get_output_data(tensor_info.uid)
    if actual is None:
        return None
    return compare(
        _compared(actual, tensor_info),
        _compared(ref.host_data(), tensor_info),
        rtol=rtol,
        atol=atol,
    )


def check_correctness(
    buffer_manager: Any,
    tensor_infos: List[TensorInfo],
    ref_outputs: Dict[int, ReferenceOutput],
    reference_provider_name: str,
    config: SuiteConfig,
    graph_json: Dict[str, Any],
) -> CorrectnessResult:
    """Compare every graph output against the precomputed reference outputs.

    Diffs and counts aggregate over outputs; ``worst_output_uid`` and the
    message come from the output with the largest absolute difference (a
    failing output wins over a passing one).
    """
    convolution_terms = convolution_terms_per_output(graph_json)
    try:
        compared: List[Tuple[int, float, float, ComparisonResult]] = []
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
            rtol, atol = tolerance_for(config, ti, convolution_terms)
            result = _compare_output(buffer_manager, ti, ref, rtol, atol)
            if result is not None:
                compared.append((ti.uid, rtol, atol, result))
    except (ValueError, RuntimeError, ExecutionError) as e:
        return mismatch(config, str(e))

    if not compared:
        return mismatch(config, "No output tensors to compare")

    uid, _, _, worst = max(compared, key=lambda c: (not c[3].passed, c[3].max_abs_diff))
    passed = all(c[3].passed for c in compared)
    return CorrectnessResult(
        tolerance_match=passed,
        rtol=max(c[1] for c in compared),
        atol=max(c[2] for c in compared),
        max_abs_diff=max(c[3].max_abs_diff for c in compared),
        max_rel_diff=max(c[3].max_rel_diff for c in compared),
        n_mismatch=sum(c[3].n_mismatch for c in compared),
        n_total=sum(c[3].n_total for c in compared),
        worst_output_uid=uid,
        error_message=None if passed else f"output {uid}: {worst.message}",
    )

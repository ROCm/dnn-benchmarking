# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tolerance comparison of an output against its reference, on the host.

Every output and every reference reaches the parent process as a host array
(the child processes save them), so the comparison uses NumPy only.
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from ..common.blocks import global_index, iter_blocks

# Comparison dtype: float32 at minimum, as hipDNN's CpuFpReferenceValidation
# does, and float64 for dtypes float32 cannot hold exactly.
_FLOAT64_DTYPES = frozenset({"float64", "int32", "int64", "uint32", "uint64"})


def _compare_dtype(*dtypes: np.dtype) -> np.dtype:
    """float64 if any dtype needs it, else float32."""
    names = {dtype.name for dtype in dtypes}
    return np.dtype("float64" if names & _FLOAT64_DTYPES else "float32")


@dataclass
class ComparisonResult:
    """Result of comparing an output against a reference.

    Attributes:
        passed: True when every element satisfies
            ``|actual - expected| <= atol + rtol * |expected|``.
        max_abs_diff: Largest ``|actual - expected|``; inf when the inputs
            are non-finite or have different shapes.
        max_rel_diff: Largest ``|actual - expected| / |expected|`` over the
            elements where ``|expected| > atol`` (0.0 if there are none), so
            near-zero references cannot dominate it.
        n_mismatch: Number of elements outside tolerance.
        n_total: Number of elements compared.
        worst_index: Index of the largest absolute difference, or None.
        message: Human-readable description of the result.
    """

    passed: bool
    max_abs_diff: float
    max_rel_diff: float
    n_mismatch: int
    n_total: int
    worst_index: Optional[Tuple[int, ...]]
    message: str


def _failure(message: str, n_total: int) -> ComparisonResult:
    inf = float("inf")
    return ComparisonResult(False, inf, inf, n_total, n_total, None, message)


def compare(
    actual: np.ndarray, expected: np.ndarray, *, rtol: float, atol: float
) -> ComparisonResult:
    """Compare ``actual`` against ``expected`` within ``atol + rtol * |expected|``.

    Values are compared in float32, or float64 when either side needs it. Any
    NaN or Inf fails.

    Memory: the work runs one block (``common.blocks``) at a time, so the
    temporaries (about 13 bytes per element for float32) cover one block,
    not the whole output.
    """
    a = np.asarray(actual)
    ref = np.asarray(expected)
    dtype = _compare_dtype(a.dtype, ref.dtype)

    def block(x: np.ndarray, index: Tuple) -> np.ndarray:
        return x[index].astype(dtype, copy=False)

    shape = tuple(a.shape)
    n_total = math.prod(shape)

    if tuple(ref.shape) != shape:
        return _failure(
            f"Shape mismatch: output={shape} vs reference={tuple(ref.shape)}", n_total
        )
    if not shape:  # 0-d NumPy arithmetic yields scalars, which have no out=.
        a, ref = a.reshape(1), ref.reshape(1)
    blocks = list(iter_blocks(tuple(a.shape)))
    if not all(bool(np.isfinite(a[i]).all()) for i in blocks):
        return _failure("output contains NaN or Inf values", n_total)
    if not all(bool(np.isfinite(block(ref, i)).all()) for i in blocks):
        return _failure("reference contains NaN or Inf values", n_total)

    tolerance = f"(rtol={rtol}, atol={atol})"
    if n_total == 0:
        return ComparisonResult(True, 0.0, 0.0, 0, 0, None, f"Match {tolerance}")

    n_mismatch = 0
    max_abs_diff = -math.inf
    worst: Tuple[int, ...] = ()
    rel_maxima = []
    # A converted block is ours to overwrite; otherwise it views the caller's
    # reference, which later engines compare against again.
    owns_e = ref.dtype != dtype
    # Overflow to inf is a real mismatch, so NumPy's warning is not useful.
    with np.errstate(over="ignore"):
        for index in blocks:
            e = block(ref, index)
            # e has the compare dtype, which is at least as wide as a's, so
            # the subtraction promotes a without a converted copy.
            diff = a[index] - e
            np.abs(diff, out=diff)
            abs_e = np.abs(e, out=e) if owns_e else np.abs(e)
            del e
            threshold = abs_e * rtol
            threshold += atol
            n_mismatch += int((diff > threshold).sum())
            del threshold
            flat = int(diff.argmax())
            value = float(diff.reshape(-1)[flat])
            if value > max_abs_diff:  # strict: the first maximum wins
                max_abs_diff = value
                worst = global_index(index, tuple(diff.shape), flat)
            # Relative difference over |e| > atol: dividing by inf zeroes the
            # rest, and every ratio is >= 0, so max() is 0.0 when none qualify.
            diff /= np.where(abs_e > atol, abs_e, math.inf)
            del abs_e
            rel_maxima.append(float(diff.max()))
            del diff
    max_rel_diff = float(np.max(rel_maxima))  # keeps a NaN ratio, as max() did
    flat_worst = int(np.ravel_multi_index(worst, tuple(a.shape)))
    worst_index = tuple(int(i) for i in np.unravel_index(flat_worst, shape))

    if n_mismatch == 0:
        message = f"Match {tolerance}"
    else:
        message = (
            f"Mismatch: {n_mismatch}/{n_total} elements outside tolerance, "
            f"max_abs_diff={max_abs_diff:.2e} at {worst_index}, "
            f"max_rel_diff={max_rel_diff:.2e} {tolerance}"
        )
    return ComparisonResult(
        passed=n_mismatch == 0,
        max_abs_diff=max_abs_diff,
        max_rel_diff=max_rel_diff,
        n_mismatch=n_mismatch,
        n_total=n_total,
        worst_index=worst_index,
        message=message,
    )

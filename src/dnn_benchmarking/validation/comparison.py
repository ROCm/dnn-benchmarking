# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tolerance comparison of an output against its reference.

One algorithm serves NumPy arrays and torch tensors: the operations used
(``abs``, ``isfinite``, arithmetic, boolean indexing, ``max``/``argmax``/
``sum``) behave the same in both libraries, so torch tensors are compared on
their own device without a host copy and host arrays use NumPy.
"""

import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np

# Comparison dtype: float32 at minimum, as hipDNN's CpuFpReferenceValidation
# does, and float64 for dtypes float32 cannot hold exactly. Names match both
# numpy (``dtype.name``) and torch (``str(dtype)`` without ``torch.``).
_FLOAT64_DTYPES = frozenset({"float64", "int32", "int64", "uint32", "uint64"})


def _compare_dtype_name(*dtypes: object) -> str:
    """Return "float64" if any dtype needs it, else "float32"."""
    names = {str(dtype).removeprefix("torch.") for dtype in dtypes}
    return "float64" if names & _FLOAT64_DTYPES else "float32"


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
    actual: Any,
    expected: Any,
    *,
    rtol: float,
    atol: float,
    actual_label: str = "output",
    expected_label: str = "reference",
) -> ComparisonResult:
    """Compare ``actual`` against ``expected`` within ``atol + rtol * |expected|``.

    A torch tensor ``actual`` is compared on its device (``expected`` is moved
    there); anything else is compared as NumPy arrays. Values are compared in
    float32, or float64 when either side needs it. Any NaN or Inf fails.

    Memory: the temporaries are full size, about 3x the output in the compare
    dtype. A torch out-of-memory error propagates so the caller can fall back
    to host arrays.
    """
    if hasattr(actual, "detach"):
        import torch as xp

        a = actual.detach()
        e = xp.as_tensor(expected, device=a.device)
        dtype = getattr(xp, _compare_dtype_name(a.dtype, e.dtype))
        e = e.to(dtype)
    else:
        xp = np
        a = np.asarray(actual)
        e = np.asarray(expected)
        dtype = np.dtype(_compare_dtype_name(a.dtype, e.dtype))
        e = e.astype(dtype, copy=False)
    shape = tuple(a.shape)
    n_total = math.prod(shape)

    if tuple(e.shape) != shape:
        return _failure(
            f"Shape mismatch: {actual_label}={shape} vs "
            f"{expected_label}={tuple(e.shape)}",
            n_total,
        )
    if not bool(xp.isfinite(a).all()):
        return _failure(f"{actual_label} contains NaN or Inf values", n_total)
    if not bool(xp.isfinite(e).all()):
        return _failure(f"{expected_label} contains NaN or Inf values", n_total)

    tolerance = f"(rtol={rtol}, atol={atol})"
    if n_total == 0:
        return ComparisonResult(True, 0.0, 0.0, 0, 0, None, f"Match {tolerance}")

    if not shape:  # 0-d NumPy arithmetic yields scalars, which have no out=.
        a, e = a.reshape(1), e.reshape(1)

    # e has the compare dtype, which is at least as wide as a's, so the
    # subtraction promotes a element by element without a converted copy.
    # Overflow to inf is a real mismatch, so NumPy's warning is not useful.
    with np.errstate(over="ignore"):
        diff = a - e
        xp.abs(diff, out=diff)
        abs_e = xp.abs(e)
        threshold = abs_e * rtol
        threshold += atol
        n_mismatch = int((diff > threshold).sum())
        del threshold
        flat_worst = int(diff.argmax())
        max_abs_diff = float(diff.reshape(-1)[flat_worst])
        significant = abs_e > atol
        max_rel_diff = (
            float((diff[significant] / abs_e[significant]).max())
            if bool(significant.any())
            else 0.0
        )
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

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

# Summing a bool mask casts it to the sum dtype first; int32 costs 4 bytes
# per element instead of int64's 8, and chunks this size cannot overflow it.
_COUNT_CHUNK = 1 << 30


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
    actual: Any, expected: Any, *, rtol: float, atol: float
) -> ComparisonResult:
    """Compare ``actual`` against ``expected`` within ``atol + rtol * |expected|``.

    A torch tensor ``actual`` is compared on its device (``expected`` is moved
    there); anything else is compared as NumPy arrays. Values are compared in
    float32, or float64 when either side needs it. Any NaN or Inf fails.

    Memory: at most three full-size temporaries in the compare dtype plus a
    bool mask (13 bytes per element for float32). A torch out-of-memory
    error propagates so the caller can fall back to host arrays.
    """
    if hasattr(actual, "detach"):
        import torch as xp

        a = actual.detach()
        ref = xp.as_tensor(expected, device=a.device)
        dtype = getattr(xp, _compare_dtype_name(a.dtype, ref.dtype))
        e = ref.to(dtype)
    else:
        xp = np
        a = np.asarray(actual)
        ref = np.asarray(expected)
        dtype = np.dtype(_compare_dtype_name(a.dtype, ref.dtype))
        e = ref.astype(dtype, copy=False)
    # A converted copy is ours to overwrite; otherwise e is the caller's data.
    owns_e = e is not ref
    del ref
    shape = tuple(a.shape)
    n_total = math.prod(shape)

    if tuple(e.shape) != shape:
        return _failure(
            f"Shape mismatch: output={shape} vs reference={tuple(e.shape)}", n_total
        )
    if not bool(xp.isfinite(a).all()):
        return _failure("output contains NaN or Inf values", n_total)
    if not bool(xp.isfinite(e).all()):
        return _failure("reference contains NaN or Inf values", n_total)

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
        abs_e = xp.abs(e, out=e) if owns_e else xp.abs(e)
        threshold = abs_e * rtol
        threshold += atol
        mismatch = (diff > threshold).reshape(-1)
        del threshold
        n_mismatch = sum(
            int(mismatch[i : i + _COUNT_CHUNK].sum(dtype=xp.int32))
            for i in range(0, n_total, _COUNT_CHUNK)
        )
        del mismatch
        flat_worst = int(diff.argmax())
        max_abs_diff = float(diff.reshape(-1)[flat_worst])
        # Relative difference over |e| > atol: dividing by inf zeroes the
        # rest, and every ratio is >= 0, so max() is 0.0 when none qualify.
        # where() rather than mask assignment, which torch turns into an
        # 8-byte-per-element index.
        denominator = xp.where(abs_e > atol, abs_e, math.inf)
        del abs_e
        diff /= denominator
        del denominator
        max_rel_diff = float(diff.max())
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

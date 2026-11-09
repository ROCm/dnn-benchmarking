# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tolerance comparison of an output against its reference.

One algorithm serves NumPy arrays and torch tensors: the operations used
(``abs``, ``isfinite``, arithmetic, ``where``, ``max``/``argmax``/``sum``)
behave the same in both libraries, so torch tensors are compared on their own
device without a host copy and host arrays use NumPy.
"""

import math
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence, Tuple

import numpy as np

from ..common.blocks import global_index, iter_blocks

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


_Setup = Tuple[Any, Any, Any, Any, bool, Callable[[Any, Tuple, bool], Any]]


def _prepare(actual: Any, expected: Any) -> _Setup:
    """Return ``(xp, a, ref, dtype, upcast_a, block)`` for the two inputs."""
    if hasattr(actual, "detach"):
        import torch as xp

        a = actual.detach()
        ref = xp.as_tensor(expected, device=a.device)
        dtype = getattr(xp, _compare_dtype_name(a.dtype, ref.dtype))
        # torch implements no isfinite/abs/sub for float8; upcast first.
        upcast_a = a.is_floating_point() and a.element_size() == 1

        def block(x: Any, index: Tuple, upcast: bool) -> Any:
            return x[index].to(dtype) if upcast else x[index]

    else:
        xp = np
        a = np.asarray(actual)
        ref = np.asarray(expected)
        dtype = np.dtype(_compare_dtype_name(a.dtype, ref.dtype))
        upcast_a = False

        def block(x: Any, index: Tuple, upcast: bool) -> Any:
            return x[index].astype(dtype, copy=False) if upcast else x[index]

    return xp, a, ref, dtype, upcast_a, block


def _unmatched_nonfinite(xp: Any, x: Any, other: Any) -> bool:
    """True when ``x`` holds a NaN or an infinity that ``other`` lacks at that position."""
    bad = ~xp.isfinite(x)
    if not bool(bad.any()):
        return False
    bad &= ~(xp.isinf(x) & (x == other))
    return bool(bad.any())


def _compare_arrays(
    setup: _Setup,
    a: Any,
    ref: Any,
    rtol: float,
    atol: float,
    matching_infinities: bool,
) -> ComparisonResult:
    """Compare equally shaped ``a`` and ``ref`` (views of the prepared inputs)."""
    xp, _, _, dtype, upcast_a, block = setup
    shape = tuple(a.shape)
    n_total = math.prod(shape)

    if not shape:  # 0-d NumPy arithmetic yields scalars, which have no out=.
        a, ref = a.reshape(1), ref.reshape(1)
    blocks = list(iter_blocks(tuple(a.shape)))
    if matching_infinities:
        # Same-sign infinities in both tensors are equal; NaN never is.
        for i in blocks:
            ab, eb = block(a, i, upcast_a), block(ref, i, True)
            if _unmatched_nonfinite(xp, ab, eb):
                return _failure("output contains NaN or Inf values", n_total)
            if _unmatched_nonfinite(xp, eb, ab):
                if bool(xp.isnan(eb).any()):
                    return _failure("reference contains NaN values", n_total)
                # Every output infinity matched above, so the output is
                # finite where the reference holds this one.
                return _failure(
                    "output is finite where the reference holds an infinity",
                    n_total,
                )
    else:
        if not all(bool(xp.isfinite(block(a, i, upcast_a)).all()) for i in blocks):
            return _failure("output contains NaN or Inf values", n_total)
        if not all(bool(xp.isfinite(block(ref, i, True)).all()) for i in blocks):
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
    # inf - inf, taken for matched infinities, is NaN and is replaced below.
    with np.errstate(
        over="ignore", **({"invalid": "ignore"} if matching_infinities else {})
    ):
        for index in blocks:
            e = block(ref, index, True)
            # e has the compare dtype, which is at least as wide as a's, so
            # the subtraction promotes a without a converted copy.
            ab = block(a, index, upcast_a)
            diff = ab - e
            if matching_infinities:
                diff = xp.where(xp.isinf(e) & (ab == e), 0.0, diff)
            xp.abs(diff, out=diff)
            abs_e = xp.abs(e, out=e) if owns_e else xp.abs(e)
            del e, ab
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
            # where() rather than mask assignment, which torch turns into an
            # 8-byte-per-element index.
            diff /= xp.where(abs_e > atol, abs_e, math.inf)
            del abs_e
            rel_maxima.append(float(diff.max()))
            del diff
    max_rel_diff = float(np.max(rel_maxima))  # np.max propagates a NaN ratio
    flat_worst = int(np.ravel_multi_index(worst, tuple(a.shape)))
    worst_index = tuple(int(i) for i in np.unravel_index(flat_worst, shape))

    return ComparisonResult(
        passed=n_mismatch == 0,
        max_abs_diff=max_abs_diff,
        max_rel_diff=max_rel_diff,
        n_mismatch=n_mismatch,
        n_total=n_total,
        worst_index=worst_index,
        message=_message(
            n_mismatch, n_total, max_abs_diff, max_rel_diff, worst_index, tolerance
        ),
    )


def _message(
    n_mismatch: int,
    n_total: int,
    max_abs_diff: float,
    max_rel_diff: float,
    worst_index: Optional[Tuple[int, ...]],
    tolerance: str,
) -> str:
    if n_mismatch == 0:
        return f"Match {tolerance}"
    return (
        f"Mismatch: {n_mismatch}/{n_total} elements outside tolerance, "
        f"max_abs_diff={max_abs_diff:.2e} at {worst_index}, "
        f"max_rel_diff={max_rel_diff:.2e} {tolerance}"
    )


def compare(
    actual: Any,
    expected: Any,
    *,
    rtol: float,
    atol: float,
    lengths: Optional[Sequence[int]] = None,
    matching_infinities: bool = False,
) -> ComparisonResult:
    """Compare ``actual`` against ``expected`` within ``atol + rtol * |expected|``.

    A torch tensor ``actual`` is compared on its device (``expected`` is moved
    there); anything else is compared as NumPy arrays. Values are compared in
    float32, or float64 when either side needs it. Any NaN or Inf fails,
    except that with ``matching_infinities`` an infinity that has the same
    sign in both tensors counts as equal.

    With ``lengths`` (one per batch, dimension 0), only the first
    ``lengths[b]`` positions along dimension -2 of batch ``b`` are compared;
    the rest is padding and is not read. ``n_total`` then counts the compared
    elements and ``worst_index`` indexes the full tensor.

    Memory: the work runs one block (``common.blocks``) at a time, so the
    temporaries (about 13 bytes per element for float32) cover one block,
    not the whole output. A torch out-of-memory error propagates so the
    caller can fall back to host arrays.
    """
    setup = _prepare(actual, expected)
    a, ref = setup[1], setup[2]
    shape = tuple(a.shape)

    if tuple(ref.shape) != shape:
        return _failure(
            f"Shape mismatch: output={shape} vs reference={tuple(ref.shape)}",
            math.prod(shape),
        )
    if lengths is None:
        return _compare_arrays(setup, a, ref, rtol, atol, matching_infinities)

    if len(shape) < 3 or len(lengths) != shape[0]:
        raise ValueError(
            f"{len(lengths)} lengths do not describe a tensor of shape {shape}"
        )
    if any(not 0 <= n <= shape[-2] for n in lengths):
        raise ValueError(f"lengths {list(lengths)} exceed dimension {shape[-2]}")
    tolerance = f"(rtol={rtol}, atol={atol})"
    n_total = sum(lengths) * math.prod(shape[1:-2]) * shape[-1]
    n_mismatch = 0
    max_abs_diff = -math.inf
    max_rel_diff = 0.0
    worst: Optional[Tuple[int, ...]] = None
    for b, length in enumerate(lengths):
        if length == 0:
            continue
        part = _compare_arrays(
            setup,
            a[b][..., :length, :],
            ref[b][..., :length, :],
            rtol,
            atol,
            matching_infinities,
        )
        if not part.passed and part.worst_index is None:  # NaN or Inf
            return _failure(part.message, n_total)
        if part.worst_index is None:  # nothing to compare in this batch
            continue
        n_mismatch += part.n_mismatch
        max_rel_diff = max(max_rel_diff, part.max_rel_diff)
        if part.max_abs_diff > max_abs_diff:  # strict: the first maximum wins
            max_abs_diff = part.max_abs_diff
            worst = (b, *part.worst_index)
    if worst is None:
        return ComparisonResult(True, 0.0, 0.0, 0, 0, None, f"Match {tolerance}")
    return ComparisonResult(
        passed=n_mismatch == 0,
        max_abs_diff=max_abs_diff,
        max_rel_diff=max_rel_diff,
        n_mismatch=n_mismatch,
        n_total=n_total,
        worst_index=worst,
        message=_message(
            n_mismatch, n_total, max_abs_diff, max_rel_diff, worst, tolerance
        ),
    )

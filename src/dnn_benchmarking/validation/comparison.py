# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Unified comparison logic for array validation.

Shared by reference validation and any direct array comparisons.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Tuple

import numpy as np

if TYPE_CHECKING:
    import torch

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
    """Result of comparing two arrays.

    Attributes:
        passed: Whether arrays match within tolerance.
        max_abs_diff: Maximum absolute difference between arrays.
        max_rel_diff: Maximum relative difference between arrays.
        message: Human-readable description of the result.
    """

    passed: bool
    max_abs_diff: float
    max_rel_diff: float
    message: str


class ArrayComparator:
    """Compares numpy arrays with tolerance-based matching.

    Handles NaN/Inf detection, shape validation, and difference calculation.
    Used by reference validation and direct output comparisons.
    """

    def __init__(self, rtol: float = 1e-5, atol: float = 1e-8) -> None:
        """Initialize comparator with tolerance settings.

        Args:
            rtol: Relative tolerance for np.allclose comparison.
            atol: Absolute tolerance for np.allclose comparison.
        """
        self._rtol = rtol
        self._atol = atol

    @property
    def rtol(self) -> float:
        """Relative tolerance."""
        return self._rtol

    @property
    def atol(self) -> float:
        """Absolute tolerance."""
        return self._atol

    def compare(
        self,
        actual: np.ndarray,
        expected: np.ndarray,
        actual_label: str = "actual",
        expected_label: str = "expected",
    ) -> ComparisonResult:
        """Compare two arrays with NaN/Inf checking and tolerance comparison.

        Args:
            actual: The array to validate (e.g., hipDNN output).
            expected: The reference array (e.g., PyTorch output).
            actual_label: Label for actual array in messages (default: "actual").
            expected_label: Label for expected array in messages (default: "expected").

        Returns:
            ComparisonResult with pass/fail status and difference metrics.
        """
        expected_in = np.asarray(expected)
        actual = np.asarray(actual)
        # float32 minimum: fp16 differences cannot overflow, and the host and
        # device paths (compare_tensors) run the same float operations.
        # Only expected is converted (asarray copies only when the dtype
        # changes), because that buffer becomes |e|. actual is cast element by
        # element inside the subtraction.
        dtype = np.dtype(_compare_dtype_name(actual.dtype, expected_in.dtype))
        expected = np.asarray(expected_in, dtype=dtype)

        if not np.isfinite(actual).all():
            return ComparisonResult(
                passed=False,
                max_abs_diff=float("inf"),
                max_rel_diff=float("inf"),
                message=f"{actual_label} contains NaN or Inf values",
            )
        if not np.isfinite(expected).all():
            return ComparisonResult(
                passed=False,
                max_abs_diff=float("inf"),
                max_rel_diff=float("inf"),
                message=f"{expected_label} contains NaN or Inf values",
            )
        if actual.shape != expected.shape:
            return ComparisonResult(
                passed=False,
                max_abs_diff=float("inf"),
                max_rel_diff=float("inf"),
                message=f"Shape mismatch: {actual_label}={actual.shape} vs {expected_label}={expected.shape}",
            )
        if actual.size == 0:
            return self._result(True, 0.0, 0.0)

        # Written-out allclose (|a - e| <= atol + rtol * |e|), so |a - e| and
        # |e| are computed once and reused for the reported diffs. Each
        # temporary is updated in place after its last read. Overflow to inf
        # in float32 is a real mismatch, so its warning is not useful.
        with np.errstate(over="ignore"):
            abs_diff = np.subtract(actual, expected, dtype=dtype)
            np.abs(abs_diff, out=abs_diff)
            max_abs_diff = float(abs_diff.max())
            # Take |e| in place only on a converted copy, never on the caller's.
            if expected is expected_in:
                abs_expected = np.abs(expected)
            else:
                abs_expected = np.abs(expected, out=expected)
            threshold = abs_expected * self._rtol
            threshold += self._atol
            passed = bool((abs_diff <= threshold).all())
            del threshold
            abs_expected += 1e-10
            abs_diff /= abs_expected
            max_rel_diff = float(abs_diff.max())

        return self._result(passed, max_abs_diff, max_rel_diff)

    def compare_tensors(
        self,
        actual: "torch.Tensor",
        expected: "torch.Tensor",
        actual_label: str = "actual",
        expected_label: str = "expected",
    ) -> ComparisonResult:
        """Compare two torch tensors on their device, with ``compare`` semantics.

        ``expected`` moves to the device of ``actual``. Both are compared in
        the dtype ``compare`` uses, with the same operations, so verdicts and
        messages are identical.

        ponytail: full-size temporaries. Extra VRAM is about 3.25x the output
        in the compare dtype (for example 13 bytes per fp16 or fp32 element).
        An out-of-memory error makes the caller fall back to the host path.
        Chunk the work if real graphs hit that limit.
        """
        import torch

        a = actual.detach()
        e = expected.detach().to(a.device)

        # isfinite is exact on the native dtype, so it runs before conversion.
        if not bool(torch.isfinite(a).all()):
            return ComparisonResult(
                passed=False,
                max_abs_diff=float("inf"),
                max_rel_diff=float("inf"),
                message=f"{actual_label} contains NaN or Inf values",
            )
        if not bool(torch.isfinite(e).all()):
            return ComparisonResult(
                passed=False,
                max_abs_diff=float("inf"),
                max_rel_diff=float("inf"),
                message=f"{expected_label} contains NaN or Inf values",
            )
        if a.shape != e.shape:
            return ComparisonResult(
                passed=False,
                max_abs_diff=float("inf"),
                max_rel_diff=float("inf"),
                message=(
                    f"Shape mismatch: {actual_label}={tuple(a.shape)} "
                    f"vs {expected_label}={tuple(e.shape)}"
                ),
            )

        if a.numel() == 0:
            return self._result(True, 0.0, 0.0)

        dtype = getattr(torch, _compare_dtype_name(a.dtype, e.dtype))
        e_conv = e.to(dtype)  # Same tensor, not a copy, when e is already dtype.
        # Mixed-dtype subtraction casts a element by element inside the
        # kernel; the result has dtype because dtype is at least as wide as a.
        abs_diff = torch.sub(a, e_conv).abs_()
        max_abs_diff = float(abs_diff.max())
        # Take |e| in place only on a converted copy, never on the caller's e.
        abs_expected = e_conv.abs_() if e_conv is not e else e_conv.abs()
        del e_conv
        # Written-out allclose, as in compare, reusing |a - e| and |e|.
        threshold = abs_expected.mul(self._rtol).add_(self._atol)
        passed = bool((abs_diff <= threshold).all())
        del threshold
        max_rel_diff = float(abs_diff.div_(abs_expected.add_(1e-10)).max())
        return self._result(passed, max_abs_diff, max_rel_diff)

    def _result(
        self, passed: bool, max_abs_diff: float, max_rel_diff: float
    ) -> ComparisonResult:
        """Build the ComparisonResult and message shared by both comparisons."""
        if passed:
            message = f"Match (rtol={self._rtol}, atol={self._atol})"
        else:
            message = (
                f"Mismatch: max_abs_diff={max_abs_diff:.2e}, "
                f"max_rel_diff={max_rel_diff:.2e} "
                f"(rtol={self._rtol}, atol={self._atol})"
            )
        return ComparisonResult(
            passed=passed,
            max_abs_diff=max_abs_diff,
            max_rel_diff=max_rel_diff,
            message=message,
        )

    def compare_with_diffs(
        self, actual: np.ndarray, expected: np.ndarray
    ) -> Tuple[bool, float, float]:
        """Simplified comparison returning just pass status and diffs.

        Convenience method for cases where full ComparisonResult isn't needed.

        Args:
            actual: The array to validate.
            expected: The reference array.

        Returns:
            Tuple of (passed, max_abs_diff, max_rel_diff).
        """
        result = self.compare(actual, expected)
        return result.passed, result.max_abs_diff, result.max_rel_diff

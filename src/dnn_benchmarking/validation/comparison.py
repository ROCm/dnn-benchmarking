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
        # Compare in float64 so fp16 differences cannot overflow and the host
        # and device paths (compare_tensors) reach the same verdict.
        actual = np.asarray(actual, dtype=np.float64)
        expected = np.asarray(expected, dtype=np.float64)

        # Check for NaN/Inf in actual
        if np.any(np.isnan(actual)) or np.any(np.isinf(actual)):
            return ComparisonResult(
                passed=False,
                max_abs_diff=float("inf"),
                max_rel_diff=float("inf"),
                message=f"{actual_label} contains NaN or Inf values",
            )

        # Check for NaN/Inf in expected
        if np.any(np.isnan(expected)) or np.any(np.isinf(expected)):
            return ComparisonResult(
                passed=False,
                max_abs_diff=float("inf"),
                max_rel_diff=float("inf"),
                message=f"{expected_label} contains NaN or Inf values",
            )

        # Ensure shapes match
        if actual.shape != expected.shape:
            return ComparisonResult(
                passed=False,
                max_abs_diff=float("inf"),
                max_rel_diff=float("inf"),
                message=f"Shape mismatch: {actual_label}={actual.shape} vs {expected_label}={expected.shape}",
            )

        # Calculate differences
        abs_diff = np.abs(actual - expected)
        max_abs_diff = float(np.max(abs_diff)) if abs_diff.size > 0 else 0.0

        # Handle division by zero for relative difference
        with np.errstate(divide="ignore", invalid="ignore"):
            rel_diff = abs_diff / (np.abs(expected) + 1e-10)
            max_rel_diff = float(np.max(rel_diff)) if rel_diff.size > 0 else 0.0

        # Perform allclose comparison
        passed = np.allclose(actual, expected, rtol=self._rtol, atol=self._atol)

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
        float64, as in ``compare``, so verdicts and messages are identical.
        """
        import torch

        a = actual.detach().to(torch.float64)
        e = expected.detach().to(device=a.device, dtype=torch.float64)

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

        abs_diff = (a - e).abs()
        max_abs_diff = float(abs_diff.max()) if abs_diff.numel() else 0.0
        max_rel_diff = (
            float((abs_diff / (e.abs() + 1e-10)).max()) if abs_diff.numel() else 0.0
        )
        passed = bool(torch.allclose(a, e, rtol=self._rtol, atol=self._atol))
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

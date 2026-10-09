# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for compare(): verdicts, mismatch statistics, and host/device parity."""

import numpy as np
import pytest

from dnn_benchmarking.validation import compare


def _both(actual: np.ndarray, expected: np.ndarray, **tol):
    """Compare on the host, and as torch tensors when torch is installed."""
    host = compare(actual, expected, **tol)
    try:
        import torch
    except ImportError:
        return host, host
    return host, compare(torch.from_numpy(actual), torch.from_numpy(expected), **tol)


def test_identical_arrays_pass() -> None:
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    result = compare(a, a.copy(), rtol=0.0, atol=0.0)

    assert result.passed
    assert (result.max_abs_diff, result.max_rel_diff, result.n_mismatch) == (0, 0, 0)
    assert result.n_total == 3


def test_tolerance_is_atol_plus_rtol_times_expected() -> None:
    expected = np.array([100.0, 100.0, 100.0], dtype=np.float64)
    # atol + rtol*|e| = 1 + 0.01*100 = 2: 2.0 is inside, 2.5 is outside, and
    # 2.01 is outside only because the limit uses |expected|, not |actual|
    # (1 + 0.01*102.01 = 2.0201 would admit it).
    actual = expected + [2.0, 2.5, 2.01]

    for result in _both(actual, expected, rtol=0.01, atol=1.0):
        assert result.n_mismatch == 2
        assert result.worst_index == (1,)


def test_mismatch_count_and_worst_index() -> None:
    expected = np.zeros((3, 4), dtype=np.float32)
    actual = expected.copy()
    actual[0, 1] = 0.5
    actual[2, 3] = -2.0  # worst
    actual[1, 0] = 0.05  # inside atol

    for result in _both(actual, expected, rtol=0.0, atol=0.1):
        assert not result.passed
        assert result.n_mismatch == 2
        assert result.n_total == 12
        assert result.worst_index == (2, 3)
        assert result.max_abs_diff == 2.0
        assert result.message == (
            "Mismatch: 2/12 elements outside tolerance, max_abs_diff=2.00e+00 "
            "at (2, 3), max_rel_diff=0.00e+00 (rtol=0.0, atol=0.1)"
        )


def test_max_rel_diff_ignores_near_zero_references() -> None:
    """|a - e| / |e| over |e| <= atol would report ~1e6 for a harmless diff."""
    expected = np.array([1e-7, 2.0], dtype=np.float32)
    actual = np.array([0.1, 2.2], dtype=np.float32)

    for result in _both(actual, expected, rtol=0.2, atol=0.2):
        assert result.passed
        assert result.max_rel_diff == pytest.approx(0.1, rel=1e-5)


def test_max_rel_diff_is_zero_when_every_reference_is_near_zero() -> None:
    result = compare(np.array([0.01]), np.array([0.0]), rtol=0.0, atol=0.1)

    assert result.passed
    assert result.max_rel_diff == 0.0


def test_failure_against_near_zero_reference_counts_mismatch() -> None:
    """max_rel_diff skips |e| <= atol, so only n_mismatch reports the failure."""
    result = compare(np.array([5.0]), np.array([0.0]), rtol=1e-5, atol=1e-6)

    assert not result.passed
    assert result.n_mismatch == 1
    assert result.max_abs_diff == 5.0
    assert result.max_rel_diff == 0.0


@pytest.mark.parametrize("side", ["actual", "expected"])
@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_non_finite_values_fail(side: str, bad: float) -> None:
    good = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    poisoned = good.copy()
    poisoned[1] = bad
    actual, expected = (poisoned, good) if side == "actual" else (good, poisoned)

    for result in _both(actual, expected, rtol=1.0, atol=1.0):
        assert not result.passed
        assert result.max_abs_diff == float("inf")
        assert result.n_mismatch == result.n_total == 3
        name = "output" if side == "actual" else "reference"
        assert result.message == f"{name} contains NaN or Inf values"


def test_shape_mismatch_fails() -> None:
    for result in _both(np.zeros((2, 3)), np.zeros((3, 2)), rtol=1.0, atol=1.0):
        assert not result.passed
        assert result.max_abs_diff == float("inf")
        assert result.message == "Shape mismatch: output=(2, 3) vs reference=(3, 2)"


def test_empty_and_scalar_arrays() -> None:
    assert compare(np.array([]), np.array([]), rtol=0.0, atol=0.0).passed
    scalar = compare(np.float32(1.0), np.float32(3.0), rtol=0.0, atol=0.5)
    assert not scalar.passed
    assert scalar.max_abs_diff == 2.0 and scalar.worst_index == ()


def test_fp16_difference_does_not_overflow() -> None:
    """fp16 subtraction would give inf; the float32 compare dtype stays finite."""
    a = np.array([60000, 1.0], dtype=np.float16)
    e = np.array([-60000, 1.0], dtype=np.float16)

    host, device = _both(a, e, rtol=1e-3, atol=1e-3)

    assert host.max_abs_diff == 120000.0
    assert host == device


def test_int32_compared_in_float64() -> None:
    """float32 would round 2**24 + 1 to 2**24 and hide the difference."""
    a = np.array([2**24 + 1], dtype=np.int32)
    e = np.array([2**24], dtype=np.int32)

    host, device = _both(a, e, rtol=0.0, atol=0.0)

    assert not host.passed
    assert host.max_abs_diff == 1.0
    assert host == device


@pytest.mark.parametrize("dtype", [np.float32, np.float16])
@pytest.mark.parametrize("fails", [False, True])
def test_strided_parity_leaves_inputs_unchanged(dtype: type, fails: bool) -> None:
    rng = np.random.default_rng(0)
    expected = rng.standard_normal((4, 6)).astype(dtype)
    actual = (expected + dtype(1e-3)).astype(dtype)
    if fails:
        actual[3, 4] += dtype(1.0)
    a, e = actual[:, ::2], expected[:, ::2]
    before = (a.copy(), e.copy())

    host, device = _both(a, e, rtol=1e-2, atol=1e-2)

    assert host.passed is not fails
    assert host == device
    # torch.from_numpy shares memory, so this covers both paths.
    np.testing.assert_array_equal(a, before[0])
    np.testing.assert_array_equal(e, before[1])


def test_torch_actual_accepts_numpy_expected() -> None:
    torch = pytest.importorskip("torch")
    expected = np.array([1.0, 2.0], dtype=np.float32)

    result = compare(
        torch.tensor([1.0, 2.5], dtype=torch.bfloat16), expected, rtol=0.0, atol=0.1
    )

    assert not result.passed
    assert result.worst_index == (1,)


@pytest.mark.parametrize(
    "name", ["float8_e4m3fn", "float8_e5m2", "float8_e4m3fnuz", "float8_e5m2fnuz"]
)
def test_torch_fp8_actual_is_compared(name: str) -> None:
    """torch has no isfinite/abs for float8, so compare() must upcast it."""
    torch = pytest.importorskip("torch")
    fp8 = getattr(torch, name, None)
    if fp8 is None:
        pytest.skip(f"torch has no {name}")
    expected = np.array([1.0, 2.0, 0.5], dtype=np.float32)
    actual = torch.tensor([1.0, 2.0, 0.75]).to(fp8)

    result = compare(actual, expected, rtol=0.0, atol=0.1)

    assert not result.passed
    assert result.n_mismatch == 1 and result.worst_index == (2,)
    assert result.max_abs_diff == 0.25
    assert compare(actual[:2], expected[:2], rtol=0.0, atol=0.0).passed

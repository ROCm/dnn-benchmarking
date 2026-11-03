# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for compare(): verdicts, mismatch statistics, and host/device parity."""

import numpy as np
import pytest

from dnn_benchmarking.validation import compare


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

    result = compare(actual, expected, rtol=0.01, atol=1.0)

    assert result.n_mismatch == 2
    assert result.worst_index == (1,)


def test_mismatch_count_and_worst_index() -> None:
    expected = np.zeros((3, 4), dtype=np.float32)
    actual = expected.copy()
    actual[0, 1] = 0.5
    actual[2, 3] = -2.0  # worst
    actual[1, 0] = 0.05  # inside atol

    result = compare(actual, expected, rtol=0.0, atol=0.1)

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

    result = compare(actual, expected, rtol=0.2, atol=0.2)

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

    result = compare(actual, expected, rtol=1.0, atol=1.0)

    assert not result.passed
    assert result.max_abs_diff == float("inf")
    assert result.n_mismatch == result.n_total == 3
    name = "output" if side == "actual" else "reference"
    assert result.message == f"{name} contains NaN or Inf values"


@pytest.mark.parametrize("side, bad", [("actual", np.nan), ("expected", np.inf)])
def test_non_finite_value_in_a_later_block_fails(
    monkeypatch, side: str, bad: float
) -> None:
    """The finite checks must cover every block, not only the first."""
    from dnn_benchmarking.common import blocks

    monkeypatch.setattr(blocks, "BLOCK_ELEMENTS", 4)
    good = np.arange(1.0, 13.0, dtype=np.float32).reshape(3, 4)
    poisoned = good.copy()
    poisoned[-1, -1] = bad
    actual, expected = (poisoned, good) if side == "actual" else (good, poisoned)

    result = compare(actual, expected, rtol=1.0, atol=1.0)

    assert not result.passed
    name = "output" if side == "actual" else "reference"
    assert result.message == f"{name} contains NaN or Inf values"


def test_shape_mismatch_fails() -> None:
    result = compare(np.zeros((2, 3)), np.zeros((3, 2)), rtol=1.0, atol=1.0)

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

    result = compare(a, e, rtol=1e-3, atol=1e-3)

    assert result.max_abs_diff == 120000.0


def test_int32_compared_in_float64() -> None:
    """float32 would round 2**24 + 1 to 2**24 and hide the difference."""
    a = np.array([2**24 + 1], dtype=np.int32)
    e = np.array([2**24], dtype=np.int32)

    result = compare(a, e, rtol=0.0, atol=0.0)

    assert not result.passed
    assert result.max_abs_diff == 1.0


@pytest.mark.parametrize("dtype", [np.float32, np.float16])
@pytest.mark.parametrize("fails", [False, True])
def test_strided_inputs_are_left_unchanged(dtype: type, fails: bool) -> None:
    rng = np.random.default_rng(0)
    expected = rng.standard_normal((4, 6)).astype(dtype)
    actual = (expected + dtype(1e-3)).astype(dtype)
    if fails:
        actual[3, 4] += dtype(1.0)
    a, e = actual[:, ::2], expected[:, ::2]
    before = (a.copy(), e.copy())

    result = compare(a, e, rtol=1e-2, atol=1e-2)

    assert result.passed is not fails
    np.testing.assert_array_equal(a, before[0])
    np.testing.assert_array_equal(e, before[1])


@pytest.mark.parametrize("block_elements", [1, 4, 7, 1 << 22])
def test_block_boundaries_do_not_change_the_result(monkeypatch, block_elements) -> None:
    """compare() works one block at a time; the statistics, the first worst
    index and the caller's reference must not depend on the block size."""
    from dnn_benchmarking.common import blocks

    monkeypatch.setattr(blocks, "BLOCK_ELEMENTS", block_elements)
    expected = np.arange(1.0, 25.0, dtype=np.float16).reshape(2, 3, 4)
    actual = expected.astype(np.float32)
    actual[0, 2, 1] += 0.5  # the first of two equal worst differences
    actual[1, 1, 3] += 0.5
    actual[1, 0, 0] += 0.25
    reference = expected.copy()

    result = compare(actual, expected, rtol=0.0, atol=0.1)

    assert (result.n_mismatch, result.n_total) == (3, 24)
    assert result.max_abs_diff == 0.5
    assert result.worst_index == (0, 2, 1)
    assert result.max_rel_diff == pytest.approx(0.5 / 10.0)
    np.testing.assert_array_equal(expected, reference)

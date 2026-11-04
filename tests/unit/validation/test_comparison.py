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

    for result in _both(actual, expected, rtol=1.0, atol=1.0):
        assert not result.passed
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

    for result in _both(actual, expected, rtol=0.0, atol=0.1):
        assert (result.n_mismatch, result.n_total) == (3, 24)
        assert result.max_abs_diff == 0.5
        assert result.worst_index == (0, 2, 1)
        assert result.max_rel_diff == pytest.approx(0.5 / 10.0)
    np.testing.assert_array_equal(expected, reference)


def _varlen_pair():
    expected = np.ones((2, 2, 4, 3), dtype=np.float32)
    return expected.copy(), expected


def test_lengths_exclude_padding_from_the_comparison() -> None:
    actual, expected = _varlen_pair()
    actual[0, :, 2:, :] = np.nan  # batch 0 is valid up to 2
    actual[1, :, 3:, :] = 1e9  # batch 1 is valid up to 3

    host = compare(actual, expected, rtol=0.0, atol=0.0, lengths=[2, 3])
    assert host.passed
    assert host.n_total == (2 + 3) * 2 * 3
    assert host.max_abs_diff == 0.0
    torch = pytest.importorskip("torch")
    device = compare(
        torch.from_numpy(actual),
        torch.from_numpy(expected),
        rtol=0.0,
        atol=0.0,
        lengths=[2, 3],
    )
    assert device.passed and device.n_total == host.n_total


def test_lengths_still_fail_a_wrong_value_inside_the_valid_range() -> None:
    actual, expected = _varlen_pair()
    actual[0, :, 2:, :] = 5.0  # padding: ignored
    actual[1, 1, 2, 1] = 4.0  # valid for batch 1 (length 3)

    host = compare(actual, expected, rtol=0.0, atol=0.1, lengths=[2, 3])

    assert not host.passed
    assert host.n_mismatch == 1
    assert host.worst_index == (1, 1, 2, 1)  # an index of the full tensor
    assert host.max_abs_diff == 3.0
    assert "1/30" in host.message


def test_lengths_report_the_cross_batch_maximum_and_the_first_worst_index() -> None:
    actual, expected = _varlen_pair()
    actual[0, 0, 0, 0] = 1.5  # abs 0.5, rel 0.5
    expected[1, 0, 0, 0] = 4.0
    actual[1, 0, 0, 0] = 4.5  # abs 0.5 (a tie), rel 0.125

    for result in _both(actual, expected, rtol=0.0, atol=0.0, lengths=[2, 3]):
        assert result.n_mismatch == 2
        assert result.max_abs_diff == 0.5
        assert result.max_rel_diff == pytest.approx(0.5)
        assert result.worst_index[0] == 0


def test_lengths_fail_a_nan_inside_the_valid_range() -> None:
    actual, expected = _varlen_pair()
    actual[1, 0, 0, 0] = np.nan

    result = compare(actual, expected, rtol=0.0, atol=0.0, lengths=[1, 4])

    assert not result.passed
    assert result.n_total == (1 + 4) * 2 * 3
    assert "NaN or Inf" in result.message


def test_zero_length_batches_contribute_nothing() -> None:
    actual, expected = _varlen_pair()
    actual[:] = np.nan

    result = compare(actual, expected, rtol=0.0, atol=0.0, lengths=[0, 0])

    assert result.passed
    assert result.n_total == 0


def test_lengths_reject_a_wrong_count_or_range() -> None:
    actual, expected = _varlen_pair()

    with pytest.raises(ValueError):
        compare(actual, expected, rtol=0.0, atol=0.0, lengths=[1])
    with pytest.raises(ValueError):
        compare(actual, expected, rtol=0.0, atol=0.0, lengths=[1, 5])


def test_matching_infinities_pass_only_when_requested() -> None:
    expected = np.array([1.0, -np.inf, 2.0], dtype=np.float32)
    actual = expected.copy()

    strict = compare(actual, expected, rtol=0.0, atol=0.0)
    lenient = compare(actual, expected, rtol=0.0, atol=0.0, matching_infinities=True)

    assert not strict.passed
    assert lenient.passed
    assert lenient.n_total == 3 and lenient.n_mismatch == 0
    assert lenient.max_abs_diff == 0.0


def test_matching_infinities_still_compare_the_finite_values() -> None:
    expected = np.array([1.0, -np.inf, 2.0], dtype=np.float32)
    actual = np.array([1.0, -np.inf, 3.0], dtype=np.float32)

    for result in _both(actual, expected, rtol=0.0, atol=0.1, matching_infinities=True):
        assert result.n_mismatch == 1
        assert result.worst_index == (2,)


@pytest.mark.parametrize(
    "actual, expected",
    [
        ([np.inf], [-np.inf]),  # opposite signs
        ([np.inf], [1.0]),  # infinity against a finite value
        ([1.0], [-np.inf]),
        ([np.nan], [np.nan]),  # NaN is never equal
    ],
)
def test_matching_infinities_reject_everything_else(actual, expected) -> None:
    a = np.array(actual, dtype=np.float32)
    e = np.array(expected, dtype=np.float32)

    for result in _both(a, e, rtol=0.0, atol=0.0, matching_infinities=True):
        assert not result.passed
        assert result.max_abs_diff == float("inf")

# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for BenchmarkStats and noise_warnings."""

import pytest

from dnn_benchmarking.reporting.statistics import BenchmarkStats, noise_warnings


class TestBenchmarkStats:
    def test_from_timings_summarizes_samples(self) -> None:
        stats = BenchmarkStats.from_timings([1.0, 2.0, 3.0, 4.0, 5.0])
        assert stats.n == 5
        assert stats.mean_ms == 3.0
        assert stats.median_ms == 3.0
        assert stats.min_ms == 1.0
        assert stats.max_ms == 5.0

    def test_single_value_has_zero_spread(self) -> None:
        stats = BenchmarkStats.from_timings([5.0])
        assert stats.std_ms == 0.0
        assert stats.cv == 0.0
        assert stats.iqr_ms == 0.0

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            BenchmarkStats.from_timings([])

    def test_std_is_sample_std(self) -> None:
        stats = BenchmarkStats.from_timings([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
        assert stats.std_ms == pytest.approx((32 / 7) ** 0.5)
        assert stats.cv == pytest.approx(stats.std_ms / 5.0)

    def test_percentiles_and_iqr(self) -> None:
        stats = BenchmarkStats.from_timings(list(range(1, 101)))
        assert stats.p25_ms == pytest.approx(25.75)
        assert stats.median_ms == 51.0  # upper median sorted[n // 2]
        assert stats.p75_ms == pytest.approx(75.25)
        assert stats.p95_ms == pytest.approx(95.05)
        assert stats.iqr_ms == pytest.approx(49.5)

    def test_median_even_count_is_upper_middle_sample(self) -> None:
        """Even counts take sorted[n // 2] like rocKE / Solera, not the mean of
        the two middle values, so the median is always an observed sample."""
        assert BenchmarkStats.from_timings([4.0, 1.0, 3.0, 2.0]).median_ms == 3.0

    def test_to_dict_nulls_p95_below_20_samples(self) -> None:
        assert BenchmarkStats.from_timings([1.0] * 19).to_dict()["p95_ms"] is None
        assert BenchmarkStats.from_timings([1.0] * 20).to_dict()["p95_ms"] == 1.0

    def test_to_dict_omits_derivable_spread(self) -> None:
        d = BenchmarkStats.from_timings(list(range(1, 101))).to_dict()
        assert "cv" not in d and "iqr_ms" not in d


class TestNoiseWarnings:
    def test_quiet_samples_have_no_warnings(self) -> None:
        assert noise_warnings(BenchmarkStats.from_timings([1.0] * 50)) == []

    def test_wide_iqr_is_flagged_with_enough_samples(self) -> None:
        stats = BenchmarkStats.from_timings([1.0, 1.3] * 10)
        assert any(w.startswith("noisy:") for w in noise_warnings(stats))

    @pytest.mark.parametrize(
        "timings, flag, expected",
        [
            ([19.0] * 10 + [20.0] * 10, "noisy:", False),  # IQR exactly 5%
            ([0.951] * 10 + [1.0] * 10, "noisy:", False),  # IQR 4.9% of median
            ([0.949] * 10 + [1.0] * 10, "noisy:", True),  # IQR 5.1% of median
            ([0.9] * 5 + [1.0] * 4, "noisy:", False),  # wide, but 9 samples
            ([0.9] * 5 + [1.0] * 5, "noisy:", True),  # wide, 10 samples
            ([1.0] * 9 + [1.9], "outlier:", False),
            ([1.0] * 9 + [2.1], "outlier:", True),
        ],
    )
    def test_thresholds(self, timings, flag, expected) -> None:
        warnings = noise_warnings(BenchmarkStats.from_timings(timings))
        assert any(w.startswith(flag) for w in warnings) == expected

    def test_few_slow_samples_do_not_make_a_tight_core_noisy(self) -> None:
        """High CV from two slow samples; the robust spread stays zero."""
        stats = BenchmarkStats.from_timings([1.0] * 18 + [1.9, 1.9])
        assert stats.cv > 0.05
        assert not any(w.startswith("noisy:") for w in noise_warnings(stats))

    def test_outlier_max_is_flagged(self) -> None:
        stats = BenchmarkStats.from_timings([1.0] * 9 + [2.5])
        assert any(w.startswith("outlier:") for w in noise_warnings(stats))

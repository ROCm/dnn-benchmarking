# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Benchmark statistics calculation."""

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


@dataclass
class BenchmarkStats:
    """Summary statistics of one timing sample set (milliseconds).

    No field has a default: a partially built instance must fail loudly
    rather than report a silent ``0.0``.

    Attributes:
        n: Number of samples.
        mean_ms: Arithmetic mean.
        std_ms: Sample standard deviation (ddof=1; 0 for n == 1).
        cv: Coefficient of variation, ``std_ms / mean_ms``.
        min_ms: Minimum.
        p25_ms: 25th percentile.
        median_ms: Upper median ``sorted(timings)[n // 2]`` (rocKE / Solera
            definition; always an observed sample); the headline number.
        p75_ms: 75th percentile.
        p95_ms: 95th percentile (serialized as null when n < 20).
        max_ms: Maximum.
    """

    n: int
    mean_ms: float
    std_ms: float
    cv: float
    min_ms: float
    p25_ms: float
    median_ms: float
    p75_ms: float
    p95_ms: float
    max_ms: float

    @classmethod
    def from_timings(cls, timings: Sequence[float]) -> "BenchmarkStats":
        """Summarize a non-empty list of timings in milliseconds.

        Raises:
            ValueError: If ``timings`` is empty.
        """
        if len(timings) == 0:
            raise ValueError("timings list cannot be empty")
        arr = np.asarray(timings, dtype=np.float64)
        mean = float(arr.mean())
        std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
        p25, p75, p95 = (float(v) for v in np.percentile(arr, [25, 75, 95]))
        median = float(np.sort(arr)[arr.size // 2])
        return cls(
            n=int(arr.size),
            mean_ms=mean,
            std_ms=std,
            cv=std / mean if mean > 0 else 0.0,
            min_ms=float(arr.min()),
            p25_ms=p25,
            median_ms=median,
            p75_ms=p75,
            p95_ms=p95,
            max_ms=float(arr.max()),
        )

    @property
    def iqr_ms(self) -> float:
        """Interquartile range."""
        return self.p75_ms - self.p25_ms

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a JSON-ready dict; p95 is null below 20 samples."""
        d: Dict[str, Any] = asdict(self)
        if self.n < 20:
            d["p95_ms"] = None
        d["iqr_ms"] = self.iqr_ms
        return d


# Heuristic thresholds. NOISY_IQR: quiet MI210 runs show IQR 1-2 % of the
# median (docs/usage.md example), so 5 % marks a disturbed run, not normal
# spread. OUTLIER_RATIO: launch jitter and clock ramp on a quiet GPU stay well
# under 2x (the slowest sample in the docs/methodology.md matmul run is 1.3x).
NOISY_IQR = 0.05
OUTLIER_RATIO = 2.0


def noise_warnings(stats: BenchmarkStats) -> List[str]:
    """Flag dispersion a reader should know about; samples are never trimmed.

    Noise uses the robust spread IQR/median, so a few slow samples (reported
    by the outlier flag) do not mark an otherwise tight distribution noisy.
    """
    warnings: List[str] = []
    if stats.n >= 10 and stats.median_ms > 0:
        spread = stats.iqr_ms / stats.median_ms
        if spread > NOISY_IQR:
            warnings.append(f"noisy: IQR {spread:.1%} of median")
    if stats.median_ms > 0 and stats.max_ms > OUTLIER_RATIO * stats.median_ms:
        warnings.append(f"outlier: max {stats.max_ms / stats.median_ms:.1f}x median")
    return warnings


@dataclass
class TimingInfo:
    """How a plan's timings were measured (serialized as the plan's ``timing``).

    Attributes mirror ``execution.timing.Measurement`` minus the samples and
    the run-wide ``cache_mode`` and ``timing_block`` (see ``run.config``).
    """

    mode: str
    timer: str
    warmup_iters: int
    first_call_ms: float
    capped: bool = False
    fallback_reason: Optional[str] = None

    @classmethod
    def from_measurement(cls, m: Any) -> "TimingInfo":
        """Copy the provenance fields of an ``execution.timing.Measurement``."""
        return cls(
            mode=m.mode,
            timer=m.timer,
            warmup_iters=m.warmup_iters,
            first_call_ms=m.first_call_ms,
            capped=m.capped,
            fallback_reason=m.fallback_reason,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a JSON-ready dict."""
        return asdict(self)

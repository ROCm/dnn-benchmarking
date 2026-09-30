# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Benchmark statistics calculation."""

import json
import socket
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


def _get_hostname() -> str:
    """Get machine hostname for result identification."""
    return socket.gethostname()


def _get_timestamp() -> str:
    """Get current UTC timestamp in ISO format."""
    return datetime.now(timezone.utc).isoformat()


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
        median_ms: Median; the headline number.
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
        p25, median, p75, p95 = (float(v) for v in np.percentile(arr, [25, 50, 75, 95]))
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


NOISY_CV = 0.05
OUTLIER_RATIO = 2.0


def noise_warnings(stats: BenchmarkStats) -> List[str]:
    """Flag dispersion a reader should know about; samples are never trimmed."""
    warnings: List[str] = []
    if stats.n >= 10 and stats.cv > NOISY_CV:
        warnings.append(f"noisy: CV {stats.cv:.1%}")
    if stats.median_ms > 0 and stats.max_ms > OUTLIER_RATIO * stats.median_ms:
        warnings.append(f"outlier: max {stats.max_ms / stats.median_ms:.1f}x median")
    return warnings


@dataclass
class TimingInfo:
    """How a row's timings were measured (serialized as the row's ``timing``).

    Attributes mirror ``execution.timing.Measurement`` minus the samples.
    """

    mode: str
    backend: str
    cache_mode: str
    warmup_iters: int
    first_call_ms: float
    capped: bool = False
    fallback_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a JSON-ready dict."""
        return asdict(self)


@dataclass
class BenchmarkMetadata:
    """Metadata for benchmark results export.

    Attributes:
        timestamp: UTC timestamp when benchmark was run.
        graph_name: Name/identifier of the graph being benchmarked.
        graph_path: Path to the graph JSON file.
        warmup_iters: Number of warmup iterations.
        benchmark_iters: Number of benchmark iterations.
        engine_id: Engine ID used for execution.
        timing_backend: GPU timer backend used ("hip" or "").
        execution_backend: Execution backend used ("hipdnn", "pytorch", or "").
        pytorch_sdpa_backend_requested: Requested PyTorch SDPA backend; None
            when PyTorch SDPA selection does not apply.
        pytorch_rocm_fa_library_requested: Requested ROCm Flash Attention
            implementation preference; None when not requested.
        hostname: Machine hostname where benchmark was run.
    """

    timestamp: str = field(default_factory=_get_timestamp)
    graph_name: str = ""
    graph_path: str = ""
    warmup_iters: int = 0
    benchmark_iters: int = 0
    engine_id: int = 0
    timing_backend: str = ""
    execution_backend: str = ""
    pytorch_sdpa_backend_requested: Optional[str] = None
    pytorch_rocm_fa_library_requested: Optional[str] = None
    hostname: str = field(default_factory=_get_hostname)


@dataclass
class BenchmarkResult:
    """Raw benchmark timing results.

    Holds host (submission) timings and optional kernel (GPU event) timings,
    with metadata for cross-device comparison. End-to-end time, when needed,
    is host + kernel.

    Attributes:
        host_timings: List of host-side submission times in milliseconds.
        kernel_timings: Optional list of GPU kernel times in milliseconds.
        metadata: Optional metadata for result identification and comparison.
    """

    host_timings: List[float]
    kernel_timings: Optional[List[float]] = None
    metadata: Optional[BenchmarkMetadata] = None

    @property
    def has_kernel_timings(self) -> bool:
        """Check if kernel timings are available."""
        return self.kernel_timings is not None and len(self.kernel_timings) > 0

    @property
    def timing_backend(self) -> str:
        """Return backend used for kernel timing."""
        if self.metadata:
            return self.metadata.timing_backend
        return ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Returns:
            Dictionary representation of the result.
        """
        result: Dict[str, Any] = {
            "host_timings": self.host_timings,
            "kernel_timings": self.kernel_timings,
        }
        if self.metadata:
            result["metadata"] = asdict(self.metadata)
        return result

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string.

        Args:
            indent: JSON indentation level.

        Returns:
            JSON string representation.
        """
        return json.dumps(self.to_dict(), indent=indent)

    def save_json(self, path: str) -> None:
        """Save results to JSON file.

        Args:
            path: Path to the output JSON file.
        """
        Path(path).write_text(self.to_json())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BenchmarkResult":
        """Create from dictionary.

        Args:
            data: Dictionary with result data.

        Returns:
            BenchmarkResult instance.
        """
        metadata = None
        if "metadata" in data and data["metadata"]:
            metadata_dict = dict(data["metadata"])
            legacy_gpu_backend = metadata_dict.pop("gpu_backend", None)
            if "timing_backend" not in metadata_dict and legacy_gpu_backend is not None:
                metadata_dict["timing_backend"] = legacy_gpu_backend
            metadata = BenchmarkMetadata(**metadata_dict)
        return cls(
            host_timings=data["host_timings"],
            kernel_timings=data.get("kernel_timings"),
            metadata=metadata,
        )

    @classmethod
    def load_json(cls, path: str) -> "BenchmarkResult":
        """Load results from JSON file.

        Args:
            path: Path to the JSON file.

        Returns:
            BenchmarkResult loaded from file.
        """
        data = json.loads(Path(path).read_text())
        return cls.from_dict(data)


@dataclass
class CombinedBenchmarkStats:
    """Combined statistics for host and kernel timing.

    Attributes:
        host_stats: Statistics from host-side submission timing.
        kernel_stats: Optional statistics from GPU kernel timing.
    """

    host_stats: BenchmarkStats
    kernel_stats: Optional[BenchmarkStats] = None

    @classmethod
    def from_result(cls, result: BenchmarkResult) -> "CombinedBenchmarkStats":
        """Create combined stats from a BenchmarkResult.

        Args:
            result: BenchmarkResult with host and optional kernel timings.

        Returns:
            CombinedBenchmarkStats with calculated statistics.
        """
        host = BenchmarkStats.from_timings(result.host_timings)
        kernel = (
            BenchmarkStats.from_timings(result.kernel_timings)
            if result.has_kernel_timings
            else None
        )
        return cls(host_stats=host, kernel_stats=kernel)

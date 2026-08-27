# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Suite result data model with JSON serialization.

Top-level structure is graph-first: SuiteResult contains metadata plus a
list of GraphResult, each holding ProviderEngineResult entries with timing
statistics and correctness data. Error entries carry status + message only.
"""

import json
import os
import socket
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, NamedTuple, Optional

from ..common import torch_support
from ..metrics.arch import detect_arch
from .statistics import BenchmarkStats


@dataclass
class CorrectnessResult:
    """Correctness tracking for a single provider/engine run.

    Attributes:
        execution_success: Did the run complete without error?
        tolerance_match: Within rtol/atol? None if execution failed or
            reference provider unavailable.
        rtol: Relative tolerance used.
        atol: Absolute tolerance used.
        max_abs_diff: Maximum absolute difference (if comparison was performed).
        max_rel_diff: Maximum relative difference (if comparison was performed).
        error_message: Explanation when tolerance_match is None.
    """

    execution_success: bool
    tolerance_match: Optional[bool]
    rtol: float
    atol: float
    max_abs_diff: Optional[float] = None
    max_rel_diff: Optional[float] = None
    error_message: Optional[str] = None

    @property
    def passed(self) -> bool:
        """Overall pass = executed successfully AND tolerance matched."""
        return self.execution_success and (self.tolerance_match is True)

    @classmethod
    def failed(
        cls, rtol: float, atol: float, error_message: str
    ) -> "CorrectnessResult":
        """Build a CorrectnessResult representing an execution failure.

        Used at error/skip sites where the GPU run did not complete and no
        comparison was performed.

        Args:
            rtol: Relative tolerance configured for comparison.
            atol: Absolute tolerance configured for comparison.
            error_message: Explanation of the failure.

        Returns:
            CorrectnessResult with execution_success=False and
            tolerance_match=None.
        """
        return cls(
            execution_success=False,
            tolerance_match=None,
            rtol=rtol,
            atol=atol,
            error_message=error_message,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        d: Dict[str, Any] = {
            "passed": self.passed,
            "execution_success": self.execution_success,
            "tolerance_match": self.tolerance_match,
            "rtol": self.rtol,
            "atol": self.atol,
        }
        if self.max_abs_diff is not None:
            d["max_abs_diff"] = self.max_abs_diff
        if self.max_rel_diff is not None:
            d["max_rel_diff"] = self.max_rel_diff
        if self.error_message is not None:
            d["error_message"] = self.error_message
        return d


@dataclass
class OracleResult:
    """Post-tuning timed run for one engine row.

    Timing fields come from a clean benchmark pass executed after the
    autotuning sweep selected and activated a winner. Sweep-time
    measurements are diagnostics only and appear as ``sweep_min_time_ms``.

    Attributes:
        engine_id: Engine the winning plan belongs to.
        engine_name: Engine name reported by the sweep.
        plan_name: Name of the plan hipDNN activated after tuning.
        compiled_plan_index: Index of the winning plan in the compiled set.
        rank: Sweep rank of the winner (0 is fastest).
        sweep_min_time_ms: Winner's fastest sweep iteration. hipDNN ranks
            candidates on this value. Diagnostic only: the reported oracle
            timing is the post-tuning benchmark pass, not the sweep.
        candidates_benchmarked: Number of candidates that benchmarked
            successfully.
        knob_settings: ``{"knob_id": str, "value": int|float|str}`` entries
            for knobs the sweep set explicitly. Normally empty: the
            compiled-plan candidate path records only explicitly-set knobs
            and this flow sets none, so ``[]`` means "engine defaults".
        cpu_build_time_ms: Graph build time of the autotune-capable build.
        gpu_kernel_stats: GPU kernel timing of the post-tuning run.
        host_stats: Host-side timing of the post-tuning run.
    """

    engine_id: int
    engine_name: str
    plan_name: str
    compiled_plan_index: int
    rank: int
    sweep_min_time_ms: float
    candidates_benchmarked: int
    knob_settings: List[Dict[str, Any]]
    cpu_build_time_ms: Optional[float] = None
    gpu_kernel_stats: Optional[BenchmarkStats] = None
    host_stats: Optional[BenchmarkStats] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "engine_id": self.engine_id,
            "engine_name": self.engine_name,
            "plan_name": self.plan_name,
            "compiled_plan_index": self.compiled_plan_index,
            "rank": self.rank,
            "sweep_min_time_ms": self.sweep_min_time_ms,
            "candidates_benchmarked": self.candidates_benchmarked,
            "knob_settings": list(self.knob_settings),
            "cpu_build_time_ms": self.cpu_build_time_ms,
            "gpu_kernel_stats": (
                self.gpu_kernel_stats.to_dict() if self.gpu_kernel_stats else None
            ),
            "host_stats": self.host_stats.to_dict() if self.host_stats else None,
        }


@dataclass
class OracleDelta:
    """OOTB-vs-oracle comparison for one engine row.

    Attributes:
        basis: Which timing pair the comparison used.
        ootb_mean_ms: Mean of the heuristic-selected run.
        oracle_mean_ms: Mean of the post-tuning run.
        delta_ms: ``ootb_mean_ms - oracle_mean_ms``; positive means the
            oracle is faster.
        speedup: ``ootb_mean_ms / oracle_mean_ms``.
    """

    basis: Literal["gpu_kernel", "host"]
    ootb_mean_ms: float
    oracle_mean_ms: float
    delta_ms: float
    speedup: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "basis": self.basis,
            "ootb_mean_ms": self.ootb_mean_ms,
            "oracle_mean_ms": self.oracle_mean_ms,
            "delta_ms": self.delta_ms,
            "speedup": self.speedup,
        }


@dataclass
class ProviderEngineResult:
    """Result for one provider/engine combination on one graph.

    Attributes:
        provider: Provider name.
        engine_id: Engine ID used.
        status: One of 'success', 'error', 'skipped'.
        engine_version: Loaded provider plugin version.
        started_at: UTC timestamp immediately before this engine run.
        role: ``engine`` for hipDNN engine rows, ``reference`` for timed
            validation-provider rows that are shown for comparison but are not
            counted as pass/fail engine combinations.
        cpu_build_time_ms: CPU graph-build time.
        gpu_kernel_stats: GPU kernel timing statistics.
        host_stats: Host-side submission timing statistics.
        correctness: Correctness comparison result.
        error_message: Error message only (no partial timing on error).
        skip_reason: Reason this combination was skipped.
        warnings: Non-fatal warnings for this row, such as reference timing
            paths that are not solely built-in PyTorch operators.
        workspace_bytes: hipDNN-reserved workspace size in bytes.
        analytical_flops: Total analytical FLOPs across compute nodes
            (None for purely bandwidth-bound graphs).
        analytical_flops_partial: True when at least one node type was
            unrecognised — ``analytical_flops`` then reflects only the
            recognised compute nodes.
        analytical_io_bytes: Sum of non-virtual tensor sizes (bytes).
        derived_tflops_per_s: Throughput derived from analytical_flops
            and the GPU kernel mean time.
        derived_gbytes_per_s: Bandwidth derived from analytical_io_bytes
            and the GPU kernel mean time.
        cpu_user_time_per_iter_us: User-space CPU time per benchmark
            iteration in microseconds (rusage delta over the loop,
            divided by ``benchmark_iters``). Mostly Python dispatch +
            sync overhead.
        cpu_kernel_time_per_iter_us: Kernel-space CPU time per
            benchmark iteration in microseconds. Usually near zero;
            useful only as a spike diagnostic (heavy syscalls / page
            faults during the loop).
        vram_used_mb: Total process-wide GPU VRAM allocated at the
            end of this engine's benchmark loop, sampled via amdsmi.
            Workspace + I/O buffers + any allocator cache. Distinct
            from ``workspace_bytes`` which is only the engine's
            scratchpad request. Note this is process-wide and may
            include cached allocations from previous engines on the
            same graph.
        extra_metrics: Opt-in profiling payload from rocprofv3 PMC /
            traces, perf, and rocprof-compute roofline. None when no
            opt-in profiling flag was supplied.
        oracle: Post-tuning result for this engine row. Set only when
            ``--oracle`` was requested and tuning succeeded.
        oracle_delta: OOTB-vs-oracle comparison. None when either side
            lacks comparable statistics.
        oracle_error: Why tuning produced no result for this row.
            Mutually exclusive with ``oracle``.

    Note:
        Process RSS, host RAM availability, and the volatile parts of
        an amdsmi snapshot (power/clocks/temps/utilisation) are *not*
        per-engine — they're either flat across a suite (RSS) or
        misleading post-loop snapshots (power/clocks/temps lag the
        workload). They live on :class:`SuiteMetadata` instead. VRAM
        is the exception: it's stable during the loop and varies
        meaningfully across engines, so it lives here.
    """

    _VALID_STATUSES = {"success", "error", "skipped"}
    _VALID_ROLES = {"engine", "reference"}

    provider: str
    engine_id: int
    status: Literal["success", "error", "skipped"]
    engine_version: str = "unavailable"
    started_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    role: Literal["engine", "reference"] = "engine"
    plugin_path: Optional[str] = None
    cpu_build_time_ms: Optional[float] = None
    gpu_kernel_stats: Optional[BenchmarkStats] = None
    host_stats: Optional[BenchmarkStats] = None
    elapsed_time_ms: float = 0.0
    correctness: Optional[CorrectnessResult] = None
    error_message: Optional[str] = None
    skip_reason: Optional[str] = None
    warnings: Optional[List[str]] = None
    # Always-on metrics (None when collection failed or skipped)
    workspace_bytes: Optional[int] = None
    analytical_flops: Optional[int] = None
    analytical_flops_partial: bool = False
    analytical_io_bytes: Optional[int] = None
    derived_tflops_per_s: Optional[float] = None
    derived_gbytes_per_s: Optional[float] = None
    cpu_user_time_per_iter_us: Optional[float] = None
    cpu_kernel_time_per_iter_us: Optional[float] = None
    vram_used_mb: Optional[float] = None
    # Opt-in profiling payload (rocprofv3 PMC / trace, perf, roofline).
    extra_metrics: Optional[Dict[str, Any]] = None
    # Opt-in oracle (auto-tuned) comparison payload.
    oracle: Optional[OracleResult] = None
    oracle_delta: Optional[OracleDelta] = None
    oracle_error: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate status field."""
        if self.status not in self._VALID_STATUSES:
            raise ValueError(
                f"Invalid status '{self.status}'. "
                f"Must be one of: {self._VALID_STATUSES}"
            )
        if self.role not in self._VALID_ROLES:
            raise ValueError(
                f"Invalid role '{self.role}'. " f"Must be one of: {self._VALID_ROLES}"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Error entries serialize status + error_message only, no timing.
        Correctness, when present, is always serialized regardless of status
        so that error/skip entries can carry their failure context.

        Always-on metrics are emitted only inside the ``success`` branch
        and only when non-None, so the JSON shape stays compact for
        runs where probes were unavailable.
        """
        d: Dict[str, Any] = {
            "provider": self.provider,
            "engine_id": self.engine_id,
            "engine_name": self.provider,
            "engine_version": self.engine_version,
            "started_at": self.started_at,
            "status": self.status,
        }
        if self.role != "engine":
            d["role"] = self.role
        if self.plugin_path is not None:
            d["plugin_path"] = self.plugin_path
        if self.warnings:
            d["warnings"] = list(self.warnings)
        # extra_metrics is exclusively populated by the opt-in
        # profiling orchestrator, which the suite runner only fires on
        # the success path. Asserting the invariant here makes it
        # load-bearing: if a future caller routes profiling onto a
        # non-success status, the assertion fires and forces a
        # decision (emit always when present, or gate explicitly)
        # rather than silently dropping the slice from the JSON.
        if self.status != "success":
            assert self.extra_metrics is None, (
                f"extra_metrics is set on status={self.status!r}; "
                "the orchestrator only runs on success today, so this "
                "indicates either a new caller or a regression in the "
                "success-gating in suite_runner.run_single_provider_engine"
            )
        if self.status == "success":
            d["cpu_build_time_ms"] = self.cpu_build_time_ms
            d["gpu_kernel_stats"] = (
                self.gpu_kernel_stats.to_dict() if self.gpu_kernel_stats else None
            )
            d["host_stats"] = self.host_stats.to_dict() if self.host_stats else None
            d["elapsed_time_ms"] = self.elapsed_time_ms

            # Always-on metric fields — emit only when populated.
            if self.workspace_bytes is not None:
                d["workspace_bytes"] = self.workspace_bytes
            if self.analytical_flops is not None:
                d["analytical_flops"] = self.analytical_flops
            if self.analytical_flops_partial:
                d["analytical_flops_partial"] = True
            if self.analytical_io_bytes is not None:
                d["analytical_io_bytes"] = self.analytical_io_bytes
            if self.derived_tflops_per_s is not None:
                d["derived_tflops_per_s"] = self.derived_tflops_per_s
            if self.derived_gbytes_per_s is not None:
                d["derived_gbytes_per_s"] = self.derived_gbytes_per_s
            if self.cpu_user_time_per_iter_us is not None:
                d["cpu_user_time_per_iter_us"] = self.cpu_user_time_per_iter_us
            if self.cpu_kernel_time_per_iter_us is not None:
                d["cpu_kernel_time_per_iter_us"] = self.cpu_kernel_time_per_iter_us
            if self.vram_used_mb is not None:
                d["vram_used_mb"] = self.vram_used_mb
            if self.extra_metrics is not None:
                d["extra_metrics"] = self.extra_metrics
            if self.oracle is not None:
                d["oracle"] = self.oracle.to_dict()
            if self.oracle_delta is not None:
                d["oracle_delta"] = self.oracle_delta.to_dict()
            if self.oracle_error is not None:
                d["oracle_error"] = self.oracle_error
        elif self.status == "error":
            d["error_message"] = self.error_message
        elif self.status == "skipped":
            d["skip_reason"] = self.skip_reason

        if self.correctness is not None:
            d["correctness"] = self.correctness.to_dict()
        return d


def build_oracle_delta(
    ootb: ProviderEngineResult, oracle: OracleResult
) -> Optional[OracleDelta]:
    """Compare an OOTB engine row against its post-tuning oracle run.

    Prefers GPU kernel time; falls back to host time when either side has
    no kernel statistics.

    Args:
        ootb: The heuristic-selected row.
        oracle: The post-tuning result for the same engine row.

    Returns:
        An OracleDelta, or None when no comparable statistics pair exists
        or the oracle mean is zero.
    """
    basis: Literal["gpu_kernel", "host"]
    if ootb.gpu_kernel_stats is not None and oracle.gpu_kernel_stats is not None:
        basis = "gpu_kernel"
        ootb_mean = ootb.gpu_kernel_stats.mean_ms
        oracle_mean = oracle.gpu_kernel_stats.mean_ms
    elif ootb.host_stats is not None and oracle.host_stats is not None:
        basis = "host"
        ootb_mean = ootb.host_stats.mean_ms
        oracle_mean = oracle.host_stats.mean_ms
    else:
        return None

    if oracle_mean == 0.0:
        return None

    return OracleDelta(
        basis=basis,
        ootb_mean_ms=ootb_mean,
        oracle_mean_ms=oracle_mean,
        delta_ms=ootb_mean - oracle_mean,
        speedup=ootb_mean / oracle_mean,
    )


class StatusCounts(NamedTuple):
    """Counts of provider/engine results bucketed by outcome.

    Attributes:
        passed: Successful runs whose correctness either matched or was not
            checked (tolerance_match is True or None).
        failed: Successful runs whose correctness comparison failed
            (tolerance_match is False).
        skipped: Runs marked as 'skipped' (unsupported combinations).
        errored: Runs marked as 'error' (hard failure).
    """

    passed: int
    failed: int
    skipped: int
    errored: int


@dataclass
class GraphResult:
    """Result for one graph across all provider/engine combinations.

    Attributes:
        graph_name: Name of the graph.
        graph_path: File path to the graph JSON.
        results: List of ProviderEngineResult for each combination.
    """

    graph_name: str
    graph_path: str
    results: List[ProviderEngineResult]
    engine_ids: List[int] = field(default_factory=list)

    def is_no_engine_graph(self) -> bool:
        """True when this graph result represents a no-engine outcome."""
        return len(self.engine_ids) == 0

    def count_by_status(self) -> StatusCounts:
        """Bucket results into pass/fail/skip/error counts.

        A 'pass' is a successful run whose correctness check either passed or
        was not performed (tolerance_match is True or None). A 'fail' is a
        successful run whose correctness check explicitly failed
        (tolerance_match is False).

        Returns:
            StatusCounts with the four bucket counts.
        """
        engine_results = [r for r in self.results if r.role == "engine"]
        passed = sum(
            1
            for r in engine_results
            if r.status == "success"
            and (r.correctness is None or r.correctness.tolerance_match is not False)
        )
        failed = sum(
            1
            for r in engine_results
            if r.status == "success"
            and r.correctness is not None
            and r.correctness.tolerance_match is False
        )
        skipped = sum(1 for r in engine_results if r.status == "skipped")
        errored = sum(1 for r in engine_results if r.status == "error")
        return StatusCounts(
            passed=passed, failed=failed, skipped=skipped, errored=errored
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Graph entry with 'results' array of provider/engine entries.
        """
        return {
            "graph_name": self.graph_name,
            "graph_path": self.graph_path,
            "results": [r.to_dict() for r in self.results],
        }


@dataclass
class SuiteMetadata:
    """Suite-level summary plus environment info.

    Attributes:
        timestamp: UTC timestamp when suite was run.
        hostname: Machine hostname.
        total_graphs: Total number of graphs in suite.
        total_combinations: Total provider/engine combinations across all graphs.
        pass_combinations: Combinations that passed correctness.
        fail_combinations: Combinations that failed correctness.
        skip_combinations: Combinations skipped (unsupported).
        error_combinations: Combinations that errored during execution.
        pytorch_sdpa_backend_requested: Requested PyTorch SDPA backend for
            PyTorch timing or reference validation; None when PyTorch was not
            selected.
        pytorch_rocm_fa_library_requested: Requested ROCm Flash Attention
            implementation preference; None when not requested.
        rocm_version: ROCm/HIP version string (None on CUDA hosts).
        cuda_version: CUDA toolkit version the torch wheel was built
            against (None on ROCm hosts).
        cudnn_version: cuDNN version string, decoded major.minor.patch
            (None on ROCm hosts or when cuDNN is unavailable).
        gpu_model: GPU model name.
        gpu_arch: GPU gfx target (e.g. "gfx90a", "gfx942"). Useful for
            keying arch-specific PMC counter sets when analysing the
            JSON downstream. "unknown" when detection failed.
        python_version: Python version string.
        hipdnn_version: hipDNN version string.
        cpu_model: CPU model string from /proc/cpuinfo.
        cpu_count: Number of logical CPUs.
        numa_nodes: Number of NUMA nodes on the host.
        total_ram_gb: Total host RAM in GiB.
        kernel_version: Linux kernel version.
        gpu_compute_units: Number of GPU compute units.
        gpu_hbm_gb: Total GPU HBM in GiB.
        gpu_pcie_link: PCIe link speed/width string (e.g. "gen4 x16").
        amdgpu_driver_version: amdgpu driver version string.
        host_rss_mb: Process RSS in MiB sampled once at suite end. Flat
            across the suite — purely a steady-state footprint figure
            (Python interpreter + torch + ROCm + hipDNN + buffers).
        host_ram_available_mb: Host RAM available system-wide at suite
            end, in MiB. Capacity hint, not a workload metric.
        vram_used_mb: GPU VRAM currently allocated to this process at
            suite end, via amdsmi. Reflects steady-state allocation, not
            per-kernel peak.
        vram_total_mb: Total VRAM on the GPU at suite end, via amdsmi.
        hipdnn_selection_env: hipDNN cache/benchmarking environment
            variables sampled at suite end, recorded only for ``--oracle``
            runs. A ``None`` value means the variable was not set, which
            is the load-bearing signal for cache-affected OOTB timings.
    """

    timestamp: str
    hostname: str
    total_graphs: int
    total_combinations: int
    pass_combinations: int
    fail_combinations: int
    skip_combinations: int
    error_combinations: int
    pytorch_sdpa_backend_requested: Optional[str] = None
    pytorch_rocm_fa_library_requested: Optional[str] = None
    rocm_version: Optional[str] = None
    cuda_version: Optional[str] = None
    cudnn_version: Optional[str] = None
    gpu_model: Optional[str] = None
    gpu_arch: Optional[str] = None
    python_version: Optional[str] = None
    hipdnn_version: Optional[str] = None
    cpu_model: Optional[str] = None
    cpu_count: Optional[int] = None
    numa_nodes: Optional[int] = None
    total_ram_gb: Optional[float] = None
    kernel_version: Optional[str] = None
    gpu_compute_units: Optional[int] = None
    gpu_hbm_gb: Optional[float] = None
    gpu_pcie_link: Optional[str] = None
    amdgpu_driver_version: Optional[str] = None
    host_rss_mb: Optional[float] = None
    host_ram_available_mb: Optional[float] = None
    vram_used_mb: Optional[float] = None
    vram_total_mb: Optional[float] = None
    hipdnn_selection_env: Optional[Dict[str, Optional[str]]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        d: Dict[str, Any] = {
            "timestamp": self.timestamp,
            "hostname": self.hostname,
            "total_graphs": self.total_graphs,
            "total_combinations": self.total_combinations,
            "pass_combinations": self.pass_combinations,
            "fail_combinations": self.fail_combinations,
            "skip_combinations": self.skip_combinations,
            "error_combinations": self.error_combinations,
            "pytorch_sdpa_backend_requested": self.pytorch_sdpa_backend_requested,
            "pytorch_rocm_fa_library_requested": (
                self.pytorch_rocm_fa_library_requested
            ),
            "rocm_version": self.rocm_version,
            "cuda_version": self.cuda_version,
            "cudnn_version": self.cudnn_version,
            "gpu_model": self.gpu_model,
            "gpu_arch": self.gpu_arch,
            "python_version": self.python_version,
            "hipdnn_version": self.hipdnn_version,
            "cpu_model": self.cpu_model,
            "cpu_count": self.cpu_count,
            "numa_nodes": self.numa_nodes,
            "total_ram_gb": self.total_ram_gb,
            "kernel_version": self.kernel_version,
            "gpu_compute_units": self.gpu_compute_units,
            "gpu_hbm_gb": self.gpu_hbm_gb,
            "gpu_pcie_link": self.gpu_pcie_link,
            "amdgpu_driver_version": self.amdgpu_driver_version,
            "host_rss_mb": self.host_rss_mb,
            "host_ram_available_mb": self.host_ram_available_mb,
            "vram_used_mb": self.vram_used_mb,
            "vram_total_mb": self.vram_total_mb,
        }
        if self.hipdnn_selection_env is not None:
            d["hipdnn_selection_env"] = dict(self.hipdnn_selection_env)
        return d


@dataclass
class SuiteResult:
    """Top-level suite result with graph-first nesting.

    Attributes:
        metadata: Suite-level metadata.
        graphs: List of per-graph results.
    """

    metadata: SuiteMetadata
    graphs: List[GraphResult]

    @classmethod
    def from_graph_results(
        cls,
        graph_results: List[GraphResult],
        total_graphs: int,
        *,
        pytorch_sdpa_backend_requested: Optional[str] = None,
        pytorch_rocm_fa_library_requested: Optional[str] = None,
        oracle: bool = False,
    ) -> "SuiteResult":
        """Build a SuiteResult from per-graph results with auto-computed metadata."""
        env_info = collect_environment_info()
        total_pass = total_fail = total_skip = total_error = 0
        for gr in graph_results:
            c = gr.count_by_status()
            total_pass += c.passed
            total_fail += c.failed
            total_skip += c.skipped
            total_error += c.errored

        # Suite-end host/VRAM snapshot. Process RSS and VRAM are flat
        # across the suite once libraries are loaded; sampling once here
        # keeps the (graph, engine) results free of redundant noise.
        # Failures fall back to None — never block metadata construction.
        host_rss_mb: Optional[float] = None
        host_ram_available_mb: Optional[float] = None
        vram_used_mb: Optional[float] = None
        vram_total_mb: Optional[float] = None
        try:
            from ..metrics.host import host_memory_snapshot

            mem = host_memory_snapshot()
            host_rss_mb = mem.get("host_rss_mb")
            host_ram_available_mb = mem.get("host_ram_available_mb")
        except Exception:
            pass
        try:
            from ..metrics.gpu_smi import GpuSmiProbe

            snap = GpuSmiProbe().snapshot()
            vram_used_mb = snap.get("vram_used_mb")
            vram_total_mb = snap.get("vram_total_mb")
        except Exception:
            pass

        # Oracle runs record the hipDNN switches that change what either
        # pass measures, so an implausible OOTB-vs-oracle gap is
        # diagnosable from the JSON alone.
        hipdnn_selection_env: Optional[Dict[str, Optional[str]]] = None
        if oracle:
            hipdnn_selection_env = {
                name: os.environ.get(name)
                for name in (
                    "HIPDNN_DISABLE_EXACT_ENGINE_CACHE",
                    "HIPDNN_CACHE_DIR",
                    "HIPDNN_DISABLE_CACHE",
                    "HIPDNN_FORCE_BENCHMARKING",
                )
            }

        metadata = SuiteMetadata(
            timestamp=datetime.now(timezone.utc).isoformat(),
            hostname=socket.gethostname(),
            total_graphs=total_graphs,
            total_combinations=total_pass + total_fail + total_skip + total_error,
            pass_combinations=total_pass,
            fail_combinations=total_fail,
            skip_combinations=total_skip,
            error_combinations=total_error,
            pytorch_sdpa_backend_requested=pytorch_sdpa_backend_requested,
            pytorch_rocm_fa_library_requested=pytorch_rocm_fa_library_requested,
            rocm_version=env_info.get("rocm_version"),
            cuda_version=env_info.get("cuda_version"),
            cudnn_version=env_info.get("cudnn_version"),
            gpu_model=env_info.get("gpu_model"),
            gpu_arch=env_info.get("gpu_arch"),
            python_version=env_info.get("python_version"),
            hipdnn_version=env_info.get("hipdnn_version"),
            cpu_model=env_info.get("cpu_model"),
            cpu_count=env_info.get("cpu_count"),
            numa_nodes=env_info.get("numa_nodes"),
            total_ram_gb=env_info.get("total_ram_gb"),
            kernel_version=env_info.get("kernel_version"),
            gpu_compute_units=env_info.get("gpu_compute_units"),
            gpu_hbm_gb=env_info.get("gpu_hbm_gb"),
            gpu_pcie_link=env_info.get("gpu_pcie_link"),
            amdgpu_driver_version=env_info.get("amdgpu_driver_version"),
            host_rss_mb=host_rss_mb,
            host_ram_available_mb=host_ram_available_mb,
            vram_used_mb=vram_used_mb,
            vram_total_mb=vram_total_mb,
            hipdnn_selection_env=hipdnn_selection_env,
        )
        return cls(metadata=metadata, graphs=graph_results)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Returns dict with "metadata" and "graphs" keys.
        """
        return {
            "metadata": self.metadata.to_dict(),
            "graphs": [g.to_dict() for g in self.graphs],
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string.

        Args:
            indent: JSON indentation level.

        Returns:
            JSON string representation.
        """
        return json.dumps(self.to_dict(), indent=indent)

    def save_json(self, path: str) -> None:
        """Write suite results to JSON file.

        Args:
            path: Output file path.
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_json())


def _format_cudnn_version(raw: Optional[int]) -> Optional[str]:
    """Decode the packed integer from ``torch.backends.cudnn.version()``.

    torch exposes cuDNN's version only as a packed int (e.g. ``92000``),
    so we decode it to a human-readable ``major.minor.patch`` string.
    cuDNN 9+ packs as ``major*10000 + minor*100 + patch``; earlier
    releases used ``major*1000 + minor*100 + patch``. Returns ``None``
    for a missing/zero version.
    """
    if not raw:
        return None
    if raw >= 90000:
        major, minor, patch = raw // 10000, (raw % 10000) // 100, raw % 100
    else:
        major, minor, patch = raw // 1000, (raw % 1000) // 100, raw % 100
    return f"{major}.{minor}.{patch}"


def collect_environment_info() -> Dict[str, Any]:
    """Collect ROCm/CUDA/GPU/Python/hipDNN versions plus static machine metadata.

    Combines the legacy version probes (torch hip, hipdnn_frontend) with
    the host- and GPU-side static info from
    :func:`metrics.machine_info.collect_machine_info`. On a CUDA host the
    ROCm/hipDNN probes stay ``None`` and ``cuda_version``/``cudnn_version``
    are populated instead (and vice versa on ROCm). Never raises; missing
    values are ``None`` so :class:`SuiteMetadata` can serialise a stable
    shape.
    """
    python_version = (
        f"{sys.version_info.major}.{sys.version_info.minor}"
        f".{sys.version_info.micro}"
    )
    rocm_version: Optional[str] = None
    cuda_version: Optional[str] = None
    cudnn_version: Optional[str] = None
    gpu_model: Optional[str] = None
    hipdnn_version: Optional[str] = None

    try:
        if torch_support.module_available():
            import torch

            if hasattr(torch.version, "hip"):
                rocm_version = torch.version.hip
            if torch_support.is_cuda_build():
                cuda_version = getattr(torch.version, "cuda", None)
                try:
                    cudnn_version = _format_cudnn_version(
                        torch.backends.cudnn.version()
                    )
                except Exception:
                    cudnn_version = None
            if torch_support.gpu_available():
                gpu_model = torch.cuda.get_device_name(0)
    except Exception:
        pass

    try:
        import hipdnn_frontend

        hipdnn_version = getattr(hipdnn_frontend, "__version__", None)
    except ImportError:
        pass

    # gfx target via the same torch -> rocminfo -> "unknown" chain used
    # by metrics.rocprof_pmc, so the JSON output and the PMC keying
    # agree on what arch this run targeted. detect_arch() never raises —
    # it returns "unknown" when no GPU is detectable.
    gpu_arch = detect_arch()

    info: Dict[str, Any] = {
        "rocm_version": rocm_version,
        "cuda_version": cuda_version,
        "cudnn_version": cudnn_version,
        "gpu_model": gpu_model,
        "gpu_arch": gpu_arch,
        "python_version": python_version,
        "hipdnn_version": hipdnn_version,
    }

    try:
        from ..metrics.machine_info import collect_machine_info

        info.update(collect_machine_info())
    except Exception:
        # machine_info already routes failures through warn_once; avoid
        # propagating any unexpected exception out of metadata building.
        pass

    return info

# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Suite result data model and result JSON schema v2.

Top-level structure is graph-first: SuiteResult holds run info, the
environment and a list of GraphResult, each holding ProviderEngineResult
rows. Python attribute names are stable; the v2 JSON key names are applied
only in ``to_dict``. Every key is always present (null when not
applicable) so consumers never probe for key presence.
"""

import csv
import hashlib
import io
import json
import math
import os
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

from .. import __version__
from .statistics import BenchmarkStats, TimingInfo

SUITE_RESULT_SCHEMA_VERSION = 2

# Keys of run.config / environment that are always emitted (null when the
# producer did not supply them). Extra producer keys are kept as-is.
RUN_CONFIG_KEYS = (
    "backend",
    "engine_filter",
    "plugin_paths",
    "warmup_iters",
    "iters",
    "min_time_ms",
    "cache_mode",
    "timing_block",
    "seed",
    "validate",
    "rtol",
    "atol",
    "oracle_mode",
    "autotune",
    "cache_dir",
    "pytorch_sdpa_backend",
    "pytorch_rocm_fa_library",
    "metrics_tier",
    "profiling",
)
PROFILING_KEYS = ("pmc", "emit_trace", "perf", "roofline")
ENVIRONMENT_KEYS = (
    "hostname",
    "cpu_model",
    "cpu_count",
    "numa_nodes",
    "total_ram_gb",
    "kernel_version",
    "gpu_model",
    "gpu_arch",
    "gpu_compute_units",
    "gpu_hbm_gb",
    "gpu_pcie_link",
    "amdgpu_driver_version",
    "gpu_power_cap_w",
    "gpu_max_sclk_mhz",
    "gpu_compute_partition",
    "rocm_version",
    "cuda_version",
    "cudnn_version",
    "hipdnn_version",
    "python_version",
    "torch_version",
    "amdsmi_available",
    "selection_env",
    "end_of_run",
)
END_OF_RUN_KEYS = (
    "host_rss_mb",
    "host_ram_available_mb",
    "vram_used_mb",
    "vram_total_mb",
)
ROW_COLUMNS = (
    "gpu_arch",
    "graph_name",
    "graph_id",
    "provider",
    "role",
    "engine_id",
    "engine_name",
    "status",
    "verdict",
    "kernel_median_ms",
    "kernel_cv",
    "host_median_ms",
    "n",
    "timing_mode",
    "cache_mode",
    "timing_block",
    "seed",
    "tflops",
    "gbps",
    "workspace_bytes",
    "max_abs_diff",
    "message",
)


def engine_id_hex(engine_id: Optional[int]) -> Optional[str]:
    """Format an engine id as ``0x%016X`` of its unsigned 64-bit value.

    hipDNN engine ids are signed 64-bit hashes; above 2**53 they lose
    precision in float-based JSON readers, so the schema carries them as
    hex strings.
    """
    if engine_id is None:
        return None
    return f"0x{engine_id & 0xFFFFFFFFFFFFFFFF:016X}"


def graph_id_for(graph_json: Dict[str, Any]) -> str:
    """Stable join key for a graph: sha256 of its canonical JSON, 12 hex chars."""
    canonical = json.dumps(graph_json, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]


def _with_keys(d: Optional[Dict[str, Any]], keys: tuple) -> Dict[str, Any]:
    """Return ``d`` with every key in ``keys`` present (null when missing)."""
    return {**dict.fromkeys(keys), **(d or {})}


def _finite(obj: Any) -> Any:
    """Map NaN/inf to None recursively so the output is strict JSON."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _finite(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_finite(v) for v in obj]
    return obj


def _stats_dict(stats: Optional[BenchmarkStats]) -> Optional[Dict[str, Any]]:
    return stats.to_dict() if stats is not None else None


@dataclass
class CorrectnessResult:
    """Correctness tracking for a single provider/engine run.

    Attributes:
        tolerance_match: Within rtol/atol? None when not checked (no
            reference requested or reference unavailable).
        rtol: Relative tolerance used.
        atol: Absolute tolerance used.
        max_abs_diff: Maximum absolute difference (if comparison was performed).
        max_rel_diff: Maximum relative difference (if comparison was performed).
        error_message: Explanation when tolerance_match is None or False.
        n_mismatch: Elements outside tolerance, summed over outputs.
        n_total: Elements compared, summed over outputs.
        worst_output_uid: Tensor UID of the output with the largest diff.
    """

    tolerance_match: Optional[bool]
    rtol: float
    atol: float
    max_abs_diff: Optional[float] = None
    max_rel_diff: Optional[float] = None
    error_message: Optional[str] = None
    n_mismatch: Optional[int] = None
    n_total: Optional[int] = None
    worst_output_uid: Optional[int] = None

    @property
    def passed(self) -> bool:
        """Validation ran and every output was within tolerance."""
        return self.tolerance_match is True

    @property
    def explicitly_failed(self) -> bool:
        """True only when validation ran and returned a negative verdict.

        ``passed`` is also False for ``tolerance_match=None``, which means
        "not checked" -- the default when no reference was requested. Callers
        that gate on a real failure must use this instead, or a plain run
        looks like a suite of failures.
        """
        return self.tolerance_match is False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to the v2 ``correctness`` object."""
        return {
            "match": self.tolerance_match,
            "rtol": self.rtol,
            "atol": self.atol,
            "max_abs_diff": self.max_abs_diff,
            "max_rel_diff": self.max_rel_diff,
            "n_mismatch": self.n_mismatch,
            "n_total": self.n_total,
            "worst_output_uid": self.worst_output_uid,
            "message": self.error_message,
        }


@dataclass(kw_only=True)
class PlanResult:
    """One built and timed plan: the OOTB plan or the tuned plan.

    The row is the OOTB plan (JSON ``ootb``) and ``OracleResult`` is the
    tuned plan (JSON ``oracle``), so both serialize to the same object.

    Attributes:
        cpu_build_time_ms: CPU time to build the plan (JSON ``build_ms``).
        timing: How the samples were measured.
        gpu_kernel_stats: Device time per launch (JSON ``kernel``).
        host_stats: Host submit time per launch (JSON ``host``).
        workspace_bytes: Workspace that hipDNN reserved for the plan.
        derived_tflops_per_s: Graph FLOPs over the kernel median.
        derived_gbytes_per_s: Graph I/O bytes over the kernel median.
        correctness: Comparison of this plan's outputs with the reference.
    """

    cpu_build_time_ms: Optional[float] = None
    timing: Optional[TimingInfo] = None
    gpu_kernel_stats: Optional[BenchmarkStats] = None
    host_stats: Optional[BenchmarkStats] = None
    workspace_bytes: Optional[int] = None
    derived_tflops_per_s: Optional[float] = None
    derived_gbytes_per_s: Optional[float] = None
    correctness: Optional[CorrectnessResult] = None

    def plan_dict(self) -> Dict[str, Any]:
        """Convert to the v2 plan object (``ootb`` and ``oracle``)."""
        return {
            "build_ms": self.cpu_build_time_ms,
            "timing": self.timing.to_dict() if self.timing is not None else None,
            "kernel": _stats_dict(self.gpu_kernel_stats),
            "host": _stats_dict(self.host_stats),
            "workspace_bytes": self.workspace_bytes,
            "tflops": self.derived_tflops_per_s,
            "gbps": self.derived_gbytes_per_s,
            "correctness": (
                self.correctness.to_dict() if self.correctness is not None else None
            ),
        }


@dataclass
class OracleResult(PlanResult):
    """Tuned plan for one engine row, plus how it was selected.

    ``sweep_min_time_ms`` is the fastest single selection-sweep iteration.
    Reported timing comes from the later ``gpu_kernel_stats`` or ``host_stats``
    benchmark.

    Candidate counts describe compiled plans, not provider-internal kernels.
    ``exhaustive_enabled`` means the selected engine advertises
    ``global.benchmarking`` and the run requested it. Providers can reuse
    cached selections, so it does not prove a fresh search occurred.

    ``warm_baseline_*`` contains the OOTB plan re-timed after selection. The
    delta uses this warm measurement, not the row's earlier OOTB timing.
    ``correctness`` is the tuned plan's verdict; the row retains the OOTB
    verdict. The ``PlanResult`` fields use the row's analytical FLOPs and
    I/O bytes with the tuned kernel median, like the row's own throughputs.
    """

    plan_name: str
    compiled_plan_index: int
    rank: int
    sweep_min_time_ms: float
    compiled_plans_benchmarked: int
    compiled_plans_total: int
    compiled_plans_failed: int
    knob_settings: List[Dict[str, Any]]
    exhaustive_requested: bool = False
    exhaustive_supported: bool = False
    warm_baseline_gpu_kernel_stats: Optional[BenchmarkStats] = None
    warm_baseline_host_stats: Optional[BenchmarkStats] = None
    warm_baseline_derived_tflops_per_s: Optional[float] = None

    @property
    def exhaustive_enabled(self) -> bool:
        """True when a capable provider was built for exhaustive selection."""
        return self.exhaustive_requested and self.exhaustive_supported

    @property
    def tuning_available(self) -> bool:
        """Return whether this pass had a tuning alternative."""
        return self.compiled_plans_total > 1 or self.exhaustive_enabled

    def to_dict(self) -> Dict[str, Any]:
        """Convert to the v2 ``oracle`` object: the plan object plus tuning."""
        return {
            **self.plan_dict(),
            "tuning_available": self.tuning_available,
            "plan_name": self.plan_name,
            "compiled_plan_index": self.compiled_plan_index,
            "rank": self.rank,
            "sweep_min_time_ms": self.sweep_min_time_ms,
            "compiled_plans_benchmarked": self.compiled_plans_benchmarked,
            "compiled_plans_total": self.compiled_plans_total,
            "compiled_plans_failed": self.compiled_plans_failed,
            "knob_settings": list(self.knob_settings),
            "exhaustive_requested": self.exhaustive_requested,
            "exhaustive_enabled": self.exhaustive_enabled,
            "exhaustive_supported": self.exhaustive_supported,
            "baseline_kernel": _stats_dict(self.warm_baseline_gpu_kernel_stats),
            "baseline_host": _stats_dict(self.warm_baseline_host_stats),
            "baseline_tflops": self.warm_baseline_derived_tflops_per_s,
        }


@dataclass
class OracleDelta:
    """Warm heuristic baseline vs tuned run for one engine row.

    Both sides are measured after the autotuning sweep, back to back on the
    same buffers, so device warmth is common to them and the ratio isolates
    the plan change. This is deliberately not the row's headline OOTB
    timing: that one is measured before the sweep exists and is the
    "what you get out of the box" number, which at low ``--warmup`` can sit
    well above steady state and would inflate the speedup.

    Attributes:
        basis: Timing pair compared; always ``kernel`` (device median).
        baseline_median_ms: Median of the heuristic plan, re-timed post-sweep.
        oracle_median_ms: Median of the post-tuning run.
        delta_ms: ``baseline_median_ms - oracle_median_ms``; positive means
            the oracle is faster.
        speedup: ``baseline_median_ms / oracle_median_ms``.
    """

    basis: Literal["kernel"]
    baseline_median_ms: float
    oracle_median_ms: float
    delta_ms: float
    speedup: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "basis": self.basis,
            "baseline_median_ms": self.baseline_median_ms,
            "oracle_median_ms": self.oracle_median_ms,
            "delta_ms": self.delta_ms,
            "speedup": self.speedup,
        }


Verdict = Literal["passed", "failed", "unchecked", "reference", "skipped", "error"]


@dataclass
class ProviderEngineResult(PlanResult):
    """Result for one provider/engine combination on one graph.

    The inherited ``PlanResult`` fields describe the OOTB plan (JSON
    ``ootb``).

    Attributes:
        provider: Backend, ``hipdnn`` or ``pytorch``.
        engine_id: Engine ID used; None when the row has no hipDNN engine.
        status: One of 'success', 'error', 'skipped'.
        engine_version: Loaded provider plugin version.
        started_at: UTC timestamp immediately before this engine run.
        role: ``engine`` for engine rows, ``reference`` for timed
            validation-provider rows that are shown for comparison but are not
            counted as pass/fail engine combinations.
        plugin_path: Plugin the engine was loaded from.
        elapsed_time_ms: Wall time of the whole row (build, timing,
            validation, oracle, profiling).
        error_message: Why the row errored.
        skip_reason: Why the row was skipped.
        warnings: Non-fatal warnings for this row (noise, throttling, ...).
        analytical_flops: Total analytical FLOPs across compute nodes
            (None for purely bandwidth-bound graphs).
        analytical_flops_partial: True when at least one node type was
            unrecognised; ``analytical_flops`` then covers only the
            recognised compute nodes.
        analytical_io_bytes: Sum of non-virtual tensor sizes (bytes).
        vram_used_mb: Process-wide VRAM allocated at the end of this
            engine's benchmark loop (may include cached allocations from
            earlier engines on the same graph).
        extra_metrics: Opt-in profiling payload (rocprofv3 PMC / trace,
            perf, roofline).
        oracle: Tuned plan; set only for oracle runs that tuned.
        oracle_delta: Warm-baseline vs tuned comparison.
        oracle_error: Why tuning produced no result; exclusive with oracle.
        clocks_before: GPU clocks sampled right before the timed loop.
        clocks_after: GPU clocks sampled right after the timed loop.
        engine_name: Display name of the engine (e.g. MIOPEN_ENGINE).
    """

    _VALID_STATUSES = {"success", "error", "skipped"}
    _VALID_ROLES = {"engine", "reference"}

    provider: str
    engine_id: Optional[int]
    status: Literal["success", "error", "skipped"]
    engine_version: str = "unavailable"
    started_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    role: Literal["engine", "reference"] = "engine"
    plugin_path: Optional[str] = None
    elapsed_time_ms: float = 0.0
    error_message: Optional[str] = None
    skip_reason: Optional[str] = None
    warnings: Optional[List[str]] = None
    analytical_flops: Optional[int] = None
    analytical_flops_partial: bool = False
    analytical_io_bytes: Optional[int] = None
    vram_used_mb: Optional[float] = None
    extra_metrics: Optional[Dict[str, Any]] = None
    oracle: Optional[OracleResult] = None
    oracle_delta: Optional[OracleDelta] = None
    oracle_error: Optional[str] = None
    clocks_before: Optional[Dict[str, Any]] = None
    clocks_after: Optional[Dict[str, Any]] = None
    engine_name: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate status and role."""
        if self.status not in self._VALID_STATUSES:
            raise ValueError(
                f"Invalid status '{self.status}'. "
                f"Must be one of: {self._VALID_STATUSES}"
            )
        if self.role not in self._VALID_ROLES:
            raise ValueError(
                f"Invalid role '{self.role}'. Must be one of: {self._VALID_ROLES}"
            )

    @classmethod
    def error_row(
        cls,
        provider: str,
        engine_id: Optional[int],
        message: str,
        *,
        engine_name: Optional[str] = None,
        role: Literal["engine", "reference"] = "engine",
        plugin_path: Optional[str] = None,
    ) -> "ProviderEngineResult":
        """Build an ``error`` row carrying only its reason."""
        return cls(
            provider=provider,
            engine_id=engine_id,
            status="error",
            error_message=message,
            engine_name=engine_name,
            role=role,
            plugin_path=plugin_path,
        )

    @classmethod
    def skipped_row(
        cls,
        provider: str,
        engine_id: Optional[int],
        reason: str,
        *,
        engine_name: Optional[str] = None,
        role: Literal["engine", "reference"] = "engine",
        plugin_path: Optional[str] = None,
    ) -> "ProviderEngineResult":
        """Build a ``skipped`` row carrying only its reason."""
        return cls(
            provider=provider,
            engine_id=engine_id,
            status="skipped",
            skip_reason=reason,
            engine_name=engine_name,
            role=role,
            plugin_path=plugin_path,
        )

    @property
    def verdict(self) -> Verdict:
        """Single outcome label used by counts, console and JSON.

        ``unchecked`` is a successful run whose output was not validated
        (no correctness, or ``tolerance_match`` None); it is not a pass.
        """
        if self.status == "error":
            return "error"
        if self.status == "skipped":
            return "skipped"
        if self.role == "reference":
            return "reference"
        c = self.correctness
        if c is not None and c.explicitly_failed:
            return "failed"
        if c is not None and c.passed:
            return "passed"
        return "unchecked"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to the v2 row object (every key always present).

        ``ootb`` and ``oracle`` are the same plan object; ``ootb`` is null
        when the row did not run (``error`` or ``skipped`` rows carry only
        their reason).
        """
        return {
            "provider": self.provider,
            "role": self.role,
            "engine": {
                "id": engine_id_hex(self.engine_id),
                "name": self.engine_name,
                "version": self.engine_version,
                "plugin_path": self.plugin_path,
            },
            "status": self.status,
            "verdict": self.verdict,
            "message": self.error_message or self.skip_reason,
            "started_at": self.started_at,
            "elapsed_s": self.elapsed_time_ms / 1000.0,
            "metrics": {
                "flops": self.analytical_flops,
                "flops_partial": self.analytical_flops_partial,
                "io_bytes": self.analytical_io_bytes,
                "vram_mb": self.vram_used_mb,
                "clocks_before": self.clocks_before,
                "clocks_after": self.clocks_after,
            },
            "ootb": self.plan_dict() if self.status == "success" else None,
            "oracle": (
                {
                    **self.oracle.to_dict(),
                    "delta": self.oracle_delta.to_dict() if self.oracle_delta else None,
                }
                if self.oracle is not None
                else None
            ),
            "oracle_error": self.oracle_error,
            "warnings": list(self.warnings or []),
            "extra_metrics": self.extra_metrics,
        }


def build_oracle_delta(oracle: OracleResult) -> Optional[OracleDelta]:
    """Compare the warm heuristic baseline against the tuned run by kernel median.

    Both operands come from ``oracle``: the sweep-adjacent re-timing of the
    heuristic plan and the post-tuning run. The row's own OOTB timing is
    deliberately not used: it is measured before the sweep, so at low
    ``--warmup`` it can sit above steady state and report a speedup that is
    accumulated warmup rather than a better plan.

    Returns None when either side lacks kernel statistics or either median
    is non-positive.
    """
    if oracle.warm_baseline_gpu_kernel_stats is None or oracle.gpu_kernel_stats is None:
        return None
    baseline = oracle.warm_baseline_gpu_kernel_stats.median_ms
    tuned = oracle.gpu_kernel_stats.median_ms
    if baseline <= 0.0 or tuned <= 0.0:
        return None
    return OracleDelta(
        basis="kernel",
        baseline_median_ms=baseline,
        oracle_median_ms=tuned,
        delta_ms=baseline - tuned,
        speedup=baseline / tuned,
    )


@dataclass
class GraphResult:
    """Result for one graph across all provider/engine combinations.

    Attributes:
        graph_name: Name of the graph.
        graph_path: File path to the graph JSON.
        results: One row per provider/engine combination.
        engine_ids: Engines applicable to the graph; empty means none.
        graph_id: Join key from :func:`graph_id_for`.
        error: Graph-level failure (load, discovery, input generation).
        message: Why no engine applied, when ``status`` is ``no_engines``.
    """

    graph_name: str
    graph_path: str
    results: List[ProviderEngineResult]
    engine_ids: List[int] = field(default_factory=list)
    graph_id: Optional[str] = None
    error: Optional[str] = None
    message: Optional[str] = None

    @property
    def status(self) -> Literal["ok", "no_engines", "error"]:
        """``error`` on a graph-level failure, ``no_engines`` when none applied."""
        if self.error is not None:
            return "error"
        if not self.engine_ids:
            return "no_engines"
        return "ok"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to the v2 graph object."""
        return {
            "graph_id": self.graph_id,
            "graph_name": self.graph_name,
            "graph_path": self.graph_path,
            "status": self.status,
            "error": self.error,
            "message": self.message,
            "results": [r.to_dict() for r in self.results],
        }


@dataclass
class RunInfo:
    """How and when the suite was run (JSON ``run``).

    Attributes:
        started_at: UTC ISO timestamp of suite start.
        argv: Command line.
        config: Effective configuration (keys per :data:`RUN_CONFIG_KEYS`).
        finished_at: UTC ISO timestamp of suite end; None while running.
        complete: False for partial (interrupted or in-progress) results.
    """

    started_at: str
    argv: List[str]
    config: Dict[str, Any]
    finished_at: Optional[str] = None
    complete: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to the v2 ``run`` object."""
        config = _with_keys(self.config, RUN_CONFIG_KEYS)
        config["profiling"] = _with_keys(config["profiling"], PROFILING_KEYS)
        return {
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "complete": self.complete,
            "argv": list(self.argv),
            "config": config,
        }


@dataclass
class SuiteResult:
    """Top-level suite result (result JSON schema v2).

    Attributes:
        run: Run info and effective config.
        environment: Machine/software snapshot (keys per
            :data:`ENVIRONMENT_KEYS`).
        graphs: Per-graph results.
    """

    run: RunInfo
    environment: Dict[str, Any]
    graphs: List[GraphResult]

    def summary(self) -> Dict[str, int]:
        """Counts recomputed from the graphs on every call.

        Row buckets count ``role == 'engine'`` rows only (reference rows are
        not pass/fail combinations) and sum to ``rows``.
        """
        verdicts = [
            r.verdict for g in self.graphs for r in g.results if r.role == "engine"
        ]
        statuses = [g.status for g in self.graphs]
        return {
            "graphs": len(self.graphs),
            "rows": len(verdicts),
            "passed": verdicts.count("passed"),
            "unchecked": verdicts.count("unchecked"),
            "failed": verdicts.count("failed"),
            "skipped": verdicts.count("skipped"),
            "errors": verdicts.count("error"),
            "graph_errors": statuses.count("error"),
            "no_engine_graphs": statuses.count("no_engines"),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to the v2 document (not yet NaN-sanitized; see to_json)."""
        env = _with_keys(self.environment, ENVIRONMENT_KEYS)
        env["end_of_run"] = _with_keys(env["end_of_run"], END_OF_RUN_KEYS)
        return {
            "schema_version": SUITE_RESULT_SCHEMA_VERSION,
            "tool": {"name": "dnn-benchmarking", "version": __version__},
            "run": self.run.to_dict(),
            "environment": env,
            "summary": self.summary(),
            "graphs": [g.to_dict() for g in self.graphs],
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize to strict JSON: NaN/inf become null."""
        return json.dumps(_finite(self.to_dict()), indent=indent, allow_nan=False)

    def to_rows(self) -> List[Dict[str, Any]]:
        """Flatten to one dict per row with :data:`ROW_COLUMNS` keys.

        A graph without rows (graph-level error, no engines) still yields
        one row carrying the graph status and error.
        """
        doc = _finite(self.to_dict())
        arch = doc["environment"]["gpu_arch"]
        seed = doc["run"]["config"]["seed"]
        rows: List[Dict[str, Any]] = []
        for g in doc["graphs"]:
            base = {
                "gpu_arch": arch,
                "graph_name": g["graph_name"],
                "graph_id": g["graph_id"],
                "seed": seed,
            }
            if not g["results"]:
                rows.append(
                    {
                        **dict.fromkeys(ROW_COLUMNS),
                        **base,
                        "status": g["status"],
                        "message": g["error"] or g["message"],
                    }
                )
            for r in g["results"]:
                ootb = r["ootb"] or {}
                kernel, host = ootb.get("kernel") or {}, ootb.get("host") or {}
                timing = ootb.get("timing") or {}
                correctness = ootb.get("correctness") or {}
                rows.append(
                    {
                        **base,
                        "provider": r["provider"],
                        "role": r["role"],
                        "engine_id": r["engine"]["id"],
                        "engine_name": r["engine"]["name"],
                        "status": r["status"],
                        "verdict": r["verdict"],
                        "kernel_median_ms": kernel.get("median_ms"),
                        "kernel_cv": kernel.get("cv"),
                        "host_median_ms": host.get("median_ms"),
                        "n": kernel.get("n"),
                        "timing_mode": timing.get("mode"),
                        "cache_mode": timing.get("cache_mode"),
                        "timing_block": timing.get("timing_block"),
                        "tflops": ootb.get("tflops"),
                        "gbps": ootb.get("gbps"),
                        "workspace_bytes": ootb.get("workspace_bytes"),
                        "max_abs_diff": correctness.get("max_abs_diff"),
                        "message": r["message"],
                    }
                )
        return rows

    def write(self, path: Union[str, Path]) -> None:
        """Write atomically: JSON, or CSV when ``path`` ends in ``.csv``.

        The content is fully serialized before a temp file in the target
        directory is written and renamed over ``path``, so readers never see
        a partial file and a failed write leaves any previous file intact.
        """
        p = Path(path)
        if p.suffix.lower() == ".csv":
            buf = io.StringIO()
            writer = csv.DictWriter(buf, fieldnames=ROW_COLUMNS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(self.to_rows())
            text = buf.getvalue()
        else:
            text = self.to_json()
        p.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=p.parent, prefix=f".{p.name}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                f.write(text)
            # mkstemp creates 0600; give the result the mode open() would.
            # os.chmod on the path, not os.fchmod: Windows lacks fchmod.
            umask = os.umask(0)
            os.umask(umask)
            os.chmod(tmp, 0o666 & ~umask)
            os.replace(tmp, p)
        except BaseException:
            os.unlink(tmp)
            raise

    @staticmethod
    def load(path: Union[str, Path]) -> Dict[str, Any]:
        """Read a v2 result document.

        Raises:
            OSError: Unreadable file.
            ValueError: Not a schema v2 JSON result file.
        """
        with open(path) as f:
            try:
                doc = json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"{path}: not a JSON result file ({e}); compare needs "
                    "-o *.json output"
                ) from None
        version = doc.get("schema_version") if isinstance(doc, dict) else None
        if version != SUITE_RESULT_SCHEMA_VERSION:
            raise ValueError(
                f"{path}: unsupported result schema_version {version!r} "
                f"(expected {SUITE_RESULT_SCHEMA_VERSION}); regenerate it with this "
                "version of dnn-benchmark"
            )
        if not doc["run"]["complete"]:
            print(
                f"warning: {path}: run.complete is false; results are partial",
                file=sys.stderr,
            )
        return doc

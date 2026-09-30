# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Benchmark configuration dataclasses."""

import argparse
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional


class ReferenceProviderName(str, Enum):
    """Supported reference provider names."""

    NONE = "none"
    PYTORCH = "pytorch"


class ExecutionBackendName(str, Enum):
    """Supported execution backend names."""

    HIPDNN = "hipdnn"
    PYTORCH = "pytorch"


class PyTorchSdpaBackendName(str, Enum):
    """Supported PyTorch scaled-dot-product-attention backend selections."""

    DEFAULT = "default"
    FLASH = "flash"
    MATH = "math"
    EFFICIENT = "efficient"
    CUDNN = "cudnn"
    OVERRIDEABLE = "overrideable"


def _one_of(flag: str, enum_cls: type[Enum]) -> str:
    return f"{flag} must be one of: " + ", ".join(e.value for e in enum_cls)


def _normalize_pytorch_sdpa_settings(
    selection: PyTorchSdpaBackendName | str,
    rocm_fa_library: Optional[str],
) -> tuple[PyTorchSdpaBackendName, Optional[str]]:
    """Normalize and validate the strict SDPA category and ROCm preference."""
    try:
        selection = PyTorchSdpaBackendName(selection)
    except ValueError as e:
        raise ValueError(
            _one_of("--pytorch-sdpa-backend", PyTorchSdpaBackendName)
        ) from e

    if rocm_fa_library is not None and not isinstance(rocm_fa_library, str):
        raise ValueError("--pytorch-rocm-fa-library must be a string when set")
    if rocm_fa_library is not None and selection is not PyTorchSdpaBackendName.FLASH:
        raise ValueError(
            "--pytorch-rocm-fa-library requires --pytorch-sdpa-backend flash"
        )
    return selection, rocm_fa_library


CACHE_MODE_CHOICES = ("warm", "cold")


class OracleMode(str, Enum):
    """Oracle comparison depth."""

    OFF = "off"
    PLAN = "plan"
    EXHAUSTIVE = "exhaustive"


class MetricsTier(str, Enum):
    """Always-on metric collection tier."""

    BASIC = "basic"
    OFF = "off"


PMC_SET_CHOICES = ("basic", "memory", "flops", "all")
EMIT_TRACE_CHOICES = ("pftrace",)


@dataclass(frozen=True)
class TimingPolicy:
    """How one timed loop runs; shared by the hipDNN and PyTorch executors.

    Attributes:
        warmup_iters: Untimed enqueues before the loop. The loop always runs
            at least one untimed enqueue (priming) even when this is 0.
        iters: Minimum number of timed iterations.
        min_time_ms: Keep sampling until the summed device time reaches this
            budget (0 disables the time budget; the loop is then fixed-count).
        max_iters: Hard cap on timed iterations.
        cache_mode: ``warm`` reuses caches between iterations; ``cold``
            flushes L2/MALL before every timed iteration.
    """

    warmup_iters: int = 10
    iters: int = 100
    min_time_ms: float = 0.0
    max_iters: int = 10_000
    cache_mode: str = "warm"

    def __post_init__(self) -> None:
        """Validate loop bounds and the cache mode."""
        if self.warmup_iters < 0:
            raise ValueError("warmup_iters must be non-negative")
        if self.iters <= 0:
            raise ValueError("iters must be positive")
        if self.min_time_ms < 0:
            raise ValueError("min_time_ms must be non-negative")
        if self.max_iters < self.iters:
            raise ValueError("max_iters must be >= iters")
        if self.cache_mode not in CACHE_MODE_CHOICES:
            raise ValueError(
                f"cache_mode must be one of {CACHE_MODE_CHOICES}, "
                f"got {self.cache_mode!r}"
            )


@dataclass
class ValidationConfig:
    """Configuration for reference validation.

    Attributes:
        provider: Reference provider for correctness checking.
        rtol: Optional relative tolerance override. If only one of rtol/atol
            is set, it is used for both; if neither is set, validation uses
            dtype-aware defaults.
        atol: Optional absolute tolerance override.
    """

    provider: ReferenceProviderName = ReferenceProviderName.NONE
    rtol: Optional[float] = None
    atol: Optional[float] = None

    def __post_init__(self) -> None:
        """Validate and normalize configuration values."""
        try:
            self.provider = ReferenceProviderName(self.provider)
        except ValueError as e:
            raise ValueError(_one_of("--validate", ReferenceProviderName)) from e
        if self.rtol is not None and self.rtol < 0:
            raise ValueError("--rtol must be >= 0")
        if self.atol is not None and self.atol < 0:
            raise ValueError("--atol must be >= 0")

    @property
    def enabled(self) -> bool:
        """True when a non-`none` reference provider is selected."""
        return self.provider is not ReferenceProviderName.NONE

    @property
    def tolerance_override(self) -> Optional[tuple[float, float]]:
        """Return explicit validation tolerances, or None for dtype-aware defaults."""
        if self.rtol is None and self.atol is None:
            return None
        value = self.rtol if self.rtol is not None else self.atol
        return (
            self.rtol if self.rtol is not None else value,
            self.atol if self.atol is not None else value,
        )


@dataclass
class MetricsConfig:
    """Controls which metric sources are collected during benchmarking.

    Two collection modes:

    * Always-on (``tier``) — zero-overhead probes wrapped around the
      timed loop: analytical FLOPs/IO, workspace, host rusage + RAM,
      amdsmi GPU snapshot, machine metadata.
    * Opt-in profiling pass (``pmc_set``, ``emit_trace``, ``perf``,
      ``roofline``) — each runs the workload again under an external
      profiling tool. Kept separate from the timed pass so PMC sampling
      and roofline replay don't pollute the headline timing.

    Attributes:
        tier: ``basic`` enables always-on probes. ``off`` disables all
            metric collection — useful for clean engine-comparison timing.
        emit_trace: ``pftrace`` — re-run benchmark under
            ``rocprofv3 --kernel-trace --memory-copy-trace`` and write a
            trace file.
        pmc_set: ``basic`` | ``memory`` | ``flops`` | ``all`` — re-run
            under ``rocprofv3 --pmc <set>`` and fold per-kernel counter
            aggregates into ``extra_metrics["pmc"]``. ``all`` requires
            ``pmc_allow_multipass`` because the union of sets crosses the
            single-pass replay budget on most arches.
        perf: Re-run wrapped in ``perf stat -x,`` to collect CPU cycles
            and instructions. Kernel-space events are dropped silently
            when ``/proc/sys/kernel/perf_event_paranoid > 1``.
        roofline: Re-run under ``rocprof-compute profile --roof-only``
            to capture empirical HBM/compute ceilings at rocprof-
            compute's default datatype (FP32). The CSV outputs
            (``roofline.csv``, ``sysinfo.csv``) and the workload
            directory land in ``extra_metrics["roofline"]`` as
            ``roofline_csv`` / ``sysinfo_csv`` / ``workload_path``. The
            PDF/HTML plot is rendered post-hoc by the user via
            ``rocprof-compute analyze --path <workload_path>
            [--roofline-data-type FP16]``; we don't expose a profile-
            time datatype knob because rocprof-compute's ``profile``
            mode doesn't accept one.
        pmc_allow_multipass: Required for ``--pmc all``. Without it,
            ``all`` is rejected at config-build time because the rocprofv3
            multi-pass replay budget is easy to exceed and the resulting
            multi-pass replay has been observed to hang for minutes on
            sub-second workloads.
        profiling_output_dir: Root directory for profiling artefacts.
            ``None`` until the orchestrator resolves a default
            (``./profiling-output/<utc-timestamp>/``) at suite start.
        profiling_timeout_s: Wall-clock budget (seconds) for each external
            profiler subprocess. Default 600 s; ``0`` disables. Sized to
            absorb a heavy graph under multi-pass PMC replay on a slow
            host while still bounding a wedged child so the suite doesn't
            block indefinitely.
    """

    tier: MetricsTier = MetricsTier.BASIC
    emit_trace: Optional[str] = None
    pmc_set: Optional[str] = None
    perf: bool = False
    roofline: bool = False
    pmc_allow_multipass: bool = False
    profiling_output_dir: Optional[Path] = None
    profiling_timeout_s: int = 600

    def __post_init__(self) -> None:
        """Validate choices, the multipass opt-in, and the timeout."""
        try:
            self.tier = MetricsTier(self.tier)
        except ValueError as e:
            raise ValueError(_one_of("--metrics-tier", MetricsTier)) from e
        if self.emit_trace is not None and self.emit_trace not in EMIT_TRACE_CHOICES:
            raise ValueError(
                "--emit-trace must be one of: " + ", ".join(EMIT_TRACE_CHOICES)
            )
        if self.pmc_set is not None and self.pmc_set not in PMC_SET_CHOICES:
            raise ValueError("--pmc must be one of: " + ", ".join(PMC_SET_CHOICES))
        # The 'all' PMC set unions every counter group; rocprofv3 falls
        # back to multi-pass replay, which has been observed to hang for
        # minutes on what should be a sub-second run. Require the explicit
        # opt-in so users discover the cost.
        if self.pmc_set == "all" and not self.pmc_allow_multipass:
            raise ValueError(
                "--pmc all requires --pmc-allow-multipass: rocprofv3 "
                "falls back to multi-pass replay for the unioned counter "
                "set, which can hang on small workloads. Pick "
                "--pmc basic|memory|flops for a single-pass run."
            )
        if isinstance(self.profiling_output_dir, str):
            self.profiling_output_dir = Path(self.profiling_output_dir)
        if self.profiling_timeout_s < 0:
            raise ValueError(
                f"--profiling-timeout must be >= 0 (0 disables); "
                f"got {self.profiling_timeout_s}"
            )

    @property
    def basic_enabled(self) -> bool:
        """True when always-on probes should run."""
        return self.tier == "basic"

    @property
    def opt_in_pass_requested(self) -> bool:
        """True when any opt-in profiling source was requested."""
        return bool(self.emit_trace or self.pmc_set or self.perf or self.roofline)

    @property
    def extra_runs_per_engine(self) -> int:
        """How many additional workload runs each opt-in source contributes.

        Each opt-in profiling source re-runs the workload once under its
        external tool. The basic always-on tier wraps the timed pass and
        does not add a run. Used by the reporter to give the user an
        upfront cost estimate.
        """
        return (
            int(self.pmc_set is not None)
            + int(self.emit_trace is not None)
            + int(self.perf)
            + int(self.roofline)
        )


@dataclass(frozen=True)
class EngineSelection:
    """One ordered engine execution selection.

    The plugin path is attached to the selection row rather than looked up by
    engine ID so repeated engine IDs can be benchmarked against different
    plugin builds.
    """

    engine_id: int
    plugin_path: Optional[Path] = None


@dataclass
class SuiteConfig:
    """Configuration for suite execution mode.

    Controls how the suite runner iterates providers/engines and validates
    correctness for each graph.

    Attributes:
        warmup_iters: Number of warmup iterations per provider/engine.
        benchmark_iters: Number of benchmark iterations for timing.
        seed: Random seed for reproducible inputs.
        engine_filter: If set, ordered engine selections to run.
        validation: Reference validation configuration (provider + tolerances).
        verbose: If True, print rich per-engine block per graph instead of summary.
        oracle_mode: Oracle comparison depth. "off" runs no comparison;
            "plan" times the auto-tuner's chosen plan against the heuristic
            plan; "exhaustive" additionally forces provider kernel
            benchmarking so providers sample kernel variants.
        metrics: Metric collection configuration. Defaults to ``basic`` tier
            (always-on probes, no extra runs).
        backend: Execution backend (``hipdnn`` runs discovered engine plugins,
            ``pytorch`` runs the graph through the PyTorch executor as a single
            engine row per graph).
        pytorch_sdpa_backend: Strict PyTorch SDPA category selection for
            PyTorch timing or reference execution.
        pytorch_rocm_fa_library: Optional ROCm Flash Attention implementation
            preference forwarded to PyTorch with the Flash category.
    """

    warmup_iters: int = 10
    benchmark_iters: int = 100
    seed: int = 0
    engine_filter: Optional[List[int]] = None
    verbose: bool = False
    oracle_mode: OracleMode = OracleMode.OFF
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    plugin_paths: Optional[List[Path]] = None
    backend: ExecutionBackendName = ExecutionBackendName.HIPDNN
    pytorch_sdpa_backend: PyTorchSdpaBackendName = PyTorchSdpaBackendName.DEFAULT
    pytorch_rocm_fa_library: Optional[str] = None
    #: Sample every knob-filtered candidate on first execute and cache the
    #: winner (HIPDNN_FORCE_BENCHMARKING=1). Off, an engine serves its cold
    #: heuristic's rank-0 pick, so the table measures the heuristic rather than
    #: what the shipped kernel set can deliver.
    autotune: bool = False
    #: Per-run HIPDNN_CACHE_DIR. The winner cache is on disk and outlives the
    #: job; reads are not gated on benchmarking while writes are, so without an
    #: explicit empty root an untuned phase can replay a previous tuned ranking.
    cache_dir: Optional[str] = None
    #: Summed device-time budget per timed loop (0 = fixed iteration count).
    min_time_ms: float = 0.0
    #: ``warm`` or ``cold`` (flush L2/MALL before each timed iteration).
    cache_mode: str = "warm"
    #: Suppress progress output; tables and the summary still print.
    quiet: bool = False

    @classmethod
    def from_namespace(cls, args: argparse.Namespace) -> "SuiteConfig":
        """Build a suite config from merged CLI/config-file arguments.

        ``args`` must hold every public option (``apply_config_file`` merges
        the defaults). hipDNN runs without ``--plugin-path`` fall back to the
        ROCm install's plugin directory.
        """
        plugin_paths = args.plugin_path
        if plugin_paths is None and args.backend != ExecutionBackendName.PYTORCH:
            from ..common.rocm_runtime import default_hipdnn_plugin_paths

            plugin_paths = default_hipdnn_plugin_paths()
        return cls(
            warmup_iters=args.warmup,
            benchmark_iters=args.iters,
            min_time_ms=args.min_time_ms,
            cache_mode=args.cache_mode,
            seed=args.seed,
            engine_filter=args.engine,
            verbose=args.verbose,
            quiet=args.quiet,
            oracle_mode=args.oracle_mode,
            metrics=MetricsConfig(
                tier=args.metrics_tier,
                emit_trace=args.emit_trace,
                pmc_set=args.pmc,
                perf=args.perf,
                roofline=args.roofline,
                pmc_allow_multipass=args.pmc_allow_multipass,
                profiling_output_dir=args.profiling_output_dir,
                profiling_timeout_s=args.profiling_timeout,
            ),
            validation=ValidationConfig(
                provider=args.validate, rtol=args.rtol, atol=args.atol
            ),
            plugin_paths=plugin_paths,
            backend=args.backend,
            pytorch_sdpa_backend=args.pytorch_sdpa_backend,
            pytorch_rocm_fa_library=args.pytorch_rocm_fa_library,
            autotune=args.autotune,
            cache_dir=str(args.cache_dir) if args.cache_dir else None,
        )

    @property
    def timing_policy(self) -> TimingPolicy:
        """Timing loop policy derived from this suite configuration."""
        return TimingPolicy(
            warmup_iters=self.warmup_iters,
            iters=self.benchmark_iters,
            min_time_ms=self.min_time_ms,
            max_iters=max(TimingPolicy.max_iters, self.benchmark_iters),
            cache_mode=self.cache_mode,
        )

    @property
    def oracle_enabled(self) -> bool:
        """True when any oracle comparison should run."""
        return self.oracle_mode is not OracleMode.OFF

    @property
    def oracle_exhaustive(self) -> bool:
        """True when the oracle pass must force provider kernel benchmarking."""
        return self.oracle_mode is OracleMode.EXHAUSTIVE

    def __post_init__(self) -> None:
        """Validate values and cross-field constraints; messages name CLI flags."""
        if self.warmup_iters < 0:
            raise ValueError("--warmup must be >= 0")
        if self.benchmark_iters <= 0:
            raise ValueError("--iters must be >= 1")
        if self.min_time_ms < 0:
            raise ValueError("--min-time-ms must be >= 0")
        if self.cache_mode not in CACHE_MODE_CHOICES:
            raise ValueError(
                "--cache-mode must be one of: " + ", ".join(CACHE_MODE_CHOICES)
            )
        if self.engine_filter is not None and len(self.engine_filter) == 0:
            raise ValueError("--engine must list at least one engine")
        if self.plugin_paths is not None:
            if len(self.plugin_paths) == 0:
                raise ValueError("--plugin-path must list at least one path")
            self.plugin_paths = [Path(p) for p in self.plugin_paths]

            if len(self.plugin_paths) > 1:
                if self.engine_filter is None:
                    raise ValueError(
                        "--plugin-path with multiple entries requires --engine"
                    )
                if len(self.plugin_paths) != len(self.engine_filter):
                    raise ValueError(
                        "--plugin-path entry count must be 1 or match --engine count"
                    )
        try:
            self.backend = ExecutionBackendName(self.backend)
        except ValueError as e:
            raise ValueError(_one_of("--backend", ExecutionBackendName)) from e
        try:
            self.oracle_mode = OracleMode(self.oracle_mode)
        except ValueError as e:
            raise ValueError(_one_of("--oracle-mode", OracleMode)) from e
        (
            self.pytorch_sdpa_backend,
            self.pytorch_rocm_fa_library,
        ) = _normalize_pytorch_sdpa_settings(
            self.pytorch_sdpa_backend,
            self.pytorch_rocm_fa_library,
        )
        if self.oracle_exhaustive and self.warmup_iters == 0:
            raise ValueError(
                "--oracle-mode exhaustive requires --warmup >= 1: with "
                "benchmarking forced, a plan's first execute() samples kernel "
                "variants, and at zero warmup that sampling lands inside the "
                "timed loop"
            )
        if self.backend is ExecutionBackendName.PYTORCH:
            rejected = [
                ("--engine", self.engine_filter is not None),
                ("--plugin-path", self.plugin_paths is not None),
                (
                    "--validate pytorch",
                    self.validation.provider is ReferenceProviderName.PYTORCH,
                ),
                ("--pmc", self.metrics.pmc_set is not None),
                ("--emit-trace", self.metrics.emit_trace is not None),
                ("--perf", self.metrics.perf),
                ("--roofline", self.metrics.roofline),
                ("--oracle-mode", self.oracle_enabled),
                ("--autotune", self.autotune),
                ("--cache-dir", self.cache_dir is not None),
            ]
            flags = [flag for flag, present in rejected if present]
            if flags:
                raise ValueError(
                    f"{', '.join(flags)} not supported with --backend pytorch "
                    "(hipDNN-only options)"
                )

    @property
    def plugin_path(self) -> Optional[Path]:
        """Return the shared plugin path when exactly one path is configured."""
        if self.plugin_paths is None or len(self.plugin_paths) != 1:
            return None
        return self.plugin_paths[0]

    def engine_selections_for(self, engine_ids: List[int]) -> List[EngineSelection]:
        """Return ordered engine selections for the provided engine IDs.

        ``engine_ids`` is either the explicit ``--engine`` list, where duplicate
        IDs are meaningful selections, or the backend-discovered engine list.
        Multiple plugin paths are only valid with an explicit engine list and
        are associated positionally with that list.
        """
        if self.plugin_paths is None:
            return [EngineSelection(engine_id) for engine_id in engine_ids]

        if len(self.plugin_paths) == 1:
            plugin_path = self.plugin_paths[0]
            return [
                EngineSelection(engine_id, plugin_path=plugin_path)
                for engine_id in engine_ids
            ]

        if self.engine_filter is None or len(engine_ids) != len(self.plugin_paths):
            raise ValueError(
                "--plugin-path entry count must be 1 or match --engine count"
            )

        return [
            EngineSelection(engine_id, plugin_path=plugin_path)
            for engine_id, plugin_path in zip(engine_ids, self.plugin_paths)
        ]

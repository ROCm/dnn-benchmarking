# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Suite CLI: config, startup checks, the per-graph loop, result writes, exit codes."""

import argparse
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config.benchmark_config import (
    ExecutionBackendName,
    PyTorchSdpaBackendName,
    ReferenceProviderName,
    SuiteConfig,
)
from ..graph.loader import GraphLoader
from ..metrics.gpu_smi import GpuSmiProbe
from ..metrics.host import host_memory_snapshot
from ..metrics.machine_info import collect_environment_info
from ..metrics.profiling_orchestrator import check_requested_tools
from ..reporting.reporter import Reporter
from ..reporting.suite_results import (
    GraphResult,
    RunInfo,
    SuiteResult,
    engine_id_hex,
    graph_id_for,
)
from .backends import BackendStartupError, GraphRunner, start_backend

#: Minimum seconds between intermediate result writes.
WRITE_INTERVAL_S = 10.0

# Match hipDNN's documented truthy values. Notably, "0" leaves a switch off.
_TRUTHY_ENV = {"1", "true", "on", "yes", "enable", "enabled"}

#: Variables that decide hipDNN kernel selection; recorded for oracle/autotune.
_SELECTION_ENV = (
    "HIPDNN_DISABLE_EXACT_ENGINE_CACHE",
    "HIPDNN_CACHE_DIR",
    "HIPDNN_DISABLE_CACHE",
    "HIPDNN_FORCE_BENCHMARKING",
    "MIOPEN_USER_DB_PATH",
    "MIOPEN_CUSTOM_CACHE_DIR",
)

#: Hazard kinds returned by :func:`_apply_tuning_environment`.
LEAKED_FORCE_BENCHMARKING = "leaked_force_benchmarking"
AUTOTUNE_WITHOUT_CACHE_DIR = "autotune_without_cache_dir"


class _Terminated(BaseException):
    """Raised by the SIGTERM handler; BaseException so per-graph isolation
    (``except Exception``) does not swallow it and the final write runs."""


def _raise_terminated(signum: int, frame: Any) -> None:
    raise _Terminated()


def _truthy_env(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _TRUTHY_ENV


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_suite_cli(
    args: argparse.Namespace,
    graph_paths: List[Path],
    reporter: Reporter,
    tarball_source: Optional[str] = None,
) -> int:
    """Build the config, run startup checks, run the suite; return the exit code."""
    try:
        config = SuiteConfig.from_namespace(args)
    except ValueError as e:
        reporter.error(str(e))
        return 2

    output_path: Optional[Path] = args.output
    if output_path is not None:
        problem = _output_problem(output_path)
        if problem:
            reporter.error(f"--output {output_path}: {problem}")
            return 2

    missing = check_requested_tools(config.metrics)
    for message in missing:
        reporter.error(message)
    if missing:
        return 2

    _warn_ignored_options(config, reporter)
    if config.backend is ExecutionBackendName.HIPDNN:
        _apply_tuning_environment(config, reporter)
        if config.oracle_enabled:
            _warn_oracle(config, reporter)

    try:
        run_graph = start_backend(config, reporter)
    except BackendStartupError as e:
        reporter.error(str(e))
        return e.exit_code

    if tarball_source:
        reporter.info(f"Graphs from {tarball_source}")
    if config.metrics.extra_runs_per_engine:
        reporter.info(
            f"Profiling: {config.metrics.extra_runs_per_engine} extra run(s) per engine"
        )
    return _run_suite(graph_paths, config, run_graph, output_path, reporter)


def _output_problem(path: Path) -> Optional[str]:
    """Why results cannot be written to ``path``, or None when they can."""
    if path.is_dir():
        return "is a directory"
    parent = path.parent
    try:
        parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        return f"cannot create {parent}: {e.strerror or e}"
    if not os.access(parent, os.W_OK | os.X_OK):
        return f"{parent} is not writable"
    return None


def _run_config(config: SuiteConfig) -> Dict[str, Any]:
    """The effective configuration recorded as ``run.config``."""
    pytorch = (
        config.backend is ExecutionBackendName.PYTORCH
        or config.validation.provider is ReferenceProviderName.PYTORCH
    )
    metrics = config.metrics
    return {
        "backend": config.backend.value,
        "engine_filter": (
            [engine_id_hex(e) for e in config.engine_filter]
            if config.engine_filter is not None
            else None
        ),
        "plugin_paths": (
            [str(p) for p in config.plugin_paths]
            if config.plugin_paths is not None
            else None
        ),
        "warmup_iters": config.warmup_iters,
        "iters": config.benchmark_iters,
        "min_time_ms": config.min_time_ms,
        "cache_mode": config.cache_mode,
        "seed": config.seed,
        "validate": (
            config.validation.provider.value if config.validation.enabled else None
        ),
        "rtol": config.validation.rtol,
        "atol": config.validation.atol,
        "oracle_mode": config.oracle_mode.value,
        "autotune": config.autotune,
        "cache_dir": config.cache_dir,
        "pytorch_sdpa_backend": config.pytorch_sdpa_backend.value if pytorch else None,
        "pytorch_rocm_fa_library": config.pytorch_rocm_fa_library if pytorch else None,
        "metrics_tier": metrics.tier.value,
        "profiling": {
            "pmc": metrics.pmc_set,
            "emit_trace": metrics.emit_trace,
            "perf": metrics.perf,
            "roofline": metrics.roofline,
        },
    }


def _run_one_graph(graph_path: Path, run_graph: GraphRunner) -> GraphResult:
    """Load and run one graph; any failure becomes a graph-level error."""
    graph_id = None
    try:
        loader = GraphLoader()
        graph_json = loader.load_json(graph_path)
        graph_id = graph_id_for(graph_json)
        loader.validate(graph_json)
        tensor_infos = loader.extract_tensor_info(graph_json)
        return run_graph(graph_path, graph_json, tensor_infos)
    except Exception as e:
        return GraphResult(
            graph_name=graph_path.stem,
            graph_path=str(graph_path),
            results=[],
            graph_id=graph_id,
            error=f"{type(e).__name__}: {e}",
        )


def _run_suite(
    graph_paths: List[Path],
    config: SuiteConfig,
    run_graph: GraphRunner,
    output_path: Optional[Path],
    reporter: Reporter,
) -> int:
    """Run every graph, writing results as it goes; return the exit code."""
    environment = collect_environment_info()
    if config.oracle_enabled or config.autotune:
        environment["selection_env"] = {n: os.environ.get(n) for n in _SELECTION_ENV}
    run_config = _run_config(config)
    total = len(graph_paths)
    reporter.print_suite_header(environment, run_config, total)
    suite = SuiteResult(
        run=RunInfo(started_at=_now(), argv=list(sys.argv), config=run_config),
        environment=environment,
        graphs=[],
    )

    def write() -> bool:
        if output_path is None:
            return True
        try:
            suite.write(output_path)
            return True
        except (OSError, TypeError, ValueError) as e:
            reporter.error(f"writing {output_path} failed: {e}")
            return False

    write_ok = True
    interrupted: Optional[int] = None
    previous_sigterm = signal.signal(signal.SIGTERM, _raise_terminated)
    try:
        last_write = time.monotonic()
        for i, graph_path in enumerate(graph_paths, start=1):
            reporter.graph_start(i, total, graph_path.stem)
            gr = _run_one_graph(graph_path, run_graph)
            suite.graphs.append(gr)
            reporter.print_graph_table(gr)
            if time.monotonic() - last_write >= WRITE_INTERVAL_S:
                write_ok = write() and write_ok
                last_write = time.monotonic()
        suite.run.finished_at = _now()
        suite.run.complete = True
    except KeyboardInterrupt:
        interrupted = 130
    except _Terminated:
        interrupted = 143
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)
        environment["end_of_run"] = {
            **host_memory_snapshot(),
            **GpuSmiProbe().snapshot(),
        }
        write_ok = write() and write_ok

    if interrupted is not None:
        where = (
            f"partial results in {output_path}"
            if output_path is not None and write_ok
            else "no results file written"
        )
        reporter.error(
            f"interrupted after {len(suite.graphs)}/{total} graph(s); {where}"
        )
        return interrupted

    reporter.print_summary(suite, str(output_path) if output_path else None)
    if config.oracle_enabled:
        reporter.print_oracle_summary(suite.graphs)
    return _exit_code(suite, write_ok)


def _exit_code(suite: SuiteResult, write_ok: bool) -> int:
    """3 on any failed verdict, else 1 on any error or write failure, else 0."""
    verdicts = {r.verdict for g in suite.graphs for r in g.results}
    if "failed" in verdicts:
        return 3
    if "error" in verdicts or any(g.error for g in suite.graphs) or not write_ok:
        return 1
    return 0


def _warn_ignored_options(config: SuiteConfig, reporter: Reporter) -> None:
    """Options that are accepted but would silently do nothing."""
    metrics = config.metrics
    if metrics.profiling_output_dir is not None and not metrics.opt_in_pass_requested:
        reporter.warning(
            "--profiling-output-dir has no effect without --pmc, --emit-trace, "
            "--perf or --roofline"
        )
    pytorch_selected = (
        config.backend is ExecutionBackendName.PYTORCH
        or config.validation.provider is ReferenceProviderName.PYTORCH
    )
    sdpa_set = (
        config.pytorch_sdpa_backend is not PyTorchSdpaBackendName.DEFAULT
        or config.pytorch_rocm_fa_library is not None
    )
    if sdpa_set and not pytorch_selected:
        reporter.warning(
            "PyTorch SDPA options have no effect without --backend pytorch or "
            "--validate pytorch"
        )


def _warn_oracle(config: SuiteConfig, reporter: Reporter) -> None:
    """One line per condition that makes the oracle's OOTB baseline non-cold."""
    if not _truthy_env("HIPDNN_DISABLE_EXACT_ENGINE_CACHE"):
        reporter.warning(
            "--oracle-mode: exact-engine cache is on, so the OOTB timing may "
            "replay a persisted ranking (HIPDNN_DISABLE_EXACT_ENGINE_CACHE=1 for cold)"
        )
    elif not _truthy_env("HIPDNN_DISABLE_CACHE"):
        reporter.warning(
            "--oracle-mode: provider kernel caches are on "
            "(HIPDNN_DISABLE_CACHE=1 for a cold comparison)"
        )
    if config.warmup_iters == 0:
        reporter.warning(
            "--oracle-mode with --warmup 0: first-execute kernel sampling lands "
            "in both timed loops"
        )
    if config.oracle_exhaustive:
        reporter.info(
            "--oracle-mode exhaustive: providers may reuse tuned selections "
            "(e.g. MIOpen FindDb); a cache miss costs candidates x variants"
        )


def _apply_tuning_environment(config: SuiteConfig, reporter: Reporter) -> List[str]:
    """Set the kernel-selection environment and state the path in effect.

    Prints one info line ``kernel selection: heuristic|autotune; cache: ...``
    plus one warning per hazard. Returns the hazard kinds
    (:data:`LEAKED_FORCE_BENCHMARKING`, :data:`AUTOTUNE_WITHOUT_CACHE_DIR`).
    """
    hazards: List[str] = []
    if config.cache_dir:
        os.environ["HIPDNN_CACHE_DIR"] = config.cache_dir
    cache = os.environ.get("HIPDNN_CACHE_DIR") or "shared per-user (~/.cache/hipdnn)"

    if config.autotune:
        os.environ["HIPDNN_FORCE_BENCHMARKING"] = "1"
    elif _truthy_env("HIPDNN_FORCE_BENCHMARKING"):
        # Process-wide and inherited from the shell: the run is NOT on the
        # heuristic path even though --autotune was not passed.
        hazards.append(LEAKED_FORCE_BENCHMARKING)
        reporter.warning(
            "HIPDNN_FORCE_BENCHMARKING is set in the environment without "
            "--autotune; kernels are benchmarked, not heuristic-selected"
        )
    autotune = config.autotune or bool(hazards)
    if config.autotune and not config.cache_dir:
        # The winner cache outlives the run and reads are not gated on
        # benchmarking, so a previous session's ranking can be reported.
        hazards.append(AUTOTUNE_WITHOUT_CACHE_DIR)
        reporter.warning(
            "--autotune without --cache-dir: winners cached by earlier runs "
            "may be reported instead of measured"
        )
    reporter.info(
        f"kernel selection: {'autotune' if autotune else 'heuristic'}; cache: {cache}"
    )
    return hazards

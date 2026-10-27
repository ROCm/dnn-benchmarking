# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Oracle pass: time a tuned run next to a row's OOTB run.

Every tuned run executes in a child process, after the row's OOTB plan is
timed and its buffers are released:

- hipDNN rows (:func:`run_hipdnn_tuned`): the child runs the same engine
  with ``--autotune`` (``HIPDNN_FORCE_BENCHMARKING=1``, the process-wide form
  of the ``global.benchmarking`` knob), so its first launch samples the
  candidate kernels and keeps the fastest.
- PyTorch rows (:func:`run_pytorch_tuned`): the child runs the graph with
  exhaustive conv search and TunableOp.

A child owns all tuning state. hipDNN keeps a tuned winner in memory for the
life of its process and serves it to every later build of the same graph and
engine. A tuned search also writes winners to the hipDNN disk cache and
MIOpen's user database. Both stay in the child: its hipDNN cache directory
and MIOpen user database are in a temporary directory. Thus no OOTB
measurement, including a stall-fallback re-measurement, can observe them.
A failed tuned run sets ``row.oracle_error`` and never fails the row.
"""

import json
import os
import subprocess
import sys
import tempfile
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config.benchmark_config import SuiteConfig
from ..metrics._diagnostic import warn_once
from ..metrics._subprocess import run_capped
from ..metrics.analytical import derive_throughputs
from ..reporting.statistics import BenchmarkStats, TimingInfo
from ..reporting.suite_results import (
    CorrectnessResult,
    OracleResult,
    ProviderEngineResult,
    engine_id_hex,
)

#: hipDNN knob that makes a provider sample its candidate kernels on the first
#: execute() and keep the fastest (kernel ingestor, MIOpen).
BENCHMARKING_KNOB = "global.benchmarking"

#: Node types whose PyTorch kernel the tuned child searches: MIOpen
#: exhaustive conv search (cudnn.benchmark) and TunableOp GEMMs.
_PYTORCH_TUNABLE_OPS = frozenset(
    {
        "ConvolutionFwdAttributes",
        "ConvolutionBwdAttributes",
        "ConvolutionWrwAttributes",
        "MatmulAttributes",
    }
)


def _record_error(row: ProviderEngineResult, what: str, e: Exception) -> None:
    row.oracle_error = f"{type(e).__name__}: {e}"
    warn_once("oracle", f"tuned run failed for {what}: {row.oracle_error}")


def _set_throughputs(oracle: OracleResult, row: ProviderEngineResult) -> None:
    """Same FLOPs, bytes and median denominator as the row's throughputs."""
    median = oracle.gpu_kernel_stats.median_ms if oracle.gpu_kernel_stats else None
    oracle.derived_tflops_per_s, oracle.derived_gbytes_per_s = derive_throughputs(
        row.analytical_flops, row.analytical_io_bytes, median
    )


def _timing_argv(graph_path: Path, config: SuiteConfig, output: Path) -> List[str]:
    """Child argv shared by every tuned run: one graph, the run's timing."""
    return [
        "--graph",
        str(graph_path),
        "--warmup",
        str(config.warmup_iters),
        "--iters",
        str(config.benchmark_iters),
        "--min-time-ms",
        str(config.min_time_ms),
        "--cache-mode",
        config.cache_mode,
        "--timing-block",
        str(config.timing_block),
        "--seed",
        str(config.seed),
        "--quiet",
        "--output",
        str(output),
    ]


def _hipdnn_tuned_argv(
    graph_path: Path,
    config: SuiteConfig,
    engine_id: int,
    plugin_path: Optional[Path],
    output: Path,
) -> List[str]:
    """CLI argv for the tuned-hipDNN child: one engine row, benchmarking on.

    The child's untimed first launch runs the search, so it gets the same
    warmup and sampling settings as the OOTB row. With validation on, the
    child validates its own output against the same reference provider.
    """
    argv = [sys.executable, "-m", "dnn_benchmarking"]
    argv += _timing_argv(graph_path, config, output)
    argv += ["--engine", engine_id_hex(engine_id), "--autotune"]
    if plugin_path is not None:
        argv += ["--plugin-path", str(plugin_path)]
    if not config.metrics.basic:
        argv.append("--no-metrics")
    if config.validation.enabled:
        argv += ["--validate", config.validation.provider.value]
        if config.validation.rtol is not None:
            argv += ["--rtol", str(config.validation.rtol)]
        if config.validation.atol is not None:
            argv += ["--atol", str(config.validation.atol)]
    return argv


def _pytorch_tuned_argv(
    graph_path: Path, config: SuiteConfig, output: Path
) -> List[str]:
    """CLI argv for the tuned-PyTorch child: one timed PyTorch row, no oracle.

    The child's untimed first launch runs the tuning search, so it gets the
    same warmup and sampling settings as the OOTB row.
    """
    argv = [sys.executable, "-m", "dnn_benchmarking.cli.pytorch_tuned_child"]
    argv += ["--runtime", "pytorch", "--no-metrics"]
    argv += _timing_argv(graph_path, config, output)
    argv += ["--pytorch-sdpa-backend", config.pytorch_sdpa_backend.value]
    if config.pytorch_rocm_fa_library is not None:
        argv += ["--pytorch-rocm-fa-library", config.pytorch_rocm_fa_library]
    return argv


def _run_tuned_child(
    what: str, argv: List[str], env: Dict[str, str], output: Path, config: SuiteConfig
) -> Dict[str, Any]:
    """Run one tuned child and return its single engine row.

    Raises:
        RuntimeError: If the child fails, times out (``--profiling-timeout``)
            or reports no successful row.
    """
    timeout_s = config.metrics.profiling_timeout_s or None
    try:
        proc = run_capped(argv, timeout_s, env=env)
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"{what} child timed out after {timeout_s} s (--profiling-timeout)"
        ) from None
    if not output.is_file():
        lines = [
            line.strip()
            for line in f"{proc.stdout or ''}\n{proc.stderr or ''}".splitlines()
            if line.strip()
        ]
        errors = [line for line in lines if "error" in line.lower()]
        detail = (errors or lines or ["no output"])[-1]
        raise RuntimeError(f"{what} child exited {proc.returncode}: {detail}")
    document = json.loads(output.read_text())
    rows = [
        row
        for graph in document["graphs"]
        for row in graph["results"]
        if row["role"] == "engine"
    ]
    if len(rows) != 1:
        raise RuntimeError(f"{what} child returned {len(rows)} engine rows")
    if rows[0]["status"] != "success":
        raise RuntimeError(rows[0]["message"] or rows[0]["status"])
    return rows[0]


def _stats(data: Optional[Dict[str, Any]], samples: int) -> Optional[BenchmarkStats]:
    """Rebuild stats from a v2 stats object and the plan's ``timing.samples``.

    The file keeps only the quartiles; the console-only fields (mean, std,
    min, p95, max) are None, and nothing reads them for a tuned row (the
    speedup, throughput and table use the median).
    """
    if not data:
        return None
    values = {**data, "n": samples}
    return BenchmarkStats(
        **{f.name: values.get(f.name) for f in fields(BenchmarkStats)}
    )


def _oracle_from_plan(plan: Dict[str, Any], tuning_available: bool) -> OracleResult:
    """The tuned run from a child row's ``ootb`` plan object."""
    timing = TimingInfo(**plan["timing"])
    return OracleResult(
        tuning_available=tuning_available,
        cpu_build_time_ms=plan.get("build_ms"),
        timing=timing,
        gpu_kernel_stats=_stats(plan["kernel"], timing.samples),
        host_stats=_stats(plan["host"], timing.samples),
        workspace_bytes=plan.get("workspace_bytes"),
    )


def run_hipdnn_tuned(
    *,
    row: ProviderEngineResult,
    graph_path: Path,
    graph_name: str,
    engine_id: int,
    plugin_path: Optional[Path],
    tuning_available: bool,
    config: SuiteConfig,
) -> None:
    """Attach the engine's tuned run, timed in a child, as ``row.oracle``.

    Call after the OOTB buffers are released, so the child's allocations do
    not stack on top of them. ``tuning_available`` is whether the engine
    exposes ``global.benchmarking`` for this graph; without it the child
    re-measures the OOTB configuration. Never fails the row.
    """
    try:
        with tempfile.TemporaryDirectory(prefix="dnn-bench-hipdnn-tuned-") as state:
            output = Path(state) / "result.json"
            env = dict(os.environ)
            # The search must neither reuse earlier tuning nor leave entries
            # that a later OOTB run would read.
            env["MIOPEN_USER_DB_PATH"] = state
            child = _run_tuned_child(
                "tuned hipDNN",
                _hipdnn_tuned_argv(graph_path, config, engine_id, plugin_path, output),
                env,
                output,
                config,
            )
        plan = child["ootb"]
        oracle = _oracle_from_plan(plan, tuning_available)
        if plan.get("correctness") is not None:
            oracle.correctness = CorrectnessResult.from_dict(plan["correctness"])
            if oracle.correctness.explicitly_failed:
                warn_once(
                    "oracle_correctness",
                    f"tuned plan for {graph_name} engine {engine_id_hex(engine_id)} "
                    "failed validation; no speedup is reported",
                )
        _set_throughputs(oracle, row)
        row.oracle = oracle
    except Exception as e:
        _record_error(row, f"{graph_name} engine {engine_id_hex(engine_id)}", e)


def run_pytorch_tuned(
    *,
    row: ProviderEngineResult,
    graph_path: Path,
    graph_json: Dict[str, Any],
    graph_name: str,
    config: SuiteConfig,
) -> None:
    """Attach a tuned PyTorch run as ``row.oracle``; never fails the row.

    Call after the OOTB buffers are released, so the child's allocations do
    not stack on top of them. Tuned outputs are not validated: they never
    enter this process. ``tuning_available`` is true only when the graph has
    a conv or matmul node, the ops the child searches.
    """
    from ..common.pytorch_tuning import tuned_subprocess_env

    try:
        with tempfile.TemporaryDirectory(prefix="dnn-bench-pytorch-tuned-") as state:
            output = Path(state) / "result.json"
            child = _run_tuned_child(
                "tuned PyTorch",
                _pytorch_tuned_argv(graph_path, config, output),
                tuned_subprocess_env(state),
                output,
                config,
            )
        oracle = _oracle_from_plan(
            child["ootb"],
            any(
                node.get("type") in _PYTORCH_TUNABLE_OPS
                for node in graph_json.get("nodes") or []
            ),
        )
        # PyTorch has no plan build; the tuned child reports none either.
        oracle.cpu_build_time_ms = None
        oracle.workspace_bytes = None
        _set_throughputs(oracle, row)
        row.oracle = oracle
    except Exception as e:
        _record_error(row, f"PyTorch on {graph_name}", e)

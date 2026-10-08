# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Oracle pass: time a tuned run next to a row's OOTB run.

hipDNN rows get a second plan for the same engine, built with the
``global.benchmarking`` knob (:func:`build_tuned_plan`) right after the OOTB
build and timed after the OOTB loop (:func:`run_tuned_plan`). PyTorch rows get
a tuned run in an isolated child process (:func:`run_pytorch_tuned`). A failed
tuned run sets ``row.oracle_error`` and never fails the row.
"""

import json
import os
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

from ..config.benchmark_config import SuiteConfig
from ..graph.tensor_info import TensorInfo
from ..metrics._diagnostic import warn_once
from ..metrics._subprocess import run_capped
from ..metrics.analytical import derive_throughputs
from ..reporting.statistics import BenchmarkStats, TimingInfo
from ..reporting.suite_results import OracleResult, ProviderEngineResult
from ..validation import ReferenceOutput
from .correctness import check_correctness
from .executor import Executor
from .timing import StallFallbackError

#: hipDNN knob that makes a provider sample its candidate kernels on the first
#: execute() and keep the fastest (kernel ingestor, MIOpen).
BENCHMARKING_KNOB = "global.benchmarking"

# A benchmarking plan writes its winner to the hipDNN disk cache, and cache
# reads are not gated on benchmarking, so a later OOTB row would serve it.
# ponytail: process-global guard; concurrent execution needs process isolation.
_TUNED_ENV = {"HIPDNN_DISABLE_CACHE": "1"}

# hipDNN also keeps each winner in memory for the life of the process, even
# with HIPDNN_DISABLE_CACHE=1, and serves it to every later build of the same
# graph content and engine, with or without the knob. An OOTB build of a pair
# listed here would time the tuned kernel. Filled when a tuned plan executes.
# ponytail: keyed by graph_id (exact JSON), so graphs that differ only in
# names or UIDs still share hipDNN's content key; a child process per tuned
# hipDNN pass removes this registry.
_TUNED_IN_PROCESS: Set[Tuple[str, int]] = set()

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


def tuned_in_process(graph_id: str, engine_id: int) -> bool:
    """True when this engine already executed a tuned plan for this graph."""
    return (graph_id, engine_id) in _TUNED_IN_PROCESS


@contextmanager
def _tuned_env() -> Iterator[None]:
    """Disable hipDNN disk caches and restore the prior environment."""
    previous = {name: os.environ.get(name) for name in _TUNED_ENV}
    os.environ.update(_TUNED_ENV)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


@dataclass
class TunedPlan:
    """A built tuned plan waiting for :func:`run_tuned_plan`."""

    executor: Executor
    handle: Any
    tuning_available: bool


def _record_error(row: ProviderEngineResult, what: str, e: Exception) -> None:
    row.oracle_error = f"{type(e).__name__}: {e}"
    warn_once("oracle", f"tuned run failed for {what}: {row.oracle_error}")


def _set_throughputs(oracle: OracleResult, row: ProviderEngineResult) -> None:
    """Same FLOPs, bytes and median denominator as the row's throughputs."""
    median = oracle.gpu_kernel_stats.median_ms if oracle.gpu_kernel_stats else None
    oracle.derived_tflops_per_s, oracle.derived_gbytes_per_s = derive_throughputs(
        row.analytical_flops, row.analytical_io_bytes, median
    )


def build_tuned_plan(
    *,
    row: ProviderEngineResult,
    handle: Any,
    engine_id: int,
    graph_json_str: str,
    graph_name: str,
    config: SuiteConfig,
) -> Optional[TunedPlan]:
    """Build this engine's tuned plan right after its OOTB build.

    The tuned plan goes through the same timed build as the OOTB plan, for
    the same engine, with ``global.benchmarking=1``. Building it before
    anything executes keeps both build times in the same process state: work
    in between (the OOTB timed loop, validation) slows a later build.

    Returns:
        The plan for :func:`run_tuned_plan`, or None after recording
        ``row.oracle_error``.
    """
    try:
        with _tuned_env():
            # Isolate MIOpen's mutable per-handle solver map from the OOTB plan.
            oracle_handle = type(handle)()
            oracle_handle.set_stream(handle.get_stream())
            executor = Executor(graph_json_str, config.timing_policy)
            executor.prepare(
                oracle_handle, engine_id=engine_id, knobs={BENCHMARKING_KNOB: 1}
            )
            # hipDNN ignores a knob the engine does not expose; such a tuned
            # run re-measures the OOTB configuration.
            available = BENCHMARKING_KNOB in executor.engine_knob_ids(engine_id)
        return TunedPlan(executor, oracle_handle, available)
    except Exception as e:
        _record_error(row, f"{graph_name} engine {engine_id}", e)
        return None


def run_tuned_plan(
    *,
    tuned: TunedPlan,
    row: ProviderEngineResult,
    graph_id: str,
    engine_id: int,
    graph_name: str,
    config: SuiteConfig,
    bm: Any,
    variant_pack: Dict[int, int],
    tensor_infos: List[TensorInfo],
    reference_outputs: Optional[Dict[int, ReferenceOutput]],
) -> None:
    """Time and validate the tuned plan; attach it as ``row.oracle``.

    ``measure`` makes one untimed first launch before the warmups. With
    benchmarking on, that launch samples every candidate and keeps the
    fastest, so the search cost lands in ``timing.first_call_ms`` and the
    tuned plan gets the same warmups as the OOTB plan.
    """
    try:
        with _tuned_env():
            if tuned.tuning_available:
                # Before the launch: a search that fails part way may still
                # leave a winner behind.
                _TUNED_IN_PROCESS.add((graph_id, engine_id))
            bm.zero_outputs()
            m = tuned.executor.benchmark(tuned.handle, variant_pack)
            oracle = OracleResult(
                tuning_available=tuned.tuning_available,
                cpu_build_time_ms=tuned.executor.build_time_ms,
                timing=TimingInfo.from_measurement(m),
                gpu_kernel_stats=BenchmarkStats.from_timings(m.kernel_ms),
                host_stats=BenchmarkStats.from_timings(m.host_ms),
                workspace_bytes=(
                    tuned.executor.workspace_size if config.metrics.basic else None
                ),
            )
            _set_throughputs(oracle, row)

            if reference_outputs is not None:
                bm.zero_outputs()
                tuned.executor.execute_once(tuned.handle, variant_pack)
                oracle.correctness = check_correctness(
                    bm,
                    tensor_infos,
                    reference_outputs,
                    config.validation.provider.value,
                    config,
                )
                if oracle.correctness.explicitly_failed:
                    warn_once(
                        "oracle_correctness",
                        f"tuned plan for {graph_name} engine {engine_id} failed "
                        "validation; no speedup is reported",
                    )
        row.oracle = oracle
    except StallFallbackError:
        raise  # The suite runner remeasures the whole graph unstalled.
    except Exception as e:
        _record_error(row, f"{graph_name} engine {engine_id}", e)


def _pytorch_tuned_argv(
    graph_path: Path, config: SuiteConfig, output: Path
) -> List[str]:
    """CLI argv for the tuned-PyTorch child: one timed PyTorch row, no oracle.

    The child's untimed first launch runs the tuning search, so it gets the
    same warmup and sampling settings as the OOTB row.
    """
    argv = [
        sys.executable,
        "-m",
        "dnn_benchmarking.cli.pytorch_tuned_child",
        "--runtime",
        "pytorch",
        "--no-metrics",
        "--quiet",
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
        "--pytorch-sdpa-backend",
        config.pytorch_sdpa_backend.value,
        "--output",
        str(output),
    ]
    if config.pytorch_rocm_fa_library is not None:
        argv += ["--pytorch-rocm-fa-library", config.pytorch_rocm_fa_library]
    return argv


def _run_pytorch_tuned_child(graph_path: Path, config: SuiteConfig) -> Dict[str, Any]:
    """Time tuned PyTorch in a fresh process and return its result row.

    The child owns every piece of tuning state (conv algorithm cache,
    TunableOp results, MIOpen user database), and the temporary directory
    that holds it is deleted afterwards, so no OOTB run can observe it.

    Raises:
        RuntimeError: If the child fails, times out (``--profiling-timeout``)
            or reports no successful row.
    """
    from ..common.pytorch_tuning import tuned_subprocess_env

    with tempfile.TemporaryDirectory(prefix="dnn-bench-pytorch-tuned-") as state_dir:
        output = Path(state_dir) / "result.json"
        timeout_s = config.metrics.profiling_timeout_s or None
        try:
            proc = run_capped(
                _pytorch_tuned_argv(graph_path, config, output),
                timeout_s,
                env=tuned_subprocess_env(state_dir),
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(
                f"tuned PyTorch child timed out after {timeout_s} s "
                "(--profiling-timeout)"
            ) from None
        if not output.is_file():
            lines = [
                line.strip()
                for line in f"{proc.stdout or ''}\n{proc.stderr or ''}".splitlines()
                if line.strip()
            ]
            errors = [line for line in lines if "error" in line.lower()]
            detail = (errors or lines or ["no output"])[-1]
            raise RuntimeError(
                f"tuned PyTorch child exited {proc.returncode}: {detail}"
            )
        document = json.loads(output.read_text())
    rows = [row for graph in document["graphs"] for row in graph["results"]]
    if len(rows) != 1:
        raise RuntimeError(f"tuned PyTorch child returned {len(rows)} rows")
    if rows[0]["status"] != "success":
        raise RuntimeError(rows[0]["message"] or rows[0]["status"])
    return rows[0]


def _stats(data: Optional[Dict[str, Any]]) -> Optional[BenchmarkStats]:
    """Rebuild stats from a v2 stats object.

    The file keeps only ``n`` and the quartiles; the console-only fields
    (mean, std, min, p95, max) are None, and nothing reads them for a tuned
    row (the speedup, throughput and table use the median).
    """
    if not data:
        return None
    return BenchmarkStats(**{f.name: data.get(f.name) for f in fields(BenchmarkStats)})


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
    try:
        plan = _run_pytorch_tuned_child(graph_path, config)["ootb"]
        oracle = OracleResult(
            tuning_available=any(
                node.get("type") in _PYTORCH_TUNABLE_OPS
                for node in graph_json.get("nodes") or []
            ),
            timing=TimingInfo(**plan["timing"]),
            gpu_kernel_stats=_stats(plan["kernel"]),
            host_stats=_stats(plan["host"]),
        )
        _set_throughputs(oracle, row)
        row.oracle = oracle
    except Exception as e:
        _record_error(row, f"PyTorch on {graph_name}", e)

# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Oracle pass: time a tuned run next to a row's OOTB run.

Every tuned run executes in a child process, so no tuning state reaches the
benchmark process. hipDNN rows get the same engine built with the
``global.benchmarking`` knob (:func:`run_hipdnn_tuned`,
``execution.hipdnn_tuned_child``): hipDNN keeps a tuned winner in memory and
serves it to every later build of the same graph and engine, so an in-process
search would leak into later OOTB rows. PyTorch rows get a tuned run of the
normal CLI (:func:`run_pytorch_tuned`). A failed tuned run sets
``row.oracle_error`` and never fails the row.
"""

import json
import os
import subprocess
import sys
import tempfile
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ..config.benchmark_config import SuiteConfig
from ..graph.tensor_info import TensorInfo
from ..metrics._diagnostic import warn_once
from ..metrics._subprocess import run_capped
from ..metrics.analytical import derive_throughputs
from ..reporting.statistics import BenchmarkStats, TimingInfo
from ..reporting.suite_results import OracleResult, ProviderEngineResult
from ..validation import ReferenceOutput
from .correctness import check_correctness
from .timing import Measurement, StallFallbackError

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


def _run_child(
    what: str,
    argv: List[str],
    env: Dict[str, str],
    output: Path,
    config: SuiteConfig,
) -> Dict[str, Any]:
    """Run a tuned child and return the JSON document it wrote to ``output``.

    Raises:
        RuntimeError: If the child times out (``--profiling-timeout``) or
            writes no output; the message carries its last error line.
    """
    timeout_s = config.metrics.profiling_timeout_s or None
    try:
        proc = run_capped(argv, timeout_s, env=env)
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"{what} timed out after {timeout_s} s (--profiling-timeout)"
        ) from None
    if not output.is_file():
        lines = [
            line.strip()
            for line in f"{proc.stdout or ''}\n{proc.stderr or ''}".splitlines()
            if line.strip()
        ]
        errors = [line for line in lines if "error" in line.lower()]
        detail = (errors or lines or ["no output"])[-1]
        raise RuntimeError(f"{what} exited {proc.returncode}: {detail}")
    return json.loads(output.read_text())


def hipdnn_tuned_env(state_dir: Path) -> Dict[str, str]:
    """Environment of the tuned-hipDNN child.

    hipDNN disk caches are off, so the search neither reads an earlier
    ranking nor writes one. MIOpen's user database is private to the child,
    so the run's OOTB rows never read the search's find results.
    """
    miopen = state_dir / "miopen"
    miopen.mkdir(exist_ok=True)
    env = dict(os.environ)
    env.update({"HIPDNN_DISABLE_CACHE": "1", "MIOPEN_USER_DB_PATH": str(miopen)})
    return env


class _ChildOutputs:
    """The child's outputs, in the form :func:`check_correctness` reads."""

    def __init__(self, arrays: Dict[int, np.ndarray]) -> None:
        self._arrays = arrays

    def get_output_tensor(self, uid: int) -> None:
        return None  # host arrays only: compare on the host

    def get_output_data(self, uid: int) -> Optional[np.ndarray]:
        return self._arrays.get(uid)


def run_hipdnn_tuned(
    *,
    row: ProviderEngineResult,
    graph_json_str: str,
    graph_name: str,
    engine_id: int,
    plugin_path: Optional[str],
    config: SuiteConfig,
    tensor_infos: List[TensorInfo],
    reference_outputs: Optional[Dict[int, ReferenceOutput]],
) -> None:
    """Time and validate the engine's tuned plan in a child; set ``row.oracle``.

    Call after the OOTB buffers and plan are released, so the child's
    allocations do not stack on top of them. The child regenerates the
    inputs from the run seed. ``measure`` makes one untimed first launch
    before the warmups; with benchmarking on, that launch samples every
    candidate and keeps the fastest, so the search cost lands in
    ``timing.first_call_ms`` and the tuned plan gets the same warmups as the
    OOTB plan.

    Raises:
        StallFallbackError: The child's stall gate failed; the suite runner
            remeasures the whole graph unstalled. Other failures set
            ``row.oracle_error``.
    """
    try:
        with tempfile.TemporaryDirectory(prefix="dnn-bench-hipdnn-tuned-") as d:
            state = Path(d)
            request = state / "request.json"
            request.write_text(
                json.dumps(
                    {
                        "graph_json": graph_json_str,
                        "engine_id": engine_id,
                        "plugin_path": plugin_path,
                        "seed": config.seed,
                        "policy": asdict(config.timing_policy),
                        "outputs": reference_outputs is not None,
                        "workspace": config.metrics.basic,
                    }
                )
            )
            result = _run_child(
                "tuned hipDNN child",
                [
                    sys.executable,
                    "-m",
                    "dnn_benchmarking.execution.hipdnn_tuned_child",
                    str(request),
                ],
                hipdnn_tuned_env(state),
                state / "result.json",
                config,
            )
            if "stall" in result:
                raise StallFallbackError(result["stall"])
            if "error" in result:
                raise RuntimeError(result["error"])
            outputs = None
            if reference_outputs is not None:
                with np.load(state / "outputs.npz") as saved:
                    outputs = {int(uid): saved[uid] for uid in saved.files}

        m = Measurement(**result["measurement"])
        oracle = OracleResult(
            tuning_available=result["tuning_available"],
            cpu_build_time_ms=result["cpu_build_time_ms"],
            timing=TimingInfo.from_measurement(m),
            gpu_kernel_stats=BenchmarkStats.from_timings(m.kernel_ms),
            host_stats=BenchmarkStats.from_timings(m.host_ms),
            workspace_bytes=result["workspace_bytes"],
        )
        _set_throughputs(oracle, row)
        if outputs is not None:
            oracle.correctness = check_correctness(
                _ChildOutputs(outputs),
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
    same warmup and sampling settings as the OOTB row. It keeps the
    selection variables its parent sets (``--allow-selection-env``).
    """
    argv = [
        sys.executable,
        "-m",
        "dnn_benchmarking.cli.pytorch_tuned_child",
        "--runtime",
        "pytorch",
        "--no-metrics",
        "--quiet",
        "--allow-selection-env",
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
        document = _run_child(
            "tuned PyTorch child",
            _pytorch_tuned_argv(graph_path, config, output),
            tuned_subprocess_env(state_dir),
            output,
            config,
        )
    rows = [row for graph in document["graphs"] for row in graph["results"]]
    if len(rows) != 1:
        raise RuntimeError(f"tuned PyTorch child returned {len(rows)} rows")
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
            timing=(timing := TimingInfo(**plan["timing"])),
            gpu_kernel_stats=_stats(plan["kernel"], timing.samples),
            host_stats=_stats(plan["host"], timing.samples),
        )
        _set_throughputs(oracle, row)
        row.oracle = oracle
    except Exception as e:
        _record_error(row, f"PyTorch on {graph_name}", e)

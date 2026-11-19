# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Oracle pass: time a tuned run next to a row's OOTB run.

Every tuned run is a child process, so it can use the tuned selection state
of the work dir (``common.work_dir``: hipDNN cache and MIOpen user database,
kept across commands) while the row's OOTB run, in this process, uses the
OOTB state: MIOpen reads ``MIOPEN_USER_DB_PATH`` once per process. A
hipDNN row's tuned run builds the same engine with the
``global.benchmarking`` knob in ``cli/hipdnn_tuned_child.py``
(:func:`run_hipdnn_tuned`); a PyTorch row's tuned run is the normal CLI in
``cli/pytorch_tuned_child.py`` (:func:`run_pytorch_tuned`). A failed tuned
run sets ``row.oracle_error`` and never fails the row.
"""

import json
import os
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ..common.work_dir import SelectionState, tuned_state, work_dir
from ..config.benchmark_config import SuiteConfig
from ..graph.tensor_info import TensorInfo
from ..metrics._diagnostic import warn_once
from ..metrics._subprocess import run_capped
from ..metrics.analytical import derive_throughputs
from ..reporting.statistics import BenchmarkStats, TimingInfo
from ..reporting.suite_results import OracleResult, ProviderEngineResult
from ..validation import ReferenceOutput
from .correctness import check_correctness
from .timing import (
    STALL_RERUN_REASON,
    Measurement,
    StallFallbackError,
    release_flush_buffers,
)

#: hipDNN knob that makes a provider sample its candidate kernels on the first
#: execute() and keep the fastest (kernel ingestor, MIOpen).
BENCHMARKING_KNOB = "global.benchmarking"

#: Files that ``cli/hipdnn_tuned_child.py`` writes in its directory.
RESULT_FILE = "result.json"
FAILURE_FILE = "failure.json"

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


def output_file(out_dir: Path, uid: int) -> Path:
    """Path of the tuned child's saved output tensor ``uid``."""
    return out_dir / f"output_{uid}.npy"


@dataclass(frozen=True)
class _SavedOutputs:
    """The tuned child's saved outputs, as ``check_correctness`` reads them."""

    out_dir: Path

    def get_output_tensor(self, uid: int) -> None:
        return None  # Host arrays only: compare on the host.

    def get_output_data(self, uid: int) -> Optional[np.ndarray]:
        path = output_file(self.out_dir, uid)
        return np.load(path) if path.is_file() else None


def _record_error(row: ProviderEngineResult, what: str, e: Exception) -> None:
    row.oracle_error = f"{type(e).__name__}: {e}"
    warn_once("oracle", f"tuned run failed for {what}: {row.oracle_error}")


def _set_throughputs(oracle: OracleResult, row: ProviderEngineResult) -> None:
    """Same FLOPs, bytes and median denominator as the row's throughputs."""
    median = oracle.gpu_kernel_stats.median_ms if oracle.gpu_kernel_stats else None
    oracle.derived_tflops_per_s, oracle.derived_gbytes_per_s = derive_throughputs(
        row.analytical_flops, row.analytical_io_bytes, median
    )


def _tuned_state(config: SuiteConfig) -> SelectionState:
    """The tuned selection state that every tuned run shares, created."""
    return tuned_state(work_dir(config.benchmark_work_dir)).create()


def _child_failure(proc: "subprocess.CompletedProcess[str]") -> str:
    """The child's last error line, or its last line, for an error message."""
    lines = [
        line.strip()
        for line in f"{proc.stdout or ''}\n{proc.stderr or ''}".splitlines()
        if line.strip()
    ]
    errors = [line for line in lines if "error" in line.lower()]
    return (errors or lines or ["no output"])[-1]


def _hipdnn_tuned_argv(
    graph_path: Path,
    engine_id: int,
    plugin_path: Optional[Path],
    config: SuiteConfig,
    out_dir: Path,
    save_outputs: bool,
) -> List[str]:
    """Argv of ``cli/hipdnn_tuned_child.py``, with the full timing policy."""
    argv = [
        sys.executable,
        "-m",
        "dnn_benchmarking.cli.hipdnn_tuned_child",
        "--graph",
        str(graph_path),
        "--engine",
        str(engine_id),
        "--seed",
        str(config.seed),
        "--timing-policy",
        json.dumps(asdict(config.timing_policy)),
        "--out-dir",
        str(out_dir),
    ]
    if plugin_path is not None:
        argv += ["--plugin-path", str(plugin_path)]
    if save_outputs:
        argv.append("--save-outputs")
    return argv


#: Selection variable that the tuned hipDNN child must run with. Forced, not
#: defaulted: providers apply ``HIPDNN_FORCE_BENCHMARKING`` over the knob, so
#: an inherited ``0`` would make the tuned run untuned.
TUNED_CHILD_ENV = {"HIPDNN_FORCE_BENCHMARKING": "1"}


def _hipdnn_tuned_env(config: SuiteConfig) -> Dict[str, str]:
    """The tuned child's environment: :data:`TUNED_CHILD_ENV` and the tuned
    selection state (``HIPDNN_CACHE_DIR``, ``MIOPEN_USER_DB_PATH``). An
    inherited ``HIPDNN_DISABLE_CACHE`` stays as it is."""
    env = dict(os.environ)
    env.update(TUNED_CHILD_ENV)
    env.update(_tuned_state(config).env())
    return env


def run_hipdnn_tuned(
    *,
    row: ProviderEngineResult,
    graph_path: Path,
    graph_name: str,
    tensor_infos: List[TensorInfo],
    reference_outputs: Optional[Dict[int, ReferenceOutput]],
    engine_id: int,
    plugin_path: Optional[Path],
    config: SuiteConfig,
) -> None:
    """Time the engine's tuned plan in a child; attach it as ``row.oracle``.

    Call after the row's OOTB buffers and plans are released, so the child's
    VRAM does not stack on them. With ``reference_outputs``, the child saves
    its outputs and this process validates them like the OOTB row's. The
    child's first, untimed launch runs the search, so the search cost lands
    in ``timing.first_call_ms``.

    Raises:
        StallFallbackError: The stall gate failed in the child; the suite
            runner then remeasures the whole graph without it. Any other
            failure only sets ``row.oracle_error``.
    """
    what = f"{graph_name} engine {engine_id}"
    with tempfile.TemporaryDirectory(prefix="dnn-bench-hipdnn-tuned-") as state:
        out_dir = Path(state)
        try:
            result = _run_hipdnn_tuned_child(
                graph_path,
                engine_id,
                plugin_path,
                config,
                out_dir,
                save_outputs=reference_outputs is not None,
            )
            m = Measurement(**result["measurement"])
            oracle = OracleResult(
                tuning_available=result["tuning_available"],
                cpu_build_time_ms=result["build_ms"],
                timing=TimingInfo.from_measurement(m),
                gpu_kernel_stats=BenchmarkStats.from_timings(m.kernel_ms),
                host_stats=BenchmarkStats.from_timings(m.host_ms),
                workspace_bytes=(
                    result["workspace_bytes"] if config.metrics.basic else None
                ),
            )
            _set_throughputs(oracle, row)
            if reference_outputs is not None:
                oracle.correctness = check_correctness(
                    _SavedOutputs(out_dir),
                    tensor_infos,
                    reference_outputs,
                    config.validation.provider.value,
                    config,
                )
                if oracle.correctness.explicitly_failed:
                    warn_once(
                        "oracle_correctness",
                        f"tuned plan for {what} failed validation; no speedup "
                        "is reported",
                    )
        except StallFallbackError:
            raise  # The suite runner remeasures the whole graph unstalled.
        except Exception as e:
            _record_error(row, what, e)
            return
    row.oracle = oracle


def _run_hipdnn_tuned_child(
    graph_path: Path,
    engine_id: int,
    plugin_path: Optional[Path],
    config: SuiteConfig,
    out_dir: Path,
    save_outputs: bool,
) -> Dict[str, Any]:
    """Run ``cli/hipdnn_tuned_child.py``; return its ``result.json``.

    Frees this process's cold-cache flush buffer first, so the child gets the
    device memory the OOTB row had; the next cold iteration reallocates it.

    Raises:
        StallFallbackError: The child reports a failed stall gate.
        RuntimeError: The child failed, timed out (``--oracle-timeout``)
            or wrote no result.
    """
    timeout_s = config.oracle_timeout_s or None
    release_flush_buffers()
    argv = _hipdnn_tuned_argv(
        graph_path, engine_id, plugin_path, config, out_dir, save_outputs
    )
    try:
        proc = run_capped(argv, timeout_s, env=_hipdnn_tuned_env(config))
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"tuned hipDNN child timed out after {timeout_s} s (--oracle-timeout)"
        ) from None
    failure_path = out_dir / FAILURE_FILE
    if failure_path.is_file():
        failure = json.loads(failure_path.read_text())
        if failure["kind"] == "stall":
            raise StallFallbackError(failure["message"])
        raise RuntimeError(failure["message"])
    result_path = out_dir / RESULT_FILE
    if proc.returncode != 0 or not result_path.is_file():
        raise RuntimeError(
            f"tuned hipDNN child exited {proc.returncode}: {_child_failure(proc)}"
        )
    return json.loads(result_path.read_text())


def _pytorch_tuned_argv(
    graph_path: Path, config: SuiteConfig, output: Path
) -> List[str]:
    """CLI argv for the tuned-PyTorch child: one timed PyTorch row, no oracle.

    The child's untimed first launch runs the tuning search, so it gets the
    same warmup and sampling settings as the OOTB row, and the same stall-gate
    state: after a stall-gate fallback both plans time unstalled.
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
        "--benchmark-work-dir",
        str(work_dir(config.benchmark_work_dir)),
    ]
    if config.pytorch_rocm_fa_library is not None:
        argv += ["--pytorch-rocm-fa-library", config.pytorch_rocm_fa_library]
    if not config.timing_policy.stall_gate:
        argv.append("--no-stall-gate")
    return argv


def _run_pytorch_tuned_child(graph_path: Path, config: SuiteConfig) -> Dict[str, Any]:
    """Time tuned PyTorch in a fresh process and return its result row.

    The child owns the conv algorithm cache and the TunableOp results, in a
    temporary directory that is deleted afterwards. It runs on the tuned
    selection state of the work dir; no OOTB run uses it. This process's
    cold-cache flush buffer is freed first, as for the hipDNN child.

    Raises:
        RuntimeError: If the child fails, times out (``--oracle-timeout``)
            or reports no successful row.
    """
    from ..common.pytorch_tuning import tuned_subprocess_env

    release_flush_buffers()
    with tempfile.TemporaryDirectory(prefix="dnn-bench-pytorch-tuned-") as state_dir:
        output = Path(state_dir) / "result.json"
        timeout_s = config.oracle_timeout_s or None
        try:
            proc = run_capped(
                _pytorch_tuned_argv(graph_path, config, output),
                timeout_s,
                env=tuned_subprocess_env(state_dir, _tuned_state(config)),
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(
                f"tuned PyTorch child timed out after {timeout_s} s "
                "(--oracle-timeout)"
            ) from None
        if not output.is_file():
            raise RuntimeError(
                f"tuned PyTorch child exited {proc.returncode}: "
                f"{_child_failure(proc)}"
            )
        document = json.loads(output.read_text())
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

    Raises:
        StallFallbackError: The child remeasured its graph unstalled after a
            stall-gate failure while this run still times staged; the suite
            runner remeasures the whole graph unstalled. Other failures set
            ``row.oracle_error``.
    """
    try:
        plan = _run_pytorch_tuned_child(graph_path, config)["ootb"]
        if (
            config.timing_policy.stall_gate
            and plan["timing"]["fallback_reason"] == STALL_RERUN_REASON
        ):
            raise StallFallbackError("stall gate failed in the tuned PyTorch child")
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
    except StallFallbackError:
        raise  # The suite runner remeasures the whole graph unstalled.
    except Exception as e:
        _record_error(row, f"PyTorch on {graph_name}", e)

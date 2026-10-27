# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Oracle pass: time a tuned run next to a row's OOTB run.

Every tuned run executes in a child script, after the row's OOTB plan is
timed and its buffers are released:

- hipDNN rows (:func:`run_hipdnn_tuned`): ``cli/hipdnn_tuned_child.py``
  builds the same engine with ``HIPDNN_FORCE_BENCHMARKING=1`` (the
  process-wide form of the ``global.benchmarking`` knob), so its first launch
  samples the candidate kernels and keeps the fastest.
- PyTorch rows (:func:`run_pytorch_tuned`): ``cli/pytorch_tuned_child.py``
  runs the graph with exhaustive conv search and TunableOp.

This module sets the environment of each child. The scripts set no tuning
variable and read no dnn-benchmark option. A child owns all tuning state.
hipDNN keeps a tuned winner in memory for the life of its process and serves
it to every later build of the same graph and engine. A tuned search also
writes winners to MIOpen's user database. The child's user database is in a
temporary directory that this module deletes, and hipDNN's disk cache is off.
Thus no OOTB measurement, including a stall-fallback re-measurement, can
observe the tuned selection.

The child gets the full timing policy, ``stall_gate`` included, so a graph
that fell back to unstalled timing times its tuned runs the same way.
This process validates the tuned hipDNN outputs that the child saves. A
failed tuned run sets ``row.oracle_error`` and never fails the row.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypeVar

import numpy as np

from ..config.benchmark_config import SuiteConfig, TimingPolicy
from ..graph.tensor_info import TensorInfo
from ..metrics._diagnostic import warn_once
from ..metrics._subprocess import run_capped
from ..metrics.analytical import derive_throughputs
from ..reporting.statistics import BenchmarkStats, TimingInfo
from ..reporting.suite_results import (
    OracleResult,
    ProviderEngineResult,
    engine_id_hex,
)
from ..validation import ReferenceOutput
from .correctness import check_correctness
from .timing import Measurement, StallFallbackError

_T = TypeVar("_T")

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

# --- Child contract ---------------------------------------------------------
# Both child scripts take ``--graph --seed --timing-policy --out-dir`` and
# write ``result.json`` in the output directory (the private temporary
# directory of the run). The hipDNN child can also save its outputs there.

RESULT_FILE = "result.json"


def output_file(out_dir: Path, uid: int) -> Path:
    """Path of the saved output tensor ``uid`` (a NumPy ``.npy`` file)."""
    return out_dir / f"output_{uid}.npy"


#: Environment the parent sets for each child: name -> required value, or
#: None for "set and not empty". The children check it and set nothing.
HIPDNN_CHILD_ENV: Dict[str, Optional[str]] = {
    "HIPDNN_FORCE_BENCHMARKING": "1",
    "HIPDNN_DISABLE_CACHE": "1",
    "MIOPEN_USER_DB_PATH": None,
}
PYTORCH_CHILD_ENV: Dict[str, Optional[str]] = {
    "PYTORCH_TUNABLEOP_ENABLED": "1",
    "PYTORCH_TUNABLEOP_TUNING": "1",
    "PYTORCH_TUNABLEOP_FILENAME": None,
    "MIOPEN_USER_DB_PATH": None,
}


def require_child_env(required: Dict[str, Optional[str]]) -> None:
    """Raise unless the parent set the child's tuning environment.

    Without this check, a missing variable would time an untuned run and
    report it as the tuned result.

    Raises:
        RuntimeError: A variable is missing or has another value.
    """
    wrong = [
        f"{name}={os.environ.get(name)!r}"
        for name, value in required.items()
        if not os.environ.get(name) or (value is not None and os.environ[name] != value)
    ]
    if wrong:
        raise RuntimeError(f"tuning environment not set by the parent: {wrong}")


def _timing_policy(text: str) -> TimingPolicy:
    return TimingPolicy(**json.loads(text))


def child_parser(prog: str) -> argparse.ArgumentParser:
    """Argument parser with the options that every child script takes."""
    parser = argparse.ArgumentParser(prog=prog)
    parser.add_argument("--graph", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument(
        "--timing-policy",
        required=True,
        type=_timing_policy,
        help="TimingPolicy fields as a JSON object",
    )
    parser.add_argument("--out-dir", required=True, type=Path)
    return parser


def write_result(out_dir: Path, m: Measurement, **plan: Any) -> None:
    """Write the timed loop and the ``plan`` values as ``result.json``."""
    result = {
        "timing": TimingInfo.from_measurement(m).to_dict(),
        "kernel": asdict(BenchmarkStats.from_timings(m.kernel_ms)),
        "host": asdict(BenchmarkStats.from_timings(m.host_ms)),
        **plan,
    }
    (out_dir / RESULT_FILE).write_text(json.dumps(result))


def with_stall_fallback(run: Callable[[TimingPolicy], _T], policy: TimingPolicy) -> _T:
    """``run(policy)``; when the stall gate fails, ``run`` again without it.

    The child does what the parent does for a graph (``suite_runner``): the
    whole run is repeated unstalled, so the tuned row still gets a result.
    The parent shows ``mixed-timing`` when the two modes then differ.
    """
    try:
        return run(policy)
    except StallFallbackError:
        if not policy.stall_gate:
            raise
    # Retry outside the except block: the caught traceback holds the first
    # attempt's frames (executor, buffers, workspace) until the handler exits.
    return run(replace(policy, stall_gate=False))


# --- Parent side ------------------------------------------------------------


class _SavedOutputs:
    """Read-only output source for ``check_correctness``: the child's files."""

    def __init__(self, out_dir: Path) -> None:
        self._out_dir = out_dir

    def get_output_tensor(self, uid: int) -> None:
        return None  # Host arrays only: compare on the host.

    def get_output_data(self, uid: int) -> Optional[np.ndarray]:
        path = output_file(self._out_dir, uid)
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


def _child_args(graph_path: Path, config: SuiteConfig) -> List[str]:
    """Arguments that every child script takes, except ``--out-dir``."""
    return [
        "--graph",
        str(graph_path),
        "--seed",
        str(config.seed),
        "--timing-policy",
        json.dumps(asdict(config.timing_policy)),
    ]


def _run_child(
    what: str,
    module: str,
    args: List[str],
    env: Dict[str, str],
    out_dir: Path,
    config: SuiteConfig,
) -> Dict[str, Any]:
    """Run ``python -m dnn_benchmarking.cli.<module>`` and return its result.

    ``--profiling-timeout`` limits the child.

    Raises:
        RuntimeError: If the child times out, exits nonzero or writes no
            ``result.json``. The message holds the last line of its stderr.
    """
    argv = [sys.executable, "-m", f"dnn_benchmarking.cli.{module}"]
    argv += [*args, "--out-dir", str(out_dir)]
    timeout_s = config.metrics.profiling_timeout_s or None
    try:
        proc = run_capped(argv, timeout_s, env=env)
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"{what} child timed out after {timeout_s} s (--profiling-timeout)"
        ) from None
    result = out_dir / RESULT_FILE
    if proc.returncode != 0 or not result.is_file():
        lines = [
            line.strip()
            for line in (proc.stderr or proc.stdout or "").splitlines()
            if line.strip()
        ]
        detail = lines[-1] if lines else "no output"
        raise RuntimeError(f"{what} child exited {proc.returncode}: {detail}")
    return json.loads(result.read_text())


def _oracle_from_result(result: Dict[str, Any], tuning_available: bool) -> OracleResult:
    """The tuned run from a child's ``result.json``."""
    return OracleResult(
        tuning_available=tuning_available,
        cpu_build_time_ms=result.get("build_ms"),
        timing=TimingInfo(**result["timing"]),
        gpu_kernel_stats=BenchmarkStats(**result["kernel"]),
        host_stats=BenchmarkStats(**result["host"]),
    )


def _hipdnn_tuned_env(state: str) -> Dict[str, str]:
    """Environment of the tuned-hipDNN child.

    Benchmarking on for every plan; no disk cache; a private MIOpen user
    database, so the search neither reuses earlier tuning nor leaves entries
    that a later OOTB run would read. Forced, not defaulted.
    """
    env = dict(os.environ)
    env.update(
        {
            "HIPDNN_FORCE_BENCHMARKING": "1",
            "HIPDNN_DISABLE_CACHE": "1",
            "MIOPEN_USER_DB_PATH": state,
        }
    )
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
    tuning_available: bool,
    config: SuiteConfig,
) -> None:
    """Attach the engine's tuned run, timed in a child, as ``row.oracle``.

    Call after the OOTB buffers are released, so the child's allocations do
    not stack on top of them. ``tuning_available`` is whether the engine
    exposes ``global.benchmarking`` for this graph; without it the child
    re-measures the OOTB configuration. With ``reference_outputs``, the child
    saves its outputs and this process compares them with the references, as
    for the OOTB row. Never fails the row.
    """
    what = f"{graph_name} engine {engine_id_hex(engine_id)}"
    try:
        with tempfile.TemporaryDirectory(prefix="dnn-bench-hipdnn-tuned-") as state:
            args = _child_args(graph_path, config)
            args += ["--engine", engine_id_hex(engine_id)]
            if plugin_path is not None:
                args += ["--plugin-path", str(plugin_path)]
            if reference_outputs is not None:
                args.append("--save-outputs")
            result = _run_child(
                "tuned hipDNN",
                "hipdnn_tuned_child",
                args,
                _hipdnn_tuned_env(state),
                Path(state),
                config,
            )
            oracle = _oracle_from_result(result, tuning_available)
            if config.metrics.basic:
                oracle.workspace_bytes = result.get("workspace_bytes")
            if reference_outputs is not None:
                oracle.correctness = check_correctness(
                    _SavedOutputs(Path(state)),
                    tensor_infos,
                    reference_outputs,
                    config.validation.provider.value,
                    config,
                )
        if oracle.correctness is not None and oracle.correctness.explicitly_failed:
            warn_once(
                "oracle_correctness",
                f"tuned plan for {what} failed validation; no speedup is reported",
            )
        _set_throughputs(oracle, row)
        row.oracle = oracle
    except Exception as e:
        _record_error(row, what, e)


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
            args = _child_args(graph_path, config)
            args += ["--pytorch-sdpa-backend", config.pytorch_sdpa_backend.value]
            if config.pytorch_rocm_fa_library is not None:
                args += ["--pytorch-rocm-fa-library", config.pytorch_rocm_fa_library]
            result = _run_child(
                "tuned PyTorch",
                "pytorch_tuned_child",
                args,
                tuned_subprocess_env(state),
                Path(state),
                config,
            )
        oracle = _oracle_from_result(
            result,
            any(
                node.get("type") in _PYTORCH_TUNABLE_OPS
                for node in graph_json.get("nodes") or []
            ),
        )
        _set_throughputs(oracle, row)
        row.oracle = oracle
    except Exception as e:
        _record_error(row, f"PyTorch on {graph_name}", e)

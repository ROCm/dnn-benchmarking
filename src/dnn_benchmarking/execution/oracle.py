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

This module sets the environment of each child (``execution.tuned_child``
holds the contract with the scripts). The scripts set no tuning variable
and read no dnn-benchmark option. A child owns all tuning state. hipDNN
keeps a tuned winner in memory for the life of its process and serves it to
every later build of the same graph and engine. A tuned search also writes
winners to MIOpen's user database. The child's user database is in a
temporary directory that this module deletes, and hipDNN's disk cache is off.
Thus no OOTB measurement, including a stall-fallback re-measurement, can
observe the tuned selection.

The child gets the full timing policy, ``stall_gate`` included, so a graph
that fell back to unstalled timing times its tuned runs the same way.
This process validates the tuned hipDNN outputs that the child saves. A
failed tuned run sets ``row.oracle_error`` and never fails the row.
"""

import json
import subprocess
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ..config.benchmark_config import SuiteConfig
from ..graph.tensor_info import TensorInfo
from ..metrics._diagnostic import warn_once
from ..metrics._subprocess import run_capped
from ..metrics.analytical import derive_throughputs
from ..reporting.suite_results import (
    OracleResult,
    ProviderEngineResult,
    engine_id_hex,
)
from ..validation import ReferenceOutput
from .correctness import check_correctness
from .executor import Executor
from .tuned_child import (
    HIPDNN_TUNED_ENV,
    PYTORCH_TUNED_ENV,
    RESULT_FILE,
    ChildEnv,
    output_file,
    read_result,
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


def hipdnn_tuning_available(
    row: ProviderEngineResult, executor: Executor, engine_id: int
) -> Optional[bool]:
    """Whether the engine exposes ``global.benchmarking`` for this graph.

    Call after ``executor.prepare``. Without the knob, the tuned child
    re-measures the OOTB configuration. A failed query sets
    ``row.oracle_error`` and returns None (no tuned run). It never fails the
    row.
    """
    try:
        return BENCHMARKING_KNOB in executor.engine_knob_ids(engine_id)
    except Exception as e:
        _record_error(row, f"engine {engine_id_hex(engine_id)}", e)
        return None


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
    env: ChildEnv,
    out_dir: Path,
    config: SuiteConfig,
) -> None:
    """Run ``python -m dnn_benchmarking.cli.<module>`` in ``env``.

    ``out_dir`` is the private directory of the run. ``--profiling-timeout``
    limits the child.

    Raises:
        RuntimeError: If the child times out, exits nonzero or writes no
            ``result.json``. The message holds the last line of its stderr.
    """
    argv = [sys.executable, "-m", f"dnn_benchmarking.cli.{module}"]
    argv += [*args, "--out-dir", str(out_dir)]
    timeout_s = config.metrics.profiling_timeout_s or None
    try:
        proc = run_capped(argv, timeout_s, env=env.build(str(out_dir)))
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"{what} child timed out after {timeout_s} s (--profiling-timeout)"
        ) from None
    if proc.returncode != 0 or not (out_dir / RESULT_FILE).is_file():
        lines = [
            line.strip()
            for line in (proc.stderr or proc.stdout or "").splitlines()
            if line.strip()
        ]
        detail = lines[-1] if lines else "no output"
        raise RuntimeError(f"{what} child exited {proc.returncode}: {detail}")


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
            out_dir = Path(state)
            args = _child_args(graph_path, config)
            args += ["--engine", engine_id_hex(engine_id)]
            if plugin_path is not None:
                args += ["--plugin-path", str(plugin_path)]
            if reference_outputs is not None:
                args.append("--save-outputs")
            _run_child(
                "tuned hipDNN",
                "hipdnn_tuned_child",
                args,
                HIPDNN_TUNED_ENV,
                out_dir,
                config,
            )
            oracle = read_result(out_dir, tuning_available)
            if not config.metrics.basic:
                oracle.workspace_bytes = None
            if reference_outputs is not None:
                oracle.correctness = check_correctness(
                    _SavedOutputs(out_dir),
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
    tuning_available = any(
        node.get("type") in _PYTORCH_TUNABLE_OPS
        for node in graph_json.get("nodes") or []
    )
    try:
        with tempfile.TemporaryDirectory(prefix="dnn-bench-pytorch-tuned-") as state:
            out_dir = Path(state)
            args = _child_args(graph_path, config)
            args += ["--pytorch-sdpa-backend", config.pytorch_sdpa_backend.value]
            if config.pytorch_rocm_fa_library is not None:
                args += ["--pytorch-rocm-fa-library", config.pytorch_rocm_fa_library]
            _run_child(
                "tuned PyTorch",
                "pytorch_tuned_child",
                args,
                PYTORCH_TUNED_ENV,
                out_dir,
                config,
            )
            oracle = read_result(out_dir, tuning_available)
        _set_throughputs(oracle, row)
        row.oracle = oracle
    except Exception as e:
        _record_error(row, f"PyTorch on {graph_name}", e)

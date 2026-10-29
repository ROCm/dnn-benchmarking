# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Oracle pass: time a tuned run next to a row's OOTB run.

Every tuned run executes in a child script, after the row's OOTB run:

- hipDNN rows (:func:`run_hipdnn_tuned`): ``cli/hipdnn_child.py --tuned``
  builds the same engine with ``HIPDNN_FORCE_BENCHMARKING=1`` (the
  process-wide form of the ``global.benchmarking`` knob), so its first launch
  samples the candidate kernels and keeps the fastest.
- PyTorch rows (:func:`run_pytorch_tuned`): ``cli/pytorch_child.py
  --tuned`` runs the graph with exhaustive conv search and TunableOp.

This module sets the environment of each child (``execution.child_protocol``
holds the contract with the scripts, ``execution.child_runner`` starts
them). The scripts set no tuning variable and read no dnn-benchmark option.
A child owns all tuning state. hipDNN keeps a tuned winner in memory for the
life of its process and serves it to every later build of the same graph
and engine. A tuned search also writes winners to MIOpen's user database.
The child's user database is in a temporary directory that is deleted after
the run, and hipDNN's disk cache is off. Thus no OOTB measurement, including
a stall-fallback re-measurement, can observe the tuned selection.

The child gets the full timing policy, ``stall_gate`` included, so a graph
that fell back to unstalled timing times its tuned runs the same way.
This process validates the tuned hipDNN outputs that the child saves. A
failed tuned run sets ``row.oracle_error`` and never fails the row.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config.benchmark_config import SuiteConfig
from ..graph.tensor_info import TensorInfo
from ..metrics._diagnostic import warn_once
from ..metrics.analytical import derive_throughputs
from ..reporting.suite_results import (
    OracleResult,
    ProviderEngineResult,
    engine_id_hex,
)
from ..validation import ReferenceOutput
from .child_protocol import HIPDNN_TUNED, PYTORCH_TUNED, ChildResult, failure_text
from .child_runner import child_run, hipdnn_child_args, pytorch_child_args
from .correctness import check_correctness

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


def _record_error(row: ProviderEngineResult, what: str, message: str) -> None:
    """Record why the row has no tuned result; never fails the row."""
    row.oracle_error = message
    warn_once("oracle", f"tuned run failed for {what}: {message}")


def _engine_label(graph_name: str, engine_id: int) -> str:
    return f"{graph_name} engine {engine_id_hex(engine_id)}"


def hipdnn_tuning_available(
    row: ProviderEngineResult, graph_name: str, engine_id: int, ootb: ChildResult
) -> Optional[bool]:
    """Whether the engine exposes ``global.benchmarking`` for this graph,
    as the OOTB child found it.

    Without the knob, the tuned child re-measures the OOTB configuration. A
    failed query sets ``row.oracle_error`` and returns None (no tuned run).
    It never fails the row.
    """
    if ootb.tuning is None:
        return None
    if ootb.tuning.error is not None:
        _record_error(row, _engine_label(graph_name, engine_id), ootb.tuning.error)
    return ootb.tuning.available


def _oracle_result(
    tuned: ChildResult, tuning_available: bool, config: SuiteConfig
) -> OracleResult:
    return OracleResult(
        tuning_available=tuning_available,
        cpu_build_time_ms=tuned.build_ms,
        timing=tuned.timing,
        gpu_kernel_stats=tuned.kernel,
        host_stats=tuned.host,
        workspace_bytes=tuned.workspace_bytes if config.metrics.basic else None,
    )


def _set_throughputs(oracle: OracleResult, row: ProviderEngineResult) -> None:
    """Same FLOPs, bytes and median denominator as the row's throughputs."""
    median = oracle.gpu_kernel_stats.median_ms if oracle.gpu_kernel_stats else None
    oracle.derived_tflops_per_s, oracle.derived_gbytes_per_s = derive_throughputs(
        row.analytical_flops, row.analytical_io_bytes, median
    )


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

    Call after the OOTB child has exited, so the two runs never hold VRAM
    at the same time. ``tuning_available`` is whether the engine exposes
    ``global.benchmarking`` for this graph; without it the child re-measures
    the OOTB configuration. With ``reference_outputs``, the child saves its
    outputs and this process compares them with the references, as for the
    OOTB row. Never fails the row.
    """
    what = _engine_label(graph_name, engine_id)
    args = hipdnn_child_args(
        graph_path,
        config,
        engine_id,
        plugin_path,
        save_outputs=reference_outputs is not None,
    )
    try:
        with child_run(
            HIPDNN_TUNED, args, config.metrics.profiling_timeout_s or None
        ) as (tuned, outputs):
            oracle = _oracle_result(tuned, tuning_available, config)
            if reference_outputs is not None:
                oracle.correctness = check_correctness(
                    outputs,
                    tensor_infos,
                    reference_outputs,
                    config.validation.provider.value,
                    config,
                )
    except Exception as e:
        _record_error(row, what, failure_text(e))
        return
    if oracle.correctness is not None and oracle.correctness.explicitly_failed:
        warn_once(
            "oracle_correctness",
            f"tuned plan for {what} failed validation; no speedup is reported",
        )
    _set_throughputs(oracle, row)
    row.oracle = oracle


def run_pytorch_tuned(
    *,
    row: ProviderEngineResult,
    graph_path: Path,
    graph_json: Dict[str, Any],
    graph_name: str,
    config: SuiteConfig,
) -> None:
    """Attach a tuned PyTorch run as ``row.oracle``; never fails the row.

    Call after the OOTB child has exited, so the two runs never hold VRAM at
    the same time. Tuned outputs are not validated: the child does not save
    them. ``tuning_available`` is true only when the graph has a conv or
    matmul node, the ops the child searches.
    """
    tuning_available = any(
        node.get("type") in _PYTORCH_TUNABLE_OPS
        for node in graph_json.get("nodes") or []
    )
    args = pytorch_child_args(graph_path, config, save_outputs=False)
    try:
        with child_run(
            PYTORCH_TUNED, args, config.metrics.profiling_timeout_s or None
        ) as (tuned, _):
            oracle = _oracle_result(tuned, tuning_available, config)
    except Exception as e:
        _record_error(row, f"PyTorch on {graph_name}", failure_text(e))
        return
    _set_throughputs(oracle, row)
    row.oracle = oracle

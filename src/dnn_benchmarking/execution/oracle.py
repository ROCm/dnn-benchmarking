# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Oracle pass: time a tuned run next to a row's OOTB run.

Every tuned run executes in a child script, after the row's OOTB run:

- hipDNN rows (:func:`run_hipdnn_tuned`): ``cli/hipdnn_child.py --tuned``
  builds the same engine with ``HIPDNN_FORCE_BENCHMARKING=1`` (the
  process-wide form of the ``global.benchmarking`` knob), so its first launch
  samples the candidate kernels and keeps the fastest.
- PyTorch rows (:func:`run_pytorch_tuned`): ``cli/pytorch_tuned_child.py``
  runs the graph with exhaustive conv search and TunableOp.

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

import tempfile
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
from .child_protocol import HIPDNN_TUNED_ENV, PYTORCH_TUNED_ENV, read_result
from .child_runner import SavedOutputs, child_args, hipdnn_child_args, run_child
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


def record_oracle_error(row: ProviderEngineResult, what: str, message: str) -> None:
    """Record why the row has no tuned result; never fails the row."""
    row.oracle_error = message
    warn_once("oracle", f"tuned run failed for {what}: {message}")


def _record_error(row: ProviderEngineResult, what: str, e: Exception) -> None:
    record_oracle_error(row, what, f"{type(e).__name__}: {e}")


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
    what = f"{graph_name} engine {engine_id_hex(engine_id)}"
    try:
        with tempfile.TemporaryDirectory(prefix="dnn-bench-hipdnn-tuned-") as state:
            out_dir = Path(state)
            args = hipdnn_child_args(
                graph_path,
                config,
                engine_id,
                plugin_path,
                save_outputs=reference_outputs is not None,
            )
            run_child(
                "tuned hipDNN",
                "hipdnn_child",
                [*args, "--tuned"],
                HIPDNN_TUNED_ENV,
                out_dir,
                config,
            )
            oracle = read_result(out_dir, tuning_available)
            if not config.metrics.basic:
                oracle.workspace_bytes = None
            if reference_outputs is not None:
                oracle.correctness = check_correctness(
                    SavedOutputs(out_dir),
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
            args = child_args(graph_path, config)
            args += ["--pytorch-sdpa-backend", config.pytorch_sdpa_backend.value]
            if config.pytorch_rocm_fa_library is not None:
                args += ["--pytorch-rocm-fa-library", config.pytorch_rocm_fa_library]
            run_child(
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

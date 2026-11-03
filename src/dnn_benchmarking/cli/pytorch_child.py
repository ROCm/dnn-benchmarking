# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""PyTorch child process: one timed PyTorch run of one graph.

``python -m dnn_benchmarking.cli.pytorch_child --graph G --seed S
--timing-policy JSON --out-dir D --pytorch-sdpa-backend B
[--pytorch-rocm-fa-library L] [--save-outputs] [--clocks] [--tuned]``

The parent (``execution.child_runner.child_run``) runs every PyTorch run in
this script: the ``--runtime pytorch`` row, the ``--validate pytorch``
reference row, and with ``--oracle`` their tuned runs (``--tuned``). An OOTB
run starts with a new, empty MIOpen user database in D; a tuned run uses the
command's shared tuned database. The OOTB run checks
the environment of ``execution.child_protocol.PYTORCH_OOTB`` (the PyTorch
selection settings, TunableOp off); ``--tuned`` checks that of
``PYTORCH_TUNED`` (TunableOp on, with a results file in D) and turns on
PyTorch's exhaustive conv search (``torch.backends.cudnn.benchmark``). The
script sets no selection variable.

The script builds the graph and fills the inputs from the seed
(``execution.timed_run``), and times one loop with the parent's timing
policy (``cli.child_script``). It writes ``result.json`` in D, with the
executor's notes.
"""

import argparse
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..config.benchmark_config import (
    PyTorchSdpaBackendName,
    ReferencePrecision,
    TimingPolicy,
)
from ..execution.buffer_manager import generate_input_data
from ..execution.child_protocol import (
    PYTORCH_OOTB,
    PYTORCH_TUNED,
    child_parser,
    output_file,
    write_result,
)
from ..execution.timed_run import pytorch_run
from ..graph.loader import GraphLoader
from ..graph.tensor_info import TensorInfo
from .child_script import TimedRun, run_timed_child, timed_loop


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse the child argv that ``execution.child_runner`` builds."""
    parser = child_parser("pytorch_child")
    parser.add_argument(
        "--pytorch-sdpa-backend", required=True, type=PyTorchSdpaBackendName
    )
    parser.add_argument("--pytorch-rocm-fa-library")
    parser.add_argument(
        "--reference-precision",
        type=ReferencePrecision,
        default=ReferencePrecision.NATIVE,
        help="with --save-outputs: arithmetic of the saved outputs",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    """Time one PyTorch run; return the exit code."""
    args = parse_args(argv)
    specs = (PYTORCH_OOTB, PYTORCH_TUNED)
    return run_timed_child(args, specs, lambda: _prepare(args))


def _prepare(args: argparse.Namespace) -> TimedRun:
    """Turn on the exhaustive conv search for a tuned run, load the graph;
    return the timed run."""
    if args.tuned:
        import torch

        torch.backends.cudnn.benchmark = True
    graph_json, tensor_infos = GraphLoader().load(args.graph)
    return lambda policy: _timed_run(args, graph_json, tensor_infos, policy)


def _timed_run(
    args: argparse.Namespace,
    graph_json: Dict[str, Any],
    tensor_infos: List[TensorInfo],
    policy: TimingPolicy,
) -> None:
    """Build and time the graph with ``policy``; write the result files."""
    input_data = generate_input_data(tensor_infos, args.seed, graph_json)
    with pytorch_run(
        graph_json,
        tensor_infos,
        input_data,
        policy,
        args.pytorch_sdpa_backend,
        args.pytorch_rocm_fa_library,
    ) as (executor, bm):
        tensors = bm.get_tensors()
        m, clocks_after = timed_loop(
            args,
            bm,
            benchmark=lambda: executor.benchmark(tensors),
            run_once=lambda: executor.execute_once(tensors),
        )
        notes = list(executor.notes)
        device = executor.device
    precision = None
    if args.save_outputs:
        # After the timed buffers are released, so the two never share VRAM.
        precision, note = _save_float64_reference(args, graph_json, input_data, device)
        notes += [note] if note else []
    write_result(
        args.out_dir,
        m,
        clocks_after=clocks_after,
        notes=notes,
        output_precision=precision.value if precision else None,
    )


def _save_float64_reference(
    args: argparse.Namespace,
    graph_json: Dict[str, Any],
    input_data: Dict[int, np.ndarray],
    device: str,
) -> Tuple[ReferencePrecision, Optional[str]]:
    """Replace the saved outputs with a float64 reference when requested and
    the graph supports it; return the precision of the saved outputs and a
    note when the float64 reference was requested but not used.

    The float64 pass computes on the GPU from the same inputs. On a failure
    (for example an out-of-memory error) the native outputs stay.
    """
    if args.reference_precision is not ReferencePrecision.FLOAT64:
        return ReferencePrecision.NATIVE, None
    provider = _float64_provider()
    if not provider.supports_float64(graph_json):
        return ReferencePrecision.NATIVE, None
    try:
        outputs = provider.compute_reference(
            graph_json, input_data, precision=ReferencePrecision.FLOAT64, device=device
        )
    except Exception as e:
        reason = f"{type(e).__name__}: {e}"
        note = f"float64 reference not used ({reason}); using the graph data types"
        return ReferencePrecision.NATIVE, note
    for uid, ref in outputs.items():
        np.save(output_file(args.out_dir, uid), ref.data)
    return ReferencePrecision.FLOAT64, None


def _float64_provider() -> Any:
    """The PyTorch reference provider (imports torch)."""
    from ..validation.providers.pytorch_provider import PyTorchReferenceProvider

    return PyTorchReferenceProvider()


if __name__ == "__main__":
    sys.exit(main())

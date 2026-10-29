# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""PyTorch child process: one timed PyTorch run of one graph.

``python -m dnn_benchmarking.cli.pytorch_child --graph G --seed S
--timing-policy JSON --out-dir D --pytorch-sdpa-backend B
[--pytorch-rocm-fa-library L] [--save-outputs] [--clocks] [--tuned]``

The parent (``execution.child_runner.child_run``) runs every PyTorch run in
this script: the ``--runtime pytorch`` row, the ``--validate pytorch``
reference row, and with ``--oracle`` their tuned runs (``--tuned``). Each
run starts with a new, empty MIOpen user database in D. The OOTB run checks
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
from typing import Any, Dict, List, Optional

from ..config.benchmark_config import PyTorchSdpaBackendName, TimingPolicy
from ..execution.buffer_manager import generate_input_data
from ..execution.child_protocol import (
    PYTORCH_OOTB,
    PYTORCH_TUNED,
    child_parser,
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
    with pytorch_run(
        graph_json,
        tensor_infos,
        generate_input_data(tensor_infos, args.seed, graph_json),
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
    write_result(args.out_dir, m, clocks_after=clocks_after, notes=notes)


if __name__ == "__main__":
    sys.exit(main())

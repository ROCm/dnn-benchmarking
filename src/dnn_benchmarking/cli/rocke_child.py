# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""rocKE child process: one timed run of one rocKE kernel on one graph.

``python -m dnn_benchmarking.cli.rocke_child --graph G --seed S
--timing-policy JSON --out-dir D --kernel JSON [--save-outputs] [--clocks]``

The parent (``execution.child_runner.child_run``) runs every rocKE row in
this script. The parent selects the kernel and passes it as JSON
(``execution.rocke_sdpa.kernel_to_json``). The script checks the environment
of ``execution.child_protocol.ROCKE_OOTB`` and sets no selection variable.

The script reads the SDPA problem from the graph, fills the inputs from the
seed, loads the kernel, and times one loop with the parent's timing policy
(``cli.child_script``). It writes ``result.json`` in D, with the kernel load
time as ``build_ms``.
"""

import argparse
import sys
from typing import Any, Dict, List, Optional

from ..config.benchmark_config import TimingPolicy
from ..execution.buffer_manager import generate_input_data
from ..execution.child_protocol import ROCKE_OOTB, child_parser, write_result
from ..execution.rocke_sdpa import RockeKernel, kernel_from_json, sdpa_problem
from ..graph.loader import GraphLoader
from ..graph.tensor_info import TensorInfo
from .child_script import TimedRun, run_timed_child, timed_loop


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse the child argv that ``execution.child_runner`` builds."""
    parser = child_parser("rocke_child")
    parser.add_argument(
        "--kernel",
        required=True,
        help="the rocKE kernel as JSON (rocke_sdpa.kernel_to_json)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    """Time one rocKE kernel; return the exit code."""
    args = parse_args(argv)
    return run_timed_child(args, (ROCKE_OOTB,), lambda: _prepare(args))


def _prepare(args: argparse.Namespace) -> TimedRun:
    """Read the kernel and load the graph; return the timed run.

    The kernel is read here, not by argparse, so an invalid one reaches the
    parent through ``failure.json`` with its message.
    """
    kernel = kernel_from_json(args.kernel)
    graph_json, tensor_infos = GraphLoader().load(args.graph)
    return lambda policy: _timed_run(args, kernel, graph_json, tensor_infos, policy)


def _timed_run(
    args: argparse.Namespace,
    kernel: RockeKernel,
    graph_json: Dict[str, Any],
    tensor_infos: List[TensorInfo],
    policy: TimingPolicy,
) -> None:
    """Load and time ``kernel`` with ``policy``; write the result files."""
    from ..execution.pytorch_buffer_manager import PyTorchCudaBufferManager
    from ..execution.rocke_executor import RockeExecutor

    problem = sdpa_problem(graph_json, tensor_infos)
    input_data = generate_input_data(tensor_infos, args.seed, graph_json)
    executor = RockeExecutor(kernel, problem, policy)
    with PyTorchCudaBufferManager(tensor_infos, device=executor.device) as bm:
        bm.allocate_all()
        bm.load_input_data(input_data)
        bm.zero_outputs()
        executor.prepare(bm.get_tensors())
        m, clocks_after = timed_loop(
            args, bm, benchmark=executor.benchmark, run_once=executor.execute_once
        )
    write_result(
        args.out_dir, m, build_ms=executor.build_time_ms, clocks_after=clocks_after
    )


if __name__ == "__main__":
    sys.exit(main())

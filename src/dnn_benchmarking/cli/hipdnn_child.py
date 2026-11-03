# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""hipDNN child process: one timed run of one engine on one graph.

``python -m dnn_benchmarking.cli.hipdnn_child --graph G --engine 0xHEX
--seed S --timing-policy JSON --out-dir D [--plugin-path P] [--save-outputs]
[--clocks] [--tuned | --autotune]``

The parent (``execution.child_runner.child_run``) runs every hipDNN engine
run in this script. An OOTB run starts with a new, empty MIOpen user
database in D; a tuned or autotune run uses the command's shared tuned
database. The flags select the spec whose environment the script
checks (``execution.child_protocol``): ``HIPDNN_OOTB`` (benchmarking off),
``HIPDNN_AUTOTUNE`` (``--autotune``) or ``HIPDNN_TUNED`` (``--tuned``, the
``--oracle`` tuned run); the last two set ``HIPDNN_FORCE_BENCHMARKING=1``.
The script sets no selection variable.

The script builds the engine and fills the inputs from the seed
(``execution.timed_run``), and times one loop with the parent's timing
policy (``cli.child_script``). It writes ``result.json`` in D, with the
plan's build time and workspace and whether the engine exposes
``global.benchmarking`` for the graph. The script does not import
``torch``.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..common.rocm_runtime import initialize_pip_rocm_runtime
from ..config.benchmark_config import TimingPolicy
from ..execution.buffer_manager import generate_input_data
from ..execution.child_protocol import (
    BENCHMARKING_KNOB,
    HIPDNN_AUTOTUNE,
    HIPDNN_OOTB,
    HIPDNN_TUNED,
    TuningQuery,
    child_parser,
    failure_text,
    write_result,
)
from ..execution.executor import Executor, set_plugin_path
from ..execution.timed_run import build_hipdnn_plan, hipdnn_buffers
from ..graph.loader import GraphLoader
from ..graph.tensor_info import TensorInfo
from .child_script import TimedRun, run_timed_child, timed_loop
from .parser import parse_engine_id


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse the child argv that ``execution.child_runner`` builds."""
    parser = child_parser("hipdnn_child")
    parser.add_argument("--engine", required=True, type=parse_engine_id)
    parser.add_argument("--plugin-path", type=Path)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    """Time one engine run; return the exit code."""
    args = parse_args(argv)
    specs = (HIPDNN_OOTB, HIPDNN_AUTOTUNE, HIPDNN_TUNED)
    return run_timed_child(args, specs, lambda: _prepare(args))


def _prepare(args: argparse.Namespace) -> TimedRun:
    """Load hipDNN and the graph; return the timed run."""
    initialize_pip_rocm_runtime()
    import hipdnn_frontend as hipdnn

    set_plugin_path(hipdnn, args.plugin_path)
    handle = hipdnn.Handle()
    graph_json, tensor_infos = GraphLoader().load(args.graph)
    return lambda policy: _timed_run(args, handle, graph_json, tensor_infos, policy)


def _tuning_query(executor: Executor, engine_id: int) -> TuningQuery:
    """Whether the plan exposes the benchmarking knob; never fails the run."""
    try:
        return TuningQuery(BENCHMARKING_KNOB in executor.engine_knob_ids(engine_id))
    except Exception as e:
        return TuningQuery(None, failure_text(e))


def _timed_run(
    args: argparse.Namespace,
    handle: Any,
    graph_json: Dict[str, Any],
    tensor_infos: List[TensorInfo],
    policy: TimingPolicy,
) -> None:
    """Build and time the engine with ``policy``; write the result files."""
    executor = build_hipdnn_plan(handle, json.dumps(graph_json), policy, args.engine)
    tuning = _tuning_query(executor, args.engine)
    input_data = generate_input_data(tensor_infos, args.seed, graph_json)
    with hipdnn_buffers(tensor_infos, input_data) as (bm, variant_pack):
        m, clocks_after = timed_loop(
            args,
            bm,
            benchmark=lambda: executor.benchmark(handle, variant_pack),
            run_once=lambda: executor.execute_once(handle, variant_pack),
        )
    write_result(
        args.out_dir,
        m,
        build_ms=executor.build_time_ms,
        workspace_bytes=executor.workspace_size,
        clocks_after=clocks_after,
        tuning=tuning,
    )


if __name__ == "__main__":
    sys.exit(main())

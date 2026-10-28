# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""hipDNN child process: one timed run of one engine on one graph.

``python -m dnn_benchmarking.cli.hipdnn_child --graph G --engine 0xHEX
--seed S --timing-policy JSON --out-dir D [--plugin-path P] [--save-outputs]
[--clocks] [--tuned]``

The parent (``execution.child_runner.child_run``) runs every hipDNN engine
run in this script, so each run starts with a new, empty MIOpen user
database in D. The OOTB run checks ``execution.child_protocol.HIPDNN_OOTB_ENV``;
``--tuned`` checks ``HIPDNN_TUNED_ENV`` (``HIPDNN_FORCE_BENCHMARKING=1``).
The script sets no selection variable.

The script builds the engine and fills the inputs from the seed with the
same steps as every hipDNN run (``execution.timed_run``), and times one loop
with the parent's timing policy. It writes ``result.json`` in D, with the
GPU clocks right after the loop (``--clocks``) and whether the engine
exposes ``global.benchmarking`` for the graph. With ``--save-outputs``, it
runs the graph once more and saves each output tensor in D.

When the stall gate fails, a tuned run times again without it. An OOTB run
reports the failure instead: the parent then remeasures the whole graph
without the gate, so all rows of the graph share one timing mode. Any
failure goes to ``failure.json`` and the exit code is 1. The script does not
import ``torch``.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ..common.rocm_runtime import initialize_pip_rocm_runtime
from ..config.benchmark_config import TimingPolicy
from ..execution.buffer_manager import generate_input_data
from ..execution.child_protocol import (
    BENCHMARKING_KNOB,
    HIPDNN_OOTB_ENV,
    HIPDNN_TUNED_ENV,
    TuningQuery,
    child_parser,
    failure_text,
    output_file,
    run_reporting_failures,
    write_result,
)
from ..execution.executor import Executor, set_plugin_path
from ..execution.timed_run import build_hipdnn_plan, hipdnn_buffers
from ..execution.timing import retry_unstalled
from ..graph.loader import GraphLoader
from ..graph.tensor_info import TensorInfo
from ..metrics import GpuSmiProbe
from .parser import parse_engine_id


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse the child argv that ``execution.child_runner`` builds."""
    parser = child_parser("hipdnn_child")
    parser.add_argument("--engine", required=True, type=parse_engine_id)
    parser.add_argument("--plugin-path", type=Path)
    parser.add_argument("--save-outputs", action="store_true")
    parser.add_argument("--clocks", action="store_true")
    parser.add_argument("--tuned", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    """Time one engine run; return the exit code."""
    args = parse_args(argv)
    return run_reporting_failures(args.out_dir, lambda: _run(args))


def _run(args: argparse.Namespace) -> None:
    (HIPDNN_TUNED_ENV if args.tuned else HIPDNN_OOTB_ENV).check()
    initialize_pip_rocm_runtime()
    import hipdnn_frontend as hipdnn

    set_plugin_path(hipdnn, args.plugin_path)
    handle = hipdnn.Handle()

    loader = GraphLoader()
    graph_json = loader.load_json(args.graph)
    loader.validate(graph_json)
    tensor_infos = loader.extract_tensor_info(graph_json)

    def run(policy: TimingPolicy) -> None:
        _timed_run(args, handle, graph_json, tensor_infos, policy)

    if args.tuned:
        retry_unstalled(run, args.timing_policy)
    else:
        run(args.timing_policy)


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
        probe = GpuSmiProbe() if args.clocks else None
        m = executor.benchmark(handle, variant_pack)
        clocks_after = probe.clocks() if probe is not None else None
        if args.save_outputs:
            bm.zero_outputs()
            executor.execute_once(handle, variant_pack)
            for tensor_info in bm.get_output_tensors():
                data = bm.get_output_data(tensor_info.uid)
                if data is not None:
                    np.save(output_file(args.out_dir, tensor_info.uid), data)
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

# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""hipDNN child process: one timed run of one engine on one graph.

``python -m dnn_benchmarking.cli.hipdnn_child --graph G --engine 0xHEX
--seed S --timing-policy JSON --out-dir D [--plugin-path P] [--save-outputs]
[--tuned | [--clocks] [--query-tuning]]``

The parent (``execution.child_runner``) runs every hipDNN engine row in this
script, so each run starts with a new, empty MIOpen user database in D.

- OOTB (default): the parent sets ``execution.child_protocol.HIPDNN_OOTB_ENV``.
  ``--clocks`` samples the GPU clocks right after the timed loop.
  ``--query-tuning`` reports whether the engine exposes
  ``global.benchmarking`` for the graph. A stall-gate failure is not retried
  here: the parent remeasures the whole graph without the gate, so all rows
  of the graph share one timing mode. On a skip, a stall or an error the
  script writes ``failure.json``.
- ``--tuned``: the parent sets ``HIPDNN_TUNED_ENV``
  (``HIPDNN_FORCE_BENCHMARKING=1``). A stall-gate failure reruns the tuned
  run without the gate. On an error the script writes it on stderr.

Both modes check the environment and set no selection variable. They build
the engine and fill the inputs from the seed with the same steps
(``execution.timed_run``), time one loop with the parent's timing policy and
write ``result.json`` in D. With ``--save-outputs``, the script runs the graph
once more and saves each output tensor in D. It does not import ``torch``.
It exits 1 on an error.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..common.exceptions import UnsupportedGraphError
from ..common.rocm_runtime import initialize_pip_rocm_runtime
from ..config.benchmark_config import TimingPolicy
from ..execution.buffer_manager import generate_input_data
from ..execution.child_protocol import (
    BENCHMARKING_KNOB,
    FAILED,
    HIPDNN_OOTB_ENV,
    HIPDNN_TUNED_ENV,
    SKIPPED,
    STALLED,
    child_parser,
    output_file,
    write_failure,
    write_result,
)
from ..execution.executor import Executor, set_plugin_path
from ..execution.timed_run import build_hipdnn_plan, hipdnn_buffers
from ..execution.timing import StallFallbackError, retry_unstalled
from ..graph.loader import GraphLoader
from ..graph.tensor_info import TensorInfo
from ..metrics import GpuSmiProbe
from ..reporting.reporter import Reporter
from ..reporting.suite_results import engine_id_hex
from .parser import parse_engine_id


def parse_args(argv: Optional[List[str]] = None):
    """Parse the child argv that ``execution.child_runner`` builds."""
    parser = child_parser("hipdnn_child")
    parser.add_argument("--engine", required=True, type=parse_engine_id)
    parser.add_argument("--plugin-path", type=Path)
    parser.add_argument("--save-outputs", action="store_true")
    parser.add_argument("--tuned", action="store_true")
    parser.add_argument("--clocks", action="store_true")
    parser.add_argument("--query-tuning", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    """Time one engine run; return the exit code."""
    args = parse_args(argv)
    reporter = Reporter(quiet=True)
    try:
        (HIPDNN_TUNED_ENV if args.tuned else HIPDNN_OOTB_ENV).check()
        initialize_pip_rocm_runtime()
        import hipdnn_frontend as hipdnn
    except (RuntimeError, ImportError) as e:
        return _fail(args, reporter, f"{type(e).__name__}: {e}")

    try:
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
    except StallFallbackError as e:
        if args.tuned:
            return _fail(args, reporter, f"{type(e).__name__}: {e}")
        write_failure(args.out_dir, STALLED, str(e))
    except UnsupportedGraphError as e:
        if args.tuned:
            return _fail(args, reporter, f"{type(e).__name__}: {e}")
        write_failure(args.out_dir, SKIPPED, str(e))
    except Exception as e:
        return _fail(args, reporter, f"{type(e).__name__}: {e}")
    return 0


def _fail(args: argparse.Namespace, reporter: Reporter, message: str) -> int:
    """Report an error: ``failure.json`` (OOTB) and stderr; return 1."""
    if not args.tuned:
        write_failure(args.out_dir, FAILED, message)
    reporter.error(f"engine {engine_id_hex(args.engine)}: {message}")
    return 1


def _tuning_query(
    executor: Executor, engine_id: int
) -> Tuple[Optional[bool], Optional[str]]:
    """``(tuning_available, error)``: whether the plan exposes the knob."""
    try:
        return BENCHMARKING_KNOB in executor.engine_knob_ids(engine_id), None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def _timed_run(
    args: argparse.Namespace,
    handle: Any,
    graph_json: Dict[str, Any],
    tensor_infos: List[TensorInfo],
    policy: TimingPolicy,
) -> None:
    """Build and time the engine with ``policy``; write the result files."""
    executor = build_hipdnn_plan(handle, json.dumps(graph_json), policy, args.engine)
    tuning_available, tuning_error = (
        _tuning_query(executor, args.engine) if args.query_tuning else (None, None)
    )
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
        tuning_available=tuning_available,
        tuning_error=tuning_error,
    )


if __name__ == "__main__":
    sys.exit(main())

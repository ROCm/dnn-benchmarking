# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tuned-hipDNN child process of ``--oracle``.

``python -m dnn_benchmarking.cli.hipdnn_tuned_child --graph G --engine 0xHEX
--seed S --timing-policy JSON --out-dir D [--plugin-path P] [--save-outputs]``

The parent (``execution.oracle.run_hipdnn_tuned``) sets the environment
``execution.tuned_child.HIPDNN_TUNED_ENV``: ``HIPDNN_FORCE_BENCHMARKING=1``,
``HIPDNN_DISABLE_CACHE=1`` and a private ``MIOPEN_USER_DB_PATH``. This script
checks it and sets no tuning variable. It builds the engine and fills the
inputs from the seed with the same steps as the OOTB row
(``execution.timed_run``), then times one loop with the parent's timing
policy. It writes
``result.json`` in D. With ``--save-outputs``, it runs the graph once more
and saves each output tensor in D. It does not import ``torch``.
On failure it writes an error on stderr and exits 1.
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
from ..execution.executor import set_plugin_path
from ..execution.timed_run import build_hipdnn_plan, hipdnn_buffers
from ..execution.timing import retry_unstalled
from ..execution.tuned_child import (
    HIPDNN_TUNED_ENV,
    child_parser,
    output_file,
    write_result,
)
from ..graph.loader import GraphLoader
from ..graph.tensor_info import TensorInfo
from ..reporting.reporter import Reporter
from ..reporting.suite_results import engine_id_hex
from .parser import parse_engine_id


def parse_args(argv: Optional[List[str]] = None):
    """Parse the child argv that ``execution.oracle`` builds."""
    parser = child_parser("hipdnn_tuned_child")
    parser.add_argument("--engine", required=True, type=parse_engine_id)
    parser.add_argument("--plugin-path", type=Path)
    parser.add_argument("--save-outputs", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    """Time one tuned engine run; return the exit code."""
    args = parse_args(argv)
    reporter = Reporter(quiet=True)
    try:
        HIPDNN_TUNED_ENV.check()
    except RuntimeError as e:
        reporter.error(str(e))
        return 1
    try:
        initialize_pip_rocm_runtime()
        import hipdnn_frontend as hipdnn
    except (RuntimeError, ImportError) as e:
        reporter.error(f"hipDNN unavailable: {e}")
        return 1

    try:
        set_plugin_path(hipdnn, args.plugin_path)
        handle = hipdnn.Handle()

        loader = GraphLoader()
        graph_json = loader.load_json(args.graph)
        loader.validate(graph_json)
        tensor_infos = loader.extract_tensor_info(graph_json)

        def run(policy: TimingPolicy) -> None:
            _tuned_run(args, handle, graph_json, tensor_infos, policy)

        retry_unstalled(run, args.timing_policy)
    except Exception as e:
        reporter.error(f"engine {engine_id_hex(args.engine)}: {type(e).__name__}: {e}")
        return 1
    return 0


def _tuned_run(
    args: argparse.Namespace,
    handle: Any,
    graph_json: Dict[str, Any],
    tensor_infos: List[TensorInfo],
    policy: TimingPolicy,
) -> None:
    """Build and time the engine with ``policy``; write the result files."""
    executor = build_hipdnn_plan(handle, json.dumps(graph_json), policy, args.engine)
    input_data = generate_input_data(tensor_infos, args.seed, graph_json)
    with hipdnn_buffers(tensor_infos, input_data) as (bm, variant_pack):
        m = executor.benchmark(handle, variant_pack)
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
    )


if __name__ == "__main__":
    sys.exit(main())

# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tuned hipDNN child process: the ``--oracle`` tuned run of one engine.

``python -m dnn_benchmarking.cli.hipdnn_tuned_child --graph G --engine ID
--seed S --timing-policy JSON --out-dir D [--plugin-path P] [--save-outputs]``

The parent (``execution.oracle.run_hipdnn_tuned``) starts this script after
the row's OOTB run. The tuned run needs its own process: MIOpen reads
``MIOPEN_USER_DB_PATH`` once per process, and the OOTB run uses the OOTB
selection state while the tuned run uses the tuned one that every tuned run
shares (``common.work_dir``). The parent sets ``HIPDNN_CACHE_DIR``,
``MIOPEN_USER_DB_PATH`` and ``execution.oracle.TUNED_CHILD_ENV``; this
script checks all of them before hipDNN loads.

The script primes the engine, builds its plan with ``global.benchmarking=1``,
fills the inputs from the seed, and times the plan with the parent's timing
policy. The first, untimed launch runs the search. It writes D/result.json
(the timed loop, build time, workspace and whether the engine exposes the
knob) and, with ``--save-outputs``, one ``output_<uid>.npy`` per output for
the parent's validation. Before that output run, the outputs are zeroed and
a Philox ``rng_dump`` output is filled with NaN instead. A failure goes to
D/failure.json: kind ``stall`` when the stall gate failed (the parent then
remeasures the whole graph without it), else ``error``.
"""

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional

import numpy as np

from ..common.rocm_runtime import initialize_pip_rocm_runtime
from ..config.benchmark_config import TimingPolicy
from ..execution.buffer_manager import BufferManager, generate_input_data
from ..execution.executor import Executor
from ..execution.oracle import (
    BENCHMARKING_KNOB,
    FAILURE_FILE,
    RESULT_FILE,
    TUNED_CHILD_ENV,
    output_file,
)
from ..execution.sdpa_dropout import find_rng_dump
from ..execution.suite_runner import set_plugin_path
from ..execution.timing import StallFallbackError
from ..graph.loader import GraphLoader


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse the argv that ``execution.oracle`` builds."""
    parser = argparse.ArgumentParser(prog="hipdnn_tuned_child")
    parser.add_argument("--graph", required=True, type=Path)
    parser.add_argument("--engine", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument(
        "--timing-policy",
        required=True,
        type=lambda text: TimingPolicy(**json.loads(text)),
        help="TimingPolicy fields as a JSON object, stall_gate included",
    )
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--plugin-path", type=Path)
    parser.add_argument("--save-outputs", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    """Run the tuned plan once; return the exit code."""
    args = parse_args(argv)
    try:
        _run(args)
    except StallFallbackError as e:
        return _fail(args.out_dir, "stall", str(e))
    except Exception as e:
        return _fail(args.out_dir, "error", f"{type(e).__name__}: {e}")
    return 0


def _fail(out_dir: Path, kind: str, message: str) -> int:
    (out_dir / FAILURE_FILE).write_text(json.dumps({"kind": kind, "message": message}))
    return 1


def _run(args: argparse.Namespace) -> None:
    wrong = [
        f"{name}={os.environ.get(name)!r}"
        for name, value in TUNED_CHILD_ENV.items()
        if os.environ.get(name) != value
    ]
    for name in ("HIPDNN_CACHE_DIR", "MIOPEN_USER_DB_PATH"):
        path = os.environ.get(name)
        if not path or not Path(path).is_dir():
            wrong.append(f"{name}={path!r} is not a directory")
    if wrong:
        raise RuntimeError(f"tuned child environment not set by the parent: {wrong}")
    initialize_pip_rocm_runtime()
    import hipdnn_frontend as hipdnn

    set_plugin_path(hipdnn, args.plugin_path)
    handle = hipdnn.Handle()
    loader = GraphLoader()
    graph_json = loader.load_json(args.graph)
    loader.validate(graph_json)
    tensor_infos = loader.extract_tensor_info(graph_json)
    rng_dump = find_rng_dump(graph_json)
    graph_json_str = json.dumps(graph_json)
    policy: TimingPolicy = args.timing_policy

    # Absorb the engine's one-time build costs before the timed build, as
    # the OOTB row does.
    Executor(graph_json_str, policy).prime(handle, args.engine)
    executor = Executor(graph_json_str, policy)
    executor.prepare(handle, engine_id=args.engine, knobs={BENCHMARKING_KNOB: 1})
    # hipDNN ignores a knob the engine does not expose; such a tuned run
    # re-measures the OOTB configuration.
    tuning_available = BENCHMARKING_KNOB in executor.engine_knob_ids(args.engine)
    with BufferManager(tensor_infos) as bm:
        bm.allocate_all()
        bm.load_input_data(generate_input_data(tensor_infos, args.seed, graph_json))
        bm.zero_outputs()
        variant_pack = bm.create_variant_pack()
        m = executor.benchmark(handle, variant_pack)
        if args.save_outputs:
            # An engine that does not write its rng_dump then leaves NaN,
            # which the parent's dump check rejects.
            if rng_dump is not None:
                bm.zero_outputs(nan_uids={rng_dump.uid})
            else:
                bm.zero_outputs()
            executor.execute_once(handle, variant_pack)
            for tensor_info in bm.get_output_tensors():
                data = bm.get_output_data(tensor_info.uid)
                if data is not None:
                    np.save(output_file(args.out_dir, tensor_info.uid), data)
    result = {
        "measurement": asdict(m),
        "build_ms": executor.build_time_ms,
        "workspace_bytes": executor.workspace_size,
        "tuning_available": tuning_available,
    }
    (args.out_dir / RESULT_FILE).write_text(json.dumps(result))


if __name__ == "__main__":
    sys.exit(main())

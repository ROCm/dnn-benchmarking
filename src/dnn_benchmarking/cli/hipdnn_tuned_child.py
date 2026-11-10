# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tuned hipDNN child process: every ``--oracle`` tuned run of a command.

``python -m dnn_benchmarking.cli.hipdnn_tuned_child --manifest M``

The parent (``execution.oracle.HipdnnTunedRuns``) starts this script after
the last graph. The tuned runs need their own process: MIOpen reads
``MIOPEN_USER_DB_PATH`` once per process, and the OOTB runs use the OOTB
selection state while the tuned runs use the tuned one (``common.work_dir``).
The parent sets ``HIPDNN_CACHE_DIR``, ``MIOPEN_USER_DB_PATH`` and
``execution.oracle.TUNED_CHILD_ENV``; this script checks all of them before
hipDNN loads.

The manifest holds the seed and one job per row, in row order: graph,
engine, plugin path, timing policy, output directory and whether to save
the outputs. For each job the script makes a new handle, primes the engine,
builds its plan with ``global.benchmarking=1``, fills the inputs from the
seed, and times the plan with the job's timing policy. The first, untimed
launch runs the search. It then writes ``result.json`` (the timed loop,
build time, workspace and whether the engine exposes the knob) and, when
asked, one ``output_<uid>.npy`` per output in the job's directory. A failed
job writes ``failure.json`` instead: kind ``stall`` when the stall gate
failed (the parent then remeasures the whole graph without it), else
``error``. A failed job does not stop the next one.
"""

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
from ..execution.suite_runner import set_plugin_path
from ..execution.timing import StallFallbackError
from ..graph.loader import GraphLoader


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse the argv that ``execution.oracle`` builds."""
    parser = argparse.ArgumentParser(prog="hipdnn_tuned_child")
    parser.add_argument("--manifest", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    """Run every job of the manifest; return the exit code.

    0 when every job wrote its result or failure file. 1 when the
    environment check or hipDNN loading failed, before any job ran.
    """
    args = parse_args(argv)
    manifest = json.loads(args.manifest.read_text())
    try:
        hipdnn = _load_hipdnn()
    except Exception as e:
        print(f"hipdnn_tuned_child: error: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    for job in manifest["jobs"]:
        failure = _run_job(hipdnn, job, manifest["seed"])
        if failure is not None:
            kind, message = failure
            (Path(job["out_dir"]) / FAILURE_FILE).write_text(
                json.dumps({"kind": kind, "message": message})
            )
    return 0


def _load_hipdnn() -> Any:
    """Check the environment that the parent set; import hipDNN."""
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

    return hipdnn


def _run_job(hipdnn: Any, job: Dict[str, Any], seed: int) -> Optional[Tuple[str, str]]:
    """Time one row's tuned plan on a new handle; return ``(kind, message)``
    when it failed, else None.

    A new handle per job, as a child per row had: no in-memory provider state
    of an earlier job reaches this one. The handle is destroyed after the
    plan and the buffers are released, before the next job starts.
    """
    try:
        set_plugin_path(
            hipdnn, Path(job["plugin_path"]) if job["plugin_path"] else None
        )
        handle = hipdnn.Handle()
    except Exception as e:
        return "error", f"{type(e).__name__}: {e}"
    failure: Optional[Tuple[str, str]] = None
    try:
        _time_tuned_plan(handle, job, seed)
    except StallFallbackError as e:
        failure = "stall", str(e)
    except Exception as e:
        failure = "error", f"{type(e).__name__}: {e}"
    # After the except blocks: their tracebacks hold the plan and buffers.
    hipdnn.destroy_handle(handle)
    return failure


def _time_tuned_plan(handle: Any, job: Dict[str, Any], seed: int) -> None:
    """Build and time the tuned plan of one row; write its files."""
    out_dir = Path(job["out_dir"])
    engine_id: int = job["engine"]
    policy = TimingPolicy(**job["timing_policy"])
    loader = GraphLoader()
    graph_json = loader.load_json(Path(job["graph"]))
    loader.validate(graph_json)
    tensor_infos = loader.extract_tensor_info(graph_json)
    graph_json_str = json.dumps(graph_json)

    # Absorb the engine's one-time build costs before the timed build, as
    # the OOTB row does.
    Executor(graph_json_str, policy).prime(handle, engine_id)
    executor = Executor(graph_json_str, policy)
    executor.prepare(handle, engine_id=engine_id, knobs={BENCHMARKING_KNOB: 1})
    # hipDNN ignores a knob the engine does not expose; such a tuned run
    # re-measures the OOTB configuration.
    tuning_available = BENCHMARKING_KNOB in executor.engine_knob_ids(engine_id)
    with BufferManager(tensor_infos) as bm:
        bm.allocate_all()
        bm.load_input_data(generate_input_data(tensor_infos, seed, graph_json))
        bm.zero_outputs()
        variant_pack = bm.create_variant_pack()
        m = executor.benchmark(handle, variant_pack)
        if job["save_outputs"]:
            bm.zero_outputs()
            executor.execute_once(handle, variant_pack)
            for tensor_info in bm.get_output_tensors():
                data = bm.get_output_data(tensor_info.uid)
                if data is not None:
                    np.save(output_file(out_dir, tensor_info.uid), data)
    result = {
        "measurement": asdict(m),
        "build_ms": executor.build_time_ms,
        "workspace_bytes": executor.workspace_size,
        "tuning_available": tuning_available,
    }
    (out_dir / RESULT_FILE).write_text(json.dumps(result))


if __name__ == "__main__":
    sys.exit(main())

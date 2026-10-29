# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Entry point of the tuned-hipDNN child process (``--oracle``).

``python -m dnn_benchmarking.execution.hipdnn_tuned_child <request.json>``
builds one engine's plan with ``global.benchmarking=1`` and times it like the
OOTB row. hipDNN keeps a tuned winner in memory and serves it to every later
build of the same graph and engine, so the search must not run in the
benchmark process. The parent (``execution.oracle``) writes the request and
supplies the environment (disk caches off, a private MIOpen user database).

Request keys: ``graph_json`` (string), ``engine_id``, ``plugin_path`` (or
null), ``seed``, ``policy`` (``TimingPolicy`` fields), ``outputs`` (write the
graph outputs for validation), ``workspace`` (report the workspace size).

The child writes ``result.json`` beside the request: ``tuning_available``,
``cpu_build_time_ms``, ``workspace_bytes`` and ``measurement``
(``timing.Measurement`` fields); or ``stall`` (the stall gate failed, so the
parent remeasures the whole graph) or ``error``. With ``outputs``, it also
writes ``outputs.npz``: one dense logical array per output UID.
"""

import dataclasses
import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np

#: hipDNN knob that makes a provider sample its candidate kernels on the first
#: execute() and keep the fastest (kernel ingestor, MIOpen).
BENCHMARKING_KNOB = "global.benchmarking"


def _run(request: Dict[str, Any], directory: Path) -> Dict[str, Any]:
    from ..common.rocm_runtime import initialize_pip_rocm_runtime
    from ..config.benchmark_config import TimingPolicy
    from ..graph.loader import GraphLoader
    from .buffer_manager import BufferManager, generate_input_data
    from .executor import Executor
    from .suite_runner import set_plugin_path

    initialize_pip_rocm_runtime()
    import hipdnn_frontend as hipdnn

    set_plugin_path(hipdnn, request["plugin_path"])
    handle = hipdnn.Handle()
    graph_json_str = request["graph_json"]
    graph_json = json.loads(graph_json_str)
    tensor_infos = GraphLoader().extract_tensor_info(graph_json)
    policy = TimingPolicy(**request["policy"])
    engine_id = request["engine_id"]

    # Same build state as the OOTB row: one untimed build absorbs the
    # engine's one-time costs, then the timed build.
    Executor(graph_json_str, policy).prime(handle, engine_id)
    executor = Executor(graph_json_str, policy)
    executor.prepare(handle, engine_id=engine_id, knobs={BENCHMARKING_KNOB: 1})
    # hipDNN ignores a knob the engine does not expose; such a tuned run
    # re-measures the OOTB configuration.
    result: Dict[str, Any] = {
        "tuning_available": BENCHMARKING_KNOB in executor.engine_knob_ids(engine_id),
        "cpu_build_time_ms": executor.build_time_ms,
        "workspace_bytes": executor.workspace_size if request["workspace"] else None,
    }
    with BufferManager(tensor_infos) as bm:
        bm.allocate_all()
        bm.load_input_data(
            generate_input_data(tensor_infos, request["seed"], graph_json)
        )
        bm.zero_outputs()
        variant_pack = bm.create_variant_pack()
        # The untimed first launch runs the search (see timing.measure).
        result["measurement"] = dataclasses.asdict(
            executor.benchmark(handle, variant_pack)
        )
        if request["outputs"]:
            bm.zero_outputs()
            executor.execute_once(handle, variant_pack)
            outputs = {
                str(ti.uid): bm.get_output_data(ti.uid)
                for ti in tensor_infos
                if ti.is_output
            }
            np.savez(
                directory / "outputs.npz",
                **{uid: data for uid, data in outputs.items() if data is not None},
            )
    return result


def main(argv: list) -> int:
    from .timing import StallFallbackError

    request_path = Path(argv[1])
    request = json.loads(request_path.read_text())
    try:
        result = _run(request, request_path.parent)
    except StallFallbackError as e:
        result = {"stall": str(e)}
    except Exception as e:
        result = {"error": f"{type(e).__name__}: {e}"}
    (request_path.parent / "result.json").write_text(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

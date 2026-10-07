# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Hidden ``--internal-profiling-run`` sub-mode.

This is the workload that the profiling orchestrator wraps under
rocprofv3 / perf / rocprof-compute. We deliberately re-exec the whole
process (rather than running another loop in-place) because the outer
profiler expects a fresh process tree and a clean address space — that
is how kernel-trace / PMC counters scope what they record.

Argv (built by ``profiling_orchestrator.build_inner_argv``):
``--internal-profiling-run --graph G --engine E --warmup W --iters I
--seed S [--plugin-path P]``. The child builds the one engine, fills the
inputs exactly as the timed pass did, and dispatches ``W + I`` plain
executions. It prints nothing on success so profiler log capture stays
clean.
"""

import argparse
import json
import sys
from pathlib import Path

from ..common.rocm_runtime import initialize_pip_rocm_runtime
from ..config.benchmark_config import TimingPolicy
from ..execution.buffer_manager import BufferManager, generate_input_data
from ..execution.executor import Executor
from ..execution.suite_runner import set_plugin_path
from ..execution.timing import device_sync
from ..graph.loader import GraphLoader


def _fail(msg: str) -> int:
    print(f"internal-profiling-run: {msg}", file=sys.stderr)
    return 1


def run_internal_profiling(args: argparse.Namespace) -> int:
    """Run a single (graph, engine) workload for the orchestrator. Returns exit code."""
    if not args.graph or len(args.graph) != 1:
        return _fail("expected exactly one --graph")
    if not args.engine or len(args.engine) != 1:
        return _fail("expected exactly one --engine")
    if args.plugin_path and len(args.plugin_path) != 1:
        return _fail("expected at most one --plugin-path")
    graph_path = Path(args.graph[0])
    engine_id = args.engine[0]
    plugin_path = args.plugin_path[0] if args.plugin_path else None
    # Warm cache, fixed count: the executor needs a policy, but nothing is timed.
    policy = TimingPolicy(
        warmup_iters=args.warmup,
        iters=args.iters,
        min_time_ms=0.0,
        cache_mode="warm",
    )

    try:
        initialize_pip_rocm_runtime()
        import hipdnn_frontend as hipdnn
    except (RuntimeError, ImportError) as e:
        return _fail(f"hipDNN unavailable: {e}")

    try:
        set_plugin_path(hipdnn, plugin_path)
        handle = hipdnn.Handle()

        loader = GraphLoader()
        graph_json = loader.load_json(graph_path)
        loader.validate(graph_json)
        tensor_infos = loader.extract_tensor_info(graph_json)

        executor = Executor(json.dumps(graph_json), policy)
        executor.prepare(handle, engine_id=engine_id)
        with BufferManager(tensor_infos) as bm:
            bm.allocate_all()
            # graph_json matters for paged SDPA: page tables and sequence
            # lengths need valid integers, not uniform noise.
            bm.load_input_data(generate_input_data(tensor_infos, args.seed, graph_json))
            vp = bm.create_variant_pack()
            # No Executor.benchmark(): rocprofv3 --pmc serializes dispatches
            # and deadlocks on the staged timer's stall gate. No execute_once()
            # either: its workspace reset would add a memset dispatch to every
            # profiled iteration. Same submissions as the timed loop, one drain.
            for _ in range(policy.warmup_iters + policy.iters):
                executor.enqueue(handle, vp)
            device_sync("hip")
    except Exception as e:
        return _fail(f"engine {engine_id}: {type(e).__name__}: {e}")
    return 0

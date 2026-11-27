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
--seed S [--plugin-path P] [--input-recipe R]
[--cache-mode rotate --rotate-buffers N]``. The child builds the one
engine, fills the inputs exactly as the timed pass did, and dispatches
``W + I`` plain executions (cycling ``N`` independent buffer sets when
``--cache-mode rotate`` is set so L2/MALL counters match rotated timings
without adding extra flush kernels). It prints nothing on success so
profiler log capture stays clean.
"""

import argparse
from contextlib import ExitStack
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
from ..graph.varlen import resolve_varlen_layouts


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
    cache_mode = getattr(args, "cache_mode", "warm")
    raw_rot = int(getattr(args, "rotate_buffers", 0) or 0)
    n_rot = max(2, raw_rot) if cache_mode == "rotate" and raw_rot > 1 else 1
    # Fixed count: the executor needs a policy, but nothing is timed.
    policy = TimingPolicy(
        warmup_iters=args.warmup,
        iters=args.iters,
        min_time_ms=0.0,
        cache_mode="rotate" if n_rot > 1 else "warm",
        rotate_buffers=n_rot if n_rot > 1 else 0,
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
        # sequence lengths need valid integers, not uniform noise.
        recipe = getattr(args, "input_recipe", "centered")
        if recipe == "centered":
            input_data = generate_input_data(tensor_infos, args.seed, graph_json)
        else:
            input_data = generate_input_data(
                tensor_infos, args.seed, graph_json, recipe=recipe
            )
        tensor_infos = resolve_varlen_layouts(tensor_infos, graph_json, input_data)
        with ExitStack() as stack:
            bms = [
                stack.enter_context(BufferManager(tensor_infos)) for _ in range(n_rot)
            ]
            for bm in bms:
                bm.allocate_all()
                bm.load_input_data(input_data)
                bm.zero_outputs()
            vps = [bm.create_variant_pack() for bm in bms]
            # No Executor.benchmark(): rocprofv3 --pmc serializes dispatches
            # and deadlocks on the staged timer's stall gate. No execute_once()
            # either: its workspace reset would add a memset dispatch to every
            # profiled iteration. Same submissions as the timed loop, one drain.
            for idx in range(policy.warmup_iters + policy.iters):
                executor.enqueue(handle, vps[idx % n_rot])
            device_sync("hip")
    except Exception as e:
        return _fail(f"engine {engine_id}: {type(e).__name__}: {e}")
    return 0

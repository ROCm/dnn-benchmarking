# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Oracle pass: autotune one engine and compare it with its heuristic plan."""

import os
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional

from ..common.exceptions import ExecutionError
from ..config.benchmark_config import SuiteConfig
from ..graph.tensor_info import TensorInfo
from ..metrics._diagnostic import warn_once
from ..metrics.analytical import derive_throughputs
from ..reporting.statistics import BenchmarkStats
from ..reporting.suite_results import (
    OracleResult,
    ProviderEngineResult,
    build_oracle_delta,
)
from ..validation import ReferenceOutput
from .correctness import check_correctness
from .executor import Executor

# Benchmarking is latched when oracle plans are built. Disable hipDNN disk
# caches so the selected provider variant cannot affect later runs.
# ponytail: process-global guard; concurrent execution needs process isolation.
_EXHAUSTIVE_ENV = {
    "HIPDNN_FORCE_BENCHMARKING": "1",
    "HIPDNN_DISABLE_CACHE": "1",
}


@contextmanager
def _exhaustive_env(enabled: bool) -> Iterator[None]:
    """Set provider benchmarking controls and restore the prior environment."""
    if not enabled:
        yield
        return
    previous = {name: os.environ.get(name) for name in _EXHAUSTIVE_ENV}
    os.environ.update(_EXHAUSTIVE_ENV)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def run_oracle_pass(
    *,
    row: ProviderEngineResult,
    handle: Any,
    engine_id: int,
    graph_json_str: str,
    graph_name: str,
    config: SuiteConfig,
    bm: Any,
    variant_pack: Dict[int, int],
    ootb_executor: Executor,
    tensor_infos: List[TensorInfo],
    reference_outputs: Optional[Dict[int, ReferenceOutput]],
) -> None:
    """Tune one engine and attach the result to ``row``; never fails the row.

    Both the heuristic plan and the tuned plan are re-timed after the sweep,
    back to back on the same buffers, so the median delta isolates the plan
    change. The tuned plan is validated after its timed loop; the row keeps
    its own (heuristic plan) verdict.
    """
    executor: Optional[Executor] = None
    oracle_handle: Any = None
    try:
        with _exhaustive_env(config.oracle_exhaustive):
            # Isolate MIOpen's mutable per-handle solver map from the baseline.
            oracle_handle = type(handle)()
            oracle_handle.set_stream(handle.get_stream())
            executor = Executor(graph_json_str, config.timing_policy)
            executor.prepare(oracle_handle, engine_id=engine_id, for_autotune=True)

            bm.zero_outputs()
            candidates = executor.autotune(oracle_handle, variant_pack, engine_id)
            eligible = [c for c in candidates if not c.excluded_by_caller]
            successful = [c for c in eligible if c.succeeded]
            if not successful:
                raise ExecutionError("autotune produced no successful candidate")
            winner = successful[0]  # hipDNN returns successes in rank order.

            bm.zero_outputs()
            baseline = ootb_executor.benchmark(handle, variant_pack)
            bm.zero_outputs()
            tuned = executor.benchmark(oracle_handle, variant_pack)

            oracle = OracleResult(
                plan_name=executor.plan_name(oracle_handle) or "",
                compiled_plan_index=int(winner.compiled_plan_index),
                rank=int(winner.rank),
                sweep_min_time_ms=float(winner.min_time_ms),
                compiled_plans_benchmarked=len(successful),
                compiled_plans_total=len(eligible),
                compiled_plans_failed=len(eligible) - len(successful),
                knob_settings=[
                    {"knob_id": str(k.knob_id), "value": k.value}
                    for k in winner.knob_settings
                ],
                cpu_build_time_ms=executor.init_time_ms,
                host_stats=BenchmarkStats.from_timings(tuned.host_ms),
                gpu_kernel_stats=BenchmarkStats.from_timings(tuned.kernel_ms),
                warm_baseline_host_stats=BenchmarkStats.from_timings(baseline.host_ms),
                warm_baseline_gpu_kernel_stats=BenchmarkStats.from_timings(
                    baseline.kernel_ms
                ),
                exhaustive_requested=config.oracle_exhaustive,
                exhaustive_supported=bool(winner.supports_exhaustive),
            )
            # Same FLOPs and median denominator as the row's own TFLOP/s.
            oracle.derived_tflops_per_s, _ = derive_throughputs(
                row.analytical_flops, None, oracle.gpu_kernel_stats.median_ms
            )
            oracle.warm_baseline_derived_tflops_per_s, _ = derive_throughputs(
                row.analytical_flops,
                None,
                oracle.warm_baseline_gpu_kernel_stats.median_ms,
            )

            if reference_outputs is not None:
                bm.zero_outputs()
                executor.execute_once(oracle_handle, variant_pack)
                oracle.correctness = check_correctness(
                    bm,
                    tensor_infos,
                    reference_outputs,
                    config.validation.provider.value,
                    config,
                )

        row.oracle = oracle
        # A speedup requires two valid operands.
        invalid = [
            side
            for side, verdict in (
                ("baseline", row.correctness),
                ("tuned plan", oracle.correctness),
            )
            if verdict is not None and verdict.explicitly_failed
        ]
        if invalid:
            warn_once(
                "oracle_correctness",
                f"oracle comparison for {graph_name} engine {engine_id} "
                f"suppressed: {' and '.join(invalid)} failed validation",
            )
        else:
            row.oracle_delta = build_oracle_delta(oracle)
    except Exception as e:
        row.oracle_error = f"{type(e).__name__}: {e}"
        warn_once(
            "oracle",
            f"oracle tuning failed for {graph_name} engine {engine_id}: "
            f"{row.oracle_error}",
        )
    finally:
        # Free the tuned plan's workspace before the next engine or profiling.
        executor = oracle_handle = None

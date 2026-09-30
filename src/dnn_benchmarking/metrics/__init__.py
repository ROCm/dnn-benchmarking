# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Metric probes and derivations for dnn-benchmarking.

Always-on probes (no GPU work):
    * Analytical FLOPs / IO bytes from graph JSON (:mod:`analytical`).
    * Host RAM snapshot (:mod:`host`).
    * GPU clocks / power / throttle and VRAM via amdsmi (:mod:`gpu_smi`).
    * One-shot environment metadata (:mod:`machine_info`).

Opt-in profiling sources (separate workload re-run, orchestrated via
:mod:`profiling_orchestrator`):
    * rocprofv3 PMC counters (:mod:`rocprof_pmc`).
    * rocprofv3 kernel/memory trace (:mod:`rocprof_trace`).
    * Linux ``perf stat`` CPU counters, process-total (:mod:`perf`).
    * ``rocprof-compute --roof-only`` roofline data (:mod:`roofline`).

Each opt-in source runs after the timed pass so PMC sampling and roof
replay can't pollute the headline timing. Results land in
``ProviderEngineResult.extra_metrics`` under per-source keys.
"""

from .analytical import compute_flops, compute_io_bytes, derive_throughputs
from .gpu_smi import GpuSmiProbe
from .host import host_memory_snapshot, is_psutil_available
from .machine_info import collect_environment_info

__all__ = [
    "compute_flops",
    "compute_io_bytes",
    "derive_throughputs",
    "GpuSmiProbe",
    "host_memory_snapshot",
    "is_psutil_available",
    "collect_environment_info",
]

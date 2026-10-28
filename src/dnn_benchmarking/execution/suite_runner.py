# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Per-graph engine iteration: one timed row per engine (plus PyTorch rows).

Uses explicit ``--engine`` IDs in caller order when provided; otherwise
discovers ranked engine IDs for the graph via ``Graph.get_ranked_engine_ids``.
Every row is timed by ``timing.measure`` (through the executors) and, with
``--validate``, compared against reference outputs computed once per graph.
Each hipDNN engine row runs its plan in a child process with a new, empty
MIOpen user database (``execution.child_runner``); this process only
discovers engines, computes references and assembles the rows.
"""

import copy
import gc
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

from ..common.exceptions import UnsupportedGraphError
from ..config.benchmark_config import (
    PyTorchSdpaBackendName,
    SuiteConfig,
    TimingPolicy,
)
from ..execution.buffer_manager import generate_input_data
from ..execution.executor import Executor, set_plugin_path
from ..execution.timing import (
    Measurement,
    StallFallbackError,
    Timer,
    retry_unstalled,
)
from ..graph.tensor_info import TensorInfo
from ..metrics import GpuSmiProbe, compute_flops, compute_io_bytes, derive_throughputs
from ..metrics._diagnostic import warn_once
from ..reporting.reporter import Reporter
from ..reporting.statistics import BenchmarkStats, TimingInfo, noise_warnings
from ..reporting.suite_results import (
    CorrectnessResult,
    GraphResult,
    ProviderEngineResult,
    engine_id_hex,
    graph_id_for,
)
from ..validation import ReferenceOutput, ReferenceProvider, ReferenceProviderRegistry
from .child_protocol import HIPDNN_OOTB_ENV, ChildResult, failure_text
from .child_runner import SavedOutputs, child_run, hipdnn_child_args
from .correctness import check_correctness, mismatch, unchecked
from .oracle import hipdnn_tuning_available, run_hipdnn_tuned, run_pytorch_tuned
from .timed_run import pytorch_run


@dataclass
class _GraphContext:
    """Per-graph state shared by every row of one graph."""

    graph_path: Path
    graph_json: Dict[str, Any]
    graph_name: str
    tensor_infos: List[TensorInfo]
    config: SuiteConfig
    reporter: Reporter
    input_data: Dict[int, Any]
    flops: Optional[int] = None
    io_bytes: Optional[int] = None
    reference_outputs: Optional[Dict[int, ReferenceOutput]] = None
    reference_error: Optional[str] = None
    # Set when validation was skipped on purpose (rows stay unchecked).
    validation_skip_reason: Optional[str] = None
    graph_json_str: str = ""  # hipDNN runs only


def _engine_identity(handle: Any, engine_id: int) -> Tuple[str, str]:
    """(display name, plugin version) of a loaded engine.

    The name comes from the handle's engine info, else the handle's name
    lookup (it asks the loaded plugins, so it also names plugin-supplied
    ``hipkernel:*`` engines), else the built-in hipDNN registry, else the
    hex ID, so a row always has a printable label.
    """
    name: Optional[str] = None
    version = "unavailable"
    try:
        info = handle.get_engine_info(engine_id)
        name = str(info.engine_name or "") or None
        version = str(info.version or "") or version
    except Exception as e:
        warn_once("suite_runner", f"engine info lookup failed for {engine_id:#x}: {e}")
    if name is None:
        try:
            name = handle.engine_id_to_name(engine_id) or None
        except IndexError:
            pass  # the handle does not carry this ID
        except Exception as e:
            warn_once(
                "suite_runner", f"handle name lookup failed for {engine_id:#x}: {e}"
            )
    if name is None:
        try:
            import hipdnn_frontend as hipdnn

            name = hipdnn.engine_id_to_name(engine_id) or None
        except Exception:
            pass
    return name or engine_id_hex(engine_id) or "", version


def _report_row(
    reporter: Reporter,
    label: str,
    run: Callable[[], ProviderEngineResult],
) -> ProviderEngineResult:
    """Run one row between progress events, stamping its start and wall time."""
    reporter.engine_start(label)
    started_at = datetime.now(timezone.utc).isoformat()
    with Timer() as t:
        row = run()
    row.started_at = started_at
    row.elapsed_time_ms = t.elapsed_ms
    reporter.engine_done(row)
    return row


def _throttled(after: Optional[Dict[str, Any]]) -> bool:
    # Only the SMU's throttle status counts: sclk alone moves with DPM demand
    # (1700 -> 1430 MHz over a 3-iteration loop on an idle MI210), so a lower
    # after-sample is not evidence of throttling.
    return bool(after and after.get("throttle_status"))


def _measure_row(
    row: ProviderEngineResult,
    ctx: _GraphContext,
    benchmark: Callable[[], Measurement],
) -> None:
    """Run an in-process timed loop and record it (PyTorch rows)."""
    probe = GpuSmiProbe() if ctx.config.metrics.basic else None
    m = benchmark()
    clocks_after = probe.clocks() if probe is not None else None
    _record_timing(
        row,
        ctx,
        TimingInfo.from_measurement(m),
        BenchmarkStats.from_timings(m.kernel_ms),
        BenchmarkStats.from_timings(m.host_ms),
        clocks_after,
    )


def _record_timing(
    row: ProviderEngineResult,
    ctx: _GraphContext,
    timing: TimingInfo,
    kernel: BenchmarkStats,
    host: BenchmarkStats,
    clocks_after: Optional[Dict[str, Any]],
) -> None:
    """Record a timed loop's stats, provenance, clocks and warnings."""
    row.clocks_after = clocks_after
    row.ootb.gpu_kernel_stats = kernel
    row.ootb.host_stats = host
    row.ootb.timing = timing
    warnings = list(row.warnings or []) + noise_warnings(kernel)
    if timing.capped:
        warnings.append("capped at max_iters")
    if timing.fallback_reason:
        # events mode: why staged timing was not used; block mode: a host
        # sync inside the timed span.
        text = f"{timing.mode} timing: {timing.fallback_reason}"
        warn_once("timing", text)
        warnings.append(text)
    if _throttled(row.clocks_after):
        warnings.append("throttled")
    row.warnings = warnings

    if ctx.config.metrics.basic:
        row.analytical_flops = ctx.flops
        row.analytical_io_bytes = ctx.io_bytes
        row.ootb.derived_tflops_per_s, row.ootb.derived_gbytes_per_s = (
            derive_throughputs(ctx.flops, ctx.io_bytes, kernel.median_ms)
        )


def _reference_provider(
    config: SuiteConfig, graph_json: Dict[str, Any]
) -> Tuple[Optional[ReferenceProvider], Optional[str]]:
    """The requested reference provider for this graph, or why there is none."""
    name = config.validation.provider.value
    try:
        provider = ReferenceProviderRegistry.get_provider(name)
    except ValueError:
        return None, f"Reference provider '{name}' not registered"
    if not provider.is_available():
        return None, f"Reference provider '{name}' not available"
    if not provider.supports_graph(graph_json):
        return None, f"Reference provider '{name}' does not support this graph"
    return provider, None


def _compute_reference_outputs_once(
    ref_provider: ReferenceProvider,
    graph_json: Dict[str, Any],
    input_data: Dict[int, Any],
    config: SuiteConfig,
) -> Tuple[Optional[Dict[int, ReferenceOutput]], Optional[str]]:
    try:
        from .pytorch_ops import PyTorchSdpaBackendState, use_pytorch_sdpa_backend

        state = PyTorchSdpaBackendState(
            config.pytorch_sdpa_backend, config.pytorch_rocm_fa_library
        )
        with use_pytorch_sdpa_backend(state):
            return ref_provider.compute_reference(graph_json, input_data), None
    except Exception as e:
        return None, str(e)


def _pytorch_reference_outputs_from_buffer(
    buffer_manager: Any,
    keep_device: bool = True,
) -> Dict[int, ReferenceOutput]:
    """Collect reference outputs from a PyTorch buffer manager.

    With ``keep_device``, a GPU output stays on the device only: each engine
    compares on the GPU, and the rare host fallback copies it to the host for
    one comparison (``ReferenceOutput.host_data``). Otherwise ``data`` is a
    host copy.
    """
    outputs: Dict[int, ReferenceOutput] = {}
    tensors = buffer_manager.get_tensors()
    for tensor_info in buffer_manager.get_output_tensors():
        uid = tensor_info.uid
        tensor = tensors.get(uid)
        if tensor is None:
            continue
        if keep_device and tensor.is_cuda:
            outputs[uid] = ReferenceOutput(
                data=None, tensor_uid=uid, device_data=_owned(tensor.detach())
            )
        else:
            outputs[uid] = ReferenceOutput(
                data=buffer_manager.get_output_data(uid), tensor_uid=uid
            )
    return outputs


def _owned(tensor: Any) -> Any:
    """``tensor`` itself, or a compact copy when it is a view of a larger storage.

    The buffer manager only drops its references on exit, so a kept tensor
    stays valid without a copy. A view would keep its whole base allocation
    alive, which can be larger than a copy of the view.
    """
    if tensor.untyped_storage().nbytes() > tensor.numel() * tensor.element_size():
        return tensor.clone()
    return tensor


def _graph_context(
    graph_path: Path,
    graph_json: Dict[str, Any],
    tensor_infos: List[TensorInfo],
    config: SuiteConfig,
    reporter: Reporter,
) -> _GraphContext:
    """Generate inputs and per-graph analytical FLOPs/IO (shape-only, once)."""
    graph_name = graph_json.get("name", graph_path.stem)
    ctx = _GraphContext(
        graph_path=graph_path,
        graph_json=graph_json,
        graph_name=graph_name,
        tensor_infos=tensor_infos,
        config=config,
        reporter=reporter,
        input_data=generate_input_data(tensor_infos, config.seed, graph_json),
    )
    if config.metrics.basic:
        try:
            ctx.flops = compute_flops(graph_json)
        except Exception as e:
            warn_once("analytical", f"compute_flops failed for {graph_name}: {e}")
        try:
            ctx.io_bytes = compute_io_bytes(tensor_infos)
        except Exception as e:
            warn_once("analytical", f"compute_io_bytes failed for {graph_name}: {e}")
    return ctx


def _torch_version() -> str:
    try:
        from importlib.metadata import version

        return version("torch")
    except Exception:
        return "unavailable"


def _run_pytorch_row(
    ctx: _GraphContext, role: Literal["engine", "reference"]
) -> Tuple[ProviderEngineResult, Optional[Dict[int, ReferenceOutput]]]:
    """Time the graph through PyTorch as one row (``_time_pytorch_row``).

    Then return PyTorch's cached VRAM to the device. The buffer manager
    empties the cache on exit, but the row's locals (and, after a failure,
    the exception's frames) still hold the executor and its intermediates at
    that point. They reach the cache only once the row returns. Without this
    release, hipDNN's own allocations (``hipMalloc``) for the same graph can
    fail although the device has room.
    """
    try:
        return _time_pytorch_row(ctx, role)
    finally:
        _release_torch_cache()


def _release_torch_cache() -> None:
    """Free the tensors no longer referenced, then return PyTorch's cached VRAM.

    ``gc.collect()`` frees tensors held only by reference cycles, which
    ``empty_cache()`` alone cannot release.
    """
    torch = sys.modules.get("torch")
    try:
        if torch is None or not torch.cuda.is_initialized():
            return
        gc.collect()
        torch.cuda.empty_cache()
    except Exception:
        pass


_OUT_OF_MEMORY = re.compile(r"out ?of ?memory", re.IGNORECASE)


def _gpu_memory_at_row_start() -> Optional[str]:
    """Device memory as a row starts, reported if the row runs out of memory.

    Read only when torch has already initialized the GPU: initializing it
    here would itself take VRAM in a run that has no PyTorch rows. The
    device-free figure counts every process on the GPU; the PyTorch cache
    figure counts only this process.
    """
    torch = sys.modules.get("torch")
    try:
        if torch is None or not torch.cuda.is_initialized():
            return None
        free, total = torch.cuda.mem_get_info()
        cached = torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
    except Exception:
        return None
    gib = float(1 << 30)
    return (
        f"at row start the GPU had {free / gib:.1f} of {total / gib:.1f} GiB free "
        f"(all processes) and this process's PyTorch cache held "
        f"{cached / gib:.1f} GiB unused"
    )


def _failure_message(error: Exception, memory_at_start: Optional[str]) -> str:
    """``Type: message``, plus the row-start memory for out-of-memory errors."""
    message = failure_text(error)
    if memory_at_start is not None and _OUT_OF_MEMORY.search(message):
        message = f"{message} [{memory_at_start}]"
    return message


def _time_pytorch_row(
    ctx: _GraphContext, role: Literal["engine", "reference"]
) -> Tuple[ProviderEngineResult, Optional[Dict[int, ReferenceOutput]]]:
    """Time the graph through PyTorch as one row.

    ``role="reference"`` additionally extracts reference outputs for the
    engine comparison. Unsupported graphs are skipped unless the SDPA
    selection is strict. Any other failure of a default-dispatch reference
    is skipped (the CPU reference may still serve); any other failure of a
    strict SDPA selection or of the ``--runtime pytorch`` engine row is an
    error.
    """
    config = ctx.config
    strict = config.pytorch_sdpa_backend is not PyTorchSdpaBackendName.DEFAULT
    row = ProviderEngineResult(
        runtime="pytorch",
        engine_id=None,
        engine_name="pytorch",
        status="success",
        role=role,
        engine_version=_torch_version(),
    )
    outputs: Optional[Dict[int, ReferenceOutput]] = None
    memory_at_start = _gpu_memory_at_row_start()
    try:
        from . import pytorch_ops

        with pytorch_run(
            ctx.graph_json,
            ctx.tensor_infos,
            ctx.input_data,
            config.timing_policy,
            config.pytorch_sdpa_backend,
            config.pytorch_rocm_fa_library,
        ) as (executor, bm):
            tensors = bm.get_tensors()
            _measure_row(row, ctx, lambda: executor.benchmark(tensors))
            row.warnings = (row.warnings or []) + executor.notes
            if role == "reference":
                bm.zero_outputs()
                executor.execute_once(tensors)
                # A profiling child allocates its own VRAM, so profiled runs
                # keep host-only references (and DeviceBuffer I/O).
                outputs = _pytorch_reference_outputs_from_buffer(
                    bm, keep_device=not config.metrics.opt_in_pass_requested
                )
        if role == "reference":
            row.warnings = (row.warnings or []) + pytorch_ops.get_reference_warnings(
                ctx.graph_json
            )
        # After the OOTB buffers are released, so the child's allocations do
        # not stack on top of them.
        if config.oracle:
            run_pytorch_tuned(
                row=row,
                graph_path=ctx.graph_path,
                graph_json=ctx.graph_json,
                graph_name=ctx.graph_name,
                config=config,
            )
    except StallFallbackError:
        raise  # run_graph_* remeasures the whole graph unstalled.
    except UnsupportedGraphError as e:
        return _failed_row(_error_or_skip(strict), row, str(e)), None
    except Exception as e:
        fatal = role == "engine" or strict
        message = _failure_message(e, memory_at_start)
        return _failed_row(_error_or_skip(fatal), row, message), None
    return row, outputs


def run_single_provider_engine(
    ctx: _GraphContext,
    engine_id: int,
    engine_name: str,
    engine_version: str,
    plugin_path: Optional[Path],
) -> ProviderEngineResult:
    """Time, validate (and optionally tune/profile) one hipDNN engine.

    The OOTB plan, the tuned plan and each profiling pass run in separate
    child processes, each with a new, empty MIOpen user database.
    """
    config = ctx.config
    plugin = str(plugin_path) if plugin_path is not None else None
    row = ProviderEngineResult(
        runtime="hipdnn",
        engine_id=engine_id,
        engine_name=engine_name,
        status="success",
        engine_version=engine_version,
        plugin_path=plugin,
    )
    memory_at_start = _gpu_memory_at_row_start()
    tuning_available: Optional[bool] = None
    args = hipdnn_child_args(
        ctx.graph_path,
        config,
        engine_id,
        plugin_path,
        save_outputs=ctx.reference_outputs is not None,
    )
    try:
        # No time limit, as for every timed OOTB row.
        with child_run("hipDNN", "hipdnn_child", args, HIPDNN_OOTB_ENV, None) as (
            ootb,
            outputs,
        ):
            _record_ootb(row, ctx, ootb)
            row.ootb.correctness = _ootb_correctness(ctx, outputs)
            if config.oracle:
                tuning_available = hipdnn_tuning_available(
                    row, ctx.graph_name, engine_id, ootb
                )
    except StallFallbackError:
        raise  # run_graph_* remeasures the whole graph unstalled.
    except UnsupportedGraphError as e:
        return _failed_row(ProviderEngineResult.skipped_row, row, str(e))
    except Exception as e:
        return _failed_row(
            ProviderEngineResult.error_row,
            row,
            _failure_message(e, memory_at_start),
        )

    # The tuned run and profiling run in their own child processes after the
    # OOTB run, so tuning state and profiler overhead cannot reach the
    # headline numbers.
    if tuning_available is not None:
        run_hipdnn_tuned(
            row=row,
            graph_path=ctx.graph_path,
            graph_name=ctx.graph_name,
            tensor_infos=ctx.tensor_infos,
            reference_outputs=ctx.reference_outputs,
            engine_id=engine_id,
            plugin_path=plugin_path,
            tuning_available=tuning_available,
            config=config,
        )
    if config.metrics.opt_in_pass_requested:
        _run_profiling(row, ctx, engine_id, engine_name, plugin_path)
    return row


def _record_ootb(
    row: ProviderEngineResult, ctx: _GraphContext, ootb: ChildResult
) -> None:
    """Record the OOTB child's build, workspace and timed loop."""
    row.ootb.cpu_build_time_ms = ootb.build_ms
    if ctx.config.metrics.basic:
        row.ootb.workspace_bytes = ootb.workspace_bytes
    _record_timing(row, ctx, ootb.timing, ootb.kernel, ootb.host, ootb.clocks_after)


def _ootb_correctness(
    ctx: _GraphContext, outputs: SavedOutputs
) -> Optional[CorrectnessResult]:
    """Compare the OOTB child's saved outputs with the references."""
    config = ctx.config
    if ctx.reference_outputs is not None:
        return check_correctness(
            outputs,
            ctx.tensor_infos,
            ctx.reference_outputs,
            config.validation.provider.value,
            config,
        )
    if ctx.validation_skip_reason is not None:
        return unchecked(config, ctx.validation_skip_reason)
    if config.validation.enabled:
        # Validation was requested but no reference is usable: keep
        # --validate a hard gate.
        return mismatch(config, ctx.reference_error or "Reference outputs unavailable")
    return None


def _error_or_skip(fatal: bool) -> Callable[..., ProviderEngineResult]:
    return ProviderEngineResult.error_row if fatal else ProviderEngineResult.skipped_row


def _failed_row(
    make: Callable[..., ProviderEngineResult],
    row: ProviderEngineResult,
    message: str,
) -> ProviderEngineResult:
    """Replace a partially filled row with a reason-only error/skipped row."""
    failed = make(
        row.runtime,
        row.engine_id,
        message,
        engine_name=row.engine_name,
        role=row.role,
        plugin_path=row.plugin_path,
    )
    failed.engine_version = row.engine_version
    return failed


def _run_profiling(
    row: ProviderEngineResult,
    ctx: _GraphContext,
    engine_id: int,
    engine_name: str,
    plugin_path: Optional[Path],
) -> None:
    """Run the opt-in profiler passes; a failure annotates the timed row."""
    from ..metrics.profiling_orchestrator import run_profiling_passes

    extra = None
    ctx.reporter.profiling_start(engine_name)
    with Timer() as t:
        try:
            extra = run_profiling_passes(
                graph_path=ctx.graph_path,
                engine_id=engine_id,
                engine_name=engine_name,
                seed=ctx.config.seed,
                warmup_iters=ctx.config.warmup_iters,
                metrics_config=ctx.config.metrics,
                plugin_path=plugin_path,
            )
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            warn_once("profiling", f"profiling pass failed: {msg}")
            row.warnings = (row.warnings or []) + [f"profiling failed: {msg}"]
    ctx.reporter.profiling_done(engine_name, t.elapsed_ms / 1000.0)
    row.extra_metrics = extra or None


def _prepare_references(
    ctx: _GraphContext,
) -> Optional[ProviderEngineResult]:
    """Compute reference outputs once per graph; return the timed PyTorch row.

    Sets ``ctx.reference_outputs`` or ``ctx.reference_error``. A strict
    PyTorch SDPA selection never falls back to the CPU reference after the
    timed native path fails. The CPU reference is skipped when its estimated
    host memory does not fit (``ctx.validation_skip_reason``): the engine rows
    are then unchecked, not failed.
    """
    config = ctx.config
    provider, ctx.reference_error = _reference_provider(config, ctx.graph_json)
    if provider is None:
        warn_once("validation", f"{ctx.graph_name}: {ctx.reference_error}")
        return None

    outputs: Optional[Dict[int, ReferenceOutput]] = None

    def run() -> ProviderEngineResult:
        nonlocal outputs
        row, outputs = _run_pytorch_row(ctx, "reference")
        return row

    row = _report_row(ctx.reporter, "pytorch reference", run)
    if outputs is not None:
        ctx.reference_outputs = outputs
    elif config.pytorch_sdpa_backend is not PyTorchSdpaBackendName.DEFAULT:
        ctx.reference_error = (
            row.error_message
            or row.skip_reason
            or (
                f"Requested PyTorch SDPA backend '{config.pytorch_sdpa_backend.value}' "
                "is unavailable; no fallback is used."
            )
        )
    else:
        ctx.validation_skip_reason = _host_memory_problem(provider, ctx.graph_json)
        if ctx.validation_skip_reason is not None:
            ctx.reference_error = ctx.validation_skip_reason
            warn_once("validation", f"{ctx.graph_name}: {ctx.validation_skip_reason}")
        else:
            ctx.reference_outputs, ctx.reference_error = (
                _compute_reference_outputs_once(
                    provider, ctx.graph_json, ctx.input_data, config
                )
            )
    return row


#: Share of the available host memory the CPU reference may use; the rest is
#: left for the engine rows of the same graph and for the estimate's error.
_REFERENCE_HOST_FRACTION = 0.8


def _host_memory_problem(
    provider: ReferenceProvider, graph_json: Dict[str, Any]
) -> Optional[str]:
    """Why the CPU reference would not fit in host memory, or None.

    Without an estimate or a reading of the available memory, the reference
    runs as before.
    """
    needed = provider.estimate_host_bytes(graph_json)
    if needed is None:
        return None
    try:
        import psutil

        available = psutil.virtual_memory().available
    except Exception as e:
        warn_once("validation", f"host memory reading failed: {e}")
        return None
    if needed <= available * _REFERENCE_HOST_FRACTION:
        return None
    gib = float(1 << 30)
    return (
        f"validation skipped: the CPU reference needs about {needed / gib:.1f} GiB "
        f"of host memory and {available / gib:.1f} GiB is available"
    )


def _with_stall_fallback(
    run: Callable[[SuiteConfig], GraphResult],
    config: SuiteConfig,
    reporter: Reporter,
) -> GraphResult:
    """Run one graph; if stall-gated timing fails, rerun every row unstalled.

    Rows of one graph are compared with each other, so they must share a
    timing mode: a declined arm or a watchdog release discards the partial
    graph instead of mixing staged and events rows. Each rerun row records
    the reason in ``timing.fallback_reason`` and ``warnings``.
    """

    def attempt(policy: TimingPolicy) -> GraphResult:
        if policy is config.timing_policy:
            return run(config)
        unstalled = copy.copy(config)
        unstalled.timing_policy = policy
        return run(unstalled)

    return retry_unstalled(
        attempt,
        config.timing_policy,
        lambda reason: reporter.warning(
            f"{reason}; remeasuring every row of this graph without stalling"
        ),
    )


def run_graph_all_providers(
    graph_path: Path,
    graph_json: Dict[str, Any],
    tensor_infos: list,
    config: SuiteConfig,
    handle: Any,
    reporter: Reporter,
) -> GraphResult:
    """Run a single graph against every selected or discovered hipDNN engine.

    An unsupported graph (no applicable engine) yields ``engine_ids=[]`` and
    no engine rows; with ``--validate pytorch`` the timed PyTorch reference
    row is still emitted when PyTorch can run the graph. Discovery and input
    generation failures set ``GraphResult.error``. A stall-gate failure
    reruns the whole graph without stalling (``_with_stall_fallback``).

    Args:
        graph_path: Path to the graph JSON file.
        graph_json: Parsed graph JSON dictionary.
        tensor_infos: TensorInfo objects for the graph.
        config: Suite configuration.
        handle: hipdnn.Handle, or None to create one per engine (per-engine
            plugin paths).
        reporter: Progress sink.
    """
    return _with_stall_fallback(
        lambda cfg: _run_graph_all_providers(
            graph_path, graph_json, tensor_infos, cfg, handle, reporter
        ),
        config,
        reporter,
    )


def _run_graph_all_providers(
    graph_path: Path,
    graph_json: Dict[str, Any],
    tensor_infos: list,
    config: SuiteConfig,
    handle: Any,
    reporter: Reporter,
) -> GraphResult:
    """One attempt of :func:`run_graph_all_providers` with ``config``'s policy."""
    graph = GraphResult(
        graph_name=graph_json.get("name", graph_path.stem),
        graph_path=str(graph_path),
        results=[],
        graph_id=graph_id_for(graph_json),
    )

    graph_json_str = json.dumps(graph_json)
    if config.engine_filter is not None:
        # Explicit --engine is a selection, not a post-discovery filter. Keep
        # the caller's order so per-engine plugin paths are deterministic.
        engine_ids = list(config.engine_filter)
    else:
        try:
            # SuiteConfig requires --engine with several plugin paths, so the
            # shared handle exists whenever discovery runs. The discovery
            # graph is a temporary: it is not kept alive for the engine rows.
            engine_ids = Executor(
                graph_json_str, config.timing_policy
            ).discover_engines(handle)
        except UnsupportedGraphError as e:
            engine_ids = []
            graph.message = str(e)
        except Exception as e:
            graph.error = f"Engine discovery failed: {type(e).__name__}: {e}"
            return graph
    graph.engine_ids = engine_ids

    try:
        ctx = _graph_context(graph_path, graph_json, tensor_infos, config, reporter)
        ctx.graph_json_str = graph_json_str
    except Exception as e:
        graph.error = f"Input data generation failed: {type(e).__name__}: {e}"
        return graph

    if config.validation.enabled:
        reference_row = _prepare_references(ctx)
        if reference_row is not None:
            graph.results.append(reference_row)

    for selection in config.engine_selections_for(engine_ids):
        graph.results.append(
            _run_engine_selection(
                ctx, handle, selection.engine_id, selection.plugin_path
            )
        )
    return graph


def _run_engine_selection(
    ctx: _GraphContext,
    handle: Any,
    engine_id: int,
    plugin_path: Optional[Path],
) -> ProviderEngineResult:
    """One engine row, creating a plugin-scoped handle when none is shared."""
    if handle is None:
        try:
            import hipdnn_frontend as hipdnn

            set_plugin_path(hipdnn, plugin_path)
            handle = hipdnn.Handle()
        except Exception as e:
            label = engine_id_hex(engine_id) or ""
            return _report_row(
                ctx.reporter,
                label,
                lambda: ProviderEngineResult.error_row(
                    "hipdnn",
                    engine_id,
                    f"{type(e).__name__}: {e}",
                    engine_name=label,
                    plugin_path=str(plugin_path) if plugin_path else None,
                ),
            )
    name, version = _engine_identity(handle, engine_id)
    return _report_row(
        ctx.reporter,
        name,
        lambda: run_single_provider_engine(ctx, engine_id, name, version, plugin_path),
    )


def run_graph_pytorch(
    graph_path: Path,
    graph_json: Dict[str, Any],
    tensor_infos: list,
    config: SuiteConfig,
    reporter: Reporter,
) -> GraphResult:
    """Run a single graph through the PyTorch executor as the sole engine row.

    The ``--runtime pytorch`` counterpart of :func:`run_graph_all_providers`:
    no hipDNN engine discovery, plugins, or reference validation.
    """
    return _with_stall_fallback(
        lambda cfg: _run_graph_pytorch(
            graph_path, graph_json, tensor_infos, cfg, reporter
        ),
        config,
        reporter,
    )


def _run_graph_pytorch(
    graph_path: Path,
    graph_json: Dict[str, Any],
    tensor_infos: list,
    config: SuiteConfig,
    reporter: Reporter,
) -> GraphResult:
    graph = GraphResult(
        graph_name=graph_json.get("name", graph_path.stem),
        graph_path=str(graph_path),
        results=[],
        engine_ids=[0],
        graph_id=graph_id_for(graph_json),
    )

    # Unsupported operations are an unsupported-graph signal, checked before
    # input generation (a static inspection).
    from . import pytorch_ops

    unsupported = sorted(pytorch_ops.get_unsupported_operations(graph_json))
    if unsupported:
        msg = f"Graph contains unsupported operations: {unsupported}"
        make = _error_or_skip(
            config.pytorch_sdpa_backend is not PyTorchSdpaBackendName.DEFAULT
        )
        graph.results.append(
            _report_row(
                reporter,
                "pytorch",
                lambda: make("pytorch", None, msg, engine_name="pytorch"),
            )
        )
        return graph

    try:
        ctx = _graph_context(graph_path, graph_json, tensor_infos, config, reporter)
    except Exception as e:
        graph.error = f"Input data generation failed: {type(e).__name__}: {e}"
        return graph

    graph.results.append(
        _report_row(reporter, "pytorch", lambda: _run_pytorch_row(ctx, "engine")[0])
    )
    return graph

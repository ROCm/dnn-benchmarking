# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Per-graph row iteration: every selected runtime adds its rows to the graph.

Each runtime turns a graph into a :class:`RuntimePlan` (its engines, or why
none applies) before inputs exist. :func:`run_graph` then generates inputs and
the ``--validate`` reference once, and appends each runtime's rows in run
order. Every row is timed by ``timing.measure`` (through the executors).
"""

import copy
import gc
import json
import re
import sys
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
)

from ..common.exceptions import UnsupportedGraphError
from ..config.benchmark_config import (
    PyTorchSdpaBackendName,
    ReferencePrecision,
    SuiteConfig,
)
from ..execution.buffer_manager import BufferManager, generate_input_data
from ..execution.executor import Executor
from ..execution.timing import Measurement, StallFallbackError, Timer
from ..graph.tensor_info import TensorInfo
from ..metrics import GpuSmiProbe, compute_flops, compute_io_bytes, derive_throughputs
from ..metrics._diagnostic import warn_once
from ..reporting.reporter import Reporter
from ..reporting.statistics import BenchmarkStats, TimingInfo, noise_warnings
from ..reporting.suite_results import (
    GraphResult,
    ProviderEngineResult,
    engine_id_hex,
    graph_id_for,
)
from ..validation import ReferenceOutput, ReferenceProvider, ReferenceProviderRegistry
from .correctness import check_correctness, mismatch, unchecked
from .oracle import (
    build_tuned_plan,
    run_pytorch_tuned,
    run_tuned_plan,
    tuned_in_process,
)
from .rocke_sdpa import (
    Catalog,
    RockeKernel,
    SdpaProblem,
    auto_kernel,
    catalog_kernels,
    sdpa_problem,
)


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
    graph_id: str = ""  # hipDNN runs only


def set_plugin_path(hipdnn: Any, plugin_path: Optional[Path]) -> None:
    """Set the process-wide hipDNN plugin search path for the next handle."""
    if plugin_path is not None:
        hipdnn.set_engine_plugin_paths(
            [str(plugin_path)], hipdnn.PluginLoadingMode.ABSOLUTE
        )


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


def _hipdnn_buffer_device(
    reference_outputs: Optional[Dict[int, ReferenceOutput]],
) -> Optional[str]:
    """Torch device for hipDNN I/O buffers, or None for hipdnn.DeviceBuffer.

    Torch storage is used only when a device reference exists, so the
    outputs can be compared on the GPU. Timing-only runs, and runs whose
    reference is host-only, keep the DeviceBuffer baseline. ``"cuda"`` is
    torch's current device, which torch reads with ``hipGetDevice``. The
    hipDNN handle uses the same in-process HIP device, so the pointers
    belong to the handle's device.
    """
    if reference_outputs and any(
        ref.device_data is not None for ref in reference_outputs.values()
    ):
        return "cuda"
    return None


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
    """Run the timed loop and record stats, provenance, clocks and warnings."""
    basic = ctx.config.metrics.basic
    probe = GpuSmiProbe() if basic else None
    m = benchmark()
    if probe is not None:
        row.clocks_after = probe.clocks()

    kernel = BenchmarkStats.from_timings(m.kernel_ms)
    row.ootb.gpu_kernel_stats = kernel
    row.ootb.host_stats = BenchmarkStats.from_timings(m.host_ms)
    row.ootb.timing = TimingInfo.from_measurement(m)
    warnings = list(row.warnings or []) + noise_warnings(kernel)
    if m.capped:
        warnings.append("capped at max_iters")
    if m.fallback_reason:
        # events mode: why staged timing was not used; block mode: a host
        # sync inside the timed span.
        text = f"{m.mode} timing: {m.fallback_reason}"
        warn_once("timing", text)
        warnings.append(text)
    if _throttled(row.clocks_after):
        warnings.append("throttled")
    row.warnings = warnings

    if probe is not None:
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
    precision: ReferencePrecision = ReferencePrecision.NATIVE,
    device: Optional[str] = None,
) -> Tuple[Optional[Dict[int, ReferenceOutput]], Optional[str]]:
    try:
        from .pytorch_ops import PyTorchSdpaBackendState, use_pytorch_sdpa_backend

        state = PyTorchSdpaBackendState(
            config.pytorch_sdpa_backend, config.pytorch_rocm_fa_library
        )
        with use_pytorch_sdpa_backend(state):
            outputs = ref_provider.compute_reference(
                graph_json, input_data, precision=precision, device=device
            )
            return outputs, None
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


def _package_version(distribution: str) -> str:
    try:
        from importlib.metadata import version

        return version(distribution)
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
    message = f"{type(error).__name__}: {error}"
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
        engine_version=_package_version("torch"),
    )
    outputs: Optional[Dict[int, ReferenceOutput]] = None
    memory_at_start = _gpu_memory_at_row_start()
    try:
        from . import pytorch_ops
        from .pytorch_buffer_manager import PyTorchCudaBufferManager
        from .pytorch_executor import PyTorchCudaExecutor

        executor = PyTorchCudaExecutor(
            ctx.graph_json,
            config.timing_policy,
            pytorch_sdpa_backend=config.pytorch_sdpa_backend,
            pytorch_rocm_fa_library=config.pytorch_rocm_fa_library,
        )
        executor.prepare()
        # The executor's device, not the buffer manager's cuda:0 default.
        with PyTorchCudaBufferManager(ctx.tensor_infos, device=executor.device) as bm:
            bm.allocate_all()
            bm.load_input_data(ctx.input_data)
            bm.zero_outputs()
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
        raise  # run_graph remeasures the whole graph unstalled.
    except UnsupportedGraphError as e:
        return _failed_row(_error_or_skip(strict), row, str(e)), None
    except Exception as e:
        fatal = role == "engine" or strict
        message = _failure_message(e, memory_at_start)
        return _failed_row(_error_or_skip(fatal), row, message), None
    return row, outputs


def run_single_provider_engine(
    ctx: _GraphContext,
    handle: Any,
    engine_id: int,
    engine_name: str,
    engine_version: str,
    plugin_path: Optional[Path],
) -> ProviderEngineResult:
    """Build, time, validate (and optionally tune/profile) one hipDNN engine."""
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
    try:
        if tuned_in_process(ctx.graph_id, engine_id):
            # Typically the stall-fallback rerun of a graph whose first
            # attempt already ran this engine's search, or a repeated graph.
            raise RuntimeError(
                "OOTB not measurable: this engine already ran a tuned search on "
                "this graph in this process, and hipDNN would serve its winner"
            )
        # Absorb the engine's one-time build costs (provider setup, cold
        # plugin and kernel file reads) before timing, so the OOTB and tuned
        # build times start from the same state.
        Executor(ctx.graph_json_str, config.timing_policy).prime(handle, engine_id)
        executor = Executor(ctx.graph_json_str, config.timing_policy)
        executor.prepare(handle, engine_id=engine_id)
        row.ootb.cpu_build_time_ms = executor.build_time_ms
        if config.metrics.basic:
            row.ootb.workspace_bytes = executor.workspace_size
        tuned = (
            build_tuned_plan(
                row=row,
                handle=handle,
                engine_id=engine_id,
                graph_json_str=ctx.graph_json_str,
                graph_name=ctx.graph_name,
                config=config,
            )
            if config.oracle
            else None
        )

        with BufferManager(
            ctx.tensor_infos, device=_hipdnn_buffer_device(ctx.reference_outputs)
        ) as bm:
            bm.allocate_all()
            bm.load_input_data(ctx.input_data)
            bm.zero_outputs()
            variant_pack = bm.create_variant_pack()
            _measure_row(row, ctx, lambda: executor.benchmark(handle, variant_pack))

            _validate_row(
                row, ctx, bm, lambda: executor.execute_once(handle, variant_pack)
            )

            if tuned is not None:
                run_tuned_plan(
                    tuned=tuned,
                    row=row,
                    graph_id=ctx.graph_id,
                    engine_id=engine_id,
                    graph_name=ctx.graph_name,
                    config=config,
                    bm=bm,
                    variant_pack=variant_pack,
                    tensor_infos=ctx.tensor_infos,
                    reference_outputs=ctx.reference_outputs,
                )
        # Release both workspaces before the profiling child allocates its own
        # VRAM; holding them roughly doubles peak VRAM on large graphs.
        del executor, tuned
    except StallFallbackError:
        raise  # run_graph remeasures the whole graph unstalled.
    except UnsupportedGraphError as e:
        return _failed_row(ProviderEngineResult.skipped_row, row, str(e))
    except Exception as e:
        return _failed_row(
            ProviderEngineResult.error_row,
            row,
            _failure_message(e, memory_at_start),
        )

    # Profiling re-runs the workload after the timed pass, so profiler
    # overhead cannot pollute the headline numbers.
    if config.metrics.opt_in_pass_requested:
        _run_profiling(row, ctx, engine_id, engine_name, plugin_path)
    return row


def _validate_row(
    row: ProviderEngineResult,
    ctx: _GraphContext,
    bm: Any,
    execute_once: Callable[[], None],
) -> None:
    """Set ``row.ootb.correctness`` for one engine row.

    With reference outputs, run the engine once more on zeroed outputs and
    compare. When the reference was skipped on purpose (it does not fit in
    host memory), the row is unchecked. When ``--validate`` has no usable
    reference for another reason, the row fails, so ``--validate`` stays a
    hard gate.
    """
    config = ctx.config
    if ctx.reference_outputs is not None:
        bm.zero_outputs()
        execute_once()
        row.ootb.correctness = check_correctness(
            bm,
            ctx.tensor_infos,
            ctx.reference_outputs,
            config.validation.provider.value,
            config,
        )
    elif ctx.validation_skip_reason is not None:
        row.ootb.correctness = unchecked(config, ctx.validation_skip_reason)
    elif config.validation.enabled:
        row.ootb.correctness = mismatch(
            config, ctx.reference_error or "Reference outputs unavailable"
        )


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

    With ``--reference-precision float64`` (the default), a graph the provider
    can compute in float64 gets a float64 reference: on the GPU after the
    timed row, else on the host. If the float64 reference fails or does not
    fit, the native reference serves and the reference row gets a note.
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
    float64 = (
        config.validation.precision is ReferencePrecision.FLOAT64
        and provider.supports_float64(ctx.graph_json)
    )
    if outputs is not None:
        # Drop the local name: ctx.reference_outputs then holds the only
        # reference to the native outputs, so the float64 pass can free them.
        ctx.reference_outputs, outputs = outputs, None
        if float64:
            # The timed row ran, so PyTorch has a GPU. The native outputs stay
            # the reference if the float64 one fails.
            _use_float64_reference(ctx, provider, row, device="cuda")
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
        if float64:
            shortfall = _host_memory_shortfall(
                provider, ctx.graph_json, ReferencePrecision.FLOAT64
            )
            if shortfall is not None:
                _note_native_reference(ctx, row, shortfall)
            elif _use_float64_reference(ctx, provider, row, device=None):
                return row
        shortfall = _host_memory_shortfall(provider, ctx.graph_json)
        if shortfall is not None:
            ctx.validation_skip_reason = f"validation skipped: {shortfall}"
            ctx.reference_error = ctx.validation_skip_reason
            warn_once("validation", f"{ctx.graph_name}: {ctx.validation_skip_reason}")
        else:
            ctx.reference_outputs, ctx.reference_error = (
                _compute_reference_outputs_once(
                    provider, ctx.graph_json, ctx.input_data, config
                )
            )
    return row


def _use_float64_reference(
    ctx: _GraphContext,
    provider: ReferenceProvider,
    row: ProviderEngineResult,
    device: Optional[str],
) -> bool:
    """Make a float64 reference ``ctx.reference_outputs``; False if it failed.

    A failure (for example a GPU out-of-memory error) keeps the current
    reference and adds a note to ``row``. Profiled runs keep host-only
    references, as the timed row does.
    """
    outputs, error = _compute_reference_outputs_once(
        provider,
        ctx.graph_json,
        ctx.input_data,
        ctx.config,
        ReferencePrecision.FLOAT64,
        device,
    )
    if outputs is None:
        _release_torch_cache()
        _note_native_reference(ctx, row, error or "no outputs")
        return False
    keep_device = not ctx.config.metrics.opt_in_pass_requested
    for ref in outputs.values():
        if ref.device_data is None:
            continue
        if keep_device:
            ref.device_data = _owned(ref.device_data)
        else:
            ref.data, ref.device_data = ref.host_data(), None
    ctx.reference_outputs = outputs
    # Return the float64 temporaries and the replaced native outputs to the
    # device before the engine rows allocate.
    _release_torch_cache()
    return True


def _note_native_reference(
    ctx: _GraphContext, row: ProviderEngineResult, reason: str
) -> None:
    """Record on the reference row why the float64 reference is not used."""
    note = f"float64 reference not used ({reason}); using the graph data types"
    row.warnings = (row.warnings or []) + [note]
    warn_once("validation", f"{ctx.graph_name}: {note}")


#: Share of the available host memory the CPU reference may use; the rest is
#: left for the engine rows of the same graph and for the estimate's error.
_REFERENCE_HOST_FRACTION = 0.8


def _host_memory_shortfall(
    provider: ReferenceProvider,
    graph_json: Dict[str, Any],
    precision: ReferencePrecision = ReferencePrecision.NATIVE,
) -> Optional[str]:
    """Why the CPU reference would not fit in host memory, or None.

    Without an estimate or a reading of the available memory, the reference
    runs as before.
    """
    needed = provider.estimate_host_bytes(graph_json, precision)
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
    kind = "float64 " if precision is ReferencePrecision.FLOAT64 else ""
    return (
        f"the {kind}CPU reference needs about {needed / gib:.1f} GiB "
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
    try:
        return run(config)
    except StallFallbackError as e:
        reason = str(e)
    # Rerun outside the except block: the caught exception's traceback holds
    # every frame of the first attempt, with its inputs, references, device
    # buffers and workspaces, until the block exits.
    reporter.warning(f"{reason}; remeasuring every row of this graph without stalling")
    unstalled = copy.copy(config)
    unstalled.timing_policy = replace(config.timing_policy, stall_gate=False)
    return run(unstalled)


@dataclass
class RuntimePlan:
    """What one runtime runs for one graph, decided before inputs exist.

    Attributes:
        runtime: Runtime name; it prefixes ``message`` and ``error`` when a
            run selects several runtimes.
        engine_ids: Engines that apply (0 for each non-hipDNN row); empty
            means none.
        rows: Builds the rows. It gets the shared graph context, or None when
            no plan of the graph needs inputs.
        message: Why no engine applies.
        error: Why selection failed; the runtime then adds no rows.
        needs_inputs: The rows, or the reference row this runtime promises
            for a graph without engines, need generated inputs.
    """

    runtime: str
    engine_ids: List[int] = field(default_factory=list)
    rows: Callable[[Optional[_GraphContext]], List[ProviderEngineResult]] = (
        lambda ctx: []
    )
    message: Optional[str] = None
    error: Optional[str] = None
    needs_inputs: bool = False


#: ``plan(graph_path, graph_json, tensor_infos, config, reporter)``. A new
#: runtime adds one planner; :func:`run_graph` composes any of them.
GraphPlanner = Callable[
    [Path, Dict[str, Any], list, SuiteConfig, Reporter], RuntimePlan
]


def run_graph(
    graph_path: Path,
    graph_json: Dict[str, Any],
    tensor_infos: list,
    config: SuiteConfig,
    reporter: Reporter,
    planners: Sequence[GraphPlanner],
) -> GraphResult:
    """Run one graph through every planner's runtime, in order.

    Inputs and the ``--validate`` reference row are computed once and shared
    by every runtime. A runtime that does not apply adds no rows; its reason
    becomes ``GraphResult.message`` only when no runtime applies. A selection
    failure sets ``GraphResult.error`` and the other runtimes still run. A
    stall-gate failure reruns the whole graph without stalling
    (``_with_stall_fallback``), so all rows share one timing mode.
    """
    return _with_stall_fallback(
        lambda cfg: _run_graph(
            graph_path, graph_json, tensor_infos, cfg, reporter, planners
        ),
        config,
        reporter,
    )


def _run_graph(
    graph_path: Path,
    graph_json: Dict[str, Any],
    tensor_infos: list,
    config: SuiteConfig,
    reporter: Reporter,
    planners: Sequence[GraphPlanner],
) -> GraphResult:
    """One attempt of :func:`run_graph` with ``config``'s policy."""
    graph = GraphResult(
        graph_name=graph_json.get("name", graph_path.stem),
        graph_path=str(graph_path),
        results=[],
        graph_id=graph_id_for(graph_json),
    )
    plans = [
        plan(graph_path, graph_json, tensor_infos, config, reporter)
        for plan in planners
    ]

    def labelled(plan: RuntimePlan, text: str) -> str:
        return f"{plan.runtime}: {text}" if len(plans) > 1 else text

    errors = [labelled(p, p.error) for p in plans if p.error is not None]
    graph.error = "; ".join(errors) or None
    graph.engine_ids = [e for p in plans for e in p.engine_ids]
    if not graph.engine_ids:
        messages = [labelled(p, p.message) for p in plans if p.message]
        graph.message = "; ".join(messages) or None

    ctx: Optional[_GraphContext] = None
    if any(p.needs_inputs for p in plans):
        try:
            ctx = _graph_context(graph_path, graph_json, tensor_infos, config, reporter)
        except Exception as e:
            errors.append(f"Input data generation failed: {type(e).__name__}: {e}")
            graph.error = "; ".join(errors)
            return graph
        ctx.graph_json_str, ctx.graph_id = json.dumps(graph_json), graph.graph_id
        if config.validation.enabled:
            reference_row = _prepare_references(ctx)
            if reference_row is not None:
                graph.results.append(reference_row)

    for plan in plans:
        graph.results.extend(plan.rows(ctx))
    return graph


def hipdnn_planner(handle: Any) -> GraphPlanner:
    """Plan every selected or discovered hipDNN engine.

    Explicit ``--engine`` IDs run in the caller's order; otherwise the graph's
    ranked engines are discovered. A graph without engines still needs inputs:
    with ``--validate pytorch`` the timed PyTorch reference row is emitted
    when PyTorch can run the graph.

    Args:
        handle: hipdnn.Handle, or None to create one per engine (per-engine
            plugin paths).
    """

    def plan(
        graph_path: Path,
        graph_json: Dict[str, Any],
        tensor_infos: list,
        config: SuiteConfig,
        reporter: Reporter,
    ) -> RuntimePlan:
        message = None
        if config.engine_filter is not None:
            # Explicit --engine is a selection, not a post-discovery filter.
            # Keep the caller's order so per-engine plugin paths are
            # deterministic.
            engine_ids = list(config.engine_filter)
        else:
            try:
                # SuiteConfig requires --engine with several plugin paths, so
                # the shared handle exists whenever discovery runs. The
                # discovery graph is a temporary: it is not kept alive for the
                # engine rows.
                engine_ids = Executor(
                    json.dumps(graph_json), config.timing_policy
                ).discover_engines(handle)
            except UnsupportedGraphError as e:
                engine_ids, message = [], str(e)
            except Exception as e:
                return RuntimePlan(
                    "hipdnn", error=f"Engine discovery failed: {type(e).__name__}: {e}"
                )
        selections = config.engine_selections_for(engine_ids)
        return RuntimePlan(
            "hipdnn",
            engine_ids=engine_ids,
            rows=lambda ctx: [
                _run_engine_selection(ctx, handle, s.engine_id, s.plugin_path)
                for s in selections
            ],
            message=message,
            needs_inputs=True,
        )

    return plan


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
        lambda: run_single_provider_engine(
            ctx, handle, engine_id, name, version, plugin_path
        ),
    )


def pytorch_plan(
    graph_path: Path,
    graph_json: Dict[str, Any],
    tensor_infos: list,
    config: SuiteConfig,
    reporter: Reporter,
) -> RuntimePlan:
    """Plan the graph through the PyTorch executor as one engine row.

    An unsupported operation gives an error or skipped row without inputs
    (a static inspection).
    """
    from . import pytorch_ops

    unsupported = sorted(pytorch_ops.get_unsupported_operations(graph_json))
    if unsupported:
        msg = f"Graph contains unsupported operations: {unsupported}"
        make = _error_or_skip(
            config.pytorch_sdpa_backend is not PyTorchSdpaBackendName.DEFAULT
        )
        return RuntimePlan(
            "pytorch",
            engine_ids=[0],
            rows=lambda ctx: [
                _report_row(
                    reporter,
                    "pytorch",
                    lambda: make("pytorch", None, msg, engine_name="pytorch"),
                )
            ],
        )
    return RuntimePlan(
        "pytorch",
        engine_ids=[0],
        rows=lambda ctx: [
            _report_row(reporter, "pytorch", lambda: _run_pytorch_row(ctx, "engine")[0])
        ],
        needs_inputs=True,
    )


@dataclass(frozen=True)
class RockeSession:
    """Per-suite rocKE state: the shipped-kernel catalog and the device.

    Attributes:
        catalog: hipDNN-shipped rocKE variants for ``arch``.
        arch: GPU architecture name (``gfx950``).
        num_cus: Compute-unit count for rocKE's dispatcher.
        version: Installed ``rocke`` package version.
        loaded: Code objects loaded so far, by ``(library, toc_key)``. A
            variant loads once per suite, and every row that runs it reuses
            the module.
    """

    catalog: Catalog
    arch: str
    num_cus: int
    version: str
    loaded: Dict[Tuple[Path, str], Any] = field(default_factory=dict)


def rocke_session(config: SuiteConfig, reporter: Reporter) -> RockeSession:
    """Read the device and the shipped rocKE catalog once per suite.

    Raises:
        OSError, ValueError, KeyError: A descriptor file is unreadable or
            malformed.
    """
    import torch

    from .rocke_executor import device_arch, device_num_cus
    from .rocke_sdpa import load_catalog

    device = torch.device("cuda")
    arch = device_arch(device)
    catalog = load_catalog(config.plugin_paths or [], arch)
    if not catalog:
        reporter.warning(
            f"no hipDNN-shipped rocKE kernels for {arch} under the plugin path; "
            "only the rocKE auto kernel runs"
        )
    return RockeSession(
        catalog=catalog,
        arch=arch,
        num_cus=device_num_cus(device),
        version=_package_version("rocke"),
    )


def rocke_planner(session: RockeSession) -> GraphPlanner:
    """Plan every rocKE kernel that runs an SDPA graph.

    One row for each hipDNN-shipped variant that runs the graph, then one row
    for rocKE's auto kernel. A graph that no rocKE kernel runs gets no rows
    and no inputs; alone, its status is ``no_engines``.
    """

    def plan(
        graph_path: Path,
        graph_json: Dict[str, Any],
        tensor_infos: list,
        config: SuiteConfig,
        reporter: Reporter,
    ) -> RuntimePlan:
        try:
            problem = sdpa_problem(graph_json, tensor_infos)
            kernels: List[RockeKernel] = list(catalog_kernels(session.catalog, problem))
            reasons = [] if kernels else ["no hipDNN-shipped rocKE variant matches"]
            try:
                kernels.append(auto_kernel(problem, session.arch, session.num_cus))
            except UnsupportedGraphError as e:
                reasons.append(str(e))
        except UnsupportedGraphError as e:
            return RuntimePlan("rocke", message=str(e))
        except Exception as e:
            return RuntimePlan(
                "rocke",
                error=f"rocKE kernel selection failed: {type(e).__name__}: {e}",
            )
        if not kernels:
            return RuntimePlan("rocke", message="; ".join(reasons))

        def rows(ctx: Optional[_GraphContext]) -> List[ProviderEngineResult]:
            return [
                _report_row(
                    reporter,
                    kernel.name,
                    lambda k=kernel: _run_rocke_row(ctx, k, problem, session),
                )
                for kernel in kernels
            ]

        # No hipDNN engine IDs: one 0 sentinel per row, as for PyTorch.
        return RuntimePlan(
            "rocke", engine_ids=[0] * len(kernels), rows=rows, needs_inputs=True
        )

    return plan


def _run_rocke_row(
    ctx: _GraphContext,
    kernel: RockeKernel,
    problem: SdpaProblem,
    session: RockeSession,
) -> ProviderEngineResult:
    """Load, time and validate one rocKE kernel."""
    config = ctx.config
    row = ProviderEngineResult(
        runtime="rocke",
        # The row name is stable across runs, so it is the engine ID.
        engine_id=kernel.name,
        engine_name=kernel.name,
        status="success",
        engine_version=session.version,
    )
    try:
        from .pytorch_buffer_manager import PyTorchCudaBufferManager
        from .rocke_executor import RockeExecutor

        executor = RockeExecutor(
            kernel, problem, config.timing_policy, loaded=session.loaded
        )
        with PyTorchCudaBufferManager(ctx.tensor_infos, device=executor.device) as bm:
            bm.allocate_all()
            bm.load_input_data(ctx.input_data)
            bm.zero_outputs()
            executor.prepare(bm.get_tensors())
            row.ootb.cpu_build_time_ms = executor.build_time_ms
            _measure_row(row, ctx, executor.benchmark)
            _validate_row(row, ctx, bm, executor.execute_once)
    except StallFallbackError:
        raise  # run_graph remeasures the whole graph unstalled.
    except UnsupportedGraphError as e:
        return _failed_row(ProviderEngineResult.skipped_row, row, str(e))
    except Exception as e:
        return _failed_row(
            ProviderEngineResult.error_row, row, f"{type(e).__name__}: {e}"
        )
    return row

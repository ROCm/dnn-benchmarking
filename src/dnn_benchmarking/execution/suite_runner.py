# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Per-graph engine iteration: one timed row per engine (plus PyTorch rows).

Uses explicit ``--engine`` IDs in caller order when provided; otherwise
discovers ranked engine IDs for the graph via ``Graph.get_ranked_engine_ids``.
Every row is timed by ``timing.measure`` (through the executors) and, with
``--validate``, compared against reference outputs computed once per graph.
"""

import copy
import json
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

from ..common.exceptions import UnsupportedGraphError
from ..config.benchmark_config import PyTorchSdpaBackendName, SuiteConfig
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
from .correctness import check_correctness, mismatch
from .oracle import (
    build_tuned_plan,
    run_pytorch_tuned,
    run_tuned_plan,
    tuned_in_process,
)


@dataclass
class _TimedPytorchRow:
    """Timed PyTorch row plus reference outputs (reference role only).

    ``reference_pass_failed`` separates the two ways the row can end without
    outputs: timing itself failed, or timing finished and the reference output
    pass failed (MATH SDPA out of memory on a large graph, for example). The
    skip reason has to name the right one.
    """

    result: ProviderEngineResult
    outputs: Optional[Dict[int, ReferenceOutput]] = None
    reference_pass_failed: bool = False


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

    ``data`` is always a host copy (one per output per graph); the host
    fallback comparison needs it. With ``keep_device``, a device clone is
    also kept so each engine can compare on the GPU without a host copy of
    its own output.
    """
    outputs: Dict[int, ReferenceOutput] = {}
    tensors = buffer_manager.get_tensors()
    for tensor_info in buffer_manager.get_output_tensors():
        data = buffer_manager.get_output_data(tensor_info.uid)
        if data is not None:
            tensor = tensors.get(tensor_info.uid)
            outputs[tensor_info.uid] = ReferenceOutput(
                data=data,
                tensor_uid=tensor_info.uid,
                # Clone: the buffer manager frees its tensors on exit.
                device_data=(
                    tensor.detach().clone()
                    if keep_device and tensor is not None and tensor.is_cuda
                    else None
                ),
            )
    return outputs


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
) -> _TimedPytorchRow:
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
    # "timing" until the timed loop is done, then "reference" while the
    # reference output pass runs. Whichever one raises is what the skip reason
    # has to name.
    stage = "timing"
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
            if role == "reference":
                stage = "reference"
                bm.zero_outputs()
                # Timing above used default dispatch; the outputs other rows
                # are graded against come from repeatable SDPA.
                with pytorch_ops.reference_sdpa_pass():
                    executor.execute_once(tensors)
                # A profiling child allocates its own VRAM, so profiled runs
                # keep host-only references (and DeviceBuffer I/O).
                outputs = _pytorch_reference_outputs_from_buffer(
                    bm, keep_device=not config.metrics.opt_in_pass_requested
                )
                stage = "post"
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
        return _TimedPytorchRow(
            result=_failed_row(_error_or_skip(strict), row, str(e)),
            reference_pass_failed=stage == "reference",
        )
    except Exception as e:
        fatal = role == "engine" or strict
        return _TimedPytorchRow(
            result=_failed_row(_error_or_skip(fatal), row, f"{type(e).__name__}: {e}"),
            reference_pass_failed=stage == "reference",
        )
    return _TimedPytorchRow(result=row, outputs=outputs)


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

            if ctx.reference_outputs is not None:
                bm.zero_outputs()
                executor.execute_once(handle, variant_pack)
                row.ootb.correctness = check_correctness(
                    bm,
                    ctx.tensor_infos,
                    ctx.reference_outputs,
                    config.validation.provider.value,
                    config,
                )
            elif config.validation.enabled:
                # Validation was requested but no reference is usable: keep
                # --validate a hard gate.
                row.ootb.correctness = mismatch(
                    config, ctx.reference_error or "Reference outputs unavailable"
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
        raise  # run_graph_* remeasures the whole graph unstalled.
    except UnsupportedGraphError as e:
        return _failed_row(ProviderEngineResult.skipped_row, row, str(e))
    except Exception as e:
        return _failed_row(
            ProviderEngineResult.error_row, row, f"{type(e).__name__}: {e}"
        )

    # Profiling re-runs the workload after the timed pass, so profiler
    # overhead cannot pollute the headline numbers.
    if config.metrics.opt_in_pass_requested:
        _run_profiling(row, ctx, engine_id, engine_name, plugin_path)
    return row


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
    timed native path fails.
    """
    config = ctx.config
    provider, ctx.reference_error = _reference_provider(config, ctx.graph_json)
    if provider is None:
        warn_once("validation", f"{ctx.graph_name}: {ctx.reference_error}")
        return None

    timed: Optional[_TimedPytorchRow] = None

    def run() -> ProviderEngineResult:
        nonlocal timed
        timed = _run_pytorch_row(ctx, "reference")
        return timed.result

    row = _report_row(ctx.reporter, "pytorch reference", run)
    assert timed is not None
    if timed.outputs is not None:
        ctx.reference_outputs = timed.outputs
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
        with Timer() as cpu_timer:
            cpu_outputs, cpu_error = _compute_reference_outputs_once(
                provider, ctx.graph_json, ctx.input_data, config
            )
        ctx.reference_outputs, ctx.reference_error = cpu_outputs, cpu_error
        # The timed row produced no outputs: say which of its two steps
        # failed, that engines are still graded, and what the CPU fallback
        # cost, since no row times it.
        if row.status == "skipped" and ctx.reference_outputs is not None:
            failed_step = (
                "Reference output pass failed"
                if timed.reference_pass_failed
                else "Timing skipped"
            )
            row.skip_reason = (
                f"{failed_step} ({row.skip_reason}). "
                "Engine outputs are still validated against reference "
                "outputs computed on the CPU, which took "
                f"{cpu_timer.elapsed_ms / 1000:.1f} s."
            )
    return row


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
        reporter.warning(f"{e}; remeasuring every row of this graph without stalling")
        unstalled = copy.copy(config)
        unstalled.timing_policy = replace(config.timing_policy, stall_gate=False)
        return run(unstalled)


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
            discovery = Executor(graph_json_str, config.timing_policy)
            # SuiteConfig requires --engine with several plugin paths, so the
            # shared handle exists whenever discovery runs.
            engine_ids = discovery.discover_engines(handle)
        except UnsupportedGraphError as e:
            engine_ids = []
            graph.message = str(e)
        except Exception as e:
            graph.error = f"Engine discovery failed: {type(e).__name__}: {e}"
            return graph
    graph.engine_ids = engine_ids

    try:
        ctx = _graph_context(graph_path, graph_json, tensor_infos, config, reporter)
        ctx.graph_json_str, ctx.graph_id = graph_json_str, graph.graph_id
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
        lambda: run_single_provider_engine(
            ctx, handle, engine_id, name, version, plugin_path
        ),
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
        _report_row(reporter, "pytorch", lambda: _run_pytorch_row(ctx, "engine").result)
    )
    return graph

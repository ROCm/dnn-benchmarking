# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Runtime startup: check each selected runtime once, return a per-graph runner."""

import importlib.util
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol

from ..common import torch_support
from ..common.rocm_runtime import initialize_pip_rocm_runtime
from ..config.benchmark_config import RuntimeName, SuiteConfig
from ..execution.oracle import HipdnnTunedRuns
from ..execution.suite_runner import (
    GraphPlanner,
    hipdnn_planner,
    pytorch_plan,
    rocke_planner,
    rocke_session,
    run_graph,
    set_plugin_path,
    without_stall_gate,
)
from ..reporting.reporter import Reporter
from ..reporting.suite_results import GraphResult, engine_id_hex
from ..validation.reference_provider import ReferenceProviderRegistry
from .parser import TYPED_ENGINE_NAMES


class GraphRunner(Protocol):
    """Runs one graph through every selected runtime.

    ``stall_gate=False`` remeasures a graph whose stall gate failed in an
    ``--oracle`` tuned run, so all its rows share one timing mode.
    """

    def __call__(
        self,
        graph_path: Path,
        graph_json: Dict[str, Any],
        tensor_infos: list,
        *,
        stall_gate: bool = True,
    ) -> GraphResult: ...


class RuntimeStartupError(Exception):
    """Startup failed before any graph ran; ``exit_code`` is the CLI exit code."""

    def __init__(self, message: str, exit_code: int = 1) -> None:
        super().__init__(message)
        self.exit_code = exit_code


def start_runtime(
    config: SuiteConfig,
    reporter: Reporter,
    resources: ExitStack,
    tuned_runs: Optional[HipdnnTunedRuns] = None,
) -> GraphRunner:
    """Check every selected runtime (and the reference provider); return the runner.

    Each graph runs through the runtimes in ``config.runtimes`` order, and
    every runtime adds its rows to the same graph result. The shared hipDNN
    handle is destroyed when ``resources`` closes, so the caller decides
    which state (the ``--fast`` MIOpen user database) outlives it. hipDNN
    engine rows queue their ``--oracle`` tuned runs in ``tuned_runs``.

    Raises:
        RuntimeStartupError: a runtime is unavailable (exit 1) or an explicit
            ``--engine`` that no loaded plugin provides (exit 2).
    """
    # The reference-provider check and the other runtimes import torch. A ROCm
    # torch wheel may preload another libhipdnn_backend, so bind the selected
    # hipDNN frontend first. Defer reporting load errors until after the
    # reference check, preserving its startup error when both are unavailable.
    _check_plugin_paths(config)
    if RuntimeName.HIPDNN in config.runtimes:
        hipdnn, load_error = _load_hipdnn_bindings()
    _check_reference_provider(config)
    planners: List[GraphPlanner] = []
    for runtime in config.runtimes:
        if runtime is RuntimeName.HIPDNN:
            handle = _create_hipdnn_handle(config, hipdnn, load_error)
            if handle is not None:
                resources.callback(hipdnn.destroy_handle, handle)
            planners.append(
                hipdnn_planner(handle, prime_once=config.fast, tuned_runs=tuned_runs)
            )
        else:
            planners.append(_STARTERS[runtime](config, reporter))

    def run(
        graph_path: Path,
        graph_json: Dict[str, Any],
        tensor_infos: list,
        *,
        stall_gate: bool = True,
    ) -> GraphResult:
        graph_config = config if stall_gate else without_stall_gate(config)
        return run_graph(
            graph_path, graph_json, tensor_infos, graph_config, reporter, planners
        )

    return run


def _start_pytorch(config: SuiteConfig, reporter: Reporter) -> GraphPlanner:
    _check_torch_gpu("pytorch")
    return pytorch_plan


def _start_rocke(config: SuiteConfig, reporter: Reporter) -> GraphPlanner:
    _check_torch_gpu("rocke")
    if not torch_support.is_rocm_build():
        raise RuntimeStartupError(
            "--runtime rocke: rocKE kernels need a ROCm build of torch"
        )
    missing = [
        name
        for name in ("rocke", "kernels", "builders", "dispatch")
        if importlib.util.find_spec(name) is None
    ]
    if missing:
        raise RuntimeStartupError(
            f"--runtime rocke: rocKE Python packages are not importable "
            f"({', '.join(missing)}); run setup_env.py on Linux, or "
            "pip install -e rocke/platform and rocke/library from "
            "rocm-libraries/dnn-providers/hip-kernel-provider"
        )
    try:
        session = rocke_session(config, reporter)
    except (OSError, ValueError) as e:
        raise RuntimeStartupError(
            f"--runtime rocke: cannot read the rocKE descriptors: {e}"
        ) from e
    return rocke_planner(session)


#: Startup of each runtime except hipDNN, whose bindings must load before
#: torch (see :func:`start_runtime`). A new runtime adds one entry here.
_STARTERS: Dict[RuntimeName, Callable[[SuiteConfig, Reporter], GraphPlanner]] = {
    RuntimeName.PYTORCH: _start_pytorch,
    RuntimeName.ROCKE: _start_rocke,
}


def _check_torch_gpu(runtime: str) -> None:
    """Require a GPU-capable torch; torch.cuda is the authoritative check.

    A CPU-only torch cannot run these benchmarks even when ROCm tools see a
    device.
    """
    if not torch_support.module_available():
        raise RuntimeStartupError(
            f"--runtime {runtime}: PyTorch is not importable; install a ROCm "
            "or CUDA build of torch"
        )
    if not torch_support.gpu_available():
        raise RuntimeStartupError(
            f"--runtime {runtime}: PyTorch sees no GPU (torch.cuda.is_available() "
            "is False); install a ROCm or CUDA build of torch"
        )


def _check_reference_provider(config: SuiteConfig) -> None:
    if not config.validation.enabled:
        return
    name = config.validation.provider.value
    try:
        ref = ReferenceProviderRegistry.get_provider(name)
    except ValueError as e:
        raise RuntimeStartupError(f"--validate {name}: {e}") from e
    if not ref.is_available():
        raise RuntimeStartupError(
            f"--validate {name}: reference runtime unavailable "
            "(check that its dependencies are installed)"
        )


def _load_hipdnn_bindings() -> tuple[Any, Optional[Exception]]:
    """Load bindings before a ROCm torch reference probe can load its backend."""
    try:
        initialize_pip_rocm_runtime()
        import hipdnn_frontend as hipdnn
    except (ImportError, OSError, RuntimeError) as e:
        return None, e
    return hipdnn, None


def _create_hipdnn_handle(
    config: SuiteConfig, hipdnn: Any, load_error: Optional[Exception]
) -> Any:
    """Create the shared hipDNN handle; None when plugin paths are per engine.

    Handle creation is the authoritative GPU/runtime check for hipDNN. With
    ``--fast`` the plugin path names one plugin library, so the handle loads
    and initialises only that plugin.
    """
    if load_error is not None:
        if isinstance(load_error, ImportError):
            raise RuntimeStartupError(
                f"hipdnn_frontend is not importable ({load_error}); install the hipDNN "
                "Python bindings or use --runtime pytorch"
            ) from load_error
        raise RuntimeStartupError(
            f"hipDNN handle creation failed: {load_error}"
        ) from load_error
    try:
        if config.plugin_paths is not None and len(config.plugin_paths) > 1:
            # The runner creates one handle per engine/plugin pair; check each
            # pair now so a bad engine or directory fails before any graph.
            for selection in config.engine_selections_for(config.engine_filter):
                set_plugin_path(hipdnn, selection.plugin_path)
                _check_engines_loaded(
                    hipdnn,
                    hipdnn.Handle(),
                    [selection.engine_id],
                    selection.plugin_path,
                )
            return None
        set_plugin_path(hipdnn, config.plugin_path)
        handle = hipdnn.Handle()
    except ImportError as e:
        raise RuntimeStartupError(
            f"hipdnn_frontend is not importable ({e}); install the hipDNN "
            "Python bindings or use --runtime pytorch"
        ) from e
    except (OSError, RuntimeError) as e:
        raise RuntimeStartupError(f"hipDNN handle creation failed: {e}") from e
    if config.engine_filter is not None:
        _check_engines_loaded(hipdnn, handle, config.engine_filter)
    return handle


def _check_plugin_paths(config: SuiteConfig) -> None:
    """Reject unusable ``--plugin-path`` entries before any runtime starts.

    hipDNN takes a plugin directory, whose libraries all load, or one plugin
    library file, which loads alone; with ``--fast`` the one path must be a
    library file. rocKE reads the kernel descriptors under plugin
    directories, so it rejects a file.
    """
    paths = config.plugin_paths or []
    if RuntimeName.HIPDNN in config.runtimes:
        unusable = [str(p) for p in paths if not (p.is_file() or p.is_dir())]
        if unusable:
            raise RuntimeStartupError(
                "hipDNN plugin path is not a plugin directory or library file: "
                f"{', '.join(unusable)} (check --plugin-path)",
                exit_code=2,
            )
        if config.fast and not config.plugin_path.is_file():
            raise RuntimeStartupError(
                f"--fast: --plugin-path must name one plugin library file "
                f"(for example .../hipdnn_plugins/engines/libmiopen_plugin.so); "
                f"got {config.plugin_path}",
                exit_code=2,
            )
    if RuntimeName.ROCKE in config.runtimes:
        files = [str(p) for p in paths if p.is_file()]
        if files:
            raise RuntimeStartupError(
                "--runtime rocke reads the kernel descriptors under plugin "
                f"directories; --plugin-path names a file: {', '.join(files)}",
                exit_code=2,
            )


def _check_engines_loaded(
    hipdnn: Any, handle: Any, engine_ids: List[int], plugin_path: Optional[Path] = None
) -> None:
    """Reject explicit ``--engine`` IDs that no loaded plugin provides."""
    get_info = getattr(handle, "get_engine_info", None)
    if get_info is None:
        return
    unknown = []
    for engine_id in dict.fromkeys(engine_ids):
        try:
            get_info(engine_id)
        except Exception:
            unknown.append(engine_id)
    if unknown:
        raise RuntimeStartupError(
            "--engine: not provided by any "
            + (f"plugin loaded from {plugin_path}" if plugin_path else "loaded plugin")
            + ": "
            + ", ".join(_engine_label(hipdnn, e) for e in unknown),
            exit_code=2,
        )


def _engine_label(hipdnn: Any, engine_id: int) -> str:
    """``NAME/0xHEX`` when the ID has a registered or typed name, else ``0xHEX``."""
    try:
        name = hipdnn.engine_id_to_name(engine_id)
    except Exception:
        name = ""
    name = name or TYPED_ENGINE_NAMES.get(engine_id, "")
    hex_id = engine_id_hex(engine_id)
    return f"{name}/{hex_id}" if name else hex_id

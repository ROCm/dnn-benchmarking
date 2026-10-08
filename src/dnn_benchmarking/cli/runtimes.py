# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Runtime startup: check the selected runtime once, return a per-graph runner."""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ..common import torch_support
from ..common.rocm_runtime import initialize_pip_rocm_runtime
from ..config.benchmark_config import RuntimeName, SuiteConfig
from ..execution.suite_runner import (
    run_graph_all_providers,
    run_graph_pytorch,
    set_plugin_path,
)
from ..reporting.reporter import Reporter
from ..reporting.suite_results import GraphResult, engine_id_hex
from ..validation.reference_provider import ReferenceProviderRegistry
from .parser import TYPED_ENGINE_NAMES

#: ``run_graph(graph_path, graph_json, tensor_infos) -> GraphResult``
GraphRunner = Callable[[Path, Dict[str, Any], list], GraphResult]


class RuntimeStartupError(Exception):
    """Startup failed before any graph ran; ``exit_code`` is the CLI exit code."""

    def __init__(self, message: str, exit_code: int = 1) -> None:
        super().__init__(message)
        self.exit_code = exit_code


def start_runtime(config: SuiteConfig, reporter: Reporter) -> GraphRunner:
    """Check the runtime (and the reference provider); return the runner.

    Raises:
        RuntimeStartupError: runtime unavailable (exit 1) or an explicit
            ``--engine`` that no loaded plugin provides (exit 2).
    """
    _check_reference_provider(config)
    if config.runtime is RuntimeName.PYTORCH:
        # torch.cuda is the authoritative GPU check here: a CPU-only torch
        # cannot run these benchmarks even when ROCm tools see a device.
        if not torch_support.module_available():
            raise RuntimeStartupError(
                "--runtime pytorch: PyTorch is not importable; install a ROCm "
                "or CUDA build of torch"
            )
        if not torch_support.gpu_available():
            raise RuntimeStartupError(
                "--runtime pytorch: PyTorch sees no GPU (torch.cuda.is_available() "
                "is False); install a ROCm or CUDA build of torch"
            )
        return lambda path, graph_json, infos: run_graph_pytorch(
            path, graph_json, infos, config, reporter
        )

    handle = _create_hipdnn_handle(config)
    return lambda path, graph_json, infos: run_graph_all_providers(
        path, graph_json, infos, config, handle, reporter
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


def _create_hipdnn_handle(config: SuiteConfig) -> Any:
    """Create the shared hipDNN handle; None when plugin paths are per engine.

    Handle creation is the authoritative GPU/runtime check for hipDNN.
    """
    missing = [str(p) for p in config.plugin_paths or [] if not p.is_dir()]
    if missing:
        raise RuntimeStartupError(
            f"hipDNN plugin path is not a directory: {', '.join(missing)} "
            "(check --plugin-path)",
            exit_code=2,
        )
    try:
        initialize_pip_rocm_runtime()
        import hipdnn_frontend as hipdnn

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

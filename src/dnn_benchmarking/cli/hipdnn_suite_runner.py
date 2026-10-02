# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""hipDNN suite benchmark startup and graph dispatch."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..common.rocm_runtime import initialize_pip_rocm_runtime
from ..config.benchmark_config import SuiteConfig
from ..execution.suite_runner import run_graph_all_providers, set_plugin_path
from ..reporting.reporter import Reporter
from ..reporting.suite_results import GraphResult
from .suite_runner_cli import _run_suite_graphs_after_startup


def _run_loaded_hipdnn_graph(
    graph_path: Path,
    graph_json: Dict[str, Any],
    tensor_infos: list,
    config: SuiteConfig,
    reporter: Reporter,
    handle: Any,
) -> GraphResult:
    return run_graph_all_providers(
        graph_path, graph_json, tensor_infos, config, handle, reporter=reporter
    )


def load_hipdnn_bindings() -> Tuple[Any, Optional[Exception]]:
    """Import hipdnn_frontend. Returns (module, None) or (None, error).

    Call this before anything imports torch. A ROCm torch wheel ships its own,
    possibly older, libhipdnn_backend; whichever library the process loads
    first owns that soname, so importing torch first binds hipdnn_frontend to
    torch's copy instead of the install setup_env put first on
    LD_LIBRARY_PATH.
    """
    try:
        initialize_pip_rocm_runtime()
        import hipdnn_frontend
    except (ImportError, RuntimeError) as e:
        return None, e
    return hipdnn_frontend, None


def run_hipdnn_suite_benchmark(
    graph_paths: List[Path],
    config: SuiteConfig,
    output_path: Optional[Path],
    reporter: Reporter,
    hipdnn: Any,
    load_error: Optional[Exception],
    tarball_source: Optional[str] = None,
) -> int:
    """Run suite graphs through hipDNN providers/engines.

    ``hipdnn`` and ``load_error`` come from :func:`load_hipdnn_bindings`, which
    the caller runs before the reference-provider probe and the suite header
    (both import torch). Load errors are reported here, after the header.
    """
    reporter.print_suite_header(
        len(graph_paths),
        tarball_source=tarball_source,
        extra_profiling_runs=config.metrics.extra_runs_per_engine,
    )

    handle = None
    reporter.print_hipdnn_init_start()
    if hipdnn is None:
        reporter.print_hipdnn_init_newline()
        if isinstance(load_error, ImportError):
            reporter.print_error(
                "hipdnn_frontend not available. Install hipDNN Python bindings "
                f"first. ({load_error})"
            )
        else:
            reporter.print_error(f"Failed to create hipDNN handle: {load_error}")
        return 1
    # hipDNN handle creation is the authoritative GPU/runtime check for
    # this backend. Apply plugin paths before constructing the handle; do
    # not depend on optional telemetry tools such as amd-smi.
    try:
        plugin_paths = config.plugin_paths
        per_engine_plugin_paths = plugin_paths is not None and len(plugin_paths) > 1

        if not per_engine_plugin_paths:
            set_plugin_path(hipdnn, config.plugin_path)
            handle = hipdnn.Handle()
    except RuntimeError as e:
        reporter.print_hipdnn_init_newline()
        reporter.print_error(f"Failed to create hipDNN handle: {e}")
        return 1

    reporter.print_hipdnn_init_done()

    def run_loaded_graph(
        graph_path: Path,
        graph_json: Dict[str, Any],
        tensor_infos: list,
        config: SuiteConfig,
        reporter: Reporter,
    ) -> GraphResult:
        return _run_loaded_hipdnn_graph(
            graph_path, graph_json, tensor_infos, config, reporter, handle
        )

    return _run_suite_graphs_after_startup(
        graph_paths=graph_paths,
        config=config,
        output_path=output_path,
        reporter=reporter,
        run_loaded_graph=run_loaded_graph,
    )

# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Main entry point for dnn-benchmark CLI."""

import os
import sys
import tempfile

from pathlib import Path


# Redirect ROCm/tool caches away from the network home directory before any
# ROCm library is imported. MIOpen, comgr, pip, and torch all default to
# ~/.cache/ or ~/.miopen/, which is a network filesystem on AMD dev machines.
# DNN_BENCH_WORKSPACE is set by setup_env.py; otherwise use a private,
# process-scoped temporary directory rather than the network home directory.
def _create_cache_base(
    workspace: str | None,
) -> tuple[Path, tempfile.TemporaryDirectory[str] | None]:
    """Return the cache base and the fallback directory object, if allocated."""
    if workspace:
        return Path(workspace), None

    temporary_directory = tempfile.TemporaryDirectory(prefix="dnn-bench-cache-")
    return Path(temporary_directory.name), temporary_directory


# Retaining this object keeps the fallback directory alive; its finalizer
# removes it when the CLI process exits.
_CACHE_BASE, _TEMPORARY_CACHE_DIRECTORY_LIFETIME = _create_cache_base(
    os.environ.get("DNN_BENCH_WORKSPACE")
)
_LOCAL_CACHE_DEFAULTS = {
    "XDG_CACHE_HOME": _CACHE_BASE / "cache",
    "MIOPEN_USER_DB_PATH": _CACHE_BASE / "miopen_cache",
    "MIOPEN_CUSTOM_CACHE_DIR": _CACHE_BASE / "miopen_cache",
    "AMD_COMGR_CACHE_DIR": _CACHE_BASE / "comgr_cache",
}
for _var, _default in _LOCAL_CACHE_DEFAULTS.items():
    if _var not in os.environ:
        _default.mkdir(parents=True, exist_ok=True)
        os.environ[_var] = str(_default)

from ..common.exceptions import GraphLoadError
from ..common.rocm_runtime import initialize_pip_rocm_runtime
from ..reporting.reporter import Reporter
from .config_file import apply_config_file
from .internal_profiling import run_internal_profiling
from .parser import create_parser
from .suite_runner_cli import run_suite_cli


def _resolve_graphs(args, reporter: Reporter):
    """Resolve --graph args to file paths. Returns (tmpdirs, files, tarball_source)."""
    from ..graph.resolver import is_tarball as _is_tarball, resolve_graph_files_multi

    if len(args.graph) == 1 and _is_tarball(args.graph[0]):
        reporter.print_extracting(args.graph[0])

    try:
        tmpdirs, files, tarball_source = resolve_graph_files_multi(args.graph)
    except GraphLoadError as e:
        reporter.print_error(str(e))
        return None, None, None

    if tmpdirs:
        reporter.print_extracted_count(len(files), str(args.graph))

    if not files:
        for td in tmpdirs:
            td.cleanup()
        reporter.print_no_graphs_found(str(args.graph))
        return tmpdirs, None, None

    return tmpdirs, files, tarball_source


def _apply_run_dir(args) -> None:
    """Point the report and both artifact roots at one directory.

    A reader that holds only the report resolves artifacts beside it, so a
    single directory is what the viewer can open without further picking.
    Anything the user set explicitly is left alone.
    """
    run_dir = getattr(args, "run_dir", None)
    if run_dir is None:
        return
    defaults = {
        "output": run_dir / "results.json",
        "tensor_output_dir": run_dir / "tensors",
        "profiling_output_dir": run_dir / "profiling-output",
    }
    for dest, value in defaults.items():
        if getattr(args, dest, None) is None:
            setattr(args, dest, value)


def main() -> int:
    """CLI entry point."""
    parser = create_parser(suppress_defaults=True)
    args = parser.parse_args()
    # PyTorch capability and host-info probes preload its SDK backend. Claim the
    # selected application's backend first, before those probes can run.
    if os.environ.get("HIPDNN_SDK"):
        try:
            initialize_pip_rocm_runtime()
        except RuntimeError as error:
            print(f"ERROR: {error}", file=sys.stderr)
            return 1

    try:
        apply_config_file(args)
    except ValueError as e:
        parser.error(str(e))
    _apply_run_dir(args)

    # Backend-specific startup is the authoritative GPU availability check:
    # PyTorch mode requires GPU-enabled torch, while hipDNN mode creates a real
    # hipdnn_frontend.Handle after applying any configured plugin paths. Do not
    # gate here on telemetry tools such as amd-smi; they are optional
    # and can be absent even when execution is valid.
    if getattr(args, "internal_profiling_run", False):
        return run_internal_profiling(args)
    if not args.graph:
        parser.error("--graph is required unless --config provides graphs")
    reporter = Reporter()

    tmpdirs, resolved_files, tarball_source = _resolve_graphs(args, reporter)
    if resolved_files is None:
        return 1

    try:
        return run_suite_cli(
            args,
            graph_paths=[Path(p) for p in resolved_files],
            reporter=reporter,
            tarball_source=tarball_source,
        )
    finally:
        if tmpdirs:
            for td in tmpdirs:
                td.cleanup()


if __name__ == "__main__":
    sys.exit(main())

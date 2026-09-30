# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Main entry point for dnn-benchmark CLI."""

import os
import sys
import tempfile
from pathlib import Path
from typing import List, Optional, Sequence

from ..common.exceptions import GraphLoadError
from .config_file import apply_config_file
from .parser import create_parser

_CACHE_ENV_SUBDIRS = {
    "XDG_CACHE_HOME": "cache",
    "MIOPEN_USER_DB_PATH": "miopen_cache",
    "MIOPEN_CUSTOM_CACHE_DIR": "miopen_cache",
    "AMD_COMGR_CACHE_DIR": "comgr_cache",
}


def _create_cache_base(
    workspace: str | None,
) -> tuple[Path, tempfile.TemporaryDirectory[str] | None]:
    """Return the cache base and the fallback directory object, if allocated."""
    if workspace:
        return Path(workspace), None

    temporary_directory = tempfile.TemporaryDirectory(prefix="dnn-bench-cache-")
    return Path(temporary_directory.name), temporary_directory


def _configure_cache_env() -> tempfile.TemporaryDirectory[str] | None:
    """Point unset ROCm/tool cache variables away from the (network) home dir.

    MIOpen, comgr and torch default to ~/.cache or ~/.miopen. Uses
    DNN_BENCH_WORKSPACE (set by setup_env.py) or a private temporary directory,
    allocated only when a variable is unset. Must run before any GPU runtime
    initialises. Keep the returned object alive for the run; its finalizer
    removes the temporary directory.
    """
    unset = [var for var in _CACHE_ENV_SUBDIRS if var not in os.environ]
    if not unset:
        return None
    base, lifetime = _create_cache_base(os.environ.get("DNN_BENCH_WORKSPACE"))
    for var in unset:
        path = base / _CACHE_ENV_SUBDIRS[var]
        path.mkdir(parents=True, exist_ok=True)
        os.environ[var] = str(path)
    return lifetime


def _resolve_graphs(
    args, reporter
) -> tuple[list, Optional[List[str]], Optional[str]]:
    """Resolve --graph args. Returns (tmpdirs, files or None, tarball_source)."""
    from ..graph.resolver import is_tarball, resolve_graph_files_multi

    graphs = " ".join(args.graph)
    if len(args.graph) == 1 and is_tarball(args.graph[0]):
        reporter.info(f"Extracting {graphs} ...")
    try:
        tmpdirs, files, tarball_source = resolve_graph_files_multi(args.graph)
    except GraphLoadError as e:
        reporter.error(str(e))
        return [], None, None
    if tmpdirs:
        reporter.info(f"Extracted {len(files)} graph file(s) from {graphs}")
    if not files:
        reporter.error(f"No graph files found in: {graphs}")
        return tmpdirs, None, None
    return tmpdirs, files, tarball_source


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["compare"]:
        from ..reporting.compare import main as compare_main

        return compare_main(argv[1:])

    _cache_lifetime = _configure_cache_env()  # noqa: F841 - keeps the dir alive
    parser = create_parser(suppress_defaults=True)
    args = parser.parse_args(argv)
    try:
        apply_config_file(args)
    except ValueError as e:
        parser.error(str(e))

    # Backend startup (in the suite runner) is the authoritative GPU check;
    # optional telemetry tools such as amd-smi are not gated here.
    if args.internal_profiling_run:
        from .internal_profiling import run_internal_profiling

        return run_internal_profiling(args)
    if not args.graph:
        parser.error("--graph is required unless --config provides graphs")

    from ..reporting.reporter import Reporter

    reporter = Reporter(quiet=args.quiet, verbose=args.verbose)
    tmpdirs, resolved_files, tarball_source = _resolve_graphs(args, reporter)
    try:
        if resolved_files is None:
            return 1
        from .suite_runner_cli import run_suite_cli

        return run_suite_cli(
            args,
            graph_paths=[Path(p) for p in resolved_files],
            reporter=reporter,
            tarball_source=tarball_source,
        )
    finally:
        for td in tmpdirs:
            td.cleanup()

# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Main entry point for dnn-benchmark CLI."""

import os
import sys
import tempfile
from pathlib import Path
from typing import List, Optional, Sequence

from ..common.exceptions import GraphLoadError
from ..common.run_state import TEMP_ROOT_ENV, run_state_dir
from .config_file import apply_config_file
from .parser import create_parser

#: Caches that only save time, kept across runs: under
#: ``--benchmark-work-dir``/``cache``, else in DNN_BENCH_WORKSPACE. The MIOpen
#: kernel cache holds compiled binaries: Find compiles before it times a
#: solver, so a cache hit shortens Find but does not change the selection.
#: Selection state (the MIOpen user database) belongs to each OOTB child run;
#: tuned child runs share the command's tuned database
#: (:func:`common.run_state.tuned_db_dir`).
_WORKSPACE_CACHE_SUBDIRS = {
    "XDG_CACHE_HOME": "cache",
    "AMD_COMGR_CACHE_DIR": "comgr_cache",
    "MIOPEN_CUSTOM_CACHE_DIR": "miopen_kernel_cache",
}


def _set_cache_dir(var: str, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    os.environ[var] = str(path)


def _configure_cache_env(
    work_dir: Optional[Path],
) -> tempfile.TemporaryDirectory[str] | None:
    """Point ROCm/tool cache variables away from the (network) home dir.

    MIOpen, comgr and torch default to ~/.cache or ~/.miopen. With
    ``work_dir`` (--benchmark-work-dir), the caches go to ``work_dir/cache``
    and replace inherited values, and this command gets one run directory
    under ``work_dir/tmp``: every temporary per-run directory of the command
    (OOTB MIOpen user databases, child result files) goes inside it, and the run
    directory is removed when the command ends. Without ``work_dir``, unset
    caches go to DNN_BENCH_WORKSPACE (set by setup_env.py), or to a temporary
    directory without a workspace. Must run before any GPU runtime
    initialises. Keep the returned object alive for the run; its finalizer
    removes the temporary directory.
    """
    if work_dir is not None:
        for var, subdir in _WORKSPACE_CACHE_SUBDIRS.items():
            _set_cache_dir(var, work_dir / "cache" / subdir)
        _set_cache_dir(TEMP_ROOT_ENV, work_dir / "tmp")
        # A run directory of its own, so commands that share the work dir
        # never remove each other's temporary state.
        run_dir = run_state_dir("dnn-bench-run-")
        os.environ[TEMP_ROOT_ENV] = run_dir.name
        return run_dir
    unset = [var for var in _WORKSPACE_CACHE_SUBDIRS if var not in os.environ]
    if not unset:
        return None
    workspace = os.environ.get("DNN_BENCH_WORKSPACE")
    lifetime = None if workspace else run_state_dir("dnn-bench-cache-")
    base = Path(workspace) if workspace else Path(lifetime.name)
    for var in unset:
        _set_cache_dir(var, base / _WORKSPACE_CACHE_SUBDIRS[var])
    return lifetime


def _resolve_graphs(args, reporter) -> tuple[list, Optional[List[str]], Optional[str]]:
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

    parser = create_parser(suppress_defaults=True)
    args = parser.parse_args(argv)
    try:
        apply_config_file(args)
    except ValueError as e:
        parser.error(str(e))
    # After parsing (the root can come from the config file), before any GPU
    # runtime initialises.
    _cache_lifetime = _configure_cache_env(  # noqa: F841 - keeps the dir alive
        args.benchmark_work_dir
    )

    # Runtime startup (in the suite runner) is the authoritative GPU check;
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
            argv=[sys.argv[0], *argv],
        )
    finally:
        for td in tmpdirs:
            td.cleanup()

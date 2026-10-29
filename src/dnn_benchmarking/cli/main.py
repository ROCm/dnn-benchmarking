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

#: Caches that only save time. Kept under --benchmark-cache-dir, else in
#: DNN_BENCH_WORKSPACE across runs. The MIOpen kernel cache holds compiled
#: binaries: Find compiles before it times a solver, so a cache hit shortens
#: Find but does not change the selection.
_WORKSPACE_CACHE_SUBDIRS = {
    "XDG_CACHE_HOME": "cache",
    "AMD_COMGR_CACHE_DIR": "comgr_cache",
    "MIOPEN_CUSTOM_CACHE_DIR": "miopen_kernel_cache",
}

#: MIOpen user find-DB and perf-DB of this process (PyTorch rows; every hipDNN
#: run has its own, see ``execution.child_runner``). It starts empty in each
#: run: MIOpen keys a record without the deterministic attribute, and a stored
#: record repeats one Find ranking in every later run. The installed system
#: DB is read-only and stays in use.
_MIOPEN_USER_DB = "MIOPEN_USER_DB_PATH"


def _set_cache_dir(var: str, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    os.environ[var] = str(path)


def _configure_cache_env(
    cache_root: Optional[Path],
) -> tempfile.TemporaryDirectory[str] | None:
    """Point ROCm/tool cache variables away from the (network) home dir.

    MIOpen, comgr and torch default to ~/.cache or ~/.miopen. The MIOpen user
    DB goes to a private per-run temporary directory. The other caches go to
    ``cache_root`` (--benchmark-cache-dir), which replaces inherited values;
    else to DNN_BENCH_WORKSPACE (set by setup_env.py), or to the same
    temporary directory without a workspace. Other variables that are already
    set are kept. Must run before any GPU runtime initialises. Keep the
    returned object alive for the run; its finalizer removes the temporary
    directory.
    """
    if cache_root is not None:
        for var, subdir in _WORKSPACE_CACHE_SUBDIRS.items():
            _set_cache_dir(var, cache_root / subdir)
    unset_workspace = [var for var in _WORKSPACE_CACHE_SUBDIRS if var not in os.environ]
    user_db_unset = _MIOPEN_USER_DB not in os.environ
    workspace = os.environ.get("DNN_BENCH_WORKSPACE")
    if not user_db_unset and (workspace or not unset_workspace):
        for var in unset_workspace:
            _set_cache_dir(var, Path(workspace) / _WORKSPACE_CACHE_SUBDIRS[var])
        return None
    lifetime = tempfile.TemporaryDirectory(prefix="dnn-bench-cache-")
    run_dir = Path(lifetime.name)
    workspace_base = Path(workspace) if workspace else run_dir
    for var in unset_workspace:
        _set_cache_dir(var, workspace_base / _WORKSPACE_CACHE_SUBDIRS[var])
    if user_db_unset:
        _set_cache_dir(_MIOPEN_USER_DB, run_dir / "miopen_user_db")
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
        args.benchmark_cache_dir
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

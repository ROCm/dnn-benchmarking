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


def _configure_cache_env(
    fast: bool, miopen_user_db: Optional[Path] = None
) -> List[tempfile.TemporaryDirectory[str]]:
    """Set the ROCm/tool cache variables; the one owner of ``MIOPEN_USER_DB_PATH``.

    MIOpen, comgr and torch default to ~/.cache or ~/.miopen. Unset variables
    point to DNN_BENCH_WORKSPACE (set by setup_env.py) or to a private
    temporary directory, allocated only when a variable is unset.

    The MIOpen user database, in order of precedence:

    1. ``miopen_user_db`` (``--miopen-user-db``): created if missing and kept
       after the run, so a later run with the same path reuses its find
       results (a warm run).
    2. With ``fast`` (``--fast``): a new, empty temporary directory, removed
       after the run, so no earlier command changes the selection.
    3. Otherwise the inherited value, or the workspace directory.

    Both 1 and 2 replace an inherited value. MIOpen reads the variable once
    per process, so every graph of the run shares the database. hipDNN's
    disk cache stays as configured (``HIPDNN_CACHE_DIR``,
    ``--hipdnn-cache-dir``).

    Must run before any GPU runtime initialises. Keep the returned objects
    alive for the run, past the hipDNN handle; their finalizers remove the
    temporary directories.
    """
    lifetimes: List[tempfile.TemporaryDirectory[str]] = []
    if miopen_user_db is not None:
        try:
            miopen_user_db.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise ValueError(f"--miopen-user-db {miopen_user_db}: {e}") from e
        os.environ["MIOPEN_USER_DB_PATH"] = str(miopen_user_db.resolve())
    elif fast:
        user_db = tempfile.TemporaryDirectory(prefix="dnn-bench-fast-db-")
        lifetimes.append(user_db)
        os.environ["MIOPEN_USER_DB_PATH"] = user_db.name
    unset = [var for var in _CACHE_ENV_SUBDIRS if var not in os.environ]
    if not unset:
        return lifetimes
    base, lifetime = _create_cache_base(os.environ.get("DNN_BENCH_WORKSPACE"))
    if lifetime is not None:
        lifetimes.append(lifetime)
    for var in unset:
        path = base / _CACHE_ENV_SUBDIRS[var]
        path.mkdir(parents=True, exist_ok=True)
        os.environ[var] = str(path)
    return lifetimes


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
    # After parsing (--fast and --miopen-user-db can come from the config
    # file), before any GPU runtime initialises.
    try:
        _cache_lifetimes = _configure_cache_env(  # noqa: F841 - keeps dirs alive
            fast=bool(args.fast), miopen_user_db=args.miopen_user_db
        )
    except ValueError as e:
        parser.error(str(e))

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

# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Main entry point for dnn-benchmark CLI."""

import os
import sys
import tempfile
from pathlib import Path
from typing import List, Optional, Sequence

from ..common.exceptions import GraphLoadError
from ..common.work_dir import (
    COMPILE_CACHE_SUBDIRS,
    OOTB_TEMP_SUBDIR,
    SelectionState,
    tuned_state,
    uses_tuned_state,
    work_dir,
)
from .config_file import apply_config_file
from .parser import create_parser


def _configure_cache_env(
    work: Path, *, tuned: bool, replace_compile_caches: bool
) -> List[tempfile.TemporaryDirectory[str]]:
    """Point the caches and the selection state into the work directory.

    The one owner of ``HIPDNN_CACHE_DIR`` and ``MIOPEN_USER_DB_PATH``
    (``common.work_dir``):

    - Compile caches go to ``<work>/cache``. They fill unset variables, and
      replace inherited values when ``replace_compile_caches``
      (``--benchmark-work-dir`` given).
    - With ``tuned``, the selection state is ``<work>/tuned``, kept across
      commands. Otherwise it is a new ``<work>/ootb/temp/run-*`` directory,
      empty and removed when the command ends. Both replace inherited values.

    MIOpen reads ``MIOPEN_USER_DB_PATH`` once per process, so every graph of
    the run shares the state. Must run before any GPU runtime initialises.
    Keep the returned objects alive for the run, past the hipDNN handle;
    their finalizers remove the run directory.

    Raises:
        ValueError: A directory under ``work`` cannot be created.
    """
    lifetimes: List[tempfile.TemporaryDirectory[str]] = []
    try:
        for var, subdir in COMPILE_CACHE_SUBDIRS.items():
            if replace_compile_caches or var not in os.environ:
                path = work / "cache" / subdir
                path.mkdir(parents=True, exist_ok=True)
                os.environ[var] = str(path)
        if tuned:
            state = tuned_state(work).create()
        else:
            temp = work / OOTB_TEMP_SUBDIR
            temp.mkdir(parents=True, exist_ok=True)
            run_dir = tempfile.TemporaryDirectory(prefix="run-", dir=temp)
            lifetimes.append(run_dir)
            state = SelectionState.under(Path(run_dir.name)).create()
    except OSError as e:
        raise ValueError(f"benchmark work dir {work}: {e}") from e
    os.environ.update(state.env())
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


def main(argv: Optional[Sequence[str]] = None, *, tuned_child: bool = False) -> int:
    """CLI entry point.

    Args:
        argv: Command line without the program name (default ``sys.argv``).
        tuned_child: Set by ``cli.pytorch_tuned_child``: the run is the tuned
            run of an ``--oracle`` row and uses the tuned selection state.
    """
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
    # After parsing (--fast, --autotune and --benchmark-work-dir can come from
    # the config file), before any GPU runtime initialises. A profiling pass
    # always starts with a new, empty OOTB state.
    tuned = not args.internal_profiling_run and (
        tuned_child or uses_tuned_state(bool(args.fast), bool(args.autotune))
    )
    work_arg = str(args.benchmark_work_dir) if args.benchmark_work_dir else None
    try:
        _cache_lifetimes = _configure_cache_env(  # noqa: F841 - keeps dirs alive
            work_dir(work_arg),
            tuned=tuned,
            replace_compile_caches=work_arg is not None,
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

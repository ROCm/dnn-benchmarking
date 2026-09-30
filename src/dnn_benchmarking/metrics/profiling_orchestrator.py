# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Re-exec orchestrator for opt-in profiling sources.

When the user passes ``--pmc``, ``--emit-trace``, ``--perf``, or
``--roofline``, the timed pass runs first to keep its numbers clean.
After it succeeds, this orchestrator runs the workload again — once per
requested source — under the corresponding external profiler (rocprofv3,
perf, rocprof-compute). The results are merged into a single dict that
populates ``ProviderEngineResult.extra_metrics``.

The child is ``python -m dnn_benchmarking --internal-profiling-run``
for a single (graph, engine) pair (see :func:`build_inner_argv`); it
carries no profiling flag, so it cannot recurse. It primes, then runs
``iters`` timed iterations with a warm cache.

Failures of individual sources never raise — each module returns a dict
slice with a ``skipped`` / ``error_tail`` / ``warnings`` key, and a
warning naming the graph and engine goes to stderr.
"""

import hashlib
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config.benchmark_config import MetricsConfig
from ._diagnostic import warn_once
from ._tool_resolver import resolve_rocm_tool
from . import perf as _perf_mod
from . import rocprof_pmc as _pmc_mod
from . import rocprof_trace as _trace_mod
from . import roofline as _roofline_mod

# Timed iterations in the profiled child unless the caller asks for more.
# Enough dispatches to see the engine kernel repeat under PMC, few enough
# that trace/roofline replay stays cheap.
PROFILING_ITERS = 5


def resolve_output_dir(metrics_config: MetricsConfig) -> Path:
    """Pick the root profiling-output directory, defaulting to a UTC stamp.

    **Mutates the shared MetricsConfig instance**: on first call with
    ``profiling_output_dir is None``, this writes the resolved path
    back into the config so every subsequent (graph, engine) pair in
    the same suite lands under the same root. Without this, each engine
    would generate its own timestamped directory and per-suite output
    would stop being a single browsable tree.

    Callers that need a fresh resolution per call (e.g. parallel suites
    sharing a config) must clone the MetricsConfig first — today's
    sequential suite runner is the only caller and shares one config
    intentionally.
    """
    if metrics_config.profiling_output_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
        metrics_config.profiling_output_dir = Path("profiling-output") / stamp
    metrics_config.profiling_output_dir.mkdir(parents=True, exist_ok=True)
    return metrics_config.profiling_output_dir


# Characters that are safe in path segments across Linux filesystems
# without quoting at the shell. Anything else (slash, space, colon,
# brackets, $, &, ...) gets collapsed to '_' so a future plugin that
# returns an awkward engine name can't break the artifact tree or
# require the user to shell-quote artifact paths.
_PATH_SAFE_RE = re.compile(r"[^A-Za-z0-9._-]")


def _safe_segment(name: str) -> str:
    return _PATH_SAFE_RE.sub("_", name) or "unnamed"


def _graph_anchor(graph_path: Path) -> str:
    """The canonical string used both for the hash and the ``.source`` file.

    Resolves to an absolute path so it's stable across runs (and useful
    on its own when read back from ``.source``). Falls back to the
    string form when ``resolve()`` errors — typically a test path that
    doesn't exist on disk.
    """
    try:
        return str(graph_path.resolve())
    except OSError:
        return str(graph_path)


def _graph_segment(graph_path: Path) -> str:
    """``<stem>-<6 hex>`` disambiguator for the per-graph subdir.

    Suite mode accepts directories and globs, so two graphs with the same
    file stem (``a/conv.json`` and ``b/conv.json``) would otherwise
    collide on ``<root>/conv/<engine>/<source>/results.db`` and silently
    overwrite each other. The hash is derived from the resolved absolute
    path so it's stable across runs and distinguishes two graphs that
    share a stem.

    Hash length is 6 hex chars (24 bits) — collision probability across a
    realistic suite (<10⁴ graphs) is negligible, and 6 chars keeps the
    segment short enough to type when chasing an artifact path. The
    hash is opaque on its own; ``_subdir`` writes a ``.source`` file in
    the graph dir so users can map it back without re-hashing.
    """
    digest = hashlib.sha256(_graph_anchor(graph_path).encode("utf-8")).hexdigest()[:6]
    return f"{_safe_segment(graph_path.stem)}-{digest}"


def build_inner_argv(
    graph_path: Path,
    engine_id: int,
    seed: int,
    warmup_iters: int,
    iters: int,
    plugin_path: Optional[Path],
) -> List[str]:
    """Argv for the ``--internal-profiling-run`` child (frozen contract).

    ``-m dnn_benchmarking --internal-profiling-run --graph G --engine E
    --warmup W --iters I --seed S [--plugin-path P]``. No profiling flag
    is ever forwarded, so the child cannot recurse.
    """
    argv = [
        sys.executable,
        "-m",
        "dnn_benchmarking",
        "--internal-profiling-run",
        "--graph",
        str(graph_path),
        "--engine",
        str(engine_id),
        "--warmup",
        str(warmup_iters),
        "--iters",
        str(iters),
        "--seed",
        str(seed),
    ]
    if plugin_path is not None:
        argv += ["--plugin-path", str(plugin_path)]
    return argv


def check_requested_tools(metrics_config: MetricsConfig) -> List[str]:
    """One message per requested profiling pass whose tool cannot be found.

    Called once at startup; the CLI exits 2 when the list is non-empty so
    a missing profiler fails fast instead of skipping every engine.
    """
    missing: List[str] = []
    rocprof_flags = [
        flag
        for flag, on in (
            ("--pmc", metrics_config.pmc_set is not None),
            ("--emit-trace", metrics_config.emit_trace is not None),
        )
        if on
    ]
    if rocprof_flags and resolve_rocm_tool("rocprofv3") is None:
        missing.append(
            f"{'/'.join(rocprof_flags)} requires rocprofv3, which was not found "
            "(rocm-sdk wheel shim, $ROCM_PATH/bin or PATH)"
        )
    if metrics_config.perf and _perf_mod._resolve_perf() is None:
        missing.append(
            "--perf requires a runnable `perf` binary (linux-tools); none found"
        )
    if metrics_config.roofline and resolve_rocm_tool("rocprof-compute") is None:
        missing.append(
            "--roofline requires rocprof-compute, which was not found "
            "(rocm-sdk wheel shim, $ROCM_PATH/bin or PATH)"
        )
    return missing


def _subdir(out_dir: Path, graph_path: Path, engine_name: str, source: str) -> Path:
    """Per-source output directory:
    ``<out_dir>/<graph_stem>-<hash6>/<engine_name>/<source>/``.

    Three semantic levels under the user-controlled root: graph, then
    engine, then profiling source. Replaces the legacy flat scheme
    ``<graph>_<engine_id>_<source>/`` whose engine_id segment was a
    19-digit hash that no one could read or type.

    The graph segment carries a 6-hex disambiguator (see
    ``_graph_segment``) so same-stem graphs from different directories
    don't collide; engine name is sanitised because a future plugin
    could return slashes/spaces.
    """
    graph_dir = out_dir / _graph_segment(graph_path)
    sub = graph_dir / _safe_segment(engine_name) / _safe_segment(source)
    sub.mkdir(parents=True, exist_ok=True)
    # Drop a `.source` file at the graph-segment level so
    # `cat conv-7a3f1c/.source` answers "which graph is this?" without
    # the user having to recompute the hash. Idempotent — multiple
    # (engine, source) calls for the same graph overwrite with
    # identical content.
    (graph_dir / ".source").write_text(_graph_anchor(graph_path) + "\n")
    return sub


def run_profiling_passes(
    graph_path: Path,
    engine_id: int,
    engine_name: str,
    seed: int,
    warmup_iters: int,
    metrics_config: MetricsConfig,
    plugin_path: Optional[Path],
    iters: int = PROFILING_ITERS,
    out_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run every requested profiling source. Returns a merged dict.

    Source slices are merged at the top level so consumers can address
    them via ``extra_metrics["pmc"]``, ``extra_metrics["trace"]`` etc.

    Args:
        graph_path: Graph file passed to the child.
        engine_id: Single engine ID for the child.
        engine_name: Human-readable engine name (e.g. ``"MIOPEN_ENGINE"``),
            used for the per-engine output subdirectory and warnings.
        seed: Input seed, forwarded so counters see the timed pass's inputs.
        warmup_iters: Child warmup iteration count.
        metrics_config: Decides which sources fire.
        plugin_path: Optional plugin path forwarded to the child.
        iters: Child timed iteration count.
        out_dir: Override the resolved profiling-output root (test hook).

    Never raises. Source-specific failures end up in their slice's
    ``skipped`` / ``error_tail`` / ``warnings`` keys.
    """
    if not metrics_config.opt_in_pass_requested:
        return {}

    if out_dir is None:
        out_dir = resolve_output_dir(metrics_config)
    inner_argv = build_inner_argv(
        graph_path, engine_id, seed, warmup_iters, iters, plugin_path
    )
    context = f"{graph_path.stem}/{engine_name}"
    sources = (
        # (requested, slice key, module, subdir, extra kwargs)
        (
            metrics_config.pmc_set is not None,
            "pmc",
            _pmc_mod,
            f"pmc_{metrics_config.pmc_set}",
            {"pmc_set": metrics_config.pmc_set},
        ),
        (
            metrics_config.emit_trace is not None,
            "trace",
            _trace_mod,
            f"trace_{metrics_config.emit_trace}",
            {},
        ),
        (metrics_config.perf, "perf", _perf_mod, "perf", {}),
        (metrics_config.roofline, "roofline", _roofline_mod, "roofline", {}),
    )

    aggregated: Dict[str, Any] = {}
    for requested, key, module, subdir, extra in sources:
        if not requested:
            continue
        try:
            aggregated.update(
                module.run(
                    inner_argv=inner_argv,
                    out_dir=_subdir(out_dir, graph_path, engine_name, subdir),
                    timeout_s=metrics_config.profiling_timeout_s,
                    context=context,
                    **extra,
                )
            )
        except Exception as e:
            warn_once(key, f"{context}: unexpected error in {key} pass: {e}")
            aggregated.setdefault(key, {})["unexpected_error"] = str(e)
    return aggregated

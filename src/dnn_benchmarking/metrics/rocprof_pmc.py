# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""rocprofv3 PMC counter collection.

Re-runs the workload under ``rocprofv3 --pmc <counters>`` and parses
the resulting rocpd SQLite database into per-kernel counter means.

Results are reported per kernel only. The profiled child also launches
input-fill and warmup dispatches, so a sum or mean across every kernel
would mix unrelated work; ``per_kernel`` keeps each kernel separate and
records its dispatch count so the engine's kernel is identifiable.

Counter sets are hardcoded per GPU family and small enough to fit a
single-pass replay. The ``"all"`` set unions every group and is gated by
``MetricsConfig.pmc_allow_multipass`` because rocprofv3's multi-pass
replay has been observed to hang for minutes on sub-second workloads.
Counter availability is not pre-validated: an unknown counter makes
rocprofv3 fail, and its stderr tail lands in ``error_tail``.

The CDNA sets are verified against rocprofiler-sdk's
``basic_counters.xml`` / ``derived_counters.xml`` plus a single-pass
``rocprofv3 --pmc`` run on gfx90a. The fallback set is conservative and
known-good across every supported arch.
"""

import sqlite3
from pathlib import Path
from typing import Any, Dict, List

from ._artifact_paths import find_first, flatten_hostname_dir
from ._diagnostic import warn_once
from ._subprocess import run_tool
from ._tool_resolver import resolve_rocm_tool
from .arch import detect_arch

_CDNA_SETS: Dict[str, List[str]] = {
    "basic": [
        "GRBM_GUI_ACTIVE",
        "SQ_WAVES",
        "SQ_INSTS_VALU",
        "SQ_BUSY_CYCLES",
    ],
    "memory": [
        "TCC_HIT_sum",
        "TCC_MISS_sum",
        "TCP_TCC_READ_REQ_sum",
        "TCC_EA_RDREQ_sum",
    ],
    "flops": [
        "SQ_INSTS_VALU_MFMA_F16",
        "SQ_INSTS_VALU_MFMA_BF16",
        "SQ_INSTS_VALU_MFMA_F32",
    ],
}

_FALLBACK_SETS: Dict[str, List[str]] = {
    "basic": ["GRBM_GUI_ACTIVE", "SQ_WAVES"],
}

# Arches whose counters match the CDNA table (MI200, MI300).
_CDNA_ARCHES = frozenset({"gfx90a", "gfx942"})


def _counter_table(arch: str) -> Dict[str, List[str]]:
    return _CDNA_SETS if arch in _CDNA_ARCHES else _FALLBACK_SETS


def _resolve_counter_groups(arch: str, pmc_set: str) -> List[List[str]]:
    """Return one counter group per intended rocprofv3 pass.

    rocprofv3 expresses multipass via repeated ``--pmc`` flags, one
    group per pass. Named sets return exactly one group; ``all`` returns
    one group per table entry. Empty list means nothing to collect.
    """
    table = _counter_table(arch)
    if pmc_set == "all":
        return [list(group) for group in table.values()]
    group = table.get(pmc_set)
    return [list(group)] if group else []


def _build_argv(
    counter_groups: List[List[str]],
    out_dir: Path,
    inner_argv: List[str],
) -> List[str]:
    """rocprofv3 arguments (without the binary): one ``--pmc`` per group.

    The output format is pinned to rocpd because the parser reads the
    SQLite db and other rocprofv3 builds default to CSV. ``-o results``
    drops the ``<pid>_`` filename prefix; ``flatten_hostname_dir`` then
    hoists any ``<hostname>/`` segment.
    """
    args: List[str] = []
    for group in counter_groups:
        args += ["--pmc", *group]
    return args + [
        "--output-format",
        "rocpd",
        "-d",
        str(out_dir),
        "-o",
        "results",
        "--",
        *inner_argv,
    ]


def _quote_sqlite_identifier(identifier: str) -> str:
    """Return a SQLite identifier quoted against interpolation attacks."""
    return '"' + identifier.replace('"', '""') + '"'


def _l2_hit_rate(counters: Dict[str, float]) -> Any:
    hit = next((v for k, v in counters.items() if k.startswith("TCC_HIT")), None)
    miss = next((v for k, v in counters.items() if k.startswith("TCC_MISS")), None)
    if hit is None or miss is None or hit + miss <= 0:
        return None
    return hit / (hit + miss)


def _parse_rocpd_db(db_path: Path) -> Dict[str, Any]:
    """Per-kernel counter means (per dispatch) and dispatch counts.

    rocpd table names carry a per-run uuid suffix, so they are found via
    ``sqlite_master``. Join keys (rocprofv3 1.2.2 schema):
    ``pmc_event.event_id`` -> ``kernel_dispatch.dispatch_id``;
    ``kernel_dispatch.kernel_id`` -> ``info_kernel_symbol.id`` (which
    carries ``kernel_name``).
    """
    # Plain path, not a `file:` URI, so `?`/`#`/`%` in the output dir are
    # not parsed as URI syntax; query_only enforces read-only.
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA query_only = ON")
    try:
        tables = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }

        def find(prefix: str) -> Any:
            return next((t for t in tables if t.startswith(prefix)), None)

        pmc_event_table = find("rocpd_pmc_event")
        kernel_table = find("rocpd_kernel_dispatch")
        symbol_table = find("rocpd_info_kernel_symbol")
        info_pmc_table = find("rocpd_info_pmc")
        if pmc_event_table is None or kernel_table is None or symbol_table is None:
            return {
                "warnings": [
                    "rocpd db missing pmc_event, kernel_dispatch or "
                    f"info_kernel_symbol table; present tables: {sorted(tables)}"
                ]
            }

        id_to_name: Dict[int, str] = {}
        if info_pmc_table is not None:
            for pmc_id, name in conn.execute(
                f"SELECT id, name FROM {_quote_sqlite_identifier(info_pmc_table)}"  # nosec B608
            ):
                id_to_name[int(pmc_id)] = str(name)

        per_kernel: Dict[str, Dict[str, Any]] = {}
        # Table identifiers come from sqlite_master and are quoted.
        rows = conn.execute(  # nosec B608
            f"""
            SELECT sym.kernel_name, p.pmc_id,
                   SUM(p.value), COUNT(DISTINCT p.event_id)
            FROM {_quote_sqlite_identifier(pmc_event_table)} p
            JOIN {_quote_sqlite_identifier(kernel_table)} k  ON p.event_id = k.dispatch_id
            JOIN {_quote_sqlite_identifier(symbol_table)} sym ON k.kernel_id = sym.id
            GROUP BY sym.kernel_name, p.pmc_id
            """  # nosec B608
        )
        for kname, pmc_id, sum_v, dispatches in rows:
            entry = per_kernel.setdefault(str(kname), {"dispatches": 0, "counters": {}})
            counter = id_to_name.get(int(pmc_id), f"pmc_id_{pmc_id}")
            entry["counters"][counter] = float(sum_v) / dispatches
            entry["dispatches"] = max(entry["dispatches"], int(dispatches))
        for entry in per_kernel.values():
            rate = _l2_hit_rate(entry["counters"])
            if rate is not None:
                entry["l2_hit_rate"] = rate
        return {"per_kernel": per_kernel}
    finally:
        conn.close()


def run(
    inner_argv: List[str],
    out_dir: Path,
    timeout_s: int,
    context: str,
    pmc_set: str,
) -> Dict[str, Any]:
    """Run rocprofv3 PMC collection and return the ``{"pmc": ...}`` slice.

    ``timeout_s`` of 0 disables the cap; ``context`` (graph/engine)
    labels warnings. Never raises.
    """
    arch = detect_arch()
    result: Dict[str, Any] = {"set": pmc_set, "arch": arch}
    counter_groups = _resolve_counter_groups(arch, pmc_set)
    if not counter_groups:
        warn_once("rocprof_pmc", f"no counters defined for arch={arch} set={pmc_set}")
        result["skipped"] = "no counters defined"
        return {"pmc": result}
    result["counters_requested"] = list(
        dict.fromkeys(c for group in counter_groups for c in group)
    )
    if pmc_set == "all" and arch not in _CDNA_ARCHES:
        # The user paid the multipass opt-in expecting a union; say why
        # the result is small.
        warn_once(
            "rocprof_pmc",
            f"arch '{arch}' has no PMC table; --pmc all narrowed to fallback basic",
        )
        result["arch_narrowed_to_fallback"] = True

    _, fields = run_tool(
        "rocprof_pmc",
        resolve_rocm_tool("rocprofv3"),
        _build_argv(counter_groups, out_dir, inner_argv),
        out_dir,
        timeout_s,
        context,
    )
    result.update(fields)
    # Hoist even on failure so partial output is reachable at a stable path.
    flatten_hostname_dir(out_dir)
    if fields:
        return {"pmc": result}

    db_path = find_first(out_dir, "*.db")
    if db_path is None:
        warn_once("rocprof_pmc", f"{context}: rocprofv3 produced no .db file")
        result["warnings"] = ["no .db file found in output directory"]
        return {"pmc": result}
    result["db_path"] = str(db_path)
    try:
        result.update(_parse_rocpd_db(db_path))
    except sqlite3.Error as e:
        warn_once("rocprof_pmc", f"{context}: rocpd db parse failed: {e}")
        result["warnings"] = [f"rocpd db parse failed: {e}"]
    return {"pmc": result}

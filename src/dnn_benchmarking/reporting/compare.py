# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""``dnn-benchmark compare A.json B.json``: compare two result files.

Convention: ``speedup = A_median / B_median`` is "B speedup vs A"; above 1
means B is faster. A pair is ``within noise`` when its relative change is
at most ``max(threshold, 2 * sqrt(cv_A**2 + cv_B**2))``; a regression is B
slower than that. Failed/error rows never enter the geomean.

Exit codes: 0 no regression, 1 regression beyond threshold, 2 usage error,
unreadable or incompatible input.
"""

import argparse
import csv
import json
import math
import sys
import textwrap
from dataclasses import asdict, dataclass
from statistics import geometric_mean
from typing import Any, Dict, List, Optional, Tuple

from .reporter import _clip, _fmt_time, _width
from .suite_results import SuiteResult

CONVENTION = "speedup = A_median / B_median (B speedup vs A; >1 means B is faster)"
_USABLE = ("passed", "unchecked", "reference")


@dataclass
class Pair:
    """One compared (graph[, engine]) pair; ms/cv are None when unusable."""

    graph: str
    engine_a: Optional[str]
    engine_b: Optional[str]
    a_ms: Optional[float]
    b_ms: Optional[float]
    a_cv: Optional[float]
    b_cv: Optional[float]
    speedup: Optional[float]
    label: str
    in_geomean: bool


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="dnn-benchmark compare",
        description=f"Compare two result JSON files. {CONVENTION}.",
    )
    p.add_argument("a", metavar="A.json")
    p.add_argument("b", metavar="B.json")
    p.add_argument(
        "--by",
        choices=("best", "ref", "engine"),
        default="best",
        help="best: fastest usable engine per graph; ref: reference rows; "
        "engine: same engine in both files (default: best)",
    )
    p.add_argument("--metric", choices=("kernel", "host"), default="kernel")
    p.add_argument(
        "--threshold",
        type=float,
        default=5.0,
        metavar="PCT",
        help="regression threshold in percent (default: 5)",
    )
    p.add_argument(
        "--allow-mismatch",
        action="store_true",
        help="compare even when run.config.cache_mode differs",
    )
    fmt = p.add_mutually_exclusive_group()
    fmt.add_argument("--csv", action="store_true", help="CSV on stdout")
    fmt.add_argument("--json", action="store_true", help="JSON on stdout")
    return p


def _engine_label(row: Dict[str, Any]) -> str:
    e = row["engine"]
    return e["name"] or e["id"] or row["provider"]


def _stat(row: Dict[str, Any], metric: str) -> Tuple[Optional[float], Optional[float]]:
    s = row[metric]
    if row["verdict"] in ("error", "skipped") or not s:
        return None, None
    return s["median_ms"], s["cv"]


def _pick(graph: Dict[str, Any], by: str, metric: str) -> Optional[Dict[str, Any]]:
    """Row to compare for ``best``/``ref``: fastest usable row of that role."""
    role = "reference" if by == "ref" else "engine"
    usable = [
        r
        for r in graph["results"]
        if r["role"] == role and r["verdict"] in _USABLE and r[metric]
    ]
    return min(usable, key=lambda r: r[metric]["median_ms"], default=None)


def _row_key(row: Dict[str, Any]) -> Tuple[Any, ...]:
    return (row["role"], row["provider"], row["engine"]["id"], row["engine"]["name"])


def _pair(
    graph: str,
    ra: Optional[Dict[str, Any]],
    rb: Optional[Dict[str, Any]],
    metric: str,
    threshold: float,
    kind: str = "engine",
) -> Pair:
    a_ms, a_cv = _stat(ra, metric) if ra else (None, None)
    b_ms, b_cv = _stat(rb, metric) if rb else (None, None)
    pair = Pair(
        graph=graph,
        engine_a=_engine_label(ra) if ra else None,
        engine_b=_engine_label(rb) if rb else None,
        a_ms=a_ms,
        b_ms=b_ms,
        a_cv=a_cv,
        b_cv=b_cv,
        speedup=None,
        label="",
        in_geomean=False,
    )
    if ra is None and rb is None:
        pair.label = f"no {kind} row in either"
    for side, row, ms in (("A", ra, a_ms), ("B", rb, b_ms)):
        if row is None:
            pair.label = pair.label or f"no {side} row"
        elif ms is None or ms <= 0:
            pair.label = pair.label or f"{side} {row['verdict']}"
    if pair.label:
        return pair
    pair.speedup = a_ms / b_ms
    failed = [s for s, r in (("A", ra), ("B", rb)) if r["verdict"] == "failed"]
    if failed:
        pair.label = f"{'+'.join(failed)} failed"
        return pair
    pair.in_geomean = True
    band = max(threshold / 100.0, 2.0 * math.hypot(a_cv or 0.0, b_cv or 0.0))
    change = b_ms / a_ms - 1.0  # positive = B slower
    if abs(change) <= band:
        pair.label = "within noise"
    else:
        pair.label = "REGRESSION" if change > 0 else "faster"
    return pair


def _match_graphs(
    a: Dict[str, Any], b: Dict[str, Any]
) -> Tuple[List[Tuple[Dict[str, Any], Dict[str, Any]]], List[str], List[str]]:
    """Join on graph_id (null only for graphs that failed to load: no rows)."""
    by_id = {g["graph_id"]: g for g in b["graphs"] if g["graph_id"]}
    matched, only_a, used = [], [], set()
    for ga in a["graphs"]:
        gb = by_id.get(ga["graph_id"]) if ga["graph_id"] else None
        if gb is None or id(gb) in used:
            only_a.append(ga["graph_name"])
            continue
        used.add(id(gb))
        matched.append((ga, gb))
    only_b = [g["graph_name"] for g in b["graphs"] if id(g) not in used]
    return matched, only_a, only_b


def compare(
    a: Dict[str, Any],
    b: Dict[str, Any],
    *,
    by: str = "best",
    metric: str = "kernel",
    threshold: float = 5.0,
) -> Dict[str, Any]:
    """Compare two loaded v2 documents; returns the JSON-shaped report."""
    matched, only_a, only_b = _match_graphs(a, b)
    pairs: List[Pair] = []
    for ga, gb in matched:
        name = ga["graph_name"]
        if by != "engine":
            ra, rb = _pick(ga, by, metric), _pick(gb, by, metric)
            kind = "ref" if by == "ref" else "engine"
            pairs.append(_pair(name, ra, rb, metric, threshold, kind))
            continue
        rows_b = {_row_key(r): r for r in gb["results"]}
        for ra in ga["results"]:
            pairs.append(_pair(name, ra, rows_b.pop(_row_key(ra), None), metric, threshold))
        pairs.extend(_pair(name, None, rb, metric, threshold) for rb in rows_b.values())
    ratios = [p.speedup for p in pairs if p.in_geomean]
    return {
        "convention": CONVENTION,
        "by": by,
        "metric": metric,
        "threshold_pct": threshold,
        "pairs": [asdict(p) for p in pairs],
        "only_in_a": only_a,
        "only_in_b": only_b,
        "geomean_speedup": geometric_mean(ratios) if ratios else None,
        "regressions": sum(p.label == "REGRESSION" for p in pairs),
    }


def _config_warnings(a: Dict[str, Any], b: Dict[str, Any]) -> List[str]:
    ca, cb = a["run"]["config"], b["run"]["config"]
    return [
        f"run.config.{k} differs: A={ca.get(k)!r} B={cb.get(k)!r}"
        for k in sorted(set(ca) | set(cb))
        if ca.get(k) != cb.get(k)
    ]


def _time(v: Optional[float]) -> str:
    return "-" if v is None else _fmt_time(v)


def _describe(label: str, path: str, doc: Dict[str, Any]) -> str:
    env, cfg = doc["environment"], doc["run"]["config"]
    return (
        f"{label}: {path}  {env.get('gpu_model') or '?'} {env.get('gpu_arch') or '?'}"
        f"  backend={cfg.get('backend')} cache={cfg.get('cache_mode')}"
    )


def _print_table(report: Dict[str, Any], a: Tuple[str, Dict], b: Tuple[str, Dict]) -> None:
    out, width = sys.stdout, _width()
    print(_clip(_describe("A", *a), width), file=out)
    print(_clip(_describe("B", *b), width), file=out)
    heading = f"{CONVENTION}; metric={report['metric']} median; threshold {report['threshold_pct']:g}%"
    for line in textwrap.wrap(heading, width):
        print(line, file=out)
    pairs = report["pairs"]
    # (header, right-aligned, cells)
    columns = [
        ("graph", False, [p["graph"] for p in pairs]),
        ("A engine", False, [p["engine_a"] or "-" for p in pairs]),
        ("A time", True, [_time(p["a_ms"]) for p in pairs]),
        ("B engine", False, [p["engine_b"] or "-" for p in pairs]),
        ("B time", True, [_time(p["b_ms"]) for p in pairs]),
        ("speedup", True, ["-" if p["speedup"] is None else f"{p['speedup']:.2f}x" for p in pairs]),
        ("note", False, [p["label"] for p in pairs]),
    ]
    widths = [max(len(h), *(len(c) for c in cells)) for h, _, cells in columns]
    # Graph and engine names share the squeeze; narrowest first so a short
    # column hands its unused share to the others.
    flex = sorted((0, 1, 3), key=lambda i: widths[i])
    avail = width - 2 * (len(columns) - 1) - sum(
        w for i, w in enumerate(widths) if i not in flex
    )
    for n, i in enumerate(flex):
        widths[i] = min(widths[i], max(8, avail // (len(flex) - n)))
        avail -= widths[i]

    def render(cells: List[str]) -> str:
        parts = []
        for (_, right, _), w, cell in zip(columns, widths, cells):
            cell = _clip(cell, w)
            parts.append(cell.rjust(w) if right else cell.ljust(w))
        return _clip("  ".join(parts).rstrip(), width)

    print(render([h for h, _, _ in columns]), file=out)
    for i in range(len(pairs)):
        print(render([cells[i] for _, _, cells in columns]), file=out)
    for side, names in (("A", report["only_in_a"]), ("B", report["only_in_b"])):
        for name in names:
            print(_clip(f"graph only in {side}: {name}", width), file=out)
    geo = report["geomean_speedup"]
    n = sum(p["in_geomean"] for p in pairs)
    footer = (
        f"{len(pairs)} pairs, {n} in geomean; geomean B speedup vs A: "
        f"{'-' if geo is None else f'{geo:.3f}x'}; "
        f"{report['regressions']} regression(s); "
        f"graphs only in A: {len(report['only_in_a'])}, "
        f"graphs only in B: {len(report['only_in_b'])}"
    )
    for line in textwrap.wrap(footer, width):
        print(line, file=out)


def main(argv: List[str]) -> int:
    """Entry point for ``dnn-benchmark compare``; returns the exit code."""
    try:
        args = _parser().parse_args(argv)
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else 2
    try:
        a, b = SuiteResult.load(args.a), SuiteResult.load(args.b)
    except (OSError, ValueError, KeyError, TypeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    for w in _config_warnings(a, b):
        print(f"warning: {w}", file=sys.stderr)
    if a["run"]["config"].get("cache_mode") != b["run"]["config"].get("cache_mode"):
        if not args.allow_mismatch:
            print(
                "error: cache_mode differs between A and B; timings are not "
                "comparable (pass --allow-mismatch to compare anyway)",
                file=sys.stderr,
            )
            return 2
    report = compare(a, b, by=args.by, metric=args.metric, threshold=args.threshold)
    if args.json:
        json.dump(report, sys.stdout, indent=2, allow_nan=False)
        print()
    elif args.csv:
        writer = csv.DictWriter(sys.stdout, fieldnames=list(Pair.__dataclass_fields__))
        writer.writeheader()
        writer.writerows(report["pairs"])
    else:
        _print_table(report, (args.a, a), (args.b, b))
    return 1 if report["regressions"] else 0

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
from dataclasses import asdict, dataclass
from statistics import geometric_mean
from typing import Any, Dict, List, Optional, Tuple

from .suite_results import SuiteResult

CONVENTION = "speedup = A_median / B_median (B speedup vs A; >1 means B is faster)"
_USABLE = ("passed", "unchecked", "reference")
_WIDTHS = (31, 20, 9, 20, 9, 9)  # graph, A engine, A ms, B engine, B ms, speedup
_MAX_WIDTH = 120


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
    """Join on graph_id; graph_name only when either side has no graph_id."""
    by_id = {g["graph_id"]: g for g in b["graphs"] if g["graph_id"]}
    by_name = {g["graph_name"]: g for g in b["graphs"]}
    matched, only_a, used = [], [], set()
    for ga in a["graphs"]:
        gb = by_id.get(ga["graph_id"]) if ga["graph_id"] else None
        if gb is None:
            cand = by_name.get(ga["graph_name"])
            if cand is not None and not (ga["graph_id"] and cand["graph_id"]):
                gb = cand
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
            pairs.append(_pair(name, ra, rb, metric, threshold))
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


def _cut(text: Optional[str], width: int) -> str:
    text = text or "-"
    return text if len(text) <= width else text[: width - 1] + "~"


def _ms(v: Optional[float]) -> str:
    return "-" if v is None else f"{v:.4g}"


def _describe(label: str, path: str, doc: Dict[str, Any]) -> str:
    env, cfg = doc["environment"], doc["run"]["config"]
    return (
        f"{label}: {path}  {env.get('gpu_model') or '?'} {env.get('gpu_arch') or '?'}"
        f"  backend={cfg.get('backend')} cache={cfg.get('cache_mode')}"
    )


def _print_table(report: Dict[str, Any], a: Tuple[str, Dict], b: Tuple[str, Dict]) -> None:
    out = sys.stdout
    print(_cut(_describe("A", *a), _MAX_WIDTH), file=out)
    print(_cut(_describe("B", *b), _MAX_WIDTH), file=out)
    print(
        f"{CONVENTION}; metric={report['metric']} median; "
        f"threshold {report['threshold_pct']:g}%",
        file=out,
    )
    g, e, m, _, _, s = _WIDTHS
    header = (
        f"{'graph':<{g}} {'A engine':<{e}} {'A ms':>{m}} "
        f"{'B engine':<{e}} {'B ms':>{m}} {'speedup':>{s}}  note"
    )
    print(header, file=out)
    for p in report["pairs"]:
        speed = "-" if p["speedup"] is None else f"{p['speedup']:.2f}x"
        print(
            f"{_cut(p['graph'], g):<{g}} {_cut(p['engine_a'], e):<{e}} "
            f"{_ms(p['a_ms']):>{m}} {_cut(p['engine_b'], e):<{e}} "
            f"{_ms(p['b_ms']):>{m}} {speed:>{s}}  {_cut(p['label'], 12)}",
            file=out,
        )
    for side, names in (("A", report["only_in_a"]), ("B", report["only_in_b"])):
        for name in names:
            print(_cut(f"only in {side}: {name}", _MAX_WIDTH), file=out)
    geo = report["geomean_speedup"]
    n = sum(p["in_geomean"] for p in report["pairs"])
    print(
        f"{len(report['pairs'])} pairs, {n} in geomean; geomean B speedup vs A: "
        f"{'-' if geo is None else f'{geo:.3f}x'}; "
        f"{report['regressions']} regression(s); "
        f"only in A: {len(report['only_in_a'])}, only in B: {len(report['only_in_b'])}",
        file=out,
    )


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

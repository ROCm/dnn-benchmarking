# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for ``dnn-benchmark compare`` on small synthetic result files."""

import json

from dnn_benchmarking.reporting.compare import main
from dnn_benchmarking.reporting.statistics import BenchmarkStats
from dnn_benchmarking.reporting.suite_results import (
    CorrectnessResult,
    GraphResult,
    ProviderEngineResult,
    RunInfo,
    SuiteResult,
)


def _stats(median: float, cv: float = 0.0) -> BenchmarkStats:
    # Two-point samples: exact median, CV ~= cv.
    d = median * cv
    return BenchmarkStats.from_timings([median - d, median + d] * 10)


def _row(name, median, *, cv=0.0, match=True, status="success", role="engine"):
    return ProviderEngineResult(
        "hipdnn",
        1,
        status,
        role=role,
        engine_name=name,
        gpu_kernel_stats=_stats(median, cv) if status == "success" else None,
        correctness=CorrectnessResult(True, match, 1e-3, 1e-5),
    )


def _write(tmp_path, fname, graphs, **config):
    suite = SuiteResult(
        run=RunInfo("t", [], {"cache_mode": "warm", "iters": 100, **config}, complete=True),
        environment={"gpu_model": "MI210", "gpu_arch": "gfx90a"},
        graphs=[
            GraphResult(name, f"{name}.json", rows, engine_ids=[1], graph_id=gid)
            for name, gid, rows in graphs
        ],
    )
    path = tmp_path / fname
    suite.write(path)
    return str(path)


def _json(capsys, argv):
    code = main(argv + ["--json"])
    return code, json.loads(capsys.readouterr().out)


def test_best_engine_speedup_and_geomean(tmp_path, capsys):
    a = _write(tmp_path, "a.json", [("g1", "id1", [_row("E1", 2.0), _row("E2", 4.0)]),
                                    ("g2", "id2", [_row("E1", 1.0)])])  # fmt: skip
    b = _write(tmp_path, "b.json", [("g1", "id1", [_row("E1", 3.0), _row("E2", 1.0)]),
                                    ("g2", "id2", [_row("E1", 0.5)])])  # fmt: skip
    code, report = _json(capsys, [a, b])
    assert code == 0
    g1, g2 = report["pairs"]
    # Best per side: A picks E1 (2.0), B picks E2 (1.0); B is 2x faster.
    assert (g1["engine_a"], g1["engine_b"], g1["speedup"]) == ("E1", "E2", 2.0)
    assert g1["label"] == "faster"
    assert g2["speedup"] == 2.0
    assert report["geomean_speedup"] == 2.0


def test_regression_exits_1(tmp_path, capsys):
    a = _write(tmp_path, "a.json", [("g", "id", [_row("E", 1.0)])])
    b = _write(tmp_path, "b.json", [("g", "id", [_row("E", 1.5)])])
    code, report = _json(capsys, [a, b])
    assert code == 1
    assert report["pairs"][0]["label"] == "REGRESSION"
    assert report["regressions"] == 1


def test_changes_within_threshold_or_noise_are_not_regressions(tmp_path, capsys):
    a = _write(tmp_path, "a.json", [("small", "i1", [_row("E", 1.0)]),
                                    ("noisy", "i2", [_row("E", 1.0, cv=0.1)])])  # fmt: skip
    b = _write(tmp_path, "b.json", [("small", "i1", [_row("E", 1.03)]),
                                    ("noisy", "i2", [_row("E", 1.2, cv=0.1)])])  # fmt: skip
    code, report = _json(capsys, [a, b])
    assert code == 0
    assert [p["label"] for p in report["pairs"]] == ["within noise", "within noise"]
    # A 1% threshold makes the quiet 3% slowdown a regression; the noisy 20%
    # slowdown stays inside its 2 * combined-CV band.
    code, report = _json(capsys, [a, b, "--threshold", "1"])
    assert code == 1
    assert [p["label"] for p in report["pairs"]] == ["REGRESSION", "within noise"]


def test_failed_rows_excluded_from_geomean(tmp_path, capsys):
    a = _write(tmp_path, "a.json", [("g", "id", [_row("E1", 1.0), _row("E2", 1.0),
                                                 _row("E3", 1.0)])])  # fmt: skip
    b = _write(tmp_path, "b.json", [("g", "id", [_row("E1", 0.5), _row("E2", 0.1, match=False),
                                                 _row("E3", 0, status="error")])])  # fmt: skip
    code, report = _json(capsys, [a, b, "--by", "engine"])
    e1, e2, e3 = report["pairs"]
    assert e1["in_geomean"] and e1["speedup"] == 2.0
    assert not e2["in_geomean"] and e2["label"] == "B failed"
    assert e3["speedup"] is None and e3["label"] == "B error"
    assert report["geomean_speedup"] == 2.0
    assert code == 0


def test_graphs_only_in_one_file_are_listed(tmp_path, capsys):
    a = _write(tmp_path, "a.json", [("shared", "s", [_row("E", 1.0)]),
                                    ("gone", "x", [_row("E", 1.0)])])  # fmt: skip
    b = _write(tmp_path, "b.json", [("shared", "s", [_row("E", 1.0)]),
                                    ("new", "y", [_row("E", 1.0)])])  # fmt: skip
    _, report = _json(capsys, [a, b])
    assert [p["graph"] for p in report["pairs"]] == ["shared"]
    assert (report["only_in_a"], report["only_in_b"]) == (["gone"], ["new"])


def test_join_on_graph_id_with_name_fallback(tmp_path, capsys):
    a = _write(tmp_path, "a.json", [("renamed_a", "same", [_row("E", 1.0)]),
                                    ("same_name", "id_a", [_row("E", 1.0)]),
                                    ("no_id", None, [_row("E", 1.0)])])  # fmt: skip
    b = _write(tmp_path, "b.json", [("renamed_b", "same", [_row("E", 1.0)]),
                                    ("same_name", "id_b", [_row("E", 1.0)]),
                                    ("no_id", "z", [_row("E", 1.0)])])  # fmt: skip
    _, report = _json(capsys, [a, b])
    # Same content under a new name joins; same name with different content does not.
    assert [p["graph"] for p in report["pairs"]] == ["renamed_a", "no_id"]
    assert report["only_in_a"] == ["same_name"]
    assert report["only_in_b"] == ["same_name"]


def test_config_mismatch_warns(tmp_path, capsys):
    a = _write(tmp_path, "a.json", [("g", "id", [_row("E", 1.0)])])
    b = _write(tmp_path, "b.json", [("g", "id", [_row("E", 1.0)])], iters=50)
    assert main([a, b]) == 0
    assert "iters" in capsys.readouterr().err


def test_cache_mode_mismatch_is_an_error_unless_allowed(tmp_path, capsys):
    a = _write(tmp_path, "a.json", [("g", "id", [_row("E", 1.0)])])
    b = _write(tmp_path, "b.json", [("g", "id", [_row("E", 1.0)])], cache_mode="cold")
    assert main([a, b]) == 2
    assert main([a, b, "--allow-mismatch"]) == 0


def test_unreadable_or_incompatible_input_exits_2(tmp_path, capsys):
    a = _write(tmp_path, "a.json", [("g", "id", [_row("E", 1.0)])])
    v1 = tmp_path / "v1.json"
    v1.write_text(json.dumps({"metadata": {}, "graphs": []}))
    assert main([a, str(tmp_path / "missing.json")]) == 2
    assert main([a, str(v1)]) == 2
    assert main([a]) == 2
    assert main([a, a, "--by", "nope"]) == 2


def test_table_fits_120_columns(tmp_path, capsys):
    long = "conv_fwd_" + "x" * 150
    a = _write(tmp_path, "a.json", [(long, "id", [_row("ENGINE_" + "Y" * 60, 123.456)])])
    b = _write(tmp_path, "b.json", [(long, "id", [_row("ENGINE_" + "Y" * 60, 0.001)])])
    assert main([a, b]) == 0
    out = capsys.readouterr().out
    assert "B speedup vs A" in out
    assert max(len(line) for line in out.splitlines()) <= 120

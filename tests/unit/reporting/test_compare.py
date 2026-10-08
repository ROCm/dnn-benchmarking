# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for ``dnn-benchmark compare`` on small synthetic result files."""

import csv
import io
import json

import pytest

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
    # Two-point samples: exact median, IQR / median = 2 * cv.
    d = median * cv
    return BenchmarkStats.from_timings([median - d, median + d] * 10)


def _row(name, median, *, cv=0.0, match=True, status="success", role="engine",
         timings=None, plugin_path=None, host=None, validated=True):  # fmt: skip
    stats = BenchmarkStats.from_timings(timings) if timings else _stats(median, cv)
    return ProviderEngineResult(
        "hipdnn",
        1,
        status,
        role=role,
        plugin_path=plugin_path,
        gpu_kernel_stats=stats if status == "success" else None,
        host_stats=_stats(host) if host else None,
        engine_name=name,
        correctness=CorrectnessResult(match, 1e-3, 1e-5) if validated else None,
    )


def _write(tmp_path, fname, graphs, **config):
    suite = SuiteResult(
        run=RunInfo(
            "t", [], {"cache_mode": "warm", "iters": 100, **config}, complete=True
        ),
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
                                    ("g2", "id2", [_row("E1", 2.0)])])  # fmt: skip
    code, report = _json(capsys, [a, b])
    assert code == 1  # g2 is a 2x regression
    g1, g2 = report["pairs"]
    # Best per side: A picks E1 (2.0), B picks E2 (1.0); B is 2x faster.
    assert (g1["engine_a"], g1["engine_b"], g1["speedup"]) == ("E1", "E2", 2.0)
    assert g1["label"] == "faster"
    assert g2["speedup"] == 0.5
    # Geometric mean of 2.0 and 0.5 (arithmetic would be 1.25).
    assert report["geomean_speedup"] == pytest.approx(1.0)


def test_regression_exits_1(tmp_path, capsys):
    a = _write(tmp_path, "a.json", [("g", "id", [_row("E", 1.0)])])
    b = _write(tmp_path, "b.json", [("g", "id", [_row("E", 1.5)])])
    code, report = _json(capsys, [a, b])
    assert code == 1
    assert report["pairs"][0]["label"] == "REGRESSION"
    assert report["regressions"] == 1


def test_unchecked_rows_are_compared(tmp_path, capsys):
    # A default run has no --validate, so every row's verdict is 'unchecked'.
    a = _write(tmp_path, "a.json", [("g", "id", [_row("E", 1.0, validated=False)])])
    b = _write(tmp_path, "b.json", [("g", "id", [_row("E", 1.2, validated=False)])])
    code, report = _json(capsys, [a, b])
    assert code == 1
    (pair,) = report["pairs"]
    assert (pair["label"], pair["in_geomean"]) == ("REGRESSION", True)


def test_changes_within_threshold_or_noise_are_not_regressions(tmp_path, capsys):
    a = _write(tmp_path, "a.json", [("small", "i1", [_row("E", 1.0)]),
                                    ("noisy", "i2", [_row("E", 1.0, cv=0.1)])])  # fmt: skip
    b = _write(tmp_path, "b.json", [("small", "i1", [_row("E", 1.03)]),
                                    ("noisy", "i2", [_row("E", 1.2, cv=0.1)])])  # fmt: skip
    code, report = _json(capsys, [a, b])
    assert code == 0
    assert [p["label"] for p in report["pairs"]] == ["within noise", "within noise"]
    # A 1% threshold makes the quiet 3% slowdown a regression; the noisy 20%
    # slowdown stays inside its 2 * combined relative-IQR band.
    code, report = _json(capsys, [a, b, "--threshold", "1"])
    assert code == 1
    assert [p["label"] for p in report["pairs"]] == ["REGRESSION", "within noise"]


def test_noise_band_is_twice_the_combined_relative_iqr(tmp_path, capsys):
    # Each side has relative IQR 0.2, so hypot = 0.283 and the band is 0.566.
    # A 40% slowdown lies between the two: noise at 2x, a regression at 1x.
    a = _write(tmp_path, "a.json", [("g", "id", [_row("E", 1.0, cv=0.1)])])
    b = _write(tmp_path, "b.json", [("g", "id", [_row("E", 1.4, cv=0.1)])])
    code, report = _json(capsys, [a, b])
    assert (code, report["pairs"][0]["label"]) == (0, "within noise")


def test_noise_band_uses_twice_the_hypot_not_a_wider_band(tmp_path, capsys):
    # Band 0.566 (2 * hypot(0.2, 0.2)); a 70% slowdown exceeds it but lies
    # inside 3 * hypot (0.849) and inside summed spreads 2 * (0.2 + 0.2) (0.8).
    a = _write(tmp_path, "a.json", [("g", "id", [_row("E", 1.0, cv=0.1)])])
    b = _write(tmp_path, "b.json", [("g", "id", [_row("E", 1.7, cv=0.1)])])
    code, report = _json(capsys, [a, b])
    assert (code, report["pairs"][0]["label"]) == (1, "REGRESSION")


def test_change_equal_to_threshold_is_within_noise(tmp_path, capsys):
    # 1.25 / 1.0 - 1 == 25 / 100 exactly in binary floating point.
    a = _write(tmp_path, "a.json", [("g", "id", [_row("E", 1.0)])])
    b = _write(tmp_path, "b.json", [("g", "id", [_row("E", 1.25)])])
    code, report = _json(capsys, [a, b, "--threshold", "25"])
    assert (code, report["pairs"][0]["label"]) == (0, "within noise")


def test_best_skips_faster_failed_row(tmp_path, capsys):
    a = _write(tmp_path, "a.json", [("g", "id", [_row("E1", 1.0)])])
    b = _write(tmp_path, "b.json",
               [("g", "id", [_row("E1", 1.0), _row("E2", 0.1, match=False)])])  # fmt: skip
    _, report = _json(capsys, [a, b])
    assert (report["pairs"][0]["engine_b"], report["pairs"][0]["label"]) == (
        "E1",
        "within noise",
    )


def test_metric_host_compares_host_medians(tmp_path, capsys):
    # Kernel and host times rank the B engines in opposite orders.
    a = _write(tmp_path, "a.json", [("g", "id", [_row("E1", 3.0, host=1.0)])])
    b = _write(tmp_path, "b.json", [("g", "id", [_row("E1", 3.0, host=2.0),
                                                 _row("E2", 1.0, host=4.0)])])  # fmt: skip
    code, report = _json(capsys, [a, b, "--metric", "host"])
    pair = report["pairs"][0]
    assert (code, pair["engine_b"], pair["speedup"]) == (1, "E1", 0.5)
    assert main([a, b]) == 0  # by kernel time, B's E2 is faster


def test_csv_output(tmp_path, capsys):
    a = _write(tmp_path, "a.json", [("g", "id", [_row("E", 2.0)])])
    b = _write(tmp_path, "b.json", [("g", "id", [_row("E", 1.0)])])
    assert main([a, b, "--csv"]) == 0
    rows = list(csv.DictReader(io.StringIO(capsys.readouterr().out)))
    assert rows == [
        {
            "graph": "g",
            "engine_a": "E",
            "engine_b": "E",
            "a_ms": "2.0",
            "b_ms": "1.0",
            "a_rel_iqr": "0.0",
            "b_rel_iqr": "0.0",
            "speedup": "2.0",
            "label": "faster",
            "in_geomean": "True",
        }
    ]


def test_no_shared_graphs_prints_table_and_warns(tmp_path, capsys):
    a = _write(tmp_path, "a.json", [("ga", "aaa", [_row("E", 1.0)])])
    b = _write(tmp_path, "b.json", [("gb", "bbb", [_row("E", 1.0)])])
    assert main([a, b]) == 0
    out, err = capsys.readouterr()
    assert "graph only in A: ga" in out and "0 pairs" in out
    assert "no timings were compared" in err


def test_one_outlier_sample_does_not_hide_a_regression(tmp_path, capsys):
    a = _write(tmp_path, "a.json", [("g", "id", [_row("E", 1.0, timings=[1.0] * 100)])])
    b = _write(tmp_path, "b.json",
               [("g", "id", [_row("E", 1.5, timings=[1.5] * 99 + [30.0])])])  # fmt: skip
    code, report = _json(capsys, [a, b])
    assert code == 1
    assert report["pairs"][0]["label"] == "REGRESSION"


def test_by_engine_pairs_duplicated_engines_in_order(tmp_path, capsys):
    # -e E,E --plugin-path a,b gives two rows with the same engine identity.
    rows = [_row("E", 1.0, plugin_path="/a"), _row("E", 2.0, plugin_path="/b")]
    a = _write(tmp_path, "a.json", [("g", "id", rows)])
    b = _write(tmp_path, "b.json", [("g", "id", rows)])
    code, report = _json(capsys, [a, b, "--by", "engine"])
    assert code == 0
    assert [(p["a_ms"], p["b_ms"], p["label"]) for p in report["pairs"]] == [
        (1.0, 1.0, "within noise"),
        (2.0, 2.0, "within noise"),
    ]


def test_by_engine_lists_engines_only_in_b(tmp_path, capsys):
    a = _write(tmp_path, "a.json", [("g", "id", [_row("E1", 1.0)])])
    b = _write(tmp_path, "b.json", [("g", "id", [_row("E1", 1.0), _row("E2", 1.0)])])
    _, report = _json(capsys, [a, b, "--by", "engine"])
    assert [(p["engine_b"], p["label"]) for p in report["pairs"]] == [
        ("E1", "within noise"),
        ("E2", "no A row"),
    ]


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


def test_join_on_graph_id(tmp_path, capsys):
    a = _write(tmp_path, "a.json", [("renamed_a", "same", [_row("E", 1.0)]),
                                    ("same_name", "id_a", [_row("E", 1.0)])])  # fmt: skip
    b = _write(tmp_path, "b.json", [("renamed_b", "same", [_row("E", 1.0)]),
                                    ("same_name", "id_b", [_row("E", 1.0)])])  # fmt: skip
    _, report = _json(capsys, [a, b])
    # Same content under a new name joins; same name with different content does not.
    assert [p["graph"] for p in report["pairs"]] == ["renamed_a"]
    assert report["only_in_a"] == ["same_name"]
    assert report["only_in_b"] == ["same_name"]


def test_duplicate_graph_ids_pair_in_order_of_appearance(tmp_path, capsys):
    a = _write(tmp_path, "a.json", [("a1", "d", [_row("E", 1.0)]),
                                    ("a2", "d", [_row("E", 2.0)])])  # fmt: skip
    b = _write(tmp_path, "b.json", [("b1", "d", [_row("E", 1.0)]),
                                    ("b2", "d", [_row("E", 2.0)]),
                                    ("b3", "d", [_row("E", 3.0)])])  # fmt: skip
    _, report = _json(capsys, [a, b])
    assert [(p["graph"], p["speedup"]) for p in report["pairs"]] == [
        ("a1", 1.0),
        ("a2", 1.0),
    ]
    assert (report["only_in_a"], report["only_in_b"]) == ([], ["b3"])


def test_config_mismatch_warns(tmp_path, capsys):
    a = _write(tmp_path, "a.json", [("g", "id", [_row("E", 1.0)])])
    b = _write(tmp_path, "b.json", [("g", "id", [_row("E", 1.0)])], iters=50)
    assert main([a, b]) == 0
    assert "iters" in capsys.readouterr().err


@pytest.mark.parametrize("key, value", [("cache_mode", "cold"), ("timing_block", 8)])
def test_timed_config_mismatch_is_an_error_unless_allowed(tmp_path, capsys, key, value):
    a = _write(tmp_path, "a.json", [("g", "id", [_row("E", 1.0)])], timing_block=1)
    b = _write(tmp_path, "b.json", [("g", "id", [_row("E", 1.0)])],
               **{"timing_block": 1, key: value})  # fmt: skip
    assert main([a, b]) == 2
    assert key in capsys.readouterr().err
    assert main([a, b, "--allow-mismatch"]) == 0


def test_unreadable_or_incompatible_input_exits_2(tmp_path, capsys):
    a = _write(tmp_path, "a.json", [("g", "id", [_row("E", 1.0)])])
    v1 = tmp_path / "v1.json"
    v1.write_text(json.dumps({"metadata": {}, "graphs": []}))
    assert main([a, str(tmp_path / "missing.json")]) == 2
    assert main([a, str(v1)]) == 2
    assert main([a]) == 2
    assert main([a, a, "--by", "nope"]) == 2
    assert main([a, a, "--threshold", "nan"]) == 2
    assert main([a, a, "--threshold", "-1"]) == 2


@pytest.mark.parametrize("by", ["best", "engine"])
def test_null_median_written_for_nan_timings_is_unusable(tmp_path, capsys, by):
    nan = float("nan")
    a = _write(tmp_path, "a.json", [("g", "id", [_row("E", 1.0)])])
    b = _write(tmp_path, "b.json", [("g", "id", [_row("E", 0, timings=[nan] * 3)])])
    b_row = SuiteResult.load(b)["graphs"][0]["results"][0]
    assert b_row["ootb"]["kernel"]["median_ms"] is None
    code, report = _json(capsys, [a, b, "--by", by])
    (pair,) = report["pairs"]
    assert code == 0 and pair["speedup"] is None
    assert pair["label"] == ("no B row" if by == "best" else "B no time")


def test_malformed_row_exits_2_not_regression(tmp_path, capsys):
    a = _write(tmp_path, "a.json", [("g", "id", [_row("E", 1.0)])])
    doc = json.loads(open(a).read())
    del doc["graphs"][0]["results"][0]["verdict"]
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(doc))
    assert main([a, str(bad)]) == 2
    assert "error:" in capsys.readouterr().err


def test_missing_ref_on_both_sides_is_one_label(tmp_path, capsys):
    a = _write(tmp_path, "a.json", [("g", "id", [_row("E", 1.0)])])
    b = _write(tmp_path, "b.json", [("g", "id", [_row("E", 1.0)])])
    _, report = _json(capsys, [a, b, "--by", "ref"])
    assert report["pairs"][0]["label"] == "no ref row in either"


def test_by_ref_compares_reference_rows(tmp_path, capsys):
    a = _write(tmp_path, "a.json", [("g", "id", [_row("E", 1.0), _row("R", 4.0, role="reference")])])  # fmt: skip
    b = _write(tmp_path, "b.json", [("g", "id", [_row("E", 1.0), _row("R", 2.0, role="reference")])])  # fmt: skip
    code, report = _json(capsys, [a, b, "--by", "ref"])
    (pair,) = report["pairs"]
    assert (code, pair["engine_b"], pair["speedup"], pair["label"]) == (
        0,
        "R",
        2.0,
        "faster",
    )


def test_zero_median_is_unusable_not_a_division_error(tmp_path, capsys):
    a = _write(tmp_path, "a.json", [("g", "id", [_row("E", 0.0)])])
    b = _write(tmp_path, "b.json", [("g", "id", [_row("E", 1.0)])])
    code, report = _json(capsys, [a, b])
    assert (code, report["pairs"][0]["label"]) == (0, "A no time")


@pytest.mark.parametrize("columns", [80, 100, 200])
def test_table_fits_terminal_width(tmp_path, capsys, monkeypatch, columns):
    monkeypatch.setenv("COLUMNS", str(columns))
    long = "conv_fwd_" + "x" * 150
    a = _write(
        tmp_path, "a.json", [(long, "id", [_row("ENGINE_" + "Y" * 60, 123.456)])]
    )
    b = _write(tmp_path, "b.json", [(long, "id", [_row("ENGINE_" + "Y" * 60, 0.001)])])
    assert main([a, b]) == 0
    out = capsys.readouterr().out
    assert max(len(line) for line in out.splitlines()) <= columns
    # Same units as the run table.
    assert "123.456 ms" in out and "1.00 µs" in out

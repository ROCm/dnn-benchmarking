# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Kernel-selection environment: what is set, and which hazards are reported."""

import io
import os

import pytest

from dnn_benchmarking.cli import suite_runner_cli
from dnn_benchmarking.cli.config_file import apply_config_file
from dnn_benchmarking.cli.parser import create_parser
from dnn_benchmarking.cli.suite_runner_cli import _apply_tuning_environment
from dnn_benchmarking.config.benchmark_config import SuiteConfig
from dnn_benchmarking.reporting.reporter import Reporter


def _lines(**kwargs):
    """The lines _apply_tuning_environment reports."""
    out = io.StringIO()
    _apply_tuning_environment(SuiteConfig(**kwargs), Reporter(output=out))
    return out.getvalue().splitlines()


def _warnings(**kwargs):
    """The WARNING lines _apply_tuning_environment reports."""
    return [line for line in _lines(**kwargs) if line.startswith("WARNING")]


def test_default_stays_on_the_heuristic_path() -> None:
    assert _warnings() == []
    assert "HIPDNN_FORCE_BENCHMARKING" not in os.environ


def test_autotune_says_earlier_tuned_results_are_reused() -> None:
    """The user learns that earlier tuned results are reused and which flag
    searches again."""
    lines = _lines(autotune=True)
    assert [line for line in lines if line.startswith("WARNING")] == []
    assert any("reuse" in line and "--fresh-state" in line for line in lines)
    assert any("kernel selection: autotune;" in line for line in lines)
    assert os.environ["HIPDNN_FORCE_BENCHMARKING"] == "1"


def test_autotune_with_the_disk_cache_off_reports_it(monkeypatch) -> None:
    """With HIPDNN_DISABLE_CACHE set, no earlier ranking can be served: the
    line says the cache is off and no reuse note is printed."""
    monkeypatch.setenv("HIPDNN_DISABLE_CACHE", "1")
    lines = _lines(autotune=True)
    assert not any("--fresh-state" in line for line in lines)
    assert any("cache: disabled" in line for line in lines)


@pytest.mark.parametrize(
    "value, warned", [("1", True), ("0", False)], ids=["truthy-leak", "off-value"]
)
def test_leaked_force_benchmarking(monkeypatch, value, warned) -> None:
    """hipDNN treats "0" as off, so only a truthy inherited value is a hazard."""
    monkeypatch.setenv("HIPDNN_FORCE_BENCHMARKING", value)
    lines = _lines()
    warnings = [line for line in lines if line.startswith("WARNING")]
    assert [("HIPDNN_FORCE_BENCHMARKING" in w) for w in warnings] == (
        [True] if warned else []
    )
    path = "autotune" if warned else "heuristic"
    assert any(f"kernel selection: {path};" in line for line in lines)


def test_pytorch_runtime_leaves_hipdnn_environment_alone(tmp_path, monkeypatch) -> None:
    applied = []
    monkeypatch.setattr(
        suite_runner_cli, "_apply_tuning_environment", lambda c, r: applied.append(c)
    )
    monkeypatch.setattr(suite_runner_cli, "collect_environment_info", lambda: {})
    monkeypatch.setattr(
        suite_runner_cli,
        "start_runtime",
        lambda c, r, s, tuned_runs=None: lambda *a, **kw: None,
    )

    args = create_parser(suppress_defaults=True).parse_args(["-r", "pytorch"])
    apply_config_file(args)
    suite_runner_cli.run_suite_cli(args, [], Reporter(output=io.StringIO()))
    assert applied == []

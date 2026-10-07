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


def test_autotune_without_cache_dir_is_a_hazard() -> None:
    [warning] = _warnings(autotune=True)
    assert "--cache-dir" in warning
    assert os.environ["HIPDNN_FORCE_BENCHMARKING"] == "1"


def test_autotune_with_cache_dir_is_isolated() -> None:
    assert _warnings(autotune=True, cache_dir="/tmp/phase-x") == []
    assert os.environ["HIPDNN_CACHE_DIR"] == "/tmp/phase-x"


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


def test_pytorch_backend_leaves_hipdnn_environment_alone(tmp_path, monkeypatch) -> None:
    applied = []
    monkeypatch.setattr(
        suite_runner_cli, "_apply_tuning_environment", lambda c, r: applied.append(c)
    )
    monkeypatch.setattr(suite_runner_cli, "collect_environment_info", lambda: {})
    monkeypatch.setattr(suite_runner_cli, "start_backend", lambda c, r: lambda *a: None)

    args = create_parser(suppress_defaults=True).parse_args(["-b", "pytorch"])
    apply_config_file(args)
    suite_runner_cli.run_suite_cli(args, [], Reporter(output=io.StringIO()))
    assert applied == []

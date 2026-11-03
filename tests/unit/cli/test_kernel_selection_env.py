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


def test_autotune_is_reported_and_left_to_the_child_spec() -> None:
    """--autotune reaches the engine runs through their child spec; this
    process sets no benchmarking variable."""
    lines = _lines(autotune=True)
    assert [line for line in lines if line.startswith("WARNING")] == []
    assert any("kernel selection: autotune; cache: disabled" in l for l in lines)
    assert "HIPDNN_FORCE_BENCHMARKING" not in os.environ


def test_hipdnn_disk_cache_is_off_for_engine_discovery() -> None:
    _lines()
    assert os.environ["HIPDNN_DISABLE_CACHE"] == "1"


@pytest.mark.parametrize("value", ["1", "0"])
def test_shell_force_benchmarking_is_ignored_with_a_warning(monkeypatch, value) -> None:
    """The child spec fixes the variable, so a shell value never changes the
    selection; the user is told so."""
    monkeypatch.setenv("HIPDNN_FORCE_BENCHMARKING", value)
    lines = _lines()
    warnings = [line for line in lines if line.startswith("WARNING")]
    assert [("HIPDNN_FORCE_BENCHMARKING" in w) for w in warnings] == [True]
    assert any("kernel selection: heuristic;" in line for line in lines)


def test_pytorch_runtime_leaves_hipdnn_environment_alone(tmp_path, monkeypatch) -> None:
    applied = []
    monkeypatch.setattr(
        suite_runner_cli, "_apply_tuning_environment", lambda c, r: applied.append(c)
    )
    monkeypatch.setattr(suite_runner_cli, "collect_environment_info", lambda: {})
    monkeypatch.setattr(suite_runner_cli, "start_runtime", lambda c, r: lambda *a: None)

    args = create_parser(suppress_defaults=True).parse_args(["-r", "pytorch"])
    apply_config_file(args)
    suite_runner_cli.run_suite_cli(args, [], Reporter(output=io.StringIO()))
    assert applied == []

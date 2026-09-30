# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Kernel-selection environment: what is set, and which hazards are reported."""

import io
import os

import pytest

from dnn_benchmarking.cli import suite_runner_cli
from dnn_benchmarking.cli.config_file import apply_config_file
from dnn_benchmarking.cli.parser import create_parser
from dnn_benchmarking.cli.suite_runner_cli import (
    AUTOTUNE_WITHOUT_CACHE_DIR,
    LEAKED_FORCE_BENCHMARKING,
    _apply_tuning_environment,
)
from dnn_benchmarking.config.benchmark_config import SuiteConfig
from dnn_benchmarking.reporting.reporter import Reporter


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    # Process-wide variables; monkeypatch restores them after each test.
    monkeypatch.delenv("HIPDNN_FORCE_BENCHMARKING", raising=False)
    monkeypatch.delenv("HIPDNN_CACHE_DIR", raising=False)


def _apply(**kwargs):
    return _apply_tuning_environment(
        SuiteConfig(**kwargs), Reporter(output=io.StringIO())
    )


def test_default_stays_on_the_heuristic_path() -> None:
    assert _apply() == []
    assert "HIPDNN_FORCE_BENCHMARKING" not in os.environ


def test_autotune_without_cache_dir_is_a_hazard() -> None:
    assert _apply(autotune=True) == [AUTOTUNE_WITHOUT_CACHE_DIR]
    assert os.environ["HIPDNN_FORCE_BENCHMARKING"] == "1"


def test_autotune_with_cache_dir_is_isolated() -> None:
    assert _apply(autotune=True, cache_dir="/tmp/phase-x") == []
    assert os.environ["HIPDNN_CACHE_DIR"] == "/tmp/phase-x"


@pytest.mark.parametrize(
    "value, hazards",
    [("1", [LEAKED_FORCE_BENCHMARKING]), ("0", [])],
    ids=["truthy-leak", "off-value"],
)
def test_leaked_force_benchmarking(monkeypatch, value, hazards) -> None:
    """hipDNN treats "0" as off, so only a truthy inherited value is a hazard."""
    monkeypatch.setenv("HIPDNN_FORCE_BENCHMARKING", value)
    assert _apply() == hazards


def test_pytorch_backend_leaves_hipdnn_environment_alone(
    tmp_path, monkeypatch
) -> None:
    applied = []
    monkeypatch.setattr(
        suite_runner_cli, "_apply_tuning_environment", lambda c, r: applied.append(c)
    )
    monkeypatch.setattr(suite_runner_cli, "collect_environment_info", lambda: {})
    monkeypatch.setattr(
        suite_runner_cli, "start_backend", lambda c, r: lambda *a: None
    )

    args = create_parser(suppress_defaults=True).parse_args(["-b", "pytorch"])
    apply_config_file(args)
    suite_runner_cli.run_suite_cli(args, [], Reporter(output=io.StringIO()))
    assert applied == []

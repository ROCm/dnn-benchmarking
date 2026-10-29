# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Kernel-selection environment: inherited variables, private caches, hazards."""

import io
import os
from pathlib import Path

import pytest

from dnn_benchmarking.cli import suite_runner_cli
from dnn_benchmarking.cli.config_file import apply_config_file
from dnn_benchmarking.cli.parser import create_parser
from dnn_benchmarking.cli.suite_runner_cli import _apply_tuning_environment
from dnn_benchmarking.common import selection_env
from dnn_benchmarking.config.benchmark_config import SuiteConfig
from dnn_benchmarking.reporting.reporter import Reporter


def _lines(**kwargs):
    """The lines _apply_tuning_environment reports inside an isolated run."""
    config = SuiteConfig(**kwargs)
    out = io.StringIO()
    with selection_env.isolate(config.hipdnn_cache_dir) as private:
        _apply_tuning_environment(config, Reporter(output=out), private)
    return out.getvalue().splitlines()


def _warnings(lines):
    return [line for line in lines if line.startswith("WARNING")]


def test_default_is_heuristic_on_a_private_cache() -> None:
    lines = _lines()

    assert _warnings(lines) == []
    assert any(
        "kernel selection: heuristic;" in line and "(private" in line for line in lines
    )


def test_autotune_is_scoped_to_the_run() -> None:
    """The run sets HIPDNN_FORCE_BENCHMARKING; a later run in the same
    process must not inherit it."""
    config = SuiteConfig(autotune=True)
    with selection_env.isolate(None) as private:
        _apply_tuning_environment(config, Reporter(output=io.StringIO()), private)
        assert os.environ["HIPDNN_FORCE_BENCHMARKING"] == "1"

    assert "HIPDNN_FORCE_BENCHMARKING" not in os.environ


def test_hipdnn_cache_dir_is_a_shared_cache() -> None:
    lines = _lines(autotune=True, hipdnn_cache_dir="/tmp/phase-x")

    assert _warnings(lines) == []
    assert any(
        "kernel selection: autotune; cache: /tmp/phase-x (shared)" in line
        for line in lines
    )


@pytest.mark.parametrize(
    "value, warned", [("1", True), ("0", False)], ids=["truthy-leak", "off-value"]
)
def test_allowed_force_benchmarking_is_reported(monkeypatch, value, warned) -> None:
    """hipDNN treats "0" as off, so only a truthy inherited value is a hazard."""
    monkeypatch.setenv("HIPDNN_FORCE_BENCHMARKING", value)

    lines = _lines()

    assert [("HIPDNN_FORCE_BENCHMARKING" in w) for w in _warnings(lines)] == (
        [True] if warned else []
    )
    path = "autotune" if warned else "heuristic"
    assert any(f"kernel selection: {path};" in line for line in lines)


# --- selection_env ------------------------------------------------------------


def test_inherited_lists_named_and_prefixed_variables_with_a_value() -> None:
    environ = {
        "HIPDNN_FORCE_BENCHMARKING": "1",
        "MIOPEN_DEBUG_CONV_DIRECT": "0",
        "PYTORCH_TUNABLEOP_ENABLED": "1",
        "HIPDNN_HEUR_POLICY_ORDER": "a,b",
        "MIOPEN_FIND_MODE": "",  # empty means unset to hipDNN and MIOpen
        "MIOPEN_LOG_LEVEL": "5",  # logging, not selection
        "MIOPEN_CUSTOM_CACHE_DIR": "/c",  # compiled kernels, shared
    }

    assert list(selection_env.inherited(environ)) == [
        "HIPDNN_FORCE_BENCHMARKING",
        "HIPDNN_HEUR_POLICY_ORDER",
        "MIOPEN_DEBUG_CONV_DIRECT",
        "PYTORCH_TUNABLEOP_ENABLED",
    ]


def test_isolate_gives_empty_private_caches_and_restores_the_environment(
    monkeypatch,
) -> None:
    monkeypatch.delenv("MIOPEN_USER_DB_PATH", raising=False)

    with selection_env.isolate(None) as private:
        dirs = [
            Path(os.environ["HIPDNN_CACHE_DIR"]),
            Path(os.environ["MIOPEN_USER_DB_PATH"]),
        ]
        assert [d.parent for d in dirs] == [private, private]
        assert all(d.is_dir() and not any(d.iterdir()) for d in dirs)

    assert not private.exists()
    assert "HIPDNN_CACHE_DIR" not in os.environ
    assert "MIOPEN_USER_DB_PATH" not in os.environ


def test_isolate_keeps_an_allowed_or_parent_value(monkeypatch) -> None:
    monkeypatch.setenv("MIOPEN_USER_DB_PATH", "/allowed/db")

    with selection_env.isolate("/shared/hipdnn"):
        assert os.environ["MIOPEN_USER_DB_PATH"] == "/allowed/db"
        assert os.environ["HIPDNN_CACHE_DIR"] == "/shared/hipdnn"

    assert os.environ["MIOPEN_USER_DB_PATH"] == "/allowed/db"
    assert "HIPDNN_CACHE_DIR" not in os.environ


# --- run_suite_cli ------------------------------------------------------------


@pytest.fixture
def suite(monkeypatch):
    """run_suite_cli with no runtime and no graphs; records the run's env."""
    seen = {}

    def start_runtime(config, reporter):
        seen.update(os.environ)
        return lambda *a: None

    monkeypatch.setattr(suite_runner_cli, "collect_environment_info", lambda: {})
    monkeypatch.setattr(suite_runner_cli, "start_runtime", start_runtime)

    def run(*argv):
        args = create_parser(suppress_defaults=True).parse_args(list(argv))
        apply_config_file(args)
        out = io.StringIO()
        code = suite_runner_cli.run_suite_cli(args, [], Reporter(output=out))
        return code, out.getvalue(), seen

    return run


def test_inherited_selection_variable_stops_the_run(suite, monkeypatch) -> None:
    monkeypatch.setenv("MIOPEN_FIND_MODE", "1")
    monkeypatch.setenv("HIPDNN_DISABLE_EXACT_ENGINE_CACHE", "1")

    code, text, seen = suite()

    assert code == 2
    assert seen == {}  # no runtime started
    assert "HIPDNN_DISABLE_EXACT_ENGINE_CACHE=1, MIOPEN_FIND_MODE=1" in text
    assert "--allow-selection-env" in text


def test_allowed_selection_variable_is_kept_and_reported(suite, monkeypatch) -> None:
    monkeypatch.setenv("MIOPEN_FIND_MODE", "1")

    code, text, seen = suite("--allow-selection-env")

    assert code == 0
    assert seen["MIOPEN_FIND_MODE"] == "1"
    assert "--allow-selection-env: keeping MIOPEN_FIND_MODE=1" in text


def test_every_runtime_selects_from_private_caches(suite) -> None:
    code, _, seen = suite("-r", "pytorch")

    assert code == 0
    assert Path(seen["MIOPEN_USER_DB_PATH"]).name == "miopen"
    assert not Path(seen["MIOPEN_USER_DB_PATH"]).exists()  # removed after the run
    assert "MIOPEN_USER_DB_PATH" not in os.environ


def test_pytorch_runtime_leaves_hipdnn_environment_alone(suite, monkeypatch) -> None:
    applied = []
    monkeypatch.setattr(
        suite_runner_cli,
        "_apply_tuning_environment",
        lambda c, r, p: applied.append(c),
    )

    suite("-r", "pytorch")

    assert applied == []

# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for the PyTorch kernel-selection environment."""

import os

import pytest

from dnn_benchmarking.common import pytorch_tuning
from dnn_benchmarking.common.work_dir import SelectionState

_TUNING_PREFIXES = ("PYTORCH_TUNABLEOP_", "MIOPEN_")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in pytorch_tuning.ENV_NAMES:
        # setenv first so monkeypatch restores what apply_pytorch_environment sets.
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)


def _tuning_vars() -> dict:
    return {k: v for k, v in os.environ.items() if k.startswith(_TUNING_PREFIXES)}


def test_defaults_enable_nhwc_and_aotriton() -> None:
    effective, replaced = pytorch_tuning.apply_pytorch_environment()
    assert replaced == {}
    assert effective["PYTORCH_MIOPEN_SUGGEST_NHWC"] == "1"
    assert effective["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] == "1"
    assert os.environ["PYTORCH_MIOPEN_SUGGEST_NHWC"] == "1"


def test_ootb_environment_never_enables_tuning_or_touches_miopen(monkeypatch) -> None:
    """OOTB PyTorch disables TunableOp and never touches MIOPEN_* variables."""
    monkeypatch.setenv("MIOPEN_LOG_LEVEL", "3")
    miopen_before = {k: v for k, v in os.environ.items() if k.startswith("MIOPEN_")}
    pytorch_tuning.apply_pytorch_environment()
    assert os.environ["PYTORCH_TUNABLEOP_ENABLED"] == "0"
    assert os.environ["PYTORCH_TUNABLEOP_TUNING"] == "0"
    miopen_after = {k: v for k, v in os.environ.items() if k.startswith("MIOPEN_")}
    assert miopen_after == miopen_before


def test_tuned_child_preserves_tunableop_variables(monkeypatch, tmp_path) -> None:
    """B3: tuned_subprocess_env followed by the child's env setup keeps TunableOp=1."""
    from dnn_benchmarking.cli.suite_runner_cli import _apply_pytorch_environment
    from dnn_benchmarking.config.benchmark_config import RuntimeName, SuiteConfig
    from dnn_benchmarking.reporting.reporter import Reporter

    tuned = SelectionState.under(tmp_path / "tuned")
    child_env = pytorch_tuning.tuned_subprocess_env(str(tmp_path), tuned)
    for key in (
        "PYTORCH_TUNABLEOP_ENABLED",
        "PYTORCH_TUNABLEOP_TUNING",
        "PYTORCH_TUNABLEOP_FILENAME",
    ):
        monkeypatch.setenv(key, child_env[key])

    cfg = SuiteConfig(runtime=RuntimeName.PYTORCH)
    _apply_pytorch_environment(cfg, Reporter(quiet=True), tuned=True)

    assert os.environ["PYTORCH_TUNABLEOP_ENABLED"] == "1"
    assert os.environ["PYTORCH_TUNABLEOP_TUNING"] == "1"
    assert os.environ["PYTORCH_MIOPEN_SUGGEST_NHWC"] == "1"
    assert os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] == "1"


def test_tuned_subprocess_env_forces_tuning_and_the_shared_state(
    monkeypatch, tmp_path
) -> None:
    """Inherited values would change what the tuned child tunes; the selection
    state is the tuned one that every tuned run shares."""
    monkeypatch.setenv("PYTORCH_TUNABLEOP_ENABLED", "0")
    monkeypatch.setenv("PYTORCH_TUNABLEOP_FILENAME", "/stale/results.csv")
    monkeypatch.setenv("MIOPEN_USER_DB_PATH", "/ootb/db")
    monkeypatch.setenv("DNN_BENCH_TEST_PASSTHROUGH", "kept")
    parent_before = dict(os.environ)
    tuned = SelectionState.under(tmp_path / "tuned")

    env = pytorch_tuning.tuned_subprocess_env(str(tmp_path), tuned)

    assert env["PYTORCH_TUNABLEOP_ENABLED"] == "1"
    assert env["PYTORCH_TUNABLEOP_TUNING"] == "1"
    assert env["PYTORCH_TUNABLEOP_FILENAME"] == str(tmp_path / "tunableop_results.csv")
    assert env["MIOPEN_USER_DB_PATH"] == str(tuned.miopen_user_db)
    assert env["HIPDNN_CACHE_DIR"] == str(tuned.hipdnn_cache)
    assert env["DNN_BENCH_TEST_PASSTHROUGH"] == "kept"
    assert dict(os.environ) == parent_before

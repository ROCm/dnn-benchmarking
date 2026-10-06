# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for the PyTorch kernel-selection environment."""

import os

import pytest

from dnn_benchmarking.common import pytorch_tuning

_TUNING_PREFIXES = ("PYTORCH_TUNABLEOP_", "MIOPEN_")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in pytorch_tuning.ENV_NAMES:
        # setenv first so monkeypatch restores what apply_pytorch_environment sets.
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)
    # No torch import: keeps cudnn.benchmark out of the process under test.
    monkeypatch.setattr(pytorch_tuning.torch_support, "module_available", lambda: False)


def _tuning_vars() -> dict:
    return {k: v for k, v in os.environ.items() if k.startswith(_TUNING_PREFIXES)}


def test_defaults_enable_nhwc_and_aotriton() -> None:
    effective = pytorch_tuning.apply_pytorch_environment()
    assert effective["PYTORCH_MIOPEN_SUGGEST_NHWC"] == "1"
    assert effective["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] == "1"
    assert os.environ["PYTORCH_MIOPEN_SUGGEST_NHWC"] == "1"


def test_caller_environment_wins(monkeypatch) -> None:
    monkeypatch.setenv("PYTORCH_MIOPEN_SUGGEST_NHWC", "0")
    effective = pytorch_tuning.apply_pytorch_environment()
    assert effective["PYTORCH_MIOPEN_SUGGEST_NHWC"] == "0"
    assert os.environ["PYTORCH_MIOPEN_SUGGEST_NHWC"] == "0"


def test_ootb_environment_never_enables_tuning_or_touches_miopen() -> None:
    """OOTB PyTorch and the hipDNN MIOpen plugin share this process."""
    before = _tuning_vars()
    pytorch_tuning.apply_pytorch_environment()
    assert _tuning_vars() == before


def test_tuned_subprocess_env_forces_isolated_state(monkeypatch, tmp_path) -> None:
    """Inherited values would let the tuned child reuse or leak tuning state."""
    monkeypatch.setenv("PYTORCH_TUNABLEOP_ENABLED", "0")
    monkeypatch.setenv("PYTORCH_TUNABLEOP_FILENAME", "/stale/results.csv")
    monkeypatch.setenv("MIOPEN_USER_DB_PATH", "/shared/miopen-db")
    monkeypatch.setenv("DNN_BENCH_TEST_PASSTHROUGH", "kept")
    parent_before = dict(os.environ)

    env = pytorch_tuning.tuned_subprocess_env(str(tmp_path))

    assert env["PYTORCH_TUNABLEOP_ENABLED"] == "1"
    assert env["PYTORCH_TUNABLEOP_TUNING"] == "1"
    assert env["PYTORCH_TUNABLEOP_FILENAME"] == str(tmp_path / "tunableop_results.csv")
    assert env["MIOPEN_USER_DB_PATH"] == str(tmp_path)
    assert env["DNN_BENCH_TEST_PASSTHROUGH"] == "kept"
    assert dict(os.environ) == parent_before


def test_snapshot_is_none_until_applied() -> None:
    assert pytorch_tuning.pytorch_environment_snapshot() is None
    pytorch_tuning.apply_pytorch_environment()
    assert (
        pytorch_tuning.pytorch_environment_snapshot()["PYTORCH_MIOPEN_SUGGEST_NHWC"]
        == "1"
    )

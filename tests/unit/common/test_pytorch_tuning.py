# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for the PyTorch kernel-selection environment."""

import os

import pytest

from dnn_benchmarking.common import pytorch_tuning


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in pytorch_tuning.ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    # No torch import: keeps cudnn.benchmark out of the process under test.
    monkeypatch.setattr(pytorch_tuning.torch_support, "module_available", lambda: False)


def test_defaults_enable_nhwc_and_aotriton_but_not_tuning() -> None:
    effective = pytorch_tuning.apply_pytorch_environment(exhaustive=False)
    assert effective["PYTORCH_MIOPEN_SUGGEST_NHWC"] == "1"
    assert effective["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] == "1"
    assert "PYTORCH_TUNABLEOP_ENABLED" not in os.environ


def test_exhaustive_enables_tunableop_with_results_under_cache_dir(tmp_path) -> None:
    effective = pytorch_tuning.apply_pytorch_environment(
        exhaustive=True, cache_dir=str(tmp_path)
    )
    assert effective["PYTORCH_TUNABLEOP_ENABLED"] == "1"
    assert effective["PYTORCH_TUNABLEOP_TUNING"] == "1"
    assert effective["PYTORCH_TUNABLEOP_FILENAME"].startswith(str(tmp_path))


def test_caller_environment_wins(monkeypatch) -> None:
    monkeypatch.setenv("PYTORCH_MIOPEN_SUGGEST_NHWC", "0")
    monkeypatch.setenv("PYTORCH_TUNABLEOP_ENABLED", "0")
    effective = pytorch_tuning.apply_pytorch_environment(exhaustive=True)
    assert effective["PYTORCH_MIOPEN_SUGGEST_NHWC"] == "0"
    assert effective["PYTORCH_TUNABLEOP_ENABLED"] == "0"


def test_never_sets_miopen_vars_shared_with_the_hipdnn_plugin() -> None:
    pytorch_tuning.apply_pytorch_environment(exhaustive=True)
    assert not [k for k in os.environ if k.startswith("MIOPEN_FIND")]


def test_snapshot_is_none_until_applied() -> None:
    assert pytorch_tuning.pytorch_environment_snapshot() is None
    pytorch_tuning.apply_pytorch_environment(exhaustive=False)
    assert pytorch_tuning.pytorch_environment_snapshot()["PYTORCH_MIOPEN_SUGGEST_NHWC"] == "1"

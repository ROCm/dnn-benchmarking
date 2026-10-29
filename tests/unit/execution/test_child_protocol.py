# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for execution.child_protocol (contract between the parent and its children)."""

import os

from dnn_benchmarking.common.pytorch_tuning import PYTORCH_SELECTION_ENV
from dnn_benchmarking.execution import child_protocol


def test_pytorch_tuned_env_forces_isolated_state(monkeypatch, tmp_path) -> None:
    """Inherited values would let the tuned child reuse or leak tuning state."""
    monkeypatch.setenv("PYTORCH_TUNABLEOP_ENABLED", "0")
    monkeypatch.setenv("PYTORCH_TUNABLEOP_FILENAME", "/stale/results.csv")
    monkeypatch.setenv("MIOPEN_USER_DB_PATH", "/shared/miopen-db")
    monkeypatch.setenv("DNN_BENCH_TEST_PASSTHROUGH", "kept")
    parent_before = dict(os.environ)

    env = child_protocol.PYTORCH_TUNED_ENV.build(str(tmp_path))

    assert env["PYTORCH_TUNABLEOP_ENABLED"] == "1"
    assert env["PYTORCH_TUNABLEOP_TUNING"] == "1"
    # The parent no longer holds these, so the child must get them directly.
    for name, value in PYTORCH_SELECTION_ENV.items():
        assert env[name] == value
    assert env["PYTORCH_TUNABLEOP_FILENAME"] == str(tmp_path / "tunableop_results.csv")
    assert env["MIOPEN_USER_DB_PATH"] == str(tmp_path)
    assert env["DNN_BENCH_TEST_PASSTHROUGH"] == "kept"
    assert dict(os.environ) == parent_before


def test_the_parent_env_satisfies_the_pytorch_child_check(tmp_path, monkeypatch):
    for name, value in child_protocol.PYTORCH_TUNED_ENV.build(str(tmp_path)).items():
        monkeypatch.setenv(name, value)

    child_protocol.PYTORCH_TUNED_ENV.check(tmp_path)


def test_pytorch_ootb_env_forces_selection_and_a_private_user_db(
    monkeypatch, tmp_path
) -> None:
    """A shell value cannot change what a PyTorch row measures; the row
    neither tunes nor shares MIOpen Find records with another run."""
    monkeypatch.setenv("PYTORCH_MIOPEN_SUGGEST_NHWC", "0")
    monkeypatch.setenv("MIOPEN_USER_DB_PATH", "/shared/miopen-db")
    monkeypatch.setenv("PYTORCH_TUNABLEOP_ENABLED", "1")
    monkeypatch.setenv("PYTORCH_TUNABLEOP_TUNING", "1")
    parent_before = dict(os.environ)

    env = child_protocol.PYTORCH_OOTB_ENV.build(str(tmp_path))

    for name, value in PYTORCH_SELECTION_ENV.items():
        assert env[name] == value
    assert env["MIOPEN_USER_DB_PATH"] == str(tmp_path)
    assert env["PYTORCH_TUNABLEOP_ENABLED"] == env["PYTORCH_TUNABLEOP_TUNING"] == "0"
    assert dict(os.environ) == parent_before

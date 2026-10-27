# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for execution.tuned_child (contract between the oracle and its children)."""

import os

from dnn_benchmarking.execution import tuned_child


def test_pytorch_tuned_env_forces_isolated_state(monkeypatch, tmp_path) -> None:
    """Inherited values would let the tuned child reuse or leak tuning state."""
    monkeypatch.setenv("PYTORCH_TUNABLEOP_ENABLED", "0")
    monkeypatch.setenv("PYTORCH_TUNABLEOP_FILENAME", "/stale/results.csv")
    monkeypatch.setenv("MIOPEN_USER_DB_PATH", "/shared/miopen-db")
    monkeypatch.setenv("DNN_BENCH_TEST_PASSTHROUGH", "kept")
    parent_before = dict(os.environ)

    env = tuned_child.PYTORCH_TUNED_ENV.build(str(tmp_path))

    assert env["PYTORCH_TUNABLEOP_ENABLED"] == "1"
    assert env["PYTORCH_TUNABLEOP_TUNING"] == "1"
    assert env["PYTORCH_TUNABLEOP_FILENAME"] == str(tmp_path / "tunableop_results.csv")
    assert env["MIOPEN_USER_DB_PATH"] == str(tmp_path)
    assert env["DNN_BENCH_TEST_PASSTHROUGH"] == "kept"
    assert dict(os.environ) == parent_before


def test_the_parent_env_satisfies_the_pytorch_child_check(tmp_path, monkeypatch):
    for name, value in tuned_child.PYTORCH_TUNED_ENV.build(str(tmp_path)).items():
        monkeypatch.setenv(name, value)

    tuned_child.PYTORCH_TUNED_ENV.check()

# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for execution.child_protocol (contract between the parent and its children)."""

import os

import pytest

from dnn_benchmarking.common.pytorch_tuning import PYTORCH_SELECTION_ENV
from dnn_benchmarking.execution import child_protocol


def test_pytorch_tuned_env_forces_its_settings_and_the_shared_tuned_db(
    monkeypatch, tmp_path
) -> None:
    """Inherited values would let the tuned child tune with other settings;
    TunableOp results stay private, the MIOpen user DB is the shared one."""
    monkeypatch.setenv("PYTORCH_TUNABLEOP_ENABLED", "0")
    monkeypatch.setenv("PYTORCH_TUNABLEOP_FILENAME", "/stale/results.csv")
    monkeypatch.setenv("MIOPEN_USER_DB_PATH", "/shared/miopen-db")
    monkeypatch.setenv("DNN_BENCH_TEST_PASSTHROUGH", "kept")
    parent_before = dict(os.environ)
    state_dir, tuned_db = tmp_path / "state", tmp_path / "tuned-db"

    env = child_protocol.PYTORCH_TUNED_ENV.build(str(state_dir), tuned_db)

    assert env["PYTORCH_TUNABLEOP_ENABLED"] == "1"
    assert env["PYTORCH_TUNABLEOP_TUNING"] == "1"
    # The parent no longer holds these, so the child must get them directly.
    for name, value in PYTORCH_SELECTION_ENV.items():
        assert env[name] == value
    assert env["PYTORCH_TUNABLEOP_FILENAME"] == str(state_dir / "tunableop_results.csv")
    assert env["MIOPEN_USER_DB_PATH"] == str(tuned_db)
    assert env["DNN_BENCH_TEST_PASSTHROUGH"] == "kept"
    assert dict(os.environ) == parent_before


@pytest.mark.parametrize(
    "spec",
    [child_protocol.HIPDNN_TUNED, child_protocol.HIPDNN_AUTOTUNE, child_protocol.PYTORCH_TUNED],
)
def test_every_tuned_child_gets_the_same_user_db_whatever_its_state_dir(
    spec, tmp_path
) -> None:
    """Tuning carries across rows: two tuned runs share one MIOpen user DB."""
    tuned_db = tmp_path / "tuned-db"

    first = spec.env.build(str(tmp_path / "row-1"), tuned_db)
    second = spec.env.build(str(tmp_path / "row-2"), tuned_db)

    assert first["MIOPEN_USER_DB_PATH"] == second["MIOPEN_USER_DB_PATH"] == str(tuned_db)


@pytest.mark.parametrize("spec", [child_protocol.HIPDNN_OOTB, child_protocol.PYTORCH_OOTB])
def test_ootb_children_keep_a_private_user_db_even_with_a_tuned_db(spec, tmp_path) -> None:
    """OOTB always selects from an empty DB, never the tuned one."""
    env = spec.env.build(str(tmp_path / "row"), tmp_path / "tuned-db")

    assert env["MIOPEN_USER_DB_PATH"] == str(tmp_path / "row")


def test_a_tuned_env_needs_the_tuned_db_dir(tmp_path) -> None:
    with pytest.raises(ValueError):
        child_protocol.HIPDNN_TUNED.env.build(str(tmp_path), None)


@pytest.mark.parametrize("spec", child_protocol.ALL_SPECS)
def test_the_parent_env_satisfies_the_child_check(spec, tmp_path, monkeypatch):
    state_dir = tmp_path / "state"
    for name, value in spec.env.build(str(state_dir), tmp_path / "tuned-db").items():
        monkeypatch.setenv(name, value)

    spec.env.check(state_dir)


def test_an_ootb_child_rejects_the_tuned_db(tmp_path, monkeypatch) -> None:
    env = child_protocol.HIPDNN_OOTB.env
    for name, value in env.build(str(tmp_path / "row"), tmp_path / "tuned-db").items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("MIOPEN_USER_DB_PATH", str(tmp_path / "tuned-db"))

    with pytest.raises(RuntimeError):
        env.check(tmp_path / "row")


@pytest.mark.parametrize("value", [None, ""])
def test_a_tuned_child_rejects_a_missing_user_db(tmp_path, monkeypatch, value) -> None:
    env = child_protocol.HIPDNN_TUNED.env
    for name, built in env.build(str(tmp_path / "row"), tmp_path / "tuned-db").items():
        monkeypatch.setenv(name, built)
    if value is None:
        monkeypatch.delenv("MIOPEN_USER_DB_PATH")
    else:
        monkeypatch.setenv("MIOPEN_USER_DB_PATH", value)

    with pytest.raises(RuntimeError):
        env.check(tmp_path / "row")


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

    env = child_protocol.PYTORCH_OOTB_ENV.build(str(tmp_path), None)

    for name, value in PYTORCH_SELECTION_ENV.items():
        assert env[name] == value
    assert env["MIOPEN_USER_DB_PATH"] == str(tmp_path)
    assert env["PYTORCH_TUNABLEOP_ENABLED"] == env["PYTORCH_TUNABLEOP_TUNING"] == "0"
    assert dict(os.environ) == parent_before

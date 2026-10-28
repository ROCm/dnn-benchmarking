# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for CLI cache directory initialization."""

import os
import stat
from pathlib import Path

import pytest

from dnn_benchmarking.cli.main import _configure_cache_env

_CACHE_VARS = (
    "XDG_CACHE_HOME",
    "AMD_COMGR_CACHE_DIR",
    "MIOPEN_USER_DB_PATH",
    "MIOPEN_CUSTOM_CACHE_DIR",
)


@pytest.fixture
def clean_cache_env(monkeypatch) -> None:
    """The autouse conftest fixture unsets the cache variables; unset this too."""
    monkeypatch.setenv("DNN_BENCH_WORKSPACE", "")
    monkeypatch.delenv("DNN_BENCH_WORKSPACE")


def test_configure_cache_env_only_fills_unset_variables(
    tmp_path: Path, monkeypatch, clean_cache_env
) -> None:
    monkeypatch.setenv("DNN_BENCH_WORKSPACE", str(tmp_path))
    for var in ("XDG_CACHE_HOME", "MIOPEN_USER_DB_PATH", "MIOPEN_CUSTOM_CACHE_DIR"):
        monkeypatch.setenv(var, "/preset")

    assert _configure_cache_env() is None  # MIOpen preset, workspace given
    assert os.environ["XDG_CACHE_HOME"] == "/preset"
    assert os.environ["MIOPEN_USER_DB_PATH"] == "/preset"
    assert Path(os.environ["AMD_COMGR_CACHE_DIR"]).parent == tmp_path
    assert Path(os.environ["AMD_COMGR_CACHE_DIR"]).is_dir()


def test_all_preset_without_workspace_allocates_nothing(
    monkeypatch, clean_cache_env
) -> None:
    for var in _CACHE_VARS:
        monkeypatch.setenv(var, "/preset")

    assert _configure_cache_env() is None
    assert all(os.environ[var] == "/preset" for var in _CACHE_VARS)


def test_miopen_state_is_fresh_per_run_and_removed_at_exit(
    tmp_path: Path, monkeypatch, clean_cache_env
) -> None:
    """A workspace keeps the kernel cache, but each run gets an empty MIOpen DB."""
    monkeypatch.setenv("DNN_BENCH_WORKSPACE", str(tmp_path))
    first = _configure_cache_env()
    assert first is not None
    first_db = Path(os.environ["MIOPEN_USER_DB_PATH"])
    assert Path(os.environ["MIOPEN_CUSTOM_CACHE_DIR"]).parent == tmp_path
    assert Path(os.environ["XDG_CACHE_HOME"]).parent == tmp_path
    assert tmp_path not in first_db.parents
    assert first_db.is_dir() and not any(first_db.iterdir())
    if os.name == "posix":
        assert stat.S_IMODE(Path(first.name).stat().st_mode) == 0o700
    (first_db / "record.ufdb.txt").write_text("stale")
    first.cleanup()
    assert not first_db.exists()

    monkeypatch.delenv("MIOPEN_USER_DB_PATH")
    second = _configure_cache_env()
    try:
        second_db = Path(os.environ["MIOPEN_USER_DB_PATH"])
        assert second_db != first_db
        assert not any(second_db.iterdir())
    finally:
        second.cleanup()

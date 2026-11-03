# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for CLI cache directory initialization."""

import os
import stat
from pathlib import Path

import pytest

from dnn_benchmarking.cli.main import _configure_cache_env, _create_cache_base


def test_cache_base_uses_configured_workspace(tmp_path: Path) -> None:
    cache_base, temporary_directory_lifetime = _create_cache_base(str(tmp_path))

    assert cache_base == tmp_path
    assert temporary_directory_lifetime is None


def test_cache_base_fallback_is_private_temporary_directory() -> None:
    cache_base, temporary_directory_lifetime = _create_cache_base(None)

    assert temporary_directory_lifetime is not None
    try:
        assert cache_base == Path(temporary_directory_lifetime.name)
        assert cache_base.is_dir()
        if os.name == "posix":
            assert stat.S_IMODE(cache_base.stat().st_mode) == 0o700
    finally:
        temporary_directory_lifetime.cleanup()


def test_configure_cache_env_only_fills_unset_variables(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("DNN_BENCH_WORKSPACE", str(tmp_path))
    for var in ("XDG_CACHE_HOME", "MIOPEN_USER_DB_PATH", "MIOPEN_CUSTOM_CACHE_DIR"):
        monkeypatch.setenv(var, "/preset")

    assert _configure_cache_env(fast=False) == []  # workspace given: no temp dir
    assert os.environ["XDG_CACHE_HOME"] == "/preset"
    assert os.environ["MIOPEN_USER_DB_PATH"] == "/preset"
    assert Path(os.environ["AMD_COMGR_CACHE_DIR"]).parent == tmp_path
    assert Path(os.environ["AMD_COMGR_CACHE_DIR"]).is_dir()


def test_fast_gets_a_new_empty_user_db_over_an_inherited_one(
    tmp_path: Path, monkeypatch
) -> None:
    """--fast replaces even a preset MIOpen user database with a new, empty
    directory that its lifetime object removes."""
    monkeypatch.setenv("DNN_BENCH_WORKSPACE", str(tmp_path))
    monkeypatch.setenv("MIOPEN_USER_DB_PATH", "/preset")

    [lifetime] = _configure_cache_env(fast=True)

    user_db = Path(os.environ["MIOPEN_USER_DB_PATH"])
    assert user_db == Path(lifetime.name)
    assert list(user_db.iterdir()) == []
    lifetime.cleanup()
    assert not user_db.exists()


def test_miopen_user_db_is_created_kept_and_wins_over_fast(
    tmp_path: Path, monkeypatch
) -> None:
    """--miopen-user-db replaces both the inherited value and the new --fast
    database, and nothing the tool removes owns it."""
    monkeypatch.setenv("DNN_BENCH_WORKSPACE", str(tmp_path / "workspace"))
    monkeypatch.setenv("MIOPEN_USER_DB_PATH", "/preset")
    user_db = tmp_path / "db" / "nested"

    lifetimes = _configure_cache_env(fast=True, miopen_user_db=user_db)

    assert os.environ["MIOPEN_USER_DB_PATH"] == str(user_db.resolve())
    assert user_db.is_dir()
    for lifetime in lifetimes:
        lifetime.cleanup()
    assert user_db.is_dir()


def test_unusable_miopen_user_db_names_the_option(tmp_path: Path) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("")
    with pytest.raises(ValueError, match=f"--miopen-user-db {blocker / 'db'}"):
        _configure_cache_env(fast=False, miopen_user_db=blocker / "db")

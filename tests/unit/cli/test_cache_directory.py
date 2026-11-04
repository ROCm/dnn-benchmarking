# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for the work-directory layout that ``cli.main`` sets up."""

import os
import re
from pathlib import Path

import pytest

from dnn_benchmarking.cli.main import _configure_cache_env
from dnn_benchmarking.common.work_dir import work_dir


def _state() -> tuple[Path, Path]:
    return (
        Path(os.environ["HIPDNN_CACHE_DIR"]),
        Path(os.environ["MIOPEN_USER_DB_PATH"]),
    )


def test_ootb_state_is_new_empty_and_removed_after_the_command(
    tmp_path: Path, monkeypatch
) -> None:
    """An OOTB run replaces inherited state with a new, empty run directory
    under ootb/temp; no earlier command can change its kernel selection."""
    monkeypatch.setenv("HIPDNN_CACHE_DIR", "/preset/hipdnn")
    monkeypatch.setenv("MIOPEN_USER_DB_PATH", "/preset/db")

    [lifetime] = _configure_cache_env(
        tmp_path, tuned=False, replace_compile_caches=False
    )

    hipdnn_cache, user_db = _state()
    run_dir = Path(lifetime.name)
    assert run_dir.parent == tmp_path / "ootb" / "temp"
    assert (hipdnn_cache.parent, user_db.parent) == (run_dir, run_dir)
    assert list(hipdnn_cache.iterdir()) == list(user_db.iterdir()) == []
    lifetime.cleanup()
    assert not run_dir.exists()


def test_two_ootb_commands_never_share_state(tmp_path: Path) -> None:
    [first] = _configure_cache_env(tmp_path, tuned=False, replace_compile_caches=False)
    first_state = _state()
    [second] = _configure_cache_env(tmp_path, tuned=False, replace_compile_caches=False)

    assert _state() != first_state
    first.cleanup()
    second.cleanup()


def test_tuned_state_is_kept_across_commands(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("MIOPEN_USER_DB_PATH", "/preset/db")

    lifetimes = _configure_cache_env(tmp_path, tuned=True, replace_compile_caches=False)
    assert lifetimes == []

    hipdnn_cache, user_db = _state()
    assert hipdnn_cache == tmp_path / "tuned" / "hipdnn-cache"
    assert user_db == tmp_path / "tuned" / "miopen-user-db"
    (user_db / "record.ufdb.txt").write_text("find result")
    _configure_cache_env(tmp_path, tuned=True, replace_compile_caches=False)
    assert (user_db / "record.ufdb.txt").read_text() == "find result"


@pytest.mark.parametrize("replace", [False, True])
def test_compile_caches_fill_unset_and_replace_only_for_an_explicit_work_dir(
    tmp_path: Path, monkeypatch, replace: bool
) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", "/preset")
    monkeypatch.delenv("AMD_COMGR_CACHE_DIR", raising=False)

    _configure_cache_env(tmp_path, tuned=True, replace_compile_caches=replace)

    xdg = tmp_path / "cache" / "xdg"
    assert os.environ["XDG_CACHE_HOME"] == (str(xdg) if replace else "/preset")
    assert Path(os.environ["AMD_COMGR_CACHE_DIR"]) == tmp_path / "cache" / "comgr"
    assert Path(os.environ["AMD_COMGR_CACHE_DIR"]).is_dir()


def test_unusable_work_dir_names_it(tmp_path: Path) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("")
    with pytest.raises(ValueError, match=re.escape(f"benchmark work dir {blocker}")):
        _configure_cache_env(blocker, tuned=False, replace_compile_caches=False)


def test_default_work_dir_is_under_the_workspace(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("DNN_BENCH_WORKSPACE", str(tmp_path))
    assert work_dir(None) == tmp_path / "work"
    assert work_dir(str(tmp_path / "w")) == tmp_path / "w"

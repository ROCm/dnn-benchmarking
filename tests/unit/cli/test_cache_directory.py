# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for the work-directory layout that ``cli.main`` sets up."""

import argparse
import os
import re
import tempfile
from pathlib import Path

import pytest

from dnn_benchmarking.cli.main import _configure_cache_env, _configure_work_dir
from dnn_benchmarking.common.work_dir import work_dir


def _state() -> tuple[Path, Path]:
    return (
        Path(os.environ["HIPDNN_CACHE_DIR"]),
        Path(os.environ["MIOPEN_USER_DB_PATH"]),
    )


def _configure(work: Path, *, tuned: bool = False, fresh: bool = False) -> list:
    return _configure_cache_env(
        work, tuned=tuned, fresh=fresh, replace_compile_caches=False
    )


@pytest.mark.parametrize("tuned", [False, True], ids=["ootb", "tuned"])
def test_kept_state_replaces_inherited_paths_and_outlives_the_command(
    tmp_path: Path, monkeypatch, tuned: bool
) -> None:
    """OOTB and tuned runs each keep their own state across commands, so a
    later command reuses the find results of an earlier one of its kind."""
    monkeypatch.setenv("HIPDNN_CACHE_DIR", "/preset/hipdnn")
    monkeypatch.setenv("MIOPEN_USER_DB_PATH", "/preset/db")

    assert _configure(tmp_path, tuned=tuned) == []

    root = tmp_path / ("tuned" if tuned else "ootb")
    hipdnn_cache, user_db = _state()
    assert (hipdnn_cache, user_db) == (root / "hipdnn-cache", root / "miopen-user-db")
    (user_db / "record.ufdb.txt").write_text("find result")
    _configure(tmp_path, tuned=tuned)
    assert (user_db / "record.ufdb.txt").read_text() == "find result"


def test_ootb_and_tuned_runs_never_share_state(tmp_path: Path) -> None:
    _configure(tmp_path, tuned=False)
    ootb = _state()
    _configure(tmp_path, tuned=True)
    assert set(ootb).isdisjoint(_state())


@pytest.mark.parametrize("tuned", [False, True], ids=["ootb", "tuned"])
def test_fresh_state_is_new_empty_outside_the_work_dir_and_removed_after(
    tmp_path: Path, monkeypatch, tuned: bool
) -> None:
    """--fresh-state wins over the kept states, so no earlier command can
    change the kernel selection; nothing of it stays in the work dir."""
    system_tmp = tmp_path / "system-tmp"
    system_tmp.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(system_tmp))
    work = tmp_path / "work"

    [first] = _configure(work, tuned=tuned, fresh=True)
    hipdnn_cache, user_db = _state()
    run_dir = Path(first.name)
    assert run_dir.parent == system_tmp
    assert (hipdnn_cache.parent, user_db.parent) == (run_dir, run_dir)
    assert list(hipdnn_cache.iterdir()) == list(user_db.iterdir()) == []

    [second] = _configure(work, tuned=tuned, fresh=True)
    assert Path(second.name) != run_dir
    first.cleanup()
    second.cleanup()
    assert list(system_tmp.iterdir()) == []
    assert not (work / "ootb").exists() and not (work / "tuned").exists()


@pytest.mark.parametrize("replace", [False, True])
def test_compile_caches_fill_unset_and_replace_only_when_asked(
    tmp_path: Path, monkeypatch, replace: bool
) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", "/preset")
    monkeypatch.delenv("AMD_COMGR_CACHE_DIR", raising=False)

    _configure_cache_env(
        tmp_path, tuned=True, fresh=False, replace_compile_caches=replace
    )

    xdg = tmp_path / "cache" / "xdg"
    assert os.environ["XDG_CACHE_HOME"] == (str(xdg) if replace else "/preset")
    assert Path(os.environ["AMD_COMGR_CACHE_DIR"]) == tmp_path / "cache" / "comgr"
    assert Path(os.environ["AMD_COMGR_CACHE_DIR"]).is_dir()


def _args(work: Path, **overrides) -> argparse.Namespace:
    values = {"benchmark_work_dir": work, "autotune": False, "fresh_state": False}
    return argparse.Namespace(**{**values, **overrides})


@pytest.mark.parametrize(
    "tuned_child, replaced", [(False, True), (True, False)], ids=["parent", "child"]
)
def test_a_tuned_child_keeps_the_compile_caches_its_parent_exported(
    tmp_path: Path, monkeypatch, tuned_child: bool, replaced: bool
) -> None:
    """The parent passes --benchmark-work-dir to its children, so only the
    command that the user started may replace inherited compile caches."""
    monkeypatch.setenv("MIOPEN_CUSTOM_CACHE_DIR", "/user/kernels")

    _configure_work_dir(_args(tmp_path), tuned_child)

    kernels = str(tmp_path / "cache" / "miopen-kernels")
    assert os.environ["MIOPEN_CUSTOM_CACHE_DIR"] == (
        kernels if replaced else "/user/kernels"
    )
    expected_state = tmp_path / ("tuned" if tuned_child else "ootb")
    assert _state()[0].parent == expected_state


def test_the_default_work_dir_is_resolved_before_xdg_cache_home_changes(
    tmp_path: Path, monkeypatch
) -> None:
    """The default reads XDG_CACHE_HOME, which the command then points into
    the work dir; later readers must still see the first resolution."""
    monkeypatch.delenv("DNN_BENCH_WORKSPACE")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    args = _args(None)

    _configure_work_dir(args, tuned_child=False)

    work = tmp_path / "xdg" / "dnn-benchmark" / "work"
    assert args.benchmark_work_dir == work
    assert work_dir(str(args.benchmark_work_dir)) == work
    assert _state()[0] == work / "ootb" / "hipdnn-cache"


def test_unusable_work_dir_names_it(tmp_path: Path) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("")
    with pytest.raises(ValueError, match=re.escape(f"benchmark work dir {blocker}")):
        _configure(blocker)


def test_default_work_dir_order(tmp_path: Path, monkeypatch) -> None:
    """Workspace, then XDG_CACHE_HOME, then a per-user directory in the
    system temporary directory; never the home directory."""
    monkeypatch.setenv("DNN_BENCH_WORKSPACE", str(tmp_path))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert work_dir(None) == tmp_path / "work"
    assert work_dir(str(tmp_path / "w")) == tmp_path / "w"

    monkeypatch.delenv("DNN_BENCH_WORKSPACE")
    assert work_dir(None) == tmp_path / "xdg" / "dnn-benchmark" / "work"

    monkeypatch.delenv("XDG_CACHE_HOME")
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "system-tmp"))
    for name in ("LOGNAME", "USER", "LNAME", "USERNAME"):
        monkeypatch.setenv(name, "alice")
    assert work_dir(None) == tmp_path / "system-tmp" / "dnn-benchmark-alice" / "work"

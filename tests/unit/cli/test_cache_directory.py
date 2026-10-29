# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for CLI cache directory initialization."""

import os
import stat
from pathlib import Path

import pytest

from dnn_benchmarking.cli.main import _configure_cache_env
from dnn_benchmarking.common.run_state import run_state_dir

_CACHE_VARS = ("XDG_CACHE_HOME", "AMD_COMGR_CACHE_DIR", "MIOPEN_CUSTOM_CACHE_DIR")


@pytest.fixture
def clean_cache_env(monkeypatch) -> None:
    """The autouse conftest fixture unsets the cache variables; unset this too."""
    monkeypatch.setenv("DNN_BENCH_WORKSPACE", "")
    monkeypatch.delenv("DNN_BENCH_WORKSPACE")


def test_configure_cache_env_only_fills_unset_variables(
    tmp_path: Path, monkeypatch, clean_cache_env
) -> None:
    monkeypatch.setenv("DNN_BENCH_WORKSPACE", str(tmp_path))
    for var in ("XDG_CACHE_HOME", "MIOPEN_CUSTOM_CACHE_DIR"):
        monkeypatch.setenv(var, "/preset")

    assert _configure_cache_env(None) is None  # workspace: nothing temporary
    assert os.environ["XDG_CACHE_HOME"] == "/preset"
    assert Path(os.environ["AMD_COMGR_CACHE_DIR"]).parent == tmp_path
    assert Path(os.environ["AMD_COMGR_CACHE_DIR"]).is_dir()


def test_all_preset_without_workspace_allocates_nothing(
    monkeypatch, clean_cache_env
) -> None:
    for var in _CACHE_VARS:
        monkeypatch.setenv(var, "/preset")

    assert _configure_cache_env(None) is None
    assert all(os.environ[var] == "/preset" for var in _CACHE_VARS)


def test_caches_without_workspace_are_private_and_removed_at_exit(
    clean_cache_env,
) -> None:
    lifetime = _configure_cache_env(None)
    assert lifetime is not None
    base = Path(lifetime.name)
    for var in _CACHE_VARS:
        assert Path(os.environ[var]).parent == base
    if os.name == "posix":
        assert stat.S_IMODE(base.stat().st_mode) == 0o700
    lifetime.cleanup()
    assert not base.exists()


def test_work_dir_keeps_caches_and_removes_temp_state_at_exit(
    tmp_path: Path, monkeypatch, clean_cache_env
) -> None:
    """--benchmark-work-dir wins over a caller's cache paths, so two calls
    share compiled kernels under ``cache``. The command's temporary state
    (every child's directory and MIOpen user DB) is in one run directory
    under ``tmp``, removed when the command ends; the caches stay. The
    parent never sets a MIOpen user DB: only children own one."""
    for var in _CACHE_VARS:
        monkeypatch.setenv(var, "/inherited")
    work_dir = tmp_path / "worker"

    lifetime = _configure_cache_env(work_dir)
    try:
        for var in _CACHE_VARS:
            assert Path(os.environ[var]).parent == work_dir / "cache"
            assert Path(os.environ[var]).is_dir()
        assert "MIOPEN_USER_DB_PATH" not in os.environ
        run_dir = Path(lifetime.name)
        assert run_dir.parent == work_dir / "tmp"
        with run_state_dir("dnn-bench-hipdnn_child-") as child_dir:
            assert Path(child_dir).parent == run_dir
    finally:
        lifetime.cleanup()
    assert list((work_dir / "tmp").iterdir()) == []
    assert (work_dir / "cache" / "miopen_kernel_cache").is_dir()

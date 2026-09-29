# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for setup_env.py's sparse rocm-libraries checkout.

The configure reads the root CMakePresets.json, so a checkout that omits it
fails before anything builds. Limit: git < 2.37 defaults to non-cone patterns,
which drop root files; the clone test fails without `sparse-checkout init
--cone` only on such a git. Newer git (CI) passes either way.
"""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

_SETUP_ENV = Path(__file__).resolve().parents[2] / "setup_env.py"


def _git(*args: str, cwd: Path) -> str:
    return subprocess.run(
        ["git", "-c", "protocol.file.allow=always", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _commit(repo: Path, *paths: str) -> str:
    """Commit `paths` (default: everything) and return the new HEAD."""
    _git("add", *(paths or ("-A",)), cwd=repo)
    _git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "c", cwd=repo)
    return _git("rev-parse", "HEAD", cwd=repo)


@pytest.fixture
def checkout(tmp_path, monkeypatch):
    """A local rocm-libraries stand-in, pinned by a superproject gitlink."""
    source = tmp_path / "source"
    for rel in (
        "CMakePresets.json",
        "cmake/a.cmake",
        "projects/hipdnn/a.txt",
        "dnn-providers/a.txt",
        "projects/other/cmake/a.cmake",
    ):
        (source / rel).parent.mkdir(parents=True, exist_ok=True)
        (source / rel).write_text(rel)
    _git("init", "-q", "-b", "main", cwd=source)
    pin = _commit(source)

    script_dir = tmp_path / "super"
    script_dir.mkdir()
    (script_dir / ".gitmodules").write_text(
        f'[submodule "rocm-libraries"]\n\tpath = rocm-libraries\n'
        f"\turl = {source}\n\tbranch = main\n"
    )
    _git("init", "-q", "-b", "main", cwd=script_dir)
    _git(
        "update-index",
        "--add",
        "--cacheinfo",
        f"160000,{pin},rocm-libraries",
        cwd=script_dir,
    )
    _commit(script_dir, ".gitmodules")

    spec = importlib.util.spec_from_file_location("setup_env_checkout", _SETUP_ENV)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "SCRIPT_DIR", script_dir)
    monkeypatch.setattr(module, "ROCM_LIBRARIES_DIR", script_dir / "rocm-libraries")
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "protocol.file.allow")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", "always")
    return module, source, pin


def test_clone_keeps_root_files_and_only_listed_dirs(checkout) -> None:
    module, _, pin = checkout
    module.Setup.__new__(module.Setup).ensure_rocm_libraries_checkout()

    root = module.ROCM_LIBRARIES_DIR
    assert (root / "CMakePresets.json").is_file()
    assert (root / "cmake/a.cmake").is_file()
    assert (root / "projects/hipdnn/a.txt").is_file()
    # A non-cone "cmake" pattern would match this nested directory.
    assert not (root / "projects/other").exists()
    assert _git("rev-parse", "HEAD", cwd=root) == pin


def test_existing_checkout_off_pin_warns(checkout, capsys) -> None:
    module, source, _ = checkout
    setup = module.Setup.__new__(module.Setup)
    setup.ensure_rocm_libraries_checkout()
    capsys.readouterr()

    (source / "cmake/b.cmake").write_text("b")
    _commit(source)
    root = module.ROCM_LIBRARIES_DIR
    _git("fetch", "-q", "origin", "main", cwd=root)
    _git("checkout", "-q", "FETCH_HEAD", cwd=root)

    setup.ensure_rocm_libraries_checkout()
    assert "not the pinned" in capsys.readouterr().err

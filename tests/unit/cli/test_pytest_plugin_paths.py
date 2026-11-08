# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for pytest dnn plugin path option parsing."""

from pathlib import Path

import pytest

from tests import conftest as project_conftest


def _plugin_dir(path: Path) -> Path:
    path.mkdir(parents=True)
    (path / "engine.so").write_bytes(b"")
    return path


def test_accepts_relative_and_absolute_entries_and_skips_blanks(
    tmp_path: Path, monkeypatch
) -> None:
    """Relative entries resolve from pytest's cwd; blank comma entries are ignored."""
    _plugin_dir(tmp_path / "plugins/relative")
    absolute = _plugin_dir(tmp_path / "plugins/absolute")
    monkeypatch.chdir(tmp_path)

    assert project_conftest._parse_plugin_paths(
        f" plugins/relative, ,{absolute}, "
    ) == [Path("plugins/relative"), absolute]


def test_rejects_empty_input() -> None:
    with pytest.raises(pytest.UsageError):
        project_conftest._parse_plugin_paths(" , ")


def test_reports_every_invalid_entry(tmp_path: Path) -> None:
    valid = _plugin_dir(tmp_path / "plugins/valid")
    missing = tmp_path / "missing"
    empty = tmp_path / "empty"
    empty.mkdir()

    with pytest.raises(pytest.UsageError) as excinfo:
        project_conftest._parse_plugin_paths(f"{valid},{missing},{empty}")

    message = str(excinfo.value)
    assert str(missing) in message
    assert str(empty) in message
    assert str(valid) not in message

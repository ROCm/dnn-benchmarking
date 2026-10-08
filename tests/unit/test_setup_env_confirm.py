# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for setup_env.py's confirmation prompt without a terminal.

Under nohup, srun or CI, stdin is closed, so the "Continue? [Y/n]" prompt
used to die with a bare EOFError traceback that never mentioned -y/--yes.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

_SETUP_ENV = Path(__file__).resolve().parents[2] / "setup_env.py"


def _run_setup(tmp_path: Path, stdin_text: str | None) -> subprocess.CompletedProcess:
    """Run setup_env.py up to its prompt; a fresh workspace means a build."""
    return subprocess.run(
        [sys.executable, str(_SETUP_ENV), "--workspace", str(tmp_path / "ws")],
        input=stdin_text,
        stdin=subprocess.DEVNULL if stdin_text is None else None,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def test_closed_stdin_fails_with_a_pointer_to_yes(tmp_path: Path) -> None:
    result = _run_setup(tmp_path, stdin_text=None)

    assert result.returncode == 1
    assert "Traceback" not in result.stderr
    assert "-y/--yes" in result.stderr
    # Failed before any work: no venv was created.
    assert not (tmp_path / "ws" / ".venv").exists()


@pytest.mark.skipif(os.name != "posix", reason="Closing fd 0 requires POSIX")
def test_closed_file_descriptor_fails_before_creating_a_venv(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(_SETUP_ENV), "--workspace", str(tmp_path / "ws")],
        preexec_fn=lambda: os.close(0),
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert result.returncode == 1
    assert "Traceback" not in result.stderr
    assert "-y/--yes" in result.stderr
    assert not (tmp_path / "ws" / ".venv").exists()


def test_a_piped_answer_is_still_read(tmp_path: Path) -> None:
    result = _run_setup(tmp_path, stdin_text="n\n")

    assert result.returncode == 0
    assert "Aborted." in result.stdout
    assert not (tmp_path / "ws" / ".venv").exists()

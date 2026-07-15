# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for the dnn-benchmarking setup script (setup_env.py).

These cover only the argument-validation contract and syntactic validity. They
never invoke a path that creates a venv or hits the network (that is CI's
install smoke, not a unit test).
"""

import subprocess
import sys
from pathlib import Path


SETUP_SCRIPT = Path(__file__).resolve().parents[3] / "setup_env.py"


def test_setup_env_parses() -> None:
    result = subprocess.run(
        [sys.executable, str(SETUP_SCRIPT), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--torch-mode" in result.stdout


def test_setup_env_rejects_unknown_torch_mode() -> None:
    result = subprocess.run(
        [sys.executable, str(SETUP_SCRIPT), "--torch-mode", "bogus"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "invalid choice" in result.stderr


def test_setup_env_rejects_unknown_flag() -> None:
    result = subprocess.run(
        [sys.executable, str(SETUP_SCRIPT), "--nope"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "unrecognized arguments" in result.stderr


def test_setup_env_is_importable() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import ast, pathlib; "
            f"ast.parse(pathlib.Path(r'{SETUP_SCRIPT}').read_text())",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr

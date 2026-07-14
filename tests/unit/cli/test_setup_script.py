# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for the dnn-benchmarking setup script."""

import os
import subprocess
from pathlib import Path


SETUP_SCRIPT = Path(__file__).resolve().parents[3] / "setup.sh"


def test_setup_script_has_valid_shell_syntax() -> None:
    result = subprocess.run(
        ["bash", "-n", str(SETUP_SCRIPT)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_setup_script_rejects_removed_force_build_flag() -> None:
    result = subprocess.run(
        ["bash", str(SETUP_SCRIPT), "--force-build"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "Unknown argument: --force-build" in result.stdout

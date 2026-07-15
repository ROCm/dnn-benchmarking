# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for CLI cache environment defaults."""

import os
import subprocess
import sys


def test_cli_cache_defaults_do_not_require_writable_workspace() -> None:
    """Importing the CLI without setup.sh must not create /workspace."""
    code = """
import os
from pathlib import Path

import dnn_benchmarking.cli.main  # noqa: F401

expected = Path('/tmp/dnn-bench-cache')
assert Path(os.environ['XDG_CACHE_HOME']) == expected / 'cache'
assert Path(os.environ['MIOPEN_USER_DB_PATH']) == expected / 'miopen_cache'
assert Path(os.environ['MIOPEN_CUSTOM_CACHE_DIR']) == expected / 'miopen_cache'
assert Path(os.environ['AMD_COMGR_CACHE_DIR']) == expected / 'comgr_cache'
for value in (
    os.environ['XDG_CACHE_HOME'],
    os.environ['MIOPEN_USER_DB_PATH'],
    os.environ['MIOPEN_CUSTOM_CACHE_DIR'],
    os.environ['AMD_COMGR_CACHE_DIR'],
):
    assert Path(value).is_dir()
"""
    env = os.environ.copy()
    for key in (
        "DNN_BENCH_WORKSPACE",
        "XDG_CACHE_HOME",
        "MIOPEN_USER_DB_PATH",
        "MIOPEN_CUSTOM_CACHE_DIR",
        "AMD_COMGR_CACHE_DIR",
    ):
        env.pop(key, None)

    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr

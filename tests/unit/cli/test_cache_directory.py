# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for CLI cache directory initialization."""

import os
import stat
from pathlib import Path

from dnn_benchmarking.cli.main import _create_cache_base


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

# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for setup_env helpers.

setup_env.py is a stdlib-only script run by the *system* interpreter
before the venv exists, so it isn't importable as a package module; load
it by path.
"""

import importlib.util
import os
import sys
from pathlib import Path

import pytest

_SETUP_ENV = Path(__file__).resolve().parents[2] / "setup_env.py"


def _load_setup_env():
    spec = importlib.util.spec_from_file_location("setup_env_under_test", _SETUP_ENV)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


setup_env = _load_setup_env()


@pytest.fixture
def wheel_libs(tmp_path):
    """A core/devel pair shaped like the ROCm wheels: hardlinked duplicates."""
    core = tmp_path / "_rocm_sdk_core" / "lib"
    devel = tmp_path / "_rocm_sdk_devel" / "lib"
    core.mkdir(parents=True)
    devel.mkdir(parents=True)

    real = core / "librocprofiler-sdk.so.1"
    real.write_bytes(b"elf")
    # The wheels ship the devel copies as hardlinks under three names.
    for name in (
        "librocprofiler-sdk.so",
        "librocprofiler-sdk.so.1",
        "librocprofiler-sdk.so.1.3.3",
    ):
        os.link(real, devel / name)
    return core, devel


@pytest.mark.skipif(
    sys.platform == "win32", reason="hardlink/symlink semantics differ on Windows"
)
class TestUnifyRocprofilerLibs:
    def test_relinks_hardlinked_duplicates_to_core(self, wheel_libs):
        """Every devel duplicate must resolve to the core path, since
        rocprofiler-register compares registered libraries by path."""
        core, devel = wheel_libs
        assert setup_env.unify_rocprofiler_libs(core, devel) == 3
        for dup in devel.glob("librocprofiler-sdk.so*"):
            assert dup.is_symlink()
            assert dup.resolve() == (core / "librocprofiler-sdk.so.1").resolve()

    def test_is_idempotent(self, wheel_libs):
        core, devel = wheel_libs
        setup_env.unify_rocprofiler_libs(core, devel)
        assert setup_env.unify_rocprofiler_libs(core, devel) == 0

    def test_leaves_distinct_builds_alone(self, wheel_libs):
        """A devel library that is a different file must never be swapped for
        the core one — that would silently change which build gets loaded."""
        core, devel = wheel_libs
        other = devel / "librocprofiler-sdk-rocpd.so.1"
        other.write_bytes(b"a different build")
        (core / "librocprofiler-sdk-rocpd.so.1").write_bytes(b"core build")

        setup_env.unify_rocprofiler_libs(core, devel)

        assert not other.is_symlink()
        assert other.read_bytes() == b"a different build"

    def test_ignores_non_rocprofiler_duplicates(self, wheel_libs):
        """Scope is the rocprofiler registration conflict, not every wheel
        duplicate; unrelated hardlinks stay untouched."""
        core, devel = wheel_libs
        hip = core / "libamdhip64.so.7"
        hip.write_bytes(b"hip")
        os.link(hip, devel / "libamdhip64.so.7")

        setup_env.unify_rocprofiler_libs(core, devel)

        assert not (devel / "libamdhip64.so.7").is_symlink()

    def test_missing_prefix_is_a_no_op(self, tmp_path):
        assert setup_env.unify_rocprofiler_libs(tmp_path / "nope", tmp_path) == 0

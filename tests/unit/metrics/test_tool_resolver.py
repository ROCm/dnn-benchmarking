# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for the ROCm tool binary resolver.

Why this matters: a profiler from one ROCm install cannot profile a
process running another one. With the rocm-sdk wheels installed (the
ROCm torch and hipDNN link against), ``/opt/rocm/bin/rocprofv3`` drags
the system LLVM runtime into the workload and it dies loading
``libamd_comgr.so.3``. So the wheel's console-script shim wins whenever
the wheels are present, and the system install only serves the
system-ROCm case.
"""

import os
import stat

import pytest

from dnn_benchmarking.metrics import _tool_resolver


@pytest.fixture
def fake_rocm_root(tmp_path, monkeypatch):
    """Build a fake $ROCM_PATH layout with an executable bin/<tool>."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    monkeypatch.setenv("ROCM_PATH", str(tmp_path))
    return bin_dir


@pytest.fixture
def no_wheel_rocm(monkeypatch):
    """Pretend the rocm-sdk wheels aren't installed (system-ROCm host)."""
    monkeypatch.setattr(_tool_resolver, "_wheel_rocm_installed", lambda: False)


@pytest.fixture
def wheel_rocm(monkeypatch):
    """Pretend the rocm-sdk wheels provide this process's ROCm runtime."""
    monkeypatch.setattr(_tool_resolver, "_wheel_rocm_installed", lambda: True)


def _make_executable(path):
    path.write_text("#!/bin/sh\necho hi\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


class TestResolveRocmTool:
    def test_prefers_wheel_shim_over_rocm_path(
        self, wheel_rocm, fake_rocm_root, monkeypatch
    ):
        """With the wheels installed, the shim next to sys.executable wins
        even when $ROCM_PATH/bin/<tool> exists — the system tool would
        crash the wheel-ROCm workload it is supposed to profile."""
        _make_executable(fake_rocm_root / "rocprofv3")
        seen = {}

        def fake_which(name, path=None):
            seen["path"] = path
            return "/venv/bin/rocprofv3" if path else "/usr/bin/rocprofv3"

        monkeypatch.setattr(_tool_resolver.shutil, "which", fake_which)
        assert _tool_resolver.resolve_rocm_tool("rocprofv3") == "/venv/bin/rocprofv3"
        assert seen["path"] == os.path.dirname(_tool_resolver.sys.executable)

    def test_wheel_without_shim_falls_back_and_warns(
        self, wheel_rocm, fake_rocm_root, monkeypatch, capsys
    ):
        """Wheels installed but no shim: the system tool is all we have,
        and it is the mismatched-install case the user must be told about."""
        from dnn_benchmarking.metrics._diagnostic import reset as reset_warn_once

        reset_warn_once()
        rocm_tool = fake_rocm_root / "rocprofv3"
        _make_executable(rocm_tool)
        monkeypatch.setattr(
            _tool_resolver.shutil, "which", lambda _name, path=None: None
        )
        assert _tool_resolver.resolve_rocm_tool("rocprofv3") == str(rocm_tool)
        captured = capsys.readouterr()
        assert "[metrics:tool_resolver]" in captured.err
        assert "cannot profile this process" in captured.err

    def test_prefers_rocm_path_over_path_resolution(
        self, no_wheel_rocm, fake_rocm_root, monkeypatch
    ):
        """Without the wheels, $ROCM_PATH/bin/<tool> beats a PATH hit —
        a stray tool elsewhere on PATH is the less coherent choice."""
        rocm_tool = fake_rocm_root / "rocprofv3"
        _make_executable(rocm_tool)
        monkeypatch.setattr(
            _tool_resolver.shutil,
            "which",
            lambda _name, path=None: "/some/other/bin/rocprofv3",
        )
        assert _tool_resolver.resolve_rocm_tool("rocprofv3") == str(rocm_tool)

    def test_falls_back_to_path_when_rocm_path_absent(
        self, no_wheel_rocm, tmp_path, monkeypatch
    ):
        """If $ROCM_PATH/bin/<tool> doesn't exist, fall back to PATH so the
        tool is still usable on hosts where the layout differs."""
        monkeypatch.setenv("ROCM_PATH", str(tmp_path))  # tmp_path/bin doesn't exist
        monkeypatch.setattr(
            _tool_resolver.shutil,
            "which",
            lambda _name, path=None: "/somewhere/else/rocprofv3",
        )
        assert (
            _tool_resolver.resolve_rocm_tool("rocprofv3") == "/somewhere/else/rocprofv3"
        )

    def test_returns_none_when_neither_present(
        self, no_wheel_rocm, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("ROCM_PATH", str(tmp_path))
        monkeypatch.setattr(
            _tool_resolver.shutil, "which", lambda _name, path=None: None
        )
        assert _tool_resolver.resolve_rocm_tool("rocprofv3") is None

    @pytest.mark.skipif(
        os.name == "nt",
        reason="executable detection is the Unix +x bit; os.access(X_OK) on "
        "Windows is not chmod-based, so a 0o644 file still reads as executable",
    )
    def test_skips_rocm_path_when_file_is_not_executable(
        self, no_wheel_rocm, fake_rocm_root, monkeypatch
    ):
        """A non-executable file at $ROCM_PATH/bin/<tool> shouldn't be picked.
        Otherwise a stray data file would mask a working PATH binary."""
        rocm_tool = fake_rocm_root / "rocprofv3"
        rocm_tool.write_text("not executable")  # no +x bit
        os.chmod(rocm_tool, 0o600)
        monkeypatch.setattr(
            _tool_resolver.shutil,
            "which",
            lambda _name, path=None: "/fallback/rocprofv3",
        )
        assert _tool_resolver.resolve_rocm_tool("rocprofv3") == "/fallback/rocprofv3"

    def test_default_rocm_root_is_opt_rocm(self, monkeypatch):
        """When ROCM_PATH isn't set, default to /opt/rocm — the ROCm install
        location on every standard host."""
        monkeypatch.delenv("ROCM_PATH", raising=False)
        assert _tool_resolver._preferred_rocm_root() == "/opt/rocm"

    def test_path_fallback_warns_loudly(
        self, no_wheel_rocm, tmp_path, monkeypatch, capsys
    ):
        """The PATH-fallback case is where the resolved tool may belong to
        an unrelated ROCm install; leave a breadcrumb so a crashed or
        empty profiling pass points back here."""
        from dnn_benchmarking.metrics._diagnostic import reset as reset_warn_once

        reset_warn_once()
        monkeypatch.setenv("ROCM_PATH", str(tmp_path))  # tmp_path/bin missing
        monkeypatch.setattr(
            _tool_resolver.shutil,
            "which",
            lambda _name, path=None: "/some/venv/bin/rocprofv3",
        )
        result = _tool_resolver.resolve_rocm_tool("rocprofv3")
        assert result == "/some/venv/bin/rocprofv3"
        captured = capsys.readouterr()
        assert "[metrics:tool_resolver]" in captured.err
        assert "PATH-resolved" in captured.err
        assert "/some/venv/bin/rocprofv3" in captured.err

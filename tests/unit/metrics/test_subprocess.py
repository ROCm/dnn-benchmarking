# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for the capped subprocess helper.

The regression these guard: profiler front-ends exec the workload as a
grandchild that inherits the captured pipes. Plain
``subprocess.run(capture_output=True, timeout=...)`` kills only the
front-end and then waits on those pipes forever, so a wedged rocprofv3
hangs the whole benchmark instead of skipping one pass.
"""

import subprocess
import sys
import time

import psutil
import pytest

from dnn_benchmarking.metrics import _subprocess as _subprocess_mod
from dnn_benchmarking.metrics._subprocess import run_capped


def _alive(pid: int) -> bool:
    # psutil, not os.kill(pid, 0): signal 0 is not a thing on Windows
    # (WinError 87). psutil is already a runtime dependency.
    return psutil.pid_exists(pid)


# Spawns a grandchild that outlives it and holds the inherited pipes,
# exactly like rocprofv3/perf/rocprof-compute do with the workload.
_SPAWNS_ORPHAN = (
    "import subprocess, sys, time; "
    "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)']); "
    "time.sleep(30)"
)

# The grandchild here calls setsid(), escaping the process group we kill.
# It still holds the inherited pipes, so the reap after the kill must be
# bounded or the "timeout" never returns.
_SPAWNS_ESCAPING_ORPHAN = (
    "import subprocess, sys, time; "
    "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'], "
    "start_new_session=True); "
    "time.sleep(30)"
)


class TestRunCapped:
    def test_timeout_fires_despite_surviving_grandchild(self):
        start = time.monotonic()
        with pytest.raises(subprocess.TimeoutExpired):
            run_capped([sys.executable, "-c", _SPAWNS_ORPHAN], 2)
        # Generous bound: the point is "returns promptly", not the exact
        # kill latency. Pre-fix this waited out the 30 s grandchild.
        assert time.monotonic() - start < 15

    def test_timeout_fires_when_grandchild_escapes_the_group(self):
        """A grandchild in its own session survives the group kill and keeps
        the pipes open. The post-kill reap must be bounded, or the timeout
        never returns and we are back to the original forever-hang."""
        start = time.monotonic()
        with pytest.raises(subprocess.TimeoutExpired):
            run_capped([sys.executable, "-c", _SPAWNS_ESCAPING_ORPHAN], 2)
        assert time.monotonic() - start < 20

    def test_returns_completed_process_on_success(self):
        proc = run_capped(
            [sys.executable, "-c", "import sys; print('out'); sys.exit(3)"], 60
        )
        assert proc.returncode == 3
        assert proc.stdout.strip() == "out"
        assert proc.stderr == ""

    def test_none_timeout_disables_the_cap(self):
        proc = run_capped([sys.executable, "-c", "print('ok')"], None)
        assert proc.returncode == 0
        assert proc.stdout.strip() == "ok"

    def test_missing_binary_raises_oserror(self, tmp_path):
        with pytest.raises(OSError):
            run_capped([str(tmp_path / "definitely-not-a-binary")], 10)

    def test_cancellation_kills_the_tool(self, monkeypatch):
        """KeyboardInterrupt/SystemExit must tear the tool down too. The
        tool runs in its own session, so a Ctrl-C in this process never
        reaches it — without explicit cleanup rocprofv3 and the GPU
        workload under it would outlive the cancelled run."""
        real_communicate = subprocess.Popen.communicate
        interrupted = []

        def interrupt_once(self, input=None, timeout=None):
            if not interrupted:
                interrupted.append(self.pid)
                raise KeyboardInterrupt
            return real_communicate(self, input=input, timeout=timeout)

        monkeypatch.setattr(subprocess.Popen, "communicate", interrupt_once)
        with pytest.raises(KeyboardInterrupt):
            run_capped([sys.executable, "-c", _SPAWNS_ORPHAN], 60)

        pid = interrupted[0]
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and _alive(pid):
            time.sleep(0.05)
        assert not _alive(pid), f"tool pid {pid} survived cancellation"


class TestKillTree:
    def test_windows_kills_the_whole_tree(self, monkeypatch):
        """taskkill /T, not Popen.kill(): on Windows the descendants keep
        the captured pipes open, and closing a pipe mid-read blocks until
        the last writer exits (CI measured 30s against a 2s cap)."""
        calls = []
        monkeypatch.setattr(_subprocess_mod.os, "name", "nt")
        monkeypatch.setattr(
            _subprocess_mod.subprocess, "run", lambda argv, **kw: calls.append(argv)
        )

        class FakeProc:
            pid = 4321

            def kill(self):
                calls.append("kill")

        _subprocess_mod._kill_tree(FakeProc())

        assert calls == [["taskkill", "/F", "/T", "/PID", "4321"]]

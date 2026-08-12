# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Capped subprocess execution for the profiling passes.

``subprocess.run(capture_output=True, timeout=...)`` does not bound the
wall clock when the child spawns its own children. On expiry it kills
only the direct child and then keeps waiting for stdout/stderr to
close, and every profiler front-end here (rocprofv3, perf,
rocprof-compute) execs the workload as a grandchild that inherits those
pipes. A wedged rocprofv3 therefore hangs the benchmark forever:
measured on MI210 with ``--profiling-timeout 90``, the pass was still
running at 500 s with no timeout warning.

``run_capped`` puts the tool in its own process group and kills the
group, so the pipes close and ``TimeoutExpired`` actually propagates.
"""

import os
import signal
import subprocess
from typing import List, Optional

# Grace period for reaping the killed process group. Bounded so a
# grandchild that escaped the group (its own session) can't hold the
# pipes — and us — open forever.
_REAP_TIMEOUT_S = 5


def _kill_tree(proc: "subprocess.Popen[str]") -> None:
    """Kill the tool and every process it spawned.

    Killing only the direct child is not enough: its descendants keep the
    captured pipes open, and closing a pipe whose read is still in flight
    blocks until the last writer exits — which is exactly the hang this
    module exists to prevent (Windows CI measured 30s against a 2s cap).
    POSIX kills the session's process group; Windows has no process
    groups worth the name, so walk the tree with ``taskkill /T``.
    """
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=_REAP_TIMEOUT_S,
            )
            return
        except (OSError, subprocess.SubprocessError):
            # taskkill missing or wedged; the direct child is still worth
            # killing even though its descendants will survive.
            pass
        try:
            proc.kill()
        except OSError:
            pass
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except OSError:
        try:
            proc.kill()
        except OSError:
            pass


def run_capped(
    argv: List[str], timeout_s: Optional[int]
) -> "subprocess.CompletedProcess[str]":
    """Run ``argv``, capturing text output, under a real wall-clock cap.

    Drop-in for ``subprocess.run(argv, capture_output=True, text=True,
    check=False, timeout=timeout_s)`` — same return value, same
    exceptions — except that the timeout also applies to grandchildren.
    Any abnormal exit tears the process tree down, so cancelling the
    caller cannot leave a profiler and its GPU workload running.

    Args:
        argv: Command to run.
        timeout_s: Wall-clock budget in seconds; ``None`` disables it.

    Returns:
        The completed process with ``stdout``/``stderr`` as text.

    Raises:
        subprocess.TimeoutExpired: after the process tree is killed.
        KeyboardInterrupt, SystemExit: likewise, after the kill.
        OSError: if the tool can't be spawned.
    """
    with subprocess.Popen(
        argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    ) as proc:
        try:
            stdout, stderr = proc.communicate(timeout=timeout_s)
        except BaseException:
            # Every abnormal exit, not just TimeoutExpired: the tool runs
            # in its own session, so a Ctrl-C or SystemExit in this
            # process never reaches it and would leave rocprofv3 (and the
            # GPU workload under it) running after we're gone.
            _kill_tree(proc)
            try:
                # Bounded reap: a grandchild that called setsid() escaped
                # the group and can still hold the pipes, and an unbounded
                # wait here would reintroduce the very hang we prevent.
                proc.communicate(timeout=_REAP_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                pass
            raise
    return subprocess.CompletedProcess(argv, proc.returncode, stdout, stderr)

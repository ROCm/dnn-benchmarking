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

``run_tool`` is the one skeleton every profiling source shares: spawn
under the cap, turn timeout / spawn failure / nonzero exit into the
slice keys (``skipped``, ``returncode``, ``error_tail``) and a warning
naming the (graph, engine) it happened on.
"""

import os
import signal
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ._diagnostic import warn_once

# Lines of captured output kept in ``error_tail``.
_TAIL_LINES = 40

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
    argv: List[str],
    timeout_s: Optional[int],
    env: Optional[Dict[str, str]] = None,
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
        env: Child environment; ``None`` passes a copy of ``os.environ``.

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
        # Explicit env, deliberately not inherited: ``os.environ`` is the
        # Python-startup snapshot, while the C-level environ also holds the
        # variables HIP set with setenv() during its init in this process.
        # Passing those to the profiled child makes rocprofv3 abort (rc=-6).
        env=env if env is not None else dict(os.environ),
    ) as proc:
        try:
            stdout, stderr = proc.communicate(timeout=timeout_s)
        except BaseException as exc:
            # Every abnormal exit, not just TimeoutExpired: the tool runs
            # in its own session, so a Ctrl-C or SystemExit in this
            # process never reaches it and would leave rocprofv3 (and the
            # GPU workload under it) running after we're gone.
            _kill_tree(proc)
            try:
                # Bounded reap: a grandchild that called setsid() escaped
                # the group and can still hold the pipes, and an unbounded
                # wait here would reintroduce the very hang we prevent.
                out, err = proc.communicate(timeout=_REAP_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                out, err = None, None
            if isinstance(exc, subprocess.TimeoutExpired):
                # Keep what the tool printed before it wedged: the last
                # stderr lines are what diagnose the hang.
                exc.output, exc.stderr = out, err
            raise
    return subprocess.CompletedProcess(argv, proc.returncode, stdout, stderr)


def _tail(*streams: Optional[str]) -> Optional[str]:
    """Last lines of the first non-empty stream (stderr first, then stdout)."""
    for text in streams:
        if text and text.strip():
            return "\n".join(text.strip().splitlines()[-_TAIL_LINES:])
    return None


def run_tool(
    source: str,
    binary: Optional[str],
    args: List[str],
    out_dir: Path,
    timeout_s: int,
    context: str,
) -> Tuple[Optional["subprocess.CompletedProcess[str]"], Dict[str, Any]]:
    """Run ``[binary, *args]`` for one profiling source.

    Returns ``(proc, fields)``. ``proc`` is None when the tool never
    completed (missing binary, spawn failure, timeout); ``fields`` then
    carries ``skipped``; on a nonzero exit ``proc`` is returned and
    ``fields`` carries ``returncode``; on success it is empty. A timeout
    or nonzero exit adds ``error_tail`` when the tool printed anything.

    ``timeout_s`` of 0 disables the cap. ``context`` (``graph/engine``)
    goes into every warning so per-engine failures are not deduplicated
    away across a suite.
    """
    if binary is None:
        # Environment-level: one warning per source, not per engine.
        warn_once(source, "profiling tool not found; pass skipped")
        return None, {"skipped": "profiling tool not found"}
    name = Path(binary).name
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        proc = run_capped([binary, *args], timeout_s or None)
    except subprocess.TimeoutExpired as e:
        msg = f"{name} timed out after {timeout_s}s"
        warn_once(source, f"{context}: {msg}; raise --profiling-timeout to extend")
        fields: Dict[str, Any] = {"skipped": msg}
        tail = _tail(e.stderr, e.output)
        if tail:
            fields["error_tail"] = tail
        return None, fields
    except (OSError, subprocess.SubprocessError) as e:
        msg = f"{name} invocation failed: {e}"
        warn_once(source, f"{context}: {msg}")
        return None, {"skipped": msg}
    if proc.returncode != 0:
        warn_once(source, f"{context}: {name} exited {proc.returncode}")
        fields = {"returncode": proc.returncode}
        tail = _tail(proc.stderr, proc.stdout)
        if tail:
            fields["error_tail"] = tail
        return proc, fields
    return proc, {}

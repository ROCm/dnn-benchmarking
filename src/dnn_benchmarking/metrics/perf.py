# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Linux perf stat wrapper for CPU-side hardware counters.

Wraps the workload in ``perf stat -x, -e <events>``, parses the CSV
output, and folds CPU cycles/instructions/IPC into ``extra_metrics["perf"]``.

Two tiers of events:

* User-space (``cycles:u``, ``instructions:u``) — always available to
  the running user.
* Kernel-space (``cycles:k``, ``instructions:k``) — require
  ``/proc/sys/kernel/perf_event_paranoid <= 1``, unless this process
  holds ``CAP_PERFMON``/``CAP_SYS_ADMIN``, which bypass the sysctl
  entirely. Dropped when neither holds; the recorded paranoid value
  tells the user why the kernel fields are None.

Missing perf binary is a single warn_once + skipped metrics dict;
nothing about ``--perf`` is fatal.
"""

import glob
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ._artifact_paths import DEFAULT_PROFILING_TIMEOUT_S
from ._diagnostic import warn_once
from ._subprocess import run_capped

PERF_EVENTS_USER = [
    "cycles:u",
    "instructions:u",
    "task-clock",
    "context-switches",
    "page-faults",
]

PERF_EVENTS_KERNEL = [
    "cycles:k",
    "instructions:k",
]


def _read_perf_paranoid() -> Optional[int]:
    """Return the kernel perf_event_paranoid setting, or None if unreadable."""
    try:
        with open("/proc/sys/kernel/perf_event_paranoid", "r") as fh:
            return int(fh.read().strip())
    except (OSError, ValueError):
        return None


# CAP_SYS_ADMIN and CAP_PERFMON bypass perf_event_paranoid outright.
_CAP_SYS_ADMIN = 21
_CAP_PERFMON = 38


def _has_perfmon_capability() -> bool:
    """True iff this process may open kernel events whatever the sysctl says.

    Benchmarking hosts routinely run the workload as root in a
    privileged container, where perf_event_paranoid is inherited from
    the (unprivileged-facing) host and says nothing about what we are
    allowed to do. Reading the sysctl alone drops half the counters on
    exactly the machines that can collect them.
    """
    try:
        with open("/proc/self/status", "r") as fh:
            for line in fh:
                if line.startswith("CapEff:"):
                    caps = int(line.split()[1], 16)
                    return bool(caps & ((1 << _CAP_SYS_ADMIN) | (1 << _CAP_PERFMON)))
    except (OSError, ValueError, IndexError):
        pass
    return False


def _kernel_events_allowed(paranoid: Optional[int]) -> bool:
    # Documented kernel rule: cycles:k / instructions:k require
    # paranoid <= 1. paranoid 2 blocks kernel events; 3 blocks all
    # unprivileged tracing; 4 blocks even cycles:u on some kernels.
    if _has_perfmon_capability():
        return True
    return paranoid is not None and paranoid <= 1


def _perf_is_runnable(binary: str) -> bool:
    """True iff ``binary --version`` succeeds.

    Distro ``perf`` is a wrapper that dispatches to a per-kernel build
    and exits non-zero when no ``linux-tools`` package matches the
    running kernel, so resolving the name proves nothing. The probe
    costs ~10 ms against a profiled workload run.
    """
    try:
        probe = subprocess.run(
            [binary, "--version"], capture_output=True, text=True, timeout=30
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return probe.returncode == 0


def _installed_perf_binaries() -> List[str]:
    """Real perf builds the distro shipped, newest version string first.

    Ubuntu installs one per kernel version under ``/usr/lib/linux-tools*``.
    In a container the running (host) kernel usually has no match — but
    the image's own build still counts the hardware events we ask for.
    """
    found = set()
    for pattern in ("/usr/lib/linux-tools/*/perf", "/usr/lib/linux-tools-*/perf"):
        found.update(glob.glob(pattern))
    return sorted(found, reverse=True)


def _resolve_perf() -> Optional[Tuple[str, Optional[str]]]:
    """Return ``(binary, substitution note)``, or None if none runs.

    Prefers whatever ``perf`` resolves to normally; only when that can't
    run does it reach for an installed build for a different kernel, and
    it says so in the slice rather than passing the numbers off as
    business as usual.
    """
    primary = shutil.which("perf")
    if primary is not None and _perf_is_runnable(primary):
        return primary, None
    for candidate in _installed_perf_binaries():
        if candidate != primary and _perf_is_runnable(candidate):
            return candidate, (
                f"{primary or 'perf'} does not run on kernel "
                f"{platform.release()}; used {candidate} instead"
            )
    return None


def _build_argv(
    events: List[str],
    csv_path: Path,
    inner_argv: List[str],
    binary: str,
) -> List[str]:
    return [
        binary,
        "stat",
        "-x,",
        "-o",
        str(csv_path),
        "-e",
        ",".join(events),
        "--",
        *inner_argv,
    ]


def _parse_perf_csv(csv_path: Path) -> Dict[str, Any]:
    """Parse the seven-column ``perf stat -x,`` output.

    Format per row: ``<value>,<unit>,<event>,<run-time-ns>,<percent>,<metric>,<metric-unit>``
    Header lines (``# ...``) and blanks are skipped. Values that report
    ``<not counted>`` or ``<not supported>`` map to None for that event.
    """
    out: Dict[str, Any] = {}
    if not csv_path.exists():
        return out
    for raw in csv_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        # perf inserts the message in the value column when it can't
        # measure (commas in messages would break naive split, so guard).
        cols = line.split(",")
        if len(cols) < 3:
            continue
        value_str, _unit, event = cols[0], cols[1], cols[2]
        if value_str.startswith("<"):
            out[event] = None
            continue
        try:
            value = float(value_str)
            out[event] = int(value) if value.is_integer() else value
        except ValueError:
            out[event] = None
    return out


def run(
    inner_argv: List[str],
    out_dir: Path,
    timeout_s: int = DEFAULT_PROFILING_TIMEOUT_S,
) -> Dict[str, Any]:
    """Run perf stat, parse CSV, return extra_metrics slice. Never raises.

    ``timeout_s`` bounds the perf subprocess; ``0`` disables.
    """
    resolved = _resolve_perf()
    if resolved is None:
        warn_once("perf", "no runnable perf binary found; skipping CPU counters")
        return {"perf": {"skipped": "no runnable perf binary found"}}
    binary, substitution = resolved
    if substitution:
        warn_once("perf", substitution)

    paranoid = _read_perf_paranoid()
    kernel_ok = _kernel_events_allowed(paranoid)
    events = list(PERF_EVENTS_USER)
    if kernel_ok:
        events.extend(PERF_EVENTS_KERNEL)

    # No hostname subdir: perf is a single CSV, the orchestrator's
    # per-(graph, engine, source) subdir already disambiguates runs,
    # and the user-facing path stays short.
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "perf.csv"
    argv = _build_argv(events, csv_path, inner_argv, binary)

    subprocess_timeout = timeout_s or None
    try:
        proc = run_capped(argv, subprocess_timeout)
    except subprocess.TimeoutExpired:
        warn_once("perf", f"perf invocation timed out after {subprocess_timeout}s")
        return {
            "perf": {
                "skipped": f"perf invocation timed out after {subprocess_timeout}s"
            }
        }
    except (OSError, subprocess.SubprocessError) as e:
        warn_once("perf", f"perf invocation failed: {e}")
        return {"perf": {"skipped": f"perf invocation failed: {e}"}}

    parsed = _parse_perf_csv(csv_path)

    def _get(name: str) -> Optional[float]:
        v = parsed.get(name)
        return v if isinstance(v, (int, float)) else None

    cycles_user = _get("cycles:u")
    instr_user = _get("instructions:u")
    ipc_user: Optional[float] = None
    if cycles_user and instr_user is not None and cycles_user > 0:
        ipc_user = float(instr_user) / float(cycles_user)

    result: Dict[str, Any] = {
        "cycles_user": cycles_user,
        "instructions_user": instr_user,
        "ipc_user": ipc_user,
        "cycles_kernel": _get("cycles:k") if kernel_ok else None,
        "instructions_kernel": _get("instructions:k") if kernel_ok else None,
        "task_clock_ms": _get("task-clock"),
        "context_switches": _get("context-switches"),
        "page_faults": _get("page-faults"),
        "kernel_perf_paranoid": paranoid,
        "binary": binary,
    }
    if substitution:
        # Counters from a perf built for another kernel are still worth
        # having, but the reader must be able to see that is what they are.
        result["binary_substituted"] = substitution
    if csv_path.exists():
        # Only advertise the artifact when perf actually wrote it: on a
        # launch failure the directory exists but the CSV does not, and a
        # consumer opening the advertised path would just get ENOENT.
        result["csv_path"] = str(csv_path)
    if not kernel_ok:
        result["kernel_events_skipped_reason"] = (
            f"perf_event_paranoid={paranoid} (kernel events need <= 1, "
            "or CAP_PERFMON)"
        )
    if proc.returncode != 0:
        result["returncode"] = proc.returncode
        tail = "\n".join(proc.stderr.strip().splitlines()[-20:])
        if tail:
            result["error_tail"] = tail
        warn_once(
            "perf",
            f"perf stat exited {proc.returncode}; partial counters may be present",
        )
    return {"perf": result}

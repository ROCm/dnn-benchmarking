# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""rocprofv3 kernel/memcpy trace export.

Wraps the workload in ``rocprofv3 --kernel-trace --memory-copy-trace
--output-format pftrace`` and records the resulting artifact path. The
``.pftrace`` file opens directly in https://ui.perfetto.dev.
"""

import subprocess
from pathlib import Path
from typing import Any, Dict, List

from ._artifact_paths import (
    DEFAULT_PROFILING_TIMEOUT_S,
    find_first,
    flatten_hostname_dir,
)
from ._diagnostic import warn_once
from ._subprocess import run_capped
from ._tool_resolver import resolve_rocm_tool


def _build_argv(
    out_dir: Path,
    inner_argv: List[str],
    rocprofv3_binary: str,
) -> List[str]:
    # `-o results` strips rocprofv3's default `<pid>_` filename prefix;
    # ``flatten_hostname_dir`` then strips the `<hostname>/` segment
    # after the subprocess returns.
    return [
        rocprofv3_binary,
        "--kernel-trace",
        "--memory-copy-trace",
        "--output-format",
        "pftrace",
        "-d",
        str(out_dir),
        "-o",
        "results",
        "--",
        *inner_argv,
    ]


def run(
    inner_argv: List[str],
    out_dir: Path,
    timeout_s: int = DEFAULT_PROFILING_TIMEOUT_S,
) -> Dict[str, Any]:
    """Run rocprofv3 trace and return the extra_metrics slice.

    ``timeout_s`` is the per-subprocess wall-clock budget; ``0`` disables it.
    Never raises.
    """
    rocprofv3_binary = resolve_rocm_tool("rocprofv3")
    if rocprofv3_binary is None:
        warn_once("rocprof_trace", "rocprofv3 binary not found; skipping trace pass")
        return {
            "trace": {
                "format": "pftrace",
                "skipped": "rocprofv3 binary not found",
            }
        }

    # rocprofv3 nests its own <hostname>/<pid>_results.* under -d. Pass
    # out_dir directly so the path doesn't double the hostname segment.
    out_dir.mkdir(parents=True, exist_ok=True)

    argv = _build_argv(out_dir, inner_argv, rocprofv3_binary)

    subprocess_timeout = timeout_s or None
    try:
        proc = run_capped(argv, subprocess_timeout)
    except subprocess.TimeoutExpired:
        warn_once(
            "rocprof_trace",
            f"rocprofv3 trace pass timed out after {subprocess_timeout}s",
        )
        return {
            "trace": {
                "format": "pftrace",
                "skipped": f"rocprofv3 trace pass timed out after {subprocess_timeout}s",
            }
        }
    except (OSError, subprocess.SubprocessError) as e:
        warn_once("rocprof_trace", f"rocprofv3 invocation failed: {e}")
        return {
            "trace": {
                "format": "pftrace",
                "skipped": f"rocprofv3 invocation failed: {e}",
            }
        }

    # Hoist results.<ext> out of <out_dir>/<hostname>/ to <out_dir>/.
    # Done on both success and error paths so anything rocprofv3
    # produced before exiting is reachable at a stable path.
    flatten_hostname_dir(out_dir)

    result: Dict[str, Any] = {"format": "pftrace"}
    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.strip().splitlines()[-40:])
        warn_once(
            "rocprof_trace",
            f"rocprofv3 exited {proc.returncode}; see extra_metrics['trace']['error_tail']",
        )
        result["returncode"] = proc.returncode
        result["error_tail"] = tail
        return {"trace": result}

    path = find_first(out_dir, "*.pftrace")
    if path is None:
        warn_once("rocprof_trace", "no .pftrace file produced")
        result["warnings"] = ["no .pftrace artifact found"]
    else:
        result["path"] = str(path)
    return {"trace": result}

# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""rocprofv3 kernel/memcpy trace export.

Wraps the workload in ``rocprofv3 --kernel-trace --memory-copy-trace
--output-format pftrace`` and records the resulting artifact path. The
``.pftrace`` file opens directly in https://ui.perfetto.dev.
"""

from pathlib import Path
from typing import Any, Dict, List

from ._artifact_paths import find_first, flatten_hostname_dir
from ._diagnostic import warn_once
from ._subprocess import run_tool
from ._tool_resolver import resolve_rocm_tool


def _build_argv(out_dir: Path, inner_argv: List[str]) -> List[str]:
    # `-o results` strips rocprofv3's `<pid>_` filename prefix;
    # ``flatten_hostname_dir`` strips the `<hostname>/` segment afterwards.
    return [
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
    timeout_s: int,
    context: str,
) -> Dict[str, Any]:
    """Run rocprofv3 trace and return the ``{"trace": ...}`` slice. Never raises."""
    result: Dict[str, Any] = {"format": "pftrace"}
    _, fields = run_tool(
        "rocprof_trace",
        resolve_rocm_tool("rocprofv3"),
        _build_argv(out_dir, inner_argv),
        out_dir,
        timeout_s,
        context,
    )
    result.update(fields)
    flatten_hostname_dir(out_dir)
    if fields:
        return {"trace": result}

    path = find_first(out_dir, "*.pftrace")
    if path is None:
        warn_once("rocprof_trace", f"{context}: no .pftrace file produced")
        result["warnings"] = ["no .pftrace artifact found"]
    else:
        result["path"] = str(path)
    return {"trace": result}

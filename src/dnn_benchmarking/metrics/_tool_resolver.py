# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""ROCm tool binary resolution.

Profiling tools must come from the same ROCm distribution the workload
loads. When the rocm-sdk wheels are installed — the ROCm torch is built
against, and the one hipDNN links here — that's the wheel's tools, not
whatever sits in ``/opt/rocm``.

Mixing the two does not degrade, it fails outright. The two installs
are independent builds even at equal version (measured: system ROCm
7.15.0 vs wheels 7.15.0a20260721), so ``/opt/rocm/bin/rocprofv3``
injects the system LLVM runtime into a wheel-ROCm process and the
workload dies before any counter is read::

    _rocm_sdk_core/lib/libamd_comgr.so.3: undefined symbol
    _ZNSt19_Sp_make_shared_tag5_S_eqERKSt9type_info, version LLVM_23.0

The wheels ship each tool twice: the raw binary under
``_rocm_sdk_core/bin`` and a console-script shim next to
``sys.executable``. Only the shim works — the raw binary can't find
aqlprofile on its own (``aqlprofile API table load failed``) because
the shim is what sets the wheel's ROCm environment up first. So the
shim is what we return.

Without the wheels (torch built against a system ROCm) the system
install is the coherent one, and we keep the old ``$ROCM_PATH/bin``
then PATH order.

Known-broken combination, for the next person who sees empty results:
the wheel's rocprofv3 currently deadlocks on hipDNN workloads (both
``--pmc`` and ``--kernel-trace``, right after ``HSA version 1.21.0
initialized``) while working fine on plain torch ones. That's an
upstream rocprofiler-sdk problem; ``_subprocess.run_capped`` bounds it
so a wedged pass is a skipped pass rather than a hung benchmark.
"""

import importlib.util
import os
import shutil
import sys
from typing import Optional

from ._diagnostic import warn_once


def _preferred_rocm_root() -> str:
    return os.environ.get("ROCM_PATH", "/opt/rocm")


def _wheel_rocm_installed() -> bool:
    """True iff the rocm-sdk wheels provide this process's ROCm runtime."""
    return importlib.util.find_spec("_rocm_sdk_core") is not None


def resolve_rocm_tool(name: str) -> Optional[str]:
    """Return the absolute path of the ROCm tool ``name``, or ``None``.

    Resolution order: the rocm-sdk wheel's console script when the
    wheels are installed, then ``$ROCM_PATH/bin/<name>``, then PATH.

    Warns once per tool whenever it has to leave the wheel install,
    since that combination cannot profile a wheel-ROCm workload (see
    module docstring) and the user would otherwise just see a crashed
    or empty profiling pass.

    Args:
        name: Bare tool name, e.g. ``"rocprofv3"`` or ``"rocprof-compute"``.

    Returns:
        Absolute path string or ``None`` if the tool isn't installed.
    """
    wheel_rocm = _wheel_rocm_installed()
    if wheel_rocm:
        shim = shutil.which(name, path=os.path.dirname(sys.executable))
        if shim is not None:
            return shim

    candidate = os.path.join(_preferred_rocm_root(), "bin", name)
    if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
        if wheel_rocm:
            warn_once(
                "tool_resolver",
                f"{name}: no rocm-sdk wheel shim next to {sys.executable!r}; "
                f"using {candidate!r} from a different ROCm install, which "
                "cannot profile this process (see "
                "metrics/_tool_resolver.py docstring)",
            )
        return candidate

    path_resolved = shutil.which(name)
    if path_resolved is not None:
        warn_once(
            "tool_resolver",
            f"{name}: not found in the wheel install or at {candidate!r}; "
            f"using PATH-resolved {path_resolved!r} (may belong to a "
            "different ROCm install — see metrics/_tool_resolver.py "
            "docstring)",
        )
    return path_resolved

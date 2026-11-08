# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""GPU identity: gfx target and model name of the live GPU.

Used by :mod:`rocprof_pmc` to pick the counter set for the device and by
:mod:`machine_info` for the environment block. Each field comes from the
first source that reports it: torch when the process has already imported
it (importing torch takes seconds, so this module never does), then amdsmi,
then rocminfo. ``"unknown"`` is the arch sentinel when no GPU can be
identified; callers that key off arch (e.g. PMC set selection) translate it
into their conservative defaults rather than raising.
"""

import functools
import re
import subprocess
import sys
from typing import Optional, Tuple

from ..common import torch_support
from ._diagnostic import warn_once
from ._tool_resolver import resolve_rocm_tool
from .gpu_smi import GpuSmiProbe

_GFX_PATTERN = re.compile(r"\b(gfx[0-9a-f]+)\b", re.IGNORECASE)

# (gfx target, model name); None where the source does not report it.
GpuIdentity = Tuple[Optional[str], Optional[str]]


def _gfx(text: Optional[str]) -> Optional[str]:
    m = _GFX_PATTERN.search(text or "")
    return m.group(1).lower() if m else None


def _detect_via_torch() -> GpuIdentity:
    if "torch" not in sys.modules or not torch_support.gpu_available():
        return None, None
    try:
        torch = sys.modules["torch"]
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        # gcnArchName is ROCm-only; CUDA builds lack it.
        return _gfx(getattr(props, "gcnArchName", None)), props.name or None
    except Exception as e:
        warn_once("arch", f"torch GPU lookup failed: {e}")
        return None, None


def _detect_via_amdsmi() -> GpuIdentity:
    target, model = GpuSmiProbe().identity()
    return _gfx(target), model


def _detect_via_rocminfo() -> GpuIdentity:
    # resolve_rocm_tool also finds rocminfo when /opt/rocm/bin is not on PATH.
    binary = resolve_rocm_tool("rocminfo")
    if binary is None:
        return None, None
    try:
        proc = subprocess.run(
            [binary], capture_output=True, text=True, timeout=10, check=False
        )
    except (OSError, subprocess.SubprocessError) as e:
        warn_once("arch", f"rocminfo invocation failed: {e}")
        return None, None
    if proc.returncode != 0:
        return None, None
    # Each agent lists "Name:" then "Marketing Name:"; only GPU agents have a
    # gfx name, so the first gfx line starts the first GPU agent.
    arch = None
    for line in proc.stdout.splitlines():
        if arch is None:
            arch = _gfx(line)
        elif line.strip().startswith("Marketing Name:"):
            return arch, line.split(":", 1)[1].strip() or None
    return arch, None


@functools.lru_cache(maxsize=None)
def detect_gpu() -> Tuple[str, Optional[str]]:
    """Return ``(gfx target or "unknown", model name or None)`` of the live GPU."""
    arch = model = None
    for probe in (_detect_via_torch, _detect_via_amdsmi, _detect_via_rocminfo):
        if arch and model:
            break
        probe_arch, probe_model = probe()
        arch, model = arch or probe_arch, model or probe_model
    return arch or "unknown", model


def detect_arch() -> str:
    """Return the gfx target of the live GPU, or ``"unknown"``.

    Callers should treat ``"unknown"`` as a signal to use their conservative
    defaults rather than as a table key.
    """
    return detect_gpu()[0]

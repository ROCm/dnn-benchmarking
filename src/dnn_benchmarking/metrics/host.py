# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Host-side probe: host memory snapshot.

``host_memory_snapshot`` returns process RSS and host RAM availability via
``psutil`` if installed. It degrades gracefully: on any failure it yields
``None`` values rather than raising.
"""

from typing import Dict, Optional

from ._diagnostic import warn_once


def host_memory_snapshot() -> Dict[str, Optional[float]]:
    """Return process RSS and host RAM availability in MB.

    Both values are ``None`` if ``psutil`` is unavailable or the read
    fails. The two-key shape is stable so callers can serialise without
    branching on availability.

    Returns:
        Dict with keys ``host_rss_mb`` and ``host_ram_available_mb``.
    """
    out: Dict[str, Optional[float]] = {
        "host_rss_mb": None,
        "host_ram_available_mb": None,
    }
    try:
        import psutil
    except ImportError:
        warn_once("psutil", "module not installed; host memory metrics disabled")
        return out

    try:
        proc = psutil.Process()
        out["host_rss_mb"] = proc.memory_info().rss / (1024.0 * 1024.0)
    except (psutil.Error, OSError) as e:
        warn_once("psutil", f"process RSS read failed: {e}")

    try:
        out["host_ram_available_mb"] = psutil.virtual_memory().available / (
            1024.0 * 1024.0
        )
    except (psutil.Error, OSError) as e:
        warn_once("psutil", f"virtual_memory read failed: {e}")

    return out

# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Single-fire warning helper for metric collection.

Metric probes can fail silently in many places (psutil missing, amdsmi
init error, /proc read denied). To avoid spamming N warnings per suite
when the same dependency is missing, ``warn_once`` deduplicates by
``(source, reason)`` for the lifetime of the process.

A console front end can install a sink with :func:`set_sink` so warnings
are written through it (e.g. to finish a pending progress line first).
"""

import sys
from typing import Callable, Optional, Set, Tuple

_seen: Set[Tuple[str, str]] = set()
_sink: Optional[Callable[[str], None]] = None


def set_sink(fn: Optional[Callable[[str], None]]) -> None:
    """Route warn_once lines through ``fn``; ``None`` restores stderr."""
    global _sink
    _sink = fn


def warn_once(source: str, reason: str) -> None:
    """Emit a warning the first time a (source, reason) is seen.

    Args:
        source: Short identifier of the metric source (e.g. "amdsmi",
            "psutil", "machine_info").
        reason: One-line explanation of the failure.
    """
    key = (source, reason)
    if key in _seen:
        return
    _seen.add(key)
    line = f"[metrics:{source}] {reason}"
    if _sink is not None:
        _sink(line)
    else:
        print(line, file=sys.stderr)


def reset() -> None:
    """Clear the seen set. Intended for tests."""
    _seen.clear()

# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Prebuilt hipDNN backend and engine plugins for one GPU architecture.

This package carries no hipDNN API. It exists to place ``libhipdnn_backend.so``
and the engine plugins somewhere a pip install can reach, and to report where
they landed. The payload is linked with ``$ORIGIN``-relative RPATHs that point
at the sibling ROCm SDK wheels, so nothing here needs ``LD_LIBRARY_PATH`` or a
shell activation step.

One wheel is published per GPU architecture (``hipdnn-runtime-gfx942`` and so
on). They all install this same package path, so exactly one may be installed at
a time; :data:`__gpu_arch__` records which one won.
"""

import os
from pathlib import Path

__all__ = ["__gpu_arch__", "backend_library", "library_dir", "plugin_path"]

_ROOT = Path(__file__).resolve().parent

#: GPU architecture this payload was compiled for. Substituted at wheel build
#: time by tools/build_release_wheels.py.
__gpu_arch__ = "@GPU_ARCH@"


def library_dir() -> Path:
    """Directory holding the hipDNN backend library."""
    return _ROOT / "lib"


def backend_library() -> Path:
    """Path to the hipDNN backend shared library in this payload."""
    name = "hipdnn_backend.dll" if os.name == "nt" else "libhipdnn_backend.so"
    return library_dir() / name


def plugin_path() -> Path:
    """Directory holding the hipDNN engine plugins."""
    return _ROOT / "lib" / "hipdnn_plugins" / "engines"

# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Kernel-selection state that a run must not inherit.

Two kinds of state decide which kernel runs:

* Environment variables (``HIPDNN_FORCE_BENCHMARKING``, ``MIOPEN_FIND_MODE``
  and others). A value inherited from the shell would change every row
  without a trace in the command line. :func:`inherited` lists them; the CLI
  rejects a run that inherits one unless ``--allow-selection-env`` is given.
* On-disk selection caches (hipDNN winners and autotune rankings, the MIOpen
  user database). Reads are not gated on tuning, so a cache that outlives a
  run lets the next run report an earlier run's choice. :func:`isolate`
  points them at a private, empty directory that is removed after the run.

Compiled-kernel caches (``MIOPEN_CUSTOM_CACHE_DIR``, ``AMD_COMGR_CACHE_DIR``,
``XDG_CACHE_HOME``) only save build work; they stay shared (see
``cli.main``) and are recorded.
"""

import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterator, Mapping, Optional

#: Variables that change which kernel a provider selects.
SELECTION_ENV = (
    "HIPDNN_CACHE_DIR",
    "HIPDNN_DISABLE_CACHE",
    "HIPDNN_DISABLE_EXACT_ENGINE_CACHE",
    "HIPDNN_FORCE_BENCHMARKING",
    "HIPDNN_HEURISTIC_PLUGIN_DIR",
    "MIOPEN_USER_DB_PATH",
    "MIOPEN_FIND_MODE",
    "MIOPEN_FIND_ENFORCE",
    "HIPBLASLT_USE_ROCROLLER",
    "HIPBLASLT_ROCROLLER_NO_CUSTOM_KERNEL",
)

#: Families of selection variables: heuristic policy, MIOpen solver toggles,
#: PyTorch GEMM tuning.
SELECTION_ENV_PREFIXES = ("HIPDNN_HEUR_", "MIOPEN_DEBUG_", "PYTORCH_TUNABLEOP_")

#: Compiled-kernel caches: shared between runs, recorded with the selection
#: variables.
SHARED_CACHE_ENV = ("MIOPEN_CUSTOM_CACHE_DIR", "AMD_COMGR_CACHE_DIR", "XDG_CACHE_HOME")


def _is_selection(name: str) -> bool:
    return name in SELECTION_ENV or name.startswith(SELECTION_ENV_PREFIXES)


def inherited(environ: Mapping[str, str]) -> Dict[str, str]:
    """Selection variables set to a non-empty value, sorted by name.

    hipDNN and MIOpen treat an empty value as unset.
    """
    return {
        name: environ[name]
        for name in sorted(environ)
        if _is_selection(name) and environ[name]
    }


def snapshot(environ: Mapping[str, str]) -> Dict[str, Optional[str]]:
    """The selection and shared-cache variables in effect, for the result file.

    Every fixed name is present (null when unset); a prefix-family variable
    is present only when set.
    """
    names = set(SELECTION_ENV) | set(SHARED_CACHE_ENV)
    names |= {n for n in environ if n.startswith(SELECTION_ENV_PREFIXES)}
    return {name: environ.get(name) for name in sorted(names)}


@contextmanager
def isolate(hipdnn_cache_dir: Optional[str]) -> Iterator[Path]:
    """Point the on-disk selection caches at a private, empty directory.

    ``HIPDNN_CACHE_DIR`` becomes ``hipdnn_cache_dir`` when given (a cache the
    caller chose to share), else ``<private>/hipdnn``. ``MIOPEN_USER_DB_PATH``
    becomes ``<private>/miopen``. A variable that is already set is kept: the
    caller allowed it (``--allow-selection-env``), or this is a child process
    of a run that set it.

    Enter before any GPU runtime initializes. On exit, every fixed selection
    variable gets back its value from before entry (including any the run set
    inside, such as ``HIPDNN_FORCE_BENCHMARKING``), and the directory is
    removed. Yields the private directory.
    """
    previous = {name: os.environ.get(name) for name in SELECTION_ENV}
    with tempfile.TemporaryDirectory(prefix="dnn-bench-selection-") as private:
        root = Path(private)
        try:
            if hipdnn_cache_dir:
                os.environ["HIPDNN_CACHE_DIR"] = hipdnn_cache_dir
            for name, sub in (
                ("HIPDNN_CACHE_DIR", "hipdnn"),
                ("MIOPEN_USER_DB_PATH", "miopen"),
            ):
                if not os.environ.get(name):
                    path = root / sub
                    path.mkdir()
                    os.environ[name] = str(path)
            yield root
        finally:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value

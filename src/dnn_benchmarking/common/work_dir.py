# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""The benchmark work directory: the caches and the kernel-selection state.

Layout of ``--benchmark-work-dir`` (default :func:`work_dir`)::

    <work>/
      cache/                 compile caches, kept: they only save time
        xdg/  miopen-kernels/  comgr/
      ootb/                  selection state of OOTB runs, kept across commands
        hipdnn-cache/        HIPDNN_CACHE_DIR
        miopen-user-db/      MIOPEN_USER_DB_PATH (MIOpen find results)
      tuned/                 selection state of tuned runs, kept across commands
        hipdnn-cache/        (tuned rankings, ingestor winners)
        miopen-user-db/

OOTB state serves every run that does not tune, ``--fast`` included, and the
OOTB plans of ``--oracle``. Only such runs write it, so a stored find result
is the one the heuristic path selects. Tuned state serves ``--autotune`` runs,
runs with a true ``HIPDNN_FORCE_BENCHMARKING`` in the environment
(:func:`benchmarks_kernels`), and the tuned runs of ``--oracle`` (child
processes, ``execution.oracle``).
With ``--fresh-state``, the process instead gets a new, empty state in the
system temporary directory (:data:`FRESH_STATE_PREFIX`), removed when the
command ends. ``cli.main`` sets the variables.
"""

import getpass
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

#: Compile caches under ``<work>/cache``: variable -> subdirectory.
COMPILE_CACHE_SUBDIRS: Dict[str, str] = {
    "XDG_CACHE_HOME": "xdg",
    "MIOPEN_CUSTOM_CACHE_DIR": "miopen-kernels",
    "AMD_COMGR_CACHE_DIR": "comgr",
}

#: Prefix of the ``--fresh-state`` directory in the system temporary directory.
FRESH_STATE_PREFIX = "dnn-bench-state-"

# hipDNN's documented truthy values. Notably, "0" leaves a switch off.
_TRUTHY_ENV = {"1", "true", "on", "yes", "enable", "enabled"}


def env_switch_on(name: str) -> bool:
    """Whether the environment sets the hipDNN switch ``name`` to a true value."""
    return os.environ.get(name, "").strip().lower() in _TRUTHY_ENV


def benchmarks_kernels(autotune: bool) -> bool:
    """Whether the process benchmarks kernels: ``--autotune``, or
    ``HIPDNN_FORCE_BENCHMARKING`` set to a true value in the environment.

    Such a run stores the winners of its search, so it must use the tuned
    state: in the OOTB state, the winners would reach later heuristic runs.
    """
    return autotune or env_switch_on("HIPDNN_FORCE_BENCHMARKING")


@dataclass(frozen=True)
class SelectionState:
    """Where hipDNN and MIOpen keep what decides a kernel selection."""

    hipdnn_cache: Path
    miopen_user_db: Path

    @classmethod
    def under(cls, root: Path) -> "SelectionState":
        return cls(root / "hipdnn-cache", root / "miopen-user-db")

    def create(self) -> "SelectionState":
        """Create both directories; return self."""
        self.hipdnn_cache.mkdir(parents=True, exist_ok=True)
        self.miopen_user_db.mkdir(parents=True, exist_ok=True)
        return self

    def env(self) -> Dict[str, str]:
        """The variables that point hipDNN and MIOpen at this state."""
        return {
            "HIPDNN_CACHE_DIR": str(self.hipdnn_cache),
            "MIOPEN_USER_DB_PATH": str(self.miopen_user_db),
        }


def work_dir(benchmark_work_dir: Optional[str]) -> Path:
    """The absolute work directory, ``~`` expanded.

    In order: ``--benchmark-work-dir``, ``$DNN_BENCH_WORKSPACE/work``,
    ``$XDG_CACHE_HOME/dnn-benchmark/work``, else a per-user directory in the
    system temporary directory. The home directory is not a default: it can
    be on a network file system, which is slow for the SQLite and comgr
    caches. ``cli.main`` resolves the path once, before it points
    ``XDG_CACHE_HOME`` into the work directory.
    """
    if benchmark_work_dir:
        return Path(benchmark_work_dir).expanduser().resolve()
    workspace = os.environ.get("DNN_BENCH_WORKSPACE")
    if workspace:
        return (Path(workspace).expanduser() / "work").resolve()
    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache:
        return (Path(xdg_cache).expanduser() / "dnn-benchmark" / "work").resolve()
    user_dir = f"dnn-benchmark-{_user()}"
    return (Path(tempfile.gettempdir()) / user_dir / "work").resolve()


def _user() -> str:
    """The user name, else the numeric uid: a container started with
    ``--user <uid>`` can have no passwd entry and no ``USER``/``LOGNAME``."""
    try:
        return getpass.getuser()
    except (KeyError, OSError, ImportError):
        if hasattr(os, "getuid"):
            return str(os.getuid())
        raise


def ootb_state(work: Path) -> SelectionState:
    """The OOTB selection state, kept across commands."""
    return SelectionState.under(work / "ootb")


def tuned_state(work: Path) -> SelectionState:
    """The tuned selection state, kept across commands."""
    return SelectionState.under(work / "tuned")

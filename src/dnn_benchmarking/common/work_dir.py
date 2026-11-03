# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""The benchmark work directory: the caches and the kernel-selection state.

Layout of ``--benchmark-work-dir`` (default :func:`work_dir`)::

    <work>/
      cache/                 compile caches, kept: they only save time
        xdg/  miopen-kernels/  comgr/
      tuned/                 selection state of tuned runs, kept across commands
        hipdnn-cache/        HIPDNN_CACHE_DIR (tuned rankings, ingestor winners)
        miopen-user-db/      MIOPEN_USER_DB_PATH (MIOpen find results)
      ootb/temp/run-*/       one per command, removed when the command ends
        hipdnn-cache/
        miopen-user-db/

Tuned state serves ``--fast`` and ``--autotune`` runs (the whole process) and
the tuned runs of ``--oracle`` (child processes, ``execution.oracle``). Every
other run, the OOTB plans of ``--oracle`` included, gets the empty state of
its own ``ootb/temp/run-*`` directory, so no earlier command, and no tuned
run, changes its kernel selection. ``cli.main`` sets the variables.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

#: Compile caches under ``<work>/cache``: variable -> subdirectory.
COMPILE_CACHE_SUBDIRS: Dict[str, str] = {
    "XDG_CACHE_HOME": "xdg",
    "MIOPEN_CUSTOM_CACHE_DIR": "miopen-kernels",
    "AMD_COMGR_CACHE_DIR": "comgr",
}

#: Directory under the work directory that holds the per-command OOTB state.
OOTB_TEMP_SUBDIR = Path("ootb") / "temp"


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
    """``--benchmark-work-dir``, else ``$DNN_BENCH_WORKSPACE/work``, else
    ``~/.cache/dnn-benchmark/work``; ``~`` expanded, absolute."""
    if benchmark_work_dir:
        return Path(benchmark_work_dir).expanduser().resolve()
    workspace = os.environ.get("DNN_BENCH_WORKSPACE")
    if workspace:
        return (Path(workspace).expanduser() / "work").resolve()
    return Path.home() / ".cache" / "dnn-benchmark" / "work"


def tuned_state(work: Path) -> SelectionState:
    """The tuned selection state, kept across commands."""
    return SelectionState.under(work / "tuned")


def uses_tuned_state(fast: bool, autotune: bool) -> bool:
    """Whether a whole run uses the tuned state: ``--fast`` and
    ``--autotune``. Other runs get a new, empty OOTB state."""
    return fast or autotune

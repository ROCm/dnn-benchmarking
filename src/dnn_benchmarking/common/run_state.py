# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Location of the per-run state: MIOpen user databases and child result
files.

Each OOTB GPU run gets a new, empty directory that is deleted after the run
(:func:`run_state_dir`). ``--benchmark-work-dir`` puts these directories under
``<work dir>/tmp`` (``cli.main`` sets :data:`TEMP_ROOT_ENV`); without it they
go to the system temporary directory. The variable is inherited, so child
processes use the same location.

Tuned GPU runs share one MIOpen user database instead (:func:`tuned_db_dir`).
"""

import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

#: Directory for the temporary per-run state; unset: the system default.
TEMP_ROOT_ENV = "DNN_BENCH_TEMP_ROOT"

#: Subdirectory of ``--benchmark-work-dir`` that holds the tuned database.
TUNED_DB_SUBDIR = "tuned-db"


def run_state_dir(prefix: str) -> "tempfile.TemporaryDirectory[str]":
    """A new, empty directory for one run's state, removed on cleanup."""
    root = os.environ.get(TEMP_ROOT_ENV) or None
    if root is not None:
        Path(root).mkdir(parents=True, exist_ok=True)
    return tempfile.TemporaryDirectory(prefix=prefix, dir=root)


@contextmanager
def tuned_db_dir(work_dir: Optional[Path]) -> Iterator[Path]:
    """The MIOpen user database that every tuned child run of a command shares.

    With ``work_dir`` (--benchmark-work-dir) it is ``work_dir/tuned-db``,
    outside ``tmp``: it outlives the command, so every command that uses the
    work dir reuses the tuning of the earlier ones. Without ``work_dir`` it
    belongs to the command, shared by its rows and removed on exit. OOTB
    child runs never use it: each gets its own empty database.
    """
    if work_dir is not None:
        tuned_db = work_dir / TUNED_DB_SUBDIR
        tuned_db.mkdir(parents=True, exist_ok=True)
        yield tuned_db
        return
    with run_state_dir("dnn-bench-tuned-db-") as state:
        yield Path(state)

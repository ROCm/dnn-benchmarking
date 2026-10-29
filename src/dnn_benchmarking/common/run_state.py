# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Location of the temporary per-run state: MIOpen user databases and child
result files.

Each GPU run gets a new, empty directory that is deleted after the run.
``--benchmark-work-dir`` puts these directories under ``<work dir>/tmp``
(``cli.main`` sets :data:`TEMP_ROOT_ENV`); without it they go to the system
temporary directory. The variable is inherited, so child processes use the
same location.
"""

import os
import tempfile
from pathlib import Path

#: Directory for the temporary per-run state; unset: the system default.
TEMP_ROOT_ENV = "DNN_BENCH_TEMP_ROOT"


def run_state_dir(prefix: str) -> "tempfile.TemporaryDirectory[str]":
    """A new, empty directory for one run's state, removed on cleanup."""
    root = os.environ.get(TEMP_ROOT_ENV) or None
    if root is not None:
        Path(root).mkdir(parents=True, exist_ok=True)
    return tempfile.TemporaryDirectory(prefix=prefix, dir=root)

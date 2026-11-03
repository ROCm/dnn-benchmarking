# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Process environment of a ``--fast`` run.

``--fast`` runs every graph through one engine in this process. The runtime
keeps one hipDNN handle with only the plugin under test loaded
(``cli.runtimes``) and primes the engine once (``execution.suite_runner``).
This module owns the process-wide state that has to be set before the handle
exists: the MIOpen user database and hipDNN's disk cache.
"""

import os
import tempfile
from contextlib import contextmanager
from typing import Dict, Iterator, Optional

from ..reporting.reporter import Reporter

#: The variable through which MIOpen finds its user database. MIOpen reads it
#: once per process.
_USER_DB_VAR = "MIOPEN_USER_DB_PATH"


@contextmanager
def fast_mode_environment(reporter: Reporter) -> Iterator[None]:
    """Give the run a new, empty MIOpen user database; turn hipDNN's disk cache off.

    MIOpen reads ``MIOPEN_USER_DB_PATH`` once per process, so every graph of
    the run shares this database, and its find results carry over from graph
    to graph. It starts empty, so no earlier command changes the selection,
    and it is deleted on exit. Enter before the hipDNN handle is created.
    The previous values return on exit.
    """
    with tempfile.TemporaryDirectory(prefix="dnn-bench-fast-db-") as user_db:
        previous = _set_env({_USER_DB_VAR: user_db, "HIPDNN_DISABLE_CACHE": "1"})
        try:
            reporter.info(
                f"--fast: one process, one handle; MIOpen user database {user_db} "
                "(new, shared by every graph); hipDNN disk cache off"
            )
            yield
        finally:
            _restore_env(previous)


def _set_env(values: Dict[str, str]) -> Dict[str, Optional[str]]:
    """Set ``values`` in ``os.environ``; return the values they replace."""
    previous = {name: os.environ.get(name) for name in values}
    os.environ.update(values)
    return previous


def _restore_env(previous: Dict[str, Optional[str]]) -> None:
    for name, value in previous.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value

# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Parent side of the child scripts: start a child, read what it wrote.

:func:`child_run` is the only way the suite starts a child. Each child runs
in a new, empty temporary directory, which is also its MIOpen user database
(``execution.child_protocol``). The directory is deleted after the parent
has read it, so every GPU run starts with an empty MIOpen user database.

The child writes to this process's stdout and stderr: MIOpen and hipDNN
logs (``MIOPEN_LOG_LEVEL``) stay visible and are not held in memory. A
failure comes back through ``failure.json``.
"""

import json
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import numpy as np

from ..config.benchmark_config import SuiteConfig
from ..metrics._subprocess import run_capped
from ..reporting.suite_results import engine_id_hex
from .child_protocol import (
    RESULT_FILE,
    ChildEnv,
    ChildResult,
    ChildRunError,
    output_file,
    read_child_result,
    read_failure,
)


class SavedOutputs:
    """Read-only output source for ``check_correctness``: the child's files."""

    def __init__(self, out_dir: Path) -> None:
        self._out_dir = out_dir

    def get_output_tensor(self, uid: int) -> None:
        return None  # Host arrays only: compare on the host.

    def get_output_data(self, uid: int) -> Optional[np.ndarray]:
        path = output_file(self._out_dir, uid)
        return np.load(path) if path.is_file() else None


def child_args(graph_path: Path, config: SuiteConfig) -> List[str]:
    """Arguments that every child script takes, except ``--out-dir``."""
    return [
        "--graph",
        str(graph_path),
        "--seed",
        str(config.seed),
        "--timing-policy",
        json.dumps(asdict(config.timing_policy)),
    ]


def hipdnn_child_args(
    graph_path: Path,
    config: SuiteConfig,
    engine_id: int,
    plugin_path: Optional[Path],
    save_outputs: bool,
) -> List[str]:
    """Arguments of ``cli/hipdnn_child.py``, the same for its OOTB and tuned
    runs (the tuned run adds ``--tuned``)."""
    args = child_args(graph_path, config)
    args += ["--engine", engine_id_hex(engine_id)]
    if plugin_path is not None:
        args += ["--plugin-path", str(plugin_path)]
    if save_outputs:
        args.append("--save-outputs")
    if config.metrics.basic:
        args.append("--clocks")
    return args


@contextmanager
def child_run(
    what: str,
    module: str,
    args: List[str],
    env: ChildEnv,
    timeout_s: Optional[int],
) -> Iterator[Tuple[ChildResult, SavedOutputs]]:
    """Run ``python -m dnn_benchmarking.cli.<module>`` in a new, empty
    directory and ``env``; yield its result and saved outputs.

    The outputs stay readable until the context exits; then the directory
    (the child's MIOpen user database) is deleted.

    Raises:
        UnsupportedGraphError: The engine does not support the graph.
        StallFallbackError: The stall gate failed and the child did not
            retry without it (an OOTB run).
        ChildRunError: Any other failure, a timeout, or a child that ended
            without ``result.json``.
    """
    with tempfile.TemporaryDirectory(prefix=f"dnn-bench-{module}-") as state:
        out_dir = Path(state)
        argv = [sys.executable, "-m", f"dnn_benchmarking.cli.{module}"]
        argv += [*args, "--out-dir", str(out_dir)]
        try:
            proc = run_capped(
                argv, timeout_s, env=env.build(str(out_dir)), capture_output=False
            )
        except subprocess.TimeoutExpired:
            raise ChildRunError(
                f"{what} child timed out after {timeout_s} s (--profiling-timeout)"
            ) from None
        failure = read_failure(out_dir)
        if failure is not None:
            raise failure.exception()
        if proc.returncode != 0 or not (out_dir / RESULT_FILE).is_file():
            raise ChildRunError(
                f"{what} child exited {proc.returncode} without a result"
            )
        yield read_child_result(out_dir), SavedOutputs(out_dir)

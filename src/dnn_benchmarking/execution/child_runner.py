# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Parent side of the child scripts: start a child, read what it wrote.

Each child runs in a new temporary directory, which is also its MIOpen user
database (``execution.child_protocol``). The directory is empty before the
child starts and is deleted after the parent reads it, so every GPU run
starts with an empty MIOpen user database.

- :func:`hipdnn_ootb_run`: the OOTB plan of one hipDNN engine row.
- :func:`run_child`: one tuned run (``execution.oracle``).
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

from ..common.exceptions import UnsupportedGraphError
from ..config.benchmark_config import SuiteConfig
from ..metrics._subprocess import run_capped
from ..reporting.suite_results import engine_id_hex
from .child_protocol import (
    HIPDNN_OOTB_ENV,
    RESULT_FILE,
    SKIPPED,
    STALLED,
    ChildEnv,
    PlanResult,
    output_file,
    read_failure,
    read_plan_result,
)
from .timing import StallFallbackError


class ChildRunError(Exception):
    """A child produced no result. The message is the child's own
    ``Type: message``, or its exit code and last stderr line."""


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
    """Arguments of ``cli/hipdnn_child.py`` shared by its OOTB and tuned runs."""
    args = child_args(graph_path, config)
    args += ["--engine", engine_id_hex(engine_id)]
    if plugin_path is not None:
        args += ["--plugin-path", str(plugin_path)]
    if save_outputs:
        args.append("--save-outputs")
    return args


def _spawn(
    module: str,
    args: List[str],
    env: ChildEnv,
    out_dir: Path,
    timeout_s: Optional[float],
) -> "subprocess.CompletedProcess[str]":
    """Run ``python -m dnn_benchmarking.cli.<module>`` in ``env``."""
    argv = [sys.executable, "-m", f"dnn_benchmarking.cli.{module}"]
    argv += [*args, "--out-dir", str(out_dir)]
    return run_capped(argv, timeout_s, env=env.build(str(out_dir)))


def _last_line(proc: "subprocess.CompletedProcess[str]") -> str:
    lines = [
        line.strip()
        for line in (proc.stderr or proc.stdout or "").splitlines()
        if line.strip()
    ]
    return lines[-1] if lines else "no output"


def run_child(
    what: str,
    module: str,
    args: List[str],
    env: ChildEnv,
    out_dir: Path,
    config: SuiteConfig,
) -> None:
    """Run a tuned child in ``env``; ``--profiling-timeout`` limits it.

    Raises:
        RuntimeError: If the child times out, exits nonzero or writes no
            ``result.json``. The message holds the last line of its stderr.
    """
    timeout_s = config.metrics.profiling_timeout_s or None
    try:
        proc = _spawn(module, args, env, out_dir, timeout_s)
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"{what} child timed out after {timeout_s} s (--profiling-timeout)"
        ) from None
    if proc.returncode != 0 or not (out_dir / RESULT_FILE).is_file():
        raise RuntimeError(f"{what} child exited {proc.returncode}: {_last_line(proc)}")


@contextmanager
def hipdnn_ootb_run(
    *,
    graph_path: Path,
    engine_id: int,
    plugin_path: Optional[Path],
    config: SuiteConfig,
    save_outputs: bool,
) -> Iterator[Tuple[PlanResult, SavedOutputs]]:
    """Time the OOTB plan of one engine in a child; yield its result.

    Yields the plan result and the saved outputs, which stay readable until
    the context exits. The OOTB run has no time limit, as in the parent.

    Raises:
        UnsupportedGraphError: The engine does not support the graph.
        StallFallbackError: The stall gate failed; the caller remeasures the
            whole graph without it.
        ChildRunError: Any other failure of the child.
    """
    args = hipdnn_child_args(graph_path, config, engine_id, plugin_path, save_outputs)
    if config.metrics.basic:
        args.append("--clocks")
    if config.oracle:
        args.append("--query-tuning")
    with tempfile.TemporaryDirectory(prefix="dnn-bench-hipdnn-ootb-") as state:
        out_dir = Path(state)
        proc = _spawn("hipdnn_child", args, HIPDNN_OOTB_ENV, out_dir, None)
        failure = read_failure(out_dir)
        if failure is not None:
            kind, message = failure
            if kind == SKIPPED:
                raise UnsupportedGraphError(message)
            if kind == STALLED:
                raise StallFallbackError(message)
            raise ChildRunError(message)
        if proc.returncode != 0 or not (out_dir / RESULT_FILE).is_file():
            raise ChildRunError(
                f"hipDNN child exited {proc.returncode}: {_last_line(proc)}"
            )
        yield read_plan_result(out_dir), SavedOutputs(out_dir)

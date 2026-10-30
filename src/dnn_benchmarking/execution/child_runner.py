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
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import numpy as np

from ..common.run_state import run_state_dir
from ..config.benchmark_config import SuiteConfig
from ..metrics._subprocess import run_capped
from ..reporting.suite_results import engine_id_hex
from .child_protocol import (
    RESULT_FILE,
    ChildSpec,
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

    def get_output_data(self, uid: int) -> Optional[np.ndarray]:
        path = output_file(self._out_dir, uid)
        return np.load(path) if path.is_file() else None


def _common_args(
    graph_path: Path, config: SuiteConfig, save_outputs: bool
) -> List[str]:
    """Arguments that every child script takes, except ``--out-dir`` and
    ``--tuned`` (:func:`child_run` adds them)."""
    args = [
        "--graph",
        str(graph_path),
        "--seed",
        str(config.seed),
        "--timing-policy",
        json.dumps(asdict(config.timing_policy)),
    ]
    if save_outputs:
        args.append("--save-outputs")
    if config.metrics.basic:
        args.append("--clocks")
    return args


def hipdnn_child_args(
    graph_path: Path,
    config: SuiteConfig,
    engine_id: int,
    plugin_path: Optional[Path],
    save_outputs: bool,
) -> List[str]:
    """Arguments of ``cli/hipdnn_child.py``, the same for its OOTB and tuned
    runs."""
    args = _common_args(graph_path, config, save_outputs)
    args += ["--engine", engine_id_hex(engine_id)]
    if plugin_path is not None:
        args += ["--plugin-path", str(plugin_path)]
    return args


def pytorch_child_args(
    graph_path: Path, config: SuiteConfig, save_outputs: bool
) -> List[str]:
    """Arguments of ``cli/pytorch_child.py``, the same for its OOTB and tuned
    runs. A child that saves its outputs (the reference row) also gets the
    reference precision: it saves a float64 reference when it can."""
    args = _common_args(graph_path, config, save_outputs)
    args += ["--pytorch-sdpa-backend", config.pytorch_sdpa_backend.value]
    if config.pytorch_rocm_fa_library is not None:
        args += ["--pytorch-rocm-fa-library", config.pytorch_rocm_fa_library]
    if save_outputs:
        args += ["--reference-precision", config.validation.precision.value]
    return args


@contextmanager
def child_run(
    spec: ChildSpec,
    args: List[str],
    timeout_s: Optional[int],
    tuned_db_dir: Optional[Path],
) -> Iterator[Tuple[ChildResult, SavedOutputs]]:
    """Run the script of ``spec`` in a new, empty directory, with the
    environment of ``spec``; yield its result and saved outputs.

    The outputs stay readable until the context exits; then the directory is
    deleted. For an OOTB spec it is also the child's MIOpen user database; a
    tuned spec uses ``tuned_db_dir``, the command's shared tuned database.

    Raises:
        ValueError: ``spec`` is tuned and ``tuned_db_dir`` is ``None``.
        UnsupportedGraphError: The engine does not support the graph.
        StallFallbackError: The stall gate failed and the child did not
            retry without it (an OOTB run).
        ChildRunError: Any other failure, a timeout, or a child that ended
            without ``result.json``.
    """
    what = spec.label
    with run_state_dir(f"dnn-bench-{spec.module}-") as state:
        out_dir = Path(state)
        argv = [sys.executable, "-m", f"dnn_benchmarking.cli.{spec.module}"]
        argv += [*args, *spec.flags(), "--out-dir", str(out_dir)]
        env = spec.env.build(str(out_dir), tuned_db_dir)
        try:
            proc = run_capped(argv, timeout_s, env=env, capture_output=False)
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

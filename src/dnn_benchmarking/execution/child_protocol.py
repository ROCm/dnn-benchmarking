# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Contract between the suite runner and its child scripts.

Every hipDNN engine row runs its plans in child processes: the OOTB plan in
``cli/hipdnn_child.py`` and, with ``--oracle``, the tuned plan in the same
script with ``--tuned``. ``cli/pytorch_tuned_child.py`` times tuned PyTorch.
Each child starts with a new, empty MIOpen user database (its private
directory), so no Find result of one run reaches another. MIOpen reads the
database path once per process, so only a new process can start empty.

The parent (``execution.child_runner``) and each child script import this
module. It holds:

- the options that every child takes (:func:`child_parser`);
- the files that a child writes in its private directory
  (:func:`write_result`, :func:`write_failure`, :func:`output_file`), and that
  the parent reads (:func:`read_plan_result`, :func:`read_failure`);
- the environment that the parent sets and the child checks
  (:class:`ChildEnv`, :data:`HIPDNN_OOTB_ENV`, :data:`HIPDNN_TUNED_ENV`,
  :data:`PYTORCH_TUNED_ENV`).

This module does not import ``torch``: the hipDNN child imports it.
"""

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

from ..config.benchmark_config import TimingPolicy
from ..reporting.statistics import BenchmarkStats, TimingInfo
from ..reporting.suite_results import OracleResult
from .timing import Measurement

RESULT_FILE = "result.json"
FAILURE_FILE = "failure.json"

#: hipDNN knob that makes a provider sample its candidate kernels on the first
#: execute() and keep the fastest (kernel ingestor, MIOpen).
BENCHMARKING_KNOB = "global.benchmarking"

#: ``failure.json`` kinds of an OOTB child: the engine does not support the
#: graph, the stall gate failed (the parent remeasures the graph unstalled),
#: or another error.
SKIPPED = "skipped"
STALLED = "stalled"
FAILED = "failed"


@dataclass(frozen=True)
class ChildEnv:
    """Environment that the parent sets and the child checks.

    The child sets no selection variable. Without the check, a missing
    variable would time another configuration than the one reported, or
    share a MIOpen user database with another run.

    Attributes:
        fixed: Variables with one exact value.
        in_dir: Variables whose value comes from the private directory of the
            run (function of the directory path).
    """

    fixed: Mapping[str, str]
    in_dir: Mapping[str, Callable[[str], str]]

    def build(self, state_dir: str) -> Dict[str, str]:
        """``os.environ`` plus the child variables for ``state_dir``.

        The child variables replace inherited values: an inherited value
        would break the isolation of the run.
        """
        env = dict(os.environ)
        env.update(self.fixed)
        env.update({name: value(state_dir) for name, value in self.in_dir.items()})
        return env

    def check(self) -> None:
        """Raise unless the parent set this environment.

        Raises:
            RuntimeError: A variable is missing or empty, or a ``fixed``
                variable has another value. The message names each one.
        """
        wrong = [
            f"{name}={os.environ.get(name)!r}"
            for name in [*self.fixed, *self.in_dir]
            if not os.environ.get(name)
            or (name in self.fixed and os.environ[name] != self.fixed[name])
        ]
        if wrong:
            raise RuntimeError(f"child environment not set by the parent: {wrong}")


def _same_dir(state_dir: str) -> str:
    return state_dir


#: OOTB hipDNN child: no hipDNN disk cache (as in the parent) and a private,
#: empty MIOpen user database, so the row selects its kernel on its own.
HIPDNN_OOTB_ENV = ChildEnv(
    fixed={"HIPDNN_DISABLE_CACHE": "1"},
    in_dir={"MIOPEN_USER_DB_PATH": _same_dir},
)

#: Tuned-hipDNN child: benchmarking on for every plan, no disk cache, and a
#: private MIOpen user database. The search then neither reuses earlier
#: tuning nor leaves entries that a later OOTB run would read.
HIPDNN_TUNED_ENV = ChildEnv(
    fixed={"HIPDNN_FORCE_BENCHMARKING": "1", "HIPDNN_DISABLE_CACHE": "1"},
    in_dir={"MIOPEN_USER_DB_PATH": _same_dir},
)

#: Tuned-PyTorch child: TunableOp tunes GEMMs on first use, into a private
#: results file. MIOpen's user database is private for the same reason as
#: for hipDNN.
PYTORCH_TUNED_ENV = ChildEnv(
    fixed={"PYTORCH_TUNABLEOP_ENABLED": "1", "PYTORCH_TUNABLEOP_TUNING": "1"},
    in_dir={
        "PYTORCH_TUNABLEOP_FILENAME": lambda d: os.path.join(
            d, "tunableop_results.csv"
        ),
        "MIOPEN_USER_DB_PATH": _same_dir,
    },
)


def _timing_policy(text: str) -> TimingPolicy:
    return TimingPolicy(**json.loads(text))


def child_parser(prog: str) -> argparse.ArgumentParser:
    """Argument parser with the options that every child script takes.

    ``--timing-policy`` is the full ``TimingPolicy`` as a JSON object,
    ``stall_gate`` included.
    """
    parser = argparse.ArgumentParser(prog=prog)
    parser.add_argument("--graph", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument(
        "--timing-policy",
        required=True,
        type=_timing_policy,
        help="TimingPolicy fields as a JSON object",
    )
    parser.add_argument("--out-dir", required=True, type=Path)
    return parser


def output_file(out_dir: Path, uid: int) -> Path:
    """Path of the saved output tensor ``uid`` (a NumPy ``.npy`` file)."""
    return out_dir / f"output_{uid}.npy"


def write_result(
    out_dir: Path,
    m: Measurement,
    build_ms: Optional[float] = None,
    workspace_bytes: Optional[int] = None,
    clocks_after: Optional[Dict[str, Any]] = None,
    tuning_available: Optional[bool] = None,
    tuning_error: Optional[str] = None,
) -> None:
    """Write the timed loop and the plan values as ``result.json``.

    ``clocks_after`` is the GPU clock sample right after the timed loop.
    ``tuning_available`` / ``tuning_error`` answer the OOTB child's
    ``--query-tuning``.
    """
    result = {
        "timing": TimingInfo.from_measurement(m).to_dict(),
        "kernel": asdict(BenchmarkStats.from_timings(m.kernel_ms)),
        "host": asdict(BenchmarkStats.from_timings(m.host_ms)),
        "build_ms": build_ms,
        "workspace_bytes": workspace_bytes,
        "clocks_after": clocks_after,
        "tuning_available": tuning_available,
        "tuning_error": tuning_error,
    }
    (out_dir / RESULT_FILE).write_text(json.dumps(result))


def write_failure(out_dir: Path, kind: str, message: str) -> None:
    """Write why an OOTB child produced no result (``SKIPPED``, ``STALLED``,
    ``FAILED``) as ``failure.json``."""
    (out_dir / FAILURE_FILE).write_text(json.dumps({"kind": kind, "message": message}))


def read_failure(out_dir: Path) -> Optional[Tuple[str, str]]:
    """``(kind, message)`` from ``failure.json``, or None without one."""
    path = out_dir / FAILURE_FILE
    if not path.is_file():
        return None
    failure = json.loads(path.read_text())
    return failure["kind"], failure["message"]


@dataclass(frozen=True)
class PlanResult:
    """A child's ``result.json``."""

    timing: TimingInfo
    kernel: BenchmarkStats
    host: BenchmarkStats
    build_ms: Optional[float]
    workspace_bytes: Optional[int]
    clocks_after: Optional[Dict[str, Any]]
    tuning_available: Optional[bool]
    tuning_error: Optional[str]


def read_plan_result(out_dir: Path) -> PlanResult:
    """The run from the ``result.json`` that :func:`write_result` wrote."""
    result = json.loads((out_dir / RESULT_FILE).read_text())
    return PlanResult(
        timing=TimingInfo(**result["timing"]),
        kernel=BenchmarkStats(**result["kernel"]),
        host=BenchmarkStats(**result["host"]),
        build_ms=result["build_ms"],
        workspace_bytes=result["workspace_bytes"],
        clocks_after=result["clocks_after"],
        tuning_available=result["tuning_available"],
        tuning_error=result["tuning_error"],
    )


def read_result(out_dir: Path, tuning_available: bool) -> OracleResult:
    """The tuned run from the ``result.json`` that :func:`write_result` wrote."""
    plan = read_plan_result(out_dir)
    return OracleResult(
        tuning_available=tuning_available,
        cpu_build_time_ms=plan.build_ms,
        timing=plan.timing,
        gpu_kernel_stats=plan.kernel,
        host_stats=plan.host,
        workspace_bytes=plan.workspace_bytes,
    )

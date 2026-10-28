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
- the files that a child writes in its private directory and that the
  parent reads: ``result.json`` (:func:`write_result`,
  :func:`read_child_result`), the saved outputs (:func:`output_file`), and
  ``failure.json`` (:func:`run_reporting_failures`, :func:`read_failure`),
  which carries a failure across the process boundary as the exception that
  the parent raises again;
- the environment that the parent sets and the child checks
  (:class:`ChildEnv`, :data:`HIPDNN_OOTB_ENV`, :data:`HIPDNN_TUNED_ENV`,
  :data:`PYTORCH_TUNED_ENV`).

The hipDNN child imports this module, so it must not import ``torch``.
"""

import argparse
import json
import os
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional

from ..common.exceptions import UnsupportedGraphError
from ..config.benchmark_config import TimingPolicy
from ..reporting.reporter import Reporter
from ..reporting.statistics import BenchmarkStats, TimingInfo
from .timing import Measurement, StallFallbackError

RESULT_FILE = "result.json"
FAILURE_FILE = "failure.json"

#: hipDNN knob that makes a provider sample its candidate kernels on the first
#: execute() and keep the fastest (kernel ingestor, MIOpen).
BENCHMARKING_KNOB = "global.benchmarking"


class ChildRunError(Exception):
    """A child produced no result. The message is the child's own
    ``Type: message``, or why the child gave no result."""


class FailureKind(Enum):
    """Why a child produced no result, as written in ``failure.json``."""

    SKIPPED = "skipped"  # The engine does not support the graph.
    STALLED = "stalled"  # The stall gate failed.
    FAILED = "failed"  # Any other error.


@dataclass(frozen=True)
class ChildFailure:
    """The content of ``failure.json``."""

    kind: FailureKind
    message: str

    @classmethod
    def from_exception(cls, e: Exception) -> "ChildFailure":
        """The failure that the parent raises again as :meth:`exception`."""
        if isinstance(e, UnsupportedGraphError):
            return cls(FailureKind.SKIPPED, str(e))
        if isinstance(e, StallFallbackError):
            return cls(FailureKind.STALLED, str(e))
        return cls(FailureKind.FAILED, failure_text(e))

    def exception(self) -> Exception:
        """The exception that the parent raises for this failure."""
        if self.kind is FailureKind.SKIPPED:
            return UnsupportedGraphError(self.message)
        if self.kind is FailureKind.STALLED:
            return StallFallbackError(self.message)
        return ChildRunError(self.message)


def failure_text(e: Exception) -> str:
    """``Type: message``. A :class:`ChildRunError` already has that form."""
    return str(e) if isinstance(e, ChildRunError) else f"{type(e).__name__}: {e}"


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


def run_reporting_failures(out_dir: Path, body: Callable[[], None]) -> int:
    """Run a child script's ``body``; return the exit code.

    ``body`` writes ``result.json``. When it raises, the failure goes to
    ``failure.json`` (and stderr), and the exit code is 1.
    """
    try:
        body()
    except Exception as e:
        failure = ChildFailure.from_exception(e)
        (out_dir / FAILURE_FILE).write_text(
            json.dumps({"kind": failure.kind.value, "message": failure.message})
        )
        Reporter(quiet=True).error(failure.message)
        return 1
    return 0


def read_failure(out_dir: Path) -> Optional[ChildFailure]:
    """The child's ``failure.json``, or None without one."""
    path = out_dir / FAILURE_FILE
    if not path.is_file():
        return None
    failure = json.loads(path.read_text())
    return ChildFailure(FailureKind(failure["kind"]), failure["message"])


@dataclass(frozen=True)
class TuningQuery:
    """Whether the plan exposes :data:`BENCHMARKING_KNOB`, or why that is not
    known (``available`` is then None)."""

    available: Optional[bool]
    error: Optional[str] = None


@dataclass(frozen=True)
class ChildResult:
    """A child's ``result.json``: one timed run and its plan values."""

    timing: TimingInfo
    kernel: BenchmarkStats
    host: BenchmarkStats
    build_ms: Optional[float] = None
    workspace_bytes: Optional[int] = None
    #: GPU clock sample right after the timed loop (``--clocks``).
    clocks_after: Optional[Dict[str, Any]] = None
    #: hipDNN children only.
    tuning: Optional[TuningQuery] = None


def write_result(
    out_dir: Path,
    m: Measurement,
    build_ms: Optional[float] = None,
    workspace_bytes: Optional[int] = None,
    clocks_after: Optional[Dict[str, Any]] = None,
    tuning: Optional[TuningQuery] = None,
) -> None:
    """Write the timed loop and the plan values as ``result.json``."""
    result = ChildResult(
        timing=TimingInfo.from_measurement(m),
        kernel=BenchmarkStats.from_timings(m.kernel_ms),
        host=BenchmarkStats.from_timings(m.host_ms),
        build_ms=build_ms,
        workspace_bytes=workspace_bytes,
        clocks_after=clocks_after,
        tuning=tuning,
    )
    (out_dir / RESULT_FILE).write_text(json.dumps(asdict(result)))


def read_child_result(out_dir: Path) -> ChildResult:
    """The run from the ``result.json`` that :func:`write_result` wrote."""
    result = json.loads((out_dir / RESULT_FILE).read_text())
    tuning = result["tuning"]
    return ChildResult(
        timing=TimingInfo(**result["timing"]),
        kernel=BenchmarkStats(**result["kernel"]),
        host=BenchmarkStats(**result["host"]),
        build_ms=result["build_ms"],
        workspace_bytes=result["workspace_bytes"],
        clocks_after=result["clocks_after"],
        tuning=TuningQuery(**tuning) if tuning is not None else None,
    )

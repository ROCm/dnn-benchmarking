# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Contract between the suite runner and its child scripts.

Every GPU run is a child process. A hipDNN engine row runs its OOTB plan in
``cli/hipdnn_child.py`` and, with ``--oracle``, its tuned plan in the same
script with ``--tuned``. A PyTorch row (``--runtime pytorch`` or the
``--validate pytorch`` reference) runs in ``cli/pytorch_child.py`` the same
way.
An OOTB child starts with a new, empty MIOpen user database (its private
directory), so no Find result of another run reaches it. MIOpen reads the
database path once per process, so only a new process can start empty.
Tuned children (``--tuned``, ``--autotune``) share the command's tuned
database instead, so tuning carries from row to row.

The parent (``execution.child_runner``) and each child script import this
module. It holds:

- the options that every child takes (:func:`child_parser`);
- the files that a child writes in its private directory and that the
  parent reads: ``result.json`` (:func:`write_result`,
  :func:`read_child_result`), the saved outputs (:func:`save_outputs`,
  :func:`output_file`), and ``failure.json`` (:func:`run_reporting_failures`,
  :func:`read_failure`), which carries a failure across the process boundary
  as the exception that the parent raises again;
- the kinds of child run (:class:`ChildSpec`: :data:`HIPDNN_OOTB`,
  :data:`HIPDNN_TUNED`, :data:`PYTORCH_OOTB`, :data:`PYTORCH_TUNED`), each
  with the environment that the parent sets and the child checks
  (:class:`ChildEnv`). The parent sets no selection variable in its own
  environment.

The hipDNN child imports this module, so it must not import ``torch``.
"""

import argparse
import json
import os
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, FrozenSet, List, Mapping, Optional, Protocol

import numpy as np

from ..common.exceptions import UnsupportedGraphError
from ..common.pytorch_tuning import PYTORCH_SELECTION_ENV
from ..config.benchmark_config import TimingPolicy
from ..graph.tensor_info import TensorInfo
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


#: The variable through which MIOpen finds its user database.
USER_DB_VAR = "MIOPEN_USER_DB_PATH"


class UserDb(Enum):
    """Which MIOpen user database a child run reads and writes."""

    #: The run's own new, empty directory, deleted after it: the selection
    #: sees no Find result of another run.
    PRIVATE = "private"
    #: The command's tuned database, shared by every tuned run: a search
    #: reuses what an earlier tuned run found for the same problem.
    TUNED = "tuned"


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
        user_db: The MIOpen user database of the run (:data:`USER_DB_VAR`).
            The child is not told the tuned path, so for :attr:`UserDb.TUNED`
            it checks only that the variable is set.
    """

    fixed: Mapping[str, str]
    in_dir: Mapping[str, Callable[[str], str]]
    user_db: UserDb

    @property
    def names(self) -> FrozenSet[str]:
        """Every variable that this environment sets."""
        return frozenset({*self.fixed, *self.in_dir, USER_DB_VAR})

    def build(self, state_dir: str, tuned_db_dir: Optional[Path]) -> Dict[str, str]:
        """``os.environ`` plus the child variables for ``state_dir`` and
        ``tuned_db_dir``.

        The child variables replace inherited values: an inherited value
        would break the isolation of the run.

        Raises:
            ValueError: :attr:`user_db` is :attr:`UserDb.TUNED` and
                ``tuned_db_dir`` is ``None``.
        """
        env = dict(os.environ)
        env.update(self._exact(state_dir))
        env[USER_DB_VAR] = self._user_db_path(state_dir, tuned_db_dir)
        return env

    def check(self, state_dir: Path) -> None:
        """Raise unless the parent set this environment for ``state_dir``.

        Raises:
            RuntimeError: A variable is missing, or has another value than
                :meth:`build` gives for ``state_dir``. The message names
                each one.
        """
        expected = self._exact(str(state_dir))
        if self.user_db is UserDb.PRIVATE:
            expected[USER_DB_VAR] = str(state_dir)
        wrong = [
            f"{name}={os.environ.get(name)!r}"
            for name, value in expected.items()
            if os.environ.get(name) != value
        ]
        if self.user_db is UserDb.TUNED and not os.environ.get(USER_DB_VAR):
            wrong.append(f"{USER_DB_VAR}={os.environ.get(USER_DB_VAR)!r}")
        if wrong:
            raise RuntimeError(f"child environment not set by the parent: {wrong}")

    def _exact(self, state_dir: str) -> Dict[str, str]:
        """The variables with one exact value for ``state_dir``."""
        return {
            **self.fixed,
            **{name: value(state_dir) for name, value in self.in_dir.items()},
        }

    def _user_db_path(self, state_dir: str, tuned_db_dir: Optional[Path]) -> str:
        if self.user_db is UserDb.PRIVATE:
            return state_dir
        if tuned_db_dir is None:
            raise ValueError("a tuned child needs the tuned MIOpen user database")
        return str(tuned_db_dir)


#: OOTB hipDNN child: the heuristic selection (benchmarking forced off, so a
#: shell value cannot turn it on), no hipDNN disk cache, and a private, empty
#: MIOpen user database, so the row selects its kernel on its own.
HIPDNN_OOTB_ENV = ChildEnv(
    fixed={"HIPDNN_FORCE_BENCHMARKING": "0", "HIPDNN_DISABLE_CACHE": "1"},
    in_dir={},
    user_db=UserDb.PRIVATE,
)

#: hipDNN child that benchmarks: the ``--autotune`` row and the ``--oracle``
#: tuned run. Benchmarking on for every plan, no hipDNN disk cache, and the
#: command's tuned MIOpen user database. An OOTB run never reads it.
HIPDNN_BENCHMARKING_ENV = ChildEnv(
    fixed={"HIPDNN_FORCE_BENCHMARKING": "1", "HIPDNN_DISABLE_CACHE": "1"},
    in_dir={},
    user_db=UserDb.TUNED,
)

#: OOTB PyTorch child (a ``--runtime pytorch`` row or the ``--validate
#: pytorch`` reference): the PyTorch selection settings, TunableOp off (a
#: value inherited from the shell would tune or reuse results), and a
#: private, empty MIOpen user database.
PYTORCH_OOTB_ENV = ChildEnv(
    fixed={
        **PYTORCH_SELECTION_ENV,
        "PYTORCH_TUNABLEOP_ENABLED": "0",
        "PYTORCH_TUNABLEOP_TUNING": "0",
    },
    in_dir={},
    user_db=UserDb.PRIVATE,
)

#: Tuned-PyTorch child: the PyTorch selection settings, and TunableOp tunes
#: GEMMs on first use, into a private results file. The MIOpen user database
#: is the command's tuned one, as for the tuned hipDNN child.
PYTORCH_TUNED_ENV = ChildEnv(
    fixed={
        **PYTORCH_SELECTION_ENV,
        "PYTORCH_TUNABLEOP_ENABLED": "1",
        "PYTORCH_TUNABLEOP_TUNING": "1",
    },
    in_dir={
        "PYTORCH_TUNABLEOP_FILENAME": lambda d: os.path.join(
            d, "tunableop_results.csv"
        ),
    },
    user_db=UserDb.TUNED,
)


@dataclass(frozen=True)
class ChildSpec:
    """One kind of child run: the script, the environment that the parent
    sets for it, and the script flags that select it.

    The parent starts a child only from a spec, and the script checks the
    environment of the spec that its flags select (:func:`select_spec`), so
    a script never runs with the environment of another kind.

    Attributes:
        tuned: The ``--oracle`` tuned run (``--tuned``): the script retries
            without the stall gate instead of failing.
        autotune: The ``--autotune`` row (``--autotune``).
    """

    label: str
    module: str
    env: ChildEnv
    tuned: bool = False
    autotune: bool = False

    def flags(self) -> List[str]:
        """The script flags that select this spec."""
        return [
            flag
            for flag, on in (("--tuned", self.tuned), ("--autotune", self.autotune))
            if on
        ]


HIPDNN_OOTB = ChildSpec("hipDNN", "hipdnn_child", HIPDNN_OOTB_ENV)
HIPDNN_AUTOTUNE = ChildSpec(
    "hipDNN autotune", "hipdnn_child", HIPDNN_BENCHMARKING_ENV, autotune=True
)
HIPDNN_TUNED = ChildSpec(
    "tuned hipDNN", "hipdnn_child", HIPDNN_BENCHMARKING_ENV, tuned=True
)
PYTORCH_OOTB = ChildSpec("PyTorch", "pytorch_child", PYTORCH_OOTB_ENV)
PYTORCH_TUNED = ChildSpec(
    "tuned PyTorch", "pytorch_child", PYTORCH_TUNED_ENV, tuned=True
)

#: Every kind of child run.
ALL_SPECS = (HIPDNN_OOTB, HIPDNN_AUTOTUNE, HIPDNN_TUNED, PYTORCH_OOTB, PYTORCH_TUNED)

#: Name prefixes of variables that hipDNN, MIOpen or PyTorch read to choose
#: a kernel. A child inherits every variable of the parent except the ones
#: its spec sets, so a shell value with one of these prefixes reaches every
#: run.
_SELECTION_PREFIXES = ("HIPDNN_", "MIOPEN_", "PYTORCH_", "TORCH_ROCM_")

#: Inherited names with those prefixes that do not choose a kernel: cache
#: locations (``cli.main`` sets them) and logging.
_NOT_SELECTION = frozenset({"MIOPEN_CUSTOM_CACHE_DIR", "HIPDNN_CACHE_DIR"})


def inherited_selection_env() -> Dict[str, str]:
    """Shell variables that every child inherits and that can change the
    kernel a run selects.

    Variables that a spec sets are not inherited and are not listed.
    """
    set_by_specs = {name for spec in ALL_SPECS for name in spec.env.names}
    return {
        name: value
        for name, value in sorted(os.environ.items())
        if name.startswith(_SELECTION_PREFIXES)
        and name not in set_by_specs
        and name not in _NOT_SELECTION
        and "LOG" not in name
    }


def select_spec(args: argparse.Namespace, *specs: ChildSpec) -> ChildSpec:
    """The spec among ``specs`` (those of one script) that the parsed flags
    select.

    Raises:
        RuntimeError: No spec of the script has these flags.
    """
    for spec in specs:
        if (spec.tuned, spec.autotune) == (args.tuned, args.autotune):
            return spec
    raise RuntimeError(
        f"no {specs[0].module} run with --tuned={args.tuned} "
        f"--autotune={args.autotune}"
    )


def _timing_policy(text: str) -> TimingPolicy:
    return TimingPolicy(**json.loads(text))


def child_parser(prog: str) -> argparse.ArgumentParser:
    """Argument parser with the options that every child script takes.

    ``--timing-policy`` is the full ``TimingPolicy`` as a JSON object,
    ``stall_gate`` included. ``--tuned`` selects the tuned spec of the script
    (:class:`ChildSpec`).
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
    parser.add_argument("--save-outputs", action="store_true")
    parser.add_argument("--clocks", action="store_true")
    parser.add_argument("--tuned", action="store_true")
    parser.add_argument("--autotune", action="store_true")
    return parser


def output_file(out_dir: Path, uid: int) -> Path:
    """Path of the saved output tensor ``uid`` (a NumPy ``.npy`` file)."""
    return out_dir / f"output_{uid}.npy"


class OutputBuffers(Protocol):
    """The buffer-manager methods that :func:`save_outputs` uses (hipDNN and
    PyTorch buffer managers)."""

    def zero_outputs(self) -> None: ...

    def get_output_tensors(self) -> List[TensorInfo]: ...

    def get_output_data(self, uid: int) -> Optional[np.ndarray]: ...


def save_outputs(
    out_dir: Path, buffers: OutputBuffers, run_once: Callable[[], None]
) -> None:
    """Run the graph once more on zeroed outputs; save each output tensor.

    The timed loop leaves the outputs of its last launch, which validation
    must not see, so the run starts from zeros.
    """
    buffers.zero_outputs()
    run_once()
    for tensor_info in buffers.get_output_tensors():
        data = buffers.get_output_data(tensor_info.uid)
        if data is not None:
            np.save(output_file(out_dir, tensor_info.uid), data)


def run_reporting_failures(out_dir: Path, body: Callable[[], None]) -> int:
    """Run a child script's ``body``; return the exit code.

    ``body`` writes ``result.json``. When it raises, the failure goes to
    ``failure.json`` only, and the exit code is 1. The parent reports it:
    a skip or a stall-gate failure is a normal outcome, not an error line.
    """
    try:
        body()
    except Exception as e:
        failure = ChildFailure.from_exception(e)
        (out_dir / FAILURE_FILE).write_text(
            json.dumps({"kind": failure.kind.value, "message": failure.message})
        )
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
    #: Executor notes for the row's ``warnings`` (PyTorch children).
    notes: List[str] = field(default_factory=list)
    #: Arithmetic of the saved outputs (``--save-outputs``): ``float64`` or
    #: ``native``; None without saved outputs.
    output_precision: Optional[str] = None


def write_result(
    out_dir: Path,
    m: Measurement,
    build_ms: Optional[float] = None,
    workspace_bytes: Optional[int] = None,
    clocks_after: Optional[Dict[str, Any]] = None,
    tuning: Optional[TuningQuery] = None,
    notes: Optional[List[str]] = None,
    output_precision: Optional[str] = None,
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
        notes=list(notes or []),
        output_precision=output_precision,
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
        notes=result["notes"],
        output_precision=result["output_precision"],
    )

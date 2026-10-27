# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Contract between the ``--oracle`` pass and its tuned-child scripts.

The parent (``execution.oracle``) and each child script
(``cli/hipdnn_tuned_child.py``, ``cli/pytorch_tuned_child.py``) import this
module. It holds:

- the options that every child takes (:func:`child_parser`);
- the files that a child writes in its output directory, the private
  temporary directory of the run (:func:`write_result`,
  :func:`output_file`), and that the parent reads (:func:`read_result`);
- the tuning environment that the parent sets and the child checks
  (:class:`ChildEnv`, :data:`HIPDNN_TUNED_ENV`, :data:`PYTORCH_TUNED_ENV`).
"""

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, Mapping, Optional

from ..config.benchmark_config import TimingPolicy
from ..reporting.statistics import BenchmarkStats, TimingInfo
from ..reporting.suite_results import OracleResult
from .timing import Measurement

RESULT_FILE = "result.json"


@dataclass(frozen=True)
class ChildEnv:
    """Tuning environment that the parent sets and the child checks.

    The child sets no tuning variable. Without the check, a missing variable
    would time an untuned run and report it as the tuned result.

    Attributes:
        fixed: Variables with one exact value.
        in_dir: Variables whose value comes from the private directory of the
            run (function of the directory path).
    """

    fixed: Mapping[str, str]
    in_dir: Mapping[str, Callable[[str], str]]

    def build(self, state_dir: str) -> Dict[str, str]:
        """``os.environ`` plus the tuning variables for ``state_dir``.

        The tuning variables replace inherited values: an inherited value
        would break the isolation of the tuned run.
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
            raise RuntimeError(f"tuning environment not set by the parent: {wrong}")


def _same_dir(state_dir: str) -> str:
    return state_dir


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
) -> None:
    """Write the timed loop and the plan values as ``result.json``."""
    result = {
        "timing": TimingInfo.from_measurement(m).to_dict(),
        "kernel": asdict(BenchmarkStats.from_timings(m.kernel_ms)),
        "host": asdict(BenchmarkStats.from_timings(m.host_ms)),
        "build_ms": build_ms,
        "workspace_bytes": workspace_bytes,
    }
    (out_dir / RESULT_FILE).write_text(json.dumps(result))


def read_result(out_dir: Path, tuning_available: bool) -> OracleResult:
    """The tuned run from the ``result.json`` that :func:`write_result` wrote."""
    result = json.loads((out_dir / RESULT_FILE).read_text())
    return OracleResult(
        tuning_available=tuning_available,
        cpu_build_time_ms=result["build_ms"],
        timing=TimingInfo(**result["timing"]),
        gpu_kernel_stats=BenchmarkStats(**result["kernel"]),
        host_stats=BenchmarkStats(**result["host"]),
        workspace_bytes=result["workspace_bytes"],
    )

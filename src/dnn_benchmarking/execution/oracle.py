# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Oracle pass: time a tuned run next to a row's OOTB run.

Every tuned run is a child process, so it can use the tuned selection state
of the work dir (``common.work_dir``: hipDNN cache and MIOpen user database,
kept across commands) while the row's OOTB run, in this process, uses the
OOTB state: MIOpen reads ``MIOPEN_USER_DB_PATH`` once per process. A hipDNN
row's tuned run builds the same engine with the ``global.benchmarking``
knob; the tuned runs of all hipDNN rows share one
``cli/hipdnn_tuned_child.py`` process after the last graph
(:class:`HipdnnTunedRuns`). A PyTorch row's tuned run is the normal CLI in
``cli/pytorch_tuned_child.py`` (:func:`run_pytorch_tuned`). A failed tuned
run sets ``row.oracle_error`` and never fails the row.
"""

import itertools
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..common.work_dir import SelectionState, tuned_state, work_dir
from ..config.benchmark_config import SuiteConfig, TimingPolicy
from ..graph.tensor_info import TensorInfo
from ..metrics._diagnostic import warn_once
from ..metrics._subprocess import run_capped
from ..metrics.analytical import derive_throughputs
from ..reporting.reporter import Reporter
from ..reporting.statistics import BenchmarkStats, TimingInfo
from ..reporting.suite_results import GraphResult, OracleResult, ProviderEngineResult
from ..validation import ReferenceOutput
from .correctness import check_correctness
from .timing import Measurement

#: hipDNN knob that makes a provider sample its candidate kernels on the first
#: execute() and keep the fastest (kernel ingestor, MIOpen).
BENCHMARKING_KNOB = "global.benchmarking"

#: Files of ``cli/hipdnn_tuned_child.py``: the job list in the tuned-runs
#: directory, and the result or failure of one run in its own directory.
MANIFEST_FILE = "manifest.json"
RESULT_FILE = "result.json"
FAILURE_FILE = "failure.json"

#: Node types whose PyTorch kernel the tuned child searches: MIOpen
#: exhaustive conv search (cudnn.benchmark) and TunableOp GEMMs.
_PYTORCH_TUNABLE_OPS = frozenset(
    {
        "ConvolutionFwdAttributes",
        "ConvolutionBwdAttributes",
        "ConvolutionWrwAttributes",
        "MatmulAttributes",
    }
)


def output_file(out_dir: Path, uid: int) -> Path:
    """Path of the tuned child's saved output tensor ``uid``."""
    return out_dir / f"output_{uid}.npy"


@dataclass(frozen=True)
class _SavedOutputs:
    """The tuned child's saved outputs, as ``check_correctness`` reads them."""

    out_dir: Path

    def get_output_tensor(self, uid: int) -> None:
        return None  # Host arrays only: compare on the host.

    def get_output_data(self, uid: int) -> Optional[np.ndarray]:
        path = output_file(self.out_dir, uid)
        return np.load(path) if path.is_file() else None


def _record_error(row: ProviderEngineResult, what: str, e: Exception) -> None:
    row.oracle_error = f"{type(e).__name__}: {e}"
    warn_once("oracle", f"tuned run failed for {what}: {row.oracle_error}")


def _set_throughputs(oracle: OracleResult, row: ProviderEngineResult) -> None:
    """Same FLOPs, bytes and median denominator as the row's throughputs."""
    median = oracle.gpu_kernel_stats.median_ms if oracle.gpu_kernel_stats else None
    oracle.derived_tflops_per_s, oracle.derived_gbytes_per_s = derive_throughputs(
        row.analytical_flops, row.analytical_io_bytes, median
    )


def _tuned_state(config: SuiteConfig) -> SelectionState:
    """The tuned selection state that every tuned run shares, created."""
    return tuned_state(work_dir(config.benchmark_work_dir)).create()


def _child_failure(proc: "subprocess.CompletedProcess[str]") -> str:
    """The child's last error line, or its last line, for an error message."""
    lines = [
        line.strip()
        for line in f"{proc.stdout or ''}\n{proc.stderr or ''}".splitlines()
        if line.strip()
    ]
    errors = [line for line in lines if "error" in line.lower()]
    return (errors or lines or ["no output"])[-1]


def _hipdnn_tuned_argv(manifest: Path) -> List[str]:
    """Argv of ``cli/hipdnn_tuned_child.py`` for the jobs in ``manifest``."""
    return [
        sys.executable,
        "-m",
        "dnn_benchmarking.cli.hipdnn_tuned_child",
        "--manifest",
        str(manifest),
    ]


#: Selection variable that the tuned hipDNN child must run with. Forced, not
#: defaulted: providers apply ``HIPDNN_FORCE_BENCHMARKING`` over the knob, so
#: an inherited ``0`` would make the tuned run untuned.
TUNED_CHILD_ENV = {"HIPDNN_FORCE_BENCHMARKING": "1"}


def _hipdnn_tuned_env(config: SuiteConfig) -> Dict[str, str]:
    """The tuned child's environment: :data:`TUNED_CHILD_ENV` and the tuned
    selection state (``HIPDNN_CACHE_DIR``, ``MIOPEN_USER_DB_PATH``). An
    inherited ``HIPDNN_DISABLE_CACHE`` stays as it is."""
    env = dict(os.environ)
    env.update(TUNED_CHILD_ENV)
    env.update(_tuned_state(config).env())
    return env


def reference_file(reference_dir: Path, uid: int) -> Path:
    """Path of the saved reference output tensor ``uid``."""
    return reference_dir / f"reference_{uid}.npy"


#: In a saved-reference directory: ``{uid: ReferenceOutput.precision}``.
_REFERENCE_PRECISIONS_FILE = "precisions.json"


def _load_references(reference_dir: Path) -> Dict[int, ReferenceOutput]:
    """The reference outputs that :meth:`HipdnnTunedRuns.save_references` wrote."""
    precisions = json.loads((reference_dir / _REFERENCE_PRECISIONS_FILE).read_text())
    return {
        int(uid): ReferenceOutput(
            data=np.load(reference_file(reference_dir, int(uid))),
            tensor_uid=int(uid),
            precision=precision,
        )
        for uid, precision in precisions.items()
    }


@dataclass
class _TunedJob:
    """One queued tuned run: the row it belongs to and what the child needs."""

    row: ProviderEngineResult
    graph_path: Path
    graph_name: str
    tensor_infos: List[TensorInfo]
    engine_id: int
    plugin_path: Optional[Path]
    timing_policy: TimingPolicy
    #: The graph's saved reference outputs; None without a reference.
    reference_dir: Optional[Path]
    #: Where the child writes this run's files; set when the run starts.
    out_dir: Path = Path()

    def manifest_entry(self) -> Dict[str, Any]:
        """This run as the child reads it from the manifest."""
        return {
            "graph": str(self.graph_path),
            "engine": self.engine_id,
            "plugin_path": str(self.plugin_path) if self.plugin_path else None,
            "timing_policy": asdict(self.timing_policy),
            "out_dir": str(self.out_dir),
            "save_outputs": self.reference_dir is not None,
        }


@dataclass(frozen=True)
class StalledGraph:
    """A graph whose stall gate failed in one of its tuned runs."""

    graph: GraphResult
    reason: str


class HipdnnTunedRuns:
    """The ``--oracle`` tuned runs of hipDNN rows, run in one child at the end.

    Each hipDNN engine row queues its tuned run (:meth:`add`) after its OOTB
    pass. After the last graph, :meth:`run` starts one child,
    ``cli/hipdnn_tuned_child.py``. The child takes the runs in row order. For
    each run it makes a new handle, primes the engine, builds its plan with
    ``global.benchmarking=1`` and times it with the row's timing policy, on
    the tuned selection state, as a child per row did. One process start and
    hipDNN import thus serve every row, and no OOTB buffer is alive while
    the child runs. The first, untimed launch runs the search, so the search
    cost lands in ``timing.first_call_ms``.

    The child writes the files of each run when the run ends. A crash or
    ``--oracle-timeout`` thus loses only the runs that did not end; their
    rows get ``oracle_error``. With a reference, the graph's reference
    outputs stay on disk (:meth:`save_references`) until the child's saved
    outputs are compared with them. :meth:`close` removes the files.
    """

    def __init__(self, config: SuiteConfig) -> None:
        self._config = config
        self._root = tempfile.TemporaryDirectory(prefix="dnn-bench-hipdnn-tuned-")
        self._jobs: List[_TunedJob] = []
        self._dir_numbers = itertools.count()

    def close(self) -> None:
        """Remove the saved references and the files of the child."""
        self._root.cleanup()

    def _new_dir(self, kind: str) -> Path:
        path = Path(self._root.name) / f"{kind}-{next(self._dir_numbers)}"
        path.mkdir()
        return path

    def save_references(self, reference_outputs: Dict[int, ReferenceOutput]) -> Path:
        """Write one graph's reference outputs to disk; return their directory.

        Call once per graph attempt and pass the directory to every
        :meth:`add` of that graph.
        """
        directory = self._new_dir("reference")
        precisions = {}
        for uid, ref in reference_outputs.items():
            np.save(reference_file(directory, uid), ref.host_data())
            precisions[str(uid)] = ref.precision
        (directory / _REFERENCE_PRECISIONS_FILE).write_text(json.dumps(precisions))
        return directory

    def add(
        self,
        *,
        row: ProviderEngineResult,
        graph_path: Path,
        graph_name: str,
        tensor_infos: List[TensorInfo],
        engine_id: int,
        plugin_path: Optional[Path],
        timing_policy: TimingPolicy,
        reference_dir: Optional[Path],
    ) -> None:
        """Queue the tuned run of ``row``, timed with ``timing_policy``."""
        self._jobs.append(
            _TunedJob(
                row=row,
                graph_path=graph_path,
                graph_name=graph_name,
                tensor_infos=tensor_infos,
                engine_id=engine_id,
                plugin_path=plugin_path,
                timing_policy=timing_policy,
                reference_dir=reference_dir,
            )
        )

    def run(
        self, graphs: Sequence[GraphResult], reporter: Reporter
    ) -> List[StalledGraph]:
        """Run the queued tuned runs of the rows in ``graphs``; attach results.

        Every queued run leaves the queue. A run whose row is not in
        ``graphs`` (from a graph attempt that was remeasured unstalled) does
        not run. A failed run sets ``row.oracle_error`` and never fails the
        row.

        Returns:
            The graphs, in ``graphs`` order, where the stall gate failed in
            a tuned run. Rows of one graph must share one timing mode, so
            the caller remeasures each of them without stalling, which
            queues new tuned runs, and calls this method again for them.
        """
        graph_of = {id(row): graph for graph in graphs for row in graph.results}
        jobs = [job for job in self._jobs if id(job.row) in graph_of]
        self._jobs = []
        if not jobs:
            return []
        for job in jobs:
            job.out_dir = self._new_dir("run")
        reporter.info(f"--oracle: {len(jobs)} tuned hipDNN run(s) in one child")
        stopped = self._run_child(jobs)
        stalled: Dict[int, StalledGraph] = {}
        for job in jobs:
            reason = self._collect(job, stopped)
            if reason is not None:
                graph = graph_of[id(job.row)]
                stalled.setdefault(id(graph), StalledGraph(graph, reason))
        return [stalled[id(g)] for g in graphs if id(g) in stalled]

    def _run_child(self, jobs: List[_TunedJob]) -> Optional[str]:
        """Run the child over ``jobs``; return why it stopped early, or None."""
        manifest = Path(self._root.name) / MANIFEST_FILE
        manifest.write_text(
            json.dumps(
                {
                    "seed": self._config.seed,
                    "jobs": [job.manifest_entry() for job in jobs],
                }
            )
        )
        timeout_s = self._config.oracle_timeout_s or None
        try:
            proc = run_capped(
                _hipdnn_tuned_argv(manifest),
                timeout_s,
                env=_hipdnn_tuned_env(self._config),
            )
        except subprocess.TimeoutExpired:
            return (
                f"tuned hipDNN child timed out after {timeout_s} s (--oracle-timeout)"
            )
        if proc.returncode != 0:
            return (
                f"tuned hipDNN child exited {proc.returncode}: {_child_failure(proc)}"
            )
        return None

    def _collect(self, job: _TunedJob, stopped: Optional[str]) -> Optional[str]:
        """Attach the result of ``job`` to its row; return a stall reason.

        Args:
            stopped: Why the child stopped early, for a run without files.
        """
        what = f"{job.graph_name} engine {job.engine_id}"
        failure_path = job.out_dir / FAILURE_FILE
        result_path = job.out_dir / RESULT_FILE
        try:
            if failure_path.is_file():
                failure = json.loads(failure_path.read_text())
                if failure["kind"] == "stall" and job.timing_policy.stall_gate:
                    return failure["message"]
                raise RuntimeError(failure["message"])
            if not result_path.is_file():
                raise RuntimeError(stopped or "tuned hipDNN child wrote no result")
            job.row.oracle = self._oracle(job, json.loads(result_path.read_text()))
        except Exception as e:
            _record_error(job.row, what, e)
        return None

    def _oracle(self, job: _TunedJob, result: Dict[str, Any]) -> OracleResult:
        """The row's tuned result from the child's ``result.json``, validated
        against the saved references when the graph has them."""
        m = Measurement(**result["measurement"])
        oracle = OracleResult(
            tuning_available=result["tuning_available"],
            cpu_build_time_ms=result["build_ms"],
            timing=TimingInfo.from_measurement(m),
            gpu_kernel_stats=BenchmarkStats.from_timings(m.kernel_ms),
            host_stats=BenchmarkStats.from_timings(m.host_ms),
            workspace_bytes=(
                result["workspace_bytes"] if self._config.metrics.basic else None
            ),
        )
        _set_throughputs(oracle, job.row)
        if job.reference_dir is not None:
            oracle.correctness = check_correctness(
                _SavedOutputs(job.out_dir),
                job.tensor_infos,
                _load_references(job.reference_dir),
                self._config.validation.provider.value,
                self._config,
            )
            if oracle.correctness.explicitly_failed:
                warn_once(
                    "oracle_correctness",
                    f"tuned plan for {job.graph_name} engine {job.engine_id} "
                    "failed validation; no speedup is reported",
                )
        return oracle


def _pytorch_tuned_argv(
    graph_path: Path, config: SuiteConfig, output: Path
) -> List[str]:
    """CLI argv for the tuned-PyTorch child: one timed PyTorch row, no oracle.

    The child's untimed first launch runs the tuning search, so it gets the
    same warmup and sampling settings as the OOTB row.
    """
    argv = [
        sys.executable,
        "-m",
        "dnn_benchmarking.cli.pytorch_tuned_child",
        "--runtime",
        "pytorch",
        "--no-metrics",
        "--quiet",
        "--graph",
        str(graph_path),
        "--warmup",
        str(config.warmup_iters),
        "--iters",
        str(config.benchmark_iters),
        "--min-time-ms",
        str(config.min_time_ms),
        "--cache-mode",
        config.cache_mode,
        "--timing-block",
        str(config.timing_block),
        "--seed",
        str(config.seed),
        "--pytorch-sdpa-backend",
        config.pytorch_sdpa_backend.value,
        "--output",
        str(output),
        "--benchmark-work-dir",
        str(work_dir(config.benchmark_work_dir)),
    ]
    if config.pytorch_rocm_fa_library is not None:
        argv += ["--pytorch-rocm-fa-library", config.pytorch_rocm_fa_library]
    return argv


def _run_pytorch_tuned_child(graph_path: Path, config: SuiteConfig) -> Dict[str, Any]:
    """Time tuned PyTorch in a fresh process and return its result row.

    The child owns the conv algorithm cache and the TunableOp results, in a
    temporary directory that is deleted afterwards. It runs on the tuned
    selection state of the work dir; no OOTB run uses it.

    Raises:
        RuntimeError: If the child fails, times out (``--oracle-timeout``)
            or reports no successful row.
    """
    from ..common.pytorch_tuning import tuned_subprocess_env

    with tempfile.TemporaryDirectory(prefix="dnn-bench-pytorch-tuned-") as state_dir:
        output = Path(state_dir) / "result.json"
        timeout_s = config.oracle_timeout_s or None
        try:
            proc = run_capped(
                _pytorch_tuned_argv(graph_path, config, output),
                timeout_s,
                env=tuned_subprocess_env(state_dir, _tuned_state(config)),
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(
                f"tuned PyTorch child timed out after {timeout_s} s "
                "(--oracle-timeout)"
            ) from None
        if not output.is_file():
            raise RuntimeError(
                f"tuned PyTorch child exited {proc.returncode}: "
                f"{_child_failure(proc)}"
            )
        document = json.loads(output.read_text())
    rows = [row for graph in document["graphs"] for row in graph["results"]]
    if len(rows) != 1:
        raise RuntimeError(f"tuned PyTorch child returned {len(rows)} rows")
    if rows[0]["status"] != "success":
        raise RuntimeError(rows[0]["message"] or rows[0]["status"])
    return rows[0]


def _stats(data: Optional[Dict[str, Any]], samples: int) -> Optional[BenchmarkStats]:
    """Rebuild stats from a v2 stats object and the plan's ``timing.samples``.

    The file keeps only the quartiles; the console-only fields (mean, std,
    min, p95, max) are None, and nothing reads them for a tuned row (the
    speedup, throughput and table use the median).
    """
    if not data:
        return None
    values = {**data, "n": samples}
    return BenchmarkStats(
        **{f.name: values.get(f.name) for f in fields(BenchmarkStats)}
    )


def run_pytorch_tuned(
    *,
    row: ProviderEngineResult,
    graph_path: Path,
    graph_json: Dict[str, Any],
    graph_name: str,
    config: SuiteConfig,
) -> None:
    """Attach a tuned PyTorch run as ``row.oracle``; never fails the row.

    Call after the OOTB buffers are released, so the child's allocations do
    not stack on top of them. Tuned outputs are not validated: they never
    enter this process. ``tuning_available`` is true only when the graph has
    a conv or matmul node, the ops the child searches.
    """
    try:
        plan = _run_pytorch_tuned_child(graph_path, config)["ootb"]
        oracle = OracleResult(
            tuning_available=any(
                node.get("type") in _PYTORCH_TUNABLE_OPS
                for node in graph_json.get("nodes") or []
            ),
            timing=(timing := TimingInfo(**plan["timing"])),
            gpu_kernel_stats=_stats(plan["kernel"], timing.samples),
            host_stats=_stats(plan["host"], timing.samples),
        )
        _set_throughputs(oracle, row)
        row.oracle = oracle
    except Exception as e:
        _record_error(row, f"PyTorch on {graph_name}", e)

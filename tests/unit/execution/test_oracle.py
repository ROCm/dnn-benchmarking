# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for execution.oracle (tuned run next to the OOTB run)."""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from dnn_benchmarking.cli.parser import create_parser
from dnn_benchmarking.common.exceptions import ExecutionError
from dnn_benchmarking.config.benchmark_config import (
    MetricsConfig,
    SuiteConfig,
    TimingPolicy,
    ValidationConfig,
)
from dnn_benchmarking.execution import oracle as oracle_mod
from dnn_benchmarking.execution.timing import Measurement, StallFallbackError
from dnn_benchmarking.graph.tensor_info import TensorInfo
from dnn_benchmarking.reporting.statistics import BenchmarkStats, TimingInfo
from dnn_benchmarking.reporting.suite_results import (
    CorrectnessResult,
    PlanResult,
    ProviderEngineResult,
)
from dnn_benchmarking.validation import ReferenceOutput

CACHE = "HIPDNN_DISABLE_CACHE"
# (call, handle) for every fake call; cleared by the ``tuned`` fixture.
CALLS = []


def _m(kernel_ms, host_ms=0.01):
    return Measurement(
        kernel_ms=list(kernel_ms),
        host_ms=[host_ms] * len(kernel_ms),
        mode="staged",
        timer="hip",
        cache_mode="warm",
        warmup_iters=1,
        first_call_ms=1.0,
    )


class _Handle:
    def get_stream(self):
        return 7

    def set_stream(self, stream):
        self.stream = stream


class _BM:
    def __init__(self, outputs=None):
        self.outputs = outputs or {}

    def zero_outputs(self):
        CALLS.append(("zero_outputs", None))

    def get_output_tensor(self, uid):
        return None

    def get_output_data(self, uid):
        return self.outputs.get(uid)


class _TunedExecutor:
    """Stands in for the oracle's own Executor; records the cache env per call."""

    knob_ids = [oracle_mod.BENCHMARKING_KNOB, "miopen.find_mode"]
    kernel_ms = [0.5, 0.5, 0.5, 9.0]
    prepare_error = None
    benchmark_error = None
    prepared = None
    policy = None

    def __init__(self, graph_json_str, policy):
        self.build_time_ms = 3.0
        self.workspace_size = 4096
        type(self).policy = policy

    def prepare(self, handle, engine_id=None, knobs=None):
        type(self).prepared = SimpleNamespace(
            handle=handle, engine_id=engine_id, knobs=knobs
        )
        self.env["prepare"] = os.environ.get(CACHE)
        if self.prepare_error is not None:
            raise self.prepare_error

    def engine_knob_ids(self, engine_id):
        return list(self.knob_ids)

    def benchmark(self, handle, variant_pack):
        CALLS.append(("benchmark", handle))
        self.env["benchmark"] = os.environ.get(CACHE)
        if self.benchmark_error is not None:
            raise self.benchmark_error
        return _m(self.kernel_ms, host_ms=0.2)

    def execute_once(self, handle, variant_pack):
        CALLS.append(("execute_once", handle))
        self.env["execute_once"] = os.environ.get(CACHE)


@pytest.fixture
def tuned(monkeypatch):
    cls = type("Tuned", (_TunedExecutor,), {"env": {}})
    monkeypatch.setattr(oracle_mod, "Executor", cls)
    CALLS.clear()
    return cls


def _config(**kw):
    kw.setdefault("validation", ValidationConfig(provider="pytorch"))
    return SuiteConfig(oracle_mode="exhaustive", **kw)


def _pytorch_config(**kw):
    return SuiteConfig(oracle_mode="exhaustive", runtime="pytorch", **kw)


def _row(correctness=None):
    return ProviderEngineResult(
        runtime="hipdnn",
        engine_id=5,
        status="success",
        ootb=PlanResult(correctness=correctness),
    )


def _build(row, handle=None, config=None):
    return oracle_mod.build_tuned_plan(
        row=row,
        handle=handle or _Handle(),
        engine_id=5,
        graph_json_str="{}",
        graph_name="g",
        config=config or _config(),
    )


def _run(row=None, refs=None, bm=None, config=None):
    row = row or _row()
    config = config or _config()
    out = TensorInfo(
        uid=1,
        name="y",
        dims=[2],
        strides=[1],
        data_type="float",
        is_virtual=False,
        is_output=True,
    )
    oracle_mod.run_tuned_plan(
        tuned=_build(row, config=config),
        row=row,
        engine_id=5,
        graph_name="g",
        config=config,
        bm=bm or _BM(),
        variant_pack={},
        tensor_infos=[out],
        reference_outputs=refs,
    )
    return row


def _refs():
    return {1: ReferenceOutput(data=np.zeros(2, np.float32), tensor_uid=1)}


# --- build_tuned_plan -------------------------------------------------------


def test_tuned_plan_is_a_benchmarking_build_of_the_row_engine_on_its_own_handle(
    tuned,
):
    """MIOpen's solver map is per handle; sharing the row handle would let the
    tuned search change the OOTB plan."""
    row_handle = _Handle()
    config = _config(warmup_iters=3, benchmark_iters=5, min_time_ms=2.0)

    plan = _build(_row(), handle=row_handle, config=config)

    assert plan.handle is tuned.prepared.handle
    assert plan.handle is not row_handle
    assert plan.handle.stream == 7
    assert tuned.prepared.engine_id == 5
    assert tuned.prepared.knobs == {"global.benchmarking": 1}
    assert tuned.policy == TimingPolicy(warmup_iters=3, iters=5, min_time_ms=2.0)


@pytest.mark.parametrize("exposed", [True, False])
def test_tuning_available_follows_the_engine_knobs(tuned, exposed):
    """hipDNN ignores a knob the engine does not expose; the tuned run then
    re-measures the OOTB configuration and must say so."""
    if not exposed:
        tuned.knob_ids = ["miopen.find_mode"]

    assert _build(_row()).tuning_available is exposed


@pytest.mark.parametrize("previous", [None, "0"], ids=["unset", "set"])
def test_cache_is_disabled_for_tuned_work_and_restored(tuned, monkeypatch, previous):
    """A benchmarking plan writes its winner to the hipDNN disk cache; a later
    OOTB row must not read it, and the OOTB timing must keep its cache."""
    if previous is None:
        monkeypatch.delenv(CACHE, raising=False)
    else:
        monkeypatch.setenv(CACHE, previous)

    _run(refs=_refs(), bm=_BM({1: np.zeros(2, np.float32)}))

    assert tuned.env == {"prepare": "1", "benchmark": "1", "execute_once": "1"}
    assert os.environ.get(CACHE) == previous


def test_build_failure_is_an_oracle_error_and_restores_the_env(tuned, monkeypatch):
    monkeypatch.delenv(CACHE, raising=False)
    tuned.prepare_error = ExecutionError("boom")
    row = _row()

    assert _build(row) is None
    assert row.oracle_error == "ExecutionError: boom"
    assert row.status == "success"
    assert CACHE not in os.environ


# --- run_tuned_plan ---------------------------------------------------------


def test_tuned_run_reports_its_own_build_timing_and_throughput(tuned):
    """TFLOP/s and GB/s use the row's analytical FLOPs/bytes over the TUNED
    kernel median (0.5 ms), not the mean or the OOTB median."""
    row = _row()
    row.analytical_flops = 10**9
    row.analytical_io_bytes = 10**6

    o = _run(row=row).oracle

    assert row.oracle_error is None
    assert o.tuning_available is True
    assert o.cpu_build_time_ms == pytest.approx(3.0)
    assert o.timing.first_call_ms == pytest.approx(1.0)
    assert o.gpu_kernel_stats.median_ms == pytest.approx(0.5)
    assert o.host_stats.median_ms == pytest.approx(0.2)
    assert o.workspace_bytes == 4096
    assert o.derived_tflops_per_s == pytest.approx(2.0)
    assert o.derived_gbytes_per_s == pytest.approx(2.0)


def test_workspace_is_not_reported_with_metrics_off(tuned):
    o = _run(config=_config(metrics=MetricsConfig(tier="off"))).oracle

    assert o.workspace_bytes is None


def test_tuned_plan_times_on_its_handle_then_validates_once_on_zeroed_outputs(
    tuned,
):
    row = _run(refs=_refs(), bm=_BM({1: np.zeros(2, np.float32)}))

    tuned_handle = tuned.prepared.handle
    assert CALLS == [
        ("zero_outputs", None),
        ("benchmark", tuned_handle),
        ("zero_outputs", None),
        ("execute_once", tuned_handle),
    ]
    assert row.oracle.correctness.tolerance_match


def test_no_references_means_no_tuned_validation(tuned):
    row = _run()

    assert ("execute_once", tuned.prepared.handle) not in CALLS
    assert row.oracle.correctness is None


def test_failing_tuned_plan_keeps_the_row_verdict(tuned):
    passed = CorrectnessResult(tolerance_match=True, rtol=1e-5, atol=1e-6)

    row = _run(row=_row(passed), refs=_refs(), bm=_BM({1: np.ones(2, np.float32)}))

    assert row.oracle.correctness.explicitly_failed
    assert row.ootb.correctness is passed
    assert row.verdict == "passed"


def test_stall_fallback_propagates_for_a_whole_graph_remeasure(tuned, monkeypatch):
    monkeypatch.delenv(CACHE, raising=False)
    tuned.benchmark_error = StallFallbackError("watchdog released the stream")

    with pytest.raises(StallFallbackError):
        _run()
    assert CACHE not in os.environ


def test_tuned_run_failure_is_an_oracle_error(tuned):
    tuned.benchmark_error = ExecutionError("launch failed")

    row = _run()

    assert row.oracle is None
    assert row.oracle_error == "ExecutionError: launch failed"
    assert row.status == "success"


# --- run_pytorch_tuned ------------------------------------------------------


def _child_row(status="success", message=None, kernel_ms=(0.25, 0.25, 0.3)):
    timing = TimingInfo(
        mode="staged",
        timer="hip",
        warmup_iters=3,
        first_call_ms=40.0,
    )
    return {
        "status": status,
        "message": message,
        "ootb": {
            "timing": timing.to_dict(),
            "kernel": BenchmarkStats.from_timings(kernel_ms).to_dict(),
            "host": BenchmarkStats.from_timings([0.5, 0.5, 0.5]).to_dict(),
        },
    }


@pytest.fixture
def child(monkeypatch):
    """Fake ``run_capped``: records the child launch and writes its result."""
    launch = SimpleNamespace(
        argv=None,
        env=None,
        result={"graphs": [{"results": [_child_row()]}]},
        proc=SimpleNamespace(returncode=0, stdout="", stderr=""),
    )

    def run_capped(argv, timeout_s, env=None):
        launch.argv, launch.env = argv, env
        if launch.result is not None:
            output = Path(argv[argv.index("--output") + 1])
            output.write_text(json.dumps(launch.result))
        return launch.proc

    monkeypatch.setattr(oracle_mod, "run_capped", run_capped)
    return launch


def _run_pytorch(tmp_path, config=None):
    row = ProviderEngineResult(runtime="pytorch", engine_id=None, status="success")
    row.analytical_flops = 10**9
    oracle_mod.run_pytorch_tuned(
        row=row,
        graph_path=tmp_path / "g.json",
        graph_name="g",
        config=config or _pytorch_config(),
    )
    return row


def test_pytorch_child_runs_one_untuned_oracle_off_row_with_the_run_timing(
    child, tmp_path
):
    """The child must time with the parent's settings, or the tuned and OOTB
    PyTorch numbers are not comparable."""
    config = _pytorch_config(
        warmup_iters=3,
        benchmark_iters=5,
        min_time_ms=2.0,
        timing_block=4,
        seed=11,
    )

    _run_pytorch(tmp_path, config)

    args = create_parser().parse_args(child.argv[3:])
    child_config = SuiteConfig.from_namespace(args)
    assert child.argv[1:3] == ["-m", "dnn_benchmarking"]
    assert args.internal_pytorch_tuned is True
    assert args.graph == [str(tmp_path / "g.json")]
    assert child_config.backend == "pytorch"
    assert child_config.oracle_mode == "off"
    assert child_config.timing_policy == config.timing_policy
    assert child_config.seed == 11


def test_pytorch_child_tunes_into_a_private_state_dir_that_is_deleted(child, tmp_path):
    """MIOpen's user db and TunableOp results must not outlive the child, or a
    later OOTB run would serve the tuned selection."""
    _run_pytorch(tmp_path)

    state_dir = Path(child.env["MIOPEN_USER_DB_PATH"])
    output = Path(child.argv[child.argv.index("--output") + 1])
    assert output.parent == state_dir
    assert child.env["PYTORCH_TUNABLEOP_ENABLED"] == "1"
    assert not state_dir.exists()


def test_pytorch_tuned_row_comes_from_the_child_ootb_plan(child, tmp_path):
    row = _run_pytorch(tmp_path)

    o = row.oracle
    assert row.oracle_error is None
    assert o.tuning_available is True
    assert o.cpu_build_time_ms is None
    assert o.correctness is None
    assert o.timing.first_call_ms == pytest.approx(40.0)
    assert o.gpu_kernel_stats.median_ms == pytest.approx(0.25)
    assert o.host_stats.median_ms == pytest.approx(0.5)
    # 1e9 FLOPs over the child's 0.25 ms kernel median.
    assert o.derived_tflops_per_s == pytest.approx(4.0)


def test_pytorch_child_without_output_reports_its_last_error_line(child, tmp_path):
    child.result = None
    child.proc = SimpleNamespace(
        returncode=1,
        stdout="loading graph\n",
        stderr="Traceback ...\nRuntimeError: HIP out of memory\n  \n",
    )

    row = _run_pytorch(tmp_path)

    assert row.oracle is None
    assert "exited 1" in row.oracle_error
    assert row.oracle_error.endswith("RuntimeError: HIP out of memory")
    assert row.status == "success"


def test_pytorch_child_row_error_carries_its_message(child, tmp_path):
    child.result = {
        "graphs": [{"results": [_child_row("error", "unsupported op: sdpa")]}]
    }

    row = _run_pytorch(tmp_path)

    assert row.oracle is None
    assert row.oracle_error == "RuntimeError: unsupported op: sdpa"

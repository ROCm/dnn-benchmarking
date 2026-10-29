# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for execution.oracle and its tuned-hipDNN child."""

import dataclasses
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from dnn_benchmarking.cli.parser import create_parser
from dnn_benchmarking.common import rocm_runtime
from dnn_benchmarking.common.exceptions import ExecutionError
from dnn_benchmarking.config.benchmark_config import (
    MetricsConfig,
    SuiteConfig,
    TimingPolicy,
    ValidationConfig,
)
from dnn_benchmarking.execution import (
    buffer_manager,
    executor,
    hipdnn_tuned_child,
    suite_runner,
)
from dnn_benchmarking.execution import oracle as oracle_mod
from dnn_benchmarking.execution.timing import Measurement, StallFallbackError
from dnn_benchmarking.graph.loader import GraphLoader
from dnn_benchmarking.graph.tensor_info import TensorInfo
from dnn_benchmarking.reporting.statistics import BenchmarkStats, TimingInfo
from dnn_benchmarking.reporting.suite_results import (
    CorrectnessResult,
    PlanResult,
    ProviderEngineResult,
)
from dnn_benchmarking.validation import ReferenceOutput

GRAPH_FILE = Path(__file__).parents[3] / "graphs" / "sample_add.json"


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


def _config(**kw):
    kw.setdefault("validation", ValidationConfig(provider="pytorch"))
    return SuiteConfig(oracle=True, **kw)


def _pytorch_config(**kw):
    return SuiteConfig(oracle=True, runtime="pytorch", **kw)


def _row(correctness=None):
    return ProviderEngineResult(
        runtime="hipdnn",
        engine_id=5,
        status="success",
        ootb=PlanResult(correctness=correctness),
    )


_OUT = TensorInfo(
    uid=1,
    name="y",
    dims=[2],
    strides=[1],
    data_type="float",
    is_virtual=False,
    is_output=True,
)


def _refs():
    return {1: ReferenceOutput(data=np.zeros(2, np.float32), tensor_uid=1)}


# --- tuned-hipDNN child process ---------------------------------------------


class _ChildFakes:
    """hipDNN, executor and buffers as the child sees them; records calls."""

    def __init__(self):
        self.calls = []
        self.knob_ids = ["global.benchmarking", "miopen.find_mode"]
        self.benchmark_error = None
        self.inputs = None
        self.output = np.full(4, 3.0, np.float32)


@pytest.fixture
def child_fakes(monkeypatch):
    f = _ChildFakes()

    class Executor:
        def __init__(self, graph_json_str, policy):
            self.policy = policy
            self.build_time_ms, self.workspace_size = 3.0, 4096

        def prime(self, handle, engine_id):
            f.calls.append(("prime", engine_id))

        def prepare(self, handle, engine_id=None, knobs=None):
            f.calls.append(("prepare", engine_id, knobs))

        def engine_knob_ids(self, engine_id):
            return list(f.knob_ids)

        def benchmark(self, handle, variant_pack):
            f.calls.append(("benchmark", self.policy))
            if f.benchmark_error is not None:
                raise f.benchmark_error
            return _m([0.5, 0.5, 0.5, 9.0], host_ms=0.2)

        def execute_once(self, handle, variant_pack):
            f.calls.append(("execute_once",))

    class BufferManager:
        def __init__(self, tensor_infos, device=None):
            self.tensor_infos = tensor_infos

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def allocate_all(self):
            pass

        def load_input_data(self, data):
            f.inputs = data

        def zero_outputs(self):
            f.calls.append(("zero_outputs",))

        def create_variant_pack(self):
            return {}

        def get_output_data(self, uid):
            return f.output

    monkeypatch.setattr(rocm_runtime, "initialize_pip_rocm_runtime", lambda: None)
    monkeypatch.setitem(
        sys.modules, "hipdnn_frontend", SimpleNamespace(Handle=lambda: "handle")
    )
    monkeypatch.setattr(
        suite_runner,
        "set_plugin_path",
        lambda hipdnn, path: f.calls.append(("plugin", path)),
    )
    monkeypatch.setattr(executor, "Executor", Executor)
    monkeypatch.setattr(buffer_manager, "BufferManager", BufferManager)
    return f


def _child(tmp_path, **request):
    """Run the child's main() in-process; return (result, request)."""
    base = {
        "graph_json": GRAPH_FILE.read_text(),
        "engine_id": 5,
        "plugin_path": "/plugins",
        "seed": 11,
        "policy": dataclasses.asdict(TimingPolicy(warmup_iters=3, iters=5)),
        "outputs": True,
        "workspace": True,
    }
    base.update(request)
    path = tmp_path / "request.json"
    path.write_text(json.dumps(base))
    assert hipdnn_tuned_child.main(["child", str(path)]) == 0
    return json.loads((tmp_path / "result.json").read_text()), base


def test_child_primes_then_times_a_benchmarking_build_of_the_engine(
    child_fakes, tmp_path
):
    """Same build state and timing as the OOTB row: a prime, then the timed
    build of the requested engine with global.benchmarking=1, timed with the
    run's policy, on the run's inputs, from the row's plugin."""
    result, request = _child(tmp_path)

    assert child_fakes.calls[:4] == [
        ("plugin", "/plugins"),
        ("prime", 5),
        ("prepare", 5, {"global.benchmarking": 1}),
        ("zero_outputs",),
    ]
    assert ("benchmark", TimingPolicy(warmup_iters=3, iters=5)) in child_fakes.calls
    graph_json = json.loads(request["graph_json"])
    expected = buffer_manager.generate_input_data(
        GraphLoader().extract_tensor_info(graph_json), 11, graph_json
    )
    assert child_fakes.inputs.keys() == expected.keys()
    for uid, data in expected.items():
        np.testing.assert_array_equal(child_fakes.inputs[uid], data)
    assert result["cpu_build_time_ms"] == 3.0
    assert result["workspace_bytes"] == 4096
    assert Measurement(**result["measurement"]) == _m([0.5, 0.5, 0.5, 9.0], 0.2)


@pytest.mark.parametrize("exposed", [True, False])
def test_child_tuning_available_follows_the_engine_knobs(
    child_fakes, tmp_path, exposed
):
    """hipDNN ignores a knob the engine does not expose; the tuned run then
    re-measures the OOTB configuration and must say so."""
    if not exposed:
        child_fakes.knob_ids = ["miopen.find_mode"]

    result, _ = _child(tmp_path)

    assert result["tuning_available"] is exposed


def test_child_writes_outputs_from_one_execution_on_zeroed_buffers(
    child_fakes, tmp_path
):
    _child(tmp_path)

    assert child_fakes.calls[-2:] == [("zero_outputs",), ("execute_once",)]
    with np.load(tmp_path / "outputs.npz") as saved:
        assert set(saved.files) == {"3"}  # the sample graph's output UID
        np.testing.assert_array_equal(saved["3"], child_fakes.output)


def test_child_without_outputs_or_workspace_writes_neither(child_fakes, tmp_path):
    result, _ = _child(tmp_path, outputs=False, workspace=False)

    assert ("execute_once",) not in child_fakes.calls
    assert not (tmp_path / "outputs.npz").exists()
    assert result["workspace_bytes"] is None


@pytest.mark.parametrize(
    ("error", "key", "message"),
    [
        (StallFallbackError("watchdog released"), "stall", "watchdog released"),
        (ExecutionError("launch failed"), "error", "ExecutionError: launch failed"),
    ],
)
def test_child_reports_failures_in_its_result(
    child_fakes, tmp_path, error, key, message
):
    child_fakes.benchmark_error = error

    result, _ = _child(tmp_path)

    assert result == {key: message}


# --- run_hipdnn_tuned (parent side) -----------------------------------------


@pytest.fixture
def hipdnn_child(monkeypatch):
    """Fake ``run_capped`` for the hipDNN child: writes result and outputs."""
    launch = SimpleNamespace(
        argv=None,
        env=None,
        request=None,
        state=None,
        result={
            "tuning_available": True,
            "cpu_build_time_ms": 3.0,
            "workspace_bytes": 4096,
            "measurement": dataclasses.asdict(_m([0.5, 0.5, 0.5, 9.0], 0.2)),
        },
        output=np.zeros(2, np.float32),
        timeout=None,
        proc=SimpleNamespace(returncode=0, stdout="", stderr=""),
    )

    def run_capped(argv, timeout_s, env=None):
        launch.argv, launch.env, launch.timeout = argv, env, timeout_s
        launch.state = Path(argv[-1]).parent
        launch.request = json.loads(Path(argv[-1]).read_text())
        if launch.result is not None:
            (launch.state / "result.json").write_text(json.dumps(launch.result))
            if launch.request["outputs"]:
                np.savez(launch.state / "outputs.npz", **{"1": launch.output})
        return launch.proc

    monkeypatch.setattr(oracle_mod, "run_capped", run_capped)
    return launch


def _tuned(row=None, refs=None, config=None, plugin_path="/plugins"):
    row = row or _row()
    oracle_mod.run_hipdnn_tuned(
        row=row,
        graph_json_str='{"name": "g"}',
        graph_name="g",
        engine_id=5,
        plugin_path=plugin_path,
        config=config or _config(),
        tensor_infos=[_OUT],
        reference_outputs=refs,
    )
    return row


def test_tuned_child_gets_the_row_engine_and_the_run_settings(hipdnn_child):
    config = _config(warmup_iters=3, benchmark_iters=5, min_time_ms=2.0, seed=11)
    config.metrics.profiling_timeout_s = 7

    _tuned(config=config, refs=_refs())

    assert hipdnn_child.argv[1:3] == [
        "-m",
        "dnn_benchmarking.execution.hipdnn_tuned_child",
    ]
    assert hipdnn_child.request == {
        "graph_json": '{"name": "g"}',
        "engine_id": 5,
        "plugin_path": "/plugins",
        "seed": 11,
        "policy": dataclasses.asdict(config.timing_policy),
        "outputs": True,
        "workspace": True,
    }
    assert hipdnn_child.timeout == 7


def test_tuned_child_writes_no_selection_state_the_run_can_read(
    hipdnn_child, monkeypatch
):
    """hipDNN disk caches are off and the MIOpen user database is the child's
    own, deleted with it: no later OOTB row can serve the tuned selection."""
    monkeypatch.setenv("MIOPEN_USER_DB_PATH", "/run/private/miopen")
    monkeypatch.delenv("HIPDNN_DISABLE_CACHE", raising=False)

    _tuned()

    env = hipdnn_child.env
    assert env["HIPDNN_DISABLE_CACHE"] == "1"
    assert Path(env["MIOPEN_USER_DB_PATH"]).parent == hipdnn_child.state
    assert not hipdnn_child.state.exists()
    assert "HIPDNN_DISABLE_CACHE" not in os.environ


def test_tuned_row_reports_the_child_build_timing_and_throughput(hipdnn_child):
    """TFLOP/s and GB/s use the row's analytical FLOPs/bytes over the TUNED
    kernel median (0.5 ms), not the mean or the OOTB median."""
    row = _row()
    row.analytical_flops = 10**9
    row.analytical_io_bytes = 10**6

    o = _tuned(row=row).oracle

    assert row.oracle_error is None
    assert o.tuning_available is True
    assert o.cpu_build_time_ms == pytest.approx(3.0)
    assert o.timing.first_call_ms == pytest.approx(1.0)
    assert o.gpu_kernel_stats.median_ms == pytest.approx(0.5)
    assert o.host_stats.median_ms == pytest.approx(0.2)
    assert o.workspace_bytes == 4096
    assert o.derived_tflops_per_s == pytest.approx(2.0)
    assert o.derived_gbytes_per_s == pytest.approx(2.0)
    assert o.correctness is None


def test_metrics_off_asks_the_child_for_no_workspace(hipdnn_child):
    _tuned(config=_config(metrics=MetricsConfig(basic=False)))

    assert hipdnn_child.request["workspace"] is False


@pytest.mark.parametrize(
    ("output", "passed"),
    [(np.zeros(2, np.float32), True), (np.ones(2, np.float32), False)],
)
def test_tuned_outputs_are_validated_and_keep_the_row_verdict(
    hipdnn_child, output, passed
):
    hipdnn_child.output = output
    ootb = CorrectnessResult(tolerance_match=True, rtol=1e-5, atol=1e-6)

    row = _tuned(row=_row(ootb), refs=_refs())

    assert row.oracle.correctness.tolerance_match is passed
    assert row.ootb.correctness is ootb
    assert row.verdict == "passed"


def test_child_stall_propagates_for_a_whole_graph_remeasure(hipdnn_child):
    hipdnn_child.result = {"stall": "watchdog released the stream"}

    with pytest.raises(StallFallbackError, match="watchdog released"):
        _tuned()


def test_child_error_is_an_oracle_error(hipdnn_child):
    hipdnn_child.result = {"error": "ExecutionError: launch failed"}

    row = _tuned()

    assert row.oracle is None
    assert row.oracle_error == "RuntimeError: ExecutionError: launch failed"
    assert row.status == "success"


def test_child_crash_reports_its_last_error_line(hipdnn_child):
    hipdnn_child.result = None
    hipdnn_child.proc = SimpleNamespace(
        returncode=-11, stdout="", stderr="Fatal: hipErrorIllegalAddress\n"
    )

    row = _tuned()

    assert row.oracle is None and row.status == "success"
    assert row.oracle_error == (
        "RuntimeError: tuned hipDNN child exited -11: Fatal: hipErrorIllegalAddress"
    )


# --- run_pytorch_tuned ------------------------------------------------------


def _child_row(status="success", message=None, kernel_ms=(0.25, 0.25, 0.3)):
    timing = TimingInfo(
        mode="staged",
        timer="hip",
        warmup_iters=3,
        samples=len(kernel_ms),
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


def _run_pytorch(tmp_path, config=None, node_type="ConvolutionFwdAttributes"):
    row = ProviderEngineResult(runtime="pytorch", engine_id=None, status="success")
    row.analytical_flops = 10**9
    oracle_mod.run_pytorch_tuned(
        row=row,
        graph_path=tmp_path / "g.json",
        graph_json={"nodes": [{"type": node_type}]},
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
    # Launching plain ``dnn_benchmarking`` would time untuned PyTorch.
    assert child.argv[1:3] == ["-m", "dnn_benchmarking.cli.pytorch_tuned_child"]
    assert args.graph == [str(tmp_path / "g.json")]
    assert child_config.runtime == "pytorch"
    assert child_config.oracle is False
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


@pytest.mark.parametrize(
    ("node_type", "available"),
    [("MatmulAttributes", True), ("SdpaFwdAttributes", False)],
)
def test_pytorch_tuning_is_available_only_for_searched_ops(
    child, tmp_path, node_type, available
):
    """The child searches convs and GEMMs only; any other graph re-measures
    the OOTB kernels and must stay out of the geomean."""
    row = _run_pytorch(tmp_path, node_type=node_type)
    assert row.oracle.tuning_available is available


def test_pytorch_child_timeout_is_an_oracle_error(monkeypatch, tmp_path):
    """A wedged search must not hang the suite: --profiling-timeout bounds it."""
    seen = []

    def run_capped(argv, timeout_s, env=None):
        seen.append(timeout_s)
        raise subprocess.TimeoutExpired(argv, timeout_s)

    monkeypatch.setattr(oracle_mod, "run_capped", run_capped)
    config = _pytorch_config()
    config.metrics.profiling_timeout_s = 7

    row = _run_pytorch(tmp_path, config)

    assert seen == [7]
    assert row.oracle is None and row.status == "success"
    assert "timed out after 7 s" in row.oracle_error


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

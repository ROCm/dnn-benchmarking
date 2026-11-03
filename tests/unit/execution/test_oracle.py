# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for execution.oracle (tuned run next to the OOTB run)."""

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from dnn_benchmarking.cli import hipdnn_tuned_child
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

OUT = TensorInfo(
    uid=1,
    name="y",
    dims=[2],
    strides=[1],
    data_type="float",
    is_virtual=False,
    is_output=True,
)


@pytest.fixture(autouse=True)
def tuned(tmp_path, monkeypatch):
    """The tuned state of the default work dir, so no test writes to ~/.cache."""
    monkeypatch.setenv("DNN_BENCH_WORKSPACE", str(tmp_path / "workspace"))
    return tmp_path / "workspace" / "work" / "tuned"


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


class _Fake:
    """Knobs and records of the fake hipDNN inside the tuned child."""

    def __init__(self):
        self.knob_ids = [oracle_mod.BENCHMARKING_KNOB, "miopen.find_mode"]
        self.kernel_ms = [0.5, 0.5, 0.5, 9.0]
        self.output = np.zeros(2, np.float32)
        self.prepare_error = None
        self.benchmark_error = None
        self.calls = []  # (call, engine_id, knobs) in order
        self.policies = []
        self.plugin_paths = []
        self.seeds = []
        self.child_env = {}  # selection variables the child ran with
        self.launches = []  # (argv, timeout_s) of each child launch


@pytest.fixture
def fake(monkeypatch):
    """Runs the real tuned child in this process, with fake hipDNN."""
    f = _Fake()

    class Executor:
        def __init__(self, graph_json_str, policy):
            f.policies.append(policy)
            self.build_time_ms, self.workspace_size = 3.0, 4096

        def prime(self, handle, engine_id):
            f.calls.append(("prime", engine_id, None))

        def prepare(self, handle, engine_id=None, knobs=None):
            f.calls.append(("prepare", engine_id, knobs))
            if f.prepare_error is not None:
                raise f.prepare_error

        def engine_knob_ids(self, engine_id):
            return list(f.knob_ids)

        def benchmark(self, handle, variant_pack):
            f.calls.append(("benchmark", None, None))
            if f.benchmark_error is not None:
                raise f.benchmark_error
            return _m(f.kernel_ms, host_ms=0.2)

        def execute_once(self, handle, variant_pack):
            f.calls.append(("execute_once", None, None))

    class BufferManager:
        def __init__(self, tensor_infos):
            self.written = False

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def allocate_all(self):
            pass

        def load_input_data(self, data):
            pass

        def zero_outputs(self):
            f.calls.append(("zero_outputs", None, None))

        def create_variant_pack(self):
            return {}

        def get_output_tensors(self):
            return [OUT]

        def get_output_data(self, uid):
            return f.output

    class Loader:
        def load_json(self, path):
            return {"nodes": []}

        def validate(self, graph_json):
            pass

        def extract_tensor_info(self, graph_json):
            return [OUT]

    def generate_input_data(tensor_infos, seed, graph_json):
        f.seeds.append(seed)
        return {}

    def run_capped(argv, timeout_s, env=None):
        """The child script, in this process, with the env the parent built."""
        assert argv[1:3] == ["-m", "dnn_benchmarking.cli.hipdnn_tuned_child"]
        f.launches.append((argv, timeout_s))
        names = [*oracle_mod.TUNED_CHILD_ENV, "HIPDNN_CACHE_DIR", "MIOPEN_USER_DB_PATH"]
        f.child_env = {name: env.get(name) for name in names}
        with monkeypatch.context() as m:
            for name in names:
                m.setenv(name, env[name])
            code = hipdnn_tuned_child.main(argv[3:])
        return SimpleNamespace(returncode=code, stdout="", stderr="")

    child = hipdnn_tuned_child
    monkeypatch.setattr(child, "Executor", Executor)
    monkeypatch.setattr(child, "BufferManager", BufferManager)
    monkeypatch.setattr(child, "GraphLoader", Loader)
    monkeypatch.setattr(child, "generate_input_data", generate_input_data)
    monkeypatch.setattr(child, "initialize_pip_rocm_runtime", lambda: None)
    monkeypatch.setattr(
        child, "set_plugin_path", lambda hipdnn, path: f.plugin_paths.append(path)
    )
    monkeypatch.setitem(
        sys.modules, "hipdnn_frontend", SimpleNamespace(Handle=SimpleNamespace)
    )
    monkeypatch.setattr(oracle_mod, "run_capped", run_capped)
    return f


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


def _run(tmp_path, row=None, refs=None, config=None, plugin_path=None):
    row = row or _row()
    oracle_mod.run_hipdnn_tuned(
        row=row,
        graph_path=tmp_path / "g.json",
        graph_name="g",
        tensor_infos=[OUT],
        reference_outputs=refs,
        engine_id=5,
        plugin_path=plugin_path,
        config=config or _config(),
    )
    return row


def _refs():
    return {1: ReferenceOutput(data=np.zeros(2, np.float32), tensor_uid=1)}


# --- run_hipdnn_tuned -------------------------------------------------------


def test_tuned_run_is_a_benchmarking_build_of_the_row_engine_with_the_run_policy(
    fake, tmp_path
):
    """The child primes the engine, builds it with the knob, and times it with
    the parent's full timing policy (stall gate included) and seed."""
    config = _config(warmup_iters=3, benchmark_iters=5, min_time_ms=2.0, seed=11)
    config.timing_policy = TimingPolicy(
        warmup_iters=3, iters=5, min_time_ms=2.0, stall_gate=False
    )

    _run(tmp_path, config=config, plugin_path=tmp_path / "libmiopen_plugin.so")

    assert fake.calls[:3] == [
        ("prime", 5, None),
        ("prepare", 5, {"global.benchmarking": 1}),
        ("zero_outputs", None, None),
    ]
    assert fake.policies[-1] == config.timing_policy
    assert fake.seeds == [11]
    assert fake.plugin_paths == [tmp_path / "libmiopen_plugin.so"]


def test_tuned_child_runs_benchmarking_on_the_shared_tuned_state(
    fake, tmp_path, monkeypatch, tuned
):
    """An inherited HIPDNN_FORCE_BENCHMARKING=0 would make the tuned run
    untuned; the tuned hipDNN cache and MIOpen database are the kept ones of
    the work dir, not this process's empty OOTB state."""
    monkeypatch.setenv("HIPDNN_FORCE_BENCHMARKING", "0")
    monkeypatch.setenv("HIPDNN_CACHE_DIR", str(tmp_path / "ootb-cache"))
    monkeypatch.setenv("MIOPEN_USER_DB_PATH", str(tmp_path / "ootb-db"))

    _run(tmp_path)

    assert fake.child_env == {
        "HIPDNN_FORCE_BENCHMARKING": "1",
        "HIPDNN_CACHE_DIR": str(tuned / "hipdnn-cache"),
        "MIOPEN_USER_DB_PATH": str(tuned / "miopen-user-db"),
    }
    assert os.environ["HIPDNN_CACHE_DIR"] == str(tmp_path / "ootb-cache")
    assert os.environ["MIOPEN_USER_DB_PATH"] == str(tmp_path / "ootb-db")


def test_the_tuned_state_follows_the_work_dir(fake, tmp_path):
    _run(tmp_path, config=_config(benchmark_work_dir=str(tmp_path / "w")))

    assert fake.child_env["HIPDNN_CACHE_DIR"] == str(
        tmp_path / "w" / "tuned" / "hipdnn-cache"
    )


def test_the_child_refuses_an_environment_the_parent_did_not_set(tmp_path, monkeypatch):
    """A child started without the tuned environment would time untuned
    kernels or use another state; it must fail, not run."""
    monkeypatch.setenv("HIPDNN_FORCE_BENCHMARKING", "0")
    monkeypatch.delenv("MIOPEN_USER_DB_PATH", raising=False)
    monkeypatch.setenv("HIPDNN_CACHE_DIR", str(tmp_path / "missing"))
    argv = ["--graph", "g.json", "--engine", "5", "--seed", "0"]
    argv += ["--timing-policy", "{}", "--out-dir", str(tmp_path)]

    assert hipdnn_tuned_child.main(argv) == 1

    failure = json.loads((tmp_path / oracle_mod.FAILURE_FILE).read_text())
    assert failure["kind"] == "error"
    names = ("HIPDNN_FORCE_BENCHMARKING", "HIPDNN_CACHE_DIR", "MIOPEN_USER_DB_PATH")
    for name in names:
        assert name in failure["message"]


@pytest.mark.parametrize("exposed", [True, False])
def test_tuning_available_follows_the_engine_knobs(fake, tmp_path, exposed):
    """hipDNN ignores a knob the engine does not expose; the tuned run then
    re-measures the OOTB configuration and must say so."""
    if not exposed:
        fake.knob_ids = ["miopen.find_mode"]

    assert _run(tmp_path).oracle.tuning_available is exposed


def test_tuned_run_reports_its_own_build_timing_and_throughput(fake, tmp_path):
    """TFLOP/s and GB/s use the row's analytical FLOPs/bytes over the TUNED
    kernel median (0.5 ms), not the mean or the OOTB median."""
    row = _row()
    row.analytical_flops = 10**9
    row.analytical_io_bytes = 10**6

    o = _run(tmp_path, row=row).oracle

    assert row.oracle_error is None
    assert o.cpu_build_time_ms == pytest.approx(3.0)
    assert o.timing.first_call_ms == pytest.approx(1.0)
    assert o.gpu_kernel_stats.median_ms == pytest.approx(0.5)
    assert o.host_stats.median_ms == pytest.approx(0.2)
    assert o.workspace_bytes == 4096
    assert o.derived_tflops_per_s == pytest.approx(2.0)
    assert o.derived_gbytes_per_s == pytest.approx(2.0)


def test_workspace_is_not_reported_with_metrics_off(fake, tmp_path):
    config = _config(metrics=MetricsConfig(basic=False))

    assert _run(tmp_path, config=config).oracle.workspace_bytes is None


def test_tuned_outputs_are_saved_from_zeroed_buffers_and_validated_here(fake, tmp_path):
    row = _run(tmp_path, refs=_refs())

    assert fake.calls[2:] == [
        ("zero_outputs", None, None),
        ("benchmark", None, None),
        ("zero_outputs", None, None),
        ("execute_once", None, None),
    ]
    assert row.oracle.correctness.tolerance_match


def test_no_references_means_no_tuned_validation(fake, tmp_path):
    row = _run(tmp_path)

    assert ("execute_once", None, None) not in fake.calls
    assert row.oracle.correctness is None


def test_failing_tuned_plan_keeps_the_row_verdict(fake, tmp_path):
    passed = CorrectnessResult(tolerance_match=True, rtol=1e-5, atol=1e-6)
    fake.output = np.ones(2, np.float32)

    row = _run(tmp_path, row=_row(passed), refs=_refs())

    assert row.oracle.correctness.explicitly_failed
    assert row.ootb.correctness is passed
    assert row.verdict == "passed"


def test_stall_in_the_child_propagates_for_a_whole_graph_remeasure(fake, tmp_path):
    fake.benchmark_error = StallFallbackError("watchdog released the stream")

    with pytest.raises(StallFallbackError, match="watchdog released the stream"):
        _run(tmp_path)


@pytest.mark.parametrize("where", ["prepare", "benchmark"])
def test_tuned_child_failure_is_an_oracle_error(fake, tmp_path, where):
    setattr(fake, f"{where}_error", ExecutionError("launch failed"))

    row = _run(tmp_path)

    assert row.oracle is None
    assert row.oracle_error == "RuntimeError: ExecutionError: launch failed"
    assert row.status == "success"


def test_tuned_child_timeout_is_an_oracle_error(monkeypatch, tmp_path):
    """A wedged search must not hang the suite: --profiling-timeout bounds it."""
    seen = []

    def run_capped(argv, timeout_s, env=None):
        seen.append(timeout_s)
        raise subprocess.TimeoutExpired(argv, timeout_s)

    monkeypatch.setattr(oracle_mod, "run_capped", run_capped)
    config = _config()
    config.metrics.profiling_timeout_s = 7

    row = _run(tmp_path, config=config)

    assert seen == [7]
    assert row.oracle is None and "timed out after 7 s" in row.oracle_error


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


def test_pytorch_child_tunes_on_the_shared_state_and_a_deleted_state_dir(
    child, tmp_path, tuned
):
    """TunableOp results must not outlive the child; MIOpen's find results go
    to the kept tuned state that every tuned run shares."""
    _run_pytorch(tmp_path)

    output = Path(child.argv[child.argv.index("--output") + 1])
    work = Path(child.argv[child.argv.index("--benchmark-work-dir") + 1])
    assert work / "tuned" == tuned
    assert child.env["MIOPEN_USER_DB_PATH"] == str(tuned / "miopen-user-db")
    assert child.env["HIPDNN_CACHE_DIR"] == str(tuned / "hipdnn-cache")
    assert (tuned / "miopen-user-db").is_dir()
    assert child.env["PYTORCH_TUNABLEOP_ENABLED"] == "1"
    assert not output.parent.exists()


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

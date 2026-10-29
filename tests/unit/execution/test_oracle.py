# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for execution.oracle (tuned run in a child script next to the OOTB run)."""

import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from dnn_benchmarking.cli import hipdnn_child, pytorch_child
from dnn_benchmarking.common.run_state import TEMP_ROOT_ENV
from dnn_benchmarking.config.benchmark_config import (
    MetricsConfig,
    SuiteConfig,
    ValidationConfig,
)
from dnn_benchmarking.execution import oracle as oracle_mod
from dnn_benchmarking.execution import child_protocol, child_runner
from dnn_benchmarking.execution.timing import Measurement
from dnn_benchmarking.graph.tensor_info import TensorInfo
from dnn_benchmarking.reporting.suite_results import (
    CorrectnessResult,
    PlanResult,
    ProviderEngineResult,
    oracle_speedup,
)
from dnn_benchmarking.validation import ReferenceOutput

# Signed 64-bit hash, as hipDNN reports it; the child takes it in hex.
ENGINE = -2
REF = np.arange(4, dtype=np.float32)


def _tensor(uid, is_output):
    return TensorInfo(
        uid=uid,
        name=f"t{uid}",
        dims=[4],
        strides=[1],
        data_type="float",
        is_virtual=False,
        is_output=is_output,
    )


TENSORS = [_tensor(1, False), _tensor(2, True)]


def _references():
    return {2: ReferenceOutput(data=REF.copy(), tensor_uid=2)}


def _measurement():
    return Measurement(
        kernel_ms=[0.25, 0.25, 0.3],
        host_ms=[0.5, 0.5, 0.5],
        mode="staged",
        timer="hip",
        cache_mode="warm",
        warmup_iters=3,
        first_call_ms=40.0,
    )


@pytest.fixture
def child(monkeypatch):
    """Fake ``run_capped``: records the child launch and writes its files.

    ``plan`` goes into ``result.json`` (None: no file). With
    ``--save-outputs`` the child saves ``outputs`` (uid -> array). An
    ``error`` is raised in the child and reported as the real scripts do.
    """
    launch = SimpleNamespace(
        argv=None,
        env=None,
        out_dir=None,
        state_existed=None,
        plan={"build_ms": 7.0, "workspace_bytes": 4096},
        outputs={2: REF.copy()},
        error=None,
        returncode=0,
    )

    def body(out_dir):
        if launch.error is not None:
            raise launch.error
        if launch.plan is not None:
            child_protocol.write_result(out_dir, _measurement(), **launch.plan)
        if "--save-outputs" in launch.argv:
            for uid, data in launch.outputs.items():
                np.save(child_protocol.output_file(out_dir, uid), data)

    def run_capped(argv, timeout_s, env=None, capture_output=True):
        launch.argv, launch.env = argv, env
        out_dir = launch.out_dir = Path(argv[argv.index("--out-dir") + 1])
        launch.state_existed = out_dir.is_dir()
        code = child_protocol.run_reporting_failures(out_dir, lambda: body(out_dir))
        return SimpleNamespace(returncode=code or launch.returncode)

    monkeypatch.setattr(child_runner, "run_capped", run_capped)
    return launch


def _config(**kw):
    kw.setdefault("validation", ValidationConfig(provider="pytorch"))
    return SuiteConfig(oracle=True, **kw)


def _pytorch_config(**kw):
    return SuiteConfig(oracle=True, runtime="pytorch", **kw)


def _row(correctness=None):
    row = ProviderEngineResult(
        runtime="hipdnn",
        engine_id=ENGINE,
        status="success",
        ootb=PlanResult(correctness=correctness),
    )
    row.analytical_flops = 10**9
    row.analytical_io_bytes = 10**6
    return row


def _run_hipdnn(
    tmp_path,
    row=None,
    config=None,
    plugin_path=None,
    tuning_available=True,
    references=None,
):
    row = row or _row()
    oracle_mod.run_hipdnn_tuned(
        row=row,
        graph_path=tmp_path / "g.json",
        graph_name="g",
        tensor_infos=TENSORS,
        reference_outputs=references,
        engine_id=ENGINE,
        plugin_path=plugin_path,
        tuning_available=tuning_available,
        config=config or _config(),
    )
    return row


def _timeout(monkeypatch, seen):
    def run_capped(argv, timeout_s, env=None, capture_output=True):
        seen.append(timeout_s)
        raise subprocess.TimeoutExpired(argv, timeout_s)

    monkeypatch.setattr(child_runner, "run_capped", run_capped)


def _set_tuned_env(monkeypatch, env, state_dir):
    """Set the tuning variables of ``env`` as the parent does."""
    built = env.build(str(state_dir))
    for name in [*env.fixed, *env.in_dir]:
        monkeypatch.setenv(name, built[name])


# --- run_hipdnn_tuned -------------------------------------------------------


@pytest.mark.parametrize("stall_gate", [True, False])
def test_hipdnn_child_times_the_row_engine_with_the_run_timing(
    child, tmp_path, stall_gate
):
    """The child must time the same engine, inputs and timing mode as the
    parent, or the tuned and OOTB numbers are not comparable. A graph that
    fell back to unstalled timing passes ``stall_gate=False`` on."""
    config = _config(
        warmup_iters=3,
        benchmark_iters=5,
        min_time_ms=2.0,
        timing_block=4,
        seed=11,
    )
    config.timing_policy = replace(config.timing_policy, stall_gate=stall_gate)

    _run_hipdnn(tmp_path, config=config, plugin_path=Path("/plugins/p.so"))

    assert child.argv[1:3] == ["-m", "dnn_benchmarking.cli.hipdnn_child"]
    args = hipdnn_child.parse_args(child.argv[3:])
    assert args.tuned
    assert args.graph == tmp_path / "g.json"
    assert args.engine == ENGINE
    assert args.plugin_path == Path("/plugins/p.so")
    assert args.seed == 11
    assert args.timing_policy == config.timing_policy
    assert args.out_dir == child.out_dir


def test_hipdnn_child_uses_the_default_plugins_without_a_plugin_path(child, tmp_path):
    _run_hipdnn(tmp_path)

    assert hipdnn_child.parse_args(child.argv[3:]).plugin_path is None


def test_hipdnn_child_env_forces_benchmarking_into_a_private_deleted_user_db(
    child, tmp_path, monkeypatch
):
    """The parent sets the child's tuning environment, also over inherited
    values. The search writes MIOpen's user db; it may not reuse earlier
    tuning or outlive the child, or a later OOTB run would serve the tuned
    kernel."""
    monkeypatch.setenv("HIPDNN_FORCE_BENCHMARKING", "0")
    monkeypatch.setenv("HIPDNN_DISABLE_CACHE", "0")

    _run_hipdnn(tmp_path)

    state_dir = Path(child.env["MIOPEN_USER_DB_PATH"])
    assert child.env["HIPDNN_FORCE_BENCHMARKING"] == "1"
    assert child.env["HIPDNN_DISABLE_CACHE"] == "1"
    assert child.state_existed and child.out_dir == state_dir
    assert not state_dir.exists()


def test_child_user_db_is_under_the_temp_root_and_deleted(child, tmp_path, monkeypatch):
    """With --benchmark-work-dir, the child's private user db lives under
    the work dir's ``tmp``; it is still new for the child and deleted after it."""
    root = tmp_path / "work-dir" / "tmp"
    monkeypatch.setenv(TEMP_ROOT_ENV, str(root))

    _run_hipdnn(tmp_path)

    state_dir = Path(child.env["MIOPEN_USER_DB_PATH"])
    assert state_dir.parent == root
    assert child.state_existed and not state_dir.exists()


def test_the_parent_env_satisfies_the_hipdnn_child_check(child, tmp_path, monkeypatch):
    _run_hipdnn(tmp_path)
    for name, value in child.env.items():
        monkeypatch.setenv(name, value)

    child_protocol.HIPDNN_TUNED.env.check(child.out_dir)


@pytest.mark.parametrize(
    "name, value",
    [
        ("HIPDNN_FORCE_BENCHMARKING", None),
        ("HIPDNN_FORCE_BENCHMARKING", "0"),
        ("HIPDNN_DISABLE_CACHE", "0"),
        ("MIOPEN_USER_DB_PATH", None),
        ("MIOPEN_USER_DB_PATH", "/shared/miopen-db"),
    ],
)
def test_hipdnn_child_refuses_to_time_without_the_tuning_env(
    tmp_path, monkeypatch, name, value
):
    """An untuned run must not be reported as the tuned result, and a run
    with another run's MIOpen user database is not isolated."""
    _set_tuned_env(monkeypatch, child_protocol.HIPDNN_TUNED.env, tmp_path)
    if value is None:
        monkeypatch.delenv(name)
    else:
        monkeypatch.setenv(name, value)
    argv = ["--graph", "g.json", "--seed", "0", "--engine", "0x1", "--tuned"]
    argv += ["--timing-policy", "{}", "--out-dir", str(tmp_path)]

    assert hipdnn_child.main(argv) == 1
    assert not (tmp_path / child_protocol.RESULT_FILE).exists()


@pytest.mark.parametrize(
    "name", ["PYTORCH_TUNABLEOP_ENABLED", "PYTORCH_TUNABLEOP_TUNING"]
)
def test_pytorch_child_refuses_to_time_without_the_tuning_env(
    tmp_path, monkeypatch, name
):
    _set_tuned_env(monkeypatch, child_protocol.PYTORCH_TUNED_ENV, tmp_path)
    monkeypatch.setenv(name, "0")
    argv = ["--graph", "g.json", "--seed", "0", "--timing-policy", "{}"]
    argv += ["--out-dir", str(tmp_path), "--pytorch-sdpa-backend", "default"]

    assert pytorch_child.main([*argv, "--tuned"]) == 1
    assert not (tmp_path / child_protocol.RESULT_FILE).exists()


def test_hipdnn_tuned_row_comes_from_the_child_result(child, tmp_path):
    """TFLOP/s and GB/s use the row's analytical FLOPs/bytes over the TUNED
    kernel median (0.25 ms), not the OOTB median."""
    row = _run_hipdnn(tmp_path)

    o = row.oracle
    assert row.oracle_error is None
    assert o.tuning_available is True
    assert o.cpu_build_time_ms == pytest.approx(7.0)
    assert o.workspace_bytes == 4096
    assert o.timing.first_call_ms == pytest.approx(40.0)
    assert o.timing.samples == 3
    assert o.gpu_kernel_stats.median_ms == pytest.approx(0.25)
    assert o.host_stats.median_ms == pytest.approx(0.5)
    assert o.derived_tflops_per_s == pytest.approx(4.0)
    assert o.derived_gbytes_per_s == pytest.approx(4.0)


def test_hipdnn_tuned_workspace_follows_the_metrics_switch(child, tmp_path):
    row = _run_hipdnn(tmp_path, config=_config(metrics=MetricsConfig(basic=False)))

    assert row.oracle.workspace_bytes is None


@pytest.mark.parametrize("available", [True, False])
def test_hipdnn_tuning_available_is_passed_through(child, tmp_path, available):
    """Without the benchmarking knob the child re-measures the OOTB
    configuration, and the row must say so."""
    row = _run_hipdnn(tmp_path, tuning_available=available)

    assert row.oracle.tuning_available is available


def test_tuned_outputs_are_validated_against_the_row_references(child, tmp_path):
    """A wrong-but-fast tuned kernel is caught by the same reference check
    and tolerances as the OOTB row."""
    row = _run_hipdnn(tmp_path, references=_references())

    assert "--save-outputs" in child.argv
    assert row.oracle.correctness.passed
    assert row.oracle.correctness.n_total == REF.size


def test_failing_tuned_outputs_keep_the_row_verdict(child, tmp_path):
    passed = CorrectnessResult(tolerance_match=True, rtol=1e-5, atol=1e-6)
    child.outputs = {2: REF + 1.0}

    row = _run_hipdnn(tmp_path, row=_row(passed), references=_references())

    assert row.oracle.correctness.explicitly_failed
    assert row.ootb.correctness is passed
    assert row.verdict == "passed"
    assert oracle_speedup(row) is None


def test_missing_tuned_output_fails_tuned_validation(child, tmp_path):
    child.outputs = {}

    row = _run_hipdnn(tmp_path, references=_references())

    assert row.oracle.correctness.explicitly_failed


def test_without_references_the_tuned_run_is_not_validated(child, tmp_path):
    row = _run_hipdnn(tmp_path, config=_config(validation=ValidationConfig()))

    assert "--save-outputs" not in child.argv
    assert row.oracle is not None and row.oracle.correctness is None


def test_hipdnn_child_timeout_is_an_oracle_error(monkeypatch, tmp_path):
    """A wedged search must not hang the suite: --profiling-timeout bounds it."""
    seen = []
    _timeout(monkeypatch, seen)
    config = _config()
    config.metrics.profiling_timeout_s = 7

    row = _run_hipdnn(tmp_path, config=config)

    assert seen == [7]
    assert row.oracle is None and row.status == "success"
    assert "timed out after 7 s" in row.oracle_error


@pytest.mark.parametrize("failure", ["error", "no-result"])
def test_hipdnn_child_failure_is_an_oracle_error(child, tmp_path, failure):
    if failure == "error":
        child.error = RuntimeError("HIP out of memory")
    else:
        child.plan = None

    row = _run_hipdnn(tmp_path, references=_references())

    assert row.oracle is None
    assert row.status == "success"
    if failure == "error":
        assert row.oracle_error == "RuntimeError: HIP out of memory"
    else:
        assert "exited 0" in row.oracle_error


# --- run_pytorch_tuned ------------------------------------------------------


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


@pytest.mark.parametrize("stall_gate", [True, False])
def test_pytorch_child_times_the_graph_with_the_run_timing(child, tmp_path, stall_gate):
    """The child must time with the parent's settings, or the tuned and OOTB
    PyTorch numbers are not comparable."""
    config = _pytorch_config(
        warmup_iters=3,
        benchmark_iters=5,
        min_time_ms=2.0,
        timing_block=4,
        seed=11,
        pytorch_sdpa_backend="flash",
        pytorch_rocm_fa_library="aotriton",
    )
    config.timing_policy = replace(config.timing_policy, stall_gate=stall_gate)

    _run_pytorch(tmp_path, config)

    assert child.argv[1:3] == ["-m", "dnn_benchmarking.cli.pytorch_child"]
    args = pytorch_child.parse_args(child.argv[3:])
    assert args.tuned
    assert args.graph == tmp_path / "g.json"
    assert args.seed == 11
    assert args.timing_policy == config.timing_policy
    assert args.pytorch_sdpa_backend is config.pytorch_sdpa_backend
    assert args.pytorch_rocm_fa_library == "aotriton"


def test_pytorch_child_tunes_into_a_private_state_dir_that_is_deleted(child, tmp_path):
    """MIOpen's user db and TunableOp results must not outlive the child, or a
    later OOTB run would serve the tuned selection."""
    _run_pytorch(tmp_path)

    state_dir = Path(child.env["MIOPEN_USER_DB_PATH"])
    assert child.state_existed and child.out_dir == state_dir
    assert child.env["PYTORCH_TUNABLEOP_ENABLED"] == "1"
    assert child.env["PYTORCH_TUNABLEOP_TUNING"] == "1"
    assert Path(child.env["PYTORCH_TUNABLEOP_FILENAME"]).parent == state_dir
    assert not state_dir.exists()


def test_pytorch_tuned_row_comes_from_the_child_result(child, tmp_path):
    child.plan = {}

    row = _run_pytorch(tmp_path)

    o = row.oracle
    assert row.oracle_error is None
    assert o.tuning_available is True
    assert (o.cpu_build_time_ms, o.workspace_bytes, o.correctness) == (None,) * 3
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
    _timeout(monkeypatch, seen)
    config = _pytorch_config()
    config.metrics.profiling_timeout_s = 7

    row = _run_pytorch(tmp_path, config)

    assert seen == [7]
    assert row.oracle is None and row.status == "success"
    assert "timed out after 7 s" in row.oracle_error


def test_pytorch_child_failure_reports_the_child_error(child, tmp_path):
    child.error = RuntimeError("HIP out of memory")

    row = _run_pytorch(tmp_path)

    assert row.oracle is None
    assert row.oracle_error == "RuntimeError: HIP out of memory"
    assert row.status == "success"

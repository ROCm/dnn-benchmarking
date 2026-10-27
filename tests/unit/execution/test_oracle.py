# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for execution.oracle (tuned run in a child next to the OOTB run)."""

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from dnn_benchmarking.cli.parser import create_parser
from dnn_benchmarking.config.benchmark_config import (
    MetricsConfig,
    SuiteConfig,
    ValidationConfig,
)
from dnn_benchmarking.execution import oracle as oracle_mod
from dnn_benchmarking.reporting.statistics import BenchmarkStats, TimingInfo
from dnn_benchmarking.reporting.suite_results import (
    CorrectnessResult,
    PlanResult,
    ProviderEngineResult,
    engine_id_hex,
    oracle_speedup,
)

# Signed 64-bit hash, as hipDNN reports it; the child takes it in hex.
ENGINE = -2


def _child_row(status="success", message=None, kernel_ms=(0.25, 0.25, 0.3), **plan):
    timing = TimingInfo(
        mode="staged",
        timer="hip",
        warmup_iters=3,
        samples=len(kernel_ms),
        first_call_ms=40.0,
    )
    return {
        "role": "engine",
        "status": status,
        "message": message,
        "ootb": {
            "timing": timing.to_dict(),
            "kernel": BenchmarkStats.from_timings(kernel_ms).to_dict(),
            "host": BenchmarkStats.from_timings([0.5, 0.5, 0.5]).to_dict(),
            **plan,
        },
    }


@pytest.fixture
def child(monkeypatch):
    """Fake ``run_capped``: records the child launch and writes its result."""
    launch = SimpleNamespace(
        argv=None,
        env=None,
        state_existed=None,
        result={"graphs": [{"results": [_child_row()]}]},
        proc=SimpleNamespace(returncode=0, stdout="", stderr=""),
    )

    def run_capped(argv, timeout_s, env=None):
        launch.argv, launch.env = argv, env
        state = (env or {}).get("MIOPEN_USER_DB_PATH")
        launch.state_existed = state is not None and Path(state).is_dir()
        if launch.result is not None:
            output = Path(argv[argv.index("--output") + 1])
            output.write_text(json.dumps(launch.result))
        return launch.proc

    monkeypatch.setattr(oracle_mod, "run_capped", run_capped)
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
    tmp_path, row=None, config=None, plugin_path=None, tuning_available=True
):
    row = row or _row()
    oracle_mod.run_hipdnn_tuned(
        row=row,
        graph_path=tmp_path / "g.json",
        graph_name="g",
        engine_id=ENGINE,
        plugin_path=plugin_path,
        tuning_available=tuning_available,
        config=config or _config(),
    )
    return row


def _child_args(child):
    assert child.argv[1:3] == ["-m", "dnn_benchmarking"]
    return create_parser().parse_args(child.argv[3:])


# --- run_hipdnn_tuned -------------------------------------------------------


def test_hipdnn_child_autotunes_the_row_engine_with_the_run_timing(child, tmp_path):
    """The child must time the same engine with the parent's settings, or the
    tuned and OOTB numbers are not comparable."""
    config = _config(
        warmup_iters=3,
        benchmark_iters=5,
        min_time_ms=2.0,
        timing_block=4,
        seed=11,
    )

    _run_hipdnn(tmp_path, config=config, plugin_path=Path("/plugins/p.so"))

    args = _child_args(child)
    child_config = SuiteConfig.from_namespace(args)
    assert args.graph == [str(tmp_path / "g.json")]
    assert child.argv[child.argv.index("--engine") + 1] == engine_id_hex(ENGINE)
    assert child_config.engine_filter == [ENGINE]
    assert child_config.autotune is True
    assert child_config.oracle is False
    assert child_config.plugin_paths == [Path("/plugins/p.so")]
    assert child_config.timing_policy == config.timing_policy
    assert child_config.seed == 11


def test_hipdnn_child_uses_the_default_plugins_without_a_plugin_path(child, tmp_path):
    _run_hipdnn(tmp_path)

    assert _child_args(child).plugin_path is None


@pytest.mark.parametrize(
    ("validation", "expected"),
    [
        (ValidationConfig(provider="pytorch"), ("pytorch", None, None)),
        (
            ValidationConfig(provider="pytorch", rtol=0.01, atol=0.002),
            ("pytorch", 0.01, 0.002),
        ),
        (ValidationConfig(), ("none", None, None)),
    ],
    ids=["default-tolerance", "tolerance", "off"],
)
def test_hipdnn_child_validates_like_the_parent(child, tmp_path, validation, expected):
    """A wrong-but-fast tuned kernel is caught by the same reference check."""
    _run_hipdnn(tmp_path, config=_config(validation=validation))

    args = _child_args(child)
    assert (args.validate, args.rtol, args.atol) == expected


@pytest.mark.parametrize("basic", [True, False])
def test_hipdnn_child_follows_the_metrics_switch(child, tmp_path, basic):
    _run_hipdnn(tmp_path, config=_config(metrics=MetricsConfig(basic=basic)))

    assert _child_args(child).metrics is basic


def test_hipdnn_child_tunes_into_private_caches_that_are_deleted(child, tmp_path):
    """The search writes hipDNN's winner cache and MIOpen's user db; neither
    may reuse earlier tuning or outlive the child, or a later OOTB run would
    serve the tuned kernel."""
    _run_hipdnn(tmp_path)

    state_dir = Path(child.env["MIOPEN_USER_DB_PATH"])
    output = Path(child.argv[child.argv.index("--output") + 1])
    cache_dir = Path(SuiteConfig.from_namespace(_child_args(child)).hipdnn_cache_dir)
    assert child.state_existed
    assert output.parent == state_dir
    assert cache_dir.is_relative_to(state_dir)
    assert not state_dir.exists()


def test_hipdnn_tuned_row_comes_from_the_child_ootb_plan(child, tmp_path):
    """TFLOP/s and GB/s use the row's analytical FLOPs/bytes over the TUNED
    kernel median (0.25 ms), not the OOTB median."""
    passed = CorrectnessResult(tolerance_match=True, rtol=1e-5, atol=1e-6)
    child.result = {
        "graphs": [
            {
                "results": [
                    _child_row(
                        build_ms=7.0,
                        workspace_bytes=4096,
                        correctness=passed.to_dict(),
                    )
                ]
            }
        ]
    }

    row = _run_hipdnn(tmp_path)

    o = row.oracle
    assert row.oracle_error is None
    assert o.tuning_available is True
    assert o.cpu_build_time_ms == pytest.approx(7.0)
    assert o.workspace_bytes == 4096
    assert o.timing.first_call_ms == pytest.approx(40.0)
    assert o.gpu_kernel_stats.median_ms == pytest.approx(0.25)
    assert o.host_stats.median_ms == pytest.approx(0.5)
    assert o.correctness.passed
    assert o.derived_tflops_per_s == pytest.approx(4.0)
    assert o.derived_gbytes_per_s == pytest.approx(4.0)


@pytest.mark.parametrize("available", [True, False])
def test_hipdnn_tuning_available_is_passed_through(child, tmp_path, available):
    """Without the benchmarking knob the child re-measures the OOTB
    configuration, and the row must say so."""
    row = _run_hipdnn(tmp_path, tuning_available=available)

    assert row.oracle.tuning_available is available


def test_failing_tuned_child_keeps_the_row_verdict(child, tmp_path):
    passed = CorrectnessResult(tolerance_match=True, rtol=1e-5, atol=1e-6)
    failed = CorrectnessResult(tolerance_match=False, rtol=1e-5, atol=1e-6)
    child.result = {"graphs": [{"results": [_child_row(correctness=failed.to_dict())]}]}

    row = _run_hipdnn(tmp_path, row=_row(passed))

    assert row.oracle.correctness.explicitly_failed
    assert row.ootb.correctness is passed
    assert row.verdict == "passed"
    assert oracle_speedup(row) is None


def test_hipdnn_child_timeout_is_an_oracle_error(monkeypatch, tmp_path):
    """A wedged search must not hang the suite: --profiling-timeout bounds it."""
    seen = []

    def run_capped(argv, timeout_s, env=None):
        seen.append(timeout_s)
        raise subprocess.TimeoutExpired(argv, timeout_s)

    monkeypatch.setattr(oracle_mod, "run_capped", run_capped)
    config = _config()
    config.metrics.profiling_timeout_s = 7

    row = _run_hipdnn(tmp_path, config=config)

    assert seen == [7]
    assert row.oracle is None and row.status == "success"
    assert row.oracle_error


@pytest.mark.parametrize("failure", ["no-output", "error-row", "no-row"])
def test_hipdnn_child_failure_is_an_oracle_error(child, tmp_path, failure):
    if failure == "no-output":
        child.result = None
        child.proc = SimpleNamespace(
            returncode=1, stdout="", stderr="RuntimeError: HIP out of memory\n"
        )
    elif failure == "error-row":
        child.result = {
            "graphs": [{"results": [_child_row("error", "no plan for engine")]}]
        }
    else:
        child.result = {"graphs": [{"results": []}]}

    row = _run_hipdnn(tmp_path)

    assert row.oracle is None
    assert row.oracle_error
    assert row.status == "success"


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

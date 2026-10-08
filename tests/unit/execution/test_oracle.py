# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for execution.oracle (autotuned plan vs heuristic plan)."""

import os
from types import SimpleNamespace

import numpy as np
import pytest

from dnn_benchmarking.common.exceptions import ExecutionError
from dnn_benchmarking.config.benchmark_config import (
    SuiteConfig,
    TimingPolicy,
    ValidationConfig,
)
from dnn_benchmarking.execution import oracle as oracle_mod
from dnn_benchmarking.execution.timing import Measurement
from dnn_benchmarking.graph.tensor_info import TensorInfo
from dnn_benchmarking.reporting.suite_results import (
    CorrectnessResult,
    PlanResult,
    ProviderEngineResult,
)
from dnn_benchmarking.validation import ReferenceOutput

ENV = ("HIPDNN_FORCE_BENCHMARKING", "HIPDNN_DISABLE_CACHE")
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


def _candidate(succeeded=True, excluded=False, rank=0):
    return SimpleNamespace(
        succeeded=succeeded,
        excluded_by_caller=excluded,
        rank=rank,
        compiled_plan_index=rank,
        min_time_ms=0.4,
        knob_settings=[],
        supports_exhaustive=True,
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
    """Stands in for the oracle's own Executor; records env at plan build."""

    candidates = [_candidate()]
    kernel_ms = [0.5, 0.5, 0.5, 9.0]
    prepare_error = None
    autotune_error = None
    env_at_prepare = None
    prepared_on = None
    policy = None
    for_autotune = None

    def __init__(self, graph_json_str, policy):
        self.init_time_ms = 3.0
        self.workspace_size = 4096
        type(self).policy = policy

    def prepare(self, handle, engine_id=None, for_autotune=False):
        type(self).env_at_prepare = {k: os.environ.get(k) for k in ENV}
        type(self).prepared_on = handle
        type(self).for_autotune = for_autotune
        if self.prepare_error is not None:
            raise self.prepare_error

    def autotune(self, handle, variant_pack, engine_id):
        CALLS.append(("tuned.autotune", handle))
        if self.autotune_error is not None:
            raise self.autotune_error
        return self.candidates

    def benchmark(self, handle, variant_pack):
        CALLS.append(("tuned.benchmark", handle))
        return _m(self.kernel_ms, host_ms=0.2)

    def execute_once(self, handle, variant_pack):
        CALLS.append(("tuned.execute_once", handle))

    def plan_name(self, handle):
        return "tuned"


@pytest.fixture
def tuned(monkeypatch):
    cls = type("Tuned", (_TunedExecutor,), {})
    monkeypatch.setattr(oracle_mod, "Executor", cls)
    CALLS.clear()
    return cls


def _run(
    mode="plan", correctness=None, bm=None, refs=None, flops=None, handle=None, **config
):
    row = ProviderEngineResult(
        runtime="hipdnn",
        engine_id=5,
        status="success",
        ootb=PlanResult(correctness=correctness),
    )
    row.analytical_flops = flops

    def baseline_benchmark(h, vp):
        CALLS.append(("baseline.benchmark", h))
        return _m([1.0, 1.0, 1.0, 10.0])

    baseline = SimpleNamespace(benchmark=baseline_benchmark)
    out = TensorInfo(
        uid=1,
        name="y",
        dims=[2],
        strides=[1],
        data_type="float",
        is_virtual=False,
        is_output=True,
    )
    oracle_mod.run_oracle_pass(
        row=row,
        handle=handle or _Handle(),
        engine_id=5,
        graph_json_str="{}",
        graph_name="g",
        config=SuiteConfig(
            oracle_mode=mode, validation=ValidationConfig(provider="pytorch"), **config
        ),
        bm=bm or _BM(),
        variant_pack={},
        ootb_executor=baseline,
        tensor_infos=[out],
        reference_outputs=refs,
    )
    return row


def test_delta_compares_post_sweep_medians(tuned):
    row = _run()

    assert row.oracle_error is None
    # Means (3.25 vs 2.625) would give 1.24x; medians give 2x.
    assert row.oracle_delta.basis == "kernel"
    assert row.oracle_delta.baseline_median_ms == pytest.approx(1.0)
    assert row.oracle_delta.oracle_median_ms == pytest.approx(0.5)
    assert row.oracle_delta.speedup == pytest.approx(2.0)


def test_tuned_plan_gets_its_own_handle_on_the_row_stream(tuned):
    """MIOpen's solver map is per handle; sharing it would let the sweep
    change the heuristic baseline's plan."""
    row_handle = _Handle()
    _run(handle=row_handle)

    assert tuned.prepared_on is not row_handle
    assert tuned.prepared_on.stream == 7


def test_tuned_plan_is_built_for_autotune_with_the_run_policy(tuned):
    _run(warmup_iters=3, benchmark_iters=5, min_time_ms=2.0)

    assert tuned.for_autotune is True
    assert tuned.policy == TimingPolicy(warmup_iters=3, iters=5, min_time_ms=2.0)


def test_baseline_times_on_row_handle_then_tuned_plan_validates_once(tuned):
    """The heuristic plan must keep the row handle (the tuned handle's solver
    map would hide the speedup), and validation must re-run the tuned plan on
    freshly zeroed outputs."""
    row_handle = _Handle()
    refs = {1: ReferenceOutput(data=np.zeros(2, np.float32), tensor_uid=1)}

    row = _run(handle=row_handle, refs=refs, bm=_BM({1: np.zeros(2, np.float32)}))

    tuned_handle = tuned.prepared_on
    assert CALLS == [
        ("zero_outputs", None),
        ("tuned.autotune", tuned_handle),
        ("zero_outputs", None),
        ("baseline.benchmark", row_handle),
        ("zero_outputs", None),
        ("tuned.benchmark", tuned_handle),
        ("zero_outputs", None),
        ("tuned.execute_once", tuned_handle),
    ]
    assert row.oracle.correctness.tolerance_match


def test_tuned_and_warm_heuristic_report_median_tflops(tuned):
    """Both oracle operands get TFLOP/s from the row's FLOPs and their own
    kernel median, so the two throughputs compare directly."""
    row = _run(flops=10**9)

    # 1e9 FLOPs: 1.0 ms median -> 1 TFLOP/s (warm heuristic); 0.5 ms -> 2 (tuned).
    assert row.oracle.warm_baseline_derived_tflops_per_s == pytest.approx(1.0)
    assert row.oracle.derived_tflops_per_s == pytest.approx(2.0)


def test_first_eligible_success_wins_and_counts_ignore_excluded_plans(tuned):
    # hipDNN returns candidates in rank order; the first eligible success wins.
    winner = _candidate(rank=1)
    winner.knob_settings = [SimpleNamespace(knob_id=4, value=8)]
    tuned.candidates = [
        _candidate(excluded=True, rank=0),
        winner,
        _candidate(succeeded=False, rank=2),
        _candidate(rank=3),
    ]
    o = _run().oracle

    assert (
        o.compiled_plans_total,
        o.compiled_plans_benchmarked,
        o.compiled_plans_failed,
    ) == (3, 2, 1)
    assert (o.rank, o.compiled_plan_index) == (1, 1)
    assert o.knob_settings == [{"knob_id": "4", "value": 8}]


def test_failed_sweep_is_an_oracle_error_and_skips_timing(tuned):
    """Executor.autotune raises when no candidate succeeded; the oracle keeps
    its message, times nothing, and leaves the row verdict alone."""
    tuned.autotune_error = ExecutionError("workspace exceeds limit")
    row = _run()

    assert row.oracle is None and row.oracle_delta is None
    assert row.oracle_error == "ExecutionError: workspace exceeds limit"
    assert [c for c, _ in CALLS] == ["zero_outputs", "tuned.autotune"]
    assert row.status == "success"


def test_failing_tuned_plan_suppresses_speedup_but_keeps_row_verdict(tuned):
    passed = CorrectnessResult(tolerance_match=True, rtol=1e-5, atol=1e-6)
    refs = {1: ReferenceOutput(data=np.zeros(2, np.float32), tensor_uid=1)}
    bm = _BM({1: np.ones(2, np.float32)})

    row = _run(correctness=passed, bm=bm, refs=refs)

    assert row.oracle.correctness.explicitly_failed
    assert row.oracle_delta is None
    assert row.verdict == "passed"


@pytest.mark.parametrize(
    "tolerance_match, has_delta",
    [(False, False), (None, True), (True, True)],
    ids=["failed", "unchecked", "passed"],
)
def test_failing_baseline_suppresses_speedup(tuned, tolerance_match, has_delta):
    """A speedup needs two valid operands: a heuristic plan that failed
    validation is not a baseline."""
    verdict = CorrectnessResult(tolerance_match=tolerance_match, rtol=1e-5, atol=1e-6)

    row = _run(correctness=verdict)

    assert row.oracle is not None
    assert (row.oracle_delta is not None) is has_delta


@pytest.mark.parametrize("mode", ["plan", "exhaustive"])
@pytest.mark.parametrize("supports", [True, False])
def test_exhaustive_flags_follow_mode_and_winner(tuned, mode, supports):
    winner = _candidate()
    winner.supports_exhaustive = supports
    tuned.candidates = [winner]

    o = _run(mode=mode).oracle

    assert (o.exhaustive_requested, o.exhaustive_supported) == (
        mode == "exhaustive",
        supports,
    )
    assert o.exhaustive_enabled == (mode == "exhaustive" and supports)


def test_host_stats_split_tuned_from_warm_baseline(tuned):
    o = _run().oracle

    assert o.host_stats.median_ms == pytest.approx(0.2)
    assert o.warm_baseline_host_stats.median_ms == pytest.approx(0.01)


@pytest.mark.parametrize(
    "mode, forced, cache_off", [("plan", None, "0"), ("exhaustive", "1", "1")]
)
def test_exhaustive_env_is_scoped_to_the_oracle_build(
    tuned, monkeypatch, mode, forced, cache_off
):
    monkeypatch.delenv("HIPDNN_FORCE_BENCHMARKING", raising=False)
    monkeypatch.setenv("HIPDNN_DISABLE_CACHE", "0")

    _run(mode=mode)

    # Exhaustive also disables hipDNN disk caches while the plans are built.
    assert tuned.env_at_prepare == {
        "HIPDNN_FORCE_BENCHMARKING": forced,
        "HIPDNN_DISABLE_CACHE": cache_off,
    }
    assert "HIPDNN_FORCE_BENCHMARKING" not in os.environ
    assert os.environ["HIPDNN_DISABLE_CACHE"] == "0"


def test_env_restored_and_error_recorded_when_build_fails(tuned, monkeypatch):
    monkeypatch.delenv("HIPDNN_FORCE_BENCHMARKING", raising=False)
    tuned.prepare_error = ExecutionError("boom")

    row = _run(mode="exhaustive")

    assert row.oracle_error == "ExecutionError: boom"
    assert "HIPDNN_FORCE_BENCHMARKING" not in os.environ

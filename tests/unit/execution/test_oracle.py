# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for execution.oracle (autotuned plan vs heuristic plan)."""

import os
from types import SimpleNamespace

import numpy as np
import pytest

from dnn_benchmarking.common.exceptions import ExecutionError
from dnn_benchmarking.config.benchmark_config import SuiteConfig, ValidationConfig
from dnn_benchmarking.execution import oracle as oracle_mod
from dnn_benchmarking.execution.timing import Measurement
from dnn_benchmarking.graph.tensor_info import TensorInfo
from dnn_benchmarking.reporting.suite_results import (
    CorrectnessResult,
    ProviderEngineResult,
)
from dnn_benchmarking.validation import ReferenceOutput

ENV = ("HIPDNN_FORCE_BENCHMARKING", "HIPDNN_DISABLE_CACHE")


def _m(kernel_ms):
    return Measurement(
        kernel_ms=list(kernel_ms),
        host_ms=[0.01] * len(kernel_ms),
        mode="staged",
        backend="hip",
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
        pass

    def get_output_tensor(self, uid):
        return None

    def get_output_data(self, uid):
        return self.outputs.get(uid)


class _TunedExecutor:
    """Stands in for the oracle's own Executor; records env at plan build."""

    candidates = [_candidate()]
    kernel_ms = [0.5, 0.5, 0.5, 9.0]
    prepare_error = None
    env_at_prepare = None

    def __init__(self, graph_json_str, policy):
        self.init_time_ms = 3.0

    def prepare(self, handle, engine_id=None, for_autotune=False):
        type(self).env_at_prepare = {k: os.environ.get(k) for k in ENV}
        if self.prepare_error is not None:
            raise self.prepare_error

    def autotune(self, handle, variant_pack, engine_id):
        return self.candidates

    def benchmark(self, handle, variant_pack):
        return _m(self.kernel_ms)

    def execute_once(self, handle, variant_pack):
        pass

    def plan_name(self, handle):
        return "tuned"


@pytest.fixture
def tuned(monkeypatch):
    cls = type("Tuned", (_TunedExecutor,), {})
    monkeypatch.setattr(oracle_mod, "Executor", cls)
    return cls


def _run(mode="plan", correctness=None, bm=None, refs=None, flops=None):
    row = ProviderEngineResult(
        provider="hipdnn", engine_id=5, status="success", correctness=correctness
    )
    row.analytical_flops = flops
    baseline = SimpleNamespace(benchmark=lambda h, vp: _m([1.0, 1.0, 1.0, 10.0]))
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
        handle=_Handle(),
        engine_id=5,
        graph_json_str="{}",
        graph_name="g",
        config=SuiteConfig(
            oracle_mode=mode, validation=ValidationConfig(provider="pytorch")
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


def test_tuned_and_warm_heuristic_report_median_tflops(tuned):
    """Both oracle operands get TFLOP/s from the row's FLOPs and their own
    kernel median, so the two throughputs compare directly."""
    row = _run(flops=10**9)

    # 1e9 FLOPs: 1.0 ms median -> 1 TFLOP/s (warm heuristic); 0.5 ms -> 2 (tuned).
    assert row.oracle.warm_baseline_derived_tflops_per_s == pytest.approx(1.0)
    assert row.oracle.derived_tflops_per_s == pytest.approx(2.0)


def test_candidate_counts_ignore_caller_excluded_plans(tuned):
    tuned.candidates = [
        _candidate(rank=0),
        _candidate(succeeded=False, rank=1),
        _candidate(excluded=True, rank=2),
    ]
    o = _run().oracle

    assert (
        o.compiled_plans_total,
        o.compiled_plans_benchmarked,
        o.compiled_plans_failed,
    ) == (2, 1, 1)
    assert o.rank == 0


@pytest.mark.parametrize(
    "candidates",
    [[], [_candidate(succeeded=False)], [_candidate(excluded=True)]],
)
def test_no_successful_candidate_is_an_oracle_error(tuned, candidates):
    tuned.candidates = candidates
    row = _run()

    assert row.oracle is None and row.oracle_delta is None
    assert "no successful candidate" in row.oracle_error
    assert row.status == "success"


def test_failing_tuned_plan_suppresses_speedup_but_keeps_row_verdict(tuned):
    passed = CorrectnessResult(tolerance_match=True, rtol=1e-5, atol=1e-6)
    refs = {1: ReferenceOutput(data=np.zeros(2, np.float32), tensor_uid=1)}
    bm = _BM({1: np.ones(2, np.float32)})

    row = _run(correctness=passed, bm=bm, refs=refs)

    assert row.oracle.correctness.explicitly_failed
    assert row.oracle_delta is None
    assert row.verdict == "passed"


@pytest.mark.parametrize("mode, forced", [("plan", None), ("exhaustive", "1")])
def test_exhaustive_env_is_scoped_to_the_oracle_build(tuned, monkeypatch, mode, forced):
    monkeypatch.delenv("HIPDNN_FORCE_BENCHMARKING", raising=False)
    monkeypatch.setenv("HIPDNN_DISABLE_CACHE", "0")

    _run(mode=mode)

    assert tuned.env_at_prepare["HIPDNN_FORCE_BENCHMARKING"] == forced
    assert "HIPDNN_FORCE_BENCHMARKING" not in os.environ
    assert os.environ["HIPDNN_DISABLE_CACHE"] == "0"


def test_env_restored_and_error_recorded_when_build_fails(tuned, monkeypatch):
    monkeypatch.delenv("HIPDNN_FORCE_BENCHMARKING", raising=False)
    tuned.prepare_error = ExecutionError("boom")

    row = _run(mode="exhaustive")

    assert row.oracle_error == "ExecutionError: boom"
    assert "HIPDNN_FORCE_BENCHMARKING" not in os.environ

# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for forced-engine selection in the hipDNN Executor.

A forced/preferred engine is a SOFT request in hipDNN: when the requested id is
not among the engines the backend ranks as applicable, the frontend silently
runs the top-ranked engine while the caller still believes the forced engine
ran -- fabricating comparison rows where several "different" forced engines are
all the same fallback.

The executor avoids this by hard-selecting a forced engine via
``Graph.create_execution_plan_ext`` (which errors instead of falling back) and
reading back the engine that actually backs the built plan via
``Graph.get_execution_plan_engine_id``.
"""

import sys
import types
from unittest.mock import patch

import pytest

import dnn_benchmarking.execution.executor as executor_module
import dnn_benchmarking.execution.timing as timing_module
from dnn_benchmarking.config.benchmark_config import TimingPolicy
from dnn_benchmarking.common.exceptions import ExecutionError, UnsupportedGraphError
from dnn_benchmarking.reporting.suite_results import engine_id_hex


class _StubResult:
    """hipDNN Error stub with a configurable bad/message state."""

    def __init__(self, bad: bool = False, message: str = ""):
        self._bad = bad
        self._message = message

    def is_bad(self) -> bool:
        return self._bad

    def get_message(self) -> str:
        return self._message


class _StubGraph:
    """Minimal hipDNN Graph stub exercising the executor's plan lifecycle.

    ``create_execution_plan_ext`` records the hard-selected engine and its knob
    settings, returning a bad Error when ``hard_fails``;
    ``create_execution_plans`` flags the heuristic path;
    ``get_execution_plan_engine_id`` reports the engine backing the built plan.
    With a ``clock``, every binding call advances it by 1 ms.
    """

    def __init__(
        self,
        ranked,
        selected=None,
        hard_fails=False,
        rank_error=None,
        plans_fail=False,
        support_fails=False,
        build_fails=False,
        engine_knobs=None,
        clock=None,
        workspace_size=0,
    ):
        self._ranked = ranked
        self._selected = selected
        self._hard_fails = hard_fails
        self._rank_error = rank_error
        self._plans_fail = plans_fail
        self._support_fails = support_fails
        self._build_fails = build_fails
        self._engine_knobs = engine_knobs or {}
        self._clock = clock
        self._workspace_size = workspace_size
        self.plans_created = False
        self.plans_built = False
        self.hard_engine_id = None
        self.knob_settings = None

    def tick(self):
        if self._clock is not None:
            self._clock.now_s += 0.001

    def from_json(self, _s):
        self.tick()
        return _StubResult()

    def validate(self):
        self.tick()
        return _StubResult()

    def build_operation_graph(self, _handle):
        self.tick()
        return _StubResult()

    def get_ranked_engine_ids(self):
        if self._rank_error is not None:
            raise RuntimeError(self._rank_error)
        return list(self._ranked)

    def create_execution_plans(self):
        self.tick()
        self.plans_created = True
        return _StubResult(bad=self._plans_fail, message="plan creation failed")

    def create_execution_plan_ext(self, engine_id, knob_settings):
        self.tick()
        if self._hard_fails:
            return _StubResult(bad=True, message="Failed to finalize engine descriptor")
        self.hard_engine_id = engine_id
        self.knob_settings = list(knob_settings)
        return _StubResult()

    def get_execution_plan_engine_id(self):
        self.tick()
        return self._selected

    def check_support(self):
        self.tick()
        return _StubResult(bad=self._support_fails, message="not supported")

    def build_plans(self):
        self.tick()
        self.plans_built = not self._build_fails
        return _StubResult(bad=self._build_fails, message="build failed")

    def get_workspace_size(self):
        self.tick()
        return self._workspace_size

    def get_knobs_for_engine(self, engine_id):
        return [
            types.SimpleNamespace(knob_id=k)
            for k in self._engine_knobs.get(engine_id, [])
        ]


class _FakeClock:
    """perf_counter stand-in that advances only when a stub binding call runs."""

    def __init__(self):
        self.now_s = 0.0

    def perf_counter(self):
        return self.now_s


def _executor(warmup_iters: int = 0):
    # "{}" -> empty graph dict: no data-type attrs / nodes to configure.
    return executor_module.Executor("{}", TimingPolicy(warmup_iters=warmup_iters))


def _fake_module(graph):
    fake = types.ModuleType("hipdnn_frontend")
    fake.Graph = lambda: graph
    fake.KnobSetting = lambda knob_id, value: types.SimpleNamespace(
        knob_id=knob_id, value=value
    )

    def device_buffer(_size):
        graph.tick()  # workspace allocation costs time too
        return types.SimpleNamespace(ptr=lambda: 0xDEADBEEF, zeros=lambda: None)

    fake.DeviceBuffer = device_buffer
    return fake


def _prepared_executor(graph, **prepare_kwargs):
    executor = _executor()
    with patch.dict(sys.modules, {"hipdnn_frontend": _fake_module(graph)}):
        executor.prepare(handle=object(), engine_id=999, **prepare_kwargs)
    return executor


def test_prepare_hard_select_uses_the_forced_engine():
    """A forced, applicable engine is hard-selected (not soft-preferred)."""
    executor = _executor()
    graph = _StubGraph(ranked=[999], selected=999)
    with patch.dict(sys.modules, {"hipdnn_frontend": _fake_module(graph)}):
        executor.prepare(handle=object(), engine_id=999)
    assert graph.hard_engine_id == 999  # hard selection was used
    assert graph.plans_created is False  # heuristic path not taken


def test_prepare_hard_select_not_applicable_is_skip():
    """A hard-select failure (engine not applicable) becomes an
    UnsupportedGraphError, i.e. a clean skip rather than a silent fallback."""
    executor = _executor()
    graph = _StubGraph(ranked=[111], hard_fails=True)
    with patch.dict(sys.modules, {"hipdnn_frontend": _fake_module(graph)}):
        with pytest.raises(UnsupportedGraphError):
            executor.prepare(handle=object(), engine_id=999)


def test_prepare_discovery_uses_heuristic_plan_creation():
    """With no forced engine, prepare uses the heuristic create_execution_plans
    path."""
    executor = _executor()
    graph = _StubGraph(ranked=[111, 222], selected=111)
    with patch.dict(sys.modules, {"hipdnn_frontend": _fake_module(graph)}):
        executor.prepare(handle=object(), engine_id=None)
    assert graph.plans_created is True  # heuristic path taken
    assert graph.hard_engine_id is None  # hard selection not used


def test_graph_data_types_reach_the_hipdnn_setters():
    """Stated graph types are set on the hipDNN graph; "unset" (no DataType
    member) is left for hipDNN inference."""
    graph = _StubGraph(ranked=[111], selected=111)
    calls = []
    for key in ("io_data_type", "intermediate_data_type", "compute_data_type"):
        setattr(graph, f"set_{key}", lambda dt, key=key: calls.append((key, dt)))
    fake = _fake_module(graph)
    fake.DataType = types.SimpleNamespace(HALF="HALF", FLOAT="FLOAT", NOT_SET="NOT_SET")
    graph_json = (
        '{"io_data_type": "half", "intermediate_data_type": "unset",'
        ' "compute_data_type": "float"}'
    )
    executor = executor_module.Executor(graph_json, TimingPolicy())
    with patch.dict(sys.modules, {"hipdnn_frontend": fake}):
        executor.prepare(handle=object(), engine_id=None)
    assert calls == [("io_data_type", "HALF"), ("compute_data_type", "FLOAT")]


def test_discover_engines_ranking_runtime_error_becomes_unsupported():
    """A backend RuntimeError while ranking surfaces as an unsupported-graph
    skip, not a hard error."""
    executor = _executor()
    graph = _StubGraph(ranked=[], rank_error="no engine has an applicable solution")
    with patch.dict(sys.modules, {"hipdnn_frontend": _fake_module(graph)}):
        with pytest.raises(UnsupportedGraphError) as exc:
            executor.discover_engines(handle=object())
    assert "applicable solution" in str(exc.value)


def test_prepare_forced_engine_mismatch_is_skip():
    """Driven through the public prepare() flow: hard-select succeeds but the
    backend reports a different engine backing the plan -> unsupported skip."""
    executor = _executor()
    graph = _StubGraph(ranked=[999], selected=111)  # hard select ok, read-back differs
    with patch.dict(sys.modules, {"hipdnn_frontend": _fake_module(graph)}):
        with pytest.raises(UnsupportedGraphError) as exc:
            executor.prepare(handle=object(), engine_id=999)
    assert graph.hard_engine_id == 999  # hard select was attempted
    assert engine_id_hex(999) in str(exc.value)
    assert engine_id_hex(111) in str(exc.value)


def test_prepare_create_execution_plans_failure_is_execution_error():
    """A bad create_execution_plans() result on the discovery path is a hard
    ExecutionError, not an unsupported-graph skip."""
    executor = _executor()
    graph = _StubGraph(ranked=[1], plans_fail=True)
    with patch.dict(sys.modules, {"hipdnn_frontend": _fake_module(graph)}):
        with pytest.raises(ExecutionError) as exc:
            executor.prepare(handle=object(), engine_id=None)
    assert "plan creation failed" in str(exc.value)


def test_prepare_check_support_failure_is_unsupported():
    """A bad check_support() result is classified as an unsupported-graph skip."""
    executor = _executor()
    graph = _StubGraph(ranked=[1], selected=1, support_fails=True)
    with patch.dict(sys.modules, {"hipdnn_frontend": _fake_module(graph)}):
        with pytest.raises(UnsupportedGraphError) as exc:
            executor.prepare(handle=object(), engine_id=None)
    assert "not supported" in str(exc.value)


def test_prepare_build_plans_failure_is_execution_error():
    """A bad build_plans() result is a hard ExecutionError."""
    executor = _executor()
    graph = _StubGraph(ranked=[1], selected=1, build_fails=True)
    with patch.dict(sys.modules, {"hipdnn_frontend": _fake_module(graph)}):
        with pytest.raises(ExecutionError) as exc:
            executor.prepare(handle=object(), engine_id=None)
    assert "build failed" in str(exc.value)


def test_prepare_applies_knobs_to_the_forced_engine_plan():
    """Knobs become hipDNN KnobSettings on the forced engine's plan; dropping
    them would silently build an untuned plan for the oracle."""
    graph = _StubGraph(ranked=[999], selected=999)
    _prepared_executor(graph, knobs={"global.benchmarking": 1})
    assert graph.hard_engine_id == 999
    assert [(s.knob_id, s.value) for s in graph.knob_settings] == [
        ("global.benchmarking", 1)
    ]


def test_prepare_without_knobs_builds_the_ootb_plan():
    graph = _StubGraph(ranked=[999], selected=999)
    _prepared_executor(graph)
    assert graph.knob_settings == []


def test_prepare_knobs_without_engine_is_rejected_before_any_graph_work():
    """Knobs only apply to a hard-selected plan; the heuristic path would
    ignore them, so reject the call before touching hipDNN."""
    fake = types.ModuleType("hipdnn_frontend")

    def _no_graph():
        raise AssertionError("graph built before knob validation")

    fake.Graph = _no_graph
    with patch.dict(sys.modules, {"hipdnn_frontend": fake}):
        with pytest.raises(ValueError):
            _executor().prepare(
                handle=object(), engine_id=None, knobs={"global.benchmarking": 1}
            )


def test_build_time_covers_only_plan_create_support_and_build():
    """build_time_ms must exclude graph setup and workspace allocation so OOTB
    and tuned builds of one engine are comparable."""
    clock = _FakeClock()
    graph = _StubGraph(ranked=[999], selected=999, clock=clock, workspace_size=64)
    with patch.object(timing_module, "time", clock):
        executor = _prepared_executor(graph)
    # create_execution_plan_ext + check_support + build_plans, 1 ms each.
    assert executor.build_time_ms == pytest.approx(3.0)
    # Control: from_json, validate, build_operation_graph, engine read-back,
    # workspace query, and workspace allocation all ran (and cost time).
    assert clock.now_s == pytest.approx(0.009)
    assert executor.workspace_size == 64


def test_prime_builds_the_engine_plan_and_leaves_the_executor_unprepared():
    """prime() pays the engine's first-build cost on a throwaway OOTB plan; the
    executor must not be usable for timing afterwards."""
    graph = _StubGraph(ranked=[999], selected=999)
    executor = _executor()
    with patch.dict(sys.modules, {"hipdnn_frontend": _fake_module(graph)}):
        executor.prime(handle=object(), engine_id=999)
    assert graph.hard_engine_id == 999
    assert graph.knob_settings == []
    assert graph.plans_built is True
    for run in (executor.execute_once, executor.benchmark):
        with pytest.raises(ExecutionError, match="Graph not prepared"):
            run(object(), {})


def test_engine_knob_ids_lists_the_requested_engines_knobs():
    graph = _StubGraph(
        ranked=[999, 111],
        selected=999,
        engine_knobs={999: ["global.benchmarking", "tile"], 111: ["split_k"]},
    )
    executor = _prepared_executor(graph)
    assert executor.engine_knob_ids(999) == ["global.benchmarking", "tile"]
    assert executor.engine_knob_ids(111) == ["split_k"]


def test_engine_knob_ids_without_prepare_raises():
    with pytest.raises(ExecutionError):
        _executor().engine_knob_ids(999)

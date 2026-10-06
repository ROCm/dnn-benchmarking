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
from dnn_benchmarking.config.benchmark_config import BenchmarkConfig
from dnn_benchmarking.common.exceptions import ExecutionError, UnsupportedGraphError


class _StubResult:
    """hipDNN Error stub with a configurable bad/message state."""

    def __init__(self, bad: bool = False, message: str = ""):
        self._bad = bad
        self._message = message

    def is_bad(self) -> bool:
        return self._bad

    def get_message(self) -> str:
        return self._message


class _FakeClock:
    """Millisecond clock that advances only when a stub binding call runs."""

    def __init__(self):
        self.now_ms = 0.0

    def timer(self):
        clock = self

        class _ClockTimer:
            def __enter__(self):
                self._start = clock.now_ms
                return self

            def __exit__(self, *_exc):
                self.elapsed_ms = clock.now_ms - self._start

        return _ClockTimer()


class _StubGraph:
    """Minimal hipDNN Graph stub exercising the executor's plan lifecycle.

    ``create_execution_plan_ext`` records the hard-selected engine and its knob
    settings, returning a bad Error when ``hard_fails``;
    ``create_execution_plans`` flags the heuristic path;
    ``get_execution_plan_engine_id`` reports the engine backing the built plan.
    With a ``clock``, every binding call costs 1 ms.
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
        self.plans_created = False
        self.hard_engine_id = None
        self.knob_settings = None
        self.plan_name_handle = "unset"

    def _tick(self):
        if self._clock is not None:
            self._clock.now_ms += 1.0

    def from_json(self, _s):
        self._tick()
        return _StubResult()

    def validate(self):
        self._tick()
        return _StubResult()

    def build_operation_graph(self, _handle):
        self._tick()
        return _StubResult()

    def get_ranked_engine_ids(self):
        if self._rank_error is not None:
            raise RuntimeError(self._rank_error)
        return list(self._ranked)

    def create_execution_plans(self):
        self._tick()
        self.plans_created = True
        return _StubResult(bad=self._plans_fail, message="plan creation failed")

    def create_execution_plan_ext(self, engine_id, knob_settings):
        self._tick()
        if self._hard_fails:
            return _StubResult(bad=True, message="Failed to finalize engine descriptor")
        self.hard_engine_id = engine_id
        self.knob_settings = list(knob_settings)
        return _StubResult()

    def get_execution_plan_engine_id(self):
        self._tick()
        return self._selected

    def check_support(self):
        self._tick()
        return _StubResult(bad=self._support_fails, message="not supported")

    def build_plans(self):
        self._tick()
        return _StubResult(bad=self._build_fails, message="build failed")

    def get_workspace_size(self):
        self._tick()
        return 0

    def get_knobs_for_engine(self, engine_id):
        return [
            types.SimpleNamespace(knob_id=k)
            for k in self._engine_knobs.get(engine_id, [])
        ]

    def get_plan_name(self, handle):
        # hipDNN needs the handle to name plugin-supplied engines; without it
        # it consults only the built-in registry and reports a hex engine ID.
        self.plan_name_handle = handle
        return "winning_plan" if handle is not None else "0xdeadbeef"


def _executor():
    config = BenchmarkConfig(graph_path="dummy.json", warmup_iters=0, benchmark_iters=1)
    # "{}" -> empty graph dict: no data-type attrs / nodes to configure.
    return executor_module.Executor("{}", config)


def _fake_module(graph):
    fake = types.ModuleType("hipdnn_frontend")
    fake.Graph = lambda: graph
    fake.KnobSetting = lambda knob_id, value: types.SimpleNamespace(
        knob_id=knob_id, value=value
    )
    return fake


def _prepared_executor(graph, **prepare_kwargs):
    executor = _executor()
    with patch.dict(sys.modules, {"hipdnn_frontend": _fake_module(graph)}):
        executor.prepare(handle=object(), engine_id=999, **prepare_kwargs)
    return executor


def test_prepare_hard_select_records_actual_engine():
    """A forced, applicable engine is hard-selected (not soft-preferred) and the
    engine the backend reports as backing the plan is recorded."""
    executor = _executor()
    graph = _StubGraph(ranked=[999], selected=999)
    with patch.dict(sys.modules, {"hipdnn_frontend": _fake_module(graph)}):
        executor.prepare(handle=object(), engine_id=999)
    assert graph.hard_engine_id == 999  # hard selection was used
    assert graph.knob_settings == []  # OOTB plan: no knob overrides
    assert graph.plans_created is False  # heuristic path not taken
    assert executor.selected_engine_id == 999


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
    path and records whichever engine the backend selected."""
    executor = _executor()
    graph = _StubGraph(ranked=[111, 222], selected=111)
    with patch.dict(sys.modules, {"hipdnn_frontend": _fake_module(graph)}):
        executor.prepare(handle=object(), engine_id=None)
    assert graph.plans_created is True  # heuristic path taken
    assert graph.hard_engine_id is None  # hard selection not used
    assert executor.selected_engine_id == 111


def test_record_selected_engine_mismatch_raises():
    """If a forced engine differs from the engine actually selected, it is
    treated as an unsupported-graph skip rather than mislabeled timings."""
    executor = _executor()
    executor._graph = _StubGraph(ranked=[111], selected=111)
    with pytest.raises(UnsupportedGraphError) as exc:
        executor._record_selected_engine(999)
    assert "999" in str(exc.value) and "111" in str(exc.value)


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
    assert "999" in str(exc.value) and "111" in str(exc.value)


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
    """build_time_ms must exclude graph setup and workspace sizing so OOTB and
    tuned builds of one engine are comparable; init_time_ms covers it all."""
    clock = _FakeClock()
    graph = _StubGraph(ranked=[999], selected=999, clock=clock)
    with patch.object(executor_module, "Timer", clock.timer):
        executor = _prepared_executor(graph)
    # create_execution_plan_ext + check_support + build_plans, 1 ms each.
    assert executor.build_time_ms == 3.0
    # Plus from_json, validate, build_operation_graph, engine read-back, workspace.
    assert executor.init_time_ms == 8.0


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


def test_plan_name_passes_the_handle_to_the_binding():
    """Newer bindings need the handle to name plugin-supplied engines, which is
    the engine class this tool benchmarks; without it they report a hex ID."""
    handle = object()
    graph = _StubGraph(ranked=[999], selected=999)
    executor = _prepared_executor(graph)

    assert executor.plan_name(handle) == "winning_plan"
    assert graph.plan_name_handle is handle


def test_plan_name_without_a_handle_gets_the_hex_fallback():
    """Control: the handle is what makes the difference, so a caller that drops
    it silently degrades to a hex engine ID."""
    graph = _StubGraph(ranked=[999], selected=999)
    executor = _prepared_executor(graph)
    assert executor.plan_name(None) == "0xdeadbeef"


def test_plan_name_without_prepare_is_none():
    assert _executor().plan_name(object()) is None

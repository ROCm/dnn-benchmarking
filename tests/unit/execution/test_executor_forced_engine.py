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


class _StubGraph:
    """Minimal hipDNN Graph stub exercising the executor's plan lifecycle.

    ``create_execution_plan_ext`` records the hard-selected engine, returning a
    bad Error when ``hard_fails``; ``create_execution_plans`` flags the
    heuristic path; ``get_execution_plan_engine_id`` reports the engine backing
    the built plan.
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
        autotune_results=(),
        autotune_error=None,
        plan_engines=None,
        plan_workspaces=None,
    ):
        self._autotune_results = autotune_results
        self._autotune_error = autotune_error
        self._ranked = ranked
        self._selected = selected
        self._hard_fails = hard_fails
        self._rank_error = rank_error
        self._plans_fail = plans_fail
        self._support_fails = support_fails
        self._build_fails = build_fails
        self.plans_created = False
        self.hard_engine_id = None
        self.build_policy = "unset"
        self.autotune_workspace_queried = False
        self.autotune_kwargs = None
        self.plan_name_handle = "unset"
        # One entry per candidate plan, giving the engine that backs it.
        # Defaults to one plan per ranked engine.
        self._plan_engines = (
            [int(e) for e in plan_engines]
            if plan_engines is not None
            else [int(e) for e in ranked]
        )
        self._plan_workspaces = (
            list(plan_workspaces)
            if plan_workspaces is not None
            else [0] * len(self._plan_engines)
        )
        self.barred_engines = set()
        self.compiled_plan_engines = []

    def from_json(self, _s):
        return _StubResult()

    def validate(self):
        return _StubResult()

    def build_operation_graph(self, _handle):
        return _StubResult()

    def get_ranked_engine_ids(self):
        if self._rank_error is not None:
            raise RuntimeError(self._rank_error)
        return list(self._ranked)

    def create_execution_plans(self):
        self.plans_created = True
        return _StubResult(bad=self._plans_fail, message="plan creation failed")

    def create_execution_plan_ext(self, engine_id):
        if self._hard_fails:
            return _StubResult(bad=True, message="Failed to finalize engine descriptor")
        self.hard_engine_id = engine_id
        return _StubResult()

    def get_execution_plan_engine_id(self):
        return self._selected

    def check_support(self):
        return _StubResult(bad=self._support_fails, message="not supported")

    def deselect_engines(self, engine_ids):
        self.barred_engines.update(int(e) for e in engine_ids)
        return self

    def build_plans(self, policy=None):
        self.build_policy = policy
        if self._build_fails:
            return _StubResult(bad=True, message="build failed")
        # BuildPlanPolicy.ALL compiles every plan whose engine is not barred;
        # the heuristic policy compiles only the active plan.
        if policy == "ALL":
            self.compiled_plan_engines = [
                e for e in self._plan_engines if e not in self.barred_engines
            ]
        else:
            self.compiled_plan_engines = self._plan_engines[:1]
        return _StubResult()

    def get_workspace_size(self):
        return 0

    def get_autotune_workspace_size(self):
        self.autotune_workspace_queried = True
        # Mirrors hipDNN: barred plans are skipped from the maximum.
        sizes = [
            ws
            for engine, ws in zip(self._plan_engines, self._plan_workspaces)
            if engine not in self.barred_engines
        ]
        return max(sizes) if sizes else 0

    def get_plan_name(self, handle):
        # hipDNN needs the handle to name plugin-supplied engines; without it
        # it consults only the built-in registry and reports a hex engine ID.
        self.plan_name_handle = handle
        return "winning_plan" if handle is not None else "0xdeadbeef"

    def autotune(self, handle, variant_pack, workspace_ptr, **kwargs):
        if self._autotune_error is not None:
            raise RuntimeError(self._autotune_error)
        self.autotune_kwargs = kwargs
        return list(self._autotune_results)


class _StubAutotuneConfig:
    """Stands in for hipdnn_frontend.AutotuneConfig."""

    def __init__(self):
        self.engine_id_filter = []


class _StubCandidate:
    """Stands in for one hipdnn_frontend.AutotuneResult entry."""

    def __init__(
        self,
        succeeded=True,
        rank=0,
        error_message="",
        engine_id=999,
        excluded_by_caller=False,
    ):
        self.succeeded = succeeded
        self.rank = rank
        self.error_message = error_message
        self.engine_id = engine_id
        # hipDNN sets this on candidates its own filters rejected without
        # benchmarking (engine_id_filter, deselect_engines, workspace ceiling).
        self.excluded_by_caller = excluded_by_caller


class _StubDeviceBuffer:
    """Stands in for hipdnn_frontend.DeviceBuffer."""

    def __init__(self, size):
        self.size = size

    def ptr(self):
        return 0xDEADBEEF

    def zeros(self):
        return None


def _executor():
    config = BenchmarkConfig(graph_path="dummy.json", warmup_iters=0, benchmark_iters=1)
    # "{}" -> empty graph dict: no data-type attrs / nodes to configure.
    return executor_module.Executor("{}", config)


def _fake_module(graph):
    fake = types.ModuleType("hipdnn_frontend")
    fake.Graph = lambda: graph
    fake.BuildPlanPolicy = types.SimpleNamespace(ALL="ALL")
    fake.AutotuneConfig = _StubAutotuneConfig
    fake.DeviceBuffer = _StubDeviceBuffer
    return fake


def test_prepare_hard_select_records_actual_engine():
    """A forced, applicable engine is hard-selected (not soft-preferred) and the
    engine the backend reports as backing the plan is recorded."""
    executor = _executor()
    graph = _StubGraph(ranked=[999], selected=999)
    with patch.dict(sys.modules, {"hipdnn_frontend": _fake_module(graph)}):
        executor.prepare(handle=object(), engine_id=999)
    assert graph.hard_engine_id == 999  # hard selection was used
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


def _prepared_autotune_executor(graph):
    executor = _executor()
    with patch.dict(sys.modules, {"hipdnn_frontend": _fake_module(graph)}):
        executor.prepare(handle=object(), engine_id=999, for_autotune=True)
    return executor


def test_prepare_for_autotune_builds_all_plans():
    """The autotune build compiles every candidate and never hard-selects."""
    graph = _StubGraph(ranked=[999], selected=999)
    executor = _prepared_autotune_executor(graph)
    assert graph.plans_created is True
    assert graph.hard_engine_id is None
    assert graph.build_policy == "ALL"
    assert graph.autotune_workspace_queried is True
    # No plan is pinned yet, so no engine is recorded by prepare().
    assert executor.selected_engine_id is None


def test_prepare_for_autotune_skips_forced_engine_mismatch_check():
    """A requested engine that differs from the reported one must not raise on
    the autotune path: no plan is pinned until autotune() picks a winner."""
    graph = _StubGraph(ranked=[111], selected=111)
    executor = _prepared_autotune_executor(graph)
    assert executor.selected_engine_id is None


def test_autotune_filters_to_engine_and_omits_workspace_size():
    graph = _StubGraph(
        ranked=[999],
        selected=999,
        # hipDNN's rankAndSelectWinner returns succeeded candidates first, in
        # ascending rank order; the stub reproduces that contract.
        autotune_results=[_StubCandidate(rank=0), _StubCandidate(rank=1)],
    )
    executor = _prepared_autotune_executor(graph)
    with patch.dict(sys.modules, {"hipdnn_frontend": _fake_module(graph)}):
        winners = executor.autotune(object(), {}, 999)

    assert graph.autotune_kwargs is not None
    assert "workspace_size" not in graph.autotune_kwargs
    assert graph.autotune_kwargs["config"].engine_id_filter == [999]
    # Rank order is preserved, so callers can take winners[0].
    assert [w.rank for w in winners] == [0, 1]
    assert executor.selected_engine_id == 999
    assert executor.plan_name(object()) == "winning_plan"


def test_autotune_drops_failed_candidates():
    graph = _StubGraph(
        ranked=[999],
        selected=999,
        autotune_results=[
            _StubCandidate(succeeded=False, rank=-1, error_message="bad plan"),
            _StubCandidate(rank=0),
        ],
    )
    executor = _prepared_autotune_executor(graph)
    with patch.dict(sys.modules, {"hipdnn_frontend": _fake_module(graph)}):
        winners = executor.autotune(object(), {}, 999)
    assert len(winners) == 1
    assert winners[0].rank == 0


def test_autotune_all_candidates_failed_raises():
    graph = _StubGraph(
        ranked=[999],
        selected=999,
        autotune_results=[
            _StubCandidate(succeeded=False, rank=-1, error_message="bad plan")
        ],
    )
    executor = _prepared_autotune_executor(graph)
    with patch.dict(sys.modules, {"hipdnn_frontend": _fake_module(graph)}):
        with pytest.raises(ExecutionError) as exc:
            executor.autotune(object(), {}, 999)
    assert "bad plan" in str(exc.value)


def test_autotune_runtime_error_becomes_execution_error():
    graph = _StubGraph(ranked=[999], selected=999, autotune_error="sweep exploded")
    executor = _prepared_autotune_executor(graph)
    with patch.dict(sys.modules, {"hipdnn_frontend": _fake_module(graph)}):
        with pytest.raises(ExecutionError) as exc:
            executor.autotune(object(), {}, 999)
    assert "sweep exploded" in str(exc.value)


def test_autotune_without_prepare_raises():
    with pytest.raises(ExecutionError) as exc:
        _executor().autotune(object(), {}, 1)
    assert "not prepared" in str(exc.value)


def test_prepare_for_autotune_bars_other_engines():
    """Only the target engine's plans are compiled, and the workspace is sized
    for those plans alone.

    build_plans(ALL) skips a barred plan before finalizing it, so barring the
    other engines removes their compiles; get_autotune_workspace_size() also
    skips barred plans, which keeps the oracle workspace off the peak while the
    OOTB workspace is still allocated.
    """
    graph = _StubGraph(
        ranked=[999, 111, 222],
        selected=999,
        plan_engines=[999, 999, 111, 222],
        plan_workspaces=[16, 32, 4096, 8192],
    )
    _prepared_autotune_executor(graph)

    assert graph.barred_engines == {111, 222}
    # Same count as engine 999's own plans, i.e. two fewer compiles here.
    assert graph.compiled_plan_engines == [999, 999]


def test_prepare_for_autotune_without_deselect_compiles_every_engine():
    """Control for the previous test: without barring, every engine's plans are
    compiled and the workspace is sized for the largest of them."""
    graph = _StubGraph(
        ranked=[999, 111, 222],
        selected=999,
        plan_engines=[999, 999, 111, 222],
        plan_workspaces=[16, 32, 4096, 8192],
    )
    executor = _executor()
    with patch.dict(sys.modules, {"hipdnn_frontend": _fake_module(graph)}):
        # engine_id=None is the only way to reach the ALL build without the
        # barring step, which is exactly the "before" case.
        executor.prepare(handle=object(), engine_id=None, for_autotune=True)

    assert graph.barred_engines == set()
    assert graph.compiled_plan_engines == [999, 999, 111, 222]
    assert executor.workspace_size == 8192


def test_prepare_for_autotune_workspace_covers_target_engine_only():
    """The allocated workspace is the target engine's maximum, not the graph's."""
    graph = _StubGraph(
        ranked=[999, 111],
        selected=999,
        plan_engines=[999, 999, 111],
        plan_workspaces=[16, 32, 8192],
    )
    executor = _prepared_autotune_executor(graph)
    assert executor.workspace_size == 32


def test_autotune_passes_run_warmup_iterations_to_the_sweep():
    """The sweep's warmup comes from the run's --warmup, not the binding
    default of 1, so first-execute kernel sampling stays out of the window
    that ranks the candidates."""
    config = BenchmarkConfig(graph_path="dummy.json", warmup_iters=7, benchmark_iters=1)
    executor = executor_module.Executor("{}", config)
    graph = _StubGraph(
        ranked=[999], selected=999, autotune_results=[_StubCandidate(rank=0)]
    )
    with patch.dict(sys.modules, {"hipdnn_frontend": _fake_module(graph)}):
        executor.prepare(handle=object(), engine_id=999, for_autotune=True)
        executor.autotune(object(), {}, 999)

    assert graph.autotune_kwargs["config"].warmup_iterations == 7


def test_autotune_error_reports_a_benchmarked_failure_not_a_filter_rejection():
    """Candidates the caller's own filters rejected are not benchmarked and
    carry the filter's message; the reported error must be the real failure."""
    graph = _StubGraph(
        ranked=[999],
        selected=999,
        autotune_results=[
            _StubCandidate(
                succeeded=False,
                rank=-1,
                error_message="Plan excluded by engineIdFilter.",
                engine_id=111,
                excluded_by_caller=True,
            ),
            _StubCandidate(
                succeeded=False,
                rank=-1,
                error_message="workspace exceeds limit",
                engine_id=999,
            ),
        ],
    )
    executor = _prepared_autotune_executor(graph)
    with patch.dict(sys.modules, {"hipdnn_frontend": _fake_module(graph)}):
        with pytest.raises(ExecutionError) as exc:
            executor.autotune(object(), {}, 999)
    assert "workspace exceeds limit" in str(exc.value)
    assert "engineIdFilter" not in str(exc.value)


def test_autotune_error_when_every_candidate_was_filtered_out():
    """With nothing benchmarked there is no real failure to report, so fall
    back to a generic message rather than parroting the filter's."""
    graph = _StubGraph(
        ranked=[999],
        selected=999,
        autotune_results=[
            _StubCandidate(
                succeeded=False,
                rank=-1,
                error_message="Plan excluded by engineIdFilter.",
                engine_id=111,
                excluded_by_caller=True,
            )
        ],
    )
    executor = _prepared_autotune_executor(graph)
    with patch.dict(sys.modules, {"hipdnn_frontend": _fake_module(graph)}):
        with pytest.raises(ExecutionError) as exc:
            executor.autotune(object(), {}, 999)
    assert "no autotune candidate benchmarked successfully" in str(exc.value)
    assert "engineIdFilter" not in str(exc.value)


def test_autotune_winner_from_another_engine_raises():
    """engine_id_filter makes this impossible; if it ever happens the oracle
    timing would carry the wrong engine label, so fail loudly."""
    graph = _StubGraph(
        ranked=[999],
        selected=999,
        autotune_results=[_StubCandidate(rank=0, engine_id=111)],
    )
    executor = _prepared_autotune_executor(graph)
    with patch.dict(sys.modules, {"hipdnn_frontend": _fake_module(graph)}):
        with pytest.raises(ExecutionError) as exc:
            executor.autotune(object(), {}, 999)
    assert "111" in str(exc.value) and "999" in str(exc.value)


def test_plan_name_passes_the_handle_to_the_binding():
    """Newer bindings need the handle to name plugin-supplied engines, which is
    the engine class this tool benchmarks; without it they report a hex ID."""
    handle = object()
    graph = _StubGraph(ranked=[999], selected=999)
    executor = _prepared_autotune_executor(graph)

    assert executor.plan_name(handle) == "winning_plan"
    assert graph.plan_name_handle is handle


def test_plan_name_without_a_handle_gets_the_hex_fallback():
    """Control: the handle is what makes the difference, so a caller that drops
    it silently degrades to a hex engine ID."""
    graph = _StubGraph(ranked=[999], selected=999)
    executor = _prepared_autotune_executor(graph)
    assert executor.plan_name(None) == "0xdeadbeef"


def test_plan_name_without_prepare_is_none():
    assert _executor().plan_name(object()) is None

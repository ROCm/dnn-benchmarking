# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for the hidden --internal-profiling-run sub-mode (the profiled child)."""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from dnn_benchmarking.cli import internal_profiling
from dnn_benchmarking.cli.parser import create_parser
from dnn_benchmarking.metrics.profiling_orchestrator import build_inner_argv


def _child_args(plugin_path=None):
    """Parse exactly what the orchestrator emits, as the child would."""
    argv = build_inner_argv(Path("/g/x.json"), 42, 11, 3, 7, plugin_path)
    return create_parser().parse_args(argv[3:])  # drop python -m dnn_benchmarking


@pytest.fixture
def stack(monkeypatch):
    """Hermetic hipdnn / loader / executor / buffers; returns the mocks."""
    mocks = {"hipdnn": MagicMock(), "executor": MagicMock(), "inputs": MagicMock()}
    calls = mocks["calls"] = []
    mocks["executor"].enqueue.side_effect = lambda *a: calls.append("enqueue")
    monkeypatch.setattr(
        internal_profiling, "device_sync", lambda backend: calls.append(backend)
    )
    mocks["hipdnn"].PluginLoadingMode.ABSOLUTE = "absolute"
    monkeypatch.setitem(sys.modules, "hipdnn_frontend", mocks["hipdnn"])
    monkeypatch.setattr(internal_profiling, "initialize_pip_rocm_runtime", lambda: None)

    loader = mocks["loader"] = MagicMock()
    loader.return_value.load_json.return_value = {"name": "g", "nodes": []}
    monkeypatch.setattr(internal_profiling, "GraphLoader", loader)

    def make_executor(graph_json_str, policy):
        mocks["policy"] = policy
        return mocks["executor"]

    monkeypatch.setattr(internal_profiling, "Executor", make_executor)
    bm = MagicMock()
    bm.allocate_all.side_effect = lambda: calls.append("allocate_all")
    bm.load_input_data.side_effect = lambda data: calls.append(("load", data))
    bm.create_variant_pack.side_effect = lambda: calls.append("variant_pack")
    buffer_manager = MagicMock()
    buffer_manager.return_value.__enter__.return_value = bm
    monkeypatch.setattr(internal_profiling, "BufferManager", buffer_manager)
    monkeypatch.setattr(internal_profiling, "generate_input_data", mocks["inputs"])
    return mocks


def test_orchestrator_argv_drives_one_engine_warm_fixed_count(stack):
    assert internal_profiling.run_internal_profiling(_child_args()) == 0

    stack["executor"].prepare.assert_called_once()
    assert stack["executor"].prepare.call_args.kwargs["engine_id"] == 42
    # Buffers allocated and loaded with the seeded inputs before the first
    # enqueue; then warmup + iters plain graph executes and one drain: no
    # timing loop (stall-gate deadlock) and no execute_once (per-iteration
    # workspace memset).
    setup = ["allocate_all", ("load", stack["inputs"].return_value), "variant_pack"]
    assert stack["calls"] == setup + ["enqueue"] * (3 + 7) + ["hip"]
    graph_json = {"name": "g", "nodes": []}
    stack["loader"].return_value.validate.assert_called_once_with(graph_json)
    stack["executor"].execute_once.assert_not_called()
    stack["executor"].benchmark.assert_not_called()
    policy = stack["policy"]
    assert (policy.min_time_ms, policy.cache_mode) == (0.0, "warm")
    # Same inputs as the timed pass: seed forwarded, and graph_json passed
    # so paged-SDPA page tables are valid.
    _, seed, inputs_graph_json = stack["inputs"].call_args.args
    assert seed == 11 and inputs_graph_json == graph_json


def test_plugin_path_replaces_default_plugin_loading(stack):
    plugin = Path("/p/x.so")
    assert internal_profiling.run_internal_profiling(_child_args(plugin)) == 0
    stack["hipdnn"].set_engine_plugin_paths.assert_called_once_with(
        [str(plugin)], "absolute"
    )


def test_execution_failure_returns_one_with_reason(stack, capsys):
    stack["executor"].prepare.side_effect = RuntimeError("kernel exploded")
    assert internal_profiling.run_internal_profiling(_child_args()) == 1
    err = capsys.readouterr().err
    assert "kernel exploded" in err and "engine 42" in err


def test_runtime_init_failure_returns_one(stack, monkeypatch, capsys):
    def fail():
        raise RuntimeError("ROCm SDK preload failed")

    monkeypatch.setattr(internal_profiling, "initialize_pip_rocm_runtime", fail)
    assert internal_profiling.run_internal_profiling(_child_args()) == 1
    assert "ROCm SDK preload failed" in capsys.readouterr().err


def test_multiple_plugin_paths_are_rejected(stack, capsys):
    args = _child_args()
    args.plugin_path = [Path("/a"), Path("/b")]
    assert internal_profiling.run_internal_profiling(args) == 1
    assert "--plugin-path" in capsys.readouterr().err


@pytest.mark.parametrize(
    "field, value, flag",
    [
        ("graph", [], "--graph"),
        ("graph", [Path("/a.json"), Path("/b.json")], "--graph"),
        ("engine", [1, 2], "--engine"),
    ],
)
def test_exactly_one_graph_and_engine_are_required(stack, capsys, field, value, flag):
    args = _child_args()
    setattr(args, field, value)
    assert internal_profiling.run_internal_profiling(args) == 1
    assert flag in capsys.readouterr().err
    stack["executor"].prepare.assert_not_called()

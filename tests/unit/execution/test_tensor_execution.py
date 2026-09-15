# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from dnn_benchmarking.config.benchmark_config import MetricsConfig, SuiteConfig
from dnn_benchmarking.execution.suite_runner import (
    _prepare_graph_inputs,
    run_single_provider_engine,
)
from dnn_benchmarking.graph.tensor_info import TensorInfo
from dnn_benchmarking.reporting.statistics import BenchmarkResult


def _tensor(uid: int, *, output: bool = False) -> TensorInfo:
    return TensorInfo(
        uid=uid,
        name=f"tensor_{uid}",
        dims=[2],
        strides=[1],
        data_type="float",
        is_virtual=False,
        is_output=output,
    )


def test_init_modes_and_manifest_replay_use_same_values(tmp_path: Path) -> None:
    graph = {"name": "g", "nodes": [], "tensors": []}
    info = [_tensor(1), _tensor(2, output=True)]
    graph_path = tmp_path / "g.json"

    ones, manifest, _ = _prepare_graph_inputs(
        graph_path,
        graph,
        info,
        SuiteConfig(input_init="ones", tensor_output_dir=tmp_path / "capture"),
    )
    replayed, _, _ = _prepare_graph_inputs(
        graph_path,
        graph,
        info,
        SuiteConfig(input_manifest=Path(manifest)),
    )

    np.testing.assert_array_equal(ones[1], np.ones(2, dtype=np.float32))
    np.testing.assert_array_equal(replayed[1], ones[1])


@patch("dnn_benchmarking.execution.suite_runner.BufferManager")
@patch("dnn_benchmarking.execution.suite_runner.Executor")
def test_output_capture_runs_after_timing_and_records_manifest(
    executor_cls: MagicMock, buffer_manager_cls: MagicMock, tmp_path: Path
) -> None:
    events: list[str] = []
    executor = MagicMock()
    executor.init_time_ms = 0.1
    executor.workspace_size = 0
    executor.benchmark.side_effect = lambda *args, **kwargs: (
        events.append("benchmark")
        or BenchmarkResult(host_timings=[1.0], kernel_timings=[], metadata=None)
    )
    executor.execute_once.side_effect = lambda *args: events.append("capture")
    executor_cls.return_value = executor

    manager = MagicMock()
    manager.__enter__.return_value = manager
    manager.__exit__.return_value = False
    manager.create_variant_pack.return_value = {1: 1, 2: 2}
    manager.get_output_data.return_value = np.array([3.0, 4.0], dtype=np.float32)
    buffer_manager_cls.return_value = manager

    graph = {"name": "g", "nodes": [], "tensors": []}
    result = run_single_provider_engine(
        graph_path=tmp_path / "g.json",
        graph_json_str=json.dumps(graph),
        graph_name="g",
        tensor_infos=[_tensor(1), _tensor(2, output=True)],
        config=SuiteConfig(
            metrics=MetricsConfig(tier="off"), tensor_output_dir=tmp_path / "capture"
        ),
        handle=MagicMock(),
        provider="engine",
        engine_id=7,
        plugin_path=None,
        reference_outputs=None,
        reference_error=None,
        input_data={1: np.array([1.0, 2.0], dtype=np.float32)},
        validation_requested=False,
        graph_json=graph,
    )

    assert events == ["benchmark", "capture"]
    assert result.status == "success"
    assert result.tensor_manifest is not None
    manifest = json.loads(Path(result.tensor_manifest).read_text())
    assert manifest["phase"] == "output"
    assert manifest["producer"] == {"engine_id": "7", "provider": "engine"}
    assert np.fromfile(
        Path(result.tensor_manifest).parent / "tensor-2.bin", dtype="<f4"
    ).tolist() == [3.0, 4.0]


@patch("dnn_benchmarking.execution.suite_runner._write_row_outputs")
@patch("dnn_benchmarking.execution.suite_runner.BufferManager")
@patch("dnn_benchmarking.execution.suite_runner.Executor")
def test_output_capture_failure_preserves_successful_timing(
    executor_cls: MagicMock,
    buffer_manager_cls: MagicMock,
    write_outputs: MagicMock,
    tmp_path: Path,
) -> None:
    executor = MagicMock()
    executor.init_time_ms = 0.1
    executor.workspace_size = 0
    executor.benchmark.return_value = BenchmarkResult(
        host_timings=[1.0], kernel_timings=[], metadata=None
    )
    executor_cls.return_value = executor
    manager = MagicMock()
    manager.__enter__.return_value = manager
    manager.__exit__.return_value = False
    manager.create_variant_pack.return_value = {1: 1, 2: 2}
    manager.get_output_data.return_value = np.array([3.0, 4.0], dtype=np.float32)
    buffer_manager_cls.return_value = manager
    write_outputs.side_effect = OSError("disk full")

    result = run_single_provider_engine(
        graph_path=tmp_path / "g.json",
        graph_json_str="{}",
        graph_name="g",
        tensor_infos=[_tensor(1), _tensor(2, output=True)],
        config=SuiteConfig(
            metrics=MetricsConfig(tier="off"), tensor_output_dir=tmp_path
        ),
        handle=MagicMock(),
        provider="engine",
        engine_id=7,
        plugin_path=None,
        reference_outputs=None,
        reference_error=None,
        input_data={1: np.ones(2, dtype=np.float32)},
        validation_requested=False,
        graph_json={"name": "g"},
    )

    assert result.status == "success"
    assert result.host_stats is not None
    assert result.tensor_manifest is None
    assert result.warnings and "disk full" in result.warnings[0]

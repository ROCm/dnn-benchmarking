# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Unit tests for suite_runner module."""

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

from dnn_benchmarking.execution.suite_runner import (
    run_graph_all_providers,
    run_graph_pytorch_backend,
    _resolve_engine_name,
    _resolve_engine_version,
    _get_reference_provider,
    _check_correctness,
    _BFLOAT16_RTOL,
    _BFLOAT16_ATOL,
    _pytorch_tuned_argv,
    _run_pytorch_oracle_pass,
    _run_pytorch_tuned_child,
    _run_timed_pytorch_row,
    _TimedPytorchRow,
    _compute_reference_outputs_once,
    _hipdnn_buffer_device,
    set_plugin_path,
    _collect_basic_metrics_post_loop,
)
from dnn_benchmarking.config.benchmark_config import (
    MetricsConfig,
    ReferenceProviderName,
    SuiteConfig,
    ValidationConfig,
)
from dnn_benchmarking.common.exceptions import ExecutionError, UnsupportedGraphError
from dnn_benchmarking.reporting.statistics import (
    BenchmarkMetadata,
    BenchmarkResult,
    BenchmarkStats,
)
from dnn_benchmarking.reporting.suite_results import (
    CorrectnessResult,
    GraphResult,
    ProviderEngineResult,
    oracle_speedup,
)
from dnn_benchmarking.validation.reference_provider import ReferenceOutput


def _make_tensor_info(
    uid: int,
    is_output: bool = False,
    is_virtual: bool = False,
    data_type: str = "float",
    dims=None,
    strides=None,
    value=None,
):
    """Create a mock TensorInfo object."""
    ti = MagicMock()
    ti.uid = uid
    ti.is_output = is_output
    ti.is_virtual = is_virtual
    ti.data_type = data_type
    ti.dims = dims or [1]
    ti.strides = strides or []
    ti.value = value
    ti.is_pass_by_value = value is not None
    ti.storage_elements = 1
    ti.size_bytes = 4
    return ti


def _make_graph_json():
    """Create a minimal graph JSON dict."""
    return {"name": "test_graph", "nodes": [], "tensors": []}


def _make_config(**overrides):
    """Create a SuiteConfig with optional overrides."""
    defaults = {
        "warmup_iters": 2,
        "benchmark_iters": 3,
        "seed": 42,
    }
    defaults.update(overrides)
    return SuiteConfig(**defaults)


def _make_bm_mock():
    """Create a BufferManager mock that supports the context-manager protocol."""
    mock_bm = MagicMock()
    mock_bm.__enter__ = MagicMock(return_value=mock_bm)
    mock_bm.__exit__ = MagicMock(return_value=False)
    mock_bm.create_variant_pack.return_value = {1: 100}
    return mock_bm


def test_resolve_engine_version_uses_loaded_plugin_metadata():
    handle = MagicMock()
    handle.get_engine_info.return_value.version = "2.3.4"

    assert _resolve_engine_version(handle, 7) == "2.3.4"
    handle.get_engine_info.assert_called_once_with(7)


def _make_exec_factory(
    engine_ids=None,
    build_time_ms: float = 1.0,
    has_kernel_timings: bool = False,
    prepare_side_effect=None,
    discover_side_effect=None,
):
    """Build a factory for Executor() that handles both discovery and execution.

    The first Executor() call (in run_graph_all_providers) is for discovery
    and only uses .discover_engines(); subsequent calls are per-engine and
    use .prepare(), .warmup(), .benchmark(). All instances share the same
    mock by default; override with side_effects when behaviour must differ.
    """

    def make_instance(*args, **kwargs):
        m = MagicMock()
        m.build_time_ms = build_time_ms
        if discover_side_effect is not None:
            m.discover_engines.side_effect = discover_side_effect
        else:
            m.discover_engines.return_value = engine_ids or []
        if prepare_side_effect is not None:
            m.prepare.side_effect = prepare_side_effect
        bench_result = MagicMock()
        bench_result.host_timings = [1.0]
        bench_result.kernel_timings = [0.5] if has_kernel_timings else None
        bench_result.has_kernel_timings = has_kernel_timings
        m.benchmark.return_value = bench_result
        return m

    return make_instance


class TestPluginPathLoading:
    """Explicit benchmark plugin paths should replace default plugin search paths."""

    def test_set_plugin_path_defaults_to_absolute_loading(self) -> None:
        hipdnn = MagicMock()
        hipdnn.PluginLoadingMode.ABSOLUTE = "absolute"

        set_plugin_path(hipdnn, Path("/plugins/engines"))

        # set_plugin_path forwards the native string form of the path; compare
        # against that rather than a hardcoded POSIX path so this holds on Windows.
        hipdnn.set_engine_plugin_paths.assert_called_once_with(
            [str(Path("/plugins/engines"))], "absolute"
        )


class TestRunGraphAllProviders:
    """Tests for run_graph_all_providers function."""

    @patch("dnn_benchmarking.execution.suite_runner._resolve_engine_name")
    @patch("dnn_benchmarking.execution.suite_runner._get_reference_provider")
    @patch("dnn_benchmarking.execution.suite_runner.Executor")
    @patch("dnn_benchmarking.execution.suite_runner.BufferManager")
    def test_one_result_per_discovered_engine(
        self,
        mock_bm_cls,
        mock_exec_cls,
        mock_get_ref,
        mock_resolve_name,
    ):
        """run_graph_all_providers returns one ProviderEngineResult per discovered engine ID."""
        mock_resolve_name.side_effect = lambda eid, handle=None: f"engine_{eid}"
        mock_get_ref.return_value = None

        mock_exec_cls.side_effect = _make_exec_factory(
            engine_ids=[0, 1, 2], has_kernel_timings=True
        )
        mock_bm_cls.return_value = _make_bm_mock()

        config = _make_config()
        result = run_graph_all_providers(
            graph_path=Path("test.json"),
            graph_json=_make_graph_json(),
            tensor_infos=[_make_tensor_info(1), _make_tensor_info(2, is_output=True)],
            config=config,
            handle=MagicMock(),
        )

        assert isinstance(result, GraphResult)
        assert len(result.results) == 3
        assert [r.engine_id for r in result.results] == [0, 1, 2]
        assert [r.provider for r in result.results] == [
            "engine_0",
            "engine_1",
            "engine_2",
        ]

    @patch("dnn_benchmarking.execution.suite_runner._resolve_engine_name")
    @patch("dnn_benchmarking.execution.suite_runner._get_reference_provider")
    @patch("dnn_benchmarking.execution.suite_runner.Executor")
    @patch("dnn_benchmarking.execution.suite_runner.BufferManager")
    def test_prepare_failure_records_error_status(
        self,
        mock_bm_cls,
        mock_exec_cls,
        mock_get_ref,
        mock_resolve_name,
    ):
        """When Executor.prepare() fails, the result is status='error' with no timing."""
        mock_resolve_name.return_value = "engine_0"
        mock_get_ref.return_value = None

        mock_exec_cls.side_effect = _make_exec_factory(
            engine_ids=[0],
            prepare_side_effect=ExecutionError("build failed"),
        )

        result = run_graph_all_providers(
            graph_path=Path("test.json"),
            graph_json=_make_graph_json(),
            tensor_infos=[_make_tensor_info(1)],
            config=_make_config(),
            handle=MagicMock(),
        )

        assert len(result.results) == 1
        r = result.results[0]
        assert r.status == "error"
        assert "build failed" in r.error_message
        assert r.build_time_ms is None
        assert r.gpu_kernel_stats is None
        assert r.host_stats is None

    @patch("dnn_benchmarking.execution.suite_runner._resolve_engine_name")
    @patch("dnn_benchmarking.execution.suite_runner._get_reference_provider")
    @patch("dnn_benchmarking.execution.suite_runner.Executor")
    @patch("dnn_benchmarking.execution.suite_runner.BufferManager")
    def test_check_support_failure_records_skipped_status(
        self,
        mock_bm_cls,
        mock_exec_cls,
        mock_get_ref,
        mock_resolve_name,
    ):
        """An UnsupportedGraphError is recorded as skipped."""
        mock_resolve_name.return_value = "engine_0"
        mock_get_ref.return_value = None

        mock_exec_cls.side_effect = _make_exec_factory(
            engine_ids=[0],
            prepare_side_effect=UnsupportedGraphError(
                "Backend support check failed: not supported"
            ),
        )

        result = run_graph_all_providers(
            graph_path=Path("test.json"),
            graph_json=_make_graph_json(),
            tensor_infos=[_make_tensor_info(1)],
            config=_make_config(),
            handle=MagicMock(),
        )

        assert len(result.results) == 1
        r = result.results[0]
        assert r.status == "skipped"
        assert r.skip_reason is not None

    @patch("dnn_benchmarking.execution.suite_runner._resolve_engine_name")
    @patch("dnn_benchmarking.execution.suite_runner._get_reference_provider")
    @patch("dnn_benchmarking.execution.suite_runner.Executor")
    @patch("dnn_benchmarking.execution.suite_runner.BufferManager")
    def test_successful_execution_records_separated_timing(
        self,
        mock_bm_cls,
        mock_exec_cls,
        mock_get_ref,
        mock_resolve_name,
    ):
        """Success: status='success' with separate build_time_ms / gpu_kernel_stats / host_stats."""
        mock_resolve_name.return_value = "engine_0"
        mock_get_ref.return_value = None

        mock_exec_cls.side_effect = _make_exec_factory(
            engine_ids=[0], build_time_ms=12.5, has_kernel_timings=True
        )
        mock_bm_cls.return_value = _make_bm_mock()

        result = run_graph_all_providers(
            graph_path=Path("test.json"),
            graph_json=_make_graph_json(),
            tensor_infos=[_make_tensor_info(1), _make_tensor_info(2, is_output=True)],
            config=_make_config(),
            handle=MagicMock(),
        )

        r = result.results[0]
        assert r.status == "success"
        assert r.build_time_ms == 12.5
        assert isinstance(r.gpu_kernel_stats, BenchmarkStats)
        assert isinstance(r.host_stats, BenchmarkStats)


class TestDiscoveryFailure:
    """Discovery-level failures are surfaced as graph-level errors."""

    @patch("dnn_benchmarking.execution.suite_runner.Executor")
    def test_discovery_exception_is_recorded_as_graph_error(self, mock_exec_cls):
        """When discover_engines raises, the graph gets a single error entry."""
        mock_exec_cls.side_effect = _make_exec_factory(
            discover_side_effect=ExecutionError("backend rejected graph")
        )

        result = run_graph_all_providers(
            graph_path=Path("test.json"),
            graph_json=_make_graph_json(),
            tensor_infos=[_make_tensor_info(1)],
            config=_make_config(),
            handle=MagicMock(),
        )

        assert len(result.results) == 1
        r = result.results[0]
        assert r.status == "error"
        assert "Engine discovery failed" in r.error_message
        assert "backend rejected graph" in r.error_message

    @patch("dnn_benchmarking.execution.suite_runner.Executor")
    def test_empty_discovery_recorded_as_graph_error(self, mock_exec_cls):
        """When discovery returns no engines, surface as a graph-level error."""
        mock_exec_cls.side_effect = _make_exec_factory(engine_ids=[])

        result = run_graph_all_providers(
            graph_path=Path("test.json"),
            graph_json=_make_graph_json(),
            tensor_infos=[_make_tensor_info(1)],
            config=_make_config(),
            handle=MagicMock(),
        )

        assert len(result.results) == 1
        assert result.results[0].status == "error"
        assert "No engines discovered" in result.results[0].error_message

    def test_input_generation_exception_recorded_as_graph_error(self):
        """Bad tensor metadata during shared input generation does not abort the suite."""
        with (
            patch("dnn_benchmarking.execution.suite_runner.Executor") as mock_exec_cls,
            patch(
                "dnn_benchmarking.execution.suite_runner._get_reference_provider",
                return_value=None,
            ),
            patch(
                "dnn_benchmarking.execution.suite_runner.generate_input_data",
                side_effect=ValueError("bad tensor strides"),
            ),
        ):
            mock_exec_cls.side_effect = _make_exec_factory(engine_ids=[7])

            result = run_graph_all_providers(
                graph_path=Path("test.json"),
                graph_json=_make_graph_json(),
                tensor_infos=[_make_tensor_info(1)],
                config=_make_config(),
                handle=MagicMock(),
            )

        assert len(result.results) == 1
        r = result.results[0]
        assert r.status == "error"
        assert r.provider == "unknown"
        assert "Input data generation failed" in r.error_message
        assert "bad tensor strides" in r.error_message
        assert r.correctness is not None
        assert r.correctness.passed is False
        assert result.engine_ids == [7]

    @patch("dnn_benchmarking.execution.suite_runner.Executor")
    def test_no_engines_unsupported_error_recorded_as_skipped(self, mock_exec_cls):
        """UnsupportedGraphError during discovery is recorded as skipped."""
        mock_exec_cls.side_effect = _make_exec_factory(
            discover_side_effect=UnsupportedGraphError(
                "Failed to get ranked engine ids: No engine configurations available for the graph."
            )
        )

        result = run_graph_all_providers(
            graph_path=Path("test.json"),
            graph_json=_make_graph_json(),
            tensor_infos=[_make_tensor_info(1)],
            config=_make_config(),
            handle=MagicMock(),
        )

        assert len(result.results) == 1
        r = result.results[0]
        assert r.status == "skipped"
        assert "No engine configurations" in (r.skip_reason or "")

    @patch("dnn_benchmarking.execution.suite_runner._resolve_engine_name")
    @patch("dnn_benchmarking.execution.suite_runner._get_reference_provider")
    @patch("dnn_benchmarking.execution.suite_runner.Executor")
    @patch("dnn_benchmarking.execution.suite_runner.BufferManager")
    def test_engine_filter_runs_explicit_id_without_discovery(
        self,
        mock_bm_cls,
        mock_exec_cls,
        mock_get_ref,
        mock_resolve_name,
    ):
        """Explicit --engine IDs run in CLI order without discovery filtering."""
        mock_resolve_name.side_effect = lambda eid, handle=None: f"engine_{eid}"
        mock_get_ref.return_value = None
        mock_exec_cls.side_effect = _make_exec_factory(engine_ids=[0, 1])
        mock_bm_cls.return_value = _make_bm_mock()

        result = run_graph_all_providers(
            graph_path=Path("test.json"),
            graph_json=_make_graph_json(),
            tensor_infos=[_make_tensor_info(1)],
            config=_make_config(engine_filter=[99]),
            handle=MagicMock(),
        )

        assert len(result.results) == 1
        assert result.results[0].status == "success"
        assert result.results[0].engine_id == 99

    @patch("dnn_benchmarking.execution.suite_runner._get_reference_provider")
    @patch("dnn_benchmarking.execution.suite_runner.Executor")
    @patch("dnn_benchmarking.execution.suite_runner.BufferManager")
    def test_engine_name_comes_from_the_per_engine_handle(
        self, mock_bm_cls, mock_exec_cls, mock_get_ref
    ):
        """With --engine and no shared handle, the row is named by the handle
        built for that engine, so plugin-supplied engines are not shown as hex."""
        mock_get_ref.return_value = None
        mock_exec_cls.side_effect = _make_exec_factory(engine_ids=[0])
        mock_bm_cls.return_value = _make_bm_mock()
        plugin_handle = MagicMock()
        plugin_handle.engine_id_to_name.return_value = "hipkernel:Gfx950AttentionDense"
        frontend = SimpleNamespace(
            Handle=MagicMock(return_value=plugin_handle),
            PluginLoadingMode=SimpleNamespace(ABSOLUTE=object()),
            engine_id_to_name=lambda _id: "",  # built-in registry: unknown
        )

        with patch.dict(sys.modules, {"hipdnn_frontend": frontend}):
            result = run_graph_all_providers(
                graph_path=Path("test.json"),
                graph_json=_make_graph_json(),
                tensor_infos=[_make_tensor_info(1)],
                config=_make_config(engine_filter=[0x7636]),
                handle=None,
            )

        assert result.results[0].provider == "hipkernel:Gfx950AttentionDense"

    @patch("dnn_benchmarking.execution.suite_runner._get_reference_provider")
    @patch("dnn_benchmarking.execution.suite_runner.Executor")
    @patch("dnn_benchmarking.execution.suite_runner.BufferManager")
    def test_setup_failure_row_keeps_the_built_in_registry_name(
        self, mock_bm_cls, mock_exec_cls, mock_get_ref
    ):
        """When the per-engine handle cannot be built there is no handle to ask,
        but the row must still carry the built-in engine name, not hex."""
        mock_get_ref.return_value = None
        mock_exec_cls.side_effect = _make_exec_factory(engine_ids=[0])
        mock_bm_cls.return_value = _make_bm_mock()
        frontend = SimpleNamespace(
            Handle=MagicMock(side_effect=RuntimeError("plugin failed to load")),
            PluginLoadingMode=SimpleNamespace(ABSOLUTE=object()),
            engine_id_to_name=lambda _id: "MIOPEN_ENGINE",
        )

        with patch.dict(sys.modules, {"hipdnn_frontend": frontend}):
            result = run_graph_all_providers(
                graph_path=Path("test.json"),
                graph_json=_make_graph_json(),
                tensor_infos=[_make_tensor_info(1)],
                config=_make_config(engine_filter=[1]),
                handle=None,
            )

        row = result.results[0]
        assert (row.status, row.provider) == ("error", "MIOPEN_ENGINE")


class TestSuiteConfigValidation:
    """Tests for SuiteConfig dataclass validation."""

    def test_valid_config(self):
        config = SuiteConfig(warmup_iters=5, benchmark_iters=10)
        assert config.warmup_iters == 5
        assert config.benchmark_iters == 10
        assert config.engine_filter is None
        assert config.validation.rtol is None
        assert config.validation.atol is None
        assert config.validation.tolerance_override is None
        assert config.validation.provider is ReferenceProviderName.NONE

    def test_negative_warmup_raises(self):
        with pytest.raises(ValueError, match="warmup_iters"):
            SuiteConfig(warmup_iters=-1, benchmark_iters=10)

    def test_zero_benchmark_iters_raises(self):
        with pytest.raises(ValueError, match="benchmark_iters"):
            SuiteConfig(warmup_iters=0, benchmark_iters=0)

    def test_engine_filter_accepts_list(self):
        config = SuiteConfig(engine_filter=[1, 2, 3])
        assert config.engine_filter == [1, 2, 3]

    def test_engine_filter_empty_list_raises(self):
        with pytest.raises(ValueError, match="engine_filter"):
            SuiteConfig(engine_filter=[])

    def test_engine_filter_accepts_negative_ids(self):
        """Engine IDs are FNV-1a hashes; negative values must be allowed."""
        config = SuiteConfig(engine_filter=[1, -1234567890])
        assert config.engine_filter == [1, -1234567890]

    def test_verbose_default_false(self):
        config = SuiteConfig()
        assert config.verbose is False

    def test_verbose_can_be_set(self):
        config = SuiteConfig(verbose=True)
        assert config.verbose is True

    def test_default_reference_provider_accepted(self):
        config = SuiteConfig()
        assert config.validation.provider is ReferenceProviderName.NONE
        for provider in ("none", "pytorch"):
            SuiteConfig(validation=ValidationConfig(provider=provider))


class TestEngineFilter:
    """Tests for engine filter behavior."""

    @patch("dnn_benchmarking.execution.suite_runner._resolve_engine_name")
    @patch("dnn_benchmarking.execution.suite_runner._get_reference_provider")
    @patch("dnn_benchmarking.execution.suite_runner.Executor")
    @patch("dnn_benchmarking.execution.suite_runner.BufferManager")
    def test_engine_filter_limits_iteration(
        self,
        mock_bm_cls,
        mock_exec_cls,
        mock_get_ref,
        mock_resolve_name,
    ):
        """When --engine filter is set, only that engine ID is iterated."""
        mock_resolve_name.side_effect = lambda eid, handle=None: f"engine_{eid}"
        mock_get_ref.return_value = None

        mock_exec_cls.side_effect = _make_exec_factory(engine_ids=[0, 1, 2])
        mock_bm_cls.return_value = _make_bm_mock()

        result = run_graph_all_providers(
            graph_path=Path("test.json"),
            graph_json=_make_graph_json(),
            tensor_infos=[_make_tensor_info(1)],
            config=_make_config(engine_filter=[2]),
            handle=MagicMock(),
        )

        assert len(result.results) == 1
        assert result.results[0].engine_id == 2

    @patch("dnn_benchmarking.execution.suite_runner._resolve_engine_name")
    @patch("dnn_benchmarking.execution.suite_runner._get_reference_provider")
    @patch("dnn_benchmarking.execution.suite_runner.Executor")
    @patch("dnn_benchmarking.execution.suite_runner.BufferManager")
    def test_engine_filter_list_keeps_intersection(
        self,
        mock_bm_cls,
        mock_exec_cls,
        mock_get_ref,
        mock_resolve_name,
    ):
        """engine_filter=[1, 3, 99] runs exactly those IDs in caller order."""
        mock_resolve_name.side_effect = lambda eid, handle=None: f"engine_{eid}"
        mock_get_ref.return_value = None

        mock_exec_cls.side_effect = _make_exec_factory(engine_ids=[0, 1, 2, 3])
        mock_bm_cls.return_value = _make_bm_mock()

        result = run_graph_all_providers(
            graph_path=Path("test.json"),
            graph_json=_make_graph_json(),
            tensor_infos=[_make_tensor_info(1)],
            config=_make_config(engine_filter=[1, 3, 99]),
            handle=MagicMock(),
        )

        engine_ids = [r.engine_id for r in result.results]
        assert engine_ids == [1, 3, 99]

    @patch("dnn_benchmarking.execution.suite_runner._resolve_engine_name")
    @patch("dnn_benchmarking.execution.suite_runner._get_reference_provider")
    @patch("dnn_benchmarking.execution.suite_runner.Executor")
    @patch("dnn_benchmarking.execution.suite_runner.BufferManager")
    def test_same_engine_runs_with_distinct_plugin_paths(
        self,
        mock_bm_cls,
        mock_exec_cls,
        mock_get_ref,
        mock_resolve_name,
    ):
        """Repeated engine IDs are separate ordered selections."""
        mock_resolve_name.side_effect = lambda eid, handle=None: f"engine_{eid}"
        mock_get_ref.return_value = None
        mock_exec_cls.side_effect = _make_exec_factory(has_kernel_timings=True)
        mock_bm_cls.return_value = _make_bm_mock()
        hipdnn = MagicMock()
        hipdnn.PluginLoadingMode.ABSOLUTE = "absolute"
        hipdnn.Handle.side_effect = [MagicMock(), MagicMock()]

        with patch.dict("sys.modules", {"hipdnn_frontend": hipdnn}):
            result = run_graph_all_providers(
                graph_path=Path("test.json"),
                graph_json=_make_graph_json(),
                tensor_infos=[_make_tensor_info(1)],
                config=_make_config(
                    engine_filter=[1, 1],
                    plugin_paths=[Path("/plugins/a"), Path("/plugins/b")],
                ),
                handle=None,
            )

        assert [r.engine_id for r in result.results] == [1, 1]
        # plugin_path is stored as str(Path(...)), so it carries the
        # platform separator.
        assert [r.plugin_path for r in result.results] == [
            str(Path("/plugins/a")),
            str(Path("/plugins/b")),
        ]
        hipdnn.set_engine_plugin_paths.assert_has_calls(
            [
                call([str(Path("/plugins/a"))], "absolute"),
                call([str(Path("/plugins/b"))], "absolute"),
            ]
        )

    @patch("dnn_benchmarking.execution.suite_runner._resolve_engine_name")
    @patch("dnn_benchmarking.execution.suite_runner._get_reference_provider")
    @patch("dnn_benchmarking.execution.suite_runner.Executor")
    @patch("dnn_benchmarking.execution.suite_runner.BufferManager")
    def test_per_engine_handle_creation_failure_records_error_result(
        self,
        mock_bm_cls,
        mock_exec_cls,
        mock_get_ref,
        mock_resolve_name,
    ):
        """A later per-engine handle failure records an error row and continues."""
        mock_resolve_name.side_effect = lambda eid, handle=None: f"engine_{eid}"
        mock_get_ref.return_value = None
        mock_exec_cls.side_effect = _make_exec_factory(has_kernel_timings=True)
        mock_bm_cls.return_value = _make_bm_mock()
        hipdnn = MagicMock()
        hipdnn.PluginLoadingMode.ABSOLUTE = "absolute"
        hipdnn.Handle.side_effect = [MagicMock(), RuntimeError("bad plugin")]

        with patch.dict("sys.modules", {"hipdnn_frontend": hipdnn}):
            result = run_graph_all_providers(
                graph_path=Path("test.json"),
                graph_json=_make_graph_json(),
                tensor_infos=[_make_tensor_info(1)],
                config=_make_config(
                    engine_filter=[1, 2],
                    plugin_paths=[Path("/plugins/a"), Path("/plugins/b")],
                ),
                handle=None,
            )

        assert [r.status for r in result.results] == ["success", "error"]
        assert result.results[0].plugin_path == str(Path("/plugins/a"))
        assert result.results[1].plugin_path == str(Path("/plugins/b"))
        assert "bad plugin" in (result.results[1].error_message or "")
        assert result.results[1].correctness is not None
        assert result.results[1].correctness.execution_success is False


class TestNoRetryOnFailure:
    """Single attempt per engine -- no automatic retry."""

    @patch("dnn_benchmarking.execution.suite_runner._resolve_engine_name")
    @patch("dnn_benchmarking.execution.suite_runner._get_reference_provider")
    @patch("dnn_benchmarking.execution.suite_runner.Executor")
    @patch("dnn_benchmarking.execution.suite_runner.BufferManager")
    def test_no_retry_on_failure(
        self,
        mock_bm_cls,
        mock_exec_cls,
        mock_get_ref,
        mock_resolve_name,
    ):
        """No retry on failure -- single attempt per engine."""
        mock_resolve_name.return_value = "engine_0"
        mock_get_ref.return_value = None

        mock_exec_cls.side_effect = _make_exec_factory(
            engine_ids=[0],
            prepare_side_effect=ExecutionError("fail"),
        )

        result = run_graph_all_providers(
            graph_path=Path("test.json"),
            graph_json=_make_graph_json(),
            tensor_infos=[_make_tensor_info(1)],
            config=_make_config(),
            handle=MagicMock(),
        )

        # One Executor for discovery + one for the single failed engine.
        assert mock_exec_cls.call_count == 2
        assert result.results[0].status == "error"


class TestCorrectnessChecking:
    """Tests for correctness checking via the reference provider path."""

    @patch("dnn_benchmarking.execution.suite_runner._resolve_engine_name")
    @patch("dnn_benchmarking.execution.suite_runner._get_reference_provider")
    @patch("dnn_benchmarking.execution.suite_runner._check_correctness")
    @patch("dnn_benchmarking.execution.suite_runner.Executor")
    @patch("dnn_benchmarking.execution.suite_runner.BufferManager")
    def test_tolerance_match_populated_from_comparator(
        self,
        mock_bm_cls,
        mock_exec_cls,
        mock_check_corr,
        mock_get_ref,
        mock_resolve_name,
    ):
        """Successful execution populates correctness.tolerance_match from the validator."""
        mock_resolve_name.return_value = "engine_0"

        mock_get_ref.return_value = MagicMock()
        mock_check_corr.return_value = CorrectnessResult(
            execution_success=True,
            tolerance_match=True,
            rtol=1e-5,
            atol=1e-8,
            max_abs_diff=1e-7,
            max_rel_diff=1e-6,
        )

        mock_exec_cls.side_effect = _make_exec_factory(engine_ids=[0])
        mock_bm_cls.return_value = _make_bm_mock()

        result = run_graph_all_providers(
            graph_path=Path("test.json"),
            graph_json=_make_graph_json(),
            tensor_infos=[_make_tensor_info(1), _make_tensor_info(2, is_output=True)],
            config=_make_config(),
            handle=MagicMock(),
        )

        r = result.results[0]
        assert r.correctness is not None
        assert r.correctness.tolerance_match is True
        assert r.correctness.execution_success is True

    @patch("dnn_benchmarking.execution.suite_runner._resolve_engine_name")
    @patch("dnn_benchmarking.execution.suite_runner._get_reference_provider")
    @patch("dnn_benchmarking.execution.suite_runner.Executor")
    @patch("dnn_benchmarking.execution.suite_runner.BufferManager")
    def test_tolerance_match_none_when_not_requested(
        self,
        mock_bm_cls,
        mock_exec_cls,
        mock_get_ref,
        mock_resolve_name,
    ):
        """When --validate is not requested, tolerance_match is None (no correctness performed)."""
        mock_resolve_name.return_value = "engine_0"
        mock_get_ref.return_value = None

        mock_exec_cls.side_effect = _make_exec_factory(engine_ids=[0])
        mock_bm_cls.return_value = _make_bm_mock()

        result = run_graph_all_providers(
            graph_path=Path("test.json"),
            graph_json=_make_graph_json(),
            tensor_infos=[_make_tensor_info(1)],
            config=_make_config(),  # reference_provider defaults to "none"
            handle=MagicMock(),
        )

        r = result.results[0]
        assert r.correctness is not None
        assert r.correctness.tolerance_match is None
        assert r.correctness.execution_success is True

    @patch("dnn_benchmarking.execution.suite_runner._resolve_engine_name")
    @patch("dnn_benchmarking.execution.suite_runner._get_reference_provider")
    @patch("dnn_benchmarking.execution.suite_runner.Executor")
    @patch("dnn_benchmarking.execution.suite_runner.BufferManager")
    def test_tolerance_match_false_when_requested_but_unsupported(
        self,
        mock_bm_cls,
        mock_exec_cls,
        mock_get_ref,
        mock_resolve_name,
    ):
        """--validate requested but provider doesn't support graph -> tolerance_match=False."""
        mock_resolve_name.return_value = "engine_0"
        mock_get_ref.return_value = None  # provider unavailable for this graph

        mock_exec_cls.side_effect = _make_exec_factory(engine_ids=[0])
        mock_bm_cls.return_value = _make_bm_mock()

        result = run_graph_all_providers(
            graph_path=Path("test.json"),
            graph_json=_make_graph_json(),
            tensor_infos=[_make_tensor_info(1)],
            config=_make_config(validation=ValidationConfig(provider="pytorch")),
            handle=MagicMock(),
        )

        r = result.results[0]
        assert r.correctness is not None
        assert r.correctness.tolerance_match is False
        assert r.correctness.execution_success is True
        assert "does not support" in (r.correctness.error_message or "")

    @patch("dnn_benchmarking.execution.suite_runner._resolve_engine_name")
    @patch("dnn_benchmarking.execution.suite_runner._get_reference_provider")
    @patch("dnn_benchmarking.execution.suite_runner.Executor")
    @patch("dnn_benchmarking.execution.suite_runner.BufferManager")
    def test_execution_success_false_on_error(
        self,
        mock_bm_cls,
        mock_exec_cls,
        mock_get_ref,
        mock_resolve_name,
    ):
        """correctness.execution_success is False when benchmark errors."""
        mock_resolve_name.return_value = "engine_0"
        mock_get_ref.return_value = None

        mock_exec_cls.side_effect = _make_exec_factory(
            engine_ids=[0],
            prepare_side_effect=ExecutionError("boom"),
        )

        result = run_graph_all_providers(
            graph_path=Path("test.json"),
            graph_json=_make_graph_json(),
            tensor_infos=[_make_tensor_info(1)],
            config=_make_config(),
            handle=MagicMock(),
        )

        r = result.results[0]
        assert r.correctness is not None
        assert r.correctness.execution_success is False
        assert r.correctness.tolerance_match is None

    @patch("dnn_benchmarking.execution.suite_runner._run_timed_pytorch_row")
    @patch("dnn_benchmarking.execution.suite_runner._resolve_engine_name")
    @patch("dnn_benchmarking.execution.suite_runner._get_reference_provider")
    @patch("dnn_benchmarking.execution.suite_runner._check_correctness")
    @patch("dnn_benchmarking.execution.suite_runner.Executor")
    @patch("dnn_benchmarking.execution.suite_runner.BufferManager")
    def test_validate_pytorch_adds_timed_reference_row(
        self,
        mock_bm_cls,
        mock_exec_cls,
        mock_check_corr,
        mock_get_ref,
        mock_resolve_name,
        mock_timed_reference,
    ):
        """--validate pytorch adds a timed reference row and reuses its outputs."""
        mock_resolve_name.return_value = "engine_1"
        ref_outputs = {
            2: ReferenceOutput(data=np.array([1.0], dtype=np.float32), tensor_uid=2)
        }
        ref_provider = MagicMock()
        ref_provider.name = "pytorch"
        mock_get_ref.return_value = ref_provider
        timed_result = ProviderEngineResult(
            provider="pytorch",
            engine_id=0,
            status="success",
            role="reference",
            host_stats=BenchmarkStats.from_timings([2.0]),
            gpu_kernel_stats=BenchmarkStats.from_timings([1.0]),
        )
        mock_timed_reference.return_value = MagicMock(
            result=timed_result,
            outputs=ref_outputs,
        )
        mock_check_corr.return_value = CorrectnessResult(
            execution_success=True,
            tolerance_match=True,
            rtol=1e-5,
            atol=1e-6,
        )
        mock_exec_cls.side_effect = _make_exec_factory(engine_ids=[1])
        mock_bm_cls.return_value = _make_bm_mock()

        result = run_graph_all_providers(
            graph_path=Path("test.json"),
            graph_json=_make_graph_json(),
            tensor_infos=[_make_tensor_info(1), _make_tensor_info(2, is_output=True)],
            config=_make_config(validation=ValidationConfig(provider="pytorch")),
            handle=MagicMock(),
        )

        assert [r.role for r in result.results] == ["reference", "engine"]
        assert result.results[0].provider == "pytorch"
        assert result.results[0].status == "success"
        ref_provider.compute_reference.assert_not_called()
        assert mock_check_corr.call_args.args[3] is ref_outputs

    @patch("dnn_benchmarking.execution.suite_runner._run_timed_pytorch_row")
    @patch("dnn_benchmarking.execution.suite_runner._resolve_engine_name")
    @patch("dnn_benchmarking.execution.suite_runner._get_reference_provider")
    @patch("dnn_benchmarking.execution.suite_runner._check_correctness")
    @patch("dnn_benchmarking.execution.suite_runner.Executor")
    @patch("dnn_benchmarking.execution.suite_runner.BufferManager")
    def test_validate_pytorch_falls_back_when_timed_reference_skips(
        self,
        mock_bm_cls,
        mock_exec_cls,
        mock_check_corr,
        mock_get_ref,
        mock_resolve_name,
        mock_timed_reference,
    ):
        """A skipped timed PyTorch row does not prevent CPU reference fallback."""
        mock_resolve_name.return_value = "engine_1"
        ref_outputs = {
            2: ReferenceOutput(data=np.array([1.0], dtype=np.float32), tensor_uid=2)
        }
        ref_provider = MagicMock()
        ref_provider.name = "pytorch"
        ref_provider.compute_reference.return_value = ref_outputs
        mock_get_ref.return_value = ref_provider
        skipped_result = ProviderEngineResult(
            provider="pytorch",
            engine_id=0,
            status="skipped",
            role="reference",
            skip_reason="PyTorch GPU not available",
        )
        mock_timed_reference.return_value = MagicMock(
            result=skipped_result,
            outputs=None,
        )
        mock_check_corr.return_value = CorrectnessResult(
            execution_success=True,
            tolerance_match=True,
            rtol=1e-5,
            atol=1e-6,
        )
        mock_exec_cls.side_effect = _make_exec_factory(engine_ids=[1])
        mock_bm_cls.return_value = _make_bm_mock()

        result = run_graph_all_providers(
            graph_path=Path("test.json"),
            graph_json=_make_graph_json(),
            tensor_infos=[_make_tensor_info(1), _make_tensor_info(2, is_output=True)],
            config=_make_config(validation=ValidationConfig(provider="pytorch")),
            handle=MagicMock(),
        )

        assert result.results[0].role == "reference"
        assert result.results[0].status == "skipped"
        assert ref_provider.compute_reference.call_count == 1
        assert mock_check_corr.call_args.args[3] is ref_outputs

    @patch("dnn_benchmarking.execution.suite_runner._run_timed_pytorch_row")
    @patch("dnn_benchmarking.execution.suite_runner._resolve_engine_name")
    @patch("dnn_benchmarking.execution.suite_runner._get_reference_provider")
    @patch("dnn_benchmarking.execution.suite_runner._check_correctness")
    @patch("dnn_benchmarking.execution.suite_runner.Executor")
    @patch("dnn_benchmarking.execution.suite_runner.BufferManager")
    def test_validate_pytorch_nondefault_never_falls_back_after_timed_failure(
        self,
        mock_bm_cls,
        mock_exec_cls,
        mock_check_corr,
        mock_get_ref,
        mock_resolve_name,
        mock_timed_reference,
    ):
        reason = "The graph did not execute a native forward SDPA call."
        ref_provider = MagicMock()
        mock_get_ref.return_value = ref_provider
        mock_timed_reference.return_value = _TimedPytorchRow(
            result=ProviderEngineResult(
                provider="pytorch",
                engine_id=0,
                status="error",
                role="reference",
                error_message=reason,
            ),
            outputs=None,
        )
        mock_resolve_name.return_value = "engine_1"
        mock_exec_cls.side_effect = _make_exec_factory(engine_ids=[1])
        mock_bm_cls.return_value = _make_bm_mock()

        result = run_graph_all_providers(
            graph_path=Path("test.json"),
            graph_json=_make_graph_json(),
            tensor_infos=[_make_tensor_info(1)],
            config=_make_config(
                validation=ValidationConfig(provider="pytorch"),
                pytorch_sdpa_backend="math",
            ),
            handle=MagicMock(),
        )

        ref_provider.compute_reference.assert_not_called()
        mock_check_corr.assert_not_called()
        assert result.results[0].error_message == reason

    def test_cpu_pytorch_reference_receives_rocm_fa_preference(self) -> None:
        from dnn_benchmarking.config import PyTorchSdpaBackendName
        from dnn_benchmarking.execution.pytorch_ops import _sdpa_backend
        from dnn_benchmarking.validation.providers.pytorch_provider import (
            PyTorchReferenceProvider,
        )

        class ScopedProvider(PyTorchReferenceProvider):
            def compute_reference(self, graph_json, input_data):
                state = _sdpa_backend._ACTIVE_SDPA_BACKEND.get()
                assert state is not None
                assert state.selection is PyTorchSdpaBackendName.FLASH
                assert state.rocm_fa_library == "aotriton"
                return {}

        outputs, error = _compute_reference_outputs_once(
            ScopedProvider(),
            _make_graph_json(),
            {},
            _make_config(
                validation=ValidationConfig(provider="pytorch"),
                pytorch_sdpa_backend="flash",
                pytorch_rocm_fa_library="aotriton",
            ),
        )

        assert outputs is None
        assert error is not None
        assert "The graph did not execute a native forward SDPA call." in error

    def test_cpu_pytorch_reference_succeeds_after_native_sdpa(self) -> None:
        import torch

        from dnn_benchmarking.execution import pytorch_ops
        from dnn_benchmarking.validation.providers.pytorch_provider import (
            PyTorchReferenceProvider,
        )

        class ExecutedProvider(PyTorchReferenceProvider):
            def compute_reference(self, graph_json, input_data):
                query = torch.rand(1, 1, 2, 4)
                return pytorch_ops.execute_selected_sdpa(
                    query,
                    query,
                    query,
                    attn_mask=None,
                    dropout_p=0.0,
                    is_causal=False,
                    scale=None,
                )

        outputs, error = _compute_reference_outputs_once(
            ExecutedProvider(),
            _make_graph_json(),
            {},
            _make_config(
                validation=ValidationConfig(provider="pytorch"),
                pytorch_sdpa_backend="math",
            ),
        )

        assert outputs is not None
        assert error is None

    def test_cpu_pytorch_reference_preserves_unavailable_selection(self) -> None:
        from contextlib import nullcontext

        import torch
        from torch.nn import attention

        from dnn_benchmarking.config import PyTorchSdpaBackendName
        from dnn_benchmarking.execution import pytorch_ops
        from dnn_benchmarking.validation.providers.pytorch_provider import (
            PyTorchReferenceProvider,
        )

        class StrictCpuProvider(PyTorchReferenceProvider):
            def compute_reference(self, graph_json, input_data):
                query = torch.rand(1, 1, 2, 4)
                return pytorch_ops.execute_selected_sdpa(
                    query,
                    query,
                    query,
                    attn_mask=None,
                    dropout_p=0.0,
                    is_causal=False,
                    scale=None,
                )

        dispatch_error = RuntimeError("No viable backend for this CPU input")
        sdpa = MagicMock(side_effect=dispatch_error)
        with (
            patch.object(attention, "sdpa_kernel", return_value=nullcontext()),
            patch.object(torch.nn.functional, "scaled_dot_product_attention", sdpa),
        ):
            outputs, error = _compute_reference_outputs_once(
                StrictCpuProvider(),
                _make_graph_json(),
                {},
                _make_config(
                    validation=ValidationConfig(provider="pytorch"),
                    pytorch_sdpa_backend=PyTorchSdpaBackendName.EFFICIENT,
                ),
            )

        assert outputs is None
        assert error is not None
        assert error.startswith(
            "Requested PyTorch SDPA backend 'efficient' is unavailable; "
            "no fallback is used."
        )
        sdpa.assert_called_once()

    @patch("dnn_benchmarking.execution.pytorch_executor.PyTorchCudaExecutor")
    @patch("dnn_benchmarking.execution.pytorch_buffer_manager.PyTorchCudaBufferManager")
    def test_timed_pytorch_reference_uses_auto_timing(
        self,
        mock_buffer_manager_cls,
        mock_pytorch_executor_cls,
    ):
        """PyTorch reference rows let the executor resolve timing from runtime."""
        executor = MagicMock()
        bench_result = MagicMock()
        bench_result.host_timings = [1.0, 2.0]
        bench_result.kernel_timings = None
        bench_result.has_kernel_timings = False
        executor.benchmark.return_value = bench_result
        bench_result.metadata = BenchmarkMetadata()
        mock_pytorch_executor_cls.return_value = executor

        buffer_manager = _make_bm_mock()
        buffer_manager.get_tensors.return_value = {}
        buffer_manager.get_output_tensors.return_value = []
        mock_buffer_manager_cls.return_value = buffer_manager

        result = _run_timed_pytorch_row(
            graph_path=Path("test.json"),
            graph_json=_make_graph_json(),
            graph_name="test_graph",
            tensor_infos=[],
            config=_make_config(
                validation=ValidationConfig(provider="pytorch"),
                metrics=MetricsConfig(tier="off"),
                pytorch_sdpa_backend="flash",
                pytorch_rocm_fa_library="aotriton",
            ),
            input_data={},
            analytical_flops=None,
            analytical_flops_partial=False,
            analytical_io_bytes=None,
        )

        mock_pytorch_executor_cls.assert_called_once()
        assert result.result.host_stats is not None
        assert result.result.gpu_kernel_stats is None
        assert result.result.status == "success"
        benchmark_config = mock_pytorch_executor_cls.call_args.args[1]
        assert benchmark_config.pytorch_sdpa_backend.value == "flash"
        assert benchmark_config.pytorch_rocm_fa_library == "aotriton"


@patch("dnn_benchmarking.execution.pytorch_executor.PyTorchCudaExecutor")
@patch("dnn_benchmarking.execution.pytorch_buffer_manager.PyTorchCudaBufferManager")
def test_timed_pytorch_reference_attaches_manual_reference_warnings(
    mock_buffer_manager_cls,
    mock_executor_cls,
):
    graph_json = {
        "nodes": [
            {
                "name": "rms_bwd",
                "type": "RMSNormBackwardAttributes",
                "inputs": {
                    "dy_tensor_uid": 1,
                    "x_tensor_uid": 1,
                    "scale_tensor_uid": 2,
                    "inv_rms_tensor_uid": 3,
                },
                "outputs": {"dx_tensor_uid": 4, "dscale_tensor_uid": 2},
            }
        ],
        "tensors": [
            {"uid": 1, "dims": [2, 3, 4]},
            {"uid": 2, "dims": [4]},
            {"uid": 3, "dims": [2, 3, 1]},
            {"uid": 4, "dims": [2, 3, 4]},
        ],
    }
    executor = MagicMock()
    executor.benchmark.return_value = MagicMock(
        host_timings=[2.0],
        kernel_timings=[],
        has_kernel_timings=False,
    )
    mock_executor_cls.return_value = executor

    buffer_manager = MagicMock()
    buffer_manager.__enter__.return_value = buffer_manager
    buffer_manager.__exit__.return_value = False
    buffer_manager.get_tensors.return_value = {}
    buffer_manager.get_output_tensors.return_value = [
        _make_tensor_info(4, is_output=True)
    ]
    buffer_manager.get_output_data.return_value = np.zeros((2, 3, 4), dtype=np.float32)
    mock_buffer_manager_cls.return_value = buffer_manager

    timed = _run_timed_pytorch_row(
        graph_path=Path("rms_bwd.json"),
        graph_json=graph_json,
        graph_name="rms_bwd",
        tensor_infos=[_make_tensor_info(4, is_output=True)],
        config=_make_config(warmup_iters=0, benchmark_iters=1),
        input_data={},
        analytical_flops=None,
        analytical_flops_partial=False,
        analytical_io_bytes=None,
    )

    assert timed.result.status == "success"
    assert timed.result.warnings
    assert "RMSNormBackwardAttributes" in timed.result.warnings[0]
    assert "not solely built-in PyTorch operator time" in timed.result.warnings[0]


class TestCheckCorrectnessOutputCount:
    """_check_correctness returns tolerance_match=False when no outputs are comparable."""

    def test_no_outputs_returns_false(self):
        bm = MagicMock()
        bm.get_output_data.return_value = None

        config = SuiteConfig(validation=ValidationConfig(provider="pytorch"))
        result = _check_correctness(
            buffer_manager=bm,
            tensor_infos=[],
            graph_json=_make_graph_json(),
            ref_outputs={},
            reference_provider_name="pytorch",
            config=config,
        )

        assert result.tolerance_match is False
        assert result.execution_success is True
        assert "No output tensors to compare" in (result.error_message or "")

    def test_missing_reference_output_returns_false(self):
        bm = MagicMock()
        bm.get_output_data.return_value = np.array([0.0], dtype=np.float32)

        result = _check_correctness(
            buffer_manager=bm,
            tensor_infos=[_make_tensor_info(7, is_output=True)],
            graph_json={
                "nodes": [
                    {
                        "type": "SdpaAttributes",
                        "outputs": {"o_tensor_uid": 7},
                    }
                ]
            },
            ref_outputs={},
            reference_provider_name="pytorch",
            config=SuiteConfig(validation=ValidationConfig(provider="pytorch")),
        )

        assert result.tolerance_match is False
        assert "did not produce output tensor UID 7" in (result.error_message or "")

    def test_zero_bf16_sdpa_forward_output_uses_bfloat16_tolerance(self):
        bm = MagicMock()
        bm.get_output_data.return_value = np.zeros((2,), dtype=np.float32)

        ref_outputs = {
            7: ReferenceOutput(
                data=np.ones((2,), dtype=np.float32),
                tensor_uid=7,
            )
        }

        result = _check_correctness(
            buffer_manager=bm,
            tensor_infos=[
                _make_tensor_info(7, is_output=True, data_type="bfloat16"),
            ],
            graph_json={
                "nodes": [
                    {
                        "type": "SdpaAttributes",
                        "outputs": {"o_tensor_uid": 7},
                    }
                ]
            },
            ref_outputs=ref_outputs,
            reference_provider_name="pytorch",
            config=SuiteConfig(validation=ValidationConfig(provider="pytorch")),
        )

        assert result.tolerance_match is False
        assert result.rtol == pytest.approx(_BFLOAT16_RTOL)
        assert result.atol == pytest.approx(_BFLOAT16_ATOL)

    def test_small_bf16_output_difference_exceeds_absolute_floor(self):
        bm = MagicMock()
        bm.get_output_data.return_value = np.zeros((1,), dtype=np.float32)

        ref_outputs = {
            7: ReferenceOutput(
                data=np.array([5e-3], dtype=np.float32),
                tensor_uid=7,
            )
        }

        result = _check_correctness(
            buffer_manager=bm,
            tensor_infos=[
                _make_tensor_info(7, is_output=True, data_type="bfloat16"),
            ],
            graph_json={
                "nodes": [{"type": "PointwiseAttributes", "outputs": {"y": 7}}]
            },
            ref_outputs=ref_outputs,
            reference_provider_name="pytorch",
            config=SuiteConfig(validation=ValidationConfig(provider="pytorch")),
        )

        assert result.tolerance_match is False
        assert result.atol == pytest.approx(1e-3)

    def test_single_explicit_tolerance_overrides_both_values(self):
        bm = MagicMock()
        bm.get_output_data.return_value = np.array([1.0], dtype=np.float32)

        ref_outputs = {
            7: ReferenceOutput(
                data=np.array([1.1], dtype=np.float32),
                tensor_uid=7,
            )
        }

        result = _check_correctness(
            buffer_manager=bm,
            tensor_infos=[_make_tensor_info(7, is_output=True, data_type="bfloat16")],
            graph_json={
                "nodes": [
                    {
                        "type": "SdpaAttributes",
                        "outputs": {"o_tensor_uid": 7},
                    }
                ]
            },
            ref_outputs=ref_outputs,
            reference_provider_name="pytorch",
            config=SuiteConfig(
                validation=ValidationConfig(provider="pytorch", rtol=0.25)
            ),
        )

        assert result.tolerance_match is True
        assert result.rtol == pytest.approx(0.25)
        assert result.atol == pytest.approx(0.25)

    def test_device_reference_is_compared_without_host_copy(self):
        torch = pytest.importorskip("torch")
        bm = MagicMock()
        bm.get_output_tensor.return_value = torch.tensor([1.0, 2.0])

        # Host data disagrees, so a pass proves the device tensors were used.
        ref_outputs = {
            7: ReferenceOutput(
                data=np.array([9.0, 9.0], dtype=np.float32),
                tensor_uid=7,
                device_data=torch.tensor([1.0, 2.0]),
            )
        }

        result = _check_correctness(
            buffer_manager=bm,
            tensor_infos=[_make_tensor_info(7, is_output=True)],
            graph_json={"nodes": []},
            ref_outputs=ref_outputs,
            reference_provider_name="pytorch",
            config=SuiteConfig(validation=ValidationConfig(provider="pytorch")),
        )

        assert result.tolerance_match is True
        assert result.max_abs_diff == 0.0
        bm.get_output_data.assert_not_called()


class TestHipdnnBufferDevice:
    """Torch I/O storage is chosen only when a GPU comparison can run."""

    def test_device_reference_selects_torch_storage(self) -> None:
        host = ReferenceOutput(data=np.zeros(1), tensor_uid=1)
        device = ReferenceOutput(data=np.zeros(1), tensor_uid=2, device_data=object())

        # Timing-only (no reference) and host-only references keep DeviceBuffer.
        assert _hipdnn_buffer_device(None) is None
        assert _hipdnn_buffer_device({1: host}) is None
        assert _hipdnn_buffer_device({1: host, 2: device}) == "cuda"


class TestResolveEngineName:
    """Tests for _resolve_engine_name fallback behavior."""

    def test_falls_back_to_hex_when_lookup_fails(self):
        """If hipdnn_frontend isn't importable, the helper falls back to a hex display."""
        # Force the import inside _resolve_engine_name to fail by injecting a
        # missing module entry. We use unittest.mock.patch on builtins.__import__
        # to surgically reject just hipdnn_frontend.
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "hipdnn_frontend":
                raise ImportError("simulated missing module")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            assert _resolve_engine_name(0xABC, None) == "engine_0xabc"

    @staticmethod
    def _frontend(registry_name):
        return SimpleNamespace(engine_id_to_name=lambda _id: registry_name)

    def test_handle_names_plugin_engine_missing_from_builtin_registry(self):
        handle = MagicMock()
        handle.engine_id_to_name.return_value = "hipkernel:Gfx950AttentionDense"
        with patch.dict(sys.modules, {"hipdnn_frontend": self._frontend("")}):
            name = _resolve_engine_name(0x7636, handle)
        assert name == "hipkernel:Gfx950AttentionDense"

    def test_falls_back_silently_when_handle_carries_no_such_engine(self, capsys):
        from dnn_benchmarking.metrics._diagnostic import reset

        reset()  # warn_once dedups process-wide; start from a clean slate.
        handle = MagicMock()
        handle.engine_id_to_name.side_effect = IndexError("not loaded")
        with patch.dict(
            sys.modules, {"hipdnn_frontend": self._frontend("MIOPEN_ENGINE")}
        ):
            assert _resolve_engine_name(1, handle) == "MIOPEN_ENGINE"
        assert capsys.readouterr().err == ""


class TestProfilingPassInvocation:
    """suite_runner.py:521-542 calls the profiling orchestrator after the
    timed pass when any opt-in metric is requested. The orchestrator's
    payload lands on result.extra_metrics; orchestrator exceptions must
    not bubble out as engine errors."""

    def _setup_mocks(self, mock_exec_cls, mock_bm_cls, mock_get_ref, mock_resolve_name):
        mock_resolve_name.side_effect = lambda eid, handle=None: f"engine_{eid}"
        mock_get_ref.return_value = None
        mock_exec_cls.side_effect = _make_exec_factory(
            engine_ids=[0], has_kernel_timings=True
        )
        mock_bm_cls.return_value = _make_bm_mock()

    @patch("dnn_benchmarking.metrics.profiling_orchestrator.run_profiling_passes")
    @patch("dnn_benchmarking.execution.suite_runner._resolve_engine_name")
    @patch("dnn_benchmarking.execution.suite_runner._get_reference_provider")
    @patch("dnn_benchmarking.execution.suite_runner.Executor")
    @patch("dnn_benchmarking.execution.suite_runner.BufferManager")
    def test_orchestrator_called_once_and_payload_lands_in_extra_metrics(
        self,
        mock_bm_cls,
        mock_exec_cls,
        mock_get_ref,
        mock_resolve_name,
        mock_orch,
    ):
        self._setup_mocks(mock_exec_cls, mock_bm_cls, mock_get_ref, mock_resolve_name)
        payload = {
            "pmc": {"set": "basic", "counters": {"GRBM_GUI_ACTIVE": {"sum": 1.0}}}
        }
        mock_orch.return_value = payload

        config = _make_config(metrics=MetricsConfig(tier="basic", pmc_set="basic"))
        result = run_graph_all_providers(
            graph_path=Path("test.json"),
            graph_json=_make_graph_json(),
            tensor_infos=[_make_tensor_info(1), _make_tensor_info(2, is_output=True)],
            config=config,
            handle=MagicMock(),
        )

        # The orchestrator runs exactly once per (graph, engine). Two
        # calls here would catch the duplicate-block bug fixed in
        # commit 196a0fb33ca.
        assert mock_orch.call_count == 1
        assert len(result.results) == 1
        assert result.results[0].extra_metrics == payload

    @patch("dnn_benchmarking.metrics.profiling_orchestrator.run_profiling_passes")
    @patch("dnn_benchmarking.execution.suite_runner._resolve_engine_name")
    @patch("dnn_benchmarking.execution.suite_runner._get_reference_provider")
    @patch("dnn_benchmarking.execution.suite_runner.Executor")
    @patch("dnn_benchmarking.execution.suite_runner.BufferManager")
    def test_orchestrator_not_called_when_no_opt_in_flag(
        self,
        mock_bm_cls,
        mock_exec_cls,
        mock_get_ref,
        mock_resolve_name,
        mock_orch,
    ):
        self._setup_mocks(mock_exec_cls, mock_bm_cls, mock_get_ref, mock_resolve_name)

        # Default MetricsConfig() — basic tier, no opt-in source set.
        config = _make_config(metrics=MetricsConfig())
        result = run_graph_all_providers(
            graph_path=Path("test.json"),
            graph_json=_make_graph_json(),
            tensor_infos=[_make_tensor_info(1), _make_tensor_info(2, is_output=True)],
            config=config,
            handle=MagicMock(),
        )

        mock_orch.assert_not_called()
        assert result.results[0].extra_metrics is None

    @patch("dnn_benchmarking.metrics.profiling_orchestrator.run_profiling_passes")
    @patch("dnn_benchmarking.execution.suite_runner._resolve_engine_name")
    @patch("dnn_benchmarking.execution.suite_runner._get_reference_provider")
    @patch("dnn_benchmarking.execution.suite_runner.Executor")
    @patch("dnn_benchmarking.execution.suite_runner.BufferManager")
    def test_orchestrator_exception_does_not_fail_engine(
        self,
        mock_bm_cls,
        mock_exec_cls,
        mock_get_ref,
        mock_resolve_name,
        mock_orch,
        capsys,
    ):
        """Orchestrator failure (tool missing, parse error, anything) must
        keep the timed pass's status='success' — the headline timing data
        already exists; profiling is best-effort."""
        self._setup_mocks(mock_exec_cls, mock_bm_cls, mock_get_ref, mock_resolve_name)
        mock_orch.side_effect = RuntimeError("rocprofv3 missing")

        config = _make_config(metrics=MetricsConfig(tier="basic", pmc_set="basic"))
        result = run_graph_all_providers(
            graph_path=Path("test.json"),
            graph_json=_make_graph_json(),
            tensor_infos=[_make_tensor_info(1), _make_tensor_info(2, is_output=True)],
            config=config,
            handle=MagicMock(),
        )

        # Engine still passes; extra_metrics stays None.
        assert result.results[0].status == "success"
        assert result.results[0].extra_metrics is None
        # warn_once writes to stderr.
        captured = capsys.readouterr()
        assert "profiling pass failed" in captured.err
        assert "rocprofv3 missing" in captured.err

    @patch("dnn_benchmarking.metrics.profiling_orchestrator.run_profiling_passes")
    @patch("dnn_benchmarking.execution.suite_runner._resolve_engine_name")
    @patch("dnn_benchmarking.execution.suite_runner._get_reference_provider")
    @patch("dnn_benchmarking.execution.suite_runner.Executor")
    @patch("dnn_benchmarking.execution.suite_runner.BufferManager")
    def test_orchestrator_runs_after_buffermanager_teardown(
        self,
        mock_bm_cls,
        mock_exec_cls,
        mock_get_ref,
        mock_resolve_name,
        mock_orch,
    ):
        """Profiling pass must fire *after* the BufferManager context
        exits — only then are the parent's I/O buffers and the
        executor's workspace freed. Without this ordering, the inner
        profiling subprocess allocates its own VRAM on top of the
        parent's still-pinned tensors, which roughly doubles peak VRAM
        and can OOM on large graphs that fit fine on the headline run.
        """
        self._setup_mocks(mock_exec_cls, mock_bm_cls, mock_get_ref, mock_resolve_name)

        # Track __exit__ vs orchestrator invocation order via shared list.
        order: list[str] = []
        bm_instance = mock_bm_cls.return_value
        original_exit = bm_instance.__exit__

        def tracking_exit(*args, **kwargs):
            order.append("bm_exit")
            return original_exit(*args, **kwargs)

        bm_instance.__exit__ = tracking_exit

        def tracking_orch(**kwargs):
            order.append("orch")
            return {"pmc": {"set": "basic"}}

        mock_orch.side_effect = tracking_orch

        config = _make_config(metrics=MetricsConfig(tier="basic", pmc_set="basic"))
        run_graph_all_providers(
            graph_path=Path("test.json"),
            graph_json=_make_graph_json(),
            tensor_infos=[_make_tensor_info(1), _make_tensor_info(2, is_output=True)],
            config=config,
            handle=MagicMock(),
        )

        # Strict ordering: bm context must exit BEFORE the orchestrator
        # runs. Reversing this (the pre-fix state) is the bug.
        assert order == [
            "bm_exit",
            "orch",
        ], f"profiling must run after BufferManager teardown; got {order}"


class TestRunGraphPytorchBackend:
    """run_graph_pytorch_backend emits one provider='pytorch' engine row."""

    @patch("dnn_benchmarking.execution.suite_runner._run_timed_pytorch_row")
    def test_single_engine_row(self, mock_timed_row):
        row = ProviderEngineResult(
            provider="pytorch",
            engine_id=0,
            status="success",
            host_stats=BenchmarkStats.from_timings([2.0]),
        )
        mock_timed_row.return_value = MagicMock(result=row, outputs=None)

        result = run_graph_pytorch_backend(
            graph_path=Path("test.json"),
            graph_json=_make_graph_json(),
            tensor_infos=[_make_tensor_info(1)],
            config=_make_config(),
        )

        assert mock_timed_row.call_args.kwargs["role"] == "engine"
        assert result.engine_ids == [0]
        assert [r.provider for r in result.results] == ["pytorch"]
        assert result.results[0].status == "success"

    @patch("dnn_benchmarking.execution.suite_runner.generate_input_data")
    @patch("dnn_benchmarking.execution.suite_runner._run_timed_pytorch_row")
    def test_unsupported_operations_skip_row(
        self, mock_timed_row, mock_gen, monkeypatch
    ):
        import sys
        import types

        import dnn_benchmarking.execution as execution_pkg

        fake_ops = types.SimpleNamespace(
            get_unsupported_operations=lambda graph_json: ["FooAttributes"]
        )
        monkeypatch.setattr(execution_pkg, "pytorch_ops", fake_ops, raising=False)
        monkeypatch.setitem(
            sys.modules, "dnn_benchmarking.execution.pytorch_ops", fake_ops
        )

        result = run_graph_pytorch_backend(
            graph_path=Path("test.json"),
            graph_json=_make_graph_json(),
            tensor_infos=[_make_tensor_info(1)],
            config=_make_config(),
        )

        mock_timed_row.assert_not_called()
        # Unsupported graphs must be skipped before any input allocation.
        mock_gen.assert_not_called()
        row = result.results[0]
        assert row.status == "skipped"
        assert "unsupported operations" in (row.skip_reason or "")
        assert result.engine_ids == [0]

    @patch("dnn_benchmarking.execution.suite_runner.generate_input_data")
    def test_input_generation_failure_is_error_row(self, mock_gen):
        mock_gen.side_effect = ValueError("boom")

        result = run_graph_pytorch_backend(
            graph_path=Path("test.json"),
            graph_json=_make_graph_json(),
            tensor_infos=[_make_tensor_info(1)],
            config=_make_config(),
        )

        row = result.results[0]
        assert row.status == "error"
        assert "Input data generation failed" in (row.error_message or "")


class TestTimedPytorchRowEngineRole:
    """Engine-role rows report failures as errors, not skips."""

    @patch("dnn_benchmarking.execution.pytorch_executor.PyTorchCudaExecutor")
    @patch("dnn_benchmarking.execution.pytorch_buffer_manager.PyTorchCudaBufferManager")
    def test_engine_role_success_has_no_reference_correctness(
        self,
        mock_buffer_manager_cls,
        mock_pytorch_executor_cls,
    ):
        executor = MagicMock()
        bench_result = MagicMock()
        bench_result.host_timings = [1.0, 2.0]
        bench_result.kernel_timings = None
        bench_result.has_kernel_timings = False
        executor.benchmark.return_value = bench_result
        mock_pytorch_executor_cls.return_value = executor

        buffer_manager = _make_bm_mock()
        buffer_manager.get_tensors.return_value = {}
        buffer_manager.get_output_tensors.return_value = []
        mock_buffer_manager_cls.return_value = buffer_manager

        row = _run_timed_pytorch_row(
            graph_path=Path("test.json"),
            graph_json=_make_graph_json(),
            graph_name="test_graph",
            tensor_infos=[],
            config=_make_config(metrics=MetricsConfig(tier="off")),
            input_data={},
            analytical_flops=None,
            analytical_flops_partial=False,
            analytical_io_bytes=None,
            role="engine",
        )

        assert row.result.status == "success"
        assert row.result.role == "engine"
        assert row.outputs is None
        # Engine rows never run the extra reference-output extraction pass.
        executor.execute_once.assert_not_called()
        assert row.result.correctness is not None
        assert row.result.correctness.tolerance_match is None
        assert "No reference provider requested" in (
            row.result.correctness.error_message or ""
        )

    @patch("dnn_benchmarking.execution.pytorch_executor.PyTorchCudaExecutor")
    def test_engine_role_failure_is_error(self, mock_pytorch_executor_cls):
        mock_pytorch_executor_cls.side_effect = RuntimeError("no GPU")

        row = _run_timed_pytorch_row(
            graph_path=Path("test.json"),
            graph_json=_make_graph_json(),
            graph_name="test_graph",
            tensor_infos=[],
            config=_make_config(
                metrics=MetricsConfig(tier="off"),
                pytorch_sdpa_backend="flash",
            ),
            input_data={},
            analytical_flops=None,
            analytical_flops_partial=False,
            analytical_io_bytes=None,
            role="engine",
        )

        assert row.result.status == "error"
        assert "no GPU" in (row.result.error_message or "")

    @patch("dnn_benchmarking.execution.pytorch_executor.PyTorchCudaExecutor")
    def test_reference_role_strict_failure_is_error(self, mock_pytorch_executor_cls):
        mock_pytorch_executor_cls.side_effect = RuntimeError("no GPU")

        row = _run_timed_pytorch_row(
            graph_path=Path("test.json"),
            graph_json=_make_graph_json(),
            graph_name="test_graph",
            tensor_infos=[],
            config=_make_config(
                metrics=MetricsConfig(tier="off"),
                pytorch_sdpa_backend="efficient",
            ),
            input_data={},
            analytical_flops=None,
            analytical_flops_partial=False,
            analytical_io_bytes=None,
            role="reference",
        )

        assert row.result.status == "error"
        assert "no GPU" in (row.result.error_message or "")

    @patch("dnn_benchmarking.execution.pytorch_buffer_manager.PyTorchCudaBufferManager")
    @patch("dnn_benchmarking.execution.pytorch_executor.PyTorchCudaExecutor")
    def test_strict_reference_output_failure_is_error(
        self,
        mock_pytorch_executor_cls,
        mock_buffer_manager_cls,
    ):
        executor = MagicMock()
        executor.benchmark.return_value = BenchmarkResult(
            host_timings=[1.0],
            kernel_timings=[0.5],
            metadata=BenchmarkMetadata(),
        )
        executor.execute_once.side_effect = RuntimeError("output pass failed")
        mock_pytorch_executor_cls.return_value = executor
        mock_buffer_manager_cls.return_value = _make_bm_mock()

        row = _run_timed_pytorch_row(
            graph_path=Path("test.json"),
            graph_json=_make_graph_json(),
            graph_name="test_graph",
            tensor_infos=[],
            config=_make_config(
                metrics=MetricsConfig(tier="off"),
                pytorch_sdpa_backend="math",
            ),
            input_data={},
            analytical_flops=None,
            analytical_flops_partial=False,
            analytical_io_bytes=None,
            role="reference",
        )

        assert row.result.status == "error"

    @patch("dnn_benchmarking.execution.pytorch_executor.PyTorchCudaExecutor")
    def test_reference_role_strict_backend_unavailable_is_marked_for_no_fallback(
        self, mock_pytorch_executor_cls
    ):
        from dnn_benchmarking.execution.pytorch_ops import (
            PyTorchSdpaBackendUnavailableError,
        )

        reason = (
            "Requested PyTorch ROCm Flash Attention library 'aotriton' is "
            "unavailable; no fallback is used."
        )
        mock_pytorch_executor_cls.side_effect = PyTorchSdpaBackendUnavailableError(
            reason
        )

        row = _run_timed_pytorch_row(
            graph_path=Path("test.json"),
            graph_json=_make_graph_json(),
            graph_name="test_graph",
            tensor_infos=[],
            config=_make_config(
                metrics=MetricsConfig(tier="off"),
                pytorch_sdpa_backend="flash",
                pytorch_rocm_fa_library="aotriton",
            ),
            input_data={},
            analytical_flops=None,
            analytical_flops_partial=False,
            analytical_io_bytes=None,
            role="reference",
        )

        assert row.result.status == "error"
        assert row.result.error_message == reason


def _make_oracle_exec_factory(
    prepare_side_effect=None, order=None, knob_ids=("global.benchmarking",)
):
    """Executor factory whose third instance is the tuned (oracle) pass.

    Instance order inside run_graph_all_providers is discovery, OOTB, then
    oracle. The oracle instance reports half the OOTB kernel time so the
    speedup is unambiguous, and its own build time so a swap with the OOTB
    plan's shows.

    Args:
        prepare_side_effect: Side effect of the tuned plan build, if any.
        order: When given, receives ``"<role>.<method>"`` labels in call
            order for warmup, benchmark and execute_once.
        knob_ids: Knobs the engine exposes, as reported by engine_knob_ids.
    """
    instances = []

    def _recorder(label, value=None):
        def _call(*a, **k):
            order.append(label)
            return value

        return _call

    def make_instance(*args, **kwargs):
        m = MagicMock()
        role = {0: "discovery", 1: "ootb", 2: "oracle"}.get(
            len(instances), f"extra{len(instances)}"
        )
        tuned = role == "oracle"
        m.build_time_ms = 7.0 if tuned else 3.0
        m.discover_engines.return_value = [0]
        m.engine_knob_ids.return_value = list(knob_ids)
        bench_result = MagicMock()
        bench_result.host_timings = [1.0]
        bench_result.kernel_timings = [0.25 if tuned else 0.5]
        bench_result.has_kernel_timings = True
        m.benchmark.return_value = bench_result
        if tuned and prepare_side_effect is not None:
            m.prepare.side_effect = prepare_side_effect

        if order is not None:
            m.warmup.side_effect = _recorder(f"{role}.warmup")
            m.benchmark.side_effect = _recorder(f"{role}.benchmark", bench_result)
            m.execute_once.side_effect = _recorder(f"{role}.execute_once")

        instances.append(m)
        return m

    return make_instance, instances


class TestOraclePass:
    """--oracle-mode adds a second tuned pass per engine row without changing OOTB."""

    def _run(self, factory, oracle_mode="exhaustive"):
        with (
            patch(
                "dnn_benchmarking.execution.suite_runner._resolve_engine_name",
                side_effect=lambda eid, handle=None: f"engine_{eid}",
            ),
            patch(
                "dnn_benchmarking.execution.suite_runner._get_reference_provider",
                return_value=None,
            ),
            patch(
                "dnn_benchmarking.execution.suite_runner.Executor", side_effect=factory
            ),
            patch(
                "dnn_benchmarking.execution.suite_runner.BufferManager",
                return_value=_make_bm_mock(),
            ),
        ):
            return run_graph_all_providers(
                graph_path=Path("test.json"),
                graph_json=_make_graph_json(),
                tensor_infos=[_make_tensor_info(1)],
                config=_make_config(oracle_mode=oracle_mode),
                handle=MagicMock(),
            )

    def test_tuned_plan_is_the_same_engine_built_with_the_benchmarking_knob(self):
        """Only the knob may differ; a different engine is not a tuned OOTB."""
        factory, instances = _make_oracle_exec_factory()
        result = self._run(factory)

        # discovery, OOTB, oracle
        assert len(instances) == 3
        ootb, oracle = instances[1], instances[2]
        assert ootb.prepare.call_args.kwargs == {"engine_id": 0}
        assert oracle.prepare.call_args.kwargs == {
            "engine_id": 0,
            "knobs": {"global.benchmarking": 1},
        }

    def test_oracle_uses_an_isolated_handle_on_the_same_stream(self):
        class Handle:
            instances = []

            def __init__(self):
                self.stream = -1
                self.__class__.instances.append(self)

            def get_stream(self):
                return self.stream

            def set_stream(self, stream):
                self.stream = stream

        factory, instances = _make_oracle_exec_factory()
        ootb_handle = Handle()
        ootb_handle.stream = 17
        with (
            patch(
                "dnn_benchmarking.execution.suite_runner._resolve_engine_name",
                side_effect=lambda eid, handle=None: f"engine_{eid}",
            ),
            patch(
                "dnn_benchmarking.execution.suite_runner._get_reference_provider",
                return_value=None,
            ),
            patch(
                "dnn_benchmarking.execution.suite_runner.Executor", side_effect=factory
            ),
            patch(
                "dnn_benchmarking.execution.suite_runner.BufferManager",
                return_value=_make_bm_mock(),
            ),
        ):
            run_graph_all_providers(
                graph_path=Path("test.json"),
                graph_json=_make_graph_json(),
                tensor_infos=[_make_tensor_info(1)],
                config=_make_config(oracle_mode="exhaustive"),
                handle=ootb_handle,
            )

        oracle_handle = instances[2].prepare.call_args.args[0]
        assert oracle_handle is not ootb_handle
        assert oracle_handle.stream == 17
        assert instances[1].benchmark.call_args_list[-1].args[0] is ootb_handle
        assert instances[2].benchmark.call_args.args[0] is oracle_handle

    def test_tuned_pass_samples_outside_its_warmup_and_never_retimes_ootb(self):
        """The tuned plan's first execute samples every candidate kernel.

        It must run before the tuned warmup, so the tuned plan gets the same
        number of ordinary warmup iterations as the OOTB plan; folding it into
        the warmup leaves the tuned side one warmup short. The OOTB plan is
        timed exactly once, so the speedup compares against the row's own run.
        """
        order = []
        factory, _ = _make_oracle_exec_factory(order=order)
        result = self._run(factory)

        assert order == [
            "ootb.warmup",
            "ootb.benchmark",
            "oracle.execute_once",
            "oracle.warmup",
            "oracle.benchmark",
        ]
        assert oracle_speedup(result.results[0]) == 2.0

    def test_tuned_plan_reports_median_tflops(self):
        """The tuned plan gets TFLOP/s from the row's FLOPs and its own kernel
        median, so tuned and OOTB throughput compare directly."""
        factory, _ = _make_oracle_exec_factory()
        with patch(
            "dnn_benchmarking.execution.suite_runner.compute_flops",
            return_value=(10**9, False),
        ):
            result = self._run(factory)

        row = result.results[0]
        # 1e9 FLOPs: 0.5 ms -> 2 TFLOP/s (OOTB); 0.25 ms -> 4 (tuned).
        assert row.derived_tflops_per_s == pytest.approx(2.0)
        assert row.oracle.derived_tflops_per_s == pytest.approx(4.0)
        assert row.oracle.to_dict()["derived_tflops_per_s"] == pytest.approx(4.0)

    def test_oracle_failure_leaves_ootb_row_intact(self):
        factory, _ = _make_oracle_exec_factory(
            prepare_side_effect=ExecutionError("plan build failed")
        )
        result = self._run(factory)

        r = result.results[0]
        assert r.status == "success"
        assert isinstance(r.gpu_kernel_stats, BenchmarkStats)
        assert r.oracle is None
        assert r.oracle_error == "ExecutionError: plan build failed"

    def test_no_oracle_pass_when_mode_is_off(self):
        factory, instances = _make_oracle_exec_factory()
        result = self._run(factory, oracle_mode="off")

        r = result.results[0]
        assert r.oracle is None
        assert r.oracle_error is None
        # discovery + OOTB only.
        assert len(instances) == 2

    @pytest.mark.parametrize(
        "knob_ids, available",
        [
            (["global.benchmarking", "SPLIT_K"], True),
            (["SPLIT_K"], False),
        ],
    )
    def test_tuning_available_follows_the_engine_benchmarking_knob(
        self, knob_ids, available
    ):
        """hipDNN ignores an unexposed knob, so that tuned run tuned nothing."""
        factory, _ = _make_oracle_exec_factory(knob_ids=knob_ids)
        oracle = self._run(factory).results[0].oracle

        assert oracle.tuning_available is available

    def test_build_times_are_reported_per_plan(self):
        """The row carries the OOTB build; the oracle carries the tuned build."""
        factory, _ = _make_oracle_exec_factory()
        r = self._run(factory).results[0]

        assert r.build_time_ms == 3.0
        assert r.oracle.build_time_ms == 7.0


class TestOracleTunedPlanValidation:
    """--validate gates the tuned plan, not only the OOTB run."""

    def _run(self, factory, correctness):
        """Run with a reference available and a pinned correctness verdict.

        ``correctness`` is one verdict for every check, or a list consumed in
        call order (OOTB first, tuned second).
        """
        check_kwargs = (
            {"side_effect": list(correctness)}
            if isinstance(correctness, list)
            else {"return_value": correctness}
        )
        with (
            patch(
                "dnn_benchmarking.execution.suite_runner._resolve_engine_name",
                side_effect=lambda eid, handle=None: f"engine_{eid}",
            ),
            patch(
                "dnn_benchmarking.execution.suite_runner._get_reference_provider",
                return_value=MagicMock(),
            ),
            patch(
                "dnn_benchmarking.execution.suite_runner."
                "_compute_reference_outputs_once",
                return_value=({1: MagicMock()}, None),
            ),
            patch(
                "dnn_benchmarking.execution.suite_runner._check_correctness",
                **check_kwargs,
            ),
            patch(
                "dnn_benchmarking.execution.suite_runner.Executor", side_effect=factory
            ),
            patch(
                "dnn_benchmarking.execution.suite_runner.BufferManager",
                return_value=_make_bm_mock(),
            ),
        ):
            return run_graph_all_providers(
                graph_path=Path("test.json"),
                graph_json=_make_graph_json(),
                tensor_infos=[_make_tensor_info(1)],
                config=_make_config(oracle_mode="exhaustive"),
                handle=MagicMock(),
            )

    @staticmethod
    def _verdict(passed: bool) -> CorrectnessResult:
        return CorrectnessResult(
            execution_success=True,
            tolerance_match=passed,
            rtol=1e-5,
            atol=1e-5,
            error_message=None if passed else "output mismatch",
        )

    def test_failing_tuned_plan_publishes_no_speedup(self):
        """Fault injection: a wrong tuned result must not carry a speedup.

        This is the regression guard for a tuned plan that computes garbage
        twice as fast and previously reported a clean 2.00x.
        """
        factory, _ = _make_oracle_exec_factory()
        r = self._run(factory, self._verdict(False)).results[0]

        assert r.oracle is not None
        assert r.oracle.correctness.passed is False
        # Timings survive as evidence; the comparison does not.
        assert r.oracle.gpu_kernel_stats is not None
        assert oracle_speedup(r) is None

    def test_failing_baseline_publishes_no_speedup(self):
        """Inverse fault injection: a wrong baseline cannot measure a gain.

        The tuned plan is correct here and the baseline is not. Comparing
        against a broken comparand produced a clean 2.00x even though one
        operand was garbage, so eligibility must consider both sides.
        """
        factory, _ = _make_oracle_exec_factory()
        # OOTB check first, tuned check second.
        r = self._run(factory, [self._verdict(False), self._verdict(True)]).results[0]

        assert r.correctness.passed is False
        assert r.oracle.correctness.passed is True
        # Both verdicts survive separately; only the comparison is refused.
        assert r.oracle.gpu_kernel_stats is not None
        assert oracle_speedup(r) is None

    def test_unchecked_correctness_does_not_suppress_the_comparison(self):
        """ "Not checked" is not "failed".

        A plain run records tolerance_match=None on the row. Treating that as
        a failure would suppress every speedup when --validate is absent.
        """
        factory, _ = _make_oracle_exec_factory()
        unchecked = CorrectnessResult(
            execution_success=True,
            tolerance_match=None,
            rtol=1e-5,
            atol=1e-5,
            error_message="No reference provider requested",
        )
        r = self._run(factory, [unchecked, self._verdict(True)]).results[0]

        assert oracle_speedup(r) == 2.0

    def test_passing_tuned_plan_still_reports_a_speedup(self):
        factory, _ = _make_oracle_exec_factory()
        r = self._run(factory, self._verdict(True)).results[0]

        assert r.oracle.correctness.passed is True
        assert oracle_speedup(r) == 2.0

    def test_a_failing_tuned_plan_leaves_the_ootb_verdict_passing(self):
        """OOTB correctness is the row's; the tuned verdict is the oracle's.

        The OOTB plan is correct and the tuned plan is not, so the two must
        disagree. A row that reports the tuned failure as its own would fail
        a passing engine.
        """
        factory, _ = _make_oracle_exec_factory()
        # First call is the OOTB check, second is the tuned check.
        r = self._run(factory, [self._verdict(True), self._verdict(False)]).results[0]

        assert r.correctness.passed is True
        assert r.oracle.correctness.passed is False
        assert r.status == "success"
        assert oracle_speedup(r) is None

    def test_tuned_plan_is_validated_after_its_timed_loop(self):
        """Validation must never land inside a measurement."""
        order = []
        factory, _ = _make_oracle_exec_factory(order=order)
        self._run(factory, self._verdict(True))

        tuned = [label for label in order if label.startswith("oracle.")]
        assert tuned[-2:] == ["oracle.benchmark", "oracle.execute_once"]

    def test_without_a_reference_no_tuned_verdict_is_recorded(self):
        """--validate off leaves the tuned check off too, and the speedup stands."""
        factory, instances = _make_oracle_exec_factory()
        r = TestOraclePass()._run(factory).results[0]

        assert r.oracle.correctness is None
        assert oracle_speedup(r) == 2.0
        # Only the sampling execute; no validation execute.
        assert instances[2].execute_once.call_count == 1


class TestOracleExhaustiveEnvGuard:
    """The tuned pass disables hipDNN disk caches and restores the environment."""

    _NAMES = ("HIPDNN_DISABLE_CACHE", "HIPDNN_FORCE_BENCHMARKING")

    @pytest.fixture(autouse=True)
    def _clean_env(self, monkeypatch: pytest.MonkeyPatch):
        for name in self._NAMES:
            monkeypatch.delenv(name, raising=False)

    def _run(self, factory):
        return TestOraclePass()._run(factory)

    def _snapshot(self):
        return {name: os.environ.get(name) for name in self._NAMES}

    def test_tuned_build_and_first_execute_run_with_the_cache_disabled(self):
        """A tuned winner written to the disk cache would serve later OOTB rows.

        Provider benchmarking comes from the plan's knob, so the process-wide
        HIPDNN_FORCE_BENCHMARKING must stay unset: it would also tune the OOTB
        re-time inside the same window.
        """
        seen = {}
        factory, instances = _make_oracle_exec_factory()

        def make_instance_with_capture(*args, **kwargs):
            m = factory(*args, **kwargs)
            if len(instances) == 3:
                m.prepare.side_effect = lambda *a, **k: seen.setdefault(
                    "build", self._snapshot()
                )
                m.warmup.side_effect = lambda *a, **k: seen.setdefault(
                    "first_execute", self._snapshot()
                )
            return m

        self._run(make_instance_with_capture)

        tuned_env = {"HIPDNN_DISABLE_CACHE": "1", "HIPDNN_FORCE_BENCHMARKING": None}
        assert seen == {"build": tuned_env, "first_execute": tuned_env}

    def test_env_is_restored_after_run(self) -> None:
        factory, _ = _make_oracle_exec_factory()
        self._run(factory)
        assert self._snapshot() == {name: None for name in self._NAMES}

    def test_preexisting_value_is_restored_not_deleted(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("HIPDNN_DISABLE_CACHE", "0")
        factory, _ = _make_oracle_exec_factory()
        self._run(factory)
        assert os.environ["HIPDNN_DISABLE_CACHE"] == "0"

    def test_exception_inside_guard_still_restores_env(self) -> None:
        factory, _ = _make_oracle_exec_factory(
            prepare_side_effect=ExecutionError("plan build failed")
        )
        result = self._run(factory)
        assert result.results[0].oracle_error == "ExecutionError: plan build failed"
        assert self._snapshot() == {name: None for name in self._NAMES}


def _tuned_child_document(*rows):
    """JSON document the tuned-PyTorch child writes to --output."""
    return {"graphs": [{"results": list(rows)}]}


_TUNED_CHILD_ROW = {
    "status": "success",
    "gpu_kernel_stats": BenchmarkStats.from_timings([0.25]).to_dict(),
    "host_stats": BenchmarkStats.from_timings([0.75]).to_dict(),
}


class TestPytorchOracle:
    """Tuned PyTorch runs in a child process; the parent only attaches its stats."""

    @staticmethod
    def _parse_child(argv):
        """Parse a child argv with the real CLI parser, as the child would."""
        from dnn_benchmarking.cli.parser import create_parser

        assert argv[:3] == [sys.executable, "-m", "dnn_benchmarking"]
        return create_parser().parse_args(argv[3:])

    def test_child_argv_is_a_non_recursive_run_with_the_parent_settings(self):
        config = _make_config(
            oracle_mode="exhaustive",
            pytorch_sdpa_backend="flash",
            pytorch_rocm_fa_library="ck",
            timing_block=4,
        )
        args = self._parse_child(
            _pytorch_tuned_argv(Path("g.json"), config, Path("out.json"))
        )

        # A child that inherited exhaustive would spawn a child of its own.
        assert args.oracle_mode == "off"
        assert args.internal_pytorch_tuned is True
        assert args.backend == "pytorch"
        assert args.graph == ["g.json"]
        assert args.output == Path("out.json")
        # One extra warmup absorbs the first-use tuning search, so the child
        # still runs as many ordinary warmups as the OOTB row.
        assert (args.warmup, args.iters, args.seed) == (3, 3, 42)
        # Both sides of the comparison must time the same block size.
        assert args.timing_block == 4
        assert args.pytorch_sdpa_backend == "flash"
        assert args.pytorch_rocm_fa_library == "ck"

    def test_child_argv_omits_unset_seed_and_fa_library(self):
        config = _make_config(oracle_mode="exhaustive", seed=None)
        args = self._parse_child(
            _pytorch_tuned_argv(Path("g.json"), config, Path("out.json"))
        )

        assert args.seed is None
        assert args.pytorch_rocm_fa_library is None
        assert args.pytorch_sdpa_backend == "default"

    @staticmethod
    def _run_child(document, returncode=0, stderr=""):
        """Run _run_pytorch_tuned_child against a fake child process.

        Returns the call's outcome (row or exception) and what the fake
        child observed.
        """
        seen = {}

        def fake_run_capped(argv, timeout_s, env=None):
            seen["env"] = env
            seen["state_dir_existed"] = os.path.isdir(env["MIOPEN_USER_DB_PATH"])
            if document is not None:
                output = Path(argv[argv.index("--output") + 1])
                output.write_text(json.dumps(document))
            return SimpleNamespace(returncode=returncode, stdout="", stderr=stderr)

        with patch(
            "dnn_benchmarking.execution.suite_runner.run_capped",
            side_effect=fake_run_capped,
        ):
            try:
                outcome = _run_pytorch_tuned_child(
                    Path("g.json"), _make_config(oracle_mode="exhaustive")
                )
            except Exception as e:
                outcome = e
        return outcome, seen

    def test_child_returns_its_success_row_and_discards_tuning_state(self):
        row, seen = self._run_child(_tuned_child_document(_TUNED_CHILD_ROW))

        assert row == _TUNED_CHILD_ROW
        env = seen["env"]
        state_dir = Path(env["MIOPEN_USER_DB_PATH"])
        assert seen["state_dir_existed"]
        assert Path(env["PYTORCH_TUNABLEOP_FILENAME"]).parent == state_dir
        # Tuning results left behind would be read by a later OOTB run.
        assert not state_dir.exists()

    @pytest.mark.parametrize(
        "document, returncode, stderr, message",
        [
            (_tuned_child_document(), 0, "", "returned 0 rows"),
            (
                _tuned_child_document(_TUNED_CHILD_ROW, _TUNED_CHILD_ROW),
                0,
                "",
                "returned 2 rows",
            ),
            (
                _tuned_child_document(
                    {"status": "error", "error_message": "HIP out of memory"}
                ),
                0,
                "",
                "HIP out of memory",
            ),
            (None, 1, "Traceback\nRuntimeError: boom", "exited 1: RuntimeError: boom"),
        ],
        ids=["no-rows", "several-rows", "failed-row", "no-output"],
    )
    def test_child_without_one_success_row_raises(
        self, document, returncode, stderr, message
    ):
        error, seen = self._run_child(document, returncode, stderr)

        assert isinstance(error, RuntimeError)
        assert message in str(error)
        assert not Path(seen["env"]["MIOPEN_USER_DB_PATH"]).exists()

    @staticmethod
    def _oracle_pass(child):
        """Run _run_pytorch_oracle_pass on a row whose OOTB kernel mean is 0.8.

        ``child`` is the tuned child's row, or the exception it raises.
        """
        result = ProviderEngineResult(
            provider="pytorch",
            engine_id=0,
            status="success",
            gpu_kernel_stats=BenchmarkStats.from_timings([0.8]),
            host_stats=BenchmarkStats.from_timings([1.5]),
            analytical_flops=10**9,
        )
        kwargs = (
            {"side_effect": child}
            if isinstance(child, Exception)
            else {"return_value": child}
        )
        with patch(
            "dnn_benchmarking.execution.suite_runner._run_pytorch_tuned_child",
            **kwargs,
        ):
            _run_pytorch_oracle_pass(
                result=result,
                graph_path=Path("g.json"),
                graph_name="g",
                config=_make_config(oracle_mode="exhaustive"),
            )
        return result

    def test_tuned_child_stats_are_compared_with_the_rows_own_ootb_run(self):
        """The tuned side is the child's; the OOTB side is the row, untouched."""
        result = self._oracle_pass(_TUNED_CHILD_ROW)

        oracle = result.oracle
        assert oracle.tuning_available is True
        assert oracle.gpu_kernel_stats.mean_ms == 0.25
        assert oracle.host_stats.mean_ms == 0.75
        assert result.oracle_error is None
        assert result.gpu_kernel_stats.mean_ms == 0.8
        assert oracle_speedup(result) == pytest.approx(3.2)
        # Same TFLOP/s basis as hipDNN oracles: 1e9 FLOPs over the tuned median.
        assert oracle.derived_tflops_per_s == pytest.approx(4.0)

    def test_child_failure_leaves_ootb_row_intact(self):
        result = self._oracle_pass(RuntimeError("tuned PyTorch child returned 0 rows"))

        assert result.oracle is None
        assert (
            result.oracle_error == "RuntimeError: tuned PyTorch child returned 0 rows"
        )
        assert result.status == "success"
        assert result.gpu_kernel_stats.mean_ms == 0.8
        assert result.host_stats.mean_ms == 1.5


def test_basic_metrics_use_kernel_median_and_per_execution_cpu_time():
    """derived_tflops_per_s divides by the kernel *median* (rocKE parity),
    and CPU time is per timed execution (iters * timing_block)."""
    result = ProviderEngineResult(provider="hipdnn", engine_id=1, status="success")
    # Mean 4 ms, median 1 ms.
    result.gpu_kernel_stats = BenchmarkStats.from_timings([1.0, 1.0, 10.0])
    probe = SimpleNamespace(
        delta=SimpleNamespace(user_time_ms=60.0, kernel_time_ms=6.0)
    )

    with patch("dnn_benchmarking.execution.suite_runner.GpuSmiProbe"):
        _collect_basic_metrics_post_loop(
            result=result,
            cpu_time_probe=probe,
            timed_executions=3 * 20,
            analytical_flops=10**12,
            analytical_flops_partial=False,
            analytical_io_bytes=None,
        )

    assert result.derived_tflops_per_s == pytest.approx(1000.0)
    assert result.cpu_user_time_per_iter_us == pytest.approx(1000.0)
    assert result.cpu_kernel_time_per_iter_us == pytest.approx(100.0)

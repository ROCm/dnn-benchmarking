# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for execution.suite_runner, asserted on the GraphResult it returns."""

import io
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from dnn_benchmarking.common.exceptions import ExecutionError, UnsupportedGraphError
from dnn_benchmarking.config.benchmark_config import (
    MetricsConfig,
    SuiteConfig,
    ValidationConfig,
)
from dnn_benchmarking.execution import suite_runner
from dnn_benchmarking.execution.timing import Measurement
from dnn_benchmarking.graph.tensor_info import TensorInfo
from dnn_benchmarking.metrics._diagnostic import reset as reset_warnings
from dnn_benchmarking.reporting.reporter import Reporter
from dnn_benchmarking.reporting.suite_results import graph_id_for

GRAPH = {"name": "g", "nodes": [], "tensors": []}
PATH = Path("g.json")
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


def _measurement(**kw):
    base = dict(
        kernel_ms=[1.0] * 10,
        host_ms=[0.01] * 10,
        mode="staged",
        backend="hip",
        cache_mode="warm",
        warmup_iters=2,
        first_call_ms=5.0,
    )
    return Measurement(**{**base, **kw})


class Fake:
    """Knobs shared by the fake executors, buffers and probes of one test."""

    def __init__(self):
        self.discovered = [1, 2]
        self.discover_error = None
        self.prepare_errors = {}
        self.bench_errors = {}
        self.measurement = {}
        self.engine_output = REF.copy()
        self.torch_error = None
        self.clocks = []


@pytest.fixture
def fake(monkeypatch):
    reset_warnings()
    f = Fake()

    class Executor:
        def __init__(self, graph_json_str, policy):
            self.init_time_ms, self.workspace_size = 2.0, 64

        def discover_engines(self, handle):
            if f.discover_error:
                raise f.discover_error
            return list(f.discovered)

        def prepare(self, handle, engine_id=None, for_autotune=False):
            self.engine_id = engine_id
            if engine_id in f.prepare_errors:
                raise f.prepare_errors[engine_id]

        def benchmark(self, handle, variant_pack):
            if self.engine_id in f.bench_errors:
                raise f.bench_errors[self.engine_id]
            return _measurement(**f.measurement)

        def execute_once(self, handle, variant_pack):
            pass

    class BufferManager:
        def __init__(self, tensor_infos, device=None):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def allocate_all(self):
            pass

        def load_input_data(self, data):
            pass

        def zero_outputs(self):
            pass

        def create_variant_pack(self):
            return {}

        def get_output_tensor(self, uid):
            return None

        def get_output_data(self, uid):
            return f.engine_output

    class Probe:
        def clocks(self):
            return f.clocks.pop(0) if f.clocks else None

        def snapshot(self):
            return {"vram_used_mb": 12.0}

    monkeypatch.setattr(suite_runner, "Executor", Executor)
    monkeypatch.setattr(suite_runner, "BufferManager", BufferManager)
    monkeypatch.setattr(suite_runner, "GpuSmiProbe", Probe)
    monkeypatch.setitem(
        sys.modules,
        "hipdnn_frontend",
        SimpleNamespace(
            engine_id_to_name=lambda eid: "",
        ),
    )
    return f


@pytest.fixture
def fake_torch(fake, monkeypatch):
    """PyTorch executor/buffers and a CPU reference provider."""
    pytest.importorskip("torch")
    from dnn_benchmarking.execution import pytorch_buffer_manager, pytorch_executor

    class TorchExecutor:
        def __init__(
            self,
            graph_json,
            policy,
            *,
            pytorch_sdpa_backend,
            pytorch_rocm_fa_library=None,
        ):
            self.init_time_ms = 1.0

        def prepare(self):
            if fake.torch_error:
                raise fake.torch_error

        def benchmark(self, tensors):
            return _measurement()

        def execute_once(self, tensors):
            pass

    class TorchBuffers:
        def __init__(self, tensor_infos):
            self._outputs = [t for t in tensor_infos if t.is_output]

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        allocate_all = load_input_data = zero_outputs = lambda self, *a: None

        def get_tensors(self):
            return {t.uid: SimpleNamespace(is_cuda=False) for t in self._outputs}

        def get_output_tensors(self):
            return self._outputs

        def get_output_data(self, uid):
            return REF.copy()

    class CpuReference:
        def compute_reference(self, graph_json, input_data):
            return {2: suite_runner.ReferenceOutput(data=REF.copy(), tensor_uid=2)}

    monkeypatch.setattr(pytorch_executor, "PyTorchCudaExecutor", TorchExecutor)
    monkeypatch.setattr(
        pytorch_buffer_manager, "PyTorchCudaBufferManager", TorchBuffers
    )
    monkeypatch.setattr(
        suite_runner,
        "_reference_provider",
        lambda config, graph_json: (CpuReference(), None),
    )
    return fake


def _handle(names=None):
    names = names or {1: "ENG_A", 2: "ENG_B"}
    return SimpleNamespace(
        get_engine_info=lambda eid: SimpleNamespace(
            engine_name=names.get(eid, ""), version="2.1"
        )
    )


def _run(handle="default", **config):
    out = io.StringIO()
    graph = suite_runner.run_graph_all_providers(
        PATH,
        GRAPH,
        TENSORS,
        SuiteConfig(**config),
        _handle() if handle == "default" else handle,
        Reporter(out),
    )
    return graph, out.getvalue()


def _validate(**config):
    return _run(validation=ValidationConfig(provider="pytorch"), **config)


def test_one_hipdnn_row_per_engine_named_from_engine_info(fake):
    graph, progress = _run()

    assert graph.graph_id == graph_id_for(GRAPH)
    assert graph.status == "ok" and graph.error is None
    assert [
        (r.provider, r.engine_id, r.engine_name, r.engine_version, r.verdict)
        for r in graph.results
    ] == [
        ("hipdnn", 1, "ENG_A", "2.1", "unchecked"),
        ("hipdnn", 2, "ENG_B", "2.1", "unchecked"),
    ]
    assert "ENG_A" in progress and "ENG_B" in progress


@pytest.mark.parametrize(
    "info_name, handle_name, registry_name, expected",
    [
        ("ENG_A", "H", "REG", "ENG_A"),
        # Plugin-supplied engines: only the handle knows the name.
        ("", "hipkernel:Attn", "", "hipkernel:Attn"),
        ("", IndexError, "REG", "REG"),
        ("", "", "", "0x0000000000000001"),
    ],
)
def test_engine_name_falls_back_through_handle_registry_then_hex(
    fake, monkeypatch, info_name, handle_name, registry_name, expected
):
    monkeypatch.setitem(
        sys.modules,
        "hipdnn_frontend",
        SimpleNamespace(engine_id_to_name=lambda eid: registry_name),
    )
    fake.discovered = [1]
    handle = _handle({1: info_name})

    def handle_lookup(eid):
        if handle_name is IndexError:
            raise IndexError(eid)
        return handle_name

    handle.engine_id_to_name = handle_lookup

    graph, progress = _run(handle=handle)

    assert graph.results[0].engine_name == expected
    assert "WARNING" not in progress


@pytest.mark.parametrize(
    "stage, error, status, message",
    [
        ("prepare", UnsupportedGraphError("no plan"), "skipped", "no plan"),
        ("prepare", ExecutionError("build"), "error", "ExecutionError: build"),
        ("bench", RuntimeError("hip"), "error", "RuntimeError: hip"),
    ],
)
def test_engine_failure_is_isolated_to_its_row(fake, stage, error, status, message):
    getattr(fake, f"{stage}_errors")[1] = error

    failed, ok = _run()[0].results

    assert (
        failed.status,
        failed.verdict,
        failed.error_message or failed.skip_reason,
    ) == (status, status, message)
    assert failed.engine_name == "ENG_A"
    assert failed.gpu_kernel_stats is None and failed.correctness is None
    assert ok.status == "success"


def test_row_timing_throughput_and_noise_from_the_measurement(fake, monkeypatch):
    monkeypatch.setattr(suite_runner, "compute_flops", lambda g: (2_000_000_000, False))
    fake.discovered = [1]
    # Median 1 ms, mean 1.4 ms: throughput must come from the median.
    fake.measurement = dict(
        kernel_ms=[1.0] * 9 + [5.0],
        capped=True,
        mode="events",
        fallback_reason="no stream wait",
    )

    row = _run()[0].results[0]

    assert row.gpu_kernel_stats.median_ms == pytest.approx(1.0)
    assert row.derived_tflops_per_s == pytest.approx(2.0)
    assert row.derived_gbytes_per_s == pytest.approx(32 / 1e-3 / 1e9)
    assert row.timing.mode == "events" and row.timing.first_call_ms == 5.0
    assert row.workspace_bytes == 64 and row.vram_used_mb == 12.0
    warnings = " | ".join(row.warnings)
    for expected in (
        "outlier: max 5.0x median",
        "capped at max_iters",
        "events timing: no stream wait",
    ):
        assert expected in warnings


def test_metrics_tier_off_skips_probes_and_throughput(fake, monkeypatch):
    monkeypatch.setattr(suite_runner, "compute_flops", lambda g: (2_000_000_000, False))
    fake.discovered = [1]
    fake.clocks = [{"sclk_mhz": 1700}] * 2

    row = _run(metrics=MetricsConfig(tier="off"))[0].results[0]

    assert row.gpu_kernel_stats is not None
    assert row.clocks_before is None and row.derived_tflops_per_s is None


CLOCK = {"sclk_mhz": 1700.0, "throttle_status": 0}


@pytest.mark.parametrize(
    "before, after, throttled",
    [
        (CLOCK, CLOCK, False),
        (CLOCK, {**CLOCK, "sclk_mhz": 1400.0}, False),  # DPM drop, not throttling
        (CLOCK, {**CLOCK, "throttle_status": 4}, True),
        (None, None, False),
    ],
)
def test_clocks_bracket_the_timed_loop(fake, before, after, throttled):
    fake.discovered = [1]
    fake.clocks = [before, after]

    row = _run()[0].results[0]

    assert (row.clocks_before, row.clocks_after) == (before, after)
    assert ("throttled" in row.warnings) is throttled


def test_unsupported_graph_has_no_engines_and_keeps_the_reason(fake):
    fake.discover_error = UnsupportedGraphError("nothing applies")

    graph, _ = _run()

    assert graph.status == "no_engines"
    assert graph.results == [] and graph.error is None
    assert graph.message == "nothing applies"


def test_unsupported_graph_still_gets_the_pytorch_reference_row(fake_torch):
    fake_torch.discover_error = UnsupportedGraphError("nothing applies")

    graph, _ = _validate()

    assert graph.status == "no_engines"
    assert [(r.provider, r.role, r.engine_id, r.verdict) for r in graph.results] == [
        ("pytorch", "reference", None, "reference")
    ]
    assert graph.results[0].gpu_kernel_stats is not None


@pytest.mark.parametrize(
    "patch, error",
    [
        ("discover", "Engine discovery failed: ExecutionError: driver"),
        ("inputs", "Input data generation failed: ValueError: bad dims"),
    ],
)
def test_graph_level_failure_sets_error_without_rows(fake, monkeypatch, patch, error):
    if patch == "discover":
        fake.discover_error = ExecutionError("driver")
    else:

        def fail(*a):
            raise ValueError("bad dims")

        monkeypatch.setattr(suite_runner, "generate_input_data", fail)

    graph, _ = _run()

    assert graph.status == "error" and graph.error == error
    assert graph.results == []


@pytest.mark.parametrize(
    "engine_output, verdict",
    [(REF.copy(), "passed"), (REF + 1.0, "failed")],
)
def test_engines_are_validated_against_the_timed_reference(
    fake_torch, engine_output, verdict
):
    fake_torch.discovered = [1]
    fake_torch.engine_output = engine_output

    reference, engine = _validate()[0].results

    assert (reference.role, reference.verdict) == ("reference", "reference")
    assert engine.verdict == verdict
    assert engine.correctness.n_total == 4


def test_missing_reference_fails_validation_with_the_reason(fake, monkeypatch):
    reason = "Reference provider 'pytorch' does not support this graph"
    monkeypatch.setattr(
        suite_runner, "_reference_provider", lambda c, g: (None, reason)
    )
    fake.discovered = [1]

    (engine,) = _validate()[0].results

    assert engine.verdict == "failed"
    assert engine.correctness.error_message == reason


@pytest.mark.parametrize(
    "sdpa_backend, reference_status, engine_verdict",
    [
        ("default", "skipped", "passed"),  # CPU reference serves as fallback
        ("math", "error", "failed"),  # strict selection never falls back
    ],
)
def test_failed_timed_reference(
    fake_torch, sdpa_backend, reference_status, engine_verdict
):
    fake_torch.discovered = [1]
    fake_torch.torch_error = ExecutionError("PyTorch GPU not available")

    reference, engine = _validate(pytorch_sdpa_backend=sdpa_backend)[0].results

    assert reference.status == reference_status
    assert (reference.error_message or reference.skip_reason) == (
        "ExecutionError: PyTorch GPU not available"
    )
    assert engine.verdict == engine_verdict


def test_per_engine_handle_failure_is_an_error_row(fake, monkeypatch):
    def no_handle():
        raise RuntimeError("plugin load failed")

    monkeypatch.setitem(
        sys.modules,
        "hipdnn_frontend",
        SimpleNamespace(
            PluginLoadingMode=SimpleNamespace(ABSOLUTE="abs"),
            set_engine_plugin_paths=lambda paths, mode: None,
            Handle=no_handle,
        ),
    )

    graph, _ = _run(
        handle=None, engine_filter=[1, 1], plugin_paths=[Path("/a"), Path("/b")]
    )

    assert [(r.status, r.plugin_path, r.error_message) for r in graph.results] == [
        ("error", str(Path("/a")), "RuntimeError: plugin load failed"),
        ("error", str(Path("/b")), "RuntimeError: plugin load failed"),
    ]


def test_profiling_payload_lands_on_the_row(fake, monkeypatch):
    """The child gets the timed run's seed (same inputs) and metric flags."""
    from dnn_benchmarking.metrics import profiling_orchestrator

    calls = []

    def run_profiling_passes(**kw):
        calls.append(kw)
        return {"perf": {"cycles": kw["engine_name"]}}

    monkeypatch.setattr(
        profiling_orchestrator, "run_profiling_passes", run_profiling_passes
    )
    fake.discovered = [1]
    metrics = MetricsConfig(perf=True, pmc_set="basic", profiling_timeout_s=42.0)

    graph, progress = _run(seed=7, metrics=metrics)

    assert graph.results[0].extra_metrics == {"perf": {"cycles": "ENG_A"}}
    assert "profiling ENG_A" in progress
    [kw] = calls
    assert kw["seed"] == 7
    assert kw["metrics_config"] == metrics


def test_profiling_failure_keeps_the_timed_row(fake, monkeypatch):
    from dnn_benchmarking.metrics import profiling_orchestrator

    def fail(**kw):
        raise FileNotFoundError("/proc/nope")

    monkeypatch.setattr(profiling_orchestrator, "run_profiling_passes", fail)

    graph, progress = _run(metrics=MetricsConfig(perf=True))

    assert [(r.status, r.engine_name) for r in graph.results] == [
        ("success", "ENG_A"),
        ("success", "ENG_B"),
    ]
    for row in graph.results:
        assert row.gpu_kernel_stats is not None and row.extra_metrics is None
        assert "profiling failed: FileNotFoundError: /proc/nope" in row.warnings
    # Every profiling progress line is completed.
    assert progress.count("profiling ENG_A ... done") == 1
    assert progress.count("profiling ENG_B ... done") == 1


class TestPytorchBackend:
    def _run(self, graph_json=GRAPH, **config):
        return suite_runner.run_graph_pytorch_backend(
            PATH,
            graph_json,
            TENSORS,
            SuiteConfig(backend="pytorch", **config),
            Reporter(io.StringIO()),
        )

    def test_single_timed_pytorch_row(self, fake_torch):
        graph = self._run()

        (row,) = graph.results
        assert graph.graph_id == graph_id_for(GRAPH) and graph.status == "ok"
        assert (
            row.provider,
            row.engine_id,
            row.engine_name,
            row.role,
            row.verdict,
        ) == ("pytorch", None, "pytorch", "engine", "unchecked")
        assert row.gpu_kernel_stats.median_ms == pytest.approx(1.0)
        assert row.timing.mode == "staged"

    def test_executor_failure_is_an_error_row(self, fake_torch):
        fake_torch.torch_error = ExecutionError("PyTorch GPU not available")

        (row,) = self._run().results

        assert row.status == "error"
        assert row.error_message == "ExecutionError: PyTorch GPU not available"

    @pytest.mark.parametrize(
        "sdpa_backend, status", [("default", "skipped"), ("math", "error")]
    )
    def test_unsupported_operations(self, fake_torch, sdpa_backend, status):
        graph_json = {**GRAPH, "nodes": [{"type": "NotARealOp"}]}

        (row,) = self._run(graph_json, pytorch_sdpa_backend=sdpa_backend).results

        assert row.status == status
        assert "NotARealOp" in (row.error_message or row.skip_reason)

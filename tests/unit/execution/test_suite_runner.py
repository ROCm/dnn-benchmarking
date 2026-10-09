# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for execution.suite_runner, asserted on the GraphResult it returns."""

import io
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from dnn_benchmarking.common.exceptions import ExecutionError, UnsupportedGraphError
from dnn_benchmarking.config.benchmark_config import (
    MetricsConfig,
    PyTorchSdpaBackendName,
    SuiteConfig,
    ValidationConfig,
)
from dnn_benchmarking.execution import oracle, suite_runner
from dnn_benchmarking.execution.timing import Measurement, StallFallbackError
from dnn_benchmarking.graph.tensor_info import TensorInfo
from dnn_benchmarking.metrics._diagnostic import reset as reset_warnings
from dnn_benchmarking.reporting.reporter import Reporter
from dnn_benchmarking.reporting.suite_results import graph_id_for, oracle_speedup

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
        timer="hip",
        cache_mode="warm",
        warmup_iters=2,
        first_call_ms=5.0,
    )
    return Measurement(**{**base, **kw})


def _stall_aware(f, key, policy):
    """Fail the stall gate for ``key`` while staging is allowed; once the
    runner disables it, time in events mode like ``timing.measure``."""
    if key in f.stall and policy.stall_gate:
        raise StallFallbackError("stall watchdog invalidated the measurement")
    if not policy.stall_gate:
        return _measurement(mode="events", fallback_reason="stall gate failed")
    return _measurement(**f.measurement)


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
        self.torch_execute_error = None
        self.clocks = []
        self.events = []
        self.torch_options = []
        self.torch_buffer_devices = []
        self.torch_outputs_written = False
        self.policies = []
        # (call, handle, HIPDNN_DISABLE_CACHE) per plan build/run, in order;
        # calls on the tuned plan are prefixed "tuned.".
        self.calls = []
        self.stall = set()  # engine ids (or 'pytorch') whose stall gate fails


@pytest.fixture
def fake(monkeypatch):
    reset_warnings()
    f = Fake()

    class Executor:
        def __init__(self, graph_json_str, policy):
            f.policies.append(policy)
            self.policy = policy
            self.tag = ""
            self.build_time_ms, self.workspace_size = 0.0, 0

        def _call(self, name, handle):
            f.calls.append(
                (self.tag + name, handle, os.environ.get("HIPDNN_DISABLE_CACHE"))
            )

        def discover_engines(self, handle):
            f.events.append("discover")
            if f.discover_error:
                raise f.discover_error
            return list(f.discovered)

        def prime(self, handle, engine_id):
            self._call("prime", handle)

        def prepare(self, handle, engine_id=None, knobs=None):
            self.tag = "tuned." if knobs else ""
            self._call("prepare", handle)
            self.engine_id = engine_id
            if engine_id in f.prepare_errors:
                raise f.prepare_errors[engine_id]
            # Distinct OOTB and tuned build times, so a row cannot report the
            # other plan's build.
            self.build_time_ms = 7.0 if knobs else 2.0
            self.workspace_size = 64

        def engine_knob_ids(self, engine_id):
            return ["global.benchmarking"]

        def benchmark(self, handle, variant_pack):
            self._call("benchmark", handle)
            f.events.append(("benchmark", self.engine_id, self.policy.stall_gate))
            if self.engine_id in f.bench_errors:
                raise f.bench_errors[self.engine_id]
            return _stall_aware(f, self.engine_id, self.policy)

        def execute_once(self, handle, variant_pack):
            self._call("execute_once", handle)
            f.events.append("execute_once")

        def __del__(self):
            f.events.append("executor_freed")

    class BufferManager:
        def __init__(self, tensor_infos, device=None):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            f.events.append("buffers_freed")
            return False

        def allocate_all(self):
            pass

        def load_input_data(self, data):
            pass

        def zero_outputs(self):
            f.events.append("zero_outputs")

        def create_variant_pack(self):
            return {}

        def get_output_tensor(self, uid):
            return None

        def get_output_data(self, uid):
            f.events.append("read_output")
            return f.engine_output

    class Probe:
        def clocks(self):
            return f.clocks.pop(0) if f.clocks else None

    monkeypatch.setattr(suite_runner, "Executor", Executor)
    monkeypatch.setattr(oracle, "Executor", Executor)
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
            fake.policies.append(policy)
            self.policy = policy
            fake.torch_options.append((pytorch_sdpa_backend, pytorch_rocm_fa_library))
            self.device = "cuda:3"  # not the buffer manager's cuda:0 default

        def prepare(self):
            if fake.torch_error:
                raise fake.torch_error

        def benchmark(self, tensors):
            fake.torch_outputs_written = True
            return _stall_aware(fake, "pytorch", self.policy)

        def execute_once(self, tensors):
            if fake.torch_execute_error:
                raise fake.torch_execute_error
            fake.torch_outputs_written = True

    class TorchBuffers:
        def __init__(self, tensor_infos, device):
            fake.torch_buffer_devices.append(device)
            self._outputs = [t for t in tensor_infos if t.is_output]

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            fake.events.append("torch_buffers_freed")
            return False

        allocate_all = load_input_data = lambda self, *a: None

        def zero_outputs(self):
            fake.torch_outputs_written = False

        def get_tensors(self):
            return {t.uid: SimpleNamespace(is_cuda=False) for t in self._outputs}

        def get_output_tensors(self):
            return self._outputs

        def get_output_data(self, uid):
            # Zeros unless the graph ran since the last zero_outputs().
            return REF.copy() if fake.torch_outputs_written else np.zeros_like(REF)

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


class _Handle:
    """hipdnn.Handle stand-in; the oracle builds a second one on this stream."""

    def __init__(self, names=None):
        self.names = names or {1: "ENG_A", 2: "ENG_B"}
        self.stream = None

    def get_engine_info(self, eid):
        return SimpleNamespace(engine_name=self.names.get(eid, ""), version="2.1")

    def get_stream(self):
        return self.stream

    def set_stream(self, stream):
        self.stream = stream


def _run(handle="default", **config):
    out = io.StringIO()
    graph = suite_runner.run_graph_all_providers(
        PATH,
        GRAPH,
        TENSORS,
        SuiteConfig(**config),
        _Handle() if handle == "default" else handle,
        Reporter(out),
    )
    return graph, out.getvalue()


def _validate(**config):
    return _run(validation=ValidationConfig(provider="pytorch"), **config)


@pytest.fixture
def reference(monkeypatch):
    """--validate with host reference outputs and no PyTorch row (no torch)."""

    def prepare(ctx):
        ctx.reference_outputs = {
            2: suite_runner.ReferenceOutput(data=REF.copy(), tensor_uid=2)
        }

    monkeypatch.setattr(suite_runner, "_prepare_references", prepare)


def test_one_hipdnn_row_per_engine_named_from_engine_info(fake):
    graph, progress = _run()

    assert graph.graph_id == graph_id_for(GRAPH)
    assert graph.status == "ok" and graph.error is None
    assert [
        (r.runtime, r.engine_id, r.engine_name, r.engine_version, r.verdict)
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
    handle = _Handle({1: info_name})

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
    assert failed.ootb.gpu_kernel_stats is None and failed.ootb.correctness is None
    assert ok.status == "success"


def test_row_timing_throughput_and_noise_from_the_measurement(fake, monkeypatch):
    monkeypatch.setattr(suite_runner, "compute_flops", lambda g: 2_000_000_000)
    fake.discovered = [1]
    # Median 1 ms, mean 1.4 ms: throughput must come from the median.
    # Every timing field off its default, so a row cannot claim a cache mode,
    # block size or cap its measurement did not use.
    fake.measurement = dict(
        kernel_ms=[1.0] * 9 + [5.0],
        capped=True,
        mode="events",
        timer="torch",
        cache_mode="cold",
        warmup_iters=7,
        timing_block=4,
        fallback_reason="no stream wait",
    )

    row = _run()[0].results[0]

    assert row.ootb.gpu_kernel_stats.median_ms == pytest.approx(1.0)
    assert row.ootb.host_stats.median_ms == pytest.approx(0.01)
    assert row.ootb.derived_tflops_per_s == pytest.approx(2.0)
    assert row.ootb.derived_gbytes_per_s == pytest.approx(32 / 1e-3 / 1e9)
    assert row.to_dict()["ootb"]["timing"] == {
        "mode": "events",
        "timer": "torch",
        "warmup_iters": 7,
        "samples": 10,  # the loop's sample count, beyond --iters when extended
        "first_call_ms": 5.0,
        "capped": True,
        "fallback_reason": "no stream wait",
    }
    assert row.ootb.workspace_bytes == 64
    assert row.ootb.cpu_build_time_ms == 2.0
    warnings = " | ".join(row.warnings)
    for expected in (
        "outlier: max 5.0x median",
        "capped at max_iters",
        "events timing: no stream wait",
    ):
        assert expected in warnings


def test_every_executor_gets_the_run_policy(fake):
    config = dict(warmup_iters=3, benchmark_iters=5, cache_mode="cold")

    _run(oracle=True, **config)

    # Discovery, then per engine the prime, the OOTB plan and the tuned plan.
    assert fake.policies == [SuiteConfig(**config).timing_policy] * 7


def test_oracle_builds_both_plans_before_anything_runs(fake, reference, monkeypatch):
    """Per engine: one untimed prime on the row handle absorbs the first
    build's one-time costs, then the timed OOTB and tuned builds run back to
    back (work in between slows a later build). The tuned plan is never
    primed, runs on its own handle with hipDNN caches disabled, and is timed
    only after the OOTB plan is timed and validated."""
    monkeypatch.delenv("HIPDNN_DISABLE_CACHE", raising=False)
    handle = _Handle()

    graph, _ = _validate(handle=handle, oracle=True)

    assert [(name, h is handle, cache) for name, h, cache in fake.calls] == [
        ("prime", True, None),
        ("prepare", True, None),
        ("tuned.prepare", False, "1"),
        ("benchmark", True, None),
        ("execute_once", True, None),
        ("tuned.benchmark", False, "1"),
        ("tuned.execute_once", False, "1"),
    ] * 2
    for row in graph.results:
        assert (row.ootb.cpu_build_time_ms, row.oracle.cpu_build_time_ms) == (2.0, 7.0)
        assert row.ootb.correctness.passed and row.oracle.correctness.passed


def test_tuned_plan_failing_validation_publishes_no_speedup(
    fake, reference, monkeypatch
):
    """A wrong-but-fast tuned plan is checked against the graph's reference:
    it keeps its own failed verdict and the OOTB row keeps its pass."""
    run_tuned_plan = suite_runner.run_tuned_plan

    def wrong_tuned_outputs(**kw):
        fake.engine_output = REF + 1.0
        run_tuned_plan(**kw)

    monkeypatch.setattr(suite_runner, "run_tuned_plan", wrong_tuned_outputs)

    graph, _ = _validate(oracle=True, engine_filter=[1])

    (row,) = graph.results
    assert (row.status, row.verdict) == ("success", "passed")
    assert row.oracle.correctness.explicitly_failed
    assert oracle_speedup(row) is None


def test_no_metrics_skips_probes_and_throughput(fake, monkeypatch):
    monkeypatch.setattr(suite_runner, "compute_flops", lambda g: 2_000_000_000)
    fake.discovered = [1]
    fake.clocks = [{"sclk_mhz": 1700}]

    row = _run(metrics=MetricsConfig(basic=False))[0].results[0]

    assert row.ootb.gpu_kernel_stats is not None
    assert row.clocks_after is None and row.ootb.derived_tflops_per_s is None


CLOCK = {"sclk_mhz": 1700.0, "throttle_status": 0}


@pytest.mark.parametrize(
    "after, throttled",
    [
        (CLOCK, False),
        ({**CLOCK, "sclk_mhz": 1400.0}, False),  # DPM drop, not throttling
        ({**CLOCK, "throttle_status": 4}, True),
        (None, False),
    ],
)
def test_clocks_after_the_timed_loop_flag_throttling(fake, after, throttled):
    fake.discovered = [1]
    fake.clocks = [after]

    row = _run()[0].results[0]

    assert row.clocks_after == after
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
    assert [(r.runtime, r.role, r.engine_id, r.verdict) for r in graph.results] == [
        ("pytorch", "reference", None, "reference")
    ]
    assert graph.results[0].ootb.gpu_kernel_stats is not None


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
    assert engine.ootb.correctness.n_total == 4


def test_missing_reference_fails_validation_with_the_reason(fake, monkeypatch):
    reason = "Reference provider 'pytorch' does not support this graph"
    monkeypatch.setattr(
        suite_runner, "_reference_provider", lambda c, g: (None, reason)
    )
    fake.discovered = [1]

    (engine,) = _validate()[0].results

    assert engine.verdict == "failed"
    assert engine.ootb.correctness.error_message == reason


@pytest.mark.parametrize(
    "sdpa_backend, reference_status, reference_reason, engine_verdict, engine_reason",
    [
        # CPU reference serves as fallback, and the row names the failed step
        (
            "default",
            "skipped",
            "Timing skipped (ExecutionError: PyTorch GPU not available).",
            "passed",
            None,
        ),
        # strict selection never falls back, and says why
        (
            "math",
            "error",
            "ExecutionError: PyTorch GPU not available",
            "failed",
            "ExecutionError: PyTorch GPU not available",
        ),
    ],
)
def test_failed_timed_reference(
    fake_torch,
    sdpa_backend,
    reference_status,
    reference_reason,
    engine_verdict,
    engine_reason,
):
    fake_torch.discovered = [1]
    fake_torch.torch_error = ExecutionError("PyTorch GPU not available")

    reference, engine = _validate(pytorch_sdpa_backend=sdpa_backend)[0].results

    assert reference.status == reference_status
    assert (reference.error_message or reference.skip_reason).startswith(
        reference_reason
    )
    assert engine.verdict == engine_verdict
    assert engine.ootb.correctness.error_message == engine_reason


def test_a_skipped_timed_reference_says_engines_are_still_graded(fake_torch):
    """The skipped row must not read as "nothing was validated"."""
    fake_torch.discovered = [1]
    fake_torch.torch_error = ExecutionError("PyTorch GPU not available")

    reference, engine = _validate()[0].results

    assert engine.verdict == "passed"
    assert "still validated" in reference.skip_reason
    assert "CPU" in reference.skip_reason


def test_a_failed_reference_pass_is_not_reported_as_skipped_timing(fake_torch):
    """Timing can finish and only the MATH output pass fail, out of memory on
    a large graph for instance. Saying "Timing skipped" there is wrong."""
    fake_torch.discovered = [1]
    fake_torch.torch_execute_error = RuntimeError("HIP out of memory")

    reference, engine = _validate()[0].results

    assert reference.status == "skipped"
    assert reference.skip_reason.startswith(
        "Reference output pass failed (RuntimeError: HIP out of memory)."
    )
    assert "Timing skipped" not in reference.skip_reason
    assert "still validated" in reference.skip_reason
    # The timed loop ran, so the row's own failure did not cost the grading.
    assert engine.verdict == "passed"


def test_reference_outputs_come_from_math_sdpa_and_timing_does_not(
    fake_torch, monkeypatch
):
    """Timing measures default dispatch; the graded outputs use MATH."""
    import torch

    from dnn_benchmarking.execution import pytorch_executor, pytorch_ops

    def math_only_at_sdpa():
        seen = []

        def recording_sdpa(*args, **kwargs):
            seen.append(
                torch.backends.cuda.math_sdp_enabled()
                and not torch.backends.cuda.flash_sdp_enabled()
            )
            return torch.empty(0)

        q = torch.rand(1, 1, 2, 4)
        with patch.object(
            torch.nn.functional, "scaled_dot_product_attention", recording_sdpa
        ):
            pytorch_ops.execute_selected_sdpa(
                q, q, q, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None
            )
        return seen[0]

    calls = {}
    executor_cls = pytorch_executor.PyTorchCudaExecutor
    timed_loop, output_pass = executor_cls.benchmark, executor_cls.execute_once

    def benchmark(self, tensors):
        calls["benchmark"] = math_only_at_sdpa()
        return timed_loop(self, tensors)

    def execute_once(self, tensors):
        calls["execute_once"] = math_only_at_sdpa()
        return output_pass(self, tensors)

    monkeypatch.setattr(executor_cls, "benchmark", benchmark)
    monkeypatch.setattr(executor_cls, "execute_once", execute_once)

    _validate()

    assert calls == {"benchmark": False, "execute_once": True}


def test_cpu_reference_failure_fails_validation_and_keeps_engine_rows(
    fake_torch, monkeypatch
):
    class FailingReference:
        def compute_reference(self, graph_json, input_data):
            raise RuntimeError("cpu reference exploded")

    monkeypatch.setattr(
        suite_runner, "_reference_provider", lambda c, g: (FailingReference(), None)
    )
    fake_torch.torch_error = ExecutionError("PyTorch GPU not available")

    graph, _ = _validate()

    assert graph.error is None
    reference, *engines = graph.results
    assert reference.status == "skipped"
    assert [
        (e.engine_id, e.verdict, e.ootb.correctness.error_message) for e in engines
    ] == [
        (1, "failed", "cpu reference exploded"),
        (2, "failed", "cpu reference exploded"),
    ]


def test_hipdnn_buffers_use_torch_storage_only_with_a_device_reference():
    host = suite_runner.ReferenceOutput(data=REF, tensor_uid=2)
    device = suite_runner.ReferenceOutput(data=REF, tensor_uid=3, device_data=REF)

    assert suite_runner._hipdnn_buffer_device(None) is None
    assert suite_runner._hipdnn_buffer_device({2: host}) is None
    assert suite_runner._hipdnn_buffer_device({2: host, 3: device}) == "cuda"


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


def test_per_engine_handle_loads_only_that_rows_plugin(fake, monkeypatch):
    """-e 1,1 --plugin-path /a,/b: each row gets a fresh handle created after
    its own plugin path is set, so an A/B row is not silently the default set."""
    calls = []

    def make_handle():
        calls.append("Handle")
        return _Handle({1: f"ENG_{len(calls)}"})

    monkeypatch.setitem(
        sys.modules,
        "hipdnn_frontend",
        SimpleNamespace(
            PluginLoadingMode=SimpleNamespace(ABSOLUTE="abs"),
            set_engine_plugin_paths=lambda paths, mode: calls.append((paths, mode)),
            Handle=make_handle,
        ),
    )
    a, b = Path("/a"), Path("/b")

    graph, _ = _run(handle=None, engine_filter=[1, 1], plugin_paths=[a, b])

    assert calls == [([str(a)], "abs"), "Handle", ([str(b)], "abs"), "Handle"]
    assert [(r.status, r.plugin_path, r.engine_name) for r in graph.results] == [
        ("success", str(a), "ENG_2"),
        ("success", str(b), "ENG_4"),
    ]


def test_default_run_skips_oracle_and_profiling(fake, monkeypatch):
    """Without --oracle or a profiling flag, no engine gets a tuned plan
    and no profiler child is spawned."""
    from dnn_benchmarking.metrics import profiling_orchestrator

    called = []
    for name in ("build_tuned_plan", "run_tuned_plan"):
        monkeypatch.setattr(
            suite_runner, name, lambda name=name, **kw: called.append(name)
        )
    monkeypatch.setattr(
        profiling_orchestrator,
        "run_profiling_passes",
        lambda **kw: called.append("profile"),
    )

    graph, _ = _run()

    assert called == []
    assert [(r.status, r.oracle, r.extra_metrics) for r in graph.results] == [
        ("success", None, None)
    ] * 2


def test_pytorch_run_options_reach_the_timed_executor(fake_torch):
    fake_torch.discovered = [1]
    config = dict(warmup_iters=3, benchmark_iters=5, cache_mode="cold")

    _validate(
        pytorch_sdpa_backend="flash", pytorch_rocm_fa_library="aotriton", **config
    )

    assert fake_torch.torch_options == [(PyTorchSdpaBackendName.FLASH, "aotriton")]
    # Discovery, the PyTorch reference row, then the engine's prime and plan.
    assert fake_torch.policies == [SuiteConfig(**config).timing_policy] * 4


def test_reference_warnings_go_on_the_reference_row_only(fake_torch, monkeypatch):
    from dnn_benchmarking.execution import pytorch_ops

    monkeypatch.setattr(
        pytorch_ops, "get_reference_warnings", lambda graph_json: ["manual op"]
    )
    fake_torch.discovered = [1]

    reference, engine = _validate()[0].results

    assert "manual op" in reference.warnings
    assert "manual op" not in (engine.warnings or [])


def test_profiling_runs_after_teardown_with_the_rows_payload(fake, monkeypatch):
    """The child gets the row's engine, plugin, seed (same inputs), warmup and
    metric flags, and runs only after the row's buffers and workspace are
    released (holding them would roughly double peak VRAM)."""
    from dnn_benchmarking.metrics import profiling_orchestrator

    calls = []

    def run_profiling_passes(**kw):
        fake.events.append("profile")
        calls.append(kw)
        return {"perf": {"cycles": kw["engine_name"]}}

    monkeypatch.setattr(
        profiling_orchestrator, "run_profiling_passes", run_profiling_passes
    )
    metrics = MetricsConfig(perf=True, pmc_set="basic", profiling_timeout_s=42.0)

    graph, progress = _run(
        seed=7,
        warmup_iters=3,
        metrics=metrics,
        engine_filter=[2],
        plugin_paths=[Path("/a")],
    )

    assert graph.results[0].extra_metrics == {"perf": {"cycles": "ENG_B"}}
    assert "profiling ENG_B" in progress
    assert calls == [
        dict(
            graph_path=PATH,
            engine_id=2,
            engine_name="ENG_B",
            seed=7,
            warmup_iters=3,
            metrics_config=metrics,
            plugin_path=Path("/a"),
        )
    ]
    assert fake.events[-3:] == ["buffers_freed", "executor_freed", "profile"]


def test_validation_reads_outputs_zeroed_after_the_timed_loop(fake, reference):
    """Outputs left by the timed loop must not satisfy the reference check."""
    _validate(engine_filter=[1])

    start = fake.events.index("zero_outputs")
    assert fake.events[start : start + 5] == [
        "zero_outputs",
        ("benchmark", 1, True),
        "zero_outputs",
        "execute_once",
        "read_output",
    ]


def test_explicit_engines_run_in_caller_order_without_discovery(fake):
    fake.discovered = [1, 2]

    graph, _ = _run(engine_filter=[2, 9, 1])

    assert graph.engine_ids == [2, 9, 1]
    assert [r.engine_id for r in graph.results] == [2, 9, 1]
    assert "discover" not in fake.events


def test_stall_failure_remeasures_every_engine_of_the_graph_unstalled(fake):
    """One engine's watchdog release must not leave a graph with staged and
    events rows side by side: every row is remeasured without stalling."""
    fake.stall = {2}

    graph, progress = _run()

    benchmarks = [e for e in fake.events if isinstance(e, tuple)]
    assert benchmarks == [
        ("benchmark", 1, True),
        ("benchmark", 2, True),
        ("benchmark", 1, False),
        ("benchmark", 2, False),
    ]
    assert [r.engine_id for r in graph.results] == [1, 2]
    assert [r.status for r in graph.results] == ["success", "success"]
    assert {r.ootb.timing.mode for r in graph.results} == {"events"}
    assert all("stall gate failed" in " ".join(r.warnings) for r in graph.results)
    assert "remeasuring every row of this graph without stalling" in progress


def test_pytorch_buffers_share_the_executor_device(fake_torch):
    """Buffers on another device than the executor would make every
    PyTorch row fail or silently copy across GPUs."""
    _validate()

    assert fake_torch.torch_buffer_devices == ["cuda:3"]


def test_stall_rerun_reports_no_ootb_for_an_engine_that_already_tuned(fake):
    """hipDNN keeps a tuned winner in memory and serves it to any later build
    of the graph: engine 1 tuned before engine 2 stalled, so its rerun OOTB
    would time the tuned kernel and must not be published."""
    fake.stall = {2}

    graph, _ = _run(oracle=True)

    e1, e2 = graph.results
    assert (e1.status, e1.to_dict()["ootb"]) == ("error", None)
    assert "already ran a tuned search" in e1.error_message
    # Engine 2 stalled in its OOTB loop, before its own search.
    assert e2.status == "success" and e2.oracle is not None


def test_stall_failure_on_the_reference_reruns_the_graph(fake_torch):
    fake_torch.stall = {"pytorch"}

    graph, _ = _validate()

    assert [r.role for r in graph.results] == ["reference", "engine", "engine"]
    assert {r.ootb.timing.mode for r in graph.results} == {"events"}
    assert [r.verdict for r in graph.results[1:]] == ["passed", "passed"]


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
        assert row.ootb.gpu_kernel_stats is not None and row.extra_metrics is None
        assert "profiling failed: FileNotFoundError: /proc/nope" in row.warnings
    # Every profiling progress line is completed.
    assert progress.count("profiling ENG_A ... done") == 1
    assert progress.count("profiling ENG_B ... done") == 1


@pytest.mark.parametrize(
    "provider, reason",
    [
        (None, "not registered"),
        (SimpleNamespace(is_available=lambda: False), "not available"),
        (
            SimpleNamespace(is_available=lambda: True, supports_graph=lambda g: False),
            "does not support this graph",
        ),
        (
            SimpleNamespace(is_available=lambda: True, supports_graph=lambda g: True),
            None,
        ),
    ],
)
def test_reference_provider_is_usable_or_says_why_not(monkeypatch, provider, reason):
    def get_provider(name):
        assert name == "pytorch"
        if provider is None:
            raise ValueError(name)
        return provider

    monkeypatch.setattr(
        suite_runner.ReferenceProviderRegistry, "get_provider", get_provider
    )
    config = SuiteConfig(validation=ValidationConfig(provider="pytorch"))

    got = suite_runner._reference_provider(config, GRAPH)

    expected = (
        (provider, None)
        if reason is None
        else (None, f"Reference provider 'pytorch' {reason}")
    )
    assert got == expected


class TestPytorchRuntime:
    def _run(self, graph_json=GRAPH, **config):
        return suite_runner.run_graph_pytorch(
            PATH,
            graph_json,
            TENSORS,
            SuiteConfig(runtime="pytorch", **config),
            Reporter(io.StringIO()),
        )

    def test_single_timed_pytorch_row(self, fake_torch):
        graph = self._run()

        (row,) = graph.results
        assert graph.graph_id == graph_id_for(GRAPH) and graph.status == "ok"
        assert (
            row.runtime,
            row.engine_id,
            row.engine_name,
            row.role,
            row.verdict,
        ) == ("pytorch", None, "pytorch", "engine", "unchecked")
        assert row.ootb.gpu_kernel_stats.median_ms == pytest.approx(1.0)
        assert row.ootb.timing.mode == "staged"
        assert row.ootb.cpu_build_time_ms is None  # PyTorch has no plan build

    @pytest.mark.parametrize(
        "oracle, events",
        [
            (False, ["torch_buffers_freed"]),
            # The child's allocations must not stack on the row's buffers.
            (True, ["torch_buffers_freed", "pytorch_tuned"]),
        ],
    )
    def test_oracle_tunes_once_after_the_buffers_are_released(
        self, fake_torch, monkeypatch, oracle, events
    ):
        rows = []

        def run_pytorch_tuned(*, row, graph_path, graph_json, graph_name, config):
            fake_torch.events.append("pytorch_tuned")
            rows.append((row, graph_path))

        monkeypatch.setattr(suite_runner, "run_pytorch_tuned", run_pytorch_tuned)

        (row,) = self._run(oracle=oracle).results

        assert fake_torch.events == events
        assert rows == [(row, PATH)] * (len(events) - 1)

    def test_stall_failure_remeasures_the_row_unstalled(self, fake_torch):
        fake_torch.stall = {"pytorch"}

        (row,) = self._run().results

        assert row.status == "success"
        assert row.ootb.timing.mode == "events"
        assert [p.stall_gate for p in fake_torch.policies] == [True, False]

    def test_executor_failure_is_an_error_row(self, fake_torch):
        fake_torch.torch_error = ExecutionError("PyTorch GPU not available")

        (row,) = self._run().results

        assert row.status == "error"
        assert row.error_message == "ExecutionError: PyTorch GPU not available"

    def test_engine_row_gets_no_reference_warnings(self, fake_torch, monkeypatch):
        from dnn_benchmarking.execution import pytorch_ops

        monkeypatch.setattr(
            pytorch_ops, "get_reference_warnings", lambda graph_json: ["manual op"]
        )

        (row,) = self._run().results

        assert row.role == "engine"
        assert "manual op" not in (row.warnings or [])

    @pytest.mark.parametrize(
        "sdpa_backend, status", [("default", "skipped"), ("math", "error")]
    )
    def test_unsupported_operations(self, fake_torch, sdpa_backend, status):
        graph_json = {**GRAPH, "nodes": [{"type": "NotARealOp"}]}

        (row,) = self._run(graph_json, pytorch_sdpa_backend=sdpa_backend).results

        assert row.status == status
        assert "NotARealOp" in (row.error_message or row.skip_reason)

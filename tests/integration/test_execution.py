# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""hipDNN graph execution, timing and validation on a real GPU."""

import io
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest

from dnn_benchmarking.common.exceptions import UnsupportedGraphError
from dnn_benchmarking.config import (
    MetricsConfig,
    SuiteConfig,
    TimingPolicy,
    ValidationConfig,
)
from dnn_benchmarking.execution import (
    BufferManager,
    Executor,
    correctness,
    suite_runner,
)
from dnn_benchmarking.execution.buffer_manager import generate_input_data
from dnn_benchmarking.execution.suite_runner import run_graph_all_providers
from dnn_benchmarking.graph import GraphLoader
from dnn_benchmarking.reporting.suite_results import GraphResult
from dnn_benchmarking.reporting.reporter import Reporter
from dnn_benchmarking.validation import compare
from tests.integration.conftest import load_graph

pytestmark = pytest.mark.gpu


def _validate_config() -> SuiteConfig:
    return SuiteConfig(
        warmup_iters=1,
        benchmark_iters=2,
        validation=ValidationConfig(provider="pytorch"),
        metrics=MetricsConfig(basic=False),
    )


def _assert_engines_match_reference(
    result: GraphResult, require_engine: bool = False
) -> None:
    """One timed reference row; every engine row passes or is skipped.

    An engine error always fails. No engine running the graph skips, unless
    ``require_engine`` (a graph the sample plugins are known to run).
    """
    assert result.error is None, result.error
    reference = [r for r in result.results if r.role == "reference"]
    assert [r.verdict for r in reference] == ["reference"], reference
    assert reference[0].runtime == "pytorch"
    assert reference[0].ootb.gpu_kernel_stats is not None
    assert reference[0].ootb.host_stats is not None

    engines = [r for r in result.results if r.role == "engine"]
    errors = [(r.engine_name, r.error_message) for r in engines if r.status == "error"]
    assert not errors, errors
    if not any(r.status == "success" for r in engines):
        reason = f"no hipDNN engine runs {result.graph_name}: " + "; ".join(
            [result.message or ""] + [r.skip_reason or "" for r in engines]
        )
        if require_engine:
            pytest.fail(reason)
        pytest.skip(reason)
    verdicts = {r.verdict for r in engines}
    assert "passed" in verdicts
    assert verdicts <= {"passed", "skipped"}, [
        (r.engine_name, r.verdict, r.error_message or r.ootb.correctness)
        for r in engines
    ]


@pytest.mark.parametrize(
    "policy",
    [
        TimingPolicy(warmup_iters=2, iters=5),
        TimingPolicy(warmup_iters=1, iters=3, min_time_ms=0.5, cache_mode="cold"),
    ],
    ids=["warm", "cold-min-time"],
)
def test_benchmark_measures_and_writes_output(
    hipdnn, sample_conv_fwd_json: Dict[str, Any], policy: TimingPolicy
) -> None:
    """benchmark() primes, times per policy, and the graph really ran."""
    tensor_infos = GraphLoader().extract_tensor_info(sample_conv_fwd_json)
    handle = hipdnn.Handle()
    executor = Executor(json.dumps(sample_conv_fwd_json), policy)
    executor.prepare(handle)
    assert executor.build_time_ms > 0

    with BufferManager(tensor_infos) as bm:
        bm.allocate_all()
        bm.load_input_data(generate_input_data(tensor_infos, seed=42))
        bm.zero_outputs()
        variant_pack = bm.create_variant_pack()
        assert all(variant_pack.values()), variant_pack
        m = executor.benchmark(handle, variant_pack)
        output = bm.get_output_data(0)

    assert len(m.kernel_ms) == len(m.host_ms) >= policy.iters
    assert all(t > 0 for t in m.kernel_ms + m.host_ms)
    if not m.capped:
        assert sum(m.kernel_ms) >= policy.min_time_ms
    assert m.timer == "hip"
    assert m.cache_mode == policy.cache_mode
    assert m.warmup_iters == policy.warmup_iters
    assert m.first_call_ms > 0
    # The default stall-gated mode must engage where the device supports it;
    # elsewhere events mode is legitimate only with a recorded reason.
    if hipdnn.hip_can_use_stream_wait_value():
        assert m.mode == "staged", m
    else:
        assert m.fallback_reason, m
    assert output.shape == (16, 16, 16, 16)
    assert not np.allclose(output, 0)


def test_compare_matches_pytorch_reference(
    hipdnn, torch_gpu, sample_conv_fwd_json: Dict[str, Any]
) -> None:
    """Executor output decoded from device buffers matches the PyTorch reference."""
    from dnn_benchmarking.validation import ReferenceProviderRegistry

    provider = ReferenceProviderRegistry.get_provider("pytorch")
    tensor_infos = GraphLoader().extract_tensor_info(sample_conv_fwd_json)
    inputs = generate_input_data(tensor_infos, seed=42)
    handle = hipdnn.Handle()
    executor = Executor(json.dumps(sample_conv_fwd_json), TimingPolicy())
    executor.prepare(handle)

    with BufferManager(tensor_infos) as bm:
        bm.allocate_all()
        bm.load_input_data(inputs)
        bm.zero_outputs()
        executor.execute_once(handle, bm.create_variant_pack())
        output = bm.get_output_data(0)

    reference = provider.compute_reference(sample_conv_fwd_json, inputs)[0].data
    # The product gate: the dtype tolerance `--validate pytorch` applies.
    output_info = next(t for t in tensor_infos if t.is_output)
    rtol, atol = correctness.tolerance_for(SuiteConfig(), output_info)
    result = compare(output, reference, rtol=rtol, atol=atol)
    assert result.passed, result.message


@pytest.mark.parametrize(
    "graph_name, require_engine",
    [
        ("sample_conv_fwd.json", True),
        ("sample_batchnorm.json", True),
        ("sample_matmul.json", False),
        ("sample_relu.json", False),
        ("sample_add.json", False),
        ("sample_sdpa.json", False),
        ("sample_mha_sdpa.json", False),
    ],
)
def test_engines_validate_against_pytorch(
    hipdnn, torch_gpu, graph_name: str, require_engine: bool
) -> None:
    """--validate pytorch: every engine that runs a sample graph passes."""
    path, graph_json, tensor_infos = load_graph(graph_name)
    result = run_graph_all_providers(
        path,
        graph_json,
        tensor_infos,
        _validate_config(),
        hipdnn.Handle(),
        Reporter(output=io.StringIO()),
    )
    _assert_engines_match_reference(result, require_engine)


def test_paged_sdpa_sample_passes_hipdnn_graph_validation(hipdnn) -> None:
    """hipDNN deserializes, validates and builds sample_sdpa_paged.json.

    No engine has to run it: "no engines" (UnsupportedGraphError) is fine, a
    graph-build ExecutionError is not.
    """
    _, graph_json, _ = load_graph("sample_sdpa_paged.json")
    try:
        Executor(json.dumps(graph_json), TimingPolicy()).discover_engines(
            hipdnn.Handle()
        )
    except UnsupportedGraphError:
        pass


def test_bfloat16_conv_validates_against_pytorch(
    hipdnn, torch_gpu, sample_conv_fwd_json: Dict[str, Any]
) -> None:
    """BF16 device buffers decode and validate against a BF16 reference."""
    graph_json = json.loads(json.dumps(sample_conv_fwd_json))
    graph_json["name"] = "sample_conv_fwd_bfloat16"
    graph_json["io_data_type"] = "bfloat16"
    graph_json["intermediate_data_type"] = "bfloat16"
    for tensor in graph_json["tensors"]:
        tensor["data_type"] = "bfloat16"

    result = run_graph_all_providers(
        Path("graph_bfloat16.json"),
        graph_json,
        GraphLoader().extract_tensor_info(graph_json),
        _validate_config(),
        hipdnn.Handle(),
        Reporter(output=io.StringIO()),
    )
    _assert_engines_match_reference(result)


def test_validation_compares_on_device(
    hipdnn,
    torch_gpu,
    sample_conv_fwd_json: Dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With a device reference, engine buffers live in torch and compare on GPU."""
    compared = []
    real_compare = correctness.compare

    def spy(actual, expected, **kwargs):
        compared.append(type(actual).__module__)
        return real_compare(actual, expected, **kwargs)

    monkeypatch.setattr(correctness, "compare", spy)
    devices = []
    pick = suite_runner._hipdnn_buffer_device
    monkeypatch.setattr(
        suite_runner,
        "_hipdnn_buffer_device",
        lambda refs: devices.append(pick(refs)) or devices[-1],
    )

    result = run_graph_all_providers(
        Path("graph.json"),
        sample_conv_fwd_json,
        GraphLoader().extract_tensor_info(sample_conv_fwd_json),
        _validate_config(),
        hipdnn.Handle(),
        Reporter(output=io.StringIO()),
    )

    _assert_engines_match_reference(result, require_engine=True)
    assert set(devices) == {"cuda"}
    assert compared and set(compared) == {"torch"}


@pytest.mark.parametrize("dtype_name", ["float16", "float32"])
def test_large_output_comparison_vram_budget(torch_gpu, dtype_name: str) -> None:
    """compare() on a 64M-element device output stays within 13 bytes per element.

    Three float32 temporaries (|a - e|, |e|, threshold) plus one bool mask
    are needed. Any redundant full-size copy exceeds the budget.
    """
    torch = torch_gpu
    dtype = getattr(torch, dtype_name)
    n = 1 << 26
    expected = torch.randn(n, device="cuda", dtype=dtype)
    actual = expected.clone()
    actual[-1] += 1.0

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    result = compare(actual, expected, rtol=1e-3, atol=1e-3)
    peak = torch.cuda.max_memory_allocated() - base

    assert result.passed is False
    assert result.n_mismatch == 1
    assert result.max_abs_diff >= 0.5
    # 3 x 4-byte temporaries + 1-byte mask per element, plus 1 MiB for the
    # scalar results of max() and all() (512-byte allocator blocks).
    budget = 13 * n + (1 << 20)
    assert peak <= budget, f"peak {peak / 2**20:.0f} MiB > {budget / 2**20:.0f} MiB"


class TestHardEngineSelectBindings:
    """Real-backend coverage for the hard-select / read-back Graph bindings.

    Drives a live Graph through create_execution_plan_ext() -> build_plans() ->
    get_execution_plan_engine_id(), exercising the actual nanobind surface and
    the C++ getter (which the executor unit tests stub out).
    """

    @staticmethod
    def _built_graph(graph_json_str: str, handle):
        """A Graph with the operation graph built (no execution plan yet)."""
        executor = Executor(graph_json_str, TimingPolicy())
        executor._build_through_operation_graph(handle)
        return executor._graph

    def test_hard_select_and_read_back_matches(
        self, hipdnn, sample_conv_fwd_json: Dict[str, Any]
    ) -> None:
        """Hard-selecting a ranked engine builds, and the read-back reports it."""
        handle = hipdnn.Handle()
        graph_json_str = json.dumps(sample_conv_fwd_json)

        discovery_graph = self._built_graph(graph_json_str, handle)
        ranked = [int(e) for e in discovery_graph.get_ranked_engine_ids()]
        assert ranked, "expected at least one ranked engine for the graph"
        engine = ranked[0]

        graph = self._built_graph(graph_json_str, handle)
        result = graph.create_execution_plan_ext(engine)
        assert not result.is_bad(), result.get_message()
        assert not graph.build_plans().is_bad()
        assert graph.get_execution_plan_engine_id() == engine

    def test_hard_select_inapplicable_engine_is_bad(
        self, hipdnn, sample_conv_fwd_json: Dict[str, Any]
    ) -> None:
        """Hard-selecting an engine id the backend cannot honor returns is_bad()."""
        handle = hipdnn.Handle()
        graph = self._built_graph(json.dumps(sample_conv_fwd_json), handle)
        assert graph.create_execution_plan_ext(123456789).is_bad()

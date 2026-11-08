# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""PyTorch executor and buffer manager on a live GPU (ROCm or CUDA).

The timer differs per platform (HIP events on ROCm, torch.cuda
events on CUDA), so assertions use ``expected_timer()``.
"""

from collections import Counter

import numpy as np
import pytest

from dnn_benchmarking.config import PyTorchSdpaBackendName, TimingPolicy
from dnn_benchmarking.graph import GraphLoader
from dnn_benchmarking.execution.buffer_manager import generate_input_data
from dnn_benchmarking.execution.pytorch_buffer_manager import PyTorchCudaBufferManager
from dnn_benchmarking.execution.pytorch_executor import PyTorchCudaExecutor
from tests.conftest import expected_timer
from tests.integration.conftest import load_graph

pytestmark = pytest.mark.gpu

POLICY = TimingPolicy(warmup_iters=1, iters=2)


def _benchmark(graph_name: str, policy: TimingPolicy = POLICY, **executor_kwargs):
    """Prepare, load seeded inputs, and time one sample graph."""
    _, graph_json, tensor_infos = load_graph(graph_name)
    executor_kwargs.setdefault("pytorch_sdpa_backend", PyTorchSdpaBackendName.DEFAULT)
    executor = PyTorchCudaExecutor(graph_json, policy, **executor_kwargs)
    executor.prepare()

    with PyTorchCudaBufferManager(tensor_infos) as bm:
        bm.allocate_all()
        bm.load_input_data(generate_input_data(tensor_infos, 42, graph_json))
        bm.zero_outputs()
        return executor.benchmark(bm.get_tensors())


def test_load_input_data_copies_inputs_to_device(torch_gpu) -> None:
    """Loaded device tensors hold exactly the generated host inputs."""
    _, graph_json, tensor_infos = load_graph("sample_conv_fwd.json")
    inputs = generate_input_data(tensor_infos, seed=123)

    with PyTorchCudaBufferManager(tensor_infos) as bm:
        bm.allocate_all()
        bm.load_input_data(inputs)
        tensors = bm.get_tensors()
        assert all(t.is_cuda for t in tensors.values())
        for uid, expected in inputs.items():
            np.testing.assert_array_equal(tensors[uid].cpu().numpy(), expected)


@pytest.mark.parametrize(
    "graph_name",
    [
        "sample_conv_fwd.json",
        "sample_conv_dgrad.json",
        "sample_conv_wgrad.json",
        "sample_matmul.json",
        "sample_matmul_batched.json",
        "sample_matmul_broadcast.json",
        "sample_relu.json",
        "sample_add.json",
        "sample_batchnorm.json",
        "sample_batchnorm_training.json",
        "sample_batchnorm_inference.json",
        "sample_batchnorm_inference_variance.json",
        "sample_batchnorm_backward.json",
        "sample_sdpa.json",
        "sample_mha_sdpa.json",
        "sample_sdpa_backward.json",
        "sample_sdpa_paged.json",
        "sample_layernorm.json",
        "sample_rmsnorm.json",
        "sample_rmsnorm_backward.json",
        "sample_reduction.json",
        "sample_resample_fwd.json",
    ],
)
# Cold runs the cache flush beside the PyTorch executor's stream: the HIP
# flush buffer on ROCm, the torch one on CUDA.
@pytest.mark.parametrize("cache_mode", ["warm", "cold"])
def test_benchmark_times_every_iteration(
    torch_gpu, graph_name: str, cache_mode: str
) -> None:
    """Every supported sample graph yields one positive timing per iteration."""
    policy = TimingPolicy(warmup_iters=1, iters=2, cache_mode=cache_mode)
    m = _benchmark(graph_name, policy)
    assert len(m.kernel_ms) == len(m.host_ms) == policy.iters
    assert all(t > 0 for t in m.kernel_ms + m.host_ms)
    # The host-sync probe may add one priming enqueue; it is counted honestly.
    assert m.warmup_iters >= policy.warmup_iters
    assert m.timer == expected_timer()
    assert m.cache_mode == cache_mode
    assert m.mode == "staged" or m.fallback_reason, m


_NO_COPY = {"aten::copy_", "aten::convolution"}


@pytest.mark.parametrize(
    "graph_name, asymmetric, channels_last, forbidden",
    [
        pytest.param("sample_conv_dgrad.json", False, False, _NO_COPY, id="dgrad"),
        pytest.param("sample_conv_dgrad.json", True, False, _NO_COPY, id="dgrad-asym"),
        pytest.param("sample_conv_dgrad.json", False, True, _NO_COPY, id="dgrad-nhwc"),
        pytest.param(
            "sample_conv_dgrad.json", True, True, _NO_COPY, id="dgrad-asym-nhwc"
        ),
        # Asymmetric wgrad must pad x, and that pad copies x.
        pytest.param(
            "sample_conv_wgrad.json",
            True,
            False,
            {"aten::convolution"},
            id="wgrad-asym",
        ),
    ],
)
def test_conv_backward_iteration_runs_only_the_backward_conv(
    torch_gpu,
    monkeypatch,
    graph_name: str,
    asymmetric: bool,
    channels_last: bool,
    forbidden: set,
) -> None:
    """A timed dgrad or wgrad iteration runs one backward conv and no extra work.

    A copy or a forward conv in the iteration adds to the PyTorch time and
    gives a false hipDNN speedup.
    """
    _, graph_json, _ = load_graph(graph_name)
    if asymmetric:
        # Same output size as the symmetric [1, 1] padding of the sample.
        params = graph_json["nodes"][0]["parameters"]
        params["pre_padding"], params["post_padding"] = [0, 0], [2, 2]
    if channels_last:
        # NHWC on every tensor, as in the graphs of the reported defect. The
        # MIOpen path keeps NHWC only with this flag, which the CLI sets.
        monkeypatch.setenv("PYTORCH_MIOPEN_SUGGEST_NHWC", "1")
        for tensor in graph_json["tensors"]:
            _, c, h, w = tensor["dims"]
            tensor["strides"] = [h * w * c, 1, w * c, c]
    tensor_infos = GraphLoader().extract_tensor_info(graph_json)
    executor = PyTorchCudaExecutor(
        graph_json, POLICY, pytorch_sdpa_backend=PyTorchSdpaBackendName.DEFAULT
    )
    executor.prepare()

    profiler = torch_gpu.profiler
    with PyTorchCudaBufferManager(tensor_infos) as bm:
        bm.allocate_all()
        bm.load_input_data(generate_input_data(tensor_infos, 42, graph_json))
        tensors = bm.get_tensors()
        # The first call does one-time work (algorithm find, lazy init).
        executor.execute_once(tensors)
        with profiler.profile(activities=[profiler.ProfilerActivity.CPU]) as prof:
            executor.execute_once(tensors)

    ops = Counter(event.name for event in prof.events())
    assert ops["aten::convolution_backward"] == 1, ops
    assert not forbidden & ops.keys(), ops


def test_flash_aotriton_preference_benchmarks_sdpa(torch_gpu) -> None:
    """Strict flash selection with an AOTriton preference runs natively."""
    from dnn_benchmarking.common import torch_support

    if not torch_support.is_rocm_build():
        pytest.skip("AOTriton SDPA selection is ROCm-only")

    m = _benchmark(
        "sample_sdpa.json",
        pytorch_sdpa_backend=PyTorchSdpaBackendName.FLASH,
        pytorch_rocm_fa_library="aotriton",
    )
    assert m.kernel_ms[0] > 0 and m.host_ms[0] > 0


def test_gqa_math_fallback_is_skipped_not_timed(torch_gpu) -> None:
    """A GQA call no fused kernel accepts would run math, which copies K/V to
    the query head count inside the timed call. The benchmark refuses it; the
    one-shot reference still computes it.

    A sliding window forces this on every arch: Flash takes no mask, and
    Efficient needs equal Q/K/V head counts. If a torch release lifts either
    limit, this test fails and the methodology note needs an update.
    """
    from dnn_benchmarking.common.exceptions import UnsupportedGraphError

    torch = torch_gpu
    graph_json = {
        "nodes": [
            {
                "type": "SdpaAttributes",
                "inputs": {"q_tensor_uid": 1, "k_tensor_uid": 2, "v_tensor_uid": 3},
                "outputs": {"o_tensor_uid": 4},
                "attributes": {
                    "attn_scale_value": 0.125,
                    "left_bound": 15,
                    "right_bound": 0,
                },
            }
        ]
    }
    tensors = {
        uid: torch.randn(1, heads, 128, 64, device="cuda", dtype=torch.bfloat16)
        for uid, heads in ((1, 8), (2, 2), (3, 2))
    }
    executor = PyTorchCudaExecutor(
        graph_json, POLICY, pytorch_sdpa_backend=PyTorchSdpaBackendName.DEFAULT
    )
    executor.prepare()

    with pytest.raises(UnsupportedGraphError, match="math backend"):
        executor.benchmark(tensors)
    executor.execute_once(tensors)
    assert tensors[4].shape == (1, 8, 128, 64)


@pytest.mark.parametrize("q_heads", [8, 2])
def test_sdpa_stats_come_from_fused_kernel(torch_gpu, q_heads: int) -> None:
    """The stats output is the log-sum-exp the fused kernel saved for backward.

    Guards its layout on this torch build: query length 100 is not a multiple
    of any kernel block, so a padded or transposed log-sum-exp shows up here.
    """
    torch = torch_gpu
    graph_json = {
        "nodes": [
            {
                "type": "SdpaAttributes",
                "inputs": {"q_tensor_uid": 1, "k_tensor_uid": 2, "v_tensor_uid": 3},
                "outputs": {"o_tensor_uid": 4, "stats_tensor_uid": 5},
                "attributes": {"attn_scale_value": 0.125, "causal_mask": True},
            }
        ]
    }
    torch.manual_seed(0)
    tensors = {
        uid: torch.randn(2, heads, 100, 64, device="cuda", dtype=torch.bfloat16)
        for uid, heads in ((1, q_heads), (2, 2), (3, 2))
    }
    executor = PyTorchCudaExecutor(
        graph_json, POLICY, pytorch_sdpa_backend=PyTorchSdpaBackendName.DEFAULT
    )
    executor.prepare()
    executor.benchmark(tensors)
    executor.execute_once(tensors)

    k = tensors[2].float().repeat_interleave(q_heads // 2, dim=-3)
    scores = (tensors[1].float() @ k.transpose(-2, -1)) * 0.125
    keep = torch.ones(100, 100, dtype=torch.bool, device="cuda").tril()
    expected = torch.logsumexp(
        scores.masked_fill(~keep, float("-inf")), dim=-1, keepdim=True
    )
    # Inputs are bf16 (about 3 significant digits) and the reference is fp32,
    # so the kernel's log-sum-exp matches it to about 1e-3, not to fp32 rounding.
    torch.testing.assert_close(tensors[5].float(), expected, rtol=1e-3, atol=1e-3)
    assert executor.notes == [], "stats fell back to a separate score matmul"


def _gqa_graph(**attributes) -> dict:
    return {
        "nodes": [
            {
                "type": "SdpaAttributes",
                "inputs": {"q_tensor_uid": 1, "k_tensor_uid": 2, "v_tensor_uid": 3},
                "outputs": {"o_tensor_uid": 4},
                "attributes": {"attn_scale_value": 0.125, **attributes},
            }
        ]
    }


def _gqa_tensors(torch, q_len: int, kv_len: int, kv_heads: int = 2):
    return {
        1: torch.randn(2, 8, q_len, 64, device="cuda", dtype=torch.bfloat16),
        2: torch.randn(2, kv_heads, kv_len, 64, device="cuda", dtype=torch.bfloat16),
        3: torch.randn(2, kv_heads, kv_len, 64, device="cuda", dtype=torch.bfloat16),
    }


def test_bottom_right_gqa_decode_is_timed_by_default(torch_gpu) -> None:
    """One query over a long cache with bottom-right causal bounds needs no
    mask, so default dispatch keeps a fused kernel and the row is timed."""
    executor = PyTorchCudaExecutor(
        _gqa_graph(causal_mask_bottom_right=True),
        POLICY,
        pytorch_sdpa_backend=PyTorchSdpaBackendName.DEFAULT,
    )
    executor.prepare()

    m = executor.benchmark(_gqa_tensors(torch_gpu, 1, 256))
    assert m.kernel_ms[0] > 0


def test_strict_efficient_runs_gqa_graph(torch_gpu) -> None:
    """Efficient rejects enable_gqa, so a strict efficient row gets K/V
    expanded up front, as before enable_gqa was used."""
    from dnn_benchmarking.execution.pytorch_ops import (
        PyTorchSdpaBackendUnavailableError,
    )

    def run(kv_heads: int) -> None:
        executor = PyTorchCudaExecutor(
            _gqa_graph(),
            POLICY,
            pytorch_sdpa_backend=PyTorchSdpaBackendName.EFFICIENT,
        )
        executor.prepare()
        executor.benchmark(_gqa_tensors(torch_gpu, 64, 64, kv_heads))

    try:
        run(8)
    except PyTorchSdpaBackendUnavailableError:
        pytest.skip("strict efficient attention is unavailable on this build")
    run(2)


def test_pytorch_row_returns_vram_after_oom(torch_gpu) -> None:
    """A PyTorch row that runs out of memory must not keep VRAM reserved.

    hipDNN rows of the same graph allocate with hipMalloc, outside PyTorch's
    caching allocator, so memory PyTorch keeps cached is lost to them. The
    graph chains two 1 GiB intermediates under a cap that fits only one, so
    the row fails while the first is still referenced.
    """
    import io
    from pathlib import Path

    from dnn_benchmarking.config import SuiteConfig
    from dnn_benchmarking.execution import suite_runner
    from dnn_benchmarking.graph import GraphLoader
    from dnn_benchmarking.reporting.reporter import Reporter

    torch = torch_gpu
    m, k = 16384, 256  # an M x M fp32 tensor is 1 GiB

    def tensor(uid, dims, virtual=False):
        return {
            "uid": uid,
            "name": f"t{uid}",
            "dims": dims,
            "strides": [dims[1], 1],
            "data_type": "float",
            "virtual": virtual,
        }

    def relu(name, x, y):
        return {
            "name": name,
            "type": "PointwiseAttributes",
            "compute_data_type": "float",
            "inputs": {"operation": "relu_fwd", "in_0_tensor_uid": x},
            "outputs": {"out_0_tensor_uid": y},
        }

    graph_json = {
        "name": "oom_chain",
        "compute_data_type": "float",
        "io_data_type": "float",
        "intermediate_data_type": "float",
        "tensors": [
            tensor(1, [m, k]),
            tensor(2, [k, m]),
            tensor(3, [m, m], virtual=True),
            tensor(4, [m, m], virtual=True),
            tensor(5, [m, m]),
        ],
        "nodes": [
            {
                "name": "mm",
                "type": "MatmulAttributes",
                "compute_data_type": "float",
                "inputs": {"a_tensor_uid": 1, "b_tensor_uid": 2},
                "outputs": {"c_tensor_uid": 3},
            },
            relu("relu1", 3, 4),
            relu("relu2", 4, 5),
        ],
    }
    infos = GraphLoader().extract_tensor_info(graph_json)
    ctx = suite_runner._graph_context(
        Path("oom_chain.json"),
        graph_json,
        infos,
        SuiteConfig(runtime="pytorch"),
        Reporter(io.StringIO()),
    )

    torch.cuda.empty_cache()
    reserved_before = torch.cuda.memory_reserved()
    total = torch.cuda.get_device_properties(0).total_memory
    # Inputs and output (~1.1 GiB), one intermediate, and some slack.
    cap = reserved_before + 2.6 * (1 << 30)
    torch.cuda.set_per_process_memory_fraction(cap / total)
    try:
        row, _ = suite_runner._run_pytorch_row(ctx, "engine")
    finally:
        torch.cuda.set_per_process_memory_fraction(1.0)

    assert "out of memory" in (row.error_message or ""), row
    assert torch.cuda.memory_reserved() <= reserved_before
    assert "at row start the GPU had" in row.error_message


@pytest.mark.parametrize(
    "graph_name",
    ["sample_conv_fwd.json", "sample_conv_dgrad.json", "sample_conv_wgrad.json"],
)
def test_default_conv_reference_is_float64_on_the_gpu(torch_gpu, graph_name) -> None:
    """With the default config, a convolution graph is validated against a
    float64 reference computed on the GPU, and that reference equals a
    float64 computation on the host."""
    import io
    from pathlib import Path

    from dnn_benchmarking.config import SuiteConfig, ValidationConfig
    from dnn_benchmarking.execution import suite_runner
    from dnn_benchmarking.reporting.reporter import Reporter
    from dnn_benchmarking.validation import ReferenceProviderRegistry
    from dnn_benchmarking.validation.reference_provider import ReferencePrecision

    path, graph_json, tensor_infos = load_graph(graph_name)
    config = SuiteConfig(validation=ValidationConfig(provider="pytorch"))
    ctx = suite_runner._graph_context(
        Path(path), graph_json, tensor_infos, config, Reporter(io.StringIO())
    )
    row = suite_runner._prepare_references(ctx)

    assert row is not None and row.status == "success", row
    assert not any("float64 reference not used" in w for w in row.warnings or [])
    assert ctx.reference_outputs
    host = ReferenceProviderRegistry.get_provider("pytorch").compute_reference(
        graph_json, ctx.input_data, precision=ReferencePrecision.FLOAT64
    )
    for uid, ref in ctx.reference_outputs.items():
        assert ref.precision == ReferencePrecision.FLOAT64.value
        assert ref.device_data is not None and ref.device_data.is_cuda
        assert ref.device_data.dtype == torch_gpu.float64
        np.testing.assert_allclose(
            ref.host_data(), host[uid].host_data(), rtol=1e-12, atol=0
        )

# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""PyTorch executor and buffer manager on a live GPU (ROCm or CUDA).

The timer differs per platform (HIP events on ROCm, torch.cuda
events on CUDA), so assertions use ``expected_timer()``.
"""

import numpy as np
import pytest

from dnn_benchmarking.config import PyTorchSdpaBackendName, TimingPolicy
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
    assert executor.init_time_ms > 0

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

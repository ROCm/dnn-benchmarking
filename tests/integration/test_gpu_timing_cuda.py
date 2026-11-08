# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""PyTorch executor timing and environment reporting on NVIDIA CUDA hosts.

CUDA-specific: asserts torch.cuda event timing (never HIP), the non-ROCm
environment sentinels, and CUDA/cuDNN labels (not ROCm) in the suite header.
"""

import io
import re

import pytest

from dnn_benchmarking.config import PyTorchSdpaBackendName, TimingPolicy
from dnn_benchmarking.execution.buffer_manager import generate_input_data
from dnn_benchmarking.execution.pytorch_buffer_manager import PyTorchCudaBufferManager
from dnn_benchmarking.execution.pytorch_executor import PyTorchCudaExecutor
from dnn_benchmarking.metrics.machine_info import collect_environment_info
from dnn_benchmarking.reporting.reporter import Reporter
from tests.conftest import skip_if_no_cuda_torch
from tests.integration.conftest import load_graph

pytestmark = [pytest.mark.gpu, pytest.mark.cuda]


@pytest.mark.parametrize(
    "graph_name",
    [
        "sample_conv_fwd.json",
        # Batchnorm must take native_batch_norm, not MIOpen, on CUDA builds.
        "sample_batchnorm_training.json",
        "sample_batchnorm_backward.json",
    ],
)
def test_pytorch_gpu_timing_cuda(graph_name: str) -> None:
    """The PyTorch executor times with torch.cuda events on CUDA."""
    skip_if_no_cuda_torch()
    _, graph_json, tensor_infos = load_graph(graph_name)
    executor = PyTorchCudaExecutor(
        graph_json,
        TimingPolicy(warmup_iters=1, iters=3),
        pytorch_sdpa_backend=PyTorchSdpaBackendName.DEFAULT,
    )
    executor.prepare()

    with PyTorchCudaBufferManager(tensor_infos) as bm:
        bm.allocate_all()
        bm.load_input_data(generate_input_data(tensor_infos, 42, graph_json))
        bm.zero_outputs()
        m = executor.benchmark(bm.get_tensors())

    assert len(m.kernel_ms) == len(m.host_ms) == 3
    assert all(t > 0.0 for t in m.kernel_ms + m.host_ms)
    assert m.timer == "torch"


def test_cuda_environment_metadata_sentinels() -> None:
    """ROCm-specific metadata carries its non-ROCm sentinels on a CUDA host."""
    skip_if_no_cuda_torch()
    info = collect_environment_info()

    assert info["rocm_version"] is None
    assert info["gpu_arch"] == "unknown"
    assert isinstance(info["cuda_version"], str) and info["cuda_version"]
    assert info["cudnn_version"] is None or re.fullmatch(
        r"\d+\.\d+\.\d+", info["cudnn_version"]
    )


def test_cuda_suite_header_shows_cuda_label_not_rocm() -> None:
    """On a CUDA host the suite header prints CUDA (and cuDNN), never ROCm."""
    skip_if_no_cuda_torch()
    output = io.StringIO()
    run_config = {
        "warmup_iters": 1,
        "iters": 1,
        "min_time_ms": 0.0,
        "cache_mode": "warm",
        "seed": 0,
        "runtime": "pytorch",
    }
    Reporter(output=output).print_suite_header(
        collect_environment_info(), run_config, 1
    )
    out = output.getvalue()

    assert "ROCm:" not in out
    assert re.search(r"^CUDA:\s+\S+", out, re.MULTILINE)

# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""PyTorch GPU executor for graph benchmarking."""

from typing import Any, Dict, Optional

import torch

from ..common import torch_support
from ..common.exceptions import ExecutionError, UnsupportedGraphError
from ..config.benchmark_config import TimingPolicy
from . import pytorch_ops
from .timing import Measurement, Timer, is_hip_available, measure


class PyTorchCudaExecutor:
    """Executes hipDNN-format graphs with PyTorch on one GPU stream.

    Timing goes through ``timing.measure`` like the hipDNN executor: HIP
    events on ROCm torch (when hipdnn_frontend is available), torch.cuda
    events otherwise.
    """

    def __init__(
        self,
        graph_json: Dict[str, Any],
        policy: TimingPolicy,
        *,
        pytorch_sdpa_backend: Any,
        pytorch_rocm_fa_library: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        """Initialize executor with graph JSON and timing policy.

        Args:
            graph_json: The graph as a parsed JSON dictionary.
            policy: How ``benchmark`` warms up and samples.
            pytorch_sdpa_backend: Strict SDPA backend selection.
            pytorch_rocm_fa_library: Preferred ROCm flash-attention library.
            device: CUDA/ROCm device (e.g. ``cuda:0``); None = current device.

        Raises:
            ExecutionError: If PyTorch GPU is not available.
        """
        if not torch_support.gpu_available():
            raise ExecutionError(
                "PyTorch GPU not available. Install PyTorch with CUDA or ROCm support."
            )

        self._graph_json = graph_json
        self._policy = policy
        self._sdpa_backend_state = pytorch_ops.PyTorchSdpaBackendState(
            pytorch_sdpa_backend, pytorch_rocm_fa_library
        )
        self._device = torch.device(device if device is not None else "cuda")
        # ROCm torch shares the HIP runtime, so HIP events (and staging) can
        # bracket its stream; CUDA torch must use torch.cuda events.
        self._backend = (
            "hip"
            if torch_support.is_rocm_build() and is_hip_available()
            else "torch"
        )
        self._init_time_ms: float = 0.0
        self._stream: Optional[Any] = None
        self._compiled: Optional[pytorch_ops.CompiledGraph] = None

    def prepare(self) -> None:
        """Validate and compile the graph and pin the execution stream.

        Raises:
            ExecutionError: If graph contains unsupported operations.
        """
        with Timer() as t:
            unsupported = pytorch_ops.get_unsupported_operations(self._graph_json)
            if unsupported:
                raise ExecutionError(
                    f"Graph contains unsupported operations: {unsupported}. "
                    f"Supported: {list(pytorch_ops.get_supported_operations())}"
                )

            self._compiled = pytorch_ops.compile_graph(self._graph_json)

            with torch.cuda.device(self._device):
                torch.cuda.init()
                self._stream = torch.cuda.default_stream(self._device)

        self._init_time_ms = t.elapsed_ms

    def execute_once(self, tensors: Dict[int, torch.Tensor]) -> None:
        """Execute the graph once and synchronize.

        Used after timed loops to collect clean reference outputs without
        including output zeroing or extraction in benchmark timings.
        """
        stream = self._get_stream()
        with torch.cuda.device(self._device), torch.cuda.stream(stream):
            self._execute_graph(tensors)
            stream.synchronize()

    def benchmark(self, tensors: Dict[int, torch.Tensor]) -> Measurement:
        """Prime and time the graph per the executor's policy.

        Inputs are wrapped in ``ReplayTensors`` so constant host reads
        (epsilon/scale ``.item()``, paged SDPA lengths) resolve once during
        priming and every timed iteration is a pure asynchronous enqueue.

        Args:
            tensors: Mapping of tensor UIDs to CUDA tensors.

        Raises:
            ExecutionError: If executor not prepared or execution fails.
        """
        stream = self._get_stream()
        replay = pytorch_ops.ReplayTensors(tensors)
        # One stream context around the whole loop: entering it per iteration
        # would put its host cost inside the submit bracket.
        with torch.cuda.device(self._device), torch.cuda.stream(stream):
            return measure(
                lambda: self._execute_graph(replay),
                stream=int(stream.cuda_stream),
                policy=self._policy,
                backend=self._backend,
                torch_stream=stream,
            )

    def _get_stream(self) -> Any:
        """Return the PyTorch stream used by all graph execution."""
        if self._stream is None:
            raise ExecutionError("Executor not prepared. Call prepare() first.")
        return self._stream

    def _execute_graph(self, tensors: Dict[int, torch.Tensor]) -> None:
        """Execute all graph operations in order.

        Raises:
            ExecutionError: If execution fails.
        """
        try:
            assert self._compiled is not None
            with pytorch_ops.use_pytorch_sdpa_backend(self._sdpa_backend_state):
                self._compiled.execute(tensors)
        except (UnsupportedGraphError, pytorch_ops.PyTorchSdpaBackendUnavailableError):
            raise
        except Exception as e:
            raise ExecutionError(f"Graph execution failed: {e}") from e

    @property
    def init_time_ms(self) -> float:
        """Get graph initialization time in milliseconds."""
        return self._init_time_ms

    @property
    def device(self) -> torch.device:
        """Get the CUDA/ROCm device being used."""
        return self._device

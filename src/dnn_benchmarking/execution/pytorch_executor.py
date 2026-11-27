# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""PyTorch GPU executor for graph benchmarking."""

from typing import Any, Dict, List, Optional, Sequence

import torch
from ..common import torch_support
from ..common.exceptions import ExecutionError, UnsupportedGraphError
from ..config.benchmark_config import TimingPolicy
from . import pytorch_dispatch, pytorch_ops
from .timing import Measurement, is_hip_available, measure


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
        self._timer = (
            "hip" if torch_support.is_rocm_build() and is_hip_available() else "torch"
        )
        self._stream: Optional[Any] = None
        self._compiled: Optional[pytorch_ops.CompiledGraph] = None
        # Row warnings that handlers found while priming the last benchmark().
        self.notes: List[str] = []
        # The last benchmark()'s primed inputs, for capture_dispatch().
        self._replay: Optional[pytorch_ops.ReplayTensors] = None

    def prepare(self) -> None:
        """Validate and compile the graph and pin the execution stream.

        Raises:
            ExecutionError: If graph contains unsupported operations.
        """
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

    def execute_once(self, tensors: Dict[int, torch.Tensor]) -> None:
        """Execute the graph once and synchronize.

        Used after timed loops to collect clean reference outputs without
        including output zeroing or extraction in benchmark timings.
        """
        stream = self._get_stream()
        with torch.cuda.device(self._device), torch.cuda.stream(stream):
            self._execute_graph(tensors)
            stream.synchronize()

    def benchmark(
        self,
        tensors: Dict[int, torch.Tensor],
        *,
        tensors_ring: Optional[Sequence[Dict[int, torch.Tensor]]] = None,
        rotated_bytes: Optional[int] = None,
    ) -> Measurement:
        """Prime and time the graph per the executor's policy.

        Inputs are wrapped in ``ReplayTensors`` so constant host reads
        (epsilon/scale ``.item()``, paged SDPA lengths) resolve once during
        priming and every timed iteration is a pure asynchronous enqueue.
        When ``tensors_ring`` holds multiple independent buffer sets
        (``--cache-mode rotate``), each launch rotates to the next set.
        """
        stream = self._get_stream()
        if tensors_ring and len(tensors_ring) > 1:
            replays = tuple(pytorch_ops.ReplayTensors(t) for t in tensors_ring)
            active_replay = replays[0]
            self.notes = active_replay.notes
            self._replay = active_replay
            idx = 0

            def enqueue_rotating() -> None:
                nonlocal idx
                r = replays[idx]
                idx = (idx + 1) % len(replays)
                self._execute_graph(r)

            enqueue_fn = enqueue_rotating
            n_rot = len(replays)
        else:
            active_replay = pytorch_ops.ReplayTensors(tensors)
            self.notes = active_replay.notes
            self._replay = active_replay
            enqueue_fn = lambda: self._execute_graph(active_replay)
            n_rot = 1

        # One stream context around the whole loop: entering it per iteration
        # would put its host cost inside the submit bracket.
        def prime_remaining_ring() -> None:
            # Prime sets 1..N-1 after the cold first call on set 0 so
            # first_call_ms still reflects one-time kernel/runtime setup while
            # every ReplayTensors populates its scalar/length host-read cache
            # before _probe_host_sync and the timed loop run.
            for extra_replay in replays[1:]:
                self._execute_graph(extra_replay)
            stream.synchronize()

        with torch.cuda.device(self._device), torch.cuda.stream(stream):
            measurement = measure(
                enqueue_fn,
                stream=int(stream.cuda_stream),
                policy=self._policy,
                timer=self._timer,
                torch_stream=stream,
                post_first_call=prime_remaining_ring if n_rot > 1 else None,
                rotated_buffers=n_rot,
                rotated_bytes=rotated_bytes if n_rot > 1 else None,
            )
            work_notes = getattr(active_replay, "timed_work_notes", None)
            if work_notes:
                measurement.timed_work_notes = list(work_notes)
            return measurement

    def capture_dispatch(self) -> Dict[str, Any]:
        """Profile one untimed launch of the timed path; return what it ran.

        The launch replays the inputs that the last ``benchmark()`` primed, so
        handler setup (fused-kernel probes, layout copies, host reads) does
        not run again. The device is drained first, so no input-fill, warmup
        or cache-flush kernel is in the trace. Call it once, after
        ``benchmark()``: it releases the primed inputs.

        Returns:
            The plan's ``dispatch`` object: ``sdpa_backend``, ``fa_library``
            (Flash on ROCm only) and ``kernels``, the distinct GPU kernel
            names in order of first launch.

        Raises:
            ExecutionError: If ``benchmark()`` did not run first.
        """
        stream = self._get_stream()
        replay, self._replay = self._replay, None
        if replay is None:
            raise ExecutionError("capture_dispatch() needs a benchmark() first.")

        def launch() -> None:
            self._execute_graph(replay)
            stream.synchronize()

        with torch.cuda.device(self._device), torch.cuda.stream(stream):
            torch.cuda.synchronize()
            events = pytorch_dispatch.profile_launch(launch)
        backend = pytorch_dispatch.sdpa_backend(events)
        flash = backend is not None and "flash" in backend.split(",")
        return {
            "sdpa_backend": backend,
            "fa_library": (
                pytorch_ops.rocm_fa_library(self._sdpa_backend_state) if flash else None
            ),
            "kernels": pytorch_dispatch.kernel_names(events),
        }

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
    def device(self) -> torch.device:
        """Get the CUDA/ROCm device being used."""
        return self._device

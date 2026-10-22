# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""rocKE executor: build one rocKE SDPA kernel and launch it without hipDNN.

The executor launches through rocKE's own runtime (``rocke.runtime``) on the
torch default stream, and times through ``timing.measure`` like the other
executors. Timed launches run under ``rocke.runtime.no_fence``: a fenced
rocKE launch synchronizes the host after every call.
"""

import hashlib
from typing import Callable, Dict, List, Optional

import torch

from ..common import torch_support
from ..common.exceptions import ExecutionError
from ..config.benchmark_config import TimingPolicy
from .rocke_sdpa import DENSE_ARCH, AutoKernel, CatalogKernel, RockeKernel, SdpaProblem
from .timing import Measurement, Timer, is_hip_available, measure


def device_arch(device: torch.device) -> str:
    """GPU architecture name without feature flags (``gfx950``)."""
    return str(torch.cuda.get_device_properties(device).gcnArchName).split(":")[0]


def device_num_cus(device: torch.device) -> int:
    """Compute-unit count, which rocKE's dispatcher uses to size grids."""
    return int(torch.cuda.get_device_properties(device).multi_processor_count)


class RockeExecutor:
    """Launch one rocKE kernel on graph tensors.

    ``build_time_ms`` is the host time to build and load the kernel. For the
    auto kernel it also includes the first launch, because rocKE compiles the
    unified kernel inside its launch call. ``warnings`` holds row warnings.
    """

    def __init__(
        self,
        kernel: RockeKernel,
        problem: SdpaProblem,
        policy: TimingPolicy,
        device: Optional[torch.device] = None,
    ) -> None:
        self._kernel = kernel
        self._problem = problem
        self._policy = policy
        self._device = device if device is not None else torch.device("cuda")
        # Same choice as the PyTorch executor: rocKE and ROCm torch share the
        # HIP runtime, so HIP events can bracket the torch stream.
        self._timer = (
            "hip" if torch_support.is_rocm_build() and is_hip_available() else "torch"
        )
        self._stream = torch.cuda.default_stream(self._device)
        self._launch: Optional[Callable[[], None]] = None
        self.build_time_ms: Optional[float] = None
        self.warnings: List[str] = []

    def prepare(self, tensors: Dict[int, torch.Tensor]) -> None:
        """Build the kernel and bind it to the graph tensors.

        Raises:
            ExecutionError: The build, the load or the first launch fails.
        """
        p = self._problem
        # Graph order is [B, H, S, D] over packed BSHD memory; the permuted
        # views are the contiguous [B, S, H, D] tensors rocKE kernels read.
        q, k, v, o = (
            tensors[uid].permute(0, 2, 1, 3)
            for uid in (p.q_uid, p.k_uid, p.v_uid, p.o_uid)
        )
        if not all(t.is_contiguous() for t in (q, k, v, o)):
            raise ExecutionError("rocKE runtime needs packed BSHD Q/K/V/O tensors")
        try:
            if isinstance(self._kernel, CatalogKernel):
                self._prepare_catalog(q, k, v, o)
            else:
                self._prepare_auto(q, k, v, o)
        except ExecutionError:
            raise
        except Exception as e:
            raise ExecutionError(f"rocKE build failed: {type(e).__name__}: {e}") from e

    def _prepare_catalog(self, q, k, v, o) -> None:
        from kernels.gfx950.attention_dense import (
            Gfx950AttentionDenseSpec,
            attention_dense_block,
            attention_dense_grid,
            attention_dense_signature,
            build_attention_dense,
        )
        from rocke.helpers.compile import compile_kernel
        from rocke.runtime import KernelLauncher, LaunchConfig

        kernel = self._kernel
        assert isinstance(kernel, CatalogKernel)
        p = self._problem
        with Timer() as t:
            spec = Gfx950AttentionDenseSpec(**kernel.spec)
            # backend="python" is the lowering hipDNN's pack step pins.
            art = compile_kernel(
                build_attention_dense(spec, arch=DENSE_ARCH),
                arch=DENSE_ARCH,
                backend="python",
                capture_ir_text=False,
            )
            launcher = KernelLauncher(
                hsaco=art.hsaco,
                kernel_name=art.kernel_name,
                signature=attention_dense_signature(spec),
            )
        self.build_time_ms = t.elapsed_ms

        built = hashlib.sha256(art.hsaco).hexdigest()
        if kernel.shipped_sha256 is None:
            self.warnings.append(
                "hipDNN code-object hash unavailable; parity not checked"
            )
        elif built != kernel.shipped_sha256:
            self.warnings.append(
                f"code object differs from hipDNN's (rocKE {built[:12]}, hipDNN "
                f"{kernel.shipped_sha256[:12]}): rocKE revision or comgr differs"
            )

        values = {
            "q_ptr": q.data_ptr(),
            "k_ptr": k.data_ptr(),
            "v_ptr": v.data_ptr(),
            "o_ptr": o.data_ptr(),
            "scale": p.scale,
            "batch": p.batch,
            "seqlen_q": p.seqlen_q,
            "seqlen_kv": p.seqlen_kv,
        }
        config = LaunchConfig(
            stream=int(self._stream.cuda_stream),
            grid=attention_dense_grid(spec),
            block=attention_dense_block(spec),
        )
        self._launch = lambda: launcher(values, config=config)

    def _prepare_auto(self, q, k, v, o) -> None:
        from kernels import UnifiedAttentionProblem, run_unified_attention_torch

        kernel = self._kernel
        assert isinstance(kernel, AutoKernel)
        p = self._problem
        bs = kernel.block_size
        blocks_per_seq = p.seqlen_kv // bs
        device = q.device
        problem = UnifiedAttentionProblem(
            total_q=p.batch * p.seqlen_q,
            num_seqs=p.batch,
            num_query_heads=p.num_query_heads,
            num_kv_heads=p.num_kv_heads,
            head_size=p.head_size,
            block_size=bs,
            max_seqlen_q=p.seqlen_q,
            max_seqlen_k=p.seqlen_kv,
            dtype=p.dtype,
            num_cus=device_num_cus(self._device),
        )
        # Packed BSHD K/V is a paged cache whose pages are in order: page j of
        # sequence b is block b * blocks_per_seq + j. No copy is needed.
        q_tokens = q.view(p.batch * p.seqlen_q, p.num_query_heads, p.head_size)
        o_tokens = o.view(p.batch * p.seqlen_q, p.num_query_heads, p.head_size)
        k_cache = k.view(p.batch * blocks_per_seq, bs, p.num_kv_heads, p.head_size)
        v_cache = v.view(p.batch * blocks_per_seq, bs, p.num_kv_heads, p.head_size)
        cu_seqlens_q = torch.arange(
            0, (p.batch + 1) * p.seqlen_q, p.seqlen_q, dtype=torch.int32, device=device
        )
        seqused_k = torch.full(
            (p.batch,), p.seqlen_kv, dtype=torch.int32, device=device
        )
        block_table = torch.arange(
            p.batch * blocks_per_seq, dtype=torch.int32, device=device
        ).view(p.batch, blocks_per_seq)
        # Force the dispatcher's path: "auto" would fall back to rocKE's
        # scalar correctness kernel when the tiled path declines.
        backend = "tiled" if kernel.path == "2d" else "3d"
        stream = int(self._stream.cuda_stream)

        def launch() -> None:
            run_unified_attention_torch(
                problem=problem,
                q=q_tokens,
                k=k_cache,
                v=v_cache,
                out=o_tokens,
                cu_seqlens_q=cu_seqlens_q,
                seqused_k=seqused_k,
                softmax_scale=p.scale,
                block_table=block_table,
                softcap=0.0,
                backend=backend,
                stream=stream,
            )

        with Timer() as t, torch.cuda.device(self._device):
            launch()
            self._release()
        self.build_time_ms = t.elapsed_ms
        self._launch = launch

    def execute_once(self) -> None:
        """Launch the kernel once and wait for it."""
        launch = self._require_prepared()
        with torch.cuda.device(self._device):
            launch()
            self._release()

    def benchmark(self) -> Measurement:
        """Time the kernel per the executor's policy."""
        from rocke.runtime import no_fence

        launch = self._require_prepared()
        try:
            with (
                torch.cuda.device(self._device),
                torch.cuda.stream(self._stream),
                no_fence(),
            ):
                return measure(
                    launch,
                    stream=int(self._stream.cuda_stream),
                    policy=self._policy,
                    timer=self._timer,
                    torch_stream=self._stream,
                )
        finally:
            self._release()

    def _release(self) -> None:
        """Drain the stream and free the kernel arguments rocKE retains."""
        from rocke.runtime import wait_stream_and_release

        wait_stream_and_release(int(self._stream.cuda_stream))

    @property
    def device(self) -> torch.device:
        """The GPU the executor launches on."""
        return self._device

    def _require_prepared(self) -> Callable[[], None]:
        if self._launch is None:
            raise ExecutionError("Executor not prepared. Call prepare() first.")
        return self._launch

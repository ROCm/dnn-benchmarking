# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""rocKE executor: load one rocKE SDPA kernel and launch it without hipDNN.

The executor launches through rocKE's own runtime (``rocke.runtime``) on the
torch default stream, and times through ``timing.measure`` like the other
executors. Timed launches run under ``rocke.runtime.no_fence``: a fenced
rocKE launch synchronizes the host after every call.
"""

import hashlib
from pathlib import Path
from typing import Any, Callable, Dict, NamedTuple, Optional, Tuple

import torch

from ..common import torch_support
from ..common.exceptions import ExecutionError
from ..config.benchmark_config import TimingPolicy
from .kpack import read_kernel
from .rocke_sdpa import (
    DENSE_ARCH,
    AutoKernel,
    CatalogKernel,
    RockeKernel,
    SdpaProblem,
    hipdnn_grid,
)
from .timing import Measurement, Timer, is_hip_available, measure


def device_arch(device: torch.device) -> str:
    """GPU architecture name without feature flags (``gfx950``)."""
    return str(torch.cuda.get_device_properties(device).gcnArchName).split(":")[0]


def device_num_cus(device: torch.device) -> int:
    """Compute-unit count, which rocKE's dispatcher uses to size grids."""
    return int(torch.cuda.get_device_properties(device).multi_processor_count)


class LoadedVariant(NamedTuple):
    """A catalog variant ready to launch with hipDNN's geometry.

    ``load_ms`` is the host time that reading and loading the code object
    took when it first loaded.
    """

    launcher: Any  # rocke.runtime.KernelLauncher
    load_ms: float
    grid: Tuple[int, int, int]
    block: Tuple[int, int, int]


def load_variant(
    kernel: CatalogKernel,
    problem: SdpaProblem,
    loaded: Dict[Tuple[Path, str], Tuple[Any, float]],
) -> LoadedVariant:
    """Check a catalog variant against hipDNN's launch and load its code object.

    ``loaded`` maps ``(library, toc_key)`` to the launcher and the load time,
    so each code object loads once.

    Raises:
        ExecutionError: rocKE's argument list or grid is not the one hipDNN
            uses, or the code object does not have the recorded SHA-256.
        OSError: The kpack archive is unreadable.
    """
    from kernels.gfx950.attention_dense import (
        Gfx950AttentionDenseSpec,
        attention_dense_block,
        attention_dense_grid,
        attention_dense_signature,
    )
    from rocke.runtime import KernelLauncher

    spec = Gfx950AttentionDenseSpec(**kernel.spec)
    # rocKE packs kernel arguments from its own typed signature; hipDNN binds
    # the descriptor's. Equal names in equal order give one layout.
    signature = attention_dense_signature(spec)
    names = tuple(a["name"] for a in signature)
    if names != kernel.arg_names:
        raise ExecutionError(
            f"rocKE's argument list {names} differs from hipDNN's {kernel.arg_names}"
        )
    grid = tuple(attention_dense_grid(spec))
    expected = hipdnn_grid(problem, int(kernel.spec["block_m"]))
    if grid != expected:
        raise ExecutionError(f"rocKE's grid {grid} differs from hipDNN's {expected}")
    key = (kernel.library, kernel.toc_key)
    if key not in loaded:
        with Timer() as t:
            hsaco = read_kernel(kernel.library, kernel.toc_key, DENSE_ARCH)
            digest = hashlib.sha256(hsaco).hexdigest()
            if digest != kernel.sha256:
                raise ExecutionError(
                    f"code object {kernel.toc_key} in {kernel.library} has "
                    f"SHA-256 {digest[:12]}, the descriptor records "
                    f"{kernel.sha256[:12]}"
                )
            launcher = KernelLauncher(
                hsaco=hsaco, kernel_name=kernel.symbol, signature=signature
            )
        loaded[key] = (launcher, t.elapsed_ms)
    launcher, load_ms = loaded[key]
    return LoadedVariant(launcher, load_ms, grid, tuple(attention_dense_block(spec)))


class PagedViews(NamedTuple):
    """Unified-attention arguments over packed BSHD tensors, without a copy."""

    q_tokens: torch.Tensor  # [B*Sq, Hq, D]
    o_tokens: torch.Tensor  # [B*Sq, Hq, D]
    k_cache: torch.Tensor  # [num_blocks, block_size, Hkv, D]
    v_cache: torch.Tensor  # [num_blocks, block_size, Hkv, D]
    cu_seqlens_q: torch.Tensor  # [B+1] int32
    seqused_k: torch.Tensor  # [B] int32
    block_table: torch.Tensor  # [B, Skv / block_size] int32


def paged_views(p: SdpaProblem, block_size: int, q, k, v, o) -> PagedViews:
    """Map contiguous ``[B, S, H, D]`` Q/K/V/O onto rocKE's paged-KV ABI.

    Packed BSHD K/V is a paged cache whose pages are in order: page ``j`` of
    sequence ``b`` is block ``b * (Skv / block_size) + j``.
    """
    blocks_per_seq = p.seqlen_kv // block_size
    device = q.device
    return PagedViews(
        q_tokens=q.view(p.batch * p.seqlen_q, p.num_query_heads, p.head_size),
        o_tokens=o.view(p.batch * p.seqlen_q, p.num_query_heads, p.head_size),
        k_cache=k.view(
            p.batch * blocks_per_seq, block_size, p.num_kv_heads, p.head_size
        ),
        v_cache=v.view(
            p.batch * blocks_per_seq, block_size, p.num_kv_heads, p.head_size
        ),
        cu_seqlens_q=torch.arange(
            0, (p.batch + 1) * p.seqlen_q, p.seqlen_q, dtype=torch.int32, device=device
        ),
        seqused_k=torch.full((p.batch,), p.seqlen_kv, dtype=torch.int32, device=device),
        block_table=torch.arange(
            p.batch * blocks_per_seq, dtype=torch.int32, device=device
        ).view(p.batch, blocks_per_seq),
    )


class RockeExecutor:
    """Launch one rocKE kernel on graph tensors.

    ``build_time_ms`` is the host time to get and load the kernel. For a
    catalog kernel that is reading hipDNN's code object from the kpack
    archive and loading it. ``loaded`` caches the loaded code objects by
    ``(library, toc_key)``, so a variant loads once; a later row reports the
    time of that first load. For the auto kernel it includes the first launch,
    because rocKE compiles the unified kernel inside its launch call.
    """

    def __init__(
        self,
        kernel: RockeKernel,
        problem: SdpaProblem,
        policy: TimingPolicy,
        device: Optional[torch.device] = None,
        loaded: Optional[Dict[Tuple[Path, str], Tuple[Any, float]]] = None,
    ) -> None:
        self._kernel = kernel
        self._problem = problem
        self._policy = policy
        self._device = device if device is not None else torch.device("cuda")
        self._loaded = loaded if loaded is not None else {}
        # Same choice as the PyTorch executor: rocKE and ROCm torch share the
        # HIP runtime, so HIP events can bracket the torch stream.
        self._timer = (
            "hip" if torch_support.is_rocm_build() and is_hip_available() else "torch"
        )
        self._stream = torch.cuda.default_stream(self._device)
        self._launch: Optional[Callable[[], None]] = None
        self.build_time_ms: Optional[float] = None

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
        from rocke.runtime import LaunchConfig

        kernel = self._kernel
        assert isinstance(kernel, CatalogKernel)
        p = self._problem
        variant = load_variant(kernel, p, self._loaded)
        self.build_time_ms = variant.load_ms
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
            grid=variant.grid,
            block=variant.block,
        )
        self._launch = lambda: variant.launcher(values, config=config)

    def _prepare_auto(self, q, k, v, o) -> None:
        from kernels import UnifiedAttentionProblem, run_unified_attention_torch

        kernel = self._kernel
        assert isinstance(kernel, AutoKernel)
        p = self._problem
        bs = kernel.block_size
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
        views = paged_views(p, bs, q, k, v, o)
        # Force the dispatcher's path: "auto" would fall back to rocKE's
        # scalar correctness kernel when the tiled path declines.
        backend = "tiled" if kernel.path == "2d" else "3d"
        stream = int(self._stream.cuda_stream)

        def launch() -> None:
            run_unified_attention_torch(
                problem=problem,
                q=views.q_tokens,
                k=views.k_cache,
                v=views.v_cache,
                out=views.o_tokens,
                cu_seqlens_q=views.cu_seqlens_q,
                seqused_k=views.seqused_k,
                softmax_scale=p.scale,
                block_table=views.block_table,
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

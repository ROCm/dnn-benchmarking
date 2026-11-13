# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""AITER executor: launch one AITER SDPA forward backend without hipDNN.

AITER launches on torch's current stream. The executor makes that the torch
default stream and times through ``timing.measure`` like the other
executors. AITER compiles the host module of a compiled backend (JIT) the
first time the backend runs in an install, and Triton compiles its kernel on
the first launch in a process. ``prepare`` makes that first launch, so
``build_time_ms`` holds the compile on a cold cache and the module load or
cache lookup otherwise.
"""

import importlib.util
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, Optional, Tuple

import torch

from ..common import torch_support
from ..common.exceptions import ExecutionError
from ..config.benchmark_config import TimingPolicy
from .aiter_sdpa import AITER_ARCHES, AiterBackend, AiterDevice, backend_rows, is_causal
from .sdpa_graph import DenseSdpa
from .timing import Measurement, Timer, is_hip_available, measure

#: One AITER call on the graph tensors.
Launch = Callable[[], Any]

#: Backends whose entry point allocates and returns the output; the others
#: write the graph's O tensor.
_ALLOCATES_OUTPUT = frozenset({"auto", "triton", "gluon"})


def prepare_aiter_environment() -> None:
    """Set the JIT directory and toolchain of AITER; call before ``import aiter``.

    AITER compiles its host modules (JIT) against the installed torch and
    rebuilds a module only when it is missing or built for another GPU. So,
    in a venv and unless ``AITER_JIT_DIR`` is set, this keeps the modules in
    ``<venv>/aiter-jit``: ``setup_env.py --clean`` removes them with the venv,
    and two venvs do not load each other's modules.

    AITER reads its ROCm root from ``ROCM_HOME``, then ``ROCM_PATH``, and its
    compiler from the first ``hipcc`` on ``PATH``. The setup_env.py
    environment sets ``ROCM_PATH`` to the hipDNN prefix, the SDK's libraries
    package, which has no HIP runtime to link (``-lamdhip64`` fails). When
    the SDK's devel package is installed and expanded, this sets
    ``ROCM_HOME`` to it (unless ``ROCM_HOME`` is set) and puts the
    interpreter's ``bin``, which holds the SDK's ``hipcc``, first on
    ``PATH``. The first ``import aiter`` compiles AITER's core module.
    """
    if sys.prefix != sys.base_prefix:
        os.environ.setdefault("AITER_JIT_DIR", str(Path(sys.prefix) / "aiter-jit"))
    spec = importlib.util.find_spec("_rocm_sdk_devel")
    if spec is None or not spec.submodule_search_locations:
        return
    devel = Path(list(spec.submodule_search_locations)[0])
    if not (devel / "bin" / "hipconfig").is_file():
        return  # `rocm-sdk path --root` has not expanded it
    os.environ.setdefault("ROCM_HOME", str(devel))
    bin_dir = Path(sys.executable).parent
    hipcc = shutil.which("hipcc")
    if (bin_dir / "hipcc").is_file() and (
        hipcc is None or Path(hipcc).parent != bin_dir
    ):
        os.environ["PATH"] = os.pathsep.join([str(bin_dir), os.environ.get("PATH", "")])


def _launch_for(
    backend: AiterBackend,
    problem: DenseSdpa,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
) -> Launch:
    """Bind one AITER entry point to the ``[B, S, H, D]`` views of Q/K/V/O.

    The arguments are the ones ``aiter.flash_attn_func`` passes for a dense
    problem without dropout, window, sink, bias or FP8 descale.
    """
    from aiter.ops import mha

    scale = problem.scale
    causal = is_causal(problem)
    name = backend.name
    if name == "auto":
        return lambda: mha.flash_attn_func(
            q, k, v, dropout_p=0.0, softmax_scale=scale, causal=causal
        )
    if name == "ck":
        return lambda: mha.mha_fwd(
            q,
            k,
            v,
            dropout_p=0.0,
            softmax_scale=scale,
            is_causal=causal,
            window_size_left=-1,
            window_size_right=-1,
            sink_size=0,
            return_softmax_lse=False,
            return_dropout_randval=False,
            out=o,
        )
    if name == "asm_v3":
        return lambda: mha.fmha_v3_fwd(
            q,
            k,
            v,
            dropout_p=0.0,
            softmax_scale=scale,
            is_causal=causal,
            window_size_left=-1,
            window_size_right=-1,
            return_softmax_lse=False,
            return_dropout_randval=False,
            how_v3_bf16_cvt=1,  # flash_attn_func's default
            out=o,
        )
    if name == "opus":
        return lambda: mha.fmha_fwd_bf16_opus_fwd(
            q, k, v, softmax_scale=scale, causal=causal, out=o
        )
    if name == "native_splitkv":
        splits = backend.num_splits
        return lambda: mha.mha_fwd_native_splitkv(
            q,
            k,
            v,
            out=o,
            softmax_scale=scale,
            causal=causal,
            return_lse=False,
            num_splits=splits,
        )
    if name in ("triton", "gluon"):
        from aiter.ops.triton.attention.mha import flash_attn_func

        return lambda: flash_attn_func(
            q,
            k,
            v,
            dropout_p=0.0,
            softmax_scale=scale,
            causal=causal,
            backend=name,
        )
    raise ExecutionError(f"unknown AITER backend {name!r}")


class AiterExecutor:
    """Launch one AITER backend on graph tensors."""

    def __init__(
        self,
        backend: AiterBackend,
        problem: DenseSdpa,
        policy: TimingPolicy,
        device: Optional[torch.device] = None,
    ) -> None:
        self._backend = backend
        self._problem = problem
        self._policy = policy
        self._device = device if device is not None else torch.device("cuda")
        # Same choice as the PyTorch executor: AITER and ROCm torch share the
        # HIP runtime, so HIP events can bracket the torch stream.
        self._timer = (
            "hip" if torch_support.is_rocm_build() and is_hip_available() else "torch"
        )
        self._stream = torch.cuda.default_stream(self._device)
        self._launch: Optional[Launch] = None
        self._o: Optional[torch.Tensor] = None
        self.build_time_ms: Optional[float] = None

    def prepare(self, tensors: Dict[int, torch.Tensor]) -> None:
        """Bind the backend to the graph tensors and launch it once.

        Raises:
            ExecutionError: The first launch fails (an AITER compile error, a
                shape that the backend rejects, a HIP error).
        """
        p = self._problem
        # Graph order is [B, H, S, D] over packed BSHD memory; the permuted
        # views are the contiguous [B, S, H, D] tensors AITER reads.
        q, k, v, o = (
            tensors[uid].permute(0, 2, 1, 3)
            for uid in (p.q_uid, p.k_uid, p.v_uid, p.o_uid)
        )
        if not all(t.is_contiguous() for t in (q, k, v, o)):
            raise ExecutionError("AITER runtime needs packed BSHD Q/K/V/O tensors")
        try:
            launch = _launch_for(self._backend, p, q, k, v, o)
            with Timer() as t:
                self._run_once(launch, o)
        except Exception as e:
            raise ExecutionError(
                f"AITER {self._backend.name} failed: {type(e).__name__}: {e}"
            ) from e
        self.build_time_ms = t.elapsed_ms
        self._launch, self._o = launch, o

    def execute_once(self) -> None:
        """Launch the backend once, write the graph's O tensor and wait."""
        launch = self._require_prepared()
        assert self._o is not None
        self._run_once(launch, self._o)

    def _run_once(self, launch: Launch, o: torch.Tensor) -> None:
        with (
            torch.no_grad(),
            torch.cuda.device(self._device),
            torch.cuda.stream(self._stream),
        ):
            out = launch()
            if self._backend.name in _ALLOCATES_OUTPUT:
                o.copy_(out)
            self._stream.synchronize()

    def benchmark(self) -> Measurement:
        """Time the backend per the executor's policy.

        A backend that allocates its output keeps the allocation in the timed
        call, as a caller of that AITER entry point does; the copy to the
        graph's O tensor is not timed.
        """
        launch = self._require_prepared()
        try:
            with (
                torch.no_grad(),
                torch.cuda.device(self._device),
                torch.cuda.stream(self._stream),
            ):
                return measure(
                    launch,
                    stream=int(self._stream.cuda_stream),
                    policy=self._policy,
                    timer=self._timer,
                    torch_stream=self._stream,
                )
        finally:
            self._stream.synchronize()

    @property
    def device(self) -> torch.device:
        """The GPU the executor launches on."""
        return self._device

    def _require_prepared(self) -> Launch:
        if self._launch is None:
            raise ExecutionError("Executor not prepared. Call prepare() first.")
        return self._launch


def aiter_device(device: torch.device) -> AiterDevice:
    """The facts of ``device`` that AITER's backend checks read.

    It imports AITER, so it prepares the environment first
    (:func:`prepare_aiter_environment`).
    """
    prepare_aiter_environment()
    from aiter.ops.triton.attention.mha import is_gluon_available

    return AiterDevice(
        arch=torch_support.device_arch(device),
        num_cus=torch_support.device_num_cus(device),
        gluon=bool(is_gluon_available()),
    )


#: ``(head_size_qk, head_size_v)`` of the prebuild problems: together they
#: select every backend that ``backend_rows`` knows.
_PREBUILD_HEAD_SIZES = ((128, 128), (192, 128), (64, 64))


def prebuild_skip_reason() -> Optional[str]:
    """Why :func:`prebuild` compiles nothing on this host, or None.

    It does not import AITER, whose import compiles AITER's core module.
    """
    if not torch_support.gpu_available():
        return "no GPU is visible; AITER compiles each backend on first use"
    arch = torch_support.device_arch(torch.device("cuda"))
    if arch not in AITER_ARCHES:
        return f"{arch} is not one of {', '.join(AITER_ARCHES)}; it has no AITER rows"
    return None


def prebuild() -> Iterator[Tuple[str, Optional[str]]]:
    """Compile every AITER backend that runs on this GPU, for fp16 and bf16.

    It runs each backend once on small problems with and without a causal
    mask, which compiles the host modules that ``--runtime aiter`` loads
    later. setup_env.py calls it after it installs AITER and prints each
    result. Call :func:`prebuild_skip_reason` first.

    Yields:
        ``(what ran, error)`` for each backend run; ``error`` is None when
        the run passed.
    """
    device = torch.device("cuda")
    info = aiter_device(device)
    # A long KV makes the native split-KV heuristic split (gfx942 bf16 d64).
    b, h, sq, skv = 1, 8, 128, 4096
    for data_type, dtype in (("half", torch.float16), ("bfloat16", torch.bfloat16)):
        for causal in (False, True):
            for dqk, dv in _PREBUILD_HEAD_SIZES:
                problem = DenseSdpa(
                    data_type=data_type,
                    batch=b,
                    num_query_heads=h,
                    num_kv_heads=h,
                    seqlen_q=sq,
                    seqlen_kv=skv,
                    head_size_qk=dqk,
                    head_size_v=dv,
                    causal=False,
                    bottom_right=causal,
                    scale=dqk**-0.5,
                    q_uid=1,
                    k_uid=2,
                    v_uid=3,
                    o_uid=4,
                )
                shapes = ((1, sq, dqk), (2, skv, dqk), (3, skv, dv), (4, sq, dv))
                # Graph order [B, H, S, D] over packed BSHD memory.
                tensors = {
                    uid: torch.randn(b, s, h, d, dtype=dtype, device=device).permute(
                        0, 2, 1, 3
                    )
                    for uid, s, d in shapes
                }
                mask = "causal" if causal else "no mask"
                for backend in backend_rows(problem, info):
                    label = f"{backend.row_name} {data_type} {mask} {dqk}/{dv}"
                    executor = AiterExecutor(backend, problem, TimingPolicy(), device)
                    try:
                        executor.prepare(tensors)
                    except ExecutionError as e:
                        yield label, str(e)
                    else:
                        yield f"{label}: {executor.build_time_ms:.0f} ms", None

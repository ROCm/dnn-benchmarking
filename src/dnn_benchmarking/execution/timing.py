# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""The one timed-loop implementation shared by every executor (``measure``).

Modes
-----
* ``staged`` (default on HIP): each iteration is enqueued behind a stalled
  stream (``StalledRegionTimer``), so the start->stop event span is gap-free
  device time with no host submission inside it.
* ``events``: start/stop events around each enqueue. Used when staging is not
  available (torch timer, missing stream-wait-value support) or when the
  enqueue synchronizes with the host, which would deadlock a stalled stream.

In both modes ``host_ms`` is ``perf_counter`` around the enqueue call only,
i.e. host submission cost.

Timers
------
* ``hip``: HIP events from ``hipdnn_frontend`` recorded on a raw stream
  pointer. Preferred on ROCm.
* ``torch``: ``torch.cuda.Event`` recorded on a torch stream; CUDA hosts,
  where ``hipdnn_frontend`` is not installed.
"""

import atexit
import ctypes
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type

from ..common.exceptions import ExecutionError
from ..config.benchmark_config import TimingPolicy


@dataclass
class Measurement:
    """Samples and provenance of one timed loop (see ``measure``).

    Attributes:
        kernel_ms: Device time per execution for each timed sample (in
            block mode, the block's elapsed time divided by its size).
        host_ms: Host submit time (enqueue only) per execution, same shape.
        mode: ``staged`` (stall-gated gap-free span), ``events``, or
            ``block`` (rocKE block timing, ``timing_block > 1``).
        timer: Event timer, ``hip`` or ``torch``.
        cache_mode: ``warm`` or ``cold``.
        warmup_iters: Untimed enqueues actually run (always >= 1): priming,
            any host-sync probe rerun, and discarded measured-path warmups.
            In block mode, priming plus every per-sample untimed execution.
        first_call_ms: Wall time of the first untimed enqueue plus sync;
            captures one-time plan compile / kernel find cost.
        capped: True when ``max_iters`` stopped the loop before the
            ``min_time_ms`` budget was met.
        fallback_reason: Why staged mode was not used, when it was not.
    """

    kernel_ms: List[float]
    host_ms: List[float]
    mode: str
    timer: str
    cache_mode: str
    warmup_iters: int
    first_call_ms: float
    capped: bool = False
    fallback_reason: Optional[str] = None
    timing_block: int = 1
    launch_mode: str = "raw"
    requested_warmup_iters: Optional[int] = None
    flush_method: Optional[str] = None
    flush_bytes: Optional[int] = None
    rotated_buffers: int = 1
    timed_work_notes: Optional[List[str]] = None


class StallFallbackError(RuntimeError):
    """The stall gate could not arm, or its watchdog released the stalled
    stream, so the staged sample is invalid. The suite runner remeasures the
    whole graph without stalling (see ``suite_runner``)."""


class MixedModeFallbackError(StallFallbackError):
    """A row in this graph fell back to events mode while another ran staged,
    so the suite runner remeasures the whole graph uniformly in events mode."""


#: ``fallback_reason`` of a row that a stall-gate failure remeasured unstalled.
STALL_RERUN_REASON = "stall gate failed in this graph; every row remeasured unstalled"

#: ``fallback_reason`` of a row remeasured in events mode because another row
#: on the same graph fell back to events mode.
MIXED_MODE_RERUN_REASON = (
    "another row in this graph fell back to events; every row remeasured in events mode"
)
_HIP_API = ("HipEvent", "hip_get_device_count", "hip_device_synchronize")
_STAGED_HIP_API = ("HipStallGate", "hip_can_use_stream_wait_value")

# Default fallback flush size (512 MiB) when the GPU arch is unknown; calibrated
# per-architecture sizes match measured L2/MALL capacity sweeps (ALMIOPEN-3437).
_FLUSH_BYTES = 512 * 1024 * 1024
_FLUSH_BYTES_BY_ARCH: Dict[str, int] = {
    "gfx90a": 24 * 1024 * 1024,  # 8 MiB L2 (no MALL); 3x L2 sweep
    "gfx942": 640 * 1024 * 1024,  # 256 MiB MALL; 2.5x MALL sweep
    "gfx950": 640 * 1024 * 1024,  # 256 MiB MALL; 2.5x MALL sweep
}
# One cold-cache flush buffer per timer, allocated on first cold iteration.
_flush_buffers: Dict[str, Any] = {}
# Cached compiled HIP read-only sweep kernel (module + function + sink), if hiprtc works.
_hip_sweep_kernel: Optional[Any] = None
_hip_sweep_probed: bool = False
# Set when the exit handler that frees the flush buffers is registered.
_flush_exit_registered = False
# Lazily imported hipdnn_frontend module. Resolved on first HIP use so the tool
# stays importable on hosts without hipDNN (e.g. CUDA machines running the
# PyTorch runtime). Tests may inject a fake module here.
hipdnn: Optional[Any] = None


def _require_hip_runtime() -> Any:
    """Return hipdnn_frontend when its HIP bindings and a HIP device exist."""
    global hipdnn
    if hipdnn is None:
        try:
            import hipdnn_frontend
        except ImportError as e:
            raise RuntimeError(
                f"HIP GPU timing not available: hipdnn_frontend is not importable: {e}"
            ) from e
        hipdnn = hipdnn_frontend
    missing = [name for name in _HIP_API if not hasattr(hipdnn, name)]
    if missing:
        raise RuntimeError(
            f"hipdnn_frontend is missing HIP bindings: {', '.join(missing)}"
        )
    if int(hipdnn.hip_get_device_count()) <= 0:
        raise RuntimeError("HIP GPU timing not available: no HIP devices are visible")
    return hipdnn


def is_hip_available() -> bool:
    """Return True when HIP event timing (hipdnn_frontend + a device) works."""
    try:
        _require_hip_runtime()
        return True
    except RuntimeError:
        return False


def _staged_unavailable_reason() -> Optional[str]:
    """Return why stall-gated staging cannot run here, or None if it can."""
    try:
        module = _require_hip_runtime()
    except RuntimeError as e:
        return str(e)
    missing = [name for name in _STAGED_HIP_API if not hasattr(module, name)]
    if not missing and not hasattr(module.HipStallGate, "timed_out"):
        # Without the watchdog state a released gate gives silent bad samples.
        missing = ["HipStallGate.timed_out"]
    if missing:
        return f"hipdnn_frontend is missing staging bindings: {', '.join(missing)}"
    if not module.hip_can_use_stream_wait_value():
        return "device does not support hipStreamWaitValue32"
    return None


class EventTimer:
    """Start/stop GPU event pair on one stream."""

    def __init__(self, timer: str, stream: int = 0, torch_stream: Any = None):
        """Create the event pair.

        Args:
            timer: ``hip`` (HIP events on ``stream``) or ``torch``
                (``torch.cuda.Event`` on ``torch_stream``, default the current
                torch stream).
            stream: HIP stream pointer encoded as an integer.
            torch_stream: torch.cuda.Stream for the torch timer.

        Raises:
            RuntimeError: If the timer's runtime is unavailable.
        """
        if timer == "hip":
            module = _require_hip_runtime()
            self._stream: Any = int(stream)
            self._start = module.HipEvent()
            self._stop = module.HipEvent()
        elif timer == "torch":
            import torch

            self._stream = (
                torch_stream
                if torch_stream is not None
                else torch.cuda.current_stream()
            )
            self._start = torch.cuda.Event(enable_timing=True)
            self._stop = torch.cuda.Event(enable_timing=True)
        else:
            raise ValueError(f"Unknown timer: {timer!r}")

    def start(self) -> None:
        self._start.record(self._stream)

    def stop(self) -> None:
        self._stop.record(self._stream)

    def elapsed_ms(self) -> float:
        """Wait for the stop event and return the start->stop span."""
        self._stop.synchronize()
        return float(self._start.elapsed_time(self._stop))


class StalledRegionTimer:
    """Stage a measured region behind a stalled queue.

    Per iteration the sequence is: arm stall, start event, CPU-start,
    <enqueue work>, CPU-stop, stop event, release. ``measure`` returns
    ``(host_submit_ms, kernel_ms)`` with no host work inside the GPU span,
    so the start->stop event span is gap-free device time and the CPU
    bracket is pure host submission cost. The caller must leave the device
    idle before each call (``measure()`` priming and cold flushes end with a
    device sync; each iteration ends with ``stop.synchronize()``).
    """

    def __init__(self, stream: int = 0) -> None:
        """Initialize the staged timer's gate and HIP events.

        Raises:
            RuntimeError: If HIP bindings or a HIP device are unavailable.
        """
        self._hipdnn = _require_hip_runtime()
        self._stream = int(stream)
        self._gate = self._hipdnn.HipStallGate()
        self._start = self._hipdnn.HipEvent()
        self._stop = self._hipdnn.HipEvent()

    def measure(self, enqueue: Callable[[], None]) -> Tuple[float, float]:
        """Measure one staged iteration; returns ``(host_submit_ms, kernel_ms)``.

        Raises:
            StallFallbackError: The gate could not arm, or the watchdog
                released it before the enqueue finished (the span then holds
                host submission gaps).
        """
        try:
            self._gate.arm(self._stream)
        except RuntimeError as e:
            raise StallFallbackError(f"stall gate could not arm: {e}") from e
        try:
            self._start.record(self._stream)
            t0 = time.perf_counter()
            enqueue()
            t1 = time.perf_counter()
            self._stop.record(self._stream)
        except BaseException:
            # Anything after arm() raised (e.g. an ExecutionError from the
            # work submission). Release the gate so the armed stream-wait is
            # satisfied, then drain the device so no pending wait still
            # references the signal memory when the gate is torn down, then
            # propagate. The stop event was not recorded, so do not
            # synchronize it.
            self._gate.release()
            try:
                self._hipdnn.hip_device_synchronize()
            except Exception:
                pass
            raise
        self._gate.release()
        self._stop.synchronize()
        if self._gate.timed_out():
            raise StallFallbackError("stall watchdog invalidated the measurement")
        return (t1 - t0) * 1000.0, float(self._start.elapsed_time(self._stop))


def device_sync(timer: str) -> None:
    """Block until all work on the current device (``hip`` or ``torch``) is done."""
    if timer == "hip":
        _require_hip_runtime().hip_device_synchronize()
    else:
        import torch

        torch.cuda.synchronize()


def _is_fake_hipdnn() -> bool:
    """True when unit tests injected a fake module in place of hipdnn_frontend."""
    return hipdnn is not None and not getattr(hipdnn, "__file__", None)


def query_free_vram_bytes(timer: str = "hip") -> Optional[int]:
    """Return currently free GPU VRAM in bytes, or None if unavailable.

    Never imports ``torch`` on a HIP-only run and never loads a second HIP
    runtime library.
    """
    if _is_fake_hipdnn():
        return None
    if timer == "torch" or "torch" in sys.modules:
        torch = sys.modules.get("torch")
        try:
            if (
                torch is not None
                and getattr(torch, "__file__", None)
                and torch.cuda.is_available()
            ):
                free, _total = torch.cuda.mem_get_info()
                return int(free)
        except Exception:
            pass
    try:
        from ..metrics.gpu_smi import _loaded_hip_runtime

        hip = _loaded_hip_runtime()
        if hip is not None:
            free = ctypes.c_size_t(0)
            total = ctypes.c_size_t(0)
            if hip.hipMemGetInfo(ctypes.byref(free), ctypes.byref(total)) == 0:
                return int(free.value)
    except Exception:
        pass
    return None


def flush_bytes_for_arch(arch: Optional[str]) -> int:
    """Calibrated cold-cache sweep size in bytes for ``arch``.

    Uses calibrated L2/MALL sizes for known architectures and falls back to
    ``_FLUSH_BYTES`` (512 MiB) when the architecture is unlisted or unknown
    (``hipDeviceAttributeL2CacheSize`` omits MALL on multi-XCD GPUs, so it is
    not used as a fallback).
    """
    if arch:
        base = str(arch).strip().lower().split(":")[0]
        if base in _FLUSH_BYTES_BY_ARCH:
            return _FLUSH_BYTES_BY_ARCH[base]
    return _FLUSH_BYTES


def _detect_flush_bytes(timer: str) -> int:
    """Determine the calibrated sweep size for the active GPU, else ``_FLUSH_BYTES``."""
    if _is_fake_hipdnn():
        return _FLUSH_BYTES
    if timer == "torch" or "torch" in sys.modules:
        torch = sys.modules.get("torch")
        try:
            if (
                torch is not None
                and getattr(torch, "__file__", None)
                and torch.cuda.is_available()
            ):
                props = torch.cuda.get_device_properties(torch.cuda.current_device())
                arch = getattr(props, "gcnArchName", None) or None
                if arch:
                    return flush_bytes_for_arch(arch)
        except Exception:
            pass
    try:
        from ..metrics.gpu_smi import GpuSmiProbe

        arch, _ = GpuSmiProbe().identity()
        if arch:
            return flush_bytes_for_arch(arch)
    except Exception:
        pass
    try:
        from ..metrics import arch as arch_mod

        detected = arch_mod.detect_arch()
        if detected and detected != "unknown":
            return flush_bytes_for_arch(detected)
    except Exception:
        pass
    return _FLUSH_BYTES


_READ_SWEEP_HIP_SRC = b"""
extern "C" __global__ void dnn_bench_read_sweep_u4(
    const unsigned int* __restrict__ src,
    unsigned long long count_u4,
    unsigned int* __restrict__ sink) {
    unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
    unsigned int acc = 0u;
    for (unsigned long long i = idx; i < count_u4; i += stride) {
        unsigned long long base = i * 4ULL;
        acc ^= src[base] ^ src[base + 1] ^ src[base + 2] ^ src[base + 3];
    }
    if (idx == 0ULL) {
        sink[0] = acc;
    }
}
"""


class _HipReadSweep:
    """Coalesced read-only VRAM sweep compiled once via hiprtc on the loaded HIP runtime."""

    def __init__(
        self, hip_lib: Any, module: ctypes.c_void_p, fn: ctypes.c_void_p
    ) -> None:
        self._hip = hip_lib
        self._module = module
        self._fn = fn

    @classmethod
    def try_build(cls) -> Optional["_HipReadSweep"]:
        if sys.platform == "win32":
            return None
        if hipdnn is None or not getattr(hipdnn, "__file__", None):
            return None
        hip_path = None
        try:
            with open("/proc/self/maps") as f:
                for line in f:
                    parts = line.split(maxsplit=5)
                    if len(parts) == 6 and "/libamdhip64.so" in parts[5]:
                        hip_path = Path(parts[5].strip())
                        break
        except OSError:
            return None
        if hip_path is None:
            return None
        rtc_candidates = sorted(hip_path.parent.glob("libhiprtc.so*"))
        if not rtc_candidates:
            return None
        try:
            hip = ctypes.CDLL(str(hip_path))
            rtc = ctypes.CDLL(str(rtc_candidates[0]))
            prog = ctypes.c_void_p()
            if (
                rtc.hiprtcCreateProgram(
                    ctypes.byref(prog),
                    _READ_SWEEP_HIP_SRC,
                    b"read_sweep.hip",
                    0,
                    None,
                    None,
                )
                != 0
            ):
                return None
            try:
                if rtc.hiprtcCompileProgram(prog, 0, None) != 0:
                    return None
                code_size = ctypes.c_size_t()
                if rtc.hiprtcGetCodeSize(prog, ctypes.byref(code_size)) != 0:
                    return None
                code = ctypes.create_string_buffer(code_size.value)
                if rtc.hiprtcGetCode(prog, code) != 0:
                    return None
            finally:
                rtc.hiprtcDestroyProgram(ctypes.byref(prog))
            module = ctypes.c_void_p()
            if hip.hipModuleLoadData(ctypes.byref(module), code) != 0:
                return None
            fn = ctypes.c_void_p()
            if (
                hip.hipModuleGetFunction(
                    ctypes.byref(fn), module, b"dnn_bench_read_sweep_u4"
                )
                != 0
            ):
                return None
            return cls(hip, module, fn)
        except Exception:
            return None

    def launch(self, buf_ptr: int, nbytes: int, sink_ptr: int, stream: int) -> bool:
        count_u4 = nbytes // 16
        if count_u4 <= 0:
            return False
        a_src = ctypes.c_uint64(buf_ptr)
        a_cnt = ctypes.c_uint64(count_u4)
        a_snk = ctypes.c_uint64(sink_ptr)
        params = (ctypes.c_void_p * 3)(
            ctypes.addressof(a_src),
            ctypes.addressof(a_cnt),
            ctypes.addressof(a_snk),
        )
        threads = 256
        blocks = min(4096, max(1, (count_u4 + threads - 1) // threads))
        rc = self._hip.hipModuleLaunchKernel(
            self._fn,
            ctypes.c_uint(blocks),
            ctypes.c_uint(1),
            ctypes.c_uint(1),
            ctypes.c_uint(threads),
            ctypes.c_uint(1),
            ctypes.c_uint(1),
            ctypes.c_uint(0),
            ctypes.c_void_p(stream),
            params,
            None,
        )
        return rc == 0


def _get_hip_sweep_kernel() -> Optional[_HipReadSweep]:
    global _hip_sweep_kernel, _hip_sweep_probed
    if not _hip_sweep_probed:
        _hip_sweep_probed = True
        _hip_sweep_kernel = _HipReadSweep.try_build()
    return _hip_sweep_kernel


def _flush_cache(timer: str, stream: int = 0) -> Tuple[str, int]:
    """Evict L2/MALL via a read-only capacity sweep (fallback: zero-fill), then drain.

    Returns ``(method, flush_bytes)`` where ``method`` is ``"read_sweep"`` or
    ``"zero_fill"``. Never mislabels a zero-fill fallback as ``"read_sweep"``.
    """
    global _flush_exit_registered
    entry = _flush_buffers.get(timer)
    if entry is None:
        nbytes = _detect_flush_bytes(timer)
        try:
            if timer == "hip":
                mod = _require_hip_runtime()
                buf = mod.DeviceBuffer(nbytes)
                sink = None
                sweep = _get_hip_sweep_kernel()
                if sweep is not None and hasattr(buf, "ptr"):
                    buf.zeros()  # pre-touch once at allocation
                    sink = mod.DeviceBuffer(16)
                entry = (buf, nbytes, sink)
            else:
                import torch

                buf = torch.empty(nbytes, dtype=torch.uint8, device="cuda")
                if (
                    hasattr(buf, "zero_")
                    and hasattr(buf, "view")
                    and hasattr(torch, "int64")
                ):
                    buf.zero_()  # pre-touch once at allocation
                entry = (buf, nbytes, None)
        except Exception as e:
            raise ExecutionError(
                f"Cannot allocate the {nbytes >> 20} MiB cold-cache flush "
                f"buffer ({e}); rerun with --cache-mode warm"
            ) from e
        if not _flush_exit_registered:
            # Free device buffers before interpreter teardown; nanobind
            # reports module-global instances still alive at exit as leaks.
            # Once per process: release_flush_buffers() empties the dict.
            atexit.register(_flush_buffers.clear)
            _flush_exit_registered = True
        _flush_buffers[timer] = entry

    buf, nbytes, sink = entry
    method = "zero_fill"
    if timer == "hip":
        sweep = _get_hip_sweep_kernel()
        if (
            sweep is not None
            and sink is not None
            and hasattr(buf, "ptr")
            and hasattr(sink, "ptr")
            and sweep.launch(int(buf.ptr()), nbytes, int(sink.ptr()), stream)
        ):
            method = "read_sweep"
        else:
            buf.zeros()
    else:
        import torch

        if (
            hasattr(buf, "view")
            and hasattr(torch, "int64")
            and hasattr(torch, "sum")
            and nbytes >= 8
        ):
            # Coalesced read-only reduction over pre-touched VRAM; writes only
            # an 8-byte scalar sink on the current stream.
            _ = torch.sum(buf[: (nbytes // 8) * 8].view(torch.int64))
            method = "read_sweep"
        else:
            buf.zero_()
    device_sync(timer)
    return method, nbytes


def release_flush_buffers() -> None:
    """Free the cold-cache flush buffers; the next cold iteration reallocates.

    Call before a child process allocates its own device memory (the
    ``--oracle`` tuned runs), so the child does not run with the flush buffer
    still held. The torch allocator keeps freed blocks, so also return them to
    the device.
    """
    had_torch = _flush_buffers.pop("torch", None) is not None
    _flush_buffers.clear()
    if had_torch:
        sys.modules["torch"].cuda.empty_cache()


def _probe_host_sync(enqueue: Callable[[], None]) -> Optional[str]:
    """Run one enqueue with torch's sync debug mode set to error.

    Returns why the enqueue synchronized with the host, or None. A host sync
    inside a stall-gated region never completes, so a syncing enqueue must be
    timed in events mode. On failure the enqueue is rerun unchecked, which
    completes the priming pass and re-raises genuine execution errors.
    """
    import torch

    set_mode = getattr(torch.cuda, "set_sync_debug_mode", None)
    if set_mode is None:
        enqueue()
        return None

    def quiet_set_mode(mode: Any) -> None:
        # torch warns on every call that the debug mode is a prototype that
        # does not catch every sync; the probe is best-effort by design.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            set_mode(mode)

    previous = torch.cuda.get_sync_debug_mode()
    quiet_set_mode("error")
    try:
        enqueue()
        return None
    except Exception as e:
        reason = f"host sync in enqueue: {e}"
    finally:
        quiet_set_mode(previous)
    enqueue()
    return reason


def measure(
    enqueue: Callable[[], None],
    *,
    stream: int,
    policy: TimingPolicy,
    timer: str = "hip",
    torch_stream: Any = None,
    after_iteration: Optional[Callable[[], None]] = None,
    post_first_call: Optional[Callable[[], None]] = None,
    launch_mode: str = "raw",
    rotated_buffers: int = 1,
    rotated_bytes: Optional[int] = None,
    timed_work_notes: Optional[Sequence[str]] = None,
) -> Measurement:
    """Prime, warm up, then time ``enqueue`` per ``policy`` (a ``TimingPolicy``).

    Priming: the first enqueue is timed with a device sync as
    ``first_call_ms``. For torch-driven enqueues (``torch_stream`` given) one
    more priming enqueue runs under ``torch.cuda.set_sync_debug_mode("error")``;
    if it synchronizes, the loop uses events mode. The rest of
    ``policy.warmup_iters`` run through the timed-iteration path (same mode,
    same cold flush) and are discarded, so clocks and caches reach the timed
    loop's steady state. The loop stops once ``policy.iters`` samples exist
    and their kernel time sums to ``policy.min_time_ms``, or at
    ``policy.max_iters`` (``capped``). In cold cache mode every warmup and
    timed iteration is preceded by a cache flush and device sync.

    ``policy.timing_block > 1`` switches to rocKE block timing instead; see
    ``_measure_blocks``.

    Args:
        enqueue: Submits one iteration of work to ``stream`` / ``torch_stream``.
            Torch callers must already be inside ``torch.cuda.stream(...)``.
        stream: HIP stream pointer (``hip`` timer, staging).
        policy: Warmup/iteration counts, time budget, and cache mode.
        timer: ``hip`` or ``torch`` timer.
        torch_stream: torch.cuda.Stream the enqueue runs on, if torch-driven.
        after_iteration: Optional callback invoked after every synchronized
            iteration outside the timed bracket (e.g. releasing rocKE's
            retained eager scratch).
        launch_mode: ``"raw"`` or ``"graph_replay"`` launch provenance.
        rotated_buffers: Number of rotated I/O buffer sets (1 = none).
        rotated_bytes: Total bytes across rotated I/O sets when rotating.
        timed_work_notes: Extra non-core work inside the timed closure.

    Raises:
        ExecutionError: If the cold-cache flush buffer cannot be allocated.
        StallFallbackError: If stall-gated timing failed (see
            ``StalledRegionTimer``); the caller remeasures without stalling.
        RuntimeError: If the timer is unavailable.
    """
    events = EventTimer(timer, stream, torch_stream)
    notes = list(timed_work_notes) if timed_work_notes else None

    t0 = time.perf_counter()
    enqueue()
    device_sync(timer)
    first_call_ms = (time.perf_counter() - t0) * 1000.0
    if after_iteration is not None:
        after_iteration()
    if post_first_call is not None:
        post_first_call()
    primed = 1
    reason: Optional[str] = None
    if torch_stream is not None:
        reason = _probe_host_sync(enqueue)
        # A detected sync costs a second (unchecked) enqueue.
        primed += 1 if reason is None else 2
        device_sync(timer)
        if after_iteration is not None:
            after_iteration()

    if policy.timing_block > 1:
        return _measure_blocks(
            enqueue,
            events,
            policy,
            timer,
            first_call_ms,
            primed,
            reason,
            after_iteration=after_iteration,
            launch_mode=launch_mode,
            rotated_buffers=rotated_buffers,
            rotated_bytes=rotated_bytes,
            timed_work_notes=notes,
        )

    staged: Optional[StalledRegionTimer] = None
    if reason is None:
        if timer != "hip":
            reason = "staged timing requires the hip timer"
        elif not policy.stall_gate:
            reason = policy.stall_fallback_reason or STALL_RERUN_REASON
        else:
            reason = _staged_unavailable_reason()
    if reason is None:
        try:
            staged = StalledRegionTimer(stream)
        except RuntimeError as e:
            raise StallFallbackError(f"stall gate unavailable: {e}") from e

    cold = policy.cache_mode == "cold"
    flush_method: Optional[str] = "rotate" if rotated_buffers > 1 else None
    flush_bytes: Optional[int] = rotated_bytes if rotated_buffers > 1 else None

    def iteration() -> Tuple[float, float]:
        """One ``(host_ms, kernel_ms)`` sample on the timed path."""
        nonlocal flush_method, flush_bytes
        if cold:
            flush_method, flush_bytes = _flush_cache(timer, stream)
        if staged is not None:
            sample = staged.measure(enqueue)
        else:
            events.start()
            t0 = time.perf_counter()
            enqueue()
            t1 = time.perf_counter()
            events.stop()
            sample = (t1 - t0) * 1000.0, events.elapsed_ms()
        if after_iteration is not None:
            after_iteration()
        return sample

    for _ in range(primed, policy.warmup_iters):
        iteration()

    kernel_ms: List[float] = []
    host_ms: List[float] = []
    total_ms = 0.0
    capped = False
    while len(kernel_ms) < policy.iters or total_ms < policy.min_time_ms:
        if len(kernel_ms) >= policy.max_iters:
            capped = True
            break
        host, kernel = iteration()
        host_ms.append(host)
        kernel_ms.append(kernel)
        total_ms += kernel

    return Measurement(
        kernel_ms=kernel_ms,
        host_ms=host_ms,
        mode="staged" if staged is not None else "events",
        timer=timer,
        cache_mode=policy.cache_mode,
        warmup_iters=max(primed, policy.warmup_iters),
        first_call_ms=first_call_ms,
        capped=capped,
        fallback_reason=reason,
        launch_mode=launch_mode,
        requested_warmup_iters=policy.warmup_iters,
        flush_method=flush_method,
        flush_bytes=flush_bytes,
        rotated_buffers=rotated_buffers,
        timed_work_notes=notes,
    )


def _measure_blocks(
    enqueue: Callable[[], None],
    events: EventTimer,
    policy: TimingPolicy,
    timer: str,
    first_call_ms: float,
    untimed: int,
    host_sync: Optional[str],
    *,
    after_iteration: Optional[Callable[[], None]] = None,
    launch_mode: str = "raw",
    rotated_buffers: int = 1,
    rotated_bytes: Optional[int] = None,
    timed_work_notes: Optional[List[str]] = None,
) -> Measurement:
    """rocKE block timing (``time_launches`` / Solera ``measure()``).

    Each sample runs ``policy.warmup_iters`` untimed executions, drains the
    device, then times ``policy.timing_block`` back-to-back executions in one
    event pair and records the per-execution average. The first sample is
    discarded. The stall gate is not used: N gated enqueues can fill the HIP
    queue and block the host before the gate opens.

    ``untimed`` counts the priming enqueues already run; every per-sample
    untimed execution is added to it for ``Measurement.warmup_iters``.
    ``host_sync`` (the priming probe's finding) is kept as the fallback
    reason: the block's event span then includes host round-trips.
    """
    block = policy.timing_block
    kernel_ms: List[float] = []
    host_ms: List[float] = []
    total_ms = 0.0
    capped = False
    discard_first = True
    while len(kernel_ms) < policy.iters or total_ms < policy.min_time_ms:
        if len(kernel_ms) >= policy.max_iters:
            capped = True
            break
        for _ in range(policy.warmup_iters):
            enqueue()
        untimed += policy.warmup_iters
        device_sync(timer)
        if after_iteration is not None:
            after_iteration()
        events.start()
        t0 = time.perf_counter()
        for _ in range(block):
            enqueue()
        t1 = time.perf_counter()
        events.stop()
        kernel = events.elapsed_ms() / block
        if after_iteration is not None:
            after_iteration()
        if discard_first:
            discard_first = False
            continue
        kernel_ms.append(kernel)
        host_ms.append((t1 - t0) * 1000.0 / block)
        total_ms += kernel * block

    return Measurement(
        kernel_ms=kernel_ms,
        host_ms=host_ms,
        mode="block",
        timer=timer,
        cache_mode=policy.cache_mode,
        warmup_iters=untimed,
        first_call_ms=first_call_ms,
        capped=capped,
        fallback_reason=host_sync,
        timing_block=block,
        launch_mode=launch_mode,
        requested_warmup_iters=policy.warmup_iters,
        flush_method="rotate" if rotated_buffers > 1 else None,
        flush_bytes=rotated_bytes if rotated_buffers > 1 else None,
        rotated_buffers=rotated_buffers,
        timed_work_notes=timed_work_notes,
    )


class Timer:
    """Context manager measuring wall-clock time with ``perf_counter``."""

    def __init__(self) -> None:
        self._start: float = 0.0
        self._end: float = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        self._end = time.perf_counter()

    @property
    def elapsed_ms(self) -> float:
        return (self._end - self._start) * 1000.0

    @property
    def elapsed_s(self) -> float:
        return self._end - self._start

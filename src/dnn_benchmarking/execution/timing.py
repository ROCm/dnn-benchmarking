# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""The one timed-loop implementation shared by every executor (``measure``).

Modes
-----
* ``staged`` (default on HIP): each iteration is enqueued behind a stalled
  stream (``StalledRegionTimer``), so the start->stop event span is gap-free
  device time with no host submission inside it.
* ``events``: start/stop events around each enqueue. Used when staging is not
  available (torch backend, missing stream-wait-value support) or when the
  enqueue synchronizes with the host, which would deadlock a stalled stream.

In both modes ``host_ms`` is ``perf_counter`` around the enqueue call only,
i.e. host submission cost.

Event backends
--------------
* ``hip``: HIP events from ``hipdnn_frontend`` recorded on a raw stream
  pointer. Preferred on ROCm.
* ``torch``: ``torch.cuda.Event`` recorded on a torch stream; CUDA hosts,
  where ``hipdnn_frontend`` is not installed.
"""

import time
import warnings
from dataclasses import dataclass
from types import TracebackType
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

from ..common.exceptions import ExecutionError
from ..config.benchmark_config import TimingPolicy


@dataclass
class Measurement:
    """Samples and provenance of one timed loop (see ``measure``).

    Attributes:
        kernel_ms: Device span per timed iteration.
        host_ms: Host submit time (enqueue only) per timed iteration.
        mode: ``staged`` (stall-gated gap-free span) or ``events``.
        backend: Event backend, ``hip`` or ``torch``.
        cache_mode: ``warm`` or ``cold``.
        warmup_iters: Untimed enqueues actually run (always >= 1).
        first_call_ms: Wall time of the first untimed enqueue plus sync;
            captures one-time plan compile / kernel find cost.
        capped: True when ``max_iters`` stopped the loop before the
            ``min_time_ms`` budget was met.
        fallback_reason: Why staged mode was not used, when it was not.
    """

    kernel_ms: List[float]
    host_ms: List[float]
    mode: str
    backend: str
    cache_mode: str
    warmup_iters: int
    first_call_ms: float
    capped: bool = False
    fallback_reason: Optional[str] = None


_HIP_API = ("HipEvent", "hip_get_device_count", "hip_device_synchronize")
_STAGED_HIP_API = ("HipStallGate", "hip_can_use_stream_wait_value")

# ponytail: fixed 512 MiB >= 2x the largest last-level cache we target (MI300X
# MALL, 256 MiB); derive from device properties if a bigger cache appears.
_FLUSH_BYTES = 512 * 1024 * 1024
# One cold-cache flush buffer per backend, allocated on first cold iteration.
_flush_buffers: Dict[str, Any] = {}

# Lazily imported hipdnn_frontend module. Resolved on first HIP use so the tool
# stays importable on hosts without hipDNN (e.g. CUDA machines running the
# PyTorch backend). Tests may inject a fake module here.
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
    if missing:
        return f"hipdnn_frontend is missing staging bindings: {', '.join(missing)}"
    if not module.hip_can_use_stream_wait_value():
        return "device does not support hipStreamWaitValue32"
    return None


class EventTimer:
    """Start/stop GPU event pair on one stream; doubles as a stream sync."""

    def __init__(self, backend: str, stream: int = 0, torch_stream: Any = None):
        """Create the event pair.

        Args:
            backend: ``hip`` (HIP events on ``stream``) or ``torch``
                (``torch.cuda.Event`` on ``torch_stream``, default the current
                torch stream).
            stream: HIP stream pointer encoded as an integer.
            torch_stream: torch.cuda.Stream for the torch backend.

        Raises:
            RuntimeError: If the backend's runtime is unavailable.
        """
        if backend == "hip":
            module = _require_hip_runtime()
            self._stream: Any = int(stream)
            self._start = module.HipEvent()
            self._stop = module.HipEvent()
        elif backend == "torch":
            import torch

            self._stream = (
                torch_stream if torch_stream is not None else torch.cuda.current_stream()
            )
            self._start = torch.cuda.Event(enable_timing=True)
            self._stop = torch.cuda.Event(enable_timing=True)
        else:
            raise ValueError(f"Unknown timing backend: {backend!r}")

    def start(self) -> None:
        self._start.record(self._stream)

    def stop(self) -> None:
        self._stop.record(self._stream)

    def elapsed_ms(self) -> float:
        """Wait for the stop event and return the start->stop span."""
        self._stop.synchronize()
        return float(self._start.elapsed_time(self._stop))

    def synchronize_stream(self) -> None:
        """Block until all work enqueued on the stream so far has completed."""
        self.stop()
        self._stop.synchronize()


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
        """Measure one staged iteration; returns ``(host_submit_ms, kernel_ms)``."""
        self._gate.arm(self._stream)
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
        return (t1 - t0) * 1000.0, float(self._start.elapsed_time(self._stop))


def _device_sync(backend: str) -> None:
    if backend == "hip":
        _require_hip_runtime().hip_device_synchronize()
    else:
        import torch

        torch.cuda.synchronize()


def _flush_cache(backend: str) -> None:
    """Evict L2/MALL by zeroing the flush buffer, then drain the device."""
    buf = _flush_buffers.get(backend)
    if buf is None:
        try:
            if backend == "hip":
                buf = _require_hip_runtime().DeviceBuffer(_FLUSH_BYTES)
            else:
                import torch

                buf = torch.empty(_FLUSH_BYTES, dtype=torch.uint8, device="cuda")
        except Exception as e:
            raise ExecutionError(
                f"Cannot allocate the {_FLUSH_BYTES >> 20} MiB cold-cache flush "
                f"buffer ({e}); rerun with --cache-mode warm"
            ) from e
        _flush_buffers[backend] = buf
    if backend == "hip":
        buf.zeros()
    else:
        buf.zero_()
    _device_sync(backend)


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
    backend: str = "hip",
    torch_stream: Any = None,
) -> Measurement:
    """Prime, then time ``enqueue`` per ``policy`` (a ``TimingPolicy``).

    Priming runs ``max(1, policy.warmup_iters)`` untimed enqueues; the first
    is timed with a device sync as ``first_call_ms``. For torch-driven
    enqueues (``torch_stream`` given) one priming enqueue after the first runs
    under ``torch.cuda.set_sync_debug_mode("error")``; if it synchronizes, the
    loop uses events mode. The loop stops once ``policy.iters`` samples exist
    and their kernel time sums to ``policy.min_time_ms``, or at
    ``policy.max_iters`` (``capped``). In cold cache mode every timed
    iteration is preceded by a cache flush and device sync.

    Args:
        enqueue: Submits one iteration of work to ``stream`` / ``torch_stream``.
            Torch callers must already be inside ``torch.cuda.stream(...)``.
        stream: HIP stream pointer (``hip`` backend, staging).
        policy: Warmup/iteration counts, time budget, and cache mode.
        backend: ``hip`` or ``torch`` event backend.
        torch_stream: torch.cuda.Stream the enqueue runs on, if torch-driven.

    Raises:
        ExecutionError: If the cold-cache flush buffer cannot be allocated.
        RuntimeError: If the event backend is unavailable.
    """
    events = EventTimer(backend, stream, torch_stream)

    t0 = time.perf_counter()
    enqueue()
    _device_sync(backend)
    first_call_ms = (time.perf_counter() - t0) * 1000.0
    primed = 1
    reason: Optional[str] = None
    if torch_stream is not None:
        reason = _probe_host_sync(enqueue)
        primed += 1
    for _ in range(primed, policy.warmup_iters):
        enqueue()
    primed = max(primed, policy.warmup_iters)
    _device_sync(backend)

    staged: Optional[StalledRegionTimer] = None
    if reason is None:
        if backend != "hip":
            reason = "staged timing requires the hip backend"
        else:
            reason = _staged_unavailable_reason()
    if reason is None:
        try:
            staged = StalledRegionTimer(stream)
        except RuntimeError as e:
            reason = str(e)

    kernel_ms: List[float] = []
    host_ms: List[float] = []
    total_ms = 0.0
    capped = False
    cold = policy.cache_mode == "cold"
    while len(kernel_ms) < policy.iters or total_ms < policy.min_time_ms:
        if len(kernel_ms) >= policy.max_iters:
            capped = True
            break
        if cold:
            _flush_cache(backend)
        if staged is not None:
            host, kernel = staged.measure(enqueue)
        else:
            events.start()
            t0 = time.perf_counter()
            enqueue()
            t1 = time.perf_counter()
            events.stop()
            host, kernel = (t1 - t0) * 1000.0, events.elapsed_ms()
        host_ms.append(host)
        kernel_ms.append(kernel)
        total_ms += kernel

    return Measurement(
        kernel_ms=kernel_ms,
        host_ms=host_ms,
        mode="staged" if staged is not None else "events",
        backend=backend,
        cache_mode=policy.cache_mode,
        warmup_iters=primed,
        first_call_ms=first_call_ms,
        capped=capped,
        fallback_reason=reason,
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

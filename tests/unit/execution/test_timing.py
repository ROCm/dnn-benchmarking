# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""CPU-only tests for ``timing.measure`` driven through a fake HIP runtime."""

import sys
import types
from typing import List

import pytest

import dnn_benchmarking.execution.timing as timing_module
from dnn_benchmarking.common.exceptions import ExecutionError
from dnn_benchmarking.config.benchmark_config import TimingPolicy
from dnn_benchmarking.execution.timing import StalledRegionTimer, measure


class FakeClock:
    """Deterministic perf_counter: only explicit advances move time."""

    def __init__(self) -> None:
        self.now = 0.0

    def perf_counter(self) -> float:
        return self.now

    def advance_ms(self, ms: float) -> None:
        self.now += ms / 1000.0


def _install_fake_hip(
    monkeypatch,
    log: List[str],
    *,
    kernel_ms: float = 1.0,
    staged: bool = True,
    buffer_error: Exception = None,
    clock: FakeClock = None,
):
    class FakeEvent:
        def record(self, stream: int) -> None:
            log.append("record")

        def synchronize(self) -> None:
            log.append("event_sync")
            if clock is not None:
                clock.advance_ms(50.0)  # waiting on the GPU is not submit time

        def elapsed_time(self, other) -> float:
            return kernel_ms

    class FakeGate:
        def arm(self, stream: int) -> None:
            log.append("arm")

        def release(self) -> None:
            log.append("release")

    class FakeBuffer:
        def __init__(self, size: int) -> None:
            if buffer_error is not None:
                raise buffer_error
            log.append(f"alloc:{size}")

        def zeros(self) -> None:
            log.append("flush")

    fake = types.SimpleNamespace(
        HipEvent=FakeEvent,
        HipStallGate=FakeGate,
        DeviceBuffer=FakeBuffer,
        hip_get_device_count=lambda: 1,
        hip_device_synchronize=lambda: log.append("device_sync"),
        hip_can_use_stream_wait_value=lambda: staged,
    )
    monkeypatch.setattr(timing_module, "hipdnn", fake)
    monkeypatch.setattr(timing_module, "_flush_buffers", {})
    if clock is not None:
        monkeypatch.setattr(timing_module, "time", clock)
    return fake


def _enqueue(log: List[str], clock: FakeClock = None, ms: float = 0.0):
    def enqueue() -> None:
        log.append("enqueue")
        if clock is not None:
            clock.advance_ms(ms)

    return enqueue


def test_zero_warmup_still_primes_before_first_gated_enqueue(monkeypatch) -> None:
    """A first-call compile inside a stalled region never drains; measure()
    must run and drain one untimed enqueue even with warmup_iters=0."""
    log: List[str] = []
    _install_fake_hip(monkeypatch, log)

    m = measure(_enqueue(log), stream=7, policy=TimingPolicy(warmup_iters=0, iters=2))

    first_arm = log.index("arm")
    assert log.index("enqueue") < first_arm
    assert "device_sync" in log[log.index("enqueue") : first_arm]
    assert m.mode == "staged"
    assert m.warmup_iters == 1
    assert log.count("enqueue") == 3


def test_warmups_run_the_timed_path_and_are_discarded(monkeypatch) -> None:
    """After one priming enqueue (never flushed), the remaining warmups use
    the same flush + stall-gated path as timed iterations, so clocks and
    caches are in steady state when sampling starts; only timed ones are kept."""
    log: List[str] = []
    _install_fake_hip(monkeypatch, log)

    m = measure(
        _enqueue(log),
        stream=7,
        policy=TimingPolicy(warmup_iters=3, iters=2, cache_mode="cold"),
    )

    priming = log.index("enqueue")
    assert log[priming + 1] == "device_sync"
    assert "flush" not in log[: priming + 2]
    arms = [i for i, e in enumerate(log) if e == "arm"]
    assert len(arms) == 2 + 2  # 2 warmups after priming, 2 timed
    for arm in arms:
        # flush, then a full device drain, then the gate is armed.
        assert log[arm - 2 : arm] == ["flush", "device_sync"]
    assert log.count("enqueue") == 1 + 2 + 2
    assert len(m.kernel_ms) == len(m.host_ms) == 2
    assert m.warmup_iters == 3
    assert log.count(f"alloc:{timing_module._FLUSH_BYTES}") == 1
    assert m.cache_mode == "cold"


def test_warm_mode_never_flushes(monkeypatch) -> None:
    log: List[str] = []
    _install_fake_hip(monkeypatch, log)

    measure(_enqueue(log), stream=7, policy=TimingPolicy(warmup_iters=1, iters=3))

    assert "flush" not in log
    assert not any(e.startswith("alloc") for e in log)


def test_min_time_extends_sample_count(monkeypatch) -> None:
    log: List[str] = []
    _install_fake_hip(monkeypatch, log, kernel_ms=1.0)

    m = measure(
        _enqueue(log),
        stream=7,
        policy=TimingPolicy(warmup_iters=1, iters=2, min_time_ms=5.0),
    )

    assert len(m.kernel_ms) == 5
    assert m.capped is False


def test_max_iters_caps_loop_and_flags_it(monkeypatch) -> None:
    log: List[str] = []
    _install_fake_hip(monkeypatch, log, kernel_ms=1.0)

    m = measure(
        _enqueue(log),
        stream=7,
        policy=TimingPolicy(warmup_iters=1, iters=2, min_time_ms=100.0, max_iters=3),
    )

    assert len(m.kernel_ms) == len(m.host_ms) == 3
    assert m.capped is True


def test_fixed_count_reaching_iters_is_not_capped(monkeypatch) -> None:
    log: List[str] = []
    _install_fake_hip(monkeypatch, log)

    m = measure(
        _enqueue(log),
        stream=7,
        policy=TimingPolicy(warmup_iters=1, iters=3, max_iters=3),
    )

    assert len(m.kernel_ms) == 3
    assert m.capped is False


def test_events_mode_host_time_brackets_only_enqueue(monkeypatch) -> None:
    """Without staging, host_ms is submit time: waiting for the stop event
    (50 ms on the fake clock) must not leak into it."""
    log: List[str] = []
    clock = FakeClock()
    _install_fake_hip(monkeypatch, log, kernel_ms=0.25, staged=False, clock=clock)

    m = measure(
        _enqueue(log, clock, ms=2.0),
        stream=7,
        policy=TimingPolicy(warmup_iters=1, iters=3),
    )

    assert m.mode == "events"
    assert m.fallback_reason == "device does not support hipStreamWaitValue32"
    assert m.host_ms == pytest.approx([2.0, 2.0, 2.0])
    assert m.kernel_ms == [0.25, 0.25, 0.25]
    assert "arm" not in log


def test_first_call_ms_covers_first_enqueue_and_its_sync(monkeypatch) -> None:
    log: List[str] = []
    clock = FakeClock()
    _install_fake_hip(monkeypatch, log, clock=clock)
    calls = []

    def enqueue() -> None:
        calls.append(1)
        clock.advance_ms(30.0 if len(calls) == 1 else 1.0)  # first call compiles

    m = measure(enqueue, stream=7, policy=TimingPolicy(warmup_iters=2, iters=1))

    assert m.first_call_ms == pytest.approx(30.0)
    assert m.warmup_iters == 2


def test_cold_flush_allocation_failure_names_warm_mode(monkeypatch) -> None:
    log: List[str] = []
    _install_fake_hip(monkeypatch, log, buffer_error=RuntimeError("out of memory"))

    with pytest.raises(ExecutionError, match="--cache-mode warm"):
        measure(
            _enqueue(log),
            stream=7,
            policy=TimingPolicy(warmup_iters=1, iters=1, cache_mode="cold"),
        )


def _install_fake_torch_sync_debug(monkeypatch, modes: List[str]):
    cuda = types.SimpleNamespace(
        get_sync_debug_mode=lambda: 0,
        set_sync_debug_mode=lambda mode: modes.append(mode),
    )
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(cuda=cuda))


def test_host_sync_in_torch_enqueue_falls_back_to_events(monkeypatch) -> None:
    """A syncing enqueue would deadlock the stalled stream, so measure() must
    detect it during priming and time in events mode instead."""
    log: List[str] = []
    _install_fake_hip(monkeypatch, log)
    modes: List[str] = []
    _install_fake_torch_sync_debug(monkeypatch, modes)

    def enqueue() -> None:
        log.append("enqueue")
        if modes and modes[-1] == "error":
            raise RuntimeError("called a synchronizing CUDA operation")

    m = measure(
        enqueue,
        stream=7,
        policy=TimingPolicy(warmup_iters=0, iters=2),
        torch_stream=object(),
    )

    assert m.mode == "events"
    assert m.fallback_reason.startswith("host sync in enqueue: ")
    assert "arm" not in log
    assert modes == ["error", 0]  # debug mode restored
    assert len(m.kernel_ms) == 2
    # Priming, the failed checked call, and its unchecked rerun.
    assert m.warmup_iters == 3 == log.count("enqueue") - len(m.kernel_ms)


def test_async_torch_enqueue_stays_staged(monkeypatch) -> None:
    log: List[str] = []
    _install_fake_hip(monkeypatch, log)
    modes: List[str] = []
    _install_fake_torch_sync_debug(monkeypatch, modes)

    m = measure(
        _enqueue(log),
        stream=7,
        policy=TimingPolicy(warmup_iters=0, iters=1),
        torch_stream=object(),
    )

    assert m.mode == "staged"
    assert m.fallback_reason is None
    # First call populates host caches unchecked; one more checked call.
    assert m.warmup_iters == 2


def test_genuine_enqueue_error_during_sync_probe_propagates(monkeypatch) -> None:
    log: List[str] = []
    _install_fake_hip(monkeypatch, log)
    _install_fake_torch_sync_debug(monkeypatch, [])
    calls = []

    def enqueue() -> None:
        calls.append(1)
        if len(calls) > 1:
            raise ExecutionError("kernel launch failed")

    with pytest.raises(ExecutionError, match="kernel launch failed"):
        measure(
            enqueue,
            stream=7,
            policy=TimingPolicy(warmup_iters=1, iters=1),
            torch_stream=object(),
        )


def test_staged_measure_releases_gate_when_enqueue_raises(monkeypatch) -> None:
    log: List[str] = []
    _install_fake_hip(monkeypatch, log)
    timer = StalledRegionTimer(stream=7)

    def boom() -> None:
        raise RuntimeError("enqueue failed")

    with pytest.raises(RuntimeError, match="enqueue failed"):
        timer.measure(boom)

    # Gate released after arm, device drained, stop event never waited on.
    assert log == ["arm", "record", "release", "device_sync"]


def test_missing_hipdnn_reports_hip_unavailable(monkeypatch) -> None:
    import builtins

    monkeypatch.setattr(timing_module, "hipdnn", None)
    real_import = builtins.__import__

    def blocking_import(name, *args, **kwargs):
        if name == "hipdnn_frontend":
            raise ImportError("blocked hipdnn_frontend")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocking_import)

    assert timing_module.is_hip_available() is False
    with pytest.raises(RuntimeError, match="not importable"):
        measure(lambda: None, stream=0, policy=TimingPolicy())


def test_block_timing_follows_rocke_protocol(monkeypatch) -> None:
    """timing_block=N follows rocKE time_launches / Solera measure(): every
    sample runs warmup_iters untimed executions, drains, then times N
    back-to-back executions in one event pair (elapsed / N); the first sample
    is discarded. The stall gate is skipped even where staging is available,
    because N gated enqueues can fill the HIP queue."""
    log: List[str] = []
    clock = FakeClock()
    fake = _install_fake_hip(monkeypatch, log, staged=True, clock=clock)
    elapsed = iter([40.0, 8.0, 12.0])
    fake.HipEvent.elapsed_time = lambda self, other: next(elapsed)

    m = measure(
        _enqueue(log, clock, ms=0.5),
        stream=7,
        policy=TimingPolicy(warmup_iters=2, iters=2, timing_block=4),
    )

    assert "arm" not in log
    sample = ["enqueue"] * 2 + ["device_sync", "record"]
    sample += ["enqueue"] * 4 + ["record", "event_sync"]
    priming = ["enqueue", "device_sync"]
    assert log == priming + sample * 3
    assert m.kernel_ms == [2.0, 3.0]
    assert m.host_ms == [pytest.approx(0.5), pytest.approx(0.5)]
    # One priming enqueue plus two untimed executions before each of 3 samples.
    assert (m.mode, m.timing_block, m.warmup_iters) == ("block", 4, 7)
    assert m.fallback_reason is None


def test_block_min_time_counts_every_execution_in_the_block(monkeypatch) -> None:
    """The --min-time-ms budget is device time: a block of 4 that takes 8 ms
    adds 8 ms, not its 2 ms per-execution average."""
    log: List[str] = []
    _install_fake_hip(monkeypatch, log, kernel_ms=8.0)

    m = measure(
        _enqueue(log),
        stream=7,
        policy=TimingPolicy(warmup_iters=0, iters=1, min_time_ms=9.0, timing_block=4),
    )

    assert m.kernel_ms == [2.0, 2.0]


def test_cold_cache_rejects_block_timing() -> None:
    """A flush before a block of N leaves only the first execution cold."""
    with pytest.raises(ValueError, match="--timing-block"):
        TimingPolicy(cache_mode="cold", timing_block=2)

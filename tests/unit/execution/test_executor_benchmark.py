# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""hipDNN Executor.benchmark / execute_once against a fake HIP runtime."""

import types
from typing import Any, Dict, List

import pytest

import dnn_benchmarking.execution.executor as executor_module
import dnn_benchmarking.execution.timing as timing_module
from dnn_benchmarking.common.exceptions import ExecutionError
from dnn_benchmarking.config.benchmark_config import TimingPolicy


class _Result:
    def __init__(self, message: str = "") -> None:
        self._message = message

    def is_bad(self) -> bool:
        return bool(self._message)

    def get_message(self) -> str:
        return self._message


class _Graph:
    def __init__(self, log: List[Any], fail: str = "") -> None:
        self._log = log
        self._fail = fail

    def execute(self, handle: Any, variant_pack: Dict[int, int], ws: int) -> _Result:
        self._log.append("execute")
        return _Result(self._fail)


class _Handle:
    def __init__(self, stream: int) -> None:
        self._stream = stream

    def get_stream(self) -> int:
        return self._stream


@pytest.fixture
def hip_log(monkeypatch) -> List[Any]:
    log: List[Any] = []

    class Event:
        def record(self, stream: int) -> None:
            log.append(("record", stream))

        def synchronize(self) -> None:
            log.append("event_sync")

        def elapsed_time(self, other) -> float:
            return 0.5

    class Gate:
        def arm(self, stream: int) -> None:
            log.append(("arm", stream))

        def release(self) -> None:
            log.append("release")

    monkeypatch.setattr(
        timing_module,
        "hipdnn",
        types.SimpleNamespace(
            HipEvent=Event,
            HipStallGate=Gate,
            hip_get_device_count=lambda: 1,
            hip_device_synchronize=lambda: log.append("device_sync"),
            hip_can_use_stream_wait_value=lambda: True,
        ),
    )
    return log


def _executor(log: List[Any], fail: str = "", **policy) -> executor_module.Executor:
    executor = executor_module.Executor("{}", TimingPolicy(**policy))
    executor._graph = _Graph(log, fail)
    return executor


def test_benchmark_times_on_the_handle_stream(hip_log) -> None:
    executor = _executor(hip_log, warmup_iters=2, iters=3)

    m = executor.benchmark(_Handle(123), {})

    assert m.mode == "staged" and m.backend == "hip"
    assert m.kernel_ms == [0.5, 0.5, 0.5]
    assert hip_log.count("execute") == 2 + 3
    streams = {entry[1] for entry in hip_log if isinstance(entry, tuple)}
    assert streams == {123}


def test_benchmark_execution_failure_is_execution_error(hip_log) -> None:
    executor = _executor(hip_log, fail="bad kernel", iters=1)

    with pytest.raises(ExecutionError, match="bad kernel"):
        executor.benchmark(_Handle(0), {})


def test_benchmark_without_prepare_raises() -> None:
    executor = executor_module.Executor("{}", TimingPolicy())

    with pytest.raises(ExecutionError, match="not prepared"):
        executor.benchmark(_Handle(0), {})


def test_handle_stream_change_after_prepare_raises(hip_log) -> None:
    """Reject hipDNN handle stream drift so every enqueue uses one stream."""
    executor = _executor(hip_log)
    executor._execution_stream = 123

    with pytest.raises(ExecutionError, match="stream changed"):
        executor.benchmark(_Handle(456), {})


def test_execute_once_resets_the_workspace_then_drains_the_device(hip_log) -> None:
    """A validation run must not read workspace left over from the timed loop."""
    executor = _executor(hip_log)
    executor._workspace = types.SimpleNamespace(
        zeros=lambda: hip_log.append("workspace_zeros")
    )

    executor.execute_once(_Handle(99), {})

    assert hip_log == ["workspace_zeros", "execute", "device_sync"]


def test_execute_once_sync_failure_is_execution_error(hip_log, monkeypatch) -> None:
    def failing_sync() -> None:
        raise RuntimeError("hipDeviceSynchronize: illegal address")

    monkeypatch.setattr(timing_module.hipdnn, "hip_device_synchronize", failing_sync)

    with pytest.raises(ExecutionError, match="illegal address"):
        _executor(hip_log).execute_once(_Handle(99), {})

# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""CPU-hermetic tests for the PyTorch GPU executor control flow."""

import importlib
import sys
import types
from contextlib import contextmanager
from typing import Any, List

import pytest

import dnn_benchmarking.execution.timing as timing_module
from dnn_benchmarking.common.exceptions import ExecutionError
from dnn_benchmarking.config.benchmark_config import TimingPolicy

# Import the reference handlers under real (CPU) torch so their module-level
# torch references resolve once and are cached. The fixtures below then swap
# in a minimal fake ``torch`` only for the executor (and timing) modules.
import dnn_benchmarking.execution.pytorch_ops  # noqa: E402,F401


class FakeStream:
    def __init__(self, ptr: int) -> None:
        self.cuda_stream = ptr


class FakeTorchEvent:
    def __init__(self, log: List[Any]) -> None:
        self._log = log

    def record(self, stream: Any) -> None:
        self._log.append(("torch_record", stream))

    def synchronize(self) -> None:
        pass

    def elapsed_time(self, other: Any) -> float:
        return 2.0


class FakeCuda:
    def __init__(self) -> None:
        self.log: List[Any] = []
        self.device_depth = 0
        self.active_stream: Any = None
        self.default_stream_obj = FakeStream(0xCAFE)
        self.sync_mode: Any = 0

    def is_available(self) -> bool:
        return True

    def init(self) -> None:
        pass

    def default_stream(self, device: Any) -> FakeStream:
        return self.default_stream_obj

    def current_stream(self) -> FakeStream:
        return self.default_stream_obj

    def synchronize(self) -> None:
        self.log.append("torch_device_sync")

    def Event(self, enable_timing: bool = False) -> FakeTorchEvent:
        return FakeTorchEvent(self.log)

    def get_sync_debug_mode(self) -> Any:
        return self.sync_mode

    def set_sync_debug_mode(self, mode: Any) -> None:
        self.sync_mode = mode

    @contextmanager
    def device(self, device: Any):
        self.device_depth += 1
        try:
            yield
        finally:
            self.device_depth -= 1

    @contextmanager
    def stream(self, stream: FakeStream):
        self.active_stream = stream
        try:
            yield
        finally:
            self.active_stream = None


def _install_fake_hip(monkeypatch, log: List[Any]) -> None:
    class Event:
        def record(self, stream: int) -> None:
            log.append(("hip_record", stream))

        def synchronize(self) -> None:
            pass

        def elapsed_time(self, other) -> float:
            return 1.0

    class Gate:
        def arm(self, stream: int) -> None:
            log.append(("arm", stream))

        def release(self) -> None:
            pass

    monkeypatch.setattr(
        timing_module,
        "hipdnn",
        types.SimpleNamespace(
            HipEvent=Event,
            HipStallGate=Gate,
            hip_get_device_count=lambda: 1,
            hip_device_synchronize=lambda: None,
            hip_can_use_stream_wait_value=lambda: True,
        ),
    )


class RecordingCompiled:
    """Fake CompiledGraph recording the tensor map of every replay."""

    def __init__(self, cuda: FakeCuda) -> None:
        self.cuda = cuda
        self.seen: List[Any] = []

    def execute(self, tensors: Any) -> None:
        # Work must be enqueued on the executor's stream on its device.
        assert self.cuda.device_depth > 0
        assert self.cuda.active_stream is self.cuda.default_stream_obj
        self.seen.append(tensors)


def _load_executor_module(monkeypatch, fake_cuda: FakeCuda, is_rocm: bool):
    fake_torch = types.ModuleType("torch")
    fake_torch.__path__ = []
    fake_torch.Tensor = object
    fake_torch.cuda = fake_cuda
    fake_torch.device = lambda device: device
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    module_name = "dnn_benchmarking.execution.pytorch_executor"
    old_module = sys.modules.pop(module_name, None)
    module = importlib.import_module(module_name)
    monkeypatch.setattr(module.torch_support, "gpu_available", lambda: True)
    monkeypatch.setattr(module.torch_support, "is_rocm_build", lambda: is_rocm)
    monkeypatch.setattr(
        module.pytorch_ops, "get_unsupported_operations", lambda graph: []
    )
    yield module

    sys.modules.pop(module_name, None)
    import dnn_benchmarking.execution as execution_pkg

    if old_module is not None:
        sys.modules[module_name] = old_module
        execution_pkg.pytorch_executor = old_module
    elif getattr(execution_pkg, "pytorch_executor", None) is module:
        del execution_pkg.pytorch_executor


@pytest.fixture
def fake_cuda() -> FakeCuda:
    return FakeCuda()


@pytest.fixture
def rocm_module(monkeypatch, fake_cuda):
    _install_fake_hip(monkeypatch, fake_cuda.log)
    yield from _load_executor_module(monkeypatch, fake_cuda, is_rocm=True)


@pytest.fixture
def cuda_module(monkeypatch, fake_cuda):
    monkeypatch.setattr(timing_module, "hipdnn", None)
    monkeypatch.setitem(sys.modules, "hipdnn_frontend", None)  # not importable
    yield from _load_executor_module(monkeypatch, fake_cuda, is_rocm=False)


def _prepared(module, monkeypatch, fake_cuda, sdpa="default", **policy):
    compiled = RecordingCompiled(fake_cuda)
    monkeypatch.setattr(module.pytorch_ops, "compile_graph", lambda graph: compiled)
    executor = module.PyTorchCudaExecutor(
        {"nodes": []},
        TimingPolicy(**{"warmup_iters": 1, "iters": 2, **policy}),
        pytorch_sdpa_backend=sdpa,
        device="cuda:1",
    )
    executor.prepare()
    return executor, compiled


def test_rocm_benchmark_is_staged_on_the_torch_stream(
    rocm_module, monkeypatch, fake_cuda
) -> None:
    executor, compiled = _prepared(rocm_module, monkeypatch, fake_cuda)

    m = executor.benchmark({1: "x"})

    assert (m.mode, m.backend) == ("staged", "hip")
    assert m.kernel_ms == [1.0, 1.0]
    hip_streams = {e[1] for e in fake_cuda.log if e[0] in ("hip_record", "arm")}
    assert hip_streams == {0xCAFE}


@pytest.mark.parametrize("syncs", [False, True], ids=["staged", "events"])
def test_every_replay_shares_one_replay_tensor_map(
    rocm_module, monkeypatch, fake_cuda, syncs: bool
) -> None:
    """Host reads are memoized on one ReplayTensors in both timing modes, so
    timed iterations never repeat a device->host sync."""
    executor, compiled = _prepared(rocm_module, monkeypatch, fake_cuda)
    if syncs:

        def execute(tensors: Any) -> None:
            compiled.seen.append(tensors)
            if fake_cuda.sync_mode == "error":
                raise RuntimeError("called a synchronizing CUDA operation")

        compiled.execute = execute

    m = executor.benchmark({1: "x"})

    assert m.mode == ("events" if syncs else "staged")
    assert all(isinstance(t, rocm_module.pytorch_ops.ReplayTensors) for t in compiled.seen)
    assert len({id(t) for t in compiled.seen}) == 1
    assert dict(compiled.seen[0]) == {1: "x"}


def test_host_sync_during_priming_selects_events_mode(
    rocm_module, monkeypatch, fake_cuda
) -> None:
    executor, compiled = _prepared(rocm_module, monkeypatch, fake_cuda)

    def execute(tensors: Any) -> None:
        if fake_cuda.sync_mode == "error":
            raise RuntimeError("called a synchronizing CUDA operation")

    compiled.execute = execute

    m = executor.benchmark({})

    assert m.mode == "events"
    assert m.fallback_reason.startswith("host sync in enqueue")
    assert fake_cuda.sync_mode == 0  # debug mode restored
    assert not any(e[0] == "arm" for e in fake_cuda.log if isinstance(e, tuple))


def test_cuda_build_times_with_torch_events(cuda_module, monkeypatch, fake_cuda) -> None:
    executor, _ = _prepared(cuda_module, monkeypatch, fake_cuda)

    m = executor.benchmark({})

    assert (m.mode, m.backend) == ("events", "torch")
    assert m.kernel_ms == [2.0, 2.0]
    assert m.fallback_reason
    recorded = {e[1] for e in fake_cuda.log if isinstance(e, tuple)}
    assert recorded == {fake_cuda.default_stream_obj}


def test_unavailable_sdpa_backend_error_propagates_unchanged(
    rocm_module, monkeypatch, fake_cuda
) -> None:
    executor, compiled = _prepared(rocm_module, monkeypatch, fake_cuda, sdpa="math")
    error = rocm_module.pytorch_ops.PyTorchSdpaBackendUnavailableError(
        "Requested PyTorch SDPA backend 'math' is unavailable; no fallback is used."
    )

    def execute(tensors: Any) -> None:
        raise error

    compiled.execute = execute

    with pytest.raises(rocm_module.pytorch_ops.PyTorchSdpaBackendUnavailableError) as e:
        executor.benchmark({})
    assert e.value is error


def test_nondefault_backend_requires_native_sdpa_execution(
    rocm_module, monkeypatch, fake_cuda
) -> None:
    executor, _ = _prepared(rocm_module, monkeypatch, fake_cuda, sdpa="math")

    with pytest.raises(
        rocm_module.pytorch_ops.PyTorchSdpaBackendUnavailableError,
        match="The graph did not execute a native forward SDPA call.",
    ):
        executor.benchmark({})


def test_execute_once_runs_on_stream_and_drains_it(
    rocm_module, monkeypatch, fake_cuda
) -> None:
    executor, compiled = _prepared(rocm_module, monkeypatch, fake_cuda)
    fake_cuda.log.clear()

    executor.execute_once({2: "y"})

    assert compiled.seen == [{2: "y"}]
    assert fake_cuda.log == [("hip_record", 0xCAFE)]


def test_benchmark_before_prepare_raises(rocm_module) -> None:
    executor = rocm_module.PyTorchCudaExecutor(
        {"nodes": []}, TimingPolicy(), pytorch_sdpa_backend="default"
    )

    with pytest.raises(ExecutionError, match="not prepared"):
        executor.benchmark({})

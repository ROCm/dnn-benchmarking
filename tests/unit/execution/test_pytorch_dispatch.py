# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""What a PyTorch row records as its dispatch: SDPA backend, ROCm Flash
library and GPU kernels, read from a Chrome trace of one launch."""

import enum

import pytest

from dnn_benchmarking.config import PyTorchSdpaBackendName
from dnn_benchmarking.execution import pytorch_dispatch
from dnn_benchmarking.execution.pytorch_ops import _sdpa_backend


def _op(name, ts):
    return {"ph": "X", "cat": "cpu_op", "name": name, "ts": ts, "dur": 1.0}


def _kernel(name, ts):
    return {"ph": "X", "cat": "kernel", "name": name, "ts": ts, "dur": 1.0}


# The events kineto writes for one Flash forward on ROCm, out of time order
# as in a real trace: metadata, CPU ops, the runtime launch call, a GPU-side
# annotation that repeats the op name, a flow event, a copy, a fill, and the
# kernel.
FLASH_FORWARD_TRACE = [
    {"ph": "M", "name": "process_name", "pid": 1, "args": {"name": "python"}},
    _kernel("attn_fwd", 220.0),
    _op("aten::scaled_dot_product_attention", 100.0),
    _op("aten::_scaled_dot_product_flash_attention", 101.0),
    _op("aten::_flash_attention_forward", 102.0),
    {"ph": "X", "cat": "cuda_runtime", "name": "hipLaunchKernel", "ts": 103.0},
    {"ph": "X", "cat": "gpu_user_annotation", "name": "aten::bmm", "ts": 200.0},
    {"ph": "s", "cat": "ac2g", "name": "ac2g", "ts": 103.0},
    {"ph": "X", "cat": "gpu_memcpy", "name": "Memcpy DtoD", "ts": 205.0},
    {"ph": "X", "cat": "gpu_memset", "name": "Memset (Device)", "ts": 210.0},
]


def test_kernels_are_the_distinct_gpu_kernels_in_order_of_first_launch():
    events = [
        _kernel("b", 300.0),
        _kernel("a", 100.0),
        _kernel("b", 200.0),
        _kernel("c", 400.0),
        _kernel("a", 500.0),
    ]
    assert pytorch_dispatch.kernel_names(events) == ["a", "b", "c"]


def test_copies_fills_and_host_events_are_not_kernels():
    assert pytorch_dispatch.kernel_names(FLASH_FORWARD_TRACE) == ["attn_fwd"]


@pytest.mark.parametrize(
    "ops, backend",
    [
        (
            [
                "aten::scaled_dot_product_attention",
                "aten::_scaled_dot_product_flash_attention",
                "aten::_flash_attention_forward",
            ],
            "flash",
        ),
        (
            [
                "aten::scaled_dot_product_attention",
                "aten::_scaled_dot_product_efficient_attention",
                "aten::_efficient_attention_forward",
            ],
            "efficient",
        ),
        (
            [
                "aten::scaled_dot_product_attention",
                "aten::_scaled_dot_product_cudnn_attention",
            ],
            "cudnn",
        ),
        (
            [
                "aten::scaled_dot_product_attention",
                "aten::_scaled_dot_product_attention_math",
                "aten::bmm",
                "aten::_softmax",
                "aten::bmm",
            ],
            "math",
        ),
        # A bound fused backward calls the ATen backward op directly.
        (
            [
                "aten::_scaled_dot_product_flash_attention_backward",
                "aten::_flash_attention_backward",
            ],
            "flash",
        ),
        (
            [
                "aten::_scaled_dot_product_efficient_attention_backward",
                "aten::_efficient_attention_backward",
            ],
            "efficient",
        ),
        # The backward formula of the tool and non-SDPA graphs call no
        # SDPA op.
        (["aten::bmm", "aten::exp", "aten::bmm"], None),
        (["aten::convolution", "aten::relu"], None),
    ],
)
def test_sdpa_backend_is_the_sdpa_op_that_ran(ops, backend):
    events = [_op(name, float(i)) for i, name in enumerate(ops)]
    assert pytorch_dispatch.sdpa_backend(events) == backend


def test_several_backends_are_listed_in_order_of_first_call():
    events = [
        _op("aten::_scaled_dot_product_flash_attention", 30.0),
        _op("aten::_scaled_dot_product_attention_math", 10.0),
        _op("aten::_scaled_dot_product_flash_attention", 20.0),
        # A GPU-side annotation names an op, but it is not a call.
        {"cat": "gpu_user_annotation", "name": "aten::_cudnn_attention_forward"},
    ]
    assert pytorch_dispatch.sdpa_backend(events) == "math,flash"


def test_a_real_trace_names_the_forced_backend():
    """The trace format of the installed torch, on the CPU: the launch runs
    the backend that it forces, and the record must name that backend."""
    torch = pytest.importorskip("torch")
    from torch.nn.attention import SDPBackend, sdpa_kernel

    q = torch.randn(1, 2, 16, 8)
    found = {}
    for forced, name in (
        (SDPBackend.MATH, "math"),
        (SDPBackend.FLASH_ATTENTION, "flash"),
    ):

        def launch():
            with sdpa_kernel(forced):
                torch.nn.functional.scaled_dot_product_attention(q, q, q)

        found[name] = pytorch_dispatch.sdpa_backend(
            pytorch_dispatch.profile_launch(launch)
        )
    assert found == {"math": "math", "flash": "flash"}


class _FaBackend(enum.Enum):
    """Stand-in for ``torch._C._ROCmFABackend``."""

    Default = 0
    AOTriton = 1
    Ck = 2


class _Preference:
    """``torch.backends.cuda.preferred_rocm_fa_library`` with process state."""

    def __init__(self, value=_FaBackend.Default, refused=()):
        self.value = value
        self.refused = set(refused)  # libraries this GPU cannot use

    def __call__(self, backend=None):
        if isinstance(backend, str):
            backend = {"default": _FaBackend.Default, "ck": _FaBackend.Ck}.get(
                backend, _FaBackend.AOTriton
            )
        if backend is not None and backend.name.lower() not in self.refused:
            self.value = backend
        return self.value


@pytest.fixture
def rocm(monkeypatch):
    import torch

    preference = _Preference()
    monkeypatch.setattr(_sdpa_backend.torch_support, "is_rocm_build", lambda: True)
    monkeypatch.setattr(
        torch.backends.cuda, "preferred_rocm_fa_library", preference, raising=False
    )
    return preference


def _state(selection=PyTorchSdpaBackendName.DEFAULT, library=None):
    return _sdpa_backend.PyTorchSdpaBackendState(selection, library)


@pytest.mark.parametrize(
    "process, expected",
    [
        (_FaBackend.Default, "aotriton"),  # PyTorch serves Default with AOTriton
        (_FaBackend.AOTriton, "aotriton"),
        (_FaBackend.Ck, "ck"),  # TORCH_ROCM_FA_PREFER_CK=1
    ],
)
def test_default_dispatch_records_the_process_preference(rocm, process, expected):
    rocm.value = process
    assert _sdpa_backend.rocm_fa_library(_state()) == expected


def test_strict_flash_with_a_library_records_it_and_restores_the_process(rocm):
    flash = _state(PyTorchSdpaBackendName.FLASH, "ck")

    assert _sdpa_backend.rocm_fa_library(flash) == "ck"
    assert rocm.value is _FaBackend.Default


def test_a_library_the_gpu_refuses_records_the_one_that_serves(rocm):
    rocm.refused = {"ck"}
    flash = _state(PyTorchSdpaBackendName.FLASH, "ck")

    assert _sdpa_backend.rocm_fa_library(flash) == "aotriton"


def test_no_fa_library_on_a_cuda_build(monkeypatch):
    monkeypatch.setattr(_sdpa_backend.torch_support, "is_rocm_build", lambda: False)
    assert _sdpa_backend.rocm_fa_library(_state()) is None

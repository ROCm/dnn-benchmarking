# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Strict PyTorch SDPA backend selection for a scoped graph execution."""

from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from threading import RLock
from typing import Any, Dict, Iterator, Optional, Tuple

import torch

from ...common import torch_support
from ...config.benchmark_config import PyTorchSdpaBackendName


class PyTorchSdpaBackendUnavailableError(RuntimeError):
    """A requested non-default PyTorch SDPA backend cannot execute."""


class PyTorchSdpaBackendState:
    """The requested strict SDPA category and ROCm Flash preference."""

    def __init__(
        self,
        selection: PyTorchSdpaBackendName,
        rocm_fa_library: Optional[str] = None,
    ) -> None:
        self.selection = PyTorchSdpaBackendName(selection)
        self.rocm_fa_library = rocm_fa_library
        self._native_forward_sdpa_executed = False


_ACTIVE_SDPA_BACKEND: ContextVar[Optional[PyTorchSdpaBackendState]] = ContextVar(
    "active_pytorch_sdpa_backend",
    default=None,
)

# torch.nn.attention.sdpa_kernel and preferred_rocm_fa_library mutate
# process-wide PyTorch state. Serialize every SDPA call so a default call cannot
# overlap and observe another thread's temporary selected-backend flags.
_SDPA_SELECTION_LOCK = RLock()

# The fused ROCm SDPA kernels (AOTriton flash and efficient) are not bitwise
# repeatable. On gfx950, fp16 D128 S4096 inputs that are identical run to run
# came back 0.8 to 7.8 fp16 ULP from an fp32 SDPA, a different amount each
# run, while MATH stayed at 0.5 ULP every run. A reference must not move, so
# the default-dispatch reference pass uses MATH. Strict selections keep the
# backend the user asked for.
_REFERENCE_PASS: ContextVar[bool] = ContextVar(
    "pytorch_sdpa_reference_pass",
    default=False,
)


@contextmanager
def reference_sdpa_pass() -> Iterator[None]:
    """Run default-dispatch SDPA calls on the repeatable MATH backend."""
    token = _REFERENCE_PASS.set(True)
    try:
        yield
    finally:
        _REFERENCE_PASS.reset(token)


@contextmanager
def use_pytorch_sdpa_backend(
    state: PyTorchSdpaBackendState,
) -> Iterator[None]:
    """Activate and require a native forward SDPA call for strict selections."""
    previous_execution = state._native_forward_sdpa_executed
    state._native_forward_sdpa_executed = False
    token = _ACTIVE_SDPA_BACKEND.set(state)
    try:
        yield
        if (
            state.selection is not PyTorchSdpaBackendName.DEFAULT
            and not state._native_forward_sdpa_executed
        ):
            raise _unavailable_error(
                state.selection,
                "The graph did not execute a native forward SDPA call.",
            )
    finally:
        state._native_forward_sdpa_executed = (
            previous_execution or state._native_forward_sdpa_executed
        )
        _ACTIVE_SDPA_BACKEND.reset(token)


def _unavailable_error(
    selection: PyTorchSdpaBackendName, reason: str
) -> PyTorchSdpaBackendUnavailableError:
    prefix = (
        f"Requested PyTorch SDPA backend '{selection.value}' is unavailable; "
        "no fallback is used."
    )
    return PyTorchSdpaBackendUnavailableError(f"{prefix} {reason}")


def _rocm_fa_library_unavailable_error(
    library: str, reason: str
) -> PyTorchSdpaBackendUnavailableError:
    prefix = (
        f"Requested PyTorch ROCm Flash Attention library '{library}' is "
        "unavailable; no fallback is used."
    )
    return PyTorchSdpaBackendUnavailableError(f"{prefix} {reason}")


def _selected_sdp_backend(
    selection: PyTorchSdpaBackendName,
) -> Tuple[object, object]:
    """Resolve one strict public PyTorch SDPA category."""

    try:
        from torch.nn import attention
    except ImportError as error:
        raise _unavailable_error(
            selection, "PyTorch does not expose torch.nn.attention."
        ) from error

    sdpa_kernel = getattr(attention, "sdpa_kernel", None)
    sdp_backend = getattr(attention, "SDPBackend", None)
    if not callable(sdpa_kernel) or sdp_backend is None:
        raise _unavailable_error(
            selection,
            "PyTorch does not expose the public sdpa_kernel API.",
        )

    member_names = {
        PyTorchSdpaBackendName.FLASH: "FLASH_ATTENTION",
        PyTorchSdpaBackendName.MATH: "MATH",
        PyTorchSdpaBackendName.EFFICIENT: "EFFICIENT_ATTENTION",
        PyTorchSdpaBackendName.CUDNN: "CUDNN_ATTENTION",
        PyTorchSdpaBackendName.OVERRIDEABLE: "OVERRIDEABLE",
    }
    member_name = member_names[selection]
    backend = getattr(sdp_backend, member_name, None)
    if backend is None:
        raise _unavailable_error(
            selection,
            f"PyTorch does not expose SDPBackend.{member_name}.",
        )
    return sdpa_kernel, backend


@contextmanager
def _prefer_rocm_fa_library(
    selection: PyTorchSdpaBackendName,
    library: Optional[str],
) -> Iterator[None]:
    """Temporarily set PyTorch's ROCm Flash implementation preference."""
    if library is None:
        yield
        return
    if selection is not PyTorchSdpaBackendName.FLASH:
        raise ValueError("ROCm Flash preference requires the Flash SDPA category")
    if not torch_support.is_rocm_build():
        raise _rocm_fa_library_unavailable_error(
            library,
            "The ROCm Flash Attention preference is available only in ROCm builds.",
        )

    cuda_backends = getattr(torch.backends, "cuda", None)
    preference = getattr(cuda_backends, "preferred_rocm_fa_library", None)
    if not callable(preference):
        raise _rocm_fa_library_unavailable_error(
            library,
            "PyTorch does not expose preferred_rocm_fa_library.",
        )

    try:
        previous = preference()
        preference(library)
    except (RuntimeError, ValueError) as error:
        raise _rocm_fa_library_unavailable_error(
            library,
            f"PyTorch could not set the ROCm Flash Attention preference: {error}",
        ) from error

    try:
        yield
    finally:
        preference(previous)


def _is_backend_unavailable_error(error: RuntimeError) -> bool:
    """Recognize PyTorch's forced-dispatch no-backend diagnostics."""
    message = str(error).lower()
    return "no available kernel" in message or "no viable backend" in message


def strict_sdpa_selection() -> Optional[PyTorchSdpaBackendName]:
    """The active strict SDPA category, or ``None`` under default dispatch."""
    state = _ACTIVE_SDPA_BACKEND.get()
    if state is None or state.selection is PyTorchSdpaBackendName.DEFAULT:
        return None
    return state.selection


def _sdpa_kwargs(
    attn_mask: Optional[torch.Tensor],
    dropout_p: float,
    is_causal: bool,
    scale: Optional[float],
    enable_gqa: bool,
) -> Dict[str, Any]:
    # enable_gqa needs torch >= 2.5, the documented floor (docs/setup.md).
    return dict(
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
        enable_gqa=enable_gqa,
    )


def default_sdpa_avoids_math(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    attn_mask: Optional[torch.Tensor],
    dropout_p: float,
    is_causal: bool,
    scale: Optional[float],
    enable_gqa: bool,
) -> bool:
    """Whether default dispatch picks a fused (Flash/Efficient) kernel.

    Default dispatch falls back to the math backend only when no fused kernel
    accepts the call, so this runs the call once restricted to the fused
    kernels. cuDNN is left out: its default priority relative to math varies
    across torch releases, so its acceptance proves nothing about the default.
    """
    from torch.nn.attention import SDPBackend, sdpa_kernel

    with _SDPA_SELECTION_LOCK:
        try:
            with sdpa_kernel(
                [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]
            ):
                torch.nn.functional.scaled_dot_product_attention(
                    query,
                    key,
                    value,
                    **_sdpa_kwargs(attn_mask, dropout_p, is_causal, scale, enable_gqa),
                )
        except RuntimeError as error:
            if _is_backend_unavailable_error(error):
                return False
            raise
    return True


# Where each fused forward saves its log-sum-exp for backward: Flash (GPU and
# CPU) uses ``logsumexp``, memory-efficient uses ``log_sumexp``.
_SAVED_LOGSUMEXP_ATTRS = ("_saved_logsumexp", "_saved_log_sumexp")


def fused_logsumexp(output: torch.Tensor, q_len: int) -> Optional[torch.Tensor]:
    """The log-sum-exp the fused SDPA forward saved for backward, ``[..., q_len, 1]``.

    A forward with ``requires_grad`` inputs makes PyTorch compute the
    log-sum-exp inside the attention kernel, as a training step does. It is
    read back from ``output.grad_fn``. ``None`` when the backend saves none
    (the math backend) or the call recorded no autograd node.
    """
    grad_fn = output.grad_fn
    for name in _SAVED_LOGSUMEXP_ATTRS:
        lse = getattr(grad_fn, name, None)
        if isinstance(lse, torch.Tensor):
            # The memory-efficient kernel may pad the query axis.
            return lse.detach()[..., :q_len].unsqueeze(-1)
    return None


def execute_selected_sdpa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    attn_mask: Optional[torch.Tensor],
    dropout_p: float,
    is_causal: bool,
    scale: Optional[float],
    enable_gqa: bool = False,
) -> torch.Tensor:
    """Execute SDPA with the current strict selection, if one is active.

    ``enable_gqa`` lets K/V carry fewer heads than Q (one shared count),
    PyTorch's own grouped-query path.
    """
    kwargs = _sdpa_kwargs(attn_mask, dropout_p, is_causal, scale, enable_gqa)
    state = _ACTIVE_SDPA_BACKEND.get()
    if state is None or state.selection is PyTorchSdpaBackendName.DEFAULT:
        backend_scope = nullcontext()
        if _REFERENCE_PASS.get():
            sdpa_kernel, math = _selected_sdp_backend(PyTorchSdpaBackendName.MATH)
            backend_scope = sdpa_kernel(math)
        with _SDPA_SELECTION_LOCK, backend_scope:
            return torch.nn.functional.scaled_dot_product_attention(
                query, key, value, **kwargs
            )

    sdpa_kernel, backend = _selected_sdp_backend(state.selection)
    with _SDPA_SELECTION_LOCK:
        try:
            with (
                _prefer_rocm_fa_library(
                    state.selection,
                    state.rocm_fa_library,
                ),
                sdpa_kernel(backend),
            ):
                result = torch.nn.functional.scaled_dot_product_attention(
                    query, key, value, **kwargs
                )
        except RuntimeError as error:
            if _is_backend_unavailable_error(error):
                raise _unavailable_error(state.selection, str(error)) from error
            raise
        state._native_forward_sdpa_executed = True
        return result

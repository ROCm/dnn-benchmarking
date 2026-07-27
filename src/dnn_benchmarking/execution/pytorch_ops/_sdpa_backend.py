# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Strict PyTorch SDPA backend selection for a scoped graph execution."""

from contextlib import contextmanager
from contextvars import ContextVar
from threading import RLock
from typing import Iterator, Optional, Tuple

import torch

from ...common import torch_support
from ...config.benchmark_config import PyTorchSdpaBackendName


class PyTorchSdpaBackendUnavailableError(RuntimeError):
    """A requested non-default PyTorch SDPA backend cannot execute."""


class PyTorchSdpaBackendState:
    """The requested selection and whether its native SDPA call completed."""

    def __init__(self, selection: PyTorchSdpaBackendName) -> None:
        self.selection = PyTorchSdpaBackendName(selection)
        self._native_forward_sdpa_executed = False


_ACTIVE_SDPA_BACKEND: ContextVar[Optional[PyTorchSdpaBackendState]] = ContextVar(
    "active_pytorch_sdpa_backend",
    default=None,
)

# torch.nn.attention.sdpa_kernel and preferred_rocm_fa_library mutate
# process-wide PyTorch state. Serialize every SDPA call so a default call cannot
# overlap and observe another thread's temporary selected-backend flags.
_SDPA_SELECTION_LOCK = RLock()


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


def _selected_sdp_backend(
    selection: PyTorchSdpaBackendName,
) -> Tuple[object, object]:
    """Resolve one strict public PyTorch SDPA category."""
    if (
        selection is PyTorchSdpaBackendName.AOTRITON_PREFERRED
        and not torch_support.is_rocm_build()
    ):
        raise _unavailable_error(
            selection, "The AOTriton preference is available only in ROCm builds."
        )

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
        PyTorchSdpaBackendName.AOTRITON_PREFERRED: "FLASH_ATTENTION",
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
def _prefer_aotriton(
    selection: PyTorchSdpaBackendName,
) -> Iterator[None]:
    """Temporarily prefer AOTriton for restricted Flash dispatch."""
    if selection is not PyTorchSdpaBackendName.AOTRITON_PREFERRED:
        yield
        return

    cuda_backends = getattr(torch.backends, "cuda", None)
    preference = getattr(cuda_backends, "preferred_rocm_fa_library", None)
    if not callable(preference):
        raise _unavailable_error(
            selection,
            "PyTorch does not expose preferred_rocm_fa_library.",
        )

    try:
        previous = preference()
        preference("aotriton")
    except (RuntimeError, ValueError) as error:
        raise _unavailable_error(
            selection,
            f"PyTorch could not set the AOTriton preference: {error}",
        ) from error

    try:
        yield
    finally:
        preference(previous)


def _is_backend_unavailable_error(error: RuntimeError) -> bool:
    """Recognize PyTorch's forced-dispatch no-backend diagnostics."""
    message = str(error).lower()
    return "no available kernel" in message or "no viable backend" in message


def execute_selected_sdpa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    attn_mask: Optional[torch.Tensor],
    dropout_p: float,
    is_causal: bool,
    scale: Optional[float],
) -> torch.Tensor:
    """Execute SDPA with the current strict selection, if one is active."""
    state = _ACTIVE_SDPA_BACKEND.get()
    if state is None or state.selection is PyTorchSdpaBackendName.DEFAULT:
        with _SDPA_SELECTION_LOCK:
            return torch.nn.functional.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
                scale=scale,
            )

    sdpa_kernel, backend = _selected_sdp_backend(state.selection)
    with _SDPA_SELECTION_LOCK:
        try:
            with _prefer_aotriton(state.selection), sdpa_kernel(backend):
                result = torch.nn.functional.scaled_dot_product_attention(
                    query,
                    key,
                    value,
                    attn_mask=attn_mask,
                    dropout_p=dropout_p,
                    is_causal=is_causal,
                    scale=scale,
                )
        except RuntimeError as error:
            if _is_backend_unavailable_error(error):
                raise _unavailable_error(state.selection, str(error)) from error
            raise
        state._native_forward_sdpa_executed = True
        return result

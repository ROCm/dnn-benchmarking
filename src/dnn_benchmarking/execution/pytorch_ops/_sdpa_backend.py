# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Strict PyTorch SDPA backend selection for a scoped graph execution."""

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator, Optional

import torch

from ...common import torch_support
from ...config.benchmark_config import PyTorchSdpaBackendName


class PyTorchSdpaBackendUnavailableError(RuntimeError):
    """A requested non-default PyTorch SDPA backend cannot execute."""


class PyTorchSdpaBackendState:
    """The PyTorch SDPA backend selection active for one execution scope."""

    def __init__(self, selection: PyTorchSdpaBackendName) -> None:
        self.selection = PyTorchSdpaBackendName(selection)


_ACTIVE_SDPA_BACKEND: ContextVar[Optional[PyTorchSdpaBackendState]] = ContextVar(
    "active_pytorch_sdpa_backend",
    default=None,
)


@contextmanager
def use_pytorch_sdpa_backend(
    state: PyTorchSdpaBackendState,
) -> Iterator[None]:
    """Activate an SDPA selection for the current context only."""
    token = _ACTIVE_SDPA_BACKEND.set(state)
    try:
        yield
    finally:
        _ACTIVE_SDPA_BACKEND.reset(token)


def _unavailable_error(
    selection: PyTorchSdpaBackendName, reason: str
) -> PyTorchSdpaBackendUnavailableError:
    prefix = (
        f"Requested PyTorch SDPA backend '{selection.value}' is unavailable; "
        "no fallback is used."
    )
    return PyTorchSdpaBackendUnavailableError(f"{prefix} {reason}")


def _selected_sdp_backend(selection: PyTorchSdpaBackendName):
    """Resolve a strict public PyTorch SDPA backend selection."""
    if (
        selection is PyTorchSdpaBackendName.AOTRITON
        and not torch_support.is_rocm_build()
    ):
        raise _unavailable_error(
            selection, "AOTriton is available only in ROCm builds."
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

    # AOTriton uses PyTorch's FLASH_ATTENTION selector on ROCm. The
    # `aotriton` choice adds the ROCm-only eligibility check; `flash` is generic.
    member_names = {
        PyTorchSdpaBackendName.AOTRITON: "FLASH_ATTENTION",
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
    try:
        with sdpa_kernel(backend):
            return torch.nn.functional.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
                scale=scale,
            )
    except RuntimeError as error:
        raise _unavailable_error(state.selection, str(error)) from error

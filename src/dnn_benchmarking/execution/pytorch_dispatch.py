# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Which SDPA backend and GPU kernels one PyTorch launch ran.

Under default dispatch, PyTorch picks the SDPA backend for each call, and
tuning (MIOpen find, TunableOp) can pick other kernels. A PyTorch row records
what one untimed launch of its timed path ran, as the plan's ``dispatch``
object (docs/results-schema.md).

This module imports torch only inside :func:`profile_launch`.
"""

import json
import os
import tempfile
import warnings
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional

#: The ATen ops that name the SDPA backend of a call, forward and backward.
#: The outer ``scaled_dot_product_attention`` op calls one of them.
SDPA_BACKEND_OPS: Dict[str, str] = {
    **dict.fromkeys(
        (
            "aten::_scaled_dot_product_flash_attention",
            "aten::_scaled_dot_product_flash_attention_backward",
            "aten::_scaled_dot_product_flash_attention_for_cpu",
            "aten::_scaled_dot_product_flash_attention_for_cpu_backward",
            "aten::_flash_attention_forward",
            "aten::_flash_attention_backward",
        ),
        "flash",
    ),
    **dict.fromkeys(
        (
            "aten::_scaled_dot_product_efficient_attention",
            "aten::_scaled_dot_product_efficient_attention_backward",
            "aten::_efficient_attention_forward",
            "aten::_efficient_attention_backward",
        ),
        "efficient",
    ),
    **dict.fromkeys(
        (
            "aten::_scaled_dot_product_cudnn_attention",
            "aten::_scaled_dot_product_cudnn_attention_backward",
            "aten::_cudnn_attention_forward",
            "aten::_cudnn_attention_backward",
        ),
        "cudnn",
    ),
    **dict.fromkeys(
        (
            "aten::_scaled_dot_product_fused_attention_overrideable",
            "aten::_scaled_dot_product_fused_attention_overrideable_backward",
        ),
        "overrideable",
    ),
    "aten::_scaled_dot_product_attention_math": "math",
}

#: Start of the row warning when a launch gives no dispatch record or no
#: kernel list. The oracle pass copies such warnings from the tuned child.
DISPATCH_NOT_RECORDED = "dispatch not recorded"

#: Kineto level above all of its messages. Kineto prints its own lines on
#: stderr at each profiler start and stop, which the console reserves for
#: progress, warnings and errors. A failed capture gets a row warning instead.
_KINETO_SILENT = "6"


def _in_order(events: Iterable[Mapping[str, Any]], category: str) -> List[str]:
    """Names of the trace events of ``category``, in start-time order."""
    matched = [e for e in events if str(e.get("cat", "")).lower() == category]
    matched.sort(key=lambda e: float(e.get("ts", 0.0)))
    return [str(e["name"]) for e in matched]


def kernel_names(events: Iterable[Mapping[str, Any]]) -> List[str]:
    """Distinct GPU kernel names of a Chrome trace, in order of first launch.

    Only ``kernel`` events count: copies, fills, runtime API calls and CPU
    ops have other categories.
    """
    return list(dict.fromkeys(_in_order(events, "kernel")))


def sdpa_backend(events: Iterable[Mapping[str, Any]]) -> Optional[str]:
    """The SDPA backend that the traced CPU ops called, or None.

    Several distinct backends are joined with ``,`` in order of first call.
    None when no SDPA backend op ran, also when a handler computes the
    attention with its own formula.
    """
    backends = (SDPA_BACKEND_OPS.get(name) for name in _in_order(events, "cpu_op"))
    return ",".join(dict.fromkeys(b for b in backends if b is not None)) or None


def profile_launch(launch: Callable[[], None]) -> List[Dict[str, Any]]:
    """Chrome trace events of one ``launch()``: CPU ops and GPU kernels.

    ``launch`` must synchronize its stream, so every kernel it starts ends
    inside the profiled span.

    Kineto runs one session per process, and a second one cancels the first.
    The check below sees a session of this thread, and a Python
    ``torch.profiler`` session of any thread on the PyTorch versions that keep
    the private ``torch.autograd.profiler._is_profiler_enabled`` flag. PyTorch
    has no public process-wide check, so ``--pytorch-dispatch`` is opt-in.

    Raises:
        RuntimeError: A profiler session is already active in this process.
            Kineto runs one session at a time, so a second one would stop the
            caller's session.
    """
    import torch
    from torch.autograd import profiler as autograd_profiler
    from torch.profiler import ProfilerActivity, profile

    if torch._C._autograd._profiler_enabled() or getattr(
        autograd_profiler, "_is_profiler_enabled", False
    ):
        raise RuntimeError("a torch.profiler session is already active")

    os.environ.setdefault("KINETO_LOG_LEVEL", _KINETO_SILENT)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            launch()
        with tempfile.TemporaryDirectory(prefix="dnn-bench-dispatch-") as tmp:
            path = os.path.join(tmp, "trace.json")
            prof.export_chrome_trace(path)
            with open(path) as f:
                return json.load(f)["traceEvents"]

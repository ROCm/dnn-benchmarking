# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Process-wide kernel-selection settings for the PyTorch path.

Only PyTorch-private switches are used. ``MIOPEN_FIND_MODE`` and
``MIOPEN_FIND_ENFORCE`` are deliberately NOT set: the hipDNN MIOpen plugin
reads them too, so they would also change the hipDNN rows being compared.
"""

import os
import sys
import tempfile
from typing import Dict, Optional

from . import torch_support

# Always on. Each value is read from the environment by PyTorch itself.
_DEFAULT_ENV = {
    # PyTorch's MIOpen conv path ignores channels-last strides unless this is
    # set, and instead transposes to NCHW inside the timed region.
    "PYTORCH_MIOPEN_SUGGEST_NHWC": "1",
    # AOTriton refuses architectures it flags experimental without this, which
    # drops SDPA to a slower backend.
    "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL": "1",
}

# Only for exhaustive mode. TunableOp tunes GEMMs on first use per shape.
_EXHAUSTIVE_ENV = {
    "PYTORCH_TUNABLEOP_ENABLED": "1",
    "PYTORCH_TUNABLEOP_TUNING": "1",
}

ENV_NAMES = (*_DEFAULT_ENV, *_EXHAUSTIVE_ENV, "PYTORCH_TUNABLEOP_FILENAME")
_CUDNN_BENCHMARK_KEY = "torch.backends.cudnn.benchmark"


def apply_pytorch_environment(
    *, exhaustive: bool, cache_dir: Optional[str] = None
) -> Dict[str, str]:
    """Set PyTorch's ROCm kernel-selection controls and return what is in effect.

    Values already present in the environment win, so a caller can opt out.
    Must run before the first GEMM or SDPA call, because PyTorch caches these.

    Args:
        exhaustive: Also enable MIOpen exhaustive conv search
            (``torch.backends.cudnn.benchmark``) and TunableOp GEMM tuning.
        cache_dir: Where TunableOp writes its results; a fresh temp directory
            when None, so a stale CSV in the working directory is never read.
    """
    env = dict(_DEFAULT_ENV)
    if exhaustive:
        env.update(_EXHAUSTIVE_ENV)
        base = cache_dir or tempfile.mkdtemp(prefix="dnn-bench-tunableop-")
        env["PYTORCH_TUNABLEOP_FILENAME"] = os.path.join(base, "tunableop_results.csv")
    for name, value in env.items():
        os.environ.setdefault(name, value)
    effective = {name: os.environ[name] for name in env}
    if exhaustive and torch_support.module_available():
        import torch

        # Passes exhaustiveSearch=true to MIOpen's Find for PyTorch convs only.
        torch.backends.cudnn.benchmark = True
        effective[_CUDNN_BENCHMARK_KEY] = "True"
    return effective


def pytorch_environment_snapshot() -> Optional[Dict[str, Optional[str]]]:
    """Return the settings in effect, or None when none were applied."""
    snapshot: Dict[str, Optional[str]] = {n: os.environ.get(n) for n in ENV_NAMES}
    torch = sys.modules.get("torch")
    if torch is not None and getattr(torch.backends.cudnn, "benchmark", False):
        snapshot[_CUDNN_BENCHMARK_KEY] = "True"
    return snapshot if any(v is not None for v in snapshot.values()) else None

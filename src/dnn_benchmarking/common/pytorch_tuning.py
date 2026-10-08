# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Kernel-selection settings for the PyTorch path.

OOTB PyTorch runs in the benchmark process with layout and backend fixes only
(:func:`apply_pytorch_environment`). Tuned PyTorch runs in a child process
(:func:`tuned_subprocess_env`, :func:`enable_tuned_pytorch`) because PyTorch
keeps conv algorithm choices in a process-wide cache whose key ignores
``cudnn.benchmark``, and MIOpen persists search results in its user database.
Running both in one process would let either measurement inherit the other's
selection.

``MIOPEN_FIND_MODE`` and ``MIOPEN_FIND_ENFORCE`` are deliberately NOT set: the
hipDNN MIOpen plugin reads them too, so they would also change hipDNN rows.
"""

import os
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

ENV_NAMES = tuple(_DEFAULT_ENV)


def apply_pytorch_environment() -> Dict[str, str]:
    """Set the always-on PyTorch ROCm controls and return what is in effect.

    Values already present in the environment win, so a caller can opt out.
    Must run before the first conv or SDPA call, because PyTorch caches these.
    """
    for name, value in _DEFAULT_ENV.items():
        os.environ.setdefault(name, value)
    return {name: os.environ[name] for name in _DEFAULT_ENV}


def tuned_subprocess_env(state_dir: str) -> Dict[str, str]:
    """Return the environment for the isolated tuned-PyTorch child process.

    TunableOp tunes GEMMs on first use. MIOpen's user database points at
    ``state_dir``, so the exhaustive conv search neither reuses earlier tuning
    nor leaves entries that a later OOTB run, PyTorch or hipDNN, would read.
    These are forced, not defaulted: an inherited value would break isolation.
    """
    env = dict(os.environ)
    env.update(
        {
            "PYTORCH_TUNABLEOP_ENABLED": "1",
            "PYTORCH_TUNABLEOP_TUNING": "1",
            "PYTORCH_TUNABLEOP_FILENAME": os.path.join(
                state_dir, "tunableop_results.csv"
            ),
            "MIOPEN_USER_DB_PATH": state_dir,
        }
    )
    return env


def enable_tuned_pytorch() -> None:
    """Make PyTorch's MIOpen Find run an exhaustive search (child process only)."""
    if torch_support.module_available():
        import torch

        torch.backends.cudnn.benchmark = True


def pytorch_environment_snapshot() -> Optional[Dict[str, Optional[str]]]:
    """Return the always-on settings in effect, or None when none were set."""
    snapshot = {n: os.environ.get(n) for n in ENV_NAMES}
    return snapshot if any(v is not None for v in snapshot.values()) else None

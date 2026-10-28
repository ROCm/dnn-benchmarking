# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Kernel-selection settings for the PyTorch path.

OOTB PyTorch runs in the benchmark process with layout and backend fixes only
(:func:`apply_pytorch_environment`). Tuned PyTorch runs in a child process
(``cli.pytorch_tuned_child``, environment
``execution.child_protocol.PYTORCH_TUNED_ENV``) because PyTorch keeps conv
algorithm choices in a process-wide cache whose key ignores
``cudnn.benchmark``, and MIOpen persists search results in its user database.
Running both in one process would let either measurement inherit the other's
selection.

``MIOPEN_FIND_MODE`` and ``MIOPEN_FIND_ENFORCE`` are deliberately NOT set: the
hipDNN MIOpen plugin reads them too, so they would also change hipDNN rows.
"""

import os
from typing import Dict

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
    """Set the always-on PyTorch ROCm controls and return them.

    Forced, not defaulted: the CLI is the only control, so a value inherited
    from the shell cannot change what PyTorch rows measure.
    Must run before the first conv or SDPA call, because PyTorch caches these.
    """
    os.environ.update(_DEFAULT_ENV)
    return dict(_DEFAULT_ENV)

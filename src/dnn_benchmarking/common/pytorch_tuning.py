# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Kernel-selection settings for the PyTorch path.

Every PyTorch GPU run is a child process, ``cli.pytorch_child``: the OOTB row
and the reference row (``execution.child_protocol.PYTORCH_OOTB``) and their
tuned runs (``--tuned``, ``PYTORCH_TUNED``). The parent sets none of
these variables in its own environment. A tuned run cannot share a process
with an OOTB run: PyTorch keeps conv algorithm choices in a process-wide
cache whose key ignores ``cudnn.benchmark``, and MIOpen persists search
results in its user database.

``MIOPEN_FIND_MODE`` and ``MIOPEN_FIND_ENFORCE`` are deliberately NOT set: the
hipDNN MIOpen plugin reads them too, so they would also change hipDNN rows.
"""

#: Always on in every PyTorch child. PyTorch reads each value from the
#: environment itself, before the first conv or SDPA call.
PYTORCH_SELECTION_ENV = {
    # PyTorch's MIOpen conv path ignores channels-last strides unless this is
    # set, and instead transposes to NCHW inside the timed region.
    "PYTORCH_MIOPEN_SUGGEST_NHWC": "1",
    # AOTriton refuses architectures it flags experimental without this, which
    # drops SDPA to a slower backend.
    "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL": "1",
}

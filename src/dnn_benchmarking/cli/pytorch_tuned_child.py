# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Entry point of the tuned-PyTorch child process.

``python -m dnn_benchmarking.cli.pytorch_tuned_child <dnn-benchmark args>``
turns on PyTorch's exhaustive conv search, then runs the normal CLI on the
tuned selection state of ``--benchmark-work-dir`` (``common.work_dir``). The
parent (``execution.oracle``) supplies the isolated tuning environment and
the arguments for one timed ``--runtime pytorch`` row.
"""

import sys

import torch

from .main import main

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    sys.exit(main(tuned_child=True))

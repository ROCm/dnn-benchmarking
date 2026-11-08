# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Execution module for dnn-benchmarking."""

from .buffer_manager import BufferManager
from .executor import Executor

# PyTorch classes removed from top-level imports to avoid ROCm version conflicts.
# Import directly from submodules when needed:
#   from dnn_benchmarking.execution.pytorch_buffer_manager import PyTorchCudaBufferManager
#   from dnn_benchmarking.execution.pytorch_executor import PyTorchCudaExecutor

from .timing import Measurement, Timer, measure

__all__ = [
    "BufferManager",
    "Executor",
    "Measurement",
    "Timer",
    "measure",
]

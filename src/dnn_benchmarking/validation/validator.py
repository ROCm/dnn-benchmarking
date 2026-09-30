# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tolerance binding for :func:`compare`, kept for the suite runner's call sites."""

from typing import Any

from ..graph.tensor_info import TensorInfo
from .comparison import ComparisonResult, compare


class Validator:
    """Compares an output against its reference with fixed tolerances.

    ``validate`` accepts NumPy arrays or torch tensors (compared on their
    device); ``validate_tensors`` is the same call.
    """

    def __init__(self, rtol: float, atol: float) -> None:
        self._rtol = rtol
        self._atol = atol

    def validate(
        self, output_data: Any, tensor_info: TensorInfo, reference_data: Any
    ) -> ComparisonResult:
        """Compare ``output_data`` against ``reference_data``."""
        return compare(output_data, reference_data, rtol=self._rtol, atol=self._atol)

    validate_tensors = validate

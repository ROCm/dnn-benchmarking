# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Validation module for dnn-benchmarking."""

from .comparison import ComparisonResult, compare, compare_normalized_rms
from .reference_provider import (
    ReferenceOutput,
    ReferenceProvider,
    ReferenceProviderRegistry,
)

# Import providers to register them with the registry
from . import providers  # noqa: F401

__all__ = [
    "ComparisonResult",
    "ReferenceOutput",
    "ReferenceProvider",
    "ReferenceProviderRegistry",
    "compare",
    "compare_normalized_rms",
]

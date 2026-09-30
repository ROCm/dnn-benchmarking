# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for ReferenceProviderRegistry lookup."""

import pytest

from dnn_benchmarking.validation import ReferenceProviderRegistry


def test_get_pytorch_provider() -> None:
    assert ReferenceProviderRegistry.get_provider("pytorch").name == "pytorch"


def test_get_unknown_provider_raises() -> None:
    with pytest.raises(ValueError, match="Unknown reference provider"):
        ReferenceProviderRegistry.get_provider("unknown_provider")

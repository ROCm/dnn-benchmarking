# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for ReferenceProviderRegistry lookup and reference input dtypes."""

import numpy as np
import pytest

from dnn_benchmarking.common.exceptions import UnsupportedGraphError
from dnn_benchmarking.validation import ReferenceProviderRegistry


def test_get_pytorch_provider() -> None:
    assert ReferenceProviderRegistry.get_provider("pytorch").name == "pytorch"


def test_get_unknown_provider_raises() -> None:
    with pytest.raises(ValueError, match="Unknown reference provider"):
        ReferenceProviderRegistry.get_provider("unknown_provider")


def _relu_graph(**graph):
    return {
        **graph,
        "tensors": [
            {"uid": 1, "name": "x", "dims": [2], "data_type": "unset"},
            {"uid": 2, "name": "y", "dims": [2], "data_type": "unset"},
        ],
        "nodes": [
            {
                "type": "PointwiseAttributes",
                "inputs": {"operation": "relu_fwd", "in_0_tensor_uid": 1},
                "outputs": {"out_0_tensor_uid": 2},
            }
        ],
    }


def test_unset_input_dtype_falls_back_to_io_data_type() -> None:
    # hipDNN fills an "unset" physical tensor from io_data_type, as GraphLoader does.
    pytest.importorskip("torch")
    provider = ReferenceProviderRegistry.get_provider("pytorch")
    x = np.array([1.001, -1.0], dtype=np.float32)  # 1.001 rounds to 1.0 in bf16
    out = provider.compute_reference(_relu_graph(io_data_type="bfloat16"), {1: x})
    np.testing.assert_array_equal(out[2].data, [1.0, 0.0])
    with pytest.raises(UnsupportedGraphError, match="unset"):
        provider.compute_reference(_relu_graph(), {1: x})

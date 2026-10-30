# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for BufferManager use before allocation."""

import pytest

from dnn_benchmarking.common.exceptions import ExecutionError
from dnn_benchmarking.execution.buffer_manager import BufferManager
from dnn_benchmarking.graph.tensor_info import TensorInfo


@pytest.mark.parametrize(
    "call",
    [
        lambda bm: bm.create_variant_pack(),
        lambda bm: bm.load_input_data({}),
        lambda bm: bm.zero_outputs(),
    ],
    ids=["create_variant_pack", "load_input_data", "zero_outputs"],
)
def test_use_before_allocate_all_raises(call) -> None:
    output = TensorInfo(
        uid=1,
        name="t1",
        dims=[2, 3],
        strides=[3, 1],
        data_type="float",
        is_virtual=False,
        is_output=True,
    )
    bm = BufferManager([output])

    with pytest.raises(ExecutionError, match="Buffers not allocated"):
        call(bm)

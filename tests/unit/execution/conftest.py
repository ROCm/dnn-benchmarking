# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Fixtures shared by the execution unit tests."""

import sys

import pytest


@pytest.fixture(autouse=True)
def _forget_tuned_winners():
    """Each test is its own process run: no tuned winner from an earlier test.

    Looked up in sys.modules, not imported: some tests here install fake
    modules that must load before the package does.
    """
    yield
    oracle = sys.modules.get("dnn_benchmarking.execution.oracle")
    if oracle is not None:
        oracle._TUNED_IN_PROCESS.clear()

# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""rocKE runtime: catalog variants build in a torch-free child process."""

import pytest

from dnn_benchmarking.execution.rocke_build import BuiltKernel, build_dense_variants

pytest.importorskip("kernels.gfx950.attention_dense")

_SPEC = {
    "dtype": "fp16",
    "head_size": 64,
    "num_query_heads": 8,
    "num_kv_heads": 1,
    "causal": True,
    "batch": 1,
    "seqlen_q": 512,
    "seqlen_kv": 512,
    "block_m": 128,
    "block_n": 64,
}


def test_a_failing_variant_does_not_fail_the_others() -> None:
    """Answers come back in spec order; a bad spec fails only its own row."""
    built = build_dense_variants([dict(_SPEC, block_m=96), _SPEC])

    assert isinstance(built[0], str) and built[0]
    good = built[1]
    assert isinstance(good, BuiltKernel)
    assert good.hsaco.startswith(b"\x7fELF")
    assert good.kernel_name
    assert good.comgr

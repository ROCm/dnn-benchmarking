# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""The dtype registry agrees with every consumer: storage, torch, hipDNN."""

import numpy as np
import pytest

from dnn_benchmarking.common.dtypes import get_dtype
from dnn_benchmarking.common.exceptions import UnsupportedGraphError
from dnn_benchmarking.execution.buffer_manager import (
    _decode_storage_bytes,
    _encode_to_storage_bytes,
    _f32_to_fp8,
    _fp8_values,
)
from dnn_benchmarking.graph.tensor_info import TensorInfo

# Every data type hipDNN graphs use that the executor and buffers must handle,
# as graph name -> (torch dtype name, host numpy dtype).
EXPECTED = {
    "float": ("float32", np.float32),
    "half": ("float16", np.float16),
    "bfloat16": ("bfloat16", np.float32),
    "double": ("float64", np.float64),
    "int8": ("int8", np.int8),
    "uint8": ("uint8", np.uint8),
    "int32": ("int32", np.int32),
    "int64": ("int64", np.int64),
    "boolean": ("bool", np.bool_),
    "fp8_e4m3": ("float8_e4m3fn", np.float32),
    "fp8_e5m2": ("float8_e5m2", np.float32),
    "fp8_e8m0": ("float8_e8m0fnu", np.float32),
    "fp8_e4m3_fnuz": ("float8_e4m3fnuz", np.float32),
    "fp8_e5m2_fnuz": ("float8_e5m2fnuz", np.float32),
}
NAMES = list(EXPECTED)
FP8 = {name: torch_name for name, (torch_name, _) in EXPECTED.items() if "fp8" in name}


@pytest.mark.parametrize("name", NAMES)
def test_storage_bytes_match_registry_size(name: str) -> None:
    tensor = TensorInfo(1, "t", [3, 5], [], name.upper(), is_virtual=False)
    values = np.ones((3, 5), dtype=tensor.dtype.numpy)

    raw = _encode_to_storage_bytes(values, tensor)

    assert len(raw) == tensor.size_bytes == 15 * get_dtype(name).size
    np.testing.assert_array_equal(_decode_storage_bytes(raw, tensor), values)


@pytest.mark.parametrize("name", NAMES)
def test_registry_maps_to_the_matching_torch_and_numpy_dtypes(name: str) -> None:
    """A swapped mapping (e.g. half -> bfloat16) would silently compute the
    PyTorch reference in the wrong precision."""
    torch_name, numpy_dtype = EXPECTED[name]
    dtype = get_dtype(name)
    assert dtype.numpy == np.dtype(numpy_dtype)

    torch = pytest.importorskip("torch")
    if not hasattr(torch, torch_name):
        pytest.skip(f"torch {torch.__version__} has no {torch_name}")
    assert dtype.torch_dtype() is getattr(torch, torch_name)
    assert dtype.torch_dtype().itemsize == dtype.size


def test_upper_case_names_are_hipdnn_enum_members() -> None:
    """The executor maps graph dtype names to hipDNN via ``name.upper()``."""
    try:
        import hipdnn_frontend as hipdnn
    except Exception:  # ImportError, or OSError without the ROCm libraries
        pytest.skip("hipdnn_frontend not importable")

    for name in NAMES:
        assert hasattr(hipdnn.DataType, get_dtype(name).name.upper()), name


@pytest.mark.parametrize("name", ["fp4_e2m1", "unset", "float32", "", None])
def test_unknown_dtype_raises_unsupported(name) -> None:
    with pytest.raises(UnsupportedGraphError):
        get_dtype(name)


@pytest.mark.parametrize("name", sorted(FP8))
def test_fp8_decode_matches_torch_for_every_code(name: str) -> None:
    torch = pytest.importorskip("torch")
    if not hasattr(torch, FP8[name]):
        pytest.skip(f"torch {torch.__version__} has no {FP8[name]}")
    codes = torch.arange(256, dtype=torch.int32).to(torch.uint8)
    expected = codes.view(getattr(torch, FP8[name])).float().numpy()

    np.testing.assert_array_equal(_fp8_values(name), expected)  # NaNs compare equal


@pytest.mark.parametrize("name", sorted(FP8))
def test_fp8_encode_rounds_to_nearest_code(name: str) -> None:
    grid = np.unique(_fp8_values(name)[np.isfinite(_fp8_values(name))])
    probes = np.random.default_rng(0).uniform(grid[0], grid[-1], 4096)
    probes = probes.astype(np.float32)

    decoded = _fp8_values(name)[_f32_to_fp8(probes, name)]

    nearest = grid[np.abs(probes[:, None] - grid[None, :]).argmin(axis=1)]
    np.testing.assert_array_equal(np.abs(decoded - probes), np.abs(nearest - probes))

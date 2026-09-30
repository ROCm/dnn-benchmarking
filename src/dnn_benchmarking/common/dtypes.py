# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""The one table of tensor data types the benchmark understands.

Keys are the lower-case names hipDNN writes in graph JSON ("float", "bfloat16",
"fp8_e4m3", ...). Every consumer (tensor sizes, host buffers, torch tensors,
hipDNN enums) resolves through :func:`get_dtype`, and an unknown name raises
instead of silently becoming a 4-byte float.
"""

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from .exceptions import UnsupportedGraphError


@dataclass(frozen=True)
class DType:
    """One graph data type; ``name.upper()`` is its ``hipdnn_frontend.DataType``.

    Attributes:
        name: Graph JSON name, lower case.
        size: Bytes per element in device storage.
        numpy: Host dtype of the logical values. NumPy has no bfloat16 or fp8,
            so those are exchanged as float32 values that are exactly
            representable in the graph type.
        torch_name: ``torch`` attribute name, or None when torch has no match.
    """

    name: str
    size: int
    numpy: np.dtype
    torch_name: Optional[str]

    def torch_dtype(self) -> Any:
        """The torch dtype; raises UnsupportedGraphError if this torch lacks it."""
        import torch

        dtype = getattr(torch, self.torch_name, None) if self.torch_name else None
        if dtype is None:
            raise UnsupportedGraphError(
                f"torch {torch.__version__} has no dtype for data type '{self.name}'"
            )
        return dtype


_F32 = np.dtype(np.float32)

_DTYPES = {
    dtype.name: dtype
    for dtype in (
        DType("float", 4, _F32, "float32"),
        DType("half", 2, np.dtype(np.float16), "float16"),
        DType("bfloat16", 2, _F32, "bfloat16"),
        DType("double", 8, np.dtype(np.float64), "float64"),
        DType("int8", 1, np.dtype(np.int8), "int8"),
        DType("uint8", 1, np.dtype(np.uint8), "uint8"),
        DType("int32", 4, np.dtype(np.int32), "int32"),
        DType("int64", 8, np.dtype(np.int64), "int64"),
        DType("boolean", 1, np.dtype(np.bool_), "bool"),
        DType("fp8_e4m3", 1, _F32, "float8_e4m3fn"),
        DType("fp8_e5m2", 1, _F32, "float8_e5m2"),
        DType("fp8_e8m0", 1, _F32, "float8_e8m0fnu"),
    )
}


def get_dtype(name: str) -> DType:
    """Look up a graph data type by name (case-insensitive).

    Raises:
        UnsupportedGraphError: If the name is not a supported data type.
    """
    dtype = _DTYPES.get(str(name).lower())
    if dtype is None:
        raise UnsupportedGraphError(
            f"Unsupported tensor data type '{name}' "
            f"(supported: {', '.join(_DTYPES)})"
        )
    return dtype

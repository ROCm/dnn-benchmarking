# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tensor information dataclass."""

from dataclasses import dataclass, field
from typing import Any, List, Optional

from ..common.dtypes import DType, get_dtype
from ..common.exceptions import GraphLoadError


@dataclass
class TensorInfo:
    """Information about a tensor extracted from graph JSON.

    Attributes:
        uid: Unique identifier for the tensor.
        name: Human-readable name of the tensor.
        dims: Dimensions of the tensor (e.g., [N, C, H, W]).
        data_type: Data type name from the graph (e.g., "float", "half").
        is_virtual: Whether this is a virtual (intermediate) tensor.
        is_output: Whether this tensor is marked as a graph output.
        value: Embedded scalar for pass-by-value tensors, else None.
        dtype: Registry entry for ``data_type``; construction raises
            UnsupportedGraphError for an unknown type.
    """

    uid: int
    name: str
    dims: List[int]
    strides: List[int]
    data_type: str
    is_virtual: bool
    is_output: bool = False
    value: Optional[Any] = None
    dtype: DType = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.dtype = get_dtype(self.data_type)

    @property
    def element_size(self) -> int:
        """Get size of one element in bytes."""
        return self.dtype.size

    @property
    def num_elements(self) -> int:
        """Get total number of elements."""
        result = 1
        for dim in self.dims:
            result *= dim
        return result

    @property
    def is_pass_by_value(self) -> bool:
        """Whether this tensor carries an embedded scalar value."""
        return self.value is not None

    @property
    def storage_elements(self) -> int:
        """Get the number of storage elements required by dims/strides."""
        if self.num_elements == 0:
            return 0
        if self.strides:
            if len(self.strides) != len(self.dims):
                raise ValueError(
                    f"Tensor {self.uid} has {len(self.dims)} dims but "
                    f"{len(self.strides)} strides"
                )
            return (
                sum((dim - 1) * stride for dim, stride in zip(self.dims, self.strides))
                + 1
            )
        return self.num_elements

    @property
    def size_bytes(self) -> int:
        """Get total storage footprint in bytes."""
        return self.storage_elements * self.element_size

    @classmethod
    def from_json(cls, tensor_json: dict) -> "TensorInfo":
        """Create TensorInfo from a JSON tensor object (``is_output`` False).

        Args:
            tensor_json: Dictionary containing tensor attributes from graph JSON.

        Returns:
            TensorInfo instance.

        Raises:
            GraphLoadError: If a required field is missing or malformed.
            UnsupportedGraphError: If the data type is not supported.
        """
        try:
            return cls(
                uid=int(tensor_json["uid"]),
                name=tensor_json.get("name", f"tensor_{tensor_json['uid']}"),
                dims=[int(d) for d in tensor_json["dims"]],
                strides=[int(s) for s in tensor_json.get("strides") or []],
                data_type=tensor_json["data_type"],
                is_virtual=bool(tensor_json.get("virtual", False)),
                value=tensor_json.get("value"),
            )
        except (KeyError, TypeError, ValueError, AttributeError) as e:
            raise GraphLoadError(
                f"Malformed tensor entry {tensor_json!r:.200}: {type(e).__name__}: {e}"
            ) from e

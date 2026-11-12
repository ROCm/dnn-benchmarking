# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tensor information dataclass."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..common.dtypes import DType, get_dtype
from ..common.exceptions import GraphLoadError, UnsupportedGraphError


def tensor_label(uid: int, name: Optional[str] = None) -> str:
    """How messages name a tensor: ``tensor '<name>' (uid <uid>)``.

    A tensor without a name is ``tensor_<uid>``, as in ``TensorInfo.from_json``.
    """
    return f"tensor '{f'tensor_{uid}' if name is None else name}' (uid {uid})"


def ragged_fields(tensor_json: Dict[str, Any]) -> Tuple[Optional[int], int]:
    """``(ragged_offset_tensor_uid, ragged_offset_multiplier)`` of a tensor's JSON.

    The UID is None for a dense or padded tensor. A missing multiplier is 1;
    callers refuse one below 1.
    """
    offset_uid = tensor_json.get("ragged_offset_tensor_uid")
    multiplier = tensor_json.get("ragged_offset_multiplier")
    return (
        None if offset_uid is None else int(offset_uid),
        1 if multiplier is None else int(multiplier),
    )


@dataclass(frozen=True)
class VarlenLayout:
    """Per-batch valid lengths along ``dims[-2]``, and for a ragged tensor
    where each batch starts.

    Attributes:
        lengths: One valid length per batch (``dims[0]`` entries).
        starts: Element offset of batch ``b`` (``off[b] * multiplier``); None
            for a padded tensor, where batch ``b`` starts at ``b * strides[0]``.
        end: ``off[B] * multiplier``; an int whenever ``starts`` is set.
    """

    lengths: Tuple[int, ...]
    starts: Optional[Tuple[int, ...]] = None
    end: Optional[int] = None


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
        ragged_offset_tensor_uid: UID of the offset table that places each batch
            in storage, or None for a dense or padded tensor.
        ragged_offset_multiplier: Elements per offset-table unit.
        varlen: Valid lengths (and batch starts when ragged), set by
            ``resolve_varlen_layouts``; None until resolved.
        matching_infinities: Whether the comparison accepts infinities that
            match in both tensors, set by ``resolve_varlen_layouts`` on the
            ``stats`` output of each SDPA forward node.
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
    ragged_offset_tensor_uid: Optional[int] = None
    ragged_offset_multiplier: int = 1
    varlen: Optional[VarlenLayout] = field(default=None, compare=False)
    matching_infinities: bool = field(default=False, compare=False)
    dtype: DType = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.dtype = get_dtype(self.data_type)

    @property
    def label(self) -> str:
        """How messages name this tensor (``tensor_label``)."""
        return tensor_label(self.uid, self.name)

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
        """Get the number of storage elements required by dims/strides.

        A ragged tensor needs its resolved ``varlen`` layout: storage reaches
        the end of the offset table or the last valid element, whichever is
        higher.
        """
        if self.is_ragged:
            return self._ragged_storage_elements()
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

    def _ragged_storage_elements(self) -> int:
        if self.varlen is None or self.varlen.starts is None:
            raise UnsupportedGraphError(
                f"Ragged {self.label} has no resolved "
                "layout, so its storage size is unknown"
            )
        if self.num_elements == 0:
            return 0
        seq = len(self.dims) - 2
        inner = sum(
            (dim - 1) * stride
            for i, (dim, stride) in enumerate(zip(self.dims, self.strides))
            if i not in (0, seq)
        )
        reach = self.varlen.end or 0
        for start, length in zip(self.varlen.starts, self.varlen.lengths):
            if length > 0:
                reach = max(reach, start + (length - 1) * self.strides[seq] + inner + 1)
        return reach

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
            offset_uid, multiplier = ragged_fields(tensor_json)
            if multiplier < 1:
                raise GraphLoadError(
                    f"Tensor {tensor_json.get('uid')} has ragged_offset_multiplier "
                    f"{multiplier}; it must be at least 1"
                )
            return cls(
                uid=int(tensor_json["uid"]),
                name=tensor_json.get("name", f"tensor_{tensor_json['uid']}"),
                dims=[int(d) for d in tensor_json["dims"]],
                strides=[int(s) for s in tensor_json.get("strides") or []],
                data_type=tensor_json["data_type"],
                is_virtual=bool(tensor_json.get("virtual", False)),
                value=tensor_json.get("value"),
                ragged_offset_tensor_uid=offset_uid,
                ragged_offset_multiplier=multiplier,
            )
        except (KeyError, TypeError, ValueError, AttributeError) as e:
            raise GraphLoadError(
                f"Malformed tensor entry {tensor_json!r:.200}: {type(e).__name__}: {e}"
            ) from e

    @property
    def is_ragged(self) -> bool:
        """Whether batches start at offsets read from a ragged offset tensor."""
        return self.ragged_offset_tensor_uid is not None

    @property
    def logical_strides(self) -> List[int]:
        """Strides of the logical (``dims``-shaped) copy of this tensor.

        A ragged tensor is packed over ``dims[1:]`` in the order of its graph
        strides (outermost axis first, ties in axis order); batch ``b`` starts
        at ``b * prod(dims[1:])``. Other tensors keep their graph strides.
        """
        if not self.is_ragged:
            return list(self.strides)
        order = sorted(range(1, len(self.dims)), key=lambda i: (-self.strides[i], i))
        packed = [0] * len(self.dims)
        running = 1
        for axis in reversed(order):
            packed[axis] = running
            running *= self.dims[axis]
        packed[0] = running
        return packed

    @property
    def valid_elements(self) -> int:
        """Elements inside each batch's valid length (all of them when dense)."""
        if self.varlen is None or self.num_elements == 0:
            return self.num_elements
        return (
            sum(self.varlen.lengths)
            * self.num_elements
            // (self.dims[0] * self.dims[-2])
        )

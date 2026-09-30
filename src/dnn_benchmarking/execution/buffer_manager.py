# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Device buffer management and input generation for graph execution.

Host data uses the *logical* representation of :class:`~..common.dtypes.DType`:
dense arrays of ``dtype.numpy`` keyed by tensor UID. NumPy has no bfloat16 or
fp8, so those travel as float32 values that are exactly representable in the
graph type, and are encoded to raw words only when written to device storage.
"""

import functools
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..common.dtypes import DType
from ..common.exceptions import ExecutionError, UnsupportedGraphError
from ..graph.tensor_info import TensorInfo

# fp8 formats as (exponent bits, mantissa bits); E8M0 is a special case.
_FP8_FORMATS = {"fp8_e4m3": (4, 3), "fp8_e5m2": (5, 2), "fp8_e8m0": (8, 0)}


def _f32_to_bf16(data_f32: np.ndarray) -> np.ndarray:
    """Round float32 values to bfloat16 words (round-to-nearest-even).

    bfloat16 is the upper 16 bits of an IEEE-754 float32; RNE matches
    ``torch.Tensor.bfloat16()``. NaN stays NaN: the quiet bit is forced on and
    the rounding bias is skipped so the exponent cannot overflow to infinity.
    """
    f32_bits = np.asarray(data_f32, dtype=np.float32).view(np.uint32)
    exp_mask = np.uint32(0x7F800000)
    mant_mask = np.uint32(0x007FFFFF)
    is_nan = ((f32_bits & exp_mask) == exp_mask) & ((f32_bits & mant_mask) != 0)
    # RNE bias: 0x7FFF rounds half away from zero; adding the LSB of the
    # eventual bf16 word ties to even.
    lsb = (f32_bits >> np.uint32(16)) & np.uint32(1)
    rounded = f32_bits + (lsb + np.uint32(0x7FFF))
    rounded = np.where(is_nan, f32_bits | np.uint32(0x00400000), rounded)
    return (rounded >> np.uint32(16)).astype(np.uint16)


@functools.lru_cache(maxsize=None)
def _fp8_values(name: str) -> np.ndarray:
    """float32 value of each of the 256 codes of an fp8 format."""
    codes = np.arange(256)
    if name == "fp8_e8m0":
        # Unsigned power-of-two scale 2^(code-127); 0xFF is NaN, no zero.
        values = np.ldexp(1.0, codes - 127)
        values[0xFF] = np.nan
        return values.astype(np.float32)
    exp_bits, man_bits = _FP8_FORMATS[name]
    bias = (1 << (exp_bits - 1)) - 1
    exp = (codes >> man_bits) & ((1 << exp_bits) - 1)
    man = (codes & ((1 << man_bits) - 1)) / (1 << man_bits)
    magnitude = np.where(
        exp == 0, man * 2.0 ** (1 - bias), (1.0 + man) * np.exp2(exp - bias)
    )
    values = np.where(codes & 0x80, -magnitude, magnitude)
    if name == "fp8_e4m3":
        # OCP E4M3 ("fn"): no infinities; S.1111.111 is NaN.
        values[(codes & 0x7F) == 0x7F] = np.nan
    else:
        # E5M2 is IEEE-like: an all-ones exponent is inf (mantissa 0) or NaN.
        top = exp == (1 << exp_bits) - 1
        values[top] = np.where(man[top] == 0, np.copysign(np.inf, values[top]), np.nan)
    return values.astype(np.float32)


@functools.lru_cache(maxsize=None)
def _fp8_grid(name: str) -> Tuple[np.ndarray, np.ndarray]:
    """Sorted distinct finite values of an fp8 format and one code for each."""
    values = _fp8_values(name)
    finite = np.nonzero(np.isfinite(values))[0]
    grid, first = np.unique(values[finite], return_index=True)
    return grid, finite[first].astype(np.uint8)


def _f32_to_fp8(data_f32: np.ndarray, name: str) -> np.ndarray:
    """Round float32 values to the nearest finite fp8 code (saturating)."""
    grid, codes = _fp8_grid(name)
    upper = np.clip(np.searchsorted(grid, data_f32), 1, len(grid) - 1)
    nearer_lower = (data_f32 - grid[upper - 1]) <= (grid[upper] - data_f32)
    return codes[upper - nearer_lower]


def _storage_dtype(dtype: DType) -> np.dtype:
    """NumPy dtype of the raw device words for ``dtype``."""
    if dtype.name == "bfloat16":
        return np.dtype(np.uint16)
    if dtype.name in _FP8_FORMATS:
        return np.dtype(np.uint8)
    return dtype.numpy


def _encode(data: np.ndarray, dtype: DType) -> np.ndarray:
    """Logical values -> device words (same shape)."""
    if dtype.name == "bfloat16":
        return _f32_to_bf16(data)
    if dtype.name in _FP8_FORMATS:
        return _f32_to_fp8(np.asarray(data, dtype=np.float32), dtype.name)
    return np.asarray(data, dtype=dtype.numpy)


def _decode(words: np.ndarray, dtype: DType) -> np.ndarray:
    """Device words -> logical values (same shape)."""
    if dtype.name == "bfloat16":
        return (words.astype(np.uint32) << np.uint32(16)).view(np.float32)
    if dtype.name in _FP8_FORMATS:
        return _fp8_values(dtype.name)[words]
    return words


def _storage_view(storage: np.ndarray, tensor_info: TensorInfo) -> np.ndarray:
    """Create a logical ndarray view over raw tensor storage."""
    if tensor_info.strides:
        byte_strides = tuple(
            stride * storage.dtype.itemsize for stride in tensor_info.strides
        )
        return np.lib.stride_tricks.as_strided(
            storage,
            shape=tuple(tensor_info.dims),
            strides=byte_strides,
        )
    return storage.reshape(tensor_info.dims)


def _encode_to_storage_bytes(data: np.ndarray, tensor_info: TensorInfo) -> bytes:
    """Encode dense logical data into raw storage bytes using graph strides."""
    storage = np.zeros(
        tensor_info.storage_elements, dtype=_storage_dtype(tensor_info.dtype)
    )
    _storage_view(storage, tensor_info)[...] = _encode(data, tensor_info.dtype)
    return storage.tobytes()


def _decode_storage_bytes(data_bytes: bytes, tensor_info: TensorInfo) -> np.ndarray:
    """Decode raw storage bytes into a dense logical ndarray."""
    storage = np.frombuffer(
        data_bytes,
        dtype=_storage_dtype(tensor_info.dtype),
        count=tensor_info.storage_elements,
    )
    return np.ascontiguousarray(
        _decode(_storage_view(storage, tensor_info), tensor_info.dtype)
    )


def _paged_input_roles(graph_json: Optional[Dict[str, Any]]) -> Dict[int, str]:
    """Map tensor UID -> paged role for every SDPA node in the graph.

    Roles are read from the node's ``inputs`` map rather than guessed from a
    tensor name, because names are free-form and a graph is under no obligation
    to call its page table PAGE_TABLE_K.
    """
    roles: Dict[int, str] = {}
    if not graph_json:
        return roles
    for node in graph_json.get("nodes", []) or []:
        if not str(node.get("type", "")).startswith("Sdpa"):
            continue
        inputs = node.get("inputs") or {}
        for key, role in (
            ("page_table_k_tensor_uid", "page_table"),
            ("page_table_v_tensor_uid", "page_table"),
            ("seq_len_q_tensor_uid", "seq_len_q"),
            ("seq_len_kv_tensor_uid", "seq_len_kv"),
        ):
            uid = inputs.get(key)
            if uid is not None:
                roles[int(uid)] = role
    return roles


def _paged_metadata(
    graph_json: Optional[Dict[str, Any]],
) -> Tuple[int, int]:
    """(page_size, num_pages) taken from the paged K container's own dims.

    hipDNN has no page-size scalar: the paged K/V container is
    ``[num_blocks, num_kv_heads, page_size, head_size]``, so both facts are read
    off that tensor instead of being assumed.
    """
    if not graph_json:
        return 0, 0
    tensors = {int(t["uid"]): t for t in graph_json.get("tensors", []) or []}
    for node in graph_json.get("nodes", []) or []:
        inputs = node.get("inputs") or {}
        if inputs.get("page_table_k_tensor_uid") is None:
            continue
        k = (
            tensors.get(int(inputs["k_tensor_uid"]))
            if inputs.get("k_tensor_uid")
            else None
        )
        if k and len(k.get("dims", [])) == 4:
            return int(k["dims"][2]), int(k["dims"][0])
    return 0, 0


def _generate_paged_input(
    tensor_info: TensorInfo,
    role: str,
    page_size: int,
    num_pages: int,
) -> np.ndarray:
    """Structured data for a paged input.

    Random page ids or lengths are degenerate for these roles: a page table
    must address distinct pages inside the cache, and a sequence length must be
    positive and fit its page allocation. They are generated to satisfy those
    invariants instead.
    """
    dims = list(tensor_info.dims)
    dtype = tensor_info.dtype.numpy

    if role == "page_table":
        # Distinct pages per sequence, so a gather that ignores the table or
        # collapses sequences produces a visibly different answer.
        num_seqs = int(dims[0])
        blocks_per_seq = int(dims[1]) if len(dims) > 1 else 1
        needed = num_seqs * blocks_per_seq
        pool = num_pages if num_pages > 0 else needed
        ids = (np.arange(needed) % max(pool, 1)).astype(dtype)
        return ids.reshape(dims)

    # Sequence lengths fill the cache: every sequence uses its whole page
    # allocation, which is the largest length the page table can legally address.
    return np.full(dims, page_size if page_size > 0 else 1, dtype=dtype)


def generate_input_data(
    tensor_infos: List[TensorInfo],
    seed: int,
    graph_json: Optional[Dict[str, Any]] = None,
) -> Dict[int, np.ndarray]:
    """Generate one graph-scoped logical input map.

    Returned arrays are dense logical ndarrays keyed by tensor UID, consumed
    by both hipDNN buffers and reference providers. Floating-point inputs are
    U[0, 1) (non-negative, so batch-norm variances stay valid), drawn as
    float32 and rounded to the graph type; bfloat16 and fp8 values are returned
    as the exactly representable float32 values. Integer and boolean inputs
    are zeros, except paged-SDPA roles.

    ``graph_json`` only affects **paged** SDPA graphs: page tables and
    sequence lengths have invariants random data cannot satisfy (see
    :func:`_generate_paged_input`).
    """
    rng = np.random.default_rng(seed)
    input_data: Dict[int, np.ndarray] = {}
    paged_roles = _paged_input_roles(graph_json)
    page_size, num_pages = _paged_metadata(graph_json) if paged_roles else (0, 0)

    for tensor_info in tensor_infos:
        if tensor_info.is_output or tensor_info.is_virtual:
            continue
        dtype = tensor_info.dtype

        if tensor_info.is_pass_by_value:
            input_data[tensor_info.uid] = np.asarray(
                [tensor_info.value], dtype=dtype.numpy
            )
            continue

        role = paged_roles.get(tensor_info.uid)
        if role is not None:
            input_data[tensor_info.uid] = _generate_paged_input(
                tensor_info, role, page_size, num_pages
            )
        elif dtype.numpy.kind in "iub":
            # ponytail: zeros are safe for index/offset tensors (no OOB reads);
            # per-role structured generation (e.g. MoE offsets) is the upgrade path.
            input_data[tensor_info.uid] = np.zeros(tensor_info.dims, dtype=dtype.numpy)
        else:
            data = rng.random(tensor_info.dims, dtype=np.float32)
            input_data[tensor_info.uid] = _decode(_encode(data, dtype), dtype)

    return input_data


class BufferManager:
    """Manages device buffer allocation and data transfer for graph execution.

    This class handles:
    - Allocating device buffers for all tensors
    - Creating variant packs (UID -> pointer mapping)
    - Copying pre-generated inputs to device buffers
    - Cleanup of device memory

    Storage is ``hipdnn.DeviceBuffer`` by default. With a torch ``device``,
    each buffer is a raw ``torch.uint8`` tensor, and outputs can be viewed on
    the device without a host copy. hipDNN receives each tensor's
    ``data_ptr()``; this manager keeps the tensors alive until ``cleanup``.
    """

    def __init__(
        self,
        tensor_infos: List[TensorInfo],
        device: Optional[str] = None,
    ) -> None:
        """Initialize buffer manager with tensor metadata.

        Args:
            tensor_infos: List of TensorInfo objects describing tensors.
            device: Torch device for torch-backed storage (for example
                ``"cuda"``), or None for ``hipdnn.DeviceBuffer`` storage.
        """
        self._tensor_infos = tensor_infos
        self._tensor_info_by_uid = {tensor.uid: tensor for tensor in tensor_infos}
        self._device = device
        self._buffers: Dict[int, Any] = {}  # UID -> DeviceBuffer or torch.Tensor

    def allocate_all(self) -> None:
        """Allocate device buffers for all tensors.

        Raises:
            ExecutionError: If hipdnn_frontend is not available.
        """
        if self._device is not None:
            import torch

            for tensor_info in self._tensor_infos:
                if tensor_info.is_virtual or tensor_info.is_pass_by_value:
                    continue
                self._buffers[tensor_info.uid] = torch.empty(
                    tensor_info.size_bytes, dtype=torch.uint8, device=self._device
                )
            return

        try:
            import hipdnn_frontend as hipdnn
        except ImportError as e:
            raise ExecutionError(
                "hipdnn_frontend not available. Install hipDNN Python bindings."
            ) from e

        for tensor_info in self._tensor_infos:
            if tensor_info.is_virtual or tensor_info.is_pass_by_value:
                continue

            buffer = hipdnn.DeviceBuffer(tensor_info.size_bytes)
            self._buffers[tensor_info.uid] = buffer

    def create_variant_pack(self) -> Dict[int, int]:
        """Create variant pack mapping tensor UIDs to device pointers.

        Returns:
            Dictionary mapping tensor UID to device pointer (as int).

        Raises:
            ExecutionError: If buffers not allocated.
        """
        if not self._buffers:
            raise ExecutionError("Buffers not allocated. Call allocate_all() first.")

        if self._device is None:
            return {uid: buffer.ptr() for uid, buffer in self._buffers.items()}
        return {uid: buffer.data_ptr() for uid, buffer in self._buffers.items()}

    def _write_bytes(self, buffer: Any, raw_bytes: bytes) -> None:
        """Copy graph-layout bytes from the host into one buffer."""
        if self._device is None:
            buffer.copy_from_host(raw_bytes)
            return
        import torch

        with warnings.catch_warnings():
            # copy_ only reads the source, so a read-only view is safe and
            # avoids a second host copy of the input.
            warnings.simplefilter("ignore", UserWarning)
            source = torch.frombuffer(raw_bytes, dtype=torch.uint8)
        buffer.copy_(source)

    def load_input_data(self, input_data: Dict[int, np.ndarray]) -> None:
        """Copy pre-generated graph input data into device buffers.

        Input generation and host-to-device copies are intentionally separate
        from benchmark timing. Call this after ``allocate_all`` and before
        creating the variant pack. Pass-by-value tensors have no buffer and
        are skipped.

        Raises:
            ExecutionError: If buffers are not allocated or an input is missing.
        """
        if not self._buffers:
            raise ExecutionError("Buffers not allocated. Call allocate_all() first.")

        for tensor_info in self._tensor_infos:
            if (
                tensor_info.is_output
                or tensor_info.is_virtual
                or tensor_info.is_pass_by_value
            ):
                continue
            data = input_data.get(tensor_info.uid)
            if data is None:
                raise ExecutionError(
                    f"Missing input data for tensor UID {tensor_info.uid}"
                )
            buffer = self._buffers.get(tensor_info.uid)
            if buffer is not None:
                self._write_bytes(buffer, _encode_to_storage_bytes(data, tensor_info))

    def zero_outputs(self) -> None:
        """Zero output tensor buffers.

        Raises:
            ExecutionError: If buffers not allocated.
        """
        if not self._buffers:
            raise ExecutionError("Buffers not allocated. Call allocate_all() first.")

        for tensor_info in self._tensor_infos:
            if not tensor_info.is_output:
                continue

            buffer = self._buffers.get(tensor_info.uid)
            if buffer is None:
                continue
            if self._device is None:
                buffer.zeros()
            else:
                buffer.zero_()

        if self._device is not None:
            import torch

            # zero_() runs on torch's stream; hipDNN runs on the handle stream.
            if torch.device(self._device).type == "cuda":
                torch.cuda.synchronize()

    def get_output_data(self, uid: int) -> Optional[np.ndarray]:
        """Copy an output tensor to the host as a dense logical array.

        Returns:
            Numpy array with output data, or None if the tensor is unknown.
        """
        buffer = self._buffers.get(uid)
        tensor_info = self._tensor_info_by_uid.get(uid)
        if buffer is None or tensor_info is None:
            return None

        if self._device is None:
            data_bytes = buffer.copy_to_host()
        else:
            data_bytes = buffer.cpu().numpy().tobytes()
        return _decode_storage_bytes(data_bytes, tensor_info)

    def get_output_tensor(self, uid: int) -> Optional["torch.Tensor"]:
        """Return a typed logical view of an output buffer without a copy.

        Returns None for DeviceBuffer storage, for an unknown UID, and for a
        data type that the installed torch cannot represent.
        """
        if self._device is None:
            return None
        buffer = self._buffers.get(uid)
        tensor_info = self._tensor_info_by_uid.get(uid)
        if buffer is None or tensor_info is None:
            return None
        try:
            torch_dtype = tensor_info.dtype.torch_dtype()
        except UnsupportedGraphError:
            return None
        typed = buffer.view(torch_dtype)
        if tensor_info.strides:
            return typed.as_strided(tensor_info.dims, tensor_info.strides)
        return typed[: tensor_info.num_elements].reshape(tensor_info.dims)

    def get_output_tensors(self) -> List[TensorInfo]:
        """Get list of output tensor infos.

        Returns:
            List of TensorInfo objects for output tensors.
        """
        return [ti for ti in self._tensor_infos if ti.is_output]

    def cleanup(self) -> None:
        """Free all device buffers."""
        self._buffers.clear()
        if self._device is not None:
            import torch

            if torch.device(self._device).type != "cuda":
                return

            # Return the freed blocks to the driver. hipDNN workspaces and
            # the next engine allocate with hipMalloc, which cannot reuse
            # torch's cached blocks.
            torch.cuda.empty_cache()

    def __enter__(self) -> "BufferManager":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - cleanup buffers."""
        self.cleanup()

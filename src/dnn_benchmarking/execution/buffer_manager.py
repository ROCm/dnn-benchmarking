# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Device buffer management and input generation for graph execution.

Host data uses the *logical* representation of :class:`~..common.dtypes.DType`:
dense arrays of ``dtype.numpy`` keyed by tensor UID. NumPy has no bfloat16 or
fp8, so those travel as float32 values that are exactly representable in the
graph type, and are encoded to raw words only when written to device storage.
"""

import functools
import hashlib
import math
import struct
import warnings
from typing import Any, Collection, Dict, Iterator, List, Optional, Set, Tuple

import numpy as np

from ..common.blocks import iter_blocks
from ..common.dtypes import DType
from ..common.exceptions import ExecutionError, UnsupportedGraphError
from ..config.benchmark_config import INPUT_RECIPE_CHOICES, INPUT_RECIPE_VERSION
from ..graph.loader import tensor_data_type
from ..graph.tensor_info import TensorInfo
from ..graph.varlen import sdpa_operands
from .sdpa_backward_inputs import set_sdpa_backward_inputs
from .sdpa_dropout import read_dropout
from .varlen_inputs import generate_varlen_inputs

# fp8 formats as (exponent bits, mantissa bits); E8M0 is a special case.
_FP8_FORMATS = {
    "fp8_e4m3": (4, 3),
    "fp8_e5m2": (5, 2),
    "fp8_e8m0": (8, 0),
    "fp8_e4m3_fnuz": (4, 3),
    "fp8_e5m2_fnuz": (5, 2),
}

# One float32 NaN in little-endian storage order.
_NAN_WORD = struct.pack("<f", math.nan)


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
    fnuz = name.endswith("_fnuz")
    # FNUZ exponent bias is 2^(e-1), one more than the OCP formats.
    bias = (1 << (exp_bits - 1)) - (0 if fnuz else 1)
    exp = (codes >> man_bits) & ((1 << exp_bits) - 1)
    man = (codes & ((1 << man_bits) - 1)) / (1 << man_bits)
    magnitude = np.where(
        exp == 0, man * 2.0 ** (1 - bias), (1.0 + man) * np.exp2(exp - bias)
    )
    values = np.where(codes & 0x80, -magnitude, magnitude)
    if fnuz:
        # No infinities and no -0: 0x80 is the only NaN.
        values[0x80] = np.nan
    elif name == "fp8_e4m3":
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


def _ragged_batch_views(
    storage: np.ndarray, tensor_info: TensorInfo
) -> Iterator[Tuple[int, np.ndarray]]:
    """Yield (batch, view) for each non-empty batch of a ragged tensor.

    The view has shape ``dims[1:]`` with the sequence dimension cut to the
    batch's length. It starts at the batch's element offset and uses
    ``strides[1:]``, so ``strides[0]`` is not used.
    """
    varlen = tensor_info.varlen
    itemsize = storage.dtype.itemsize
    byte_strides = tuple(s * itemsize for s in tensor_info.strides[1:])
    for batch, (start, length) in enumerate(zip(varlen.starts, varlen.lengths)):
        if length <= 0:
            continue
        shape = (tensor_info.dims[1], length, tensor_info.dims[3])
        yield batch, np.lib.stride_tricks.as_strided(
            storage[start:], shape=shape, strides=byte_strides
        )


def _is_packed(tensor_info: TensorInfo) -> bool:
    """True when the graph strides are the dense C-order strides of the dims."""
    if not tensor_info.strides:
        return True
    if len(tensor_info.strides) != len(tensor_info.dims):
        return False  # storage_elements reports the malformed entry
    expected = 1
    for dim, stride in zip(reversed(tensor_info.dims), reversed(tensor_info.strides)):
        if dim != 1 and stride != expected:
            return False
        expected *= dim
    return True


def _fill_repeating(
    storage: np.ndarray, data: np.ndarray, tensor_info: TensorInfo
) -> None:
    """Fill 1-D ``storage`` with the encoded values of ``data``, repeated in C
    order.

    A ragged tensor's padding (past each batch's length, and the gaps between
    batches) thus holds generated values, as the padding of a padded tensor
    and hipDNN's golden data do, rather than zeros an engine might rely on.
    """
    if not storage.size:
        return
    flat = data.reshape(-1)
    for start in range(0, storage.size, flat.size):
        part = storage[start : start + flat.size]
        source = flat[: part.size]
        for index in iter_blocks(part.shape):
            part[index] = _encode(source[index], tensor_info.dtype)


def _encode_to_storage(data: np.ndarray, tensor_info: TensorInfo) -> np.ndarray:
    """Encode dense logical data into 1-D storage words in the graph layout.

    Data that already holds the storage words in a packed layout is returned
    as a flat view, without a copy. Otherwise the words are encoded block by
    block straight into the storage array, so no full-size encoded copy or
    conversion temporary exists.
    """
    storage_dtype = _storage_dtype(tensor_info.dtype)
    data = np.asarray(data)
    ragged = tensor_info.varlen is not None and tensor_info.varlen.starts is not None
    if (
        not ragged
        and data.dtype == storage_dtype
        and data.shape == tuple(tensor_info.dims)
        and data.flags.c_contiguous
        and _is_packed(tensor_info)
    ):
        return data.reshape(-1)
    if ragged:
        storage = np.empty(tensor_info.storage_elements, dtype=storage_dtype)
        data = np.broadcast_to(data, tuple(tensor_info.dims))
        _fill_repeating(storage, data, tensor_info)
        for batch, view in _ragged_batch_views(storage, tensor_info):
            source = data[batch, :, : view.shape[1], :]
            for index in iter_blocks(view.shape):
                view[index] = _encode(source[index], tensor_info.dtype)
        return storage
    storage = np.zeros(tensor_info.storage_elements, dtype=storage_dtype)
    view = _storage_view(storage, tensor_info)
    data = np.broadcast_to(data, view.shape)
    for index in iter_blocks(view.shape):
        view[index] = _encode(data[index], tensor_info.dtype)
    return storage


def _decode_storage(raw: Any, tensor_info: TensorInfo) -> np.ndarray:
    """Decode raw storage into a dense logical ndarray.

    ``raw`` is any buffer-protocol object (``bytes``, a NumPy array). The
    words are decoded block by block into the result. A ragged tensor
    decodes each batch's valid positions from its start; the rest is 0.
    """
    storage = np.frombuffer(
        raw,
        dtype=_storage_dtype(tensor_info.dtype),
        count=tensor_info.storage_elements,
    )
    if tensor_info.varlen is not None and tensor_info.varlen.starts is not None:
        out = np.zeros(tensor_info.dims, dtype=tensor_info.dtype.numpy)
        for batch, view in _ragged_batch_views(storage, tensor_info):
            target = out[batch, :, : view.shape[1], :]
            for index in iter_blocks(view.shape):
                target[index] = _decode(view[index], tensor_info.dtype)
        return out
    view = _storage_view(storage, tensor_info)
    if storage.dtype == tensor_info.dtype.numpy:
        return np.ascontiguousarray(view)
    out = np.empty(view.shape, dtype=tensor_info.dtype.numpy)
    for index in iter_blocks(view.shape):
        out[index] = _decode(view[index], tensor_info.dtype)
    return out


def _page_table_uids(graph_json: Optional[Dict[str, Any]]) -> Set[int]:
    """UIDs of every SDPA page table.

    Page tables are read from the node's operand keys rather than guessed from
    a tensor name: names are free-form, and a graph is under no obligation to
    call its page table PAGE_TABLE_K.
    """
    uids: Set[int] = set()
    if not graph_json:
        return uids
    for node in graph_json.get("nodes", []) or []:
        if not str(node.get("type", "")).startswith("Sdpa"):
            continue
        operands = sdpa_operands(node)
        for uid in (operands.page_table_k_tensor_uid, operands.page_table_v_tensor_uid):
            if uid is not None:
                uids.add(uid)
    return uids


# Keep rate of a generated dropout mask when the graph gives no dropout_scale
# value. An explicit-mask graph has no dropout_probability to take it from.
_DEFAULT_DROPOUT_KEEP = 0.9


def _dropout_input_roles(graph_json: Optional[Dict[str, Any]]) -> Dict[int, str]:
    """Map tensor UID -> dropout role for every SDPA node in the graph.

    The roles are ``mask``, ``scale``, ``scale_inv``, ``seed`` and ``offset``.
    Forward and backward nodes both count: a backward node also carries the
    mask, the scale and the scale_inv of its forward pass.
    """
    roles: Dict[int, str] = {}
    if not graph_json:
        return roles
    for node in graph_json.get("nodes", []) or []:
        if not str(node.get("type", "")).startswith("Sdpa"):
            continue
        dropout = read_dropout(node)
        for uid, role in (
            (dropout.mask_uid, "mask"),
            (dropout.scale_uid, "scale"),
            (dropout.scale_inv_uid, "scale_inv"),
            (dropout.seed_uid, "seed"),
            (dropout.offset_uid, "offset"),
        ):
            if uid is not None:
                roles[uid] = role
    return roles


def _dropout_keep_rate(tensor_infos: List[TensorInfo], roles: Dict[int, str]) -> float:
    """Keep rate of the generated dropout masks.

    A ``dropout_scale`` value in the graph sets the keep rate to 1 / scale.
    The first value sets the rate, and every value is checked. Without a
    value the keep rate is 0.9.

    Raises:
        ValueError: A ``dropout_scale`` value gives a keep rate outside (0, 1].
    """
    keep: Optional[float] = None
    for tensor_info in tensor_infos:
        if not tensor_info.is_pass_by_value or roles.get(tensor_info.uid) != "scale":
            continue
        scale = float(tensor_info.value)
        if not 1.0 <= scale < math.inf:
            raise ValueError(
                f"dropout_scale tensor {tensor_info.uid} has the value {scale}: "
                "the keep rate 1/dropout_scale must be in (0, 1]"
            )
        if keep is None:
            keep = 1.0 / scale
    return _DEFAULT_DROPOUT_KEEP if keep is None else keep


def _generate_dropout_input(
    tensor_info: TensorInfo,
    role: str,
    keep: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Data for an SDPA dropout input.

    The mask holds exact 0 and 1 values; each element is 1 with probability
    ``keep``. The scale is 1 / keep and the scale_inv is keep, so the mask and
    both scales agree. Integer seed and offset values are random and at least
    1; a seed or offset of another type gets 1.
    """
    dims = list(tensor_info.dims)
    dtype = tensor_info.dtype

    if role == "mask":
        # Draw one block at a time, so no full-size float32 draw exists.
        data = np.empty(dims, dtype=dtype.numpy)
        for index in iter_blocks(data.shape):
            draw = rng.random(data[index].shape, dtype=np.float32)
            data[index] = draw < np.float32(keep)
        return data

    if role in ("seed", "offset"):
        high = 1
        if dtype.numpy.kind in "iu":
            high = min(2**31 - 1, int(np.iinfo(dtype.numpy).max))
        values = rng.integers(1, high, size=dims, dtype=np.int64, endpoint=True)
        return values.astype(dtype.numpy)

    value = 1.0 / keep if role == "scale" else keep
    return _decode(_encode(np.full(dims, value, dtype=np.float32), dtype), dtype)


def _paged_metadata(graph_json: Optional[Dict[str, Any]]) -> int:
    """Number of pages, taken from the paged K container's own dims.

    hipDNN has no page-count scalar: the paged K/V container is
    ``[num_blocks, num_kv_heads, page_size, head_size]``, so the count is read
    off that tensor instead of being assumed. Returns 0 when there is none.
    """
    if not graph_json:
        return 0
    tensors = {int(t["uid"]): t for t in graph_json.get("tensors", []) or []}
    for node in graph_json.get("nodes", []) or []:
        if not str(node.get("type", "")).startswith("Sdpa"):
            continue
        operands = sdpa_operands(node)
        if not operands.is_paged:
            continue
        k = tensors.get(operands.k) if operands.k is not None else None
        if k and len(k.get("dims", [])) == 4:
            return int(k["dims"][0])
    return 0


# Convolution operands that can take signed data: the data, filter and
# incoming-gradient inputs of every convolution direction.
_CONVOLUTION_OPERAND_KEYS = ("x_tensor_uid", "w_tensor_uid", "dy_tensor_uid")
# The output of each convolution direction.
_CONVOLUTION_OUTPUT_KEYS = ("y_tensor_uid", "dx_tensor_uid", "dw_tensor_uid")


def _signed_input_uids(graph_json: Optional[Dict[str, Any]]) -> Set[int]:
    """UIDs of inputs that get zero-mean U[-1, 1) data instead of U[0, 1).

    Non-negative data makes every product in a convolution sum positive, so
    the sum grows with the number of terms. A backward-weights sum over
    N*Ho*Wo terms then passes the fp16 maximum (65504) at about 262,000
    terms. Signed data lets the terms cancel, so the sum grows like
    sqrt(N*Ho*Wo) instead.

    Only the operands of a convolution with an fp16 output are signed: the
    overflow happens when the sum is stored as fp16. Other convolutions keep
    U[0, 1), so their outputs stay far from zero, where the fixed fp32 atol
    would apply. A tensor that also feeds any other node, including a
    convolution with a wider output, keeps U[0, 1). A non-convolution node
    can need non-negative values (for example a batch-norm variance).
    """
    if not graph_json:
        return set()
    tensors = {
        int(tensor["uid"]): tensor
        for tensor in graph_json.get("tensors", []) or []
        if "uid" in tensor
    }
    convolution: Set[int] = set()
    other: Set[int] = set()
    for node in graph_json.get("nodes", []) or []:
        inputs = node.get("inputs") or {}
        is_convolution = str(node.get("type", "")).startswith("Convolution")
        if is_convolution and _has_fp16_output(node, tensors, graph_json):
            convolution.update(
                int(inputs[key])
                for key in _CONVOLUTION_OPERAND_KEYS
                if inputs.get(key) is not None
            )
        else:
            other.update(_input_tensor_uids(inputs))
    return convolution - other


def _fp32_conv_input_uids(graph_json: Optional[Dict[str, Any]]) -> Set[int]:
    """Convolution operand UIDs whose output is fp32/fp64.

    Under the default ``centered`` recipe these keep ``U[0, 1)`` so outputs
    stay away from zero where the fixed fp32 validation ``atol=1e-6`` would
    reject legitimate accumulation cancellation against the float64 reference.
    """
    if not graph_json:
        return set()
    tensors = {
        int(tensor["uid"]): tensor
        for tensor in graph_json.get("tensors", []) or []
        if "uid" in tensor
    }
    uids: Set[int] = set()
    for node in graph_json.get("nodes", []) or []:
        if not str(node.get("type", "")).startswith("Convolution"):
            continue
        outputs = node.get("outputs") or {}
        out_dtype = None
        for key in _CONVOLUTION_OUTPUT_KEYS:
            uid = outputs.get(key)
            if uid is not None and int(uid) in tensors:
                out_dtype = _output_data_type(tensors[int(uid)], graph_json)
                break
        if out_dtype in ("float", "double"):
            inputs = node.get("inputs") or {}
            uids.update(
                int(inputs[key])
                for key in _CONVOLUTION_OPERAND_KEYS
                if inputs.get(key) is not None
            )
    return uids


def _has_fp16_output(
    node: Dict[str, Any],
    tensors: Dict[int, Dict[str, Any]],
    graph_json: Dict[str, Any],
) -> bool:
    """True when the convolution node stores its output as fp16."""
    outputs = node.get("outputs") or {}
    for key in _CONVOLUTION_OUTPUT_KEYS:
        uid = outputs.get(key)
        if uid is not None and int(uid) in tensors:
            return _output_data_type(tensors[int(uid)], graph_json) == "half"
    return False


def _output_data_type(
    tensor: Dict[str, Any], graph_json: Dict[str, Any]
) -> Optional[str]:
    """The data type a convolution stores its output in.

    hipDNN fills an "unset" virtual (fused intermediate) tensor from the
    graph's intermediate_data_type, and any other "unset" tensor from its
    io_data_type (:func:`tensor_data_type`).
    """
    if tensor.get("virtual") and tensor.get("data_type") == "unset":
        return graph_json.get("intermediate_data_type") or "unset"
    return tensor_data_type(tensor, graph_json)


def _input_tensor_uids(inputs: Dict[str, Any]) -> Set[int]:
    """Tensor UIDs in a node's ``inputs`` map. Other keys in that map are
    attributes (for example a pointwise ``operation``), and a ``*_tensor_uid``
    key can hold one UID or a list of UIDs."""
    uids: Set[int] = set()
    for key, value in inputs.items():
        if not key.endswith("_tensor_uid") or value is None:
            continue
        values = value if isinstance(value, list) else [value]
        uids.update(int(uid) for uid in values if uid is not None)
    return uids


def _generate_paged_input(
    tensor_info: TensorInfo,
    num_pages: int,
) -> np.ndarray:
    """Structured data for a page table.

    A page table must address distinct pages inside the cache, so random page
    ids are degenerate. Its first dim is the sequence count and the product
    of the others is the width: hipDNN's ``[B, 1, max_blocks_per_seq, 1]``
    and a rank-2 ``[B, max_blocks_per_seq]`` hold the same ids.
    """
    dims = list(tensor_info.dims)
    dtype = tensor_info.dtype.numpy

    # Distinct pages per sequence, so a gather that ignores the table or
    # collapses sequences produces a visibly different answer.
    num_seqs = int(dims[0])
    blocks_per_seq = int(np.prod(dims[1:], dtype=np.int64))
    needed = num_seqs * blocks_per_seq
    pool = num_pages if num_pages > 0 else needed
    ids = (np.arange(needed) % max(pool, 1)).astype(dtype)
    return ids.reshape(dims)


# Node input keys that require strictly positive values (e.g. norm variance,
# inverse variance, epsilon, momentum, quantization scales).
_POSITIVE_VAR_KEYS = frozenset(
    {
        "in_running_var_tensor_uid",
        "prev_running_variance_tensor_uid",
        "variance_tensor_uid",
        "inv_variance_tensor_uid",
        "inv_rms_tensor_uid",
        "epsilon_tensor_uid",
        "momentum_tensor_uid",
        "dropout_scale_tensor_uid",
        "descale_q_tensor_uid",
        "descale_k_tensor_uid",
        "descale_v_tensor_uid",
        "descale_s_tensor_uid",
        "scale_s_tensor_uid",
        "scale_o_tensor_uid",
        "scale_tensor_uid",
        "in_scale_tensor_uid",
    }
)
_NORM_NODE_PREFIXES = ("Batchnorm", "Layernorm", "RMSNorm")
_WEIGHT_KEYS = frozenset({"w_tensor_uid", "b_tensor_uid"})
_SDPA_QK_KEYS = frozenset({"q_tensor_uid", "k_tensor_uid"})
_SDPA_V_KEYS = frozenset({"v_tensor_uid"})


def _tensor_roles(graph_json: Optional[Dict[str, Any]]) -> Dict[int, str]:
    """Classify input tensor UIDs by graph role for realistic recipes.

    Positive-required roles (variance/epsilon/scales) take priority over every
    other role so a shared tensor never receives negative values where positive
    inputs are required.
    """
    if not graph_json:
        return {}
    roles: Dict[int, str] = {}
    priority = {
        "activation": 0,
        "grad": 1,
        "weight": 2,
        "sdpa_qk": 3,
        "sdpa_v": 3,
        "norm_scale": 4,
        "positive_var": 5,
    }

    def assign(uid: int, role: str) -> None:
        if priority[role] >= priority.get(roles.get(uid, "activation"), -1):
            roles[uid] = role

    for node in graph_json.get("nodes", []) or []:
        ntype = str(node.get("type", ""))
        inputs = node.get("inputs") or {}
        is_norm = ntype.startswith(_NORM_NODE_PREFIXES)
        is_sdpa = ntype.startswith("Sdpa")
        is_matmul_or_conv = ntype.startswith(
            ("Convolution", "Matmul", "MoeGroupedMatmul")
        )
        for key, value in inputs.items():
            if not key.endswith("_tensor_uid") or value is None:
                continue
            uids = value if isinstance(value, list) else [value]
            for raw_uid in uids:
                if raw_uid is None:
                    continue
                uid = int(raw_uid)
                if is_norm and key == "scale_tensor_uid":
                    assign(uid, "norm_scale")
                elif key in _POSITIVE_VAR_KEYS:
                    assign(uid, "positive_var")
                elif is_sdpa and key in _SDPA_QK_KEYS:
                    assign(uid, "sdpa_qk")
                elif is_sdpa and key in _SDPA_V_KEYS:
                    assign(uid, "sdpa_v")
                elif key.startswith("d") and key != "descale_q_tensor_uid":
                    assign(uid, "grad")
                elif is_matmul_or_conv and key in _WEIGHT_KEYS:
                    assign(uid, "weight")
    return roles


def _draw_float_block(
    rng: np.random.Generator,
    shape: Tuple[int, ...],
    *,
    recipe: str,
    role: str,
    is_signed_conv: bool,
    is_fp32_conv: bool = False,
    head_dim: int = 1,
    dtype: DType,
) -> np.ndarray:
    """Draw one float32 block for ``(recipe, role)`` before dtype encoding."""
    if dtype.name == "fp8_e8m0":
        # E8M0 is an unsigned power-of-two scale format with no zero code.
        if recipe == "zeros":
            return np.ones(shape, dtype=np.float32)
        if recipe == "legacy":
            return rng.random(shape, dtype=np.float32)
        return rng.uniform(0.5, 1.5, size=shape).astype(np.float32)

    if recipe == "legacy":
        block = rng.random(shape, dtype=np.float32)
        if is_signed_conv:
            block = block * np.float32(2) - np.float32(1)
        return block

    if role == "positive_var":
        if recipe == "zeros":
            return np.ones(shape, dtype=np.float32)
        return rng.uniform(0.5, 1.5, size=shape).astype(np.float32)

    if role == "norm_scale":
        if recipe == "zeros":
            return np.ones(shape, dtype=np.float32)
        if recipe == "llm":
            block = rng.normal(1.0, 0.25, size=shape).astype(np.float32)
            return np.clip(block, np.float32(0.1), np.float32(4.0))
        return rng.uniform(0.5, 1.5, size=shape).astype(np.float32)

    if recipe == "zeros":
        return np.zeros(shape, dtype=np.float32)

    if recipe == "centered":
        if is_signed_conv:
            return rng.uniform(-1.0, 1.0, size=shape).astype(np.float32)
        if is_fp32_conv:
            return rng.random(shape, dtype=np.float32)
        if role == "sdpa_qk":
            # Scale Q and K by d^-0.25 so Q @ K^T has unit variance even when a
            # graph omits attn_scale_value (hipDNN default scale = 1.0). Keeps
            # Q/K zero-mean signed Gaussian while preventing unscaled logits of
            # O(sqrt(d)) from amplifying bf16 pre-scaled q*log2(e) rounding in
            # FlashAttention/AITER kernels.
            scale = np.float32(max(1, head_dim) ** -0.25)
            return rng.standard_normal(size=shape, dtype=np.float32) * scale
        if role == "sdpa_v":
            # Keep V in U[0, 1) under centered: softmax weights are
            # non-negative and sum to 1, so O = P @ V stays centered near 0.5
            # instead of cancelling near zero where bf16 tolerance (3e-2, 1e-3)
            # rejects 1-2 ULP flash/AITER vs MATH differences.
            return rng.random(shape, dtype=np.float32)
        return rng.standard_normal(size=shape, dtype=np.float32)
    if recipe == "heavy_tail":
        base = rng.normal(0.0, 0.8, size=shape).astype(np.float32)
        tail = rng.normal(0.0, 6.0, size=shape).astype(np.float32)
        pick = rng.random(shape, dtype=np.float32) < np.float32(0.01)
        return np.clip(np.where(pick, tail, base), np.float32(-16.0), np.float32(16.0))

    if recipe == "llm":
        # Fitted to measured DeepSeek-V2-Lite / Qwen2.5 / OLMo-2 distributions.
        if role == "weight":
            block = (rng.standard_t(10.0, size=shape) * 0.03).astype(np.float32)
            return np.clip(block, np.float32(-1.5), np.float32(1.5))
        if role == "sdpa_qk":
            block = (rng.standard_t(3.0, size=shape) * 0.75).astype(np.float32)
            return np.clip(block, np.float32(-25.0), np.float32(25.0))
        if role == "sdpa_v":
            block = rng.laplace(0.0, 0.35, size=shape).astype(np.float32)
            return np.clip(block, np.float32(-16.0), np.float32(16.0))
        if role == "grad":
            block = (rng.standard_t(2.5, size=shape) * 1e-3).astype(np.float32)
            return np.clip(block, np.float32(-1.0), np.float32(1.0))
        block = (rng.standard_t(2.5, size=shape) * 0.35).astype(np.float32)
        return np.clip(block, np.float32(-16.0), np.float32(16.0))

    raise ValueError(
        f"Unknown input recipe {recipe!r}; expected one of: {', '.join(INPUT_RECIPE_CHOICES)}"
    )


def input_fingerprint(input_data: Dict[int, np.ndarray], recipe: str, seed: int) -> str:
    """Deterministic 12-hex SHA-256 fingerprint of generated graph inputs."""
    h = hashlib.sha256()
    h.update(f"v{INPUT_RECIPE_VERSION}:{recipe}:{seed}".encode("utf-8"))
    for uid in sorted(input_data):
        arr = np.asarray(input_data[uid])
        h.update(f"|{uid}:{arr.shape}:{arr.dtype.str}:".encode("utf-8"))
        flat = np.ascontiguousarray(arr).reshape(-1)
        nbytes = flat.nbytes
        if nbytes <= 65536:
            h.update(flat.tobytes())
        else:
            raw = flat.view(np.uint8)
            step = max(1, nbytes // 4096)
            h.update(raw[:4096].tobytes())
            h.update(raw[::step][:4096].tobytes())
            h.update(raw[-4096:].tobytes())
    return h.hexdigest()[:12]


def buffer_set_bytes(tensor_infos: List[TensorInfo]) -> int:
    """Total device storage bytes for one non-virtual tensor set.

    Uses resolved ``storage_elements`` so varlen and ragged tensors count
    their real device buffer footprint, independent of ``--no-metrics``.
    """
    return sum(
        ti.storage_elements * ti.element_size
        for ti in tensor_infos
        if not ti.is_virtual and not ti.is_pass_by_value
    )


def generate_input_data(
    tensor_infos: List[TensorInfo],
    seed: int,
    graph_json: Optional[Dict[str, Any]] = None,
    *,
    recipe: str = "centered",
) -> Dict[int, np.ndarray]:
    """Generate one graph-scoped logical input map.

    Returned arrays are dense logical ndarrays keyed by tensor UID, consumed
    by both hipDNN buffers and reference providers. The default ``centered``
    recipe fills unconstrained floating-point operands from seeded ``N(0, 1)``
    (signed ``U[-1, 1)`` for fp16-output convolutions, ``N(0, d^-0.25)`` for
    SDPA Q/K, ``U[0, 1)`` for SDPA V and fp32/fp64 convolutions) and keeps
    positive-only normalization variance/scale inputs strictly positive in
    ``U[0.5, 1.5)``. Use ``recipe="legacy"`` for the historical ``U[0, 1)``
    distribution, ``"llm"`` for heavy-tailed model-like fills, ``"heavy_tail"``
    for an outlier stress mixture, or ``"zeros"`` for diagnostic zero inputs.

    ``graph_json`` selects the signed convolution operands and the SDPA
    tensors with invariants random data cannot satisfy. Lengths and offset
    tables come from :func:`generate_varlen_inputs`, which uses its own random
    stream, so the float values do not change. Page tables come from
    :func:`_generate_paged_input`. It also makes the O and stats inputs of an
    SDPA backward node the exact forward outputs of its Q, K and V (see
    :mod:`.sdpa_backward_inputs`). Dropout inputs get a 0/1 mask and scales
    that agree with one keep rate, and nonzero seed and offset values (see
    :func:`_generate_dropout_input`). The dropout inputs draw from their own
    generator, spawned from ``seed``, so they do not change the values of the
    other inputs.

    Raises:
        ValueError: A ``dropout_scale`` value gives a keep rate outside (0, 1].
    """
    if recipe not in INPUT_RECIPE_CHOICES:
        raise ValueError(
            f"Unknown input recipe {recipe!r}; expected one of: {', '.join(INPUT_RECIPE_CHOICES)}"
        )
    rng = np.random.default_rng(seed)
    input_data: Dict[int, np.ndarray] = {}
    page_tables = _page_table_uids(graph_json)
    dropout_roles = _dropout_input_roles(graph_json)
    dropout_keep = _dropout_keep_rate(tensor_infos, dropout_roles)
    dropout_rng = np.random.default_rng(np.random.SeedSequence(seed).spawn(1)[0])
    signed_uids = _signed_input_uids(graph_json)
    fp32_conv_uids = _fp32_conv_input_uids(graph_json)
    roles = _tensor_roles(graph_json)
    varlen_values = generate_varlen_inputs(graph_json, tensor_infos, seed)
    num_pages = _paged_metadata(graph_json) if page_tables else 0

    for tensor_info in tensor_infos:
        if tensor_info.is_output or tensor_info.is_virtual:
            continue
        dtype = tensor_info.dtype

        if tensor_info.is_pass_by_value:
            input_data[tensor_info.uid] = np.asarray(
                [tensor_info.value], dtype=dtype.numpy
            )
            continue

        dropout_role = dropout_roles.get(tensor_info.uid)
        if tensor_info.uid in varlen_values:
            input_data[tensor_info.uid] = varlen_values[tensor_info.uid]
        elif tensor_info.uid in page_tables:
            input_data[tensor_info.uid] = _generate_paged_input(tensor_info, num_pages)
        elif dropout_role is not None:
            input_data[tensor_info.uid] = _generate_dropout_input(
                tensor_info, dropout_role, dropout_keep, dropout_rng
            )
        elif dtype.numpy.kind in "iub":
            # ponytail: zeros are safe for index/offset tensors (no OOB reads);
            # per-role structured generation (e.g. MoE offsets) is the upgrade path.
            input_data[tensor_info.uid] = np.zeros(tensor_info.dims, dtype=dtype.numpy)
        else:
            # Draw and round one block at a time straight into the result, so
            # no full-size float32 draw or conversion temporary exists. The
            # blocks are drawn in C order, so the values match one full draw.
            role = roles.get(tensor_info.uid, "activation")
            is_signed = tensor_info.uid in signed_uids
            is_fp32_conv = tensor_info.uid in fp32_conv_uids
            head_dim = tensor_info.dims[-1] if tensor_info.dims else 1
            data = np.empty(tensor_info.dims, dtype=dtype.numpy)
            for index in iter_blocks(data.shape):
                block = _draw_float_block(
                    rng,
                    data[index].shape,
                    recipe=recipe,
                    role=role,
                    is_signed_conv=is_signed,
                    is_fp32_conv=is_fp32_conv,
                    head_dim=head_dim,
                    dtype=dtype,
                )
                data[index] = _decode(_encode(block, dtype), dtype)
            input_data[tensor_info.uid] = data

    set_sdpa_backward_inputs(graph_json, tensor_infos, input_data, _rounded)
    return input_data


def _rounded(values: np.ndarray, tensor_info: TensorInfo) -> np.ndarray:
    """float32 ``values`` rounded to the logical values of the tensor's type."""
    data = np.empty(tensor_info.dims, dtype=tensor_info.dtype.numpy)
    for index in iter_blocks(data.shape):
        data[index] = _decode(
            _encode(values[index], tensor_info.dtype), tensor_info.dtype
        )
    return data


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

    def _write_storage(self, buffer: Any, storage: np.ndarray) -> None:
        """Copy 1-D graph-layout storage words from the host into one buffer."""
        if self._device is None:
            # DeviceBuffer.copy_from_host accepts only ``bytes``.
            buffer.copy_from_host(storage.tobytes())
            return
        import torch

        with warnings.catch_warnings():
            # copy_ only reads the source, so a read-only view is safe and
            # avoids a host copy of the input.
            warnings.simplefilter("ignore", UserWarning)
            source = torch.frombuffer(storage, dtype=torch.uint8)
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
                self._write_storage(buffer, _encode_to_storage(data, tensor_info))

    def zero_outputs(self, nan_uids: Collection[int] = ()) -> None:
        """Zero output tensor buffers.

        Outputs in ``nan_uids`` are filled with float32 NaN instead: every
        4-byte word of the buffer, padding included. An engine that does not
        write such an output then leaves NaN, which no 0/1 check accepts.

        Raises:
            ExecutionError: If buffers not allocated.
            ValueError: If an output in ``nan_uids`` is not FLOAT.
        """
        if not self._buffers:
            raise ExecutionError("Buffers not allocated. Call allocate_all() first.")

        for uid in nan_uids:
            tensor_info = self._tensor_info_by_uid.get(uid)
            if tensor_info is None or tensor_info.dtype.name != "float":
                data_type = None if tensor_info is None else tensor_info.data_type
                raise ValueError(
                    f"Cannot fill tensor UID {uid} with NaN: only a FLOAT "
                    f"output can be filled (data type {data_type})"
                )

        for tensor_info in self._tensor_infos:
            if not tensor_info.is_output:
                continue

            buffer = self._buffers.get(tensor_info.uid)
            if buffer is None:
                continue
            fill_nan = tensor_info.uid in nan_uids
            if self._device is None:
                if fill_nan:
                    # One bytes object of the full storage size, padding
                    # included: DeviceBuffer.copy_from_host accepts only bytes.
                    buffer.copy_from_host(_NAN_WORD * (tensor_info.size_bytes // 4))
                else:
                    buffer.zeros()
            elif fill_nan:
                import torch

                buffer.view(torch.float32).fill_(math.nan)
            else:
                buffer.zero_()

        if self._device is not None:
            import torch

            # The fills run on torch's stream; hipDNN runs on the handle stream.
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
            raw = buffer.copy_to_host()
        else:
            raw = buffer.cpu().numpy()
        return _decode_storage(raw, tensor_info)

    def get_output_tensor(self, uid: int) -> Optional["torch.Tensor"]:
        """Return a typed logical view of an output buffer without a copy.

        Returns None for DeviceBuffer storage, for an unknown UID, for a
        ragged tensor, and for a data type that the installed torch cannot
        represent.
        """
        if self._device is None:
            return None
        buffer = self._buffers.get(uid)
        tensor_info = self._tensor_info_by_uid.get(uid)
        if buffer is None or tensor_info is None or tensor_info.is_ragged:
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

# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Element-space tensor artifact manifests: producer-side JSON + ``.bin`` I/O.

A manifest is a small, versioned JSON document plus sibling
``<graph>.tensor<uid>.bin`` files holding little-endian tensor storage. The
payload includes stride gaps so it is byte-compatible with hipDNN golden test
bundles. Input manifests let a caller supply reproducible external data for
every non-virtual, non-output tensor a graph needs, including pass-by-value
scalars. Output/reference manifests record what a run produced, keyed by phase
and optional producer identity.

Manifests are untrusted input: every field is validated before any byte is
trusted, decoded, or returned to a caller.
"""

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Literal, Mapping, Optional, Tuple

import numpy as np

from ..common.exceptions import ValidationError
from ..graph.tensor_info import DTYPE_SIZES, TensorInfo
from .buffer_manager import DTYPE_MAP, _f32_to_bf16_bytes

MANIFEST_FORMAT = "dnn-benchmarking-tensors"
MANIFEST_VERSION = 1
MANIFEST_LAYOUT = "element-space"
MANIFEST_BYTE_ORDER = "little"
MANIFEST_FILENAME = "manifest.json"
MANIFEST_PHASES = ("input", "output", "reference")

# Per-dtype canonical wire tag. Tensor bytes use graph strides and include
# deterministic zero-filled storage gaps.
_ENCODING_BY_DTYPE = {
    "float": "f32",
    "half": "f16",
    "bfloat16": "bf16",
    "double": "f64",
    "int8": "int8",
    "int32": "int32",
    "uint8": "uint8",
}

# Explicit little-endian storage encodings. hipDNN golden files are native
# little-endian on supported hosts; spelling it here keeps manifests portable.
_WIRE_DTYPE_CODE = {
    "float": "f4",
    "half": "f2",
    "double": "f8",
    "int8": "i1",
    "int32": "i4",
    "uint8": "u1",
}


_UID_RE = re.compile(r"-?[0-9]+")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def graph_fingerprint(graph_json: Dict[str, Any]) -> str:
    """Stable content hash of a graph, used to match manifests to graphs."""
    canonical = json.dumps(graph_json, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_input_tensors(
    source: Path, graph_json: Dict[str, Any], tensor_infos: List[TensorInfo]
) -> Dict[int, np.ndarray]:
    """Load every non-virtual, non-output tensor from an input manifest.

    ``source`` is either a single manifest JSON file, or a directory
    searched recursively for exactly one manifest with ``phase == "input"``
    whose recorded graph hash matches ``graph_json``. The manifest must
    supply exactly the graph's non-virtual, non-output tensors (including
    pass-by-value ones, whose decoded value is cross-checked against the
    value already embedded in the graph).

    Returns dense, C-contiguous logical arrays keyed by tensor UID. The
    manifest payload itself preserves graph stride gaps. bfloat16 values are
    exposed as float32, matching the rest of the codebase.
    """
    source = Path(source)
    fingerprint = graph_fingerprint(graph_json)

    if source.is_dir():
        manifest_path, manifest = _find_input_manifest(source, fingerprint)
    elif source.is_file():
        manifest_path = source
        manifest = _load_manifest_json(manifest_path)
        phase, graph, _tensors = _validate_manifest_document(manifest, manifest_path)
        if phase != "input":
            raise ValidationError(
                f"{manifest_path}: expected phase 'input', found {phase!r}"
            )
        if graph["sha256"] != fingerprint:
            raise ValidationError(
                f"{manifest_path}: manifest graph sha256 does not match the supplied graph"
            )
    else:
        raise ValidationError(f"{source}: input manifest source does not exist")

    manifest_dir = manifest_path.parent
    by_uid = _tensor_entries_by_uid(manifest["tensors"], manifest_path)

    info_by_uid = {ti.uid: ti for ti in tensor_infos}
    required_uids = {
        ti.uid for ti in tensor_infos if not ti.is_virtual and not ti.is_output
    }
    manifest_uids = set(by_uid)

    unknown = manifest_uids - set(info_by_uid)
    if unknown:
        raise ValidationError(
            f"{manifest_path}: manifest references unknown tensor uid(s): {sorted(unknown)}"
        )
    missing = required_uids - manifest_uids
    if missing:
        raise ValidationError(
            f"{manifest_path}: manifest missing required input tensor uid(s): {sorted(missing)}"
        )
    extra = manifest_uids - required_uids
    if extra:
        raise ValidationError(
            f"{manifest_path}: manifest supplies non-input tensor uid(s): {sorted(extra)}"
        )

    result: Dict[int, np.ndarray] = {}
    for uid in required_uids:
        entry = by_uid[uid]
        tensor_info = info_by_uid[uid]
        _validate_entry_matches_tensor_info(
            entry, tensor_info, manifest["graph"]["path"], manifest_path
        )
        data_bytes = _read_and_verify_tensor_bytes(manifest_dir, entry, manifest_path)
        data_type = tensor_info.data_type.lower()
        array = _decode_element_space(data_bytes, tensor_info, data_type)
        if tensor_info.is_pass_by_value:
            _verify_pass_by_value(array, tensor_info, data_type, manifest_path)
        result[uid] = array
    return result


def write_tensor_manifest(
    root: Path,
    graph_path: Path,
    graph_json: Dict[str, Any],
    tensor_infos: List[TensorInfo],
    tensors: Mapping[int, np.ndarray],
    *,
    phase: Literal["input", "output", "reference"],
    provider: Optional[str] = None,
    engine_id: Optional[int] = None,
    artifact_key: Optional[str] = None,
) -> Path:
    """Write an element-space manifest and sibling ``.bin`` files.

    Every tensor must match its graph shape and logical dtype. The target
    directory is derived from the graph name, its content hash, ``phase``,
    and the optional ``provider``/``engine_id``/``artifact_key`` so repeated
    engines or suites never collide; user-controlled segments are sanitized.
    The manifest file itself is replaced atomically.
    """
    if phase not in MANIFEST_PHASES:
        raise ValueError(f"Unsupported manifest phase: {phase!r}")

    info_by_uid = {ti.uid: ti for ti in tensor_infos}
    unknown = set(tensors) - set(info_by_uid)
    if unknown:
        raise ValidationError(
            f"write_tensor_manifest: unknown tensor uid(s) requested: {sorted(unknown)}"
        )
    required = {
        ti.uid
        for ti in tensor_infos
        if not ti.is_virtual
        and (
            (phase == "input" and not ti.is_output)
            or (phase != "input" and ti.is_output)
        )
    }
    if set(tensors) != required:
        missing = sorted(required - set(tensors))
        extra = sorted(set(tensors) - required)
        raise ValidationError(
            f"write_tensor_manifest: {phase} tensor set mismatch; "
            f"missing={missing}, extra={extra}"
        )

    fingerprint = graph_fingerprint(graph_json)
    graph_name = str(graph_json.get("name", "unnamed_graph"))

    segments = [f"{_sanitize_segment(graph_name)}-{fingerprint[:16]}", phase]
    if provider is not None:
        segments.append(_sanitize_segment(provider))
    if engine_id is not None:
        segments.append(str(int(engine_id)))
    if artifact_key is not None:
        segments.append(_sanitize_segment(artifact_key))

    manifest_dir = Path(root).joinpath(*segments)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    tensor_entries = []
    for uid in sorted(tensors):
        array = tensors[uid]
        tensor_info = info_by_uid[uid]
        data_type = tensor_info.data_type.lower()
        if data_type not in DTYPE_SIZES:
            raise ValidationError(
                f"write_tensor_manifest: unsupported data_type {tensor_info.data_type!r} for uid {uid}"
            )
        if list(array.shape) != list(tensor_info.dims):
            raise ValidationError(
                f"write_tensor_manifest: tensor uid {uid} shape {list(array.shape)} "
                f"does not match graph shape {list(tensor_info.dims)}"
            )
        if phase == "input" and tensor_info.is_pass_by_value:
            _verify_pass_by_value(array, tensor_info, data_type, Path("write"))
        expected_dtype = _canonical_dtype(data_type)
        if np.dtype(array.dtype) != expected_dtype:
            raise ValidationError(
                f"write_tensor_manifest: tensor uid {uid} dtype {array.dtype} does not match "
                f"expected {expected_dtype} for data_type {data_type!r}"
            )
        data_bytes = _encode_element_space(array, tensor_info, data_type)
        digest = hashlib.sha256(data_bytes).hexdigest()
        file_name = _tensor_file_name(graph_path, uid)
        _atomic_write_bytes(manifest_dir / file_name, data_bytes)
        tensor_entries.append(
            {
                "uid": str(uid),
                "name": str(tensor_info.name),
                "data_type": data_type,
                "shape": list(tensor_info.dims),
                "graph_strides": list(tensor_info.strides),
                "storage_elements": tensor_info.storage_elements,
                "encoding": _ENCODING_BY_DTYPE[data_type],
                "file": file_name,
                "byte_length": len(data_bytes),
                "sha256": digest,
            }
        )

    manifest: Dict[str, Any] = {
        "format": MANIFEST_FORMAT,
        "version": MANIFEST_VERSION,
        "layout": MANIFEST_LAYOUT,
        "byte_order": MANIFEST_BYTE_ORDER,
        "graph": {
            "name": graph_name,
            "path": str(graph_path),
            "sha256": fingerprint,
        },
        "phase": phase,
        "tensors": tensor_entries,
    }
    if provider is not None or engine_id is not None:
        producer: Dict[str, str] = {}
        if provider is not None:
            producer["provider"] = provider
        if engine_id is not None:
            producer["engine_id"] = str(int(engine_id))
        manifest["producer"] = producer

    manifest_path = manifest_dir / MANIFEST_FILENAME
    payload = json.dumps(manifest, sort_keys=True, indent=2) + "\n"
    _atomic_write_bytes(manifest_path, payload.encode("utf-8"))
    return manifest_path


# --------------------------------------------------------------------------
# Element-space encode/decode. The graph strides define logical offsets into
# the payload. Storage gaps are zero-filled when writing.
# --------------------------------------------------------------------------


def _canonical_dtype(data_type: str) -> np.dtype:
    """Numpy dtype for logical (decoded) values of ``data_type``.

    bfloat16 has no native numpy dtype; its logical values are exposed as
    float32, matching buffer_manager's convention.
    """
    if data_type == "bfloat16":
        return np.dtype(np.float32)
    dtype = DTYPE_MAP.get(data_type)
    if dtype is None:
        raise ValidationError(f"Unsupported tensor data_type: {data_type!r}")
    return np.dtype(dtype)


def _storage_view(storage: np.ndarray, tensor_info: TensorInfo) -> np.ndarray:
    if tensor_info.strides:
        return np.lib.stride_tricks.as_strided(
            storage,
            shape=tuple(tensor_info.dims),
            strides=tuple(
                stride * storage.dtype.itemsize for stride in tensor_info.strides
            ),
        )
    return storage.reshape(tensor_info.dims)


def _encode_element_space(
    data: np.ndarray, tensor_info: TensorInfo, data_type: str
) -> bytes:
    """Encode logical values into zero-filled, little-endian element storage."""
    if data_type == "bfloat16":
        logical_words = np.frombuffer(
            _f32_to_bf16_bytes(np.ascontiguousarray(data, dtype=np.float32)),
            dtype=np.uint16,
        ).reshape(tensor_info.dims)
        storage = np.zeros(tensor_info.storage_elements, dtype="<u2")
        _storage_view(storage, tensor_info)[...] = logical_words
    else:
        wire_dtype = np.dtype("<" + _WIRE_DTYPE_CODE[data_type])
        storage = np.zeros(tensor_info.storage_elements, dtype=wire_dtype)
        _storage_view(storage, tensor_info)[...] = data
    return storage.tobytes()


def _decode_element_space(
    data_bytes: bytes, tensor_info: TensorInfo, data_type: str
) -> np.ndarray:
    """Decode little-endian graph storage into a dense logical ndarray."""
    if data_type == "bfloat16":
        storage = np.frombuffer(
            data_bytes, dtype="<u2", count=tensor_info.storage_elements
        )
        logical_words = np.ascontiguousarray(_storage_view(storage, tensor_info))
        f32_bits = logical_words.astype(np.uint32) << np.uint32(16)
        return np.ascontiguousarray(f32_bits.view(np.float32).reshape(tensor_info.dims))
    wire_dtype = np.dtype("<" + _WIRE_DTYPE_CODE[data_type])
    storage = np.frombuffer(
        data_bytes, dtype=wire_dtype, count=tensor_info.storage_elements
    )
    logical = _storage_view(storage, tensor_info)
    return np.ascontiguousarray(logical, dtype=DTYPE_MAP[data_type])


def _load_manifest_json(path: Path) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValidationError(f"{path}: invalid manifest JSON: {e}") from e
    except OSError as e:
        raise ValidationError(f"{path}: cannot read manifest: {e}") from e


def _validate_manifest_document(
    manifest: Any, path: Path
) -> Tuple[str, Dict[str, Any], List[Any]]:
    """Validate manifest-level structure. Returns (phase, graph, tensors)."""
    if not isinstance(manifest, dict):
        raise ValidationError(f"{path}: manifest root must be a JSON object")
    if manifest.get("format") != MANIFEST_FORMAT:
        raise ValidationError(
            f"{path}: unsupported manifest format {manifest.get('format')!r}"
        )
    if manifest.get("version") != MANIFEST_VERSION:
        raise ValidationError(
            f"{path}: unsupported manifest version {manifest.get('version')!r}"
        )
    if manifest.get("layout") != MANIFEST_LAYOUT:
        raise ValidationError(
            f"{path}: unsupported manifest layout {manifest.get('layout')!r}"
        )
    if manifest.get("byte_order") != MANIFEST_BYTE_ORDER:
        raise ValidationError(
            f"{path}: unsupported manifest byte_order {manifest.get('byte_order')!r}"
        )

    phase = manifest.get("phase")
    if phase not in MANIFEST_PHASES:
        raise ValidationError(f"{path}: invalid manifest phase {phase!r}")

    graph = manifest.get("graph")
    if not isinstance(graph, dict):
        raise ValidationError(f"{path}: manifest.graph must be an object")
    for key in ("name", "path", "sha256"):
        if not isinstance(graph.get(key), str):
            raise ValidationError(f"{path}: manifest.graph.{key} must be a string")
    if not _SHA256_RE.fullmatch(graph["sha256"]):
        raise ValidationError(
            f"{path}: manifest.graph.sha256 must be a 64-hex-char digest"
        )

    producer = manifest.get("producer")
    if producer is not None:
        if not isinstance(producer, dict):
            raise ValidationError(f"{path}: manifest.producer must be an object")
        if "provider" in producer and not isinstance(producer["provider"], str):
            raise ValidationError(
                f"{path}: manifest.producer.provider must be a string"
            )
        engine_id = producer.get("engine_id")
        if engine_id is not None and not (
            isinstance(engine_id, str) and _UID_RE.fullmatch(engine_id)
        ):
            raise ValidationError(
                f"{path}: manifest.producer.engine_id must be a decimal string"
            )

    tensors = manifest.get("tensors")
    if not isinstance(tensors, list):
        raise ValidationError(f"{path}: manifest.tensors must be a list")
    for index, entry in enumerate(tensors):
        _validate_tensor_entry_shape(entry, path, index)

    return phase, graph, tensors


def _validate_tensor_entry_shape(entry: Any, manifest_path: Path, index: int) -> None:
    """Validate one manifest tensor entry's JSON shape, independent of any graph."""
    if not isinstance(entry, dict):
        raise ValidationError(f"{manifest_path}: tensors[{index}] must be an object")
    for key in ("uid", "name", "data_type", "encoding", "file", "sha256"):
        if not isinstance(entry.get(key), str):
            raise ValidationError(
                f"{manifest_path}: tensors[{index}].{key} must be a string"
            )

    if not _UID_RE.fullmatch(entry["uid"]):
        raise ValidationError(
            f"{manifest_path}: tensors[{index}].uid must be a decimal string"
        )

    data_type = entry["data_type"]
    if data_type not in DTYPE_SIZES:
        raise ValidationError(
            f"{manifest_path}: tensors[{index}].data_type {data_type!r} is unsupported"
        )
    if entry["encoding"] != _ENCODING_BY_DTYPE[data_type]:
        raise ValidationError(
            f"{manifest_path}: tensors[{index}].encoding {entry['encoding']!r} does not match "
            f"data_type {data_type!r}"
        )

    shape = entry.get("shape")
    if not isinstance(shape, list) or not all(
        isinstance(d, int) and not isinstance(d, bool) and d >= 0 for d in shape
    ):
        raise ValidationError(
            f"{manifest_path}: tensors[{index}].shape must be a list of non-negative ints"
        )
    strides = entry.get("graph_strides")
    if not isinstance(strides, list) or not all(
        isinstance(s, int) and not isinstance(s, bool) for s in strides
    ):
        raise ValidationError(
            f"{manifest_path}: tensors[{index}].graph_strides must be a list of ints"
        )

    storage_elements = entry.get("storage_elements")
    if (
        not isinstance(storage_elements, int)
        or isinstance(storage_elements, bool)
        or storage_elements < 0
    ):
        raise ValidationError(
            f"{manifest_path}: tensors[{index}].storage_elements must be a non-negative int"
        )
    byte_length = entry.get("byte_length")
    if (
        not isinstance(byte_length, int)
        or isinstance(byte_length, bool)
        or byte_length < 0
    ):
        raise ValidationError(
            f"{manifest_path}: tensors[{index}].byte_length must be a non-negative int"
        )
    if not _SHA256_RE.fullmatch(entry["sha256"]):
        raise ValidationError(
            f"{manifest_path}: tensors[{index}].sha256 must be a 64-hex-char digest"
        )

    file_name = entry["file"]
    if not file_name or file_name in (".", "..") or Path(file_name).name != file_name:
        raise ValidationError(
            f"{manifest_path}: tensors[{index}].file must be a plain basename"
        )


def _tensor_entries_by_uid(
    tensors: List[Dict[str, Any]], manifest_path: Path
) -> Dict[int, Dict[str, Any]]:
    by_uid: Dict[int, Dict[str, Any]] = {}
    for entry in tensors:
        uid = int(entry["uid"])
        if uid in by_uid:
            raise ValidationError(
                f"{manifest_path}: duplicate tensor uid {uid} in manifest"
            )
        by_uid[uid] = entry
    return by_uid


def _validate_entry_matches_tensor_info(
    entry: Dict[str, Any],
    tensor_info: TensorInfo,
    graph_path: str,
    manifest_path: Path,
) -> None:
    """Cross-check a manifest tensor entry against the graph's own TensorInfo."""
    if entry["data_type"] != tensor_info.data_type.lower():
        raise ValidationError(
            f"{manifest_path}: tensor uid {tensor_info.uid} data_type mismatch"
        )
    if entry["shape"] != list(tensor_info.dims):
        raise ValidationError(
            f"{manifest_path}: tensor uid {tensor_info.uid} shape mismatch"
        )
    if entry["graph_strides"] != list(tensor_info.strides):
        raise ValidationError(
            f"{manifest_path}: tensor uid {tensor_info.uid} graph_strides mismatch"
        )

    expected_file = _tensor_file_name(Path(graph_path), tensor_info.uid)
    if entry["file"] != expected_file:
        raise ValidationError(
            f"{manifest_path}: tensor uid {tensor_info.uid} file must be "
            f"{expected_file!r}"
        )

    if entry["storage_elements"] != tensor_info.storage_elements:
        raise ValidationError(
            f"{manifest_path}: tensor uid {tensor_info.uid} storage_elements mismatch: "
            f"expected {tensor_info.storage_elements}, got {entry['storage_elements']}"
        )
    expected_len = tensor_info.size_bytes
    if entry["byte_length"] != expected_len:
        raise ValidationError(
            f"{manifest_path}: tensor uid {tensor_info.uid} byte_length mismatch: "
            f"expected {expected_len}, got {entry['byte_length']}"
        )


def _read_and_verify_tensor_bytes(
    manifest_dir: Path, entry: Dict[str, Any], manifest_path: Path
) -> bytes:
    """Read a tensor's ``.bin`` file, enforcing containment, exact size, and hash."""
    manifest_dir_resolved = manifest_dir.resolve()
    file_path = (manifest_dir / entry["file"]).resolve()
    if file_path.parent != manifest_dir_resolved:
        raise ValidationError(
            f"{manifest_path}: tensor file {entry['file']!r} escapes the manifest directory"
        )
    try:
        actual_size = file_path.stat().st_size
    except OSError as e:
        raise ValidationError(
            f"{manifest_path}: cannot stat tensor file {entry['file']!r}: {e}"
        ) from e
    if actual_size != entry["byte_length"]:
        raise ValidationError(
            f"{manifest_path}: tensor file {entry['file']!r} size mismatch: "
            f"expected {entry['byte_length']} bytes, got {actual_size}"
        )
    try:
        data = file_path.read_bytes()
    except OSError as e:
        raise ValidationError(
            f"{manifest_path}: cannot read tensor file {entry['file']!r}: {e}"
        ) from e
    digest = hashlib.sha256(data).hexdigest()
    if digest != entry["sha256"]:
        raise ValidationError(
            f"{manifest_path}: tensor file {entry['file']!r} sha256 mismatch"
        )
    return data


def _verify_pass_by_value(
    array: np.ndarray, tensor_info: TensorInfo, data_type: str, manifest_path: Path
) -> None:
    """Cross-check a decoded pass-by-value tensor against the graph's embedded value."""
    expected = np.asarray(tensor_info.value, dtype=_canonical_dtype(data_type)).reshape(
        array.shape
    )
    if data_type == "bfloat16":
        # bf16 has no native numpy dtype: round the embedded value through the
        # same RNE storage path a device buffer would take before comparing.
        rounded_bytes = _f32_to_bf16_bytes(
            np.ascontiguousarray(expected, dtype=np.float32)
        )
        expected = _bfloat16_bytes_to_ndarray(rounded_bytes, list(expected.shape))
    if not np.array_equal(array, expected):
        raise ValidationError(
            f"{manifest_path}: pass-by-value tensor uid {tensor_info.uid} does not match "
            "the value embedded in the graph"
        )


def _find_input_manifest(root: Path, fingerprint: str) -> Tuple[Path, Dict[str, Any]]:
    """Recursively find the single input manifest matching ``fingerprint`` under ``root``."""
    matches: List[Path] = []
    matched_manifest: Optional[Dict[str, Any]] = None
    for path in sorted(root.rglob("*.json")):
        try:
            with open(path, "r", encoding="utf-8") as f:
                candidate = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        if (
            not isinstance(candidate, dict)
            or candidate.get("format") != MANIFEST_FORMAT
        ):
            continue
        try:
            phase, graph, _tensors = _validate_manifest_document(candidate, path)
        except ValidationError:
            continue
        if phase != "input" or graph["sha256"] != fingerprint:
            continue
        matches.append(path)
        matched_manifest = candidate

    if not matches:
        raise ValidationError(
            f"{root}: no input manifest found matching the supplied graph"
        )
    if len(matches) > 1:
        raise ValidationError(
            f"{root}: ambiguous input manifests match the supplied graph: "
            + ", ".join(str(p) for p in matches)
        )
    assert matched_manifest is not None
    return matches[0], matched_manifest


# --------------------------------------------------------------------------
# Writer path helpers.
# --------------------------------------------------------------------------


def _sanitize_segment(value: str) -> str:
    """Collapse a user/provider-supplied string into a safe path segment."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "_", value).replace("..", "_")
    cleaned = cleaned.strip("._")
    return cleaned or "unknown"


def _tensor_file_name(graph_path: Path, uid: int) -> str:
    """Return the hipDNN golden-bundle tensor filename for one UID."""
    return f"{_sanitize_segment(graph_path.stem)}.tensor{uid}.bin"


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path`` via a same-directory temp file + rename."""
    tmp_path = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    tmp_path.write_bytes(data)
    os.replace(tmp_path, path)

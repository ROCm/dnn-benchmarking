# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for the element-space tensor artifact manifest contract.

Exercises round-trip storage, hipDNN-compatible filenames, multi-graph
selection, and trust-boundary rejection paths.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from dnn_benchmarking.common.exceptions import ValidationError
from dnn_benchmarking.execution.tensor_artifacts import (
    graph_fingerprint,
    load_input_tensors,
    write_tensor_manifest,
)
from dnn_benchmarking.graph.tensor_info import TensorInfo


def _tensor_info(
    uid, dims, strides, data_type, is_output=False, is_virtual=False, value=None
):
    return TensorInfo(
        uid=uid,
        name=f"tensor_{uid}",
        dims=dims,
        strides=strides,
        data_type=data_type,
        is_virtual=is_virtual,
        is_output=is_output,
        value=value,
    )


def _read_manifest(manifest_path: Path) -> dict:
    return json.loads(manifest_path.read_text())


def test_graph_fingerprint_is_stable_regardless_of_key_order():
    """Fingerprint depends only on content, not on dict insertion order."""
    assert graph_fingerprint({"a": 1, "b": 2}) == graph_fingerprint({"b": 2, "a": 1})
    assert graph_fingerprint({"a": 1}) != graph_fingerprint({"a": 2})


def test_roundtrip_strided_float_tensor(tmp_path):
    """Element-space bytes preserve strided logical values and zero padding."""
    graph_json = {"name": "graphA", "nodes": []}
    tensor_info = _tensor_info(uid=1, dims=[2, 2], strides=[3, 1], data_type="float")
    array = np.arange(4, dtype=np.float32).reshape(2, 2)

    manifest_path = write_tensor_manifest(
        root=tmp_path,
        graph_path=Path("/graphs/a.hipdnn.json"),
        graph_json=graph_json,
        tensor_infos=[tensor_info],
        tensors={1: array},
        phase="input",
    )

    manifest = _read_manifest(manifest_path)
    assert manifest["format"] == "dnn-benchmarking-tensors"
    assert manifest["version"] == 1
    assert manifest["layout"] == "element-space"
    assert manifest["byte_order"] == "little"
    assert manifest["phase"] == "input"
    assert manifest["graph"]["name"] == "graphA"
    assert manifest["graph"]["path"] == "/graphs/a.hipdnn.json"
    assert manifest["graph"]["sha256"] == graph_fingerprint(graph_json)

    (entry,) = manifest["tensors"]
    assert entry["uid"] == "1"
    assert entry["shape"] == [2, 2]
    assert entry["graph_strides"] == [3, 1]
    assert entry["encoding"] == "f32"
    assert entry["file"] == "a.hipdnn.tensor1.bin"
    assert entry["storage_elements"] == 5
    assert entry["byte_length"] == 20
    assert len(entry["sha256"]) == 64

    loaded = load_input_tensors(manifest_path, graph_json, [tensor_info])
    assert set(loaded) == {1}
    assert loaded[1].flags["C_CONTIGUOUS"]
    np.testing.assert_array_equal(loaded[1], array)
    raw = (manifest_path.parent / entry["file"]).read_bytes()
    assert raw == np.array([0.0, 1.0, 0.0, 2.0, 3.0], dtype="<f4").tobytes()

    # A directory source finds the same manifest recursively.
    loaded_from_dir = load_input_tensors(tmp_path, graph_json, [tensor_info])
    np.testing.assert_array_equal(loaded_from_dir[1], array)


def test_roundtrip_bfloat16_tensor(tmp_path):
    """bfloat16 values round-trip exactly for values already bf16-representable."""
    graph_json = {"name": "graphBf16"}
    tensor_info = _tensor_info(uid=2, dims=[4], strides=[1], data_type="bfloat16")
    array = np.array([1.0, -2.0, 0.5, 0.0], dtype=np.float32)

    manifest_path = write_tensor_manifest(
        root=tmp_path,
        graph_path=Path("/g.json"),
        graph_json=graph_json,
        tensor_infos=[tensor_info],
        tensors={2: array},
        phase="input",
    )
    entry = _read_manifest(manifest_path)["tensors"][0]
    assert entry["data_type"] == "bfloat16"
    assert entry["encoding"] == "bf16"
    assert entry["byte_length"] == 8  # 4 elements * 2-byte bf16 words
    assert entry["storage_elements"] == 4
    assert (manifest_path.parent / entry["file"]).read_bytes() == np.array(
        [0x3F80, 0xC000, 0x3F00, 0x0000], dtype="<u2"
    ).tobytes()

    loaded = load_input_tensors(manifest_path, graph_json, [tensor_info])
    assert loaded[2].dtype == np.float32
    np.testing.assert_array_equal(loaded[2], array)


def test_directory_selects_matching_graph_among_several(tmp_path):
    """Directory search picks the one manifest whose graph hash matches."""
    graph_a = {"name": "graphA"}
    graph_b = {"name": "graphB"}
    info_a = [_tensor_info(uid=1, dims=[2], strides=[1], data_type="float")]
    info_b = [_tensor_info(uid=10, dims=[3], strides=[1], data_type="int32")]
    arr_a = np.array([1.0, 2.0], dtype=np.float32)
    arr_b = np.array([7, 8, 9], dtype=np.int32)

    write_tensor_manifest(
        root=tmp_path,
        graph_path=Path("/a.json"),
        graph_json=graph_a,
        tensor_infos=info_a,
        tensors={1: arr_a},
        phase="input",
    )
    write_tensor_manifest(
        root=tmp_path,
        graph_path=Path("/b.json"),
        graph_json=graph_b,
        tensor_infos=info_b,
        tensors={10: arr_b},
        phase="input",
    )

    loaded = load_input_tensors(tmp_path, graph_a, info_a)
    assert set(loaded) == {1}
    np.testing.assert_array_equal(loaded[1], arr_a)


def test_graph_mismatch_is_rejected(tmp_path):
    """A manifest whose recorded graph hash differs from the supplied graph errors."""
    graph_a = {"name": "graphA"}
    graph_b = {"name": "graphB"}
    info_a = [_tensor_info(uid=1, dims=[2], strides=[1], data_type="float")]
    arr_a = np.array([1.0, 2.0], dtype=np.float32)

    manifest_path = write_tensor_manifest(
        root=tmp_path,
        graph_path=Path("/a.json"),
        graph_json=graph_a,
        tensor_infos=info_a,
        tensors={1: arr_a},
        phase="input",
    )

    with pytest.raises(ValidationError, match="sha256"):
        load_input_tensors(manifest_path, graph_b, info_a)


def test_path_traversal_file_field_is_rejected(tmp_path):
    """A tensor entry whose file field escapes the manifest directory is rejected."""
    graph_json = {"name": "graphA"}
    tensor_info = _tensor_info(uid=1, dims=[2], strides=[1], data_type="float")
    manifest_path = write_tensor_manifest(
        root=tmp_path,
        graph_path=Path("/a.json"),
        graph_json=graph_json,
        tensor_infos=[tensor_info],
        tensors={1: np.array([1.0, 2.0], dtype=np.float32)},
        phase="input",
    )

    manifest = _read_manifest(manifest_path)
    manifest["tensors"][0]["file"] = "../escape.bin"
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValidationError, match="basename"):
        load_input_tensors(manifest_path, graph_json, [tensor_info])


def test_checksum_tampering_is_rejected(tmp_path):
    """A bin file whose bytes no longer hash to the manifest's sha256 is rejected."""
    graph_json = {"name": "graphA"}
    tensor_info = _tensor_info(uid=1, dims=[2], strides=[1], data_type="float")
    manifest_path = write_tensor_manifest(
        root=tmp_path,
        graph_path=Path("/a.json"),
        graph_json=graph_json,
        tensor_infos=[tensor_info],
        tensors={1: np.array([1.0, 2.0], dtype=np.float32)},
        phase="input",
    )

    bin_path = manifest_path.parent / "a.tensor1.bin"
    data = bytearray(bin_path.read_bytes())
    data[0] ^= 0xFF  # Same length, different content -> checksum mismatch.
    bin_path.write_bytes(bytes(data))

    with pytest.raises(ValidationError, match="sha256 mismatch"):
        load_input_tensors(manifest_path, graph_json, [tensor_info])


def test_size_tampering_is_rejected(tmp_path):
    """A bin file whose size no longer matches byte_length is rejected."""
    graph_json = {"name": "graphA"}
    tensor_info = _tensor_info(uid=1, dims=[2], strides=[1], data_type="float")
    manifest_path = write_tensor_manifest(
        root=tmp_path,
        graph_path=Path("/a.json"),
        graph_json=graph_json,
        tensor_infos=[tensor_info],
        tensors={1: np.array([1.0, 2.0], dtype=np.float32)},
        phase="input",
    )

    bin_path = manifest_path.parent / "a.tensor1.bin"
    bin_path.write_bytes(bin_path.read_bytes() + b"\x00\x00\x00\x00")

    with pytest.raises(ValidationError, match="size mismatch"):
        load_input_tensors(manifest_path, graph_json, [tensor_info])


def test_directory_ignores_malformed_unrelated_manifest(tmp_path):
    graph_json = {"name": "graphA"}
    tensor_info = _tensor_info(uid=1, dims=[2], strides=[1], data_type="float")
    valid = write_tensor_manifest(
        root=tmp_path,
        graph_path=Path("/a.json"),
        graph_json=graph_json,
        tensor_infos=[tensor_info],
        tensors={1: np.array([1.0, 2.0], dtype=np.float32)},
        phase="input",
    )
    malformed = tmp_path / "unrelated" / "manifest.json"
    malformed.parent.mkdir()
    malformed.write_text(
        json.dumps(
            {
                "format": "dnn-benchmarking-tensors",
                "version": 99,
            }
        )
    )

    loaded = load_input_tensors(tmp_path, graph_json, [tensor_info])

    assert valid.exists()
    np.testing.assert_array_equal(loaded[1], [1.0, 2.0])


@pytest.mark.parametrize("field", ["shape", "graph_strides"])
def test_manifest_dimension_booleans_are_rejected(tmp_path, field):
    graph_json = {"name": "graphA"}
    tensor_info = _tensor_info(uid=1, dims=[1], strides=[1], data_type="float")
    manifest_path = write_tensor_manifest(
        root=tmp_path,
        graph_path=Path("/a.json"),
        graph_json=graph_json,
        tensor_infos=[tensor_info],
        tensors={1: np.array([1.0], dtype=np.float32)},
        phase="input",
    )
    manifest = _read_manifest(manifest_path)
    manifest["tensors"][0][field] = [True]
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValidationError, match=field):
        load_input_tensors(manifest_path, graph_json, [tensor_info])


def test_storage_element_mismatch_is_rejected(tmp_path):
    graph_json = {"name": "graphA"}
    tensor_info = _tensor_info(uid=1, dims=[2], strides=[1], data_type="float")
    manifest_path = write_tensor_manifest(
        root=tmp_path,
        graph_path=Path("/a.json"),
        graph_json=graph_json,
        tensor_infos=[tensor_info],
        tensors={1: np.array([1.0, 2.0], dtype=np.float32)},
        phase="input",
    )
    manifest = _read_manifest(manifest_path)
    manifest["tensors"][0]["storage_elements"] = 3
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValidationError, match="storage_elements mismatch"):
        load_input_tensors(manifest_path, graph_json, [tensor_info])


def test_writer_rejects_manifest_missing_required_input(tmp_path):
    """The writer requires a complete replayable input set."""
    graph_json = {"name": "graphA"}
    info = [
        _tensor_info(uid=1, dims=[2], strides=[1], data_type="float"),
        _tensor_info(uid=2, dims=[2], strides=[1], data_type="float"),
    ]

    with pytest.raises(ValidationError, match=r"missing=\[2\]"):
        write_tensor_manifest(
            root=tmp_path,
            graph_path=Path("/a.json"),
            graph_json=graph_json,
            tensor_infos=info,
            tensors={1: np.array([1.0, 2.0], dtype=np.float32)},
            phase="input",
        )


def test_duplicate_tensor_uid_is_rejected(tmp_path):
    """Two tensor entries sharing a uid are rejected rather than silently merged."""
    graph_json = {"name": "graphA"}
    tensor_info = _tensor_info(uid=1, dims=[2], strides=[1], data_type="float")
    manifest_path = write_tensor_manifest(
        root=tmp_path,
        graph_path=Path("/a.json"),
        graph_json=graph_json,
        tensor_infos=[tensor_info],
        tensors={1: np.array([1.0, 2.0], dtype=np.float32)},
        phase="input",
    )

    manifest = _read_manifest(manifest_path)
    manifest["tensors"].append(dict(manifest["tensors"][0]))
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValidationError, match="duplicate tensor uid"):
        load_input_tensors(manifest_path, graph_json, [tensor_info])


def test_writer_rejects_pass_by_value_mismatch(tmp_path):
    """The writer refuses replay data that disagrees with the graph scalar."""
    graph_json = {"name": "graphA"}
    tensor_info = _tensor_info(
        uid=5, dims=[1], strides=[], data_type="float", value=3.0
    )

    with pytest.raises(ValidationError, match="pass-by-value tensor uid 5"):
        write_tensor_manifest(
            root=tmp_path,
            graph_path=Path("/a.json"),
            graph_json=graph_json,
            tensor_infos=[tensor_info],
            tensors={5: np.array([9.0], dtype=np.float32)},
            phase="input",
        )


def test_pass_by_value_matching_is_accepted(tmp_path):
    """A pass-by-value tensor whose manifest value matches the embedded scalar loads fine."""
    graph_json = {"name": "graphA"}
    tensor_info = _tensor_info(
        uid=5, dims=[1], strides=[], data_type="float", value=3.0
    )
    manifest_path = write_tensor_manifest(
        root=tmp_path,
        graph_path=Path("/a.json"),
        graph_json=graph_json,
        tensor_infos=[tensor_info],
        tensors={5: np.array([3.0], dtype=np.float32)},
        phase="input",
    )

    loaded = load_input_tensors(manifest_path, graph_json, [tensor_info])
    np.testing.assert_array_equal(loaded[5], np.array([3.0], dtype=np.float32))


def test_unknown_supplied_uid_is_rejected_by_writer(tmp_path):
    """The writer refuses tensors keyed by a uid absent from tensor_infos."""
    graph_json = {"name": "graphA"}
    tensor_info = _tensor_info(uid=1, dims=[2], strides=[1], data_type="float")

    with pytest.raises(ValidationError, match="unknown tensor uid"):
        write_tensor_manifest(
            root=tmp_path,
            graph_path=Path("/a.json"),
            graph_json=graph_json,
            tensor_infos=[tensor_info],
            tensors={99: np.array([1.0, 2.0], dtype=np.float32)},
            phase="input",
        )


def test_shape_mismatch_is_rejected_by_writer(tmp_path):
    """The writer refuses an array whose shape disagrees with the tensor's graph shape."""
    graph_json = {"name": "graphA"}
    tensor_info = _tensor_info(uid=1, dims=[2, 3], strides=[3, 1], data_type="float")

    with pytest.raises(ValidationError, match="shape"):
        write_tensor_manifest(
            root=tmp_path,
            graph_path=Path("/a.json"),
            graph_json=graph_json,
            tensor_infos=[tensor_info],
            tensors={1: np.zeros((3, 2), dtype=np.float32)},
            phase="input",
        )


def test_engine_id_serializes_as_decimal_string(tmp_path):
    """producer.engine_id is written as a decimal string, not a JSON number."""
    graph_json = {"name": "graphA"}
    tensor_info = _tensor_info(
        uid=1, dims=[2], strides=[1], data_type="float", is_output=True
    )
    manifest_path = write_tensor_manifest(
        root=tmp_path,
        graph_path=Path("/a.json"),
        graph_json=graph_json,
        tensor_infos=[tensor_info],
        tensors={1: np.array([1.0, 2.0], dtype=np.float32)},
        phase="output",
        provider="hipblaslt",
        engine_id=7,
        artifact_key="run-0",
    )
    manifest = _read_manifest(manifest_path)
    assert manifest["producer"] == {"provider": "hipblaslt", "engine_id": "7"}
    assert manifest["phase"] == "output"


def test_writer_sanitizes_hostile_provider_segment(tmp_path):
    """A provider string containing path-traversal characters cannot escape root."""
    graph_json = {"name": "graphA"}
    tensor_info = _tensor_info(
        uid=1, dims=[2], strides=[1], data_type="float", is_output=True
    )
    manifest_path = write_tensor_manifest(
        root=tmp_path,
        graph_path=Path("/a.json"),
        graph_json=graph_json,
        tensor_infos=[tensor_info],
        tensors={1: np.array([1.0, 2.0], dtype=np.float32)},
        phase="output",
        provider="../../etc",
    )
    assert tmp_path.resolve() in manifest_path.resolve().parents

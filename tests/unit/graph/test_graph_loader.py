# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for graph loading: malformed input fails as GraphLoadError."""

import json
from pathlib import Path

import pytest

from dnn_benchmarking.common.exceptions import GraphLoadError, UnsupportedGraphError
from dnn_benchmarking.graph import GraphLoader, output_uids


def _tensor(uid, data_type="float", **extra):
    return {"uid": uid, "name": f"t{uid}", "dims": [2], "data_type": data_type, **extra}


@pytest.mark.parametrize("payload", ["[]", "3", '"graph"', "null"])
def test_load_json_rejects_non_object_json(tmp_path: Path, payload: str) -> None:
    path = tmp_path / "g.json"
    path.write_text(payload)

    with pytest.raises(GraphLoadError, match="JSON object"):
        GraphLoader().load_json(path)


@pytest.mark.parametrize("graph", [{"nodes": []}, {}])
def test_validate_rejects_graph_without_nodes(graph) -> None:
    with pytest.raises(GraphLoadError, match="no operation nodes"):
        GraphLoader().validate(graph)


@pytest.mark.parametrize(
    "tensor",
    [
        {"uid": 1, "dims": [2]},  # data_type missing: no silent float default
        {"uid": 1, "data_type": "float"},  # dims missing
        {"uid": 1, "dims": None, "data_type": "float"},
        {"uid": "x", "dims": [2], "data_type": "float"},
        [1, 2],
    ],
)
def test_malformed_tensor_raises_graph_load_error(tensor) -> None:
    graph = {"nodes": [{"type": "X"}], "tensors": [tensor]}

    with pytest.raises(GraphLoadError):
        GraphLoader().extract_tensor_info(graph)


def test_unknown_dtype_is_unsupported_not_a_crash() -> None:
    graph = {"nodes": [{"type": "X"}], "tensors": [_tensor(1, "fp4_e2m1")]}

    with pytest.raises(UnsupportedGraphError, match="fp4_e2m1"):
        GraphLoader().extract_tensor_info(graph)


def test_unset_physical_dtype_resolves_from_io_data_type() -> None:
    # hipDNN's fill_from_context fills an unset non-virtual tensor from
    # io_data_type; the buffer must be sized for that type.
    graph = {"nodes": [{"type": "X"}], "io_data_type": "half"}
    graph["tensors"] = [_tensor(1, "unset")]

    (info,) = GraphLoader().extract_tensor_info(graph)
    assert (info.data_type, info.element_size) == ("half", 2)

    del graph["io_data_type"]
    with pytest.raises(UnsupportedGraphError, match="unset"):
        GraphLoader().extract_tensor_info(graph)


def test_output_uids_reads_int_and_list_outputs() -> None:
    graph = {
        "nodes": [
            {"outputs": {"y_tensor_uid": 3, "stats": [4, 5], "absent": None}},
            {"outputs": {"flag": True}},
            {},
        ]
    }

    assert output_uids(graph) == {3, 4, 5}


def test_extract_marks_list_outputs_and_drops_virtual(tmp_path: Path) -> None:
    # hipDNN serialises an unset virtual tensor's type as "unset"; it gets no
    # buffer, so it must be dropped before its dtype is resolved.
    graph = {
        "nodes": [{"inputs": {"x": 1}, "outputs": {"y": [2]}}],
        "tensors": [
            _tensor(1),
            _tensor(2, "BFLOAT16"),
            _tensor(9, "unset", virtual=True),
        ],
    }
    path = tmp_path / "g.json"
    path.write_text(json.dumps(graph))
    loader = GraphLoader()

    infos = {t.uid: t for t in loader.extract_tensor_info(loader.load_json(path))}

    assert set(infos) == {1, 2}
    assert not infos[1].is_output
    assert infos[2].is_output
    assert infos[2].element_size == 2

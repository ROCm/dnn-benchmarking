# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for tools/sdpa_fwd_to_bwd.py, which builds the SDPA backward corpus.

The tool decides what the corpus holds: which forward graphs have a backward
form, which flags the backward node needs, and which graphs are duplicates.
"""

import importlib.util
from pathlib import Path

import pytest

from dnn_benchmarking.graph import GraphLoader

_TOOL = Path(__file__).resolve().parents[2] / "tools" / "sdpa_fwd_to_bwd.py"


@pytest.fixture(scope="module")
def tool():
    spec = importlib.util.spec_from_file_location("sdpa_fwd_to_bwd", _TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tensor(uid, name, dims, data_type="half"):
    strides, step = [], 1
    for dim in reversed(dims):
        strides.insert(0, step)
        step *= dim
    return {
        "uid": uid,
        "name": name,
        "dims": dims,
        "strides": strides,
        "data_type": data_type,
        "virtual": False,
    }


def _forward(seq_lens=False, **inputs):
    """A GQA forward graph: Q [2, 4, 8, 16], K/V [2, 2, 12, 16]."""
    tensors = [
        _tensor(1, "q", [2, 4, 8, 16]),
        _tensor(2, "k", [2, 2, 12, 16]),
        _tensor(3, "v", [2, 2, 12, 16]),
        _tensor(4, "o", [2, 4, 8, 16]),
    ]
    node_inputs = {"q_tensor_uid": 1, "k_tensor_uid": 2, "v_tensor_uid": 3}
    if seq_lens:
        tensors += [
            _tensor(5, "seq_q", [2, 1, 1, 1], "int32"),
            _tensor(6, "seq_kv", [2, 1, 1, 1], "int32"),
        ]
        node_inputs.update(seq_len_q_tensor_uid=5, seq_len_kv_tensor_uid=6)
    node_inputs.update(inputs)
    return {
        "name": "fwd",
        "compute_data_type": "float",
        "io_data_type": "half",
        "intermediate_data_type": "half",
        "tensors": tensors,
        "nodes": [
            {
                "name": "sdpa",
                "type": "SdpaAttributes",
                "inputs": node_inputs,
                "outputs": {"o_tensor_uid": 4},
                "attributes": {
                    "attn_scale_value": 0.25,
                    "causal_mask": True,
                    "diagonal_alignment": "BOTTOM_RIGHT",
                },
            }
        ],
    }


def test_converted_graph_loads_with_gradients_shaped_like_their_inputs(tool) -> None:
    bwd = tool.convert(_forward())
    loader = GraphLoader()
    loader.validate(bwd)
    infos = {info.uid: info for info in loader.extract_tensor_info(bwd)}
    node = bwd["nodes"][0]
    inputs, outputs = node["inputs"], node["outputs"]

    for grad, operand in (("dq", "q"), ("dk", "k"), ("dv", "v")):
        grad_info = infos[outputs[f"{grad}_tensor_uid"]]
        assert grad_info.dims == infos[inputs[f"{operand}_tensor_uid"]].dims
        assert grad_info.is_output
    stats = infos[inputs["stats_tensor_uid"]]
    assert (stats.dims, stats.data_type) == ([2, 4, 8, 1], "float")
    assert infos[inputs["do_tensor_uid"]].dims == [2, 4, 8, 16]
    # The mask converts unchanged: the band rule is hipDNN's, not the tool's.
    assert node["parameters"]["causal_mask"] is True
    assert node["parameters"]["diagonal_alignment"] == "BOTTOM_RIGHT"


@pytest.mark.parametrize("seq_lens", [True, False])
def test_padding_mask_is_set_only_with_both_seq_len_tensors(tool, seq_lens) -> None:
    """hipDNN's SdpaBwdNode checks the seq_len tensors only with
    padding_mask, and cuDNN requires the flag with them."""
    bwd = tool.convert(_forward(seq_lens=seq_lens))

    assert bwd["nodes"][0]["parameters"]["padding_mask"] is seq_lens


@pytest.mark.parametrize(
    "inputs, reason",
    [
        ({"page_table_k_tensor_uid": 9, "page_table_v_tensor_uid": 10}, "paged"),
        ({"sink_token_tensor_uid": 9}, "sink"),
        ({"seed_tensor_uid": 9, "offset_tensor_uid": 10}, "dropout"),
    ],
)
def test_forward_only_features_are_skipped(tool, inputs, reason) -> None:
    with pytest.raises(tool.Skip) as skipped:
        tool.convert(_forward(**inputs))

    assert str(skipped.value) == reason


def test_captured_dropout_metadata_adds_philox_dropout_once(tool) -> None:
    bwd = tool.convert(_forward())
    captured = {**bwd, "metadata": {"dropout": 0.1}}

    adopted = tool.adopt(captured)
    node = adopted["nodes"][0]
    assert node["parameters"]["dropout_probability"] == 0.1
    uids = {t["uid"] for t in adopted["tensors"]}
    assert {node["inputs"]["seed_tensor_uid"], node["inputs"]["offset_tensor_uid"]} <= (
        uids
    )
    # A graph that already has dropout keeps its own probability and inputs.
    again = tool.adopt({**adopted, "metadata": {"dropout": 0.3}})
    assert again["nodes"][0] == node
    assert len(again["tensors"]) == len(adopted["tensors"])


def test_identity_ignores_name_and_metadata_only(tool) -> None:
    bwd = tool.convert(_forward())
    renamed = {**bwd, "name": "other", "metadata": {"model": ["x"]}}
    changed = tool.convert(_forward())
    changed["nodes"][0]["parameters"]["attn_scale_value"] = 0.5

    assert tool.identity(renamed) == tool.identity(bwd)
    assert tool.identity(changed) != tool.identity(bwd)

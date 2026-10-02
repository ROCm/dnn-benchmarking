# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for tools/check_deserialize.py's causal-diagonal check.

A decode graph written as causal + TOP_LEFT with Sq = 1 attends only to key 0
of its cache. The cudnn_attention_inference extraction wrote 228 decode graphs
that way. Sq = 1 top-left fails the check; other Sq != Skv top-left graphs are
reported as warnings.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_TOOL = Path(__file__).resolve().parents[2] / "tools" / "check_deserialize.py"


@pytest.fixture(scope="module")
def tool():
    spec = importlib.util.spec_from_file_location("check_deserialize_causal", _TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _graph(sq, skv, paged=False, **attrs):
    inputs = {"q_tensor_uid": 1, "k_tensor_uid": 2, "v_tensor_uid": 3}
    if paged:
        inputs["page_table_k_tensor_uid"] = 5
    return {
        "tensors": [
            {"uid": 1, "dims": [1, 8, sq, 64]},
            {"uid": 2, "dims": [1, 8, skv, 64]},
            {"uid": 3, "dims": [1, 8, skv, 64]},
        ],
        "nodes": [
            {
                "name": "sdpa",
                "type": "SdpaAttributes",
                "attributes": {"attn_scale_value": 0.125, **attrs},
                "inputs": inputs,
                "outputs": {"o_tensor_uid": 4},
            }
        ],
    }


_TOP_LEFT = {"causal_mask": True, "diagonal_alignment": "TOP_LEFT"}
_BOTTOM_RIGHT = {"causal_mask": True, "diagonal_alignment": "BOTTOM_RIGHT"}


@pytest.mark.parametrize(
    "graph",
    [
        _graph(1, 4096, **_TOP_LEFT),
        _graph(1, 64, paged=True, **_TOP_LEFT),
        _graph(1, 4096, left_bound=127, **_TOP_LEFT),
    ],
)
def test_top_left_decode_with_one_query_is_an_error(tool, graph) -> None:
    errors, warnings = tool.sdpa_top_left_mismatches(graph)

    assert len(errors) == 1 and "key 0" in errors[0]
    assert warnings == []


def test_top_left_with_unequal_lengths_is_a_warning(tool) -> None:
    errors, warnings = tool.sdpa_top_left_mismatches(_graph(171, 4096, **_TOP_LEFT))

    assert errors == []
    assert len(warnings) == 1 and "Sq=171, Skv=4096" in warnings[0]


@pytest.mark.parametrize(
    "graph",
    [
        _graph(1, 4096, **_BOTTOM_RIGHT),
        _graph(171, 4096, causal_mask_bottom_right=True),
        _graph(1, 4096),
        _graph(2048, 2048, **_TOP_LEFT),
    ],
)
def test_bottom_right_unmasked_or_square_graphs_pass(tool, graph) -> None:
    assert tool.sdpa_top_left_mismatches(graph) == ([], [])


def test_the_cli_fails_on_errors_and_only_reports_warnings(
    tool, tmp_path, monkeypatch, capsys
) -> None:
    (tmp_path / "decode.json").write_text(json.dumps(_graph(1, 4096, **_TOP_LEFT)))
    (tmp_path / "chunk.json").write_text(json.dumps(_graph(171, 4096, **_TOP_LEFT)))
    argv = ["check_deserialize.py", "--level", "json", "--src", "src"]

    monkeypatch.setattr(sys, "argv", [*argv, str(tmp_path / "chunk.json")])
    assert tool.main() == 0
    assert "WARN" in capsys.readouterr().out

    monkeypatch.setattr(sys, "argv", [*argv, str(tmp_path / "decode.json")])
    assert tool.main() == 1
    assert "key 0" in capsys.readouterr().out

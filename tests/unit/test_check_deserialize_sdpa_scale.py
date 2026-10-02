# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for tools/check_deserialize.py's SDPA scale requirement.

A graph whose SDPA node omits the scale runs at the backend's default, which
need not be the scale its source benchmarked, and engines that require an
explicit scale decline it. The corpus check must name such graphs.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_TOOL = Path(__file__).resolve().parents[2] / "tools" / "check_deserialize.py"


@pytest.fixture(scope="module")
def tool():
    spec = importlib.util.spec_from_file_location("check_deserialize_under_test", _TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sdpa_graph(scale=None, scale_uid=None, backward=False):
    key = "parameters" if backward else "attributes"
    return {
        "nodes": [
            {
                "name": "sdpa",
                "type": "SdpaBackwardAttributes" if backward else "SdpaAttributes",
                key: {"causal_mask": False, "attn_scale_value": scale},
                "inputs": {"q_tensor_uid": 1, "scale_tensor_uid": scale_uid},
                "outputs": {"o_tensor_uid": 4},
            }
        ]
    }


@pytest.mark.parametrize(
    "graph",
    [
        _sdpa_graph(scale=0.125),
        _sdpa_graph(scale_uid=9),
        _sdpa_graph(scale=0.07216878364870323, backward=True),
        {"nodes": [{"name": "mm", "type": "MatmulAttributes", "attributes": {}}]},
    ],
)
def test_a_stated_scale_or_a_non_sdpa_node_passes(tool, graph) -> None:
    assert tool.sdpa_nodes_without_scale(graph) == []


@pytest.mark.parametrize("backward", [False, True])
def test_an_sdpa_node_without_a_scale_is_named(tool, backward) -> None:
    assert tool.sdpa_nodes_without_scale(_sdpa_graph(backward=backward)) == ["sdpa"]


def test_the_cli_fails_and_names_the_unscaled_file(tool, tmp_path, monkeypatch, capsys):
    (tmp_path / "unscaled.json").write_text(json.dumps(_sdpa_graph()))
    monkeypatch.setattr(
        sys, "argv", ["check_deserialize.py", "--level", "json", str(tmp_path)]
    )

    assert tool.main() == 1
    out = capsys.readouterr().out
    assert "unscaled.json" in out
    assert "attn_scale_value" in out

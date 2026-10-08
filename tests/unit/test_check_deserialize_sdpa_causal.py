# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for tools/check_deserialize.py's causal-diagonal check.

A decode graph whose effective diagonal is top-left with Sq = 1 attends only to
key 0 of its cache. The cudnn_attention_inference extraction wrote 228 decode
graphs that way. "Effective" is what hipDNN runs: causal_mask overrides
diagonal_alignment and the written bounds, so causal_mask + BOTTOM_RIGHT is
still top-left. Sq = 1 fails the check; other Sq != Skv top-left graphs are
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


# What the JSON says versus what hipDNN runs. causal_mask pins the diagonal
# top-left whatever diagonal_alignment says, so _FIX_KIT (the form 180 of the
# 251 #70 fix-kit graphs use) is a top-left graph wearing a bottom-right label.
_TOP_LEFT = {"causal_mask": True, "diagonal_alignment": "TOP_LEFT"}
_FIX_KIT = {"causal_mask": True, "diagonal_alignment": "BOTTOM_RIGHT"}
_BOTTOM_RIGHT = {
    "causal_mask": False,
    "left_bound": -1,
    "right_bound": 0,
    "diagonal_alignment": "BOTTOM_RIGHT",
}


@pytest.mark.parametrize(
    "graph",
    [
        _graph(1, 4096, **_TOP_LEFT),
        _graph(1, 64, paged=True, **_TOP_LEFT),
        _graph(1, 4096, left_bound=127, **_TOP_LEFT),
        _graph(1, 4096, left_bound=-1, right_bound=0, diagonal_alignment="TOP_LEFT"),
        # The #70 fix kit's form: the alignment claims bottom-right, hipDNN
        # discards it and the query still sees only key 0.
        _graph(1, 4096, **_FIX_KIT),
        _graph(1, 64, paged=True, **_FIX_KIT),
        # The fix kit's windowed decode graph (gpt-oss SWA sq1_skv4096): the
        # bounds are discarded along with the alignment.
        _graph(1, 4096, left_bound=127, right_bound=0, **_FIX_KIT),
    ],
)
def test_top_left_decode_with_one_query_is_an_error(tool, graph) -> None:
    errors, warnings = tool.sdpa_top_left_mismatches(graph)

    assert len(errors) == 1 and "key 0" in errors[0]
    assert warnings == []


@pytest.mark.parametrize(
    "graph",
    [
        _graph(171, 4096, **_TOP_LEFT),
        _graph(171, 4096, **_FIX_KIT),
    ],
)
def test_top_left_with_unequal_lengths_is_a_warning(tool, graph) -> None:
    errors, warnings = tool.sdpa_top_left_mismatches(graph)

    assert errors == []
    assert len(warnings) == 1 and "Sq=171, Skv=4096" in warnings[0]


@pytest.mark.parametrize(
    "graph",
    [
        _graph(1, 4096, **_BOTTOM_RIGHT),
        _graph(171, 4096, causal_mask_bottom_right=True),
        _graph(
            1,
            4096,
            causal_mask=False,
            left_bound=127,
            right_bound=0,
            diagonal_alignment="BOTTOM_RIGHT",
        ),
        _graph(1, 4096),
        _graph(1, 4096, left_bound=-1, right_bound=-1, diagonal_alignment="TOP_LEFT"),
        _graph(2048, 2048, **_TOP_LEFT),
        _graph(2048, 2048, **_FIX_KIT),
    ],
)
def test_bottom_right_unmasked_or_square_graphs_pass(tool, graph) -> None:
    assert tool.sdpa_top_left_mismatches(graph) == ([], [])


@pytest.mark.parametrize(
    "graph",
    [
        # hipDNN masks the right side only when right_bound >= 0, so a lone
        # left_bound is a window that stays open on the right. With Sq = 1 the
        # query still sees the whole cache.
        _graph(1, 4096, left_bound=127, diagonal_alignment="TOP_LEFT"),
        _graph(1, 4096, left_bound=127, right_bound=-1),
        _graph(1, 64, paged=True, left_bound=127),
        _graph(171, 4096, left_bound=127),
    ],
)
def test_a_right_open_window_is_not_causal(tool, graph) -> None:
    assert tool.sdpa_top_left_mismatches(graph) == ([], [])


def test_both_causal_flags_are_a_conflict(tool) -> None:
    graph = _graph(1, 4096, causal_mask=True, causal_mask_bottom_right=True)

    assert tool.sdpa_mask_flag_conflicts(graph) == ["sdpa"]
    assert tool.sdpa_mask_flag_conflicts(_graph(1, 4096, **_BOTTOM_RIGHT)) == []


def test_the_cli_fails_on_errors_and_only_reports_warnings(
    tool, tmp_path, monkeypatch, capsys
) -> None:
    (tmp_path / "decode.json").write_text(json.dumps(_graph(1, 4096, **_FIX_KIT)))
    (tmp_path / "chunk.json").write_text(json.dumps(_graph(171, 4096, **_TOP_LEFT)))
    (tmp_path / "fixed.json").write_text(json.dumps(_graph(1, 4096, **_BOTTOM_RIGHT)))
    (tmp_path / "both.json").write_text(
        json.dumps(_graph(1, 4096, causal_mask=True, causal_mask_bottom_right=True))
    )
    argv = ["check_deserialize.py", "--level", "json", "--src", "src"]

    monkeypatch.setattr(sys, "argv", [*argv, str(tmp_path / "chunk.json")])
    assert tool.main() == 0
    assert "WARN" in capsys.readouterr().out

    monkeypatch.setattr(sys, "argv", [*argv, str(tmp_path / "decode.json")])
    assert tool.main() == 1
    assert "key 0" in capsys.readouterr().out

    monkeypatch.setattr(sys, "argv", [*argv, str(tmp_path / "both.json")])
    assert tool.main() == 1
    assert "causal_mask_bottom_right" in capsys.readouterr().out

    # Negative control: the same decode graph written the way hipDNN reads as
    # bottom-right passes with no warning.
    monkeypatch.setattr(sys, "argv", [*argv, str(tmp_path / "fixed.json")])
    assert tool.main() == 0
    out = capsys.readouterr().out
    assert "WARN" not in out and "FAIL" not in out

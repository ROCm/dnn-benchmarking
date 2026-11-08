# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""AITER runtime: SDPA problem extraction and the backend rows of each problem."""

import importlib.util
import shutil
import subprocess
from pathlib import Path

import pytest

from dnn_benchmarking.common.exceptions import UnsupportedGraphError
from dnn_benchmarking.execution.aiter_sdpa import (
    VERIFIED_AITER_COMMIT,
    AiterDevice,
    _native_splits,
    aiter_problem,
    backend_rows,
    is_causal,
)
from dnn_benchmarking.graph.loader import GraphLoader
from tests.rocke_graphs import sdpa_graph

GFX942 = AiterDevice(arch="gfx942", num_cus=304, gluon=False)
GFX950 = AiterDevice(arch="gfx950", num_cus=256, gluon=True)
GFX90A = AiterDevice(arch="gfx90a", num_cus=104, gluon=False)


def _problem(graph):
    return aiter_problem(graph, GraphLoader().extract_tensor_info(graph))


def _rows(device, **graph):
    return [b.name for b in backend_rows(_problem(sdpa_graph(**graph)), device)]


def test_problem_reads_both_head_sizes_and_the_kv_length() -> None:
    p = _problem(
        sdpa_graph(
            b=2, hq=16, hkv=2, s=128, s_kv=512, d=192, d_v=128, causal_mask=False
        )
    )
    assert (p.batch, p.num_query_heads, p.num_kv_heads) == (2, 16, 2)
    assert (p.seqlen_q, p.seqlen_kv, p.head_size_qk, p.head_size_v) == (
        128,
        512,
        192,
        128,
    )


def test_top_left_causal_needs_equal_sequence_lengths() -> None:
    """AITER aligns causal masks bottom right, so only a square top-left fits."""
    with pytest.raises(UnsupportedGraphError, match="bottom right"):
        _problem(sdpa_graph(s=128, s_kv=512, causal_mask=True))
    assert is_causal(_problem(sdpa_graph(s=512, causal_mask=True)))


def test_bottom_right_causal_runs_on_any_shape() -> None:
    p = _problem(
        sdpa_graph(s=128, s_kv=512, causal_mask=False, causal_mask_bottom_right=True)
    )
    assert is_causal(p) and not p.causal


def test_unsupported_features_name_the_aiter_runtime() -> None:
    with pytest.raises(UnsupportedGraphError, match="AITER runtime .*dropout"):
        _problem(sdpa_graph(dropout_probability=0.1))


@pytest.mark.parametrize(
    "device, graph, rows",
    [
        pytest.param(
            GFX950,
            dict(dtype="bfloat16", d=128),
            ["auto", "ck", "asm_v3", "opus", "triton", "gluon"],
            id="gfx950-bf16-d128",
        ),
        pytest.param(
            GFX950,
            dict(dtype="half", d=128),
            ["auto", "ck", "triton", "gluon"],
            id="gfx950-fp16-has-no-bf16-only-kernels",
        ),
        pytest.param(
            GFX942,
            dict(dtype="bfloat16", d=192, d_v=128),
            ["auto", "ck", "asm_v3", "triton"],
            id="gfx942-bf16-d192-v128",
        ),
        pytest.param(
            GFX942,
            dict(
                b=1,
                hq=8,
                hkv=8,
                s=128,
                s_kv=4096,
                dtype="bfloat16",
                d=64,
                causal_mask=False,
            ),
            ["auto", "ck", "native_splitkv", "triton"],
            id="gfx942-split-kv",
        ),
        pytest.param(
            GFX942,
            dict(b=8, hq=32, hkv=8, s=2048, dtype="bfloat16", d=64),
            ["auto", "ck", "triton"],
            id="gfx942-no-split",
        ),
        pytest.param(
            GFX90A,
            dict(dtype="bfloat16", d=128),
            ["auto", "ck"],
            id="gfx90a-has-no-triton-config",
        ),
        pytest.param(GFX950, dict(d=60), ["auto", "triton", "gluon"], id="d-not-x8"),
        pytest.param(GFX950, dict(d=512), ["auto", "triton", "gluon"], id="d512"),
        pytest.param(
            GFX950, dict(d=176, d_v=128), ["auto", "ck"], id="pe-not-power-of-two"
        ),
    ],
)
def test_backend_rows(device, graph, rows) -> None:
    assert _rows(device, **graph) == rows


def test_asm_v3_drops_a_kv_extent_of_4_gib() -> None:
    """192/128 KV rows of 2**32 bytes wrap the kernel's 32-bit offsets."""
    # 16 KV heads * 192 * 2 bytes = 6 KiB per token; 2**32 / 6 KiB = 699051.
    fits = dict(hq=16, hkv=16, s=1, s_kv=699050, causal_mask=False)
    fits.update(dtype="bfloat16", d=192, d_v=128)
    assert "asm_v3" in _rows(GFX950, **fits)
    assert "asm_v3" not in _rows(GFX950, **dict(fits, s_kv=699051))


def test_other_arches_get_no_rows() -> None:
    with pytest.raises(UnsupportedGraphError, match="gfx1100"):
        backend_rows(_problem(sdpa_graph()), AiterDevice("gfx1100", 48, False))


def test_split_kv_count_matches_aiter() -> None:
    """The copy of AITER's split-KV heuristic gives AITER's split counts."""
    spec = importlib.util.find_spec("aiter")
    if spec is None or spec.origin is None:  # an empty aiter/ is a namespace
        pytest.skip("aiter is not installed")
    from dnn_benchmarking.execution.aiter_executor import prepare_aiter_environment

    prepare_aiter_environment()  # import aiter compiles its core module
    mha = pytest.importorskip("aiter.ops.mha")
    for b in (1, 2):
        for hq in (8, 64):
            for sq in (1, 128, 2048):
                for skv in (64, 4096, 65536):
                    p = _problem(
                        sdpa_graph(b=b, hq=hq, hkv=8, s=sq, s_kv=skv, causal_mask=False)
                    )
                    assert _native_splits(p, 304) == mha._native_splitkv_heuristic(
                        b, hq, sq, skv, 304
                    ), (b, hq, sq, skv)


def test_backend_checks_were_verified_against_the_pinned_aiter() -> None:
    """A submodule bump fails here until someone checks the copied AITER code."""
    repo = Path(__file__).resolve().parents[3]
    if shutil.which("git") is None or not (repo / ".git").exists():
        pytest.skip("no git checkout of this repository")
    entry = subprocess.run(
        ["git", "-C", str(repo), "ls-files", "--stage", "aiter"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    assert entry[:1] == ["160000"], "aiter is not a submodule"
    assert entry[1] == VERIFIED_AITER_COMMIT, (
        f"the aiter submodule pins {entry[1]}; check the backend checks in "
        "execution/aiter_sdpa.py and the arguments in aiter_executor._launch_for "
        "against it, then update VERIFIED_AITER_COMMIT"
    )

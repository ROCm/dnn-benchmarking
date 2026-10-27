# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""rocKE runtime: SDPA problem extraction and shipped-kernel catalog matching."""

import copy
import json

import pytest

from dnn_benchmarking.common.exceptions import UnsupportedGraphError
from dnn_benchmarking.execution.rocke_sdpa import (
    DENSE_ENGINE,
    catalog_kernels,
    load_catalog,
    sdpa_problem,
)
from dnn_benchmarking.graph.loader import GraphLoader
from tests.rocke_graphs import bshd_tensor, sdpa_graph


def _problem(graph):
    return sdpa_problem(graph, GraphLoader().extract_tensor_info(graph))


def test_problem_reads_shape_and_hipdnn_default_scale() -> None:
    p = _problem(sdpa_graph(b=4, hq=64, hkv=8, s=1024, d=64, dtype="bfloat16"))
    assert (p.dtype, p.batch, p.num_query_heads, p.num_kv_heads) == ("bf16", 4, 64, 8)
    assert (p.seqlen_q, p.seqlen_kv, p.head_size, p.causal) == (1024, 1024, 64, True)
    # An absent attn_scale_value is 1.0 in hipDNN, not 1/sqrt(d).
    assert p.scale == 1.0
    assert _problem(sdpa_graph(attn_scale_value=0.125)).scale == 0.125


def test_size_one_dimension_strides_do_not_matter() -> None:
    graph = sdpa_graph(hq=1, hkv=1)
    for tensor in graph["tensors"]:
        tensor["strides"][1] = 999
    assert _problem(graph).num_query_heads == 1


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda g: g["tensors"][0].update(strides=[1, 2, 3, 4]), id="bhsd"),
        pytest.param(
            lambda g: g["nodes"][0]["outputs"].update(stats_tensor_uid=4), id="stats"
        ),
        pytest.param(
            lambda g: g["nodes"][0]["attributes"].update(dropout_probability=0.1),
            id="dropout",
        ),
        pytest.param(
            lambda g: g["nodes"][0].update(type="SdpaBackwardAttributes"), id="bwd"
        ),
        pytest.param(lambda g: g["tensors"][3].update(data_type="float"), id="dtype"),
        pytest.param(
            lambda g: g["nodes"][0]["attributes"].update(
                causal_mask=False, left_bound=127, right_bound=0
            ),
            id="window",
        ),
    ],
)
def test_unsupported_graphs_are_rejected(mutate) -> None:
    graph = sdpa_graph()
    mutate(graph)
    with pytest.raises(UnsupportedGraphError):
        _problem(graph)


def test_causal_needs_equal_sequence_lengths() -> None:
    graph = sdpa_graph()
    for index in (0, 3):  # query and output: shorter seqlen_q
        tensor = graph["tensors"][index]
        tensor.update(
            bshd_tensor(tensor["uid"], tensor["name"], 1, 32, 1024, 128, "half")
        )
    with pytest.raises(UnsupportedGraphError):
        _problem(graph)
    graph["nodes"][0]["attributes"]["causal_mask"] = False
    assert _problem(graph).seqlen_q == 1024


_ARGS = ("q_ptr", "k_ptr", "v_ptr", "o_ptr", "scale", "batch", "seqlen_q", "seqlen_kv")


def _metadata(dtype="FP16", d=128, hq=32, hkv=8, causal=1, bm=256, bn=64):
    """Variant metadata as the packer installs it: upper-case dtype, 0/1 flags."""
    return {
        "dtype": dtype,
        "head_size": d,
        "num_query_heads": hq,
        "num_kv_heads": hkv,
        "causal": causal,
        "ragged": 0,
        "sliding_window": 0,
        "batch": 1,
        "seqlen_q": 512,
        "seqlen_kv": 512,
        "block_m": bm,
        "block_n": bn,
    }


def _variant(name, metadata, kind="kpack"):
    return {
        "id": f"id-{name}",
        "name": name,
        "metadata": metadata,
        "kernel_source": {
            "kind": kind,
            "library": "../../kpack/hip_kernel_provider_gfx950.kpack",
            "toc_key": f"toc_{name}",
            "symbol": f"sym_{name}",
            "sha256": "cd" * 32,
            "signature": [{"name": n} for n in _ARGS],
        },
    }


def _install(plugins, engine_name, variants, arch="gfx950", name="dense"):
    """One engine and its variant set, at the authored ``artifacts/`` subpath."""
    base = plugins / "arch_content" / "hip-kernel-provider" / arch / "rocKE" / name
    (base / "artifacts").mkdir(parents=True)
    engine = {"id": f"engine-{name}", "name": engine_name}
    (base / f"{name}.ued.json").write_text(json.dumps(engine))
    kdp = {"engine": engine["id"], "kernelDescriptors": variants}
    (base / "artifacts" / f"{name}.kdp.json").write_text(json.dumps(kdp))
    return base / "artifacts"


def _catalog(tmp_path, metadata):
    """A dense catalog whose variants have ``metadata``, keyed by name."""
    plugins = tmp_path / "plugins"
    _install(plugins, DENSE_ENGINE, [_variant(n, m) for n, m in metadata.items()])
    return load_catalog([plugins], "gfx950")


def test_catalog_reads_the_dense_engine_variants(tmp_path) -> None:
    plugins = tmp_path / "plugins"
    where = _install(
        plugins,
        DENSE_ENGINE,
        [_variant("a", _metadata(dtype="BF16", causal=0)), _variant("h", {}, "hip")],
    )
    _install(plugins, "hipkernel:Other", [_variant("x", _metadata())], name="other")

    (kernel,) = load_catalog([plugins], "gfx950")
    assert kernel.name == "rocke:a"
    assert kernel.spec == dict(_metadata(), dtype="bf16", causal=False, ragged=False)
    # The library path is relative to the descriptor file, not the plugin root.
    assert (
        kernel.library
        == (where / "../../kpack/hip_kernel_provider_gfx950.kpack").resolve()
    )
    assert (kernel.toc_key, kernel.symbol, kernel.sha256) == (
        "toc_a",
        "sym_a",
        "cd" * 32,
    )
    assert kernel.arg_names == _ARGS
    assert load_catalog([plugins], "gfx942") == ()


def test_catalog_matches_problem_and_tile_divisibility(tmp_path) -> None:
    pytest.importorskip("kernels.gfx950.attention_dense")
    catalog = _catalog(
        tmp_path,
        {
            "bm256_bn64": _metadata(bm=256, bn=64),
            "bm128_bn32": _metadata(bm=128, bn=32),
            "full": _metadata(causal=0),
            "d64": _metadata(d=64),
            "bf16": _metadata(dtype="BF16"),
        },
    )

    found = catalog_kernels(catalog, _problem(sdpa_graph(b=2, s=2048)))
    assert [k.name for k in found] == ["rocke:bm256_bn64", "rocke:bm128_bn32"]
    # The graph's shape replaces the descriptor's placeholder shape.
    assert {(k.spec["batch"], k.spec["seqlen_q"]) for k in found} == {(2, 2048)}

    # 1920 is a multiple of 128 but not of 256.
    found = catalog_kernels(catalog, _problem(sdpa_graph(s=1920)))
    assert [k.name for k in found] == ["rocke:bm128_bn32"]

    graph = copy.deepcopy(sdpa_graph())
    graph["nodes"][0]["attributes"]["causal_mask"] = False
    assert [k.name for k in catalog_kernels(catalog, _problem(graph))] == ["rocke:full"]


def test_unknown_spec_field_is_not_reported_as_no_match(tmp_path) -> None:
    """A descriptor from another rocKE version must fail loudly, not match nothing."""
    pytest.importorskip("kernels.gfx950.attention_dense")
    catalog = _catalog(
        tmp_path, {"newer": dict(_metadata(), knob_from_another_rocke=1)}
    )
    with pytest.raises(TypeError):
        catalog_kernels(catalog, _problem(sdpa_graph()))

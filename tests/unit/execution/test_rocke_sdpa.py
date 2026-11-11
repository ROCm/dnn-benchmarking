# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""rocKE runtime: SDPA problem extraction and shipped-kernel catalog matching."""

import copy
import json

import pytest

from dnn_benchmarking.common.exceptions import UnsupportedGraphError
from dnn_benchmarking.execution.rocke_sdpa import (
    DENSE_ENGINE,
    auto_kernel,
    catalog_kernels,
    hipdnn_grid,
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


def _set_seqlen_q(graph, seqlen_q):
    for index in (0, 3):  # query and output
        tensor = graph["tensors"][index]
        tensor.update(
            bshd_tensor(tensor["uid"], tensor["name"], 1, 32, seqlen_q, 128, "half")
        )


def test_causal_needs_equal_sequence_lengths() -> None:
    graph = sdpa_graph()
    _set_seqlen_q(graph, 1024)
    with pytest.raises(UnsupportedGraphError):
        _problem(graph)
    graph["nodes"][0]["attributes"]["causal_mask"] = False
    assert _problem(graph).seqlen_q == 1024


_BOTTOM_RIGHT_BAND = dict(
    causal_mask=False, diagonal_alignment="BOTTOM_RIGHT", right_bound=0
)


@pytest.mark.parametrize(
    "attributes, causal",
    [
        pytest.param(
            dict(causal_mask=False, causal_mask_bottom_right=True),
            True,
            id="deprecated-flag",
        ),
        pytest.param(_BOTTOM_RIGHT_BAND, True, id="bounds"),
        # hipDNN reads the deprecated flag first, so this mask is top-left.
        pytest.param(dict(diagonal_alignment="BOTTOM_RIGHT"), True, id="flag-wins"),
        pytest.param(
            dict(causal_mask=False, diagonal_alignment="BOTTOM_RIGHT"),
            False,
            id="no-bounds",
        ),
    ],
)
def test_bottom_right_alignment_is_causal_on_square_problems(
    attributes, causal
) -> None:
    """On a square problem the bottom-right and top-left masks are equal."""
    assert _problem(sdpa_graph(**attributes)).causal is causal


@pytest.mark.parametrize(
    "attributes",
    [dict(causal_mask=False, causal_mask_bottom_right=True), _BOTTOM_RIGHT_BAND],
)
def test_bottom_right_causal_needs_equal_sequence_lengths(attributes) -> None:
    graph = sdpa_graph(**attributes)
    _set_seqlen_q(graph, 1024)
    with pytest.raises(UnsupportedGraphError, match="seqlen_q == seqlen_kv"):
        _problem(graph)


def test_query_heads_must_be_a_multiple_of_kv_heads() -> None:
    with pytest.raises(UnsupportedGraphError, match="Hq divisible by Hkv"):
        _problem(sdpa_graph(hq=12, hkv=8))


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


def test_a_variant_shipped_under_two_plugin_paths_gives_one_row(tmp_path) -> None:
    first, second = tmp_path / "first", tmp_path / "second"
    _install(first, DENSE_ENGINE, [_variant("a", _metadata())])
    second_variants = [_variant("a", _metadata()), _variant("b", _metadata())]
    _install(second, DENSE_ENGINE, second_variants)

    catalog = load_catalog([first, second], "gfx950")

    assert [k.name for k in catalog] == ["rocke:a", "rocke:b"]
    assert first in catalog[0].library.parents


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
            "hq64": _metadata(hq=64),
            "kv4": _metadata(hkv=4),
            "window": dict(_metadata(), sliding_window=128),
            "ragged": dict(_metadata(), ragged=1),
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

    hq64 = _problem(sdpa_graph(hq=64, hkv=8))
    assert [k.name for k in catalog_kernels(catalog, hq64)] == ["rocke:hq64"]
    kv4 = _problem(sdpa_graph(hq=32, hkv=4))
    assert [k.name for k in catalog_kernels(catalog, kv4)] == ["rocke:kv4"]


@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("b, hq, hkv", [(1, 32, 8), (4, 16, 16), (2, 40, 8)])
def test_matched_variants_launch_on_hipdnns_grid(tmp_path, causal, b, hq, hkv) -> None:
    """hipDNN launches every variant query-block-major; rocKE's auto block
    order would put the batch on x for causal problems."""
    dense = pytest.importorskip("kernels.gfx950.attention_dense")
    catalog = _catalog(
        tmp_path,
        {
            "bm256": _metadata(hq=hq, hkv=hkv, causal=int(causal), bm=256),
            "bm128": _metadata(hq=hq, hkv=hkv, causal=int(causal), bm=128, bn=32),
        },
    )
    problem = _problem(sdpa_graph(b=b, hq=hq, hkv=hkv, s=2048, causal_mask=causal))

    found = catalog_kernels(catalog, problem)

    assert len(found) == 2
    for kernel in found:
        spec = dense.Gfx950AttentionDenseSpec(**kernel.spec)
        assert tuple(dense.attention_dense_grid(spec)) == hipdnn_grid(
            problem, kernel.spec["block_m"]
        )


def test_auto_kernel_needs_a_tiled_arch_and_a_causal_graph() -> None:
    """Both checks run before rocKE's dispatcher, so they need no rocKE."""
    causal = _problem(sdpa_graph())
    with pytest.raises(UnsupportedGraphError, match="no tiled attention kernel"):
        auto_kernel(causal, "gfx90a", 104)
    full = _problem(sdpa_graph(causal_mask=False))
    with pytest.raises(UnsupportedGraphError, match="causal-only"):
        auto_kernel(full, "gfx950", 256)


def test_unknown_spec_field_is_not_reported_as_no_match(tmp_path) -> None:
    """A descriptor from another rocKE version must fail loudly, not match nothing."""
    pytest.importorskip("kernels.gfx950.attention_dense")
    catalog = _catalog(
        tmp_path, {"newer": dict(_metadata(), knob_from_another_rocke=1)}
    )
    with pytest.raises(TypeError):
        catalog_kernels(catalog, _problem(sdpa_graph()))

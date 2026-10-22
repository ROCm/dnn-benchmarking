# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""rocKE runtime: SDPA problem extraction and shipped-kernel catalog matching."""

import copy
import json

import pytest

from dnn_benchmarking.common.exceptions import UnsupportedGraphError
from dnn_benchmarking.execution.rocke_sdpa import (
    DENSE_BUILDER,
    DENSE_SOURCE,
    catalog_kernels,
    descriptor_files,
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


def _spec(dtype="fp16", d=128, hq=32, hkv=8, causal=True, bm=256, bn=64):
    return {
        "dtype": dtype,
        "head_size": d,
        "num_query_heads": hq,
        "num_kv_heads": hkv,
        "causal": causal,
        "batch": 1,
        "seqlen_q": 512,
        "seqlen_kv": 512,
        "block_m": bm,
        "block_n": bn,
    }


def _packed(uid, name, sha="ab" * 32):
    """An installed variant: packed into a kpack, no rocKE spec inline."""
    return {
        "id": uid,
        "name": name,
        "kernel_source": {"kind": "kpack", "sha256": sha},
    }


def _authored(uid, spec, builder=DENSE_BUILDER):
    """A variant as authored in the rocm-libraries checkout."""
    return {
        "id": uid,
        "kernel_source": {
            "kind": "rocke",
            "source": DENSE_SOURCE,
            "builder": builder,
            "spec": spec,
        },
    }


def _write(path, bundle_id, ukds):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"id": bundle_id, "kernelDescriptors": ukds}))
    return path


def _install(root, arch, bundle_id, ukds, name="dense"):
    # The installed layout keeps the authored subpath, which puts the DVC
    # bundle under <descriptor>/artifacts/.
    base = root / "arch_content" / "hip-kernel-provider" / arch / "rocKE"
    _write(base / name / "artifacts" / f"{name}.kdp.json", bundle_id, ukds)


def _catalog(tmp_path, specs):
    """Install and author one dense bundle whose variants have ``specs``."""
    names = list(specs)
    _install(tmp_path / "plugins", "gfx950", "kdp", [_packed(n, n) for n in names])
    authored = _write(
        tmp_path / "src" / "dense.kdp.json",
        "kdp",
        [_authored(n, specs[n]) for n in names],
    )
    installed = descriptor_files([tmp_path / "plugins"], "gfx950")
    return load_catalog(installed, [authored])


def test_catalog_joins_authored_spec_with_shipped_hash(tmp_path) -> None:
    spec = _spec()
    plugins = tmp_path / "plugins"
    _install(
        plugins,
        "gfx950",
        "kdp",
        [
            _packed("a", "dense_a", sha="cd" * 32),
            _packed("other", "other_builder"),
            _packed("gone", "not_authored"),
            {"id": "h", "kernel_source": {"kind": "hip", "source": "x.cpp"}},
        ],
    )
    _install(plugins, "gfx950", "unpulled", [_packed("u", "u")], name="new_bundle")
    authored = _write(
        tmp_path / "src" / "dense.kdp.json",
        "kdp",
        [_authored("a", spec), _authored("other", spec, builder="build_other")],
    )

    assert descriptor_files([plugins], "gfx942") == []
    catalog = load_catalog(descriptor_files([plugins], "gfx950"), [authored])
    assert catalog.kernels == (("dense_a", spec, "cd" * 32),)
    # Another builder's variant is not a dense kernel and not a mismatch.
    assert catalog.missing_spec == 1
    assert catalog.unmatched == ("new_bundle.kdp.json",)


def test_catalog_matches_problem_and_tile_divisibility(tmp_path) -> None:
    pytest.importorskip("kernels.gfx950.attention_dense")
    catalog = _catalog(
        tmp_path,
        {
            "bm256_bn64": _spec(bm=256, bn=64),
            "bm128_bn32": _spec(bm=128, bn=32),
            "full": _spec(causal=False),
            "d64": _spec(d=64),
            "bf16": _spec(dtype="bf16"),
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
    catalog = _catalog(tmp_path, {"newer": dict(_spec(), knob_from_another_rocke=1)})
    with pytest.raises(TypeError):
        catalog_kernels(catalog, _problem(sdpa_graph()))

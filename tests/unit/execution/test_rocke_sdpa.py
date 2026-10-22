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


def _bshd(uid, name, b, h, s, d, dtype):
    return {
        "uid": uid,
        "name": name,
        "dims": [b, h, s, d],
        "strides": [s * h * d, d, h * d, 1],
        "data_type": dtype,
        "virtual": False,
    }


def _graph(b=1, hq=32, hkv=8, s=2048, d=128, dtype="half", **attributes):
    """An SDPA forward graph laid out like the rocKE perf-oracle workloads."""
    attrs = {"causal_mask": True, "attn_scale_value": None}
    attrs.update(attributes)
    return {
        "name": "sdpa",
        "io_data_type": dtype,
        "compute_data_type": "float",
        "tensors": [
            _bshd(1, "query", b, hq, s, d, dtype),
            _bshd(2, "key", b, hkv, s, d, dtype),
            _bshd(3, "value", b, hkv, s, d, dtype),
            _bshd(4, "output", b, hq, s, d, dtype),
        ],
        "nodes": [
            {
                "name": "sdpa_node",
                "type": "SdpaAttributes",
                "inputs": {"q_tensor_uid": 1, "k_tensor_uid": 2, "v_tensor_uid": 3},
                "outputs": {"o_tensor_uid": 4, "stats_tensor_uid": None},
                "attributes": attrs,
            }
        ],
    }


def _problem(graph):
    return sdpa_problem(graph, GraphLoader().extract_tensor_info(graph))


def test_problem_reads_shape_and_hipdnn_default_scale() -> None:
    p = _problem(_graph(b=4, hq=64, hkv=8, s=1024, d=64, dtype="bfloat16"))
    assert (p.dtype, p.batch, p.num_query_heads, p.num_kv_heads) == ("bf16", 4, 64, 8)
    assert (p.seqlen_q, p.seqlen_kv, p.head_size, p.causal) == (1024, 1024, 64, True)
    # An absent attn_scale_value is 1.0 in hipDNN, not 1/sqrt(d).
    assert p.scale == 1.0
    assert _problem(_graph(attn_scale_value=0.125)).scale == 0.125


def test_size_one_dimension_strides_do_not_matter() -> None:
    graph = _graph(hq=1, hkv=1)
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
    graph = _graph()
    mutate(graph)
    with pytest.raises(UnsupportedGraphError):
        _problem(graph)


def test_causal_needs_equal_sequence_lengths() -> None:
    graph = _graph()
    for index in (0, 3):  # query and output: shorter seqlen_q
        tensor = graph["tensors"][index]
        tensor.update(_bshd(tensor["uid"], tensor["name"], 1, 32, 1024, 128, "half"))
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


def _packed(name, spec, sha="ab" * 32, provenance=True):
    ukd = {"name": name, "kernel_source": {"kind": "kpack", "sha256": sha}}
    if provenance:
        ukd["provenance"] = {
            "origin_kind": "rocke",
            "source": DENSE_SOURCE,
            "builder": DENSE_BUILDER,
            "spec": spec,
        }
    return ukd


def _write_descriptors(root, arch, ukds):
    path = root / "arch_content" / "hip-kernel-provider" / arch / "rocKE" / "dense"
    path.mkdir(parents=True)
    (path / "dense.kdp.json").write_text(json.dumps({"kernelDescriptors": ukds}))


def test_catalog_reads_spec_and_shipped_hash(tmp_path) -> None:
    spec = _spec()
    _write_descriptors(
        tmp_path,
        "gfx950",
        [
            _packed("packed", spec, sha="cd" * 32),
            _packed("no_provenance", spec, provenance=False),
            {
                "name": "authored",
                "kernel_source": {
                    "kind": "rocke",
                    "source": DENSE_SOURCE,
                    "builder": DENSE_BUILDER,
                    "spec": spec,
                },
            },
            {"name": "hip", "kernel_source": {"kind": "hip", "source": "x.cpp"}},
        ],
    )
    assert descriptor_files([tmp_path], "gfx942") == []
    catalog = load_catalog(descriptor_files([tmp_path], "gfx950"))
    assert catalog.kernels == (("packed", spec, "cd" * 32), ("authored", spec, None))
    assert catalog.missing_provenance == 1


def test_catalog_matches_problem_and_tile_divisibility(tmp_path) -> None:
    pytest.importorskip("kernels.gfx950.attention_dense")
    ukds = [
        _packed("bm256_bn64", _spec(bm=256, bn=64)),
        _packed("bm128_bn32", _spec(bm=128, bn=32)),
        _packed("full", _spec(causal=False)),
        _packed("d64", _spec(d=64)),
        _packed("bf16", _spec(dtype="bf16")),
    ]
    _write_descriptors(tmp_path, "gfx950", ukds)
    catalog = load_catalog(descriptor_files([tmp_path], "gfx950"))

    found = catalog_kernels(catalog, _problem(_graph(b=2, s=2048)))
    assert [k.name for k in found] == ["rocke:bm256_bn64", "rocke:bm128_bn32"]
    # The graph's shape replaces the descriptor's placeholder shape.
    assert {(k.spec["batch"], k.spec["seqlen_q"]) for k in found} == {(2, 2048)}

    # 1920 is a multiple of 128 but not of 256.
    found = catalog_kernels(catalog, _problem(_graph(s=1920)))
    assert [k.name for k in found] == ["rocke:bm128_bn32"]

    graph = copy.deepcopy(_graph())
    graph["nodes"][0]["attributes"]["causal_mask"] = False
    assert [k.name for k in catalog_kernels(catalog, _problem(graph))] == ["rocke:full"]

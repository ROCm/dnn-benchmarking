# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for tools/workloads_to_store.py, the graph-store migration.

The tool decides the store key of each graph, its tags, the name of a merged
geometry, and whether the result is valid enough to upload.
"""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

_TOOL = Path(__file__).resolve().parents[2] / "tools" / "workloads_to_store.py"


@pytest.fixture(scope="module")
def tool():
    spec = importlib.util.spec_from_file_location("workloads_to_store", _TOOL)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    yield module
    del sys.modules[spec.name]


def _graph(*node_types, name="g", io="half", metadata=None, dims=(1, 2)):
    graph = {
        "name": name,
        "io_data_type": io,
        "compute_data_type": "float",
        "intermediate_data_type": "float",
        "tensors": [{"uid": 1, "dims": list(dims)}],
        "nodes": [{"type": t, "name": f"n{i}"} for i, t in enumerate(node_types)],
    }
    if metadata is not None:
        graph["metadata"] = metadata
    return graph


@pytest.mark.parametrize(
    "nodes, expected",
    [
        (("MoeGroupedMatmulAttributes", "PointwiseAttributes"), "moe"),
        (("PointwiseAttributes", "MatmulAttributes"), "matmul"),
        (("RMSNormAttributes",), "norm"),
        (("BlockScaleDequantizeAttributes",), "quantize"),
        (("PointwiseAttributes", "PointwiseAttributes"), "pointwise"),
    ],
)
def test_operation_is_the_primary_node_kind(tool, nodes, expected):
    assert tool.derive_operation(_graph(*nodes))[0] == expected


@pytest.mark.parametrize(
    "nodes",
    [
        ("MatmulAttributes", "SdpaAttributes"),
        ("PointwiseAttributes", "ReductionAttributes"),
        ("NewOpAttributes",),
    ],
)
def test_ambiguous_or_unknown_operation_stops(tool, nodes):
    with pytest.raises(tool.MigrationError):
        tool.derive_operation(_graph(*nodes))


def test_convert_maps_input_tags_to_the_store_vocabulary(tool):
    metadata = {
        "operation": ["moe_grouped_matmul", "pointwise"],
        "direction": ["wgrad"],
        "dtype": ["float8_e4m3"],
        "model": ["Kimi_K26"],
        "attention_type": ["gqa"],
        "mask": ["causal"],
    }
    graph = _graph(
        "MoeGroupedMatmulAttributes",
        "PointwiseAttributes",
        io="fp8_e4m3",
        metadata=metadata,
    )
    copy = tool.convert("headline/moe", "a.json", graph)
    assert copy.tags == {
        "operation": ["moe"],
        "direction": ["wgrad"],
        "dtype": ["fp8_e4m3"],
        "model": ["kimi_k2_6"],
        "workload": ["headline"],
        "source": ["bench_cases"],
    }
    assert copy.notes == {"attention_type": ["gqa"], "mask": ["causal"]}


def test_convert_derives_missing_direction(tool):
    copy = tool.convert(
        "microbench/rocke", "c.json", _graph("ConvolutionWrwAttributes")
    )
    assert copy.tags["direction"] == ["wgrad"]


@pytest.mark.parametrize(
    "set_name, metadata",
    [
        ("microbench/unknown_set", {}),
        ("headline/attn", {"operation": ["matmul"]}),
        ("headline/attn", {"dtype": ["bfloat16"]}),
    ],
)
def test_convert_rejects_unmapped_sets_and_contradicting_tags(tool, set_name, metadata):
    with pytest.raises(tool.MigrationError):
        tool.convert(set_name, "a.json", _graph("SdpaAttributes", metadata=metadata))


def test_merge_takes_the_headline_name_and_combines_tags(tool):
    micro = tool.convert(
        "microbench/aiter",
        "a.json",
        _graph("SdpaAttributes", name="aiter_name", metadata={"model": ["m1"]}),
    )
    head = tool.convert(
        "headline/attn",
        "z.json",
        _graph("SdpaAttributes", name="headline_name", metadata={"model": ["m2"]}),
    )
    corpus = tool.convert(
        "corpus/SDPA", "b.json", _graph("SdpaAttributes", name="corpus_name")
    )
    (key, obj), *rest = tool.merge([micro, corpus, head]).items()
    assert not rest
    assert obj["name"] == "headline_name"
    assert obj["metadata"]["workload"] == ["corpus", "headline", "microbench"]
    assert obj["metadata"]["source"] == ["aiter", "bench_cases", "rocke"]
    assert obj["metadata"]["model"] == ["m1", "m2"]
    assert obj["metadata"]["notes"]["sets"] == [
        "corpus/SDPA",
        "headline/attn",
        "microbench/aiter",
    ]
    assert tool.validate(key, obj) == []


def test_validate_rejects_objects_the_store_must_not_hold(tool):
    copy = tool.convert("headline/attn", "a.json", _graph("SdpaAttributes"))
    ((key, obj),) = tool.merge([copy]).items()
    assert tool.validate(key, obj) == []

    no_source = {**obj, "metadata": {**obj["metadata"]}}
    del no_source["metadata"]["source"]
    assert tool.validate(key, no_source)

    off_vocabulary = {**obj, "metadata": {**obj["metadata"], "colour": ["red"]}}
    assert tool.validate(key, off_vocabulary)

    bad_value = {**obj, "metadata": {**obj["metadata"], "dtype": ["float8_e4m3"]}}
    assert tool.validate(key, bad_value)

    renamed_key = "f" * 64 + ".json"
    assert tool.validate(renamed_key, obj)
    assert tool.validate(key, {**obj, "name": " "})


def test_conversion_query_reports_extra_and_missing(tool):
    attn = tool.convert("headline/attn", "a.json", _graph("SdpaAttributes"))
    conv = tool.convert("headline/conv", "b.json", _graph("ConvolutionFwdAttributes"))
    objects = tool.merge([attn, conv])
    exact = tool.check_queries(
        objects, {"headline/attn": {attn.key}, "headline/conv": {conv.key}}
    )
    assert exact == {"headline/attn": (1, 0, 0), "headline/conv": (1, 0, 0)}

    wrong = tool.check_queries(objects, {"headline/attn": {conv.key}})
    assert wrong == {"headline/attn": (1, 1, 1)}


def test_pointers_are_read_at_the_ref_after_workloads_left_the_tree(tool, tmp_path):
    """The migration reads the DVC pointers from a pinned git ref, so it still
    works when the current tree has no ``Workloads/``."""

    def git(*args):
        return subprocess.run(
            [
                "git",
                "-C",
                str(tmp_path),
                "-c",
                "user.name=t",
                "-c",
                "user.email=t@t",
                *args,
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    git("init", "-q")
    pointer = tmp_path / "Workloads" / "headline" / "conv.tar.gz.dvc"
    pointer.parent.mkdir(parents=True)
    pointer.write_text(
        "outs:\n- md5: " + "a" * 32 + "\n  size: 123\n  path: conv.tar.gz\n"
    )
    git("add", "-A")
    git("commit", "-q", "-m", "pointers")
    pinned = git("rev-parse", "HEAD")
    git("rm", "-q", "-r", "Workloads")
    git("commit", "-q", "-m", "remove Workloads")

    assert not (tmp_path / "Workloads").exists()
    assert tool.read_pointers(tmp_path, pinned) == {
        "headline/conv": tool.Archive("headline/conv", "a" * 32, 123)
    }
    assert tool.read_pointers(tmp_path, "HEAD") == {}

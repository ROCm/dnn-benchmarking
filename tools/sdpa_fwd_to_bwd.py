#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT
"""Build the SDPA backward corpus as graph-store objects.

The input is a set of graphs: JSON files, directories, globs or tarballs,
usually a local copy of the graph store (``docs/graph-store.md``). The
output is one store object for each backward graph, ``<key>.json`` in the
``--out`` folder, ready for the upload procedure of the store.

The corpus has three origins, recorded in ``metadata.notes.origin``:

- ``converted_fwd``: every dense ``SdpaAttributes`` graph becomes one
  ``SdpaBackwardAttributes`` graph with the same Q, K, V, mask, scale, bounds
  and sequence-length inputs. The forward output O becomes a backward input,
  and the backward adds dO, the fp32 log-sum-exp stats and dQ, dK, dV.
  A graph with both ``seq_len_q`` and ``seq_len_kv`` gets
  ``padding_mask = true``, as cuDNN requires.
  Forward graphs with no backward equivalent in hipDNN are skipped: paged
  K/V caches, attention sinks, block masks, fp8 descale/scale tensors and
  dropout.
- ``captured_bwd``: ``SdpaBackwardAttributes`` input graphs, unchanged,
  except that a dropout probability the capture did not write into the graph
  (``_CAPTURED_DROPOUT``) is added (probability, seed and offset).
- ``model_spec``: training graphs with features no captured workload has
  (``_MODEL_SPECS``): attention dropout, additive bias with and without
  dBias, and ALiBi, at the published shapes of the named models.

An input graph whose only ``source`` tag is ``sdpa_fwd_to_bwd`` is an output
of an earlier run: the tool skips it and makes it again.

Each object has the tags ``workload=corpus``, ``source=sdpa_fwd_to_bwd``,
``operation=sdpa``, ``direction=bwd`` and ``mode=training``. The backward
graph gives ``dtype``, ``layout`` and ``phase``. ``model`` and
``model_family`` are the union over the input graphs. ``metadata.notes``
holds ``origin``, ``attention_type`` (``mla`` when an input graph says so),
``mask`` and ``sink``. Duplicates (same object key) are kept once.

Usage:
  python tools/sdpa_fwd_to_bwd.py --out /tmp/sdpa-bwd \\
      <work>/graph-store/therock-dvc/dnn-benchmarking-graphs
"""

import argparse
import collections
import copy
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from dnn_benchmarking.common.exceptions import GraphLoadError
from dnn_benchmarking.graph import store
from dnn_benchmarking.graph.resolver import resolve_graph_files_multi

#: The ``source`` tag of the objects that this tool writes.
SOURCE = "sdpa_fwd_to_bwd"


#: Forward inputs that have no backward equivalent, with the skip reason.
_FORWARD_ONLY_INPUTS = {
    "page_table_k_tensor_uid": "paged",
    "page_table_v_tensor_uid": "paged",
    "sink_token_tensor_uid": "sink",
    "block_mask_tensor_uid": "block_mask",
    "descale_q_tensor_uid": "fp8",
    "descale_k_tensor_uid": "fp8",
    "descale_v_tensor_uid": "fp8",
    "descale_s_tensor_uid": "fp8",
    "scale_s_tensor_uid": "fp8",
    "scale_o_tensor_uid": "fp8",
    "seed_tensor_uid": "dropout",
    "offset_tensor_uid": "dropout",
    "dropout_mask_tensor_uid": "dropout",
    "dropout_scale_tensor_uid": "dropout",
}

#: Forward inputs that the backward node takes unchanged.
_SHARED_INPUTS = (
    "scale_tensor_uid",
    "attn_mask_tensor_uid",
    "seq_len_q_tensor_uid",
    "seq_len_kv_tensor_uid",
)

#: Attributes the backward node shares with the forward node.
_SHARED_PARAMETERS = (
    "alibi_mask",
    "padding_mask",
    "causal_mask",
    "causal_mask_bottom_right",
    "dropout_probability",
    "attn_scale_value",
    "left_bound",
    "right_bound",
    "diagonal_alignment",
)

_BWD_INPUT_KEYS = (
    "q_tensor_uid",
    "k_tensor_uid",
    "v_tensor_uid",
    "o_tensor_uid",
    "do_tensor_uid",
    "stats_tensor_uid",
    "scale_tensor_uid",
    "attn_mask_tensor_uid",
    "seq_len_q_tensor_uid",
    "seq_len_kv_tensor_uid",
    "seed_tensor_uid",
    "offset_tensor_uid",
    "dropout_mask_tensor_uid",
    "dropout_scale_tensor_uid",
    "dropout_scale_inv_tensor_uid",
)

#: Training graphs with features no captured workload has. ``bias`` is the
#: additive bias shape, ``dbias`` adds its gradient as an output, ``dropout``
#: is the attention-probability dropout. Shapes are the models' published
#: configurations (Hugging Face ``config.json`` or the paper).
_MODEL_SPECS: Tuple[Dict[str, Any], ...] = (
    {
        "model": "bert-large-uncased",
        "source": "config.json: 16 heads, hidden 1024, "
        "attention_probs_dropout_prob 0.1; pre-training at 512 tokens",
        "variant": "pretrain_padding_mask_dropout",
        "dtype": "half",
        "shape": (16, 16, 16, 512, 512, 64, 64),
        "bias": "padding",
        "dropout": 0.1,
    },
    {
        "model": "t5-large",
        "source": "config.json: 16 heads, d_kv 64, dropout_rate 0.1; learned "
        "relative position bias shared over the batch",
        "variant": "encoder_relative_bias_dbias_dropout",
        "dtype": "bfloat16",
        "shape": (16, 16, 16, 512, 512, 64, 64),
        "bias": "shared",
        "dbias": True,
        "dropout": 0.1,
    },
    {
        "model": "swin-base",
        "source": "Swin-B stage 1: 7x7 windows (49 tokens), 4 heads of 32, "
        "learned relative position bias; 64 images of 56x56 = 4096 windows",
        "variant": "window_relative_bias_dbias",
        "dtype": "half",
        "shape": (4096, 4, 4, 49, 49, 32, 32),
        "bias": "shared",
        "dbias": True,
    },
    {
        "model": "alphafold2",
        "source": "Evoformer MSA row attention with pair bias: 8 heads of 32, "
        "256-residue crop, 512 MSA rows; the bias comes from the pair stack",
        "variant": "msa_row_pair_bias_dbias",
        "dtype": "bfloat16",
        "shape": (512, 8, 8, 256, 256, 32, 32),
        "bias": "shared",
        "dbias": True,
    },
    {
        "model": "bloom-176b",
        "source": "config.json: 112 heads, hidden 14336, ALiBi, 2048 tokens",
        "variant": "causal_alibi",
        "dtype": "bfloat16",
        "shape": (1, 112, 112, 2048, 2048, 128, 128),
        "causal": True,
        "alibi": True,
    },
    {
        "model": "mpt-7b",
        "source": "config.json: 32 heads, d_model 4096, ALiBi, max_seq_len 2048",
        "variant": "causal_alibi",
        "dtype": "bfloat16",
        "shape": (4, 32, 32, 2048, 2048, 128, 128),
        "causal": True,
        "alibi": True,
    },
)


class Skip(Exception):
    """The graph has no backward equivalent."""


#: Tags that every object of the corpus has.
_CORPUS_TAGS = {
    "workload": ["corpus"],
    "source": [SOURCE],
    "operation": ["sdpa"],
    "direction": ["bwd"],
    "mode": ["training"],
}


def _param(node: Dict[str, Any], key: str) -> Any:
    for section in ("inputs", "outputs", "attributes", "parameters"):
        value = (node.get(section) or {}).get(key)
        if value is not None:
            return value
    return None


def _contiguous_strides(dims: List[int]) -> List[int]:
    strides, step = [], 1
    for dim in reversed(dims):
        strides.append(step)
        step *= max(int(dim), 1)
    return strides[::-1]


def _tensor(uid: int, name: str, dims: List[int], data_type: str) -> Dict[str, Any]:
    """A non-virtual tensor; rank-4 attention tensors [B, H, S, D] use BSHD."""
    if name in ("stats", "seed", "offset") or len(dims) != 4 or "bias" in name:
        strides = _contiguous_strides(dims)
    else:
        batch, heads, seq, dim = dims
        strides = [seq * heads * dim, dim, heads * dim, 1]
    return {
        "uid": uid,
        "name": name,
        "dims": list(dims),
        "strides": strides,
        "data_type": data_type,
        "virtual": False,
    }


def _add_dropout(graph: Dict[str, Any], probability: float) -> None:
    """Philox dropout as cuDNN's ``set_dropout``: probability, seed, offset."""
    node = graph["nodes"][0]
    uid = max(int(t["uid"]) for t in graph["tensors"]) + 1
    for offset, name in enumerate(("seed", "offset")):
        graph["tensors"].append(_tensor(uid + offset, name, [1, 1, 1, 1], "int64"))
        node["inputs"][f"{name}_tensor_uid"] = uid + offset
    node["parameters"]["dropout_probability"] = probability


#: Attention dropout of captured backward graphs whose capture did not write
#: it into the graph, by graph name. The source is the ``dropout`` field of
#: the ``bench_cases_attn_bwd.json`` capture of the headline attention graphs.
_CAPTURED_DROPOUT = {
    "llama-3-8b-sft__bf16_bwd_causal_gqa_dropout_bf16_idx18_b4_hq32_kv8"
    "_sq4096_skv4096_d128v128_causal_bwd": 0.1,
    "llama-3-8b-sft__bf16_bwd_causal_gqa_dropout_sweep_bf16_idx35_b2_hq32_kv8"
    "_sq8192_skv8192_d128v128_causal_bwd": 0.1,
    "vit-l-14__fp16_bwd_vision_noncausal_dropout_fp16_idx21_b8_hq16_kv16"
    "_sq257_skv257_d64v64_full_bwd": 0.1,
}


def adopt(bwd: Dict[str, Any]) -> Dict[str, Any]:
    """Return a captured backward graph, with its ``_CAPTURED_DROPOUT`` applied."""
    nodes = bwd.get("nodes", [])
    if len(nodes) != 1 or nodes[0].get("type") != "SdpaBackwardAttributes":
        raise Skip("not_single_sdpa_bwd")
    graph = {k: copy.deepcopy(v) for k, v in bwd.items() if k != "metadata"}
    dropout = _CAPTURED_DROPOUT.get(bwd.get("name"), 0.0)
    node = graph["nodes"][0]
    if dropout and not float(_param(node, "dropout_probability") or 0.0):
        _add_dropout(graph, dropout)
    return graph


def spec_graph(spec: Dict[str, Any]) -> Dict[str, Any]:
    """Build the backward graph of one ``_MODEL_SPECS`` entry."""
    batch, heads_q, heads_kv, seq_q, seq_kv, dim, dim_v = spec["shape"]
    dtype = spec["dtype"]
    tensors: List[Dict[str, Any]] = []

    def add(name: str, dims: List[int], data_type: str = dtype) -> int:
        tensors.append(_tensor(len(tensors) + 1, name, dims, data_type))
        return len(tensors)

    inputs: Dict[str, Optional[int]] = dict.fromkeys(_BWD_INPUT_KEYS)
    inputs["q_tensor_uid"] = add("query", [batch, heads_q, seq_q, dim])
    inputs["k_tensor_uid"] = add("key", [batch, heads_kv, seq_kv, dim])
    inputs["v_tensor_uid"] = add("value", [batch, heads_kv, seq_kv, dim_v])
    inputs["o_tensor_uid"] = add("output", [batch, heads_q, seq_q, dim_v])
    inputs["do_tensor_uid"] = add("d_output", [batch, heads_q, seq_q, dim_v])
    inputs["stats_tensor_uid"] = add("stats", [batch, heads_q, seq_q, 1], "float")
    bias_dims = {
        "padding": [batch, 1, 1, seq_kv],
        "shared": [1, heads_q, seq_q, seq_kv],
    }.get(spec.get("bias"))
    if bias_dims:
        inputs["attn_mask_tensor_uid"] = add("bias", bias_dims)
    outputs = {
        "dq_tensor_uid": add("d_query", [batch, heads_q, seq_q, dim]),
        "dk_tensor_uid": add("d_key", [batch, heads_kv, seq_kv, dim]),
        "dv_tensor_uid": add("d_value", [batch, heads_kv, seq_kv, dim_v]),
        "dbias_tensor_uid": add("d_bias", bias_dims) if spec.get("dbias") else None,
    }
    parameters = dict.fromkeys(_SHARED_PARAMETERS)
    parameters.update(
        alibi_mask=bool(spec.get("alibi")),
        padding_mask=False,
        causal_mask=bool(spec.get("causal")),
        causal_mask_bottom_right=False,
        attn_scale_value=dim**-0.5,
        diagonal_alignment="TOP_LEFT",
    )
    name = (
        f"{spec['model']}__{dtype}_bwd_{spec['variant']}_b{batch}_hq{heads_q}"
        f"_kv{heads_kv}_sq{seq_q}_skv{seq_kv}_d{dim}v{dim_v}"
    )
    graph = {
        "name": name,
        "compute_data_type": "float",
        "io_data_type": dtype,
        "intermediate_data_type": dtype,
        "tensors": tensors,
        "nodes": [
            {
                "name": "sdpa_backward_node",
                "type": "SdpaBackwardAttributes",
                "compute_data_type": "float",
                "inputs": inputs,
                "outputs": outputs,
                "parameters": parameters,
            }
        ],
    }
    if spec.get("dropout"):
        _add_dropout(graph, spec["dropout"])
    return graph


def _graph_tags(graph: Dict[str, Any]) -> Tuple[Dict[str, List[str]], Dict[str, Any]]:
    """``(tags, notes)`` that the backward graph itself gives."""
    node = graph["nodes"][0]
    tensors = {int(t["uid"]): t for t in graph["tensors"]}
    q, k, v = (tensors[int(_param(node, f"{n}_tensor_uid"))] for n in ("q", "k", "v"))
    heads_q, seq_q, dim = (int(d) for d in q["dims"][1:])
    heads_kv, seq_kv = int(k["dims"][1]), int(k["dims"][2])
    if int(v["dims"][3]) != dim:
        attention = "mla"
    elif heads_kv == heads_q:
        attention = "mha"
    else:
        attention = "mqa" if heads_kv == 1 else "gqa"
    if (
        _param(node, "left_bound") is not None
        or _param(node, "right_bound") is not None
    ):
        mask = "sliding_window"
    elif _param(node, "causal_mask") or _param(node, "causal_mask_bottom_right"):
        mask = "causal"
    else:
        mask = "none"
    if seq_q == 1:
        phase = "decode"
    else:
        phase = "chunked_prefill" if seq_q < seq_kv else "prefill"
    tags = {
        **_CORPUS_TAGS,
        "dtype": [store.DTYPES[graph["io_data_type"]]],
        "phase": [phase],
    }
    layout = _layout(q, k)
    if layout:
        tags["layout"] = [layout]
    notes = {"attention_type": [attention], "mask": [mask], "sink": ["false"]}
    return tags, notes


def _layout(q: Dict[str, Any], k: Dict[str, Any]) -> Optional[str]:
    """``bshd`` or ``bhsd`` from the Q (or K) strides; None when they cannot tell."""
    for tensor in (q, k):
        dims, strides = tensor["dims"], tensor["strides"]
        if dims[1] != 1 and dims[2] != 1:
            return "bshd" if strides[1] < strides[2] else "bhsd"
    # One query row and one KV head: the size-1 sequence stride still shows
    # the layout of the generator (H*D for bshd, D for bhsd).
    heads, dim = q["dims"][1], q["dims"][3]
    if heads > 1 and q["strides"][1] == dim:
        return {heads * dim: "bshd", dim: "bhsd"}.get(q["strides"][2])
    return None


def metadata(
    graph: Dict[str, Any], origins: List[str], sources: List[Any]
) -> Dict[str, Any]:
    """Store metadata of one backward graph: its tags and ``notes``.

    ``sources`` holds the metadata of every source graph: the tags of an
    input graph, or the ``_MODEL_SPECS`` entry of a ``model_spec`` graph.
    """
    tags, notes = _graph_tags(graph)
    union: Dict[str, set] = {"model": set(), "model_family": set()}
    for source in sources:
        if not isinstance(source, dict):
            continue
        union["model"].update(_model_tags(source))
        union["model_family"].update(store.tag_values(source, "model_family"))
        if "mla" in store.tag_values(source, "attention_type") | store.tag_values(
            source.get("notes"), "attention_type"
        ):
            notes["attention_type"] = ["mla"]
    tags.update({key: sorted(values) for key, values in union.items() if values})
    notes["origin"] = sorted(set(origins))
    return {**tags, "notes": notes}


def convert(fwd: Dict[str, Any]) -> Dict[str, Any]:
    """Return the backward graph of one forward SDPA graph, or raise Skip."""
    nodes = fwd.get("nodes", [])
    if len(nodes) != 1 or nodes[0].get("type") != "SdpaAttributes":
        raise Skip("not_single_sdpa_fwd")
    node = nodes[0]
    for key, reason in _FORWARD_ONLY_INPUTS.items():
        if _param(node, key) is not None:
            raise Skip(reason)
    if float(_param(node, "dropout_probability") or 0.0) != 0.0:
        raise Skip("dropout")
    for key in ("max_tensor_uid", "sum_exp_tensor_uid", "amax_s_tensor_uid"):
        if _param(node, key) is not None:
            raise Skip("fp8")

    tensors = {int(t["uid"]): t for t in fwd["tensors"]}
    next_uid = max(tensors) + 1

    def new_tensor(like: Dict[str, Any], name: str) -> int:
        nonlocal next_uid
        uid, next_uid = next_uid, next_uid + 1
        tensor = copy.deepcopy(like)
        tensor.update(uid=uid, name=name, virtual=False)
        tensors[uid] = tensor
        return uid

    q, k, v, o = (
        tensors[int(_param(node, f"{n}_tensor_uid"))] for n in ("q", "k", "v", "o")
    )
    o["virtual"] = False
    batch, heads_q, seq_q = (int(d) for d in q["dims"][:3])
    stats_dims = [batch, heads_q, seq_q, 1]

    inputs: Dict[str, Optional[int]] = dict.fromkeys(_BWD_INPUT_KEYS)
    for key in ("q_tensor_uid", "k_tensor_uid", "v_tensor_uid", *_SHARED_INPUTS):
        inputs[key] = _param(node, key)
    inputs["o_tensor_uid"] = o["uid"]
    inputs["do_tensor_uid"] = new_tensor(o, "d_output")
    inputs["stats_tensor_uid"] = new_tensor(
        {
            "dims": stats_dims,
            "strides": _contiguous_strides(stats_dims),
            "data_type": "float",
        },
        "stats",
    )
    outputs = {
        "dq_tensor_uid": new_tensor(q, "d_query"),
        "dk_tensor_uid": new_tensor(k, "d_key"),
        "dv_tensor_uid": new_tensor(v, "d_value"),
        "dbias_tensor_uid": None,
    }
    parameters = {key: _param(node, key) for key in _SHARED_PARAMETERS}
    for flag in (
        "alibi_mask",
        "padding_mask",
        "causal_mask",
        "causal_mask_bottom_right",
    ):
        parameters[flag] = bool(parameters[flag])
    parameters["diagonal_alignment"] = parameters["diagonal_alignment"] or "TOP_LEFT"
    # cuDNN requires padding_mask with seq_len tensors (SEM-008), and hipDNN's
    # SdpaBwdNode checks the seq_len shapes only when padding_mask is set.
    seq_lens = (inputs["seq_len_q_tensor_uid"], inputs["seq_len_kv_tensor_uid"])
    if None not in seq_lens:
        parameters["padding_mask"] = True

    used = {u for u in (*inputs.values(), *outputs.values()) if u is not None}
    return {
        "name": f"{fwd['name']}_bwd",
        "compute_data_type": fwd["compute_data_type"],
        "io_data_type": fwd["io_data_type"],
        "intermediate_data_type": fwd["intermediate_data_type"],
        "tensors": [tensors[uid] for uid in sorted(used)],
        "nodes": [
            {
                "name": "sdpa_backward_node",
                "type": "SdpaBackwardAttributes",
                "compute_data_type": node.get("compute_data_type", "float"),
                "inputs": inputs,
                "outputs": outputs,
                "parameters": parameters,
            }
        ],
    }


def _model_tags(metadata: Any) -> List[str]:
    model = metadata.get("model") if isinstance(metadata, dict) else None
    names = model if isinstance(model, list) else [model] if model else []
    return [re.sub(r"[^a-z0-9]+", "_", str(n).lower()).strip("_") for n in names]


def iter_graphs(paths: List[str]) -> Iterator[Tuple[str, Dict[str, Any]]]:
    """Yield ``(label, graph)`` for each input graph file, in path order.

    Skips the outputs of an earlier run: graphs whose only ``source`` tag is
    ``sdpa_fwd_to_bwd``.

    Raises:
        GraphLoadError: A tarball cannot be read.
    """
    tmpdirs, files, _ = resolve_graph_files_multi(paths)
    try:
        for path in files:
            with open(path, encoding="utf-8") as f:
                graph = json.load(f)
            if store.tag_values(graph.get("metadata"), "source") == {SOURCE}:
                continue
            yield path, graph
    finally:
        for tmpdir in tmpdirs:
            tmpdir.cleanup()


def _graph_origins(
    paths: List[str], skipped: Dict[str, int]
) -> Iterator[Tuple[str, str, Dict[str, Any], Any]]:
    """Yield ``(origin, label, backward graph, source metadata)``."""
    for label, graph in iter_graphs(paths):
        types = {n.get("type") for n in graph.get("nodes", [])}
        if "SdpaAttributes" in types:
            origin, make = "converted_fwd", convert
        elif "SdpaBackwardAttributes" in types:
            origin, make = "captured_bwd", adopt
        else:
            continue
        try:
            yield origin, label, make(graph), graph.get("metadata")
        except Skip as reason:
            skipped[str(reason)] += 1
    for spec in _MODEL_SPECS:
        yield "model_spec", f"model_spec:{spec['source']}", spec_graph(spec), spec


def build(paths: List[str]) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, int]]:
    """Collect and de-duplicate.

    Returns ``{key: store object}`` and the skip counts. The first input in
    path order names a duplicate.
    """
    found: Dict[str, Dict[str, Any]] = {}
    skipped: Dict[str, int] = collections.Counter()
    for origin, _, bwd, source in _graph_origins(paths, skipped):
        entry = found.setdefault(
            store.geometry_key(bwd), {"graph": bwd, "origins": [], "sources": []}
        )
        entry["origins"].append(origin)
        entry["sources"].append(source)
    objects = {
        key: {
            **e["graph"],
            "metadata": metadata(e["graph"], e["origins"], e["sources"]),
        }
        for key, e in found.items()
    }
    return objects, dict(skipped)


def write_objects(objects: Dict[str, Dict[str, Any]], out: Path) -> None:
    """Write each object as ``<out>/<key>``. ``out`` must be empty or new."""
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError(f"{out} is not empty")
    for key, obj in objects.items():
        (out / key).write_bytes(store.object_bytes(obj))


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "graphs", nargs="+", help="input graph files, directories, globs or tarballs"
    )
    parser.add_argument(
        "--out", type=Path, required=True, help="empty folder for the store objects"
    )
    args = parser.parse_args(argv)

    try:
        objects, skipped = build(args.graphs)
    except (GraphLoadError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    problems = [
        f"{key}: {problem}"
        for key in sorted(objects)
        for problem in store.validate_object(key, objects[key])
    ]
    if problems:
        for problem in problems:
            print(f"error: {problem}", file=sys.stderr)
        return 1
    try:
        write_objects(objects, args.out)
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    origins = collections.Counter(
        origin
        for obj in objects.values()
        for origin in obj["metadata"]["notes"]["origin"]
    )
    print(
        f"{len(objects)} backward graphs written to {args.out}: "
        + ", ".join(f"{n} {o}" for o, n in sorted(origins.items())),
        file=sys.stderr,
    )
    for reason, count in sorted(skipped.items()):
        print(f"skipped {count} graphs: {reason}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

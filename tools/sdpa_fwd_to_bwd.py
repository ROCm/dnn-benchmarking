#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT
"""Build the SDPA backward corpus from the workload tarballs.

The corpus has three origins, recorded in ``metadata.origin``:

- ``converted_fwd``: every dense ``SdpaAttributes`` graph becomes one
  ``SdpaBackwardAttributes`` graph with the same Q, K, V, mask, scale, bounds
  and sequence-length inputs. The forward output O becomes a backward input,
  and the backward adds dO, the fp32 log-sum-exp stats and dQ, dK, dV.
  A graph with both ``seq_len_q`` and ``seq_len_kv`` gets
  ``padding_mask = true``, as cuDNN requires.
  Forward graphs with no backward equivalent in hipDNN are skipped: paged
  K/V caches, attention sinks, block masks, fp8 descale/scale tensors and
  dropout.
- ``captured_bwd``: ``SdpaBackwardAttributes`` graphs already in the
  tarballs, unchanged, except that a ``metadata.dropout`` probability the
  capture did not write into the graph is added (probability, seed and
  offset).
- ``model_spec``: training graphs with features no captured workload has
  (``_MODEL_SPECS``): attention dropout, additive bias with and without
  dBias, and ALiBi, at the published shapes of the named models.

``metadata.shape`` is ``decode`` (Sq = 1), ``sq_lt_skv``, ``sq_eq_skv`` or
``sq_gt_skv``. Duplicates (canonical JSON without ``name`` and
``metadata``) are kept once, with the sources of every copy in
``metadata.source``.

Usage:
  python tools/sdpa_fwd_to_bwd.py -o Workloads/corpus/SDPABackward/files.tar.gz \\
      'Workloads/**/*.tar.gz'
"""

import argparse
import collections
import copy
import glob
import hashlib
import gzip
import io
import json
import re
import sys
import tarfile
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

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


def adopt(bwd: Dict[str, Any]) -> Dict[str, Any]:
    """Return a captured backward graph, with its ``metadata.dropout`` applied."""
    nodes = bwd.get("nodes", [])
    if len(nodes) != 1 or nodes[0].get("type") != "SdpaBackwardAttributes":
        raise Skip("not_single_sdpa_bwd")
    graph = {k: copy.deepcopy(v) for k, v in bwd.items() if k != "metadata"}
    metadata = bwd.get("metadata")
    dropout = float(metadata.get("dropout") or 0.0) if isinstance(metadata, dict) else 0
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


def _shape_tag(graph: Dict[str, Any]) -> str:
    node = graph["nodes"][0]
    dims = {int(t["uid"]): t["dims"] for t in graph["tensors"]}
    seq_q = int(dims[int(_param(node, "q_tensor_uid"))][2])
    seq_kv = int(dims[int(_param(node, "k_tensor_uid"))][2])
    if seq_q == 1:
        return "decode"
    return {-1: "sq_lt_skv", 0: "sq_eq_skv", 1: "sq_gt_skv"}[
        (seq_q > seq_kv) - (seq_q < seq_kv)
    ]


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


def identity(graph: Dict[str, Any]) -> str:
    """SHA-256 of the canonical executable JSON (no name or metadata)."""
    body = {k: v for k, v in graph.items() if k not in ("name", "metadata")}
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _model_tags(metadata: Any) -> List[str]:
    model = metadata.get("model") if isinstance(metadata, dict) else None
    names = model if isinstance(model, list) else [model] if model else []
    return [re.sub(r"[^a-z0-9]+", "_", str(n).lower()).strip("_") for n in names]


def iter_graphs(
    patterns: List[str], root: Path, exclude: Optional[Path] = None
) -> Iterator[Tuple[str, Dict[str, Any]]]:
    """Yield ``(source, graph)`` for each JSON member of the matched tarballs.

    ``exclude`` (the output archive) is never read, so a rebuild does not
    adopt the previous corpus.
    """
    paths = sorted(
        {Path(p) for pat in patterns for p in glob.glob(pat, recursive=True)}
    )
    for path in paths:
        if exclude is not None and path.resolve() == exclude.resolve():
            continue
        try:
            label = path.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            label = path.as_posix()
        with tarfile.open(path) as archive:
            for member in sorted(archive.getmembers(), key=lambda m: m.name):
                if member.isfile() and member.name.endswith(".json"):
                    graph = json.load(archive.extractfile(member))
                    yield f"{label}:{member.name}", graph


def _graph_origins(
    patterns: List[str], root: Path, exclude: Optional[Path], skipped: Dict[str, int]
) -> Iterator[Tuple[str, str, Dict[str, Any], Any]]:
    """Yield ``(origin, source, backward graph, source metadata)``."""
    for source, graph in iter_graphs(patterns, root, exclude):
        types = {n.get("type") for n in graph.get("nodes", [])}
        if "SdpaAttributes" in types:
            origin, make = "converted_fwd", convert
        elif "SdpaBackwardAttributes" in types:
            origin, make = "captured_bwd", adopt
        else:
            continue
        try:
            yield origin, source, make(graph), graph.get("metadata")
        except Skip as reason:
            skipped[str(reason)] += 1
    for spec in _MODEL_SPECS:
        yield "model_spec", f"model_spec:{spec['source']}", spec_graph(spec), spec


def build(
    patterns: List[str], root: Path, exclude: Optional[Path] = None
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, int]]:
    """Collect and de-duplicate; return graphs by identity and skip counts."""
    kept: Dict[str, Dict[str, Any]] = {}
    skipped: Dict[str, int] = collections.Counter()
    for origin, source, bwd, metadata in _graph_origins(
        patterns, root, exclude, skipped
    ):
        entry = kept.setdefault(
            identity(bwd),
            {
                **bwd,
                "metadata": {
                    "operation": ["sdpa"],
                    "direction": ["bwd"],
                    "dtype": [bwd["io_data_type"]],
                    "shape": [_shape_tag(bwd)],
                    "origin": [],
                    "model": [],
                    "source": [],
                },
            },
        )
        tags = entry["metadata"]
        tags["origin"] = sorted(set(tags["origin"]) | {origin})
        tags["source"].append(source)
        tags["model"] = sorted(set(tags["model"]) | set(_model_tags(metadata)))
    for entry in kept.values():
        if not entry["metadata"]["model"]:
            del entry["metadata"]["model"]
    return kept, dict(skipped)


def _unique_names(graphs: List[Dict[str, Any]]) -> None:
    seen: Dict[str, int] = collections.Counter()
    for graph in graphs:
        base = re.sub(r"[^A-Za-z0-9_.-]+", "_", graph["name"])
        seen[base] += 1
        graph["name"] = base if seen[base] == 1 else f"{base}_{seen[base]}"


def write_archive(graphs: List[Dict[str, Any]], output: Path, prefix: str) -> None:
    """Write the graphs as a byte-reproducible ``.tar.gz`` (fixed mtimes)."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with (
        open(output, "wb") as raw,
        gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed,
        tarfile.open(fileobj=compressed, mode="w") as archive,
    ):
        for graph in graphs:
            data = (json.dumps(graph, indent=1) + "\n").encode()
            info = tarfile.TarInfo(f"{prefix}/{graph['name']}.json")
            info.size, info.mode, info.mtime = len(data), 0o644, 0
            archive.addfile(info, io.BytesIO(data))


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("tarballs", nargs="+", help="workload tarball paths or globs")
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument(
        "--prefix", default="sdpa_bwd", help="directory inside the archive"
    )
    parser.add_argument(
        "--root", type=Path, default=Path("."), help="base of source labels"
    )
    args = parser.parse_args(argv)

    kept, skipped = build(args.tarballs, args.root, args.output)
    graphs = sorted(kept.values(), key=lambda g: g["metadata"]["source"][0])
    _unique_names(graphs)
    write_archive(graphs, args.output, args.prefix)
    origins = collections.Counter(
        origin for g in graphs for origin in g["metadata"]["origin"]
    )
    sources = sum(len(g["metadata"]["source"]) for g in graphs)
    print(
        f"{sources} source graphs, {sources - len(graphs)} duplicates, "
        f"{len(graphs)} unique backward graphs written to {args.output}: "
        + ", ".join(f"{n} {o}" for o, n in sorted(origins.items())),
        file=sys.stderr,
    )
    for reason, count in sorted(skipped.items()):
        print(f"skipped {count} graphs: {reason}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

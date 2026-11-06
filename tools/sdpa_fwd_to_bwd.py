#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT
"""Convert SDPA forward workload graphs into their backward equivalents.

Every dense ``SdpaAttributes`` graph in the given tarballs becomes one
``SdpaBackwardAttributes`` graph with the same Q, K, V, mask, scale, bounds
and sequence-length inputs. The forward output O becomes a backward input,
and the backward adds dO, the fp32 log-sum-exp stats and dQ, dK, dV.

Forward graphs with no backward equivalent in hipDNN are skipped: paged K/V
caches, attention sinks, block masks, fp8 descale/scale tensors and dropout.
Duplicates (canonical JSON without ``name`` and ``metadata``) are kept once,
with the sources of every copy in ``metadata.source``.

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


class Skip(Exception):
    """The forward graph has no backward equivalent."""


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


def iter_forward_graphs(
    patterns: List[str], root: Path
) -> Iterator[Tuple[str, Dict[str, Any]]]:
    """Yield ``(source, graph)`` for each JSON member of the matched tarballs."""
    paths = sorted(
        {Path(p) for pat in patterns for p in glob.glob(pat, recursive=True)}
    )
    for path in paths:
        try:
            label = path.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            label = path.as_posix()
        with tarfile.open(path) as archive:
            for member in sorted(archive.getmembers(), key=lambda m: m.name):
                if member.isfile() and member.name.endswith(".json"):
                    graph = json.load(archive.extractfile(member))
                    yield f"{label}:{member.name}", graph


def build(
    patterns: List[str], root: Path
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, int]]:
    """Convert and de-duplicate; return graphs by identity and skip counts."""
    kept: Dict[str, Dict[str, Any]] = {}
    skipped: Dict[str, int] = collections.Counter()
    for source, fwd in iter_forward_graphs(patterns, root):
        if not any(n.get("type") == "SdpaAttributes" for n in fwd.get("nodes", [])):
            continue
        try:
            bwd = convert(fwd)
        except Skip as reason:
            skipped[str(reason)] += 1
            continue
        key = identity(bwd)
        metadata = fwd.get("metadata")
        entry = kept.setdefault(
            key,
            {
                **bwd,
                "metadata": {
                    "operation": ["sdpa"],
                    "direction": ["bwd"],
                    "dtype": [bwd["io_data_type"]],
                    "model": [],
                    "source": [],
                },
            },
        )
        tags = entry["metadata"]
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

    kept, skipped = build(args.tarballs, args.root)
    graphs = sorted(kept.values(), key=lambda g: g["metadata"]["source"][0])
    _unique_names(graphs)
    write_archive(graphs, args.output, args.prefix)
    converted = sum(len(g["metadata"]["source"]) for g in graphs)
    print(
        f"{converted} forward graphs converted, {converted - len(graphs)} duplicates, "
        f"{len(graphs)} unique backward graphs written to {args.output}",
        file=sys.stderr,
    )
    for reason, count in sorted(skipped.items()):
        print(f"skipped {count} forward graphs: {reason}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

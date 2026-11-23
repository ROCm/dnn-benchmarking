#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT
"""Convert the ``Workloads/`` DVC tarballs into flat graph-store objects.

The graph store holds one JSON object per graph geometry, with the key
``<geometry_sha>.json``. Tags in ``metadata`` select the graphs.

Procedure:

1. Read the ``Workloads/**/*.tar.gz.dvc`` pointers at ``--ref``. Download
   each tarball anonymously from the DVC remote, and check its md5 and size.
   ``--archive SET=PATH@MD5`` adds a set, or replaces the archive of a set,
   from a local file with a known md5.
2. For each graph file:

   - Compute the key: sha256 of the canonical JSON without the top-level
     ``name`` and ``metadata`` (``geometry_key``).
   - Set ``workload`` and ``source`` from the set (``SETS``).
   - Derive ``operation`` (exactly one value) and ``dtype`` from the graph,
     and ``direction`` when the graph has no ``direction`` tag
     (``NODE_TYPES``, ``DTYPES``). An input ``dtype`` of ``float8_e4m3``
     becomes ``fp8_e4m3``.
   - Normalize the tags: sorted, unique, lowercase string arrays.
   - Apply the model alias table (``MODEL_ALIASES``).
   - Move keys that are not in the vocabulary (``CLOSED_TAGS``,
     ``OPEN_TAGS``) into ``metadata.notes``.

3. Merge the copies of each key. The name comes from the set with the highest
   priority (headline, corpus, microbench, models); ties go to the sorted
   set path, then the member path. Tags and notes are combined, and
   ``metadata.notes.sets`` lists every source set.
4. Validate every object: key, vocabulary, non-empty name, and a ``source``
   tag. Then check that each set's conversion query (``SETS``) returns
   exactly that set. Stop on any failure.
5. Write the objects to ``--out`` (the dry run), then read 1% of them back and
   check their keys.
6. Only with ``--upload s3://BUCKET/PREFIX/``: upload every object with
   ``If-None-Match: *``, 16 in parallel. This needs write credentials and a
   ``boto3`` that supports ``IfNoneMatch`` (1.35 or later). Then list the
   prefix anonymously, check the key count, and re-hash a 1% sample.

Every ``operation`` value comes from the node types (``NODE_TYPES``). An
input ``operation`` tag (one value per node type) must agree with it, after
these renames (``VALUE_ALIASES``): ``moe_grouped_matmul`` -> ``moe``;
``rmsnorm``, ``layernorm`` -> ``norm``; ``block_scale_quantize``,
``block_scale_dequantize`` -> ``quantize``. The values ``convolution``,
``sdpa``, ``matmul``, ``batchnorm``, ``resample``, ``reduction`` and
``pointwise`` stay. A graph with several node kinds gets its primary
operation: ``pointwise``, ``quantize`` and ``reduction`` count only when the
graph has no other operation, so a MoE grouped matmul with a pointwise
activation is ``moe``. Two primary operations stop the migration.

Usage:
  python tools/workloads_to_store.py --ref <commit> --out /tmp/store \\
      --cache /tmp/dvc-cache \\
      --archive microbench/rocke=/path/to/rocke.tar.gz@<md5>
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import dataclasses
import hashlib
import json
import random
import re
import subprocess
import sys
import tarfile
import tempfile
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ElementTree
from pathlib import Path
from typing import Any, Iterable, Iterator

#: Anonymous HTTPS base of the DVC remote ``s3://therock-dvc/dnn-benchmarking``.
DVC_REMOTE_URL = "https://therock-dvc.s3.amazonaws.com/dnn-benchmarking"

#: Name priority of the top-level set directories (lowest number wins).
NAME_PRIORITY = {"headline": 0, "corpus": 1, "microbench": 2, "models": 3}

PARALLEL = 16
TOKEN = re.compile(r"^[a-z0-9][a-z0-9._+-]{0,63}$")
KEY = re.compile(r"^[0-9a-f]{64}\.json$")


def _query(**tags: str) -> dict[str, tuple[str, ...]]:
    return {key: tuple(values.split("|")) for key, values in tags.items()}


@dataclasses.dataclass(frozen=True)
class SetMapping:
    """Tags that a set gives its graphs, and the query that returns the set."""

    workload: str
    source: str
    query: dict[str, tuple[str, ...]]


def _microbench(source: str, **extra: str) -> SetMapping:
    return SetMapping(
        "microbench", source, _query(workload="microbench", source=source, **extra)
    )


def _model(source: str, model: str) -> SetMapping:
    return SetMapping(
        "model", source, _query(workload="model", source=source, model=model)
    )


#: Every ``Workloads/`` set: its ``workload`` and ``source`` tags, and its
#: conversion query (AND across keys, OR within a key). A set that is missing
#: from this table stops the migration, so every graph gets a ``source``.
SETS: dict[str, SetMapping] = {
    "headline/attn": SetMapping(
        "headline", "bench_cases", _query(workload="headline", operation="sdpa")
    ),
    "headline/bnorm": SetMapping(
        "headline", "synthetic", _query(workload="headline", operation="batchnorm")
    ),
    "headline/conv": SetMapping(
        "headline",
        "bench_cases",
        _query(workload="headline", operation="convolution"),
    ),
    "headline/hipblaslt": SetMapping(
        "headline", "hipblaslt", _query(workload="headline", source="hipblaslt")
    ),
    "headline/moe": SetMapping(
        "headline",
        "bench_cases",
        _query(workload="headline", source="bench_cases", operation="moe|matmul"),
    ),
    "headline/norm": SetMapping(
        "headline", "cudnn_frontend", _query(workload="headline", operation="norm")
    ),
    "microbench/aiter": _microbench("aiter"),
    "microbench/aotriton": _microbench("aotriton"),
    "microbench/conv_sweep": _microbench("conv_sweep"),
    "microbench/cudnn_attention_inference": _microbench("cudnn_attention_inference"),
    "microbench/cudnn_attention_training": _microbench("cudnn_attention_training"),
    "microbench/cudnn_bench_moe": _microbench("cudnn_bench_moe"),
    "microbench/cudnn_frontend": _microbench("cudnn_frontend"),
    "microbench/cudnn_gemm": _microbench("cudnn_gemm"),
    "microbench/hipkittens": _microbench("hipkittens"),
    "microbench/pytorch": _microbench("pytorch"),
    # corpus/SDPA also has source=rocke, and 2 of its graphs are also in
    # another microbench set.
    "microbench/rocke": _microbench("rocke", operation="convolution"),
    # The 8 sets without their own capture are copies of
    # cudnn_attention_training shapes.
    "models/auto_regressive_dit": _model(
        "cudnn_attention_training", "auto_regressive_dit"
    ),
    "models/dsv3": _model("cudnn_attention_training", "deepseek_v3"),
    "models/gpt_oss": _model("cudnn_attention_training", "gpt_oss"),
    "models/kimiK26": _model("cudnn_attention_training", "kimi_k2_6"),
    "models/llama3.1": _model("cudnn_attention_training", "llama3_1"),
    "models/ltx2": _model("cudnn_attention_training", "ltx2"),
    "models/qwen35": _model("cudnn_attention_training", "qwen3_5"),
    "models/wan22_a14b": _model("cudnn_attention_training", "wan2_2_a14b"),
    "models/glm_5_2_moe": _model("hf_config", "glm5_2"),
    "models/qwen3_235b_a22b_moe": _model("hf_config", "qwen3_235b_a22b"),
    "models/qwen3_30b_a3b_moe": _model("hf_config", "qwen3_30b_a3b"),
    "models/qwen3_32b": _model("hf_config", "qwen3_32b"),
    "models/qwen3_8b": _model("hf_config", "qwen3_8b"),
    "models/mad_llama3_1_8b_train": _model("mad", "llama3_1_8b"),
    "corpus/Convolution": SetMapping(
        "corpus",
        "brad_corpus",
        _query(workload="corpus", operation="convolution"),
    ),
    "corpus/SDPA": SetMapping(
        "corpus", "rocke", _query(workload="corpus", operation="sdpa", direction="fwd")
    ),
    "corpus/SDPABackward": SetMapping(
        "corpus",
        "sdpa_fwd_to_bwd",
        _query(workload="corpus", operation="sdpa", direction="bwd"),
    ),
}

#: Tag keys with a closed list of values.
CLOSED_TAGS: dict[str, frozenset[str]] = {
    "workload": frozenset({"headline", "microbench", "corpus", "model", "integration"}),
    "source": frozenset(
        {
            "bench_cases",
            "synthetic",
            "hipblaslt",
            "cudnn_frontend",
            "aiter",
            "aotriton",
            "conv_sweep",
            "cudnn_attention_inference",
            "cudnn_attention_training",
            "cudnn_bench_moe",
            "cudnn_gemm",
            "hipkittens",
            "pytorch",
            "rocke",
            "hf_config",
            "mad",
            "brad_corpus",
            "sdpa_fwd_to_bwd",
            "integration_tests",
        }
    ),
    "tier": frozenset({"quick", "standard", "full", "comprehensive"}),
    "operation": frozenset(
        {
            "convolution",
            "sdpa",
            "matmul",
            "moe",
            "norm",
            "batchnorm",
            "pointwise",
            "reduction",
            "resample",
            "quantize",
        }
    ),
    "direction": frozenset({"fwd", "bwd", "dgrad", "wgrad"}),
    "dtype": frozenset({"float32", "float16", "bfloat16", "fp8_e4m3"}),
    "layout": frozenset({"nchw", "nhwc", "ncdhw", "ndhwc", "bshd", "bhsd"}),
    "mode": frozenset({"training", "inference"}),
    "phase": frozenset({"prefill", "chunked_prefill", "decode"}),
}

#: Tag keys with free lowercase tokens.
OPEN_TAGS = frozenset(
    {
        "model_family",
        "model",
        "use_case",
        "capability",
        "variant",
        "priority",
        "purpose",
        "suite",
    }
)

#: Node type -> (operation, direction). A direction of None means that the
#: node type does not decide it.
NODE_TYPES: dict[str, tuple[str, str | None]] = {
    "ConvolutionFwdAttributes": ("convolution", "fwd"),
    "ConvolutionBwdAttributes": ("convolution", "dgrad"),
    "ConvolutionWrwAttributes": ("convolution", "wgrad"),
    "SdpaAttributes": ("sdpa", "fwd"),
    "SdpaBackwardAttributes": ("sdpa", "bwd"),
    "MatmulAttributes": ("matmul", None),
    "MoeGroupedMatmulAttributes": ("moe", None),
    "MoeGroupedMatmulBwdAttributes": ("moe", None),
    "BatchnormAttributes": ("batchnorm", "fwd"),
    "BatchnormInferenceAttributes": ("batchnorm", "fwd"),
    "BatchnormBackwardAttributes": ("batchnorm", "bwd"),
    "LayernormAttributes": ("norm", "fwd"),
    "LayernormBackwardAttributes": ("norm", "bwd"),
    "RMSNormAttributes": ("norm", "fwd"),
    "RMSNormBackwardAttributes": ("norm", "bwd"),
    "ResampleFwdAttributes": ("resample", "fwd"),
    "PointwiseAttributes": ("pointwise", None),
    "ReductionAttributes": ("reduction", None),
    "BlockScaleQuantizeAttributes": ("quantize", None),
    "BlockScaleDequantizeAttributes": ("quantize", None),
}

#: Operations that count only when a graph has no other operation.
AUXILIARY_OPERATIONS = frozenset({"pointwise", "quantize", "reduction"})

#: Graph ``io_data_type`` -> ``dtype`` tag.
DTYPES = {
    "float": "float32",
    "half": "float16",
    "bfloat16": "bfloat16",
    "fp8_e4m3": "fp8_e4m3",
}

#: Other spellings of tag values, per key.
VALUE_ALIASES: dict[str, dict[str, str]] = {
    "operation": {
        "moe_grouped_matmul": "moe",
        "rmsnorm": "norm",
        "layernorm": "norm",
        "block_scale_quantize": "quantize",
        "block_scale_dequantize": "quantize",
    },
    "dtype": {
        "float8_e4m3": "fp8_e4m3",
        "float": "float32",
        "half": "float16",
        "fp16": "float16",
        "bf16": "bfloat16",
        "fp32": "float32",
    },
    "phase": {"context": "prefill", "full_prefill": "prefill", "generation": "decode"},
}

#: One spelling for each model. Keys are the canonical ``model`` values;
#: values are other spellings, after ``model_token`` normalization.
MODEL_ALIASES: dict[str, tuple[str, ...]] = {
    "auto_regressive_dit": ("ar_dit",),
    "deepseek_v3": (
        "dsv3",
        "deepseek_v3_131072x128",
        "deepseek_v3_2048x4096",
        "deepseek_v3_8192x128",
    ),
    "deepseek_v4_flash": (
        "dsv4_flash",
        "dsv4_flash_tp2",
        "dsv4_flash_tp4",
        "dsv4_flash_tp8",
    ),
    "deepseek_v4_pro": ("dsv4_pro", "dsv4_pro_tp2", "dsv4_pro_tp4", "dsv4_pro_tp8"),
    "efficientnet_v2_m": ("efficientnetv2_m",),
    "efficientnet_v2_s": ("efficientnetv2_s",),
    "flux2_vae": ("flux_2_vae",),
    "ghostnet_v2": ("ghostnetv2",),
    "glm5": ("glm_5",),
    "glm5_2": ("glm_5_2", "glm_5_2_moe"),
    "gpt3_13b_class": ("gpt_3_13b_class",),
    "gpt_oss": (
        "gpt_oss_full",
        "gpt_oss_full_tp2",
        "gpt_oss_full_tp4",
        "gpt_oss_full_tp8",
        "gpt_oss_swa",
        "gpt_oss_swa_tp2",
        "gpt_oss_swa_tp4",
        "gpt_oss_swa_tp8",
    ),
    "hunyuan_video_vae": ("hunyuanvideo_vae",),
    "kimi_k2_6": ("kimik26", "kimi_k26"),
    "kimi_k3": ("kimi_k3_tp2", "kimi_k3_tp4", "kimi_k3_tp8"),
    "llama2_7b": ("llama_2_7b",),
    "llama3_1": ("llama3_1_tp2", "llama3_1_tp4", "llama3_1_tp8"),
    "llama3_1_405b": ("llama_3_1_405b",),
    "llama3_1_8b": ("llama_3_1_8b",),
    "llama3_70b": ("llama_3_70b",),
    "llama3_8b": ("llama_3_8b", "llama_3_8b_sft", "llama3_8b_sft"),
    "mobilenet_v4_conv_l": ("mobilenetv4_conv_l",),
    "mobilenet_v4_conv_s": ("mobilenetv4_conv_s",),
    "mochi1_vae": ("mochi_1_vae",),
    "nemotron_h_56b": ("nemotronh_56b",),
    "qwen3_235b_a22b": ("qwen3_235b", "qwen3_235b_a22b_moe"),
    "qwen3_30b_a3b": (
        "qwen3_30b_a3b_moe",
        "qwen3_30b_131072x128",
        "qwen3_30b_16384x128",
        "qwen3_30b_4096x2048",
    ),
    "qwen3_5": ("qwen35", "qwen3_5_tp2", "qwen3_5_tp4", "qwen3_5_tp8"),
    "qwen3_next_80b_a3b": ("qwen3_next",),
    "qwen3_vl": ("qwen3vl_vit",),
    "resnet152": ("resnet_152",),
    "resnet50": ("resnet_50",),
    "resnext101_32x8d": ("resnext_101_32x8d",),
    "resnext50_32x4d": ("resnext_50_32x4d",),
    "sd15_vae": ("sd1_5_vae",),
    "sdxl_unet": ("sdxl_cross", "sdxl_self_16x16", "sdxl_self_32x32"),
    "synthetic": (
        "d64_serving_gqa8",
        "gfx950_dense_test",
        "synthetic_resnet_scale_shapes",
    ),
    "vit_b_16": ("vit_b_16_edge",),
    "wan2_2_a14b": ("wan22_a14b",),
}

_MODEL_CANONICAL = {
    alias: canonical
    for canonical, aliases in MODEL_ALIASES.items()
    for alias in aliases
}


class MigrationError(Exception):
    """An input or a result that the migration must not accept."""


# ---------------------------------------------------------------------------
# Keys and tags


def geometry_key(graph: dict[str, Any]) -> str:
    """Return the store key: sha256 of the graph without ``name``/``metadata``."""
    body = {k: v for k, v in graph.items() if k not in ("name", "metadata")}
    canonical = json.dumps(
        body, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest() + ".json"


def model_token(value: str) -> str:
    """Return the canonical ``model`` value of one spelling."""
    token = re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")
    return _MODEL_CANONICAL.get(token, token)


def _tag_values(key: str, value: Any) -> list[str]:
    """Normalize one tag value to a list of lowercase strings."""
    items = value if isinstance(value, list) else [value]
    out = []
    for item in items:
        if item is None:
            continue
        if isinstance(item, bool):
            text = "true" if item else "false"
        elif isinstance(item, (int, float, str)):
            text = item.strip().lower() if isinstance(item, str) else json.dumps(item)
        else:
            raise MigrationError(f"tag {key!r} has a non-scalar value {item!r}")
        if key == "model":
            text = model_token(text)
        out.append(VALUE_ALIASES.get(key, {}).get(text, text))
    return out


def derive_operation(graph: dict[str, Any]) -> tuple[str, str | None]:
    """Return the one ``operation`` of a graph and the direction its nodes give."""
    found: dict[str, set[str | None]] = collections.defaultdict(set)
    for node in graph.get("nodes") or []:
        kind = node.get("type")
        if kind not in NODE_TYPES:
            raise MigrationError(f"unknown node type {kind!r}")
        operation, direction = NODE_TYPES[kind]
        found[operation].add(direction)
    primary = set(found) - AUXILIARY_OPERATIONS
    candidates = primary or set(found)
    if len(candidates) != 1:
        raise MigrationError(f"no single operation in {sorted(found)}")
    (operation,) = candidates
    directions = found[operation] - {None}
    return operation, directions.pop() if len(directions) == 1 else None


def _union(values: Iterable[Any]) -> list[Any]:
    """Unique values in JSON-text order."""
    unique = {json.dumps(v, sort_keys=True): v for v in values}
    return [unique[text] for text in sorted(unique)]


@dataclasses.dataclass
class Copy:
    """One graph file of one set, converted to store tags."""

    set_name: str
    member: str
    key: str
    graph: dict[str, Any]
    tags: dict[str, list[str]]
    notes: dict[str, list[Any]]
    changes: collections.Counter


def convert(set_name: str, member: str, graph: dict[str, Any]) -> Copy:
    """Convert one graph file to its key, tags and notes (design step 2)."""
    if not isinstance(graph, dict):
        raise MigrationError("the graph is not a JSON object")
    if set_name not in SETS:
        raise MigrationError(f"set {set_name!r} has no workload/source mapping")
    mapping = SETS[set_name]
    metadata = graph.get("metadata") or {}
    if not isinstance(metadata, dict):
        raise MigrationError("metadata is not an object")
    tags: dict[str, list[str]] = {}
    notes: dict[str, list[Any]] = {}
    changes: collections.Counter = collections.Counter()
    for key, value in metadata.items():
        if key == "notes" and isinstance(value, dict):
            for note_key, note_value in value.items():
                notes.setdefault(note_key, []).extend(
                    note_value if isinstance(note_value, list) else [note_value]
                )
        elif key in CLOSED_TAGS or key in OPEN_TAGS:
            tags[key] = _tag_values(key, value)
        else:
            notes.setdefault(key, []).extend(
                value if isinstance(value, list) else [value]
            )
            changes[f"{key}: moved to notes"] += 1

    operation, node_direction = derive_operation(graph)
    if "operation" in tags:
        tagged = set(tags["operation"])
        if (tagged - AUXILIARY_OPERATIONS or tagged) != {operation}:
            raise MigrationError(
                f"operation tag {metadata['operation']} disagrees with the nodes"
                f" ({operation})"
            )
    tags["operation"] = [operation]

    dtype = DTYPES.get(graph.get("io_data_type"))
    if dtype is None:
        raise MigrationError(f"unknown io_data_type {graph.get('io_data_type')!r}")
    if tags.get("dtype", [dtype]) != [dtype]:
        raise MigrationError(f"dtype tag {metadata['dtype']} disagrees with {dtype}")
    tags["dtype"] = [dtype]

    if "direction" not in tags and node_direction is not None:
        tags["direction"] = [node_direction]

    tags["workload"] = [mapping.workload]
    tags["source"] = [mapping.source]
    tags = {k: sorted(set(v)) for k, v in tags.items() if v}
    for key, values in tags.items():
        if key not in metadata:
            changes[f"{key}: added"] += 1
        elif metadata[key] != values:
            changes[f"{key}: changed"] += 1
        else:
            changes[f"{key}: kept"] += 1
    return Copy(set_name, member, geometry_key(graph), graph, tags, notes, changes)


# ---------------------------------------------------------------------------
# Merge and validation


def _priority(copy: Copy) -> tuple[int, str, str]:
    top = copy.set_name.split("/", 1)[0]
    return NAME_PRIORITY.get(top, len(NAME_PRIORITY)), copy.set_name, copy.member


def merge(copies: list[Copy]) -> dict[str, dict[str, Any]]:
    """Merge the copies of each key into one store object (design step 3)."""
    by_key: dict[str, list[Copy]] = collections.defaultdict(list)
    for copy in copies:
        by_key[copy.key].append(copy)
    objects = {}
    for key, group in by_key.items():
        group.sort(key=_priority)
        tags: dict[str, set[str]] = collections.defaultdict(set)
        notes: dict[str, list[Any]] = collections.defaultdict(list)
        for copy in group:
            for tag, values in copy.tags.items():
                tags[tag].update(values)
            for note, values in copy.notes.items():
                notes[note].extend(values)
        metadata: dict[str, Any] = {k: sorted(v) for k, v in tags.items()}
        metadata["notes"] = {k: _union(v) for k, v in notes.items()}
        metadata["notes"]["sets"] = sorted({c.set_name for c in group})
        head = group[0].graph
        body = {k: v for k, v in head.items() if k not in ("name", "metadata")}
        objects[key] = {**body, "name": head.get("name"), "metadata": metadata}
    return objects


def validate(key: str, obj: dict[str, Any]) -> list[str]:
    """Return the problems of one store object (design step 4)."""
    problems = []
    if not KEY.match(key) or geometry_key(obj) != key:
        problems.append("key is not the geometry hash")
    if not isinstance(obj.get("name"), str) or not obj["name"].strip():
        problems.append("empty name")
    if not obj.get("nodes"):
        problems.append("no nodes")
    metadata = obj.get("metadata")
    if not isinstance(metadata, dict) or not isinstance(metadata.get("notes"), dict):
        return problems + ["metadata or metadata.notes is not an object"]
    for tag, values in metadata.items():
        if tag == "notes":
            continue
        if tag not in CLOSED_TAGS and tag not in OPEN_TAGS:
            problems.append(f"tag {tag!r} is not in the vocabulary")
            continue
        if (
            not isinstance(values, list)
            or not values
            or values != sorted(set(values))
            or not all(isinstance(v, str) and TOKEN.match(v) for v in values)
        ):
            problems.append(f"tag {tag!r} is not a sorted unique token array")
            continue
        unknown = set(values) - CLOSED_TAGS.get(tag, set(values))
        if unknown:
            problems.append(f"tag {tag!r} has values outside the vocabulary {unknown}")
    if not metadata.get("source"):
        problems.append("no source tag")
    if len(metadata.get("operation", [])) != 1:
        problems.append("operation does not have exactly one value")
    return problems


def matches(obj: dict[str, Any], query: dict[str, tuple[str, ...]]) -> bool:
    """AND across keys, OR within a key."""
    metadata = obj["metadata"]
    return all(set(metadata.get(k, ())) & set(v) for k, v in query.items())


def check_queries(
    objects: dict[str, dict[str, Any]], set_keys: dict[str, set[str]]
) -> dict[str, tuple[int, int, int]]:
    """Return ``set -> (graphs, extra, missing)`` for every conversion query."""
    result = {}
    for set_name, keys in set_keys.items():
        selected = {
            k for k, obj in objects.items() if matches(obj, SETS[set_name].query)
        }
        result[set_name] = (len(keys), len(selected - keys), len(keys - selected))
    return result


# ---------------------------------------------------------------------------
# Inputs


@dataclasses.dataclass(frozen=True)
class Archive:
    set_name: str
    md5: str
    size: int | None
    path: Path | None = None


def _set_name(pointer: str) -> str:
    name = pointer.removeprefix("Workloads/")
    for suffix in ("/files.tar.gz.dvc", ".tar.gz.dvc"):
        if name.endswith(suffix):
            return name.removesuffix(suffix)
    raise MigrationError(f"unexpected pointer path {pointer}")


def read_pointers(repo: Path, ref: str) -> dict[str, Archive]:
    """Read the ``Workloads/`` DVC pointers at a git ref."""

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
        ).stdout

    archives = {}
    for path in git("ls-tree", "-r", "--name-only", ref, "--", "Workloads").split():
        if not path.endswith(".tar.gz.dvc"):
            continue
        text = git("show", f"{ref}:{path}")
        md5 = re.search(r"^\s*-?\s*md5:\s*([0-9a-f]{32})\s*$", text, re.M)
        size = re.search(r"^\s*size:\s*(\d+)\s*$", text, re.M)
        if not md5 or not size:
            raise MigrationError(f"{path}: no md5 or size")
        name = _set_name(path)
        archives[name] = Archive(name, md5.group(1), int(size.group(1)))
    return archives


def parse_archive_option(text: str) -> Archive:
    """Parse ``SET=PATH@MD5``."""
    match = re.fullmatch(r"([^=]+)=(.+)@([0-9a-f]{32})", text)
    if not match:
        raise argparse.ArgumentTypeError(f"expected SET=PATH@MD5, got {text!r}")
    return Archive(match.group(1), match.group(3), None, Path(match.group(2)))


def _https_get(url: str, timeout: float) -> bytes:
    """Return the body of an HTTPS GET. Reject every other URL scheme."""
    if urllib.parse.urlparse(url).scheme != "https":
        raise MigrationError(f"only https URLs are allowed, got {url!r}")
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read()


def fetch(archive: Archive, cache: Path) -> Path:
    """Return a local copy of an archive whose md5 and size are checked."""
    path = archive.path or cache / archive.md5
    if archive.path is None and not path.exists():
        url = f"{DVC_REMOTE_URL}/files/md5/{archive.md5[:2]}/{archive.md5[2:]}"
        partial = path.with_suffix(".part")
        partial.write_bytes(_https_get(url, timeout=120))
        partial.replace(path)
    data = path.read_bytes()
    # md5 is DVC's integrity checksum here, not a security control.
    if hashlib.md5(data, usedforsecurity=False).hexdigest() != archive.md5:
        raise MigrationError(f"{archive.set_name}: md5 mismatch for {path}")
    if archive.size is not None and len(data) != archive.size:
        raise MigrationError(f"{archive.set_name}: size mismatch for {path}")
    return path


def read_archive(path: Path) -> Iterator[tuple[str, Any]]:
    """Yield ``(member, graph)`` for every ``.json`` file in a tarball."""
    with tarfile.open(path) as tar:
        for member in sorted(tar.getmembers(), key=lambda m: m.name):
            if member.isfile() and member.name.endswith(".json"):
                yield member.name, json.load(tar.extractfile(member))


# ---------------------------------------------------------------------------
# Output and upload


def object_bytes(obj: dict[str, Any]) -> bytes:
    return (json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def sample(keys: list[str], fraction: float) -> list[str]:
    count = max(1, round(len(keys) * fraction))
    return random.Random(0).sample(sorted(keys), min(count, len(keys)))


def write_dry_run(objects: dict[str, dict[str, Any]], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise MigrationError(f"{out} is not empty")
    for key, obj in objects.items():
        (out / key).write_bytes(object_bytes(obj))


def verify_dry_run(objects: dict[str, Any], out: Path, fraction: float) -> int:
    files = {p.name for p in out.iterdir()}
    if files != set(objects):
        raise MigrationError(f"{out} has {len(files)} files, expected {len(objects)}")
    checked = sample(list(files), fraction)
    for key in checked:
        if geometry_key(json.loads((out / key).read_bytes())) != key:
            raise MigrationError(f"{out / key}: key is not the geometry hash")
    return len(checked)


def _split_s3(url: str) -> tuple[str, str]:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "s3" or not parsed.netloc or not url.endswith("/"):
        raise argparse.ArgumentTypeError(f"expected s3://BUCKET/PREFIX/, got {url!r}")
    return parsed.netloc, parsed.path.lstrip("/")


def upload(objects: dict[str, dict[str, Any]], url: str) -> None:
    """PUT every object with ``If-None-Match: *`` (design step 6)."""
    import boto3  # Only the upload needs it.

    bucket, prefix = _split_s3(url)
    client = boto3.client("s3")

    def put(key: str) -> str | None:
        try:
            client.put_object(
                Bucket=bucket,
                Key=prefix + key,
                Body=object_bytes(objects[key]),
                ContentType="application/json",
                IfNoneMatch="*",
            )
        except client.exceptions.ClientError as error:
            return f"{key}: {error}"
        return None

    with concurrent.futures.ThreadPoolExecutor(PARALLEL) as pool:
        failures = [f for f in pool.map(put, sorted(objects)) if f]
    if failures:
        raise MigrationError(f"{len(failures)} uploads failed, first: {failures[0]}")


def _https(bucket: str) -> str:
    return f"https://{bucket}.s3.amazonaws.com"


def list_prefix(url: str) -> list[str]:
    """List the keys under an S3 prefix anonymously (ListObjectsV2)."""
    bucket, prefix = _split_s3(url)
    namespace = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
    keys, token = [], None
    while True:
        params = {"list-type": "2", "prefix": prefix}
        if token:
            params["continuation-token"] = token
        query = urllib.parse.urlencode(params)
        body = _https_get(f"{_https(bucket)}/?{query}", timeout=60)
        root = ElementTree.fromstring(body)
        keys += [
            e.text.removeprefix(prefix)
            for e in root.findall("s3:Contents/s3:Key", namespace)
        ]
        token = root.findtext("s3:NextContinuationToken", namespaces=namespace)
        if not token:
            return keys


def verify_upload(objects: dict[str, Any], url: str, fraction: float) -> int:
    """Check the key count and re-hash a sample of the uploaded objects (step 7)."""
    bucket, prefix = _split_s3(url)
    keys = set(list_prefix(url))
    if keys != set(objects):
        raise MigrationError(f"{url} has {len(keys)} keys, expected {len(objects)}")
    checked = sample(list(keys), fraction)
    for key in checked:
        body = _https_get(f"{_https(bucket)}/{prefix}{key}", timeout=60)
        if geometry_key(json.loads(body)) != key:
            raise MigrationError(f"{url}{key}: key is not the geometry hash")
    return len(checked)


# ---------------------------------------------------------------------------
# Report


def report(
    copies: list[Copy],
    objects: dict[str, dict[str, Any]],
    queries: dict[str, tuple[int, int, int]],
) -> str:
    """Return the dry-run measurements as Markdown."""
    lines = []
    files = collections.Counter(c.set_name for c in copies)
    set_keys: dict[str, set[str]] = collections.defaultdict(set)
    key_sets: dict[str, set[str]] = collections.defaultdict(set)
    for c in copies:
        set_keys[c.set_name].add(c.key)
        key_sets[c.key].add(c.set_name)
    copies_per_key = collections.Counter(c.key for c in copies)
    multi = [k for k, n in copies_per_key.items() if n > 1]
    lines += [
        f"- Sets: {len(files)}",
        f"- Graph files: {len(copies)}",
        f"- Objects (unique keys): {len(objects)}",
        f"- Keys with more than one copy: {len(multi)} "
        f"({sum(copies_per_key[k] - 1 for k in multi)} redundant copies, "
        f"at most {max(copies_per_key.values(), default=0)} copies of one key)",
        f"- Size of all objects: {sum(len(object_bytes(o)) for o in objects.values())} bytes",
        "",
        "| Set | Files | Keys | Same-set duplicates | Query | Extra | Missing |",
        "|---|---|---|---|---|---|---|",
    ]
    for name in sorted(files):
        query = ", ".join(f"{k}={'|'.join(v)}" for k, v in SETS[name].query.items())
        graphs, extra, missing = queries[name]
        lines.append(
            f"| {name} | {files[name]} | {graphs} | {files[name] - graphs} "
            f"| `{query}` | {extra} | {missing} |"
        )

    pairs = collections.Counter()
    for sets in key_sets.values():
        ordered = sorted(sets)
        for i, a in enumerate(ordered):
            for b in ordered[i + 1 :]:
                pairs[(a, b)] += 1
    lines += ["", "Cross-set pairs with shared keys:", "", "| Set A | Set B | Shared |"]
    lines.append("|---|---|---|")
    for (a, b), n in sorted(pairs.items(), key=lambda p: (-p[1], p[0])):
        lines.append(f"| {a} | {b} | {n} |")

    by_name: dict[str, set[str]] = collections.defaultdict(set)
    for c in copies:
        by_name[c.key].add(c.graph.get("name"))
    renamed = sum(1 for names in by_name.values() if len(names) > 1)
    winners = collections.Counter(
        min(key_sets[k], key=lambda s: (NAME_PRIORITY[s.split("/")[0]], s)).split("/")[
            0
        ]
        for k in objects
    )
    lines += [
        "",
        f"- Keys whose copies have different names: {renamed}",
        "- Name taken from: "
        + ", ".join(f"{t} {n}" for t, n in sorted(winners.items())),
    ]

    changes = collections.Counter()
    for c in copies:
        changes.update(c.changes)
    lines += ["", "Changes to the input tags (per graph file):", ""]
    lines += [f"- {k}: {n}" for k, n in sorted(changes.items())]

    tag_counts: dict[str, collections.Counter] = collections.defaultdict(
        collections.Counter
    )
    multi_valued = collections.Counter()
    notes_keys = collections.Counter()
    for obj in objects.values():
        for tag, values in obj["metadata"].items():
            if tag == "notes":
                notes_keys.update(values.keys())
                continue
            tag_counts[tag].update(values)
            if len(values) > 1:
                multi_valued[tag] += 1
    lines += ["", "Tags on the objects (objects per value):", ""]
    for tag in sorted(tag_counts):
        values = tag_counts[tag]
        shown = ", ".join(f"{v} {n}" for v, n in values.most_common(12))
        more = f", ... ({len(values)} values)" if len(values) > 12 else ""
        lines.append(
            f"- `{tag}` on {sum(1 for o in objects.values() if tag in o['metadata'])}"
            f" objects, {multi_valued[tag]} with more than one value: {shown}{more}"
        )
    lines += ["", "Keys in `metadata.notes` (objects): "]
    lines.append(", ".join(f"`{k}` {n}" for k, n in sorted(notes_keys.items())))
    return "\n".join(lines)


# ---------------------------------------------------------------------------


def run(args: argparse.Namespace, cache: Path) -> int:
    archives = read_pointers(args.repo, args.ref)
    for archive in args.archive:
        archives[archive.set_name] = archive
    unmapped = sorted(set(archives) - set(SETS))
    absent = sorted(set(SETS) - set(archives))
    if unmapped or absent:
        raise MigrationError(
            f"sets without a mapping: {unmapped}; absent sets: {absent}"
        )

    with concurrent.futures.ThreadPoolExecutor(PARALLEL) as pool:
        paths = dict(
            zip(archives, pool.map(lambda a: fetch(a, cache), archives.values()))
        )
    print(f"Fetched and checked {len(paths)} archives", file=sys.stderr)

    copies, failures = [], []
    for set_name in sorted(paths):
        for member, graph in read_archive(paths[set_name]):
            try:
                copies.append(convert(set_name, member, graph))
            except MigrationError as error:
                failures.append(f"{set_name}:{member}: {error}")
    objects = merge(copies)
    for key in sorted(objects):
        failures += [f"{key}: {p}" for p in validate(key, objects[key])]
    set_keys: dict[str, set[str]] = collections.defaultdict(set)
    for c in copies:
        set_keys[c.set_name].add(c.key)
    queries = check_queries(objects, set_keys)
    failures += [
        f"{name}: conversion query gives {extra} extra and {missing} missing graphs"
        for name, (_, extra, missing) in sorted(queries.items())
        if extra or missing
    ]
    print(report(copies, objects, queries))
    if failures:
        for failure in failures:
            print(f"error: {failure}", file=sys.stderr)
        print(f"{len(failures)} validation failures; nothing written", file=sys.stderr)
        return 1

    write_dry_run(objects, args.out)
    checked = verify_dry_run(objects, args.out, args.sample)
    print(
        f"Wrote {len(objects)} objects to {args.out}; re-hashed {checked}",
        file=sys.stderr,
    )
    if args.upload:
        upload(objects, args.upload)
        checked = verify_upload(objects, args.upload, args.sample)
        print(
            f"Uploaded {len(objects)} objects to {args.upload}; re-hashed {checked}",
            file=sys.stderr,
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--ref", required=True, help="git ref whose Workloads/ pointers are read"
    )
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="git repository that has --ref (default: this repository)",
    )
    parser.add_argument(
        "--archive",
        type=parse_archive_option,
        action="append",
        default=[],
        metavar="SET=PATH@MD5",
        help="use a local archive for a set, checked against its md5",
    )
    parser.add_argument(
        "--out", type=Path, required=True, help="empty dry-run folder for the objects"
    )
    parser.add_argument(
        "--cache", type=Path, help="keep the downloaded archives here (default: temp)"
    )
    parser.add_argument(
        "--sample", type=float, default=0.01, help="fraction of objects to re-hash"
    )
    parser.add_argument(
        "--upload",
        metavar="s3://BUCKET/PREFIX/",
        help="after the dry run, upload the objects (needs boto3 and credentials)",
    )
    args = parser.parse_args(argv)
    if args.upload:
        try:
            _split_s3(args.upload)
        except argparse.ArgumentTypeError as error:
            parser.error(str(error))
    try:
        if args.cache:
            args.cache.mkdir(parents=True, exist_ok=True)
            return run(args, args.cache)
        with tempfile.TemporaryDirectory() as tmp:
            return run(args, Path(tmp))
    except MigrationError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

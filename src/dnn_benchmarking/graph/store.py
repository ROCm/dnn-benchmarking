# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""The graph store: one JSON object per graph, selected by tags.

``docs/graph-store.md`` gives the rules. This module holds them as code: the
object key, the tag vocabulary, the ``--tag`` query, and the anonymous,
read-only copy of an S3 prefix into a local cache. It uses only the standard
library, and it never writes to S3.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ElementTree
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, FrozenSet, Iterable, List, Mapping, Tuple

from ..common.exceptions import GraphLoadError

#: Scheme of a store URL: ``s3://BUCKET/PREFIX/``.
STORE_SCHEME = "s3://"

#: An object key: the geometry hash plus ``.json``.
KEY = re.compile(r"^[0-9a-f]{64}\.json$")

#: One tag value: a lowercase token.
TOKEN = re.compile(r"^[a-z0-9][a-z0-9._+-]{0,63}$")

#: Tag keys with a closed list of values.
CLOSED_TAGS: Dict[str, FrozenSet[str]] = {
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
OPEN_TAGS: FrozenSet[str] = frozenset(
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

#: Graph ``io_data_type`` -> ``dtype`` tag.
DTYPES: Dict[str, str] = {
    "float": "float32",
    "half": "float16",
    "bfloat16": "bfloat16",
    "fp8_e4m3": "fp8_e4m3",
}

#: Parallel downloads of a store sync.
PARALLEL = 16

_TIMEOUT_S = 60
#: ETag of each cached object. Not ``*.json``: ``-g <cache folder>`` must
#: see only graphs.
_INDEX = ".etags"
_S3_NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}

#: A ``--tag`` query: key -> accepted values. AND across keys, OR within one.
TagQuery = Mapping[str, FrozenSet[str]]


# ---------------------------------------------------------------------------
# Keys and tags


def geometry_key(graph: Mapping[str, Any]) -> str:
    """Return the object key: sha256 of the graph without ``name``/``metadata``.

    The canonical JSON has sorted keys, no spaces and ASCII escapes. NaN and
    infinity are not valid JSON, so they raise ``ValueError``.
    """
    body = {k: v for k, v in graph.items() if k not in ("name", "metadata")}
    canonical = json.dumps(
        body, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest() + ".json"


def object_bytes(obj: Mapping[str, Any]) -> bytes:
    """The file content of a store object: indented JSON, sorted keys."""
    return (json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def validate_object(key: str, obj: Mapping[str, Any]) -> List[str]:
    """Return the problems of one store object against the store rules.

    Checks the key, a non-empty name, the nodes, and that every tag is a
    sorted array of unique tokens in the vocabulary, with one ``operation``.
    """
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
    if len(metadata.get("operation", [])) != 1:
        problems.append("operation does not have exactly one value")
    return problems


def parse_tag(text: str) -> Tuple[str, str]:
    """Parse one ``KEY=VALUE`` tag term into lowercase ``(key, value)``.

    Raises:
        ValueError: The text is not ``KEY=VALUE`` with a non-empty key and value.
    """
    key, sep, value = text.partition("=")
    key, value = key.strip().lower(), value.strip().lower()
    if not sep or not key or not value:
        raise ValueError(f"expected KEY=VALUE, got {text!r}")
    return key, value


def tag_query(terms: Iterable[str]) -> Dict[str, FrozenSet[str]]:
    """Group ``KEY=VALUE`` terms by key: AND across keys, OR within a key."""
    query: Dict[str, set[str]] = {}
    for term in terms:
        key, value = parse_tag(term)
        query.setdefault(key, set()).add(value)
    return {key: frozenset(values) for key, values in query.items()}


def tag_values(metadata: Any, key: str) -> FrozenSet[str]:
    """The lowercase values of one tag in a graph's ``metadata``.

    Store objects hold arrays of tokens. Other graph files can hold one
    scalar; booleans read as ``true``/``false``.
    """
    if not isinstance(metadata, dict) or key not in metadata:
        return frozenset()
    value = metadata[key]
    items = value if isinstance(value, list) else [value]
    out = set()
    for item in items:
        if isinstance(item, bool):
            out.add("true" if item else "false")
        elif isinstance(item, (str, int, float)):
            out.add(str(item).strip().lower())
    return frozenset(out)


def matches(metadata: Any, query: TagQuery) -> bool:
    """Whether a graph's ``metadata`` satisfies every key of ``query``."""
    return all(tag_values(metadata, key) & values for key, values in query.items())


def select_by_tags(paths: Iterable[str], query: TagQuery) -> List[str]:
    """Keep the graph files whose ``metadata`` satisfies ``query``.

    A file that is not a readable JSON object is kept, so the graph loader
    reports it as a graph error instead of a silent drop.
    """
    selected = []
    for path in paths:
        try:
            with open(path, encoding="utf-8") as f:
                graph = json.load(f)
        except (OSError, ValueError):
            selected.append(path)
            continue
        if not isinstance(graph, dict) or matches(graph.get("metadata"), query):
            selected.append(path)
    return selected


# ---------------------------------------------------------------------------
# S3 store source


def is_store_url(text: str) -> bool:
    return text.startswith(STORE_SCHEME)


def split_store_url(url: str) -> Tuple[str, str]:
    """Return ``(bucket, prefix)`` of ``s3://BUCKET/PREFIX/``.

    A missing final ``/`` is added, so ``s3://b/graphs`` does not also list
    ``s3://b/graphs-old/``.

    Raises:
        ValueError: The URL has no bucket.
    """
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(f"expected s3://BUCKET/PREFIX/, got {url!r}")
    prefix = parsed.path.lstrip("/")
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    return parsed.netloc, prefix


def bucket_url(bucket: str) -> str:
    """The anonymous HTTPS endpoint of a bucket."""
    return f"https://{bucket}.s3.amazonaws.com"


def https_get(url: str, timeout: float = _TIMEOUT_S) -> bytes:
    """Return the body of an anonymous HTTPS GET. Reject every other scheme."""
    if urllib.parse.urlparse(url).scheme != "https":
        raise ValueError(f"only https URLs are allowed, got {url!r}")
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read()


def list_objects(bucket: str, prefix: str) -> Dict[str, str]:
    """List every object under ``prefix`` (ListObjectsV2, all pages).

    Returns:
        Object name relative to ``prefix`` -> ETag.
    """
    objects: Dict[str, str] = {}
    token = None
    while True:
        params = {"list-type": "2", "prefix": prefix}
        if token:
            params["continuation-token"] = token
        body = https_get(f"{bucket_url(bucket)}/?{urllib.parse.urlencode(params)}")
        root = ElementTree.fromstring(body)
        for item in root.findall("s3:Contents", _S3_NS):
            key = item.findtext("s3:Key", default="", namespaces=_S3_NS)
            etag = item.findtext("s3:ETag", default="", namespaces=_S3_NS)
            objects[key[len(prefix) :]] = etag
        token = root.findtext("s3:NextContinuationToken", namespaces=_S3_NS)
        if not token:
            return objects


@dataclass
class SyncResult:
    """What :func:`sync` did."""

    #: Cached file of every valid store object, sorted.
    paths: List[Path]
    #: Objects fetched in this sync (new or changed ETag).
    downloaded: int = 0
    #: Cached objects removed because the store no longer lists them.
    removed: int = 0
    #: Keys whose content does not hash to the key, or is not a JSON object.
    rejected: List[str] = field(default_factory=list)
    #: Names under the prefix that are not object keys.
    ignored: List[str] = field(default_factory=list)


def store_cache_dir(cache_root: Path, url: str) -> Path:
    """The cache folder of one store URL: ``<cache_root>/<bucket>/<prefix>``."""
    bucket, prefix = split_store_url(url)
    return cache_root.joinpath(bucket, *[p for p in prefix.split("/") if p])


def sync(url: str, cache_root: Path) -> SyncResult:
    """Copy a store prefix into a local cache; return the cached objects.

    Lists the prefix, then downloads only the objects whose ETag is new or
    changed since the last sync. Each download must hash to its key. Cached
    objects that the store no longer lists are removed.

    Raises:
        GraphLoadError: The listing or a download fails.
    """
    bucket, prefix = split_store_url(url)
    cache = store_cache_dir(cache_root, url)
    try:
        listing = list_objects(bucket, prefix)
    except (OSError, ValueError, ElementTree.ParseError) as e:
        raise GraphLoadError(f"Cannot list {url}: {e}") from e
    keys = {name: etag for name, etag in listing.items() if KEY.match(name)}
    result = SyncResult(paths=[], ignored=sorted(set(listing) - set(keys)))

    cache.mkdir(parents=True, exist_ok=True)
    index = _read_index(cache / _INDEX)
    stale = sorted(
        key
        for key, etag in keys.items()
        if index.get(key) != etag or not (cache / key).is_file()
    )

    def fetch(key: str) -> bool:
        target = cache / key
        target.unlink(missing_ok=True)
        try:
            body = https_get(f"{bucket_url(bucket)}/{urllib.parse.quote(prefix + key)}")
        except (OSError, ValueError) as e:
            raise GraphLoadError(f"Cannot download {url}{key}: {e}") from e
        try:
            graph = json.loads(body)
            valid = isinstance(graph, dict) and geometry_key(graph) == key
        except ValueError:
            valid = False
        if valid:
            _write_atomic(target, body)
        return valid

    with concurrent.futures.ThreadPoolExecutor(PARALLEL) as pool:
        for key, valid in zip(stale, pool.map(fetch, stale)):
            if valid:
                index[key] = keys[key]
                result.downloaded += 1
            else:
                index.pop(key, None)
                result.rejected.append(key)

    for cached in cache.iterdir():
        if KEY.match(cached.name) and cached.name not in keys:
            cached.unlink()
            index.pop(cached.name, None)
            result.removed += 1
    index = {key: etag for key, etag in index.items() if key in keys}
    _write_atomic(cache / _INDEX, json.dumps(index, sort_keys=True).encode())

    rejected = set(result.rejected)
    result.paths = [cache / key for key in sorted(keys) if key not in rejected]
    return result


def _read_index(path: Path) -> Dict[str, str]:
    try:
        index = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return index if isinstance(index, dict) else {}


def _write_atomic(path: Path, data: bytes) -> None:
    partial = path.with_name(f".{path.name}.{os.getpid()}.part")
    partial.write_bytes(data)
    os.replace(partial, path)

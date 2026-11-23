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
import http.client
import json
import os
import re
import ssl
import threading
import time
import urllib.parse
import xml.etree.ElementTree as ElementTree
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, FrozenSet, Iterable, List, Mapping, Optional, Tuple

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

#: Parallel downloads of a store sync; each worker keeps one connection.
PARALLEL = 16

#: Attempts of one GET on a transient error, and the first backoff.
ATTEMPTS = 6
_BACKOFF_S = 0.5
_BACKOFF_MAX_S = 8.0
#: HTTP statuses that a retry can clear.
_TRANSIENT_STATUSES = frozenset({429, 500, 502, 503, 504})

_TIMEOUT_S = 60
#: ETag of each cached object. Not ``*.json``: ``-g <cache folder>`` must
#: see only graphs.
_INDEX = ".etags"
#: Downloads between two writes of the index, so a stopped sync resumes.
_INDEX_FLUSH = 500
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


class HTTPStatusError(OSError):
    """A GET answered with a status other than 200."""

    def __init__(self, url: str, status: int) -> None:
        super().__init__(f"HTTP {status} for {url}")
        self.status = status


_local = threading.local()


def https_get(url: str, timeout: float = _TIMEOUT_S) -> bytes:
    """Return the body of one anonymous HTTPS GET. Reject every other scheme.

    Each thread keeps one keep-alive connection per host, so a sync does not
    open a TLS connection for each object.

    Raises:
        ValueError: The URL is not ``https``.
        HTTPStatusError: The status is not 200.
        OSError, http.client.HTTPException: The connection failed.
    """
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https":
        raise ValueError(f"only https URLs are allowed, got {url!r}")
    connections = _local.__dict__.setdefault("connections", {})
    connection = connections.get(parsed.netloc)
    if connection is None:
        connection = http.client.HTTPSConnection(
            parsed.netloc, timeout=timeout, context=ssl.create_default_context()
        )
        connections[parsed.netloc] = connection
    target = parsed.path or "/"
    if parsed.query:
        target += f"?{parsed.query}"
    try:
        connection.request("GET", target)
        response = connection.getresponse()
        body = response.read()
    except (OSError, http.client.HTTPException):
        connection.close()
        del connections[parsed.netloc]
        raise
    if response.status != 200:
        raise HTTPStatusError(url, response.status)
    return body


def _transient(error: Exception) -> bool:
    """Whether a retry can clear the error of a GET."""
    if isinstance(error, HTTPStatusError):
        return error.status in _TRANSIENT_STATUSES
    return isinstance(error, (OSError, http.client.HTTPException))


def _sleep(seconds: float) -> None:
    time.sleep(seconds)


def get_with_retry(url: str) -> bytes:
    """:func:`https_get`, retried on transient errors with exponential backoff.

    Connection resets, timeouts, 429 and 5xx are retried up to
    :data:`ATTEMPTS` times in total. Other errors (403, 404, ...) are not.
    """
    delay = _BACKOFF_S
    for attempt in range(1, ATTEMPTS + 1):
        try:
            return https_get(url)
        except (OSError, http.client.HTTPException) as e:
            if attempt == ATTEMPTS or not _transient(e):
                raise
        _sleep(delay)
        delay = min(delay * 2, _BACKOFF_MAX_S)
    raise AssertionError("unreachable")


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
        body = get_with_retry(f"{bucket_url(bucket)}/?{urllib.parse.urlencode(params)}")
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
    changed since the last sync. Each download must hash to its key.
    Transient errors are retried (:func:`get_with_retry`). The ETag index is
    written as downloads land and when the sync stops, so a failed sync
    keeps its progress and the next one fetches only what is missing.
    Cached objects that the store no longer lists are removed.

    Raises:
        GraphLoadError: The listing fails, or a download fails after its
            retries.
    """
    bucket, prefix = split_store_url(url)
    cache = store_cache_dir(cache_root, url)
    try:
        listing = list_objects(bucket, prefix)
    except (OSError, http.client.HTTPException, ElementTree.ParseError) as e:
        raise GraphLoadError(f"Cannot list {url}: {e}") from e
    keys = {name: etag for name, etag in listing.items() if KEY.match(name)}
    result = SyncResult(paths=[], ignored=sorted(set(listing) - set(keys)))

    cache.mkdir(parents=True, exist_ok=True)
    index_path = cache / _INDEX
    index = _read_index(index_path)
    stale = sorted(
        key
        for key, etag in keys.items()
        if index.get(key) != etag or not (cache / key).is_file()
    )
    for key in stale:
        index.pop(key, None)

    def fetch(key: str) -> bool:
        target = cache / key
        target.unlink(missing_ok=True)
        try:
            body = get_with_retry(
                f"{bucket_url(bucket)}/{urllib.parse.quote(prefix + key)}"
            )
        except (OSError, http.client.HTTPException) as e:
            raise GraphLoadError(f"Cannot download {url}{key}: {e}") from e
        try:
            graph = json.loads(body)
            valid = isinstance(graph, dict) and geometry_key(graph) == key
        except ValueError:
            valid = False
        if valid:
            _write_atomic(target, body)
        return valid

    error: Optional[GraphLoadError] = None
    pool = concurrent.futures.ThreadPoolExecutor(PARALLEL)
    try:
        futures = {pool.submit(fetch, key): key for key in stale}
        for future in concurrent.futures.as_completed(futures):
            key = futures[future]
            try:
                valid = future.result()
            except concurrent.futures.CancelledError:
                continue
            except GraphLoadError as e:
                if error is None:
                    error = e
                    for pending in futures:
                        pending.cancel()
                continue
            if valid:
                index[key] = keys[key]
                result.downloaded += 1
                if result.downloaded % _INDEX_FLUSH == 0:
                    _write_index(index_path, index, keys)
            else:
                result.rejected.append(key)
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
        _write_index(index_path, index, keys)
    if error is not None:
        raise error

    for cached in cache.iterdir():
        if KEY.match(cached.name) and cached.name not in keys:
            cached.unlink()
            result.removed += 1
    _write_index(index_path, index, keys)

    result.rejected.sort()
    rejected = set(result.rejected)
    result.paths = [cache / key for key in sorted(keys) if key not in rejected]
    return result


def _write_index(path: Path, index: Dict[str, str], keys: Mapping[str, str]) -> None:
    """Write the ETags of the listed keys that are cached."""
    current = {key: etag for key, etag in index.items() if key in keys}
    _write_atomic(path, json.dumps(current, sort_keys=True).encode())


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

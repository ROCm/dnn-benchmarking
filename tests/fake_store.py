# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""A fake S3 graph store for unit tests: replaces ``graph.store.https_get``."""

import hashlib
import json
import urllib.parse
from typing import Optional
from xml.sax.saxutils import escape

from dnn_benchmarking.graph import store

URL = "s3://bucket/graphs/"


def graph(n: int, **tags: str) -> dict:
    """A small graph; ``tag="a|b"`` gives ``metadata.tag = ["a", "b"]``."""
    return {
        "name": f"g{n}",
        "nodes": [{"type": "MatmulAttributes", "n": n}],
        "metadata": {k: v.split("|") for k, v in tags.items()},
    }


class FakeBucket:
    """Serves ListObjectsV2 pages and object GETs for ``URL``.

    ``faults[key]`` lists exceptions that the next GETs of ``key`` raise, in
    order, before the object is served.
    """

    def __init__(self, page_size: int = 2) -> None:
        self.objects: dict[str, bytes] = {}
        self.page_size = page_size
        self.gets: list[str] = []
        self.faults: dict[str, list[Exception]] = {}

    def put(self, graph: dict, key: Optional[str] = None) -> str:
        key = key or store.geometry_key(graph)
        self.objects[f"graphs/{key}"] = json.dumps(graph).encode()
        return key

    def __call__(self, url: str, timeout: float = 0) -> bytes:
        parsed = urllib.parse.urlparse(url)
        assert parsed.scheme == "https"
        assert parsed.netloc == "bucket.s3.amazonaws.com"
        if parsed.path == "/":
            return self._page(urllib.parse.parse_qs(parsed.query))
        name = urllib.parse.unquote(parsed.path[1:])
        key = name.rsplit("/", 1)[-1]
        self.gets.append(key)
        if self.faults.get(key):
            raise self.faults[key].pop(0)
        if name not in self.objects:
            raise store.HTTPStatusError(url, 404)
        return self.objects[name]

    def _page(self, query: dict) -> bytes:
        prefix = query["prefix"][0]
        names = sorted(n for n in self.objects if n.startswith(prefix))
        start = int(query.get("continuation-token", ["0"])[0])
        page = names[start : start + self.page_size]
        items = "".join(
            f"<Contents><Key>{escape(n)}</Key><ETag>&quot;"
            f"{hashlib.md5(self.objects[n], usedforsecurity=False).hexdigest()}"
            "&quot;</ETag></Contents>"
            for n in page
        )
        token = ""
        if start + self.page_size < len(names):
            token = (
                f"<NextContinuationToken>{start + self.page_size}"
                "</NextContinuationToken>"
            )
        return (
            '<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
            f"{items}{token}</ListBucketResult>"
        ).encode()

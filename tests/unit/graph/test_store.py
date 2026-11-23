# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Unit tests for the graph store: keys, the --tag query, and the S3 copy.

A fake bucket replaces HTTPS, so no test reaches the network.
"""

import json
import urllib.parse
from pathlib import Path

import pytest

from dnn_benchmarking.common.exceptions import GraphLoadError
from dnn_benchmarking.graph import store
from tests.fake_store import URL, FakeBucket, graph as _graph


@pytest.fixture
def bucket(monkeypatch) -> FakeBucket:
    fake = FakeBucket()
    monkeypatch.setattr(store, "https_get", fake)
    monkeypatch.setattr(store, "_sleep", lambda seconds: None)
    return fake


# ---------------------------------------------------------------------------
# Keys


def test_key_ignores_name_and_metadata_but_not_the_geometry() -> None:
    base = _graph(1, workload="headline")
    renamed = {**base, "name": "other", "metadata": {"workload": ["corpus"]}}
    changed = _graph(2, workload="headline")

    assert store.geometry_key(base) == store.geometry_key(renamed)
    assert store.geometry_key(base) != store.geometry_key(changed)
    assert store.KEY.match(store.geometry_key(base))


def test_key_rejects_nan() -> None:
    with pytest.raises(ValueError):
        store.geometry_key({"nodes": [{"alpha": float("nan")}]})


# ---------------------------------------------------------------------------
# --tag query


def test_query_is_and_across_keys_and_or_within_a_key() -> None:
    query = store.tag_query(["operation=moe", "operation=matmul", "workload=headline"])

    assert store.matches({"operation": ["matmul"], "workload": ["headline"]}, query)
    assert store.matches({"operation": ["moe"], "workload": ["headline"]}, query)
    assert not store.matches({"operation": ["matmul"], "workload": ["corpus"]}, query)
    assert not store.matches({"operation": ["matmul"]}, query)
    assert not store.matches(None, query)


def test_query_and_values_compare_lowercase_and_accept_scalars() -> None:
    query = store.tag_query(["DType=BFloat16", "sink=false"])

    assert store.matches({"dtype": "bfloat16", "sink": False}, query)
    assert store.matches({"dtype": ["BFLOAT16", "float16"], "sink": ["false"]}, query)


@pytest.mark.parametrize("term", ["operation", "=sdpa", "operation=", " = "])
def test_a_term_without_key_and_value_is_rejected(term: str) -> None:
    with pytest.raises(ValueError):
        store.parse_tag(term)


def test_select_keeps_matches_and_unreadable_files(tmp_path: Path) -> None:
    conv = tmp_path / "conv.json"
    conv.write_text(json.dumps(_graph(1, operation="convolution")))
    sdpa = tmp_path / "sdpa.json"
    sdpa.write_text(json.dumps(_graph(2, operation="sdpa")))
    broken = tmp_path / "broken.json"
    broken.write_text("{not json")

    selected = store.select_by_tags(
        [str(conv), str(sdpa), str(broken)], store.tag_query(["operation=sdpa"])
    )

    assert selected == [str(sdpa), str(broken)]


# ---------------------------------------------------------------------------
# Store URLs and sync


def test_store_url_gets_a_final_slash_and_needs_a_bucket() -> None:
    assert store.split_store_url("s3://b/graphs") == ("b", "graphs/")
    assert store.split_store_url("s3://b/") == ("b", "")
    with pytest.raises(ValueError):
        store.split_store_url("s3:///graphs/")


def test_https_get_rejects_other_schemes() -> None:
    with pytest.raises(ValueError):
        store.https_get("file:///etc/passwd")


def test_first_sync_lists_every_page_and_downloads_every_object(
    bucket: FakeBucket, tmp_path: Path
) -> None:
    keys = [bucket.put(_graph(n)) for n in range(5)]

    result = store.sync(URL, tmp_path)

    assert result.downloaded == 5
    assert [p.name for p in result.paths] == sorted(keys)
    for path in result.paths:
        assert store.geometry_key(json.loads(path.read_text())) == path.name


def test_second_sync_downloads_only_new_or_changed_objects(
    bucket: FakeBucket, tmp_path: Path
) -> None:
    first = bucket.put(_graph(1, workload="headline"))
    bucket.put(_graph(2))
    store.sync(URL, tmp_path)
    bucket.gets.clear()

    bucket.put(_graph(1, workload="corpus"))  # retag: same key, new ETag
    added = bucket.put(_graph(3))
    result = store.sync(URL, tmp_path)

    assert sorted(bucket.gets) == sorted([first, added])
    assert result.downloaded == 2
    retagged = json.loads((store.store_cache_dir(tmp_path, URL) / first).read_text())
    assert retagged["metadata"]["workload"] == ["corpus"]

    bucket.gets.clear()
    assert store.sync(URL, tmp_path).downloaded == 0
    assert bucket.gets == []


def test_objects_gone_from_the_store_leave_the_cache(
    bucket: FakeBucket, tmp_path: Path
) -> None:
    kept = bucket.put(_graph(1))
    gone = bucket.put(_graph(2))
    store.sync(URL, tmp_path)

    del bucket.objects[f"graphs/{gone}"]
    result = store.sync(URL, tmp_path)

    assert result.removed == 1
    assert [p.name for p in result.paths] == [kept]
    assert not (store.store_cache_dir(tmp_path, URL) / gone).exists()


def test_an_object_that_does_not_hash_to_its_key_is_skipped_and_not_cached(
    bucket: FakeBucket, tmp_path: Path
) -> None:
    good = bucket.put(_graph(1))
    wrong = bucket.put(_graph(2), key="0" * 64 + ".json")

    result = store.sync(URL, tmp_path)

    assert result.rejected == [wrong]
    assert [p.name for p in result.paths] == [good]
    assert not (store.store_cache_dir(tmp_path, URL) / wrong).exists()
    bucket.gets.clear()
    store.sync(URL, tmp_path)
    assert bucket.gets == [wrong]  # retried, never trusted


def test_names_that_are_not_keys_are_ignored(
    bucket: FakeBucket, tmp_path: Path
) -> None:
    key = bucket.put(_graph(1))
    bucket.objects["graphs/README.md"] = b"text"
    bucket.objects["graphs/old/" + key] = bucket.objects[f"graphs/{key}"]

    result = store.sync(URL, tmp_path)

    assert result.ignored == ["README.md", f"old/{key}"]
    assert [p.name for p in result.paths] == [key]


def test_transient_errors_are_retried_until_the_object_arrives(
    bucket: FakeBucket, tmp_path: Path
) -> None:
    flaky = bucket.put(_graph(1))
    bucket.put(_graph(2))
    bucket.faults[flaky] = [
        ConnectionResetError(104, "Connection reset by peer"),
        TimeoutError("timed out"),
        store.HTTPStatusError("https://x", 503),
        store.HTTPStatusError("https://x", 429),
    ]

    result = store.sync(URL, tmp_path)

    assert result.downloaded == 2
    assert bucket.gets.count(flaky) == 5


def test_a_transient_error_that_persists_stops_the_sync_after_the_retries(
    bucket: FakeBucket, tmp_path: Path
) -> None:
    key = bucket.put(_graph(1))
    bucket.faults[key] = [ConnectionResetError(104, "reset")] * store.ATTEMPTS

    with pytest.raises(GraphLoadError, match="Cannot download"):
        store.sync(URL, tmp_path)
    assert bucket.gets.count(key) == store.ATTEMPTS


@pytest.mark.parametrize("status", [403, 404])
def test_a_permanent_error_is_not_retried(
    bucket: FakeBucket, tmp_path: Path, status: int
) -> None:
    key = bucket.put(_graph(1))
    bucket.faults[key] = [store.HTTPStatusError("https://x", status)]

    with pytest.raises(GraphLoadError, match=f"HTTP {status}"):
        store.sync(URL, tmp_path)
    assert bucket.gets.count(key) == 1


def test_a_failed_sync_keeps_its_progress_and_the_next_one_resumes(
    bucket: FakeBucket, tmp_path: Path
) -> None:
    keys = [bucket.put(_graph(n)) for n in range(8)]
    broken = keys[3]
    bucket.faults[broken] = [store.HTTPStatusError("https://x", 403)]
    with pytest.raises(GraphLoadError):
        store.sync(URL, tmp_path)
    first = set(bucket.gets)
    bucket.gets.clear()

    result = store.sync(URL, tmp_path)

    # Only the broken object and the ones the failed sync did not reach.
    assert broken in bucket.gets
    assert set(bucket.gets) == {broken} | (set(keys) - first)
    assert result.downloaded == len(bucket.gets)
    assert [p.name for p in result.paths] == sorted(keys)


def test_an_unreachable_store_is_a_graph_load_error(
    tmp_path: Path, monkeypatch
) -> None:
    calls = []

    def offline(url: str, timeout: float = 0) -> bytes:
        calls.append(url)
        raise OSError("network is unreachable")

    monkeypatch.setattr(store, "https_get", offline)
    monkeypatch.setattr(store, "_sleep", lambda seconds: None)
    with pytest.raises(GraphLoadError, match="Cannot list"):
        store.sync(URL, tmp_path)
    assert len(calls) == store.ATTEMPTS

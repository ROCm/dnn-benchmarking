# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""--graph store URLs and --tag: what the CLI runs, and its exit codes.

The suite runner is replaced, so the tests see the graph list it would get.
"""

import importlib
import json
from pathlib import Path

import pytest

from dnn_benchmarking.graph import store
from tests.fake_store import URL, FakeBucket, graph

main_module = importlib.import_module("dnn_benchmarking.cli.main")


@pytest.fixture
def ran(monkeypatch) -> list:
    """Graph names that run_suite_cli receives, one list per call."""
    calls: list = []

    def fake_run_suite_cli(args, graph_paths, **_kwargs) -> int:
        calls.append(sorted(json.loads(p.read_text())["name"] for p in graph_paths))
        return 0

    suite_runner_cli = importlib.import_module("dnn_benchmarking.cli.suite_runner_cli")
    monkeypatch.setattr(suite_runner_cli, "run_suite_cli", fake_run_suite_cli)
    return calls


@pytest.fixture
def bucket(monkeypatch) -> FakeBucket:
    fake = FakeBucket()
    monkeypatch.setattr(store, "https_get", fake)
    return fake


def _write(folder: Path, *graphs: dict) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    for g in graphs:
        (folder / store.geometry_key(g)).write_text(json.dumps(g))
    return folder


def test_tags_select_from_a_local_store_folder(tmp_path: Path, ran: list) -> None:
    folder = _write(
        tmp_path / "store",
        graph(1, workload="headline", operation="convolution"),
        graph(2, workload="headline", operation="sdpa"),
        graph(3, workload="corpus", operation="convolution"),
        graph(4, workload="headline", operation="matmul"),
    )

    code = main_module.main(
        [
            "-g",
            str(folder),
            "--tag",
            "workload=headline",
            "--tag",
            "operation=convolution",
            "--tag",
            "operation=matmul",
        ]
    )

    assert code == 0
    assert ran == [["g1", "g4"]]


def test_a_store_url_runs_its_objects_and_tags_select_among_them(
    tmp_path: Path, ran: list, bucket: FakeBucket
) -> None:
    for n, workload in enumerate(["headline", "corpus", "headline"]):
        bucket.put(graph(n, workload=workload))
    work = tmp_path / "work"

    code = main_module.main(
        ["-g", URL, "--tag", "workload=headline", "--benchmark-work-dir", str(work)]
    )

    assert code == 0
    assert ran == [["g0", "g2"]]
    assert len(list((work / "graph-store" / "bucket" / "graphs").glob("*.json"))) == 3


def test_tags_from_the_config_file_apply_and_store_urls_stay_urls(
    tmp_path: Path, ran: list, bucket: FakeBucket
) -> None:
    bucket.put(graph(1, operation="sdpa"))
    bucket.put(graph(2, operation="convolution"))
    config = tmp_path / "bench.toml"
    config.write_text(f'version = 1\ngraphs = ["{URL}"]\ntags = ["Operation=SDPA"]\n')

    code = main_module.main(
        ["--config", str(config), "--benchmark-work-dir", str(tmp_path / "w")]
    )

    assert code == 0
    assert ran == [["g1"]]


def test_no_graph_matching_the_tags_is_exit_1(tmp_path: Path, ran: list) -> None:
    folder = _write(tmp_path / "store", graph(1, operation="convolution"))

    assert main_module.main(["-g", str(folder), "--tag", "operation=sdpa"]) == 1
    assert ran == []


def test_an_unreachable_store_is_exit_1(tmp_path: Path, ran: list, monkeypatch) -> None:
    def offline(url: str, timeout: float = 0) -> bytes:
        raise OSError("network is unreachable")

    monkeypatch.setattr(store, "https_get", offline)

    assert main_module.main(["-g", URL]) == 1
    assert ran == []


@pytest.mark.parametrize(
    "argv",
    [
        ["-g", "graphs/sample_conv_fwd.json", "--tag", "operation"],
        ["-g", "s3:///no-bucket/"],
    ],
)
def test_a_malformed_tag_or_store_url_is_a_usage_error(argv: list, ran: list) -> None:
    with pytest.raises(SystemExit) as exc:
        main_module.main(argv)

    assert exc.value.code == 2
    assert ran == []

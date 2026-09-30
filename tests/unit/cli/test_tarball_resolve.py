# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Unit tests for graph file resolution and tarball extraction."""

import io
import json
import tarfile
from pathlib import Path

import pytest

from dnn_benchmarking.common.exceptions import GraphLoadError
from dnn_benchmarking.graph.resolver import (
    extract_tarball,
    is_tarball,
    resolve_graph_files,
    resolve_graph_files_multi,
)


def _make_tarball(dest: Path, members: dict) -> Path:
    """Create a real tarball at dest containing the given {name: content} members."""
    with tarfile.open(str(dest), "w:gz") as tf:
        for name, content in members.items():
            data = content.encode() if isinstance(content, str) else content
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return dest


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("graphs.tar", True),
        ("graphs.tar.gz", True),
        ("graphs.tgz", True),
        ("graphs.tar.bz2", True),
        ("graphs.tar.xz", True),
        ("GRAPHS.TAR.GZ", True),
        ("graph.json", False),
        ("shapes.txt", False),
    ],
)
def test_is_tarball(name: str, expected: bool) -> None:
    assert is_tarball(name) is expected


class TestExtractTarball:
    """Tests for extract_tarball()."""

    def test_extracts_json_files(self, tmp_path: Path) -> None:
        graph = json.dumps({"name": "g", "nodes": [], "tensors": []})
        tb = _make_tarball(tmp_path / "graphs.tar.gz", {"graph.json": graph})

        tmpdir, extracted = extract_tarball(str(tb))
        try:
            assert len(extracted) == 1
            assert extracted[0].endswith(".json")
        finally:
            tmpdir.cleanup()

    def test_nonexistent_path_raises_graph_load_error(self) -> None:
        with pytest.raises(GraphLoadError, match="not found"):
            extract_tarball("/nonexistent/path/graphs.tar.gz")

    def test_not_a_tarball_raises_graph_load_error(self, tmp_path: Path) -> None:
        bad_file = tmp_path / "fake.tar.gz"
        bad_file.write_text("this is not a tarball")
        with pytest.raises(GraphLoadError, match="Not a valid tarball"):
            extract_tarball(str(bad_file))


class TestResolveGraphFiles:
    """Tests for resolve_graph_files()."""

    def test_single_json_file(self, tmp_path: Path) -> None:
        f = tmp_path / "graph.json"
        f.write_text(json.dumps({"name": "g", "nodes": [], "tensors": []}))

        tmpdirs, files, tarball_source = resolve_graph_files(str(f))
        assert tmpdirs == []
        assert len(files) == 1
        assert files[0] == str(f)
        assert tarball_source is None

    def test_glob_matches_multiple_json_files(self, tmp_path: Path) -> None:
        for i in range(3):
            (tmp_path / f"g{i}.json").write_text(
                json.dumps({"name": f"g{i}", "nodes": [], "tensors": []})
            )
        pattern = str(tmp_path / "*.json")

        tmpdirs, files, _ = resolve_graph_files(pattern)
        assert len(files) == 3
        assert all(f.endswith(".json") for f in files)
        for td in tmpdirs:
            td.cleanup()

    def test_glob_collects_json_and_tarballs(self, tmp_path: Path) -> None:
        (tmp_path / "loose.json").write_text(
            json.dumps({"name": "loose", "nodes": [], "tensors": []})
        )
        graph = json.dumps({"name": "packed", "nodes": [], "tensors": []})
        _make_tarball(tmp_path / "packed.tar.gz", {"packed.json": graph})

        pattern = str(tmp_path / "*")
        tmpdirs, files, tarball_source = resolve_graph_files(pattern)
        try:
            assert len(files) == 2
            assert any("loose.json" in f for f in files)
            assert any("packed.json" in f for f in files)
            assert tarball_source == pattern
        finally:
            for td in tmpdirs:
                td.cleanup()

    def test_directory_collects_json_recursively(self, tmp_path: Path) -> None:
        subdir = tmp_path / "sub"
        subdir.mkdir()
        for name, parent in [("a.json", tmp_path), ("b.json", subdir)]:
            (parent / name).write_text(
                json.dumps({"name": name, "nodes": [], "tensors": []})
            )
        (tmp_path / "readme.txt").write_text("not a graph")

        tmpdirs, files, tarball_source = resolve_graph_files(str(tmp_path))
        assert len(files) == 2
        assert all(f.endswith(".json") for f in files)
        assert tarball_source is None
        assert tmpdirs == []

    def test_directory_collects_tarballs_and_json(self, tmp_path: Path) -> None:
        (tmp_path / "loose.json").write_text(
            json.dumps({"name": "loose", "nodes": [], "tensors": []})
        )
        graph = json.dumps({"name": "packed", "nodes": [], "tensors": []})
        _make_tarball(tmp_path / "packed.tar.gz", {"packed.json": graph})

        tmpdirs, files, tarball_source = resolve_graph_files(str(tmp_path))
        try:
            assert len(files) == 2
            assert any("loose.json" in f for f in files)
            assert any("packed.json" in f for f in files)
            assert tarball_source == str(tmp_path)
        finally:
            for td in tmpdirs:
                td.cleanup()

    def test_empty_directory_returns_no_files(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()

        tmpdirs, files, tarball_source = resolve_graph_files(str(empty))
        assert files == []
        assert tmpdirs == []

    def test_tarball_path_extracts_and_returns_json(self, tmp_path: Path) -> None:
        graph = json.dumps({"name": "g", "nodes": [], "tensors": []})
        tb = _make_tarball(tmp_path / "graphs.tar.gz", {"graph.json": graph})

        tmpdirs, files, tarball_source = resolve_graph_files(str(tb))
        try:
            assert len(files) == 1
            assert files[0].endswith(".json")
            assert tarball_source == str(tb)
        finally:
            for td in tmpdirs:
                td.cleanup()


class TestResolveGraphFilesMulti:
    """Tests for resolve_graph_files_multi()."""

    def test_single_json_arg(self, tmp_path: Path) -> None:
        f = tmp_path / "graph.json"
        f.write_text(json.dumps({"name": "g", "nodes": [], "tensors": []}))

        tmpdirs, files, tarball_source = resolve_graph_files_multi([str(f)])
        assert len(files) == 1
        assert files[0] == str(f)
        assert tarball_source is None
        for td in tmpdirs:
            td.cleanup()

    def test_keeps_command_line_order_and_dedupes_by_resolved_path(
        self, tmp_path: Path
    ) -> None:
        for name in ("g0", "g1", "g2"):
            (tmp_path / f"{name}.json").write_text(
                json.dumps({"name": name, "nodes": [], "tensors": []})
            )
        g2 = str(tmp_path / "g2.json")
        g1_alias = str(tmp_path / "sub" / ".." / "g1.json")
        (tmp_path / "sub").mkdir()

        tmpdirs, files, _ = resolve_graph_files_multi(
            [g2, str(tmp_path / "g*.json"), g1_alias]
        )
        # g2 first (as given), then the glob's sorted remainder; the alias of
        # g1 is a duplicate and is dropped.
        assert [Path(f).name for f in files] == ["g2.json", "g0.json", "g1.json"]
        assert files[0] == g2
        assert tmpdirs == []

    def test_mixed_json_and_tarball(self, tmp_path: Path) -> None:
        loose = tmp_path / "loose.json"
        loose.write_text(json.dumps({"name": "loose", "nodes": [], "tensors": []}))
        graph = json.dumps({"name": "packed", "nodes": [], "tensors": []})
        tb = _make_tarball(tmp_path / "packed.tar.gz", {"packed.json": graph})

        tmpdirs, files, tarball_source = resolve_graph_files_multi(
            [str(loose), str(tb)]
        )
        try:
            assert len(files) == 2
            assert any("loose.json" in f for f in files)
            assert any("packed.json" in f for f in files)
            assert tarball_source == str(tb)
        finally:
            for td in tmpdirs:
                td.cleanup()

    def test_empty_args_returns_empty(self) -> None:
        tmpdirs, files, tarball_source = resolve_graph_files_multi([])
        assert files == []
        assert tmpdirs == []
        assert tarball_source is None

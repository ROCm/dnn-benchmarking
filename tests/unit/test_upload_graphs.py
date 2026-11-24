# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

import json
import runpy
import subprocess
from pathlib import Path


_REPO = Path(__file__).resolve().parents[2]


def test_merge_preserves_metadata_and_stops_on_stale_etag(
    tmp_path, monkeypatch, capsys
):
    graph = json.loads((_REPO / "graphs/sample_sdpa.json").read_text())
    graph["metadata"] = {
        "operation": ["sdpa"],
        "dtype": ["bfloat16"],
        "workload": ["microbench"],
        "notes": {"source": "stored"},
    }
    local = {
        **graph,
        "name": "local name",
        "metadata": {
            **graph["metadata"],
            "workload": ["headline"],
            "notes": {"source": "local"},
        },
    }
    graph_file = tmp_path / "local.json"
    graph_file.write_text(json.dumps(local))
    puts = []

    def aws(command, **kwargs):
        if "get-object" in command:
            Path(command[-1]).write_text(json.dumps(graph))
            return subprocess.CompletedProcess(command, 0, '{"ETag":"\\"old\\""}', "")
        assert "put-object" in command
        puts.append(command)
        assert command[command.index("--if-match") + 1] == '"old"'
        body = Path(command[command.index("--body") + 1])
        assert json.loads(body.read_text()) == {
            **graph,
            "metadata": {**graph["metadata"], "workload": ["headline", "microbench"]},
        }
        return subprocess.CompletedProcess(command, 254, "", "(PreconditionFailed)")

    monkeypatch.setattr(subprocess, "run", aws)
    cli = runpy.run_path(str(_REPO / "tools/upload_graphs.py"))["main"]
    args = [str(graph_file), "--profile", "test"]
    assert cli(args) == 0
    assert capsys.readouterr().out.startswith("would merge ")
    assert not puts
    assert cli([*args, "--apply"]) == 1
    assert len(puts) == 1
    assert "S3 object changed" in capsys.readouterr().err

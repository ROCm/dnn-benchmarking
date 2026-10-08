# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""End-to-end smoke of the dnn-benchmark CLI in a subprocess."""

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import List

import pytest

from tests.conftest import expected_timer
from tests.integration.conftest import GRAPHS_DIR, PROJECT_ROOT


@pytest.mark.gpu
def test_cli_smoke(
    hipdnn, torch_gpu, plugin_path_cli_args: List[str], tmp_path: Path
) -> None:
    """One good graph with --validate plus one broken graph.

    Pins the user-facing contract: results on stdout, progress on stderr,
    per-graph isolation, exit code 1 for a graph error, and a v2 JSON file.
    """
    conv = GRAPHS_DIR / "sample_conv_fwd.json"
    broken = tmp_path / "broken.json"
    broken.write_text("{")
    output = tmp_path / "results.json"

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "dnn_benchmarking",
            "--graph",
            str(conv),
            str(broken),
            "--warmup",
            "1",
            "--iters",
            "3",
            "--validate",
            "pytorch",
            "-o",
            str(output),
            *plugin_path_cli_args,
        ],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
        timeout=600,
    )
    assert proc.returncode == 1, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"

    # Progress lines go to stderr only; the results table and summary to stdout.
    assert re.search(r"^\[1/2\] ", proc.stderr, re.MULTILINE)
    assert re.search(r"^\[2/2\] ", proc.stderr, re.MULTILINE)
    assert not re.search(r"^\[\d/2\] ", proc.stdout, re.MULTILINE)
    assert "sample_conv_fwd_16x16x16x16_k16_3x3" in proc.stdout
    assert re.search(r"^Summary: 2 graph\(s\)", proc.stdout, re.MULTILINE)

    data = json.loads(output.read_text())
    assert data["schema_version"] == 2
    assert data["run"]["complete"] is True
    config = data["run"]["config"]
    assert (config["validate"], config["iters"], config["seed"]) == ("pytorch", 3, 0)
    assert data["summary"]["graphs"] == 2
    assert data["summary"]["graph_errors"] == 1
    assert data["summary"]["failed"] == 0

    graphs = {g["graph_path"]: g for g in data["graphs"]}
    bad = graphs[str(broken)]
    assert (bad["status"], bad["results"]) == ("error", [])
    assert bad["error"]

    good = graphs[str(conv)]
    assert good["status"] == "ok"
    reference = [r for r in good["results"] if r["role"] == "reference"]
    engines = [r for r in good["results"] if r["role"] == "engine"]
    assert [(r["runtime"], r["verdict"]) for r in reference] == [
        ("pytorch", "reference")
    ]
    assert engines
    for row in engines:
        assert row["verdict"] == "passed", row
        assert re.fullmatch(r"0x[0-9A-F]{16}", row["engine"]["id"])
        ootb = row["ootb"]
        assert ootb["correctness"]["match"] is True
        assert ootb["kernel"]["n"] == ootb["host"]["n"] == 3
        assert ootb["timing"]["warmup_iters"] == 1


@pytest.mark.gpu
def test_cli_pytorch_runtime(torch_gpu, tmp_path: Path) -> None:
    """--runtime pytorch reaches the PyTorch runner: one timed pytorch row."""
    output = tmp_path / "results.json"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "dnn_benchmarking",
            "--runtime",
            "pytorch",
            "-g",
            str(GRAPHS_DIR / "sample_relu.json"),
            "--warmup",
            "1",
            "--iters",
            "3",
            "-o",
            str(output),
        ],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
        timeout=600,
    )
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"

    [graph] = json.loads(output.read_text())["graphs"]
    [row] = graph["results"]
    assert (row["runtime"], row["role"], row["status"]) == (
        "pytorch",
        "engine",
        "success",
    ), row
    assert row["ootb"]["kernel"]["n"] == 3
    assert row["ootb"]["timing"]["timer"] == expected_timer()

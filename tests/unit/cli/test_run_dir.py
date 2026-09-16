# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""`--run-dir` writes one directory a reader can open without further picking."""

from pathlib import Path

from dnn_benchmarking.cli.main import _apply_run_dir
from dnn_benchmarking.cli.parser import create_parser


def _args(argv: list[str]):
    args = create_parser(suppress_defaults=True).parse_args(["-g", "graph.json", *argv])
    _apply_run_dir(args)
    return args


def test_run_dir_anchors_the_report_and_both_artifact_roots():
    args = _args(["--run-dir", "/tmp/run7"])

    assert args.output == Path("/tmp/run7/results.json")
    assert args.tensor_output_dir == Path("/tmp/run7/tensors")
    assert args.profiling_output_dir == Path("/tmp/run7/profiling-output")


def test_explicit_paths_win_over_the_run_dir():
    args = _args(
        [
            "--run-dir",
            "/tmp/run7",
            "--output",
            "/tmp/report.json",
            "--profiling-output-dir",
            "/tmp/traces",
        ]
    )

    assert args.output == Path("/tmp/report.json")
    assert args.profiling_output_dir == Path("/tmp/traces")
    # The one left unset still follows the run directory.
    assert args.tensor_output_dir == Path("/tmp/run7/tensors")


def test_without_run_dir_nothing_is_invented():
    args = _args([])

    assert getattr(args, "output", None) is None
    assert getattr(args, "tensor_output_dir", None) is None

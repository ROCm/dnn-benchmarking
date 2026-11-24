#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Preview or upload tagged graph JSON files to an S3 graph store."""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from dnn_benchmarking.graph import store

DEFAULT_STORE = "s3://therock-dvc/dnn-benchmarking-graphs/"


def validate(key: str, graph: dict) -> None:
    problems = store.validate_object(key, graph)
    if problems:
        raise ValueError("; ".join(problems))
    dtype = store.DTYPES.get(graph.get("io_data_type"))
    if dtype is None or graph["metadata"].get("dtype") != [dtype]:
        raise ValueError("metadata.dtype must match io_data_type")


def merge_tags(current: dict, incoming: dict) -> dict:
    metadata = current["metadata"].copy()
    for key, values in incoming["metadata"].items():
        if key != "notes":
            metadata[key] = sorted(set(metadata.get(key, [])) | set(values))
    return {**current, "metadata": metadata}


def read_graph(path: Path) -> dict:
    graph = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(graph, dict):
        raise ValueError("expected a graph JSON object")
    return graph


def load_inputs(paths: list[Path]) -> dict[str, dict]:
    graphs = {}
    for path in paths:
        files = sorted(path.glob("*.json")) if path.is_dir() else [path]
        if not files:
            raise ValueError(f"{path}: no graph JSON files")
        for file in files:
            try:
                graph = read_graph(file)
                key = store.geometry_key(graph)
                validate(key, graph)
                if key in graphs:
                    graph = merge_tags(graphs[key], graph)
                    validate(key, graph)
                graphs[key] = graph
            except (OSError, TypeError, ValueError) as exc:
                raise ValueError(f"{file}: {exc}") from exc
    return graphs


def upload_graph(
    common: list[str],
    object_args: list[str],
    key: str,
    incoming: dict,
    body: Path,
    apply: bool,
) -> str:
    get = subprocess.run(
        [*common, "get-object", *object_args, str(body)],
        capture_output=True,
        text=True,
    )
    if get.returncode == 0:
        current = read_graph(body)
        validate(key, current)
        graph = merge_tags(current, incoming)
        if graph == current:
            return "skip"
        etag = json.loads(get.stdout).get("ETag")
        if not etag:
            raise ValueError("S3 did not return an ETag for the existing graph")
        condition = ["--if-match", etag]
        action = "merge"
    elif "(NoSuchKey)" in get.stderr:
        graph = incoming
        condition = ["--if-none-match", "*"]
        action = "create"
    else:
        raise RuntimeError(
            get.stderr.strip() or f"aws get-object exited {get.returncode}"
        )
    validate(key, graph)
    payload = store.object_bytes(graph)
    if apply:
        body.write_bytes(payload)
        put = subprocess.run(
            [
                *common,
                "put-object",
                *object_args,
                "--body",
                str(body),
                "--content-type",
                "application/json",
                *condition,
            ],
            capture_output=True,
            text=True,
        )
        if put.returncode:
            if any(
                f"({code})" in put.stderr
                for code in ("PreconditionFailed", "ConditionalRequestConflict")
            ):
                raise RuntimeError(
                    "S3 object changed; rerun the preview before retrying"
                )
            raise RuntimeError(
                put.stderr.strip() or f"aws put-object exited {put.returncode}"
            )
    return action


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "graphs", nargs="+", type=Path, help="JSON files or directories"
    )
    parser.add_argument("--store", default=DEFAULT_STORE, help="destination S3 URL")
    parser.add_argument("--profile", help="AWS CLI profile (required with --apply)")
    parser.add_argument(
        "--apply", action="store_true", help="write instead of previewing"
    )
    args = parser.parse_args(argv)
    if args.apply and not args.profile:
        parser.error("--apply requires --profile")
    try:
        bucket, prefix = store.split_store_url(args.store)
        graphs = load_inputs(args.graphs)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    aws = ["aws", "--no-cli-pager", "--output", "json"]
    aws.extend(["--profile", args.profile] if args.profile else ["--no-sign-request"])
    aws.append("s3api")
    with TemporaryDirectory(prefix="dnn-upload-") as directory:
        body = Path(directory) / "graph.json"
        for key, graph in graphs.items():
            args_for_key = ["--bucket", bucket, "--key", prefix + key]
            try:
                action = upload_graph(aws, args_for_key, key, graph, body, args.apply)
            except (OSError, TypeError, ValueError, RuntimeError) as exc:
                print(f"{key}: {exc}", file=sys.stderr)
                return 1
            print(f"{action if args.apply else 'would ' + action} {key}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

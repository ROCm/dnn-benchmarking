#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Preview or upload tagged graph JSON files to an S3 graph store."""

import argparse
import json
import sys
from pathlib import Path

import boto3
from botocore import UNSIGNED
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

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


def parse_graph(data: bytes) -> dict:
    graph = json.loads(data)
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
                graph = parse_graph(file.read_bytes())
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
    s3, bucket: str, prefix: str, key: str, incoming: dict, apply: bool
) -> str:
    object_args = {"Bucket": bucket, "Key": prefix + key}
    try:
        response = s3.get_object(**object_args)
    except ClientError as exc:
        if exc.response["Error"]["Code"] != "NoSuchKey":
            raise
        graph = incoming
        condition = {"IfNoneMatch": "*"}
        action = "create"
    else:
        with response["Body"] as body:
            current = parse_graph(body.read())
        validate(key, current)
        graph = merge_tags(current, incoming)
        if graph == current:
            return "skip"
        etag = response.get("ETag")
        if not etag:
            raise ValueError("S3 did not return an ETag for the existing graph")
        condition = {"IfMatch": etag}
        action = "merge"
    validate(key, graph)
    payload = store.object_bytes(graph)
    if apply:
        try:
            s3.put_object(
                **object_args,
                Body=payload,
                ContentType="application/json",
                **condition,
            )
        except ClientError as exc:
            if exc.response["Error"]["Code"] in {
                "PreconditionFailed",
                "ConditionalRequestConflict",
            }:
                raise RuntimeError(
                    "S3 object changed; rerun the preview before retrying"
                ) from exc
            raise
    return action


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "graphs", nargs="+", type=Path, help="JSON files or directories"
    )
    parser.add_argument("--store", default=DEFAULT_STORE, help="destination S3 URL")
    parser.add_argument("--profile", help="override the AWS credential profile")
    parser.add_argument(
        "--apply", action="store_true", help="write instead of previewing"
    )
    args = parser.parse_args(argv)
    try:
        bucket, prefix = store.split_store_url(args.store)
        graphs = load_inputs(args.graphs)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    try:
        session = boto3.Session(profile_name=args.profile)
        config = (
            None if args.apply or args.profile else Config(signature_version=UNSIGNED)
        )
        s3 = session.client("s3", config=config)
    except (BotoCoreError, ClientError) as exc:
        print(exc, file=sys.stderr)
        return 1
    for key, graph in graphs.items():
        try:
            action = upload_graph(s3, bucket, prefix, key, graph, args.apply)
        except (
            BotoCoreError,
            ClientError,
            OSError,
            TypeError,
            ValueError,
            RuntimeError,
        ) as exc:
            print(f"{key}: {exc}", file=sys.stderr)
            return 1
        print(f"{action if args.apply else 'would ' + action} {key}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

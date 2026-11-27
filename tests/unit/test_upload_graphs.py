# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

import json
import runpy
from io import BytesIO
from pathlib import Path

import pytest

boto3 = pytest.importorskip("boto3")
from botocore.response import StreamingBody
from botocore.stub import Stubber

from dnn_benchmarking.graph import store


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
    s3 = boto3.client(
        "s3",
        region_name="us-east-1",
        aws_access_key_id="test",
        aws_secret_access_key="test",
    )
    monkeypatch.setattr(boto3.Session, "client", lambda self, *args, **kwargs: s3)
    key = store.geometry_key(graph)
    object_args = {"Bucket": "therock-dvc", "Key": f"dnn-benchmarking-graphs/{key}"}
    raw = store.object_bytes(graph)
    merged = {
        **graph,
        "metadata": {**graph["metadata"], "workload": ["headline", "microbench"]},
    }
    cli = runpy.run_path(str(_REPO / "tools/upload_graphs.py"))["main"]
    args = [str(graph_file)]
    with Stubber(s3) as stub:
        for _ in range(2):
            stub.add_response(
                "get_object",
                {"Body": StreamingBody(BytesIO(raw), len(raw)), "ETag": '"old"'},
                object_args,
            )
        stub.add_client_error(
            "put_object",
            service_error_code="PreconditionFailed",
            http_status_code=412,
            expected_params={
                **object_args,
                "IfMatch": '"old"',
                "Body": store.object_bytes(merged),
                "ContentType": "application/json",
            },
        )
        assert cli(args) == 0
        assert capsys.readouterr().out.startswith("would merge ")
        assert cli([*args, "--apply"]) == 1
        assert "S3 object changed" in capsys.readouterr().err
        stub.assert_no_pending_responses()

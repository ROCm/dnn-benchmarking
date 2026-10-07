# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Graph loading from JSON files."""

import json
from pathlib import Path
from typing import Any, Dict, List, Set

from ..common.exceptions import GraphLoadError
from .tensor_info import TensorInfo


def output_uids(graph_json: Dict[str, Any]) -> Set[int]:
    """UIDs every node writes, from int- or list-valued ``outputs`` entries."""
    uids: Set[int] = set()
    for node in graph_json.get("nodes") or []:
        for value in (node.get("outputs") or {}).values():
            for uid in value if isinstance(value, list) else [value]:
                if isinstance(uid, int) and not isinstance(uid, bool):
                    uids.add(uid)
    return uids


class GraphLoader:
    """Loads and parses hipDNN graph JSON files.

    Handles JSON loading, validation, and tensor info extraction.
    """

    def load_json(self, path: Path) -> Dict[str, Any]:
        """Load and parse a graph JSON file.

        Args:
            path: Path to the JSON file.

        Returns:
            Parsed JSON object.

        Raises:
            GraphLoadError: If the file cannot be read or parsed, or is not a
                JSON object.
        """
        if not path.exists():
            raise GraphLoadError(f"Graph file not found: {path}")

        try:
            with open(path, "r") as f:
                graph_json = json.load(f)
        except json.JSONDecodeError as e:
            raise GraphLoadError(f"Invalid JSON in graph file: {e}") from e
        except OSError as e:
            raise GraphLoadError(f"Cannot read graph file: {e}") from e
        if not isinstance(graph_json, dict):
            raise GraphLoadError(
                f"Graph file must contain a JSON object, got "
                f"{type(graph_json).__name__}: {path}"
            )
        return graph_json

    def validate(self, graph_json: Dict[str, Any]) -> None:
        """Check basic graph structure; operation-level checks are hipDNN's.

        Raises:
            GraphLoadError: If the graph has no operation nodes.
        """
        if not graph_json.get("nodes"):
            raise GraphLoadError("Graph contains no operation nodes")

    def extract_tensor_info(self, graph_json: Dict[str, Any]) -> List[TensorInfo]:
        """Extract tensor information from graph JSON.

        Args:
            graph_json: Parsed graph JSON dictionary.

        Returns:
            List of TensorInfo objects for all non-virtual tensors.

        Raises:
            GraphLoadError: If a tensor entry is malformed.
            UnsupportedGraphError: If a tensor has an unsupported data type.
        """
        outputs = output_uids(graph_json)
        result = []
        for tensor_json in graph_json.get("tensors", []):
            # Virtual tensors get no buffer, and hipDNN may leave their
            # data_type "unset" (filled from intermediate_data_type), so skip
            # them before resolving the dtype.
            if isinstance(tensor_json, dict):
                if tensor_json.get("virtual"):
                    continue
                # hipDNN fills an "unset" physical tensor from io_data_type
                # (TensorAttributes::fill_from_context); only both unset raises.
                if tensor_json.get("data_type") == "unset" and graph_json.get(
                    "io_data_type"
                ):
                    tensor_json = {
                        **tensor_json,
                        "data_type": graph_json["io_data_type"],
                    }
            tensor_info = TensorInfo.from_json(tensor_json)
            tensor_info.is_output = tensor_info.uid in outputs
            result.append(tensor_info)
        return result

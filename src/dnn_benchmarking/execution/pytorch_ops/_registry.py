# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Operation-handler registry and graph execution."""

from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Set, Tuple

import torch

from ...common.exceptions import UnsupportedGraphError


# Type aliases for operation planners.
# CompiledOp replays a planned op against a live tensor map.
CompiledOp = Callable[[Dict[int, torch.Tensor]], None]
# OpHandler plans an op once from its node + graph JSON, returning a CompiledOp.
OpHandler = Callable[[Dict[str, Any], Dict[str, Any]], CompiledOp]

# Registry of operation handlers
_OP_HANDLERS: Dict[str, OpHandler] = {}


def register_handler(op_type: str) -> Callable[[OpHandler], OpHandler]:
    """Decorator to register an operation handler.

    Args:
        op_type: The node type string to handle (e.g., "ConvolutionFwdAttributes").

    Returns:
        Decorator function.
    """

    def decorator(func: OpHandler) -> OpHandler:
        _OP_HANDLERS[op_type] = func
        return func

    return decorator


def get_handler(op_type: str) -> Optional[OpHandler]:
    """Get handler for operation type.

    Args:
        op_type: The node type string.

    Returns:
        Handler function or None if not found.
    """
    return _OP_HANDLERS.get(op_type)


def get_supported_operations() -> Set[str]:
    """Get set of supported operation types.

    Returns:
        Set of operation type strings that have handlers.
    """
    return set(_OP_HANDLERS.keys())


def supports_graph(graph_json: Dict[str, Any]) -> bool:
    """Check if all graph operations are supported.

    Args:
        graph_json: The graph as a parsed JSON dictionary.

    Returns:
        True if all node types have handlers.
    """
    for node in graph_json.get("nodes", []):
        if node.get("type") not in _OP_HANDLERS:
            return False
    return True


def get_unsupported_operations(graph_json: Dict[str, Any]) -> List[str]:
    """Get list of unsupported operation types in graph.

    Args:
        graph_json: The graph as a parsed JSON dictionary.

    Returns:
        List of unsupported operation type strings.
    """
    unsupported = []
    for node in graph_json.get("nodes", []):
        op_type = node.get("type")
        if op_type not in _OP_HANDLERS:
            unsupported.append(op_type)
    return unsupported


class CompiledGraph:
    """A graph compiled once into replayable per-op closures."""

    def __init__(
        self,
        ops: List[Tuple[str, CompiledOp]],
        releases: Optional[Sequence[Tuple[int, ...]]] = None,
    ) -> None:
        """Args:
        ops: ``(op_type, closure)`` in execution order.
        releases: For each op, the UIDs to drop from the tensor map after
            it runs (virtual tensors no later op reads). None drops nothing.
        """
        self._ops = ops
        self._releases = releases or [()] * len(ops)

    def execute(self, tensors: Dict[int, torch.Tensor]) -> None:
        """Replay every compiled op against the provided tensor map.

        Virtual tensors leave ``tensors`` after their last reader, so peak
        memory is the live intermediates, not every intermediate of the graph.

        Args:
            tensors: Mapping of tensor UID to torch.Tensor.

        Raises:
            UnsupportedGraphError: If an op fails at runtime.
        """
        for (op_type, run), release in zip(self._ops, self._releases):
            try:
                run(tensors)
            except UnsupportedGraphError:
                raise
            except ValueError as e:
                # A handler's ValueError signals an unsupported graph feature;
                # normalize to UnsupportedGraphError so callers skip it. Any
                # other exception is a real failure and propagates as an error.
                raise UnsupportedGraphError(
                    f"PyTorch reference could not execute {op_type!r} with the provided "
                    f"dtypes/parameters: {e}"
                ) from e
            for uid in release:
                tensors.pop(uid, None)


def _referenced_uids(value: Any, key: str = "") -> Iterator[int]:
    """Every tensor UID a node references, in any section, under a ``*uid*`` key."""
    if isinstance(value, dict):
        for child_key, child in value.items():
            yield from _referenced_uids(child, str(child_key))
    elif isinstance(value, list):
        for child in value:
            yield from _referenced_uids(child, key)
    elif "uid" in key and not isinstance(value, bool):
        try:
            yield int(value)
        except (TypeError, ValueError):
            pass


def _virtual_releases(graph_json: Dict[str, Any]) -> List[Tuple[int, ...]]:
    """For each node, the virtual tensor UIDs whose last reference it is."""
    virtual = {
        int(tensor["uid"])
        for tensor in graph_json.get("tensors", [])
        if isinstance(tensor, dict) and tensor.get("virtual") and "uid" in tensor
    }
    nodes = graph_json.get("nodes", [])
    last_use: Dict[int, int] = {}
    for index, node in enumerate(nodes):
        for uid in _referenced_uids(node):
            if uid in virtual:
                last_use[uid] = index
    releases: List[List[int]] = [[] for _ in nodes]
    for uid, index in last_use.items():
        releases[index].append(uid)
    return [tuple(uids) for uids in releases]


def compile_graph(graph_json: Dict[str, Any]) -> CompiledGraph:
    """Compile all graph operations into replayable closures once.

    Args:
        graph_json: The graph as a parsed JSON dictionary.

    Returns:
        A CompiledGraph that replays the planned ops on each execute().

    Raises:
        UnsupportedGraphError: If the graph contains an operation, attribute, or
            parameter the PyTorch reference does not support.
    """
    ops: List[Tuple[str, CompiledOp]] = []
    for node in graph_json.get("nodes", []):
        op_type = node.get("type")
        planner = _OP_HANDLERS.get(op_type)
        if planner is None:
            raise UnsupportedGraphError(f"Unsupported operation type: {op_type}")
        try:
            op = planner(node, graph_json)
        except UnsupportedGraphError:
            raise
        except ValueError as e:
            # A planner raises ValueError to signal a graph feature it cannot
            # represent; normalize to UnsupportedGraphError so callers skip it.
            # Any other exception is an unexpected failure and propagates.
            raise UnsupportedGraphError(
                f"PyTorch reference could not execute {op_type!r} with the provided "
                f"dtypes/parameters: {e}"
            ) from e
        ops.append((op_type, op))
    return CompiledGraph(ops, _virtual_releases(graph_json))


def execute_graph(
    graph_json: Dict[str, Any],
    tensors: Dict[int, torch.Tensor],
) -> None:
    """Compile and execute all graph operations in order.

    Args:
        graph_json: The graph as a parsed JSON dictionary.
        tensors: Mapping of tensor UID to torch.Tensor.

    Raises:
        UnsupportedGraphError: If the graph contains an operation, attribute, or
            parameter the PyTorch reference does not support.
    """
    compile_graph(graph_json).execute(tensors)

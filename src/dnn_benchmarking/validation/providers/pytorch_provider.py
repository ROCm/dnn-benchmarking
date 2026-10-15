# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""PyTorch reference provider for hipDNN graph validation.

Computes reference outputs by parsing graph JSON and executing
equivalent PyTorch operations on CPU.
"""

from math import prod
from typing import Any, Dict, List, Optional, Set

import numpy as np

from ...common import torch_support
from ...common.dtypes import get_dtype
from ...common.exceptions import UnsupportedGraphError
from ...config.benchmark_config import ReferenceProviderName
from ...graph.loader import output_uids, tensor_data_type

from ..reference_provider import (
    ReferenceOutput,
    ReferenceProvider,
    ReferenceProviderRegistry,
)


def _get_pytorch_ops():
    """Import PyTorch operation handlers only when the provider is used."""
    from ...execution import pytorch_ops

    return pytorch_ops


def _tensor_metadata(graph_json: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    return {
        int(tensor["uid"]): tensor
        for tensor in graph_json.get("tensors", [])
        if "uid" in tensor
    }


def _uid_tensor(
    inputs: Dict[str, Any], key: str, tensors: Dict[int, Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    try:
        return tensors.get(int(inputs[key]))
    except (KeyError, TypeError, ValueError):
        return None


def _dims(tensor_json: Optional[Dict[str, Any]]) -> List[int]:
    return [int(d) for d in (tensor_json or {}).get("dims", [])]


def _element_size(tensor_json: Dict[str, Any], graph_json: Dict[str, Any]) -> int:
    try:
        return get_dtype(tensor_data_type(tensor_json, graph_json)).size
    except (UnsupportedGraphError, KeyError, TypeError, ValueError):
        return 4


def _sdpa_scratch_bytes(
    node: Dict[str, Any], tensors: Dict[int, Dict[str, Any]]
) -> int:
    """float32 scratch of one SDPA node beyond its graph tensors.

    Per query block: scores/probabilities, d_scores and one reduction
    temporary. For the whole node: K and V expanded to the query heads, and
    the backward dK/dV accumulators.
    """
    from ...execution.pytorch_ops.handlers.sdpa import _QUERY_BLOCK

    inputs = node.get("inputs") or {}
    q = _dims(_uid_tensor(inputs, "q_tensor_uid", tensors))
    k = _dims(_uid_tensor(inputs, "k_tensor_uid", tensors))
    if len(q) < 3 or len(k) < 2:
        return 0
    rows = prod(q[:-3]) * q[-3]  # batch * query heads
    q_len, kv_len, head_dim = q[-2], k[-2], q[-1]
    block = rows * min(q_len, _QUERY_BLOCK) * kv_len * 4 * 3
    expanded = rows * kv_len * head_dim * 4 * 2
    if node.get("type") == "SdpaBackwardAttributes":
        expanded *= 2
    return block + expanded


def estimate_reference_host_bytes(graph_json: Dict[str, Any]) -> int:
    """Estimate the host memory of the CPU reference, inputs excluded.

    Each tensor counts once in its graph dtype and once as a float32 working
    copy (most handlers compute in float32); each graph output adds its
    float32 host copy; each SDPA node adds its scratch. The estimate ignores
    that virtual tensors are freed after their last reader, so it is high.
    """
    tensors = _tensor_metadata(graph_json)
    outputs = output_uids(graph_json)
    total = 0
    for uid, tensor_json in tensors.items():
        numel = prod(_dims(tensor_json))
        total += numel * (_element_size(tensor_json, graph_json) + 4)
        if uid in outputs and not tensor_json.get("virtual", False):
            total += numel * 4
    for node in graph_json.get("nodes", []):
        if str(node.get("type", "")).startswith("Sdpa"):
            total += _sdpa_scratch_bytes(node, tensors)
    return total


@ReferenceProviderRegistry.register(ReferenceProviderName.PYTORCH.value)
class PyTorchReferenceProvider(ReferenceProvider):
    """Reference provider using PyTorch for computation.

    Parses hipDNN graph JSON and executes equivalent PyTorch operations
    to produce reference outputs for validation.

    Uses the shared operation handlers from pytorch_ops module, executing
    on CPU tensors for reference computation.

    Supported operations:
    - ConvolutionFwdAttributes: 2D convolution forward pass
    - MatmulAttributes: Matrix multiplication
    - PointwiseAttributes: Element-wise operations (relu, add, mul, etc.)
    """

    @property
    def name(self) -> str:
        """Provider name."""
        return ReferenceProviderName.PYTORCH.value

    def is_available(self) -> bool:
        """Check if PyTorch is available.

        Returns:
            True if torch can be imported.
        """
        return torch_support.module_available()

    def supported_operations(self) -> Set[str]:
        """Get set of supported operation types.

        Returns:
            Set of operation type strings that have handlers.
        """
        return _get_pytorch_ops().get_supported_operations()

    def supports_graph(self, graph_json: Dict[str, Any]) -> bool:
        """Check if all graph operations are supported.

        Args:
            graph_json: The graph as a parsed JSON dictionary.

        Returns:
            True if all node types have handlers.
        """
        return _get_pytorch_ops().supports_graph(graph_json)

    def get_unsupported_operations(self, graph_json: Dict[str, Any]) -> List[str]:
        """Get list of unsupported operation types in graph.

        Args:
            graph_json: The graph as a parsed JSON dictionary.

        Returns:
            List of unsupported operation type strings.
        """
        return _get_pytorch_ops().get_unsupported_operations(graph_json)

    def estimate_host_bytes(self, graph_json: Dict[str, Any]) -> Optional[int]:
        """See :func:`estimate_reference_host_bytes`."""
        return estimate_reference_host_bytes(graph_json)

    def compute_reference(
        self,
        graph_json: Dict[str, Any],
        input_data: Dict[int, np.ndarray],
    ) -> Dict[int, ReferenceOutput]:
        """Compute reference outputs using PyTorch on CPU.

        Args:
            graph_json: The graph as a parsed JSON dictionary.
            input_data: Mapping of tensor UID to input numpy arrays.

        Returns:
            Mapping of output tensor UID to ReferenceOutput.

        Raises:
            ImportError: If PyTorch is not available.
            UnsupportedGraphError: If the graph contains operations, attributes,
                or parameters the PyTorch reference does not support.
        """
        if not self.is_available():
            raise ImportError(
                "PyTorch is not available. Install with: pip install torch"
            )

        import torch

        # Check for unsupported operations
        unsupported = self.get_unsupported_operations(graph_json)
        if unsupported:
            raise UnsupportedGraphError(
                f"Graph contains unsupported operations: {unsupported}. "
                f"Supported: {list(self.supported_operations())}"
            )

        # Convert input data to torch tensors on CPU. When graph metadata names
        # a dtype, force PyTorch to execute with that dtype; validating a BF16
        # hipDNN graph against FP32 PyTorch math is a different computation.
        # from_numpy shares the caller's memory: no handler writes its inputs
        # in place (each stores a new tensor), so no defensive copy is made.
        tensor_json_by_uid = _tensor_metadata(graph_json)
        tensors: Dict[int, torch.Tensor] = {}
        for uid, data in input_data.items():
            tensor = torch.from_numpy(data)
            data_type = tensor_data_type(tensor_json_by_uid.get(uid, {}), graph_json)
            if data_type is not None:
                tensor = tensor.to(get_dtype(data_type).torch_dtype())
            tensors[uid] = tensor

        # Execute graph using shared handlers (works on CPU tensors)
        _get_pytorch_ops().execute_graph(graph_json, tensors)

        # NumPy has no bfloat16 or fp8, so those outputs are returned as the
        # float32 values BufferManager also uses for comparison. Virtual
        # (intermediate) tensors are not graph outputs and get no host copy.
        from ...execution.pytorch_buffer_manager import host_numpy

        return {
            uid: ReferenceOutput(data=host_numpy(tensors[uid]), tensor_uid=uid)
            for uid in output_uids(graph_json)
            if uid in tensors
            and not tensor_json_by_uid.get(uid, {}).get("virtual", False)
        }

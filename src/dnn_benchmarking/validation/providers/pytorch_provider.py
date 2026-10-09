# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""PyTorch reference provider for hipDNN graph validation.

Computes reference outputs by parsing graph JSON and executing
equivalent PyTorch operations on CPU.
"""

from typing import Any, Dict, List, Set

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
        tensor_json_by_uid = _tensor_metadata(graph_json)
        tensors: Dict[int, torch.Tensor] = {}
        for uid, data in input_data.items():
            tensor = torch.from_numpy(data.copy())
            data_type = tensor_data_type(tensor_json_by_uid.get(uid, {}), graph_json)
            if data_type is not None:
                tensor = tensor.to(get_dtype(data_type).torch_dtype())
            tensors[uid] = tensor

        # Execute graph using shared handlers (works on CPU tensors)
        _get_pytorch_ops().execute_graph(graph_json, tensors)

        # NumPy has no bfloat16 or fp8, so those outputs are returned as the
        # float32 values BufferManager also uses for comparison.
        from ...execution.pytorch_buffer_manager import host_numpy

        return {
            uid: ReferenceOutput(data=host_numpy(tensors[uid]), tensor_uid=uid)
            for uid in output_uids(graph_json)
            if uid in tensors
        }

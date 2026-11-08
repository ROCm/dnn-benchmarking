# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Abstract base class and registry for reference computation providers.

Reference providers compute expected outputs for hipDNN graph operations,
enabling validation of GPU execution against known-correct implementations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Optional, Type

from ..config.benchmark_config import ReferencePrecision

if TYPE_CHECKING:
    import numpy as np


@dataclass
class ReferenceOutput:
    """Output from a reference provider for a single tensor.

    At least one of ``data`` and ``device_data`` is set.

    Attributes:
        data: The computed output data as numpy array, or None when only
            ``device_data`` holds the values (see :meth:`host_data`).
        tensor_uid: UID of the tensor in the graph.
        device_data: Same logical values as ``data`` as a torch GPU tensor,
            or None when only host data exists.
        precision: Arithmetic of the values: ``float64`` (float64 values of
            a float64 computation) or ``native`` (the graph data types).
    """

    data: Optional["np.ndarray"]
    tensor_uid: int
    device_data: Optional[Any] = None
    precision: str = ReferencePrecision.NATIVE.value

    def host_data(self) -> "np.ndarray":
        """Return the values as a dense host array in the validation representation.

        A device-only reference is copied to the host on each call and the
        copy is not kept, so the host holds it only during one comparison.
        """
        if self.data is not None:
            return self.data
        from ..execution.pytorch_buffer_manager import host_numpy

        return host_numpy(self.device_data)


class ReferenceProvider(ABC):
    """Abstract base for reference computation backends.

    Implementations compute reference outputs for hipDNN graphs using
    alternative backends such as PyTorch.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name for display and logging."""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this provider can be used.

        Returns:
            True if the provider's dependencies are available.
        """
        ...

    @abstractmethod
    def compute_reference(
        self,
        graph_json: Dict[str, Any],
        input_data: Dict[int, "np.ndarray"],
        *,
        precision: ReferencePrecision = ReferencePrecision.NATIVE,
        device: Optional[str] = None,
    ) -> Dict[int, ReferenceOutput]:
        """Compute reference outputs for given graph and inputs.

        Args:
            graph_json: The graph as a parsed JSON dictionary.
            input_data: Mapping of tensor UID to input numpy arrays.
            precision: ``FLOAT64`` requires :meth:`supports_float64`.
            device: Device to compute on; None is the host. Outputs computed
                on a GPU stay there (``ReferenceOutput.device_data``).

        Returns:
            Mapping of output tensor UID to ReferenceOutput.

        Raises:
            NotImplementedError: If provider is not available.
            UnsupportedGraphError: If the graph contains operations, attributes,
                or parameters the provider does not support.
        """
        ...

    def supports_graph(self, graph_json: Dict[str, Any]) -> bool:
        """Check if provider supports all operations in graph.

        Args:
            graph_json: The graph as a parsed JSON dictionary.

        Returns:
            True if all operations are supported.
        """
        return True  # Default: assume supported, let compute_reference fail if not

    def supports_float64(self, graph_json: Dict[str, Any]) -> bool:
        """True when ``compute_reference`` can compute this graph in float64."""
        return False

    def estimate_host_bytes(
        self,
        graph_json: Dict[str, Any],
        precision: ReferencePrecision = ReferencePrecision.NATIVE,
    ) -> Optional[int]:
        """Estimate the host memory ``compute_reference`` needs beyond its inputs.

        Returns:
            Bytes, or None when the provider has no estimate (the caller then
            runs the reference without a memory check).
        """
        return None


class ReferenceProviderRegistry:
    """Registry of available reference providers.

    Allows dynamic registration and lookup of provider implementations.
    """

    _providers: Dict[str, Type[ReferenceProvider]] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register a provider class.

        Args:
            name: Name to register the provider under.

        Returns:
            Decorator function.

        Example:
            @ReferenceProviderRegistry.register("pytorch")
            class PyTorchReferenceProvider(ReferenceProvider):
                ...
        """

        def decorator(provider_cls: Type[ReferenceProvider]) -> Type[ReferenceProvider]:
            cls._providers[name] = provider_cls
            return provider_cls

        return decorator

    @classmethod
    def get_provider(cls, name: str, **kwargs: Any) -> ReferenceProvider:
        """Get instance of named provider.

        Args:
            name: Registered provider name.
            **kwargs: Arguments passed to provider constructor.

        Returns:
            Provider instance.

        Raises:
            ValueError: If provider name is not registered.
        """
        if name not in cls._providers:
            available = ", ".join(cls._providers.keys()) or "(none)"
            raise ValueError(
                f"Unknown reference provider: '{name}'. Available: {available}"
            )
        return cls._providers[name](**kwargs)

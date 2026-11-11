# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""PyTorch CUDA tensor management for graph execution."""

from typing import Dict, List, Optional, Union

import numpy as np
import torch

from ..common import torch_support
from ..common.blocks import iter_blocks
from ..graph.tensor_info import TensorInfo

# Floating dtypes NumPy can hold; the rest (bfloat16, fp8) go to host as float32.
_NUMPY_FLOATS = (torch.float16, torch.float32, torch.float64)


def host_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Copy a tensor to a dense host array in the validation representation.

    bfloat16 and fp8 tensors become float32. They are copied and converted
    one block at a time into the result, so no full-size host copy in the
    source dtype exists next to it.
    """
    tensor = tensor.detach()
    if not tensor.dtype.is_floating_point or tensor.dtype in _NUMPY_FLOATS:
        return tensor.cpu().contiguous().numpy()
    shape = tuple(tensor.shape)
    host = torch.empty(shape, dtype=torch.float32)
    for index in iter_blocks(shape):
        host[index] = tensor[index].cpu()
    return host.numpy()


class PyTorchCudaBufferManager:
    """Manages PyTorch CUDA tensor allocation for graph execution.

    This class handles:
    - Allocating CUDA tensors for all non-virtual tensors
    - Copying pre-generated inputs to the device
    - Zeroing output tensors
    - Providing tensors for graph execution
    """

    def __init__(
        self,
        tensor_infos: List[TensorInfo],
        device: Union[str, torch.device] = "cuda:0",
    ) -> None:
        """Initialize buffer manager with tensor metadata.

        Args:
            tensor_infos: List of TensorInfo objects describing tensors.
            device: CUDA device to use (e.g., "cuda:0").
        """
        self._tensor_infos = tensor_infos
        self._device = torch.device(device)
        self._tensors: Dict[int, torch.Tensor] = {}

    def allocate_all(self) -> None:
        """Allocate CUDA tensors for all non-virtual tensors.

        Raises:
            UnsupportedGraphError: If torch has no dtype for a tensor.
        """
        for tensor_info in self._tensor_infos:
            if tensor_info.is_virtual:
                continue

            dtype = tensor_info.dtype.torch_dtype()
            if tensor_info.is_pass_by_value:
                self._tensors[tensor_info.uid] = torch.tensor(
                    [tensor_info.value], dtype=dtype, device=self._device
                )
                continue

            if tensor_info.strides:
                tensor = torch.empty_strided(
                    tensor_info.dims,
                    tensor_info.strides,
                    dtype=dtype,
                    device=self._device,
                )
            else:
                tensor = torch.empty(
                    tensor_info.dims,
                    dtype=dtype,
                    device=self._device,
                )
            self._tensors[tensor_info.uid] = tensor

    def load_input_data(self, input_data: Dict[int, np.ndarray]) -> None:
        """Copy pre-generated graph input data into CUDA tensors.

        The host-to-device copy happens before timing starts so PyTorch
        reference rows preserve the same timing semantics as hipDNN rows.
        Pass-by-value tensors keep the value set by ``allocate_all`` unless
        ``input_data`` supplies one.

        Raises:
            ValueError: If a non-scalar input is missing from ``input_data``.
        """
        for tensor_info in self._tensor_infos:
            if tensor_info.is_output or tensor_info.is_virtual:
                continue

            data = input_data.get(tensor_info.uid)
            if data is None:
                if tensor_info.is_pass_by_value:
                    continue
                raise ValueError(f"Missing input data for tensor UID {tensor_info.uid}")

            tensor = self._tensors.get(tensor_info.uid)
            if tensor is None:
                continue
            source = torch.from_numpy(np.asarray(data))
            if tuple(source.shape) != tuple(tensor.shape):
                tensor.copy_(source)  # broadcast copy
                continue
            # A host-to-device copy_ that changes the dtype (float32 -> bfloat16
            # or fp8) converts on the host first. Copying in blocks keeps that
            # temporary at block size instead of the full tensor.
            for index in iter_blocks(tuple(source.shape)):
                tensor[index].copy_(source[index])

    def zero_outputs(self) -> None:
        """Zero output tensor buffers."""
        for tensor_info in self._tensor_infos:
            if not tensor_info.is_output:
                continue

            tensor = self._tensors.get(tensor_info.uid)
            if tensor is not None:
                tensor.zero_()

    def get_tensors(self) -> Dict[int, torch.Tensor]:
        """Get mapping of tensor UIDs to CUDA tensors.

        Returns:
            Dictionary mapping tensor UID to torch.Tensor on CUDA.
        """
        return self._tensors

    def get_output_data(self, uid: int) -> Optional[np.ndarray]:
        """Copy output tensor data from CUDA to numpy array.

        Args:
            uid: Tensor UID.

        Returns:
            Numpy array with output data, or None if tensor not found.
        """
        tensor = self._tensors.get(uid)
        if tensor is None:
            return None

        return host_numpy(tensor)

    def get_output_tensor(self, uid: int) -> Optional[torch.Tensor]:
        """Return the CUDA output tensor for a device-side comparison.

        Args:
            uid: Tensor UID.

        Returns:
            The typed, strided CUDA tensor, or None if tensor not found.
        """
        return self._tensors.get(uid)

    def get_output_tensors(self) -> List[TensorInfo]:
        """Get list of output tensor infos.

        Returns:
            List of TensorInfo objects for output tensors.
        """
        return [ti for ti in self._tensor_infos if ti.is_output]

    def cleanup(self) -> None:
        """Free all tensors."""
        self._tensors.clear()
        # Let PyTorch handle CUDA memory cleanup via garbage collection.
        # CPU-only torch installs expose the torch package but not a usable
        # CUDA/ROCm backend; cleanup must remain a no-op there.
        try:
            if torch_support.gpu_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def __enter__(self) -> "PyTorchCudaBufferManager":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - cleanup tensors."""
        self.cleanup()

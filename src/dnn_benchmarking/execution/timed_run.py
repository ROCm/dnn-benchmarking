# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Steps of a child script before its timed loop.

The OOTB and the tuned run of a row execute the same child script
(``cli/hipdnn_child.py`` or ``cli/pytorch_child.py``, with and without
``--tuned``), so both take these steps. Each caller then times with
``executor.benchmark`` and can use the same buffers after the timed loop
(for example, to save the outputs).

- hipDNN: :func:`build_hipdnn_plan`, then :func:`hipdnn_buffers`. The child
  reads the plan metrics between the two steps.
- PyTorch: :func:`pytorch_run`.

This module does not import ``torch`` at startup: the hipDNN child imports
this module and must not load torch.
"""

from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional, Tuple

from ..config.benchmark_config import PyTorchSdpaBackendName, TimingPolicy
from ..graph.tensor_info import TensorInfo
from .buffer_manager import BufferManager
from .executor import Executor


def build_hipdnn_plan(
    handle: Any, graph_json_str: str, policy: TimingPolicy, engine_id: int
) -> Executor:
    """Prime the engine, then build its plan with a timed build.

    The prime absorbs the one-time build costs of the engine (provider
    setup, cold plugin and kernel file reads) before the timed build.
    """
    Executor(graph_json_str, policy).prime(handle, engine_id)
    executor = Executor(graph_json_str, policy)
    executor.prepare(handle, engine_id=engine_id)
    return executor


@contextmanager
def hipdnn_buffers(
    tensor_infos: List[TensorInfo], input_data: Dict[int, Any]
) -> Iterator[Tuple[BufferManager, Dict[int, int]]]:
    """Allocate the graph buffers, load the inputs and zero the outputs.

    Yields the buffer manager and its variant pack.
    """
    with BufferManager(tensor_infos) as bm:
        bm.allocate_all()
        bm.load_input_data(input_data)
        bm.zero_outputs()
        yield bm, bm.create_variant_pack()


@contextmanager
def pytorch_run(
    graph_json: Dict[str, Any],
    tensor_infos: List[TensorInfo],
    input_data: Dict[int, Any],
    policy: TimingPolicy,
    pytorch_sdpa_backend: PyTorchSdpaBackendName,
    pytorch_rocm_fa_library: Optional[str],
) -> Iterator[Tuple[Any, Any]]:
    """Build the graph in PyTorch and fill its buffers.

    Yields the ``PyTorchCudaExecutor`` and the ``PyTorchCudaBufferManager``,
    with the inputs loaded and the outputs zeroed.
    """
    from .pytorch_buffer_manager import PyTorchCudaBufferManager
    from .pytorch_executor import PyTorchCudaExecutor

    executor = PyTorchCudaExecutor(
        graph_json,
        policy,
        pytorch_sdpa_backend=pytorch_sdpa_backend,
        pytorch_rocm_fa_library=pytorch_rocm_fa_library,
    )
    executor.prepare()
    # The executor's device, not the buffer manager's cuda:0 default.
    with PyTorchCudaBufferManager(tensor_infos, device=executor.device) as bm:
        bm.allocate_all()
        bm.load_input_data(input_data)
        bm.zero_outputs()
        yield executor, bm

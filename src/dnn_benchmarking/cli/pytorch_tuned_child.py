# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tuned-PyTorch child process of ``--oracle``.

``python -m dnn_benchmarking.cli.pytorch_tuned_child --graph G --seed S
--timing-policy JSON --out-dir D --pytorch-sdpa-backend B
[--pytorch-rocm-fa-library L]``

The parent (``execution.oracle.run_pytorch_tuned``) sets the environment
``execution.tuned_child.PYTORCH_TUNED_ENV``: TunableOp on, and a private
``MIOPEN_USER_DB_PATH`` and TunableOp results file. This script checks it
and sets no tuning variable. It turns on
PyTorch's exhaustive conv search (``torch.backends.cudnn.benchmark``), builds
the graph as the OOTB PyTorch row does, fills the inputs from the seed and
times one loop with the parent's timing policy. It writes ``result.json`` in
D. On failure it writes an error on stderr and exits 1.
"""

import sys
from typing import List, Optional

from ..config.benchmark_config import PyTorchSdpaBackendName, TimingPolicy
from ..execution.buffer_manager import generate_input_data
from ..execution.timing import retry_unstalled
from ..execution.tuned_child import PYTORCH_TUNED_ENV, child_parser, write_result
from ..graph.loader import GraphLoader
from ..reporting.reporter import Reporter


def parse_args(argv: Optional[List[str]] = None):
    """Parse the child argv that ``execution.oracle`` builds."""
    parser = child_parser("pytorch_tuned_child")
    parser.add_argument(
        "--pytorch-sdpa-backend", required=True, type=PyTorchSdpaBackendName
    )
    parser.add_argument("--pytorch-rocm-fa-library")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    """Time one tuned PyTorch run; return the exit code."""
    args = parse_args(argv)
    try:
        # Before torch loads: TunableOp reads its variables at startup.
        PYTORCH_TUNED_ENV.check()
        import torch

        from ..execution.pytorch_buffer_manager import PyTorchCudaBufferManager
        from ..execution.pytorch_executor import PyTorchCudaExecutor

        torch.backends.cudnn.benchmark = True

        loader = GraphLoader()
        graph_json = loader.load_json(args.graph)
        loader.validate(graph_json)
        tensor_infos = loader.extract_tensor_info(graph_json)

        def run(policy: TimingPolicy) -> None:
            executor = PyTorchCudaExecutor(
                graph_json,
                policy,
                pytorch_sdpa_backend=args.pytorch_sdpa_backend,
                pytorch_rocm_fa_library=args.pytorch_rocm_fa_library,
            )
            executor.prepare()
            with PyTorchCudaBufferManager(tensor_infos, device=executor.device) as bm:
                bm.allocate_all()
                bm.load_input_data(
                    generate_input_data(tensor_infos, args.seed, graph_json)
                )
                bm.zero_outputs()
                m = executor.benchmark(bm.get_tensors())
            write_result(args.out_dir, m)

        retry_unstalled(run, args.timing_policy)
    except Exception as e:
        Reporter(quiet=True).error(f"{type(e).__name__}: {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

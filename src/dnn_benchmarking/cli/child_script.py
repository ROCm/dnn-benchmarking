# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Body that the timed child scripts share (``hipdnn_child``, ``pytorch_child``).

A script parses its options with ``execution.child_protocol.child_parser``
and calls :func:`run_timed_child` with its specs and a ``prepare`` function
that builds its runtime. Inside its timed run it calls :func:`timed_loop`.
Only the runtime-specific steps stay in the script: the plan or graph build,
the buffers, and the extra values of ``result.json``.

This module does not import ``torch``: the hipDNN child imports it.
"""

import argparse
from typing import Any, Callable, Dict, Optional, Tuple

from ..config.benchmark_config import TimingPolicy
from ..execution.child_protocol import (
    ChildSpec,
    OutputBuffers,
    run_reporting_failures,
    save_outputs,
    select_spec,
)
from ..execution.timing import Measurement, retry_unstalled
from ..metrics import GpuSmiProbe

#: One timed run with the given timing policy; writes the result files.
TimedRun = Callable[[TimingPolicy], None]


def run_timed_child(
    args: argparse.Namespace,
    specs: Tuple[ChildSpec, ...],
    prepare: Callable[[], TimedRun],
) -> int:
    """Run one child; return the exit code.

    Selects the spec that the flags of ``args`` select, checks its
    environment before any runtime loads, then calls ``prepare`` once and
    times. A tuned run times again without the stall gate when the gate
    fails. Any other run reports the failure: the parent then remeasures the
    whole graph without the gate, so all rows of the graph share one timing
    mode. A failure goes to ``failure.json``.
    """

    def body() -> None:
        spec = select_spec(args, *specs)
        spec.env.check(args.out_dir)
        run = prepare()
        if spec.tuned:
            retry_unstalled(run, args.timing_policy)
        else:
            run(args.timing_policy)

    return run_reporting_failures(args.out_dir, body)


def timed_loop(
    args: argparse.Namespace,
    buffers: OutputBuffers,
    benchmark: Callable[[], Measurement],
    run_once: Callable[[], None],
) -> Tuple[Measurement, Optional[Dict[str, Any]]]:
    """Time ``benchmark``; return the measurement and the GPU clocks right
    after it (``--clocks``). With ``--save-outputs``, run the graph once more
    with ``run_once`` and save its outputs."""
    probe = GpuSmiProbe() if args.clocks else None
    m = benchmark()
    clocks_after = probe.clocks() if probe is not None else None
    if args.save_outputs:
        save_outputs(args.out_dir, buffers, run_once)
    return m, clocks_after

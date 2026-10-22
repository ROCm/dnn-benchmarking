# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Build hipDNN's dense-attention variants with rocKE in a child process.

The shipped code objects come from rocKE's pack step, which compiles with the
comgr that ``setup_env.py`` passes to it. A process that has loaded torch
maps torch's comgr, and rocKE refuses to open a second one there. A different
comgr gives different code-object bytes from the same IR. The rocKE runtime
therefore builds catalog variants here, in a process without torch, with
``ROCKE_COMGR_LIB`` set from ``DNN_BENCH_ROCKE_COMGR_LIB``.

The parent sends a JSON list of specs on stdin. The child answers on the last
line of stdout with one entry per spec: ``hsaco`` (base64), ``kernel_name``
and ``compile_ms``, or ``error``. ``comgr`` names the library the child loaded.

This module must not import torch.
"""

import base64
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Union

#: The comgr of rocKE's pack step; ``setup_env.py`` exports it.
COMGR_ENV = "DNN_BENCH_ROCKE_COMGR_LIB"


@dataclass(frozen=True)
class BuiltKernel:
    """One code object built by the child."""

    hsaco: bytes
    kernel_name: str
    compile_ms: float
    comgr: Optional[str]


def build_dense_variants(
    specs: Sequence[Dict[str, Any]],
) -> List[Union[BuiltKernel, str]]:
    """Build ``specs`` in one child; a string is that variant's build error.

    Raises:
        RuntimeError: The child failed before it answered.
    """
    env = dict(os.environ)
    comgr = env.get(COMGR_ENV)
    if comgr:
        env["ROCKE_COMGR_LIB"] = comgr
    proc = subprocess.run(
        [sys.executable, "-m", __name__],
        input=json.dumps(list(specs)),
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    lines = proc.stdout.strip().splitlines()
    if proc.returncode != 0 or not lines:
        tail = proc.stderr.strip().splitlines()[-3:]
        raise RuntimeError(
            f"rocKE build child exited {proc.returncode}: {' | '.join(tail)}"
        )
    answer = json.loads(lines[-1])
    if len(answer["kernels"]) != len(specs):
        raise RuntimeError(
            f"rocKE build child answered {len(answer['kernels'])} of "
            f"{len(specs)} variants"
        )
    built: List[Union[BuiltKernel, str]] = []
    for entry in answer["kernels"]:
        if "error" in entry:
            built.append(entry["error"])
            continue
        built.append(
            BuiltKernel(
                hsaco=base64.b64decode(entry["hsaco"]),
                kernel_name=entry["kernel_name"],
                compile_ms=entry["compile_ms"],
                comgr=answer["comgr"],
            )
        )
    return built


def _main() -> None:
    from kernels.gfx950.attention_dense import (
        Gfx950AttentionDenseSpec,
        build_attention_dense,
    )
    from rocke.helpers.compile import compile_kernel
    from rocke.runtime.comgr import resolved_lib_path

    from .rocke_sdpa import DENSE_ARCH

    kernels = []
    for spec in json.load(sys.stdin):
        start = time.perf_counter()
        try:
            # backend="python" is the lowering hipDNN's pack step pins.
            art = compile_kernel(
                build_attention_dense(
                    Gfx950AttentionDenseSpec(**spec), arch=DENSE_ARCH
                ),
                arch=DENSE_ARCH,
                backend="python",
                capture_ir_text=False,
            )
        except Exception as e:
            kernels.append({"error": f"{type(e).__name__}: {e}"})
            continue
        kernels.append(
            {
                "hsaco": base64.b64encode(art.hsaco).decode("ascii"),
                "kernel_name": art.kernel_name,
                "compile_ms": (time.perf_counter() - start) * 1000.0,
            }
        )
    # A library may print to stdout; the answer is the last line.
    answer = {"comgr": resolved_lib_path(), "kernels": kernels}
    sys.stdout.write("\n" + json.dumps(answer) + "\n")


if __name__ == "__main__":
    _main()

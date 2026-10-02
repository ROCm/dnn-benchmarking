# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""dnn-benchmark must load the hipDNN bindings before it imports torch.

A ROCm torch wheel ships its own libhipdnn_backend. Whichever library the
process loads first owns that soname, so importing torch first binds
hipdnn_frontend to torch's copy rather than the hipDNN install on
LD_LIBRARY_PATH (seen as `undefined symbol: hipdnnGetEngineNameById_ext`).
"""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[3]

pytest.importorskip("torch")


# --validate pytorch probes the reference provider, which imports torch,
# before the hipDNN runner starts.
@pytest.mark.parametrize("extra_args", [[], ["--validate", "pytorch"]])
def test_bindings_are_imported_before_torch(tmp_path: Path, extra_args) -> None:
    fake = tmp_path / "fake" / "hipdnn_frontend"
    fake.mkdir(parents=True)
    record = tmp_path / "torch_loaded_at_import.txt"
    (fake / "__init__.py").write_text(
        textwrap.dedent(
            f"""
            import sys
            with open({str(record)!r}, "w") as f:
                f.write(str("torch" in sys.modules))

            def __getattr__(name):
                raise RuntimeError("fake hipdnn_frontend has no " + name)
            """
        )
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(fake.parent), str(_REPO / "src"), env.get("PYTHONPATH", "")]
    )
    env["DNN_BENCH_WORKSPACE"] = str(tmp_path / "ws")
    env["ROCM_PATH"] = str(tmp_path / "rocm")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "dnn_benchmarking",
            "--graph",
            str(_REPO / "graphs" / "sample_sdpa.json"),
            "--cache-dir",
            str(tmp_path / "cache"),
            *extra_args,
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )

    # The fake cannot create a handle; the run stops there.
    assert result.returncode == 1, result.stdout + result.stderr
    assert "Failed to create hipDNN handle" in result.stdout + result.stderr
    assert record.read_text() == "False"

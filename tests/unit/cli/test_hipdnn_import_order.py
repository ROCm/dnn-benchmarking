# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""hipDNN bindings must load before a PyTorch reference probe imports torch.

ROCm torch and an external hipDNN prefix can provide different copies of
libhipdnn_backend. The subprocesses exercise the real CLI startup order, so a
torch import in the test runner cannot hide a regression.
"""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[3]

pytest.importorskip("torch")


def _run_with_fake_frontend(
    tmp_path: Path, *extra_args: str, import_error: str | None = None
) -> tuple[subprocess.CompletedProcess[str], Path]:
    fake = tmp_path / "fake" / "hipdnn_frontend"
    fake.mkdir(parents=True)
    record = tmp_path / "torch_loaded_at_frontend_import.txt"
    code = textwrap.dedent(
        f"""
        import sys
        from pathlib import Path

        Path({str(record)!r}).write_text(str("torch" in sys.modules))

        class PluginLoadingMode:
            ABSOLUTE = "absolute"

        def set_engine_plugin_paths(paths, mode):
            pass

        class Handle:
            def __init__(self):
                raise RuntimeError("synthetic handle has no GPU")
        """
    )
    if import_error:
        code += f"raise ImportError({import_error!r})\n"
    (fake / "__init__.py").write_text(code)

    plugin_path = tmp_path / "plugins"
    plugin_path.mkdir()
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(fake.parent), str(_REPO / "src"), env.get("PYTHONPATH")) if p
    )
    env["DNN_BENCH_WORKSPACE"] = str(tmp_path / "workspace")
    env["ROCM_PATH"] = str(tmp_path / "other-rocm")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "dnn_benchmarking",
            "--graph",
            str(_REPO / "graphs" / "sample_relu.json"),
            "--plugin-path",
            str(plugin_path),
            "--warmup",
            "1",
            "--iters",
            "2",
            *extra_args,
        ],
        cwd=_REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    return result, record


@pytest.mark.parametrize(
    "extra_args",
    [(), ("--validate", "pytorch")],
    ids=("hipdnn", "hipdnn-with-pytorch-reference"),
)
def test_bindings_are_imported_before_torch(
    tmp_path: Path, extra_args: tuple[str, ...]
) -> None:
    result, record = _run_with_fake_frontend(tmp_path, *extra_args)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "hipDNN handle creation failed: synthetic handle has no GPU" in (
        result.stdout + result.stderr
    )
    assert record.read_text() == "False"


def test_pytorch_runtime_does_not_import_hipdnn(tmp_path: Path) -> None:
    result, record = _run_with_fake_frontend(tmp_path, "--runtime", "pytorch")
    if result.returncode:
        assert result.returncode == 1, result.stdout + result.stderr
        assert "--runtime pytorch: PyTorch sees no GPU" in (
            result.stdout + result.stderr
        )
    else:
        assert "Summary: 1 graph(s)" in result.stdout
    assert not record.exists(), "PyTorch-only CLI imported hipdnn_frontend"


def test_native_binding_import_error_is_reported(tmp_path: Path) -> None:
    symbol = "undefined symbol: hipdnnGetEngineNameById_ext"
    result, record = _run_with_fake_frontend(
        tmp_path, "--validate", "pytorch", import_error=symbol
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert symbol in result.stdout + result.stderr
    assert record.read_text() == "False"

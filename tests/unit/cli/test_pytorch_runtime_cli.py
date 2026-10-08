# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""PyTorch runtime startup: unavailable torch/GPU fails fast, hipDNN untouched."""

import builtins
import io
import sys
import types

import pytest

from dnn_benchmarking.cli import suite_runner_cli
from dnn_benchmarking.cli.config_file import apply_config_file
from dnn_benchmarking.cli.parser import create_parser
from dnn_benchmarking.reporting.reporter import Reporter


@pytest.fixture(autouse=True)
def _no_hipdnn(monkeypatch):
    """Fail any hipDNN import: the PyTorch runtime must never create a handle."""
    real_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.startswith("hipdnn_frontend"):
            raise AssertionError("hipdnn_frontend imported by the pytorch runtime")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    monkeypatch.setattr(suite_runner_cli, "collect_environment_info", lambda: {})


def _run(*argv: str) -> tuple:
    args = create_parser(suppress_defaults=True).parse_args(list(argv))
    apply_config_file(args)
    out = io.StringIO()
    code = suite_runner_cli.run_suite_cli(args, [], Reporter(output=out))
    return code, out.getvalue()


def test_missing_torch_exits_1(monkeypatch) -> None:
    real_import = builtins.__import__

    def blocking(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            raise ImportError("blocked torch")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocking)
    code, text = _run("-r", "pytorch")
    assert code == 1
    assert "ERROR:" in text and "PyTorch" in text
    assert "Traceback" not in text


def test_torch_without_gpu_exits_1(monkeypatch) -> None:
    fake = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))
    monkeypatch.setitem(sys.modules, "torch", fake)
    code, text = _run("-r", "pytorch")
    assert code == 1
    assert "ERROR:" in text and "GPU" in text


def test_hipdnn_only_option_is_usage_error() -> None:
    code, text = _run("-r", "pytorch", "--autotune")
    assert code == 2
    assert "--autotune" in text

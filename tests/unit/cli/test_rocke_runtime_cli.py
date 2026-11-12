# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""rocKE runtime startup: missing prerequisites fail fast, hipDNN untouched."""

import builtins
import importlib.util
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
    """Fail any hipDNN import: the rocKE runtime must never create a handle."""
    real_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.startswith("hipdnn_frontend"):
            raise AssertionError("hipdnn_frontend imported by the rocke runtime")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    monkeypatch.setattr(suite_runner_cli, "collect_environment_info", lambda: {})


def _run(*argv: str) -> tuple:
    args = create_parser(suppress_defaults=True).parse_args(list(argv))
    apply_config_file(args)
    out = io.StringIO()
    code = suite_runner_cli.run_suite_cli(args, [], Reporter(output=out))
    return code, out.getvalue()


def _fake_torch(monkeypatch, *, gpu: bool, hip: object) -> None:
    fake = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: gpu),
        version=types.SimpleNamespace(hip=hip, cuda=None if hip else "12.4"),
    )
    monkeypatch.setitem(sys.modules, "torch", fake)


def test_torch_without_gpu_exits_1(monkeypatch) -> None:
    _fake_torch(monkeypatch, gpu=False, hip="7.0")
    code, text = _run("-r", "rocke")
    assert code == 1
    assert "GPU" in text and "Traceback" not in text


def test_cuda_torch_exits_1(monkeypatch) -> None:
    _fake_torch(monkeypatch, gpu=True, hip=None)
    code, text = _run("-r", "rocke")
    assert code == 1
    assert "ROCm" in text


def test_missing_rocke_packages_exit_1(monkeypatch) -> None:
    _fake_torch(monkeypatch, gpu=True, hip="7.0")
    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *a: None if name == "dispatch" else real_find_spec(name, *a),
    )
    code, text = _run("-r", "rocke")
    assert code == 1
    assert "dispatch" in text and "Traceback" not in text


@pytest.mark.parametrize(
    "flag", [("--engine", "MIOPEN_ENGINE"), ("--oracle",), ("--autotune",)]
)
def test_hipdnn_only_option_is_usage_error(flag) -> None:
    code, text = _run("-r", "rocke", *flag)
    assert code == 2
    assert flag[0] in text


@pytest.mark.parametrize("runtime", ["rocke", "hipdnn,rocke"])
def test_plugin_library_file_is_usage_error(tmp_path, runtime) -> None:
    """rocKE reads descriptors under plugin directories, so a file is refused
    before any runtime starts: no GPU check, and no hipDNN handle in a mixed
    run (the autouse guard fails any hipDNN import)."""
    plugin = tmp_path / "libhip_kernel_provider.so"
    plugin.write_bytes(b"")
    code, text = _run("-r", runtime, "--plugin-path", str(plugin))
    assert code == 2
    assert f"--plugin-path names a file: {plugin}" in text

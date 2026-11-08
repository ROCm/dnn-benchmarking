# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""AITER executor: the JIT directory and toolchain set before ``import aiter``."""

import importlib.machinery
import importlib.util
import os
import sys

import pytest

from dnn_benchmarking.execution import aiter_executor


def _executable(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\n")
    path.chmod(0o755)
    return path


@pytest.fixture
def venv(tmp_path, monkeypatch):
    """A venv interpreter whose bin holds the pip ROCm SDK's hipcc, plus a
    system hipcc first on PATH and an expanded _rocm_sdk_devel package."""
    prefix = tmp_path / "venv"
    python = _executable(prefix / "bin" / "python")
    _executable(prefix / "bin" / "hipcc")
    system_bin = _executable(tmp_path / "opt" / "bin" / "hipcc").parent
    devel = tmp_path / "site" / "_rocm_sdk_devel"
    _executable(devel / "bin" / "hipconfig")

    monkeypatch.setattr(sys, "prefix", str(prefix))
    monkeypatch.setattr(sys, "base_prefix", str(tmp_path / "base"))
    monkeypatch.setattr(sys, "executable", str(python))
    monkeypatch.setenv("PATH", str(system_bin))
    for var in ("AITER_JIT_DIR", "ROCM_HOME"):
        # setenv first so that teardown removes what the function sets.
        monkeypatch.setenv(var, "")
        monkeypatch.delenv(var)
    real_find_spec = importlib.util.find_spec
    devel_spec = importlib.machinery.ModuleSpec(
        "_rocm_sdk_devel", None, is_package=True
    )
    devel_spec.submodule_search_locations = [str(devel)]
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *a: (
            devel_spec if name == "_rocm_sdk_devel" else real_find_spec(name, *a)
        ),
    )
    return prefix, devel


def test_venv_sdk_compiles_and_the_jit_dir_is_per_venv(venv) -> None:
    prefix, devel = venv
    aiter_executor.prepare_aiter_environment()
    # Without these, AITER links against the hipDNN prefix, which has no
    # libamdhip64, or loads modules that another torch built.
    assert os.environ["ROCM_HOME"] == str(devel)
    assert os.environ["PATH"].split(os.pathsep)[0] == str(prefix / "bin")
    assert os.environ["AITER_JIT_DIR"] == str(prefix / "aiter-jit")


def test_user_settings_win(venv, monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("AITER_JIT_DIR", str(tmp_path / "jit"))
    monkeypatch.setenv("ROCM_HOME", str(tmp_path / "rocm"))
    aiter_executor.prepare_aiter_environment()
    assert os.environ["AITER_JIT_DIR"] == str(tmp_path / "jit")
    assert os.environ["ROCM_HOME"] == str(tmp_path / "rocm")


def test_outside_a_venv_aiter_keeps_its_default_jit_dir(venv, monkeypatch) -> None:
    monkeypatch.setattr(sys, "base_prefix", sys.prefix)
    aiter_executor.prepare_aiter_environment()
    assert "AITER_JIT_DIR" not in os.environ


def test_unexpanded_sdk_changes_no_toolchain(venv, monkeypatch) -> None:
    _, devel = venv
    (devel / "bin" / "hipconfig").unlink()
    path = os.environ["PATH"]
    aiter_executor.prepare_aiter_environment()
    assert "ROCM_HOME" not in os.environ
    assert os.environ["PATH"] == path

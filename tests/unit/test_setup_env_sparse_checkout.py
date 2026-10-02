# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for the rocm-libraries sparse checkout made by setup_env.py.

The hipDNN configure reads rocm-libraries/shared for its test category YAML.
A sparse checkout without it configures with legacy CTest labels and says so
only in a configure message.
"""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

_SETUP_ENV = Path(__file__).resolve().parents[2] / "setup_env.py"


@pytest.fixture()
def setup_env(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("setup_env_sparse", _SETUP_ENV)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "ROCM_LIBRARIES_DIR", tmp_path / "rocm-libraries")
    return module


def _record_git(module, monkeypatch, sparse: bool) -> list:
    calls = []

    def fake_git_output(args):
        if args[-1] == "core.sparseCheckout":
            if not sparse:
                raise subprocess.CalledProcessError(1, ["git", *args])
            return "true"
        return "abc123"

    monkeypatch.setattr(module, "git_output", fake_git_output)
    monkeypatch.setattr(module, "run_git", lambda args, **kw: calls.append(args))
    return calls


def _setup(module):
    return module.Setup(module.build_parser().parse_args(["--workspace", "ws"]))


def test_a_fresh_checkout_includes_shared(setup_env, monkeypatch) -> None:
    calls = _record_git(setup_env, monkeypatch, sparse=True)

    _setup(setup_env).ensure_rocm_libraries_checkout()

    (sparse_set,) = [c for c in calls if c[2:4] == ["sparse-checkout", "set"]]
    assert {"cmake", "shared", "projects/hipdnn", "dnn-providers"} <= set(
        sparse_set[4:]
    )


def test_an_older_sparse_checkout_gains_shared(setup_env, monkeypatch) -> None:
    root = setup_env.ROCM_LIBRARIES_DIR
    (root / ".git").mkdir(parents=True)
    (root / "cmake").mkdir()
    calls = _record_git(setup_env, monkeypatch, sparse=True)

    _setup(setup_env).ensure_rocm_libraries_checkout()

    assert [c for c in calls if "sparse-checkout" in c] == [
        ["-C", str(root), "sparse-checkout", "add", "shared"]
    ]


def test_a_full_checkout_is_left_alone(setup_env, monkeypatch) -> None:
    """A non-sparse checkout of a ref without shared/ must not break setup."""
    root = setup_env.ROCM_LIBRARIES_DIR
    (root / ".git").mkdir(parents=True)
    (root / "cmake").mkdir()
    calls = _record_git(setup_env, monkeypatch, sparse=False)

    _setup(setup_env).ensure_rocm_libraries_checkout()

    assert calls == []

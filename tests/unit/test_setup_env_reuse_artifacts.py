# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for setup_env.py --reuse-artifacts and the hipDNN Python bindings.

--reuse-artifacts builds nothing, including the hipdnn_frontend bindings. With
a fresh venv (--torch-mode cpu) setup used to warn and then print "Setup
complete" for an environment that cannot import hipdnn_frontend.
"""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

_SETUP_ENV = Path(__file__).resolve().parents[2] / "setup_env.py"


@pytest.fixture()
def setup_env():
    spec = importlib.util.spec_from_file_location("setup_env_reuse", _SETUP_ENV)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _reuse_setup(module, tmp_path, monkeypatch, frontend_importable: bool):
    prefix = tmp_path / "install"
    args = module.build_parser().parse_args(
        [
            "--workspace",
            str(tmp_path / "ws"),
            "--torch-mode",
            "cpu",
            "--rocm-prefix",
            str(prefix),
            "--reuse-artifacts",
        ]
    )
    setup = module.Setup(args)
    monkeypatch.setattr(setup, "prefix_has_hipdnn", lambda prefix: True)
    monkeypatch.setattr(setup, "maybe_install_amdsmi", lambda *a: None)
    monkeypatch.setattr(setup, "write_activate_local", lambda *a: None)
    monkeypatch.setattr(
        setup,
        "probe",
        lambda code, *a: subprocess.CompletedProcess(
            [], 0 if frontend_importable else 1, "", ""
        ),
    )
    return setup, str(prefix)


def test_reuse_without_bindings_fails_instead_of_claiming_success(
    setup_env, tmp_path, monkeypatch, capsys
) -> None:
    setup, prefix = _reuse_setup(
        setup_env, tmp_path, monkeypatch, frontend_importable=False
    )

    with pytest.raises(SystemExit) as caught:
        setup.build_and_install(prefix)

    assert caught.value.code == 1
    assert "--torch-mode existing" in capsys.readouterr().err


def test_reuse_with_bindings_already_in_the_venv_proceeds(
    setup_env, tmp_path, monkeypatch
) -> None:
    setup, prefix = _reuse_setup(
        setup_env, tmp_path, monkeypatch, frontend_importable=True
    )

    setup.build_and_install(prefix)

    assert setup.env["ROCM_PATH"] == prefix

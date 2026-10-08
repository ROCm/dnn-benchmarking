# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Reuse requires an installed binding package in the preserved venv.

The binding may be present even when its native libraries cannot be loaded on
the setup host. Checking for its package must not import those libraries.
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


def _installed_setup(module, tmp_path, monkeypatch):
    prefix = tmp_path / "install"
    for name in ("hipdnn_frontend", "hipdnn_backend"):
        config = prefix / "lib" / "cmake" / name / f"{name}Config.cmake"
        config.parent.mkdir(parents=True)
        config.write_text("", encoding="utf-8")

    args = module.build_parser().parse_args(
        [
            "--workspace",
            str(tmp_path / "ws"),
            "--torch-mode",
            "existing",
            "--rocm-prefix",
            str(prefix),
            "--reuse-artifacts",
        ]
    )
    setup = module.Setup(args)
    subprocess.run([sys.executable, "-m", "venv", str(setup.venv_dir)], check=True)
    monkeypatch.setattr(setup, "maybe_install_amdsmi", lambda *prefixes: None)
    return setup, str(prefix)


def test_reuse_without_installed_bindings_fails(
    setup_env, tmp_path, monkeypatch, capsys
):
    setup, prefix = _installed_setup(setup_env, tmp_path, monkeypatch)

    with pytest.raises(SystemExit) as caught:
        setup.build_and_install(prefix)

    assert caught.value.code == 1
    assert "hipdnn_frontend is not installed" in capsys.readouterr().err


def test_reuse_detects_bindings_without_loading_native_libraries(
    setup_env, tmp_path, monkeypatch
):
    setup, prefix = _installed_setup(setup_env, tmp_path, monkeypatch)
    site = setup.probe("import sysconfig; print(sysconfig.get_path('purelib'))")
    assert site.returncode == 0, site.stderr
    package = Path(site.stdout.strip()) / "hipdnn_frontend"
    package.mkdir()
    (package / "__init__.py").write_text(
        "raise OSError('native ROCm runtime is unavailable on this host')\n",
        encoding="utf-8",
    )
    native_import = setup.probe("import hipdnn_frontend")
    assert native_import.returncode != 0
    assert "native ROCm runtime is unavailable" in native_import.stderr

    setup.build_and_install(prefix)

    assert setup.env["ROCM_PATH"] == prefix


@pytest.mark.parametrize("mode", ("rocm", "cpu", "none"))
def test_reuse_with_recreated_venv_fails_before_modifying_it(mode, tmp_path):
    workspace = tmp_path / "ws"
    venv = workspace / ".venv"
    venv.mkdir(parents=True)
    marker = venv / "keep"
    marker.write_text("original", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(_SETUP_ENV),
            "--workspace",
            str(workspace),
            "--torch-mode",
            mode,
            "--reuse-artifacts",
            "--yes",
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert result.returncode == 1
    assert "--torch-mode existing" in result.stderr
    assert "Traceback" not in result.stderr
    assert marker.read_text(encoding="utf-8") == "original"

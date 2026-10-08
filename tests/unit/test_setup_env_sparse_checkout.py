# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""The setup checkout must stage hipDNN's CTest categories, without all of shared/."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

_SETUP_ENV = Path(__file__).resolve().parents[2] / "setup_env.py"


def _git(*args: str) -> str:
    result = subprocess.run(["git", *args], capture_output=True, text=True, check=True)
    return result.stdout.strip()


@pytest.fixture()
def checkout_source(tmp_path):
    source = tmp_path / "rocm-libraries-source"
    _git("init", "--quiet", "-b", "main", str(source))
    files = {
        "CMakePresets.json": "{}\n",
        "cmake/Settings.cmake": "# root CMake helper\n",
        "shared/ctest/TestCategories.cmake": "# required by hipDNN configure\n",
        "shared/tensile/large_unneeded_file.txt": "not a configure input\n",
        "projects/hipdnn/CMakeLists.txt": "# hipDNN\n",
        "dnn-providers/CMakeLists.txt": "# providers\n",
    }
    for name, content in files.items():
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    _git("-C", str(source), "add", ".")
    _git(
        "-C",
        str(source),
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.org",
        "commit",
        "--quiet",
        "-m",
        "local sparse-checkout fixture",
    )
    return source


@pytest.fixture()
def setup_env(tmp_path, monkeypatch, checkout_source):
    root = tmp_path / "dnn-benchmarking"
    root.mkdir()
    (root / ".gitmodules").write_text(
        '[submodule "rocm-libraries"]\n'
        "    path = rocm-libraries\n"
        f"    url = {checkout_source.as_posix()}\n"
        "    branch = main\n",
        encoding="utf-8",
    )
    spec = importlib.util.spec_from_file_location("setup_env_sparse", _SETUP_ENV)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "SCRIPT_DIR", root)
    monkeypatch.setattr(module, "ROCM_LIBRARIES_DIR", root / "rocm-libraries")
    return module, root


def _setup(module, root):
    args = module.build_parser().parse_args(["--workspace", str(root / "workspace")])
    return module.Setup(args)


def test_fresh_sparse_checkout_stages_categories_without_tensile(setup_env):
    module, root = setup_env

    _setup(module, root).ensure_rocm_libraries_checkout()

    checkout = root / "rocm-libraries"
    assert (checkout / "shared/ctest/TestCategories.cmake").is_file()
    assert not (checkout / "shared/tensile/large_unneeded_file.txt").exists()
    assert (checkout / "cmake/Settings.cmake").is_file()
    assert (checkout / "CMakePresets.json").is_file()


def test_existing_sparse_checkout_gains_missing_categories_without_tensile(
    setup_env, checkout_source
):
    module, root = setup_env
    checkout = root / "rocm-libraries"
    _git("clone", "--quiet", "--no-checkout", str(checkout_source), str(checkout))
    _git("-C", str(checkout), "sparse-checkout", "init", "--cone")
    _git(
        "-C",
        str(checkout),
        "sparse-checkout",
        "set",
        "cmake",
        "projects/hipdnn",
        "dnn-providers",
    )
    _git("-C", str(checkout), "checkout", "--quiet", "main")
    assert not (checkout / "shared/ctest/TestCategories.cmake").exists()

    _setup(module, root).ensure_rocm_libraries_checkout()

    assert (checkout / "shared/ctest/TestCategories.cmake").is_file()
    assert not (checkout / "shared/tensile/large_unneeded_file.txt").exists()

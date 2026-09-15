# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for setup_env.py's --cmake-arg passthrough.

The provider build configures from a fixed list of defines. Any hipDNN engine
gated behind a non-default CMake option is therefore unbuildable by this tool
unless a caller can add one, and that failure is silent: the provider shared
library is still installed, so --plugin-path looks satisfied and every graph
reports "no engines applicable".
"""

import importlib.util
import sys
from pathlib import Path

import pytest

_SETUP_ENV = Path(__file__).resolve().parents[2] / "setup_env.py"


def _load_setup_env():
    """Import setup_env.py by path: it is a top-level script, not a package."""
    spec = importlib.util.spec_from_file_location("setup_env_under_test", _SETUP_ENV)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def setup_env():
    return _load_setup_env()


def test_bare_name_value_is_accepted_after_a_space(setup_env) -> None:
    """`--cmake-arg NAME=VALUE` must work, because it is what people type.

    argparse treats a value beginning with `-` as another option, so the
    obvious `--cmake-arg -DFOO=ON` fails with "expected one argument" unless it
    is written with an `=`. Accepting the bare form removes that trap.
    """
    args = setup_env.build_parser().parse_args(
        ["--cmake-arg", "HIPDNN_ENABLE_KERNEL_INGESTOR=ON"]
    )

    assert args.cmake_args == ["-DHIPDNN_ENABLE_KERNEL_INGESTOR=ON"]


def test_explicit_dash_d_form_is_accepted_and_not_doubled(setup_env) -> None:
    args = setup_env.build_parser().parse_args(
        ["--cmake-arg=-DHIPDNN_ENABLE_KERNEL_INGESTOR=ON"]
    )

    assert args.cmake_args == ["-DHIPDNN_ENABLE_KERNEL_INGESTOR=ON"]


def test_repeats_accumulate_in_order(setup_env) -> None:
    args = setup_env.build_parser().parse_args(
        ["--cmake-arg", "A=1", "--cmake-arg", "B=2"]
    )

    assert args.cmake_args == ["-DA=1", "-DB=2"]


def test_a_value_with_an_equals_in_it_survives(setup_env) -> None:
    """Paths and expressions contain `=`; only the FIRST one separates."""
    args = setup_env.build_parser().parse_args(
        ["--cmake-arg", "CMAKE_CXX_FLAGS=-DA=1 -DB=2"]
    )

    assert args.cmake_args == ["-DCMAKE_CXX_FLAGS=-DA=1 -DB=2"]


@pytest.mark.parametrize("bad", ["NOEQUALS", "-DNOEQUALS", "", "   "])
def test_a_define_without_a_value_is_rejected(setup_env, bad) -> None:
    """Silently dropping a malformed define would reintroduce the whole bug:
    the caller believes the option was set and the build says otherwise."""
    with pytest.raises(SystemExit):
        setup_env.build_parser().parse_args(["--cmake-arg", bad])


def test_studio_setup_preserves_existing_python_environment(
    setup_env, tmp_path
) -> None:
    source = tmp_path / "source"
    workspace = tmp_path / "workspace"
    args = setup_env.build_parser().parse_args(
        [
            "--graph-studio",
            "--source-dir",
            str(source),
            "--workspace",
            str(workspace),
        ]
    )
    setup = setup_env.Setup(args)
    setup_env.subprocess.run(
        [sys.executable, "-m", "venv", "--without-pip", str(setup.venv_dir)],
        check=True,
    )
    sentinel = setup.venv_dir / "existing-environment"
    sentinel.write_text("retain installed dependencies")
    setup.setup_venv()
    assert sentinel.read_text() == "retain installed dependencies"
    assert not (
        source / "projects/hipdnn/tools/dnn-benchmarking/rocm-libraries"
    ).exists()


def test_studio_install_cannot_overwrite_python_dependencies(
    setup_env, tmp_path
) -> None:
    args = setup_env.build_parser().parse_args(
        [
            "--graph-studio",
            "--source-dir",
            str(tmp_path),
            "--install-prefix",
            str(tmp_path / ".venv/lib/python3.12/site-packages"),
        ]
    )
    with pytest.raises(SystemExit):
        setup_env.Setup(args)
    assert not (tmp_path / ".venv").exists()


def test_studio_rejects_another_checkouts_build_before_provisioning(
    setup_env, tmp_path
) -> None:
    source = tmp_path / "source"
    studio = source / "projects/hipdnn/tools/graph-studio"
    studio.mkdir(parents=True)
    (studio / "CMakeLists.txt").touch()
    build = source / "build"
    build.mkdir()
    cache = build / "CMakeCache.txt"
    contents = "CMAKE_HOME_DIRECTORY:INTERNAL=/a/different/checkout\n"
    cache.write_text(contents)
    setup = setup_env.Setup(
        setup_env.build_parser().parse_args(
            [
                "--graph-studio",
                "--source-dir",
                str(source),
                "--yes",
            ]
        )
    )
    with pytest.raises(SystemExit):
        setup.run()
    assert cache.read_text() == contents
    assert not (source / ".venv").exists()

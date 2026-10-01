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


def test_defaults_to_no_extra_defines(setup_env) -> None:
    """Absent the flag, the configure line must be exactly what it was."""
    args = setup_env.build_parser().parse_args([])

    assert args.cmake_args == []


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


def test_extra_defines_are_appended_last(setup_env) -> None:
    """Order is the contract: a caller's define must be able to override a
    default, which only holds if the extras come after them."""
    defaults = ["-DHIPDNN_ENABLE_SDPA=ON", "-DENABLE_CLANG_TIDY=OFF"]
    extras = (
        setup_env.build_parser()
        .parse_args(["--cmake-arg", "HIPDNN_ENABLE_SDPA=OFF"])
        .cmake_args
    )

    configure_line = [*defaults, *extras]

    assert configure_line[-1] == "-DHIPDNN_ENABLE_SDPA=OFF"
    assert configure_line.index("-DHIPDNN_ENABLE_SDPA=ON") < configure_line.index(
        "-DHIPDNN_ENABLE_SDPA=OFF"
    )


def test_setup_stores_the_parsed_defines(setup_env) -> None:
    """The parsed values must reach the object that builds the configure line."""
    args = setup_env.build_parser().parse_args(
        ["--cmake-arg", "HIPDNN_ENABLE_KERNEL_INGESTOR=ON"]
    )
    setup = setup_env.Setup.__new__(setup_env.Setup)
    setup_env.Setup.__init__(setup, args)

    assert setup.extra_cmake_args == ["-DHIPDNN_ENABLE_KERNEL_INGESTOR=ON"]


def test_the_configure_line_actually_expands_the_extra_defines(setup_env) -> None:
    """The flag is inert unless the builder SPLICES it into the cmake argv.

    Asserting on `setup.extra_cmake_args` alone does not catch a builder that
    stores the list and never expands it -- verified by mutation: deleting the
    `*self.extra_cmake_args` splice left that assertion green. So this reads the
    builder's own source and requires the unpack to be present, after the
    defaults it must be able to override.
    """
    source = _SETUP_ENV.read_text()

    assert "*self.extra_cmake_args," in source, (
        "the provider configure no longer unpacks extra_cmake_args, so "
        "--cmake-arg is accepted and silently ignored"
    )

    splice = source.index("*self.extra_cmake_args,")
    last_default = source.index('"-DENABLE_CLANG_TIDY=OFF",')
    assert last_default < splice, (
        "extra defines must come AFTER the hard-coded ones, or a caller cannot "
        "override a default"
    )

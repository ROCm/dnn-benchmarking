# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Behavioural tests for setup_env.py (no GPU, no network).

setup_env.py is a top-level script, not a package, so it is imported by path.
"""

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_SETUP_ENV = Path(__file__).resolve().parents[2] / "setup_env.py"


@pytest.fixture(scope="module")
def setup_env():
    spec = importlib.util.spec_from_file_location("setup_env_under_test", _SETUP_ENV)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _setup(setup_env, tmp_path, *argv):
    args = setup_env.build_parser().parse_args(
        ["--workspace", str(tmp_path / "ws"), "--gpu-arch", "gfx90a", *argv]
    )
    return setup_env.Setup(args)


class _Stdin:
    def __init__(self, tty: bool) -> None:
        self.tty = tty

    def isatty(self) -> bool:
        return self.tty


# --- --cmake-arg -------------------------------------------------------------


@pytest.mark.parametrize(
    "argv, expected",
    [
        # The space form is what people type; argparse would reject a bare
        # `-DFOO=ON` value, so NAME=VALUE must work.
        (
            ["--cmake-arg", "HIPDNN_ENABLE_KERNEL_INGESTOR=ON"],
            ["-DHIPDNN_ENABLE_KERNEL_INGESTOR=ON"],
        ),
        (
            ["--cmake-arg=-DHIPDNN_ENABLE_KERNEL_INGESTOR=ON"],
            ["-DHIPDNN_ENABLE_KERNEL_INGESTOR=ON"],
        ),
        (["--cmake-arg", "A=1", "--cmake-arg", "B=2"], ["-DA=1", "-DB=2"]),
        # Only the first `=` separates name from value.
        (
            ["--cmake-arg", "CMAKE_CXX_FLAGS=-DA=1 -DB=2"],
            ["-DCMAKE_CXX_FLAGS=-DA=1 -DB=2"],
        ),
    ],
)
def test_cmake_arg_normalises_to_defines(setup_env, argv, expected) -> None:
    assert setup_env.build_parser().parse_args(argv).cmake_args == expected


@pytest.mark.parametrize("bad", ["NOEQUALS", "-DNOEQUALS", "", "   "])
def test_cmake_arg_without_a_value_is_rejected(setup_env, bad) -> None:
    with pytest.raises(SystemExit):
        setup_env.build_parser().parse_args(["--cmake-arg", bad])


@pytest.mark.parametrize(
    "clean, recorded_prefix, build_dir_kept",
    [(False, "prefix", True), (True, "prefix", False), (False, "other", False)],
    ids=["same-prefix", "clean", "other-prefix"],
)
def test_superbuild_configure_lets_extra_defines_override_defaults(
    setup_env, tmp_path, monkeypatch, clean, recorded_prefix, build_dir_kept
) -> None:
    rocm_libraries = tmp_path / "rocm-libraries"
    cache = rocm_libraries / "build" / "CMakeCache.txt"
    cache.parent.mkdir(parents=True)
    cache.write_text("")
    toolchain = tmp_path / "toolchain"
    (toolchain / "lib").mkdir(parents=True)
    (toolchain / "lib" / "libamd_comgr.so.3").write_text("")
    record = {
        "install_prefix": str(tmp_path / "prefix"),
        "toolchain_prefix": str(toolchain),
    }
    # A build dir configured by another workspace keeps that workspace's
    # compiler in its CMakeCache, so it must not be reused.
    (cache.parent / setup_env.BUILD_RECORD).write_text(
        json.dumps({**record, "install_prefix": str(tmp_path / recorded_prefix)})
    )
    commands = []
    monkeypatch.setattr(setup_env, "ROCM_LIBRARIES_DIR", rocm_libraries)
    monkeypatch.setattr(
        setup_env, "run", lambda cmd, **kwargs: commands.append(list(cmd))
    )
    monkeypatch.setattr(setup_env, "require_working_cmake", lambda: "cmake")
    monkeypatch.setattr(setup_env.shutil, "which", lambda name: name)
    argv = ["--cmake-arg", "HIPDNN_ENABLE_SDPA=OFF"] + (["--clean"] if clean else [])
    setup = _setup(setup_env, tmp_path, *argv)
    monkeypatch.setattr(setup, "_build_env", lambda: {})

    setup.build_superbuild(record["install_prefix"], record["toolchain_prefix"])

    configure = commands[0]
    assert configure[-1] == "-DHIPDNN_ENABLE_SDPA=OFF"
    assert configure.index("-DHIPDNN_ENABLE_SDPA=ON") < configure.index(
        "-DHIPDNN_ENABLE_SDPA=OFF"
    )
    # Incremental by default; --clean or other prefixes start the build over.
    assert cache.exists() == build_dir_kept
    # The next run with these prefixes reuses the directory.
    assert json.loads((cache.parent / setup_env.BUILD_RECORD).read_text()) == record


# --- confirmation ------------------------------------------------------------


@pytest.mark.parametrize("answer", ["", "y", "Y", "yes", " YES "])
def test_confirm_proceeds_on_yes_or_enter(
    setup_env, tmp_path, monkeypatch, answer
) -> None:
    monkeypatch.setattr(sys, "stdin", _Stdin(tty=True))
    monkeypatch.setattr("builtins.input", lambda prompt: answer)

    _setup(setup_env, tmp_path).confirm_build()


@pytest.mark.parametrize("answer", ["n", "no", "q", "nope", "yess"])
def test_confirm_aborts_on_anything_else(
    setup_env, tmp_path, monkeypatch, answer
) -> None:
    monkeypatch.setattr(sys, "stdin", _Stdin(tty=True))
    monkeypatch.setattr("builtins.input", lambda prompt: answer)

    with pytest.raises(SystemExit) as exc:
        _setup(setup_env, tmp_path).confirm_build()
    assert exc.value.code == 0


def _no_prompt(prompt):
    raise AssertionError("prompted")


def test_confirm_without_a_terminal_requires_yes_flag(
    setup_env, tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(sys, "stdin", _Stdin(tty=False))
    monkeypatch.setattr("builtins.input", _no_prompt)

    with pytest.raises(SystemExit) as exc:
        _setup(setup_env, tmp_path).confirm_build()
    assert exc.value.code == 1

    _setup(setup_env, tmp_path, "-y").confirm_build()


# --- GPU arch detection ------------------------------------------------------

_ROCMINFO_MI300 = """\
Agent 1
  Name:                    AMD EPYC 9654 96-Core Processor
Agent 2
  Name:                    gfx942
  Marketing Name:          AMD Instinct MI300X
  ISA Info:
    ISA 1
      Name:                    amdgcn-amd-amdhsa--gfx942:sramecc+:xnack-
    ISA 2
      Name:                    amdgcn-amd-amdhsa--gfx9-4-generic:sramecc+:xnack-
"""


def _fake_tools(setup_env, monkeypatch, outputs):
    monkeypatch.setattr(
        setup_env.shutil, "which", lambda tool: tool if tool in outputs else None
    )

    def fake_run(cmd, **kwargs):
        assert kwargs.get("timeout"), "detection must not wait forever"
        out = outputs[cmd[0]]
        if isinstance(out, BaseException):
            raise out
        return subprocess.CompletedProcess(cmd, 0, stdout=out, stderr="")

    monkeypatch.setattr(setup_env.subprocess, "run", fake_run)


@pytest.mark.parametrize(
    "outputs, expected",
    [
        ({"rocm_agent_enumerator": "gfx000\ngfx1151\n"}, "gfx1151"),
        ({"rocm_agent_enumerator": "gfx000\ngfx90a\ngfx90a\n"}, "gfx90a"),
        # No GPU agent from the enumerator: rocminfo decides, and a generic
        # ISA name is not mistaken for an arch.
        ({"rocm_agent_enumerator": "gfx000\n", "rocminfo": _ROCMINFO_MI300}, "gfx942"),
        (
            {
                "rocm_agent_enumerator": subprocess.TimeoutExpired(
                    "rocm_agent_enumerator", 30
                ),
                "rocminfo": _ROCMINFO_MI300,
            },
            "gfx942",
        ),
        ({"rocminfo": "  Name: gfx000\n"}, ""),
        ({}, ""),
    ],
)
def test_detect_gpu_arch(setup_env, monkeypatch, outputs, expected) -> None:
    _fake_tools(setup_env, monkeypatch, outputs)

    assert setup_env.Setup._detect_gpu_arch() == expected


def test_detect_gpu_arch_refuses_to_pick_among_several(setup_env, monkeypatch) -> None:
    _fake_tools(
        setup_env, monkeypatch, {"rocm_agent_enumerator": "gfx000\ngfx90a\ngfx942\n"}
    )

    with pytest.raises(SystemExit) as exc:
        setup_env.Setup._detect_gpu_arch()
    assert exc.value.code == 1


# --- rocprofiler library unification -----------------------------------------


def _lib_dirs(tmp_path):
    core = tmp_path / "core" / "lib"
    devel = tmp_path / "devel" / "lib"
    core.mkdir(parents=True)
    devel.mkdir(parents=True)
    sdk = core / "librocprofiler-sdk.so.1"
    sdk.write_bytes(b"sdk")
    os.link(sdk, devel / sdk.name)
    (core / "librocprofiler-register.so.0").write_bytes(b"core build")
    (devel / "librocprofiler-register.so.0").write_bytes(b"other build")
    return core, devel


def test_unify_relinks_only_same_file_duplicates(setup_env, tmp_path) -> None:
    probe = tmp_path / "symlink-probe"
    try:
        probe.symlink_to(tmp_path)
    except OSError:
        pytest.skip("symlinks unavailable")
    core, devel = _lib_dirs(tmp_path)

    assert setup_env.unify_rocprofiler_libs(core, devel) == 1

    dup = devel / "librocprofiler-sdk.so.1"
    assert dup.is_symlink()
    assert dup.resolve() == (core / "librocprofiler-sdk.so.1").resolve()
    other = devel / "librocprofiler-register.so.0"
    assert not other.is_symlink()
    assert other.read_bytes() == b"other build"
    assert setup_env.unify_rocprofiler_libs(core, devel) == 0


def test_unify_keeps_the_library_when_symlinking_fails(
    setup_env, tmp_path, monkeypatch
) -> None:
    core, devel = _lib_dirs(tmp_path)

    def refuse(self, target):
        raise OSError("symlinks not permitted")

    monkeypatch.setattr(Path, "symlink_to", refuse)

    assert setup_env.unify_rocprofiler_libs(core, devel) == 0
    dup = devel / "librocprofiler-sdk.so.1"
    assert not dup.is_symlink()
    assert dup.read_bytes() == b"sdk"
    assert sorted(p.name for p in devel.iterdir()) == [
        "librocprofiler-register.so.0",
        "librocprofiler-sdk.so.1",
    ]


# --- activate.local ----------------------------------------------------------


@pytest.mark.skipif(
    sys.platform == "win32" or shutil.which("bash") is None,
    reason="activate.local is a bash script written on Linux only",
)
def test_activate_local_is_idempotent(setup_env, tmp_path) -> None:
    setup = _setup(setup_env, tmp_path)
    activate = setup.venv_dir / "bin" / "activate"
    activate.parent.mkdir(parents=True)
    activate.write_text("# venv activate\n")
    prefix = tmp_path / "rocm prefix"
    lib = prefix / "lib"
    lib.mkdir(parents=True)

    setup.write_activate_local(str(prefix), (str(lib),))
    setup.write_activate_local(str(prefix), (str(lib),))

    assert activate.read_text().count("activate.local") == 1
    ld_path = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; source "$1"; printf %s "$LD_LIBRARY_PATH"',
            "_",
            str(activate),
        ],
        env={"PATH": os.environ["PATH"], "LD_LIBRARY_PATH": "/usr/lib"},
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert ld_path == f"{lib}:/usr/lib"


# --- rocm-libraries checkout -------------------------------------------------


def _git(*args, cwd=None):
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@t",
            "-c",
            "commit.gpgsign=false",
            *args,
        ],
        cwd=cwd,
        check=True,
        capture_output=True,
    )


@pytest.fixture
def rocm_libraries(setup_env, tmp_path, monkeypatch):
    """Point setup_env at a local bare origin; return the checkout path."""
    if shutil.which("git") is None:
        pytest.skip("git unavailable")
    src = tmp_path / "src"
    for rel in (
        "CMakePresets.json",
        "cmake/x.cmake",
        "projects/hipdnn/CMakeLists.txt",
        "dnn-providers/CMakeLists.txt",
        "projects/other/big.bin",
    ):
        (src / rel).parent.mkdir(parents=True, exist_ok=True)
        (src / rel).write_text(rel)
    _git("init", "-q", "-b", "main", cwd=src)
    _git("add", ".", cwd=src)
    _git("commit", "-q", "-m", "init", cwd=src)
    origin = tmp_path / "origin.git"
    _git("clone", "-q", "--bare", str(src), str(origin))

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".gitmodules").write_text(
        '[submodule "rocm-libraries"]\n'
        "\tpath = rocm-libraries\n"
        f"\turl = {origin.as_uri()}\n"
        "\tbranch = main\n"
    )
    monkeypatch.setattr(setup_env, "SCRIPT_DIR", repo)
    monkeypatch.setattr(setup_env, "ROCM_LIBRARIES_DIR", repo / "rocm-libraries")
    return repo / "rocm-libraries"


def test_failed_fetch_leaves_no_checkout_to_reuse(
    setup_env, tmp_path, rocm_libraries
) -> None:
    setup = _setup(setup_env, tmp_path)
    # Branch names bypass the --rocm-libraries-ref SHA check; the local
    # origin has no fixed commit id to pin.
    setup.rocm_libraries_ref = "no-such-ref"

    with pytest.raises(subprocess.CalledProcessError):
        setup.ensure_rocm_libraries_checkout()

    assert not rocm_libraries.exists()
    assert not rocm_libraries.with_name("rocm-libraries.partial").exists()

    # The retry fetches instead of reusing a half-made checkout.
    setup.rocm_libraries_ref = "main"
    setup.ensure_rocm_libraries_checkout()
    assert (rocm_libraries / "CMakePresets.json").is_file()
    assert (rocm_libraries / "projects" / "hipdnn" / "CMakeLists.txt").is_file()
    assert not (rocm_libraries / "projects" / "other").exists()


def test_non_git_directory_is_kept_without_confirmation(
    setup_env, tmp_path, monkeypatch, rocm_libraries
) -> None:
    rocm_libraries.mkdir()
    (rocm_libraries / "notes.txt").write_text("mine")
    monkeypatch.setattr(sys, "stdin", _Stdin(tty=False))
    setup = _setup(setup_env, tmp_path)
    setup.rocm_libraries_ref = "main"

    with pytest.raises(SystemExit):
        setup.ensure_rocm_libraries_checkout()

    assert (rocm_libraries / "notes.txt").read_text() == "mine"


def test_rocm_libraries_ref_must_be_a_full_sha(setup_env) -> None:
    parser = setup_env.build_parser()
    args = parser.parse_args(["--rocm-libraries-ref", "A" * 40])
    assert args.rocm_libraries_ref == "a" * 40
    for bad in ("a" * 12, "main"):
        with pytest.raises(SystemExit):
            parser.parse_args(["--rocm-libraries-ref", bad])


def _head(repo: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def test_rocm_libraries_ref_pins_the_fetched_commit(
    setup_env, tmp_path, capsys, rocm_libraries
) -> None:
    src = tmp_path / "src"
    pinned = _head(src)
    # Move the origin branch past the pin: only the SHA fetches `pinned`.
    (src / "projects/hipdnn/new.txt").write_text("new")
    _git("add", ".", cwd=src)
    _git("commit", "-q", "-m", "next", cwd=src)
    _git("fetch", "-q", str(src), "main:main", cwd=tmp_path / "origin.git")
    tip = _head(src)

    _setup(
        setup_env, tmp_path, "--rocm-libraries-ref", pinned
    ).ensure_rocm_libraries_checkout()

    assert _head(rocm_libraries) == pinned
    assert "WARNING" not in capsys.readouterr().err

    # A reused checkout is not refetched, but one off the pin is reported.
    _setup(
        setup_env, tmp_path, "--rocm-libraries-ref", tip
    ).ensure_rocm_libraries_checkout()
    assert _head(rocm_libraries) == pinned
    assert f"not the pinned {tip[:12]}" in capsys.readouterr().err


# --- venv reuse ----------------------------------------------------------------


def _fake_venv(setup_env, venv_dir: Path) -> None:
    python = setup_env.venv_python(venv_dir)
    python.parent.mkdir(parents=True, exist_ok=True)
    python.write_text("")
    (venv_dir / "bin").mkdir(exist_ok=True)
    (venv_dir / "bin" / "activate").write_text("# venv activate\n")


@pytest.mark.parametrize("clean", [False, True])
def test_setup_venv_reuses_unless_clean(
    setup_env, tmp_path, monkeypatch, clean
) -> None:
    setup = _setup(setup_env, tmp_path, *(["--clean"] if clean else []))
    _fake_venv(setup_env, setup.venv_dir)
    marker = setup.venv_dir / "marker"
    marker.write_text("old venv")
    setup.installed_torch_mode = "rocm"
    created = []

    def fake_run(cmd, **kwargs):
        created.append(cmd)
        _fake_venv(setup_env, setup.venv_dir)

    monkeypatch.setattr(setup_env, "run", fake_run)

    setup.setup_venv()

    assert marker.exists() is not clean
    assert bool(created) is clean
    assert setup.installed_torch_mode == ("missing" if clean else "rocm")


def test_clean_with_existing_torch_mode_is_a_usage_error(setup_env) -> None:
    with pytest.raises(SystemExit) as exc:
        setup_env.main(["--clean", "--torch-mode", "existing"])
    assert exc.value.code == 2


def _reused_rocm_venv(setup_env, tmp_path, monkeypatch, *argv):
    """A Setup over a venv that setup filled with gfx90a ROCm torch."""
    setup = _setup(setup_env, tmp_path, *argv)
    setup.venv_dir.mkdir(parents=True)
    (setup.venv_dir / setup_env.TORCH_RECORD).write_text(
        json.dumps(
            {"torch_index_url": setup_env.ROCM_TORCH_INDEX_URL, "gpu_arch": "gfx90a"}
        )
    )
    setup.installed_torch_mode = "rocm"
    pip_calls = []
    monkeypatch.setattr(setup, "pip", lambda *a, **kw: pip_calls.append(a))
    return setup, pip_calls


@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["--gpu-arch", "gfx90a"],
        ["--torch-index-url", "https://nightly.repo.amd.com/rocm/whl-next/"],
    ],
    ids=["no-flags", "same-arch", "same-index"],
)
def test_reused_venv_keeps_torch_when_flags_match(
    setup_env, tmp_path, monkeypatch, argv
) -> None:
    setup, pip_calls = _reused_rocm_venv(setup_env, tmp_path, monkeypatch, *argv)
    setup.install_torch()
    assert pip_calls == []


@pytest.mark.parametrize(
    "argv",
    [["--gpu-arch", "gfx942"], ["--torch-index-url", "https://example.invalid/"]],
    ids=["other-arch", "other-index"],
)
def test_reused_venv_refuses_a_different_explicit_torch_source(
    setup_env, tmp_path, monkeypatch, capsys, argv
) -> None:
    setup, pip_calls = _reused_rocm_venv(setup_env, tmp_path, monkeypatch, *argv)
    with pytest.raises(SystemExit):
        setup.install_torch()
    assert pip_calls == []
    assert "--clean" in capsys.readouterr().err


def test_fresh_torch_install_records_its_source(
    setup_env, tmp_path, monkeypatch
) -> None:
    setup = _setup(setup_env, tmp_path, "--gpu-arch", "gfx942")
    setup.venv_dir.mkdir(parents=True)
    monkeypatch.setattr(setup, "pip", lambda *a, **kw: None)
    monkeypatch.setattr(setup, "get_torch_mode", lambda: "rocm")

    setup.install_torch()

    # The next run on this venv keeps torch for the same arch only.
    same = _setup(setup_env, tmp_path, "--gpu-arch", "gfx942")
    same.installed_torch_mode = "rocm"
    same.install_torch()
    other = _setup(setup_env, tmp_path, "--gpu-arch", "gfx90a")
    other.installed_torch_mode = "rocm"
    with pytest.raises(SystemExit):
        other.install_torch()


@pytest.mark.parametrize("installed", ["rocm", "cuda", "cpu"])
def test_torch_mode_none_refuses_a_venv_with_torch(
    setup_env, tmp_path, capsys, installed
) -> None:
    # Building for "none" on a venv with ROCm wheel torch would mix the wheel
    # and system ROCm; with CUDA torch it would skip the hipDNN build.
    setup = _setup(setup_env, tmp_path, "--torch-mode", "none")
    setup.installed_torch_mode = installed
    with pytest.raises(SystemExit):
        setup.install_torch()
    assert "--clean" in capsys.readouterr().err

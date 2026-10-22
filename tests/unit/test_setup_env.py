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


_BASE_ARGV = ["--gpu-arch", "gfx942", "--cmake-arg", "HIPDNN_ENABLE_SDPA=OFF"]


@pytest.mark.parametrize(
    "rerun_argv, rerun_prefix, build_dir_kept",
    [
        (_BASE_ARGV, "prefix", True),
        ([*_BASE_ARGV, "--clean"], "prefix", False),
        # A build dir configured by another workspace keeps that workspace's
        # compiler in its CMakeCache, so it must not be reused.
        (_BASE_ARGV, "other", False),
        # CMakeCache keeps a -D define after it drops off the configure line,
        # whether it came from --cmake-arg or from setup's own defines (here
        # GPU_TARGETS, gone when no --gpu-arch is given and none is detected).
        (_BASE_ARGV[:2], "prefix", False),
        (_BASE_ARGV[2:], "prefix", False),
    ],
    ids=["same-config", "clean", "other-prefix", "other-extra", "other-default"],
)
def test_superbuild_reuses_its_build_dir_only_for_the_same_configure_line(
    setup_env, tmp_path, monkeypatch, rerun_argv, rerun_prefix, build_dir_kept
) -> None:
    rocm_libraries = tmp_path / "rocm-libraries"
    cache = rocm_libraries / "build" / "CMakeCache.txt"
    toolchain = tmp_path / "toolchain"
    (toolchain / "lib").mkdir(parents=True)
    (toolchain / "lib" / "libamd_comgr.so.3").write_text("")
    commands = []
    monkeypatch.setattr(setup_env, "ROCM_LIBRARIES_DIR", rocm_libraries)
    monkeypatch.setattr(setup_env.Setup, "_detect_gpu_arch", staticmethod(lambda: ""))
    monkeypatch.setattr(
        setup_env, "run", lambda cmd, **kwargs: commands.append(list(cmd))
    )
    monkeypatch.setattr(setup_env, "require_working_cmake", lambda: "cmake")
    monkeypatch.setattr(setup_env.shutil, "which", lambda name: name)

    def build(*argv, prefix="prefix"):
        # Not _setup: it always passes --gpu-arch.
        args = setup_env.build_parser().parse_args(
            ["--workspace", str(tmp_path / "ws"), *argv]
        )
        setup = setup_env.Setup(args)
        monkeypatch.setattr(setup, "_build_env", lambda: {})
        commands.clear()
        setup.build_superbuild(str(tmp_path / prefix), str(toolchain))
        return commands[0]

    configure = build(*_BASE_ARGV)
    # --cmake-arg defines come LAST so they override setup's defaults.
    assert configure[-1] == "-DHIPDNN_ENABLE_SDPA=OFF"
    assert configure.index("-DHIPDNN_ENABLE_SDPA=ON") < configure.index(
        "-DHIPDNN_ENABLE_SDPA=OFF"
    )
    cache.write_text("")

    configure = build(*rerun_argv, prefix=rerun_prefix)

    # Incremental by default; --clean or any other configure argument starts
    # the build over.
    assert cache.exists() == build_dir_kept
    # The record is the configure line just run, so the next identical run
    # reuses the directory.
    record = json.loads((cache.parent / setup_env.BUILD_RECORD).read_text())
    assert record == {"configure_args": configure[1:]}


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


def _raise(error):
    def prompt(_text):
        raise error

    return prompt


def test_confirm_without_a_terminal_requires_yes_flag(
    setup_env, tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(sys, "stdin", _Stdin(tty=False))
    monkeypatch.setattr("builtins.input", _no_prompt)

    with pytest.raises(SystemExit) as exc:
        _setup(setup_env, tmp_path).confirm_build()
    assert exc.value.code == 1

    _setup(setup_env, tmp_path, "-y").confirm_build()


def test_confirm_without_any_stdin_requires_yes_flag(
    setup_env, tmp_path, monkeypatch, capsys
) -> None:
    """Closing fd 0 (nohup, srun, some CI) leaves sys.stdin as None."""
    monkeypatch.setattr(sys, "stdin", None)
    monkeypatch.setattr("builtins.input", _no_prompt)

    with pytest.raises(SystemExit) as exc:
        _setup(setup_env, tmp_path).confirm_build()
    assert exc.value.code == 1
    assert "-y" in capsys.readouterr().err


@pytest.mark.parametrize(
    "error",
    [
        EOFError(),
        OSError(5, "Input/output error"),
        RuntimeError("input(): lost sys.stdin"),
    ],
)
def test_confirm_reports_a_stdin_lost_after_the_check(
    setup_env, tmp_path, monkeypatch, capsys, error
) -> None:
    """A terminal can close between isatty() and the read."""
    monkeypatch.setattr(sys, "stdin", _Stdin(tty=True))
    monkeypatch.setattr("builtins.input", _raise(error))

    with pytest.raises(SystemExit) as exc:
        _setup(setup_env, tmp_path).confirm_build()
    assert exc.value.code == 1
    assert "-y" in capsys.readouterr().err


def test_confirm_does_not_swallow_an_unrelated_runtime_error(
    setup_env, tmp_path, monkeypatch
) -> None:
    """Only a lost stdin becomes the -y message; other failures must surface."""
    monkeypatch.setattr(sys, "stdin", _Stdin(tty=True))
    monkeypatch.setattr("builtins.input", _raise(RuntimeError("readline is broken")))

    with pytest.raises(RuntimeError, match="readline is broken"):
        _setup(setup_env, tmp_path).confirm_build()


@pytest.mark.skipif(os.name != "posix", reason="closing fd 0 requires POSIX")
def test_setup_without_stdin_fails_before_creating_a_venv(tmp_path) -> None:
    closed_probe = (
        "import errno, os\n"
        "try:\n"
        "    os.fstat(0)\n"
        "except OSError as exc:\n"
        "    assert exc.errno == errno.EBADF\n"
        "else:\n"
        "    raise AssertionError('fd 0 is open')\n"
        "print('fd 0 closed')\n"
    )
    control = subprocess.run(
        [sys.executable, "-c", closed_probe],
        preexec_fn=lambda: os.close(0),
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert control.returncode == 0, control.stderr
    assert control.stdout.strip() == "fd 0 closed"

    result = subprocess.run(
        [sys.executable, str(_SETUP_ENV), "--workspace", str(tmp_path / "ws")],
        preexec_fn=lambda: os.close(0),
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 1
    assert "Traceback" not in result.stderr
    assert "-y" in result.stderr
    assert not (tmp_path / "ws" / ".venv").exists()


@pytest.mark.skipif(os.name != "nt", reason="Windows closed-fd 0 handling")
def test_windows_setup_with_closed_fd0_requires_yes(tmp_path) -> None:
    # Run setup in the same process that proves its CRT descriptor is closed.
    script = (
        "import errno, os, runpy, sys\n"
        "try:\n"
        "    os.close(0)\n"
        "except OSError as exc:\n"
        "    assert exc.errno == errno.EBADF\n"
        "try:\n"
        "    os.fstat(0)\n"
        "except OSError as exc:\n"
        "    assert exc.errno == errno.EBADF\n"
        "else:\n"
        "    raise AssertionError('fd 0 is open')\n"
        "print('fd 0 closed', flush=True)\n"
        "sys.argv = [sys.argv[1], '--workspace', sys.argv[2]]\n"
        "runpy.run_path(sys.argv[0], run_name='__main__')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script, str(_SETUP_ENV), str(tmp_path / "ws")],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 1
    assert "fd 0 closed" in result.stdout
    assert "Traceback" not in result.stderr
    assert "-y" in result.stderr
    assert not (tmp_path / "ws" / ".venv").exists()


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
        "shared/ctest/TestCategories.cmake",
        "shared/tensile/big.bin",
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


def test_sparse_checkout_takes_only_the_shared_subtree_configure_reads(
    setup_env, tmp_path, rocm_libraries
) -> None:
    """hipDNN's configure reads shared/ctest. The rest of shared/ is dead weight."""
    setup = _setup(setup_env, tmp_path)
    setup.rocm_libraries_ref = "main"

    setup.ensure_rocm_libraries_checkout()

    assert (rocm_libraries / "shared/ctest/TestCategories.cmake").is_file()
    assert not (rocm_libraries / "shared/tensile").exists()
    assert (rocm_libraries / "cmake/x.cmake").is_file()
    assert (rocm_libraries / "CMakePresets.json").is_file()


def test_an_older_sparse_checkout_gains_the_root_dirs_added_since(
    setup_env, tmp_path, rocm_libraries
) -> None:
    setup = _setup(setup_env, tmp_path)
    setup.rocm_libraries_ref = "main"
    setup.ensure_rocm_libraries_checkout()
    # What a setup_env.py from before cmake/ and shared/ctest/ staged.
    _git(
        "sparse-checkout",
        "set",
        "projects/hipdnn",
        "dnn-providers",
        cwd=rocm_libraries,
    )
    assert not (rocm_libraries / "shared/ctest").exists()

    setup.ensure_rocm_libraries_checkout()

    assert (rocm_libraries / "cmake/x.cmake").is_file()
    assert (rocm_libraries / "shared/ctest/TestCategories.cmake").is_file()
    assert not (rocm_libraries / "shared/tensile").exists()


def test_a_full_checkout_is_not_made_sparse(
    setup_env, tmp_path, rocm_libraries
) -> None:
    """`git submodule update --init` gives a full checkout; leave it alone."""
    _git("clone", "-q", str(tmp_path / "origin.git"), str(rocm_libraries))
    setup = _setup(setup_env, tmp_path)
    setup.rocm_libraries_ref = "main"

    setup.ensure_rocm_libraries_checkout()

    assert (rocm_libraries / "shared/tensile/big.bin").is_file()
    assert (rocm_libraries / "projects/other/big.bin").is_file()
    assert not setup._rocm_libraries_is_sparse()


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


def _reused_venv(setup_env, tmp_path, monkeypatch, mode, *argv, record=True):
    """A Setup over a venv that setup filled with ``mode`` torch from its
    default index (gfx90a for rocm). ``record=False`` drops the TORCH_RECORD,
    as on a venv set up before setup wrote one."""
    setup = _setup(setup_env, tmp_path, "--torch-mode", mode, *argv)
    setup.venv_dir.mkdir(parents=True)
    if record:
        index_url = {
            "rocm": setup_env.ROCM_TORCH_INDEX_URL,
            "cpu": "https://download.pytorch.org/whl/cpu",
            "cuda": "",
        }[mode]
        arch = "gfx90a" if mode == "rocm" else ""
        (setup.venv_dir / setup_env.TORCH_RECORD).write_text(
            json.dumps({"torch_index_url": index_url, "gpu_arch": arch})
        )
    setup.installed_torch_mode = mode
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
    setup, pip_calls = _reused_venv(setup_env, tmp_path, monkeypatch, "rocm", *argv)
    setup.install_torch()
    assert pip_calls == []


@pytest.mark.parametrize(
    "mode, argv",
    [
        ("rocm", ["--gpu-arch", "gfx942"]),
        ("rocm", ["--torch-index-url", "https://example.invalid/"]),
        ("cpu", ["--torch-index-url", "https://example.invalid/"]),
        ("cuda", ["--torch-index-url", "https://example.invalid/"]),
    ],
    ids=["rocm-other-arch", "rocm-other-index", "cpu-other-index", "cuda-other-index"],
)
def test_reused_venv_refuses_a_different_explicit_torch_source(
    setup_env, tmp_path, monkeypatch, capsys, mode, argv
) -> None:
    setup, pip_calls = _reused_venv(setup_env, tmp_path, monkeypatch, mode, *argv)
    with pytest.raises(SystemExit):
        setup.install_torch()
    assert pip_calls == []
    assert "--clean" in capsys.readouterr().err


def test_reused_venv_without_a_record_warns_and_keeps_torch(
    setup_env, tmp_path, monkeypatch, capsys
) -> None:
    # Every venv set up before TORCH_RECORD existed takes this path on rerun.
    setup, pip_calls = _reused_venv(
        setup_env,
        tmp_path,
        monkeypatch,
        "rocm",
        "--torch-index-url",
        "https://example.invalid/",
        record=False,
    )
    setup.install_torch()
    assert pip_calls == []
    err = capsys.readouterr().err
    assert f"no {setup_env.TORCH_RECORD}" in err
    assert "--torch-index-url https://example.invalid/" in err


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


def test_reused_venv_without_gpu_arch_builds_for_the_recorded_arch(
    setup_env, tmp_path, monkeypatch
) -> None:
    args = setup_env.build_parser().parse_args(["--workspace", str(tmp_path / "ws")])
    setup = setup_env.Setup(args)
    setup.venv_dir.mkdir(parents=True)
    (setup.venv_dir / setup_env.TORCH_RECORD).write_text(
        json.dumps(
            {"torch_index_url": setup_env.ROCM_TORCH_INDEX_URL, "gpu_arch": "gfx1201"}
        )
    )
    setup.installed_torch_mode = "rocm"
    monkeypatch.setattr(setup, "_detect_gpu_arch", lambda: "gfx942")

    setup.install_torch()

    assert setup.hip_arch_args == [
        "-DGPU_TARGETS=gfx1201",
        "-DAMDGPU_TARGETS=gfx1201",
    ]


def test_reused_venv_without_index_url_takes_the_toolchain_from_the_recorded_index(
    setup_env, tmp_path, monkeypatch
) -> None:
    # rocm-sdk-devel must come from the index the venv's ROCm wheels came from.
    setup = _setup(setup_env, tmp_path)
    setup.venv_dir.mkdir(parents=True)
    (setup.venv_dir / setup_env.TORCH_RECORD).write_text(
        json.dumps(
            {"torch_index_url": "https://example.invalid/", "gpu_arch": "gfx90a"}
        )
    )
    setup.installed_torch_mode = "rocm"
    monkeypatch.setattr(setup, "ensure_rocm_wheel_devel_prefix", lambda url: url)

    setup.install_torch()

    assert setup.toolchain_prefix == "https://example.invalid/"


@pytest.mark.skipif(
    sys.platform == "win32", reason="activate.local is written on Linux only"
)
def test_rejected_rerun_keeps_the_venv_activate_local(
    setup_env, tmp_path, monkeypatch
) -> None:
    setup = _setup(setup_env, tmp_path, "--gpu-arch", "gfx942", "-y")
    _fake_venv(setup_env, setup.venv_dir)
    (setup.venv_dir / setup_env.TORCH_RECORD).write_text(
        json.dumps(
            {"torch_index_url": setup_env.ROCM_TORCH_INDEX_URL, "gpu_arch": "gfx90a"}
        )
    )
    activate_local = setup.venv_dir / "bin" / "activate.local"
    activate_local.write_text("export ROCM_PATH=/wheel/prefix\n")
    monkeypatch.setattr(setup, "get_torch_mode", lambda: "rocm")

    with pytest.raises(SystemExit):
        setup.run()

    assert activate_local.read_text() == "export ROCM_PATH=/wheel/prefix\n"


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


# --- --reuse-artifacts bindings check ----------------------------------------


def _reuse_setup(setup_env, tmp_path):
    """A --reuse-artifacts Setup over a real venv and a prefix holding hipDNN."""
    prefix = tmp_path / "install"
    for name in ("hipdnn_frontend", "hipdnn_backend"):
        config = prefix / "lib" / "cmake" / name / f"{name}Config.cmake"
        config.parent.mkdir(parents=True)
        config.write_text("")
    setup = _setup(
        setup_env,
        tmp_path,
        "--torch-mode",
        "existing",
        "--rocm-prefix",
        str(prefix),
        "--reuse-artifacts",
    )
    subprocess.run(
        [sys.executable, "-m", "venv", "--without-pip", str(setup.venv_dir)],
        check=True,
    )
    return setup, str(prefix)


def _installed_venv_package(setup, name: str, source: str) -> Path:
    site = setup.probe("import sysconfig; print(sysconfig.get_path('purelib'))")
    assert site.returncode == 0, site.stderr
    package = Path(site.stdout.strip()) / name
    package.mkdir()
    (package / "__init__.py").write_text(source)
    return package


def test_reuse_without_installed_bindings_fails(setup_env, tmp_path, capsys) -> None:
    """Reuse builds no bindings, so an absent package cannot be a warning."""
    setup, prefix = _reuse_setup(setup_env, tmp_path)

    setup.build_hipdnn()
    with pytest.raises(SystemExit) as exc:
        setup.install_bindings()

    assert exc.value.code == 1
    assert setup.env["ROCM_PATH"] == prefix
    assert "hipdnn_frontend is not installed" in capsys.readouterr().err


def test_reuse_accepts_bindings_whose_native_libraries_do_not_load(
    setup_env, tmp_path
) -> None:
    """Presence is the question; the ROCm runtime behind it may be absent here."""
    setup, _ = _reuse_setup(setup_env, tmp_path)
    _installed_venv_package(
        setup,
        "hipdnn_frontend",
        "raise OSError('native ROCm runtime is unavailable on this host')\n",
    )
    # The import this check replaced fails on exactly this installed package.
    native = setup.probe("import hipdnn_frontend")
    assert native.returncode != 0
    assert "native ROCm runtime is unavailable" in native.stderr

    setup.build_hipdnn()
    setup.install_bindings()


@pytest.mark.parametrize(
    "mode, torch_source",
    [
        (
            "cpu",
            "from types import SimpleNamespace\n"
            "version = SimpleNamespace(hip=None, cuda=None)\n",
        ),
        (
            "rocm",
            "from types import SimpleNamespace\n"
            "version = SimpleNamespace(hip='6.4', cuda=None)\n",
        ),
        ("none", None),
    ],
)
def test_hipdnn_reuse_keeps_prepared_venv_and_does_not_fetch_sources(
    setup_env, tmp_path, monkeypatch, mode, torch_source
) -> None:
    prepared, prefix = _reuse_setup(setup_env, tmp_path)
    if torch_source is not None:
        _installed_venv_package(prepared, "torch", torch_source)
        (prepared.venv_dir / setup_env.TORCH_RECORD).write_text(
            json.dumps(
                {
                    "torch_index_url": (
                        setup_env.ROCM_TORCH_INDEX_URL
                        if mode == "rocm"
                        else "https://download.pytorch.org/whl/cpu"
                    ),
                    "gpu_arch": "gfx90a" if mode == "rocm" else "",
                }
            )
            + "\n"
        )
    frontend = _installed_venv_package(
        prepared, "hipdnn_frontend", "raise OSError('no native runtime')\n"
    )
    # A package that cannot load natively must still pass the presence check.
    assert prepared.probe("import hipdnn_frontend").returncode != 0
    marker = prepared.venv_dir / "keep-me"
    marker.write_text("original venv")
    checkout = tmp_path / "full-sources"
    checkout.mkdir()
    source_marker = checkout / "keep-me"
    source_marker.write_text("original checkout")
    monkeypatch.setattr(setup_env, "ROCM_LIBRARIES_DIR", checkout)
    monkeypatch.setattr(
        setup_env,
        "run_git",
        lambda *args, **kwargs: pytest.fail("reuse fetched rocm-libraries"),
    )
    setup = _setup(
        setup_env,
        tmp_path,
        "--torch-mode",
        mode,
        "--rocm-prefix",
        prefix,
        "--reuse-artifacts",
        "-y",
    )
    # Keep this test offline; the real CLI's package install and native import
    # are exercised separately on the actual ROCm image.
    monkeypatch.setattr(setup, "install_package", lambda: None)
    monkeypatch.setattr(setup, "install_runtime_extras", lambda: None)
    monkeypatch.setattr(setup, "verify", lambda: None)
    monkeypatch.setattr(setup, "report_profiling_sources", lambda: None)
    monkeypatch.setattr(
        setup, "pip", lambda *args, **kwargs: pytest.fail("reuse installed torch")
    )

    assert setup.run() == 0
    assert setup.env["ROCM_PATH"] == prefix
    assert setup.get_torch_mode() == (mode if mode != "none" else "missing")
    assert marker.read_text() == "original venv"
    assert (frontend / "__init__.py").read_text() == (
        "raise OSError('no native runtime')\n"
    )
    assert source_marker.read_text() == "original checkout"
    assert not checkout.with_name("full-sources.partial").exists()


def test_rocm_reuse_does_not_install_missing_toolchain(
    setup_env, tmp_path, monkeypatch
) -> None:
    setup = _setup(setup_env, tmp_path, "--torch-mode", "rocm", "--reuse-artifacts")
    setup.resolved_torch_index_url = setup_env.ROCM_TORCH_INDEX_URL
    monkeypatch.setattr(setup, "_rocm_sdk_devel_root", lambda: None)
    monkeypatch.setattr(
        setup, "pip", lambda *args, **kwargs: pytest.fail("downloaded ROCm SDK")
    )

    with pytest.raises(SystemExit):
        _ = setup.toolchain_prefix


# --- --reuse-artifacts preflight and main() failure reporting ----------------


@pytest.mark.parametrize("mode", ("rocm", "cpu", "none", "existing"))
def test_reuse_rejects_fresh_venv_before_workspace_creation(tmp_path, mode) -> None:
    workspace = tmp_path / "fresh"
    result = subprocess.run(
        [
            sys.executable,
            str(_SETUP_ENV),
            "--workspace",
            str(workspace),
            "--torch-mode",
            mode,
            "--reuse-artifacts",
            "-y",
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 1
    assert "--reuse-artifacts" in result.stderr
    assert "Traceback" not in result.stderr
    assert not workspace.exists()


@pytest.mark.parametrize(
    "mode, installed, flags",
    [
        ("cpu", "cpu", ["--clean"]),
        ("cpu", "cpu", ["--torch-index-url", "https://example.invalid/wrong"]),
        ("cpu", "rocm", []),
        ("rocm", "cpu", []),
        ("rocm", "rocm", ["--gpu-arch", "gfx942"]),
        ("none", "cpu", []),
        ("cuda", "cpu", []),
    ],
)
def test_reuse_rejects_clean_or_mismatched_venv_without_touching_it(
    setup_env, tmp_path, mode, installed, flags
) -> None:
    prepared, prefix = _reuse_setup(setup_env, tmp_path)
    hip = "'6.4'" if installed == "rocm" else "None"
    _installed_venv_package(
        prepared,
        "torch",
        f"from types import SimpleNamespace\n"
        f"version = SimpleNamespace(hip={hip}, cuda=None)\n",
    )
    frontend = _installed_venv_package(
        prepared, "hipdnn_frontend", "raise OSError('no native runtime')\n"
    )
    record = prepared.venv_dir / setup_env.TORCH_RECORD
    record.write_text(
        json.dumps(
            {
                "torch_index_url": (
                    setup_env.ROCM_TORCH_INDEX_URL
                    if installed == "rocm"
                    else "https://download.pytorch.org/whl/cpu"
                ),
                "gpu_arch": "gfx90a" if installed == "rocm" else "",
            }
        )
        + "\n"
    )
    record_before = record.read_bytes()
    marker = prepared.venv_dir / "keep-me"
    marker.write_text("unchanged")
    activate = prepared.venv_dir / (
        "Scripts/Activate.ps1" if setup_env.IS_WINDOWS else "bin/activate.local"
    )
    if not setup_env.IS_WINDOWS:
        activate.write_text("export ROCM_PATH=/original\n")
    activate_before = activate.read_bytes()
    setup = _setup(
        setup_env,
        tmp_path,
        "--torch-mode",
        mode,
        "--rocm-prefix",
        prefix,
        "--reuse-artifacts",
        *flags,
    )

    with pytest.raises(SystemExit) as exc:
        setup.run()

    assert exc.value.code == 1
    assert marker.read_text() == "unchanged"
    assert record.read_bytes() == record_before
    assert activate.read_bytes() == activate_before
    assert (frontend / "__init__.py").read_text() == (
        "raise OSError('no native runtime')\n"
    )


def test_existing_reuse_needs_installed_torch_before_modifying_venv(
    setup_env, tmp_path
) -> None:
    setup, _ = _reuse_setup(setup_env, tmp_path)
    marker = setup.venv_dir / "keep-me"
    marker.write_text("unchanged")

    with pytest.raises(SystemExit):
        setup.run()

    assert marker.read_text() == "unchanged"
    if not setup_env.IS_WINDOWS:
        assert not (setup.venv_dir / "bin/activate.local").exists()


def test_none_reuse_rejects_a_broken_torch_package(setup_env, tmp_path) -> None:
    prepared, prefix = _reuse_setup(setup_env, tmp_path)
    _installed_venv_package(prepared, "torch", "raise OSError('broken torch')\n")
    assert prepared.get_torch_mode() == "missing"
    found = prepared.probe(
        "import importlib.util; print(importlib.util.find_spec('torch') is not None)"
    )
    assert found.returncode == 0 and found.stdout.strip() == "True"
    marker = prepared.venv_dir / "keep-me"
    marker.write_text("unchanged")
    setup = _setup(
        setup_env,
        tmp_path,
        "--torch-mode",
        "none",
        "--rocm-prefix",
        prefix,
        "--reuse-artifacts",
    )

    with pytest.raises(SystemExit):
        setup.run()

    assert marker.read_text() == "unchanged"


@pytest.mark.parametrize(
    "error, rc",
    [
        (subprocess.CalledProcessError(3, ["pip", "install", "torch"]), 1),
        (OSError("disk full"), 1),
        (KeyboardInterrupt(), 130),
    ],
)
def test_main_names_the_failed_stage(
    setup_env, tmp_path, monkeypatch, capsys, error, rc
) -> None:
    def fail_torch(self):
        raise error

    monkeypatch.setattr(setup_env.Setup, "setup_venv", lambda self: None)
    monkeypatch.setattr(setup_env.Setup, "install_torch", fail_torch)

    assert (
        setup_env.main(["--workspace", str(tmp_path / "ws"), "--torch-mode", "cuda"])
        == rc
    )
    err = capsys.readouterr().err
    assert "PyTorch (--torch-mode cuda)" in err
    if isinstance(error, subprocess.CalledProcessError):
        assert "pip install torch" in err


@pytest.mark.skipif(
    sys.platform == "win32" or shutil.which("bash") is None,
    reason="activate.local is a bash script written on Linux only",
)
def test_activate_puts_the_install_prefix_before_the_toolchain(
    setup_env, tmp_path
) -> None:
    """A rebuilt binding must load the fresh libhipdnn_backend.so, not the
    toolchain's older copy (#54)."""
    setup = _setup(setup_env, tmp_path)
    activate = setup.venv_dir / "bin" / "activate"
    activate.parent.mkdir(parents=True)
    activate.write_text("# venv activate\n")

    setup.write_activate_local("/install", ("/install/lib", "/toolchain/lib"))

    ld_path = subprocess.run(
        ["bash", "-c", 'source "$1"; printf %s "$LD_LIBRARY_PATH"', "_", str(activate)],
        env={"PATH": os.environ["PATH"]},
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert ld_path == "/install/lib:/toolchain/lib"

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
from types import SimpleNamespace

import pytest

_SETUP_ENV = Path(__file__).resolve().parents[2] / "setup_env.py"
# Quartz nightly status.json files 20261113 to 20261119, recorded from
# ROCm/quartz. Their `pipelines` block keeps only the rocm and pytorch test
# entries of gfx94X-dcgpu, gfx950-dcgpu, gfx90a and gfx1151. Linux facts the
# tests use: 20261116 has a ROCm build in progress, 20261114 a failed PyTorch
# build. gfx94X-dcgpu ROCm tests fail on 20261114 and 20261115, and its
# PyTorch tests fail on 20261113. gfx950-dcgpu has no ROCm test entry, and
# its PyTorch tests fail on 20261113. gfx90a ROCm tests fail on every day.
_QUARTZ_NIGHTLY = Path(__file__).resolve().parent / "data" / "quartz" / "nightly"


@pytest.fixture(scope="module")
def setup_env():
    spec = importlib.util.spec_from_file_location("setup_env_under_test", _SETUP_ENV)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def _no_quartz_network(setup_env, monkeypatch):
    def refuse_url(url):
        raise AssertionError(f"unit test fetched {url}")

    def refuse_therock(
        therock_commit, repo_url=setup_env.THEROCK_REPO_URL, *, required=True
    ):
        raise AssertionError(
            f"unit test resolved TheRock {therock_commit} from {repo_url}"
        )

    monkeypatch.setattr(setup_env, "fetch_quartz_text", refuse_url)
    monkeypatch.setattr(
        setup_env, "resolve_therock_rocm_libraries_commit", refuse_therock
    )


def _quartz_document(build_date: str) -> dict:
    return json.loads((_QUARTZ_NIGHTLY / build_date / "status.json").read_text())


@pytest.fixture
def quartz(setup_env, monkeypatch):
    """Serve the recorded nightlies. ``latest`` is the build date that
    nightly/latest.json names; ``documents`` replaces a recorded document by
    build date; ``fetched`` lists every URL read. Setup reads the Linux
    summary of the recorded nightlies on every host, Windows included."""
    served = SimpleNamespace(latest="20261119", documents={}, fetched=[])

    def fetch(url):
        served.fetched.append(url)
        relative = url.removeprefix(setup_env.QUARTZ_NIGHTLY_URL)
        if relative == "latest.json":
            # A git symlink: raw GitHub serves its target path.
            return f"{served.latest}/status.json"
        build_date = relative.split("/", 1)[0]
        if build_date in served.documents:
            return json.dumps(served.documents[build_date])
        path = _QUARTZ_NIGHTLY / relative
        return path.read_text() if path.is_file() else None

    monkeypatch.setattr(setup_env, "fetch_quartz_text", fetch)
    monkeypatch.setattr(setup_env, "quartz_platform", lambda: "linux")
    return served


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


# --- cmake version -----------------------------------------------------------


@pytest.mark.parametrize(
    "version, ok",
    [
        # 3.22 cannot read CMakePresets.json version 6 ("Unrecognized
        # "version" field"); 3.25 cannot configure the hipDNN bindings.
        ("3.22.1", False),
        ("3.25.2", False),
        ("3.26.0", True),
        ("4.1.2", True),
    ],
)
def test_cmake_below_the_build_floor_fails_before_the_build(
    setup_env, monkeypatch, capsys, version, ok
) -> None:
    monkeypatch.setattr(setup_env.shutil, "which", lambda name: "/usr/bin/cmake")
    monkeypatch.setattr(
        setup_env.subprocess,
        "run",
        lambda cmd, **kw: subprocess.CompletedProcess(
            cmd, 0, stdout=f"cmake version {version}\n\nCMake suite ...\n", stderr=""
        ),
    )
    if ok:
        assert setup_env.require_working_cmake() == "/usr/bin/cmake"
        return
    with pytest.raises(SystemExit):
        setup_env.require_working_cmake()
    err = capsys.readouterr().err
    assert version.rsplit(".", 1)[0] in err and "3.26" in err


def test_cmake_check_runs_before_the_first_stage(
    setup_env, tmp_path, monkeypatch
) -> None:
    """A fresh workspace must not download wheels before an old cmake fails."""
    setup = _setup(setup_env, tmp_path, "-y")
    monkeypatch.setattr(
        setup_env, "require_working_cmake", lambda: setup_env.fail("cmake too old")
    )
    monkeypatch.setattr(setup, "setup_venv", lambda: pytest.fail("a stage ran"))

    with pytest.raises(SystemExit):
        setup.run()


def test_cmake_is_checked_once_per_setup(setup_env, tmp_path, monkeypatch) -> None:
    checks = []
    monkeypatch.setattr(
        setup_env, "require_working_cmake", lambda: checks.append(1) or "cmake"
    )
    setup = _setup(setup_env, tmp_path)

    assert (setup.cmake, setup.cmake) == ("cmake", "cmake")
    assert checks == [1]


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
    pin_file = repo / "rocm-pin.json"
    entry = {
        "rocm_version": _RECORDED_ROCM_VERSION,
        "build_date": "20261115",
        "therock_commit": "62d1ac21a0eda4df3871c6c8d4570cb0ee9a4637",
        "rocm_libraries_commit": _head(src),
        "index_url": setup_env.ROCM_TORCH_INDEX_URL,
    }
    pin_file.write_text(
        json.dumps(
            {
                "default": "gfx950",
                "archs": {
                    "gfx90a": entry,
                    "gfx942": entry,
                    "gfx950": entry,
                    "gfx1151": entry,
                },
            }
        )
        + "\n"
    )
    monkeypatch.setattr(setup_env, "SCRIPT_DIR", repo)
    monkeypatch.setattr(setup_env, "ROCM_PIN_FILE", pin_file)
    monkeypatch.setattr(setup_env, "ROCM_LIBRARIES_DIR", repo / "rocm-libraries")
    monkeypatch.setattr(setup_env, "ROCM_LIBRARIES_REPO_URL", origin.as_uri())
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


# --- AITER -------------------------------------------------------------------

_AITER_TAG = "v0.1.28.post1"


def _pin_aiter(repo: Path, commit: str) -> None:
    _git("update-index", "--add", "--cacheinfo", f"160000,{commit},aiter", cwd=repo)
    _git("commit", "-q", "--allow-empty", "-m", f"pin {commit[:8]}", cwd=repo)


@pytest.fixture
def aiter(setup_env, tmp_path, monkeypatch):
    """A local AITER origin whose tagged release commit has a CK submodule
    and is no longer the branch tip; this repository pins that commit.

    Returns ``(checkout path, pinned commit, branch tip)``.
    """
    if shutil.which("git") is None:
        pytest.skip("git unavailable")
    # git 2.38.1 and later refuse file:// submodule URLs by default.
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "protocol.file.allow")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", "always")

    ck = tmp_path / "ck-src"
    (ck / "include").mkdir(parents=True)
    (ck / "include" / "ck.hpp").write_text("// ck\n")
    _git("init", "-q", "-b", "develop", cwd=ck)
    _git("add", ".", cwd=ck)
    _git("commit", "-q", "-m", "ck", cwd=ck)
    ck_origin = tmp_path / "ck.git"
    _git("clone", "-q", "--bare", str(ck), str(ck_origin))

    src = tmp_path / "aiter-src"
    src.mkdir()
    (src / "setup.py").write_text("# aiter\n")
    _git("init", "-q", "-b", "main", cwd=src)
    _git(
        "submodule",
        "add",
        "-q",
        ck_origin.as_uri(),
        "3rdparty/composable_kernel",
        cwd=src,
    )
    _git("add", ".", cwd=src)
    _git("commit", "-q", "-m", "release", cwd=src)
    _git("tag", "-a", _AITER_TAG, "-m", "release", cwd=src)
    pinned = _head(src)
    (src / "next.txt").write_text("next")
    _git("add", ".", cwd=src)
    _git("commit", "-q", "-m", "next", cwd=src)
    origin = tmp_path / "aiter.git"
    _git("clone", "-q", "--bare", str(src), str(origin))

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".gitmodules").write_text(
        '[submodule "aiter"]\n'
        "\tpath = aiter\n"
        f"\turl = {origin.as_uri()}\n"
        "\tbranch = main\n"
    )
    _git("init", "-q", "-b", "main", cwd=repo)
    _git("add", ".gitmodules", cwd=repo)
    _pin_aiter(repo, pinned)
    monkeypatch.setattr(setup_env, "SCRIPT_DIR", repo)
    monkeypatch.setattr(setup_env, "AITER_DIR", repo / "aiter")
    return repo / "aiter", pinned, _head(src)


def test_aiter_fetch_pins_the_release_with_its_tag_and_ck(
    setup_env, tmp_path, capsys, aiter
) -> None:
    checkout, pinned, _ = aiter
    _setup(setup_env, tmp_path).ensure_aiter_checkout()

    assert _head(checkout) == pinned
    assert (checkout / "3rdparty/composable_kernel/include/ck.hpp").is_file()
    # setuptools_scm versions AITER from the tag; without it the result
    # files record 0.1.1.dev1 for a release.
    describe = subprocess.run(
        ["git", "-C", str(checkout), "describe", "--tags", "--exact-match"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert describe.stdout.strip() == _AITER_TAG
    assert not checkout.with_name("aiter.partial").exists()
    assert "WARNING" not in capsys.readouterr().err


def test_aiter_failed_fetch_leaves_no_checkout_to_reuse(
    setup_env, tmp_path, aiter
) -> None:
    checkout, pinned, _ = aiter
    _pin_aiter(setup_env.SCRIPT_DIR, "1" * 40)  # not in the origin
    setup = _setup(setup_env, tmp_path)

    with pytest.raises(subprocess.CalledProcessError):
        setup.ensure_aiter_checkout()
    assert not checkout.exists()
    assert not checkout.with_name("aiter.partial").exists()

    _pin_aiter(setup_env.SCRIPT_DIR, pinned)
    setup.ensure_aiter_checkout()
    assert _head(checkout) == pinned


def test_aiter_non_git_directory_is_kept_without_confirmation(
    setup_env, tmp_path, monkeypatch, aiter
) -> None:
    checkout, _, _ = aiter
    checkout.mkdir()
    (checkout / "notes.txt").write_text("mine")
    monkeypatch.setattr(sys, "stdin", _Stdin(tty=False))

    with pytest.raises(SystemExit):
        _setup(setup_env, tmp_path).ensure_aiter_checkout()
    assert (checkout / "notes.txt").read_text() == "mine"


def test_aiter_checkout_off_the_pin_is_kept_and_reported(
    setup_env, tmp_path, capsys, aiter
) -> None:
    """A user checkout survives; only its missing CK submodule is fetched."""
    checkout, pinned, tip = aiter
    _git("clone", "-q", str(tmp_path / "aiter.git"), str(checkout))

    _setup(setup_env, tmp_path).ensure_aiter_checkout()

    assert _head(checkout) == tip
    assert f"not the pinned {pinned[:12]}" in capsys.readouterr().err
    assert (checkout / "3rdparty/composable_kernel/include/ck.hpp").is_file()


@pytest.mark.parametrize(
    "torch_mode, argv",
    [
        pytest.param("cpu", (), id="cpu-torch"),
        pytest.param("rocm", ("--reuse-artifacts",), id="reuse-without-checkout"),
    ],
)
def test_install_aiter_skips_without_fetching(
    setup_env, tmp_path, monkeypatch, torch_mode, argv
) -> None:
    monkeypatch.setattr(setup_env, "AITER_DIR", tmp_path / "aiter")
    monkeypatch.setattr(
        setup_env, "run_git", lambda *a, **k: pytest.fail("fetched AITER")
    )
    setup = _setup(setup_env, tmp_path, *argv)
    setup.installed_torch_mode = torch_mode
    monkeypatch.setattr(setup, "pip", lambda *a, **k: pytest.fail("installed AITER"))

    setup.install_aiter()

    assert not (tmp_path / "aiter").exists()


def _install_aiter(setup_env, tmp_path, monkeypatch, toolchain: str = "/toolchain"):
    """Prepare install_aiter with pip and the prebuild faked.

    Returns ``(aiter dir, pip calls, prebuild envs, Setup)``.
    """
    aiter = tmp_path / "aiter"
    aiter.mkdir()
    (aiter / "pyproject.toml").write_text(
        '[build-system]\nrequires = ["setuptools>=45", "flydsl==0.4.2"]\n'
    )
    monkeypatch.setattr(setup_env, "AITER_DIR", aiter)
    setup = _setup(setup_env, tmp_path)
    setup.installed_torch_mode = "rocm"
    setup.toolchain_prefix = toolchain
    monkeypatch.setattr(setup, "ensure_aiter_checkout", lambda: None)
    installs = []
    monkeypatch.setattr(setup, "pip", lambda *a, env=None: installs.append((a, env)))
    prebuilds = []

    def fake_run(cmd, env=None, **k):
        prebuilds.append(env)
        return subprocess.CompletedProcess(cmd, returncode=1)

    monkeypatch.setattr(setup_env, "run", fake_run)
    return aiter, installs, prebuilds, setup


def test_install_aiter_installs_without_isolation_and_reports_prebuild_failure(
    setup_env, tmp_path, monkeypatch, capsys
) -> None:
    aiter, installs, _, setup = _install_aiter(setup_env, tmp_path, monkeypatch)

    setup.install_aiter()

    # An isolated build hides the venv's Triton from AITER's setup.py, which
    # then reinstalls Triton; so the build requirements go into the venv.
    assert installs[-2][0] == ("install", "setuptools>=45", "flydsl==0.4.2")
    # The default editable finder loses to the aiter/ checkout, which imports
    # as an empty namespace package from the repository root.
    args, env = installs[-1]
    assert args == (
        "install",
        "--no-build-isolation",
        "--config-settings",
        "editable_mode=compat",
        "-e",
        str(aiter),
    )
    assert env["AITER_USE_SYSTEM_TRITON"] == "1"
    assert "Some AITER backends did not compile" in capsys.readouterr().err


def _aiter_rocm_root(env: dict) -> str:
    """The ROCm root where AITER's setup.py looks for the HIP version."""
    return env.get("ROCM_HOME") or env.get("ROCM_PATH")


@pytest.mark.parametrize("user_rocm_home", [None, "/opt/my-rocm"])
def test_install_aiter_reads_the_hip_version_from_the_toolchain(
    setup_env, tmp_path, monkeypatch, user_rocm_home
) -> None:
    """With ROCm torch, ROCM_PATH is the hipDNN prefix, the libraries wheel.

    It has no hipconfig and no version file. Without /opt/rocm, AITER's
    setup.py then failed with "ROCm version file not found". A ROCM_HOME
    that the user sets still wins.
    """
    libraries = tmp_path / "_rocm_sdk_libraries"
    devel = tmp_path / "_rocm_sdk_devel"
    (devel / "bin").mkdir(parents=True)
    (devel / "bin" / "hipconfig").write_text("")
    _, installs, prebuilds, setup = _install_aiter(
        setup_env, tmp_path, monkeypatch, toolchain=str(devel)
    )
    setup.env["ROCM_PATH"] = str(libraries)
    setup.env.pop("ROCM_HOME", None)
    if user_rocm_home:
        setup.env["ROCM_HOME"] = user_rocm_home

    setup.install_aiter()

    build_env, prebuild_env = installs[-1][1], prebuilds[-1]
    expected = user_rocm_home or str(devel)
    for env in (build_env, prebuild_env):
        assert _aiter_rocm_root(env) == expected
        # Plugin discovery and the activation script keep the hipDNN prefix.
        assert env["ROCM_PATH"] == str(libraries)


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


_RECORDED_ROCM_VERSION = "10.3.0a20261115"


def _reused_venv(setup_env, tmp_path, monkeypatch, mode, *argv, record=True):
    """A Setup over a venv that setup filled with ``mode`` torch from its
    default index (gfx90a and ROCm _RECORDED_ROCM_VERSION for rocm).
    ``record=False`` drops the records, as on a venv set up before setup
    wrote them."""
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
        if mode == "rocm":
            (setup.venv_dir / setup_env.ROCM_BUILD_RECORD).write_text(
                json.dumps({"rocm_version": _RECORDED_ROCM_VERSION})
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
        ["--rocm-version", _RECORDED_ROCM_VERSION],
    ],
    ids=["no-flags", "same-arch", "same-index", "same-rocm-version"],
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
        ("rocm", ["--rocm-version", "10.3.0a20261110"]),
        ("cpu", ["--torch-index-url", "https://example.invalid/"]),
        ("cuda", ["--torch-index-url", "https://example.invalid/"]),
    ],
    ids=[
        "rocm-other-arch",
        "rocm-other-index",
        "rocm-other-version",
        "cpu-other-index",
        "cuda-other-index",
    ],
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
    setup_env, tmp_path, monkeypatch, quartz
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


# --- ROCm nightly selection (Quartz) ----------------------------------------


@pytest.mark.parametrize(
    "latest, arch, build_date, therock_commit, untested",
    [
        (
            "20261119",
            "gfx942",
            "20261119",
            "93b7a6b2f355e70b833b7b057ef406f6e6d3024e",
            (),
        ),
        (
            "20261119",
            "gfx950",
            "20261119",
            "93b7a6b2f355e70b833b7b057ef406f6e6d3024e",
            ("rocm",),
        ),
        (
            "20261116",
            "gfx950",
            "20261115",
            "62d1ac21a0eda4df3871c6c8d4570cb0ee9a4637",
            ("rocm",),
        ),
        (
            "20261124",
            "gfx942",
            "20261119",
            "93b7a6b2f355e70b833b7b057ef406f6e6d3024e",
            (),
        ),
    ],
    ids=[
        "arch-tests-pass",
        "no-rocm-test-entry",
        "rocm-build-in-progress",
        "days-without-a-nightly",
    ],
)
def test_selection_takes_the_newest_nightly_that_passes_for_the_arch(
    setup_env, quartz, latest, arch, build_date, therock_commit, untested
) -> None:
    quartz.latest = latest
    build = setup_env.select_quartz_nightly("linux", arch)
    assert build.build_date == build_date
    assert build.rocm_version == f"10.3.0a{build_date}"
    assert build.therock_commit == therock_commit
    assert build.untested == untested
    assert build.index_url == "https://nightly.repo.amd.com/rocm/whl-next/"
    assert build.source == "quartz"
    assert (
        build.status_url
        == f"https://raw.githubusercontent.com/ROCm/quartz/main/nightly/{build_date}/status.json"
    )


@pytest.mark.parametrize(
    "suite, status",
    [
        ("rocm", "failure"),
        ("pytorch", "failure"),
        ("pytorch", "in_progress"),
        ("rocm", "cancelled"),
    ],
)
def test_selection_skips_a_nightly_whose_arch_tests_did_not_pass(
    setup_env, quartz, suite, status
) -> None:
    document = _quartz_document("20261119")
    document["pipelines"][suite]["test"]["linux"]["gfx94X-dcgpu"]["status"] = status
    quartz.documents["20261119"] = document

    assert setup_env.select_quartz_nightly("linux", "gfx942").build_date == "20261118"
    # gfx950 does not read the gfx94X-dcgpu entry.
    assert setup_env.select_quartz_nightly("linux", "gfx950").build_date == "20261119"


def test_selection_accepts_an_arch_without_test_entries(setup_env, quartz) -> None:
    document = _quartz_document("20261119")
    del document["pipelines"]["pytorch"]["test"]["linux"]["gfx950-dcgpu"]
    quartz.documents["20261119"] = document

    build = setup_env.select_quartz_nightly("linux", "gfx950")

    assert build.build_date == "20261119"
    assert build.untested == ("rocm", "pytorch")


def test_selection_skips_a_nightly_without_the_gpu_arch(setup_env, quartz) -> None:
    document = _quartz_document("20261117")
    document["summary"]["linux"]["architectures"].remove("gfx950-dcgpu")
    quartz.documents["20261117"] = document
    quartz.latest = "20261117"
    # 20261116 has the arch but a Linux ROCm build in progress.
    build = setup_env.select_quartz_nightly("linux", "gfx950")
    assert build.build_date == "20261115"


@pytest.mark.parametrize(
    "arch, target, matches",
    [
        ("gfx942", "gfx94X-dcgpu", True),
        ("gfx950", "gfx950-dcgpu", True),
        ("gfx1101", "gfx110X-all", True),
        ("gfx90a", "gfx90a", True),
        ("gfx90a", "gfx94X-dcgpu", False),
        ("gfx942", "gfx950-dcgpu", False),
        ("gfx9500", "gfx950-dcgpu", False),
        ("gfx11011", "gfx110X-all", False),
    ],
)
def test_arch_matches_a_therock_build_target(setup_env, arch, target, matches) -> None:
    assert setup_env.quartz_arch_matches(arch, target) is matches


@pytest.mark.parametrize(
    "latest, arch, rejections",
    [
        (
            "20261116",
            "gfx942",
            [
                "20261116 (10.3.0a20261116): linux rocm build is in_progress\n",
                "20261115 (10.3.0a20261115): linux rocm tests of gfx94X-dcgpu "
                "are failure\n",
                "20261114 (10.3.0a20261114): linux pytorch build is failure; "
                "linux rocm tests of gfx94X-dcgpu are failure\n",
                "20261113 (10.3.0a20261113): linux pytorch tests of gfx94X-dcgpu "
                "are failure\n",
                "20261103: no status.json\n",
            ],
        ),
        (
            "20261114",
            "gfx950",
            [
                "20261114 (10.3.0a20261114): linux pytorch build is failure\n",
                "20261113 (10.3.0a20261113): linux pytorch tests of gfx950-dcgpu "
                "are failure\n",
            ],
        ),
        (
            "20261119",
            "gfx90a",
            [
                f"{day} (10.3.0a{day}): linux rocm tests of gfx90a are failure\n"
                for day in ("20261119", "20261118", "20261117", "20261115")
            ],
        ),
        (
            "20261119",
            "gfx1151",
            [
                "20261119 (10.3.0a20261119): linux rocm tests of gfx1151 are "
                "failure; linux pytorch tests of gfx1151 are in_progress\n",
                "20261113 (10.3.0a20261113): linux rocm tests of gfx1151 are "
                "cancelled; linux pytorch tests of gfx1151 are failure\n",
            ],
        ),
    ],
    ids=["gfx942", "gfx950", "gfx90a", "gfx1151"],
)
def test_selection_fails_and_names_each_failed_check(
    setup_env, quartz, capsys, latest, arch, rejections
) -> None:
    quartz.latest = latest
    with pytest.raises(SystemExit) as exc:
        setup_env.select_quartz_nightly("linux", arch)
    assert exc.value.code == 1
    err = capsys.readouterr().err
    for rejection in rejections:
        assert f"  {rejection}" in err


def test_selection_fails_when_no_nightly_in_the_window_qualifies(
    setup_env, quartz, capsys
) -> None:
    # No recorded Windows build lists a gfx94X target.
    with pytest.raises(SystemExit) as exc:
        setup_env.select_quartz_nightly("windows", "gfx942")
    assert exc.value.code == 1
    err = capsys.readouterr().err
    for day in range(13, 20):
        build_date = f"202611{day}"
        assert f"{build_date} (10.3.0a{build_date}): gfx942 is not among" in err
    assert "20261106: no status.json" in err
    # latest.json, then exactly one bounded window of nightlies.
    assert len(quartz.fetched) == 1 + setup_env.QUARTZ_LOOKBACK_DAYS


def test_selection_stops_at_an_unsupported_schema_major(
    setup_env, quartz, capsys
) -> None:
    document = _quartz_document("20261119")
    document["schema_version"] = "3.0"
    quartz.documents["20261119"] = document
    with pytest.raises(SystemExit):
        setup_env.select_quartz_nightly("linux", "gfx942")
    assert "schema '3.0'" in capsys.readouterr().err


def _fresh_rocm_setup(setup_env, tmp_path, monkeypatch, *argv, arch="gfx942"):
    """A Setup that installs ROCm torch for ``arch`` into an empty venv."""
    setup = _setup(setup_env, tmp_path, "--gpu-arch", arch, *argv)
    setup.venv_dir.mkdir(parents=True)
    pip_calls = []
    monkeypatch.setattr(setup, "pip", lambda *a, **kw: pip_calls.append(a))
    monkeypatch.setattr(setup, "get_torch_mode", lambda: "rocm")
    return setup, pip_calls


def _recorded_environment(setup, monkeypatch) -> dict:
    from dnn_benchmarking.metrics import machine_info

    monkeypatch.setattr(sys, "prefix", str(setup.venv_dir))
    info = machine_info.collect_environment_info()
    return {
        key: info[key]
        for key in (
            "rocm_build_version",
            "rocm_build_date",
            "therock_commit",
            "rocm_build_untested",
        )
    }


@pytest.mark.parametrize(
    "arch, latest, environment",
    [
        (
            "gfx942",
            "20261119",
            {
                "rocm_build_version": "10.3.0a20261119",
                "rocm_build_date": "20261119",
                "therock_commit": "93b7a6b2f355e70b833b7b057ef406f6e6d3024e",
                "rocm_build_untested": [],
            },
        ),
        (
            "gfx950",
            "20261116",
            {
                "rocm_build_version": "10.3.0a20261115",
                "rocm_build_date": "20261115",
                "therock_commit": "62d1ac21a0eda4df3871c6c8d4570cb0ee9a4637",
                "rocm_build_untested": ["rocm"],
            },
        ),
    ],
)
def test_rocm_install_latest_pins_the_selected_nightly_and_records_it(
    setup_env, tmp_path, monkeypatch, quartz, capsys, arch, latest, environment
) -> None:
    quartz.latest = latest
    setup, pip_calls = _fresh_rocm_setup(
        setup_env, tmp_path, monkeypatch, "--rocm-version", "latest", arch=arch
    )

    setup.install_torch()
    assert pip_calls == [
        (
            "install",
            "--pre",
            f"torch[device-{arch}]",
            f"rocm[libraries,devel,device-{arch}]=={environment['rocm_build_version']}",
            "--index-url",
            "https://nightly.repo.amd.com/rocm/whl-next/",
        )
    ]
    assert _recorded_environment(setup, monkeypatch) == environment
    out = capsys.readouterr().out
    assert ("Quartz has no rocm test result" in out) is bool(
        environment["rocm_build_untested"]
    )


@pytest.mark.parametrize(
    "version, warnings, untested",
    [
        (
            "10.3.0a20261115",
            ["linux rocm tests of gfx94X-dcgpu are failure"],
            [],
        ),
        (
            "10.3.0a20261114",
            [
                "linux pytorch build is failure",
                "linux rocm tests of gfx94X-dcgpu are failure",
            ],
            ["pytorch"],
        ),
    ],
    ids=["rocm-tests-failed", "pytorch-build-failed"],
)
def test_rocm_version_installs_a_failed_nightly_with_its_metadata(
    setup_env, tmp_path, monkeypatch, quartz, capsys, version, warnings, untested
) -> None:
    setup, pip_calls = _fresh_rocm_setup(
        setup_env, tmp_path, monkeypatch, "--rocm-version", version
    )

    setup.install_torch()

    build_date = version.removeprefix("10.3.0a")
    assert pip_calls[0][3] == f"rocm[libraries,devel,device-gfx942]=={version}"
    err = capsys.readouterr().err
    for warning in warnings:
        assert warning in err
    # Only the named build is read: no search for a newer one.
    assert quartz.fetched == [f"{setup_env.QUARTZ_NIGHTLY_URL}{build_date}/status.json"]
    assert _recorded_environment(setup, monkeypatch) == {
        "rocm_build_version": version,
        "rocm_build_date": build_date,
        "therock_commit": _quartz_document(build_date)["therock_commit"],
        "rocm_build_untested": untested,
    }


def test_rocm_version_unknown_to_quartz_records_no_build_metadata(
    setup_env, tmp_path, monkeypatch, quartz, capsys
) -> None:
    setup, pip_calls = _fresh_rocm_setup(
        setup_env, tmp_path, monkeypatch, "--rocm-version", "10.3.0a20261001"
    )

    setup.install_torch()

    assert pip_calls[0][3:] == (
        "rocm[libraries,devel,device-gfx942]==10.3.0a20261001",
        "--index-url",
        setup_env.ROCM_TORCH_INDEX_URL,
    )
    assert "Quartz has no nightly status.json" in capsys.readouterr().err
    assert _recorded_environment(setup, monkeypatch) == {
        "rocm_build_version": "10.3.0a20261001",
        "rocm_build_date": None,
        "therock_commit": None,
        "rocm_build_untested": None,
    }


def test_torch_index_url_installs_its_newest_rocm_without_quartz(
    setup_env, tmp_path, monkeypatch
) -> None:
    # The autouse guard fails the test on any Quartz fetch.
    setup, pip_calls = _fresh_rocm_setup(
        setup_env, tmp_path, monkeypatch, "--torch-index-url", "https://mirror.invalid/"
    )
    monkeypatch.setattr(setup, "installed_rocm_version", lambda: "10.3.0a20261117")

    setup.install_torch()

    assert pip_calls[0][3:] == (
        "rocm[libraries,devel,device-gfx942]",
        "--index-url",
        "https://mirror.invalid/",
    )
    assert _recorded_environment(setup, monkeypatch)["rocm_build_version"] == (
        "10.3.0a20261117"
    )


def test_rocm_version_needs_rocm_torch_mode(setup_env) -> None:
    with pytest.raises(SystemExit) as exc:
        setup_env.main(["--torch-mode", "cpu", "--rocm-version", "10.3.0a20261115"])
    assert exc.value.code == 2


def test_missing_toolchain_is_installed_at_the_recorded_rocm_version(
    setup_env, tmp_path, monkeypatch
) -> None:
    setup, pip_calls = _reused_venv(setup_env, tmp_path, monkeypatch, "rocm")
    roots = iter([None, "/devel"])
    monkeypatch.setattr(setup, "_rocm_sdk_devel_root", lambda **kw: next(roots))

    assert setup.ensure_rocm_wheel_devel_prefix("https://index.invalid/") == "/devel"
    assert pip_calls == [
        (
            "install",
            "--pre",
            f"rocm-sdk-devel=={_RECORDED_ROCM_VERSION}",
            "--index-url",
            "https://index.invalid/",
        )
    ]


# --- Checked-in rocm-pin.json & Quartz -> TheRock -> rocm-libraries ---------


def test_rocm_install_defaults_to_rocm_pin_without_network(
    setup_env, tmp_path, monkeypatch, rocm_libraries
) -> None:
    # The autouse guard fails on any Quartz or TheRock network call.
    pin = setup_env.read_rocm_pin("gfx950")
    setup, pip_calls = _fresh_rocm_setup(
        setup_env, tmp_path, monkeypatch, arch="gfx950"
    )

    setup.install_torch()
    setup.ensure_rocm_libraries_checkout()

    assert pip_calls == [
        (
            "install",
            "--pre",
            "torch[device-gfx950]",
            f"rocm[libraries,devel,device-gfx950]=={pin.rocm_version}",
            "--index-url",
            pin.index_url,
        )
    ]
    assert _recorded_environment(setup, monkeypatch) == {
        "rocm_build_version": pin.rocm_version,
        "rocm_build_date": pin.build_date,
        "therock_commit": pin.therock_commit,
        "rocm_build_untested": None,
    }
    saved = json.loads((setup.venv_dir / setup_env.ROCM_BUILD_RECORD).read_text())
    assert saved["source"] == "rocm-pin.json"
    assert saved["rocm_libraries_commit"] == pin.rocm_libraries_commit
    assert _head(rocm_libraries) == pin.rocm_libraries_commit


def test_read_rocm_pin_selects_per_arch_and_falls_back_to_default(
    setup_env, tmp_path, monkeypatch, capsys
) -> None:
    pin_file = tmp_path / "rocm-pin.json"
    pin_file.write_text(
        json.dumps(
            {
                "default": "gfx950",
                "archs": {
                    "gfx90a": {
                        "rocm_version": "10.3.0a20261120",
                        "build_date": "20261120",
                        "therock_commit": "a" * 40,
                        "rocm_libraries_commit": "b" * 40,
                        "index_url": setup_env.ROCM_TORCH_INDEX_URL,
                    },
                    "gfx950": {
                        "rocm_version": "10.3.0a20261123",
                        "build_date": "20261123",
                        "therock_commit": "c" * 40,
                        "rocm_libraries_commit": "d" * 40,
                        "index_url": setup_env.ROCM_TORCH_INDEX_URL,
                    },
                },
            }
        )
    )
    monkeypatch.setattr(setup_env, "ROCM_PIN_FILE", pin_file)

    pin_90a = setup_env.read_rocm_pin("gfx90a")
    assert pin_90a.rocm_version == "10.3.0a20261120"
    assert pin_90a.rocm_libraries_commit == "b" * 40

    pin_950 = setup_env.read_rocm_pin("gfx950")
    assert pin_950.rocm_version == "10.3.0a20261123"
    assert pin_950.rocm_libraries_commit == "d" * 40

    # Unlisted arch falls back to default (gfx950) and prints a note.
    setup, _ = _fresh_rocm_setup(setup_env, tmp_path, monkeypatch, arch="gfx1201")
    fallback = setup.select_rocm_build("gfx1201")
    assert fallback.rocm_version == "10.3.0a20261123"
    assert fallback.rocm_libraries_commit == "d" * 40
    assert "no entry for gfx1201; using default (gfx950)" in capsys.readouterr().out


def test_read_rocm_pin_rejects_missing_or_invalid_fields(
    setup_env, tmp_path, capsys
) -> None:
    missing = tmp_path / "missing.json"
    with pytest.raises(SystemExit) as exc:
        setup_env.read_rocm_pin(path=missing)
    assert exc.value.code == 1
    assert "cannot read ROCm pin file" in capsys.readouterr().err

    bad_commit = tmp_path / "bad-commit.json"
    bad_commit.write_text(
        json.dumps(
            {
                "default": "gfx950",
                "archs": {
                    "gfx950": {
                        "rocm_version": "10.3.0a20261123",
                        "rocm_libraries_commit": "short",
                    }
                },
            }
        )
    )
    with pytest.raises(SystemExit) as exc:
        setup_env.read_rocm_pin(path=bad_commit)
    assert exc.value.code == 1
    assert "invalid rocm_libraries_commit" in capsys.readouterr().err

    bad_default = tmp_path / "bad-default.json"
    bad_default.write_text(
        json.dumps(
            {
                "default": "gfx942",
                "archs": {
                    "gfx950": {
                        "rocm_version": "10.3.0a20261123",
                        "rocm_libraries_commit": "a" * 40,
                    }
                },
            }
        )
    )
    with pytest.raises(SystemExit) as exc:
        setup_env.read_rocm_pin(path=bad_default)
    assert exc.value.code == 1
    assert "invalid default arch" in capsys.readouterr().err


def test_reused_venv_refuses_rocm_version_latest(
    setup_env, tmp_path, monkeypatch, capsys
) -> None:
    setup, _ = _reused_venv(
        setup_env, tmp_path, monkeypatch, "rocm", "--rocm-version", "latest"
    )
    with pytest.raises(SystemExit) as exc:
        setup.install_torch()
    assert exc.value.code == 1
    assert "--rocm-version latest" in capsys.readouterr().err


def test_find_quartz_nightly_per_arch_and_extra_platform_builds(
    setup_env, quartz
) -> None:
    # On 20261116, rocm build is in_progress; on 20261115, gfx950 passes while
    # gfx942 fails rocm tests -> per-arch lookup picks 20261115 for gfx950 and
    # returns None for gfx942 (no passing nightly in recorded window).
    quartz.latest = "20261116"
    assert setup_env.find_quartz_nightly("linux", "gfx950").build_date == "20261115"
    assert setup_env.find_quartz_nightly("linux", "gfx942") is None

    # require_platform_builds checks extra platform builds/wheels without
    # failing on flaky tests: on 20261119 Windows pytorch build is in_progress,
    # so requiring Windows gfx1151 builds walks back to 20261118.
    quartz.latest = "20261119"
    build = setup_env.find_quartz_nightly(
        "linux",
        "gfx950",
        require_platform_builds=(("windows", setup_env.WINDOWS_DEFAULT_GPU_ARCH),),
    )
    assert build is not None
    assert build.build_date == "20261118"


def _make_fake_therock_repo(
    tmp_path: Path, rocm_libraries_commit: str | None
) -> tuple[Path, str]:
    """Create a local bare TheRock repo whose tip optionally pins rocm-libraries."""
    src = tmp_path / "therock-src"
    src.mkdir()
    _git("init", "-q", "-b", "main", cwd=src)
    if rocm_libraries_commit is not None:
        _git(
            "update-index",
            "--add",
            "--cacheinfo",
            f"160000,{rocm_libraries_commit},rocm-libraries",
            cwd=src,
        )
    else:
        (src / "README.md").write_text("no submodule\n")
        _git("add", "README.md", cwd=src)
    _git("commit", "-q", "-m", "therock pin", cwd=src)
    commit = _head(src)
    bare = tmp_path / "therock.git"
    _git("clone", "-q", "--bare", str(src), str(bare))
    _git("config", "uploadpack.allowfilter", "true", cwd=bare)
    return bare, commit


def test_resolve_therock_rocm_libraries_commit_reads_submodule_gitlink(
    setup_env, tmp_path, monkeypatch
) -> None:
    if shutil.which("git") is None:
        pytest.skip("git unavailable")
    monkeypatch.undo()
    expected_sha = "1234567890abcdef" * 2 + "12345678"
    bare, therock_commit = _make_fake_therock_repo(tmp_path, expected_sha)

    resolved = setup_env.resolve_therock_rocm_libraries_commit(
        therock_commit, repo_url=bare.as_uri()
    )

    assert resolved == expected_sha


def test_resolve_therock_rocm_libraries_commit_fails_when_submodule_missing(
    setup_env, tmp_path, monkeypatch, capsys
) -> None:
    if shutil.which("git") is None:
        pytest.skip("git unavailable")
    monkeypatch.undo()
    bare, therock_commit = _make_fake_therock_repo(tmp_path, None)

    with pytest.raises(SystemExit) as exc:
        setup_env.resolve_therock_rocm_libraries_commit(
            therock_commit, repo_url=bare.as_uri()
        )

    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "no 'rocm-libraries' submodule entry" in err
    assert "--rocm-libraries-ref" in err


def test_resolve_therock_rocm_libraries_commit_rejects_bad_sha(
    setup_env, monkeypatch, capsys
) -> None:
    monkeypatch.undo()
    with pytest.raises(SystemExit) as exc:
        setup_env.resolve_therock_rocm_libraries_commit("not-a-sha")

    assert exc.value.code == 1
    assert "40-character TheRock commit SHA" in capsys.readouterr().err


def test_resolve_therock_rocm_libraries_commit_handles_missing_git(
    setup_env, monkeypatch, capsys
) -> None:
    monkeypatch.undo()

    def no_git(*_args, **_kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(setup_env, "git_output", no_git)

    with pytest.raises(SystemExit) as exc:
        setup_env.resolve_therock_rocm_libraries_commit("f" * 40)

    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "git not found on PATH" in err
    assert "--rocm-libraries-ref" in err


def _two_commit_rocm_libraries(tmp_path: Path) -> tuple[str, str]:
    """Advance the local rocm_libraries origin by one commit; return (first, second)."""
    src = tmp_path / "src"
    first = _head(src)
    (src / "projects/hipdnn/matched.txt").write_text("matched\n")
    _git("add", ".", cwd=src)
    _git("commit", "-q", "-m", "matched commit", cwd=src)
    _git("fetch", "-q", str(src), "main:main", cwd=tmp_path / "origin.git")
    return first, _head(src)


def test_rocm_mode_checks_out_quartz_matched_rocm_libraries_and_caches_it(
    setup_env, tmp_path, monkeypatch, rocm_libraries
) -> None:
    _, quartz_sha = _two_commit_rocm_libraries(tmp_path)
    therock_sha = "a" * 40
    calls = []
    monkeypatch.setattr(
        setup_env,
        "resolve_therock_rocm_libraries_commit",
        lambda sha, repo_url=setup_env.THEROCK_REPO_URL, *, required=True: (
            calls.append(sha),
            quartz_sha,
        )[1],
    )

    setup = _setup(setup_env, tmp_path)
    setup.venv_dir.mkdir(parents=True)
    setup.installed_torch_mode = "rocm"
    record_path = setup.venv_dir / setup_env.ROCM_BUILD_RECORD
    record_path.write_text(
        json.dumps(
            {"rocm_version": _RECORDED_ROCM_VERSION, "therock_commit": therock_sha}
        )
    )

    setup.ensure_rocm_libraries_checkout()

    assert calls == [therock_sha]
    assert _head(rocm_libraries) == quartz_sha
    saved = json.loads(record_path.read_text())
    assert saved["rocm_libraries_commit"] == quartz_sha

    # A second run against the same venv reuses the cached rocm_libraries_commit
    # without querying TheRock again.
    setup.ensure_rocm_libraries_checkout()
    assert calls == [therock_sha]


def test_rocm_mode_warns_when_existing_checkout_differs_from_quartz_match(
    setup_env, tmp_path, monkeypatch, capsys, rocm_libraries
) -> None:
    older_sha, quartz_sha = _two_commit_rocm_libraries(tmp_path)
    therock_sha = "b" * 40
    _setup(
        setup_env, tmp_path, "--rocm-libraries-ref", older_sha
    ).ensure_rocm_libraries_checkout()
    assert _head(rocm_libraries) == older_sha

    monkeypatch.setattr(
        setup_env,
        "resolve_therock_rocm_libraries_commit",
        lambda sha, repo_url=setup_env.THEROCK_REPO_URL, *, required=True: quartz_sha,
    )
    setup = _setup(setup_env, tmp_path)
    setup.venv_dir.mkdir(parents=True)
    setup.installed_torch_mode = "rocm"
    (setup.venv_dir / setup_env.ROCM_BUILD_RECORD).write_text(
        json.dumps(
            {"rocm_version": _RECORDED_ROCM_VERSION, "therock_commit": therock_sha}
        )
    )

    setup.ensure_rocm_libraries_checkout()

    assert _head(rocm_libraries) == older_sha
    err = capsys.readouterr().err
    assert f"not {quartz_sha[:12]} (matched to TheRock {therock_sha[:12]})" in err


def test_rocm_mode_reuses_existing_checkout_when_therock_lookup_fails(
    setup_env, tmp_path, monkeypatch, capsys, rocm_libraries
) -> None:
    older_sha, _ = _two_commit_rocm_libraries(tmp_path)
    _setup(
        setup_env, tmp_path, "--rocm-libraries-ref", older_sha
    ).ensure_rocm_libraries_checkout()
    assert _head(rocm_libraries) == older_sha

    missing_repo = (tmp_path / "no-such-therock.git").as_uri()
    monkeypatch.undo()
    monkeypatch.setattr(setup_env, "ROCM_LIBRARIES_DIR", rocm_libraries)
    monkeypatch.setattr(setup_env, "THEROCK_REPO_URL", missing_repo)
    setup = _setup(setup_env, tmp_path)
    setup.venv_dir.mkdir(parents=True)
    setup.installed_torch_mode = "rocm"
    (setup.venv_dir / setup_env.ROCM_BUILD_RECORD).write_text(
        json.dumps({"rocm_version": _RECORDED_ROCM_VERSION, "therock_commit": "e" * 40})
    )

    setup.ensure_rocm_libraries_checkout()

    assert _head(rocm_libraries) == older_sha
    err = capsys.readouterr().err
    assert "WARNING:" in err
    assert missing_repo in err
    assert "Keeping the existing rocm-libraries/ checkout." in err


def test_explicit_rocm_libraries_ref_overrides_quartz_therock_commit(
    setup_env, tmp_path, rocm_libraries
) -> None:
    explicit_sha, _ = _two_commit_rocm_libraries(tmp_path)
    setup = _setup(setup_env, tmp_path, "--rocm-libraries-ref", explicit_sha)
    setup.venv_dir.mkdir(parents=True)
    setup.installed_torch_mode = "rocm"
    record_path = setup.venv_dir / setup_env.ROCM_BUILD_RECORD
    record_path.write_text(
        json.dumps({"rocm_version": _RECORDED_ROCM_VERSION, "therock_commit": "c" * 40})
    )

    setup.ensure_rocm_libraries_checkout()

    assert _head(rocm_libraries) == explicit_sha
    assert "rocm_libraries_commit" not in json.loads(record_path.read_text())


@pytest.mark.parametrize(
    "torch_mode, record",
    [
        ("cpu", {"rocm_version": _RECORDED_ROCM_VERSION, "therock_commit": "d" * 40}),
        ("rocm", {"rocm_version": _RECORDED_ROCM_VERSION, "therock_commit": None}),
        ("rocm", None),
    ],
    ids=["cpu-mode", "rocm-without-therock-commit", "rocm-without-record"],
)
def test_rocm_libraries_falls_back_to_rocm_pin_without_quartz_therock_commit(
    setup_env, tmp_path, rocm_libraries, torch_mode, record
) -> None:
    pin_sha, _ = _two_commit_rocm_libraries(tmp_path)
    setup = _setup(setup_env, tmp_path, "--torch-mode", torch_mode)
    setup.venv_dir.mkdir(parents=True)
    setup.installed_torch_mode = torch_mode
    if record is not None:
        (setup.venv_dir / setup_env.ROCM_BUILD_RECORD).write_text(json.dumps(record))

    setup.ensure_rocm_libraries_checkout()

    assert _head(rocm_libraries) == pin_sha


def _host_gpu_archs(setup_env, monkeypatch, archs):
    """rocm_agent_enumerator reports ``archs`` (plus the CPU agent)."""
    real_run, real_which = setup_env.subprocess.run, setup_env.shutil.which

    def which(name, *a, **kw):
        if name in ("rocm_agent_enumerator", "rocminfo"):
            return name if name == "rocm_agent_enumerator" and archs else None
        return real_which(name, *a, **kw)

    def run(cmd, *a, **kw):
        if list(cmd) == ["rocm_agent_enumerator"]:
            out = "\n".join(["gfx000", *archs]) + "\n"
            return subprocess.CompletedProcess(cmd, 0, stdout=out, stderr="")
        return real_run(cmd, *a, **kw)

    monkeypatch.setattr(setup_env.shutil, "which", which)
    monkeypatch.setattr(setup_env.subprocess, "run", run)


@pytest.mark.parametrize(
    "host, recorded",
    [
        ([], "gfx1201"),
        (["gfx1201"], "gfx1201"),
        # A multi-arch host keeps a venv recorded for one of its archs.
        (["gfx90a", "gfx942"], "gfx942"),
    ],
    ids=["no-detection", "match", "multi-arch-match"],
)
def test_reused_venv_without_gpu_arch_builds_for_the_recorded_arch(
    setup_env, tmp_path, monkeypatch, host, recorded
) -> None:
    setup = _venv_recorded_for(setup_env, tmp_path, recorded)
    _host_gpu_archs(setup_env, monkeypatch, host)

    setup.install_torch()

    assert setup.hip_arch_args == [
        f"-DGPU_TARGETS={recorded}",
        f"-DAMDGPU_TARGETS={recorded}",
    ]


@pytest.mark.parametrize("host", [["gfx942"], ["gfx90a", "gfx942"]])
def test_reused_venv_refuses_a_detected_gpu_of_another_arch(
    setup_env, tmp_path, monkeypatch, capsys, host
) -> None:
    setup = _venv_recorded_for(setup_env, tmp_path, "gfx1201")
    _host_gpu_archs(setup_env, monkeypatch, host)
    monkeypatch.setattr(setup, "pip", lambda *a, **kw: pytest.fail("pip ran"))

    with pytest.raises(SystemExit) as exc:
        setup.install_torch()

    assert exc.value.code != 0
    err = capsys.readouterr().err
    assert all(arch in err for arch in host) and "gfx1201" in err
    assert "--clean" in err and "--gpu-arch gfx1201" in err


def _venv_recorded_for(setup_env, tmp_path, arch):
    """A Setup without --gpu-arch over a ROCm venv recorded for ``arch``."""
    args = setup_env.build_parser().parse_args(["--workspace", str(tmp_path / "ws")])
    setup = setup_env.Setup(args)
    setup.venv_dir.mkdir(parents=True)
    (setup.venv_dir / setup_env.TORCH_RECORD).write_text(
        json.dumps(
            {"torch_index_url": setup_env.ROCM_TORCH_INDEX_URL, "gpu_arch": arch}
        )
    )
    setup.installed_torch_mode = "rocm"
    return setup


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
    monkeypatch.setattr(setup_env, "require_working_cmake", lambda: "cmake")

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


def _wheel_core(setup, packaged: bool) -> Path:
    """A rocm-sdk-core wheel layout in the venv. ``packaged=False`` is the
    layout since about rocm 10.1: share/amd_smi/amdsmi with no setup.py or
    pyproject.toml."""
    site = setup.probe("import sysconfig; print(sysconfig.get_path('purelib'))")
    assert site.returncode == 0, site.stderr
    core = Path(site.stdout.strip()) / "_rocm_sdk_core"
    (core / "lib").mkdir(parents=True)
    (core / "lib" / "libamd_smi.so.27").write_text("")
    package = core / "share" / "amd_smi" / "amdsmi"
    package.mkdir(parents=True)
    # Like the real amdsmi_wrapper: the library is found relative to the
    # resolved package path, so a copy of the package does not import.
    (package / "__init__.py").write_text(
        "from pathlib import Path\n"
        "_lib = Path(__file__).resolve().parents[3] / 'lib' / 'libamd_smi.so.27'\n"
        "if not _lib.is_file():\n"
        "    raise ImportError(f'{_lib} not found')\n"
    )
    if packaged:
        (package.parent / "pyproject.toml").write_text("")
    return core


@pytest.mark.parametrize("stale_link", [False, True])
def test_amdsmi_from_an_unpackaged_core_wheel_is_importable(
    setup_env, tmp_path, monkeypatch, capsys, stale_link
) -> None:
    probe = tmp_path / "symlink-probe"
    try:
        probe.symlink_to(tmp_path)
    except OSError:
        pytest.skip("symlinks unavailable")
    setup, _ = _reuse_setup(setup_env, tmp_path)
    core = _wheel_core(setup, packaged=False)
    if stale_link:
        # A link left by an earlier setup to a package that no longer exists.
        (core.parent / "amdsmi").symlink_to(tmp_path / "gone", True)
    monkeypatch.setattr(setup, "pip", lambda *a, **kw: pytest.fail("pip ran"))

    setup.maybe_install_amdsmi()

    assert setup.amdsmi_importable()
    assert "not installed" not in capsys.readouterr().err


def test_amdsmi_from_a_packaged_core_wheel_is_pip_installed(
    setup_env, tmp_path, monkeypatch
) -> None:
    setup, _ = _reuse_setup(setup_env, tmp_path)
    core = _wheel_core(setup, packaged=True)
    installs = []
    monkeypatch.setattr(setup, "pip", lambda *a, **kw: installs.append(a))

    setup.maybe_install_amdsmi()

    assert installs == [("install", "-e", str(core / "share" / "amd_smi"))]


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
    # No aiter/ checkout: the AITER stage must skip, not fetch.
    monkeypatch.setattr(setup_env, "AITER_DIR", tmp_path / "no-aiter")
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


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (
            (),
            [
                (
                    "install",
                    "--config-settings",
                    "editable_mode=compat",
                    "-e",
                    "platform",
                ),
                (
                    "install",
                    "--no-deps",
                    "--config-settings",
                    "editable_mode=compat",
                    "-e",
                    "library",
                ),
            ],
        ),
        (
            ("--no-editable",),
            [("install", "platform"), ("install", "--no-deps", "library")],
        ),
    ],
)
def test_rocke_install_honours_no_editable(
    setup_env, tmp_path, monkeypatch, argv, expected
) -> None:
    """A copied --no-editable venv must not point back at the rocKE sources."""
    rocke = tmp_path / "rl" / "dnn-providers" / "hip-kernel-provider" / "rocke"
    (rocke / "library").mkdir(parents=True)
    (rocke / "library" / "pyproject.toml").write_text("")
    monkeypatch.setattr(setup_env, "ROCM_LIBRARIES_DIR", tmp_path / "rl")
    setup = _setup(setup_env, tmp_path, *argv)
    calls = []
    monkeypatch.setattr(setup, "pip", lambda *args: calls.append(args))

    setup.install_rocke_packages()

    package_dirs = {name: str(rocke / name) for name in ("platform", "library")}
    assert calls == [tuple(package_dirs.get(a, a) for a in call) for call in expected]


# --- hybrid prebuilt hipDNN mode --------------------------------------------


def _populate_prebuilt_hipdnn(
    setup_env,
    install_prefix: Path,
    toolchain_prefix: Path,
    *,
    windows: bool = False,
    include_hip_kernel: bool = True,
) -> None:
    for rel in setup_env._PREBUILT_HIPDNN_CMAKE_CONFIGS:
        cfg = toolchain_prefix / rel
        cfg.parent.mkdir(parents=True, exist_ok=True)
        cfg.write_text("")
    if not windows:
        comgr = toolchain_prefix / "lib" / "libamd_comgr.so.3"
        comgr.parent.mkdir(parents=True, exist_ok=True)
        comgr.write_text("")
    if windows:
        files = [
            install_prefix / "bin" / "hipdnn_backend.dll",
            install_prefix / "bin" / "hipdnn_plugins" / "engines" / "miopen_plugin.dll",
            install_prefix
            / "bin"
            / "hipdnn_plugins"
            / "engines"
            / "hipblaslt_plugin.dll",
        ]
        if include_hip_kernel:
            files.append(
                install_prefix
                / "bin"
                / "hipdnn_plugins"
                / "engines"
                / "hip_kernel_provider.dll"
            )
    else:
        files = [
            install_prefix / "lib" / "libhipdnn_backend.so",
            install_prefix
            / "lib"
            / "hipdnn_plugins"
            / "engines"
            / "libmiopen_plugin.so",
            install_prefix
            / "lib"
            / "hipdnn_plugins"
            / "engines"
            / "libhipblaslt_plugin.so",
        ]
        if include_hip_kernel:
            files.append(
                install_prefix
                / "lib"
                / "hipdnn_plugins"
                / "engines"
                / "libhip_kernel_provider.so"
            )
    for path in files:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("")


def test_hybrid_mode_linux_builds_only_hip_kernel_provider(
    setup_env, tmp_path, monkeypatch
) -> None:
    install_prefix = tmp_path / "libs"
    toolchain_prefix = tmp_path / "devel"
    _populate_prebuilt_hipdnn(setup_env, install_prefix, toolchain_prefix)
    monkeypatch.setattr(setup_env, "IS_WINDOWS", False)
    monkeypatch.setattr(setup_env, "ROCM_LIBRARIES_DIR", tmp_path / "rl")
    monkeypatch.setattr(setup_env, "require_working_cmake", lambda: "cmake")
    monkeypatch.setattr(setup_env.shutil, "which", lambda name: name)
    commands = []
    monkeypatch.setattr(
        setup_env, "run", lambda cmd, **kwargs: commands.append(list(cmd))
    )

    setup = _setup(setup_env, tmp_path)
    monkeypatch.setattr(setup, "_build_env", lambda: {})
    setup.build_superbuild(str(install_prefix), str(toolchain_prefix))

    assert commands[0][1:3] == [
        "-S",
        str(tmp_path / "rl" / "dnn-providers" / "hip-kernel-provider"),
    ]
    assert "--preset" not in commands[0]
    assert len(commands) == 3  # configure, build, install


def test_hybrid_mode_windows_skips_superbuild_when_all_plugins_prebuilt(
    setup_env, tmp_path, monkeypatch
) -> None:
    install_prefix = tmp_path / "libs"
    toolchain_prefix = tmp_path / "devel"
    _populate_prebuilt_hipdnn(setup_env, install_prefix, toolchain_prefix, windows=True)
    monkeypatch.setattr(setup_env, "IS_WINDOWS", True)
    commands = []
    monkeypatch.setattr(
        setup_env, "run", lambda cmd, **kwargs: commands.append(list(cmd))
    )

    setup = _setup(setup_env, tmp_path)
    setup.build_superbuild(str(install_prefix), str(toolchain_prefix))

    assert commands == []


@pytest.mark.parametrize(
    "extra_argv, include_hip_kernel",
    [
        ((), False),
        (("--cmake-arg", "ENABLE_ASM_SDPA_ENGINE=OFF"), True),
    ],
)
def test_hybrid_mode_windows_builds_hip_kernel_provider_when_needed(
    setup_env, tmp_path, monkeypatch, extra_argv, include_hip_kernel
) -> None:
    install_prefix = tmp_path / "libs"
    toolchain_prefix = tmp_path / "devel"
    _populate_prebuilt_hipdnn(
        setup_env,
        install_prefix,
        toolchain_prefix,
        windows=True,
        include_hip_kernel=include_hip_kernel,
    )
    monkeypatch.setattr(setup_env, "IS_WINDOWS", True)
    setup = _setup(setup_env, tmp_path, *extra_argv)

    assert setup.select_superbuild_components(
        str(install_prefix), str(toolchain_prefix)
    ) == ("hip-kernel-provider",)


@pytest.mark.parametrize(
    "extra_argv",
    [
        ("--build-from-source",),
        ("--rocm-libraries-ref", "a" * 40),
    ],
)
def test_explicit_source_flags_force_full_superbuild(
    setup_env, tmp_path, monkeypatch, extra_argv
) -> None:
    install_prefix = tmp_path / "libs"
    toolchain_prefix = tmp_path / "devel"
    _populate_prebuilt_hipdnn(setup_env, install_prefix, toolchain_prefix)
    monkeypatch.setattr(setup_env, "IS_WINDOWS", False)
    setup = _setup(setup_env, tmp_path, *extra_argv)

    assert (
        setup.select_superbuild_components(str(install_prefix), str(toolchain_prefix))
        == setup_env.SUPERBUILD_ALL_COMPONENTS
    )


def test_local_rocm_libraries_divergence_forces_full_superbuild(
    setup_env, tmp_path, monkeypatch
) -> None:
    install_prefix = tmp_path / "libs"
    toolchain_prefix = tmp_path / "devel"
    _populate_prebuilt_hipdnn(setup_env, install_prefix, toolchain_prefix)
    monkeypatch.setattr(setup_env, "IS_WINDOWS", False)
    setup = _setup(setup_env, tmp_path)
    setup._rocm_libraries_matches_pin = False

    assert (
        setup.select_superbuild_components(str(install_prefix), str(toolchain_prefix))
        == setup_env.SUPERBUILD_ALL_COMPONENTS
    )


def test_missing_prebuilt_plugin_falls_back_to_full_superbuild(
    setup_env, tmp_path, monkeypatch
) -> None:
    install_prefix = tmp_path / "libs"
    toolchain_prefix = tmp_path / "devel"
    _populate_prebuilt_hipdnn(setup_env, install_prefix, toolchain_prefix)
    (
        install_prefix / "lib" / "hipdnn_plugins" / "engines" / "libmiopen_plugin.so"
    ).unlink()
    monkeypatch.setattr(setup_env, "IS_WINDOWS", False)
    setup = _setup(setup_env, tmp_path)

    assert (
        setup.select_superbuild_components(str(install_prefix), str(toolchain_prefix))
        == setup_env.SUPERBUILD_ALL_COMPONENTS
    )


def test_build_from_source_and_reuse_artifacts_rejected(setup_env, tmp_path) -> None:
    with pytest.raises(SystemExit) as exc:
        setup_env.main(
            [
                "--workspace",
                str(tmp_path / "ws"),
                "--build-from-source",
                "--reuse-artifacts",
            ]
        )
    assert exc.value.code == 2


def test_reuse_artifacts_accepts_hipdnn_configs_in_toolchain_prefix(
    setup_env, tmp_path, monkeypatch
) -> None:
    """Hybrid mode leaves CMake configs in rocm-sdk-devel (toolchain_prefix)."""
    install_prefix = tmp_path / "libs"
    toolchain_prefix = tmp_path / "devel"
    _populate_prebuilt_hipdnn(setup_env, install_prefix, toolchain_prefix)
    monkeypatch.setattr(setup_env, "IS_WINDOWS", False)
    setup = _setup(setup_env, tmp_path, "--reuse-artifacts")
    monkeypatch.setattr(setup, "select_binding_prefix", lambda: str(install_prefix))
    setup.__dict__["toolchain_prefix"] = str(toolchain_prefix)
    monkeypatch.setattr(setup, "write_activate_local", lambda *a, **kw: None)

    setup.build_hipdnn()

    assert setup.install_prefix == str(install_prefix)
    assert setup.plugin_engines_dir == (
        install_prefix / "lib" / "hipdnn_plugins" / "engines"
    )

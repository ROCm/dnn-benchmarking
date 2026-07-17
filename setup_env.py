#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Cross-platform setup for the dnn-benchmark tool.

Invoke directly with the *system* interpreter, before any venv exists::

    python3 setup_env.py [options]        # Linux
    py -3 setup_env.py [options]          # Windows

Creates and owns a venv under the workspace, installs torch per ``--torch-mode``,
editable-installs the benchmark package, and (for ROCm/CPU/none source builds)
builds hipDNN + the provider plugins and wires up the hipDNN Python bindings.

Stdlib only: this file must import and run under the system interpreter before
the package it installs exists. Never import torch/numpy/third-party at module
top level; any such probe runs in a subprocess against the *venv* interpreter.
"""

import argparse
import functools
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from typing import NoReturn


IS_WINDOWS = platform.system() == "Windows"

SCRIPT_DIR = Path(__file__).resolve().parent
ROCM_LIBRARIES_DIR = SCRIPT_DIR / "rocm-libraries"
HIPDNN_ROOT = ROCM_LIBRARIES_DIR / "projects" / "hipdnn"
BUILD_DIR = HIPDNN_ROOT / "build"
DEFAULT_ROCM_PREFIX = "/opt/rocm"

PROVIDERS_DIR = ROCM_LIBRARIES_DIR / "dnn-providers"
MIOPEN_PROVIDER_DIR = PROVIDERS_DIR / "miopen-provider"

# Engine plugin providers built from source alongside hipDNN, as
# (name, source dir, extra configure flags). Each builds in <source dir>/build.
# Windows builds only the MIOpen provider (see build_and_install_windows).
PROVIDERS = (
    (
        "MIOpen provider",
        MIOPEN_PROVIDER_DIR,
        ["-DMIOPENPROVIDER_SKIP_TESTS=ON"],
    ),
    (
        "hipBLASLt provider",
        PROVIDERS_DIR / "hipblaslt-provider",
        ["-DHIPDNN_SKIP_TESTS=ON"],
    ),
    (
        "hip-kernel-provider",
        PROVIDERS_DIR / "hip-kernel-provider",
        ["-DHIPKERNELPROVIDER_ENABLE_TESTS=OFF", "-DENABLE_ASM_SDPA_ENGINE=ON"],
    ),
)

ROCM_NIGHTLY_BASE = "https://rocm.nightlies.amd.com"

# Only archs with published Windows torch wheels work (gfx1151 has them,
# gfx1150 does not). Matches wheel_build_setup.ps1's default target.
WINDOWS_DEFAULT_GPU_ARCH = "gfx1151"

# ROCm nightly bucket per GPU arch. gfx90a's current torch + ROCm SDK builds
# live in the bare "gfx90a" bucket; the older "gfx90X-dcgpu" family bucket is
# frozen at a release that predates several SDK libraries (e.g. hipdnn).
# gfx942/gfx950 are still served by their "-dcgpu" family buckets.
LINUX_TORCH_BUCKETS = {
    "gfx90a": "gfx90a",
    "gfx942": "gfx94X-dcgpu",
    "gfx950": "gfx950-dcgpu",
}


# --- Small process helpers -------------------------------------------------


def fail(*lines: str) -> NoReturn:
    """Print an error (possibly multi-line) to stderr and exit 1."""
    for line in lines:
        print(line, file=sys.stderr)
    sys.exit(1)


def run(cmd, *, env=None, check=True, **kwargs):
    """Run a command; raise CalledProcessError on failure."""
    return subprocess.run(list(cmd), env=env, check=check, **kwargs)


def run_git(args, **kwargs):
    """Run git with the given args; raise on nonzero exit."""
    return run(["git", *args], **kwargs)


def git_output(args) -> str:
    """Run git and return its stdout, stripped; raise on nonzero exit."""
    result = subprocess.run(["git", *args], capture_output=True, text=True, check=True)
    return result.stdout.strip()


def default_workspace() -> Path:
    """Env override, else /workspace when present and writable (Linux only),
    else <script_dir>/.workspace."""
    env_ws = os.environ.get("DNN_BENCH_WORKSPACE")
    if env_ws:
        return Path(env_ws)
    if not IS_WINDOWS:
        ws = Path("/workspace")
        if ws.is_dir() and os.access(ws, os.W_OK):
            return ws
    return SCRIPT_DIR / ".workspace"


def venv_python(venv_dir: Path) -> Path:
    """Platform interpreter path inside a venv."""
    if IS_WINDOWS:
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


# --- Probes ----------------------------------------------------------------
# These inspect the interpreter's own sys.prefix/sysconfig, so they must run in
# the VENV interpreter, never the system one.
#
# The devel prefix has no probe: `rocm-sdk path --root` is a supported entry
# point that resolves it (see _rocm_sdk_devel_root).

_FIND_ROCM_WHEEL_PREFIX = r"""
from pathlib import Path
import sys
import sysconfig

kind = sys.argv[1]
venv_root = Path(sys.prefix).resolve()

roots = []
for key in ("purelib", "platlib"):
    value = sysconfig.get_path(key)
    if value:
        path = Path(value).resolve()
        if path == venv_root or venv_root in path.parents:
            roots.append(path)

matches = {}
for root in roots:
    if not root.is_dir():
        continue
    for child in root.iterdir():
        if not child.is_dir():
            continue
        if kind == "libraries":
            if not child.name.startswith("_rocm_sdk_libraries_"):
                continue
            lib_dir = child.joinpath("lib")
            if not lib_dir.is_dir() or not any(lib_dir.glob("libMIOpen.so*")):
                continue
        elif kind == "core":
            if not child.name.startswith("_rocm_sdk_core"):
                continue
            amd_smi_dir = child.joinpath("share/amd_smi")
            if not (
                child.joinpath("lib").is_dir()
                and any(child.joinpath("lib").glob("libamd_smi.so*"))
                and amd_smi_dir.is_dir()
                and (
                    amd_smi_dir.joinpath("setup.py").is_file()
                    or amd_smi_dir.joinpath("pyproject.toml").is_file()
                )
            ):
                continue
        else:
            print(f"ERROR: unknown ROCm wheel prefix kind: {kind}", file=sys.stderr)
            sys.exit(2)
        matches[child.resolve()] = child

if len(matches) == 1:
    print(next(iter(matches.values())))
    sys.exit(0)
if len(matches) > 1:
    print(f"ERROR: multiple usable ROCm SDK {kind} prefixes found:", file=sys.stderr)
    for path in sorted(matches.values()):
        print(f"  {path}", file=sys.stderr)
    print("Use a clean workspace/venv so setup cannot mix ROCm SDK packages.", file=sys.stderr)
    sys.exit(2)
sys.exit(1)
"""

_AMDSMI_IMPORTABLE = r"""
import sys

try:
    import amdsmi  # noqa: F401
except Exception:
    sys.exit(1)
"""

# `import torch` from a ROCm wheel with no visible GPU can print SDK probe
# warnings to stdout, which is captured here; emit the mode on its own final
# line (the leading newline guards a warning lacking one) and read only that.
_GET_TORCH_MODE = r"""
try:
    import torch
except Exception:
    mode = "missing"
else:
    if getattr(torch.version, "hip", None):
        mode = "rocm"
    elif getattr(torch.version, "cuda", None):
        mode = "cuda"
    else:
        mode = "cpu"
print("\n" + mode)
"""

# Reads the installed rocm-sdk-core version, so the devel toolchain can be
# re-pinned to match it (see _sync_rocm_sdk_devel_version). Empty output means
# rocm-sdk-core isn't installed.
_GET_ROCM_SDK_CORE_VERSION = r"""
try:
    from importlib.metadata import version
    print(version("rocm-sdk-core"))
except Exception:
    pass
"""


# --- CLI -------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="setup_env.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Set up the dnn-benchmark tool. Requires Python 3.12 or newer.",
        epilog=(
            "The selected ROCm prefix is exported as ROCM_PATH and its lib\n"
            "directory is prepended to LD_LIBRARY_PATH by the venv activation\n"
            "script (Linux). dnn-benchmarking infers plugins from:\n"
            "  $ROCM_PATH/lib/hipdnn_plugins/engines/"
        ),
    )
    parser.add_argument(
        "--torch-mode",
        choices=["rocm", "cuda", "cpu", "existing", "none"],
        default="rocm",
        help=(
            "Select how torch is provided. Default: rocm\n"
            "  rocm: install ROCm torch nightly, use ROCm libraries/toolchain "
            "from the torch wheel's bundled ROCm SDK packages, and build local "
            "hipDNN/provider artifacts when absent.\n"
            "  cuda: install CUDA torch from PyPI (or --torch-index-url) for the "
            "PyTorch execution backend only. hipDNN bindings, engine plugins, "
            "and ROCm setup are skipped.\n"
            "  cpu:  install CPU-only torch and build bindings against installed "
            "ROCm/hipDNN.\n"
            "  existing: reuse torch already present in the venv. ROCm torch uses "
            "its bundled SDK libraries; CUDA torch skips hipDNN/ROCm setup; CPU "
            "torch uses installed ROCm/hipDNN.\n"
            "  none: leave torch uninstalled and build bindings against installed "
            "ROCm/hipDNN."
        ),
    )
    parser.add_argument(
        "--reuse-venv",
        action="store_true",
        help="Reuse an existing venv instead of deleting it.",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=default_workspace(),
        help=(
            "Workspace root for the venv, Python bytecode cache, and runtime "
            "benchmark caches. The virtual environment is <path>/.venv. "
            "Default: %(default)s"
        ),
    )
    parser.add_argument(
        "--torch-index-url",
        default="",
        help="Override the pip index URL used for torch.",
    )
    parser.add_argument(
        "--gpu-arch",
        default="",
        help=(
            "Override GPU architecture detection for ROCm torch nightly "
            "selection. Supported on Linux: gfx90a, gfx942, gfx950. On Windows "
            f"any arch with published wheels; defaults to {WINDOWS_DEFAULT_GPU_ARCH}."
        ),
    )
    parser.add_argument(
        "--rocm-prefix",
        default="",
        help=(
            "Explicit ROCm/hipDNN prefix for binding/provider builds. Takes "
            "precedence over venv discovery."
        ),
    )
    parser.add_argument(
        "--reuse-artifacts",
        action="store_true",
        help=(
            "Skip building hipDNN/the provider plugins from source and use "
            "whatever is already installed in the selected ROCm prefix (e.g. a "
            "prior build in the same workspace). Fails if hipDNN is absent there "
            "-- this never falls back to building."
        ),
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Skip confirmation prompts.",
    )
    return parser


# --- Orchestration ---------------------------------------------------------


class Setup:
    """Holds resolved config and venv state for one setup run."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.torch_mode = args.torch_mode
        self.reuse_artifacts = args.reuse_artifacts
        self.reuse_venv = args.reuse_venv
        self.auto_yes = args.yes
        self.rocm_prefix = args.rocm_prefix
        self.gpu_arch_override = args.gpu_arch
        self.torch_index_url = args.torch_index_url
        self.resolved_torch_index_url = ""
        self.installed_torch_mode = "missing"
        self.plugin_engines_dir = None

        # do_build defaults on; --reuse-artifacts / cuda mode turn it off.
        self.do_build = not self.reuse_artifacts
        if self.torch_mode == "cuda":
            self.do_build = False
        if self.torch_mode == "existing":
            self.reuse_venv = True

        # Resolve: this path is written into activate.local, which is sourced
        # from arbitrary working directories.
        self.workspace = Path(args.workspace).resolve()
        self.venv_dir = self.workspace / ".venv"

        # Child-process environment; PYTHONPYCACHEPREFIX/DNN_BENCH_WORKSPACE and
        # (later) ROCM_PATH are layered onto this before subprocess use.
        self.env = dict(os.environ)

    # -- interpreters -------------------------------------------------------

    @property
    def py(self) -> str:
        """The venv interpreter path (as str) for all post-venv subprocesses."""
        return str(venv_python(self.venv_dir))

    def pip(self, *args: str, env=None) -> None:
        run([self.py, "-m", "pip", *args], env=env if env is not None else self.env)

    def probe(self, code: str, *probe_args: str):
        """Run a probe body in the venv interpreter; return CompletedProcess."""
        return subprocess.run(
            [self.py, "-c", code, *probe_args],
            capture_output=True,
            text=True,
            env=self.env,
        )

    # -- python version -----------------------------------------------------

    @staticmethod
    def require_python_version() -> None:
        if sys.version_info < (3, 12):
            version = ".".join(str(p) for p in sys.version_info[:3])
            fail(
                f"ERROR: setup_env.py requires Python >= 3.12, but the invoking "
                f"interpreter is {version}. Run setup with a Python 3.12+ environment."
            )

    # -- rocm-libraries checkout --------------------------------------------

    def ensure_rocm_libraries_checkout(self) -> None:
        """Fetch rocm-libraries if absent.

        It is a git submodule (see .gitmodules) tracking develop by default, so
        `git submodule update --init` also works. This is the fast path used only
        when the directory isn't already populated: a sparse, blobless clone of
        the .gitmodules-pinned branch limited to the two subtrees this tool builds
        (projects/hipdnn, dnn-providers), skipping the rest of the ~9GB monorepo.
        To build against a different ref, check it out directly, e.g.
        `git -C rocm-libraries fetch --depth 1 origin <ref> &&
         git -C rocm-libraries checkout FETCH_HEAD`.
        """
        if (ROCM_LIBRARIES_DIR / ".git").exists():
            return
        gitmodules = str(SCRIPT_DIR / ".gitmodules")
        url = git_output(["config", "-f", gitmodules, "submodule.rocm-libraries.url"])
        branch = git_output(
            ["config", "-f", gitmodules, "submodule.rocm-libraries.branch"]
        )
        print(
            f"Fetching rocm-libraries ({branch}) via sparse checkout "
            "(projects/hipdnn, dnn-providers)..."
        )
        if ROCM_LIBRARIES_DIR.exists():
            shutil.rmtree(ROCM_LIBRARIES_DIR)
        run_git(
            [
                "clone",
                "--quiet",
                "--filter=blob:none",
                "--sparse",
                "--no-checkout",
                url,
                str(ROCM_LIBRARIES_DIR),
            ]
        )
        run_git(
            [
                "-C",
                str(ROCM_LIBRARIES_DIR),
                "sparse-checkout",
                "set",
                "projects/hipdnn",
                "dnn-providers",
            ]
        )
        run_git(
            [
                "-C",
                str(ROCM_LIBRARIES_DIR),
                "fetch",
                "--quiet",
                "--depth",
                "1",
                "origin",
                branch,
            ]
        )
        run_git(["-C", str(ROCM_LIBRARIES_DIR), "checkout", "--quiet", "FETCH_HEAD"])

    # -- venv lifecycle -----------------------------------------------------

    def setup_venv(self) -> None:
        # Windows ROCm source builds install into the ROCm wheel env that
        # wheel_build_setup.ps1 owns. Every other mode keeps the workspace venv,
        # so a stray ROCM_WHEEL_VENV in the environment cannot hijack --workspace.
        if IS_WINDOWS and self.torch_mode == "rocm" and self.do_build:
            self._select_windows_wheel_venv()

        if self.torch_mode == "existing" and not self.venv_dir.is_dir():
            fail(
                f"ERROR: --torch-mode existing requires an existing virtual "
                f"environment at {self.venv_dir}.",
                "Use --torch-mode rocm or --torch-mode cpu to create one and "
                "install torch automatically.",
            )
        if self.venv_dir.is_dir():
            if self.reuse_venv:
                print(f"Reusing existing virtual environment at {self.venv_dir}...")
            else:
                print(f"Removing existing virtual environment at {self.venv_dir}...")
                shutil.rmtree(self.venv_dir)
        if not self.venv_dir.is_dir():
            print(f"Creating virtual environment at {self.venv_dir}...")
            run([sys.executable, "-m", "venv", str(self.venv_dir)])

        self.env["PYTHONPYCACHEPREFIX"] = str(self.workspace / "pycache")
        self.env["DNN_BENCH_WORKSPACE"] = str(self.workspace)
        if not IS_WINDOWS:
            # Windows has no activate.local sourcing; the child env above covers
            # subprocesses there instead.
            self.write_activate_local()

        self.installed_torch_mode = self.get_torch_mode()
        # An existing venv with CUDA torch reaches the CUDA skip path even when
        # --torch-mode existing was passed; hipDNN can't be built there, so keep
        # the build decision consistent.
        if self.installed_torch_mode == "cuda":
            self.do_build = False

    def _select_windows_wheel_venv(self) -> None:
        wheel_venv = os.environ.get("ROCM_WHEEL_VENV")
        if not wheel_venv:
            wheel_venv_path = self.workspace / "rocm-wheel-venv"
            if wheel_venv_path.is_dir():
                # wheel_build_setup.ps1 prompts (Read-Host "Pull new wheels?")
                # whenever its venv already exists, and -y cannot answer that;
                # reuse the env rather than re-running the bootstrap.
                print(f"Reusing existing ROCm wheel env at {wheel_venv_path}")
            else:
                # wheel_build_setup.ps1 lives inside the rocm-libraries submodule,
                # so the checkout has to precede the bootstrap.
                self.ensure_rocm_libraries_checkout()
                print(
                    "ROCM_WHEEL_VENV not set; bootstrapping a wheel env via "
                    "wheel_build_setup.ps1"
                )
                # The script publishes its venv path as $env:ROCM_WHEEL_VENV in
                # its own process, which cannot cross the subprocess boundary;
                # pass -VenvPath so the target is known without reading it back.
                run(
                    [
                        "pwsh",
                        str(
                            HIPDNN_ROOT
                            / "scripts"
                            / "windows"
                            / "wheel_build_setup.ps1"
                        ),
                        "-GpuTarget",
                        self.gpu_arch,
                        "-VenvPath",
                        str(wheel_venv_path),
                    ],
                    env=self.env,
                )
            wheel_venv = str(wheel_venv_path)
        # Reuse the wheel env in place; never recreate it.
        self.venv_dir = Path(wheel_venv)
        self.reuse_venv = True

    def write_activate_local(self, rocm_prefix: str = "", lib_dir: str = "") -> None:
        """Write the venv's activate.local and make activate source it.

        PYTHONPYCACHEPREFIX redirects Python's bytecode cache away from a network
        home directory. It must be set before the interpreter starts (setting it
        from Python is too late for that process's own imports), so it belongs in
        the activation script rather than the child env.
        """
        lines = [
            f"export PYTHONPYCACHEPREFIX={shlex.quote(str(self.workspace / 'pycache'))}",
            f"export DNN_BENCH_WORKSPACE={shlex.quote(str(self.workspace))}",
        ]
        if rocm_prefix:
            lines.append(f"export ROCM_PATH={shlex.quote(rocm_prefix)}")
        if lib_dir:
            lines += [
                'case ":${LD_LIBRARY_PATH:-}:" in',
                f"    *:{lib_dir}:*) ;;",
                f"    *) export LD_LIBRARY_PATH={shlex.quote(lib_dir)}"
                "${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} ;;",
                "esac",
            ]
        (self.venv_dir / "bin" / "activate.local").write_text("\n".join(lines) + "\n")

        activate = self.venv_dir / "bin" / "activate"
        if "activate.local" not in activate.read_text():
            with activate.open("a") as fh:
                fh.write(
                    'source "$(dirname "${BASH_SOURCE[0]}")/activate.local" '
                    "2>/dev/null || true\n"
                )

    # -- torch mode probe ---------------------------------------------------

    def get_torch_mode(self) -> str:
        result = self.probe(_GET_TORCH_MODE)
        if result.returncode != 0:
            return "missing"
        lines = [ln for ln in result.stdout.splitlines() if ln.strip()]
        return lines[-1].strip() if lines else "missing"

    def require_torch_mode(self, expected: str) -> None:
        if self.installed_torch_mode != expected:
            fail(
                f"ERROR: --torch-mode {expected} requested, but {self.venv_dir} "
                f"contains torch mode '{self.installed_torch_mode}'.",
                "Use a clean workspace or remove the existing virtual environment "
                "before changing torch modes.",
            )

    # -- GPU arch -----------------------------------------------------------

    @functools.cached_property
    def gpu_arch(self) -> str:
        """--gpu-arch, else detection, else the Windows default.

        Linux keeps "" when detection fails so the unsupported-arch error below
        can name it. On Windows there is no rocm_agent_enumerator to detect with,
        so the default stands in.
        """
        arch = self.gpu_arch_override or self._detect_gpu_arch()
        if not arch and IS_WINDOWS:
            arch = WINDOWS_DEFAULT_GPU_ARCH
        return arch

    @functools.cached_property
    def hip_arch_args(self):
        """GPU_TARGETS/AMDGPU_TARGETS flags for the HIP device-code builds.

        The wheel-bundled ROCm SDK ships no rocm_agent_enumerator/offload-arch on
        PATH and the build may run with no GPU, so HIP cannot autodetect the
        offload arch. Pass it explicitly via hipDNN's documented GPU_TARGETS
        rather than letting HIP fall back to a default target list.
        """
        if not self.gpu_arch:
            return []
        return [
            f"-DGPU_TARGETS={self.gpu_arch}",
            f"-DAMDGPU_TARGETS={self.gpu_arch}",
        ]

    @staticmethod
    def _detect_gpu_arch() -> str:
        if shutil.which("rocm_agent_enumerator"):
            out = subprocess.run(
                ["rocm_agent_enumerator"], capture_output=True, text=True, check=False
            ).stdout
            for line in out.splitlines():
                if "gfx9" in line:
                    return line.strip()
        if shutil.which("rocminfo"):
            out = subprocess.run(
                ["rocminfo"], capture_output=True, text=True, check=False
            ).stdout
            match = re.search(r"gfx\d+[a-z0-9]*", out)
            if match:
                return match.group(0)
        return ""

    # -- ROCm wheel prefix discovery ----------------------------------------

    def find_rocm_wheel_prefix(self, kind: str):
        """Return (prefix_or_None, status): 0 found, 1 none found, 2 error
        (already reported to stderr by the probe)."""
        result = self.probe(_FIND_ROCM_WHEEL_PREFIX, kind)
        if result.returncode == 0:
            return result.stdout.strip(), 0
        if result.stderr:
            sys.stderr.write(result.stderr)
        return None, result.returncode

    def require_rocm_wheel_libraries_prefix(self) -> str:
        prefix, status = self.find_rocm_wheel_prefix("libraries")
        if status == 0:
            return prefix
        if status != 1:
            # Several prefixes, which setup must not silently mix; the probe
            # has already said which.
            sys.exit(1)
        fail(
            "ERROR: no usable ROCm SDK libraries package found in this venv.",
            "Expected exactly one _rocm_sdk_libraries_* package containing "
            "MIOpen libraries.",
            "Use a ROCm torch wheel that includes ROCm SDK libraries, or pass "
            "--rocm-prefix explicitly.",
        )

    def _rocm_sdk_devel_root(self, report_errors: bool = False):
        """The rocm-sdk-devel root prefix, or None when it isn't installed.

        rocm-sdk-devel ships its payload as a tarball (wheels cannot carry the
        symlinks it needs) that has to be expanded before use. `rocm-sdk path
        --root` does that expansion on first call and returns the same
        _rocm_sdk_devel_<platform> prefix on every later one, so this single
        supported entry point covers both discovery and initialization.
        """
        result = subprocess.run(
            [self.py, "-m", "rocm_sdk", "path", "--root"],
            capture_output=True,
            text=True,
            env=self.env,
        )
        if result.returncode != 0:
            if report_errors and result.stderr:
                sys.stderr.write(result.stderr)
            return None
        lines = [ln for ln in result.stdout.splitlines() if ln.strip()]
        return lines[-1].strip() if lines else None

    def ensure_rocm_wheel_devel_prefix(self, index_url: str) -> str:
        root = self._rocm_sdk_devel_root()
        if root:
            return root

        if not index_url:
            fail(
                "ERROR: no ROCm SDK compiler/toolchain prefix found in this venv.",
                "Expected the rocm-sdk-devel package (rocm[devel]) alongside the "
                "rocm_sdk module.",
                "Install rocm-sdk-devel from the same ROCm torch index, or pass "
                "--rocm-prefix.",
            )

        print(
            "ROCm SDK compiler/toolchain prefix not found; installing "
            f"rocm-sdk-devel from {index_url}...",
            file=sys.stderr,
        )
        self.pip("install", "--pre", "rocm-sdk-devel", "--index-url", index_url)

        root = self._rocm_sdk_devel_root(report_errors=True)
        if root:
            return root
        fail(
            "ERROR: rocm-sdk-devel installed, but `rocm-sdk path --root` could "
            "not resolve the devel prefix (see the error above).",
            "The rocm_sdk module comes from the `rocm` package; installing "
            "rocm[devel] pins both at matching versions.",
        )

    def sync_rocm_sdk_devel_version(self, index_url: str) -> None:
        """Re-pin rocm-sdk-devel to match the installed rocm-sdk-core version.

        Windows bootstraps its wheel venv via wheel_build_setup.ps1, which
        installs rocm-sdk-devel from the latest nightly *before* install_torch()
        runs. install_torch() then installs torch pinned to a (usually older,
        independently-versioned) nightly that pulls its own rocm-sdk-core,
        downgrading it in the same venv -- but the already-expanded devel tree
        is never resynced. hipDNN then builds against newer devel headers than
        the runtime DLLs actually installed, and the compiled extension fails
        to import with a DLL symbol mismatch ("procedure could not be found").
        Linux doesn't hit this: it installs devel from resolved_torch_index_url
        *after* install_torch(), so the two are pinned together from the start.

        No-op when index_url is empty (e.g. --torch-mode existing, where there
        is no resolved nightly index to re-pin from).
        """
        if not index_url:
            return
        core_version = self.probe(_GET_ROCM_SDK_CORE_VERSION).stdout.strip()
        if not core_version:
            return
        devel_version = self.probe(
            "from importlib.metadata import version, PackageNotFoundError\n"
            "try:\n"
            '    print(version("rocm-sdk-devel"))\n'
            "except PackageNotFoundError:\n"
            "    pass\n"
        ).stdout.strip()
        if devel_version == core_version:
            return
        print(
            f"rocm-sdk-devel ({devel_version or 'missing'}) doesn't match the "
            f"installed rocm-sdk-core ({core_version}); re-pinning devel to "
            f"{core_version} from {index_url}..."
        )
        # --force-reinstall: pip treats an already-satisfied exact-version pin
        # as a no-op otherwise, which would leave a stale expanded devel tree
        # (rocm-sdk-devel's own package dir) in place with the old payload.
        self.pip(
            "install",
            "--pre",
            "--force-reinstall",
            f"rocm-sdk-devel=={core_version}",
            "--index-url",
            index_url,
        )

    # -- amdsmi (powers the GPU SMI snapshot; optional) ---------------------

    def amdsmi_importable(self) -> bool:
        return self.probe(_AMDSMI_IMPORTABLE).returncode == 0

    @staticmethod
    def _is_amdsmi_source(path: Path) -> bool:
        return path.is_dir() and (
            (path / "setup.py").is_file() or (path / "pyproject.toml").is_file()
        )

    def maybe_install_amdsmi(self, *prefixes: str) -> None:
        if self.amdsmi_importable():
            return

        candidates = []
        prefix, status = self.find_rocm_wheel_prefix("core")
        if status == 0:
            candidates.append(Path(prefix) / "share" / "amd_smi")
        elif status != 1:
            print(
                "Warning: ROCm SDK core discovery failed; skipping SDK amdsmi "
                "candidate.",
                file=sys.stderr,
            )
        candidates += [
            Path(prefix) / "share" / "amd_smi" for prefix in prefixes if prefix
        ]

        seen = set()
        for candidate in candidates:
            if candidate in seen or not self._is_amdsmi_source(candidate):
                continue
            seen.add(candidate)
            print(f"Installing amdsmi Python bindings from {candidate}...")
            try:
                self.pip("install", "-e", str(candidate))
            except subprocess.CalledProcessError:
                pass
            else:
                if self.amdsmi_importable():
                    return
            print(
                f"Warning: amdsmi install from {candidate} failed; trying next "
                "candidate.",
                file=sys.stderr,
            )

        print(
            "Warning: amdsmi Python bindings were not installed; GPU SMI "
            "snapshot will be disabled.",
            file=sys.stderr,
        )

    # -- hipDNN prefix selection --------------------------------------------

    @staticmethod
    def hipdnn_config_path(prefix: str) -> Path:
        return Path(prefix) / "lib/cmake/hipdnn_frontend/hipdnn_frontendConfig.cmake"

    @staticmethod
    def hipdnn_backend_config_path(prefix: str) -> Path:
        return Path(prefix) / "lib/cmake/hipdnn_backend/hipdnn_backendConfig.cmake"

    def prefix_has_hipdnn(self, prefix: str) -> bool:
        return (
            self.hipdnn_config_path(prefix).is_file()
            and self.hipdnn_backend_config_path(prefix).is_file()
        )

    def resolve_installed_rocm_prefix(self) -> str:
        if self.rocm_prefix:
            return self.rocm_prefix
        return self.env.get("ROCM_PATH") or DEFAULT_ROCM_PREFIX

    def select_binding_prefix(self) -> str:
        # An explicit --rocm-prefix wins. Otherwise the prefix follows the torch
        # mode (where ROCm comes from), NOT whether the build is on (the default)
        # or skipped (--reuse-artifacts): that only controls *whether* hipDNN is
        # rebuilt, not *where*. In rocm/existing-rocm mode that is the torch
        # wheel's bundled SDK, so a from-source build works with no system ROCm.
        if self.rocm_prefix:
            return self.resolve_installed_rocm_prefix()
        if self.torch_mode == "rocm":
            return self.require_rocm_wheel_libraries_prefix()
        if self.torch_mode == "existing" and self.installed_torch_mode == "rocm":
            return self.require_rocm_wheel_libraries_prefix()
        return self.resolve_installed_rocm_prefix()

    @functools.cached_property
    def toolchain_prefix(self) -> str:
        """ROCm compiler/devel prefix for the binding and provider builds.

        Same rule as select_binding_prefix: it follows the torch mode. For
        rocm/existing-rocm that is the wheel's devel SDK (clang + lib/cmake/hip),
        which a from-source build must use too -- the libraries wheel ships no
        compiler.
        """
        if self.rocm_prefix:
            return self.resolve_installed_rocm_prefix()
        if self.torch_mode == "rocm":
            return self.ensure_rocm_wheel_devel_prefix(self.resolved_torch_index_url)
        if self.torch_mode == "existing" and self.installed_torch_mode == "rocm":
            return self.ensure_rocm_wheel_devel_prefix("")
        return self.resolve_installed_rocm_prefix()

    # -- torch install ------------------------------------------------------

    def _rocm_torch_index_url(self) -> str:
        """Nightly index for the resolved GPU arch."""
        print(f"GPU arch: {self.gpu_arch}")
        if IS_WINDOWS:
            # Windows torch wheels are published per-arch under v2/, not in the
            # dcgpu family buckets Linux uses.
            return f"{ROCM_NIGHTLY_BASE}/v2/{self.gpu_arch}/"
        bucket = LINUX_TORCH_BUCKETS.get(self.gpu_arch)
        if bucket is None:
            fail(
                f"ERROR: Unsupported GPU architecture '{self.gpu_arch or 'none'}'.",
                "Supported: gfx90a (MI200/MI210/MI250), gfx942 (MI300X/MI300A), "
                "gfx950 (MI350)",
                "Pass --gpu-arch or --torch-index-url to override detection.",
            )
        return f"{ROCM_NIGHTLY_BASE}/v2-staging/{bucket}/"

    def install_torch(self) -> None:
        mode = self.torch_mode
        if mode == "none":
            print("Leaving torch uninstalled.")
            return

        if mode == "existing":
            if self.installed_torch_mode == "missing":
                fail(
                    "ERROR: --torch-mode existing requires torch to already be "
                    f"installed in {self.venv_dir}.",
                    "Use --torch-mode rocm or --torch-mode cpu to install torch "
                    "automatically.",
                )
            print(f"Using existing PyTorch in {self.venv_dir}.")
            return

        if mode == "cpu":
            index_url = self.torch_index_url or "https://download.pytorch.org/whl/cpu"
            if self.installed_torch_mode != "missing":
                self.require_torch_mode("cpu")
                print(f"Using existing CPU-only PyTorch in {self.venv_dir}.")
                return
            print(f"Installing CPU-only PyTorch from {index_url}")
            self.pip("install", "torch", "--index-url", index_url)
        elif mode == "cuda":
            if self.installed_torch_mode != "missing":
                self.require_torch_mode("cuda")
                print(f"Using existing CUDA PyTorch in {self.venv_dir}.")
                return
            if self.torch_index_url:
                print(f"Installing CUDA PyTorch from {self.torch_index_url}")
                self.pip("install", "torch", "--index-url", self.torch_index_url)
            else:
                print("Installing CUDA PyTorch from PyPI")
                self.pip("install", "torch")
        elif mode == "rocm":
            index_url = self.torch_index_url or self._rocm_torch_index_url()
            self.resolved_torch_index_url = index_url
            if self.installed_torch_mode != "missing":
                self.require_torch_mode("rocm")
                print(f"Using existing ROCm PyTorch in {self.venv_dir}.")
                return
            print(f"Installing ROCm PyTorch from {index_url}")
            # ROCm nightlies are pre-release, so --pre lets pip select them.
            self.pip("install", "--pre", "torch", "--index-url", index_url)

        self.installed_torch_mode = self.get_torch_mode()
        self.require_torch_mode(mode)

    # -- Linux build backend ------------------------------------------------

    def _build_env(self):
        # hipDNN's ClangToolChain warns when ROCM_PATH leaks via the environment
        # (e.g. from a prior run's activate.local); clear it for the build and
        # pass the prefix as -DROCM_PATH instead.
        build_env = dict(self.env)
        build_env.pop("ROCM_PATH", None)
        return build_env

    def _cmake_paths(self, install_prefix: str, toolchain_prefix: str):
        cmake_prefix_path = install_prefix
        if toolchain_prefix != install_prefix:
            cmake_prefix_path = f"{install_prefix};{toolchain_prefix}"
        cmake_program_path = f"{toolchain_prefix}/bin;{toolchain_prefix}/lib/llvm/bin"
        return cmake_prefix_path, cmake_program_path

    def build_hipdnn(self, install_prefix: str, toolchain_prefix: str) -> None:
        cmake_prefix_path, cmake_program_path = self._cmake_paths(
            install_prefix, toolchain_prefix
        )
        print(f"Building and installing hipDNN to {install_prefix}...")
        print(f"Using ROCm compiler/devel prefix: {toolchain_prefix}")
        if BUILD_DIR.exists():
            shutil.rmtree(BUILD_DIR)
        build_env = self._build_env()
        run(
            [
                "cmake",
                "-S",
                str(HIPDNN_ROOT),
                "-B",
                str(BUILD_DIR),
                "-DCMAKE_BUILD_TYPE=Release",
                *self.hip_arch_args,
                f"-DCMAKE_INSTALL_PREFIX={install_prefix}",
                f"-DCMAKE_PREFIX_PATH={cmake_prefix_path}",
                f"-DCMAKE_PROGRAM_PATH={cmake_program_path}",
                f"-DROCM_PATH={toolchain_prefix}",
                "-DHIPDNN_SKIP_TESTS=ON",
                "-DHIPDNN_ENABLE_SDPA=ON",
                "-DENABLE_CLANG_FORMAT=OFF",
                "-DENABLE_CLANG_TIDY=OFF",
            ],
            env=build_env,
        )
        run(["cmake", "--build", str(BUILD_DIR)], env=build_env)
        run(["cmake", "--install", str(BUILD_DIR)], env=build_env)

    def build_provider(
        self,
        name: str,
        provider_dir: Path,
        install_prefix: str,
        toolchain_prefix: str,
        extra_args,
    ) -> bool:
        if not provider_dir.is_dir():
            print(f"Warning: {name} not found at {provider_dir}", file=sys.stderr)
            return False

        build_dir = provider_dir / "build"
        cmake_prefix_path, cmake_program_path = self._cmake_paths(
            install_prefix, toolchain_prefix
        )
        print(f"Building and installing {name} to {install_prefix}...")
        print(f"Using ROCm compiler/devel prefix: {toolchain_prefix}")
        if build_dir.exists():
            shutil.rmtree(build_dir)
        build_env = self._build_env()
        try:
            run(
                [
                    "cmake",
                    "-S",
                    str(provider_dir),
                    "-B",
                    str(build_dir),
                    "-DCMAKE_BUILD_TYPE=Release",
                    *self.hip_arch_args,
                    f"-DCMAKE_INSTALL_PREFIX={install_prefix}",
                    f"-DCMAKE_PREFIX_PATH={cmake_prefix_path}",
                    f"-DCMAKE_PROGRAM_PATH={cmake_program_path}",
                    f"-DROCM_PATH={toolchain_prefix}",
                    "-DENABLE_CLANG_FORMAT=OFF",
                    "-DENABLE_CLANG_TIDY=OFF",
                    *extra_args,
                ],
                env=build_env,
            )
            run(["cmake", "--build", str(build_dir)], env=build_env)
            run(["cmake", "--install", str(build_dir)], env=build_env)
        except subprocess.CalledProcessError:
            return False
        return True

    @staticmethod
    def has_engine_plugins(plugin_dir: Path) -> bool:
        return plugin_dir.is_dir() and any(plugin_dir.glob("*.so"))

    @staticmethod
    def prepend_ld_library_path(env: dict, lib_dir: str) -> None:
        current = env.get("LD_LIBRARY_PATH", "")
        parts = current.split(":") if current else []
        if lib_dir in parts:
            return
        env["LD_LIBRARY_PATH"] = lib_dir + (f":{current}" if current else "")

    def build_and_install_linux(self, binding_prefix: str) -> None:
        if self.do_build:
            self.build_hipdnn(binding_prefix, self.toolchain_prefix)
        elif not self.prefix_has_hipdnn(binding_prefix):
            # Reachable only via --reuse-artifacts (cuda mode exits earlier).
            fail(
                "ERROR: --reuse-artifacts was passed, but hipDNN CMake configs "
                f"were not found under {binding_prefix}.",
                "Expected:",
                f"  {self.hipdnn_config_path(binding_prefix)}",
                f"  {self.hipdnn_backend_config_path(binding_prefix)}",
                "There is nothing to reuse there yet (e.g. the ROCm wheel "
                "currently ships hipDNN's runtime but not its devel/CMake "
                "artifacts).",
                "Drop --reuse-artifacts to build from source, or point "
                "--rocm-prefix at a prefix that has them.",
            )

        # Provider plugins are only ever built alongside hipDNN: --reuse-artifacts
        # promises never to build, and any other path has just rebuilt hipDNN, so
        # its plugins must be rebuilt against it.
        provider_build_failed = False
        if self.do_build:
            for name, provider_dir, extra_args in PROVIDERS:
                if not self.build_provider(
                    name,
                    provider_dir,
                    binding_prefix,
                    self.toolchain_prefix,
                    extra_args,
                ):
                    print(
                        f"Warning: {name} plugin build failed; continuing with "
                        "any available providers.",
                        file=sys.stderr,
                    )
                    provider_build_failed = True

        plugin_dir = Path(binding_prefix) / "lib/hipdnn_plugins/engines"
        plugins_available = self.has_engine_plugins(plugin_dir)
        if not plugins_available:
            print(
                f"Warning: no native hipDNN engine plugins were found in "
                f"{plugin_dir}.",
                file=sys.stderr,
            )
            print(
                "Setup will still finish, but default hipDNN benchmark runs need "
                "engine plugins.",
                file=sys.stderr,
            )
            print(
                "Pass --plugin-path or config plugin_path to use custom provider "
                "plugins.",
                file=sys.stderr,
            )
        if provider_build_failed:
            print(
                "Warning: one or more provider plugins failed to build.",
                file=sys.stderr,
            )
            print(
                "Continuing with available or user-specified plugins.",
                file=sys.stderr,
            )

        print("")
        if plugins_available:
            print(f"hipDNN plugins available at: {plugin_dir}/")
        else:
            print(f"hipDNN plugin search path: {plugin_dir}/ (no .so files found)")

        self.env["ROCM_PATH"] = binding_prefix
        self.prepend_ld_library_path(self.env, f"{binding_prefix}/lib")
        self.write_activate_local(binding_prefix, f"{binding_prefix}/lib")

        self.maybe_install_amdsmi(
            binding_prefix,
            self.toolchain_prefix,
            self.rocm_prefix,
            DEFAULT_ROCM_PREFIX,
        )
        self.build_and_install_bindings_linux(binding_prefix, self.toolchain_prefix)

    def build_and_install_bindings_linux(
        self, binding_prefix: str, toolchain_prefix: str
    ) -> None:
        # The bindings are a standalone CMake project (python/frontend_bindings)
        # plus a wheel packer (python/frontend_wheel_package/pack_frontend_wheel.py):
        # build the nanobind extension against the installed hipDNN, pack a wheel,
        # then install it. `pip install -e HIPDNN_ROOT/python` does not work --
        # python/ has no pyproject.toml.
        python_dir = HIPDNN_ROOT / "python"
        bindings_src = python_dir / "frontend_bindings"
        bindings_build_dir = python_dir / "build" / "frontend_bindings"
        wheel_dir = python_dir / "build" / "wheel_package"
        packer = python_dir / "frontend_wheel_package" / "pack_frontend_wheel.py"

        cmake_prefix_path = binding_prefix
        if toolchain_prefix != binding_prefix:
            cmake_prefix_path = f"{binding_prefix};{toolchain_prefix}"

        self.pip("install", "build")
        # Wipe any stale CMake cache (it can reference deleted pip temp envs).
        py_build_root = python_dir / "build"
        if py_build_root.exists():
            shutil.rmtree(py_build_root)

        binding_env = dict(self.env)
        binding_env["ROCM_PATH"] = toolchain_prefix
        run(
            [
                "cmake",
                "-S",
                str(bindings_src),
                "-B",
                str(bindings_build_dir),
                "-GNinja",
                "-DCMAKE_BUILD_TYPE=Release",
                f"-DCMAKE_PREFIX_PATH={cmake_prefix_path}",
                f"-DPython_EXECUTABLE={self.py}",
            ],
            env=binding_env,
        )
        run(["cmake", "--build", str(bindings_build_dir)], env=binding_env)
        run(
            [
                self.py,
                str(packer),
                "--build-dir",
                str(bindings_build_dir),
                "--wheel-dir",
                str(wheel_dir),
            ],
            env=self.env,
        )
        wheels = sorted(wheel_dir.glob("hipdnn_frontend-*.whl"))
        if len(wheels) != 1:
            fail(f"ERROR: expected exactly one hipdnn_frontend wheel in {wheel_dir}")
        self.pip("install", "--force-reinstall", str(wheels[0]))

    # -- Windows build backend ----------------------------------------------
    # Unverified against a real Windows/ROCm host; CI's rocm-build job exercises
    # compile + link only (no GPU).

    @staticmethod
    def _fwd(path: str) -> str:
        """Forward-slash a Windows path: backslashes would be read as escapes
        inside the .bat command lines."""
        return path.replace("\\", "/")

    def _windows_toolchain(self):
        """Locate cmake, ninja, vcvars64 and the Windows SDK."""
        cmake_exe = shutil.which("cmake")
        if not cmake_exe:
            fail("cmake not found on PATH.")
        ninja_exe = shutil.which("ninja")
        if not ninja_exe:
            fail("ninja not found on PATH.")

        vcvars = None
        vswhere = (
            r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
        )
        if Path(vswhere).exists():
            vs_path = subprocess.run(
                [
                    vswhere,
                    "-latest",
                    "-products",
                    "*",
                    "-requires",
                    "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                    "-property",
                    "installationPath",
                ],
                capture_output=True,
                text=True,
                check=False,
            ).stdout.strip()
            if vs_path:
                vcvars = str(Path(vs_path) / "VC/Auxiliary/Build/vcvars64.bat")
        if not vcvars or not Path(vcvars).exists():
            fallback = r"C:\develop\dist\vs-buildtools\VC\Auxiliary\Build\vcvars64.bat"
            if Path(fallback).exists():
                vcvars = fallback
        if not vcvars or not Path(vcvars).exists():
            fail("vcvars64.bat not found.")

        winsdk_root = r"C:\Program Files (x86)\Windows Kits\10"
        lib_root = Path(winsdk_root) / "Lib"
        versions = []
        if lib_root.is_dir():
            for child in lib_root.iterdir():
                if child.is_dir() and child.name[:1].isdigit() and "." in child.name:
                    versions.append(child.name)
        if not versions:
            fail(f"No Windows SDK found under {winsdk_root}\\Lib.")
        return cmake_exe, ninja_exe, vcvars, winsdk_root, sorted(versions)[-1]

    @staticmethod
    def _cmake_stages(cmake_exe: str, source: str, build: str, configure_args):
        """Turn one CMake project into its configure/build/install command lines
        for the toolchain .bat."""
        cmake = f'"{cmake_exe}"'
        src = '"{0}"'.format(Setup._fwd(source))
        bld = '"{0}"'.format(Setup._fwd(build))
        return [
            f"{cmake} -S {src} -B {bld} {' '.join(configure_args)}",
            f"{cmake} --build {bld}",
            f"{cmake} --install {bld}",
        ]

    def _invoke_toolchain_build(
        self, title, commands, vcvars, winsdk_root, winsdk_version, best_effort=False
    ) -> bool:
        """Run CMake inside an MSVC + Windows SDK env via a throwaway .bat
        (vcvars64 can't be sourced into this process); append the SDK lib/include
        paths an unregistered BuildTools instance can't locate."""
        lines = [
            "@echo off",
            'set "PATH=C:\\Program Files (x86)\\Microsoft Visual Studio\\Installer;%PATH%"',
            f'call "{vcvars}" >nul || (echo VCVARS FAILED & exit /b 1)',
            f'set "WINSDK={winsdk_root}"',
            f'set "WINSDKVER={winsdk_version}"',
            'set "LIB=%LIB%;%WINSDK%\\Lib\\%WINSDKVER%\\um\\x64;%WINSDK%\\Lib\\%WINSDKVER%\\ucrt\\x64"',
            'set "INCLUDE=%INCLUDE%;%WINSDK%\\Include\\%WINSDKVER%\\um;%WINSDK%\\Include\\%WINSDKVER%\\ucrt;%WINSDK%\\Include\\%WINSDKVER%\\shared"',
            # Clear a leaked ROCM_PATH; the prefix is passed as -DROCM_PATH.
            'set "ROCM_PATH="',
        ]
        for command in commands:
            lines.append(command)
            lines.append("if errorlevel 1 exit /b 1")

        bat = Path(tempfile.gettempdir()) / f"hipdnn_build_{uuid.uuid4().hex}.bat"
        bat.write_text("\r\n".join(lines) + "\r\n", encoding="ascii")
        print(f"==> {title}")
        code = subprocess.run(
            ["cmd", "/c", str(bat)], env=self.env, check=False
        ).returncode
        bat.unlink(missing_ok=True)
        if code != 0:
            if best_effort:
                print(
                    f"WARNING: {title} failed (exit {code}); continuing.",
                    file=sys.stderr,
                )
                return False
            fail(f"{title} failed (exit {code}).")
        return True

    def build_and_install_windows(self) -> None:
        bindings_root = HIPDNN_ROOT / "python"
        bindings_src = bindings_root / "frontend_bindings"
        bindings_build = BUILD_DIR / "python"
        bindings_package = bindings_build / "wheel_package"
        bindings_pack_script = (
            bindings_root / "frontend_wheel_package" / "pack_frontend_wheel.py"
        )
        build_type = "Release"
        install_dir = str(HIPDNN_ROOT / "install")

        if self.do_build:
            # devel prefix: --rocm-prefix, else _rocm_sdk_devel wheel discovery.
            if self.rocm_prefix:
                wheel = self.rocm_prefix
            else:
                # See sync_rocm_sdk_devel_version: wheel_build_setup.ps1's
                # bootstrap and install_torch() can pin rocm-sdk-devel and
                # rocm-sdk-core to different nightlies in the same venv.
                self.sync_rocm_sdk_devel_version(self.resolved_torch_index_url)
                wheel = self._rocm_sdk_devel_root(report_errors=True)
                if not wheel:
                    fail(
                        "Building from source needs the ROCm devel wheel "
                        f"(_rocm_sdk_devel) in {self.py}'s env, or pass "
                        "--rocm-prefix."
                    )

            cmake_exe, ninja_exe, vcvars, winsdk_root, winsdk_version = (
                self._windows_toolchain()
            )
            print(f"Toolchain: cmake={cmake_exe} ninja={ninja_exe}")
            print(f"           vcvars={vcvars}  winsdk={winsdk_version}")
            print(
                f"           rocm={wheel}  gpu={self.gpu_arch}  install={install_dir}"
            )

            wheel_fwd = self._fwd(wheel)
            ninja_fwd = self._fwd(ninja_exe)
            python_fwd = self._fwd(self.py)
            install_fwd = self._fwd(install_dir)

            hipdnn_args = [
                "-GNinja",
                f"-DCMAKE_BUILD_TYPE={build_type}",
                f'-DCMAKE_CXX_COMPILER="{wheel_fwd}/lib/llvm/bin/clang++.exe"',
                f'-DCMAKE_MAKE_PROGRAM="{ninja_fwd}"',
                f'-DCMAKE_PREFIX_PATH="{wheel_fwd}"',
                f'-DROCM_CMAKE_PATH="{wheel_fwd}"',
                f'-DROCM_PATH="{wheel_fwd}"',
                f'-DPython_EXECUTABLE="{python_fwd}"',
                f"-DGPU_TARGETS={self.gpu_arch}",
                f"-DAMDGPU_TARGETS={self.gpu_arch}",
                "-DHIPDNN_SKIP_TESTS=ON",
                # clang-format/-tidy are required-by-default dev lints that
                # hard-fail configure when the tools aren't on PATH.
                "-DENABLE_CLANG_FORMAT=OFF",
                "-DENABLE_CLANG_TIDY=OFF",
                f'-DCMAKE_INSTALL_PREFIX="{install_fwd}"',
            ]
            self._invoke_toolchain_build(
                "Building + installing hipDNN",
                self._cmake_stages(
                    cmake_exe, str(HIPDNN_ROOT), str(BUILD_DIR), hipdnn_args
                ),
                vcvars,
                winsdk_root,
                winsdk_version,
            )

            # Python bindings: standalone CMake project against the installed
            # hipDNN artifacts, packed into the frontend wheel layout.
            bindings_args = [
                "-GNinja",
                f"-DCMAKE_BUILD_TYPE={build_type}",
                f'-DCMAKE_CXX_COMPILER="{wheel_fwd}/lib/llvm/bin/clang++.exe"',
                f'-DCMAKE_MAKE_PROGRAM="{ninja_fwd}"',
                f'-DCMAKE_PREFIX_PATH="{install_fwd};{wheel_fwd}"',
                f'-DPython_EXECUTABLE="{python_fwd}"',
            ]
            bindings_stages = self._cmake_stages(
                cmake_exe, str(bindings_src), str(bindings_build), bindings_args
            )
            # The packer stages the extension into a wheel via `python -m build`.
            self.pip("install", "build")
            bindings_pack_cmd = (
                '"{0}" "{1}" --build-dir "{2}" --wheel-dir "{3}"'.format(
                    python_fwd,
                    self._fwd(str(bindings_pack_script)),
                    self._fwd(str(bindings_build)),
                    self._fwd(str(bindings_package)),
                )
            )
            self._invoke_toolchain_build(
                "Building hipDNN Python bindings",
                [bindings_stages[0], bindings_stages[1], bindings_pack_cmd],
                vcvars,
                winsdk_root,
                winsdk_version,
            )

            if MIOPEN_PROVIDER_DIR.is_dir():
                prov_args = [
                    "-GNinja",
                    f"-DCMAKE_BUILD_TYPE={build_type}",
                    f'-DCMAKE_MAKE_PROGRAM="{ninja_fwd}"',
                    f'-DCMAKE_PREFIX_PATH="{install_fwd};{wheel_fwd}"',
                    f'-DROCM_CMAKE_PATH="{wheel_fwd}"',
                    f'-DROCM_PATH="{wheel_fwd}"',
                    f"-DGPU_TARGETS={self.gpu_arch}",
                    f"-DAMDGPU_TARGETS={self.gpu_arch}",
                    "-DMIOPENPROVIDER_SKIP_TESTS=ON",
                    "-DENABLE_CLANG_FORMAT=OFF",
                    "-DENABLE_CLANG_TIDY=OFF",
                    f'-DCMAKE_INSTALL_PREFIX="{install_fwd}"',
                ]
                self._invoke_toolchain_build(
                    "Building + installing MIOpen provider",
                    self._cmake_stages(
                        cmake_exe,
                        str(MIOPEN_PROVIDER_DIR),
                        str(MIOPEN_PROVIDER_DIR / "build"),
                        prov_args,
                    ),
                    vcvars,
                    winsdk_root,
                    winsdk_version,
                    best_effort=True,
                )
                # Report on the artifact, not the build's exit status -- the
                # plugin .dll landing in engines/ is the signal that matters. The
                # RUNTIME .dll installs under bin/; lib/ is the Linux layout.
                for candidate in (
                    Path(install_dir) / "bin/hipdnn_plugins/engines",
                    Path(install_dir) / "lib/hipdnn_plugins/engines",
                ):
                    if candidate.is_dir() and any(candidate.glob("*.dll")):
                        self.plugin_engines_dir = candidate
                        break
                if self.plugin_engines_dir:
                    print(f"==> MIOpen plugin installed to {self.plugin_engines_dir}")
                else:
                    print(
                        "WARNING: MIOpen provider produced no plugin under "
                        f"{install_dir}\\{{bin,lib}}\\hipdnn_plugins\\engines "
                        "(see build output above).",
                        file=sys.stderr,
                    )
            else:
                print(
                    f"WARNING: MIOpen provider not found at {MIOPEN_PROVIDER_DIR}; "
                    "skipping.",
                    file=sys.stderr,
                )

            self._install_windows_bindings(bindings_package, install_dir)
        elif self.probe("import hipdnn_frontend").returncode != 0:
            print(
                "WARNING: --reuse-artifacts was passed and hipdnn_frontend is not "
                "importable in this env. Re-run without --reuse-artifacts to build "
                "and install the bindings from source.",
                file=sys.stderr,
            )

        # amdsmi ships with the HIP SDK rather than PyPI, so point the shared
        # installer at the SDK roots. Its ROCm-wheel probe globs libamd_smi.so*
        # and simply finds nothing here, which it treats as "no candidate".
        self.maybe_install_amdsmi(
            self.env.get("HIP_PATH", ""), self.env.get("ROCM_PATH", "")
        )

    def _install_windows_bindings(
        self, bindings_package: Path, install_dir: str
    ) -> None:
        """Install the packed wheel, then register the backend DLL directory.

        pack_frontend_wheel.py writes both the staged package and the .whl into
        bindings_package, so the wheel installs exactly as it does on Linux. The
        extension links hipdnn_backend.dll out of install_dir/bin, which it cannot
        find alone: extension modules load with LOAD_LIBRARY_SEARCH_DEFAULT_DIRS,
        so neither PATH nor RPATH applies. Register that directory from a .pth,
        stashing the handle on sys so it isn't GC'd before the import. The ROCm
        deps are preloaded by the package __init__ (rocm_sdk / ROCM_PATH).
        """
        wheels = sorted(bindings_package.glob("hipdnn_frontend-*.whl"))
        if len(wheels) != 1:
            fail(
                "ERROR: expected exactly one hipdnn_frontend wheel in "
                f"{bindings_package}"
            )
        self.pip("install", "--force-reinstall", str(wheels[0]))

        backend_bin = Path(install_dir) / "bin"
        if not (backend_bin / "hipdnn_backend.dll").exists():
            return
        site_pkgs = self.probe(
            "import sysconfig; print(sysconfig.get_path('purelib'))"
        ).stdout.strip()
        pth = Path(site_pkgs) / "hipdnn_frontend_dll_path.pth"
        print(f"==> Registering {backend_bin} for hipdnn_backend.dll via {pth}")
        pth.write_text(
            "import os, sys; _p = r'{0}'; "
            "sys.__dict__.setdefault('_hipdnn_dll_dirs', [])"
            ".append(os.add_dll_directory(_p))\n".format(backend_bin),
            encoding="ascii",
        )

    # -- confirmation prompt ------------------------------------------------

    def confirm_build(self) -> None:
        if not self.do_build or self.auto_yes:
            return
        # The exact install prefix depends on the torch mode and is only resolved
        # after torch is installed, so it is reported at build time, not here.
        confirm = input(
            "This will build and install hipDNN and provider plugins from "
            "source into the selected ROCm prefix. Continue? [Y/n] "
        )
        if confirm.strip() in ("n", "N"):
            print("Aborted.")
            sys.exit(0)

    # -- verify -------------------------------------------------------------

    def verify(self) -> None:
        print("==> Verifying installation")
        result = self.probe("import dnn_benchmarking; print('dnn_benchmarking OK')")
        sys.stdout.write(result.stdout)
        if result.returncode != 0:
            sys.stderr.write(result.stderr)
            fail("dnn_benchmarking failed to import.")

        result = self.probe("import hipdnn_frontend; print('hipdnn_frontend OK')")
        if result.returncode == 0:
            sys.stdout.write(result.stdout)
        else:
            print(
                "WARNING: hipdnn_frontend could not be imported (ROCm runtime or "
                "bindings missing).",
                file=sys.stderr,
            )

        result = subprocess.run(
            [self.py, "-m", "dnn_benchmarking", "--help"],
            capture_output=True,
            text=True,
            env=self.env,
        )
        if result.returncode != 0:
            sys.stderr.write(result.stderr)
            fail("dnn-benchmark CLI failed to run.")

    # -- top-level flow -----------------------------------------------------

    def run(self) -> int:
        self.require_python_version()
        self.workspace.mkdir(parents=True, exist_ok=True)

        self.confirm_build()
        self.setup_venv()
        print(f"Torch mode: {self.torch_mode}")

        # pyproject.toml intentionally omits torch so pip never replaces the
        # selected wheel; install it explicitly, before the package.
        self.install_torch()
        self.pip("install", "-e", str(SCRIPT_DIR))

        # CUDA torch supports only the PyTorch execution backend: no hipDNN
        # Python bindings, engine plugins, amdsmi, or ROCm prefix.
        if self.installed_torch_mode == "cuda" or self.torch_mode == "cuda":
            print("")
            print("CUDA torch selected: skipping hipDNN/provider builds, hipDNN Python")
            print("bindings, and ROCm environment setup.")
            print("")
            self.verify()
            self._print_complete(cuda=True)
            return 0

        # rocm-libraries provides the hipDNN sources and provider plugins.
        self.ensure_rocm_libraries_checkout()

        if self.gpu_arch:
            # Belt-and-suspenders for any torch C++/HIP extension compile (none
            # today: the bindings are nanobind host code linking hip::host).
            self.env.setdefault("PYTORCH_ROCM_ARCH", self.gpu_arch)

        if IS_WINDOWS:
            self.build_and_install_windows()
        else:
            binding_prefix = self.select_binding_prefix()
            print(f"Using hipDNN/ROCm prefix: {binding_prefix}")
            self.build_and_install_linux(binding_prefix)

        self.verify()
        self._print_complete()
        return 0

    def _print_complete(self, cuda: bool = False) -> None:
        print("")
        print("Setup complete. Activate the virtual environment with:")
        if IS_WINDOWS:
            print(f"  {self.venv_dir}\\Scripts\\Activate.ps1")
        else:
            print(f"  source {self.venv_dir}/bin/activate")
        print("")

        if cuda:
            print("Run PyTorch-backend benchmarks with:")
            print("  python -m dnn_benchmarking --graph <graph.json> --backend pytorch")
            return

        print("Run benchmarks with:")
        print("  python -m dnn_benchmarking --graph <graph.json>")

        if IS_WINDOWS:
            if self.plugin_engines_dir:
                # Windows installs plugins outside any prefix the runtime infers
                # from ROCM_PATH, so the path has to be passed explicitly.
                print("  python -m dnn_benchmarking --graph <graph.json> \\")
                print(f"      --plugin-path {self.plugin_engines_dir}")
            return

        print("")
        rocm_path = self.env.get("ROCM_PATH", "")
        print("The activation script sets:")
        print(f"  ROCM_PATH={rocm_path}")
        print(f"  LD_LIBRARY_PATH={rocm_path}/lib:${{LD_LIBRARY_PATH}}")
        print(
            "dnn-benchmarking infers plugins from "
            "$ROCM_PATH/lib/hipdnn_plugins/engines."
        )
        print(
            "Pass --plugin-path explicitly only when overriding the "
            "setup-installed plugins."
        )


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return Setup(args).run()
    except subprocess.CalledProcessError as exc:
        cmd = exc.cmd if isinstance(exc.cmd, str) else " ".join(str(c) for c in exc.cmd)
        print(f"ERROR: command failed (exit {exc.returncode}): {cmd}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nAborted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())

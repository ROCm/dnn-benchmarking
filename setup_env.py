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
from pathlib import Path
from typing import NoReturn


IS_WINDOWS = platform.system() == "Windows"

SCRIPT_DIR = Path(__file__).resolve().parent
ROCM_LIBRARIES_DIR = SCRIPT_DIR / "rocm-libraries"
HIPDNN_ROOT = ROCM_LIBRARIES_DIR / "projects" / "hipdnn"
DEFAULT_ROCM_PREFIX = "/opt/rocm"


ROCM_NIGHTLY_BASE = "https://rocm.nightlies.amd.com"

# Single multi-arch nightly index for every OS: GPU selection is a pip extra
# (torch[device-<arch>]) rather than a per-arch/per-OS URL bucket, so Linux
# and Windows always resolve the exact same torch/rocm-sdk-* build.
ROCM_TORCH_INDEX_URL = f"{ROCM_NIGHTLY_BASE}/whl-multi-arch/"

# Windows has no rocm_agent_enumerator/rocminfo to detect the local arch with,
# so fall back to a default target known to have published wheels.
WINDOWS_DEFAULT_GPU_ARCH = "gfx1151"


# --- Small process helpers -------------------------------------------------


def fail(*lines: str) -> NoReturn:
    """Print an error (possibly multi-line) to stderr and exit 1."""
    for line in lines:
        print(line, file=sys.stderr)
    sys.exit(1)


def run(cmd, *, env=None, check=True, **kwargs):
    """Run a command; raise CalledProcessError on failure."""
    return subprocess.run(list(cmd), env=env, check=check, **kwargs)


def require_working_cmake() -> str:
    """Return the PATH-selected CMake executable after a startup check."""
    cmake = shutil.which("cmake")
    if not cmake:
        fail("cmake not found on PATH.")
    try:
        result = subprocess.run(
            [cmake, "--version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    except OSError as error:
        fail(f"cmake at {cmake} cannot run: {error}")
    if result.returncode:
        fail(
            f"cmake at {cmake} failed its startup check (exit {result.returncode}).",
            result.stderr.strip()
            or "Fix the selected cmake executable or PATH before retrying.",
        )
    return cmake


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
            # whl-multi-arch ships a single arch-agnostic "_rocm_sdk_libraries"
            # package; older per-arch indexes (e.g. v2-staging) name it
            # "_rocm_sdk_libraries_<arch>". Match both.
            if child.name != "_rocm_sdk_libraries" and not child.name.startswith(
                "_rocm_sdk_libraries_"
            ):
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
        "--no-editable",
        dest="editable_install",
        action="store_false",
        help="Install dnn-benchmarking into the venv instead of linking to the source tree.",
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
            "selection (selects the torch[device-<arch>] extra). On Linux, "
            "detected via rocm_agent_enumerator/rocminfo when not passed; on "
            f"Windows, defaults to {WINDOWS_DEFAULT_GPU_ARCH} when not detected."
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
        self.auto_yes = args.yes
        self.rocm_prefix = args.rocm_prefix
        self.gpu_arch_override = args.gpu_arch
        self.torch_index_url = args.torch_index_url
        self.editable_install = args.editable_install
        self.resolved_torch_index_url = ""
        self.installed_torch_mode = "missing"
        self.plugin_engines_dir = None

        self.do_build = not self.reuse_artifacts and self.torch_mode != "cuda"

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
        the .gitmodules-pinned branch limited to the hipDNN/provider sources and
        root CMake tooling this script builds.
        To build against a different ref, check it out directly, e.g.
        `git -C rocm-libraries fetch --depth 1 origin <ref> &&
         git -C rocm-libraries checkout FETCH_HEAD`.
        """

        if (ROCM_LIBRARIES_DIR / ".git").exists():
            if not (ROCM_LIBRARIES_DIR / "cmake").is_dir():
                run_git(
                    [
                        "-C",
                        str(ROCM_LIBRARIES_DIR),
                        "sparse-checkout",
                        "add",
                        "cmake",
                    ]
                )
            return
        gitmodules = str(SCRIPT_DIR / ".gitmodules")
        url = git_output(["config", "-f", gitmodules, "submodule.rocm-libraries.url"])
        branch = git_output(
            ["config", "-f", gitmodules, "submodule.rocm-libraries.branch"]
        )
        print(
            f"Fetching rocm-libraries ({branch}) via sparse checkout "
            "(cmake, projects/hipdnn, dnn-providers)..."
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
                "cmake",
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
        if self.torch_mode == "existing" and not self.venv_dir.is_dir():
            fail(
                f"ERROR: --torch-mode existing requires an existing virtual "
                f"environment at {self.venv_dir}.",
                "Use --torch-mode rocm or --torch-mode cpu to create one and "
                "install torch automatically.",
            )
        if self.venv_dir.is_dir() and self.torch_mode != "existing":
            print(f"Removing existing virtual environment at {self.venv_dir}...")
            shutil.rmtree(self.venv_dir)
        if not self.venv_dir.is_dir():
            print(f"Creating virtual environment at {self.venv_dir}...")
            run([sys.executable, "-m", "venv", str(self.venv_dir)])

        self.env["PYTHONPYCACHEPREFIX"] = str(self.workspace / "pycache")
        self.env["DNN_BENCH_WORKSPACE"] = str(self.workspace)
        if not IS_WINDOWS:
            self.write_activate_local()

        self.installed_torch_mode = self.get_torch_mode()
        if self.installed_torch_mode == "cuda":
            self.do_build = False

    def write_activate_local(
        self, rocm_prefix: str = "", lib_dirs: tuple[str, ...] = ()
    ) -> None:
        """Write the venv's activate.local and make activate source it."""
        lines = [
            f"export PYTHONPYCACHEPREFIX={shlex.quote(str(self.workspace / 'pycache'))}",
            f"export DNN_BENCH_WORKSPACE={shlex.quote(str(self.workspace))}",
        ]
        if rocm_prefix:
            lines.append(f"export ROCM_PATH={shlex.quote(rocm_prefix)}")
        for lib_dir in reversed(lib_dirs):
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

        Linux keeps "" when detection fails so install_torch's missing-arch
        error can name it. On Windows there is no rocm_agent_enumerator to
        detect with, so the default stands in.
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
            "Expected a _rocm_sdk_libraries (or _rocm_sdk_libraries_<arch>) "
            "package containing MIOpen libraries.",
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

    def _require_gpu_arch(self) -> str:
        """The resolved --gpu-arch, or a fatal error naming what's missing."""
        if not self.gpu_arch:
            fail(
                "ERROR: could not detect a GPU architecture.",
                "Pass --gpu-arch (e.g. gfx90a, gfx942, gfx950) or --torch-index-url "
                "to override detection.",
            )
        return self.gpu_arch

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
            index_url = self.torch_index_url or ROCM_TORCH_INDEX_URL
            self.resolved_torch_index_url = index_url
            if self.installed_torch_mode != "missing":
                self.require_torch_mode("rocm")
                print(f"Using existing ROCm PyTorch in {self.venv_dir}.")
                return
            arch = self._require_gpu_arch()
            print(f"GPU arch: {arch}")
            print(f"Installing ROCm PyTorch from {index_url}")
            # ROCm nightlies are pre-release, so --pre lets pip select them.
            # device-<arch> is the pip extra that pulls the matching
            # rocm-sdk-device-<arch> package; the base rocm-sdk-core/-libraries
            # packages are OS/arch-agnostic on this index, so Linux and
            # Windows always resolve the exact same nightly.
            self.pip(
                "install",
                "--pre",
                f"torch[device-{arch}]",
                f"rocm[libraries,devel,device-{arch}]",
                "--index-url",
                index_url,
            )

        self.installed_torch_mode = self.get_torch_mode()
        self.require_torch_mode(mode)

    # -- Source build -------------------------------------------------------

    def _windows_host_env(self) -> dict:
        vswhere = Path(
            r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
        )
        vcvars = None
        if vswhere.is_file():
            result = subprocess.run(
                [
                    str(vswhere),
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
            )
            if result.stdout.strip():
                vcvars = (
                    Path(result.stdout.strip())
                    / "VC"
                    / "Auxiliary"
                    / "Build"
                    / "vcvars64.bat"
                )
        if not vcvars or not vcvars.is_file():
            for candidate in (
                Path(
                    r"C:\Program Files\Microsoft Visual Studio\2022"
                    r"\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
                ),
                Path(
                    r"C:\develop\dist\vs-buildtools" r"\VC\Auxiliary\Build\vcvars64.bat"
                ),
            ):
                if candidate.is_file():
                    vcvars = candidate
                    break
        if not vcvars or not vcvars.is_file():
            fail("Visual Studio 2022 C++ Build Tools and the Windows SDK are required.")

        result = subprocess.run(
            ["cmd", "/d", "/s", "/c", f'call "{vcvars}" >nul && set'],
            capture_output=True,
            text=True,
            check=True,
        )
        env = dict(self.env)
        for line in result.stdout.splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                env[key] = value
        return env

    def _build_env(self) -> dict:
        env = self._windows_host_env() if IS_WINDOWS else dict(self.env)
        env.pop("ROCM_PATH", None)
        return env

    @staticmethod
    def _cmake_paths(install_prefix: str, toolchain_prefix: str):
        prefix_path = install_prefix
        if toolchain_prefix != install_prefix:
            prefix_path = f"{install_prefix};{toolchain_prefix}"
        program_path = f"{toolchain_prefix}/bin;{toolchain_prefix}/lib/llvm/bin"
        return prefix_path, program_path

    def build_superbuild(self, install_prefix: str, toolchain_prefix: str) -> None:
        cmake = require_working_cmake()
        if not shutil.which("ninja"):
            fail("ninja not found on PATH.")

        build_dir = ROCM_LIBRARIES_DIR / "build"
        if build_dir.exists():
            shutil.rmtree(build_dir)
        prefix_path, program_path = self._cmake_paths(install_prefix, toolchain_prefix)
        print(f"Building hipDNN and providers to {install_prefix}...")
        run(
            [
                cmake,
                "--preset",
                "hipdnn-providers-all",
                "-GNinja",
                f"-DROCM_PATH={toolchain_prefix}",
                f"-DCMAKE_PREFIX_PATH={prefix_path}",
                f"-DCMAKE_PROGRAM_PATH={program_path}",
                f"-DCMAKE_INSTALL_PREFIX={install_prefix}",
                "-DROCM_LIBS_ENABLE_COMPONENTS=hipdnn;miopen-provider;"
                "hipblaslt-provider;hip-kernel-provider",
                *self.hip_arch_args,
                "-DHIPDNN_SKIP_TESTS=ON",
                "-DHIPDNN_ENABLE_SDPA=ON",
                "-DMIOPENPROVIDER_SKIP_TESTS=ON",
                "-DHIPKERNELPROVIDER_ENABLE_TESTS=OFF",
                "-DENABLE_ASM_SDPA_ENGINE=ON",
                "-DENABLE_CLANG_FORMAT=OFF",
                "-DENABLE_CLANG_TIDY=OFF",
            ],
            cwd=ROCM_LIBRARIES_DIR,
            env=self._build_env(),
        )
        run(
            [cmake, "--build", str(build_dir)],
            env=self._build_env(),
        )
        run(
            [cmake, "--install", str(build_dir)],
            env=self._build_env(),
        )

    def build_and_install_bindings(
        self, install_prefix: str, toolchain_prefix: str
    ) -> None:
        cmake = require_working_cmake()
        python_dir = HIPDNN_ROOT / "python"
        bindings_source = python_dir / "frontend_bindings"
        if not bindings_source.is_dir():
            fail(f"hipDNN frontend bindings not found at {bindings_source}")
        bindings_build = python_dir / "build" / "frontend_bindings"
        wheel_dir = python_dir / "build" / "wheel_package"
        build_root = python_dir / "build"
        if build_root.exists():
            shutil.rmtree(build_root)

        prefix_path, program_path = self._cmake_paths(install_prefix, toolchain_prefix)
        self.pip("install", "build")
        run(
            [
                cmake,
                "-S",
                str(bindings_source),
                "-B",
                str(bindings_build),
                "-GNinja",
                "-DCMAKE_BUILD_TYPE=Release",
                f"-DCMAKE_TOOLCHAIN_FILE={ROCM_LIBRARIES_DIR / 'cmake/toolchains/rocm-clang.cmake'}",
                f"-DROCM_PATH={toolchain_prefix}",
                f"-DCMAKE_PREFIX_PATH={prefix_path}",
                f"-DCMAKE_PROGRAM_PATH={program_path}",
                f"-DPython_EXECUTABLE={self.py}",
            ],
            env=self._build_env(),
        )
        run([cmake, "--build", str(bindings_build)], env=self._build_env())
        run(
            [
                self.py,
                str(python_dir / "frontend_wheel_package" / "pack_frontend_wheel.py"),
                "--build-dir",
                str(bindings_build),
                "--wheel-dir",
                str(wheel_dir),
            ],
            env=self.env,
        )
        wheels = sorted(wheel_dir.glob("hipdnn_frontend-*.whl"))
        if len(wheels) != 1:
            fail(f"ERROR: expected exactly one hipdnn_frontend wheel in {wheel_dir}")
        self.pip("install", "--force-reinstall", str(wheels[0]))

        if IS_WINDOWS:
            backend_bin = Path(install_prefix) / "bin"
            if (backend_bin / "hipdnn_backend.dll").exists():
                site_pkgs = self.probe(
                    "import sysconfig; print(sysconfig.get_path('purelib'))"
                ).stdout.strip()
                pth = Path(site_pkgs) / "hipdnn_frontend_dll_path.pth"
                pth.write_text(
                    "import os, sys; _p = r'{0}'; "
                    "sys.__dict__.setdefault('_hipdnn_dll_dirs', [])"
                    ".append(os.add_dll_directory(_p))\n".format(backend_bin),
                    encoding="ascii",
                )

    def build_and_install(self, install_prefix: str) -> None:
        toolchain_prefix = self.toolchain_prefix
        if self.do_build:
            self.build_superbuild(install_prefix, toolchain_prefix)
        elif not self.prefix_has_hipdnn(install_prefix):
            fail(
                "ERROR: --reuse-artifacts was passed, but hipDNN CMake configs "
                f"were not found under {install_prefix}.",
                "Drop --reuse-artifacts to build from source, or pass a usable "
                "--rocm-prefix.",
            )

        plugin_candidates = (
            Path(install_prefix) / "bin/hipdnn_plugins/engines",
            Path(install_prefix) / "lib/hipdnn_plugins/engines",
        )
        suffix = "*.dll" if IS_WINDOWS else "*.so"
        for candidate in plugin_candidates:
            if candidate.is_dir() and any(candidate.glob(suffix)):
                self.plugin_engines_dir = candidate
                break
        if not self.plugin_engines_dir:
            print(
                f"Warning: no native hipDNN engine plugins found under "
                f"{install_prefix}.",
                file=sys.stderr,
            )

        self.env["ROCM_PATH"] = install_prefix
        if not IS_WINDOWS:
            lib_dirs = tuple(
                str(Path(prefix) / "lib")
                for prefix in (toolchain_prefix, install_prefix)
                if prefix and (Path(prefix) / "lib").is_dir()
            )
            current = self.env.get("LD_LIBRARY_PATH", "")
            existing = current.split(":") if current else []
            ordered = [path for path in lib_dirs if path not in existing] + existing
            self.env["LD_LIBRARY_PATH"] = ":".join(ordered)
            self.write_activate_local(install_prefix, lib_dirs)

        self.maybe_install_amdsmi(
            install_prefix,
            toolchain_prefix,
            self.rocm_prefix,
            DEFAULT_ROCM_PREFIX,
        )
        if self.do_build:
            self.build_and_install_bindings(install_prefix, toolchain_prefix)
        elif self.probe("import hipdnn_frontend").returncode != 0:
            print(
                "WARNING: hipdnn_frontend is not importable in this environment.",
                file=sys.stderr,
            )

    # -- confirmation prompt ------------------------------------------------

    def confirm_build(self) -> None:
        if self.auto_yes:
            return
        actions = []
        if self.venv_dir.is_dir() and self.torch_mode != "existing":
            actions.append(f"replace the virtual environment at {self.venv_dir}")
        if self.do_build:
            actions.append("build and install hipDNN and provider plugins from source")
        if not actions:
            return
        confirm = input(f"This will {' and '.join(actions)}. Continue? [Y/n] ")
        if confirm.strip().lower() == "n":
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
        if self.editable_install:
            self.pip("install", "-e", str(SCRIPT_DIR))
        else:
            self.pip("install", str(SCRIPT_DIR))

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

        binding_prefix = self.select_binding_prefix()
        print(f"Using hipDNN/ROCm prefix: {binding_prefix}")
        self.build_and_install(binding_prefix)

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
        print(f"  LD_LIBRARY_PATH={self.env.get('LD_LIBRARY_PATH', '')}")
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

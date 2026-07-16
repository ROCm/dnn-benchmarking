#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Cross-platform setup for the dnn-benchmark tool.

Single entry point replacing the former ``setup.sh`` (bash) and ``setup.ps1``
(PowerShell). Invoke directly with the *system* interpreter before any venv
exists::

    python3 setup_env.py [options]        # Linux
    py -3 setup_env.py [options]           # Windows

It creates/owns a venv under the workspace, installs torch per ``--torch-mode``,
editable-installs the benchmark package, and (for ROCm/CPU/none source builds)
builds hipDNN + the provider plugins and wires the hipDNN Python bindings.

Stdlib only: this file must import and run under the system interpreter before
the package it installs exists. Never import torch/numpy/third-party at module
top level; any such probe runs in a subprocess against the *venv* interpreter.
"""

import argparse
import os
import platform
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


IS_WINDOWS = platform.system() == "Windows"

SCRIPT_DIR = Path(__file__).resolve().parent
ROCM_LIBRARIES_DIR = SCRIPT_DIR / "rocm-libraries"
HIPDNN_ROOT = ROCM_LIBRARIES_DIR / "projects" / "hipdnn"
WORKSPACE_ROOT = ROCM_LIBRARIES_DIR
BUILD_DIR = HIPDNN_ROOT / "build"
DEFAULT_ROCM_PREFIX = "/opt/rocm"

MIOPEN_PROVIDER_DIR = WORKSPACE_ROOT / "dnn-providers" / "miopen-provider"
MIOPEN_BUILD_DIR = MIOPEN_PROVIDER_DIR / "build"
HIP_KERNEL_PROVIDER_DIR = WORKSPACE_ROOT / "dnn-providers" / "hip-kernel-provider"
HIP_KERNEL_BUILD_DIR = HIP_KERNEL_PROVIDER_DIR / "build"
HIPBLASLT_PROVIDER_DIR = WORKSPACE_ROOT / "dnn-providers" / "hipblaslt-provider"
HIPBLASLT_BUILD_DIR = HIPBLASLT_PROVIDER_DIR / "build"


# --- Small process helpers -------------------------------------------------


def fail(*lines: str) -> "NoReturn":  # type: ignore[name-defined]
    """Print an error (possibly multi-line) to stderr and exit 1."""
    for line in lines:
        print(line, file=sys.stderr)
    sys.exit(1)


def run(cmd, *, env=None, check=True, **kwargs):
    """Run a command, echoing nothing extra; raise CalledProcessError on failure."""
    return subprocess.run(list(cmd), env=env, check=check, **kwargs)


def run_git(args, **kwargs):
    """Run git with the given args; raise on nonzero exit."""
    return run(["git", *args], **kwargs)


def default_workspace() -> Path:
    """Env default, else /workspace when present+writable (Linux only), else
    <script_dir>/.workspace (`setup.sh:9-16`)."""
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


def _sh_quote(value: str) -> str:
    """POSIX-shell-quote a value for the activate.local export lines."""
    return shlex.quote(value)


# --- Probes run in the VENV interpreter (inspect its sys.prefix/sysconfig) --
# Lifted verbatim in behavior from the setup.sh heredocs. They MUST run via the
# venv python, not the system python.

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
        elif kind == "devel":
            if not child.name.startswith("_rocm_sdk_devel"):
                continue
            if not (
                child.joinpath("lib/llvm/bin/clang").is_file()
                and child.joinpath("lib/llvm/bin/clang++").is_file()
                and child.joinpath("lib/cmake/hip/hip-config.cmake").is_file()
            ):
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

_EXPAND_ROCM_SDK_DEVEL = r"""
import importlib.util
import subprocess
import sys

if importlib.util.find_spec("rocm_sdk_devel") is None:
    sys.exit(1)

subprocess.run([sys.executable, "-m", "rocm_sdk", "init"], check=True)
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
    default_ws = default_workspace()
    default_venv = default_ws / ".venv"
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
        default=os.environ.get("DNN_BENCH_TORCH_MODE", "rocm"),
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
        default=None,
        help=(
            "Workspace root for the venv, Python bytecode cache, and runtime "
            f"benchmark caches. Default: {default_ws} "
            f"(the virtual environment is <path>/.venv, e.g. {default_venv})."
        ),
    )
    parser.add_argument(
        "--torch-index-url",
        default=os.environ.get("DNN_BENCH_TORCH_INDEX_URL", ""),
        help="Override the pip index URL used for torch.",
    )
    parser.add_argument(
        "--gpu-arch",
        default=os.environ.get("DNN_BENCH_GPU_ARCH", ""),
        help=(
            "Override GPU architecture detection for ROCm torch nightly "
            "selection. Supported: gfx90a, gfx942, gfx950."
        ),
    )
    parser.add_argument(
        "--rocm-prefix",
        default=os.environ.get("DNN_BENCH_ROCM_PREFIX", ""),
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
    """Holds resolved config + venv state; methods mirror the shell functions."""

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

        # do_build defaults on; --reuse-artifacts / cuda mode turn it off.
        self.do_build = not self.reuse_artifacts
        if self.torch_mode == "cuda":
            self.do_build = False
        if self.torch_mode == "existing":
            self.reuse_venv = True

        ws = args.workspace if args.workspace else str(default_workspace())
        self.workspace = Path(ws)
        self.venv_dir = self.workspace / ".venv"

        # Child-process environment; PYTHONPYCACHEPREFIX/DNN_BENCH_WORKSPACE and
        # (later) ROCM_PATH/CMAKE args are layered onto this before subprocess use.
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
                f"interpreter is {version}. "
                "Run setup with a Python 3.12+ environment."
            )

    # -- rocm-libraries submodule fast path (setup.sh:35-50) ----------------

    def ensure_rocm_libraries_checkout(self) -> None:
        if (ROCM_LIBRARIES_DIR / ".git").exists():
            return
        gitmodules = str(SCRIPT_DIR / ".gitmodules")
        url = subprocess.run(
            ["git", "config", "-f", gitmodules, "submodule.rocm-libraries.url"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        branch = subprocess.run(
            ["git", "config", "-f", gitmodules, "submodule.rocm-libraries.branch"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
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

    # -- venv lifecycle (setup.sh:823-868) ----------------------------------

    def setup_venv(self) -> None:
        # Windows back-compat (setup.ps1:244-268): if ROCM_WHEEL_VENV is set,
        # honor it as the target env exactly as setup.ps1 did. The
        # wheel_build_setup.ps1 bootstrap runs ONLY for rocm-mode source builds
        # — never for cuda/cpu/none (so CI's Windows cuda smoke never touches
        # ROCm infra). unverified — confirm on a Windows ROCm host.
        if IS_WINDOWS:
            wheel_venv = os.environ.get("ROCM_WHEEL_VENV")
            if not wheel_venv and self.torch_mode == "rocm" and self.do_build:
                # wheel_build_setup.ps1 lives inside the rocm-libraries
                # submodule; ensure it's checked out before invoking it (fixes
                # a pre-existing setup.ps1 ordering bug: Step 1's bootstrap ran
                # before Step 3's checkout, so a fresh clone with no venv
                # published yet couldn't find the script that publishes one).
                self.ensure_rocm_libraries_checkout()
                wheel_setup = (
                    HIPDNN_ROOT / "scripts" / "windows" / "wheel_build_setup.ps1"
                )
                print(
                    "ROCM_WHEEL_VENV not set; bootstrapping a wheel env via "
                    "wheel_build_setup.ps1"
                )
                gpu_arch = self.gpu_arch_override or "gfx1151"
                # wheel_build_setup.ps1 publishes its venv path by setting
                # $env:ROCM_WHEEL_VENV in its own process; that can never
                # propagate back to us across the subprocess boundary (unlike
                # setup.ps1, which dot-sourced it in the same PowerShell
                # session). Pass -VenvPath explicitly instead, so the target
                # is known without reading state back from the child.
                wheel_venv_path = self.workspace / "rocm-wheel-venv"
                run(
                    [
                        "pwsh",
                        str(wheel_setup),
                        "-GpuTarget",
                        gpu_arch,
                        "-VenvPath",
                        str(wheel_venv_path),
                    ],
                    env=self.env,
                )
                wheel_venv = str(wheel_venv_path)
            if wheel_venv:
                # Reuse the wheel venv in place (setup.ps1 never recreated it).
                self.venv_dir = Path(wheel_venv)
                self.reuse_venv = True

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

        pycache = str(self.workspace / "pycache")
        # Export cache/workspace into the child env used by every later step.
        self.env["PYTHONPYCACHEPREFIX"] = pycache
        self.env["DNN_BENCH_WORKSPACE"] = str(self.workspace)

        if not IS_WINDOWS:
            # Redirect Python's bytecode cache away from the network home dir by
            # injecting it into the venv activate script so it is set before the
            # interpreter starts (setting it in Python is too late for the
            # process's own imports). Windows has no activate.local sourcing
            # today; the child env above covers subprocesses there instead.
            self._write_activate_local_cache(pycache)

        self.installed_torch_mode = self.get_torch_mode()
        # An existing venv with CUDA torch reaches the CUDA skip path even when
        # --torch-mode existing was passed; hipDNN can't be built there, so keep
        # the build decision consistent.
        if self.installed_torch_mode == "cuda":
            self.do_build = False

    def _write_activate_local_cache(self, pycache: str) -> None:
        activate_local = self.venv_dir / "bin" / "activate.local"
        activate_local.write_text(
            f"export PYTHONPYCACHEPREFIX={_sh_quote(pycache)}\n"
            f"export DNN_BENCH_WORKSPACE={_sh_quote(str(self.workspace))}\n"
        )
        activate = self.venv_dir / "bin" / "activate"
        if "activate.local" not in activate.read_text():
            with activate.open("a") as fh:
                fh.write(
                    'source "$(dirname "${BASH_SOURCE[0]}")/activate.local" '
                    "2>/dev/null || true\n"
                )

    # -- torch mode probe (setup.sh:495-514) --------------------------------

    def get_torch_mode(self) -> str:
        result = self.probe(_GET_TORCH_MODE)
        if result.returncode != 0:
            return "missing"
        lines = [ln for ln in result.stdout.splitlines() if ln.strip()]
        return lines[-1].strip() if lines else "missing"

    def require_torch_mode(self, expected: str) -> None:
        status = self.installed_torch_mode or self.get_torch_mode()
        if status != expected:
            fail(
                f"ERROR: --torch-mode {expected} requested, but {self.venv_dir} "
                f"contains torch mode '{status}'.",
                "Use a clean workspace or remove the existing virtual environment "
                "before changing torch modes.",
            )

    # -- GPU arch detection (setup.sh:472-493) ------------------------------

    def detect_gpu_arch(self) -> str:
        import re

        if self.gpu_arch_override:
            return self.gpu_arch_override
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
            m = re.search(r"gfx\d+[a-z0-9]*", out)
            if m:
                return m.group(0)
        return ""

    # -- ROCm wheel prefix discovery (setup.sh:245-410) ---------------------

    def find_rocm_wheel_prefix(self, kind: str):
        """Return (prefix_or_None, exit_status) mirroring find_rocm_wheel_prefix.

        exit_status: 0 found (prefix set), 1 none found, 2 error (already
        printed to stderr by the probe / a hard fail here)."""
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
            sys.exit(1)
        fail(
            "ERROR: no usable ROCm SDK libraries package found in this venv.",
            "Expected exactly one _rocm_sdk_libraries_* package containing "
            "MIOpen libraries.",
            "Use a ROCm torch wheel that includes ROCm SDK libraries, pass "
            "--rocm-prefix explicitly, or pass --reuse-artifacts.",
        )

    def expand_rocm_sdk_devel(self) -> bool:
        result = self.probe(_EXPAND_ROCM_SDK_DEVEL)
        if result.stderr:
            sys.stderr.write(result.stderr)
        return result.returncode == 0

    def ensure_rocm_wheel_devel_prefix(self, index_url: str) -> str:
        prefix, status = self.find_rocm_wheel_prefix("devel")
        if status == 0:
            return prefix
        if status != 1:
            sys.exit(1)

        if self.expand_rocm_sdk_devel():
            prefix, status = self.find_rocm_wheel_prefix("devel")
            if status == 0:
                return prefix
            if status != 1:
                sys.exit(1)

        if not index_url:
            fail(
                "ERROR: no ROCm SDK compiler/toolchain prefix found in this venv.",
                "Expected exactly one _rocm_sdk_devel package with "
                "lib/llvm/bin/clang++ and hip CMake configs.",
                "Install rocm-sdk-devel from the same ROCm torch index, or pass "
                "--rocm-prefix.",
            )

        print(
            "ROCm SDK compiler/toolchain prefix not found; installing "
            f"rocm-sdk-devel from {index_url}...",
            file=sys.stderr,
        )
        self.pip("install", "--pre", "rocm-sdk-devel", "--index-url", index_url)

        if not self.expand_rocm_sdk_devel():
            fail(
                "ERROR: rocm-sdk-devel installed, but its devel payload could "
                "not be expanded."
            )

        prefix, status = self.find_rocm_wheel_prefix("devel")
        if status == 0:
            return prefix
        if status != 1:
            sys.exit(1)
        fail(
            "ERROR: rocm-sdk-devel installed, but no usable ROCm SDK "
            "compiler/toolchain prefix was found.",
            "Expected lib/llvm/bin/clang, lib/llvm/bin/clang++, and hip CMake "
            "configs under _rocm_sdk_devel.",
        )

    def amdsmi_importable(self) -> bool:
        return self.probe(_AMDSMI_IMPORTABLE).returncode == 0

    def maybe_install_amdsmi(self, *prefixes: str) -> None:
        if self.amdsmi_importable():
            return

        candidates = []
        prefix, status = self.find_rocm_wheel_prefix("core")
        if status == 0:
            candidates.append(str(Path(prefix) / "share" / "amd_smi"))
        elif status != 1:
            print(
                "Warning: ROCm SDK core discovery failed; skipping SDK amdsmi "
                "candidate.",
                file=sys.stderr,
            )

        for prefix in prefixes:
            if not prefix:
                continue
            candidate = Path(prefix) / "share" / "amd_smi"
            if candidate.is_dir() and (
                (candidate / "setup.py").is_file()
                or (candidate / "pyproject.toml").is_file()
            ):
                candidates.append(str(candidate))

        seen = set()
        for candidate in candidates:
            if candidate in seen:
                continue
            seen.add(candidate)
            cpath = Path(candidate)
            if not cpath.is_dir() or not (
                (cpath / "setup.py").is_file() or (cpath / "pyproject.toml").is_file()
            ):
                continue
            print(f"Installing amdsmi Python bindings from {candidate}...")
            try:
                self.pip("install", "-e", candidate)
            except subprocess.CalledProcessError:
                print(
                    f"Warning: amdsmi install from {candidate} failed; trying "
                    "next candidate.",
                    file=sys.stderr,
                )
                continue
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

    # -- hipDNN prefix predicates (setup.sh:230-243) ------------------------

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
        rocm_path = self.env.get("ROCM_PATH") or os.environ.get("ROCM_PATH")
        if rocm_path:
            return rocm_path
        return DEFAULT_ROCM_PREFIX

    def select_binding_prefix(self) -> str:
        # An explicit --rocm-prefix wins. Otherwise the prefix follows the torch
        # mode (where ROCm comes from), NOT whether the build is on/skipped.
        if self.rocm_prefix:
            return self.resolve_installed_rocm_prefix()
        if self.torch_mode == "rocm":
            return self.require_rocm_wheel_libraries_prefix()
        if self.torch_mode == "existing":
            if self.installed_torch_mode == "rocm":
                return self.require_rocm_wheel_libraries_prefix()
            return self.resolve_installed_rocm_prefix()
        # cpu | none
        return self.resolve_installed_rocm_prefix()

    def select_provider_toolchain_prefix(self) -> str:
        # Same rule as select_binding_prefix for the compiler/devel toolchain.
        if self.rocm_prefix:
            return self.resolve_installed_rocm_prefix()
        if self.torch_mode == "rocm":
            return self.ensure_rocm_wheel_devel_prefix(self.resolved_torch_index_url)
        if self.torch_mode == "existing":
            if self.installed_torch_mode == "rocm":
                return self.ensure_rocm_wheel_devel_prefix("")
            return self.resolve_installed_rocm_prefix()
        # cpu | none
        return self.resolve_installed_rocm_prefix()

    # -- torch install (setup.sh:540-618) -----------------------------------

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
            self.installed_torch_mode = self.get_torch_mode()
            self.require_torch_mode("cpu")
            return
        if mode == "cuda":
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
            self.installed_torch_mode = self.get_torch_mode()
            self.require_torch_mode("cuda")
            return
        if mode == "rocm":
            index_url = self.torch_index_url
            if not index_url:
                gpu_arch = self.detect_gpu_arch()
                # ROCm nightly bucket per GPU arch. gfx90a's current torch + ROCm
                # SDK builds live in the bare "gfx90a" bucket; the older
                # "gfx90X-dcgpu" family bucket is frozen at a release that
                # predates several SDK libraries (e.g. hipdnn). gfx942/gfx950 are
                # still served by their "-dcgpu" family buckets.
                buckets = {
                    "gfx90a": "gfx90a",
                    "gfx942": "gfx94X-dcgpu",
                    "gfx950": "gfx950-dcgpu",
                }
                index_bucket = buckets.get(gpu_arch)
                if index_bucket is None:
                    fail(
                        f"ERROR: Unsupported GPU architecture '{gpu_arch or 'none'}'.",
                        "Supported: gfx90a (MI200/MI210/MI250), gfx942 "
                        "(MI300X/MI300A), gfx950 (MI350)",
                        "Pass --gpu-arch or --torch-index-url to override "
                        "detection.",
                    )
                index_url = f"https://rocm.nightlies.amd.com/v2-staging/{index_bucket}/"
                print(f"Detected GPU: {gpu_arch}")
            self.resolved_torch_index_url = index_url
            if self.installed_torch_mode != "missing":
                self.require_torch_mode("rocm")
                print(f"Using existing ROCm PyTorch in {self.venv_dir}.")
                return
            print(f"Installing ROCm PyTorch from {index_url}")
            self.pip("install", "--pre", "torch", "--index-url", index_url)
            self.installed_torch_mode = self.get_torch_mode()
            self.require_torch_mode("rocm")
            return

    # -- Linux build backend (setup.sh:677-1013) ----------------------------

    def _build_env_no_rocm_path(self):
        # hipDNN's ClangToolChain warns when ROCM_PATH leaks via the environment;
        # clear it for the build and pass the prefix as -DROCM_PATH instead
        # (equivalent of `env -u ROCM_PATH`).
        build_env = dict(self.env)
        build_env.pop("ROCM_PATH", None)
        return build_env

    def build_hipdnn(self, install_prefix: str, toolchain_prefix: str) -> None:
        cmake_prefix_path = install_prefix
        cmake_program_path = f"{toolchain_prefix}/bin;{toolchain_prefix}/lib/llvm/bin"
        if toolchain_prefix != install_prefix:
            cmake_prefix_path = f"{install_prefix};{toolchain_prefix}"

        print(f"Building and installing hipDNN to {install_prefix}...")
        print(f"Using ROCm compiler/devel prefix: {toolchain_prefix}")
        if BUILD_DIR.exists():
            shutil.rmtree(BUILD_DIR)
        build_env = self._build_env_no_rocm_path()
        configure = [
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
        ]
        run(configure, env=build_env)
        run(["cmake", "--build", str(BUILD_DIR)], env=build_env)
        run(["cmake", "--install", str(BUILD_DIR)], env=build_env)

    def build_provider(
        self,
        name: str,
        provider_dir: Path,
        build_dir: Path,
        install_prefix: str,
        toolchain_prefix: str,
        extra_args,
    ) -> bool:
        cmake_prefix_path = install_prefix
        cmake_program_path = f"{toolchain_prefix}/bin;{toolchain_prefix}/lib/llvm/bin"
        if toolchain_prefix != install_prefix:
            cmake_prefix_path = f"{install_prefix};{toolchain_prefix}"

        if not provider_dir.is_dir():
            print(f"Warning: {name} not found at {provider_dir}", file=sys.stderr)
            return False

        print(f"Building and installing {name} to {install_prefix}...")
        print(f"Using ROCm compiler/devel prefix: {toolchain_prefix}")
        if build_dir.exists():
            shutil.rmtree(build_dir)
        build_env = self._build_env_no_rocm_path()
        configure = [
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
        ]
        try:
            run(configure, env=build_env)
            run(["cmake", "--build", str(build_dir)], env=build_env)
            run(["cmake", "--install", str(build_dir)], env=build_env)
        except subprocess.CalledProcessError:
            return False
        return True

    def build_miopen_provider(self, install_prefix: str, toolchain_prefix: str) -> bool:
        ok = self.build_provider(
            "MIOpen provider",
            MIOPEN_PROVIDER_DIR,
            MIOPEN_BUILD_DIR,
            install_prefix,
            toolchain_prefix,
            ["-DMIOPENPROVIDER_SKIP_TESTS=ON"],
        )
        if ok:
            print("")
            print(
                f"MIOpen plugin installed to: {install_prefix}/lib/hipdnn_plugins/engines/"
            )
        return ok

    def build_hipblaslt_provider(
        self, install_prefix: str, toolchain_prefix: str
    ) -> bool:
        return self.build_provider(
            "hipBLASLt provider",
            HIPBLASLT_PROVIDER_DIR,
            HIPBLASLT_BUILD_DIR,
            install_prefix,
            toolchain_prefix,
            ["-DHIPDNN_SKIP_TESTS=ON"],
        )

    def build_hip_kernel_provider(
        self, install_prefix: str, toolchain_prefix: str
    ) -> bool:
        return self.build_provider(
            "hip-kernel-provider",
            HIP_KERNEL_PROVIDER_DIR,
            HIP_KERNEL_BUILD_DIR,
            install_prefix,
            toolchain_prefix,
            ["-DHIPKERNELPROVIDER_ENABLE_TESTS=OFF", "-DENABLE_ASM_SDPA_ENGINE=ON"],
        )

    @staticmethod
    def try_build_optional_provider(name: str, fn, *args) -> bool:
        if fn(*args):
            return True
        print(
            f"Warning: {name} plugin build failed; continuing with any available "
            "providers.",
            file=sys.stderr,
        )
        return False

    @staticmethod
    def has_engine_plugins(plugin_dir: Path) -> bool:
        if not plugin_dir.is_dir():
            return False
        return any(plugin_dir.glob("*.so"))

    @staticmethod
    def warn_no_native_engine_plugins(plugin_dir: Path) -> None:
        print(
            f"Warning: no native hipDNN engine plugins were found in {plugin_dir}.",
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

    @staticmethod
    def prepend_ld_library_path(env: dict, lib_dir: str) -> None:
        current = env.get("LD_LIBRARY_PATH", "")
        parts = current.split(":") if current else []
        if lib_dir in parts:
            return
        env["LD_LIBRARY_PATH"] = lib_dir + (f":{current}" if current else "")

    def write_activation_local(self, rocm_prefix: str, lib_dir: str) -> None:
        activate_local = self.venv_dir / "bin" / "activate.local"
        pycache = str(self.workspace / "pycache")
        lines = [
            f"export PYTHONPYCACHEPREFIX={_sh_quote(pycache)}",
            f"export DNN_BENCH_WORKSPACE={_sh_quote(str(self.workspace))}",
            f"export ROCM_PATH={_sh_quote(rocm_prefix)}",
            'case ":${LD_LIBRARY_PATH:-}:" in',
            f"    *:{lib_dir}:*) ;;",
            f"    *) export LD_LIBRARY_PATH={_sh_quote(lib_dir)}"
            "${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} ;;",
            "esac",
        ]
        activate_local.write_text("\n".join(lines) + "\n")

    def build_and_install_linux(self, binding_prefix: str) -> None:
        provider_toolchain_prefix = ""
        if self.do_build or not self.prefix_has_hipdnn(binding_prefix):
            provider_toolchain_prefix = self.select_provider_toolchain_prefix()

        built_hipdnn = False
        if self.do_build:
            self.build_hipdnn(binding_prefix, provider_toolchain_prefix)
            built_hipdnn = True
        elif not self.prefix_has_hipdnn(binding_prefix):
            # Reachable only via --reuse-artifacts (cuda mode exits earlier).
            fail(
                "ERROR: --reuse-artifacts was passed, but hipDNN CMake configs "
                "were not",
                f"found under {binding_prefix}.",
                "Expected:",
                f"  {self.hipdnn_config_path(binding_prefix)}",
                f"  {self.hipdnn_backend_config_path(binding_prefix)}",
                "There is nothing to reuse there yet (e.g. the ROCm wheel "
                "currently ships",
                "hipDNN's runtime but not its devel/CMake artifacts). Drop "
                "--reuse-artifacts",
                "to build from source, or point --rocm-prefix at a prefix that "
                "has them.",
            )

        # 4. Build/install provider plugins if the selected prefix lacks them.
        plugin_dir = Path(binding_prefix) / "lib/hipdnn_plugins/engines"
        miopen_plugin = plugin_dir / "libmiopen_plugin.so"
        hipblaslt_plugin = plugin_dir / "libhipblaslt_plugin.so"
        hip_kernel_plugin = plugin_dir / "libhip_kernel_provider.so"

        if (
            self.do_build
            or built_hipdnn
            or not miopen_plugin.is_file()
            or not hipblaslt_plugin.is_file()
            or not hip_kernel_plugin.is_file()
        ):
            if not provider_toolchain_prefix:
                provider_toolchain_prefix = self.select_provider_toolchain_prefix()

        provider_build_failed = False
        if self.do_build or built_hipdnn or not miopen_plugin.is_file():
            if not self.try_build_optional_provider(
                "MIOpen provider",
                self.build_miopen_provider,
                binding_prefix,
                provider_toolchain_prefix,
            ):
                provider_build_failed = True
        if self.do_build or built_hipdnn or not hipblaslt_plugin.is_file():
            if not self.try_build_optional_provider(
                "hipBLASLt provider",
                self.build_hipblaslt_provider,
                binding_prefix,
                provider_toolchain_prefix,
            ):
                provider_build_failed = True
        if self.do_build or built_hipdnn or not hip_kernel_plugin.is_file():
            if not self.try_build_optional_provider(
                "hip-kernel-provider",
                self.build_hip_kernel_provider,
                binding_prefix,
                provider_toolchain_prefix,
            ):
                provider_build_failed = True

        native_engine_plugins_available = True
        if not self.has_engine_plugins(plugin_dir):
            native_engine_plugins_available = False
            self.warn_no_native_engine_plugins(plugin_dir)

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
        if native_engine_plugins_available:
            print(f"hipDNN plugins available at: {plugin_dir}/")
        else:
            print(f"hipDNN plugin search path: {plugin_dir}/ (no .so files found)")

        self.env["ROCM_PATH"] = binding_prefix
        self.prepend_ld_library_path(self.env, f"{binding_prefix}/lib")
        self.write_activation_local(binding_prefix, f"{binding_prefix}/lib")

        # 6. Install hipDNN Python bindings.
        if not provider_toolchain_prefix:
            provider_toolchain_prefix = self.select_provider_toolchain_prefix()
        self.maybe_install_amdsmi(
            binding_prefix,
            provider_toolchain_prefix,
            self.rocm_prefix or "",
            DEFAULT_ROCM_PREFIX,
        )

        self.build_and_install_bindings_linux(binding_prefix, provider_toolchain_prefix)

    def build_and_install_bindings_linux(
        self, binding_prefix: str, toolchain_prefix: str
    ) -> None:
        # hipDNN's Python bindings are a standalone CMake project
        # (python/frontend_bindings) plus a wheel packer
        # (python/frontend_wheel_package/pack_frontend_wheel.py): configure +
        # build the nanobind extension against the installed hipDNN, pack a
        # wheel, then install it (setup.sh:1003-1037). This replaces the
        # former `pip install -e HIPDNN_ROOT/python`, which the current
        # rocm-libraries develop no longer supports (python/ has no
        # pyproject.toml).
        python_dir = HIPDNN_ROOT / "python"
        bindings_src = python_dir / "frontend_bindings"
        bindings_build_dir = python_dir / "build" / "frontend_bindings"
        wheel_package_dir = python_dir / "frontend_wheel_package"
        wheel_dir = python_dir / "build" / "wheel_package"
        packer = wheel_package_dir / "pack_frontend_wheel.py"

        cmake_prefix_path = binding_prefix
        if toolchain_prefix != binding_prefix:
            cmake_prefix_path = f"{binding_prefix};{toolchain_prefix}"

        self.pip("install", "build")
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

    # -- Windows build backend (setup.ps1:270-447) --------------------------
    # unverified — confirm on a Windows ROCm host. Ported 1:1 from setup.ps1;
    # no Windows/ROCm host was available to exercise this path (same status
    # PR #2 shipped setup.ps1 under). CI does NOT run it (no GPU/ROCm).

    @staticmethod
    def _fwd(path: str) -> str:
        """Forward-slash a Windows path (backslashes would be read as escapes
        inside the .bat command lines). Mirrors setup.ps1 Fwd."""
        return path.replace("\\", "/")

    def _windows_toolchain(self):
        """Locate cmake/ninja/vcvars64/WinSDK (setup.ps1:289-311)."""
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
        winsdk_version = sorted(versions)[-1]
        return cmake_exe, ninja_exe, vcvars, winsdk_root, winsdk_version

    @staticmethod
    def _cmake_stages(cmake_exe: str, source: str, build: str, configure_args):
        """One CMake project -> configure/build/install command lines for the
        toolchain .bat (setup.ps1 New-CMakeStages)."""
        cmake = f'"{cmake_exe}"'
        src = '"{0}"'.format(Setup._fwd(source))
        bld = '"{0}"'.format(Setup._fwd(build))
        configure = f"{cmake} -S {src} -B {bld} {' '.join(configure_args)}"
        build_line = f"{cmake} --build {bld}"
        install_line = f"{cmake} --install {bld}"
        return [configure, build_line, install_line]

    def _invoke_toolchain_build(
        self, title, commands, vcvars, winsdk_root, winsdk_version, best_effort=False
    ) -> bool:
        """Run CMake inside an MSVC + WinSDK env via a throwaway .bat
        (setup.ps1 Invoke-ToolchainBuild)."""
        import tempfile
        import uuid

        lines = [
            "@echo off",
            'set "PATH=C:\\Program Files (x86)\\Microsoft Visual Studio\\Installer;%PATH%"',
            f'call "{vcvars}" >nul || (echo VCVARS FAILED & exit /b 1)',
            f'set "WINSDK={winsdk_root}"',
            f'set "WINSDKVER={winsdk_version}"',
            'set "LIB=%LIB%;%WINSDK%\\Lib\\%WINSDKVER%\\um\\x64;%WINSDK%\\Lib\\%WINSDKVER%\\ucrt\\x64"',
            'set "INCLUDE=%INCLUDE%;%WINSDK%\\Include\\%WINSDKVER%\\um;%WINSDK%\\Include\\%WINSDKVER%\\ucrt;%WINSDK%\\Include\\%WINSDKVER%\\shared"',
            # Clear a leaked ROCM_PATH; pass -DROCM_PATH in the args instead.
            'set "ROCM_PATH="',
        ]
        for c in commands:
            lines.append(c)
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

    def build_and_install_windows(self, gpu_arch: str) -> None:
        hipdnn_root = HIPDNN_ROOT
        build_dir = hipdnn_root / "build"
        bindings_root = hipdnn_root / "python"
        bindings_src = bindings_root / "frontend_bindings"
        bindings_build = build_dir / "python"
        bindings_package = bindings_build / "wheel_package"
        bindings_pkg_dir = bindings_package / "hipdnn_frontend"
        bindings_pack_script = (
            bindings_root / "frontend_wheel_package" / "pack_frontend_wheel.py"
        )
        provider_dir = ROCM_LIBRARIES_DIR / "dnn-providers" / "miopen-provider"
        build_type = "Release"
        install_dir = str(hipdnn_root / "install")

        if self.do_build:
            # devel prefix: --rocm-prefix else _rocm_sdk_devel discovery.
            if self.rocm_prefix:
                wheel = self.rocm_prefix
            else:
                result = self.probe(
                    "import os,_rocm_sdk_devel as d; print(os.path.dirname(d.__file__))"
                )
                if result.returncode != 0 or not result.stdout.strip():
                    fail(
                        "Building from source needs the ROCm devel wheel "
                        f"(_rocm_sdk_devel) in {self.py}'s env, or pass "
                        "--rocm-prefix."
                    )
                wheel = result.stdout.strip()

            cmake_exe, ninja_exe, vcvars, winsdk_root, winsdk_version = (
                self._windows_toolchain()
            )
            print(f"Toolchain: cmake={cmake_exe} ninja={ninja_exe}")
            print(f"           vcvars={vcvars}  winsdk={winsdk_version}")
            print(f"           rocm={wheel}  gpu={gpu_arch}  install={install_dir}")

            resolved_provider_dir = provider_dir if provider_dir.is_dir() else None
            provider_build = (
                str(resolved_provider_dir / "build") if resolved_provider_dir else None
            )

            wheel_fwd = self._fwd(wheel)
            ninja_fwd = self._fwd(ninja_exe)
            python_fwd = self._fwd(self.py)
            install_fwd = self._fwd(install_dir)

            # Hand the GPU arch to the HIP device-code builds explicitly.
            self.env.setdefault("PYTORCH_ROCM_ARCH", gpu_arch)

            hipdnn_args = [
                "-GNinja",
                f"-DCMAKE_BUILD_TYPE={build_type}",
                f'-DCMAKE_CXX_COMPILER="{wheel_fwd}/lib/llvm/bin/clang++.exe"',
                f'-DCMAKE_MAKE_PROGRAM="{ninja_fwd}"',
                f'-DCMAKE_PREFIX_PATH="{wheel_fwd}"',
                f'-DROCM_CMAKE_PATH="{wheel_fwd}"',
                f'-DROCM_PATH="{wheel_fwd}"',
                f'-DPython_EXECUTABLE="{python_fwd}"',
                f"-DGPU_TARGETS={gpu_arch}",
                f"-DAMDGPU_TARGETS={gpu_arch}",
                "-DENABLE_CLANG_FORMAT=OFF",
                "-DHIPDNN_SKIP_TESTS=ON",
                f'-DCMAKE_INSTALL_PREFIX="{install_fwd}"',
            ]
            hipdnn_stages = self._cmake_stages(
                cmake_exe, str(hipdnn_root), str(build_dir), hipdnn_args
            )
            self._invoke_toolchain_build(
                "Building + installing hipDNN",
                hipdnn_stages,
                vcvars,
                winsdk_root,
                winsdk_version,
            )

            # Python bindings: standalone CMake project against the installed
            # hipDNN artifacts (setup.ps1's Python-bindings restructure).
            bindings_prefix_path = f"{install_fwd};{wheel_fwd}"
            bindings_args = [
                "-GNinja",
                f"-DCMAKE_BUILD_TYPE={build_type}",
                f'-DCMAKE_CXX_COMPILER="{wheel_fwd}/lib/llvm/bin/clang++.exe"',
                f'-DCMAKE_MAKE_PROGRAM="{ninja_fwd}"',
                f'-DCMAKE_PREFIX_PATH="{bindings_prefix_path}"',
                f'-DPython_EXECUTABLE="{python_fwd}"',
            ]
            bindings_stages = self._cmake_stages(
                cmake_exe, str(bindings_src), str(bindings_build), bindings_args
            )
            # The packer stages the built extension into a wheel via `python -m
            # build`; ensure that build tool is present in the venv (matches
            # the Linux bindings method).
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

            plugin_engines_dir = None
            if resolved_provider_dir:
                prov_args = [
                    "-GNinja",
                    f"-DCMAKE_BUILD_TYPE={build_type}",
                    f'-DCMAKE_MAKE_PROGRAM="{ninja_fwd}"',
                    f'-DCMAKE_PREFIX_PATH="{install_fwd};{wheel_fwd}"',
                    f'-DROCM_CMAKE_PATH="{wheel_fwd}"',
                    f'-DROCM_PATH="{wheel_fwd}"',
                    f"-DGPU_TARGETS={gpu_arch}",
                    f"-DAMDGPU_TARGETS={gpu_arch}",
                    "-DMIOPENPROVIDER_SKIP_TESTS=ON",
                    "-DENABLE_CLANG_FORMAT=OFF",
                    "-DENABLE_CLANG_TIDY=OFF",
                    f'-DCMAKE_INSTALL_PREFIX="{install_fwd}"',
                ]
                prov_stages = self._cmake_stages(
                    cmake_exe, str(resolved_provider_dir), provider_build, prov_args
                )
                self._invoke_toolchain_build(
                    "Building + installing MIOpen provider",
                    prov_stages,
                    vcvars,
                    winsdk_root,
                    winsdk_version,
                    best_effort=True,
                )
                # Success is the plugin .dll landing in engines/, not exit status.
                for cand in (
                    Path(install_dir) / "bin/hipdnn_plugins/engines",
                    Path(install_dir) / "lib/hipdnn_plugins/engines",
                ):
                    if cand.is_dir() and any(cand.glob("*.dll")):
                        plugin_engines_dir = cand
                        break
                if plugin_engines_dir:
                    print(f"==> MIOpen plugin installed to {plugin_engines_dir}")
                else:
                    print(
                        "WARNING: MIOpen provider produced no plugin under "
                        f"{install_dir}\\{{bin,lib}}\\hipdnn_plugins\\engines "
                        "(see build output above).",
                        file=sys.stderr,
                    )
            else:
                print(
                    f"WARNING: MIOpen provider not found at {provider_dir}; skipping.",
                    file=sys.stderr,
                )

        # 4. Wire the compiled bindings onto the env via a .pth. Bindings build
        # out-of-tree as a standalone CMake project; the staged wheel package
        # root holds the configured hipdnn_frontend/__init__.py and the
        # compiled extension. Probe legacy and multi-config extension
        # locations too, for already-built trees.
        pyd_dir = None
        for cand in (
            bindings_pkg_dir,
            bindings_build / "lib",
            build_dir / "release" / "lib",
            build_dir / "debug" / "lib",
        ):
            if cand.is_dir() and any(cand.glob("hipdnn_frontend_python*.pyd")):
                pyd_dir = cand
                break

        already_importable = self.probe("import hipdnn_frontend").returncode == 0

        if already_importable:
            print("==> hipdnn_frontend already importable; leaving it as-is.")
        elif pyd_dir:
            # hipdnn_backend.dll is installed to install_dir/bin; register it
            # via os.add_dll_directory from the .pth (handle stashed on sys so
            # it isn't GC'd before import).
            lines = [str(bindings_package)]
            if pyd_dir != bindings_pkg_dir:
                lines.append(str(pyd_dir))
            backend_bin = Path(install_dir) / "bin"
            if (backend_bin / "hipdnn_backend.dll").exists():
                lines.append(
                    "import os, sys; _p = r'{0}'; "
                    "sys.__dict__.setdefault('_hipdnn_dll_dirs', [])"
                    ".append(os.add_dll_directory(_p))".format(backend_bin)
                )
            site_pkgs = self.probe(
                "import sysconfig; print(sysconfig.get_path('purelib'))"
            ).stdout.strip()
            pth = Path(site_pkgs) / "hipdnn_frontend.pth"
            print(
                f"==> Wiring hipdnn_frontend onto the env via {pth} "
                f"(package in {bindings_package})"
            )
            pth.write_text("\n".join(lines) + "\n", encoding="ascii")
        else:
            print(
                "WARNING: hipdnn_frontend is not importable and no compiled "
                f"extension was found under {bindings_pkg_dir} or legacy build "
                "lib directories. Re-run without --reuse-artifacts to build "
                f"from source, or build the standalone CMake project under "
                f"{bindings_src}.",
                file=sys.stderr,
            )

        # Best-effort amdsmi (setup.ps1:502-514).
        if self.probe("import amdsmi").returncode != 0:
            hip_root = self.env.get("HIP_PATH") or self.env.get("ROCM_PATH")
            amdsmi_dir = Path(hip_root) / "share/amd_smi" if hip_root else None
            if amdsmi_dir and amdsmi_dir.is_dir():
                print(f"==> Installing amdsmi Python bindings from {amdsmi_dir}")
                try:
                    self.pip("install", str(amdsmi_dir))
                except subprocess.CalledProcessError:
                    print(
                        "WARNING: amdsmi install failed; GPU SMI snapshot disabled.",
                        file=sys.stderr,
                    )
            else:
                print(
                    "WARNING: amdsmi not found; GPU SMI snapshot disabled (optional).",
                    file=sys.stderr,
                )

    # -- confirmation prompt (setup.sh:812-821) -----------------------------

    def confirm_build(self) -> None:
        if not (self.do_build and not self.auto_yes):
            return
        confirm = input(
            "This will build and install hipDNN and provider plugins from "
            "source into the selected ROCm prefix. Continue? [Y/n] "
        )
        if confirm.strip() in ("n", "N"):
            print("Aborted.")
            sys.exit(0)

    # -- verify (setup.ps1:517-525) -----------------------------------------

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

    # -- top-level flow (setup.sh:225-1027) ---------------------------------

    def run(self) -> int:
        self.require_python_version()
        self.workspace.mkdir(parents=True, exist_ok=True)

        self.confirm_build()
        self.setup_venv()
        print(f"Torch mode: {self.torch_mode}")

        # 2. Install torch, then editable-install the benchmark package.
        self.install_torch()
        self.pip("install", "-e", str(SCRIPT_DIR))

        # CUDA torch supports only the PyTorch backend: no hipDNN/ROCm setup.
        if self.installed_torch_mode == "cuda" or self.torch_mode == "cuda":
            print("")
            print("CUDA torch selected: skipping hipDNN/provider builds, hipDNN Python")
            print("bindings, and ROCm environment setup.")
            print("")
            self._print_cuda_complete()
            self.verify()
            return 0

        # rocm-libraries provides the hipDNN sources + provider plugins.
        self.ensure_rocm_libraries_checkout()

        gpu_arch = self.detect_gpu_arch()
        self.hip_arch_args = []
        if gpu_arch:
            self.hip_arch_args = [
                f"-DGPU_TARGETS={gpu_arch}",
                f"-DAMDGPU_TARGETS={gpu_arch}",
            ]
            self.env.setdefault("PYTORCH_ROCM_ARCH", gpu_arch)

        if IS_WINDOWS:
            self.build_and_install_windows(gpu_arch)
        else:
            # 3. Select the hipDNN/ROCm prefix for bindings + provider builds.
            binding_prefix = self.select_binding_prefix()
            print(f"Using hipDNN/ROCm prefix: {binding_prefix}")
            self.build_and_install_linux(binding_prefix)

        self.verify()
        self._print_complete()
        return 0

    def _print_cuda_complete(self) -> None:
        print("Setup complete. Activate the virtual environment with:")
        if IS_WINDOWS:
            print(f"  {self.venv_dir}\\Scripts\\Activate.ps1")
        else:
            print(f"  source {self.venv_dir}/bin/activate")
        print("")
        print("Run PyTorch-backend benchmarks with:")
        print("  python -m dnn_benchmarking --graph <graph.json> --backend pytorch")

    def _print_complete(self) -> None:
        print("")
        print("Setup complete. Activate the virtual environment with:")
        if IS_WINDOWS:
            print(f"  {self.venv_dir}\\Scripts\\Activate.ps1")
            print("")
            print("Run benchmarks with:")
            print("  python -m dnn_benchmarking --graph <graph.json>")
            return
        print(f"  source {self.venv_dir}/bin/activate")
        print("")
        print("Run benchmarks with:")
        print("  python -m dnn_benchmarking --graph <graph.json>")
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
    parser = build_parser()
    args = parser.parse_args(argv)
    return Setup(args).run()


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Cross-platform setup for the dnn-benchmark tool.

Invoke directly with the *system* interpreter, before any venv exists::

    python3 setup_env.py [options]        # Linux
    py -3 setup_env.py [options]          # Windows

Creates (or, on rerun, reuses) a venv under the workspace, installs torch per
``--torch-mode``, editable-installs the benchmark package, and (for ROCm/CPU/none
source builds) builds hipDNN + the provider plugins incrementally and wires up
the hipDNN Python bindings. ``--clean`` starts over from an empty venv and build
directories.

Stdlib only: this file must import and run under the system interpreter before
the package it installs exists. Never import torch/numpy/third-party at module
top level; any such probe runs in a subprocess against the *venv* interpreter.
"""

import argparse
import contextlib
import dataclasses
import datetime
import functools
import json
import os
import platform
import re
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, NoReturn, Optional, Tuple


IS_WINDOWS = platform.system() == "Windows"

SCRIPT_DIR = Path(__file__).resolve().parent
ROCM_PIN_FILE = SCRIPT_DIR / "rocm-pin.json"
ROCM_LIBRARIES_DIR = SCRIPT_DIR / "rocm-libraries"
ROCM_LIBRARIES_REPO_URL = "https://github.com/ROCm/rocm-libraries.git"
# Root directories that setup and the hipDNN/provider configure read. Only
# shared/ctest supplies a file from shared; the rest of that tree is unnecessary.
ROCM_LIBRARIES_ROOT_DIRS = (".dvc", "cmake", "shared/ctest")
ROCM_LIBRARIES_SPARSE_DIRS = (
    *ROCM_LIBRARIES_ROOT_DIRS,
    "projects/hipdnn",
    "dnn-providers",
)
HIPDNN_ROOT = ROCM_LIBRARIES_DIR / "projects" / "hipdnn"
PROVIDER_DESCRIPTOR_ROOT = (
    ROCM_LIBRARIES_DIR
    / "dnn-providers"
    / "hip-kernel-provider"
    / "src"
    / "engines"
    / "kernel_ingestor_engine"
    / "descriptors"
)

AITER_DIR = SCRIPT_DIR / "aiter"
# AITER submodules that its JIT compiles against: the CK headers and the CK
# code generator of its CK forward kernels.
AITER_SUBMODULES = ("3rdparty/composable_kernel",)


def aiter_build_requires(aiter_dir: Path) -> list[str]:
    """AITER's build requirements (``[build-system] requires``).

    AITER is installed without build isolation: in an isolated build its
    setup.py sees neither torch nor torch's Triton, ignores
    ``AITER_USE_SYSTEM_TRITON`` and runs its install_triton.sh, which
    uninstalls and reinstalls Triton with the first ``python3`` on ``PATH``.
    So these requirements go into the venv first.
    """
    import tomllib

    with open(aiter_dir / "pyproject.toml", "rb") as f:
        return list(tomllib.load(f)["build-system"]["requires"])


DEFAULT_ROCM_PREFIX = "/opt/rocm"


ROCM_NIGHTLY_BASE = "https://nightly.repo.amd.com/rocm"

# Single multi-arch nightly index for every OS: GPU selection is a pip extra
# (torch[device-<arch>]) rather than a per-arch/per-OS URL bucket, so Linux
# and Windows always resolve the exact same torch/rocm-sdk-* build.
ROCM_TORCH_INDEX_URL = f"{ROCM_NIGHTLY_BASE}/whl-next/"

# Quartz (https://github.com/ROCm/Quartz) publishes one status.json for each
# TheRock nightly. Setup reads it to pick the ROCm PyTorch nightly. The
# consumer guide is docs/status-json in that repository.
QUARTZ_NIGHTLY_URL = "https://raw.githubusercontent.com/ROCm/quartz/main/nightly/"
# The status.json schema major that setup reads. A newer minor only adds
# optional fields; another major can move the fields that setup reads.
QUARTZ_SCHEMA_MAJOR = "2"
# Setup searches the newest nightly and the nightlies of this many days in
# total, then stops with an error.
QUARTZ_LOOKBACK_DAYS = 14
QUARTZ_TIMEOUT_S = 30
# The Quartz test suites (pipelines.<suite>.test) whose result for the GPU
# architecture setup checks.
QUARTZ_TEST_SUITES = ("rocm", "pytorch")
# The ROCm wheel version of a nightly, for example 10.3.0a20261117. The group
# is the build date.
_NIGHTLY_VERSION = re.compile(r"\d+\.\d+\.\d+a(\d{8})")
# TheRock repository whose rocm-libraries submodule gitlink supplies the
# rocm-libraries commit that a Quartz-selected ROCm nightly was built from.
THEROCK_REPO_URL = "https://github.com/ROCm/TheRock.git"
THEROCK_ROCM_LIBRARIES_PATH = "rocm-libraries"

# Windows has no rocm_agent_enumerator/rocminfo to detect the local arch with,
# so fall back to a default target known to have published wheels.
WINDOWS_DEFAULT_GPU_ARCH = "gfx1151"

# A whole gfx target name as the ROCm tools print it (gfx90a, gfx942,
# gfx1151), not the prefix of a generic ISA such as gfx9-4-generic.
_GFX_ARCH = re.compile(r"\bgfx[0-9a-f]+(?![\w-])")

# Bound for the arch-detection tools, which hang on a wedged driver.
DETECT_TIMEOUT_S = 30

# Written into the venv after setup installs torch: the GPU arch and index URL
# it used, so a reused venv can refuse a different explicit --gpu-arch or
# --torch-index-url instead of silently mixing builds.
TORCH_RECORD = "dnn-bench-torch.json"

# Written into each reused CMake build dir: the full configure argument list it
# was configured with (prefixes, defaults, detected arch, --cmake-arg). The
# build dirs live in the checkout, shared by every --workspace. A CMakeCache
# keeps its compiler and <Pkg>_DIR paths when the prefixes change, and keeps a
# -D define after it drops off the configure line, so a dir configured
# differently is wiped.
BUILD_RECORD = "dnn-bench-prefixes.json"

# Written into the venv after setup installs ROCm PyTorch: the ROCm wheel
# version, the TheRock build date and commit from Quartz, the Quartz test
# suites with no result for the GPU architecture, and what selected
# the build. dnn_benchmarking.metrics.machine_info reads it into the result
# environment, and a reused venv checks --rocm-version against it.
ROCM_BUILD_RECORD = "dnn-bench-rocm-build.json"


# --- Small process helpers -------------------------------------------------


def warn(*lines: str) -> None:
    """Print a warning (possibly multi-line) to stderr."""
    print(f"WARNING: {lines[0]}", *lines[1:], sep="\n", file=sys.stderr)


def fail(*lines: str) -> NoReturn:
    """Print an error (possibly multi-line) to stderr and exit 1."""
    print(f"ERROR: {lines[0]}", *lines[1:], sep="\n", file=sys.stderr)
    sys.exit(1)


def rmtree(path: Path) -> None:
    """shutil.rmtree that also removes read-only files (git objects on Windows)."""

    def retry_writable(func, failed_path, _exc):
        os.chmod(failed_path, stat.S_IWRITE)
        func(failed_path)

    shutil.rmtree(path, onexc=retry_writable)


def run(cmd, *, env=None, check=True, **kwargs):
    """Run a command; raise CalledProcessError on failure."""
    return subprocess.run(list(cmd), env=env, check=check, **kwargs)


#: Oldest CMake that builds everything setup builds. rocm-libraries needs
#: 3.25.2 (CMakeLists.txt, CMakePresets.json "version": 6); the hipDNN Python
#: bindings need 3.26 (projects/hipdnn/python/frontend_bindings/CMakeLists.txt).
CMAKE_MIN_VERSION = (3, 26)
_CMAKE_VERSION = re.compile(r"cmake version (\d+)\.(\d+)")


def require_working_cmake() -> str:
    """Return the PATH-selected CMake executable after a startup check."""
    cmake = shutil.which("cmake")
    if not cmake:
        fail("cmake not found on PATH.")
    try:
        result = subprocess.run(
            [cmake, "--version"],
            stdout=subprocess.PIPE,
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
    floor = ".".join(map(str, CMAKE_MIN_VERSION))
    found = _CMAKE_VERSION.search(result.stdout or "")
    if found is None:
        warn(f"cannot read the version of cmake at {cmake}; setup needs {floor}+.")
    elif tuple(map(int, found.groups())) < CMAKE_MIN_VERSION:
        # An old cmake fails later with only "Could not read presets ...:
        # Unrecognized "version" field".
        fail(
            f"cmake at {cmake} is version {found.group(1)}.{found.group(2)}; "
            f"setup needs cmake {floor} or later.",
            f"Put a newer cmake first on PATH, for example from "
            f"`python3 -m pip install 'cmake>={floor}'`.",
        )
    return cmake


def run_git(args, **kwargs):
    """Run git with the given args; raise on nonzero exit."""
    return run(["git", *args], **kwargs)


def git_output(args) -> str:
    """Run git and return its stdout, stripped; raise on nonzero exit."""
    result = subprocess.run(["git", *args], capture_output=True, text=True, check=True)
    return result.stdout.strip()


# --- ROCm nightly selection (Quartz) ----------------------------------------


@dataclasses.dataclass(frozen=True)
class RocmBuild:
    """The TheRock build that setup installs ROCm PyTorch from.

    ``rocm_version`` is the ROCm wheel version that pip pins; "" installs the
    newest build on ``index_url``. ``build_date``, ``therock_commit``,
    ``status_url``, ``untested`` and ``rocm_libraries_commit`` come from
    ``rocm-pin.json`` or Quartz, or are None when unavailable. ``untested``
    names the Quartz test suites with no result for the GPU architecture.
    ``source`` names what selected the build.
    """

    rocm_version: str
    index_url: str
    source: str
    build_date: Optional[str] = None
    therock_commit: Optional[str] = None
    status_url: Optional[str] = None
    untested: Optional[Tuple[str, ...]] = None
    rocm_libraries_commit: Optional[str] = None


def quartz_platform() -> str:
    """The ``summary`` key of this host's platform in a Quartz status.json."""
    return "windows" if IS_WINDOWS else "linux"


def fetch_quartz_text(url: str) -> Optional[str]:
    """The body of ``url``, or None when the server answers 404 Not Found.

    A nightly that did not run has no status.json, so 404 is an expected
    answer. Every other failure stops setup with the URL.
    """
    try:
        with urllib.request.urlopen(url, timeout=QUARTZ_TIMEOUT_S) as response:
            return response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        fail(f"cannot read {url}: {exc}")
    except (urllib.error.URLError, TimeoutError) as exc:
        fail(f"cannot read {url}: {exc}")


def _parse_quartz_status(url: str, body: str) -> dict:
    """The status.json document in ``body``; exit when setup cannot read it."""
    try:
        document = json.loads(body)
    except ValueError as exc:
        fail(f"{url} is not valid JSON: {exc}")
    major = str(document.get("schema_version", "")).split(".")[0]
    if major != QUARTZ_SCHEMA_MAJOR:
        fail(
            f"{url} has status.json schema {document.get('schema_version')!r}; "
            f"setup reads schema {QUARTZ_SCHEMA_MAJOR}.x only.",
            "Update the Quartz reader in setup_env.py, or pass --rocm-version.",
        )
    return document


def _quartz_status_url(build_date: str) -> str:
    return f"{QUARTZ_NIGHTLY_URL}{build_date}/status.json"


def quartz_arch_matches(gpu_arch: str, target: str) -> bool:
    """True when the TheRock build target ``target`` covers ``gpu_arch``.

    Quartz lists the targets that TheRock built: one GPU (``gfx90a``,
    ``gfx950-dcgpu``) or a family whose last character is ``X``
    (``gfx94X-dcgpu`` covers gfx942, ``gfx110X-all`` covers gfx1100 to
    gfx1103). The part after ``-`` names a product line, not a GPU.
    """
    family = target.split("-", 1)[0].lower()
    arch = gpu_arch.lower()
    if family.endswith("x"):
        return len(arch) == len(family) and arch.startswith(family[:-1])
    return arch == family


def quartz_target(targets, gpu_arch: str) -> Optional[str]:
    """The TheRock build target among ``targets`` that covers ``gpu_arch``.

    A target for the one GPU wins over a family target. None when no
    target covers ``gpu_arch``.
    """
    matches = [target for target in targets if quartz_arch_matches(gpu_arch, target)]
    exact = [t for t in matches if not t.split("-", 1)[0].lower().endswith("x")]
    return (exact or matches or [None])[0]


def quartz_arch_tests(
    document: dict, platform_name: str, gpu_arch: str
) -> Tuple[Dict[str, Tuple[str, str]], Tuple[str, ...]]:
    """The Quartz test results of ``gpu_arch`` on ``platform_name``.

    Returns ``{suite: (target, status)}`` for each suite in QUARTZ_TEST_SUITES
    that has a test entry for a target that covers ``gpu_arch``, and the
    suites without one. Setup reads the status of the target only. Quartz
    rolls the PyTorch variants up into it: ``failure`` when a variant failed,
    else ``in_progress`` while a variant runs.
    """
    pipelines = document.get("pipelines", {})
    results = {}
    untested = []
    for suite in QUARTZ_TEST_SUITES:
        entries = pipelines.get(suite, {}).get("test", {}).get(platform_name, {})
        target = quartz_target(entries, gpu_arch)
        if target is None:
            untested.append(suite)
        else:
            results[suite] = (target, str(entries[target].get("status") or "absent"))
    return results, tuple(untested)


def quartz_rejection(
    document: dict,
    platform_name: str,
    gpu_arch: str,
    *,
    check_tests: bool = True,
) -> str:
    """Why the build cannot supply ROCm PyTorch for ``gpu_arch``; "" if it can.

    Setup needs the ROCm and the PyTorch build of ``platform_name`` to be
    ``success``, and ``gpu_arch`` among the built architectures. When
    ``check_tests`` is True, the ROCm and PyTorch tests of ``gpu_arch`` must
    also be ``success`` when Quartz has them; a suite with no entry for
    ``gpu_arch`` does not reject the build. Setup does not read the test
    counters or ``overall_status``: they fold in every architecture, and some
    are routinely red. The result names every check that fails, separated by
    "; ".
    """
    if document.get("release_type") != "nightly":
        return f"release_type is {document.get('release_type')!r}, not 'nightly'"
    if document.get("build_variant", "release") != "release":
        return f"build_variant is {document.get('build_variant')!r}"
    if not _NIGHTLY_VERSION.fullmatch(str(document.get("rocm_version", ""))):
        return f"rocm_version {document.get('rocm_version')!r} is not a nightly"
    summary = document.get("summary", {}).get(platform_name)
    if not summary:
        return f"no {platform_name} build"
    reasons = []
    if quartz_target(summary.get("architectures", []), gpu_arch) is None:
        reasons.append(f"{gpu_arch} is not among its {platform_name} architectures")
    for pipeline in ("rocm", "pytorch"):
        status = summary.get(pipeline, {}).get("build", {}).get("status")
        if status != "success":
            reasons.append(f"{platform_name} {pipeline} build is {status or 'absent'}")
    if not summary.get("urls", {}).get("wheels"):
        reasons.append(f"no {platform_name} wheels URL")
    if check_tests:
        results, _ = quartz_arch_tests(document, platform_name, gpu_arch)
        for suite, (target, status) in results.items():
            if status != "success":
                reasons.append(
                    f"{platform_name} {suite} tests of {target} are {status}"
                )
    return "; ".join(reasons)


def _rocm_build_from_quartz(
    document: dict,
    url: str,
    platform_name: str,
    gpu_arch: str,
    source: str,
    index_url: str = "",
) -> RocmBuild:
    """The build that ``document`` describes. An explicit ``index_url`` wins
    over the wheels URL of the document."""
    urls = document.get("summary", {}).get(platform_name, {}).get("urls", {})
    _, untested = quartz_arch_tests(document, platform_name, gpu_arch)
    return RocmBuild(
        rocm_version=document["rocm_version"],
        index_url=index_url or urls.get("wheels") or ROCM_TORCH_INDEX_URL,
        source=source,
        build_date=document.get("build_date") or None,
        therock_commit=document.get("therock_commit") or None,
        status_url=url,
        untested=untested,
    )


def _newest_quartz_nightly() -> datetime.date:
    """The build date of the nightly that ``nightly/latest.json`` names.

    The pointer is a git symlink, so raw GitHub serves its target path (for
    example ``20261118/status.json``) instead of the document.
    """
    url = f"{QUARTZ_NIGHTLY_URL}latest.json"
    body = fetch_quartz_text(url)
    if body is None:
        fail(f"Quartz publishes no {url}.", "Pass --rocm-version to install a build.")
    if body.lstrip().startswith("{"):
        build_date = str(_parse_quartz_status(url, body).get("build_date", ""))
    else:
        build_date = body.strip().split("/", 1)[0]
    try:
        return datetime.datetime.strptime(build_date, "%Y%m%d").date()
    except ValueError:
        fail(f"{url} names no nightly build date: {body.strip()[:80]!r}")


def _find_quartz_nightly(
    platform_name: str,
    gpu_arch: str,
    index_url: str = "",
    *,
    require_platform_builds: Tuple[Tuple[str, str], ...] = (),
) -> Tuple[Optional[RocmBuild], List[str], datetime.date]:
    """Search back QUARTZ_LOOKBACK_DAYS for a usable nightly for ``gpu_arch``.

    ``require_platform_builds`` takes additional ``(platform, gpu_arch)`` pairs
    whose ROCm and PyTorch builds and wheels URL must also be ``success``
    without rejecting on test entries for those extra platforms.
    """
    newest = _newest_quartz_nightly()
    rejected: List[str] = []
    for days_back in range(QUARTZ_LOOKBACK_DAYS):
        build_date = (newest - datetime.timedelta(days=days_back)).strftime("%Y%m%d")
        url = _quartz_status_url(build_date)
        body = fetch_quartz_text(url)
        if body is None:
            rejected.append(f"  {build_date}: no status.json")
            continue
        document = _parse_quartz_status(url, body)
        reasons = []
        reason = quartz_rejection(document, platform_name, gpu_arch)
        if reason:
            reasons.append(reason)
        for extra_platform, extra_arch in require_platform_builds:
            extra_reason = quartz_rejection(
                document, extra_platform, extra_arch, check_tests=False
            )
            if extra_reason and extra_reason not in reasons:
                reasons.append(extra_reason)
        if reasons:
            rejected.append(
                f"  {build_date} ({document.get('rocm_version')}): {'; '.join(reasons)}"
            )
            continue
        return (
            _rocm_build_from_quartz(
                document, url, platform_name, gpu_arch, "quartz", index_url
            ),
            rejected,
            newest,
        )
    return None, rejected, newest


def find_quartz_nightly(
    platform_name: str,
    gpu_arch: str,
    index_url: str = "",
    *,
    require_platform_builds: Tuple[Tuple[str, str], ...] = (),
) -> Optional[RocmBuild]:
    """The newest usable Quartz nightly for ``gpu_arch``, or None if none passed."""
    build, _, _ = _find_quartz_nightly(
        platform_name,
        gpu_arch,
        index_url,
        require_platform_builds=require_platform_builds,
    )
    return build


def select_quartz_nightly(
    platform_name: str, gpu_arch: str, index_url: str = ""
) -> RocmBuild:
    """The newest nightly that Quartz reports usable for ``gpu_arch``."""
    build, rejected, newest = _find_quartz_nightly(platform_name, gpu_arch, index_url)
    if build is not None:
        return build
    fail(
        f"Quartz reports no usable ROCm nightly for {platform_name} {gpu_arch} in "
        f"the {QUARTZ_LOOKBACK_DAYS} days up to {newest:%Y%m%d}:",
        *rejected,
        "Pass --rocm-version VERSION to install a specific build.",
    )


def describe_rocm_version(
    rocm_version: str, index_url: str, platform_name: str, gpu_arch: str
) -> RocmBuild:
    """The --rocm-version build, with its Quartz metadata when Quartz has it.

    An explicit version is installed even when Quartz reports it unusable,
    so an older run can be reproduced; setup only warns. A version that is
    not a nightly, or a nightly from before Quartz, has no build date or
    TheRock commit. ``index_url`` is --torch-index-url, "" when not given.
    """
    match = _NIGHTLY_VERSION.fullmatch(rocm_version)
    url = _quartz_status_url(match[1]) if match else ""
    body = fetch_quartz_text(url) if url else None
    document = _parse_quartz_status(url, body) if body is not None else {}
    if document.get("rocm_version") != rocm_version:
        warn(
            f"Quartz has no nightly status.json for ROCm {rocm_version}.",
            "The results record no TheRock build date, commit or test coverage "
            "for it.",
        )
        return RocmBuild(
            rocm_version, index_url or ROCM_TORCH_INDEX_URL, "--rocm-version"
        )
    reason = quartz_rejection(document, platform_name, gpu_arch)
    if reason:
        warn(
            f"Quartz reports that ROCm {rocm_version} is not usable: {reason}.",
            "Installing it because --rocm-version names it.",
        )
    return _rocm_build_from_quartz(
        document, url, platform_name, gpu_arch, "--rocm-version", index_url
    )


def _parse_rocm_pin_entry(entry: object, label: str, source_name: str) -> RocmBuild:
    if not isinstance(entry, dict):
        fail(f"{label} must be a JSON object.")
    rocm_version = str(entry.get("rocm_version") or "").strip()
    if not re.fullmatch(r"[0-9][0-9A-Za-z.+!-]*", rocm_version):
        fail(
            f"{label} has invalid rocm_version {entry.get('rocm_version')!r}.",
            "Expected a ROCm wheel version such as 10.3.0a20261123.",
        )
    rocm_libraries_commit = (
        str(entry.get("rocm_libraries_commit") or "").strip().lower()
    )
    if not re.fullmatch(r"[0-9a-f]{40}", rocm_libraries_commit):
        fail(
            f"{label} has invalid rocm_libraries_commit "
            f"{entry.get('rocm_libraries_commit')!r}.",
            "Expected a 40-character commit SHA.",
        )
    therock_raw = entry.get("therock_commit")
    therock_commit: Optional[str] = None
    if therock_raw:
        therock_commit = str(therock_raw).strip().lower()
        if not re.fullmatch(r"[0-9a-f]{40}", therock_commit):
            fail(
                f"{label} has invalid therock_commit {therock_raw!r}.",
                "Expected a 40-character commit SHA.",
            )
    build_date = str(entry.get("build_date") or "").strip() or None
    index_url = str(entry.get("index_url") or "").strip() or ROCM_TORCH_INDEX_URL
    status_url = _quartz_status_url(build_date) if build_date else None
    return RocmBuild(
        rocm_version=rocm_version,
        index_url=index_url,
        source=source_name,
        build_date=build_date,
        therock_commit=therock_commit,
        status_url=status_url,
        untested=None,
        rocm_libraries_commit=rocm_libraries_commit,
    )


def read_rocm_pins(
    path: Optional[Path] = None,
) -> Tuple[str, Dict[str, RocmBuild]]:
    """Read all per-arch pins and the default arch key from ``rocm-pin.json``."""
    pin_path = ROCM_PIN_FILE if path is None else path
    try:
        data = json.loads(pin_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        fail(
            f"cannot read ROCm pin file {pin_path}: {exc}.",
            "Restore rocm-pin.json, or pass --rocm-version and --rocm-libraries-ref.",
        )
    if not isinstance(data, dict):
        fail(f"{pin_path} must contain a JSON object.")
    archs_raw = data.get("archs")
    if not isinstance(archs_raw, dict) or not archs_raw:
        fail(f"{pin_path} must contain a non-empty 'archs' object.")
    pins: Dict[str, RocmBuild] = {}
    for arch_key, entry in archs_raw.items():
        if not isinstance(arch_key, str) or not arch_key.strip():
            fail(f"{pin_path} has an empty architecture key in 'archs'.")
        norm_arch = arch_key.strip().lower()
        pins[norm_arch] = _parse_rocm_pin_entry(
            entry, f"{pin_path}['archs'][{norm_arch!r}]", pin_path.name
        )
    default_arch = str(data.get("default") or "").strip().lower()
    if default_arch not in pins:
        fail(
            f"{pin_path} has invalid default arch {data.get('default')!r}.",
            f"Expected one of: {', '.join(pins)}.",
        )
    return default_arch, pins


def read_rocm_pin(
    gpu_arch: Optional[str] = None, path: Optional[Path] = None
) -> RocmBuild:
    """Read the checked-in ROCm + rocm-libraries pin for ``gpu_arch``.

    Falls back to the ``default`` arch in ``rocm-pin.json`` when ``gpu_arch``
    is None or has no explicit entry.
    """
    default_arch, pins = read_rocm_pins(path)
    if gpu_arch:
        norm = gpu_arch.strip().lower()
        if norm in pins:
            return pins[norm]
    return pins[default_arch]


def _fetch_therock_rocm_libraries_commit(
    therock_commit: str, repo_url: Optional[str] = None
) -> str:
    """Resolve the rocm-libraries gitlink at ``therock_commit`` in TheRock.

    Raises ``RuntimeError`` on invalid input, missing git, fetch failure, or a
    missing submodule entry so callers can choose fatal vs advisory handling.
    """
    url = repo_url or THEROCK_REPO_URL
    commit = therock_commit.strip().lower()
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise RuntimeError(
            f"expected a 40-character TheRock commit SHA, got {therock_commit!r}."
        )
    with tempfile.TemporaryDirectory(prefix="therock-ref-") as tmp:
        git = ["-C", tmp]
        try:
            git_output(["init", "--bare", "--quiet", tmp])
            git_output(
                [
                    *git,
                    "fetch",
                    "--quiet",
                    "--depth",
                    "1",
                    "--filter=tree:0",
                    url,
                    commit,
                ]
            )
            entry = git_output(
                [*git, "ls-tree", "FETCH_HEAD", "--", THEROCK_ROCM_LIBRARIES_PATH]
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"git not found on PATH; cannot resolve {THEROCK_ROCM_LIBRARIES_PATH} "
                f"commit for TheRock {commit}."
            ) from exc
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or exc.stdout or "").strip()
            msg = (
                f"cannot resolve {THEROCK_ROCM_LIBRARIES_PATH} commit for "
                f"TheRock {commit} from {url}."
            )
            if detail:
                msg = f"{msg}\n{detail}"
            raise RuntimeError(msg) from exc
    parts = entry.split()
    if (
        len(parts) != 4
        or parts[0] != "160000"
        or parts[1] != "commit"
        or not re.fullmatch(r"[0-9a-f]{40}", parts[2])
        or parts[3] != THEROCK_ROCM_LIBRARIES_PATH
    ):
        raise RuntimeError(
            f"TheRock commit {commit} has no '{THEROCK_ROCM_LIBRARIES_PATH}' "
            f"submodule entry in {url} (got {entry!r})."
        )
    return parts[2]


def resolve_therock_rocm_libraries_commit(
    therock_commit: str,
    repo_url: Optional[str] = None,
    *,
    required: bool = True,
) -> Optional[str]:
    """The rocm-libraries submodule commit pinned by ``therock_commit`` in TheRock.

    Uses a bare, tree-filtered git fetch of the single commit (~10 KB, no
    GitHub REST API rate limit) and reads the ``rocm-libraries`` gitlink via
    ``git ls-tree``. When ``required`` is True, exits with an actionable error
    on failure; when False (advisory check for an existing checkout), warns and
    returns None.
    """
    try:
        return _fetch_therock_rocm_libraries_commit(therock_commit, repo_url)
    except RuntimeError as exc:
        lines = str(exc).splitlines()
        if required:
            fail(
                *lines,
                "Pass --rocm-libraries-ref <sha> to select rocm-libraries explicitly.",
            )
        warn(*lines, "Keeping the existing rocm-libraries/ checkout.")
        return None


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


def unify_rocprofiler_libs(core_lib: Path, devel_lib: Path) -> int:
    """Point the devel prefix's rocprofiler libraries at the core copies.

    The ROCm wheels hardlink one set of rocprofiler libraries into both
    ``_rocm_sdk_core/lib`` (what torch loads) and ``_rocm_sdk_devel/lib``
    (the RUNPATH the hipDNN plugins are built against).
    rocprofiler-register identifies the registered library by path
    string, so a profiled run registers the core path for torch and then
    aborts when the plugins bring in the devel path — even though both
    names are the same file::

        ROCPROFILER_REGISTER_LIBRARY is already set to
        '.../_rocm_sdk_core/lib/librocprofiler-sdk.so.1', not overriding
        with '.../_rocm_sdk_devel/lib/librocprofiler-sdk.so'

    Replacing the devel duplicates with symlinks leaves exactly one path
    in play, which is what makes rocprofv3 trace capture work at all.
    Only exact same-file duplicates (same device + inode) are relinked,
    so a genuinely different build is never swapped underneath the user.

    Args:
        core_lib: ``_rocm_sdk_core/lib``.
        devel_lib: ``_rocm_sdk_devel/lib``.

    Returns:
        Number of duplicates relinked.
    """
    if not core_lib.is_dir() or not devel_lib.is_dir():
        return 0

    core_by_id = {}
    for path in core_lib.glob("librocprofiler-*"):
        if path.is_symlink() or not path.is_file():
            continue
        info = path.stat()
        core_by_id[(info.st_dev, info.st_ino)] = path

    relinked = 0
    for dup in sorted(devel_lib.glob("librocprofiler-*")):
        if dup.is_symlink() or not dup.is_file():
            continue
        info = dup.stat()
        target = core_by_id.get((info.st_dev, info.st_ino))
        if target is None:
            continue
        # Build the replacement first and swap it in atomically: unlinking
        # the hardlink up front would destroy the only remaining name for
        # the library if symlink creation then failed (Windows without
        # developer mode, restricted permissions), leaving the devel
        # prefix short a library it can never get back.
        staged = dup.with_name(dup.name + ".unify-tmp")
        try:
            if staged.is_symlink() or staged.exists():
                staged.unlink()
            staged.symlink_to(target)
            os.replace(staged, dup)
        except OSError as e:
            if staged.is_symlink() or staged.exists():
                staged.unlink()
            warn(f"could not relink {dup} -> {target}: {e}")
            return relinked
        relinked += 1
    return relinked


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
            # Multi-arch indexes ship one arch-agnostic "_rocm_sdk_libraries"
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
            # Not gated on how amdsmi is packaged: since about rocm 10.1 the
            # core wheel ships share/amd_smi/amdsmi without setup.py or
            # pyproject.toml, and comgr and rocprofiler also use this prefix.
            if not child.name.startswith("_rocm_sdk_core"):
                continue
            if not child.joinpath("lib").is_dir():
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
    print("Rerun setup with --clean so it cannot mix ROCm SDK packages.", file=sys.stderr)
    sys.exit(2)
sys.exit(1)
"""


# Reports what each opt-in profiling flag can do in this venv. Runs in the
# venv interpreter so it sees the same resolution the benchmark will.
_PROFILING_SOURCES = r"""
from dnn_benchmarking.metrics._tool_resolver import resolve_rocm_tool
from dnn_benchmarking.metrics.perf import (
    _kernel_events_allowed,
    _read_perf_paranoid,
    _resolve_perf,
)


def state(ok, detail):
    return ("  [ok]   " if ok else "  [skip] ") + detail


rocprofv3 = resolve_rocm_tool("rocprofv3")
print(state(rocprofv3 is not None, f"--trace / --pmc: rocprofv3 {rocprofv3 or 'not found'}"))


rocprof_compute = resolve_rocm_tool("rocprof-compute")
print(state(rocprof_compute is not None, "--roofline: " + (
    str(rocprof_compute)
    if rocprof_compute
    else "rocprof-compute not found; install the rocprofiler-compute system package"
)))

# Same resolution the benchmark performs: distro `perf` is a wrapper
# that refuses to run when no linux-tools matches the running kernel,
# so resolving the name proves nothing.
resolved_perf = _resolve_perf()
if resolved_perf is None:
    print(state(False, "--perf: no runnable perf; install linux-tools matching $(uname -r)"))
else:
    perf, substitution = resolved_perf
    paranoid = _read_perf_paranoid()
    if not _kernel_events_allowed(paranoid):
        print(state(False, f"--perf: {perf} records user-space counters only (perf_event_paranoid={paranoid}, kernel events need <= 1 or CAP_PERFMON)"))
    else:
        print(state(True, f"--perf: {perf}"))
    if substitution:
        print(f"         {substitution}")
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

# Compile the AITER backends that --runtime aiter runs on this GPU (see
# aiter_executor.prebuild). Exit 1 when any of them fails.
_AITER_PREBUILD = r"""
import sys

from dnn_benchmarking.execution import aiter_executor

reason = aiter_executor.prebuild_skip_reason()
if reason:
    print(f"  Nothing to compile: {reason}.")
    sys.exit(0)
failed = 0
for label, error in aiter_executor.prebuild():
    failed += error is not None
    print(f"  {label}: {error}" if error else f"  {label}", flush=True)
sys.exit(1 if failed else 0)
"""


# --- CLI -------------------------------------------------------------------


def _commit_sha(value: str) -> str:
    """--rocm-libraries-ref: a full 40-hex commit id, lowercased."""
    sha = value.strip().lower()
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise argparse.ArgumentTypeError(
            f"expected a full 40-character commit SHA, got {value!r}"
        )
    return sha


def _rocm_version(value: str) -> str:
    """--rocm-version: 'latest' (Quartz) or a wheel version such as 10.3.0a20261123."""
    version = value.strip()
    if version.lower() == "latest":
        return "latest"
    if not re.fullmatch(r"[0-9][0-9A-Za-z.+!-]*", version):
        raise argparse.ArgumentTypeError(
            f"expected 'latest' or a ROCm wheel version such as 10.3.0a20261123, "
            f"got {value!r}"
        )
    return version


def _cmake_define(value: str) -> str:
    """Normalise a --cmake-arg value to a `-DNAME=VALUE` CMake define.

    Accepts both spellings. The bare `-D` form only survives argparse when
    written `--cmake-arg=-DFOO=ON`; with a space, argparse sees a token starting
    with `-` and reports "expected one argument". Accepting `NAME=VALUE` makes
    the space form work too, which is the form everyone reaches for first.
    """
    text = value.strip()
    if not text:
        raise argparse.ArgumentTypeError("--cmake-arg needs a NAME=VALUE define")
    if text.startswith("-D"):
        text = text[2:]
    if "=" not in text:
        raise argparse.ArgumentTypeError(
            f"--cmake-arg expects NAME=VALUE, got {value!r} (no '=' found)"
        )
    return f"-D{text}"


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
            "  rocm: install the ROCm torch nightly pinned in rocm-pin.json "
            "(or --rocm-version latest / VERSION), use ROCm libraries/toolchain "
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
        help=(
            "Install dnn-benchmarking and the rocKE packages into the venv "
            "instead of linking to the source tree."
        ),
    )
    parser.add_argument(
        "--torch-index-url",
        default="",
        help=(
            "Override the pip index URL used for torch. In rocm mode without "
            "--rocm-version, setup does not read Quartz and installs the "
            "newest ROCm torch on this index."
        ),
    )
    parser.add_argument(
        "--rocm-version",
        type=_rocm_version,
        default=None,
        metavar="VERSION",
        help=(
            "ROCm wheel version for --torch-mode rocm: 'latest' (the newest "
            f"nightly of the last {QUARTZ_LOOKBACK_DAYS} days that Quartz "
            "reports usable for the GPU architecture) or a specific version "
            "such as 10.3.0a20261123 (installed even when Quartz reports it "
            "failed, to reproduce an older run). Default: the version pinned "
            "in rocm-pin.json."
        ),
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
            "Skip building hipDNN, providers and Python bindings; use the "
            "installed hipDNN prefix and hipdnn_frontend package instead. "
            "ROCm, CPU and none modes require a pre-existing venv with the "
            "requested torch mode and no --clean. --torch-mode existing keeps "
            "the installed torch; CUDA-only setups need no hipDNN bindings."
        ),
    )
    parser.add_argument(
        "--no-aiter",
        dest="aiter",
        action="store_false",
        help=(
            "Skip the AITER stage (fetch, install and kernel compile for "
            "--runtime aiter). Other runtimes do not need it."
        ),
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help=(
            "Delete and recreate the virtual environment and wipe the hipDNN/"
            "provider and binding CMake build directories. By default setup "
            "reuses the existing venv (and its torch) and rebuilds incrementally; "
            "changing its --torch-mode, --gpu-arch, --torch-index-url or "
            "--rocm-version then "
            "requires --clean. A build directory configured with other CMake "
            "arguments (another --workspace, --rocm-prefix, GPU arch or "
            "--cmake-arg) is wiped without --clean."
        ),
    )
    parser.add_argument(
        "--rocm-libraries-ref",
        type=_commit_sha,
        metavar="SHA",
        help=(
            "Full 40-character rocm-libraries commit to fetch when "
            "rocm-libraries/ is absent. "
            "Default: in rocm mode with --rocm-version latest or a "
            "Quartz-known --rocm-version, the rocm-libraries commit pinned by "
            "that nightly's TheRock commit; otherwise rocm_libraries_commit "
            "from rocm-pin.json."
        ),
    )
    parser.add_argument(
        "--cmake-arg",
        dest="cmake_args",
        action="append",
        default=[],
        type=_cmake_define,
        metavar="NAME=VALUE",
        help=(
            "Extra CMake define appended to the hipDNN/provider configure. "
            "Repeatable. Appended after the defaults, so it can override one. "
            "Accepts NAME=VALUE or -DNAME=VALUE; the leading -D is added when "
            "absent, because argparse reads a bare '-DFOO=ON' as an option "
            "rather than a value unless it is written '--cmake-arg=-DFOO=ON'. "
            "On Linux, rocKE (HIPKERNELPROVIDER_ENABLE_ROCKE, "
            "HIPDNN_ENABLE_KERNEL_INGESTOR) is ON by default; set both OFF to "
            "skip it."
        ),
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help=(
            "Answer yes to every confirmation prompt (source build, --clean, "
            "replacing a non-git rocm-libraries/ directory). Required when "
            "stdin is not a terminal and a prompt would be shown."
        ),
    )
    return parser


# --- Orchestration ---------------------------------------------------------


class Setup:
    """Holds resolved config and venv state for one setup run."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.torch_mode = args.torch_mode
        self.reuse_artifacts = args.reuse_artifacts
        self.with_aiter = args.aiter
        self.auto_yes = args.yes
        self.clean = args.clean
        self.rocm_prefix = args.rocm_prefix
        self.rocm_libraries_ref = args.rocm_libraries_ref
        self.gpu_arch_override = args.gpu_arch
        self.torch_index_url = args.torch_index_url
        self.rocm_version = args.rocm_version
        self.editable_install = args.editable_install
        self.extra_cmake_args = list(args.cmake_args)
        self.resolved_torch_index_url = ""
        self.installed_torch_mode = "missing"
        self.plugin_engines_dir = None
        self.install_prefix = ""
        self.current_stage = "preflight"
        self.n_stages = 0
        self._stage_index = 0

        self.do_build = not self.reuse_artifacts and self.torch_mode != "cuda"

        # Resolve: this path is written into activate.local, which is sourced
        # from arbitrary working directories.
        self.workspace = Path(args.workspace).resolve()
        self.venv_dir = self.workspace / ".venv"

        # Child-process environment; ROCM_PATH is layered onto this once the
        # hipDNN prefix is known.
        self.env = dict(os.environ)
        self.env["PYTHONPYCACHEPREFIX"] = str(self.workspace / "pycache")
        self.env["DNN_BENCH_WORKSPACE"] = str(self.workspace)

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
                f"setup_env.py requires Python >= 3.12, but the invoking "
                f"interpreter is {version}. Run setup with a Python 3.12+ environment."
            )

    # -- rocm-libraries checkout --------------------------------------------

    def _resolve_rocm_libraries_ref(
        self, *, required: bool = True
    ) -> Tuple[str, str, Optional[str]]:
        """Return ``(ref, source, therock_commit)`` for the rocm-libraries checkout.

        Precedence:
        1. Explicit ``--rocm-libraries-ref``.
        2. In ``rocm`` mode (or ``existing`` on a ROCm venv), the cached
           ``rocm_libraries_commit`` in ``ROCM_BUILD_RECORD`` (already set when
           installed from ``rocm-pin.json``), or resolved from its Quartz
           ``therock_commit`` via ``ROCm/TheRock`` so ``miopen-provider`` and
           hipDNN stay in lockstep with the wheel's MIOpen.
        3. ``rocm_libraries_commit`` from the checked-in ``rocm-pin.json``.
        """
        if self.rocm_libraries_ref:
            return self.rocm_libraries_ref, "--rocm-libraries-ref", None
        if self.installed_torch_mode == "rocm":
            record = self._read_venv_record(ROCM_BUILD_RECORD)
            if record:
                therock_commit = record.get("therock_commit")
                if not isinstance(therock_commit, str) or not therock_commit:
                    therock_commit = None
                recorded_ref = record.get("rocm_libraries_commit")
                if isinstance(recorded_ref, str) and re.fullmatch(
                    r"[0-9a-f]{40}", recorded_ref
                ):
                    source = str(record.get("source") or "quartz")
                    return recorded_ref, source, therock_commit
                if therock_commit:
                    print(
                        "Resolving rocm-libraries commit for TheRock "
                        f"{therock_commit[:12]}..."
                    )
                    ref = resolve_therock_rocm_libraries_commit(
                        therock_commit, required=required
                    )
                    if ref is None:
                        return "", "quartz", therock_commit
                    record["rocm_libraries_commit"] = ref
                    try:
                        (self.venv_dir / ROCM_BUILD_RECORD).write_text(
                            json.dumps(record) + "\n"
                        )
                    except OSError:
                        pass
                    return ref, "quartz", therock_commit
        pin = read_rocm_pin(self.gpu_arch or self.gpu_arch_override)
        return pin.rocm_libraries_commit or "", pin.source, pin.therock_commit

    def ensure_rocm_libraries_checkout(self) -> None:
        """Fetch rocm-libraries if absent.

        Clones a sparse, blobless checkout of the resolved commit
        (--rocm-libraries-ref, else the commit matched to the installed ROCm
        build, else rocm-pin.json), limited to the hipDNN/provider sources and
        root CMake tooling this script builds. To build against a different
        ref, check it out directly, e.g.
        `git -C rocm-libraries fetch --depth 1 origin <ref> &&
         git -C rocm-libraries checkout FETCH_HEAD`.
        """
        has_checkout = (ROCM_LIBRARIES_DIR / ".git").exists()
        ref, source, therock_commit = self._resolve_rocm_libraries_ref(
            required=not has_checkout
        )
        # Only a full commit id can be compared with HEAD; a branch moves.
        pinned = re.fullmatch(r"[0-9a-f]{40}", ref) is not None

        if has_checkout:
            # A sparse clone made by an older setup lacks root dirs added since.
            missing = [
                path
                for path in ROCM_LIBRARIES_ROOT_DIRS
                if not (ROCM_LIBRARIES_DIR / path).is_dir()
            ]
            if missing and self._rocm_libraries_is_sparse():
                run_git(
                    ["-C", str(ROCM_LIBRARIES_DIR), "sparse-checkout", "add", *missing]
                )
            # An existing checkout is reused as-is, so a pin bump or a broken
            # sparse clone would otherwise go unnoticed.
            head = git_output(["-C", str(ROCM_LIBRARIES_DIR), "rev-parse", "HEAD"])
            if pinned and head != ref:
                if source == "quartz":
                    therock_note = (
                        f" (matched to TheRock {therock_commit[:12]})"
                        if therock_commit
                        else " (matched to the Quartz ROCm nightly)"
                    )
                    warn(
                        f"rocm-libraries is at {head[:12]}, not {ref[:12]}"
                        f"{therock_note}. Check out that commit in "
                        "rocm-libraries/ or delete the directory to fetch it "
                        "(or pass --rocm-libraries-ref to override)."
                    )
                else:
                    warn(
                        f"rocm-libraries is at {head[:12]}, not the pinned "
                        f"{ref[:12]}. Check out that commit in rocm-libraries/ "
                        "or delete the directory to fetch it."
                    )
            if not (ROCM_LIBRARIES_DIR / "CMakePresets.json").is_file():
                warn(
                    "rocm-libraries/CMakePresets.json is missing (a non-cone "
                    "sparse checkout). Delete rocm-libraries/ and rerun setup."
                )
            return

        # Leave any non-empty non-git directory intact unless confirmed.
        if ROCM_LIBRARIES_DIR.exists() and any(ROCM_LIBRARIES_DIR.iterdir()):
            self._confirm(
                f"replace {ROCM_LIBRARIES_DIR}, which is not a git checkout and "
                "is not empty"
            )
        url = ROCM_LIBRARIES_REPO_URL
        print(
            f"Fetching rocm-libraries ({ref}) via sparse checkout "
            f"({', '.join(ROCM_LIBRARIES_SPARSE_DIRS)})..."
        )
        # Clone beside the target and move it in only once checked out: a
        # failed or interrupted fetch must not leave a .git directory behind
        # that the next run would mistake for a usable checkout.
        staging = ROCM_LIBRARIES_DIR.with_name(ROCM_LIBRARIES_DIR.name + ".partial")
        if staging.exists():
            rmtree(staging)
        git = ["-C", str(staging)]
        try:
            run_git(
                [
                    "clone",
                    "--quiet",
                    "--filter=blob:none",
                    "--sparse",
                    "--no-checkout",
                    url,
                    str(staging),
                ]
            )
            # Cone mode explicitly: git < 2.37 defaults to non-cone patterns,
            # which match "cmake" at any depth and drop root files such as
            # CMakePresets.json. `set --cone` needs git 2.35; `init --cone` is
            # older.
            run_git([*git, "sparse-checkout", "init", "--cone"])
            run_git([*git, "sparse-checkout", "set", *ROCM_LIBRARIES_SPARSE_DIRS])
            run_git([*git, "fetch", "--quiet", "--depth", "1", "origin", ref])
            run_git([*git, "checkout", "--quiet", "FETCH_HEAD"])
        except BaseException:
            if staging.exists():
                rmtree(staging)
            raise
        if ROCM_LIBRARIES_DIR.exists():
            rmtree(ROCM_LIBRARIES_DIR)
        os.replace(staging, ROCM_LIBRARIES_DIR)

    @staticmethod
    def _rocm_libraries_is_sparse() -> bool:
        try:
            return (
                git_output(
                    [
                        "-C",
                        str(ROCM_LIBRARIES_DIR),
                        "config",
                        "--get",
                        "core.sparseCheckout",
                    ]
                )
                == "true"
            )
        except subprocess.CalledProcessError:
            return False

    @staticmethod
    def _pinned_submodule_commit(name: str, fallback_note: str) -> str:
        """The commit this repository pins for submodule ``name``.

        Without git metadata (a Docker build or a source tarball) there is no
        pin to read, so warn and return the submodule's moving branch.
        """
        try:
            return git_output(["-C", str(SCRIPT_DIR), "rev-parse", f"HEAD:{name}"])
        except subprocess.CalledProcessError:
            branch = git_output(
                [
                    "config",
                    "-f",
                    str(SCRIPT_DIR / ".gitmodules"),
                    f"submodule.{name}.branch",
                ]
            )
            warn(
                f"could not resolve the pinned {name} commit (no git metadata, "
                "e.g. a Docker build or source tarball); falling back to the "
                f"MOVING branch '{branch}'.",
                fallback_note,
            )
            return branch

    # -- AITER ----------------------------------------------------------------

    def ensure_aiter_checkout(self) -> None:
        """Fetch AITER at its pinned submodule commit if absent.

        `git submodule update --init --recursive aiter` also works. This path
        fetches only the pinned commit and the AITER submodules that its JIT
        compiles against (AITER_SUBMODULES), each at depth 1. An existing
        checkout is reused as-is.
        """
        ref = self._pinned_submodule_commit(
            "aiter", "The AITER rows are not reproducible."
        )
        pinned = re.fullmatch(r"[0-9a-f]{40}", ref) is not None
        if (AITER_DIR / ".git").exists():
            head = git_output(["-C", str(AITER_DIR), "rev-parse", "HEAD"])
            if pinned and head != ref:
                warn(
                    f"aiter is at {head[:12]}, not the pinned {ref[:12]}. Check "
                    "out that commit in aiter/ (e.g. `git submodule update "
                    "--recursive aiter`) or delete the directory to fetch it."
                )
            self._init_aiter_submodules(AITER_DIR)
            return

        # An uninitialised submodule leaves an empty directory; anything else
        # here is not ours to delete without asking.
        if AITER_DIR.exists() and any(AITER_DIR.iterdir()):
            self._confirm(
                f"replace {AITER_DIR}, which is not a git checkout and is not empty"
            )
        url = git_output(
            ["config", "-f", str(SCRIPT_DIR / ".gitmodules"), "submodule.aiter.url"]
        )
        print(f"Fetching aiter ({ref}) at depth 1...")
        # Same staging as rocm-libraries: an interrupted fetch must not leave
        # a .git directory that the next run would reuse.
        staging = AITER_DIR.with_name(AITER_DIR.name + ".partial")
        if staging.exists():
            rmtree(staging)
        git = ["-C", str(staging)]
        try:
            run_git(["init", "--quiet", str(staging)])
            run_git([*git, "remote", "add", "origin", url])
            run_git([*git, "fetch", "--quiet", "--depth", "1", "origin", ref])
            run_git([*git, "checkout", "--quiet", "FETCH_HEAD"])
            self._fetch_tags_at(staging, ref)
            self._init_aiter_submodules(staging)
        except BaseException:
            if staging.exists():
                rmtree(staging)
            raise
        if AITER_DIR.exists():
            rmtree(AITER_DIR)
        os.replace(staging, AITER_DIR)

    @staticmethod
    def _fetch_tags_at(checkout: Path, commit: str) -> None:
        """Fetch the tags of ``origin`` that point at ``commit``.

        AITER's package version comes from ``git describe`` (setuptools_scm).
        A depth-1 fetch has no tags, so a release commit would install as
        ``0.1.1.dev1+g<sha>`` and the result files would record that version.
        """
        listing = git_output(["-C", str(checkout), "ls-remote", "--tags", "origin"])
        tags = set()
        for line in listing.splitlines():
            sha, _, name = line.partition("\t")
            if sha == commit:
                tags.add(name.removeprefix("refs/tags/").removesuffix("^{}"))
        for tag in sorted(tags):
            ref = f"refs/tags/{tag}"
            run_git(
                ["-C", str(checkout), "fetch", "--quiet", "--depth", "1", "origin"]
                + [f"{ref}:{ref}"]
            )

    @staticmethod
    def _init_aiter_submodules(checkout: Path) -> None:
        """Fetch each AITER submodule that is not checked out, at depth 1.

        A checked-out submodule is left alone, so local changes survive.
        """
        missing = [
            path
            for path in AITER_SUBMODULES
            if not (checkout / path).is_dir() or not any((checkout / path).iterdir())
        ]
        if missing:
            run_git(
                [
                    "-C",
                    str(checkout),
                    "submodule",
                    "update",
                    "--quiet",
                    "--init",
                    "--depth",
                    "1",
                    *missing,
                ]
            )

    # -- venv lifecycle -----------------------------------------------------

    def setup_venv(self) -> None:
        """Reuse the workspace venv (and its torch), or create it.

        --clean deletes it first. A fresh venv holds no torch, so
        installed_torch_mode (probed in run() for a reused venv) stays valid.
        """
        if self.clean and self.venv_dir.is_dir():
            print(f"Removing virtual environment at {self.venv_dir} (--clean)...")
            rmtree(self.venv_dir)
            self.installed_torch_mode = "missing"
        if venv_python(self.venv_dir).exists():
            print(f"Reusing virtual environment at {self.venv_dir}.")
        else:
            print(f"Creating virtual environment at {self.venv_dir}...")
            run([sys.executable, "-m", "venv", str(self.venv_dir)])
        # build_hipdnn rewrites it with ROCM_PATH and LD_LIBRARY_PATH; a reused
        # venv keeps the full file even when a later stage fails.
        if not IS_WINDOWS and not (self.venv_dir / "bin" / "activate.local").exists():
            self.write_activate_local()

    def write_activate_local(
        self, rocm_prefix: str = "", lib_dirs: tuple[str, ...] = ()
    ) -> None:
        """Write the venv's activate.local and make activate source it."""
        lines = [
            f"export PYTHONPYCACHEPREFIX={shlex.quote(str(self.workspace / 'pycache'))}",
            f"export DNN_BENCH_WORKSPACE={shlex.quote(str(self.workspace))}",
            # Where the AITER stage compiled AITER's modules (see
            # aiter_executor.prepare_aiter_environment): a plain `import
            # aiter` in the venv then loads them instead of compiling again.
            f"export AITER_JIT_DIR={shlex.quote(str(self.venv_dir / 'aiter-jit'))}",
        ]
        if rocm_prefix:
            lines.append(f"export ROCM_PATH={shlex.quote(rocm_prefix)}")
        for lib_dir in reversed(lib_dirs):
            lines += [
                'case ":${LD_LIBRARY_PATH:-}:" in',
                f"    *:{shlex.quote(lib_dir)}:*) ;;",
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
                f"--torch-mode {expected} requested, but {self.venv_dir} "
                f"contains torch mode '{self.installed_torch_mode}'.",
                "Pass --clean to recreate the virtual environment, or use another "
                "--workspace, before changing torch modes.",
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
    def cmake(self) -> str:
        """The checked cmake executable; the check runs once per Setup."""
        return require_working_cmake()

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
        """The one GPU arch the ROCm tools report, "" when none.

        A host with several distinct archs has no single right answer, so the
        user must choose.
        """
        tool, archs = Setup._detected_gpu_archs()
        if len(archs) > 1:
            fail(
                f"{tool} reports several GPU architectures: {', '.join(archs)}.",
                "Pass --gpu-arch to choose the one to install and build for.",
            )
        return archs[0] if archs else ""

    @staticmethod
    def _detected_gpu_archs() -> tuple:
        """(tool, sorted GPU archs) from the first ROCm tool that reports any,
        or ("", []). gfx000 is the CPU agent."""
        for tool in ("rocm_agent_enumerator", "rocminfo"):
            if not shutil.which(tool):
                continue
            try:
                out = subprocess.run(
                    [tool],
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=DETECT_TIMEOUT_S,
                ).stdout
            except subprocess.TimeoutExpired:
                warn(f"{tool} did not finish within {DETECT_TIMEOUT_S}s; ignoring it.")
                continue
            archs = sorted(set(_GFX_ARCH.findall(out)) - {"gfx000"})
            if archs:
                return tool, archs
        return "", []

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
            "no usable ROCm SDK libraries package found in this venv.",
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
                "no ROCm SDK compiler/toolchain prefix found in this venv.",
                "Expected the rocm-sdk-devel package (rocm[devel]) alongside the "
                "rocm_sdk module.",
                "Install rocm-sdk-devel from the same ROCm torch index, or pass "
                "--rocm-prefix.",
            )

        # The devel package must match the ROCm wheels already in the venv.
        version = (self._read_venv_record(ROCM_BUILD_RECORD) or {}).get("rocm_version")
        package = f"rocm-sdk-devel=={version}" if version else "rocm-sdk-devel"
        print(
            "ROCm SDK compiler/toolchain prefix not found; installing "
            f"{package} from {index_url}...",
            file=sys.stderr,
        )
        self.pip("install", "--pre", package, "--index-url", index_url)

        root = self._rocm_sdk_devel_root(report_errors=True)
        if root:
            return root
        fail(
            "rocm-sdk-devel installed, but `rocm-sdk path --root` could "
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

    def _link_wheel_amdsmi(self, package: Path, link: Path) -> bool:
        """Put the SDK wheel's ready-made amdsmi package on the venv path.

        pip cannot install it: it has no setup.py or pyproject.toml. The
        package loads libamd_smi relative to its resolved path, so a symlink
        keeps that lookup and a copy would break it.
        """
        print(f"Linking amdsmi Python bindings from {package}...")
        try:
            if link.is_symlink():  # stale link from an earlier setup
                link.unlink()
            link.symlink_to(package, target_is_directory=True)
        except OSError as error:
            warn(f"cannot link {link} to {package}: {error}")
            return False
        if self.amdsmi_importable():
            return True
        link.unlink()
        warn(f"amdsmi from {package} does not import; trying other sources.")
        return False

    def maybe_install_amdsmi(self, *prefixes: str) -> None:
        if self.amdsmi_importable():
            return

        candidates = []
        prefix, status = self.find_rocm_wheel_prefix("core")
        if status == 0:
            source = Path(prefix) / "share" / "amd_smi"
            package = source / "amdsmi"
            if self._is_amdsmi_source(source):
                candidates.append(source)
            elif (package / "__init__.py").is_file() and self._link_wheel_amdsmi(
                package, Path(prefix).parent / "amdsmi"
            ):
                return
        elif status != 1:
            warn("ROCm SDK core discovery failed; skipping SDK amdsmi candidate.")
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
            warn(f"amdsmi install from {candidate} failed; trying next candidate.")

        warn(
            "amdsmi Python bindings were not installed; GPU SMI snapshot will "
            "be disabled."
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
            # Reuse must not install the devel SDK into an existing venv.
            index_url = "" if self.reuse_artifacts else self.resolved_torch_index_url
            return self.ensure_rocm_wheel_devel_prefix(index_url)
        if self.torch_mode == "existing" and self.installed_torch_mode == "rocm":
            return self.ensure_rocm_wheel_devel_prefix("")
        return self.resolve_installed_rocm_prefix()

    # -- torch install ------------------------------------------------------

    def _require_gpu_arch(self) -> str:
        """The resolved --gpu-arch, or a fatal error naming what's missing."""
        if not self.gpu_arch:
            fail(
                "could not detect a GPU architecture.",
                "Pass --gpu-arch (e.g. gfx90a, gfx942, gfx950) or --torch-index-url "
                "to override detection.",
            )
        return self.gpu_arch

    def install_torch(self) -> None:
        mode = self.torch_mode
        if mode == "none":
            # A reused venv keeps its torch; "none" must not build against it.
            if self.installed_torch_mode != "missing":
                self.require_torch_mode("none")
            print("Leaving torch uninstalled.")
            return

        if mode == "existing":
            if self.installed_torch_mode == "missing":
                fail(
                    "--torch-mode existing requires torch to already be "
                    f"installed in {self.venv_dir}.",
                    "Use --torch-mode rocm or --torch-mode cpu to install torch "
                    "automatically.",
                )
            print(f"Using existing PyTorch in {self.venv_dir}.")
            return

        arch = ""
        if mode == "cpu":
            index_url = self.torch_index_url or "https://download.pytorch.org/whl/cpu"
            if self.installed_torch_mode != "missing":
                self.require_torch_mode("cpu")
                self.require_same_torch_source()
                print(f"Using existing CPU-only PyTorch in {self.venv_dir}.")
                return
            print(f"Installing CPU-only PyTorch from {index_url}")
            self.pip("install", "torch", "--index-url", index_url)
        elif mode == "cuda":
            index_url = self.torch_index_url  # "" = PyPI
            if self.installed_torch_mode != "missing":
                self.require_torch_mode("cuda")
                self.require_same_torch_source()
                print(f"Using existing CUDA PyTorch in {self.venv_dir}.")
                return
            if index_url:
                print(f"Installing CUDA PyTorch from {index_url}")
                self.pip("install", "torch", "--index-url", index_url)
            else:
                print("Installing CUDA PyTorch from PyPI")
                self.pip("install", "torch")
        elif mode == "rocm":
            index_url = self.torch_index_url or ROCM_TORCH_INDEX_URL
            self.resolved_torch_index_url = index_url
            if self.installed_torch_mode != "missing":
                self.require_torch_mode("rocm")
                self.require_same_torch_source(check_arch=True)
                print(f"Using existing ROCm PyTorch in {self.venv_dir}.")
                return
            arch = self._require_gpu_arch()
            print(f"GPU arch: {arch}")
            build = self.select_rocm_build(arch)
            index_url = self.resolved_torch_index_url = build.index_url
            self.install_rocm_torch(build, arch)

        self.installed_torch_mode = self.get_torch_mode()
        self.require_torch_mode(mode)
        (self.venv_dir / TORCH_RECORD).write_text(
            json.dumps({"torch_index_url": index_url, "gpu_arch": arch}) + "\n"
        )
        if mode == "rocm":
            self.write_rocm_build_record(build)

    def select_rocm_build(self, arch: str) -> RocmBuild:
        """The ROCm build to install: --rocm-version (latest or explicit), else
        the newest build on --torch-index-url, else rocm-pin.json."""
        platform_name = quartz_platform()
        if self.rocm_version == "latest":
            print(
                f"Selecting the newest ROCm nightly from Quartz ({QUARTZ_NIGHTLY_URL})"
            )
            return select_quartz_nightly(platform_name, arch, self.torch_index_url)
        if self.rocm_version:
            if ROCM_PIN_FILE.is_file():
                pin = read_rocm_pin(arch)
                if pin.rocm_version == self.rocm_version:
                    return dataclasses.replace(
                        pin,
                        index_url=self.torch_index_url or pin.index_url,
                    )
            return describe_rocm_version(
                self.rocm_version, self.torch_index_url, platform_name, arch
            )
        if self.torch_index_url:
            return RocmBuild("", self.torch_index_url, "--torch-index-url")
        default_arch, pins = read_rocm_pins()
        norm_arch = arch.strip().lower()
        if norm_arch in pins:
            return pins[norm_arch]
        print(
            f"{ROCM_PIN_FILE.name} has no entry for {arch}; "
            f"using default ({default_arch})."
        )
        return pins[default_arch]

    def install_rocm_torch(self, build: RocmBuild, arch: str) -> None:
        """Install ROCm PyTorch and the ROCm SDK wheels of ``build``."""
        if build.rocm_version:
            print(
                f"Installing ROCm PyTorch for ROCm {build.rocm_version} "
                f"(TheRock {build.therock_commit or 'commit unknown'}, "
                f"built {build.build_date or 'on an unknown date'}) "
                f"from {build.index_url}"
            )
        else:
            print(f"Installing the newest ROCm PyTorch from {build.index_url}")
        if build.untested:
            print(
                f"Quartz has no {' or '.join(build.untested)} test result for "
                f"{arch} in this nightly; setup records it as untested."
            )
        # The torch wheel requires rocm[libraries]==<its ROCm version>, so the
        # pin on rocm also selects a torch of that build.
        pin = f"=={build.rocm_version}" if build.rocm_version else ""
        # ROCm nightlies are pre-release, so --pre lets pip select them.
        # device-<arch> is the pip extra that pulls the matching
        # rocm-sdk-device-<arch> package; the base rocm-sdk-core/-libraries
        # packages are OS/arch-agnostic on this index, so Linux and
        # Windows always resolve the exact same nightly.
        self.pip(
            "install",
            "--pre",
            f"torch[device-{arch}]",
            f"rocm[libraries,devel,device-{arch}]{pin}",
            "--index-url",
            build.index_url,
        )

    def installed_rocm_version(self) -> str:
        """The version of the ``rocm`` package in the venv."""
        result = self.probe("import importlib.metadata as m; print(m.version('rocm'))")
        if result.returncode != 0:
            fail(
                "cannot read the version of the installed rocm package.",
                result.stderr.strip(),
            )
        return result.stdout.strip()

    def write_rocm_build_record(self, build: RocmBuild) -> None:
        """Record the installed ROCm build for the result environment."""
        record = {
            "rocm_version": build.rocm_version or self.installed_rocm_version(),
            "build_date": build.build_date,
            "therock_commit": build.therock_commit,
            "rocm_libraries_commit": build.rocm_libraries_commit,
            "source": build.source,
            "status_url": build.status_url,
            "index_url": build.index_url,
            "untested": None if build.untested is None else list(build.untested),
        }
        (self.venv_dir / ROCM_BUILD_RECORD).write_text(json.dumps(record) + "\n")

    def _read_venv_record(self, name: str) -> Optional[dict]:
        """A JSON record that setup wrote into the venv, or None when absent
        or unreadable."""
        try:
            return json.loads((self.venv_dir / name).read_text())
        except (OSError, ValueError):
            return None

    def require_same_torch_source(self, check_arch: bool = False) -> None:
        """Fail when an explicit flag differs from what the reused venv holds.

        Compares --torch-index-url (and, with ``check_arch``, --gpu-arch and
        --rocm-version) with the TORCH_RECORD and ROCM_BUILD_RECORD written
        when setup installed torch. A venv without the record (torch
        installed some other way) cannot be checked. With
        ``check_arch`` and no --gpu-arch, the build uses the recorded arch,
        and fails when the detected GPU has another arch; with no
        --torch-index-url, the toolchain uses the recorded index.
        """
        record = self._read_venv_record(TORCH_RECORD)
        if (
            check_arch
            and not self.gpu_arch_override
            and record
            and record.get("gpu_arch")
        ):
            recorded = record["gpu_arch"]
            # Detection alone, not self.gpu_arch: its Windows default is no
            # evidence of the GPU in this host. A multi-arch host may keep a
            # venv recorded for any of its archs.
            _, detected = self._detected_gpu_archs()
            if detected and recorded not in detected:
                found = ", ".join(detected)
                fail(
                    f"this host has {found} GPU(s), but {self.venv_dir} was "
                    f"set up for {recorded}.",
                    "Pass --clean to set up the venv again (with --gpu-arch "
                    f"when the host has several archs), or --gpu-arch {recorded} "
                    f"to keep the {recorded} build.",
                )
            self.gpu_arch = recorded
            print(f"GPU arch: {self.gpu_arch} (recorded in {self.venv_dir})")
        if check_arch and not self.torch_index_url and record:
            self.resolved_torch_index_url = record.get(
                "torch_index_url", self.resolved_torch_index_url
            )
        checks = [
            ("--torch-index-url", self.torch_index_url, TORCH_RECORD, "torch_index_url")
        ]
        if check_arch:
            if self.rocm_version == "latest":
                fail(
                    "--rocm-version latest requests a fresh Quartz nightly and "
                    f"cannot reuse {self.venv_dir}.",
                    "Pass --clean to recreate the virtual environment, or omit "
                    "--rocm-version to keep the installed build.",
                )
            checks.append(
                ("--gpu-arch", self.gpu_arch_override, TORCH_RECORD, "gpu_arch")
            )
            checks.append(
                ("--rocm-version", self.rocm_version, ROCM_BUILD_RECORD, "rocm_version")
            )
        for flag, value, record_name, key in checks:
            if not value:
                continue
            recorded = self._read_venv_record(record_name)
            if recorded is None:
                warn(
                    f"{self.venv_dir} has no {record_name}, so setup cannot check "
                    f"that its torch matches {flag} {value}.",
                    "Pass --clean if it does not.",
                )
            elif recorded.get(key) != value:
                fail(
                    f"{flag} {value} differs from the "
                    f"'{recorded.get(key) or 'default'}' that {self.venv_dir} "
                    "was set up with.",
                    "Pass --clean to recreate the virtual environment, or use "
                    "another --workspace.",
                )

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
            f'cmd /d /s /c "call "{vcvars}" >nul && set"',
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

    def install_build_deps(self) -> None:
        """pip packages the source build runs in the venv interpreter."""
        # `build` packs the hipdnn_frontend wheel. rocKE (Linux only, see
        # rocke_args) packs its kernels with rocm_kpack, which the configure
        # imports from Python3_EXECUTABLE; its msgpack/zstandard deps are not
        # pulled in.
        deps = ["build"]
        if not IS_WINDOWS:
            deps += [
                "dvc[s3]>=3,<4",
                "msgpack>=1.0.0",
                "zstandard>=0.20.0",
            ]
        self.pip("install", *deps)

    def pull_provider_artifacts(self) -> None:
        """Materialize every DVC-backed kernel-ingestor descriptor."""
        pointers = sorted(PROVIDER_DESCRIPTOR_ROOT.glob("**/*.dvc"))
        if not pointers:
            return
        targets = [
            pointer.relative_to(ROCM_LIBRARIES_DIR).as_posix() for pointer in pointers
        ]
        env = dict(self.env)
        ca_bundle = env.get("PIP_CERT") or env.get("SSL_CERT_FILE")
        if ca_bundle:
            env.setdefault("AWS_CA_BUNDLE", ca_bundle)
        # Keep the DVC cache inside the writable workspace when no global DVC
        # config dir is set, avoiding EROFS failures on read-only repo mounts.
        if not env.get("DVC_GLOBAL_CONFIG_DIR"):
            dvc_cfg_dir = self.workspace / "dvc-config"
            dvc_cache = self.workspace / "dvc-cache"
            dvc_cfg_dir.mkdir(parents=True, exist_ok=True)
            dvc_cache.mkdir(parents=True, exist_ok=True)
            (dvc_cfg_dir / "config").write_text(
                f"[cache]\n    dir = {dvc_cache}\n", encoding="utf-8"
            )
            env["DVC_GLOBAL_CONFIG_DIR"] = str(dvc_cfg_dir)
        print(f"Pulling {len(targets)} provider descriptor artifact(s)...")
        run(
            [self.py, "-m", "dvc", "pull", *targets],
            cwd=ROCM_LIBRARIES_DIR,
            env=env,
        )

    def rocke_args(self, toolchain_prefix: str) -> list[str]:
        """Configure defines that build rocKE and its ingestor engines.

        Every rocKE engine, including descriptor-backed ones such as
        hipkernel:Gfx950AttentionDense, ships only with both gates ON. Linux
        only: the Windows ROCm build of rocKE is untested, so Windows keeps the
        rocm-libraries defaults (both OFF).
        """
        if IS_WINDOWS:
            return []
        args = [
            "-DHIPKERNELPROVIDER_ENABLE_ROCKE=ON",
            "-DHIPDNN_ENABLE_KERNEL_INGESTOR=ON",
            "-DHIPKERNELPROVIDER_KPACK_ALLOW_FETCH=ON",
            f"-DPython3_EXECUTABLE={self.py}",
            # The ingestor headers' std::stable_sort trips libstdc++ 12's
            # deprecated get_temporary_buffer under the wheel clang's -Werror
            # (Ubuntu 22.04). _INIT keeps the user's $CXXFLAGS, which CMake
            # puts in front of it.
            "-DCMAKE_CXX_FLAGS_INIT=-Wno-error=deprecated-declarations",
        ]
        comgr = self._comgr_lib(toolchain_prefix)
        if comgr:
            args.append(f"-DHIPKERNELPROVIDER_ROCKE_COMGR_LIB={comgr}")
        return args

    def _comgr_lib(self, toolchain_prefix: str):
        """libamd_comgr for rocKE's pack step, or None.

        rocKE loads comgr through ctypes from $ROCM_PATH or /opt/rocm*, and
        neither names the wheel SDK. Some devel wheels lack the runtime .so,
        which then ships in the sibling core wheel. The core wheel is probed
        only when the toolchain has no comgr.
        """
        libs = sorted((Path(toolchain_prefix) / "lib").glob("libamd_comgr.so*"))
        if not libs:
            core_prefix, status = self.find_rocm_wheel_prefix("core")
            if status == 0:
                libs = sorted((Path(core_prefix) / "lib").glob("libamd_comgr.so*"))
        return libs[0] if libs else None

    def _reset_build_dir(self, build_dir: Path, configure_args: list) -> None:
        """Keep build_dir for an incremental build only when it was configured
        with exactly ``configure_args`` (see BUILD_RECORD); wipe it otherwise or
        on --clean."""
        wanted = {"configure_args": list(configure_args)}
        record = build_dir / BUILD_RECORD
        if build_dir.exists():
            try:
                same = json.loads(record.read_text()) == wanted
            except (OSError, ValueError):
                same = False
            if not same and not self.clean:
                print(f"Removing {build_dir}: configured with other arguments.")
            if self.clean or not same:
                rmtree(build_dir)
        build_dir.mkdir(parents=True, exist_ok=True)
        record.write_text(json.dumps(wanted) + "\n")

    def build_superbuild(self, install_prefix: str, toolchain_prefix: str) -> None:
        cmake = self.cmake
        if not shutil.which("ninja"):
            fail("ninja not found on PATH.")

        build_dir = ROCM_LIBRARIES_DIR / "build"
        prefix_path, program_path = self._cmake_paths(install_prefix, toolchain_prefix)
        configure_args = [
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
            *self.rocke_args(toolchain_prefix),
            "-DENABLE_CLANG_FORMAT=OFF",
            "-DENABLE_CLANG_TIDY=OFF",
            # LAST, so a caller's -D overrides a default above rather than
            # being silently overridden by it.
            *self.extra_cmake_args,
        ]
        self._reset_build_dir(build_dir, configure_args)
        print(f"Building hipDNN and providers to {install_prefix}...")
        run(
            [cmake, *configure_args],
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
        cmake = self.cmake
        python_dir = HIPDNN_ROOT / "python"
        bindings_source = python_dir / "frontend_bindings"
        if not bindings_source.is_dir():
            fail(f"hipDNN frontend bindings not found at {bindings_source}")
        build_root = python_dir / "build"
        bindings_build = build_root / "frontend_bindings"
        wheel_dir = build_root / "wheel_package"
        prefix_path, program_path = self._cmake_paths(install_prefix, toolchain_prefix)
        configure_args = [
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
        ]
        self._reset_build_dir(bindings_build, configure_args)
        # The packer's output, not a build cache: a wheel left by an earlier
        # run would make the pick below ambiguous or stale.
        if wheel_dir.exists():
            rmtree(wheel_dir)

        run([cmake, *configure_args], env=self._build_env())
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
            fail(f"expected exactly one hipdnn_frontend wheel in {wheel_dir}")
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

    def build_hipdnn(self) -> None:
        """Build (or, with --reuse-artifacts, locate) hipDNN and the provider
        plugins, then point the venv's environment at them."""
        if self.gpu_arch:
            # Belt-and-suspenders for any torch C++/HIP extension compile (none
            # today: the bindings are nanobind host code linking hip::host).
            self.env.setdefault("PYTORCH_ROCM_ARCH", self.gpu_arch)
        install_prefix = self.install_prefix = self.select_binding_prefix()
        print(f"Using hipDNN/ROCm prefix: {install_prefix}")
        toolchain_prefix = self.toolchain_prefix
        if self.do_build:
            self.build_superbuild(install_prefix, toolchain_prefix)
        elif not self.prefix_has_hipdnn(install_prefix):
            fail(
                "--reuse-artifacts was passed, but hipDNN CMake configs "
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
            warn(f"no native hipDNN engine plugins found under {install_prefix}.")

        self.env["ROCM_PATH"] = install_prefix
        if not IS_WINDOWS:
            # Install prefix first: the ROCm devel wheel ships its own, older
            # libhipdnn_backend.so, which must not shadow the one just built.
            lib_dirs = tuple(
                str(Path(prefix) / "lib")
                for prefix in (install_prefix, toolchain_prefix)
                if prefix and (Path(prefix) / "lib").is_dir()
            )
            current = self.env.get("LD_LIBRARY_PATH", "")
            existing = current.split(":") if current else []
            ordered = [path for path in lib_dirs if path not in existing] + existing
            self.env["LD_LIBRARY_PATH"] = ":".join(ordered)
            self.write_activate_local(install_prefix, lib_dirs)

    def install_bindings(self) -> None:
        if self.do_build:
            self.build_and_install_bindings(self.install_prefix, self.toolchain_prefix)
            return
        # --reuse-artifacts builds no bindings, so the venv has to carry them
        # already. Look for the installed package instead of importing it: the
        # native libraries behind it may need a GPU or a ROCm runtime that this
        # host cannot initialise, which says nothing about the install.
        installed = self.probe(
            "import importlib.util, sys; "
            "sys.exit(importlib.util.find_spec('hipdnn_frontend') is None)"
        )
        if installed.returncode == 0:
            return
        if installed.stderr.strip():
            fail(
                f"could not inspect {self.venv_dir} for hipdnn_frontend.",
                installed.stderr.strip(),
            )
        fail(
            "--reuse-artifacts builds no hipDNN Python bindings, and "
            f"hipdnn_frontend is not installed in {self.venv_dir}.",
            "Install hipdnn_frontend in this venv first, or drop "
            "--reuse-artifacts to build hipDNN and its bindings.",
        )

    def install_rocke_packages(self) -> None:
        """Install rocKE's Python packages for ``--runtime rocke``.

        They come from the rocm-libraries checkout that the plugins build
        from, so rocKE's specs and runtime match the kernels hipDNN ships. The
        library package provides the generic top-level names ``kernels``,
        ``builders`` and ``dispatch``; rocKE keeps them in a dedicated venv.
        ``--no-editable`` copies both packages into the venv so it survives
        without the source tree.
        """
        rocke = ROCM_LIBRARIES_DIR / "dnn-providers" / "hip-kernel-provider" / "rocke"
        if not (rocke / "library" / "pyproject.toml").is_file():
            warn(
                f"rocKE sources not found under {rocke}.",
                "--runtime rocke needs them; check the rocm-libraries checkout.",
            )
            return
        platform, library = str(rocke / "platform"), str(rocke / "library")
        if self.editable_install:
            # compat mode writes a .pth to the source root, as rocKE's
            # BUILDING.md requires.
            compat = ("--config-settings", "editable_mode=compat")
            platform_args: tuple[str, ...] = (*compat, "-e", platform)
            library_args: tuple[str, ...] = (*compat, "-e", library)
        else:
            platform_args, library_args = (platform,), (library,)
        # The library resolves `rocke` from the platform install.
        self.pip("install", *platform_args)
        self.pip("install", "--no-deps", *library_args)

    def install_aiter(self) -> None:
        """Install the pinned AITER for ``--runtime aiter`` and compile its kernels.

        AITER compiles its host modules (JIT) against the installed torch, so
        it is installed from source (the ``aiter`` submodule) and only with
        ROCm torch, without build isolation (see :func:`aiter_build_requires`).
        On a GPU that AITER supports, the stage then runs every AITER backend
        once on small problems (``aiter_executor.prebuild``); the compiled
        modules stay in ``<venv>/aiter-jit``. --reuse-artifacts fetches no
        AITER sources: it installs an existing ``aiter/`` checkout or skips.
        """
        if self.installed_torch_mode != "rocm":
            print(
                f"Skipping AITER: --runtime aiter needs ROCm torch, not "
                f"'{self.installed_torch_mode}'."
            )
            return
        if self.reuse_artifacts and not (AITER_DIR / ".git").exists():
            print(
                "Skipping AITER: --reuse-artifacts fetches no sources and aiter/ "
                "is not checked out (`git submodule update --init --recursive "
                "aiter`)."
            )
            return
        self.ensure_aiter_checkout()
        env = dict(self.env, AITER_USE_SYSTEM_TRITON="1")
        # AITER's setup.py reads the HIP version from ROCM_HOME, else
        # ROCM_PATH. ROCM_PATH is the hipDNN prefix (the libraries wheel with
        # ROCm torch), which has no hipconfig and no version file. Without
        # /opt/rocm or a hipconfig on PATH, the build then stops with "ROCm
        # version file not found". The toolchain prefix has both.
        env.setdefault("ROCM_HOME", self.toolchain_prefix)
        # AITER's runtime imports (pyyaml, ...) are in requirements.txt only.
        self.pip("install", "-r", str(AITER_DIR / "requirements.txt"), env=env)
        self.pip("install", *aiter_build_requires(AITER_DIR), env=env)
        # compat mode puts aiter/ on sys.path through a .pth file. The default
        # finder loses to the aiter/ checkout itself, which imports as an empty
        # namespace package from the repository root (python -c, python -m).
        compat = ("--config-settings", "editable_mode=compat")
        self.pip(
            "install", "--no-build-isolation", *compat, "-e", str(AITER_DIR), env=env
        )
        prebuild = run([self.py, "-c", _AITER_PREBUILD], env=env, check=False)
        if prebuild.returncode != 0:
            warn(
                "Some AITER backends did not compile or run (see above).",
                "Their --runtime aiter rows report the same error.",
            )

    def install_runtime_extras(self) -> None:
        self.maybe_install_amdsmi(
            self.install_prefix,
            self.toolchain_prefix,
            self.rocm_prefix,
            DEFAULT_ROCM_PREFIX,
        )
        self.unify_wheel_rocprofiler_libs()

    # -- confirmation prompt ------------------------------------------------

    def _confirm(self, action: str) -> None:
        """Return if the user agrees to `action`; exit otherwise.

        Only an explicit y/yes (or Enter, the default) proceeds. With no
        terminal to ask, -y is the only way to agree.
        """
        if self.auto_yes:
            return
        if sys.stdin is None or not sys.stdin.isatty():
            fail(
                f"setup needs confirmation to {action}, but stdin is not a terminal.",
                "Pass -y to proceed non-interactively.",
            )
        try:
            answer = input(f"This will {action}. Continue? [Y/n] ")
        except (EOFError, OSError, RuntimeError) as exc:
            # A tty can still be closed between the check and the read. Only a
            # genuinely lost stdin is reported this way; input() raises
            # RuntimeError for other reasons, which must not be swallowed.
            if isinstance(exc, RuntimeError) and "lost sys.stdin" not in str(exc):
                raise
            fail(
                f"setup needs confirmation to {action}, but stdin could not be read.",
                "Pass -y to proceed non-interactively.",
            )
        if answer.strip().lower() not in ("", "y", "yes"):
            print("Aborted.")
            sys.exit(0)

    def confirm_build(self) -> None:
        actions = []
        if self.clean and self.venv_dir.is_dir():
            actions.append(
                f"delete and recreate the virtual environment at {self.venv_dir}"
            )
        if self.do_build:
            actions.append(
                "build and install hipDNN and provider plugins from source"
                + (" from scratch" if self.clean else "")
            )
        if actions:
            self._confirm(" and ".join(actions))

    # -- verify -------------------------------------------------------------

    def verify(self) -> None:
        result = self.probe("import dnn_benchmarking; print('dnn_benchmarking OK')")
        sys.stdout.write(result.stdout)
        if result.returncode != 0:
            sys.stderr.write(result.stderr)
            fail("dnn_benchmarking failed to import.")

        result = self.probe("import hipdnn_frontend; print('hipdnn_frontend OK')")
        if result.returncode == 0:
            sys.stdout.write(result.stdout)
        else:
            warn(
                "hipdnn_frontend could not be imported (ROCm runtime or bindings missing)."
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

    def unify_wheel_rocprofiler_libs(self) -> None:
        """Deduplicate the wheels' rocprofiler libraries (no-op elsewhere).

        Only meaningful for a wheel ROCm install; a system ``--rocm-prefix``
        has one copy of each library to begin with.
        """
        core_prefix, status = self.find_rocm_wheel_prefix("core")
        if status != 0:
            return
        devel_root = self._rocm_sdk_devel_root()
        if not devel_root:
            return
        relinked = unify_rocprofiler_libs(
            Path(core_prefix) / "lib", Path(devel_root) / "lib"
        )
        if relinked:
            print(
                f"Relinked {relinked} rocprofiler libraries in the devel "
                "prefix onto the core prefix copies (rocprofv3 aborts when "
                "one process registers both paths)."
            )

    def report_profiling_sources(self) -> None:
        """Print which opt-in profiling sources this install can actually run.

        Every source degrades to a ``skipped`` slice at run time, which is
        correct but only visible after paying for a benchmark. Setup knows
        the answer now, and the two host-level gaps (perf, rocprof-compute)
        need a root install this script must not attempt.
        """
        report = self.probe(_PROFILING_SOURCES)
        if report.returncode != 0:
            return
        print("Profiling sources:")
        sys.stdout.write(report.stdout)
        print("")

    @contextlib.contextmanager
    def stage(self, name: str):
        """Announce one numbered setup stage and time it.

        current_stage stays set when the body raises, so main() can say
        which stage failed.
        """
        self._stage_index += 1
        self.current_stage = name
        print(f"==> [{self._stage_index}/{self.n_stages}] {name}", flush=True)
        start = time.monotonic()
        yield
        print(f"    done in {time.monotonic() - start:.1f}s", flush=True)

    def install_package(self) -> None:
        # pyproject.toml intentionally omits torch so pip never replaces the
        # selected wheel; torch is installed explicitly, before the package.
        if self.editable_install:
            self.pip("install", "-e", str(SCRIPT_DIR))
        else:
            self.pip("install", str(SCRIPT_DIR))

    def run(self) -> int:
        self.require_python_version()
        has_venv = venv_python(self.venv_dir).exists()
        if (
            self.reuse_artifacts
            and self.torch_mode != "cuda"
            and (self.clean or not has_venv)
        ):
            fail(
                f"--reuse-artifacts with --torch-mode {self.torch_mode} needs "
                "a pre-existing virtual environment that will not be cleaned.",
                f"No reusable venv at {self.venv_dir}. Prepare one with "
                "hipdnn_frontend, or drop --reuse-artifacts to build its bindings.",
            )
        if has_venv and not self.clean:
            self.installed_torch_mode = self.get_torch_mode()
        elif self.torch_mode == "existing":
            fail(
                f"--torch-mode existing requires an existing virtual "
                f"environment at {self.venv_dir}.",
                "Use --torch-mode rocm or --torch-mode cpu to create one and "
                "install torch automatically.",
            )
        if self.reuse_artifacts and has_venv and not self.clean:
            if self.torch_mode in ("rocm", "cpu", "cuda"):
                if self.installed_torch_mode != self.torch_mode:
                    fail(
                        f"--reuse-artifacts with --torch-mode {self.torch_mode} "
                        f"cannot reuse {self.venv_dir}: it contains torch mode "
                        f"'{self.installed_torch_mode}'.",
                        "Use --torch-mode existing to keep that torch, or drop "
                        "--reuse-artifacts and pass --clean to change modes.",
                    )
                self.require_same_torch_source(check_arch=self.torch_mode == "rocm")
            elif (
                self.torch_mode == "existing" and self.installed_torch_mode == "missing"
            ):
                fail(
                    "--reuse-artifacts with --torch-mode existing needs torch "
                    f"already installed in {self.venv_dir}.",
                    "Install torch in that venv first, or drop --reuse-artifacts.",
                )
            elif self.torch_mode == "none":
                if self.installed_torch_mode != "missing":
                    fail(
                        "--reuse-artifacts with --torch-mode none needs a venv "
                        f"without torch; found '{self.installed_torch_mode}'.",
                        "Use --torch-mode existing to keep its installed torch.",
                    )
                # An installed but broken torch also probes as "missing".
                torch = self.probe(
                    "import importlib.util, sys; "
                    "sys.exit(importlib.util.find_spec('torch') is not None)"
                )
                if torch.returncode:
                    fail(
                        "--reuse-artifacts with --torch-mode none needs a venv "
                        "without torch.",
                        torch.stderr.strip()
                        or "Remove torch from this venv before reusing it.",
                    )
        self.workspace.mkdir(parents=True, exist_ok=True)
        # CUDA torch supports only the PyTorch execution backend: no hipDNN
        # Python bindings, engine plugins, amdsmi, or ROCm prefix.
        cuda = self.torch_mode == "cuda" or (
            self.torch_mode == "existing" and self.installed_torch_mode == "cuda"
        )
        if cuda:
            self.do_build = False
        self.confirm_build()
        if self.do_build:
            # Before any stage downloads wheels: an old cmake fails only at
            # the hipDNN build, many minutes later.
            self.cmake

        stages = [
            ("Virtual environment", self.setup_venv),
            (f"PyTorch (--torch-mode {self.torch_mode})", self.install_torch),
            ("dnn-benchmarking package", self.install_package),
        ]
        if cuda:
            print(
                "CUDA torch: skipping hipDNN/provider builds, hipDNN Python "
                "bindings, and ROCm environment setup."
            )
        else:
            if self.do_build:
                stages += [
                    ("rocm-libraries sources", self.ensure_rocm_libraries_checkout),
                    ("Build dependencies", self.install_build_deps),
                    ("Provider descriptor artifacts", self.pull_provider_artifacts),
                ]
            stages += [
                ("hipDNN and provider plugins", self.build_hipdnn),
                ("hipDNN Python bindings", self.install_bindings),
            ]
            if not IS_WINDOWS:
                # The Windows ROCm build of rocKE is untested (see rocke_args).
                stages.append(("rocKE Python packages", self.install_rocke_packages))
            # AITER's source install needs FlyDSL, which ships Linux wheels only.
            if not IS_WINDOWS and self.with_aiter:
                stages.append(("AITER", self.install_aiter))
            stages.append(
                ("amdsmi and rocprofiler libraries", self.install_runtime_extras)
            )
        stages.append(("Verify installation", self.verify))

        self.n_stages = len(stages)
        for name, step in stages:
            with self.stage(name):
                step()

        self._print_complete(cuda=cuda)
        if not cuda:
            self.report_profiling_sources()
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
            print("  python -m dnn_benchmarking --graph <graph.json> --runtime pytorch")
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
    # Keep in-stage prints in order with child-process output when piped (CI).
    sys.stdout.reconfigure(line_buffering=True)
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.clean and args.torch_mode == "existing":
        parser.error(
            "--clean deletes the virtual environment --torch-mode existing reuses"
        )
    if args.rocm_version and args.torch_mode != "rocm":
        parser.error("--rocm-version needs --torch-mode rocm")
    setup = Setup(args)
    try:
        return setup.run()
    except subprocess.CalledProcessError as exc:
        cmd = exc.cmd if isinstance(exc.cmd, str) else " ".join(str(c) for c in exc.cmd)
        reason = f"command exited {exc.returncode}: {cmd}"
    except (EOFError, OSError) as exc:
        reason = str(exc) or type(exc).__name__
    except KeyboardInterrupt:
        print(f"\nAborted during stage '{setup.current_stage}'.", file=sys.stderr)
        return 130
    print(f"ERROR: stage '{setup.current_stage}' failed: {reason}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())

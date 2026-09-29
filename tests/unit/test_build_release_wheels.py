# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for tools/build_release_wheels.py.

The wheels it builds have names nobody has registered on PyPI, and the
requirements file adds PyPI as an extra index. Any of those names left as a
bare requirement would let a package published there under that name replace
the released one; an unpinned torch could resolve to a PyPI build.
"""

import hashlib
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "tools" / "build_release_wheels.py"


def _load_script():
    """Import the script by path: it is a tool, not a package."""
    spec = importlib.util.spec_from_file_location("build_release_wheels", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _FakeBuildEnv:
    index_url = "https://rocm.nightlies.amd.com/whl-multi-arch/"

    def distribution_version(self, name: str) -> str:
        assert name == "rocm"
        return "10.1.0a20260822"


def test_built_wheels_are_pinned_by_url_and_hash(tmp_path: Path) -> None:
    script = _load_script()
    wheels = {}
    for name, filename in (
        ("hipdnn-runtime-gfx942", "hipdnn_runtime_gfx942-0.1.0-cp312-abi3.whl"),
        ("hipdnn-frontend", "hipdnn_frontend-0.2.0-cp312-abi3.whl"),
        ("dnn-benchmarking", "dnn_benchmarking-0.1.0-py3-none-any.whl"),
    ):
        wheel = tmp_path / filename
        wheel.write_bytes(filename.encode())
        wheels[name] = wheel

    path = script.write_requirements(
        _FakeBuildEnv(),
        "gfx942",
        wheels,
        "e807507",
        "2.15.0a0+rocm10.1.0a20260822",
        "https://example.invalid/download/v1",
        tmp_path,
    )

    requirements = [
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith(("#", "-"))
    ]
    for name, wheel in wheels.items():
        digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
        expected = (
            f"{name} @ https://example.invalid/download/v1/{wheel.name}"
            f"#sha256={digest}"
        )
        assert [r for r in requirements if r.split()[0] == name] == [expected]
    assert "torch[device-gfx942]==2.15.0a0+rocm10.1.0a20260822" in requirements
    assert "rocm[libraries,device-gfx942]==10.1.0a20260822" in requirements


def _git(*args: str, cwd: Path) -> str:
    return subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def test_fresh_clone_checks_out_a_pin_equal_to_the_default_tip(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A --no-checkout clone already has HEAD at the tip. When the tip is the
    pin, skipping checkout leaves an empty tree and no CMakePresets.json."""
    origin = tmp_path / "origin"
    (origin / "projects" / "hipdnn").mkdir(parents=True)
    (origin / "CMakePresets.json").write_text("{}")
    (origin / "projects" / "hipdnn" / "CMakeLists.txt").write_text("")
    _git("init", "-q", cwd=origin)
    _git("add", ".", cwd=origin)
    _git("commit", "-qm", "tip", cwd=origin)
    pin = _git("rev-parse", "HEAD", cwd=origin)

    superproject = tmp_path / "super"
    superproject.mkdir()
    _git("init", "-q", cwd=superproject)
    _git(
        "update-index",
        "--add",
        "--cacheinfo",
        f"160000,{pin},rocm-libraries",
        cwd=superproject,
    )
    _git("commit", "-qm", "pin", cwd=superproject)

    script = _load_script()
    checkout = superproject / "rocm-libraries"
    monkeypatch.setattr(script, "REPO_ROOT", superproject)
    monkeypatch.setattr(script, "ROCM_LIBRARIES_DIR", checkout)
    monkeypatch.setattr(script, "ROCM_LIBRARIES_URL", origin.as_uri())

    assert script.ensure_pinned_rocm_libraries() == pin
    assert (checkout / "CMakePresets.json").is_file()
    assert (checkout / "projects" / "hipdnn" / "CMakeLists.txt").is_file()

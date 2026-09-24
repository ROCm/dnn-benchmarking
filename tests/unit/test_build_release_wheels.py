# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for the requirements file tools/build_release_wheels.py publishes.

The wheels it builds have names nobody has registered on PyPI, and the file adds
PyPI as an extra index. Any of those names left as a bare requirement would let
a package published there under that name replace the released one.
"""

import hashlib
import importlib.util
import sys
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[2] / "tools" / "build_release_wheels.py"


def _load_script():
    """Import the script by path: it is a tool, not a package."""
    spec = importlib.util.spec_from_file_location("build_release_wheels", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _FakeBuildEnv:
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

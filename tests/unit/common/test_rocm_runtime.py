# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

import sys
from pathlib import Path
from types import ModuleType

import pytest

from dnn_benchmarking.common import rocm_runtime


class FakeRocmSdk(ModuleType):
    def __init__(self, paths: dict[str, Path]) -> None:
        super().__init__("rocm_sdk")
        self._paths = paths
        self.initialize_calls: list[dict[str, object]] = []

    def find_libraries(self, shortname: str):
        try:
            return [self._paths[shortname]]
        except KeyError as e:
            raise FileNotFoundError(shortname) from e

    def initialize_process(self, **kwargs) -> None:
        self.initialize_calls.append(kwargs)


@pytest.fixture(autouse=True)
def reset_rocm_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rocm_runtime, "_INITIALIZED_PIP_ROCM", False)
    monkeypatch.delenv("ROCM_PATH", raising=False)


def test_rocm_path_wins_for_default_plugin_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    sdk_prefix = tmp_path / "sdk"
    plugin_dir = sdk_prefix / "lib" / "hipdnn_plugins" / "engines"
    plugin_dir.mkdir(parents=True)
    fake_sdk = FakeRocmSdk({"hipdnn": sdk_prefix / "lib" / "libhipdnn_backend.so"})
    monkeypatch.setitem(sys.modules, "rocm_sdk", fake_sdk)
    monkeypatch.setenv("ROCM_PATH", "/custom/rocm")

    assert rocm_runtime.default_hipdnn_plugin_paths() == [
        Path("/custom/rocm/lib/hipdnn_plugins/engines")
    ]
    assert rocm_runtime.initialize_pip_rocm_runtime() is False
    assert fake_sdk.initialize_calls == []


def test_pip_rocm_plugin_path_discovered_from_hipdnn_library(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    sdk_prefix = (
        tmp_path / "venv" / "site-packages" / "_rocm_sdk_libraries_gfx94X"
    )
    lib_dir = sdk_prefix / "lib"
    plugin_dir = lib_dir / "hipdnn_plugins" / "engines"
    plugin_dir.mkdir(parents=True)
    hipdnn_library = lib_dir / "libhipdnn_backend.so"
    hipdnn_library.touch()
    monkeypatch.setitem(
        sys.modules,
        "rocm_sdk",
        FakeRocmSdk({"hipdnn": hipdnn_library}),
    )

    assert rocm_runtime.pip_rocm_plugin_path() == plugin_dir
    assert rocm_runtime.default_hipdnn_plugin_paths() == [plugin_dir]


def test_pip_rocm_initialize_preloads_available_libraries(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    lib_dir = tmp_path / "sdk" / "lib"
    lib_dir.mkdir(parents=True)
    fake_sdk = FakeRocmSdk(
        {
            "amdhip64": lib_dir / "libamdhip64.so",
            "miopen": lib_dir / "libMIOpen.so",
            "hipdnn": lib_dir / "libhipdnn_backend.so",
        }
    )
    monkeypatch.setitem(sys.modules, "rocm_sdk", fake_sdk)

    assert rocm_runtime.initialize_pip_rocm_runtime() is True

    assert fake_sdk.initialize_calls == [
        {
            "preload_shortnames": ["amdhip64", "miopen", "hipdnn"],
            "env_override": True,
        }
    ]


def test_pip_rocm_initialize_is_idempotent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    lib_dir = tmp_path / "sdk" / "lib"
    lib_dir.mkdir(parents=True)
    fake_sdk = FakeRocmSdk({"hipdnn": lib_dir / "libhipdnn_backend.so"})
    monkeypatch.setitem(sys.modules, "rocm_sdk", fake_sdk)

    assert rocm_runtime.initialize_pip_rocm_runtime() is True
    assert rocm_runtime.initialize_pip_rocm_runtime() is True

    assert len(fake_sdk.initialize_calls) == 1

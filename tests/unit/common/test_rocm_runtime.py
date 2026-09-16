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


class FakeHipdnnRuntime(ModuleType):
    """Stands in for an installed hipdnn-runtime-<arch> wheel."""

    def __init__(self, root: Path) -> None:
        super().__init__("hipdnn_runtime")
        self._root = root
        self.__gpu_arch__ = "gfx942"

    def library_dir(self) -> Path:
        return self._root / "lib"

    def backend_library(self) -> Path:
        return self.library_dir() / "libhipdnn_backend.so"

    def plugin_path(self) -> Path:
        return self.library_dir() / "hipdnn_plugins" / "engines"


def install_fake_runtime_wheel(
    monkeypatch: pytest.MonkeyPatch, root: Path
) -> list[str]:
    """Stage a runtime wheel payload on disk and record what gets dlopened."""
    plugin_dir = root / "lib" / "hipdnn_plugins" / "engines"
    plugin_dir.mkdir(parents=True)
    (root / "lib" / "libhipdnn_backend.so").touch()
    monkeypatch.setitem(sys.modules, "hipdnn_runtime", FakeHipdnnRuntime(root))

    loaded: list[str] = []
    monkeypatch.setattr(
        rocm_runtime.ctypes, "CDLL", lambda path, mode=0: loaded.append(path)
    )
    return loaded


@pytest.fixture(autouse=True)
def reset_rocm_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rocm_runtime, "_INITIALIZED_PIP_ROCM", False)
    monkeypatch.delenv("ROCM_PATH", raising=False)
    monkeypatch.delenv("HIPDNN_SDK", raising=False)
    # Keep the tests independent of whether a real runtime wheel is installed
    # in the environment running them.
    monkeypatch.setitem(sys.modules, "hipdnn_runtime", None)


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


def test_rocm_sdk_not_installed_disables_pip_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "rocm_sdk", None)

    assert rocm_runtime.pip_rocm_plugin_path() is None
    assert rocm_runtime.default_hipdnn_plugin_paths() is None
    assert rocm_runtime.initialize_pip_rocm_runtime() is False


def test_pip_rocm_plugin_path_discovered_from_hipdnn_library(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    sdk_prefix = tmp_path / "venv" / "site-packages" / "_rocm_sdk_libraries_gfx94X"
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


@pytest.mark.parametrize("error_type", [FileNotFoundError, ModuleNotFoundError])
def test_pip_rocm_plugin_path_ignores_find_libraries_failure(
    monkeypatch: pytest.MonkeyPatch,
    error_type: type[Exception],
) -> None:
    fake_sdk = FakeRocmSdk({})

    def fail_find_libraries(_shortname: str):
        raise error_type("hipdnn unavailable")

    monkeypatch.setattr(fake_sdk, "find_libraries", fail_find_libraries)
    monkeypatch.setitem(sys.modules, "rocm_sdk", fake_sdk)

    assert rocm_runtime.pip_rocm_plugin_path() is None
    assert rocm_runtime.default_hipdnn_plugin_paths() is None


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


def test_pip_rocm_initialize_skips_one_missing_preload_library(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    missing_shortname = "hiprtc"
    expected_shortnames = [
        shortname
        for shortname in rocm_runtime._ROCM_PRELOAD_ORDER
        if shortname != missing_shortname
    ]
    fake_sdk = FakeRocmSdk(
        {
            shortname: tmp_path / f"lib{shortname}.so"
            for shortname in expected_shortnames
        }
    )
    monkeypatch.setitem(sys.modules, "rocm_sdk", fake_sdk)

    assert rocm_runtime.initialize_pip_rocm_runtime() is True
    assert fake_sdk.initialize_calls == [
        {
            "preload_shortnames": expected_shortnames,
            "env_override": True,
        }
    ]


def test_pip_rocm_initialize_wraps_initialize_process_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    lib_dir = tmp_path / "sdk" / "lib"
    fake_sdk = FakeRocmSdk({"hipdnn": lib_dir / "libhipdnn_backend.so"})

    def fail_initialize_process(**_kwargs) -> None:
        raise OSError("dlopen failed")

    monkeypatch.setattr(fake_sdk, "initialize_process", fail_initialize_process)
    monkeypatch.setitem(sys.modules, "rocm_sdk", fake_sdk)

    with pytest.raises(
        RuntimeError,
        match="Failed to initialize pip-installed ROCm runtime: dlopen failed",
    ) as exc_info:
        rocm_runtime.initialize_pip_rocm_runtime()

    assert isinstance(exc_info.value.__cause__, OSError)


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


def test_runtime_wheel_plugin_path_beats_rocm_sdk(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    sdk_plugin_dir = tmp_path / "sdk" / "lib" / "hipdnn_plugins" / "engines"
    sdk_plugin_dir.mkdir(parents=True)
    sdk_library = tmp_path / "sdk" / "lib" / "libhipdnn_backend.so"
    sdk_library.touch()
    monkeypatch.setitem(sys.modules, "rocm_sdk", FakeRocmSdk({"hipdnn": sdk_library}))

    wheel_root = tmp_path / "hipdnn_runtime"
    install_fake_runtime_wheel(monkeypatch, wheel_root)

    assert rocm_runtime.default_hipdnn_plugin_paths() == [
        wheel_root / "lib" / "hipdnn_plugins" / "engines"
    ]


def test_runtime_wheel_backend_preloads_before_and_instead_of_sdk_hipdnn(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The wheel's backend must claim the shared SONAME.

    Both copies are called libhipdnn_backend.so, so if the SDK's older one is
    preloaded the bindings resolve against it and fail on any newer entry point.
    """
    lib_dir = tmp_path / "sdk" / "lib"
    lib_dir.mkdir(parents=True)
    fake_sdk = FakeRocmSdk(
        {
            "amdhip64": lib_dir / "libamdhip64.so",
            "hipdnn": lib_dir / "libhipdnn_backend.so",
            "miopen": lib_dir / "libMIOpen.so",
        }
    )
    monkeypatch.setitem(sys.modules, "rocm_sdk", fake_sdk)

    wheel_root = tmp_path / "hipdnn_runtime"
    loaded = install_fake_runtime_wheel(monkeypatch, wheel_root)

    assert rocm_runtime.initialize_pip_rocm_runtime() is True
    assert loaded == [str(wheel_root / "lib" / "libhipdnn_backend.so")]
    assert "hipdnn" not in fake_sdk.initialize_calls[0]["preload_shortnames"]
    assert "miopen" in fake_sdk.initialize_calls[0]["preload_shortnames"]


def test_sdk_hipdnn_still_preloads_without_a_runtime_wheel(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    lib_dir = tmp_path / "sdk" / "lib"
    lib_dir.mkdir(parents=True)
    fake_sdk = FakeRocmSdk({"hipdnn": lib_dir / "libhipdnn_backend.so"})
    monkeypatch.setitem(sys.modules, "rocm_sdk", fake_sdk)

    assert rocm_runtime.initialize_pip_rocm_runtime() is True
    assert fake_sdk.initialize_calls[0]["preload_shortnames"] == ["hipdnn"]


def test_separate_hipdnn_install_owns_plugins(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    install = tmp_path / "application"
    monkeypatch.setenv("HIPDNN_SDK", str(install))
    monkeypatch.setenv("ROCM_PATH", str(tmp_path / "dependencies"))
    install_fake_runtime_wheel(monkeypatch, tmp_path / "released-runtime")

    assert rocm_runtime.default_hipdnn_plugin_paths() == [
        install / "lib" / "hipdnn_plugins" / "engines"
    ]


def test_windows_hipdnn_install_keeps_plugins_next_to_the_dlls(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    install = tmp_path / "application"
    plugin_dir = install / "bin" / "hipdnn_plugins" / "engines"
    plugin_dir.mkdir(parents=True)
    monkeypatch.setattr(rocm_runtime.os, "name", "nt")
    monkeypatch.setenv("HIPDNN_SDK", str(install))

    assert rocm_runtime.default_hipdnn_plugin_paths() == [plugin_dir]


def test_missing_explicit_backend_does_not_fall_back_to_wheels(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HIPDNN_SDK", str(tmp_path / "missing-install"))
    install_fake_runtime_wheel(monkeypatch, tmp_path / "released-runtime")
    monkeypatch.setitem(sys.modules, "rocm_sdk", FakeRocmSdk({}))

    with pytest.raises(RuntimeError, match="HIPDNN_SDK contains no hipDNN backend"):
        rocm_runtime.initialize_pip_rocm_runtime()


def test_broken_explicit_backend_reports_loader_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    backend = tmp_path / "lib" / "libhipdnn_backend.so"
    if sys.platform == "win32":
        backend = tmp_path / "bin" / "hipdnn_backend.dll"
    backend.parent.mkdir()
    backend.write_bytes(b"not a native library")
    monkeypatch.setenv("HIPDNN_SDK", str(tmp_path))
    monkeypatch.setenv("ROCM_PATH", str(tmp_path / "dependencies"))
    monkeypatch.setitem(sys.modules, "rocm_sdk", None)

    with pytest.raises(
        RuntimeError, match="Failed to load HIPDNN_SDK backend"
    ) as error:
        rocm_runtime.initialize_pip_rocm_runtime()
    assert isinstance(error.value.__cause__, OSError)

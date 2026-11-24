# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for tools/bump_rocm_pin.py."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_QUARTZ_NIGHTLY = Path(__file__).resolve().parent / "data" / "quartz" / "nightly"


@pytest.fixture(scope="module")
def bump_mod():
    spec = importlib.util.spec_from_file_location(
        "bump_rocm_pin", _REPO_ROOT / "tools" / "bump_rocm_pin.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def setup_env(bump_mod):
    return bump_mod.load_setup_env(_REPO_ROOT)


def _write_pin(path: Path, archs: dict, default: str = "gfx950") -> Path:
    path.write_text(
        json.dumps({"default": default, "archs": archs}, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def _entry(date: str, therock: str, rl: str) -> dict:
    return {
        "rocm_version": f"10.3.0a{date}",
        "build_date": date,
        "therock_commit": therock,
        "rocm_libraries_commit": rl,
        "index_url": "https://nightly.repo.amd.com/rocm/whl-next/",
    }


def _serve_recorded_quartz(setup_env, monkeypatch, latest: str, overrides: dict):
    fetched: list[str] = []

    def fetch(url: str):
        fetched.append(url)
        if url.endswith("/latest.json"):
            return f"{latest}/status.json\n"
        day = url.removesuffix("/status.json").rsplit("/", 1)[-1]
        if day in overrides:
            return json.dumps(overrides[day])
        recorded = _QUARTZ_NIGHTLY / day / "status.json"
        return recorded.read_text(encoding="utf-8") if recorded.is_file() else None

    monkeypatch.setattr(setup_env, "fetch_quartz_text", fetch)
    return fetched


def test_bump_pins_advances_passing_archs_and_leaves_failing_untouched(
    bump_mod, setup_env, tmp_path, monkeypatch
) -> None:
    # Recorded 20261115: gfx950 passes, gfx942 fails rocm tests.
    # Recorded 20261116: rocm build is in_progress for all archs.
    fetched = _serve_recorded_quartz(setup_env, monkeypatch, "20261116", {})
    rl_calls: list[str] = []

    def fake_resolve(therock_commit: str, repo_url=None, *, required=True):
        rl_calls.append(therock_commit)
        return "f" * 40

    monkeypatch.setattr(
        setup_env, "resolve_therock_rocm_libraries_commit", fake_resolve
    )

    pin_path = _write_pin(
        tmp_path / "rocm-pin.json",
        {
            "gfx942": _entry("20261114", "1" * 40, "a" * 40),
            "gfx950": _entry("20261114", "1" * 40, "a" * 40),
        },
    )

    res = bump_mod.bump_pins(setup_env, pin_path)

    assert res.changed is True
    assert res.bumped_archs == ("gfx950",)
    assert res.title == "chore(deps): bump ROCm pin (gfx950 -> 10.3.0a20261115)"
    saved = json.loads(pin_path.read_text(encoding="utf-8"))
    # gfx942 had no passing nightly -> untouched
    assert saved["archs"]["gfx942"]["rocm_version"] == "10.3.0a20261114"
    assert saved["archs"]["gfx942"]["rocm_libraries_commit"] == "a" * 40
    # gfx950 bumped to 20261115
    assert saved["archs"]["gfx950"]["rocm_version"] == "10.3.0a20261115"
    assert saved["archs"]["gfx950"]["rocm_libraries_commit"] == "f" * 40
    assert len(rl_calls) == 1
    # Quartz URLs were memoized across the two arch loops (each URL fetched once).
    assert len(fetched) == len(set(fetched))


def test_bump_pins_never_downgrades_on_automatic_run(
    bump_mod, setup_env, tmp_path, monkeypatch
) -> None:
    _serve_recorded_quartz(setup_env, monkeypatch, "20261119", {})
    monkeypatch.setattr(
        setup_env,
        "resolve_therock_rocm_libraries_commit",
        lambda *_a, **_kw: (_ for _ in ()).throw(AssertionError("should not resolve")),
    )

    pin_path = _write_pin(
        tmp_path / "rocm-pin.json",
        {"gfx950": _entry("20261123", "2" * 40, "b" * 40)},
    )

    res = bump_mod.bump_pins(setup_env, pin_path)

    assert res.changed is False
    assert res.bumped_archs == ()
    assert res.title == ""
    assert any("newest passing 20261119 < pinned 20261123" in r for r in res.rows)


def test_bump_pins_dry_run_does_not_modify_file(
    bump_mod, setup_env, tmp_path, monkeypatch
) -> None:
    _serve_recorded_quartz(setup_env, monkeypatch, "20261119", {})
    monkeypatch.setattr(
        setup_env,
        "resolve_therock_rocm_libraries_commit",
        lambda *_a, **_kw: "e" * 40,
    )
    pin_path = _write_pin(
        tmp_path / "rocm-pin.json",
        {"gfx950": _entry("20261118", "1" * 40, "a" * 40)},
    )
    before = pin_path.read_text(encoding="utf-8")

    res = bump_mod.bump_pins(setup_env, pin_path, dry_run=True)

    assert res.changed is True
    assert res.bumped_archs == ("gfx950",)
    assert pin_path.read_text(encoding="utf-8") == before


def test_bump_pins_caches_therock_lookup_across_archs(
    bump_mod, setup_env, tmp_path, monkeypatch
) -> None:
    _serve_recorded_quartz(setup_env, monkeypatch, "20261119", {})
    rl_calls: list[str] = []

    def fake_resolve(therock_commit: str, repo_url=None, *, required=True):
        rl_calls.append(therock_commit)
        return "c" * 40

    monkeypatch.setattr(
        setup_env, "resolve_therock_rocm_libraries_commit", fake_resolve
    )
    pin_path = _write_pin(
        tmp_path / "rocm-pin.json",
        {
            "gfx942": _entry("20261118", "1" * 40, "a" * 40),
            "gfx950": _entry("20261118", "1" * 40, "a" * 40),
        },
    )

    res = bump_mod.bump_pins(setup_env, pin_path)

    assert res.changed is True
    assert res.bumped_archs == ("gfx942", "gfx950")
    assert res.title == "chore(deps): bump ROCm pin (gfx942, gfx950 -> 10.3.0a20261119)"
    # Both archs landed on 20261119 (same therock_commit) -> only 1 TheRock lookup.
    assert len(rl_calls) == 1

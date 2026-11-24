#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Advance per-arch ROCm + rocm-libraries pins in rocm-pin.json from Quartz.

Each architecture in rocm-pin.json is checked independently against Quartz
(walking back up to QUARTZ_LOOKBACK_DAYS days, never downgrading on automatic
runs). Architectures without a newer passing nightly stay untouched.

Used by both `.github/workflows/bump-rocm-pin.yml` (manual dispatch) and
scheduled automation.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent

# Per-arch Quartz gate: (primary_platform, extra_build_only_platforms).
# Linux datacenter archs gate on Linux build+tests; gfx1151 gates on Windows
# build+tests plus Linux wheel builds. Unlisted archs default to ("linux", ()).
ARCH_GATES: Dict[str, Tuple[str, Tuple[Tuple[str, str], ...]]] = {
    "gfx90a": ("linux", ()),
    "gfx942": ("linux", ()),
    "gfx950": ("linux", ()),
    "gfx1151": ("windows", (("linux", "gfx1151"),)),
}


@dataclasses.dataclass(frozen=True)
class BumpResult:
    changed: bool
    title: str
    body: str
    bumped_archs: Tuple[str, ...]
    rows: Tuple[str, ...]
    payload: Dict[str, object]


def load_setup_env(repo_root: Path = REPO_ROOT) -> ModuleType:
    setup_path = repo_root / "setup_env.py"
    spec = importlib.util.spec_from_file_location("setup_env", setup_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {setup_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def bump_pins(
    setup_env: ModuleType,
    pin_path: Path,
    *,
    requested_version: str = "",
    requested_archs: Sequence[str] = (),
    dry_run: bool = False,
) -> BumpResult:
    default_arch, current_pins = setup_env.read_rocm_pins(pin_path)

    requested = requested_version.strip()
    if requested:
        requested = setup_env._rocm_version(requested)
    is_manual = bool(requested and requested != "latest")

    if requested_archs:
        target_archs = [a.strip().lower() for a in requested_archs if a.strip()]
        unknown = [a for a in target_archs if a not in current_pins]
        if unknown:
            setup_env.fail(
                f"Unknown arch(s) in --archs: {', '.join(unknown)}.",
                f"Expected a subset of: {', '.join(current_pins)}.",
            )
    else:
        target_archs = list(current_pins)

    # Memoize Quartz HTTP fetches across arch checks without leaking global state.
    orig_fetch = setup_env.fetch_quartz_text
    fetch_cache: Dict[str, Optional[str]] = {}

    def cached_fetch(url: str) -> Optional[str]:
        if url not in fetch_cache:
            fetch_cache[url] = orig_fetch(url)
        return fetch_cache[url]

    therock_cache: Dict[str, str] = {}

    def resolve_rl(therock_commit: str) -> str:
        if therock_commit not in therock_cache:
            therock_cache[therock_commit] = (
                setup_env.resolve_therock_rocm_libraries_commit(therock_commit)
            )
        return therock_cache[therock_commit]

    new_archs_data: Dict[str, Dict[str, Optional[str]]] = {}
    bumped_archs: List[str] = []
    bumped_versions: set[str] = set()
    rows: List[str] = []

    setup_env.fetch_quartz_text = cached_fetch
    try:
        for arch, current in current_pins.items():
            current_entry: Dict[str, Optional[str]] = {
                "rocm_version": current.rocm_version,
                "build_date": current.build_date,
                "therock_commit": current.therock_commit,
                "rocm_libraries_commit": current.rocm_libraries_commit,
                "index_url": current.index_url,
            }
            cur_rl_short = (current.rocm_libraries_commit or "")[:12]

            if arch not in target_archs:
                new_archs_data[arch] = current_entry
                rows.append(
                    f"| `{arch}` | skipped | `{current.rocm_version}` | "
                    f"`{cur_rl_short}` |"
                )
                continue

            plat, extra_builds = ARCH_GATES.get(arch, ("linux", ()))
            if is_manual:
                build = setup_env.describe_rocm_version(requested, "", plat, arch)
            else:
                build = setup_env.find_quartz_nightly(
                    plat, arch, require_platform_builds=extra_builds
                )

            if build is None:
                new_archs_data[arch] = current_entry
                rows.append(
                    f"| `{arch}` | unchanged "
                    f"(no passing nightly in {setup_env.QUARTZ_LOOKBACK_DAYS}d) | "
                    f"`{current.rocm_version}` | `{cur_rl_short}` |"
                )
                continue

            if not build.therock_commit:
                if is_manual:
                    setup_env.fail(
                        f"Quartz has no therock_commit for ROCm {build.rocm_version}."
                    )
                new_archs_data[arch] = current_entry
                rows.append(
                    f"| `{arch}` | unchanged (no therock_commit) | "
                    f"`{current.rocm_version}` | `{cur_rl_short}` |"
                )
                continue

            current_date = current.build_date or ""
            new_date = build.build_date or ""
            # Scheduled/latest bumps only move forward; explicit version may pin any build.
            if not is_manual and current_date and new_date and new_date < current_date:
                new_archs_data[arch] = current_entry
                rows.append(
                    f"| `{arch}` | unchanged "
                    f"(newest passing {new_date} < pinned {current_date}) | "
                    f"`{current.rocm_version}` | `{cur_rl_short}` |"
                )
                continue

            rl_commit = resolve_rl(build.therock_commit)
            entry_changed = (
                build.rocm_version != current.rocm_version
                or rl_commit != current.rocm_libraries_commit
                or build.therock_commit != current.therock_commit
            )
            if entry_changed:
                new_archs_data[arch] = {
                    "rocm_version": build.rocm_version,
                    "build_date": build.build_date,
                    "therock_commit": build.therock_commit,
                    "rocm_libraries_commit": rl_commit,
                    "index_url": build.index_url,
                }
                bumped_archs.append(arch)
                bumped_versions.add(build.rocm_version)
                rows.append(
                    f"| `{arch}` | **bumped** | "
                    f"`{current.rocm_version}` -> `{build.rocm_version}` | "
                    f"`{cur_rl_short}` -> `{rl_commit[:12]}` |"
                )
            else:
                new_archs_data[arch] = current_entry
                rows.append(
                    f"| `{arch}` | unchanged | `{current.rocm_version}` | "
                    f"`{cur_rl_short}` |"
                )
    finally:
        setup_env.fetch_quartz_text = orig_fetch

    payload: Dict[str, object] = {"default": default_arch, "archs": new_archs_data}
    changed = bool(bumped_archs)
    if changed:
        if not dry_run:
            pin_path.write_text(
                json.dumps(payload, indent=2) + "\n",
                encoding="utf-8",
            )
        if len(bumped_versions) == 1:
            ver = next(iter(bumped_versions))
            title = f"chore(deps): bump ROCm pin ({', '.join(bumped_archs)} -> {ver})"
        else:
            title = f"chore(deps): bump ROCm pin ({', '.join(bumped_archs)})"
        body = (
            "\n".join(
                [
                    "Bump per-arch entries in `rocm-pin.json` to the newest passing "
                    "Quartz nightly (archs without a newer passing nightly stay untouched):",
                    "",
                    "| Arch | Status | ROCm / PyTorch | rocm-libraries |",
                    "|---|---|---|---|",
                    *rows,
                ]
            )
            + "\n"
        )
    else:
        title = ""
        body = ""

    return BumpResult(
        changed=changed,
        title=title,
        body=body,
        bumped_archs=tuple(bumped_archs),
        rows=tuple(rows),
        payload=payload,
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Update per-arch ROCm + rocm-libraries pins in rocm-pin.json."
    )
    parser.add_argument(
        "--pin-file",
        type=Path,
        default=REPO_ROOT / "rocm-pin.json",
        help="Path to rocm-pin.json (default: repo root rocm-pin.json).",
    )
    parser.add_argument(
        "--rocm-version",
        default="",
        help=(
            "Optional ROCm nightly version (e.g. 10.3.0a20261123); blank or "
            "'latest' advances each arch independently from Quartz."
        ),
    )
    parser.add_argument(
        "--archs",
        default="",
        help="Optional comma-separated arch keys to update (default: all).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve and print changes without modifying rocm-pin.json.",
    )
    parser.add_argument(
        "--body-file",
        type=Path,
        default=None,
        help="Write the Markdown PR body here when changes are found.",
    )
    parser.add_argument(
        "--github-output",
        type=Path,
        default=None,
        help="Append changed and title outputs in GitHub Actions format.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Write structured JSON result (changed, title, bumped_archs, rows).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    setup_env = load_setup_env()
    requested_archs = [a for a in args.archs.split(",") if a.strip()]
    result = bump_pins(
        setup_env,
        args.pin_file,
        requested_version=args.rocm_version,
        requested_archs=requested_archs,
        dry_run=args.dry_run,
    )
    if result.changed and args.body_file is not None:
        args.body_file.write_text(result.body, encoding="utf-8")
    if args.github_output is not None:
        with args.github_output.open("a", encoding="utf-8") as handle:
            handle.write(f"changed={'true' if result.changed else 'false'}\n")
            handle.write(f"title={result.title}\n")
    if args.json_out is not None:
        args.json_out.write_text(
            json.dumps(
                {
                    "changed": result.changed,
                    "title": result.title,
                    "bumped_archs": list(result.bumped_archs),
                    "rows": list(result.rows),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    print("\n".join(result.rows))
    print(f"changed={result.changed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""TOML config file support for the dnn-benchmark CLI."""

from __future__ import annotations

import argparse
import tomllib
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

from .parser import CONFIG_OPTIONS, OPTION_DEFAULTS, CliOption, ConfigKind, parse_engine_id

_ALLOWED_TOP_LEVEL_KEYS: Set[str] = {
    str(option.config_key) for option in CONFIG_OPTIONS
} | {"version", "engines"}

_ALLOWED_ENGINE_KEYS: Set[str] = {"id", "plugin_path"}


def apply_config_file(args: argparse.Namespace) -> None:
    """Merge ``args.config`` into parsed CLI args without overriding CLI values.

    ``args`` must come from ``create_parser(suppress_defaults=True)`` so that
    ``vars(args)`` holds only explicitly supplied options. Precedence is
    ``defaults < config < explicit CLI``. The config ``[[engines]]`` matrix
    (engine IDs plus any per-engine plugin paths) is dropped as a unit when
    ``--engine`` or ``--plugin-path`` is given; a top-level ``plugin_path``
    is kept unless ``--plugin-path`` overrides it.
    """
    overrides: Dict[str, Any] = {}
    if getattr(args, "config", None) is not None:
        path = Path(args.config)
        overrides, matrix = _normalise_config(_load_toml(path), path)
        if "engine" not in args and "plugin_path" not in args:
            overrides.update(matrix)

    merged = dict(OPTION_DEFAULTS)
    merged.update(overrides)
    merged.update(vars(args))
    vars(args).clear()
    vars(args).update(merged)


def _load_toml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise ValueError(f"Config file not found: {path}")
    try:
        with path.open("rb") as f:
            return tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        raise ValueError(f"Invalid TOML config {path}: {e}") from e


def _normalise_config(
    raw: Dict[str, Any], path: Path
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Return ``(options, engine_matrix)`` keyed by argparse dest."""
    version = raw.get("version", 1)
    if type(version) is not int or version != 1:
        raise ValueError(f"Unsupported config version in {path}: {version!r}")
    _reject_unknown_keys(raw.keys(), _ALLOWED_TOP_LEVEL_KEYS, "config")

    base_dir = path.parent
    out = {
        option.dest: _convert(raw[option.config_key], option, base_dir)
        for option in CONFIG_OPTIONS
        if option.config_key in raw
    }
    matrix = _normalise_engines(raw.get("engines"), base_dir)
    if "plugin_path" in out and "plugin_path" in matrix:
        raise ValueError(
            "Config cannot set both top-level plugin_path and engine plugin_path"
        )
    return out, matrix


def _reject_unknown_keys(keys: Iterable[str], allowed: Set[str], context: str) -> None:
    unknown = sorted(set(keys) - allowed)
    if unknown:
        label = "fields" if len(unknown) > 1 else "field"
        raise ValueError(f"Unknown {context} {label}: {', '.join(unknown)}")


def _convert(value: Any, option: CliOption, base_dir: Path) -> Any:
    """Validate one config value and convert it to the argparse dest value."""
    key = option.config_key
    match option.config_kind:
        case ConfigKind.SCALAR | ConfigKind.CHOICE:
            if value is None and option.config_optional:
                return None
            typ = option.config_type
            if not _matches_type(value, typ):
                raise ValueError(f"Config field '{key}' must be {typ.__name__}")
            if option.choices is not None and value not in option.choices:
                raise ValueError(
                    f"Config field '{key}' must be one of: {', '.join(option.choices)}"
                )
            if option.parser_type is None or option.parser_type is typ:
                return value
            try:
                return option.parser_type(value)
            except argparse.ArgumentTypeError as e:
                raise ValueError(f"Config field '{key}' {e}") from e
        case ConfigKind.PATH:
            if not isinstance(value, str) or not value:
                raise ValueError(f"Config field '{key}' must be string path")
            return _path_from_config(base_dir, value)
        case ConfigKind.PATH_LIST:
            if not _is_path_list(value):
                raise ValueError(
                    f"Config field '{key}' must be a non-empty list of string paths"
                )
            return [str(_path_from_config(base_dir, item)) for item in value]
        case ConfigKind.PATH_OR_PATH_LIST:
            if isinstance(value, str) and value:
                value = [value]
            if not _is_path_list(value):
                raise ValueError(
                    f"Config field '{key}' must be a string or non-empty list of strings"
                )
            return [_path_from_config(base_dir, item) for item in value]
    raise AssertionError(f"unhandled config kind {option.config_kind}")


def _is_path_list(value: Any) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(item, str) and item for item in value)
    )


def _matches_type(value: Any, typ: Any) -> bool:
    if typ is float:
        return type(value) in {int, float}
    if typ in {int, bool}:
        return type(value) is typ
    return isinstance(value, typ)


def _path_from_config(base_dir: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base_dir / path


def _normalise_engines(engines: Any, base_dir: Path) -> Dict[str, Any]:
    """Convert the ``[[engines]]`` table to ``engine`` (+ ``plugin_path``) dests."""
    if engines is None:
        return {}
    if not isinstance(engines, list) or not engines:
        raise ValueError("Config field 'engines' must be a non-empty array of tables")

    ids: List[int] = []
    plugin_paths: List[Optional[Path]] = []
    for index, engine in enumerate(engines):
        if not isinstance(engine, dict):
            raise ValueError("Each config engine entry must be a table")
        _reject_unknown_keys(
            engine.keys(), _ALLOWED_ENGINE_KEYS, f"config engine {index}"
        )
        engine_id = engine.get("id")
        if type(engine_id) not in {int, str}:
            raise ValueError(
                f"Config engine {index} must include id (engine name or integer)"
            )
        try:
            ids.append(parse_engine_id(str(engine_id)))
        except argparse.ArgumentTypeError as e:
            raise ValueError(f"Config engine {index}: {e}") from e

        plugin_path = engine.get("plugin_path")
        if plugin_path is not None and (not isinstance(plugin_path, str) or not plugin_path):
            raise ValueError(f"Config engine {index} plugin_path must be a string")
        plugin_paths.append(
            None if plugin_path is None else _path_from_config(base_dir, plugin_path)
        )

    matrix: Dict[str, Any] = {"engine": ids}
    if any(p is not None for p in plugin_paths):
        if any(p is None for p in plugin_paths):
            raise ValueError(
                "Every config engine must set plugin_path when any engine does"
            )
        matrix["plugin_path"] = plugin_paths
    return matrix

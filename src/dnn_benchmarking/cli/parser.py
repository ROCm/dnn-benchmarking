# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""CLI argument parsing for dnn-benchmarking."""

import argparse
import importlib.metadata
import math
import re
from dataclasses import dataclass, fields
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ..config.benchmark_config import (
    CACHE_MODE_CHOICES,
    RuntimeName,
    MetricsConfig,
    OracleMode,
    PMC_SET_CHOICES,
    PyTorchSdpaBackendName,
    ReferenceProviderName,
    SuiteConfig,
    ValidationConfig,
)


class ConfigKind(str, Enum):
    """Config-file value normalization strategies."""

    SCALAR = "scalar"
    CHOICE = "choice"
    PATH = "path"
    PATH_LIST = "path_list"
    PATH_OR_PATH_LIST = "path_or_path_list"


@dataclass(frozen=True)
class CliOption:
    """Single source of truth for one public CLI option."""

    flags: tuple[str, ...]
    help: str
    dest: str
    group: str
    default: Any = None
    parser_type: Optional[Callable[[Any], Any]] = None
    action: Any = None
    nargs: Any = None
    metavar: Optional[str] = None
    choices: Optional[tuple[str, ...]] = None
    config_key: Optional[str] = None
    config_kind: Optional[ConfigKind] = None
    config_type: Optional[type] = None
    config_optional: bool = False

    def __post_init__(self) -> None:
        if (self.config_key is None) != (self.config_kind is None):
            raise ValueError(
                f"{self.dest}: config_key and config_kind must be set together"
            )
        if self.config_kind in {ConfigKind.SCALAR, ConfigKind.CHOICE}:
            if self.config_type is None:
                raise ValueError(
                    f"{self.dest}: {self.config_kind.value} requires config_type"
                )
        if self.config_kind is ConfigKind.CHOICE and self.choices is None:
            raise ValueError(f"{self.dest}: choice config fields require choices")

    @property
    def is_configurable(self) -> bool:
        return self.config_key is not None


def _default(cls: type, name: str) -> Any:
    """Return a dataclass field default (enum members as their string value)."""
    value = next(f for f in fields(cls) if f.name == name).default
    return value.value if isinstance(value, Enum) else value


def _values(enum_cls: type[Enum]) -> tuple[str, ...]:
    return tuple(member.value for member in enum_cls)


def _at_least(kind: type, minimum: float) -> Callable[[Any], Any]:
    """argparse ``type=`` converter that rejects values below ``minimum``.

    NaN and inf are rejected too: ``diff > nan`` is never true, so a NaN
    tolerance would pass every comparison.
    """

    def convert(text: Any) -> Any:
        try:
            value = kind(text)
        except (TypeError, ValueError):
            raise argparse.ArgumentTypeError(
                f"expected {kind.__name__}, got {text!r}"
            ) from None
        if not (math.isfinite(value) and value >= minimum):
            raise argparse.ArgumentTypeError(f"must be >= {minimum}, got {value}")
        return value

    return convert


_UINT64 = 1 << 64
_INT64_MIN = -(1 << 63)


def _fnv1a64(name: str) -> int:
    """64-bit FNV-1a hash; hipDNN derives engine IDs from engine names this way."""
    value = 0xCBF29CE484222325
    for byte in name.encode():
        value = ((value ^ byte) * 0x100000001B3) % _UINT64
    return value


def _signed64(value: int, token: str) -> int:
    """Wrap an unsigned 64-bit engine ID to the signed int64 hipDNN uses."""
    if not _INT64_MIN <= value < _UINT64:
        raise argparse.ArgumentTypeError(f"engine ID {token!r} is outside 64 bits")
    return value - _UINT64 if value >= 1 << 63 else value


#: Engine names parse_engine_id hashed, by ID, so an error about an engine no
#: plugin provides can show the name the user typed.
TYPED_ENGINE_NAMES: Dict[int, str] = {}


def parse_engine_id(token: str) -> int:
    """Parse one engine as a decimal ID, a 0x hex ID, or an engine name.

    Any other token is a name, as in hipDNN's engineNameOrIdToId (names such
    as ``hipkernel:ConvFwd`` contain ``:``). Names resolve to FNV-1a-64 of the
    exact (case-sensitive) name. Hex and names above 2**63 wrap to signed
    int64, matching hipDNN's engine IDs.
    """
    text = token.strip()
    if re.fullmatch(r"[+-]?\d+", text):
        return _signed64(int(text), token)
    if re.fullmatch(r"0[xX][0-9a-fA-F]+", text):
        return _signed64(int(text, 16), token)
    engine_id = _signed64(_fnv1a64(text), token)
    TYPED_ENGINE_NAMES[engine_id] = text
    return engine_id


def _parse_engine_list(s: str) -> List[int]:
    """Parse ``--engine`` as an ordered comma list; duplicates are kept."""
    parts = [p for p in (p.strip() for p in s.split(",")) if p]
    if not parts:
        raise argparse.ArgumentTypeError("requires at least one engine")
    return [parse_engine_id(p) for p in parts]


def _parse_plugin_path_list(s: str) -> List[Path]:
    """Parse --plugin-path as a comma-separated list of plugin directories."""
    parts = [p for p in (p.strip() for p in s.split(",")) if p]
    if not parts:
        raise argparse.ArgumentTypeError("requires at least one path")
    return [Path(p) for p in parts]


def _bool_option(
    flags: tuple[str, ...], dest: str, group: str, cls: type, field: str, help: str
) -> CliOption:
    return CliOption(
        flags=flags,
        dest=dest,
        group=group,
        action=argparse.BooleanOptionalAction,
        default=_default(cls, field),
        help=help,
        config_key=dest,
        config_kind=ConfigKind.SCALAR,
        config_type=bool,
    )


def _choice_option(
    flags: tuple[str, ...],
    dest: str,
    group: str,
    choices: tuple[str, ...],
    default: Any,
    help: str,
    *,
    optional: bool = False,
) -> CliOption:
    return CliOption(
        flags=flags,
        dest=dest,
        group=group,
        parser_type=str,
        choices=choices,
        default=default,
        help=help,
        config_key=dest,
        config_kind=ConfigKind.CHOICE,
        config_type=str,
        config_optional=optional,
    )


CLI_OPTIONS: tuple[CliOption, ...] = (
    # Input
    CliOption(
        flags=("--graph", "-g"),
        dest="graph",
        group="Input",
        nargs="+",
        metavar="PATH",
        help="graph JSON files, directories, globs, or tarballs "
        "(.tar, .tar.gz, .tgz, .tar.bz2, .tar.xz)",
        config_key="graphs",
        config_kind=ConfigKind.PATH_LIST,
    ),
    CliOption(
        flags=("--config",),
        dest="config",
        group="Input",
        parser_type=Path,
        metavar="PATH",
        help="TOML recipe; explicit CLI flags override its values",
    ),
    # Run
    CliOption(
        flags=("--warmup", "-w"),
        dest="warmup",
        group="Run",
        parser_type=_at_least(int, 0),
        default=_default(SuiteConfig, "warmup_iters"),
        metavar="N",
        help="untimed warmup launches per engine",
        config_key="warmup",
        config_kind=ConfigKind.SCALAR,
        config_type=int,
    ),
    CliOption(
        flags=("--iters", "-i"),
        dest="iters",
        group="Run",
        parser_type=_at_least(int, 1),
        default=_default(SuiteConfig, "benchmark_iters"),
        metavar="N",
        help="minimum timed iterations per engine",
        config_key="iters",
        config_kind=ConfigKind.SCALAR,
        config_type=int,
    ),
    CliOption(
        flags=("--min-time-ms",),
        dest="min_time_ms",
        group="Run",
        parser_type=_at_least(float, 0),
        default=_default(SuiteConfig, "min_time_ms"),
        metavar="MS",
        help="keep timing until summed kernel time reaches MS (0 = exactly --iters)",
        config_key="min_time_ms",
        config_kind=ConfigKind.SCALAR,
        config_type=float,
    ),
    _choice_option(
        ("--cache-mode",),
        "cache_mode",
        "Run",
        CACHE_MODE_CHOICES,
        _default(SuiteConfig, "cache_mode"),
        "GPU L2/MALL: warm keeps them; cold flushes them before each timed iteration",
    ),
    CliOption(
        flags=("--timing-block",),
        dest="timing_block",
        group="Run",
        parser_type=_at_least(int, 1),
        default=_default(SuiteConfig, "timing_block"),
        metavar="N",
        help="time N back-to-back launches per sample (rocKE block timing; 1 = per launch)",
        config_key="timing_block",
        config_kind=ConfigKind.SCALAR,
        config_type=int,
    ),
    CliOption(
        flags=("--seed", "-s"),
        dest="seed",
        group="Run",
        parser_type=int,
        default=_default(SuiteConfig, "seed"),
        metavar="SEED",
        help="random seed for input data",
        config_key="seed",
        config_kind=ConfigKind.SCALAR,
        config_type=int,
    ),
    # Runtime/Selection
    _choice_option(
        ("--runtime", "-r"),
        "runtime",
        "Runtime/Selection",
        _values(RuntimeName),
        _default(SuiteConfig, "runtime"),
        "hipdnn runs engine plugins; pytorch runs the graph through PyTorch",
    ),
    CliOption(
        flags=("--engine", "-e"),
        dest="engine",
        group="Runtime/Selection",
        parser_type=_parse_engine_list,
        default=_default(SuiteConfig, "engine_filter"),
        metavar="ENGINES",
        help="comma list of engine names, decimal IDs or 0x hex IDs, run in order "
        "(default: all discovered)",
    ),
    CliOption(
        flags=("--plugin-path",),
        dest="plugin_path",
        group="Runtime/Selection",
        parser_type=_parse_plugin_path_list,
        metavar="PATHS",
        help="plugin dir, or comma list matching --engine order "
        "(default: $ROCM_PATH/lib/hipdnn_plugins/engines, else the pip ROCm SDK)",
        config_key="plugin_path",
        config_kind=ConfigKind.PATH_OR_PATH_LIST,
    ),
    _bool_option(
        ("--autotune",),
        "autotune",
        "Runtime/Selection",
        SuiteConfig,
        "autotune",
        "benchmark candidate kernels on first execute and cache the winner "
        "(HIPDNN_FORCE_BENCHMARKING=1); pair with --hipdnn-cache-dir",
    ),
    CliOption(
        flags=("--hipdnn-cache-dir",),
        dest="hipdnn_cache_dir",
        group="Runtime/Selection",
        parser_type=Path,
        metavar="PATH",
        help="per-run HIPDNN_CACHE_DIR so tuned winners do not leak between runs",
        config_key="hipdnn_cache_dir",
        config_kind=ConfigKind.PATH,
    ),
    _choice_option(
        ("--pytorch-sdpa-backend",),
        "pytorch_sdpa_backend",
        "Runtime/Selection",
        _values(PyTorchSdpaBackendName),
        _default(SuiteConfig, "pytorch_sdpa_backend"),
        "PyTorch SDPA category; non-default categories are strict (no fallback)",
    ),
    CliOption(
        flags=("--pytorch-rocm-fa-library",),
        dest="pytorch_rocm_fa_library",
        group="Runtime/Selection",
        parser_type=str,
        default=_default(SuiteConfig, "pytorch_rocm_fa_library"),
        metavar="LIBRARY",
        help="ROCm Flash Attention implementation passed to PyTorch "
        "(e.g. aotriton); requires --pytorch-sdpa-backend flash",
        config_key="pytorch_rocm_fa_library",
        config_kind=ConfigKind.SCALAR,
        config_type=str,
    ),
    # Validation
    _choice_option(
        ("--validate",),
        "validate",
        "Validation",
        _values(ReferenceProviderName),
        _default(ValidationConfig, "provider"),
        "reference runtime for correctness checks; pytorch adds a timed "
        "reference row",
    ),
    CliOption(
        flags=("--rtol",),
        dest="rtol",
        group="Validation",
        parser_type=_at_least(float, 0),
        default=_default(ValidationConfig, "rtol"),
        metavar="TOL",
        help="relative tolerance (default: dtype-aware; alone it sets both)",
        config_key="rtol",
        config_kind=ConfigKind.SCALAR,
        config_type=float,
        config_optional=True,
    ),
    CliOption(
        flags=("--atol",),
        dest="atol",
        group="Validation",
        parser_type=_at_least(float, 0),
        default=_default(ValidationConfig, "atol"),
        metavar="TOL",
        help="absolute tolerance (default: dtype-aware; alone it sets both)",
        config_key="atol",
        config_kind=ConfigKind.SCALAR,
        config_type=float,
        config_optional=True,
    ),
    # Comparison
    _choice_option(
        ("--oracle-mode",),
        "oracle_mode",
        "Comparison",
        _values(OracleMode),
        _default(SuiteConfig, "oracle_mode"),
        "exhaustive: also build and time a global.benchmarking=1 plan per "
        "engine, and a tuned PyTorch run in a child process (much slower)",
    ),
    # Output
    CliOption(
        flags=("--output", "-o"),
        dest="output",
        group="Output",
        parser_type=Path,
        metavar="PATH",
        help="write results to PATH (CSV when PATH ends in .csv, else JSON)",
        config_key="output",
        config_kind=ConfigKind.PATH,
    ),
    _bool_option(
        ("--compact-json",),
        "compact_json",
        "Output",
        SuiteConfig,
        "compact_json",
        "write the JSON result without whitespace (default: one-space indent)",
    ),
    _bool_option(
        ("-v", "--verbose"),
        "verbose",
        "Output",
        SuiteConfig,
        "verbose",
        "add a per-engine detail block under each graph table",
    ),
    _bool_option(
        ("-q", "--quiet"),
        "quiet",
        "Output",
        SuiteConfig,
        "quiet",
        "suppress progress lines; tables and the summary still print",
    ),
    _bool_option(
        ("--metrics",),
        "metrics",
        "Output",
        MetricsConfig,
        "basic",
        "FLOPs/IO, workspace, host and GPU snapshots at no timing cost "
        "(default: on)",
    ),
    # Profiling
    _choice_option(
        ("--pmc",),
        "pmc",
        "Profiling",
        PMC_SET_CHOICES,
        _default(MetricsConfig, "pmc_set"),
        "re-run under rocprofv3 with this counter set (~30%% extra wall time)",
        optional=True,
    ),
    _bool_option(
        ("--pmc-allow-multipass",),
        "pmc_allow_multipass",
        "Profiling",
        MetricsConfig,
        "pmc_allow_multipass",
        "allow --pmc all (multi-pass replay; can hang for minutes)",
    ),
    _bool_option(
        ("--trace",),
        "trace",
        "Profiling",
        MetricsConfig,
        "trace",
        "re-run under rocprofv3 and write a Perfetto kernel/memcpy trace",
    ),
    _bool_option(
        ("--perf",),
        "perf",
        "Profiling",
        MetricsConfig,
        "perf",
        "re-run under 'perf stat' for CPU cycles, instructions and IPC",
    ),
    _bool_option(
        ("--roofline",),
        "roofline",
        "Profiling",
        MetricsConfig,
        "roofline",
        "re-run under 'rocprof-compute --roof-only' for HBM/compute ceilings "
        "(~3 extra runs)",
    ),
    CliOption(
        flags=("--profiling-output-dir",),
        dest="profiling_output_dir",
        group="Profiling",
        parser_type=Path,
        default=_default(MetricsConfig, "profiling_output_dir"),
        metavar="DIR",
        help="profiling artefact root (default: ./profiling-output/<utc-timestamp>/)",
        config_key="profiling_output_dir",
        config_kind=ConfigKind.PATH,
    ),
    CliOption(
        flags=("--profiling-timeout",),
        dest="profiling_timeout",
        group="Profiling",
        parser_type=_at_least(int, 0),
        default=_default(MetricsConfig, "profiling_timeout_s"),
        metavar="SECONDS",
        help="timeout per profiler subprocess; 0 disables",
        config_key="profiling_timeout",
        config_kind=ConfigKind.SCALAR,
        config_type=int,
    ),
)

CONFIG_OPTIONS: tuple[CliOption, ...] = tuple(
    option for option in CLI_OPTIONS if option.is_configurable
)

OPTION_DEFAULTS: dict[str, Any] = {
    option.dest: option.default for option in CLI_OPTIONS
}

_EPILOG = """\
examples:
  dnn-benchmark -g graphs/sample_conv_fwd.json
  dnn-benchmark -g graphs/sample_conv_fwd.json -w 20 -i 200 --cache-mode cold
  dnn-benchmark -g graphs/sample_conv_fwd.json -e MIOPEN_ENGINE -v
  dnn-benchmark -g graphs/sample_conv_fwd.json -e MIOPEN_ENGINE,MIOPEN_ENGINE \\
      --plugin-path /path/to/build_a/engines,/path/to/build_b/engines
  dnn-benchmark -g 'graphs/*.json' --validate pytorch -o results.json
  dnn-benchmark -g graphs/sample_sdpa.json --runtime pytorch --pytorch-sdpa-backend flash
  dnn-benchmark --config sample_configs/basic.toml.example -g graphs/sample_conv_fwd.json

engines:
  hipdnn_list_engines --plugin-dir <plugin dir> lists engine names and IDs.

compare two result files:
  dnn-benchmark compare --help

exit codes:
  0 ok; 1 error (error row, graph error or write failure); 2 usage or config error;
  3 correctness mismatch; 130/143 interrupted by SIGINT/SIGTERM
"""


def _version() -> str:
    try:
        return importlib.metadata.version("dnn-benchmarking")
    except importlib.metadata.PackageNotFoundError:
        return "0+unknown"


def _add_cli_option(
    groups: dict[str, Any], option: CliOption, *, suppress_defaults: bool
) -> None:
    help_text = option.help
    if option.default is not None and not isinstance(option.default, bool):
        help_text += f" (default: {option.default})"
    kwargs: dict[str, Any] = {
        "dest": option.dest,
        "default": argparse.SUPPRESS if suppress_defaults else option.default,
        "help": help_text,
    }
    for name in ("action", "nargs", "metavar", "choices"):
        value = getattr(option, name)
        if value is not None:
            kwargs[name] = value
    if option.parser_type is not None:
        kwargs["type"] = option.parser_type
    groups[option.group].add_argument(*option.flags, **kwargs)


def create_parser(*, suppress_defaults: bool = False) -> argparse.ArgumentParser:
    """Create the argument parser for dnn-benchmark CLI.

    Args:
        suppress_defaults: When True, absent public options are omitted from
            the parsed namespace. The CLI entry point uses this so config-file
            values can be merged as ``defaults < config < explicit CLI``.
    """
    parser = argparse.ArgumentParser(
        prog="dnn-benchmark",
        usage="%(prog)s -g PATH [PATH ...] [options]\n"
        "       %(prog)s --config FILE [options]\n"
        "       %(prog)s compare ...",
        description="Benchmark and validate hipDNN graphs "
        "(early development; not for build workflows or CI gating).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_EPILOG,
        allow_abbrev=False,
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {_version()}")
    # Help groups appear in the order CLI_OPTIONS first uses them.
    groups = {
        name: parser.add_argument_group(name)
        for name in dict.fromkeys(o.group for o in CLI_OPTIONS)
    }
    for option in CLI_OPTIONS:
        _add_cli_option(groups, option, suppress_defaults=suppress_defaults)

    # Hidden re-exec mode used by the profiling orchestrator:
    #   --internal-profiling-run --graph G --engine E --warmup W --iters I --seed S
    parser.add_argument(
        "--internal-profiling-run",
        action="store_true",
        default=False,
        help=argparse.SUPPRESS,
    )
    # Hidden child mode for the tuned PyTorch run: the parent re-execs the CLI
    # with --runtime pytorch so tuning state stays out of its own process.
    parser.add_argument(
        "--internal-pytorch-tuned",
        action="store_true",
        default=False,
        help=argparse.SUPPRESS,
    )
    return parser

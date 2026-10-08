# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Unit tests for the dnn-benchmark CLI parser and TOML config files."""

import importlib
import re
from pathlib import Path
from unittest.mock import patch

import pytest

from dnn_benchmarking.cli.config_file import apply_config_file
from dnn_benchmarking.cli.parser import CONFIG_OPTIONS, create_parser

ROOT = Path(__file__).resolve().parents[3]
MIOPEN_ENGINE_ID = 0x15B46865C717A122  # FNV-1a-64("MIOPEN_ENGINE"), below 2**63


def _write_config(path: Path, text: str) -> Path:
    path.write_text(text)
    return path


def _parse_with_config(argv: list[str]):
    args = create_parser(suppress_defaults=True).parse_args(argv)
    apply_config_file(args)
    return args


def _config_error(tmp_path: Path, body: str) -> str:
    config = _write_config(tmp_path / "bench.toml", f"version = 1\n{body}\n")
    args = create_parser(suppress_defaults=True).parse_args(["--config", str(config)])
    with pytest.raises(ValueError) as excinfo:
        apply_config_file(args)
    return str(excinfo.value)


@pytest.mark.parametrize(
    ("token", "expected"),
    [
        ("MIOPEN_ENGINE", MIOPEN_ENGINE_ID),
        ("0x15B46865C717A122", MIOPEN_ENGINE_ID),
        (str(MIOPEN_ENGINE_ID), MIOPEN_ENGINE_ID),
        ("-4567890123456789012", -4567890123456789012),
        # Unsigned hex/decimal above 2**63 wraps to hipDNN's signed int64 IDs.
        ("0xFFFFFFFFFFFFFFFF", -1),
        (str(2**64 - 2), -2),
        ("0x8000000000000000", -(1 << 63)),
        # FNV-1a-64("HIP_MLOPS_ENGINE") = 0xDD993EF5525F7BF9, above 2**63; hipDNN's
        # engineNameToId casts it to int64.
        ("HIP_MLOPS_ENGINE", 0xDD993EF5525F7BF9 - (1 << 64)),
        # Any token that is not a number is a name, as in hipDNN.
        ("hipkernel:ConvFwd", 0xF6975AB2C79B088E - (1 << 64)),
    ],
)
def test_engine_tokens_resolve_identically_from_cli_and_config(
    tmp_path: Path, token: str, expected: int
) -> None:
    cli_args = _parse_with_config([f"--engine={token},7"])
    config = _write_config(
        tmp_path / "bench.toml",
        f'version = 1\n[[engines]]\nid = "{token}"\n[[engines]]\nid = 7\n',
    )
    config_args = _parse_with_config(["--config", str(config)])

    assert cli_args.engine == [expected, 7]
    assert config_args.engine == [expected, 7]


@pytest.mark.parametrize("token", [str(2**64), "0x1" + "0" * 16, ","])
def test_invalid_engine_tokens_are_usage_errors(token: str) -> None:
    with pytest.raises(SystemExit) as exc:
        create_parser().parse_args(["--engine", token])
    assert exc.value.code == 2


def test_engine_list_keeps_order_and_duplicates() -> None:
    args = create_parser().parse_args(["-e", "3, 1,3,MIOPEN_ENGINE"])
    assert args.engine == [3, 1, 3, MIOPEN_ENGINE_ID]


def test_plugin_path_is_a_comma_list() -> None:
    args = create_parser().parse_args(["--plugin-path", "/a, /b,"])
    assert args.plugin_path == [Path("/a"), Path("/b")]
    with pytest.raises(SystemExit) as exc:
        create_parser().parse_args(["--plugin-path", " , "])
    assert exc.value.code == 2


@pytest.mark.parametrize(
    "argv",
    [
        ["--iters", "0"],
        ["--warmup", "-1"],
        ["--min-time-ms", "-0.5"],
        ["--profiling-timeout", "-1"],
        ["--rtol=-1e-3"],  # "--rtol -1e-3" fails in argparse, not the bound
        ["--atol=-1"],
        ["--rtol", "nan"],  # NaN would pass every comparison
        ["--atol", "inf"],
        ["--min-time-ms", "nan"],
        ["--cache-mode", "lukewarm"],
        ["--iter", "5"],  # abbreviations are rejected (allow_abbrev=False)
    ],
)
def test_invalid_cli_values_are_usage_errors(argv: list[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        create_parser().parse_args(argv)
    assert exc.value.code == 2


def test_config_populates_args_when_cli_does_not_override(tmp_path: Path) -> None:
    config = _write_config(
        tmp_path / "bench.toml",
        """
version = 1
graphs = ["graphs/a.json", "graphs/b.json"]
warmup = 3
iters = 7
min_time_ms = 25
cache_mode = "cold"
seed = 42
quiet = true
metrics = false

[[engines]]
id = 1
plugin_path = "/plugins/b"

[[engines]]
id = 1
plugin_path = "/plugins/a"
""",
    )

    args = _parse_with_config(["--config", str(config)])

    assert args.graph == [
        str(tmp_path / "graphs/a.json"),
        str(tmp_path / "graphs/b.json"),
    ]
    assert (args.warmup, args.iters, args.seed) == (3, 7, 42)
    assert args.min_time_ms == 25.0
    assert args.cache_mode == "cold"
    assert args.quiet is True
    assert args.metrics is False
    assert args.engine == [1, 1]
    # Driveless absolute paths from the config keep their order and tail;
    # on Windows the loader anchors them to the config dir's drive, so
    # compare in POSIX form by suffix rather than as exact paths.
    plugin_paths = [p.as_posix() for p in args.plugin_path]
    assert len(plugin_paths) == 2
    assert plugin_paths[0].endswith("plugins/b")
    assert plugin_paths[1].endswith("plugins/a")


def test_cli_values_override_config_values(tmp_path: Path) -> None:
    config = _write_config(
        tmp_path / "bench.toml",
        """
version = 1
graphs = ["from_config.json"]
iters = 7
oracle_mode = "exhaustive"
""",
    )

    args = _parse_with_config(
        [
            "--config",
            str(config),
            "--graph",
            "from_cli.json",
            "--iters",
            "11",
            "--oracle-mode",
            "plan",
        ]
    )

    assert args.graph == ["from_cli.json"]
    assert args.iters == 11
    assert args.oracle_mode == "plan"


@pytest.mark.parametrize(
    ("key", "flag"),
    [
        ("verbose", "--no-verbose"),
        ("quiet", "--no-quiet"),
        ("autotune", "--no-autotune"),
        ("perf", "--no-perf"),
        ("roofline", "--no-roofline"),
        ("pmc_allow_multipass", "--no-pmc-allow-multipass"),
    ],
)
def test_cli_can_turn_off_a_boolean_the_config_turns_on(
    tmp_path: Path, key: str, flag: str
) -> None:
    config = _write_config(tmp_path / "bench.toml", f"version = 1\n{key} = true\n")

    assert getattr(_parse_with_config(["--config", str(config)]), key) is True
    assert getattr(_parse_with_config(["--config", str(config), flag]), key) is False


def test_cli_engine_replaces_config_engine_matrix(tmp_path: Path) -> None:
    config = _write_config(
        tmp_path / "bench.toml",
        """
version = 1

[[engines]]
id = 2
plugin_path = "/plugins/b"

[[engines]]
id = 1
plugin_path = "/plugins/a"
""",
    )

    args = _parse_with_config(["--config", str(config), "--engine", "9,8"])

    assert args.engine == [9, 8]
    assert args.plugin_path is None


def test_cli_engine_keeps_top_level_config_plugin_path(tmp_path: Path) -> None:
    """Regression (CLI-03): --engine alone must not drop the recipe's plugin dir."""
    config = _write_config(
        tmp_path / "bench.toml",
        """
version = 1
plugin_path = "plugins"

[[engines]]
id = 2
""",
    )

    args = _parse_with_config(["--config", str(config), "--engine", "9"])

    assert args.engine == [9]
    assert args.plugin_path == [tmp_path / "plugins"]


def test_every_engine_sets_plugin_path_when_any_engine_does(tmp_path: Path) -> None:
    message = _config_error(
        tmp_path,
        '[[engines]]\nid = 1\nplugin_path = "/plugins/a"\n[[engines]]\nid = 2',
    )
    assert "Every config engine must set plugin_path" in message


@pytest.mark.parametrize(
    ("body", "field"),
    [
        ("itres = 1000", "itres"),
        ("[profiling]\nenabled = true", "profiling"),
        ("[comparison]\nbaseline = 'x'", "comparison"),
    ],
)
def test_unknown_top_level_config_fields_are_rejected(
    tmp_path: Path, body: str, field: str
) -> None:
    assert f"Unknown config field: {field}" in _config_error(tmp_path, body)


@pytest.mark.parametrize("field", ["plugin_pat", "label", "name"])
def test_unknown_engine_config_fields_are_rejected(tmp_path: Path, field: str) -> None:
    message = _config_error(tmp_path, f'[[engines]]\nid = 1\n{field} = "x"')
    assert f"Unknown config engine 0 field: {field}" in message


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("runtime", '"pytoch"'),
        ("oracle_mode", '"full"'),
        ("validate", '"torch"'),
        ("pmc", '"everything"'),
        ("cache_mode", '"hot"'),
        ("pytorch_sdpa_backend", '"aotriton"'),
    ],
)
def test_invalid_config_choice_values_are_rejected(
    tmp_path: Path, field: str, value: str
) -> None:
    message = _config_error(tmp_path, f"{field} = {value}")
    assert f"Config field '{field}' must be one of" in message


@pytest.mark.parametrize(
    "body",
    ["iters = true", "iters = 0", "warmup = -1", "min_time_ms = -1", "seed = 1.5"],
)
def test_invalid_config_scalars_are_rejected(tmp_path: Path, body: str) -> None:
    key = body.split()[0]
    assert f"Config field '{key}'" in _config_error(tmp_path, body)


def test_config_paths_are_relative_to_config_file(tmp_path: Path) -> None:
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config = _write_config(
        config_dir / "bench.toml",
        """
version = 1
graphs = ["../graphs/g.json"]
output = "results/out.json"
plugin_path = "../plugins"
profiling_output_dir = "profiles"
hipdnn_cache_dir = "hipdnn-cache"
""",
    )

    args = _parse_with_config(["--config", str(config)])

    assert args.graph == [str(config_dir / "../graphs/g.json")]
    assert args.output == config_dir / "results/out.json"
    assert args.plugin_path == [config_dir / "../plugins"]
    assert args.profiling_output_dir == config_dir / "profiles"
    assert args.hipdnn_cache_dir == config_dir / "hipdnn-cache"


def test_sample_configs_parse_and_cover_every_config_key() -> None:
    full_config = ROOT / "sample_configs" / "config.toml.example"
    full_args = _parse_with_config(["--config", str(full_config)])
    assert full_args.graph
    for graph in full_args.graph:
        assert Path(graph).exists()

    graph = ROOT / "graphs" / "sample_conv_fwd.json"
    basic_args = _parse_with_config(
        [
            "--config",
            str(ROOT / "sample_configs" / "basic.toml.example"),
            "--graph",
            str(graph),
        ]
    )
    assert basic_args.graph == [str(graph)]

    text = full_config.read_text()
    missing = [
        option.config_key
        for option in CONFIG_OPTIONS
        if not re.search(rf"^#?\s*{option.config_key}\s*=", text, re.MULTILINE)
    ]
    assert missing == []


@pytest.fixture
def cache_env(tmp_path: Path, monkeypatch) -> None:
    """Pre-set cache variables so main() leaves the process environment alone."""
    for var in (
        "XDG_CACHE_HOME",
        "MIOPEN_USER_DB_PATH",
        "MIOPEN_CUSTOM_CACHE_DIR",
        "AMD_COMGR_CACHE_DIR",
    ):
        monkeypatch.setenv(var, str(tmp_path / "cache"))


def test_invalid_config_is_a_usage_error_before_graph_resolution(
    tmp_path: Path, cache_env
) -> None:
    config = _write_config(
        tmp_path / "bench.toml", 'version = 1\ngraphs = ["g.json"]\nruntime = "x"\n'
    )
    main_module = importlib.import_module("dnn_benchmarking.cli.main")

    with (
        patch.object(main_module, "_resolve_graphs") as mock_resolve,
        pytest.raises(SystemExit) as exc,
    ):
        main_module.main(["--config", str(config)])

    assert exc.value.code == 2
    mock_resolve.assert_not_called()


def test_missing_graph_without_config_is_a_usage_error(cache_env) -> None:
    main_module = importlib.import_module("dnn_benchmarking.cli.main")

    with pytest.raises(SystemExit) as exc:
        main_module.main([])

    assert exc.value.code == 2

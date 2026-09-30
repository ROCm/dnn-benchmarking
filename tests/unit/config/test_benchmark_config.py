# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for SuiteConfig, ValidationConfig and TimingPolicy."""

from pathlib import Path

import pytest

from dnn_benchmarking.cli.config_file import apply_config_file
from dnn_benchmarking.cli.parser import create_parser
from dnn_benchmarking.config import (
    ExecutionBackendName,
    OracleMode,
    PyTorchSdpaBackendName,
    SuiteConfig,
    TimingPolicy,
    ValidationConfig,
)


def _namespace(argv: list[str]):
    args = create_parser(suppress_defaults=True).parse_args(argv)
    apply_config_file(args)
    return args


class TestFromNamespace:
    """SuiteConfig.from_namespace maps merged CLI/config args onto the config."""

    def test_config_file_and_cli_values_reach_suite_config(self, tmp_path: Path) -> None:
        config = tmp_path / "bench.toml"
        config.write_text(
            """
version = 1
warmup = 1
iters = 2
min_time_ms = 5
cache_mode = "cold"
oracle_mode = "plan"
metrics_tier = "off"
rtol = 1e-3

[[engines]]
id = "MIOPEN_ENGINE"

[[engines]]
id = 1
"""
        )
        args = _namespace(
            ["--config", str(config), "--seed", "7", "--plugin-path", "/plugins"]
        )

        suite = SuiteConfig.from_namespace(args)

        assert suite.timing_policy == TimingPolicy(
            warmup_iters=1, iters=2, min_time_ms=5.0, cache_mode="cold"
        )
        assert suite.seed == 7
        assert suite.engine_filter is None  # --plugin-path replaces the matrix
        assert suite.plugin_paths == [Path("/plugins")]
        assert suite.oracle_mode is OracleMode.PLAN
        assert suite.metrics.basic_enabled is False
        assert suite.validation.tolerance_override == (1e-3, 1e-3)

    def test_hipdnn_without_plugin_path_uses_rocm_path(self, monkeypatch) -> None:
        monkeypatch.setenv("ROCM_PATH", "/opt/rocm-x")
        suite = SuiteConfig.from_namespace(_namespace(["-g", "g.json"]))
        assert suite.plugin_paths == [Path("/opt/rocm-x/lib/hipdnn_plugins/engines")]

    def test_pytorch_backend_gets_no_default_plugin_path(self, monkeypatch) -> None:
        monkeypatch.setenv("ROCM_PATH", "/opt/rocm-x")
        suite = SuiteConfig.from_namespace(_namespace(["-b", "pytorch"]))
        assert suite.plugin_paths is None

    @pytest.mark.parametrize(
        ("argv", "flag"),
        [
            (["-e", "1"], "--engine"),
            (["--plugin-path", "/p"], "--plugin-path"),
            (["--validate", "pytorch"], "--validate pytorch"),
            (["--pmc", "basic"], "--pmc"),
            (["--emit-trace", "pftrace"], "--emit-trace"),
            (["--perf"], "--perf"),
            (["--roofline"], "--roofline"),
            (["--oracle-mode", "plan"], "--oracle-mode"),
            (["--autotune"], "--autotune"),
            (["--cache-dir", "/c"], "--cache-dir"),
        ],
    )
    def test_pytorch_backend_rejects_hipdnn_only_options(
        self, argv: list[str], flag: str
    ) -> None:
        args = _namespace(["-b", "pytorch", *argv])
        with pytest.raises(ValueError, match=f"{flag}.*--backend pytorch"):
            SuiteConfig.from_namespace(args)


class TestSuiteConfigValidation:
    @pytest.mark.parametrize(
        ("kwargs", "flag"),
        [
            ({"warmup_iters": -1}, "--warmup"),
            ({"benchmark_iters": 0}, "--iters"),
            ({"min_time_ms": -1.0}, "--min-time-ms"),
            ({"cache_mode": "hot"}, "--cache-mode"),
            ({"engine_filter": []}, "--engine"),
            ({"backend": "tensorflow"}, "--backend"),
            ({"oracle_mode": "full"}, "--oracle-mode"),
            ({"pytorch_sdpa_backend": "aotriton"}, "--pytorch-sdpa-backend"),
            (
                {"pytorch_sdpa_backend": "math", "pytorch_rocm_fa_library": "x"},
                "--pytorch-rocm-fa-library",
            ),
        ],
    )
    def test_invalid_values_name_the_flag(self, kwargs: dict, flag: str) -> None:
        with pytest.raises(ValueError, match=flag):
            SuiteConfig(**kwargs)

    def test_exhaustive_oracle_requires_warmup(self) -> None:
        with pytest.raises(ValueError, match="--oracle-mode exhaustive requires --warmup"):
            SuiteConfig(oracle_mode="exhaustive", warmup_iters=0)
        assert SuiteConfig(oracle_mode="exhaustive", warmup_iters=1).oracle_exhaustive

    def test_iters_above_the_default_cap_raise_the_cap(self) -> None:
        policy = SuiteConfig(benchmark_iters=20_000).timing_policy
        assert policy.max_iters >= policy.iters == 20_000

    @pytest.mark.parametrize("backend", [b.value for b in PyTorchSdpaBackendName])
    def test_accepts_every_sdpa_backend(self, backend: str) -> None:
        assert SuiteConfig(pytorch_sdpa_backend=backend).pytorch_sdpa_backend.value == (
            backend
        )

    def test_rocm_fa_library_accepted_with_flash(self) -> None:
        config = SuiteConfig(
            pytorch_sdpa_backend="flash", pytorch_rocm_fa_library="third-party"
        )
        assert config.pytorch_sdpa_backend is PyTorchSdpaBackendName.FLASH

    def test_backend_string_is_normalised(self) -> None:
        assert SuiteConfig(backend="pytorch").backend is ExecutionBackendName.PYTORCH


class TestSuiteConfigPluginPaths:
    """Tests for SuiteConfig engine/plugin path selection."""

    def test_single_plugin_path_applies_to_all_engines(self) -> None:
        config = SuiteConfig(engine_filter=[1, 2], plugin_paths=[Path("/plugins/a")])
        selections = config.engine_selections_for([1, 2])

        assert [s.plugin_path for s in selections] == [Path("/plugins/a")] * 2
        assert config.plugin_path == Path("/plugins/a")

    def test_repeated_engine_ids_keep_distinct_plugin_paths_in_order(self) -> None:
        config = SuiteConfig(
            engine_filter=[1, 1],
            plugin_paths=[Path("/plugins/a"), Path("/plugins/b")],
        )

        selections = config.engine_selections_for([1, 1])

        assert [s.engine_id for s in selections] == [1, 1]
        assert [s.plugin_path for s in selections] == [
            Path("/plugins/a"),
            Path("/plugins/b"),
        ]
        assert config.plugin_path is None

    def test_multiple_plugin_paths_require_engine_filter(self) -> None:
        with pytest.raises(ValueError, match="requires --engine"):
            SuiteConfig(plugin_paths=[Path("/plugins/a"), Path("/plugins/b")])

    def test_plugin_path_count_must_match_engine_count(self) -> None:
        with pytest.raises(ValueError, match="entry count"):
            SuiteConfig(
                engine_filter=[1, 2, 3],
                plugin_paths=[Path("/plugins/a"), Path("/plugins/b")],
            )


class TestValidationConfig:
    def test_both_tolerances_override_dtype_defaults(self) -> None:
        assert ValidationConfig(rtol=1e-3, atol=1e-4).tolerance_override == (1e-3, 1e-4)

    def test_single_tolerance_applies_to_both_values(self) -> None:
        assert ValidationConfig(rtol=1e-3).tolerance_override == (1e-3, 1e-3)
        assert ValidationConfig(atol=1e-4).tolerance_override == (1e-4, 1e-4)

    def test_enabled_tracks_provider(self) -> None:
        assert ValidationConfig(provider="none").enabled is False
        assert ValidationConfig(provider="pytorch").enabled is True

    @pytest.mark.parametrize(
        ("kwargs", "flag"),
        [
            ({"provider": "invalid"}, "--validate"),
            ({"rtol": -1e-5}, "--rtol"),
            ({"atol": -1e-8}, "--atol"),
        ],
    )
    def test_invalid_values_name_the_flag(self, kwargs: dict, flag: str) -> None:
        with pytest.raises(ValueError, match=flag):
            ValidationConfig(**kwargs)


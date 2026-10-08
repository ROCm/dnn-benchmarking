# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for SuiteConfig, ValidationConfig and TimingPolicy."""

from pathlib import Path

import pytest

from dnn_benchmarking.cli.config_file import apply_config_file
from dnn_benchmarking.cli.parser import create_parser
from dnn_benchmarking.config import (
    RuntimeName,
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

    def test_config_file_and_cli_values_reach_suite_config(
        self, tmp_path: Path
    ) -> None:
        config = tmp_path / "bench.toml"
        config.write_text(
            """
version = 1
warmup = 1
iters = 2
min_time_ms = 5
cache_mode = "cold"
oracle_mode = "plan"
metrics = false
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
        assert suite.metrics.basic is False
        assert suite.validation.tolerance_override == (1e-3, 1e-3)

    def test_default_seed_is_fixed(self) -> None:
        """Inputs are reproducible without --seed (pre-fix default was random)."""
        assert SuiteConfig.from_namespace(_namespace(["-g", "g.json"])).seed == 0

    def test_shipped_run_defaults(self) -> None:
        """docs/usage.md documents these; default-path trend numbers depend on them."""
        suite = SuiteConfig.from_namespace(_namespace(["-g", "g.json"]))

        policy = suite.timing_policy
        assert (
            policy.warmup_iters,
            policy.iters,
            policy.min_time_ms,
            policy.max_iters,
            policy.cache_mode,
            policy.timing_block,
        ) == (10, 100, 0.0, 10_000, "warm", 1)
        assert suite.metrics.profiling_timeout_s == 600

    def test_hipdnn_without_plugin_path_uses_rocm_path(self, monkeypatch) -> None:
        monkeypatch.setenv("ROCM_PATH", "/opt/rocm-x")
        suite = SuiteConfig.from_namespace(_namespace(["-g", "g.json"]))
        assert suite.plugin_paths == [Path("/opt/rocm-x/lib/hipdnn_plugins/engines")]

    def test_pytorch_runtime_gets_no_default_plugin_path(self, monkeypatch) -> None:
        monkeypatch.setenv("ROCM_PATH", "/opt/rocm-x")
        suite = SuiteConfig.from_namespace(_namespace(["-r", "pytorch"]))
        assert suite.plugin_paths is None

    def test_pmc_all_with_multipass_reaches_metrics_config(self) -> None:
        args = _namespace(["-g", "g.json", "--pmc", "all", "--pmc-allow-multipass"])
        metrics = SuiteConfig.from_namespace(args).metrics
        assert (metrics.pmc_set, metrics.pmc_allow_multipass) == ("all", True)

    @pytest.mark.parametrize(
        ("argv", "flag"),
        [
            (["-e", "1"], "--engine"),
            (["--plugin-path", "/p"], "--plugin-path"),
            (["--validate", "pytorch"], "--validate pytorch"),
            (["--pmc", "basic"], "--pmc"),
            (["--trace"], "--trace"),
            (["--perf"], "--perf"),
            (["--roofline"], "--roofline"),
            (["--oracle-mode", "plan"], "--oracle-mode"),
            (["--autotune"], "--autotune"),
            (["--hipdnn-cache-dir", "/c"], "--hipdnn-cache-dir"),
        ],
    )
    def test_pytorch_runtime_rejects_hipdnn_only_options(
        self, argv: list[str], flag: str
    ) -> None:
        args = _namespace(["-r", "pytorch", *argv])
        with pytest.raises(ValueError, match=f"{flag}.*--runtime pytorch"):
            SuiteConfig.from_namespace(args)


class TestSuiteConfigValidation:
    @pytest.mark.parametrize(
        ("kwargs", "flag"),
        [
            ({"warmup_iters": -1}, "--warmup"),
            ({"benchmark_iters": 0}, "--iters"),
            ({"min_time_ms": -1.0}, "--min-time-ms"),
            ({"cache_mode": "hot"}, "--cache-mode"),
            ({"timing_block": 0}, "--timing-block"),
            ({"cache_mode": "cold", "timing_block": 2}, "--timing-block"),
            ({"engine_filter": []}, "--engine"),
            ({"runtime": "tensorflow"}, "--runtime"),
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

    def test_max_iters_below_iters_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="max_iters must be >= iters"):
            TimingPolicy(iters=5, max_iters=4)

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

    def test_runtime_string_is_normalised(self) -> None:
        assert SuiteConfig(runtime="pytorch").runtime is RuntimeName.PYTORCH


class TestSuiteConfigPluginPaths:
    """Tests for SuiteConfig engine/plugin path selection."""

    def test_single_plugin_path_applies_to_all_engines(self) -> None:
        config = SuiteConfig(engine_filter=[1, 2], plugin_paths=[Path("/plugins/a")])
        selections = config.engine_selections_for([1, 2])

        assert [s.plugin_path for s in selections] == [Path("/plugins/a")] * 2
        assert config.plugin_path == Path("/plugins/a")

    @pytest.mark.parametrize(
        "engines, paths", [([1, 1], ["a", "b"]), ([2, 1], ["b", "a"])]
    )
    def test_plugin_paths_pair_with_engines_in_caller_order(
        self, engines: list[int], paths: list[str]
    ) -> None:
        plugin_paths = [Path("/plugins") / p for p in paths]
        config = SuiteConfig(engine_filter=engines, plugin_paths=plugin_paths)

        selections = config.engine_selections_for(engines)

        assert [s.engine_id for s in selections] == engines
        assert [s.plugin_path for s in selections] == plugin_paths
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

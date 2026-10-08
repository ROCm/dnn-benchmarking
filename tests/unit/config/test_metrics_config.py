# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for MetricsConfig and its embedding in SuiteConfig."""

from pathlib import Path

import pytest

from dnn_benchmarking.config.benchmark_config import MetricsConfig, SuiteConfig


def test_off_tier_disables_basic_probes() -> None:
    assert MetricsConfig(basic=False).basic is False
    assert SuiteConfig(metrics=MetricsConfig(basic=False)).metrics.basic is False


@pytest.mark.parametrize(
    ("kwargs", "flag"),
    [
        ({"pmc_set": "everything"}, "--pmc"),
        ({"pmc_set": " "}, "--pmc"),
        ({"profiling_timeout_s": -1}, "--profiling-timeout"),
    ],
)
def test_invalid_values_name_the_flag(kwargs: dict, flag: str) -> None:
    with pytest.raises(ValueError, match=flag):
        MetricsConfig(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"trace": True},
        {"pmc_set": "basic"},
        {"perf": True},
        {"roofline": True},
    ],
)
def test_any_opt_in_source_requests_a_profiling_pass(kwargs: dict) -> None:
    assert MetricsConfig(**kwargs).opt_in_pass_requested is True


def test_pmc_all_requires_multipass_opt_in() -> None:
    with pytest.raises(ValueError, match="--pmc all requires --pmc-allow-multipass"):
        MetricsConfig(pmc_set="all")
    assert MetricsConfig(pmc_set="all", pmc_allow_multipass=True).pmc_set == "all"


def test_profiling_output_dir_str_coerced_to_path() -> None:
    assert MetricsConfig(profiling_output_dir="/tmp/out").profiling_output_dir == Path(
        "/tmp/out"
    )

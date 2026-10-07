# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for the JSON shape of ProviderEngineResult.to_dict.

A row nests its plan-run measurements under ``ootb`` (and ``oracle``); both
keys are always present and null when not measured. Row-level console
metrics never reach the JSON.
"""

import json

import pytest

from dnn_benchmarking.reporting.statistics import BenchmarkStats
from dnn_benchmarking.reporting.suite_results import (
    CorrectnessResult,
    OotbResult,
    ProviderEngineResult,
    SuiteMetadata,
)

_ROW_KEYS = {
    "provider",
    "engine_id",
    "engine_name",
    "engine_version",
    "started_at",
    "status",
    "ootb",
    "oracle",
}

_PLAN_RUN_KEYS = {
    "build_time_ms",
    "gpu_kernel_stats",
    "host_stats",
    "workspace_bytes",
    "analytical_flops",
    "derived_tflops_per_s",
    "correctness",
}


def _bench_stats(mean: float = 1.0) -> BenchmarkStats:
    return BenchmarkStats(
        mean_ms=mean,
        std_ms=0.1,
        min_ms=mean - 0.1,
        max_ms=mean + 0.1,
        p95_ms=mean + 0.05,
        p99_ms=mean + 0.09,
    )


class TestSuccessRowShape:
    def test_top_level_keys(self):
        pe = ProviderEngineResult(
            provider="miopen",
            engine_id=1,
            status="success",
            ootb=OotbResult(build_time_ms=12.3),
        )
        assert set(pe.to_dict()) == _ROW_KEYS

    def test_ootb_emits_every_key_with_null_for_unmeasured(self):
        pe = ProviderEngineResult(
            provider="miopen",
            engine_id=1,
            status="success",
            ootb=OotbResult(build_time_ms=12.3, gpu_kernel_stats=_bench_stats(0.5)),
        )
        ootb = pe.to_dict()["ootb"]
        assert set(ootb) == _PLAN_RUN_KEYS | {"extra_metrics"}
        assert ootb["build_time_ms"] == 12.3
        assert ootb["gpu_kernel_stats"]["mean_ms"] == 0.5
        for key in (
            "host_stats",
            "workspace_bytes",
            "analytical_flops",
            "derived_tflops_per_s",
            "correctness",
            "extra_metrics",
        ):
            assert ootb[key] is None

    def test_ootb_serializes_all_measurements(self):
        corr = CorrectnessResult(
            execution_success=True, tolerance_match=True, rtol=1e-5, atol=1e-8
        )
        pe = ProviderEngineResult(
            provider="miopen",
            engine_id=1,
            status="success",
            ootb=OotbResult(
                build_time_ms=10.0,
                gpu_kernel_stats=_bench_stats(0.5),
                host_stats=_bench_stats(1.0),
                workspace_bytes=4096,
                analytical_flops=10**9,
                derived_tflops_per_s=2.0,
                correctness=corr,
            ),
        )
        ootb = pe.to_dict()["ootb"]
        assert ootb["build_time_ms"] == 10.0
        assert ootb["gpu_kernel_stats"]["mean_ms"] == 0.5
        assert ootb["host_stats"]["mean_ms"] == 1.0
        assert ootb["workspace_bytes"] == 4096
        assert ootb["analytical_flops"] == 10**9
        assert ootb["derived_tflops_per_s"] == 2.0
        assert ootb["correctness"] == corr.to_dict()

    def test_console_only_fields_not_serialized(self):
        pe = ProviderEngineResult(
            provider="miopen",
            engine_id=1,
            status="success",
            elapsed_time_ms=200.0,
            analytical_flops_partial=True,
            analytical_io_bytes=10**6,
            derived_gbytes_per_s=2.0,
            cpu_user_time_per_iter_us=40.0,
            cpu_kernel_time_per_iter_us=2.5,
            vram_used_mb=4096.0,
            ootb=OotbResult(analytical_flops=42),
        )
        d = pe.to_dict()
        assert set(d) == _ROW_KEYS
        for key in (
            "elapsed_time_ms",
            "analytical_flops_partial",
            "analytical_io_bytes",
            "derived_gbytes_per_s",
            "cpu_user_time_per_iter_us",
            "cpu_kernel_time_per_iter_us",
            "vram_used_mb",
        ):
            assert key not in d["ootb"]


class TestExtraMetrics:
    def test_ootb_on_non_success_status_asserts(self):
        """The runner attaches ootb only on the success path, so a
        non-success row carrying one is a regression in that gating.
        Serialization must fail loudly rather than silently drop the
        measurements from the JSON."""
        pe = ProviderEngineResult(
            provider="miopen",
            engine_id=1,
            status="error",
            error_message="boom",
            ootb=OotbResult(extra_metrics={"pmc": {"set": "basic"}}),
        )
        with pytest.raises(AssertionError, match="ootb is set"):
            pe.to_dict()

    def test_extra_metrics_combined_payload_round_trips_through_json(self):
        """Regression check: the realistic shape produced by the
        orchestrator (PMC counters + per-kernel maps, perf with int +
        float + None, roofline path strings, trace path strings) must
        survive json.dumps -> json.loads unchanged. Catches any future
        switch to a non-serialisable type (np.float32, Path, Decimal,
        ...) in any of the four source modules."""
        payload = {
            "pmc": {
                "set": "basic",
                "arch": "gfx90a",
                "counters_requested": ["GRBM_GUI_ACTIVE", "SQ_WAVES"],
                "counters": {
                    "GRBM_GUI_ACTIVE": {"sum": 12345.0, "mean_per_kernel": 4115.0},
                    "SQ_WAVES": {"sum": 832.0, "mean_per_kernel": 104.0},
                },
                "per_kernel": {
                    "conv_kernel": {"GRBM_GUI_ACTIVE": 4115.0, "SQ_WAVES": 104.0},
                    "gemm_kernel": {"GRBM_GUI_ACTIVE": 8230.0, "SQ_WAVES": 728.0},
                },
                "db_path": "/tmp/profiling-output/x/results.db",
            },
            "perf": {
                "cycles_user": 9999,
                "instructions_user": 8101,
                "ipc_user": 0.81,
                "cycles_kernel": None,
                "instructions_kernel": None,
                "task_clock_ms": 123.45,
                "context_switches": 12,
                "page_faults": 3,
                "kernel_perf_paranoid": 4,
                "kernel_events_skipped_reason": "kernel.perf_event_paranoid=4 > 1",
                "csv_path": "/tmp/profiling-output/x/perf.csv",
            },
            "roofline": {
                "roofline_csv": "/tmp/profiling-output/x/roofline.csv",
                "sysinfo_csv": "/tmp/profiling-output/x/sysinfo.csv",
                "workload_path": "/tmp/profiling-output/x",
            },
            "trace": {
                "format": "pftrace",
                "path": "/tmp/profiling-output/x/results.pftrace",
            },
        }
        pe = ProviderEngineResult(
            provider="miopen",
            engine_id=1,
            status="success",
            ootb=OotbResult(extra_metrics=payload),
        )
        d = pe.to_dict()
        # JSON round-trip: any non-serializable nested value would raise.
        round_tripped = json.loads(json.dumps(d))
        assert round_tripped["ootb"]["extra_metrics"] == payload


class TestErrorAndSkipRows:
    def test_error_status_emits_only_error_message(self):
        pe = ProviderEngineResult(
            provider="miopen",
            engine_id=1,
            status="error",
            error_message="boom",
            # Console metrics set on an error path must NOT leak to JSON.
            analytical_io_bytes=999,
            vram_used_mb=1.0,
        )
        d = pe.to_dict()
        assert set(d) == _ROW_KEYS | {"error_message"}
        assert d["status"] == "error"
        assert d["error_message"] == "boom"
        assert d["ootb"] is None
        assert d["oracle"] is None

    def test_skipped_status_emits_only_skip_reason(self):
        pe = ProviderEngineResult(
            provider="miopen",
            engine_id=1,
            status="skipped",
            skip_reason="unsupported",
            analytical_io_bytes=999,
        )
        d = pe.to_dict()
        assert set(d) == _ROW_KEYS | {"skip_reason"}
        assert d["status"] == "skipped"
        assert d["skip_reason"] == "unsupported"
        assert d["ootb"] is None
        assert d["oracle"] is None


class TestSuiteMetadataMachineFields:
    def test_machine_fields_in_to_dict(self):
        meta = SuiteMetadata(
            timestamp="2026-05-11T00:00:00Z",
            hostname="test-host",
            total_graphs=1,
            total_combinations=1,
            pass_combinations=1,
            fail_combinations=0,
            skip_combinations=0,
            error_combinations=0,
            cpu_model="EPYC 9654",
            cpu_count=192,
            numa_nodes=2,
            total_ram_gb=1536.0,
            kernel_version="6.8.0-31-generic",
            gpu_compute_units=304,
            gpu_hbm_gb=192.0,
            gpu_pcie_link="gen4 x16",
            amdgpu_driver_version="6.14.5",
        )
        d = meta.to_dict()
        assert d["cpu_model"] == "EPYC 9654"
        assert d["cpu_count"] == 192
        assert d["numa_nodes"] == 2
        assert d["total_ram_gb"] == 1536.0
        assert d["kernel_version"] == "6.8.0-31-generic"
        assert d["gpu_compute_units"] == 304
        assert d["gpu_hbm_gb"] == 192.0
        assert d["gpu_pcie_link"] == "gen4 x16"
        assert d["amdgpu_driver_version"] == "6.14.5"

    def test_footprint_fields_in_to_dict(self):
        # Process RSS / VRAM live on SuiteMetadata (not per-engine)
        # because they're flat across the suite.
        meta = SuiteMetadata(
            timestamp="2026-05-11T00:00:00Z",
            hostname="test-host",
            total_graphs=1,
            total_combinations=1,
            pass_combinations=1,
            fail_combinations=0,
            skip_combinations=0,
            error_combinations=0,
            host_rss_mb=843.3,
            host_ram_available_mb=2177515.0,
            vram_used_mb=4096.0,
            vram_total_mb=196608.0,
        )
        d = meta.to_dict()
        assert d["host_rss_mb"] == 843.3
        assert d["host_ram_available_mb"] == 2177515.0
        assert d["vram_used_mb"] == 4096.0
        assert d["vram_total_mb"] == 196608.0

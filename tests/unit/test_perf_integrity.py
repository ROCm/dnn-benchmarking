# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Regression tests for timing integrity, compare gates, runtime guards, and telemetry."""

from __future__ import annotations

import json
import signal
import types
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest

from dnn_benchmarking.cli import hipdnn_tuned_child, suite_runner_cli
from dnn_benchmarking.config.benchmark_config import (
    SuiteConfig,
    TimingPolicy,
    ValidationConfig,
)
from dnn_benchmarking.execution import aiter_sdpa, oracle, suite_runner, timing
from dnn_benchmarking.execution.correctness import check_correctness
from dnn_benchmarking.execution.sdpa_graph import DenseSdpa
from dnn_benchmarking.graph.tensor_info import TensorInfo
from dnn_benchmarking.metrics import gpu_smi
from dnn_benchmarking.metrics.analytical import (
    ANALYTICAL_FORMULA_VERSION,
    compute_flops,
    derive_throughputs,
)
from dnn_benchmarking.metrics.analytical._common import tensor_dim_product
from dnn_benchmarking.reporting import compare
from dnn_benchmarking.reporting.reporter import Reporter
from dnn_benchmarking.reporting.statistics import (
    BenchmarkStats,
    TimingInfo,
    noise_warnings,
)
from dnn_benchmarking.reporting.suite_results import (
    CorrectnessResult,
    GraphResult,
    PlanResult,
    ProviderEngineResult,
    RunInfo,
    SuiteResult,
)
from dnn_benchmarking.validation import ReferenceOutput


def _stats(median: float, spread: float = 0.0) -> BenchmarkStats:
    return BenchmarkStats.from_timings([median - spread, median + spread] * 10)


def _row(
    name: str,
    median: float,
    *,
    mode: str = "staged",
    timer: str = "hip",
    launch_mode: str = "raw",
    match: bool | None = True,
    validated: bool = True,
    role: str = "engine",
    timings: List[float] | None = None,
) -> ProviderEngineResult:
    stats = BenchmarkStats.from_timings(timings) if timings else _stats(median)
    return ProviderEngineResult(
        "hipdnn",
        1,
        "success",
        role=role,
        engine_name=name,
        ootb=PlanResult(
            gpu_kernel_stats=stats,
            timing=TimingInfo(
                mode=mode,
                timer=timer,
                warmup_iters=10,
                samples=stats.n,
                first_call_ms=1.0,
                launch_mode=launch_mode,
            ),
            correctness=(CorrectnessResult(match, 1e-3, 1e-5) if validated else None),
        ),
    )


def _write_suite(tmp_path: Path, fname: str, graphs, **cfg) -> str:
    suite = SuiteResult(
        run=RunInfo(
            "t",
            [],
            {
                "cache_mode": "warm",
                "timing_block": 1,
                "input_recipe": "centered",
                **cfg,
            },
            complete=True,
        ),
        environment={"gpu_model": "MI210", "gpu_arch": "gfx90a"},
        graphs=[
            GraphResult(name, f"{name}.json", rows, engine_ids=[1], graph_id=gid)
            for name, gid, rows in graphs
        ],
    )
    out = tmp_path / fname
    suite.write(out)
    return str(out)


# --- P0: Compare compatibility, validation loopholes, and exact thresholds ---


def test_compare_excludes_mixed_timing_modes_by_default(tmp_path, capsys) -> None:
    a = _write_suite(
        tmp_path, "a.json", [("g", "id", [_row("E", 0.08, mode="staged")])]
    )
    b = _write_suite(
        tmp_path, "b.json", [("g", "id", [_row("E", 0.12, mode="events")])]
    )

    code = compare.main([a, b, "--json"])
    cap = capsys.readouterr()
    report = json.loads(cap.out)
    assert code == 0
    assert report["mixed_timing"] == 1
    assert report["pairs"][0]["in_geomean"] is False
    assert "mixed timing" in report["pairs"][0]["label"]
    assert "mixed timing" in cap.err

    # Explicit --allow-mismatch permits scoring the pair.
    code_allowed = compare.main([a, b, "--allow-mismatch", "--json"])
    report_allowed = json.loads(capsys.readouterr().out)
    assert code_allowed == 1
    assert report_allowed["pairs"][0]["label"] == "REGRESSION"


def test_compare_excludes_mixed_launch_modes(tmp_path, capsys) -> None:
    a = _write_suite(
        tmp_path, "a.json", [("g", "id", [_row("E", 0.08, launch_mode="raw")])]
    )
    b = _write_suite(
        tmp_path,
        "b.json",
        [("g", "id", [_row("E", 0.05, launch_mode="graph_replay")])],
    )

    code = compare.main([a, b, "--json"])
    report = json.loads(capsys.readouterr().out)
    assert code == 0
    assert report["mixed_timing"] == 1
    assert report["pairs"][0]["in_geomean"] is False
    assert "mixed launch" in report["pairs"][0]["label"]


def test_compare_excludes_mixed_passed_vs_unchecked(tmp_path, capsys) -> None:
    a = _write_suite(
        tmp_path, "a.json", [("g", "id", [_row("E", 1.0, validated=True, match=True)])]
    )
    b = _write_suite(
        tmp_path, "b.json", [("g", "id", [_row("E", 1.5, validated=False)])]
    )

    code = compare.main([a, b, "--json"])
    report = json.loads(capsys.readouterr().out)
    assert code == 0
    assert report["mixed_validation"] == 1
    assert report["pairs"][0]["in_geomean"] is False
    assert "mixed validation" in report["pairs"][0]["label"]


def test_compare_failed_only_b_reports_failed_and_strict_coverage_exits_1(
    tmp_path, capsys
) -> None:
    a = _write_suite(tmp_path, "a.json", [("g", "id", [_row("E", 6.0, match=True)])])
    b = _write_suite(tmp_path, "b.json", [("g", "id", [_row("E", 7.0, match=False)])])

    code = compare.main([a, b, "--json"])
    report = json.loads(capsys.readouterr().out)
    assert code == 0
    assert report["validation_failures"] == 1
    assert report["pairs"][0]["label"] == "B failed"
    assert report["pairs"][0]["in_geomean"] is False

    assert compare.main([a, b, "--strict-coverage", "--json"]) == 1
    capsys.readouterr()


def test_compare_zero_median_does_not_preempt_valid_sibling(tmp_path, capsys) -> None:
    a = _write_suite(tmp_path, "a.json", [("g", "id", [_row("E1", 0.9)])])
    b = _write_suite(
        tmp_path, "b.json", [("g", "id", [_row("ZERO", 0.0), _row("E1", 0.3)])]
    )

    code = compare.main([a, b, "--json"])
    report = json.loads(capsys.readouterr().out)
    assert code == 0
    assert report["pairs"][0]["engine_b"] == "E1"
    assert report["pairs"][0]["speedup"] == pytest.approx(3.0)
    assert report["pairs"][0]["in_geomean"] is True


def test_compare_exact_threshold_is_within_noise(tmp_path, capsys) -> None:
    # 2.1 / 2.0 - 1.0 == 0.050000000000000044 in binary float; must not regress at 5%.
    a = _write_suite(tmp_path, "a.json", [("g", "id", [_row("E", 2.0)])])
    b = _write_suite(tmp_path, "b.json", [("g", "id", [_row("E", 2.1)])])

    code = compare.main([a, b, "--threshold", "5", "--json"])
    report = json.loads(capsys.readouterr().out)
    assert code == 0
    assert report["pairs"][0]["label"] == "within noise"


def test_compare_missing_quartile_is_unknown_spread_not_zero(tmp_path, capsys) -> None:
    a = _write_suite(tmp_path, "a.json", [("g", "id", [_row("E", 2.0)])])
    b = _write_suite(
        tmp_path,
        "b.json",
        [("g", "id", [_row("E", 2.4, timings=[2.4] * 7 + [float("inf")] * 5)])],
    )

    code = compare.main([a, b, "--json"])
    report = json.loads(capsys.readouterr().out)
    assert code == 0
    assert report["unknown_spread"] == 1
    assert report["pairs"][0]["label"] == "unknown spread"
    assert report["pairs"][0]["in_geomean"] is False


def test_partial_missing_actual_output_fails_validation() -> None:
    class PartialBM:
        def get_output_tensor(self, uid):
            return None

        def get_output_data(self, uid):
            return np.zeros(4, np.float32) if uid == 1 else None

    infos = [
        TensorInfo(1, "o1", [4], [1], "float", is_virtual=False, is_output=True),
        TensorInfo(2, "o2", [4], [1], "float", is_virtual=False, is_output=True),
    ]
    refs = {
        1: ReferenceOutput(data=np.zeros(4, np.float32), tensor_uid=1),
        2: ReferenceOutput(data=np.zeros(4, np.float32), tensor_uid=2),
    }
    cfg = SuiteConfig(validation=ValidationConfig(provider="pytorch"))
    res = check_correctness(PartialBM(), infos, refs, "pytorch", cfg)
    assert res.tolerance_match is False
    assert "UID 2" in (res.error_message or "")


def test_reporter_refuses_vs_best_across_mixed_modes() -> None:
    import io

    out = io.StringIO()
    gr = GraphResult(
        "g",
        "/tmp/g.json",
        [_row("E1", 0.05, mode="staged"), _row("E2", 0.08, mode="events")],
        engine_ids=[1, 2],
    )
    Reporter(out, io.StringIO()).print_graph_table(gr)
    text = out.getvalue()
    assert "mixed modes: staged, events" in text
    assert "1.00x" not in text


# --- P1: AITER guards, AMD SMI handle/in-run telemetry, signal restore, fresh oracle ---


def _sdpa(**kw) -> DenseSdpa:
    defaults = dict(
        data_type="bfloat16",
        batch=1,
        num_query_heads=8,
        num_kv_heads=2,
        seqlen_q=1,
        seqlen_kv=2048,
        head_size_qk=192,
        head_size_v=128,
        causal=False,
        bottom_right=True,
        scale=0.125,
        q_uid=1,
        k_uid=2,
        v_uid=3,
        o_uid=4,
    )
    defaults.update(kw)
    return DenseSdpa(**defaults)


def test_aiter_opus_excludes_nonpositive_scale(monkeypatch) -> None:
    dev = aiter_sdpa.AiterDevice(arch="gfx950", num_cus=256, gluon=False)
    monkeypatch.setenv("AITER_ENABLE_FMHA_OPUS", "1")
    zero_128 = _sdpa(
        seqlen_q=128, seqlen_kv=128, head_size_qk=128, head_size_v=128, scale=0.0
    )
    names = [b.name for b in aiter_sdpa.backend_rows(zero_128, dev)]
    assert "opus" not in names
    assert "auto" not in names

    pos_128 = _sdpa(
        seqlen_q=128, seqlen_kv=128, head_size_qk=128, head_size_v=128, scale=0.125
    )
    pos_names = [b.name for b in aiter_sdpa.backend_rows(pos_128, dev)]
    assert "opus" in pos_names and "auto" in pos_names


def test_aiter_asymmetric_gqa_decode_excluded_from_ck_and_asm_v3() -> None:
    asym = _sdpa(
        seqlen_q=1, num_query_heads=8, num_kv_heads=2, head_size_qk=192, head_size_v=128
    )
    dev942 = aiter_sdpa.AiterDevice(arch="gfx942", num_cus=304, gluon=False)
    names942 = [b.name for b in aiter_sdpa.backend_rows(asym, dev942)]
    assert "ck" not in names942 and "asm_v3" not in names942 and "auto" not in names942

    # On gfx950 BF16 192/128, OPUS handles it cleanly; CK and ASM v3 remain excluded.
    dev950 = aiter_sdpa.AiterDevice(arch="gfx950", num_cus=256, gluon=False)
    names950 = [b.name for b in aiter_sdpa.backend_rows(asym, dev950)]
    assert "ck" not in names950 and "asm_v3" not in names950
    assert "opus" in names950 and "auto" in names950

    # Symmetric width or Sq=2 remains enabled.
    sym = _sdpa(seqlen_q=1, head_size_qk=128, head_size_v=128)
    assert "ck" in [b.name for b in aiter_sdpa.backend_rows(sym, dev942)]
    sq2 = _sdpa(seqlen_q=2, head_size_qk=192, head_size_v=128)
    assert "ck" in [b.name for b in aiter_sdpa.backend_rows(sq2, dev942)]


def test_amdsmi_unavailable_with_zero_handles(monkeypatch) -> None:
    fake = types.SimpleNamespace(amdsmi_get_processor_handles=lambda: [])
    monkeypatch.setattr(gpu_smi, "_amdsmi", lambda: fake)
    assert gpu_smi.is_amdsmi_available() is False


def test_unresolved_pre_init_amdsmi_handle_is_not_pinned_forever(monkeypatch) -> None:
    gpu_smi._handle_for.cache_clear()
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "1")
    fake = types.SimpleNamespace(
        amdsmi_get_processor_handles=lambda: ["h0", "h1"],
        amdsmi_get_gpu_device_bdf=lambda h: (
            "0000:09:00.0" if h == "h0" else "0000:0c:00.0"
        ),
    )
    monkeypatch.setattr(gpu_smi, "_amdsmi", lambda: fake)
    # Before HIP init, BDF is unavailable -> None, not cached permanently.
    monkeypatch.setattr(gpu_smi, "_hip_device_bdf", lambda _i: None)
    assert gpu_smi.GpuSmiProbe()._handle is None
    # After HIP init, BDF resolves -> h1!
    monkeypatch.setattr(gpu_smi, "_hip_device_bdf", lambda _i: "0000:0c:00")
    assert gpu_smi.GpuSmiProbe()._handle == "h1"
    gpu_smi._handle_for.cache_clear()


def test_gpu_smi_sample_during_detects_in_run_throttle() -> None:
    """Background telemetry polling catches mid-loop throttling deterministically."""
    import threading

    readings = [
        {
            "sclk_mhz": 1300.0,
            "mclk_mhz": 1600.0,
            "power_w": 750.0,
            "temp_hotspot_c": 80.0,
            "throttle_status": 4,
        },
        {
            "sclk_mhz": 1700.0,
            "mclk_mhz": 1600.0,
            "power_w": 200.0,
            "temp_hotspot_c": 65.0,
            "throttle_status": 0,
        },
    ]
    probe = gpu_smi.GpuSmiProbe.__new__(gpu_smi.GpuSmiProbe)
    probe._handle = "h0"
    idx = {"n": 0}
    sampled = threading.Event()

    def clocks():
        i = min(idx["n"], len(readings) - 1)
        idx["n"] += 1
        sampled.set()
        return readings[i]

    probe.clocks = clocks
    with probe.sample_during(interval_s=0.0) as telemetry:
        assert sampled.wait(timeout=2.0)
    assert telemetry["throttle_status"] == 0
    assert telemetry["throttle_any"] is True
    assert telemetry["power_max_w"] == 750.0
    assert telemetry["sclk_min_mhz"] == 1300.0
    assert suite_runner._throttled(telemetry) is True


def test_fresh_state_isolates_oracle_tuned_state(tmp_path) -> None:
    cfg = SuiteConfig(
        benchmark_work_dir=str(tmp_path / "work"),
        fresh_state=True,
        oracle=True,
    )
    fresh_root = tmp_path / "child_tmp"
    state = oracle._tuned_state(cfg, fresh_root)
    assert fresh_root in state.hipdnn_cache.parents
    assert (tmp_path / "work" / "tuned") not in state.hipdnn_cache.parents
    argv = oracle._pytorch_tuned_argv(tmp_path / "g.json", cfg, tmp_path / "out.json")
    assert "--fresh-state" in argv


_MINIMAL_GRAPH = {
    "name": "g",
    "compute_data_type": "float",
    "intermediate_data_type": "float",
    "io_data_type": "float",
    "tensors": [
        {"uid": 1, "name": "x", "dims": [4], "strides": [1], "data_type": "float"},
        {"uid": 2, "name": "y", "dims": [4], "strides": [1], "data_type": "float"},
    ],
    "nodes": [
        {
            "name": "relu",
            "type": "PointwiseAttributes",
            "inputs": {"in_0_tensor_uid": 1, "operation": "relu_fwd"},
            "outputs": {"out_0_tensor_uid": 2},
        }
    ],
}


def test_hipdnn_tuned_child_resolves_varlen_before_allocation(
    tmp_path, monkeypatch
) -> None:
    events: List[str] = []
    graph_path = tmp_path / "g.json"
    graph_path.write_text(json.dumps(_MINIMAL_GRAPH))
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    cache_dir = tmp_path / "cache"
    db_dir = tmp_path / "db"
    cache_dir.mkdir()
    db_dir.mkdir()
    monkeypatch.setenv("HIPDNN_FORCE_BENCHMARKING", "1")
    monkeypatch.setenv("HIPDNN_CACHE_DIR", str(cache_dir))
    monkeypatch.setenv("MIOPEN_USER_DB_PATH", str(db_dir))
    monkeypatch.setattr(hipdnn_tuned_child, "initialize_pip_rocm_runtime", lambda: None)
    monkeypatch.setattr(hipdnn_tuned_child, "set_plugin_path", lambda h, p: None)

    class FakeExec:
        build_time_ms = 1.0
        workspace_size = 0

        def __init__(self, *a, **kw):
            pass

        def prime(self, *a, **kw):
            pass

        def prepare(self, *a, **kw):
            pass

        def engine_knob_ids(self, eid):
            return []

        def benchmark(self, handle, vp):
            return timing.Measurement([1.0], [0.1], "staged", "hip", "warm", 1, 1.0)

    class FakeBM:
        def __init__(self, infos):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

        def allocate_all(self):
            events.append("allocate_all")

        def load_input_data(self, data):
            events.append("load_input_data")

        def zero_outputs(self):
            pass

        def create_variant_pack(self):
            return {}

    import sys

    monkeypatch.setitem(
        sys.modules, "hipdnn_frontend", types.SimpleNamespace(Handle=lambda: object())
    )
    monkeypatch.setattr(hipdnn_tuned_child, "Executor", FakeExec)
    monkeypatch.setattr(hipdnn_tuned_child, "BufferManager", FakeBM)
    monkeypatch.setattr(
        hipdnn_tuned_child,
        "generate_input_data",
        lambda *a, **kw: (events.append("generate_input_data"), {})[1],
    )
    monkeypatch.setattr(
        hipdnn_tuned_child,
        "resolve_varlen_layouts",
        lambda infos, gj, data: (events.append("resolve_varlen_layouts"), infos)[1],
    )

    rc = hipdnn_tuned_child.main(
        [
            "--graph",
            str(graph_path),
            "--engine",
            "1",
            "--seed",
            "0",
            "--timing-policy",
            json.dumps({"warmup_iters": 1, "iters": 1}),
            "--out-dir",
            str(out_dir),
        ]
    )
    assert rc == 0
    assert events.index("generate_input_data") < events.index("resolve_varlen_layouts")
    assert events.index("resolve_varlen_layouts") < events.index("allocate_all")


def test_signal_restore_skips_native_none_handlers(tmp_path, monkeypatch) -> None:
    """When rocprofv3 installs a native C signal handler, getsignal() returns None."""
    calls = []
    monkeypatch.setattr(signal, "getsignal", lambda sig: None)
    monkeypatch.setattr(signal, "signal", lambda sig, h: calls.append((sig, h)))
    monkeypatch.setattr(
        suite_runner_cli, "_arm_sigterm_watchdog", lambda: (lambda: None)
    )
    monkeypatch.setattr(
        suite_runner_cli, "collect_environment_info", lambda: {"gpu_arch": "gfx90a"}
    )
    gpath = tmp_path / "g.json"
    gpath.write_text(json.dumps(_MINIMAL_GRAPH))
    cfg = SuiteConfig()
    rc = suite_runner_cli._run_suite(
        [gpath],
        cfg,
        lambda p, gj, ti: GraphResult("g", str(p), [], engine_ids=[1]),
        None,
        Reporter(),
        ["dnn-benchmark"],
    )
    assert rc == 0
    assert calls == []


# --- Analytical FLOPs and timer resolution warnings ---


def test_packed_paged_sdpa_without_lengths_returns_none() -> None:
    graph = {
        "tensors": [
            {"uid": 1, "dims": [1, 1, 4, 128], "data_type": "half"},
            {"uid": 2, "dims": [2, 1, 8, 128], "data_type": "half"},
            {"uid": 3, "dims": [2, 1, 8, 128], "data_type": "half"},
            {"uid": 4, "dims": [1, 1, 4, 128], "data_type": "half"},
            {"uid": 5, "dims": [2, 1, 1, 1], "data_type": "int32"},
        ],
        "nodes": [
            {
                "type": "SdpaAttributes",
                "inputs": {
                    "q_tensor_uid": 1,
                    "k_tensor_uid": 2,
                    "v_tensor_uid": 3,
                    "page_table_k_tensor_uid": 5,
                },
                "outputs": {"o_tensor_uid": 4},
                "attributes": {"causal_mask": True},
            }
        ],
    }
    assert compute_flops(graph) is None


def test_missing_dims_scalar_and_known_zero_rates() -> None:
    assert tensor_dim_product({}) is None
    assert tensor_dim_product({"dims": []}) == 1
    assert tensor_dim_product({"dims": [0, 4]}) == 0
    assert ANALYTICAL_FORMULA_VERSION == 2

    # Missing dims on a node output makes whole-graph FLOPs None, never partial.
    graph = {
        "tensors": [
            {"uid": 1, "dims": [4, 4]},
            {"uid": 2, "dims": [4, 4]},
            {"uid": 3, "dims": [4, 4]},
            {"uid": 4},  # missing dims
        ],
        "nodes": [
            {
                "type": "MatmulAttributes",
                "inputs": {"a_tensor_uid": 1, "b_tensor_uid": 2},
                "outputs": {"c_tensor_uid": 3},
            },
            {
                "type": "PointwiseAttributes",
                "inputs": {"in_0_tensor_uid": 3},
                "outputs": {"out_0_tensor_uid": 4},
            },
        ],
    }
    assert compute_flops(graph) is None
    assert derive_throughputs(0, 0, 2.0) == (0.0, 0.0)


def test_zero_sample_emits_timer_resolution_warning() -> None:
    stats = BenchmarkStats.from_timings([0.0] + [0.016] * 19)
    assert any("timer resolution" in w for w in noise_warnings(stats))


# --- Reporter, DVC setup, AITER env flags, and runtime invariants ---


def test_reporter_nonfinite_iqr_and_launch_mode_legend() -> None:
    import io

    bad_stats = BenchmarkStats.from_timings([2.4] * 7 + [float("inf")] * 5)
    bad_row = ProviderEngineResult(
        "hipdnn",
        1,
        "success",
        engine_name="E1",
        ootb=PlanResult(
            gpu_kernel_stats=bad_stats,
            timing=TimingInfo(
                mode="staged",
                timer="hip",
                warmup_iters=5,
                samples=12,
                first_call_ms=1.0,
            ),
        ),
    )
    # Non-finite IQR must not raise TypeError in _outcome.
    text = Reporter._outcome(bad_row)
    assert "iqr" not in text and "unchecked" in text

    # Legend names launch_mode when rows mix raw vs graph_replay.
    out = io.StringIO()
    gr = GraphResult(
        "g",
        "/tmp/g.json",
        [
            _row("E1", 0.05, mode="staged", launch_mode="raw"),
            _row("E2", 0.04, mode="staged", launch_mode="graph_replay"),
        ],
        engine_ids=[1, 2],
    )
    Reporter(out, io.StringIO()).print_graph_table(gr)
    rendered = out.getvalue()
    assert "staged/raw" in rendered and "staged/graph_replay" in rendered
    assert "1.00x" not in rendered


def test_setup_env_sets_dvc_global_config_dir(tmp_path, monkeypatch) -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "setup_env_under_test",
        Path(__file__).resolve().parents[2] / "setup_env.py",
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    dvc_root = tmp_path / "rocm-libs"
    desc_dir = dvc_root / "desc"
    desc_dir.mkdir(parents=True)
    (desc_dir / "a.dvc").write_text("md5: 123\n")
    monkeypatch.setattr(mod, "ROCM_LIBRARIES_DIR", dvc_root)
    monkeypatch.setattr(mod, "PROVIDER_DESCRIPTOR_ROOT", desc_dir)

    captured: Dict[str, Any] = {}
    monkeypatch.setattr(mod, "run", lambda cmd, **kw: captured.update(cmd=cmd, **kw))

    monkeypatch.setattr(mod.Setup, "py", property(lambda self: "/usr/bin/python3"))
    setup = mod.Setup.__new__(mod.Setup)
    setup.workspace = tmp_path / "ws"
    setup.env = {}
    setup.pull_provider_artifacts()
    cfg_dir = Path(captured["env"]["DVC_GLOBAL_CONFIG_DIR"])
    assert (cfg_dir / "config").is_file()
    assert str(setup.workspace / "dvc-cache") in (cfg_dir / "config").read_text()


def test_aiter_opus_env_flags_scope_and_int_parsing(monkeypatch) -> None:
    """AITER_DISABLE_FMHA_OPUS gates 192/128 only; 128/128 follows AITER_ENABLE_FMHA_OPUS != 0."""
    dev = aiter_sdpa.AiterDevice(arch="gfx950", num_cus=256, gluon=False)
    p_192 = _sdpa(seqlen_q=128, seqlen_kv=128, head_size_qk=192, head_size_v=128)
    p_128 = _sdpa(
        seqlen_q=128, seqlen_kv=128, head_size_qk=128, head_size_v=128, scale=0.0
    )
    monkeypatch.delenv("AITER_DISABLE_FMHA_OPUS", raising=False)
    monkeypatch.delenv("AITER_ENABLE_FMHA_OPUS", raising=False)
    assert aiter_sdpa._auto_routes_to_opus(p_192, dev) is True
    # Non-zero integers other than "1" also disable 192/128 OPUS (int(...) != 0).
    monkeypatch.setenv("AITER_DISABLE_FMHA_OPUS", "2")
    assert aiter_sdpa._auto_routes_to_opus(p_192, dev) is False
    monkeypatch.setenv("AITER_DISABLE_FMHA_OPUS", "0")
    assert aiter_sdpa._auto_routes_to_opus(p_192, dev) is True
    # 128/128 depends only on AITER_ENABLE_FMHA_OPUS, even if DISABLE is set.
    monkeypatch.setenv("AITER_DISABLE_FMHA_OPUS", "1")
    monkeypatch.setenv("AITER_ENABLE_FMHA_OPUS", "2")
    assert aiter_sdpa._auto_routes_to_opus(p_128, dev) is True
    names = [b.name for b in aiter_sdpa.backend_rows(p_128, dev)]
    assert "auto" not in names and "opus" not in names


def test_per_graph_mode_unification_reruns_with_mixed_mode_reason(
    tmp_path, monkeypatch
) -> None:
    """Mixed staged/events rows trigger a graph rerun with MIXED_MODE_RERUN_REASON in timing and warnings."""
    calls: List[bool] = []
    emitted_warnings: List[str] = []
    monkeypatch.setattr(
        suite_runner, "warn_once", lambda key, msg: emitted_warnings.append(msg)
    )

    class FakeEventTimer:
        def __init__(self, *a, **kw):
            pass

        def start(self):
            pass

        def stop(self):
            pass

        def elapsed_ms(self):
            return 0.06

    monkeypatch.setattr(timing, "EventTimer", FakeEventTimer)
    monkeypatch.setattr(timing, "device_sync", lambda _t: None)

    def planner(gp, gj, ti, cfg, rep):
        def rows(ctx):
            calls.append(cfg.timing_policy.stall_gate)
            if cfg.timing_policy.stall_gate:
                return [
                    _row("E1", 0.05, mode="staged"),
                    _row("E2", 0.08, mode="events"),
                ]
            r1 = ProviderEngineResult("hipdnn", 1, "success", engine_name="E1")
            suite_runner._measure_row(
                r1,
                ctx,
                lambda: timing.measure(
                    lambda: None,
                    stream=0,
                    policy=cfg.timing_policy,
                    timer="hip",
                ),
            )
            r2 = _row("E2", 0.08, mode="events")
            r2.ootb.timing.fallback_reason = "host sync in enqueue"
            return [r1, r2]

        return suite_runner.RuntimePlan(
            "hipdnn", engine_ids=[1, 2], rows=rows, needs_inputs=True
        )

    gr = suite_runner.run_graph(
        tmp_path / "g.json",
        _MINIMAL_GRAPH,
        [],
        SuiteConfig(warmup_iters=1, benchmark_iters=2),
        Reporter(quiet=True),
        [planner],
    )
    assert calls == [True, False]
    assert [r.ootb.timing.mode for r in gr.results] == ["events", "events"]
    assert gr.results[0].ootb.timing.fallback_reason == timing.MIXED_MODE_RERUN_REASON
    assert gr.results[1].ootb.timing.fallback_reason == "host sync in enqueue"
    assert any(timing.MIXED_MODE_RERUN_REASON in w for w in gr.results[0].warnings)
    assert not any(timing.STALL_RERUN_REASON in w for w in gr.results[0].warnings)
    assert not any(timing.STALL_RERUN_REASON in w for w in emitted_warnings)


def test_suite_runner_cli_tuned_child_preserves_tunableop_env(
    tmp_path, monkeypatch
) -> None:
    """run_suite_cli(tuned_child=True) keeps PYTORCH_TUNABLEOP_* set by tuned_subprocess_env."""
    import os
    from dnn_benchmarking.cli.parser import create_parser

    gpath = tmp_path / "g.json"
    gpath.write_text(json.dumps(_MINIMAL_GRAPH))
    monkeypatch.setenv("PYTORCH_TUNABLEOP_ENABLED", "1")
    monkeypatch.setenv("PYTORCH_TUNABLEOP_TUNING", "1")
    seen_env: Dict[str, Any] = {}

    def fake_start_runtime(config, reporter, resources):
        seen_env["enabled"] = os.environ.get("PYTORCH_TUNABLEOP_ENABLED")
        seen_env["tuning"] = os.environ.get("PYTORCH_TUNABLEOP_TUNING")
        return lambda p, gj, ti: GraphResult("g", str(p), [], engine_ids=[0])

    monkeypatch.setattr(suite_runner_cli, "start_runtime", fake_start_runtime)
    monkeypatch.setattr(suite_runner_cli, "is_amdsmi_available", lambda: False)
    monkeypatch.setattr(
        suite_runner_cli, "collect_environment_info", lambda: {"gpu_arch": "gfx90a"}
    )
    args = create_parser().parse_args(
        [
            "--graph",
            str(gpath),
            "--runtime",
            "pytorch",
            "--warmup",
            "1",
            "--iters",
            "1",
            "--no-metrics",
            "--quiet",
            "--benchmark-work-dir",
            str(tmp_path / "work"),
        ]
    )
    rc = suite_runner_cli.run_suite_cli(
        args,
        [gpath],
        Reporter(quiet=True),
        argv=["dnn-benchmark"],
        tuned_child=True,
    )
    assert rc == 0
    assert seen_env == {"enabled": "1", "tuning": "1"}


def test_rocke_passes_after_iteration_scratch_release(monkeypatch) -> None:
    """RockeExecutor passes after_iteration scratch cleanup into measure()."""
    import contextlib
    import sys
    from dnn_benchmarking.execution import rocke_executor

    released: List[Any] = []
    fake_rt = types.SimpleNamespace(
        release_retained_for_stream=lambda s: released.append(s),
        wait_stream_and_release=lambda s: None,
        no_fence=contextlib.nullcontext,
    )
    monkeypatch.setitem(sys.modules, "rocke", types.SimpleNamespace(runtime=fake_rt))
    monkeypatch.setitem(sys.modules, "rocke.runtime", fake_rt)

    captured_hook = {}

    def fake_measure(launch, **kw):
        captured_hook["fn"] = kw.get("after_iteration")
        return timing.Measurement([1.0], [0.1], "staged", "torch", "warm", 1, 1.0)

    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(
            device=lambda _d: contextlib.nullcontext(),
            stream=lambda _s: contextlib.nullcontext(),
        )
    )
    monkeypatch.setattr(rocke_executor, "torch", fake_torch)
    monkeypatch.setattr(rocke_executor, "measure", fake_measure)

    ex = rocke_executor.RockeExecutor.__new__(rocke_executor.RockeExecutor)
    ex._launch = lambda: None
    ex._device = "cuda:0"
    ex._stream = types.SimpleNamespace(cuda_stream=99)
    ex._policy = TimingPolicy()
    ex._launch_mode = "raw"
    ex._timer = "torch"

    ex.benchmark()
    assert callable(captured_hook["fn"])
    captured_hook["fn"]()
    assert released == [99]


def test_pytorch_executor_surfaces_timed_work_notes_on_measurement(monkeypatch) -> None:
    """PyTorchCudaExecutor attaches handler timed_work_notes (e.g. varlen copy) to Measurement."""
    import contextlib
    from dnn_benchmarking.execution import pytorch_executor
    from dnn_benchmarking.execution.pytorch_ops import ReplayTensors

    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(
            device=lambda _d: contextlib.nullcontext(),
            stream=lambda _s: contextlib.nullcontext(),
        )
    )
    monkeypatch.setattr(pytorch_executor, "torch", fake_torch)

    def fake_measure(enqueue_fn, **kw):
        enqueue_fn()
        return timing.Measurement([0.5], [0.1], "staged", "torch", "warm", 1, 0.5)

    monkeypatch.setattr(pytorch_executor, "measure", fake_measure)
    ex = pytorch_executor.PyTorchCudaExecutor.__new__(
        pytorch_executor.PyTorchCudaExecutor
    )
    ex._device = "cuda:0"
    ex._stream = types.SimpleNamespace(cuda_stream=5, synchronize=lambda: None)
    ex._policy = TimingPolicy()
    ex._timer = "torch"
    ex._sdpa_backend_state = None
    ex.notes = []

    def fake_execute(tensors):
        if isinstance(tensors, ReplayTensors):
            tensors.timed_work_note("varlen_per_batch_copy")
            tensors.timed_work_note("varlen_per_batch_copy")  # deduplicated

    ex._execute_graph = fake_execute
    m = ex.benchmark({1: object()})
    assert m.timed_work_notes == ["varlen_per_batch_copy"]

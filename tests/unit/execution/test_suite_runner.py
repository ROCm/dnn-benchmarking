# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for execution.suite_runner, asserted on the GraphResult it returns."""

import io
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from dnn_benchmarking.cli import child_script, hipdnn_child, pytorch_child
from dnn_benchmarking.common.exceptions import ExecutionError, UnsupportedGraphError
from dnn_benchmarking.config.benchmark_config import (
    MetricsConfig,
    PyTorchSdpaBackendName,
    ReferencePrecision,
    SuiteConfig,
    ValidationConfig,
)
from dnn_benchmarking.execution import child_runner, suite_runner, timed_run
from dnn_benchmarking.execution.child_protocol import (
    HIPDNN_AUTOTUNE,
    HIPDNN_OOTB,
    PYTORCH_OOTB,
    output_file,
    write_result,
)
from dnn_benchmarking.execution.timing import Measurement, StallFallbackError
from dnn_benchmarking.graph.tensor_info import TensorInfo
from dnn_benchmarking.metrics._diagnostic import reset as reset_warnings
from dnn_benchmarking.reporting.reporter import Reporter
from dnn_benchmarking.reporting.suite_results import (
    engine_id_hex,
    graph_id_for,
    oracle_speedup,
)

GRAPH = {"name": "g", "nodes": [], "tensors": []}
PATH = Path("g.json")
REF = np.arange(4, dtype=np.float32)
NATIVE, FLOAT64 = ReferencePrecision.NATIVE, ReferencePrecision.FLOAT64


def _tensor(uid, is_output):
    return TensorInfo(
        uid=uid,
        name=f"t{uid}",
        dims=[4],
        strides=[1],
        data_type="float",
        is_virtual=False,
        is_output=is_output,
    )


TENSORS = [_tensor(1, False), _tensor(2, True)]


def _measurement(**kw):
    base = dict(
        kernel_ms=[1.0] * 10,
        host_ms=[0.01] * 10,
        mode="staged",
        timer="hip",
        cache_mode="warm",
        warmup_iters=2,
        first_call_ms=5.0,
    )
    return Measurement(**{**base, **kw})


def _stall_aware(f, key, policy):
    """Fail the stall gate for ``key`` while staging is allowed; once the
    runner disables it, time in events mode like ``timing.measure``."""
    if key in f.stall and policy.stall_gate:
        raise StallFallbackError("stall watchdog invalidated the measurement")
    if not policy.stall_gate:
        return _measurement(mode="events", fallback_reason="stall gate failed")
    return _measurement(**f.measurement)


class Fake:
    """Knobs shared by the fake executors, buffers and probes of one test."""

    def __init__(self):
        self.discovered = [1, 2]
        self.discover_error = None
        self.prepare_errors = {}
        self.bench_errors = {}
        self.measurement = {}
        self.engine_output = REF.copy()
        self.torch_error = None
        self.clocks = []
        self.events = []
        self.torch_options = []
        self.torch_buffer_devices = []
        self.torch_outputs_written = False
        self.policies = []
        self.knob_ids = {}  # engine id -> knob ids; default: benchmarking
        self.stall = set()  # engine ids (or 'pytorch') whose stall gate fails
        self.tuned_child = None  # fake run_capped for --tuned children
        self.child_envs = []  # env of each OOTB child launch


@pytest.fixture
def fake(monkeypatch):
    reset_warnings()
    f = Fake()

    class Executor:
        def __init__(self, graph_json_str, policy):
            f.policies.append(policy)
            self.policy = policy
            self.build_time_ms, self.workspace_size = 0.0, 0

        def discover_engines(self, handle):
            f.events.append("discover")
            if f.discover_error:
                raise f.discover_error
            return list(f.discovered)

        def prime(self, handle, engine_id):
            pass

        def prepare(self, handle, engine_id=None, knobs=None):
            self.engine_id = engine_id
            if engine_id in f.prepare_errors:
                raise f.prepare_errors[engine_id]
            self.build_time_ms = 2.0
            self.workspace_size = 64

        def engine_knob_ids(self, engine_id):
            knobs = f.knob_ids.get(engine_id, ["global.benchmarking"])
            if isinstance(knobs, Exception):
                raise knobs
            return list(knobs)

        def benchmark(self, handle, variant_pack):
            f.events.append(("benchmark", self.engine_id, self.policy.stall_gate))
            if self.engine_id in f.bench_errors:
                raise f.bench_errors[self.engine_id]
            return _stall_aware(f, self.engine_id, self.policy)

        def execute_once(self, handle, variant_pack):
            f.events.append("execute_once")

        def __del__(self):
            f.events.append("executor_freed")

    class BufferManager:
        def __init__(self, tensor_infos):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            f.events.append("buffers_freed")
            return False

        def allocate_all(self):
            pass

        def load_input_data(self, data):
            pass

        def zero_outputs(self):
            f.events.append("zero_outputs")

        def create_variant_pack(self):
            return {}

        def get_output_data(self, uid):
            f.events.append("read_output")
            return f.engine_output

        def get_output_tensors(self):
            return [t for t in TENSORS if t.is_output]

    class Probe:
        def clocks(self):
            return f.clocks.pop(0) if f.clocks else None

    class Loader:
        def load(self, path):
            return GRAPH, TENSORS

    untuned = {
        (f"dnn_benchmarking.cli.{spec.module}", tuple(spec.flags())): spec
        for spec in (HIPDNN_OOTB, HIPDNN_AUTOTUNE, PYTORCH_OOTB)
    }
    scripts = {
        "dnn_benchmarking.cli.hipdnn_child": hipdnn_child,
        "dnn_benchmarking.cli.pytorch_child": pytorch_child,
    }

    def run_capped(argv, timeout_s, env=None, capture_output=True):
        """Run the real untuned child script in this process, in its env."""
        module, args = argv[2], argv[3:]
        if "--tuned" in args:
            assert f.tuned_child is not None, "tuned child launched without fixture"
            return f.tuned_child(argv, timeout_s, env)
        f.child_envs.append(env)
        flags = tuple(a for a in args if a == "--autotune")
        child_env = untuned[(module, flags)].env
        with monkeypatch.context() as m:
            for name in [*child_env.fixed, *child_env.in_dir]:
                m.setenv(name, env[name])
            code = scripts[module].main(args)
        return SimpleNamespace(returncode=code)

    monkeypatch.setattr(suite_runner, "Executor", Executor)
    monkeypatch.setattr(timed_run, "Executor", Executor)
    monkeypatch.setattr(timed_run, "BufferManager", BufferManager)
    monkeypatch.setattr(child_script, "GpuSmiProbe", Probe)
    monkeypatch.setattr(hipdnn_child, "GraphLoader", Loader)
    monkeypatch.setattr(pytorch_child, "GraphLoader", Loader)
    monkeypatch.setattr(hipdnn_child, "initialize_pip_rocm_runtime", lambda: None)
    monkeypatch.setattr(child_runner, "run_capped", run_capped)
    monkeypatch.setitem(
        sys.modules,
        "hipdnn_frontend",
        SimpleNamespace(
            engine_id_to_name=lambda eid: "",
            Handle=SimpleNamespace,
            set_engine_plugin_paths=lambda paths, mode: None,
            PluginLoadingMode=SimpleNamespace(ABSOLUTE="abs"),
        ),
    )
    return f


@pytest.fixture
def fake_torch(fake, monkeypatch):
    """PyTorch executor/buffers and a CPU reference provider."""
    pytest.importorskip("torch")
    from dnn_benchmarking.execution import pytorch_buffer_manager, pytorch_executor

    class TorchExecutor:
        def __init__(
            self,
            graph_json,
            policy,
            *,
            pytorch_sdpa_backend,
            pytorch_rocm_fa_library=None,
        ):
            fake.policies.append(policy)
            self.policy = policy
            fake.torch_options.append((pytorch_sdpa_backend, pytorch_rocm_fa_library))
            self.device = "cuda:3"  # not the buffer manager's cuda:0 default
            self.notes = []

        def prepare(self):
            if fake.torch_error:
                raise fake.torch_error

        def benchmark(self, tensors):
            fake.torch_outputs_written = True
            return _stall_aware(fake, "pytorch", self.policy)

        def execute_once(self, tensors):
            fake.torch_outputs_written = True

    class TorchBuffers:
        def __init__(self, tensor_infos, device):
            fake.torch_buffer_devices.append(device)
            self._outputs = [t for t in tensor_infos if t.is_output]

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            fake.events.append("torch_buffers_freed")
            return False

        allocate_all = load_input_data = lambda self, *a: None

        def zero_outputs(self):
            fake.torch_outputs_written = False

        def get_tensors(self):
            return {t.uid: SimpleNamespace(is_cuda=False) for t in self._outputs}

        def get_output_tensors(self):
            return self._outputs

        def get_output_data(self, uid):
            # Zeros unless the graph ran since the last zero_outputs().
            return REF.copy() if fake.torch_outputs_written else np.zeros_like(REF)

    class CpuReference:
        """``estimated_bytes`` by precision: an int sets both."""

        estimated_bytes = 0
        float64 = False  # supports_float64
        float64_error = None  # raised by a float64 compute
        float64_output = REF.astype(np.float64)

        def supports_float64(self, graph_json):
            return self.float64

        def estimate_host_bytes(self, graph_json, precision=NATIVE):
            if isinstance(self.estimated_bytes, dict):
                return self.estimated_bytes[precision]
            return self.estimated_bytes

        def compute_reference(self, graph_json, input_data, *, precision, device=None):
            if precision is NATIVE:
                fake.events.append("cpu_reference")
                return {2: suite_runner.ReferenceOutput(data=REF.copy(), tensor_uid=2)}
            fake.events.append(("float64_reference", device))
            if self.float64_error is not None:
                raise self.float64_error
            return {
                2: suite_runner.ReferenceOutput(
                    data=self.float64_output.copy(), tensor_uid=2, precision="float64"
                )
            }

    fake.cpu_reference = CpuReference()

    monkeypatch.setattr(pytorch_executor, "PyTorchCudaExecutor", TorchExecutor)
    monkeypatch.setattr(
        pytorch_buffer_manager, "PyTorchCudaBufferManager", TorchBuffers
    )
    monkeypatch.setattr(
        suite_runner,
        "_reference_provider",
        lambda config, graph_json: (fake.cpu_reference, None),
    )
    monkeypatch.setattr(pytorch_child, "_float64_provider", lambda: fake.cpu_reference)
    return fake


class _Handle:
    """hipdnn.Handle stand-in."""

    def __init__(self, names=None):
        self.names = names or {1: "ENG_A", 2: "ENG_B"}

    def get_engine_info(self, eid):
        return SimpleNamespace(engine_name=self.names.get(eid, ""), version="2.1")


def _run(handle="default", **config):
    out = io.StringIO()
    graph = suite_runner.run_graph_all_providers(
        PATH,
        GRAPH,
        TENSORS,
        SuiteConfig(**config),
        _Handle() if handle == "default" else handle,
        Reporter(out),
    )
    return graph, out.getvalue()


def _validate(**config):
    return _run(validation=ValidationConfig(provider="pytorch"), **config)


@pytest.fixture
def reference(monkeypatch):
    """--validate with host reference outputs and no PyTorch row (no torch)."""

    def prepare(ctx):
        ctx.reference_outputs = {
            2: suite_runner.ReferenceOutput(data=REF.copy(), tensor_uid=2)
        }

    monkeypatch.setattr(suite_runner, "_prepare_references", prepare)


def test_one_hipdnn_row_per_engine_named_from_engine_info(fake):
    graph, progress = _run()

    assert graph.graph_id == graph_id_for(GRAPH)
    assert graph.status == "ok" and graph.error is None
    assert [
        (r.runtime, r.engine_id, r.engine_name, r.engine_version, r.verdict)
        for r in graph.results
    ] == [
        ("hipdnn", 1, "ENG_A", "2.1", "unchecked"),
        ("hipdnn", 2, "ENG_B", "2.1", "unchecked"),
    ]
    assert "ENG_A" in progress and "ENG_B" in progress


@pytest.mark.parametrize(
    "info_name, handle_name, registry_name, expected",
    [
        ("ENG_A", "H", "REG", "ENG_A"),
        # Plugin-supplied engines: only the handle knows the name.
        ("", "hipkernel:Attn", "", "hipkernel:Attn"),
        ("", IndexError, "REG", "REG"),
        ("", "", "", "0x0000000000000001"),
    ],
)
def test_engine_name_falls_back_through_handle_registry_then_hex(
    fake, monkeypatch, info_name, handle_name, registry_name, expected
):
    monkeypatch.setitem(
        sys.modules,
        "hipdnn_frontend",
        SimpleNamespace(engine_id_to_name=lambda eid: registry_name),
    )
    fake.discovered = [1]
    handle = _Handle({1: info_name})

    def handle_lookup(eid):
        if handle_name is IndexError:
            raise IndexError(eid)
        return handle_name

    handle.engine_id_to_name = handle_lookup

    graph, progress = _run(handle=handle)

    assert graph.results[0].engine_name == expected
    assert "WARNING" not in progress


@pytest.mark.parametrize(
    "stage, error, status, message",
    [
        ("prepare", UnsupportedGraphError("no plan"), "skipped", "no plan"),
        ("prepare", ExecutionError("build"), "error", "ExecutionError: build"),
        ("bench", RuntimeError("hip"), "error", "RuntimeError: hip"),
    ],
)
def test_engine_failure_is_isolated_to_its_row(fake, stage, error, status, message):
    getattr(fake, f"{stage}_errors")[1] = error

    failed, ok = _run()[0].results

    assert (
        failed.status,
        failed.verdict,
        failed.error_message or failed.skip_reason,
    ) == (status, status, message)
    assert failed.engine_name == "ENG_A"
    assert failed.ootb.gpu_kernel_stats is None and failed.ootb.correctness is None
    assert ok.status == "success"


@pytest.mark.parametrize("outcome", ["skip", "stall"])
def test_handled_child_failures_print_no_error_line(fake, capsys, outcome):
    """A skip or a stall-gate failure is a normal outcome: the child reports
    it only to the parent, which turns it into a skipped row or a rerun."""
    if outcome == "skip":
        fake.prepare_errors[1] = UnsupportedGraphError("no plan")
    else:
        fake.stall = {1}

    graph, _ = _run()

    assert graph.results[0].status == ("skipped" if outcome == "skip" else "success")
    assert "ERROR" not in capsys.readouterr().err


def test_row_timing_throughput_and_noise_from_the_measurement(fake, monkeypatch):
    monkeypatch.setattr(suite_runner, "compute_flops", lambda g: 2_000_000_000)
    fake.discovered = [1]
    # Median 1 ms, mean 1.4 ms: throughput must come from the median.
    # Every timing field off its default, so a row cannot claim a cache mode,
    # block size or cap its measurement did not use.
    fake.measurement = dict(
        kernel_ms=[1.0] * 9 + [5.0],
        capped=True,
        mode="events",
        timer="torch",
        cache_mode="cold",
        warmup_iters=7,
        timing_block=4,
        fallback_reason="no stream wait",
    )

    row = _run()[0].results[0]

    assert row.ootb.gpu_kernel_stats.median_ms == pytest.approx(1.0)
    assert row.ootb.host_stats.median_ms == pytest.approx(0.01)
    assert row.ootb.derived_tflops_per_s == pytest.approx(2.0)
    assert row.ootb.derived_gbytes_per_s == pytest.approx(32 / 1e-3 / 1e9)
    assert row.to_dict()["ootb"]["timing"] == {
        "mode": "events",
        "timer": "torch",
        "warmup_iters": 7,
        "samples": 10,  # the loop's sample count, beyond --iters when extended
        "first_call_ms": 5.0,
        "capped": True,
        "fallback_reason": "no stream wait",
    }
    assert row.ootb.workspace_bytes == 64
    assert row.ootb.cpu_build_time_ms == 2.0
    warnings = " | ".join(row.warnings)
    for expected in (
        "outlier: max 5.0x median",
        "capped at max_iters",
        "events timing: no stream wait",
    ):
        assert expected in warnings


@pytest.fixture
def tuned_child(fake, monkeypatch):
    """Fake tuned-hipDNN child: 7 ms build, 0.5 ms kernels, output ``output``.

    Records each launch's engine (hex) in ``engines`` and in ``fake.events``,
    and its timing policy's ``stall_gate`` in ``stall_gates``.
    """
    child = SimpleNamespace(engines=[], stall_gates=[], output=REF.copy())

    def run_capped(argv, timeout_s, env=None):
        args = hipdnn_child.parse_args(argv[3:])
        engine = engine_id_hex(args.engine)
        child.engines.append(engine)
        child.stall_gates.append(args.timing_policy.stall_gate)
        fake.events.append(("tuned_child", engine))
        m = _measurement(kernel_ms=[0.5] * 4, host_ms=[0.01] * 4)
        write_result(args.out_dir, m, build_ms=7.0, workspace_bytes=64)
        if args.save_outputs:
            np.save(output_file(args.out_dir, 2), child.output)
        return SimpleNamespace(returncode=0)

    fake.tuned_child = run_capped
    return child


def test_every_executor_gets_the_run_policy(fake, tuned_child):
    config = dict(warmup_iters=3, benchmark_iters=5, cache_mode="cold")

    _run(oracle=True, **config)

    # Discovery, then per engine the prime and the OOTB plan; the tuned run
    # is a child process.
    assert fake.policies == [SuiteConfig(**config).timing_policy] * 5


def test_oracle_tunes_each_engine_in_a_child_after_its_row_is_released(
    fake, reference, tuned_child
):
    """The tuned child runs once per engine, after the OOTB plan is timed and
    validated and its buffers and workspace are released (tuning state must
    not reach the OOTB numbers; held VRAM would stack on the child's)."""
    graph, _ = _validate(oracle=True)

    assert tuned_child.engines == [engine_id_hex(1), engine_id_hex(2)]
    for eid in (1, 2):
        at = fake.events.index(("tuned_child", engine_id_hex(eid)))
        before = fake.events[:at]
        assert ("benchmark", eid, True) in before
        assert before[-3:] == ["read_output", "buffers_freed", "executor_freed"]
    for row in graph.results:
        assert row.status == "success" and row.oracle_error is None
        assert (row.ootb.cpu_build_time_ms, row.oracle.cpu_build_time_ms) == (2.0, 7.0)
        assert row.oracle.gpu_kernel_stats.median_ms == pytest.approx(0.5)
        # The parent compares the child's saved outputs with its references.
        assert row.ootb.correctness.passed and row.oracle.correctness.passed
        assert oracle_speedup(row) == pytest.approx(2.0)


@pytest.mark.parametrize(
    "autotune, shell, expected", [(False, "1", "0"), (True, "0", "1")]
)
def test_engine_row_benchmarks_only_with_autotune(
    fake, monkeypatch, autotune, shell, expected
):
    """The command line alone decides whether a row benchmarks its candidate
    kernels: the child spec overrides a shell value."""
    monkeypatch.setenv("HIPDNN_FORCE_BENCHMARKING", shell)

    graph, _ = _run(autotune=autotune)

    assert {r.status for r in graph.results} == {"success"}
    assert {env["HIPDNN_FORCE_BENCHMARKING"] for env in fake.child_envs} == {expected}


def test_every_gpu_run_starts_with_its_own_empty_miopen_user_db(
    fake, tuned_child, monkeypatch
):
    """Each OOTB and tuned run gets a new, empty MIOpen user database that no
    other run shares and that is deleted after it. MIOpen keys its records
    without the deterministic attribute, so a shared database lets one
    engine's Find select another engine's kernel."""
    launches = []
    inner = child_runner.run_capped

    def record(argv, timeout_s, env=None, capture_output=True):
        db = Path(env["MIOPEN_USER_DB_PATH"])
        launches.append((db, sorted(p.name for p in db.iterdir())))
        return inner(argv, timeout_s, env)

    monkeypatch.setattr(child_runner, "run_capped", record)

    _run(oracle=True)

    dbs = [db for db, _ in launches]
    assert len(dbs) == 4 and len(set(dbs)) == 4  # 2 engines x (OOTB, tuned)
    assert all(contents == [] for _, contents in launches)
    assert not any(db.exists() for db in dbs)


def test_tuning_available_follows_each_engines_knobs(fake, tuned_child):
    """hipDNN ignores a knob the engine does not expose; its tuned run then
    re-measures the OOTB configuration and must say so."""
    fake.knob_ids = {2: ["miopen.find_mode"]}

    graph, _ = _run(oracle=True)

    assert [r.oracle.tuning_available for r in graph.results] == [True, False]
    # No references: the tuned run is not validated either.
    assert [r.oracle.correctness for r in graph.results] == [None, None]


def test_failed_knob_query_is_an_oracle_error_without_a_tuned_run(fake, tuned_child):
    fake.knob_ids = {1: ExecutionError("no knobs")}

    e1, e2 = _run(oracle=True)[0].results

    assert tuned_child.engines == [engine_id_hex(2)]
    assert (e1.status, e1.oracle) == ("success", None)
    assert e1.oracle_error and e1.ootb.gpu_kernel_stats is not None
    assert e2.oracle is not None and e2.oracle_error is None


def test_tuned_child_failing_validation_publishes_no_speedup(
    fake, reference, tuned_child
):
    """A wrong-but-fast tuned plan keeps its own failed verdict and the OOTB
    row keeps its pass."""
    tuned_child.output = REF + 1.0

    graph, _ = _validate(oracle=True, engine_filter=[1])

    (row,) = graph.results
    assert (row.status, row.verdict) == ("success", "passed")
    assert row.oracle.correctness.explicitly_failed
    assert oracle_speedup(row) is None


def test_duplicate_engine_with_a_failed_knob_query_gets_no_tuned_run(fake, tuned_child):
    """Duplicate selections are separate rows: the second row's failed knob
    query must not inherit the first row's tuning availability."""

    class FailsSecondTime:
        calls = 0

        def __iter__(self):
            FailsSecondTime.calls += 1
            if FailsSecondTime.calls > 1:
                raise ExecutionError("no knobs")
            return iter(["global.benchmarking"])

    fake.knob_ids = {1: FailsSecondTime()}

    first, second = _run(oracle=True, engine_filter=[1, 1])[0].results

    assert tuned_child.engines == [engine_id_hex(1)]
    assert first.oracle is not None and first.oracle_error is None
    assert (second.status, second.oracle) == ("success", None)
    assert second.oracle_error


def test_no_metrics_skips_probes_and_throughput(fake, monkeypatch):
    monkeypatch.setattr(suite_runner, "compute_flops", lambda g: 2_000_000_000)
    fake.discovered = [1]
    fake.clocks = [{"sclk_mhz": 1700}]

    row = _run(metrics=MetricsConfig(basic=False))[0].results[0]

    assert row.ootb.gpu_kernel_stats is not None
    assert row.clocks_after is None and row.ootb.derived_tflops_per_s is None


CLOCK = {"sclk_mhz": 1700.0, "throttle_status": 0}


@pytest.mark.parametrize(
    "after, throttled",
    [
        (CLOCK, False),
        ({**CLOCK, "sclk_mhz": 1400.0}, False),  # DPM drop, not throttling
        ({**CLOCK, "throttle_status": 4}, True),
        (None, False),
    ],
)
def test_clocks_after_the_timed_loop_flag_throttling(fake, after, throttled):
    fake.discovered = [1]
    fake.clocks = [after]

    row = _run()[0].results[0]

    assert row.clocks_after == after
    assert ("throttled" in row.warnings) is throttled


def test_unsupported_graph_has_no_engines_and_keeps_the_reason(fake):
    fake.discover_error = UnsupportedGraphError("nothing applies")

    graph, _ = _run()

    assert graph.status == "no_engines"
    assert graph.results == [] and graph.error is None
    assert graph.message == "nothing applies"


def test_unsupported_graph_still_gets_the_pytorch_reference_row(fake_torch):
    fake_torch.discover_error = UnsupportedGraphError("nothing applies")

    graph, _ = _validate()

    assert graph.status == "no_engines"
    assert [(r.runtime, r.role, r.engine_id, r.verdict) for r in graph.results] == [
        ("pytorch", "reference", None, "reference")
    ]
    assert graph.results[0].ootb.gpu_kernel_stats is not None


@pytest.mark.parametrize(
    "patch, error",
    [
        ("discover", "Engine discovery failed: ExecutionError: driver"),
        ("inputs", "Input data generation failed: ValueError: bad dims"),
    ],
)
def test_graph_level_failure_sets_error_without_rows(fake, monkeypatch, patch, error):
    if patch == "discover":
        fake.discover_error = ExecutionError("driver")
    else:

        def fail(*a):
            raise ValueError("bad dims")

        monkeypatch.setattr(suite_runner, "generate_input_data", fail)

    graph, _ = _run()

    assert graph.status == "error" and graph.error == error
    assert graph.results == []


@pytest.mark.parametrize(
    "engine_output, verdict",
    [(REF.copy(), "passed"), (REF + 1.0, "failed")],
)
def test_engines_are_validated_against_the_timed_reference(
    fake_torch, engine_output, verdict
):
    fake_torch.discovered = [1]
    fake_torch.engine_output = engine_output

    reference, engine = _validate()[0].results

    assert (reference.role, reference.verdict) == ("reference", "reference")
    assert engine.verdict == verdict
    assert engine.ootb.correctness.n_total == 4


def test_missing_reference_fails_validation_with_the_reason(fake, monkeypatch):
    reason = "Reference provider 'pytorch' does not support this graph"
    monkeypatch.setattr(
        suite_runner, "_reference_provider", lambda c, g: (None, reason)
    )
    fake.discovered = [1]

    (engine,) = _validate()[0].results

    assert engine.verdict == "failed"
    assert engine.ootb.correctness.error_message == reason


@pytest.mark.parametrize(
    "sdpa_backend, reference_status, engine_verdict, engine_reason",
    [
        # CPU reference serves as fallback
        ("default", "skipped", "passed", None),
        # strict selection never falls back, and says why
        ("math", "error", "failed", "ExecutionError: PyTorch GPU not available"),
    ],
)
def test_failed_timed_reference(
    fake_torch, sdpa_backend, reference_status, engine_verdict, engine_reason
):
    fake_torch.discovered = [1]
    fake_torch.torch_error = ExecutionError("PyTorch GPU not available")

    reference, engine = _validate(pytorch_sdpa_backend=sdpa_backend)[0].results

    assert reference.status == reference_status
    assert (reference.error_message or reference.skip_reason) == (
        "ExecutionError: PyTorch GPU not available"
    )
    assert engine.verdict == engine_verdict
    assert engine.ootb.correctness.error_message == engine_reason


def test_cpu_reference_failure_fails_validation_and_keeps_engine_rows(
    fake_torch, monkeypatch
):
    class FailingReference:
        def supports_float64(self, graph_json):
            return False

        def estimate_host_bytes(self, graph_json, precision=NATIVE):
            return 0

        def compute_reference(self, graph_json, input_data, **kwargs):
            raise RuntimeError("cpu reference exploded")

    monkeypatch.setattr(
        suite_runner, "_reference_provider", lambda c, g: (FailingReference(), None)
    )
    fake_torch.torch_error = ExecutionError("PyTorch GPU not available")

    graph, _ = _validate()

    assert graph.error is None
    reference, *engines = graph.results
    assert reference.status == "skipped"
    assert [
        (e.engine_id, e.verdict, e.ootb.correctness.error_message) for e in engines
    ] == [
        (1, "failed", "cpu reference exploded"),
        (2, "failed", "cpu reference exploded"),
    ]


def test_cpu_reference_too_large_for_host_memory_leaves_rows_unchecked(
    fake_torch, monkeypatch
):
    """A CPU reference that would exhaust host memory is not run, and its
    absence is not a validation failure."""
    import psutil

    monkeypatch.setattr(
        psutil, "virtual_memory", lambda: SimpleNamespace(available=10 << 30)
    )
    fake_torch.torch_error = ExecutionError("PyTorch GPU not available")
    fake_torch.cpu_reference.estimated_bytes = 9 << 30  # over 80% of 10 GiB

    graph, _ = _validate()

    _, *engines = graph.results
    assert "cpu_reference" not in fake_torch.events
    assert [e.verdict for e in engines] == ["unchecked", "unchecked"]
    assert all("host memory" in e.ootb.correctness.error_message for e in engines)


def test_cpu_reference_that_fits_in_host_memory_runs(fake_torch, monkeypatch):
    import psutil

    monkeypatch.setattr(
        psutil, "virtual_memory", lambda: SimpleNamespace(available=10 << 30)
    )
    fake_torch.torch_error = ExecutionError("PyTorch GPU not available")
    fake_torch.cpu_reference.estimated_bytes = 7 << 30

    graph, _ = _validate()

    _, *engines = graph.results
    assert "cpu_reference" in fake_torch.events
    assert [e.verdict for e in engines] == ["passed", "passed"]


@pytest.mark.parametrize("unknown", ["estimate", "available_memory"])
def test_cpu_reference_runs_when_host_memory_cannot_be_judged(
    fake_torch, monkeypatch, unknown
):
    """Without an estimate or a memory reading the reference runs as before."""
    import psutil

    def virtual_memory():
        if unknown == "available_memory":
            raise RuntimeError("no /proc/meminfo")
        return SimpleNamespace(available=10 << 30)

    monkeypatch.setattr(psutil, "virtual_memory", virtual_memory)
    fake_torch.torch_error = ExecutionError("PyTorch GPU not available")
    fake_torch.cpu_reference.estimated_bytes = (
        None if unknown == "estimate" else 9 << 30
    )

    graph, _ = _validate()

    _, *engines = graph.results
    assert "cpu_reference" in fake_torch.events
    assert [e.verdict for e in engines] == ["passed", "passed"]


@pytest.mark.parametrize(
    "supported, precision, verdict, used",
    [
        (True, "float64", "passed", "float64"),
        (True, "native", "failed", "native"),
        (False, "float64", "failed", "native"),
    ],
)
def test_engines_compare_against_the_float64_reference_of_a_supported_graph(
    fake_torch, supported, precision, verdict, used
):
    """The engine output equals the float64 reference and is 1.0 off the
    native one, so the verdict shows which reference the engine met."""
    fake_torch.discovered = [1]
    fake_torch.cpu_reference.float64 = supported
    fake_torch.cpu_reference.float64_output = (REF + 1.0).astype(np.float64)
    fake_torch.engine_output = REF + 1.0

    graph, _ = _run(
        validation=ValidationConfig(provider="pytorch", precision=precision)
    )

    reference, engine = graph.results
    assert engine.verdict == verdict
    assert engine.ootb.correctness.reference_precision == used
    float64_calls = [e for e in fake_torch.events if e[0] == "float64_reference"]
    # The reference child computes the float64 reference on its GPU.
    assert float64_calls == (
        [("float64_reference", "cuda:3")] if used == "float64" else []
    )
    assert not reference.warnings


def test_failed_float64_reference_keeps_the_native_reference_with_a_note(fake_torch):
    fake_torch.discovered = [1]
    fake_torch.cpu_reference.float64 = True
    fake_torch.cpu_reference.float64_error = RuntimeError("HIP out of memory")

    graph, _ = _validate()

    reference, engine = graph.results
    assert (engine.verdict, engine.ootb.correctness.reference_precision) == (
        "passed",
        "native",
    )
    (note,) = reference.warnings
    assert "HIP out of memory" in note


@pytest.mark.parametrize("float64_fits", [True, False])
def test_cpu_reference_is_float64_when_it_fits_in_host_memory(
    fake_torch, monkeypatch, float64_fits
):
    import psutil

    monkeypatch.setattr(
        psutil, "virtual_memory", lambda: SimpleNamespace(available=10 << 30)
    )
    fake_torch.torch_error = ExecutionError("PyTorch GPU not available")
    fake_torch.cpu_reference.float64 = True
    fake_torch.cpu_reference.estimated_bytes = {
        NATIVE: 1 << 30,
        FLOAT64: (7 if float64_fits else 9) << 30,  # 80% of 10 GiB is the limit
    }

    graph, _ = _validate()

    reference, *engines = graph.results
    assert [e.verdict for e in engines] == ["passed", "passed"]
    used = {e.ootb.correctness.reference_precision for e in engines}
    if float64_fits:
        assert fake_torch.events.count(("float64_reference", None)) == 1
        assert "cpu_reference" not in fake_torch.events
        assert used == {"float64"}
    else:
        assert ("float64_reference", None) not in fake_torch.events
        assert "cpu_reference" in fake_torch.events
        assert used == {"native"}
        assert any("host memory" in note for note in reference.warnings)


def test_failed_cpu_float64_reference_falls_back_to_the_native_cpu_reference(
    fake_torch, monkeypatch
):
    import psutil

    monkeypatch.setattr(
        psutil, "virtual_memory", lambda: SimpleNamespace(available=10 << 30)
    )
    fake_torch.torch_error = ExecutionError("PyTorch GPU not available")
    fake_torch.cpu_reference.float64 = True
    fake_torch.cpu_reference.float64_error = MemoryError("host allocation failed")
    fake_torch.cpu_reference.estimated_bytes = 1 << 30

    graph, _ = _validate()

    reference, *engines = graph.results
    assert ("float64_reference", None) in fake_torch.events
    assert "cpu_reference" in fake_torch.events
    assert [e.verdict for e in engines] == ["passed", "passed"]
    assert {e.ootb.correctness.reference_precision for e in engines} == {"native"}
    assert any("host allocation failed" in note for note in reference.warnings)


def test_per_engine_handle_failure_is_an_error_row(fake, monkeypatch):
    def no_handle():
        raise RuntimeError("plugin load failed")

    monkeypatch.setitem(
        sys.modules,
        "hipdnn_frontend",
        SimpleNamespace(
            PluginLoadingMode=SimpleNamespace(ABSOLUTE="abs"),
            set_engine_plugin_paths=lambda paths, mode: None,
            Handle=no_handle,
        ),
    )

    graph, _ = _run(
        handle=None, engine_filter=[1, 1], plugin_paths=[Path("/a"), Path("/b")]
    )

    assert [(r.status, r.plugin_path, r.error_message) for r in graph.results] == [
        ("error", str(Path("/a")), "RuntimeError: plugin load failed"),
        ("error", str(Path("/b")), "RuntimeError: plugin load failed"),
    ]


def test_per_engine_handle_loads_only_that_rows_plugin(fake, monkeypatch):
    """-e 1,1 --plugin-path /a,/b: each row's handle, and the handle of its
    OOTB child, is created right after that row's plugin path is set, so an
    A/B row is not silently the default set."""
    calls = []

    def make_handle():
        calls.append("Handle")
        return _Handle({1: f"ENG_{len(calls)}"})

    monkeypatch.setitem(
        sys.modules,
        "hipdnn_frontend",
        SimpleNamespace(
            PluginLoadingMode=SimpleNamespace(ABSOLUTE="abs"),
            set_engine_plugin_paths=lambda paths, mode: calls.append((paths, mode)),
            Handle=make_handle,
        ),
    )
    a, b = Path("/a"), Path("/b")

    graph, _ = _run(handle=None, engine_filter=[1, 1], plugin_paths=[a, b])

    # Per row: the parent's handle (engine name), then the OOTB child's.
    row_a = [([str(a)], "abs"), "Handle"] * 2
    row_b = [([str(b)], "abs"), "Handle"] * 2
    assert calls == row_a + row_b
    assert [(r.status, r.plugin_path, r.engine_name) for r in graph.results] == [
        ("success", str(a), "ENG_2"),
        ("success", str(b), "ENG_6"),
    ]


def test_default_run_skips_oracle_and_profiling(fake, tuned_child, monkeypatch):
    """Without --oracle or a profiling flag, no tuned child and no profiler
    child is spawned."""
    from dnn_benchmarking.metrics import profiling_orchestrator

    called = []
    monkeypatch.setattr(
        profiling_orchestrator,
        "run_profiling_passes",
        lambda **kw: called.append("profile"),
    )

    graph, _ = _run()

    assert called == [] and tuned_child.engines == []
    assert [(r.status, r.oracle, r.extra_metrics) for r in graph.results] == [
        ("success", None, None)
    ] * 2


def test_pytorch_run_options_reach_the_timed_executor(fake_torch):
    fake_torch.discovered = [1]
    config = dict(warmup_iters=3, benchmark_iters=5, cache_mode="cold")

    _validate(
        pytorch_sdpa_backend="flash", pytorch_rocm_fa_library="aotriton", **config
    )

    assert fake_torch.torch_options == [(PyTorchSdpaBackendName.FLASH, "aotriton")]
    # Discovery, the PyTorch reference row, then the engine's prime and plan.
    assert fake_torch.policies == [SuiteConfig(**config).timing_policy] * 4


def test_reference_warnings_go_on_the_reference_row_only(fake_torch, monkeypatch):
    from dnn_benchmarking.execution import pytorch_ops

    monkeypatch.setattr(
        pytorch_ops, "get_reference_warnings", lambda graph_json: ["manual op"]
    )
    fake_torch.discovered = [1]

    reference, engine = _validate()[0].results

    assert "manual op" in reference.warnings
    assert "manual op" not in (engine.warnings or [])


def test_profiling_runs_after_teardown_with_the_rows_payload(fake, monkeypatch):
    """The child gets the row's engine, plugin, seed (same inputs), warmup and
    metric flags, and runs only after the row's buffers and workspace are
    released (holding them would roughly double peak VRAM)."""
    from dnn_benchmarking.metrics import profiling_orchestrator

    calls = []

    def run_profiling_passes(**kw):
        fake.events.append("profile")
        calls.append(kw)
        return {"perf": {"cycles": kw["engine_name"]}}

    monkeypatch.setattr(
        profiling_orchestrator, "run_profiling_passes", run_profiling_passes
    )
    metrics = MetricsConfig(perf=True, pmc_set="basic", profiling_timeout_s=42.0)

    graph, progress = _run(
        seed=7,
        warmup_iters=3,
        metrics=metrics,
        engine_filter=[2],
        plugin_paths=[Path("/a")],
    )

    assert graph.results[0].extra_metrics == {"perf": {"cycles": "ENG_B"}}
    assert "profiling ENG_B" in progress
    assert calls == [
        dict(
            graph_path=PATH,
            engine_id=2,
            engine_name="ENG_B",
            seed=7,
            warmup_iters=3,
            metrics_config=metrics,
            plugin_path=Path("/a"),
            autotune=False,
        )
    ]
    assert fake.events[-3:] == ["buffers_freed", "executor_freed", "profile"]


def test_validation_reads_outputs_zeroed_after_the_timed_loop(fake, reference):
    """Outputs left by the timed loop must not satisfy the reference check."""
    _validate(engine_filter=[1])

    start = fake.events.index("zero_outputs")
    assert fake.events[start : start + 5] == [
        "zero_outputs",
        ("benchmark", 1, True),
        "zero_outputs",
        "execute_once",
        "read_output",
    ]


def test_explicit_engines_run_in_caller_order_without_discovery(fake):
    fake.discovered = [1, 2]

    graph, _ = _run(engine_filter=[2, 9, 1])

    assert graph.engine_ids == [2, 9, 1]
    assert [r.engine_id for r in graph.results] == [2, 9, 1]
    assert "discover" not in fake.events


def test_stall_failure_remeasures_every_engine_of_the_graph_unstalled(fake):
    """One engine's watchdog release must not leave a graph with staged and
    events rows side by side: every row is remeasured without stalling."""
    fake.stall = {2}

    graph, progress = _run()

    benchmarks = [e for e in fake.events if isinstance(e, tuple)]
    assert benchmarks == [
        ("benchmark", 1, True),
        ("benchmark", 2, True),
        ("benchmark", 1, False),
        ("benchmark", 2, False),
    ]
    assert [r.engine_id for r in graph.results] == [1, 2]
    assert [r.status for r in graph.results] == ["success", "success"]
    assert {r.ootb.timing.mode for r in graph.results} == {"events"}
    assert all("stall gate failed" in " ".join(r.warnings) for r in graph.results)
    assert "remeasuring every row of this graph without stalling" in progress


def test_pytorch_buffers_share_the_executor_device(fake_torch):
    """Buffers on another device than the executor would make every
    PyTorch row fail or silently copy across GPUs."""
    _validate()

    assert fake_torch.torch_buffer_devices == ["cuda:3"]


def test_stall_rerun_measures_ootb_for_every_engine_and_tunes_again(fake, tuned_child):
    """Tuning state lives in the child, so a whole-graph rerun after engine 2
    stalled times every engine's OOTB plan again, and each row gets the tuned
    run of the attempt it is reported from."""
    fake.stall = {2}

    graph, _ = _run(oracle=True)

    assert [(r.status, r.ootb.timing.mode) for r in graph.results] == [
        ("success", "events")
    ] * 2
    assert all(r.oracle is not None for r in graph.results)
    # Engine 1 tuned before engine 2 stalled in its OOTB loop; the rerun
    # tunes both.
    assert tuned_child.engines == [engine_id_hex(e) for e in (1, 1, 2)]
    # The tuned runs of the rerun time unstalled too, like their rows.
    assert tuned_child.stall_gates == [True, False, False]


def test_stall_failure_on_the_reference_reruns_the_graph(fake_torch):
    fake_torch.stall = {"pytorch"}

    graph, _ = _validate()

    assert [r.role for r in graph.results] == ["reference", "engine", "engine"]
    assert {r.ootb.timing.mode for r in graph.results} == {"events"}
    assert [r.verdict for r in graph.results[1:]] == ["passed", "passed"]


def test_profiling_failure_keeps_the_timed_row(fake, monkeypatch):
    from dnn_benchmarking.metrics import profiling_orchestrator

    def fail(**kw):
        raise FileNotFoundError("/proc/nope")

    monkeypatch.setattr(profiling_orchestrator, "run_profiling_passes", fail)

    graph, progress = _run(metrics=MetricsConfig(perf=True))

    assert [(r.status, r.engine_name) for r in graph.results] == [
        ("success", "ENG_A"),
        ("success", "ENG_B"),
    ]
    for row in graph.results:
        assert row.ootb.gpu_kernel_stats is not None and row.extra_metrics is None
        assert "profiling failed: FileNotFoundError: /proc/nope" in row.warnings
    # Every profiling progress line is completed.
    assert progress.count("profiling ENG_A ... done") == 1
    assert progress.count("profiling ENG_B ... done") == 1


@pytest.mark.parametrize(
    "provider, reason",
    [
        (None, "not registered"),
        (SimpleNamespace(is_available=lambda: False), "not available"),
        (
            SimpleNamespace(is_available=lambda: True, supports_graph=lambda g: False),
            "does not support this graph",
        ),
        (
            SimpleNamespace(is_available=lambda: True, supports_graph=lambda g: True),
            None,
        ),
    ],
)
def test_reference_provider_is_usable_or_says_why_not(monkeypatch, provider, reason):
    def get_provider(name):
        assert name == "pytorch"
        if provider is None:
            raise ValueError(name)
        return provider

    monkeypatch.setattr(
        suite_runner.ReferenceProviderRegistry, "get_provider", get_provider
    )
    config = SuiteConfig(validation=ValidationConfig(provider="pytorch"))

    got = suite_runner._reference_provider(config, GRAPH)

    expected = (
        (provider, None)
        if reason is None
        else (None, f"Reference provider 'pytorch' {reason}")
    )
    assert got == expected


class TestPytorchRuntime:
    def _run(self, graph_json=GRAPH, **config):
        return suite_runner.run_graph_pytorch(
            PATH,
            graph_json,
            TENSORS,
            SuiteConfig(runtime="pytorch", **config),
            Reporter(io.StringIO()),
        )

    def test_single_timed_pytorch_row(self, fake_torch):
        graph = self._run()

        (row,) = graph.results
        assert graph.graph_id == graph_id_for(GRAPH) and graph.status == "ok"
        assert (
            row.runtime,
            row.engine_id,
            row.engine_name,
            row.role,
            row.verdict,
        ) == ("pytorch", None, "pytorch", "engine", "unchecked")
        assert row.ootb.gpu_kernel_stats.median_ms == pytest.approx(1.0)
        assert row.ootb.timing.mode == "staged"
        assert row.ootb.cpu_build_time_ms is None  # PyTorch has no plan build

    @pytest.mark.parametrize(
        "oracle, events",
        [
            (False, ["torch_buffers_freed"]),
            # The child's allocations must not stack on the row's buffers.
            (True, ["torch_buffers_freed", "pytorch_tuned"]),
        ],
    )
    def test_oracle_tunes_once_after_the_buffers_are_released(
        self, fake_torch, monkeypatch, oracle, events
    ):
        rows = []

        def run_pytorch_tuned(*, row, graph_path, graph_json, graph_name, config):
            fake_torch.events.append("pytorch_tuned")
            rows.append((row, graph_path))

        monkeypatch.setattr(suite_runner, "run_pytorch_tuned", run_pytorch_tuned)

        (row,) = self._run(oracle=oracle).results

        assert fake_torch.events == events
        assert rows == [(row, PATH)] * (len(events) - 1)

    def test_stall_failure_remeasures_the_row_unstalled(self, fake_torch):
        fake_torch.stall = {"pytorch"}

        (row,) = self._run().results

        assert row.status == "success"
        assert row.ootb.timing.mode == "events"
        assert [p.stall_gate for p in fake_torch.policies] == [True, False]

    def test_executor_failure_is_an_error_row(self, fake_torch):
        fake_torch.torch_error = ExecutionError("PyTorch GPU not available")

        (row,) = self._run().results

        assert row.status == "error"
        assert row.error_message == "ExecutionError: PyTorch GPU not available"

    def test_engine_row_gets_no_reference_warnings(self, fake_torch, monkeypatch):
        from dnn_benchmarking.execution import pytorch_ops

        monkeypatch.setattr(
            pytorch_ops, "get_reference_warnings", lambda graph_json: ["manual op"]
        )

        (row,) = self._run().results

        assert row.role == "engine"
        assert "manual op" not in (row.warnings or [])

    @pytest.mark.parametrize(
        "sdpa_backend, status", [("default", "skipped"), ("math", "error")]
    )
    def test_unsupported_operations(self, fake_torch, sdpa_backend, status):
        graph_json = {**GRAPH, "nodes": [{"type": "NotARealOp"}]}

        (row,) = self._run(graph_json, pytorch_sdpa_backend=sdpa_backend).results

        assert row.status == status
        assert "NotARealOp" in (row.error_message or row.skip_reason)

# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Suite CLI: startup checks, per-graph isolation, result writes, exit codes."""

import io
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import types
from pathlib import Path
from typing import Callable, List

import pytest

from dnn_benchmarking.cli import runtimes, suite_runner_cli
from dnn_benchmarking.cli.config_file import apply_config_file
from dnn_benchmarking.cli.main import main as cli_main
from dnn_benchmarking.cli.parser import create_parser
from dnn_benchmarking.reporting import compare
from dnn_benchmarking.reporting.reporter import Reporter
from dnn_benchmarking.reporting.suite_results import (
    CorrectnessResult,
    GraphResult,
    PlanResult,
    ProviderEngineResult,
    SuiteResult,
)

SAMPLE_GRAPH = Path(__file__).parents[3] / "graphs" / "sample_conv_fwd.json"


def _args(*argv: str):
    args = create_parser(suppress_defaults=True).parse_args(["-g", "unused", *argv])
    apply_config_file(args)
    return args


def _graphs(tmp_path: Path, n: int) -> List[Path]:
    paths = []
    for i in range(n):
        p = tmp_path / f"g{i}.json"
        shutil.copy(SAMPLE_GRAPH, p)
        paths.append(p)
    return paths


def _passed(engine_id: int = 1) -> ProviderEngineResult:
    return ProviderEngineResult(
        runtime="hipdnn",
        engine_id=engine_id,
        status="success",
        ootb=PlanResult(
            correctness=CorrectnessResult(tolerance_match=True, rtol=1e-3, atol=1e-3)
        ),
    )


def _failed() -> ProviderEngineResult:
    return ProviderEngineResult(
        runtime="hipdnn",
        engine_id=2,
        status="success",
        ootb=PlanResult(
            correctness=CorrectnessResult(tolerance_match=False, rtol=1e-3, atol=1e-3)
        ),
    )


def _graph(path: Path, rows: List[ProviderEngineResult]) -> GraphResult:
    return GraphResult(
        graph_name=path.stem, graph_path=str(path), results=rows, engine_ids=[1]
    )


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    monkeypatch.setattr(suite_runner_cli, "collect_environment_info", lambda: {})


@pytest.fixture
def runtime(monkeypatch):
    """Install a fake runtime; returns a setter taking run_graph(path)->GraphResult."""
    calls = []

    def install(run: Callable[[Path], GraphResult]) -> None:
        def start(config, reporter):
            calls.append(config)
            return lambda path, graph_json, infos: run(path)

        monkeypatch.setattr(suite_runner_cli, "start_runtime", start)

    install.calls = calls  # type: ignore[attr-defined]
    return install


def _run(args, graphs) -> tuple:
    out = io.StringIO()
    code = suite_runner_cli.run_suite_cli(args, graphs, Reporter(output=out))
    return code, out.getvalue()


@pytest.mark.parametrize(
    "rows, expected",
    [
        ([_passed()], 0),
        ([ProviderEngineResult.skipped_row("hipdnn", 1, "unsupported")], 0),
        ([_passed(), ProviderEngineResult.error_row("hipdnn", 2, "boom")], 1),
        ([_failed(), ProviderEngineResult.error_row("hipdnn", 3, "boom")], 3),
    ],
    ids=["passed", "all-skipped", "error-row", "failed-beats-error"],
)
def test_exit_code_follows_row_verdicts(tmp_path, runtime, rows, expected) -> None:
    runtime(lambda path: _graph(path, rows))
    code, _ = _run(_args(), _graphs(tmp_path, 1))
    assert code == expected


def test_graph_exception_is_isolated_and_written(tmp_path, runtime) -> None:
    g0, g1 = _graphs(tmp_path, 2)
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    out = tmp_path / "res" / "out.json"

    def run(path):
        if path == g0:
            raise RuntimeError("boom")
        return _graph(path, [_passed()])

    runtime(run)
    code, _ = _run(_args("-o", str(out)), [g0, bad, g1])

    doc = SuiteResult.load(out)
    graphs = doc["graphs"]
    assert code == 1
    assert doc["run"]["complete"] is True and doc["run"]["finished_at"]
    assert graphs[0]["status"] == "error"
    assert graphs[0]["error"] == "RuntimeError: boom"
    assert graphs[0]["graph_id"]  # the JSON parsed before the failure
    assert graphs[1]["status"] == "error" and graphs[1]["graph_id"] is None
    assert graphs[2]["status"] == "ok"
    assert doc["summary"]["graph_errors"] == 2
    assert set(doc["environment"]["end_of_run"]) >= {"host_rss_mb", "vram_used_mb"}


def test_intermediate_write_is_a_loadable_partial_file(
    tmp_path, runtime, monkeypatch
) -> None:
    monkeypatch.setattr(suite_runner_cli, "WRITE_INTERVAL_S", 0.0)
    g0, g1 = _graphs(tmp_path, 2)
    out = tmp_path / "out.json"
    seen = []

    def run(path):
        if path == g1:
            doc = SuiteResult.load(out)
            seen.append((len(doc["graphs"]), doc["run"]["complete"]))
        return _graph(path, [_passed()])

    runtime(run)
    code, _ = _run(_args("-o", str(out)), [g0, g1])

    assert code == 0
    assert seen == [(1, False)]
    assert SuiteResult.load(out)["run"]["complete"] is True


_SIGNALS = (signal.SIGINT, signal.SIGTERM)


def _sentinel_handler(signum, frame) -> None:
    raise AssertionError(f"signal {signum} reached the caller's handler")


@pytest.fixture
def caller_handlers():
    """Install known SIGINT/SIGTERM handlers; the run must hand them back.

    Comparing against whatever was installed before would let a leaked
    handler from an earlier test pass as "restored"."""
    saved = {s: signal.signal(s, _sentinel_handler) for s in _SIGNALS}
    yield
    for s, handler in saved.items():
        signal.signal(s, handler)


@pytest.mark.parametrize(
    "interrupt, expected",
    [
        (lambda: (_ for _ in ()).throw(KeyboardInterrupt()), 130),
        # raise_signal runs the Python handler on every OS; os.kill(SIGTERM)
        # terminates the whole process on Windows.
        (lambda: signal.raise_signal(signal.SIGTERM), 143),
    ],
    ids=["sigint", "sigterm"],
)
def test_interrupt_writes_partial_file(
    tmp_path, runtime, caller_handlers, interrupt, expected
) -> None:
    g0, g1, g2 = _graphs(tmp_path, 3)
    out = tmp_path / "out.json"

    def run(path):
        if path == g1:
            interrupt()
        return _graph(path, [_passed()])

    runtime(run)
    code, text = _run(_args("-o", str(out)), [g0, g1, g2])

    doc = SuiteResult.load(out)
    assert code == expected
    assert doc["run"]["complete"] is False and doc["run"]["finished_at"] is None
    assert [g["graph_name"] for g in doc["graphs"]] == ["g0"]
    assert str(out) in text
    assert all(signal.getsignal(s) is _sentinel_handler for s in _SIGNALS)


def test_interrupt_without_output_claims_no_partial_file(
    tmp_path, runtime, caller_handlers
) -> None:
    def run(path):
        raise KeyboardInterrupt

    runtime(run)
    code, text = _run(_args(), _graphs(tmp_path, 1))
    assert code == 130
    assert "no results file written" in text and "partial results" not in text


_NATIVE_BLOCK_RUN = """
import ctypes, io, sys
from pathlib import Path
from dnn_benchmarking.cli import suite_runner_cli as cli
from dnn_benchmarking.cli.config_file import apply_config_file
from dnn_benchmarking.cli.parser import create_parser
from dnn_benchmarking.reporting.reporter import Reporter

def run_graph(path, graph_json, infos):
    print("blocked", flush=True)
    mutex = ctypes.create_string_buffer(64)  # zeroed: a default pthread mutex
    libc = ctypes.CDLL(None)
    libc.pthread_mutex_lock(mutex)
    libc.pthread_mutex_lock(mutex)  # self-deadlock; futex waits survive signals

cli.SIGTERM_GRACE_S = 0.5
cli.collect_environment_info = lambda: {}
cli.start_runtime = lambda config, reporter: run_graph
args = create_parser(suppress_defaults=True).parse_args(["-g", "unused"])
apply_config_file(args)
code = cli.run_suite_cli(args, [Path(sys.argv[1])], Reporter(output=io.StringIO()))
sys.exit(code)
"""


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signal delivery")
def test_sigterm_ends_a_run_blocked_in_native_code(tmp_path) -> None:
    """The Python handler cannot run while native code holds the main thread."""
    src = str(Path(suite_runner_cli.__file__).parents[2])
    env = {**os.environ, "PYTHONPATH": src}
    proc = subprocess.Popen(
        [sys.executable, "-c", _NATIVE_BLOCK_RUN, str(_graphs(tmp_path, 1)[0])],
        stdout=subprocess.PIPE,
        text=True,
        env=env,
    )
    try:
        assert proc.stdout.readline().strip() == "blocked"
        time.sleep(0.3)  # let it enter the native wait
        proc.send_signal(signal.SIGTERM)
        assert proc.wait(timeout=4) == 143
    finally:
        proc.kill()
        proc.wait()


def test_final_write_failure_exits_1(tmp_path, runtime) -> None:
    out = tmp_path / "out.json"

    def run(path):
        out.mkdir()  # the output path turns into a directory mid-run
        return _graph(path, [_passed()])

    runtime(run)
    code, _ = _run(_args("-o", str(out)), _graphs(tmp_path, 1))
    assert code == 1


def test_non_os_write_error_is_reported_and_exits_1(
    tmp_path, runtime, monkeypatch
) -> None:
    def bad_write(self, path):
        raise ValueError("Out of range float values are not JSON compliant")

    monkeypatch.setattr(SuiteResult, "write", bad_write)
    runtime(lambda path: _graph(path, [_passed()]))
    code, text = _run(_args("-o", str(tmp_path / "out.json")), _graphs(tmp_path, 1))
    assert code == 1
    assert "not JSON compliant" in text


def test_intermediate_writes_are_throttled(tmp_path, runtime, monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(SuiteResult, "write", lambda self, path: calls.append(path))
    runtime(lambda path: _graph(path, [_passed()]))
    _run(_args("-o", str(tmp_path / "out.json")), _graphs(tmp_path, 3))
    assert len(calls) == 1  # well inside WRITE_INTERVAL_S: only the final write


def test_failed_intermediate_write_does_not_override_final_write(
    tmp_path, runtime, monkeypatch
) -> None:
    monkeypatch.setattr(suite_runner_cli, "WRITE_INTERVAL_S", 0.0)
    real_write = SuiteResult.write
    calls = []

    def flaky_write(self, path):
        calls.append(path)
        if len(calls) == 1:
            raise OSError(28, "No space left on device")
        real_write(self, path)

    monkeypatch.setattr(SuiteResult, "write", flaky_write)
    out = tmp_path / "out.json"
    runtime(lambda path: _graph(path, [_passed()]))
    code, text = _run(_args("-o", str(out)), _graphs(tmp_path, 2))

    assert code == 0
    assert "No space left on device" in text  # the transient failure is reported
    assert SuiteResult.load(out)["run"]["complete"] is True
    assert f"Results: {out}" in text


@pytest.mark.parametrize(
    "flag, argv",
    [
        ("--output", lambda d: ["-o", str(d / "out.json")]),
        (
            "--profiling-output-dir",
            lambda d: ["--perf", "--profiling-output-dir", str(d / "x")],
        ),
    ],
    ids=["output", "profiling-output-dir"],
)
def test_unwritable_output_is_usage_error(
    tmp_path, runtime, monkeypatch, flag, argv
) -> None:
    monkeypatch.setattr(suite_runner_cli, "check_requested_tools", lambda m: [])
    blocker = tmp_path / "file"
    blocker.write_text("")
    runtime(lambda path: _graph(path, [_passed()]))
    code, text = _run(_args(*argv(blocker)), _graphs(tmp_path, 1))
    assert code == 2
    assert f"ERROR: {flag} " in text
    assert runtime.calls == []


def test_output_that_is_a_directory_is_usage_error(tmp_path, runtime) -> None:
    runtime(lambda path: _graph(path, [_passed()]))
    code, text = _run(_args("-o", str(tmp_path)), _graphs(tmp_path, 1))
    assert code == 2
    assert "is a directory" in text
    assert runtime.calls == []


@pytest.mark.skipif(
    sys.platform == "win32" or os.geteuid() == 0, reason="POSIX mode bits as non-root"
)
def test_read_only_profiling_dir_is_usage_error(tmp_path, runtime, monkeypatch) -> None:
    monkeypatch.setattr(suite_runner_cli, "check_requested_tools", lambda m: [])
    ro = tmp_path / "ro"
    ro.mkdir(mode=0o500)
    runtime(lambda path: _graph(path, [_passed()]))
    try:
        code, text = _run(
            _args("--perf", "--profiling-output-dir", str(ro)), _graphs(tmp_path, 1)
        )
    finally:
        ro.chmod(0o700)
    assert code == 2
    assert "is not writable" in text


def test_plain_run_creates_no_profiling_directory(
    tmp_path, runtime, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    runtime(lambda path: _graph(path, [_passed()]))
    graphs = _graphs(tmp_path, 1)
    assert _run(_args(), graphs)[0] == 0
    assert not (tmp_path / "profiling-output").exists()


def test_config_error_is_usage_error(tmp_path, runtime) -> None:
    runtime(lambda path: _graph(path, [_passed()]))
    code, _ = _run(_args("-r", "pytorch", "-e", "1"), _graphs(tmp_path, 1))
    assert code == 2
    assert runtime.calls == []


def test_missing_profiling_tool_exits_2_before_runtime_startup(
    tmp_path, runtime, monkeypatch
) -> None:
    monkeypatch.setattr(
        suite_runner_cli, "check_requested_tools", lambda m: ["--perf needs perf"]
    )
    runtime(lambda path: _graph(path, [_passed()]))
    code, text = _run(_args("--perf"), _graphs(tmp_path, 1))
    assert code == 2
    assert "--perf needs perf" in text
    assert runtime.calls == []


def test_main_routes_compare_subcommand(monkeypatch) -> None:
    seen = []
    monkeypatch.setattr(compare, "main", lambda argv: seen.append(argv) or 7)
    assert cli_main(["compare", "a.json", "b.json"]) == 7
    assert seen == [["a.json", "b.json"]]


def test_main_without_matching_graphs_exits_1(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("DNN_BENCH_WORKSPACE", str(tmp_path))
    assert cli_main(["--graph", str(tmp_path / "none*.json")]) == 1
    assert "No graph files found" in capsys.readouterr().err


def test_main_records_its_argv_and_wires_verbose(
    tmp_path, runtime, monkeypatch, capsys
) -> None:
    monkeypatch.setenv("DNN_BENCH_WORKSPACE", str(tmp_path))
    runtime(lambda path: _graph(path, [_passed()]))
    out = tmp_path / "out.json"
    argv = ["--graph", str(_graphs(tmp_path, 1)[0]), "-o", str(out), "-v"]

    assert cli_main(argv) == 0

    assert SuiteResult.load(out)["run"]["argv"] == [sys.argv[0], *argv]
    assert "correctness passed" in capsys.readouterr().out  # verbose block


def test_run_config_records_effective_values_that_compare_checks(
    tmp_path, runtime, capsys
) -> None:
    runtime(lambda path: _graph(path, [_passed()]))
    warm, cold = tmp_path / "warm.json", tmp_path / "cold.json"
    _run(_args("-o", str(warm), "--timing-block", "4"), _graphs(tmp_path, 1))
    _run(
        _args(
            *("-o", str(cold), "--cache-mode", "cold", "--seed", "7"),
            *("--iters", "11", "--warmup", "3", "--min-time-ms", "5"),
            *("--validate", "pytorch", "--pytorch-sdpa-backend", "flash"),
            *("--pytorch-rocm-fa-library", "aotriton"),
        ),
        _graphs(tmp_path, 1),
    )

    warm_config = SuiteResult.load(warm)["run"]["config"]
    assert warm_config["timing_block"] == 4
    assert warm_config["pytorch_sdpa_backend"] is None  # PyTorch not used
    assert warm_config["pytorch_rocm_fa_library"] is None
    config = SuiteResult.load(cold)["run"]["config"]
    del config["plugin_paths"]  # the default depends on the installed ROCm
    assert config == {
        "runtime": "hipdnn",
        "engine_filter": None,
        "warmup_iters": 3,
        "iters": 11,
        "min_time_ms": 5.0,
        "cache_mode": "cold",
        "timing_block": 1,
        "seed": 7,
        "validate": "pytorch",
        "rtol": None,
        "atol": None,
        "oracle_mode": "off",
        "autotune": False,
        "hipdnn_cache_dir": None,
        "pytorch_sdpa_backend": "flash",
        "pytorch_rocm_fa_library": "aotriton",
        "metrics": True,
        "profiling": {
            "pmc": None,
            "trace": False,
            "perf": False,
            "roofline": False,
        },
    }
    capsys.readouterr()
    assert compare.main([str(warm), str(cold)]) == 2
    assert "cache_mode differs" in capsys.readouterr().err


def test_run_config_records_non_default_selection_and_validation(
    tmp_path, runtime, monkeypatch
) -> None:
    # The run sets these process-wide; register them so teardown restores them.
    monkeypatch.setenv("HIPDNN_FORCE_BENCHMARKING", "")
    monkeypatch.setenv("HIPDNN_CACHE_DIR", "")
    runtime(lambda path: _graph(path, [_passed()]))
    out, cache = tmp_path / "out.json", str(tmp_path / "cache")
    _run(
        _args(
            *("-o", str(out), "-e", "MIOPEN_ENGINE", "--plugin-path", str(tmp_path)),
            *("--validate", "pytorch", "--rtol", "1e-3", "--oracle-mode", "plan"),
            *("--autotune", "--hipdnn-cache-dir", cache),
        ),
        _graphs(tmp_path, 1),
    )
    config = SuiteResult.load(out)["run"]["config"]
    picked = {k: config[k] for k in ("engine_filter", "rtol", "oracle_mode")}
    assert picked == {
        "engine_filter": ["0x15B46865C717A122"],
        "rtol": 1e-3,
        "oracle_mode": "plan",
    }
    assert (config["autotune"], config["hipdnn_cache_dir"]) == (True, cache)


def test_selection_env_recorded_only_for_autotune_or_oracle(tmp_path, runtime) -> None:
    runtime(lambda path: _graph(path, [_passed()]))
    plain, oracle = tmp_path / "plain.json", tmp_path / "oracle.json"
    tuned = tmp_path / "tuned.json"
    _run(_args("-o", str(plain)), _graphs(tmp_path, 1))
    _run(_args("-o", str(oracle), "--oracle-mode", "plan"), _graphs(tmp_path, 1))
    _run(_args("-o", str(tuned), "--autotune"), _graphs(tmp_path, 1))

    assert SuiteResult.load(plain)["environment"]["selection_env"] is None
    env = SuiteResult.load(oracle)["environment"]["selection_env"]
    assert set(env) == set(suite_runner_cli._SELECTION_ENV)
    env = SuiteResult.load(tuned)["environment"]["selection_env"]
    assert env["HIPDNN_FORCE_BENCHMARKING"] == "1"


def _fake_hipdnn(monkeypatch, *, loaded=(), handle_error=None) -> list:
    """Install a fake hipdnn_frontend whose handle knows only ``loaded`` IDs
    (or ``loaded[plugin_path]``); return the plugin-path/Handle call log."""
    calls: list = []

    class Handle:
        def __init__(self):
            calls.append("Handle")
            if handle_error is not None:
                raise handle_error
            paths = [c for c in calls if isinstance(c, list)]
            self.loaded = loaded[paths[-1][0]] if isinstance(loaded, dict) else loaded

        def get_engine_info(self, engine_id):
            if engine_id not in self.loaded:
                raise IndexError("Engine ID is not loaded")
            return object()

    names = {0x15B46865C717A122: "MIOPEN_ENGINE"}
    module = types.SimpleNamespace(
        Handle=Handle,
        PluginLoadingMode=types.SimpleNamespace(ABSOLUTE="abs"),
        set_engine_plugin_paths=lambda paths, mode: calls.append(paths),
        engine_id_to_name=lambda engine_id: names.get(engine_id, ""),
    )
    monkeypatch.setitem(sys.modules, "hipdnn_frontend", module)
    monkeypatch.setattr(runtimes, "initialize_pip_rocm_runtime", lambda: None)
    return calls


def test_per_engine_plugin_paths_check_each_pair_without_a_shared_handle(
    monkeypatch,
) -> None:
    a, b = str(Path("/a")), str(Path("/b"))
    calls = _fake_hipdnn(monkeypatch, loaded={a: {1}, b: {1}})
    handles = []
    monkeypatch.setattr(
        runtimes, "run_graph_all_providers", lambda *args: handles.append(args[4])
    )
    config = suite_runner_cli.SuiteConfig.from_namespace(
        _args("-e", "1,1", "--plugin-path", "/a,/b")
    )

    runtimes.start_runtime(config, Reporter(output=io.StringIO()))(None, {}, [])

    assert calls == [[a], "Handle", [b], "Handle"]
    assert handles == [None]  # the runner creates one handle per pair
    selections = config.engine_selections_for(config.engine_filter)
    assert [str(s.plugin_path) for s in selections] == [a, b]


def test_unknown_engine_in_one_plugin_path_is_usage_error(
    tmp_path, monkeypatch
) -> None:
    b = str(Path("/b"))
    _fake_hipdnn(monkeypatch, loaded={str(Path("/a")): {1}, b: set()})
    code, text = _run(
        _args("-e", "1,1", "--plugin-path", "/a,/b"), _graphs(tmp_path, 1)
    )
    assert code == 2
    assert f"plugin loaded from {b}: 0x0000000000000001" in text


def test_unknown_engine_is_usage_error(tmp_path, monkeypatch) -> None:
    _fake_hipdnn(monkeypatch, loaded=())
    code, text = _run(
        _args("-e", "MIOPEN_ENGINE,0x63,NOT_AN_ENGINE", "--plugin-path", str(tmp_path)),
        _graphs(tmp_path, 1),
    )
    assert code == 2
    assert "MIOPEN_ENGINE/0x15B46865C717A122" in text
    assert ", 0x0000000000000063," in text  # an ID typed as an ID stays bare
    # Not registered, so only the parser knows the name the user typed.
    assert "NOT_AN_ENGINE/0x" in text


def test_loaded_engine_passes_startup(tmp_path, monkeypatch) -> None:
    _fake_hipdnn(monkeypatch, loaded=(0x15B46865C717A122,))
    seen = []
    monkeypatch.setattr(
        runtimes, "run_graph_all_providers", lambda *args: seen.append(args) or "ran"
    )
    config = suite_runner_cli.SuiteConfig.from_namespace(
        _args("-e", "MIOPEN_ENGINE", "--plugin-path", str(tmp_path))
    )
    runner = runtimes.start_runtime(config, Reporter(output=io.StringIO()))
    assert runner("g.json", {}, []) == "ran"
    # The hipDNN runner, bound to the handle that startup created and checked.
    ((path, _, _, run_config, handle, _),) = seen
    assert (path, run_config, handle.loaded) == (
        "g.json",
        config,
        (0x15B46865C717A122,),
    )


def test_hipdnn_handle_failure_exits_1(tmp_path, monkeypatch) -> None:
    _fake_hipdnn(monkeypatch, handle_error=RuntimeError("no GPU"))
    code, text = _run(_args("--plugin-path", str(tmp_path)), _graphs(tmp_path, 1))
    assert code == 1
    assert "no GPU" in text


def test_reference_row_error_does_not_set_exit_code(tmp_path, runtime) -> None:
    # Exit code counts engine rows only, like the summary.
    ref = ProviderEngineResult.error_row("pytorch", None, "boom", role="reference")
    runtime(lambda path: _graph(path, [_passed(), ref]))
    code, _ = _run(_args(), _graphs(tmp_path, 1))
    assert code == 0


def test_failed_final_write_reports_no_results_path(tmp_path, runtime) -> None:
    out = tmp_path / "out.json"

    def run(path):
        out.mkdir()
        return _graph(path, [_passed()])

    runtime(run)
    code, text = _run(_args("-o", str(out)), _graphs(tmp_path, 1))
    assert code == 1
    assert "Results:" not in text


def test_interrupt_with_failed_write_claims_no_partial_file(tmp_path, runtime) -> None:
    out = tmp_path / "out.json"

    def run(path):
        out.mkdir()
        raise KeyboardInterrupt

    runtime(run)
    code, text = _run(_args("-o", str(out)), _graphs(tmp_path, 1))
    assert code == 130
    assert "partial results" not in text and "no results file written" in text


def test_unavailable_reference_provider_fails_before_any_graph(
    tmp_path, monkeypatch
) -> None:
    unavailable = types.SimpleNamespace(is_available=lambda: False)
    monkeypatch.setattr(
        runtimes.ReferenceProviderRegistry, "get_provider", lambda name: unavailable
    )

    def no_runtime(config):
        raise AssertionError("runtime started despite the unavailable provider")

    monkeypatch.setattr(runtimes, "_create_hipdnn_handle", no_runtime)
    code, text = _run(_args("--validate", "pytorch"), _graphs(tmp_path, 1))
    assert code == 1
    assert "--validate pytorch" in text


def test_available_reference_provider_passes_startup(tmp_path, monkeypatch) -> None:
    _fake_hipdnn(monkeypatch)
    asked = []
    available = types.SimpleNamespace(is_available=lambda: True)
    monkeypatch.setattr(
        runtimes.ReferenceProviderRegistry,
        "get_provider",
        lambda name: asked.append(name) or available,
    )
    config = suite_runner_cli.SuiteConfig.from_namespace(
        _args("--validate", "pytorch", "--plugin-path", str(tmp_path))
    )
    assert callable(runtimes.start_runtime(config, Reporter(output=io.StringIO())))
    assert asked == ["pytorch"]


def test_cli_flags_reach_suite_config() -> None:
    config = suite_runner_cli.SuiteConfig.from_namespace(
        _args("--timing-block", "4", "--profiling-timeout", "99", "--perf")
    )
    assert config.timing_policy.timing_block == 4
    assert config.metrics.profiling_timeout_s == 99


@pytest.mark.parametrize("pmc_set", ["basic", "memory", "flops"])
def test_single_pass_pmc_sets_are_accepted(pmc_set) -> None:
    config = suite_runner_cli.SuiteConfig.from_namespace(_args("--pmc", pmc_set))
    assert config.metrics.pmc_set == pmc_set


def _warnings(text: str) -> str:
    return "\n".join(line for line in text.splitlines() if "WARNING:" in line)


@pytest.mark.parametrize(
    "env, argv, expected",
    [
        ({}, [], "HIPDNN_DISABLE_EXACT_ENGINE_CACHE"),
        (  # hipDNN reads "0" as off
            {"HIPDNN_DISABLE_EXACT_ENGINE_CACHE": "0"},
            [],
            "HIPDNN_DISABLE_EXACT_ENGINE_CACHE",
        ),
        ({"HIPDNN_DISABLE_EXACT_ENGINE_CACHE": "1"}, [], "HIPDNN_DISABLE_CACHE=1"),
        (  # hipDNN reads "0" as off
            {"HIPDNN_DISABLE_EXACT_ENGINE_CACHE": "1", "HIPDNN_DISABLE_CACHE": "0"},
            [],
            "HIPDNN_DISABLE_CACHE=1",
        ),
        (
            {"HIPDNN_DISABLE_EXACT_ENGINE_CACHE": "1", "HIPDNN_DISABLE_CACHE": "1"},
            ["--warmup", "0"],  # priming always runs untimed: nothing to warn
            None,
        ),
    ],
    ids=[
        "exact-cache-on",
        "exact-cache-zero",
        "provider-cache-on",
        "provider-cache-zero",
        "cold",
    ],
)
def test_oracle_warns_on_non_cold_baseline(
    tmp_path, runtime, monkeypatch, env, argv, expected
) -> None:
    for name in ("HIPDNN_DISABLE_EXACT_ENGINE_CACHE", "HIPDNN_DISABLE_CACHE"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    runtime(lambda path: _graph(path, [_passed()]))
    _, text = _run(_args("--oracle-mode", "plan", *argv), _graphs(tmp_path, 1))
    warnings = _warnings(text)
    if expected is None:
        assert warnings == ""
    else:
        assert expected in warnings


def test_final_write_survives_second_signal_and_snapshot_error(
    tmp_path, runtime, monkeypatch, caller_handlers
) -> None:
    def probe():
        signal.raise_signal(signal.SIGTERM)
        signal.raise_signal(signal.SIGINT)
        raise RuntimeError("amdsmi gone")

    monkeypatch.setattr(suite_runner_cli, "GpuSmiProbe", probe)
    runtime(lambda path: (_ for _ in ()).throw(KeyboardInterrupt()))
    out = tmp_path / "out.json"

    code, text = _run(_args("-o", str(out)), _graphs(tmp_path, 1))

    assert code == 130
    assert SuiteResult.load(out)["run"]["complete"] is False
    assert "end-of-run snapshot failed: amdsmi gone" in text
    assert all(signal.getsignal(s) is _sentinel_handler for s in _SIGNALS)


@pytest.mark.parametrize("mode", ["exhaustive", "plan"])
def test_only_exhaustive_oracle_states_the_provider_cache_cost(
    tmp_path, runtime, mode
) -> None:
    runtime(lambda path: _graph(path, [_passed()]))
    _, text = _run(_args("--oracle-mode", mode), _graphs(tmp_path, 1))
    notice = "--oracle-mode exhaustive: providers may reuse tuned selections"
    assert (notice in text) == (mode == "exhaustive")


def test_empty_nodes_graph_is_a_graph_error(tmp_path, runtime) -> None:
    (empty,) = _graphs(tmp_path, 1)
    doc = json.loads(empty.read_text())
    doc["nodes"] = []
    empty.write_text(json.dumps(doc))
    out = tmp_path / "out.json"
    runtime(lambda path: _graph(path, [_passed()]))
    code, _ = _run(_args("-o", str(out)), [empty])
    (graph,) = SuiteResult.load(out)["graphs"]
    assert code == 1 and graph["status"] == "error" and graph["error"]


def test_unsupported_tensor_dtype_is_no_engines_not_an_error(tmp_path, runtime) -> None:
    (graph_path,) = _graphs(tmp_path, 1)
    doc = json.loads(graph_path.read_text())
    doc["tensors"][0]["data_type"] = "int3"
    graph_path.write_text(json.dumps(doc))
    out = tmp_path / "out.json"
    runtime(lambda path: _graph(path, [_passed()]))
    code, _ = _run(_args("-o", str(out)), [graph_path])
    (graph,) = SuiteResult.load(out)["graphs"]
    assert (code, graph["status"], graph["results"]) == (0, "no_engines", [])
    assert "int3" in graph["message"] and graph["graph_id"]


def test_internal_profiling_flag_is_hidden_from_help() -> None:
    assert "--internal-profiling-run" not in create_parser().format_help()


@pytest.mark.parametrize(
    "argv, expected",
    [
        (["--profiling-output-dir", "x"], "--profiling-output-dir"),
        (["--pytorch-sdpa-backend", "math"], "--runtime pytorch"),
        (["--pytorch-sdpa-backend", "math", "--validate", "pytorch"], None),
        (["--pytorch-sdpa-backend", "math", "-r", "pytorch"], None),
    ],
    ids=[
        "profiling-output-dir",
        "sdpa-without-pytorch",
        "sdpa-validate",
        "sdpa-backend",
    ],
)
def test_ignored_options_warn(tmp_path, runtime, argv, expected) -> None:
    runtime(lambda path: _graph(path, [_passed()]))
    _, text = _run(_args(*argv), _graphs(tmp_path, 1))
    if expected is None:
        assert "SDPA" not in _warnings(text)
    else:
        assert expected in _warnings(text)

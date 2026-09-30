# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Suite CLI: startup checks, per-graph isolation, result writes, exit codes."""

import io
import os
import shutil
import signal
import sys
import types
from pathlib import Path
from typing import Callable, List

import pytest

from dnn_benchmarking.cli import backends, suite_runner_cli
from dnn_benchmarking.cli.config_file import apply_config_file
from dnn_benchmarking.cli.parser import create_parser
from dnn_benchmarking.reporting.reporter import Reporter
from dnn_benchmarking.reporting.suite_results import (
    CorrectnessResult,
    GraphResult,
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
        provider="hipdnn",
        engine_id=engine_id,
        status="success",
        correctness=CorrectnessResult(tolerance_match=True, rtol=1e-3, atol=1e-3),
    )


def _failed() -> ProviderEngineResult:
    return ProviderEngineResult(
        provider="hipdnn",
        engine_id=2,
        status="success",
        correctness=CorrectnessResult(tolerance_match=False, rtol=1e-3, atol=1e-3),
    )


def _graph(path: Path, rows: List[ProviderEngineResult]) -> GraphResult:
    return GraphResult(
        graph_name=path.stem, graph_path=str(path), results=rows, engine_ids=[1]
    )


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    # Kernel-selection variables are process-wide; monkeypatch restores them.
    for name in ("HIPDNN_FORCE_BENCHMARKING", "HIPDNN_CACHE_DIR"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(suite_runner_cli, "collect_environment_info", lambda: {})


@pytest.fixture
def backend(monkeypatch):
    """Install a fake backend; returns a setter taking run_graph(path)->GraphResult."""
    calls = []

    def install(run: Callable[[Path], GraphResult]) -> None:
        def start(config, reporter):
            calls.append(config)
            return lambda path, graph_json, infos: run(path)

        monkeypatch.setattr(suite_runner_cli, "start_backend", start)

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
def test_exit_code_follows_row_verdicts(tmp_path, backend, rows, expected) -> None:
    backend(lambda path: _graph(path, rows))
    code, _ = _run(_args(), _graphs(tmp_path, 1))
    assert code == expected


def test_graph_exception_is_isolated_and_written(tmp_path, backend) -> None:
    g0, g1 = _graphs(tmp_path, 2)
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    out = tmp_path / "res" / "out.json"

    def run(path):
        if path == g0:
            raise RuntimeError("boom")
        return _graph(path, [_passed()])

    backend(run)
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
    tmp_path, backend, monkeypatch
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

    backend(run)
    code, _ = _run(_args("-o", str(out)), [g0, g1])

    assert code == 0
    assert seen == [(1, False)]
    assert SuiteResult.load(out)["run"]["complete"] is True


@pytest.mark.parametrize(
    "interrupt, expected",
    [
        (lambda: (_ for _ in ()).throw(KeyboardInterrupt()), 130),
        (lambda: os.kill(os.getpid(), signal.SIGTERM), 143),
    ],
    ids=["sigint", "sigterm"],
)
def test_interrupt_writes_partial_file(tmp_path, backend, interrupt, expected) -> None:
    g0, g1, g2 = _graphs(tmp_path, 3)
    out = tmp_path / "out.json"
    before = signal.getsignal(signal.SIGTERM)

    def run(path):
        if path == g1:
            interrupt()
        return _graph(path, [_passed()])

    backend(run)
    code, text = _run(_args("-o", str(out)), [g0, g1, g2])

    doc = SuiteResult.load(out)
    assert code == expected
    assert doc["run"]["complete"] is False and doc["run"]["finished_at"] is None
    assert [g["graph_name"] for g in doc["graphs"]] == ["g0"]
    assert str(out) in text
    assert signal.getsignal(signal.SIGTERM) is before


def test_final_write_failure_exits_1(tmp_path, backend) -> None:
    out = tmp_path / "out.json"

    def run(path):
        out.mkdir()  # the output path turns into a directory mid-run
        return _graph(path, [_passed()])

    backend(run)
    code, _ = _run(_args("-o", str(out)), _graphs(tmp_path, 1))
    assert code == 1


@pytest.mark.parametrize(
    "flag, argv",
    [
        ("--output", lambda d: ["-o", str(d / "out.json")]),
        ("--profiling-output-dir", lambda d: ["--perf", "--profiling-output-dir", str(d / "x")]),
    ],
    ids=["output", "profiling-output-dir"],
)
def test_unwritable_output_is_usage_error(
    tmp_path, backend, monkeypatch, flag, argv
) -> None:
    monkeypatch.setattr(suite_runner_cli, "check_requested_tools", lambda m: [])
    blocker = tmp_path / "file"
    blocker.write_text("")
    backend(lambda path: _graph(path, [_passed()]))
    code, text = _run(_args(*argv(blocker)), _graphs(tmp_path, 1))
    assert code == 2
    assert f"ERROR: {flag} " in text
    assert backend.calls == []


def test_config_error_is_usage_error(tmp_path, backend) -> None:
    backend(lambda path: _graph(path, [_passed()]))
    code, _ = _run(_args("-b", "pytorch", "-e", "1"), _graphs(tmp_path, 1))
    assert code == 2
    assert backend.calls == []


def test_missing_profiling_tool_exits_2_before_backend(
    tmp_path, backend, monkeypatch
) -> None:
    monkeypatch.setattr(
        suite_runner_cli, "check_requested_tools", lambda m: ["--perf needs perf"]
    )
    backend(lambda path: _graph(path, [_passed()]))
    code, text = _run(_args("--perf"), _graphs(tmp_path, 1))
    assert code == 2
    assert "--perf needs perf" in text
    assert backend.calls == []


def test_selection_env_recorded_only_for_autotune(tmp_path, backend) -> None:
    backend(lambda path: _graph(path, [_passed()]))
    plain, tuned = tmp_path / "plain.json", tmp_path / "tuned.json"
    _run(_args("-o", str(plain)), _graphs(tmp_path, 1))
    _run(_args("-o", str(tuned), "--autotune"), _graphs(tmp_path, 1))

    assert SuiteResult.load(plain)["environment"]["selection_env"] is None
    env = SuiteResult.load(tuned)["environment"]["selection_env"]
    assert env["HIPDNN_FORCE_BENCHMARKING"] == "1"


def _fake_hipdnn(monkeypatch, *, loaded=(), handle_error=None):
    """Install a fake hipdnn_frontend whose handle knows only ``loaded`` IDs."""

    class Handle:
        def __init__(self):
            if handle_error is not None:
                raise handle_error

        def get_engine_info(self, engine_id):
            if engine_id not in loaded:
                raise IndexError("Engine ID is not loaded")
            return object()

    names = {0x15B46865C717A122: "MIOPEN_ENGINE"}
    module = types.SimpleNamespace(
        Handle=Handle,
        PluginLoadingMode=types.SimpleNamespace(ABSOLUTE="abs"),
        set_engine_plugin_paths=lambda paths, mode: None,
        engine_id_to_name=lambda engine_id: names.get(engine_id, ""),
    )
    monkeypatch.setitem(sys.modules, "hipdnn_frontend", module)
    monkeypatch.setattr(backends, "initialize_pip_rocm_runtime", lambda: None)


def test_unknown_engine_is_usage_error(tmp_path, monkeypatch) -> None:
    _fake_hipdnn(monkeypatch, loaded=())
    code, text = _run(
        _args("-e", "MIOPEN_ENGINE,0x63", "--plugin-path", str(tmp_path)),
        _graphs(tmp_path, 1),
    )
    assert code == 2
    assert "MIOPEN_ENGINE/0x15B46865C717A122" in text
    assert "0x0000000000000063" in text


def test_loaded_engine_passes_startup(tmp_path, monkeypatch) -> None:
    _fake_hipdnn(monkeypatch, loaded=(0x15B46865C717A122,))
    config = suite_runner_cli.SuiteConfig.from_namespace(
        _args("-e", "MIOPEN_ENGINE", "--plugin-path", str(tmp_path))
    )
    assert callable(backends.start_backend(config, Reporter(output=io.StringIO())))


def test_hipdnn_handle_failure_exits_1(tmp_path, monkeypatch) -> None:
    _fake_hipdnn(monkeypatch, handle_error=RuntimeError("no GPU"))
    code, text = _run(_args("--plugin-path", str(tmp_path)), _graphs(tmp_path, 1))
    assert code == 1
    assert "no GPU" in text

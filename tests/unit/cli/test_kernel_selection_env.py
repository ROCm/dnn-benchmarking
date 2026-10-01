# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for kernel-selection environment setup in the suite CLI."""

import pytest


@pytest.fixture(autouse=True)
def _isolate_selection_env(monkeypatch):
    """HIPDNN_FORCE_BENCHMARKING is process-wide with no per-engine granularity,
    so a value left behind by one test changes an unrelated one. Caught for real:
    without this, these tests leaked the flag into test_suite_cli.py and made it
    fail on an extra warning. monkeypatch restores whatever was there before."""
    monkeypatch.delenv("HIPDNN_FORCE_BENCHMARKING", raising=False)
    monkeypatch.delenv("HIPDNN_CACHE_DIR", raising=False)
    yield


class TestKernelSelectionEnvironment:
    """The selection path must be set deliberately and stated out loud.

    An engine has two selection paths that answer different questions: the cold
    heuristic (which measures how good the heuristic is) and benchmarking (which
    measures what the shipped kernel set can deliver). The default is the first,
    silently. A perf table whose path is unstated is not interpretable, and a
    kernel set that gains good variants can measure *slower* on the heuristic
    path when the tie-break is arbitrary -- so the path is announced on every
    run, not only when it is requested.
    """

    @staticmethod
    def _config(**kwargs):
        from dnn_benchmarking.config.benchmark_config import SuiteConfig

        return SuiteConfig(**kwargs)

    @staticmethod
    def _reporter():
        class _R:
            def __init__(self):
                self.messages = []

            def print_warning(self, message):
                self.messages.append(message)

            def print_error(self, message):
                self.messages.append(message)

        return _R()

    def _apply(self, monkeypatch, **kwargs):
        from dnn_benchmarking.cli.suite_runner_cli import _apply_tuning_environment

        monkeypatch.delenv("HIPDNN_FORCE_BENCHMARKING", raising=False)
        monkeypatch.delenv("HIPDNN_CACHE_DIR", raising=False)
        reporter = self._reporter()
        _apply_tuning_environment(self._config(**kwargs), reporter)
        return reporter

    def test_default_does_not_force_benchmarking(self, monkeypatch) -> None:
        import os

        self._apply(monkeypatch)
        assert "HIPDNN_FORCE_BENCHMARKING" not in os.environ

    def test_default_states_the_heuristic_path(self, monkeypatch) -> None:
        reporter = self._apply(monkeypatch)
        assert any("COLD HEURISTIC" in m for m in reporter.messages)

    def test_autotune_forces_benchmarking(self, monkeypatch) -> None:
        import os

        self._apply(monkeypatch, autotune=True)
        assert os.environ["HIPDNN_FORCE_BENCHMARKING"] == "1"

    def test_autotune_states_the_benchmarked_path(self, monkeypatch) -> None:
        reporter = self._apply(monkeypatch, autotune=True)
        assert any("BENCHMARKED" in m for m in reporter.messages)

    def test_autotune_without_cache_dir_warns(self, monkeypatch) -> None:
        """The winner cache outlives the run, and reads are not gated on
        benchmarking while writes are -- so a tuned run with no explicit root can
        report a previous kernel set's winners as if it measured them."""
        reporter = self._apply(monkeypatch, autotune=True)
        assert any("outlives this run" in m for m in reporter.messages)

    def test_cache_dir_is_exported(self, monkeypatch) -> None:
        import os

        self._apply(monkeypatch, autotune=True, cache_dir="/tmp/phase-x")
        assert os.environ["HIPDNN_CACHE_DIR"] == "/tmp/phase-x"

    def test_cache_dir_suppresses_the_warning(self, monkeypatch) -> None:
        reporter = self._apply(monkeypatch, autotune=True, cache_dir="/tmp/phase-x")
        assert not any("outlives this run" in m for m in reporter.messages)

    def test_a_leaked_force_flag_is_reported(self, monkeypatch) -> None:
        """The variable is process-wide with no per-engine granularity, so a value
        left by another shell or test silently changes the selection path. Report
        what is in effect, not what was asked for."""
        from dnn_benchmarking.cli.suite_runner_cli import _apply_tuning_environment

        monkeypatch.setenv("HIPDNN_FORCE_BENCHMARKING", "1")
        reporter = self._reporter()
        _apply_tuning_environment(self._config(), reporter)
        assert any("is set in the environment" in m for m in reporter.messages)
        assert not any("COLD HEURISTIC" in m for m in reporter.messages)

    def test_missing_cache_dir_warns_about_the_shared_cache(self, monkeypatch) -> None:
        """The winner cache is keyed by graph content and device -- not by
        checkout, engine or session. Two agents benchmarking the same graphs on
        one box read and write each other's rankings through the per-user
        default, and reads are ungated while writes are gated on benchmarking,
        so an untuned run can report another session's tuned result as its own."""
        reporter = self._apply(monkeypatch)
        assert any("shared per-user cache" in m for m in reporter.messages)

    def test_explicit_cache_dir_suppresses_the_shared_warning(
        self, monkeypatch
    ) -> None:
        reporter = self._apply(monkeypatch, cache_dir="/tmp/phase-x")
        assert not any("shared per-user cache" in m for m in reporter.messages)

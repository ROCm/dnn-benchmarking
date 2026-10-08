# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for the perf stat wrapper."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from dnn_benchmarking.metrics import _subprocess
from dnn_benchmarking.metrics import perf as perf_mod
from dnn_benchmarking.metrics._diagnostic import reset as reset_warn_once

# Grabbed before the autouse fixture stubs it out on the module.
_REAL_CAP_PROBE = perf_mod._has_perfmon_capability

# A minimal seven-column perf-stat -x, sample.
SAMPLE_CSV = """\
# started on Mon Jan  1 00:00:00 2026
1234567890,,cycles:u,123456789,100.00,,
987654321,,instructions:u,123456789,100.00,0.80,insn per cycle
123.45,msec,task-clock,123456789,100.00,,
12,,context-switches,123456789,100.00,,
3,,page-faults,123456789,100.00,,
"""


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    reset_warn_once()
    # The suite often runs as root in a privileged container, where the
    # real probe says "kernel events are fine". Pin it off so the
    # paranoid tests assert the sysctl rule, not the host's capabilities.
    monkeypatch.setattr(perf_mod, "_has_perfmon_capability", lambda: False)
    # Pin binary resolution rather than shelling out to the host's perf.
    monkeypatch.setattr(perf_mod, "_perf_is_runnable", lambda _: True)
    monkeypatch.setattr(perf_mod, "_installed_perf_binaries", lambda: [])
    monkeypatch.setattr(perf_mod.shutil, "which", lambda _: "/usr/bin/perf")
    monkeypatch.setattr(perf_mod, "_read_perf_paranoid", lambda: 1)


def _run(tmp_path, returncode=0, stderr="", write_csv=True):
    """Run perf.run with a fake perf that writes SAMPLE_CSV at its ``-o``.

    Returns ``(slice, argv)``.
    """
    captured = {}

    def fake_run(argv, timeout_s=None):
        captured["argv"] = argv
        if write_csv:
            # Write at the exact `-o` value so a moved flag fails loudly.
            Path(argv[argv.index("-o") + 1]).write_text(SAMPLE_CSV)
        return MagicMock(returncode=returncode, stdout="", stderr=stderr)

    with patch.object(_subprocess, "run_capped", side_effect=fake_run):
        extra = perf_mod.run(
            inner_argv=["python"], out_dir=tmp_path, timeout_s=60, context="g/E"
        )
    return extra["perf"], captured.get("argv")


class TestParseCsv:
    def test_parses_user_events(self, tmp_path):
        csv = tmp_path / "perf.csv"
        csv.write_text(SAMPLE_CSV)
        parsed = perf_mod._parse_perf_csv(csv)
        assert parsed["cycles:u"] == 1234567890
        assert parsed["instructions:u"] == 987654321
        assert parsed["task-clock"] == 123.45
        assert parsed["context-switches"] == 12
        assert parsed["page-faults"] == 3

    def test_handles_not_counted_marker(self, tmp_path):
        csv = tmp_path / "perf.csv"
        csv.write_text("<not counted>,,cycles:u,0,0.00,,\n")
        assert perf_mod._parse_perf_csv(csv)["cycles:u"] is None


class TestKernelEventGate:
    def test_paranoid_high_drops_kernel_events(self, monkeypatch, tmp_path):
        monkeypatch.setattr(perf_mod, "_read_perf_paranoid", lambda: 4)
        perf, argv = _run(tmp_path)
        assert "cycles:k" not in argv[argv.index("-e") + 1]
        assert perf["kernel_perf_paranoid"] == 4
        assert perf["kernel_events_skipped_reason"]
        assert perf["cycles_kernel"] is None

    def test_paranoid_low_includes_kernel_events(self, tmp_path):
        _, argv = _run(tmp_path)
        events = argv[argv.index("-e") + 1]
        assert "cycles:k" in events and "instructions:k" in events

    def test_perfmon_capability_overrides_paranoid(self, monkeypatch, tmp_path):
        """CAP_PERFMON/CAP_SYS_ADMIN bypass perf_event_paranoid in the
        kernel (measured: root in the MI210 container, paranoid=4,
        `cycles:k` counted fine)."""
        monkeypatch.setattr(perf_mod, "_read_perf_paranoid", lambda: 4)
        monkeypatch.setattr(perf_mod, "_has_perfmon_capability", lambda: True)
        perf, argv = _run(tmp_path)
        assert "cycles:k" in argv[argv.index("-e") + 1]
        assert "kernel_events_skipped_reason" not in perf


class TestPerfmonCapabilityProbe:
    """The autouse fixture stubs the probe out; these exercise the real one."""

    def test_reads_capeff_bitmask(self, tmp_path):
        """CapEff is a hex bitmask; CAP_SYS_ADMIN is bit 21 and
        CAP_PERFMON bit 38. Anything else present must not count."""
        status = tmp_path / "status"

        def probe(capeff: int) -> bool:
            status.write_text(f"Name:\tpython\nCapEff:\t{capeff:016x}\n")
            with patch("builtins.open", lambda *a, **k: status.open()):
                return _REAL_CAP_PROBE()

        assert probe(0) is False
        assert probe(1 << 12) is False  # CAP_NET_ADMIN alone
        assert probe(1 << 21) is True  # CAP_SYS_ADMIN
        assert probe(1 << 38) is True  # CAP_PERFMON

    def test_unreadable_status_is_not_privileged(self):
        def boom(*a, **k):
            raise OSError("no /proc")

        with patch("builtins.open", boom):
            assert _REAL_CAP_PROBE() is False


class TestBinaryResolution:
    def test_no_runnable_perf_returns_skipped(self, monkeypatch, tmp_path):
        monkeypatch.setattr(perf_mod.shutil, "which", lambda _: None)
        perf, argv = _run(tmp_path)
        assert argv is None
        assert "skipped" in perf

    def test_unrunnable_wrapper_falls_back_to_an_installed_build(
        self, monkeypatch, tmp_path
    ):
        """Ubuntu's /usr/bin/perf exits 2 when no linux-tools matches the
        running kernel — routine in a container. The image's own build
        still counts these events, so use it and say so."""
        monkeypatch.setattr(
            perf_mod,
            "_installed_perf_binaries",
            lambda: ["/usr/lib/linux-tools-6.8.0-136/perf"],
        )
        monkeypatch.setattr(
            perf_mod, "_perf_is_runnable", lambda b: b != "/usr/bin/perf"
        )
        perf, argv = _run(tmp_path)
        assert argv[0] == "/usr/lib/linux-tools-6.8.0-136/perf"
        assert perf["binary"] == "/usr/lib/linux-tools-6.8.0-136/perf"
        assert "/usr/bin/perf" in perf["binary_substituted"]

    def test_runnable_perf_is_never_substituted(self, monkeypatch, tmp_path):
        """Swapping in a build for another kernel when the real one works
        would change what the numbers mean."""
        monkeypatch.setattr(
            perf_mod, "_installed_perf_binaries", lambda: ["/usr/lib/other/perf"]
        )
        perf, _ = _run(tmp_path)
        assert perf["binary"] == "/usr/bin/perf"
        assert "binary_substituted" not in perf


class TestResultSlice:
    def test_counts_are_labelled_process_total_with_ipc(self, tmp_path):
        perf, _ = _run(tmp_path)
        assert perf["scope"] == "process_total"
        assert perf["ipc_user"] == pytest.approx(987654321 / 1234567890)
        assert perf["csv_path"] == str(tmp_path / "perf.csv")

    def test_nonzero_exit_keeps_partial_counters_and_tail(self, tmp_path):
        perf, _ = _run(tmp_path, returncode=2, stderr="perf: bad event\n")
        assert perf["returncode"] == 2
        assert "perf: bad event" in perf["error_tail"]
        assert perf["cycles_user"] == 1234567890

    def test_launch_failure_omits_csv_path(self, tmp_path):
        """A perf that never starts leaves the directory but no CSV;
        advertising the path anyway hands consumers an ENOENT."""
        perf, _ = _run(tmp_path, returncode=2, stderr="perf: x\n", write_csv=False)
        assert "csv_path" not in perf
        assert perf["returncode"] == 2

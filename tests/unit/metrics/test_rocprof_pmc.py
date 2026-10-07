# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for rocprofv3 PMC counter collection.

Avoids requiring a real rocprofv3 binary or rocpd db: the subprocess
is mocked, and the rocpd schema is reproduced just well enough that
the parser exercises its real SQL path against an in-test sqlite db.
"""

import os
import sqlite3
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from dnn_benchmarking.metrics import _subprocess, rocprof_pmc
from dnn_benchmarking.metrics._diagnostic import reset as reset_warn_once


@pytest.fixture(autouse=True)
def _reset():
    reset_warn_once()


def _run(out_dir, pmc_set="basic"):
    return rocprof_pmc.run(
        inner_argv=["python", "-m", "dnn_benchmarking"],
        out_dir=out_dir,
        timeout_s=60,
        context="g/E",
        pmc_set=pmc_set,
    )


class TestArgvBuild:
    def test_pins_rocpd_output_and_separates_inner_argv(self, tmp_path):
        args = rocprof_pmc._build_argv(
            counter_groups=[["GRBM_GUI_ACTIVE", "SQ_WAVES"]],
            out_dir=tmp_path,
            inner_argv=["python", "-m", "dnn_benchmarking", "--internal-profiling-run"],
        )
        assert args.count("--pmc") == 1
        # The parser reads the rocpd db; other rocprofv3 defaults are CSV.
        assert args[args.index("--output-format") + 1] == "rocpd"
        assert args[args.index("-o") + 1] == "results"
        sep = args.index("--")
        assert "GRBM_GUI_ACTIVE" in args[:sep]
        assert args[sep + 1 :] == [
            "python",
            "-m",
            "dnn_benchmarking",
            "--internal-profiling-run",
        ]

    def test_multi_group_emits_one_pmc_flag_per_group(self, tmp_path):
        """rocprofv3 expresses multipass as ``--pmc G1 --pmc G2 …`` — one
        flag per pass. A single flattened ``--pmc <every counter>`` is a
        single-pass request regardless of count, and silently overflows
        the hardware budget on arches with >hw-limit unioned counters.
        This guards against regressing to the historical flat emit.
        """
        argv = rocprof_pmc._build_argv(
            counter_groups=[
                ["GRBM_GUI_ACTIVE", "SQ_WAVES"],
                ["TCC_HIT_sum", "TCC_MISS_sum"],
                ["SQ_INSTS_VALU_MFMA_F16"],
            ],
            out_dir=tmp_path,
            inner_argv=["python"],
        )
        assert argv.count("--pmc") == 3
        # Each group's counters follow its --pmc and don't cross into
        # the next group.
        sep = argv.index("--")
        pmc_indices = [i for i, tok in enumerate(argv[:sep]) if tok == "--pmc"]
        # Group 1: GRBM_GUI_ACTIVE, SQ_WAVES sit between pmc[0]+1 and pmc[1].
        assert argv[pmc_indices[0] + 1 : pmc_indices[1]] == [
            "GRBM_GUI_ACTIVE",
            "SQ_WAVES",
        ]
        assert argv[pmc_indices[1] + 1 : pmc_indices[2]] == [
            "TCC_HIT_sum",
            "TCC_MISS_sum",
        ]
        # Group 3 runs from pmc[2]+1 to the first non-counter.
        fmt_idx = argv.index("--output-format")
        assert argv[pmc_indices[2] + 1 : fmt_idx] == ["SQ_INSTS_VALU_MFMA_F16"]


class TestResolveCounterGroups:
    def test_named_set_returns_single_group(self):
        groups = rocprof_pmc._resolve_counter_groups("gfx942", "basic")
        assert len(groups) == 1
        assert "GRBM_GUI_ACTIVE" in groups[0]

    def test_mi200_and_mi300_share_the_cdna_table(self):
        for pmc_set in ("basic", "memory", "flops", "all"):
            assert rocprof_pmc._resolve_counter_groups(
                "gfx90a", pmc_set
            ) == rocprof_pmc._resolve_counter_groups("gfx942", pmc_set)

    def test_all_returns_one_group_per_source_group(self):
        """``all`` must preserve pass boundaries, otherwise
        ``--pmc-allow-multipass`` is a lie: basic + memory + flops."""
        groups = rocprof_pmc._resolve_counter_groups("gfx942", "all")
        assert len(groups) == 3
        assert all(groups)
        basic_g = next(g for g in groups if "GRBM_GUI_ACTIVE" in g)
        memory_g = next(g for g in groups if "TCC_HIT_sum" in g)
        assert basic_g is not memory_g

    def test_unknown_arch_falls_back_to_basic_only(self):
        assert rocprof_pmc._resolve_counter_groups("gfx-mystery", "all") == [
            ["GRBM_GUI_ACTIVE", "SQ_WAVES"]
        ]
        assert rocprof_pmc._resolve_counter_groups("gfx-mystery", "memory") == []

    def test_unknown_set_returns_empty_outer_list(self):
        # Empty outer list signals "nothing to collect" to the caller —
        # not a single empty group, which would emit ``--pmc`` with no
        # counters and confuse rocprofv3.
        assert rocprof_pmc._resolve_counter_groups("gfx942", "bogus") == []


class TestSqliteIdentifierQuoting:
    def test_escapes_embedded_quotes(self):
        assert (
            rocprof_pmc._quote_sqlite_identifier('rocpd_pmc_event_"quoted"')
            == '"rocpd_pmc_event_""quoted"""'
        )


class TestRunHappyPath:
    def _make_synthetic_rocpd_db(self, db_path: Path) -> None:
        """Mirror the rocpd schema closely enough to exercise the parser.

        Uses uuid-suffixed table names so the parser's sqlite_master walk
        runs against shapes that match production output. Schema matches
        rocprofv3 1.2.2:
          * ``kernel_dispatch`` has ``dispatch_id`` + ``kernel_id`` but
            no ``kernel_name`` column.
          * Kernel names live in ``info_kernel_symbol.kernel_name`` and
            are joined via ``kernel_dispatch.kernel_id = info_kernel_symbol.id``.
          * ``pmc_event.event_id`` references ``kernel_dispatch.dispatch_id``.
        """
        suffix = "_abc123"
        conn = sqlite3.connect(db_path)
        try:
            conn.executescript(
                f"""
                CREATE TABLE rocpd_pmc_event{suffix} (
                    event_id INTEGER, pmc_id INTEGER, value REAL
                );
                CREATE TABLE rocpd_kernel_dispatch{suffix} (
                    id INTEGER PRIMARY KEY, kernel_id INTEGER, dispatch_id INTEGER
                );
                CREATE TABLE rocpd_info_kernel_symbol{suffix} (
                    id INTEGER PRIMARY KEY, kernel_name TEXT
                );
                CREATE TABLE rocpd_info_pmc{suffix} (
                    id INTEGER PRIMARY KEY, name TEXT
                );
                INSERT INTO rocpd_info_pmc{suffix} VALUES (1, 'GRBM_GUI_ACTIVE');
                INSERT INTO rocpd_info_pmc{suffix} VALUES (2, 'TCC_HIT_sum');
                INSERT INTO rocpd_info_pmc{suffix} VALUES (3, 'TCC_MISS_sum');
                INSERT INTO rocpd_info_kernel_symbol{suffix} VALUES (100, 'gemm_kernel');
                INSERT INTO rocpd_info_kernel_symbol{suffix} VALUES (200, 'fill_kernel');
                -- (kd.id, kd.kernel_id, kd.dispatch_id): gemm dispatched twice
                INSERT INTO rocpd_kernel_dispatch{suffix} VALUES (1, 100, 10);
                INSERT INTO rocpd_kernel_dispatch{suffix} VALUES (2, 100, 12);
                INSERT INTO rocpd_kernel_dispatch{suffix} VALUES (3, 200, 11);
                -- (pmc.event_id matches kd.dispatch_id, pmc.pmc_id, pmc.value)
                INSERT INTO rocpd_pmc_event{suffix} VALUES (10, 1, 1000);
                INSERT INTO rocpd_pmc_event{suffix} VALUES (12, 1, 3000);
                INSERT INTO rocpd_pmc_event{suffix} VALUES (10, 2, 30);
                INSERT INTO rocpd_pmc_event{suffix} VALUES (10, 3, 10);
                INSERT INTO rocpd_pmc_event{suffix} VALUES (12, 2, 30);
                INSERT INTO rocpd_pmc_event{suffix} VALUES (12, 3, 10);
                INSERT INTO rocpd_pmc_event{suffix} VALUES (11, 1, 7);
                """
            )
            conn.commit()
        finally:
            conn.close()

    def test_per_kernel_means_dispatch_counts_and_hit_rate(self, tmp_path, monkeypatch):
        monkeypatch.setattr(rocprof_pmc, "detect_arch", lambda: "gfx942")
        monkeypatch.setattr(rocprof_pmc, "resolve_rocm_tool", lambda name: "rocprofv3")
        out_dir = tmp_path / "pmc_out"

        def fake_run(argv, timeout_s=None):
            # rocprofv3 nests under <hostname>/; the run must hoist it.
            host_dir = Path(argv[argv.index("-d") + 1]) / "host"
            host_dir.mkdir(parents=True, exist_ok=True)
            self._make_synthetic_rocpd_db(host_dir / "results_results.db")
            return MagicMock(returncode=0, stdout="", stderr="")

        with patch.object(_subprocess, "run_capped", side_effect=fake_run):
            pmc = _run(out_dir)["pmc"]
        assert pmc["db_path"] == str(out_dir / "results.db")
        # No cross-kernel aggregate: fills and warmups would pollute it.
        assert "counters" not in pmc
        gemm = pmc["per_kernel"]["gemm_kernel"]
        assert gemm["dispatches"] == 2
        assert gemm["counters"]["GRBM_GUI_ACTIVE"] == 2000.0
        assert gemm["l2_hit_rate"] == pytest.approx(0.75)
        fill = pmc["per_kernel"]["fill_kernel"]
        assert fill == {"dispatches": 1, "counters": {"GRBM_GUI_ACTIVE": 7.0}}

    def test_missing_info_kernel_symbol_returns_warning(self, tmp_path, monkeypatch):
        """If the rocpd db omits info_kernel_symbol, the parser must not
        fall back to a broken SQL path — it should report the missing
        table and skip aggregation cleanly."""
        monkeypatch.setattr(rocprof_pmc, "detect_arch", lambda: "gfx942")
        monkeypatch.setattr(rocprof_pmc, "resolve_rocm_tool", lambda name: "rocprofv3")

        def fake_run(argv, timeout_s=None):
            db = Path(argv[argv.index("-d") + 1]) / "results.db"
            conn = sqlite3.connect(db)
            try:
                conn.executescript(
                    """
                    CREATE TABLE rocpd_pmc_event_x (event_id INTEGER, pmc_id INTEGER, value REAL);
                    CREATE TABLE rocpd_kernel_dispatch_x (id INTEGER, kernel_id INTEGER, dispatch_id INTEGER);
                    """
                )
                conn.commit()
            finally:
                conn.close()
            return MagicMock(returncode=0, stdout="", stderr="")

        with patch.object(_subprocess, "run_capped", side_effect=fake_run):
            pmc = _run(tmp_path)["pmc"]
        assert "info_kernel_symbol" in pmc["warnings"][0]


class TestRunFailureModes:
    def test_rocprofv3_nonzero_exit_records_error_tail(self, tmp_path, monkeypatch):
        monkeypatch.setattr(rocprof_pmc, "detect_arch", lambda: "gfx942")
        monkeypatch.setattr(rocprof_pmc, "resolve_rocm_tool", lambda name: "rocprofv3")
        proc = MagicMock(
            returncode=1,
            stdout="",
            stderr="rocprofv3: counter 'BOGUS' unsupported on this device\n",
        )
        with patch.object(_subprocess, "run_capped", return_value=proc):
            pmc = _run(tmp_path)["pmc"]
        assert pmc["returncode"] == 1
        assert "BOGUS" in pmc["error_tail"]
        assert "per_kernel" not in pmc

    def test_no_counters_for_arch_returns_skipped(self, tmp_path, monkeypatch):
        monkeypatch.setattr(rocprof_pmc, "detect_arch", lambda: "gfx-mystery")
        # Fallback table only defines 'basic'.
        assert _run(tmp_path, "memory")["pmc"]["skipped"] == "no counters defined"

    def test_invocation_raises_oserror_returns_skipped(self, tmp_path, monkeypatch):
        monkeypatch.setattr(rocprof_pmc, "detect_arch", lambda: "gfx942")
        monkeypatch.setattr(rocprof_pmc, "resolve_rocm_tool", lambda name: "rocprofv3")
        with patch.object(_subprocess, "run_capped", side_effect=OSError("boom")):
            pmc = _run(tmp_path)["pmc"]
        assert "boom" in pmc["skipped"]

    def test_rocprofv3_binary_missing_returns_skipped(self, tmp_path, monkeypatch):
        monkeypatch.setattr(rocprof_pmc, "detect_arch", lambda: "gfx942")
        monkeypatch.setattr(rocprof_pmc, "resolve_rocm_tool", lambda name: None)
        assert _run(tmp_path)["pmc"]["skipped"] == "profiling tool not found"

    def test_timeout_returns_skipped(self, tmp_path, monkeypatch):
        """A wedged rocprofv3 surfaces as skipped; --profiling-timeout
        reaches the cap."""
        monkeypatch.setattr(rocprof_pmc, "detect_arch", lambda: "gfx942")
        monkeypatch.setattr(rocprof_pmc, "resolve_rocm_tool", lambda name: "rocprofv3")
        seen = []

        def wedge(argv, timeout_s):
            seen.append(timeout_s)
            raise subprocess.TimeoutExpired(argv, timeout_s)

        with patch.object(_subprocess, "run_capped", side_effect=wedge):
            pmc = rocprof_pmc.run(
                inner_argv=["python"],
                out_dir=tmp_path,
                timeout_s=123,
                context="g/E",
                pmc_set="basic",
            )["pmc"]
        assert seen == [123]
        assert "timed out after 123s" in pmc["skipped"]


class TestArchNarrowing:
    """`--pmc all` on an arch without a PMC table silently narrows to
    the 2-counter fallback set. The user paid the
    --pmc-allow-multipass opt-in cost expecting a union of all groups
    and would otherwise see no diagnostic. The narrowing should both
    fire warn_once and set arch_narrowed_to_fallback in the result."""

    def test_all_on_unknown_arch_marks_narrowed(self, tmp_path, monkeypatch):
        monkeypatch.setattr(rocprof_pmc, "detect_arch", lambda: "gfx-mystery")
        monkeypatch.setattr(rocprof_pmc, "resolve_rocm_tool", lambda name: "rocprofv3")
        ok = MagicMock(returncode=0, stdout="", stderr="")
        with patch.object(_subprocess, "run_capped", return_value=ok):
            pmc = _run(tmp_path, "all")["pmc"]
        assert pmc.get("arch_narrowed_to_fallback") is True
        assert pmc["counters_requested"] == ["GRBM_GUI_ACTIVE", "SQ_WAVES"]

    def test_all_on_known_arch_does_not_mark_narrowed(self, tmp_path, monkeypatch):
        monkeypatch.setattr(rocprof_pmc, "detect_arch", lambda: "gfx942")
        monkeypatch.setattr(rocprof_pmc, "resolve_rocm_tool", lambda name: "rocprofv3")
        ok = MagicMock(returncode=0, stdout="", stderr="")
        with patch.object(_subprocess, "run_capped", return_value=ok):
            pmc = _run(tmp_path, "all")["pmc"]
        assert "arch_narrowed_to_fallback" not in pmc


class TestSqlitePathEscaping:
    @pytest.mark.skipif(
        os.name == "nt",
        reason="'?', '#', '%' are reserved in Windows filenames; the "
        "pathological dir cannot be created to exercise the URI-parser path",
    )
    def test_db_path_with_question_mark_opens_cleanly(self, tmp_path, monkeypatch):
        """sqlite3 URI parsing treats `?` as the start of a query
        string. The earlier `file:<path>?mode=ro` form would break on
        any user-controlled profiling output directory containing a
        `?`, `#`, or `%`. Opening by str path with PRAGMA query_only
        sidesteps the URI parser entirely."""
        monkeypatch.setattr(rocprof_pmc, "detect_arch", lambda: "gfx942")
        monkeypatch.setattr(rocprof_pmc, "resolve_rocm_tool", lambda name: "rocprofv3")
        # Pathological dir name with characters that broke the URI form.
        out_dir = tmp_path / "weird?dir#name%2F"
        out_dir.mkdir()
        conn = sqlite3.connect(out_dir / "results.db")
        try:
            conn.executescript(
                """
                CREATE TABLE rocpd_pmc_event_z (event_id INTEGER, pmc_id INTEGER, value REAL);
                CREATE TABLE rocpd_kernel_dispatch_z (id INTEGER, kernel_id INTEGER, dispatch_id INTEGER);
                CREATE TABLE rocpd_info_kernel_symbol_z (id INTEGER, kernel_name TEXT);
                """
            )
            conn.commit()
        finally:
            conn.close()
        ok = MagicMock(returncode=0, stdout="", stderr="")
        with patch.object(_subprocess, "run_capped", return_value=ok):
            pmc = _run(out_dir)["pmc"]
        assert pmc["per_kernel"] == {}

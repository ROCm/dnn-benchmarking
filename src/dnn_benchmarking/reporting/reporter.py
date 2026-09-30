# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Console output for benchmark suites.

Results (suite header, per-graph tables, verbose detail, summaries) go to
``output``; progress, info, warnings and errors go to ``err``. A progress
line left open on a TTY is always terminated before anything else is
written, including ``warn_once`` diagnostics routed through the sink.
"""

import shutil
import sys
import textwrap
from pathlib import Path
from statistics import geometric_mean
from typing import Any, Dict, List, Optional, Sequence, TextIO, Tuple

from ..metrics import _diagnostic
from .statistics import BenchmarkStats, noise_warnings
from .suite_results import GraphResult, ProviderEngineResult, SuiteResult, engine_id_hex

_TIMING_MODE_LABEL = {"staged": "staged stall-gate", "events": "per-launch events"}
_CLOCK_KEYS = (
    ("sclk_mhz", "sclk", "MHz"),
    ("mclk_mhz", "mclk", "MHz"),
    ("power_w", "power", "W"),
    ("temp_hotspot_c", "hotspot", "C"),
    ("throttle_status", "throttle", ""),
)


def _width() -> int:
    return shutil.get_terminal_size((120, 24)).columns


def _clip(text: str, width: int) -> str:
    """Truncate to ``width`` columns, marking the cut with an ellipsis."""
    if len(text) <= width:
        return text
    return text[: max(width - 1, 0)] + "…" if width > 0 else ""


def _one_line(text: str) -> str:
    return " ".join(text.split())


def _unit(ms: float) -> Tuple[float, str]:
    """Scale and suffix that render ``ms`` with a readable magnitude."""
    if ms < 1.0:
        return 1e3, "µs"
    if ms < 1e3:
        return 1.0, "ms"
    return 1e-3, "s"


def _fmt_time(ms: float) -> str:
    """Per-launch time with an automatic µs/ms/s unit."""
    scale, suffix = _unit(ms)
    digits = 2 if suffix == "µs" else 3
    return f"{ms * scale:.{digits}f} {suffix}"


def _fmt_duration(ms: float) -> str:
    """Wall-clock cost (setup, build): 2 significant digits below 10 ms."""
    if ms < 10:
        return f"{ms:.2g} ms"
    return f"{ms:.0f} ms" if ms < 1e3 else f"{ms / 1e3:.1f} s"


def _iqr_pct(stats: BenchmarkStats) -> float:
    return stats.iqr_ms / stats.median_ms * 100 if stats.median_ms > 0 else 0.0


def _fmt_mib(mib: float) -> str:
    return f"{mib / 1024:.2f} GiB" if mib >= 1024 else f"{mib:.1f} MiB"


def _display_name(pe: ProviderEngineResult) -> str:
    return pe.engine_name or pe.provider


def _reason(pe: ProviderEngineResult) -> Optional[str]:
    """Why a row did not produce timings, or None for successful rows."""
    if pe.status == "skipped":
        return pe.skip_reason or "no reason given"
    if pe.status == "error":
        return pe.error_message or "no error message"
    return None


def _oracle_state(pe: ProviderEngineResult) -> Optional[str]:
    """Why the oracle speedup is not reportable, or None when it is."""
    tuned_verdict = pe.oracle.correctness if pe.oracle is not None else None
    if any(
        v is not None and v.explicitly_failed for v in (pe.correctness, tuned_verdict)
    ):
        # A wrong baseline or tuned plan cannot measure a gain.
        return "invalid"
    if pe.oracle is not None and not pe.oracle.tuning_available:
        # One fixed configuration: the ratio is run-to-run noise.
        return "no-search"
    if pe.oracle_delta is not None:
        return None
    if pe.oracle_error is not None:
        return "failed"
    return "n/a"


class Reporter:
    """Formats benchmark progress and results for the console."""

    def __init__(
        self,
        output: Optional[TextIO] = None,
        err: Optional[TextIO] = None,
        *,
        quiet: bool = False,
        verbose: bool = False,
    ) -> None:
        """Bind output streams.

        Args:
            output: Results stream (default: sys.stdout).
            err: Progress/diagnostics stream. Defaults to ``output`` when
                ``output`` is given explicitly, else sys.stderr.
            quiet: Suppress progress lines and info messages.
            verbose: Add a per-engine detail block under each graph table.
        """
        if err is None:
            err = output if output is not None else sys.stderr
        self._out = output if output is not None else sys.stdout
        self._err = err
        self._quiet = quiet
        self._verbose = verbose
        self._tty = err.isatty()
        self._engine_head: Optional[str] = None  # engine progress line in flight
        self._pending: Optional[str] = None  # TTY line written without its newline
        self._legend_done = False

    # Stream plumbing

    def _break(self) -> None:
        """Terminate a pending progress line so the next write starts clean."""
        if self._pending is not None:
            self._pending = None
            _diagnostic.set_sink(None)
            self._err.write("\n")
            self._err.flush()

    def _print(self, text: str = "") -> None:
        self._break()
        print(text, file=self._out, flush=True)

    def _print_err(self, text: str) -> None:
        self._break()
        print(text, file=self._err, flush=True)

    def _begin(self, head: str) -> None:
        if self._quiet or not self._tty:
            return
        self._break()
        self._err.write(head)
        self._err.flush()
        self._pending = head
        _diagnostic.set_sink(self._print_err)

    def _finish(self, head: str, outcome: str) -> None:
        if self._quiet:
            return
        outcome = _clip(outcome, _width() - len(head) - 1)
        if self._pending == head:
            # Nothing interrupted the line: complete it in place.
            self._pending = None
            _diagnostic.set_sink(None)
            self._err.write(f" {outcome}\n")
            self._err.flush()
        else:
            self._print_err(f"{head} {outcome}")

    # Messages

    def info(self, msg: str) -> None:
        """Progress-level note (suppressed by ``quiet``)."""
        if not self._quiet:
            self._print_err(msg)

    def warning(self, msg: str) -> None:
        """Non-fatal problem; always shown."""
        self._print_err(f"WARNING: {msg}")

    def error(self, msg: str) -> None:
        """Fatal or row-level error; always shown."""
        self._print_err(f"ERROR: {msg}")

    # Progress

    def graph_start(self, index: int, total: int, name: str) -> None:
        """Announce a graph: ``[i/n] name``."""
        if not self._quiet:
            self._print_err(f"[{index}/{total}] {name}")

    def engine_start(self, label: str) -> None:
        """Open the progress line for one engine row."""
        self._engine_head = f"  {label} ..."
        self._begin(self._engine_head)

    def engine_done(self, result: ProviderEngineResult) -> None:
        """Complete the engine progress line with the row outcome."""
        head = self._engine_head or f"  {_display_name(result)} ..."
        self._engine_head = None
        self._finish(head, self._outcome(result))

    def profiling_start(self, label: str) -> None:
        """Open the progress line for an opt-in profiling pass."""
        self._begin(f"    profiling {label} ...")

    def profiling_done(self, label: str, seconds: float) -> None:
        """Complete the profiling progress line."""
        self._finish(f"    profiling {label} ...", f"done ({seconds:.1f} s)")

    @staticmethod
    def _outcome(pe: ProviderEngineResult) -> str:
        reason = _reason(pe)
        if reason is not None:
            return f"{pe.status}: {_one_line(reason)}"
        parts = [pe.verdict]
        kernel = pe.gpu_kernel_stats
        if kernel is not None:
            parts.append(_fmt_time(kernel.median_ms))
        detail = []
        if kernel is not None:
            detail.append(f"iqr {_iqr_pct(kernel):.1f}%")
        if pe.timing is not None:
            detail.append(f"setup {_fmt_duration(pe.timing.first_call_ms)}")
        elif pe.elapsed_time_ms:
            detail.append(f"took {_fmt_duration(pe.elapsed_time_ms)}")
        if detail:
            parts.append(f"({', '.join(detail)})")
        return "  ".join(parts)

    # Suite header and summaries

    def print_suite_header(
        self, env: Dict[str, Any], run_config: Dict[str, Any], n_graphs: int
    ) -> None:
        """Print the machine and methodology lines that head every run."""
        self._print(f"dnn-benchmark: {n_graphs} graph(s)")
        self._print(f"Host:    {env.get('cpu_model') or 'unknown CPU'}")
        gpu_extras = [
            text
            for text in (
                env.get("gpu_arch"),
                (
                    f"{env['gpu_compute_units']} CUs"
                    if env.get("gpu_compute_units")
                    else None
                ),
                f"{env['gpu_hbm_gb']:g} GB HBM" if env.get("gpu_hbm_gb") else None,
            )
            if text
        ]
        gpu = env.get("gpu_model") or "unknown GPU"
        self._print(
            f"GPU:     {gpu}" + (f" ({', '.join(gpu_extras)})" if gpu_extras else "")
        )
        # A CUDA wheel reports cuda_version; a ROCm wheel does not.
        if env.get("cuda_version"):
            cudnn = env.get("cudnn_version")
            self._print(
                f"CUDA:    {env['cuda_version']}"
                + (f", cuDNN {cudnn}" if cudnn else "")
            )
        else:
            self._print(f"ROCm:    {env.get('rocm_version') or 'unknown'}")
        rc = run_config
        self._print(
            f"Timing:  warmup {rc['warmup_iters']}, iters {rc['iters']} "
            f"(min-time {rc['min_time_ms']:g} ms), cache {rc['cache_mode']}, "
            f"seed {rc['seed']}, backend {rc['backend']}"
        )
        self._print()

    def print_summary(
        self, suite_result: SuiteResult, output_path: Optional[str]
    ) -> None:
        """Print row/graph counts and where the results were written."""
        s = suite_result.summary()
        line = (
            f"Summary: {s['graphs']} graph(s), {s['rows']} row(s): "
            f"{s['passed']} passed, {s['unchecked']} unchecked, {s['failed']} failed, "
            f"{s['skipped']} skipped, {s['errors']} error(s)"
        )
        extras = []
        if s["graph_errors"]:
            extras.append(f"{s['graph_errors']} graph error(s)")
        if s["no_engine_graphs"]:
            extras.append(f"{s['no_engine_graphs']} graph(s) without engines")
        if extras:
            line += "; " + ", ".join(extras)
        self._print(line)
        if output_path is not None:
            self._print(f"Results: {output_path}")

    def print_oracle_summary(self, graphs: Sequence[GraphResult]) -> None:
        """Print the suite-wide geomean of reportable oracle speedups."""
        rows = [pe for gr in graphs for pe in gr.results if pe.oracle_delta is not None]
        if not rows:
            return
        speedups = [pe.oracle_delta.speedup for pe in rows if _oracle_state(pe) is None]
        excluded = len(rows) - len(speedups)
        if not speedups:
            self._print(
                f"Oracle: no reportable speedup on any of {excluded} tuned row(s) "
                "(no tuning alternatives or invalid results)"
            )
            return
        suffix = (
            f"; {excluded} row(s) excluded (no search or invalid)" if excluded else ""
        )
        self._print(
            f"Oracle: {len(speedups)} tuned row(s), geomean speedup "
            f"{geometric_mean(speedups):.2f}x{suffix}"
        )

    # Per-graph table

    def print_graph_table(self, graph_result: GraphResult) -> None:
        """Print one row per engine; with ``verbose`` also the detail blocks."""
        gr = graph_result
        title = gr.graph_name
        stem = Path(gr.graph_path).stem
        if stem != gr.graph_name:
            # The progress line announced the file stem; show both to link them.
            title = f"{stem} ({gr.graph_name})"
        self._print(title + (f"  [{gr.graph_id}]" if gr.graph_id else ""))
        if gr.error:
            self._print(_clip(f"  graph error: {_one_line(gr.error)}", _width()))
        elif not gr.results:
            msg = f": {_one_line(gr.message)}" if gr.message else ""
            self._print(_clip(f"  no engines applicable{msg}", _width()))
        if not gr.results:
            self._print()
            return

        rows = gr.results
        best_candidates = [
            pe.gpu_kernel_stats.median_ms
            for pe in rows
            if pe.role == "engine"
            and pe.verdict in ("passed", "unchecked")
            and pe.gpu_kernel_stats is not None
            and pe.gpu_kernel_stats.median_ms > 0
        ]
        best = min(best_candidates) if best_candidates else None
        with_oracle = any(pe.oracle is not None or pe.oracle_error for pe in rows)

        # (header, right-aligned, cells)
        columns: List[Tuple[str, bool, List[str]]] = [
            ("engine", False, [_display_name(pe) for pe in rows]),
            ("verdict", False, [pe.verdict for pe in rows]),
            ("kernel_med", True, [self._kernel_cell(pe) for pe in rows]),
            (
                "iqr%",
                True,
                [
                    (
                        "-"
                        if pe.gpu_kernel_stats is None
                        else f"{_iqr_pct(pe.gpu_kernel_stats):.1f}"
                    )
                    for pe in rows
                ],
            ),
            (
                "submit",
                True,
                [
                    "-" if pe.host_stats is None else _fmt_time(pe.host_stats.median_ms)
                    for pe in rows
                ],
            ),
            ("tflops", True, [self._tflops_cell(pe) for pe in rows]),
            (
                "gbps",
                True,
                [
                    (
                        "-"
                        if pe.derived_gbytes_per_s is None
                        else f"{pe.derived_gbytes_per_s:.1f}"
                    )
                    for pe in rows
                ],
            ),
            ("vs_best", True, [self._vs_best_cell(pe, best) for pe in rows]),
        ]
        if with_oracle:
            columns.append(("oracle", True, [self._oracle_cell(pe) for pe in rows]))
        notes = [self._note(pe) for pe in rows]
        if any(notes):
            columns.append(("note", False, notes))

        width = _width()
        natural = [max(len(h), *(len(c) for c in cells)) for h, _, cells in columns]
        # Engine name and note absorb the squeeze; numeric columns never do.
        has_note = columns[-1][0] == "note"
        flexible = {0, len(columns) - 1} if has_note else {0}
        budget = (
            width
            - 2
            - 2 * (len(columns) - 1)
            - sum(w for i, w in enumerate(natural) if i not in flexible)
        )
        if has_note:
            # The name keeps up to 32 columns; the note gets what remains.
            natural[0] = min(natural[0], max(16, min(32, budget - len("note"))))
            natural[-1] = max(len("note"), min(natural[-1], budget - natural[0]))
        else:
            natural[0] = min(natural[0], max(16, budget))

        def render(cells: List[str]) -> str:
            parts = []
            for (_, right, _), w, cell in zip(columns, natural, cells):
                cell = _clip(cell, w)
                parts.append(cell.rjust(w) if right else cell.ljust(w))
            return _clip(("  " + "  ".join(parts)).rstrip(), width)

        self._print(render([h for h, _, _ in columns]))
        for i in range(len(rows)):
            self._print(render([cells[i] for _, _, cells in columns]))
        if not self._legend_done:
            self._legend_done = True
            for line in textwrap.wrap(
                self._legend(rows), width, subsequent_indent="  "
            ):
                self._print(line)
        self._print()
        if self._verbose:
            self.print_graph_verbose(gr)

    @staticmethod
    def _legend(rows: Sequence[ProviderEngineResult]) -> str:
        timing = next((pe.timing for pe in rows if pe.timing is not None), None)
        how = (
            f" ({_TIMING_MODE_LABEL.get(timing.mode, timing.mode)}; cache {timing.cache_mode})"
            if timing is not None
            else ""
        )
        return (
            f"  kernel_med = median device time per launch{how}; "
            "submit = host enqueue time; vs_best = best median / row median; * = noisy"
        )

    @staticmethod
    def _kernel_cell(pe: ProviderEngineResult) -> str:
        stats = pe.gpu_kernel_stats
        if stats is None:
            return "- "
        # Always one marker column so the numbers stay aligned.
        return _fmt_time(stats.median_ms) + ("*" if noise_warnings(stats) else " ")

    @staticmethod
    def _tflops_cell(pe: ProviderEngineResult) -> str:
        if pe.derived_tflops_per_s is None:
            return "-"
        return (
            "~" if pe.analytical_flops_partial else ""
        ) + f"{pe.derived_tflops_per_s:.2f}"

    @staticmethod
    def _vs_best_cell(pe: ProviderEngineResult, best: Optional[float]) -> str:
        if pe.role == "reference":
            return "ref"
        stats = pe.gpu_kernel_stats
        if (
            best is None
            or pe.status != "success"
            or stats is None
            or stats.median_ms <= 0
        ):
            return "-"
        return f"{best / stats.median_ms:.2f}x"

    @staticmethod
    def _oracle_cell(pe: ProviderEngineResult) -> str:
        state = _oracle_state(pe)
        return state if state is not None else f"{pe.oracle_delta.speedup:.2f}x"

    @staticmethod
    def _note(pe: ProviderEngineResult) -> str:
        reason = _reason(pe)
        if reason is not None:
            return _one_line(reason)
        # The kernel_med '*' marker already flags noise.
        warnings = [w for w in pe.warnings or [] if not w.startswith("noisy:")]
        if not warnings:
            return ""
        more = f" (+{len(warnings) - 1})" if len(warnings) > 1 else ""
        return _one_line(warnings[0]) + more

    # Verbose detail

    def print_graph_verbose(self, graph_result: GraphResult) -> None:
        """Print a compact detail block for every row of a graph."""
        for pe in graph_result.results:
            for line in self._detail_lines(pe):
                self._print(line)
            self._print()

    def _detail_lines(self, pe: ProviderEngineResult) -> List[str]:
        hex_id = engine_id_hex(pe.engine_id)
        title = _display_name(pe) + (f" ({hex_id})" if hex_id else "")
        if pe.role == "reference":
            title += "  [reference]"
        lines = [f"  {title}"]

        def add(key: str, text: str) -> None:
            lines.append(f"    {key:<12}{text}")

        if pe.plugin_path:
            add("plugin", pe.plugin_path)
        reason = _reason(pe)
        if reason is not None:
            add(pe.status, reason)
        costs = []
        if pe.cpu_build_time_ms is not None:
            costs.append(f"build {_fmt_duration(pe.cpu_build_time_ms)}")
        if pe.timing is not None:
            costs.append(f"first call {_fmt_duration(pe.timing.first_call_ms)}")
        if pe.elapsed_time_ms:
            costs.append(f"row total {_fmt_duration(pe.elapsed_time_ms)}")
        if costs:
            add("cost", ", ".join(costs))
        if pe.timing is not None:
            t = pe.timing
            text = (
                f"{t.mode}/{t.backend}, cache {t.cache_mode}, warmup {t.warmup_iters}"
            )
            if t.capped:
                text += ", capped at max iters"
            if t.fallback_reason:
                text += f", fallback: {t.fallback_reason}"
            add("timing", text)
        lines.extend(self._stats_lines(pe))
        clocks = self._clocks_text(pe.clocks_before, pe.clocks_after)
        if clocks:
            add("clocks", clocks)
        metrics = self._metrics_text(pe)
        if metrics:
            add("metrics", metrics)
        if pe.correctness is not None and pe.role == "engine":
            add("correctness", self._correctness_text(pe))
        for line in self._oracle_lines(pe):
            add("oracle", line)
        for line in self._profiling_lines(pe.extra_metrics or {}):
            add("profiling", line)
        for warning in pe.warnings or []:
            add("warning", warning)
        return lines

    @staticmethod
    def _stats_lines(pe: ProviderEngineResult) -> List[str]:
        named = [
            (n, s)
            for n, s in (("kernel", pe.gpu_kernel_stats), ("submit", pe.host_stats))
            if s
        ]
        if not named:
            return []
        cols = ("mean", "median", "std", "min", "p95", "max")
        lines = [f"    {'':<12}{'n':>6}" + "".join(f"{c:>10}" for c in cols)]
        for name, s in named:
            scale, suffix = _unit(s.median_ms)
            values = (
                s.mean_ms,
                s.median_ms,
                s.std_ms,
                s.min_ms,
                s.p95_ms if s.n >= 20 else None,
                s.max_ms,
            )
            cells = "".join(
                f"{'-':>10}" if v is None else f"{v * scale:>10.3f}" for v in values
            )
            lines.append(f"    {name:<12}{s.n:>6}{cells}  {suffix}")
        return lines

    @staticmethod
    def _clocks_text(
        before: Optional[Dict[str, Any]], after: Optional[Dict[str, Any]]
    ) -> str:
        before, after = before or {}, after or {}
        parts = []
        for key, label, unit in _CLOCK_KEYS:
            a, b = before.get(key), after.get(key)
            if a is None and b is None:
                continue
            a_text, b_text = ("?" if v is None else f"{v:g}" for v in (a, b))
            parts.append(f"{label} {a_text}->{b_text}" + (f" {unit}" if unit else ""))
        return ", ".join(parts)

    @staticmethod
    def _metrics_text(pe: ProviderEngineResult) -> str:
        parts = []
        if pe.workspace_bytes is not None:
            parts.append(f"workspace {_fmt_mib(pe.workspace_bytes / 2**20)}")
        if pe.analytical_flops is not None:
            partial = " (partial)" if pe.analytical_flops_partial else ""
            parts.append(f"flops {pe.analytical_flops:,}{partial}")
        elif pe.analytical_flops_partial:
            parts.append("flops n/a (no analytical model)")
        if pe.analytical_io_bytes is not None:
            parts.append(f"io {_fmt_mib(pe.analytical_io_bytes / 2**20)}")
        if pe.vram_used_mb is not None:
            parts.append(f"vram {_fmt_mib(pe.vram_used_mb)}")
        return ", ".join(parts)

    @staticmethod
    def _correctness_text(pe: ProviderEngineResult) -> str:
        c = pe.correctness
        if c.tolerance_match is None:
            return f"unchecked ({c.error_message or 'no comparison performed'})"
        parts = [f"rtol {c.rtol:.0e}", f"atol {c.atol:.0e}"]
        if c.max_abs_diff is not None:
            parts.append(f"max_abs_diff {c.max_abs_diff:.2e}")
        if c.max_rel_diff is not None:
            parts.append(f"max_rel_diff {c.max_rel_diff:.2e}")
        if c.n_mismatch is not None:
            total = f"/{c.n_total}" if c.n_total is not None else ""
            parts.append(f"n_mismatch {c.n_mismatch}{total}")
        if c.worst_output_uid is not None:
            parts.append(f"worst output {c.worst_output_uid}")
        verdict = "passed" if c.tolerance_match else "FAILED"
        return f"{verdict} ({', '.join(parts)})"

    @staticmethod
    def _oracle_lines(pe: ProviderEngineResult) -> List[str]:
        o = pe.oracle
        if o is None:
            return [f"unavailable: {pe.oracle_error}"] if pe.oracle_error else []
        knobs = (
            ", ".join(f"{k['knob_id']}={k['value']}" for k in o.knob_settings)
            if o.knob_settings
            else "engine defaults"
        )
        lines = [
            f"plan {o.plan_name} (index {o.compiled_plan_index}, rank {o.rank}); knobs {knobs}",
            f"{o.compiled_plans_benchmarked}/{o.compiled_plans_total} compiled plans benchmarked "
            f"({o.compiled_plans_failed} failed); sweep min {_fmt_time(o.sweep_min_time_ms)}",
        ]
        if o.exhaustive_requested:
            lines.append(
                "exhaustive search enabled (a cached selection may be reused)"
                if o.exhaustive_supported
                else "exhaustive unsupported by this engine; plan-level tuning only"
            )
        if not o.tuning_available:
            lines.append(
                "no tuning alternative: re-measured the heuristic plan; delta is noise"
            )
        if o.correctness is not None and not o.correctness.passed:
            detail = o.correctness.error_message or "output mismatch"
            lines.append(
                f"tuned plan FAILED validation ({detail}); no speedup reported"
            )
        d = pe.oracle_delta
        if d is not None:
            lines.append(
                f"{_fmt_time(d.baseline_median_ms)} warm heuristic -> "
                f"{_fmt_time(d.oracle_median_ms)} tuned = {d.speedup:.2f}x (basis {d.basis})"
            )
        return lines

    @staticmethod
    def _profiling_lines(extra: Dict[str, Any]) -> List[str]:
        """One-line summaries per opt-in profiling source; full data is in JSON."""
        lines: List[str] = []

        def failures(name: str, slc: Dict[str, Any]) -> None:
            if "skipped" in slc:
                lines.append(f"{name}: skipped — {slc['skipped']}")
            if "unexpected_error" in slc:
                lines.append(f"{name}: unexpected error — {slc['unexpected_error']}")
            if "error_tail" in slc:
                if "skipped" not in slc:
                    lines.append(f"{name}: failed (rc={slc.get('returncode', '?')})")
                lines.extend(
                    f"  | {t}" for t in str(slc["error_tail"]).splitlines()[-3:]
                )

        trace = extra.get("trace")
        if isinstance(trace, dict):
            name = f"trace ({trace.get('format', '?')})"
            if trace.get("path"):
                lines.append(
                    f"{name}: {trace['path']}  (open in https://ui.perfetto.dev/)"
                )
            failures(name, trace)

        pmc = extra.get("pmc")
        if isinstance(pmc, dict):
            name = f"pmc ({pmc.get('set', '?')}, {pmc.get('arch', '?')})"
            per_kernel = pmc.get("per_kernel") or {}
            ranked = sorted(
                per_kernel.items(), key=lambda kv: -kv[1].get("dispatches", 0)
            )
            for kernel, data in ranked[:3]:
                counters = data.get("counters") or {}
                shown = "  ".join(
                    f"{c}={v:,.4g}" for c, v in list(counters.items())[:3]
                )
                more = f"  [+{len(counters) - 3}]" if len(counters) > 3 else ""
                l2 = data.get("l2_hit_rate")
                l2_text = f"  l2_hit {l2:.1%}" if isinstance(l2, (int, float)) else ""
                lines.append(
                    f"{name}: {_clip(kernel, 40)} x{data.get('dispatches', '?')}: "
                    f"{shown}{more}{l2_text}"
                )
            if len(ranked) > 3:
                lines.append(f"{name}: [{len(ranked) - 3} more kernel(s), see JSON]")
            failures(name, pmc)
            if pmc.get("db_path"):
                db = pmc["db_path"]
                lines.append(f"pmc db: {db}")
                lines.append(f"  -> rocprof-compute analyze --path {Path(db).parent}")

        perf = extra.get("perf")
        if isinstance(perf, dict):
            bits = []
            for key, label, fmt in (
                ("ipc_user", "IPC", "{:.2f}"),
                ("cycles_user", "cycles_u", "{:,.0f}"),
                ("instructions_user", "instr_u", "{:,.0f}"),
                ("task_clock_ms", "task_clock", "{:.1f}ms"),
            ):
                value = perf.get(key)
                if isinstance(value, (int, float)):
                    bits.append(f"{label}={fmt.format(value)}")
            if bits:
                scope = f"  ({perf['scope']})" if perf.get("scope") else ""
                lines.append(f"perf: {'  '.join(bits)}{scope}")
            failures("perf", perf)

        roofline = extra.get("roofline")
        if isinstance(roofline, dict):
            if roofline.get("roofline_csv"):
                lines.append(f"roofline: {roofline['roofline_csv']}")
            if roofline.get("workload_path"):
                lines.append(
                    f"  -> rocprof-compute analyze --path {roofline['workload_path']} "
                    "--block 4  (ASCII; add --gui for web UI)"
                )
            failures("roofline", roofline)
        return lines

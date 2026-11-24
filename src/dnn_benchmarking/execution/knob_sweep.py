# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Exact candidate-tuple knob discovery and strict knob validation for hipDNN.

Hard engine selection in hipDNN warns and drops unknown knobs rather than
failing. This module validates user-supplied ``--knob`` settings strictly
against the engine's exposed knobs, and discovers exact candidate tuples for
``--sweep-knobs`` without blindly expanding a Cartesian product of marginal
knob domains.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from ..common.exceptions import ExecutionError, InvalidKnobError, KnobConstraintError


@dataclass(frozen=True)
class KnobCandidate:
    """One exact candidate kernel tuple for a hipDNN engine."""

    candidate_id: str
    knobs: Dict[str, Any]


def _parse_candidate_entry(index: int, entry: Any) -> KnobCandidate:
    """Normalize one candidate entry from JSON or ``hipdnn_bench enumerate``."""
    if not isinstance(entry, dict):
        raise ExecutionError(
            f"Candidate entry {index} must be a JSON object, got {type(entry).__name__}"
        )
    if "knobs" in entry and isinstance(entry["knobs"], dict):
        knobs = dict(entry["knobs"])
        cid = str(
            entry.get("id")
            or entry.get("candidate_id")
            or entry.get("descriptor_id")
            or entry.get("name")
            or f"c{index}"
        )
        return KnobCandidate(candidate_id=cid, knobs=knobs)
    if "knob_settings" in entry and isinstance(entry["knob_settings"], dict):
        knobs = dict(entry["knob_settings"])
        cid = str(
            entry.get("id")
            or entry.get("candidate_id")
            or entry.get("descriptor_id")
            or entry.get("name")
            or f"c{index}"
        )
        return KnobCandidate(candidate_id=cid, knobs=knobs)
    # Plain dict of knob_id -> value.
    cid = ",".join(f"{k}={v}" for k, v in sorted(entry.items())) or f"c{index}"
    return KnobCandidate(candidate_id=cid, knobs=dict(entry))


def load_knob_candidates(path: Path) -> List[KnobCandidate]:
    """Load exact candidate knob tuples from a JSON file."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise ExecutionError(f"Cannot read --knob-candidates {path}: {e}") from e
    if isinstance(raw, dict) and isinstance(raw.get("candidates"), list):
        items = raw["candidates"]
    elif isinstance(raw, list):
        items = raw
    else:
        raise ExecutionError(
            f"--knob-candidates {path} must contain a JSON array or "
            "{'candidates': [...]}"
        )
    if not items:
        raise ExecutionError(f"--knob-candidates {path} contains no candidates")
    return [_parse_candidate_entry(i, item) for i, item in enumerate(items)]


def _find_hipdnn_bench() -> Optional[str]:
    """Locate ``hipdnn_bench`` on PATH or under ``$ROCM_PATH/bin``."""
    found = shutil.which("hipdnn_bench")
    if found:
        return found
    rocm_path = os.environ.get("ROCM_PATH")
    if rocm_path:
        candidate = Path(rocm_path) / "bin" / "hipdnn_bench"
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def enumerate_candidates(
    *,
    graph_path: Path,
    engine_name: str,
    plugin_path: Optional[Path] = None,
    engine_id: Optional[int] = None,
    candidates_file: Optional[Path] = None,
    preloaded_candidates: Optional[Sequence[KnobCandidate]] = None,
) -> List[KnobCandidate]:
    """Enumerate exact candidate tuples for one hipDNN engine on one graph.

    Discovery order:
    1. Preloaded candidates (parsed at CLI startup from ``--knob-candidates``).
    2. Explicit ``--knob-candidates`` JSON file.
    3. ``hipdnn_bench enumerate --json`` subprocess (discovery only; timing
       always runs through ``dnn_benchmarking``).
    """
    if preloaded_candidates is not None:
        return list(preloaded_candidates)
    if candidates_file is not None:
        return load_knob_candidates(candidates_file)

    bench = _find_hipdnn_bench()
    if bench is None:
        raise ExecutionError(
            "--sweep-knobs requires exact candidate enumeration via "
            "hipdnn_bench on PATH or --knob-candidates <path>"
        )
    cmd = [
        bench,
        "enumerate",
        "--json",
        "--graph",
        str(graph_path),
        "--engine-name",
        engine_name,
    ]
    if plugin_path is not None:
        pdir = plugin_path.parent if plugin_path.is_file() else plugin_path
        cmd += ["--plugin-dir", str(pdir)]
    try:
        proc = subprocess.run(
            cmd, check=False, capture_output=True, text=True, timeout=60
        )
    except (OSError, subprocess.SubprocessError) as e:
        raise ExecutionError(f"hipdnn_bench enumerate failed: {e}") from e
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        raise ExecutionError(
            f"hipdnn_bench enumerate exited {proc.returncode}: {detail}"
        )
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        raise ExecutionError(
            f"hipdnn_bench enumerate returned invalid JSON: {e}"
        ) from e
    items = (
        payload.get("candidates", [])
        if isinstance(payload, dict)
        else payload if isinstance(payload, list) else []
    )
    return [_parse_candidate_entry(i, item) for i, item in enumerate(items)]


def _coerce_and_check_constraint(knob_id: str, knob: Any, value: Any) -> Any:
    """Validate ``value`` against a hipDNN ``Knob`` value type and constraint."""
    vtype = str(getattr(knob, "value_type", "")).upper()
    if "INT" in vtype:
        if isinstance(value, bool) or not isinstance(value, int):
            raise InvalidKnobError(
                f"Knob {knob_id!r} expects an integer, got {value!r}"
            )
    elif "FLOAT" in vtype or "DOUBLE" in vtype:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise InvalidKnobError(f"Knob {knob_id!r} expects a float, got {value!r}")
        value = float(value)
    elif "STRING" in vtype:
        if not isinstance(value, str):
            value = str(value)

    constraint = getattr(knob, "constraint", None)
    if constraint is None:
        return value
    valid_values = getattr(constraint, "valid_values", None)
    if valid_values:
        allowed = list(valid_values)
        if value not in allowed:
            raise KnobConstraintError(
                f"Knob {knob_id!r}={value!r} is not in allowed values {allowed}"
            )
    min_val = getattr(constraint, "min_value", None)
    max_val = getattr(constraint, "max_value", None)
    if min_val is not None and value < min_val:
        raise KnobConstraintError(
            f"Knob {knob_id!r}={value!r} is below minimum {min_val!r}"
        )
    if max_val is not None and value > max_val:
        raise KnobConstraintError(
            f"Knob {knob_id!r}={value!r} exceeds maximum {max_val!r}"
        )
    return value


def validate_knobs_strict(
    exposed_knobs: Sequence[Any], requested: Dict[str, Any]
) -> Dict[str, Any]:
    """Verify every requested knob exists on the engine and meets its constraint.

    Raises:
        InvalidKnobError: Any requested knob is unknown or out of domain.
    """
    by_id = {str(k.knob_id): k for k in exposed_knobs}
    validated: Dict[str, Any] = {}
    for knob_id, raw_value in requested.items():
        if knob_id not in by_id:
            avail = ", ".join(sorted(by_id)) or "none"
            raise InvalidKnobError(
                f"Knob {knob_id!r} is not exposed by this engine (available: {avail})"
            )
        validated[knob_id] = _coerce_and_check_constraint(
            knob_id, by_id[knob_id], raw_value
        )
    return validated

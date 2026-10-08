# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""hipDNN graph build and timed execution."""

import json
from typing import Any, Dict, List, Optional

from ..common.exceptions import ExecutionError, UnsupportedGraphError
from ..config.benchmark_config import TimingPolicy
from ..reporting.suite_results import engine_id_hex
from .timing import Measurement, StallFallbackError, Timer, device_sync, measure


def _get_handle_stream(handle: Any) -> int:
    """Return a hipDNN handle's HIP stream pointer encoded as an integer."""
    get_stream = getattr(handle, "get_stream", None)
    if callable(get_stream):
        return int(get_stream())
    return 0


class Executor:
    """Builds a hipDNN graph for one engine and times it with ``measure``."""

    def __init__(self, graph_json_str: str, policy: TimingPolicy) -> None:
        """Initialize executor with graph JSON and timing policy.

        Args:
            graph_json_str: The graph as a JSON string.
            policy: How ``benchmark`` warms up and samples.
        """
        self._graph_json_str = graph_json_str
        self._policy = policy
        self._execution_stream: Optional[int] = None
        self._graph: Any = None
        self._workspace: Any = None
        self._workspace_ptr: int = 0
        self._workspace_size: int = 0
        self._build_time_ms: float = 0.0

    def _build_through_operation_graph(self, handle: Any) -> Any:
        """Create the hipdnn graph and run it up to ``build_operation_graph``.

        Shared by ``discover_engines`` and ``prepare``. Configures only the
        data-type attributes that are explicitly present in the graph JSON
        (other types are left for hipDNN inference). Engine selection happens
        afterwards: ``prepare`` hard-selects a forced engine once the operation
        graph is built, and ``discover_engines`` queries the ranked list.

        Args:
            handle: hipdnn.Handle instance.

        Returns:
            The hipdnn module (so callers can keep using its enums/types
            without re-importing).

        Raises:
            ExecutionError: If hipdnn_frontend is unavailable or any of the
                graph-build steps fail.
        """
        try:
            import hipdnn_frontend as hipdnn
        except ImportError as e:
            raise ExecutionError(
                "hipdnn_frontend not available. Install hipDNN Python bindings."
            ) from e

        self._graph = hipdnn.Graph()

        try:
            graph_dict = json.loads(self._graph_json_str)
        except (json.JSONDecodeError, TypeError):
            graph_dict = {}

        # Configure only the data types the JSON states; unknown names are
        # left for hipDNN inference.
        for key in ("io_data_type", "intermediate_data_type", "compute_data_type"):
            if key in graph_dict:
                data_type = getattr(hipdnn.DataType, str(graph_dict[key]).upper(), None)
                if data_type is not None:
                    getattr(self._graph, f"set_{key}")(data_type)

        # Normalise node compute_data_type: from_json rejects "unset", which
        # hipDNN emits when the caller leaves the field unset. Promote to the
        # graph-level compute type so the serialised form round-trips cleanly.
        if graph_dict.get("nodes"):
            graph_cdt = graph_dict.get("compute_data_type", "float")
            changed = False
            for node in graph_dict["nodes"]:
                if node.get("compute_data_type", "").lower() == "unset":
                    node["compute_data_type"] = graph_cdt
                    changed = True
            if changed:
                self._graph_json_str = json.dumps(graph_dict)

        result = self._graph.from_json(self._graph_json_str)
        if result.is_bad():
            raise ExecutionError(f"Failed to deserialize graph: {result.get_message()}")

        result = self._graph.validate()
        if result.is_bad():
            raise ExecutionError(f"Graph validation failed: {result.get_message()}")

        result = self._graph.build_operation_graph(handle)
        if result.is_bad():
            raise ExecutionError(
                f"Failed to build operation graph: {result.get_message()}"
            )

        return hipdnn

    def discover_engines(self, handle: Any) -> List[int]:
        """Build the operation graph and return ranked engine IDs.

        Runs the same setup as ``prepare`` up to ``build_operation_graph``,
        then queries ``get_ranked_engine_ids``. Does not allocate a workspace
        or set a preferred engine; callers iterate the returned IDs and create
        a fresh Executor per engine for execution.

        Args:
            handle: hipdnn.Handle instance.

        Returns:
            List of int engine IDs ranked by the backend's heuristics.

        Raises:
            ExecutionError: If any graph-build step fails.
        """
        self._build_through_operation_graph(handle)
        try:
            return [int(eid) for eid in self._graph.get_ranked_engine_ids()]
        except RuntimeError as e:
            raise UnsupportedGraphError(str(e)) from e

    def prime(self, handle: Any, engine_id: int) -> None:
        """Build this engine's OOTB plan once, untimed, and discard it.

        The first plan build of an engine in a process also pays one-time
        costs: provider setup and, on a cold page cache, reading the plugin
        and kernel files. Whichever build comes first absorbs them, so call
        this before the timed OOTB ``prepare`` to keep the OOTB and tuned
        build times comparable. The plan is never executed, so no provider
        tuning (MIOpen find, kernel sampling) runs.

        Raises:
            ExecutionError: If graph building fails.
            UnsupportedGraphError: If the engine cannot build this graph.
        """
        hipdnn = self._build_through_operation_graph(handle)
        self._build_plan(hipdnn, engine_id, None)
        self._graph = None

    def _build_plan(
        self, hipdnn: Any, engine_id: Optional[int], knobs: Optional[Dict[str, Any]]
    ) -> None:
        """Create, support-check, and compile the execution plan."""
        if engine_id is not None:
            # Hard engine selection: build the plan for exactly this engine.
            # create_execution_plan_ext reports a bad result if the engine is
            # not valid/applicable, so it can never silently fall back to a
            # different engine the way the soft preferred-engine path could.
            settings = [
                hipdnn.KnobSetting(knob_id, value)
                for knob_id, value in (knobs or {}).items()
            ]
            result = self._graph.create_execution_plan_ext(engine_id, settings)
            if result.is_bad():
                raise UnsupportedGraphError(
                    f"Forced engine {engine_id_hex(engine_id)} not applicable "
                    f"to this graph: {result.get_message()}"
                )
        else:
            result = self._graph.create_execution_plans()
            if result.is_bad():
                raise ExecutionError(
                    f"Failed to create execution plans: {result.get_message()}"
                )

        result = self._graph.check_support()
        if result.is_bad():
            raise UnsupportedGraphError(
                f"Backend support check failed: {result.get_message()}"
            )

        result = self._graph.build_plans()
        if result.is_bad():
            raise ExecutionError(f"Failed to build plans: {result.get_message()}")

    def prepare(
        self,
        handle: Any,
        engine_id: Optional[int] = None,
        knobs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Build the operation graph and one execution plan.

        ``build_time_ms`` times only the plan build: create, support check
        and compile. Graph setup and workspace allocation are not timed.

        Args:
            handle: hipdnn.Handle instance.
            engine_id: Build the plan for exactly this engine. When None, the
                backend heuristic picks the engine.
            knobs: Knob settings for the ``engine_id`` plan, keyed by knob id.
                hipDNN ignores knobs the engine does not expose.

        Raises:
            ValueError: If ``knobs`` is given without ``engine_id``.
            ExecutionError: If graph building fails.
        """
        if knobs and engine_id is None:
            raise ValueError("knobs require an explicit engine_id")
        self._execution_stream = _get_handle_stream(handle)
        hipdnn = self._build_through_operation_graph(handle)

        with Timer() as build_timer:
            self._build_plan(hipdnn, engine_id, knobs)
        self._build_time_ms = build_timer.elapsed_ms

        self._check_selected_engine(engine_id)

        workspace_size = self._graph.get_workspace_size()
        self._workspace_size = int(workspace_size)
        if workspace_size > 0:
            self._workspace = hipdnn.DeviceBuffer(workspace_size)
            self._workspace_ptr = self._workspace.ptr()

    def engine_knob_ids(self, engine_id: int) -> List[str]:
        """Knob ids the engine exposes for this graph. Call after prepare()."""
        if self._graph is None:
            raise ExecutionError("Graph not prepared. Call prepare() first.")
        return [str(k.knob_id) for k in self._graph.get_knobs_for_engine(engine_id)]

    def _check_selected_engine(self, requested_engine_id: Optional[int]) -> None:
        """Reject a plan backed by an engine other than the forced one.

        ``get_execution_plan_engine_id`` is the authoritative source for the
        engine that will run. A mismatch should be impossible on the
        hard-select path, so it is treated as an unsupported-graph skip
        rather than mislabeled timings.
        """
        if requested_engine_id is None:
            return
        actual = int(self._graph.get_execution_plan_engine_id())
        if actual != requested_engine_id:
            raise UnsupportedGraphError(
                f"Forced engine {engine_id_hex(requested_engine_id)} was not "
                f"selected; the backend ran engine {engine_id_hex(actual)} "
                f"(silent fallback). Skipping to avoid mislabeled results."
            )

    def _get_execution_stream(self, handle: Any) -> int:
        """Return the prepared hipDNN handle stream and reject stream drift."""
        stream = _get_handle_stream(handle)
        if self._execution_stream is None:
            self._execution_stream = stream
        elif stream != self._execution_stream:
            raise ExecutionError(
                "hipDNN handle stream changed after prepare: "
                f"prepared stream {self._execution_stream}, current stream {stream}"
            )
        return stream

    def execute_once(self, handle: Any, variant_pack: Dict[int, int]) -> None:
        """Execute the prepared graph once without collecting timings."""
        if self._graph is None:
            raise ExecutionError("Graph not prepared. Call prepare() first.")
        if self._workspace is not None:
            self._workspace.zeros()

        self._get_execution_stream(handle)
        self.enqueue(handle, variant_pack)
        try:
            device_sync("hip")
        except RuntimeError as e:
            raise ExecutionError(str(e)) from e

    def enqueue(self, handle: Any, variant_pack: Dict[int, int]) -> None:
        """Submit one graph execution: no workspace reset, no device sync."""
        result = self._graph.execute(handle, variant_pack, self._workspace_ptr)
        if result.is_bad():
            raise ExecutionError(f"Graph execution failed: {result.get_message()}")

    def benchmark(self, handle: Any, variant_pack: Dict[int, int]) -> Measurement:
        """Prime and time the prepared graph per the executor's policy.

        Args:
            handle: hipdnn.Handle instance.
            variant_pack: Mapping of tensor UIDs to device pointers.

        Raises:
            ExecutionError: If graph not prepared, execution fails, or HIP
                timing is unavailable.
        """
        if self._graph is None:
            raise ExecutionError("Graph not prepared. Call prepare() first.")
        stream = self._get_execution_stream(handle)
        try:
            return measure(
                lambda: self.enqueue(handle, variant_pack),
                stream=stream,
                policy=self._policy,
                timer="hip",
            )
        except StallFallbackError:
            raise
        except RuntimeError as e:
            raise ExecutionError(str(e)) from e

    @property
    def build_time_ms(self) -> float:
        """Plan create, support check and compile time in milliseconds."""
        return self._build_time_ms

    @property
    def workspace_size(self) -> int:
        """Bytes hipDNN reserved for the operation graph workspace.

        Zero before :meth:`prepare` runs. Surfaced so the suite runner
        can record it as an always-on metric without re-querying the
        graph object.
        """
        return self._workspace_size

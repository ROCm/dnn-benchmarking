#!/usr/bin/env python3
"""Verify hipDNN graphs deserialize/validate without running any kernel.

Levels:
  --level json   : pure-Python GraphLoader.load_json + validate (no hipDNN build)
  --level opgraph: hipdnn_frontend from_json + validate + build_operation_graph,
                   i.e. assemble and finalize the backend operation-graph descriptor
                   (NO plan build, NO kernel execution). Needs a built hipDNN.

Both levels also require every SDPA node to state its scale (an
attn_scale_value or a scale tensor): an absent scale is the backend's default,
not necessarily the scale the workload's source measured.

They also check the causal diagonal of SDPA graphs with Sq != Skv, using the
mask hipDNN actually runs: causal_mask overrides diagonal_alignment and the
written bounds, and setting both causal_mask and causal_mask_bottom_right is
rejected. An effective top-left diagonal with Sq = 1 attends only to key 0,
which no decode step means: that fails. Any other effective top-left causal
graph with Sq != Skv is printed as a warning, since decode and chunked prefill
almost always mean bottom-right.

Usage:
  python tools/check_deserialize.py --level opgraph 'Workloads/**/*.json'
"""
import argparse, glob, json, os, sys
from pathlib import Path


def iter_files(patterns):
    for pat in patterns:
        pat = os.path.expanduser(pat)
        if os.path.isdir(pat):
            yield from glob.glob(os.path.join(pat, "**", "*.json"), recursive=True)
        else:
            yield from glob.glob(pat, recursive=True)


_SDPA_NODE_TYPES = ("SdpaAttributes", "SdpaBackwardAttributes")


def sdpa_nodes_without_scale(graph):
    """Names of SDPA nodes with neither attn_scale_value nor scale_tensor_uid."""
    missing = []
    for node in graph.get("nodes") or []:
        if node.get("type") not in _SDPA_NODE_TYPES:
            continue
        # Forward nodes keep it under "attributes", backward under "parameters".
        attrs = node.get("attributes") or node.get("parameters") or {}
        inputs = node.get("inputs") or {}
        if (
            attrs.get("attn_scale_value") is None
            and inputs.get("scale_tensor_uid") is None
        ):
            missing.append(node.get("name", "<unnamed>"))
    return missing


def sdpa_mask_flag_conflicts(graph):
    """Names of SDPA nodes that set both deprecated causal flags.

    hipDNN's extractDiagonalBandParams rejects the pair outright, so the graph
    cannot run at all.
    """
    conflicts = []
    for node in graph.get("nodes") or []:
        if node.get("type") not in _SDPA_NODE_TYPES:
            continue
        attrs = node.get("attributes") or node.get("parameters") or {}
        if attrs.get("causal_mask") and attrs.get("causal_mask_bottom_right"):
            conflicts.append(node.get("name", "<unnamed>"))
    return conflicts


# hipDNN resolves the mask before any plan sees it (PlanUtils.hpp
# ::extractDiagonalBandParams): causal_mask forces left_bound=-1, right_bound=0
# and top-left, overriding diagonal_alignment and any written bounds;
# causal_mask_bottom_right forces the same band bottom-right. The band itself
# masks the right side only when right_bound >= 0
# (CpuFpReferenceSdpa.hpp::isMasked), so a left_bound on its own is a window
# that stays open on the right and is not causal.
def _sdpa_is_causal(attrs):
    return bool(
        attrs.get("causal_mask")
        or attrs.get("causal_mask_bottom_right")
        or attrs.get("right_bound") == 0
    )


def _sdpa_is_bottom_right(attrs):
    if attrs.get("causal_mask_bottom_right"):
        return True
    # causal_mask wins over the alignment, so causal_mask + BOTTOM_RIGHT is
    # still run top-left.
    return not attrs.get("causal_mask") and (
        attrs.get("diagonal_alignment") == "BOTTOM_RIGHT"
    )


def sdpa_top_left_mismatches(graph):
    """(errors, warnings) for SDPA nodes whose effective causal diagonal is
    top-left while Sq != Skv.

    "Effective" is what hipDNN runs, not what the JSON names: causal_mask
    overrides diagonal_alignment. Sq and Skv come from the Q and K dims. A
    paged K is a block container, so its Skv is unknown and only the Sq = 1
    rule applies.
    """
    dims = {t.get("uid"): t.get("dims") for t in graph.get("tensors") or []}
    errors, warnings = [], []
    for node in graph.get("nodes") or []:
        if node.get("type") not in _SDPA_NODE_TYPES:
            continue
        attrs = node.get("attributes") or node.get("parameters") or {}
        if not _sdpa_is_causal(attrs) or _sdpa_is_bottom_right(attrs):
            continue
        inputs = node.get("inputs") or {}
        q_dims = dims.get(inputs.get("q_tensor_uid"))
        k_dims = dims.get(inputs.get("k_tensor_uid"))
        if not q_dims or not k_dims:
            continue
        sq = q_dims[-2]
        paged = inputs.get("page_table_k_tensor_uid") is not None
        skv = None if paged else k_dims[-2]
        name = node.get("name", "<unnamed>")
        if sq == 1 and skv != 1:
            errors.append(
                f"SDPA node {name!r}: effective top-left causal mask with Sq=1, "
                f"Skv={skv or 'paged'} attends only to key 0; decode is "
                "bottom-right or unmasked"
            )
        elif skv is not None and sq != skv:
            warnings.append(
                f"SDPA node {name!r}: effective top-left causal mask with "
                f"Sq={sq}, Skv={skv}; "
                "decode and chunked prefill are usually bottom-right"
            )
    return errors, warnings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--level", choices=("json", "opgraph"), default="opgraph")
    ap.add_argument(
        "--src", help="dnn-benchmarking src/ dir for --level json", default=None
    )
    ap.add_argument("--show", type=int, default=20, help="max failures to print")
    args = ap.parse_args()

    files = sorted(set(iter_files(args.paths)))
    if not files:
        print("no files matched", file=sys.stderr)
        return 2

    loader = handle = hipdnn = None
    if args.level == "json":
        if args.src:
            sys.path.insert(0, args.src)
        from dnn_benchmarking.graph import GraphLoader

        loader = GraphLoader()
    else:
        try:
            import hipdnn_frontend as hipdnn
        except ImportError:
            print(
                "hipdnn_frontend not importable; build hipDNN (setup_env.py) first.",
                file=sys.stderr,
            )
            return 3
        # This check only exercises from_json/validate/build_operation_graph,
        # which assemble the backend graph descriptor from JSON and never
        # touch an engine. Engine plugins (HIPBLASLT_ENGINE, MIOPEN_ENGINE,
        # ASM_SDPA_ENGINE, ...) are otherwise loaded eagerly on Handle()
        # construction and some initialize a real GPU context on load,
        # which aborts hard on GPU-less machines/CI runners. Must be called
        # before any Handle exists.
        hipdnn.set_engine_plugin_paths([], mode=hipdnn.PluginLoadingMode.ABSOLUTE)
        handle = hipdnn.Handle()

    ok = fail = 0
    warned = []
    failures = []
    for f in files:
        try:
            graph = json.loads(Path(f).read_text())
            unscaled = sdpa_nodes_without_scale(graph)
            if unscaled:
                raise ValueError(
                    f"SDPA node(s) {unscaled} set neither attn_scale_value nor "
                    "scale_tensor_uid; write the scale the workload used"
                )
            conflicting = sdpa_mask_flag_conflicts(graph)
            if conflicting:
                raise ValueError(
                    f"SDPA node(s) {conflicting} set both causal_mask and "
                    "causal_mask_bottom_right; hipDNN rejects that pair. Use "
                    "diagonal_alignment with left_bound=-1, right_bound=0"
                )
            errors, warnings = sdpa_top_left_mismatches(graph)
            if errors:
                raise ValueError("; ".join(errors))
            warned.extend((f, w) for w in warnings)
            if args.level == "json":
                g = loader.load_json(Path(f))
                loader.validate(g)
                loader.extract_tensor_info(g)
            else:
                s = Path(f).read_text()
                g = hipdnn.Graph()
                r = g.from_json(s)
                if r.is_bad():
                    raise RuntimeError(f"from_json: {r.get_message()}")
                r = g.validate()
                if r.is_bad():
                    raise RuntimeError(f"validate: {r.get_message()}")
                r = g.build_operation_graph(handle)
                if r.is_bad():
                    raise RuntimeError(f"build_operation_graph: {r.get_message()}")
            ok += 1
        except Exception as e:  # noqa: BLE001
            fail += 1
            if len(failures) < args.show:
                failures.append((f, str(e)))
    for f, w in warned[: args.show]:
        print(f"WARN {f}\n     {w}")
    for f, e in failures:
        print(f"FAIL {f}\n     {e}")
    print(
        f"\nlevel={args.level}  files={len(files)}  ok={ok}  fail={fail}  "
        f"warn={len(warned)}"
    )
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())

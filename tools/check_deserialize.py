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
                "hipdnn_frontend not importable; build hipDNN (setup.sh) first.",
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
    failures = []
    for f in files:
        try:
            unscaled = sdpa_nodes_without_scale(json.loads(Path(f).read_text()))
            if unscaled:
                raise ValueError(
                    f"SDPA node(s) {unscaled} set neither attn_scale_value nor "
                    "scale_tensor_uid; write the scale the workload used"
                )
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
    for f, e in failures:
        print(f"FAIL {f}\n     {e}")
    print(f"\nlevel={args.level}  files={len(files)}  ok={ok}  fail={fail}")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())

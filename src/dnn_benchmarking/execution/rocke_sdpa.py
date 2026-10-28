# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Select the rocKE kernels that run one SDPA forward graph (``--runtime rocke``).

Two kinds of kernel apply to a graph:

* Catalog kernels: the rocKE kernels that hipDNN ships for the
  ``hipkernel:Gfx950AttentionDense`` engine. Each variant in the installed
  descriptor file (``*.kdp.json`` under the plugin directory) records its
  problem fields and tile in ``metadata``, and where its code object is in
  ``kernel_source``: the kpack archive, the archive key, the symbol, the
  SHA-256 and the argument list. A catalog row reads that code object and
  launches it through rocKE's runtime, so it runs hipDNN's own binary.
* The auto kernel: the kernel that rocKE's own production dispatcher
  (``dispatch.attention.dispatch_attention``) selects for the problem.

This module needs neither torch nor rocKE at import time.
"""

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple, Union

from ..common.exceptions import UnsupportedGraphError
from ..graph.tensor_info import TensorInfo

DENSE_ARCH = "gfx950"

#: hipDNN engine whose descriptors make up the catalog.
DENSE_ENGINE = "hipkernel:Gfx950AttentionDense"

#: Descriptor root of the hip-kernel-provider plugin, relative to a plugin path.
_DESCRIPTOR_SUBDIR = Path("arch_content") / "hip-kernel-provider"

#: Descriptor metadata stored as 0/1 that ``Gfx950AttentionDenseSpec`` takes as bool.
_BOOL_FIELDS = ("causal", "ragged")

_ROCKE_DTYPES = {"half": "fp16", "bfloat16": "bf16"}

#: Arches with a tiled unified-attention builder. rocKE routes any other arch
#: to the gfx950 builder (``kernels.common.attention_unified._tiled_2d_impl``),
#: whose code object does not load there.
_AUTO_ARCHES = ("gfx942", "gfx950", "gfx1250")

_UNSUPPORTED_INPUTS = (
    "attn_mask_tensor_uid",
    "scale_tensor_uid",
    "seq_len_q_tensor_uid",
    "seq_len_kv_tensor_uid",
    "seed_tensor_uid",
    "offset_tensor_uid",
    "dropout_mask_tensor_uid",
    "dropout_scale_tensor_uid",
    "page_table_k_tensor_uid",
    "page_table_v_tensor_uid",
    "block_mask_tensor_uid",
    "sink_token_tensor_uid",
    "descale_q_tensor_uid",
    "descale_k_tensor_uid",
    "descale_v_tensor_uid",
    "descale_s_tensor_uid",
    "scale_s_tensor_uid",
    "scale_o_tensor_uid",
)
_UNSUPPORTED_OUTPUTS = (
    "stats_tensor_uid",
    "max_tensor_uid",
    "sum_exp_tensor_uid",
    "rng_dump_tensor_uid",
    "amax_s_tensor_uid",
    "amax_o_tensor_uid",
)


@dataclass(frozen=True)
class SdpaProblem:
    """One dense SDPA forward problem in rocKE terms.

    Q and O are ``[batch, num_query_heads, seqlen_q, head_size]`` and K and V
    are ``[batch, num_kv_heads, seqlen_kv, head_size]`` in graph order. Their
    memory layout is packed BSHD, which is the layout rocKE kernels read.
    """

    dtype: str
    batch: int
    num_query_heads: int
    num_kv_heads: int
    seqlen_q: int
    seqlen_kv: int
    head_size: int
    causal: bool
    scale: float
    q_uid: int
    k_uid: int
    v_uid: int
    o_uid: int


@dataclass(frozen=True)
class CatalogKernel:
    """One hipDNN-shipped rocKE variant.

    ``spec`` holds the keyword arguments of ``Gfx950AttentionDenseSpec``: the
    variant's metadata, with the graph's batch and sequence lengths once it
    matches a graph. ``library`` is the kpack archive; ``toc_key``,
    ``symbol``, ``sha256`` and ``arg_names`` are the archive key, the kernel
    symbol, the code-object hash and the argument names hipDNN binds.
    """

    name: str
    spec: Dict[str, Any]
    library: Path
    toc_key: str
    symbol: str
    sha256: str
    arg_names: Tuple[str, ...]


@dataclass(frozen=True)
class AutoKernel:
    """The unified-attention kernel that rocKE's dispatcher selects.

    ``path`` is ``2d`` or ``3d``; ``block_size`` is the paged-KV block size.
    """

    name: str
    path: str
    block_size: int


RockeKernel = Union[CatalogKernel, AutoKernel]


Catalog = Tuple[CatalogKernel, ...]


def _is_packed_bshd(dims: Sequence[int], strides: Sequence[int]) -> bool:
    """Whether ``[B, H, S, D]`` (graph order) is packed BSHD in memory.

    The stride of a size-1 dimension never addresses memory, so it is free.
    """
    _, heads, seq, head_size = dims
    packed = (seq * heads * head_size, head_size, heads * head_size, 1)
    return all(n == 1 or s == p for n, s, p in zip(dims, strides, packed))


def sdpa_problem(
    graph_json: Dict[str, Any], tensor_infos: List[TensorInfo]
) -> SdpaProblem:
    """Read the SDPA forward problem from a graph.

    Raises:
        UnsupportedGraphError: The graph is not one dense SDPA forward node
            that rocKE kernels can run.
    """
    from .pytorch_ops._common import _node_param, _optional_uid
    from .pytorch_ops.handlers.sdpa import _sdpa_derive_mask

    nodes = graph_json.get("nodes") or []
    if len(nodes) != 1 or nodes[0].get("type") != "SdpaAttributes":
        raise UnsupportedGraphError(
            "rocKE runtime runs graphs with one SdpaAttributes node only"
        )
    node = nodes[0]

    present = [
        key
        for key in _UNSUPPORTED_INPUTS + _UNSUPPORTED_OUTPUTS
        if _optional_uid(node, key) is not None
    ]
    for flag in ("alibi_mask", "padding_mask"):
        if _node_param(node, flag, False):
            present.append(flag)
    if _node_param(node, "generate_stats", False):
        present.append("generate_stats")
    if float(_node_param(node, "dropout_probability", 0.0) or 0.0) != 0.0:
        present.append("dropout_probability")
    if present:
        raise UnsupportedGraphError(
            f"rocKE runtime does not support SDPA features: {', '.join(present)}"
        )
    # The mask resolves as hipDNN resolves it: the deprecated causal booleans
    # win over the bounds and the alignment.
    try:
        causal, window, bottom_right = _sdpa_derive_mask(node)
    except ValueError as e:
        raise UnsupportedGraphError(str(e)) from e
    if window is not None:
        raise UnsupportedGraphError("rocKE runtime does not support sliding windows")
    causal = causal or bottom_right

    infos = {info.uid: info for info in tensor_infos}
    uids = {}
    for role, key in (
        ("q", "q_tensor_uid"),
        ("k", "k_tensor_uid"),
        ("v", "v_tensor_uid"),
        ("o", "o_tensor_uid"),
    ):
        uid = _optional_uid(node, key)
        if uid is None or uid not in infos:
            raise UnsupportedGraphError(f"SDPA node has no {role} tensor")
        uids[role] = uid
    q, k, v, o = (infos[uids[r]] for r in ("q", "k", "v", "o"))

    dtypes = {t.data_type for t in (q, k, v, o)}
    if len(dtypes) != 1 or next(iter(dtypes)) not in _ROCKE_DTYPES:
        raise UnsupportedGraphError(
            f"rocKE runtime needs one fp16 or bf16 dtype for Q/K/V/O, got {sorted(dtypes)}"
        )
    if any(len(t.dims) != 4 for t in (q, k, v, o)):
        raise UnsupportedGraphError("rocKE runtime needs 4-D Q/K/V/O tensors")
    batch, hq, sq, d = q.dims
    _, hkv, skv, dk = k.dims
    if list(o.dims) != list(q.dims) or list(v.dims) != list(k.dims) or dk != d:
        raise UnsupportedGraphError(
            "rocKE runtime needs O shaped like Q, V shaped like K, and one head size"
        )
    if k.dims[0] != batch or hkv <= 0 or hq % hkv != 0:
        raise UnsupportedGraphError(
            f"rocKE runtime needs matching batch and Hq divisible by Hkv "
            f"(Hq={hq}, Hkv={hkv})"
        )
    for t in (q, k, v, o):
        if not _is_packed_bshd(t.dims, t.strides):
            raise UnsupportedGraphError(
                f"rocKE runtime needs packed BSHD tensors; '{t.name}' has strides "
                f"{list(t.strides)}"
            )
    if causal and sq != skv:
        # Catalog kernels align the diagonal top-left, the unified kernels
        # bottom-right; the two masks are the same only for square problems.
        raise UnsupportedGraphError(
            "rocKE runtime runs causal SDPA only with seqlen_q == seqlen_kv"
        )

    scale = _node_param(node, "attn_scale_value", None)
    return SdpaProblem(
        dtype=_ROCKE_DTYPES[q.data_type],
        batch=batch,
        num_query_heads=hq,
        num_kv_heads=hkv,
        seqlen_q=sq,
        seqlen_kv=skv,
        head_size=d,
        causal=causal,
        # hipDNN reads an absent attn_scale as 1.0 (cuDNN semantics).
        scale=1.0 if scale is None else float(scale),
        q_uid=uids["q"],
        k_uid=uids["k"],
        v_uid=uids["v"],
        o_uid=uids["o"],
    )


def hipdnn_grid(problem: SdpaProblem, block_m: int) -> Tuple[int, int, int]:
    """The grid hipDNN launches a catalog variant with.

    hip-kernel-provider's ``hipkernel.gfx950_attention_dense.dispatch.v1``
    contract (``Gfx950AttentionDenseGeometry.hpp``): one CTA per query block,
    query head and batch, with query blocks on x.
    """
    return (-(-problem.seqlen_q // block_m), problem.num_query_heads, problem.batch)


def load_catalog(plugin_paths: Iterable[Path], arch: str) -> Catalog:
    """Read every shipped variant of the ``DENSE_ENGINE`` engine for ``arch``.

    An engine descriptor (``*.ued.json``) holds the engine name; a variant
    set (``*.kdp.json``) names its engine by id. The packer keeps each
    descriptor's authored subpath (for example
    ``gfx950_attention_dense/artifacts/``), so search the whole tree. A
    ``kernel_source.library`` path is relative to its descriptor file. When
    several plugin paths ship a variant of one name, the first path wins, so
    each variant gives one row.

    Raises:
        OSError, ValueError, KeyError: A descriptor file is unreadable, not
            JSON, or lacks a field of a packed variant.
    """
    import json

    def read(path: Path) -> Dict[str, Any]:
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    kernels: Dict[str, CatalogKernel] = {}
    for root in plugin_paths:
        base = Path(root) / _DESCRIPTOR_SUBDIR / arch / "rocKE"
        engines = [read(p) for p in sorted(base.rglob("*.ued.json"))]
        ids = {str(e.get("id")) for e in engines if e.get("name") == DENSE_ENGINE}
        for path in sorted(base.rglob("*.kdp.json")):
            descriptor = read(path)
            if str(descriptor.get("engine")) not in ids:
                continue
            for ukd in descriptor.get("kernelDescriptors", []):
                source = ukd.get("kernel_source") or {}
                if source.get("kind") != "kpack":
                    continue
                spec = dict(ukd["metadata"])
                spec["dtype"] = str(spec["dtype"]).lower()
                for field in _BOOL_FIELDS:
                    if field in spec:
                        spec[field] = bool(spec[field])
                name = f"rocke:{ukd['name']}"
                kernels.setdefault(
                    name,
                    CatalogKernel(
                        name=name,
                        spec=spec,
                        library=(path.parent / source["library"]).resolve(),
                        toc_key=source["toc_key"],
                        symbol=source["symbol"],
                        sha256=source["sha256"],
                        arg_names=tuple(a.get("name") for a in source["signature"]),
                    ),
                )
    return tuple(kernels.values())


def catalog_kernels(catalog: Catalog, problem: SdpaProblem) -> List[CatalogKernel]:
    """Catalog variants that run ``problem``, in descriptor order.

    A variant applies when its problem fields equal the graph's, it has no
    sliding window and no ragged layout (the graph never has them), and rocKE
    accepts the spec with the graph's batch and sequence lengths. The shipped
    kernels read those three values as kernel arguments, so the shape does not
    change the code object.
    """
    from kernels.gfx950.attention_dense import (
        Gfx950AttentionDenseSpec,
        supports_attention_dense,
    )

    found = []
    for kernel in catalog:
        spec = kernel.spec
        if (
            spec.get("dtype") != problem.dtype
            or spec.get("head_size") != problem.head_size
            or spec.get("num_query_heads") != problem.num_query_heads
            or spec.get("num_kv_heads") != problem.num_kv_heads
            or bool(spec.get("causal")) != problem.causal
            or spec.get("sliding_window")
            or spec.get("ragged")
        ):
            continue
        shaped = dict(
            spec,
            batch=problem.batch,
            seqlen_q=problem.seqlen_q,
            seqlen_kv=problem.seqlen_kv,
            # hipDNN launches every variant on the qb_minor grid (its
            # dispatch.v1 contract), and the descriptor does not record the
            # block order. rocKE's "auto" picks another order for causal
            # problems, whose grid this code object does not decode.
            nonpersist_decode="qb_minor",
        )
        # rocKE reports a tile that does not divide the sequence lengths (and
        # other unsupported spec values) as ValueError. A TypeError means the
        # descriptor names a spec field this rocKE does not know: a rocKE
        # version mismatch, which must reach the graph error, not hide here.
        try:
            ok, _ = supports_attention_dense(
                Gfx950AttentionDenseSpec(**shaped), arch=DENSE_ARCH
            )
        except ValueError:
            ok = False
        if ok:
            found.append(replace(kernel, spec=shaped))
    return found


def auto_kernel(problem: SdpaProblem, arch: str, num_cus: int) -> AutoKernel:
    """The kernel rocKE's production dispatcher selects for ``problem``.

    Raises:
        UnsupportedGraphError: rocKE selects no kernel this runtime can launch.
    """
    if arch not in _AUTO_ARCHES:
        raise UnsupportedGraphError(
            f"rocKE has no tiled attention kernel for {arch} "
            f"(supported: {', '.join(_AUTO_ARCHES)})"
        )
    if not problem.causal:
        raise UnsupportedGraphError(
            "rocKE auto dispatch selects unified attention, which is causal-only"
        )
    from dispatch.attention import AttentionRequest, dispatch_attention

    request = AttentionRequest(
        batch=problem.batch,
        nhead_q=problem.num_query_heads,
        nhead_k=problem.num_kv_heads,
        seqlen_q=problem.seqlen_q,
        seqlen_k=problem.seqlen_kv,
        hdim_q=problem.head_size,
        hdim_v=problem.head_size,
        arch=arch,
        mask_type=1,
        dtype=problem.dtype,
        num_cus=num_cus,
    )
    try:
        result = dispatch_attention(request)
    except ValueError as e:
        raise UnsupportedGraphError(f"rocKE auto dispatch: {e}") from e
    spec = result.spec
    path = getattr(spec, "path", "")
    if path not in ("2d", "3d"):
        raise UnsupportedGraphError(
            f"rocKE auto dispatch selected {result.candidate.name}, which this "
            "runtime cannot launch"
        )
    if problem.seqlen_kv % spec.block_size:
        raise UnsupportedGraphError(
            f"seqlen_kv {problem.seqlen_kv} is not a multiple of the paged-KV "
            f"block size {spec.block_size}"
        )
    return AutoKernel(f"rocke:auto:{spec.kernel_name()}", path, spec.block_size)

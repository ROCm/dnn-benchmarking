# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Select the AITER kernels that run one SDPA forward graph (``--runtime aiter``).

Each graph gets one row for AITER's own dispatch (``aiter.flash_attn_func``)
and one row for each AITER forward backend that can run the graph:

========================  ======================================================
Row                       AITER entry point
========================  ======================================================
``aiter:auto``            ``aiter.flash_attn_func`` (AITER selects the backend)
``aiter:ck``              ``aiter.mha_fwd`` (CK tile)
``aiter:asm_v3``          ``aiter.fmha_v3_fwd`` (hand-written assembly)
``aiter:opus``            ``aiter.fmha_fwd_bf16_opus_fwd`` (OPUS, gfx950)
``aiter:native_splitkv``  ``aiter.mha_fwd_native_splitkv`` (HIP split-KV, gfx942)
``aiter:triton``          ``aiter.ops.triton.attention.mha.flash_attn_func``
``aiter:gluon``           the same function with ``backend="gluon"`` (gfx950)
========================  ======================================================

AITER decides between its compiled backends inside ``_flash_attn_forward``
with nested ``can_impl_*`` checks that callers cannot reach. :func:`backend_rows`
repeats those checks for the graphs this runtime accepts (dense, no bias, no
dropout, no window, no sink, no FP8). They follow the AITER commit pinned by
the ``aiter`` submodule; a submodule bump must check them again.

This module needs neither torch nor AITER at import time.
"""

from dataclasses import dataclass
from typing import Any, Dict, List

from ..common.exceptions import UnsupportedGraphError
from ..graph.tensor_info import TensorInfo
from .sdpa_graph import DenseSdpa, dense_sdpa

#: Arches where AITER's CK forward kernels, and so ``aiter:auto``, run.
#: AITER's other backends are subsets of these.
AITER_ARCHES = ("gfx90a", "gfx942", "gfx950")

#: KV byte extent at which the 32-bit buffer offsets of some kernels wrap
#: (AITER ``_fmha_kv_byte_extent_ge_u32``).
_KV_EXTENT_LIMIT = 1 << 32

#: Largest head size that AITER's CK forward instances cover.
_CK_MAX_HEAD_SIZE = 256


@dataclass(frozen=True)
class AiterBackend:
    """One AITER row.

    ``name`` is the backend (``auto``, ``ck``, ...); the row is
    ``aiter:<name>``. ``num_splits`` is the split-KV group count of
    ``native_splitkv`` and 0 for every other backend.
    """

    name: str
    num_splits: int = 0

    @property
    def row_name(self) -> str:
        """The row and engine name, stable across runs."""
        return f"aiter:{self.name}"


@dataclass(frozen=True)
class AiterDevice:
    """The device facts that AITER's backend checks read.

    Attributes:
        arch: GPU architecture name (``gfx950``).
        num_cus: Compute-unit count (AITER ``get_cu_num``).
        gluon: AITER's Gluon MHA forward kernel can run here
            (``aiter.ops.triton.attention.mha.is_gluon_available``).
    """

    arch: str
    num_cus: int
    gluon: bool


def aiter_problem(
    graph_json: Dict[str, Any], tensor_infos: List[TensorInfo]
) -> DenseSdpa:
    """Read the SDPA forward problem that AITER kernels can run.

    AITER aligns a causal mask to the bottom right, as flash-attention does.
    A top-left causal mask is the same mask only when ``seqlen_q ==
    seqlen_kv``.

    Raises:
        UnsupportedGraphError: AITER kernels cannot run the graph.
    """
    problem = dense_sdpa(graph_json, tensor_infos, "AITER")
    if problem.causal and problem.seqlen_q != problem.seqlen_kv:
        raise UnsupportedGraphError(
            "AITER aligns causal masks to the bottom right; a top-left causal "
            "mask needs seqlen_q == seqlen_kv"
        )
    return problem


def is_causal(problem: DenseSdpa) -> bool:
    """The ``causal`` argument of every AITER entry point."""
    return problem.causal or problem.bottom_right


def _kv_extent_fits(problem: DenseSdpa, head_size: int) -> bool:
    """Whether one KV head row of ``head_size`` stays under 2**32 bytes.

    In packed BSHD the sequence stride is ``num_kv_heads * head_size`` elements
    of 2 bytes.
    """
    stride = problem.num_kv_heads * head_size
    return problem.seqlen_kv * stride * 2 < _KV_EXTENT_LIMIT


def _native_splits(problem: DenseSdpa, num_cus: int) -> int:
    """AITER ``_native_splitkv_heuristic``: the split-KV group count, or 0."""
    sq_tile, kv_tile = 128, 64

    def snap(x: float) -> int:
        g = 0
        for c in (2, 4, 8, 16):
            if c <= x:
                g = c
        return g

    sqt = -(-problem.seqlen_q // sq_tile)
    skvt = -(-problem.seqlen_kv // kv_tile)
    nwg = problem.batch * problem.num_query_heads * sqt
    kv_cap = snap(skvt / (10 if nwg < 24 else 28))
    if nwg < num_cus:
        return min(snap(3.5 * num_cus / nwg), kv_cap)
    if problem.batch >= 2:
        return 0
    over = nwg / num_cus
    if skvt < 10 * over and over < 30:
        return 0
    g = 4 if problem.num_query_heads <= 8 else 2
    if over >= 30:
        g = max(g, snap(skvt / 160))
    return min(g, kv_cap) if kv_cap > 0 else 0


def _triton_head_sizes_ok(problem: DenseSdpa) -> bool:
    """AITER's Triton kernels take a QK head size above V's only as a PE split.

    The extra QK part (``dqk - dv``) and V's head size must then both be
    unpadded powers of two of at least 16.
    """
    dqk, dv = problem.head_size_qk, problem.head_size_v
    if dqk == dv:
        return True
    pe = dqk - dv

    def pow2(n: int) -> bool:
        return n >= 16 and n & (n - 1) == 0

    return pe > 0 and pow2(pe) and pow2(dv)


def _ck_runs(p: DenseSdpa, device: AiterDevice) -> bool:
    # aiter.flash_attn_func pads other head sizes to a multiple of 8 before it
    # calls mha_fwd; this row launches mha_fwd itself.
    heads = (p.head_size_qk, p.head_size_v)
    return all(h % 8 == 0 for h in heads) and max(heads) <= _CK_MAX_HEAD_SIZE


def _asm_v3_runs(p: DenseSdpa, device: AiterDevice) -> bool:
    # AITER can_impl_fmha_v3_fwd, without the features this runtime rejects.
    heads = (p.head_size_qk, p.head_size_v)
    if heads == (192, 128):
        fits = _kv_extent_fits(p, 192) and _kv_extent_fits(p, 128)
    else:
        fits = heads == (128, 128)
    return device.arch in ("gfx942", "gfx950") and p.data_type == "bfloat16" and fits


def _opus_runs(p: DenseSdpa, device: AiterDevice) -> bool:
    # AITER can_impl_fmha_fwd_bf16_opus. aiter:auto runs the 128/128 kernel
    # only with AITER_ENABLE_FMHA_OPUS=1; this row runs it directly.
    heads = (p.head_size_qk, p.head_size_v)
    if heads == (128, 128):
        fits = _kv_extent_fits(p, 128)
    else:
        fits = heads == (192, 128)
    return device.arch == "gfx950" and p.data_type == "bfloat16" and fits


def _native_runs(p: DenseSdpa, device: AiterDevice) -> bool:
    # AITER can_impl_fmha_native, and the split it then needs: AITER runs the
    # kernel only when its heuristic splits the KV.
    return (
        device.arch == "gfx942"
        and p.data_type == "bfloat16"
        and (p.head_size_qk, p.head_size_v) == (64, 64)
        and not (is_causal(p) and p.seqlen_kv < p.seqlen_q)
        and _native_splits(p, device.num_cus) > 1
    )


def _triton_runs(p: DenseSdpa, device: AiterDevice) -> bool:
    # AITER ships Triton MHA configs (ops/triton/configs/<arch>/...) for these
    # arches only; others fail with "Required config file doesn't exist".
    return device.arch in ("gfx942", "gfx950") and _triton_head_sizes_ok(p)


def _gluon_runs(p: DenseSdpa, device: AiterDevice) -> bool:
    return device.gluon and _triton_head_sizes_ok(p)


#: Each backend row's name and whether it runs a problem, in row order.
_BACKENDS = (
    ("ck", _ck_runs),
    ("asm_v3", _asm_v3_runs),
    ("opus", _opus_runs),
    ("native_splitkv", _native_runs),
    ("triton", _triton_runs),
    ("gluon", _gluon_runs),
)


def backend_rows(problem: DenseSdpa, device: AiterDevice) -> List[AiterBackend]:
    """The AITER rows that run ``problem``: ``aiter:auto``, then each backend.

    Raises:
        UnsupportedGraphError: The device is not one of :data:`AITER_ARCHES`.
    """
    if device.arch not in AITER_ARCHES:
        raise UnsupportedGraphError(
            f"AITER runtime runs on {', '.join(AITER_ARCHES)}, not {device.arch}"
        )
    rows = [AiterBackend("auto")]
    for name, runs in _BACKENDS:
        if runs(problem, device):
            splits = _native_splits(problem, device.num_cus)
            rows.append(AiterBackend(name, splits if name == "native_splitkv" else 0))
    return rows

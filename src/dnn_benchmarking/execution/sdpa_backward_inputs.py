# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Forward-consistent O and stats inputs for SDPA backward graphs.

An SDPA backward node reads the forward output O and the forward
log-sum-exp (``stats``) as inputs. Independent random values for them do not
describe any forward pass: ``P = exp(S - stats)`` is then not a softmax, the
gradients grow to 1e4-1e5, fp16 gradients overflow to infinity, and no
tolerance can tell a correct kernel from a wrong one. On gfx942 the kernel
times were the same for random and for consistent inputs (ALMIOPEN-2976).

This module computes O and stats with an exact float32 forward over the
generated Q, K and V, as a training step would produce them. The forward
costs as much as the attention itself, so it runs in PyTorch, on the GPU
when one is visible. torch is imported only for a graph with an SDPA
backward node. Without torch, a NumPy forward (single-threaded, much
slower) gives the same values to float32 rounding.

With U[0, 1) inputs and many keys per query (decode shapes), every dV and
dK element is below 1e-3, so the fixed ``atol`` accepts an engine that
never writes them. Every gradient is linear in dO, so the module scales dO
by a power of two until the largest dV element is near 1, as AMP loss
scaling does (:func:`_d_out_scale`).
"""

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..common import torch_support
from ..graph.tensor_info import TensorInfo
from ..metrics.analytical._common import node_param
from ..metrics.analytical.sdpa import Band, alibi_slopes, sdpa_band

#: Float32 elements of one score block (256 MiB).
_SCORE_BLOCK_ELEMENTS = 1 << 26
#: Largest dO scale. dO stays below 2^14, inside the fp16 range (65504).
_MAX_D_OUT_SCALE = 1 << 14


def _uid(node: Dict[str, Any], key: str) -> Optional[int]:
    value = (node.get("inputs") or {}).get(key)
    return None if value is None else int(value)


def _scale(node: Dict[str, Any], input_data: Dict[int, np.ndarray]) -> float:
    """hipDNN reads an absent scale as 1.0, not 1/sqrt(d)."""
    uid = _uid(node, "scale_tensor_uid")
    if uid is not None and uid in input_data:
        return float(np.asarray(input_data[uid], dtype=np.float64).reshape(-1)[0])
    value = node_param(node, "attn_scale_value")
    return 1.0 if value is None else float(value)


def _torch() -> Any:
    """The torch module, or None when it cannot be imported.

    A broken install often fails with OSError or RuntimeError, not
    ImportError; the NumPy forward then gives the same values.
    """
    if not torch_support.module_available():
        return None
    import torch

    return torch


def attention_forward(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    d_out: np.ndarray,
    scale: float,
    mask: Optional[np.ndarray] = None,
    band: Band = (None, None, False),
    alibi: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Exact float32 attention: ``(O [B, Hq, Sq, Dv], stats [B, Hq, Sq, 1],
    max |dV|)``, where ``dV = P^T dO`` summed over the query heads of each
    K/V head.

    K and V can have fewer heads than Q (grouped-query attention). ``mask``
    is an additive bias that broadcasts to ``[B, Hq, Sq, Skv]``. ``alibi``
    holds per-query-head slopes (:func:`alibi_slopes`); head ``h`` adds
    ``alibi[h] * (key - query)``. A row with no visible key gets O = 0 and
    stats = 0, so a backward reads P = 0 for it.
    """
    torch = _torch()
    if torch is None:
        return _numpy_forward(q, k, v, d_out, scale, mask, band, alibi)
    return _torch_forward(torch, q, k, v, d_out, scale, mask, band, alibi)


def _torch_forward(
    torch: Any,
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    d_out: np.ndarray,
    scale: float,
    mask: Optional[np.ndarray],
    band: Band,
    alibi: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, float]:
    """``attention_forward`` in PyTorch, all heads of one batch at a time."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch, heads, q_len, _ = q.shape
    kv_len = k.shape[2]
    rep_k, rep_v = heads // k.shape[1], heads // v.shape[1]
    out = np.empty((batch, heads, q_len, v.shape[3]), dtype=np.float32)
    stats = np.empty((batch, heads, q_len, 1), dtype=np.float32)
    dv_max = 0.0
    left, right, bottom_right = band
    offset = kv_len - q_len if bottom_right else 0
    block = (
        (_SCORE_BLOCK_ELEMENTS << 2) if device.type == "cuda" else _SCORE_BLOCK_ELEMENTS
    )
    rows = max(1, block // max(heads * kv_len, 1))
    full_mask = (
        None if mask is None else np.broadcast_to(mask, (batch, heads, q_len, kv_len))
    )
    slopes = (
        None
        if alibi is None
        else torch.from_numpy(alibi.astype(np.float32)).to(device)[:, None, None]
    )

    def on_device(array: np.ndarray, rep: int = 1) -> Any:
        tensor = torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32))
        tensor = tensor.to(device)
        return tensor.repeat_interleave(rep, dim=0) if rep > 1 else tensor

    with torch.no_grad():
        keys = torch.arange(kv_len, device=device)
        key_f = keys.to(torch.float32)
        for b in range(batch):
            q_b = on_device(q[b])
            k_t = on_device(k[b], rep_k).transpose(-2, -1)
            v_b = on_device(v[b], rep_v)
            d_out_b = on_device(d_out[b])
            dv_b = torch.zeros(
                (heads, kv_len, v.shape[3]), dtype=torch.float32, device=device
            )
            for start in range(0, q_len, rows):
                end = min(start + rows, q_len)
                scores = torch.matmul(q_b[:, start:end], k_t).mul_(scale)
                if full_mask is not None:
                    scores += on_device(full_mask[b, :, start:end])
                diagonal = torch.arange(start, end, device=device)[:, None] + offset
                if slopes is not None:
                    rows_f = torch.arange(start, end, device=device)[:, None]
                    scores += slopes * (key_f - rows_f.to(torch.float32))
                if left is not None:
                    scores.masked_fill_(keys <= diagonal - left, float("-inf"))
                if right is not None:
                    scores.masked_fill_(keys > diagonal + right, float("-inf"))
                lse = torch.logsumexp(scores, dim=-1, keepdim=True)
                lse = torch.where(torch.isfinite(lse), lse, torch.zeros_like(lse))
                probs = scores.sub_(lse).exp_()
                out[b, :, start:end] = torch.matmul(probs, v_b).cpu().numpy()
                stats[b, :, start:end] = lse.cpu().numpy()
                dv_b += torch.matmul(probs.transpose(-2, -1), d_out_b[:, start:end])
                del scores, probs
            dv_b = dv_b.view(v.shape[1], rep_v, kv_len, v.shape[3]).sum(dim=1)
            dv_max = max(dv_max, float(dv_b.abs().max()) if dv_b.numel() else 0.0)
            del dv_b
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out, stats, dv_max


def _numpy_forward(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    d_out: np.ndarray,
    scale: float,
    mask: Optional[np.ndarray],
    band: Band,
    alibi: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, float]:
    """``attention_forward`` in NumPy, one head at a time."""
    batch, heads, q_len, _ = q.shape
    kv_len = k.shape[2]
    rep_k, rep_v = heads // k.shape[1], heads // v.shape[1]
    out = np.empty((batch, heads, q_len, v.shape[3]), dtype=np.float32)
    stats = np.empty((batch, heads, q_len, 1), dtype=np.float32)
    dv_max = 0.0
    left, right, bottom_right = band
    offset = kv_len - q_len if bottom_right else 0
    keys = np.arange(kv_len)
    rows_per_block = max(1, _SCORE_BLOCK_ELEMENTS // max(kv_len, 1))
    full_mask = (
        None if mask is None else np.broadcast_to(mask, (batch, heads, q_len, kv_len))
    )

    for b in range(batch):
        dv_b = np.zeros((v.shape[1], kv_len, v.shape[3]), dtype=np.float32)
        for h in range(heads):
            q_bh = np.asarray(q[b, h], dtype=np.float32)
            k_t = np.asarray(k[b, h // rep_k], dtype=np.float32).T
            v_bh = np.asarray(v[b, h // rep_v], dtype=np.float32)
            d_out_bh = np.asarray(d_out[b, h], dtype=np.float32)
            for start in range(0, q_len, rows_per_block):
                end = min(start + rows_per_block, q_len)
                scores = q_bh[start:end] @ k_t
                scores *= np.float32(scale)
                if full_mask is not None:
                    scores += full_mask[b, h, start:end]
                if alibi is not None:
                    query = np.arange(start, end, dtype=np.float32)[:, None]
                    scores += np.float32(alibi[h]) * (keys[None, :] - query)
                diagonal = np.arange(start, end)[:, None] + offset
                if left is not None:
                    scores[keys[None, :] <= diagonal - left] = -np.inf
                if right is not None:
                    scores[keys[None, :] > diagonal + right] = -np.inf
                peak = scores.max(axis=-1, keepdims=True)
                visible = np.isfinite(peak)
                peak = np.where(visible, peak, np.float32(0))
                np.subtract(scores, peak, out=scores)
                np.exp(scores, out=scores)
                total = scores.sum(axis=-1, keepdims=True)
                safe = np.where(visible, total, np.float32(1))
                scores /= safe
                out[b, h, start:end] = scores @ v_bh
                stats[b, h, start:end] = np.where(
                    visible, peak + np.log(safe), np.float32(0)
                )
                dv_b[h // rep_v] += scores.T @ d_out_bh[start:end]
        dv_max = max(dv_max, float(np.abs(dv_b).max()) if dv_b.size else 0.0)
    return out, stats, dv_max


def _d_out_scale(dv_max: float) -> int:
    """Power of two that brings the largest dV element to [0.5, 1).

    Only small gradients are scaled up; a power of two keeps dO exact at
    its graph type. The scale is at most ``_MAX_D_OUT_SCALE``.
    """
    if not 0.0 < dv_max < 0.5:
        return 1
    return min(1 << int(math.floor(-math.log2(dv_max))), _MAX_D_OUT_SCALE)


def set_sdpa_backward_inputs(
    graph_json: Optional[Dict[str, Any]],
    tensor_infos: List[TensorInfo],
    input_data: Dict[int, np.ndarray],
    round_to: Any,
) -> None:
    """Replace the O and stats inputs of each SDPA backward node in place,
    and scale its dO input by :func:`_d_out_scale`.

    ``round_to(values, tensor_info)`` rounds float32 values to the tensor's
    logical type. Nodes whose O, stats or dO another node computes are
    skipped. The forward applies the band, the attention bias and ALiBi.
    Dropout and sequence-length inputs are not applied: the forward is dense
    and dropout-free, so with dropout O differs from the dropped-out forward.
    """
    if not graph_json:
        return
    infos = {info.uid: info for info in tensor_infos}
    for node in graph_json.get("nodes", []) or []:
        if node.get("type") != "SdpaBackwardAttributes":
            continue
        uids = {
            name: _uid(node, f"{name}_tensor_uid")
            for name in ("q", "k", "v", "o", "do", "stats", "attn_mask")
        }
        if any(
            uids[name] not in input_data for name in ("q", "k", "v", "o", "do", "stats")
        ):
            continue
        try:
            band = sdpa_band(node)
        except ValueError:
            continue  # hipDNN rejects the mask, so no engine reads O or stats
        mask_uid = uids["attn_mask"]
        q, v = input_data[uids["q"]], input_data[uids["v"]]
        out_shape = (*q.shape[:3], v.shape[3])
        d_out = input_data[uids["do"]]
        if d_out.size != math.prod(out_shape):
            raise ValueError(
                f"SDPA backward dO tensor {uids['do']} has dims {list(d_out.shape)}; "
                f"the forward gives {list(out_shape)}"
            )
        out, stats, dv_max = attention_forward(
            q,
            input_data[uids["k"]],
            v,
            d_out.reshape(out_shape),
            _scale(node, input_data),
            None if mask_uid is None else input_data.get(mask_uid),
            band,
            alibi_slopes(q.shape[1]) if node_param(node, "alibi_mask") else None,
        )
        for name, values in (("o", out), ("stats", stats)):
            info = infos[uids[name]]
            if math.prod(info.dims) != values.size:
                raise ValueError(
                    f"SDPA backward {name} tensor {info.uid} has dims {info.dims}; "
                    f"the forward gives {list(values.shape)}"
                )
            input_data[info.uid] = round_to(values.reshape(info.dims), info)
        d_out_scale = _d_out_scale(dv_max)
        if d_out_scale > 1:
            info = infos[uids["do"]]
            input_data[info.uid] = round_to(
                d_out.astype(np.float32) * np.float32(d_out_scale), info
            )

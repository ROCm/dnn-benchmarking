# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""rocKE runtime: paged-KV mapping, and the checks before a variant launches."""

import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from dnn_benchmarking.common.exceptions import ExecutionError
from dnn_benchmarking.execution.rocke_sdpa import (
    CatalogKernel,
    SdpaProblem,
    sdpa_problem,
)
from dnn_benchmarking.graph.loader import GraphLoader
from tests.rocke_graphs import sdpa_graph

torch = pytest.importorskip("torch")

from dnn_benchmarking.execution import rocke_executor  # noqa: E402
from dnn_benchmarking.execution.rocke_executor import (  # noqa: E402
    load_variant,
    paged_views,
)


def _graph_tensor(info, generator):
    """A graph tensor with the graph's dims and packed BSHD strides."""
    b, h, s, d = info.dims
    return torch.randn(b, s, h, d, generator=generator).permute(0, 2, 1, 3)


def _paged_attention(p, views, causal):
    """Attention that reads K/V only through ``block_table`` and the caches."""
    group = p.num_query_heads // p.num_kv_heads
    for b in range(p.batch):
        start, end = views.cu_seqlens_q[b].item(), views.cu_seqlens_q[b + 1].item()
        pages = views.block_table[b].long()
        used = views.seqused_k[b].item()
        k = views.k_cache[pages].reshape(-1, p.num_kv_heads, p.head_size)[:used]
        v = views.v_cache[pages].reshape(-1, p.num_kv_heads, p.head_size)[:used]
        q = views.q_tokens[start:end].transpose(0, 1)  # [Hq, Sq, D]
        out = torch.nn.functional.scaled_dot_product_attention(
            q,
            k.transpose(0, 1).repeat_interleave(group, dim=0),
            v.transpose(0, 1).repeat_interleave(group, dim=0),
            is_causal=causal,
            scale=p.scale,
        )
        views.o_tokens[start:end] = out.transpose(0, 1)


@pytest.mark.parametrize("causal", [True, False])
def test_paged_views_alias_graph_tensors_in_page_order(causal) -> None:
    graph = sdpa_graph(b=3, hq=4, hkv=2, s=8, d=4, causal_mask=causal)
    infos = GraphLoader().extract_tensor_info(graph)
    p = sdpa_problem(graph, infos)
    generator = torch.Generator().manual_seed(0)
    q, k, v, o = (_graph_tensor(info, generator) for info in infos)
    o.zero_()

    bshd = [t.permute(0, 2, 1, 3) for t in (q, k, v, o)]  # as prepare() binds them
    views = paged_views(p, 4, *bshd)

    # Zero copy: the kernel reads and writes the graph tensors themselves.
    assert views.k_cache.data_ptr() == k.data_ptr()
    assert views.o_tokens.data_ptr() == o.data_ptr()

    _paged_attention(p, views, causal)

    expected = torch.nn.functional.scaled_dot_product_attention(
        q,
        k.repeat_interleave(2, dim=1),
        v.repeat_interleave(2, dim=1),
        is_causal=causal,
        scale=p.scale,
    )
    torch.testing.assert_close(o, expected)


_ARGS = ("q_ptr", "k_ptr", "v_ptr", "o_ptr", "scale", "batch", "seqlen_q", "seqlen_kv")
_HSACO = b"code object"
#: s=512, block_m=256, hq=32, b=1: hipDNN's grid is (2, 32, 1).
_PROBLEM = SdpaProblem("fp16", 1, 32, 8, 512, 512, 128, True, 1.0, 1, 2, 3, 4)


def _kernel(toc_key="toc", sha256=hashlib.sha256(_HSACO).hexdigest(), args=_ARGS):
    return CatalogKernel(
        name="rocke:v",
        spec={"block_m": 256},
        library=Path("/plugins/kpack/gfx950.kpack"),
        toc_key=toc_key,
        symbol="sym",
        sha256=sha256,
        arg_names=args,
    )


@pytest.fixture
def rocke(monkeypatch):
    """rocKE stand-in: its signature, grid and launcher, and the kpack reads."""
    fake = SimpleNamespace(grid=(2, 32, 1), reads=[], launchers=[])
    monkeypatch.setitem(
        sys.modules,
        "kernels.gfx950.attention_dense",
        SimpleNamespace(
            Gfx950AttentionDenseSpec=lambda **spec: spec,
            attention_dense_signature=lambda spec: [{"name": n} for n in _ARGS],
            attention_dense_grid=lambda spec: fake.grid,
            attention_dense_block=lambda spec: (512, 1, 1),
        ),
    )

    def launcher(**kwargs):
        fake.launchers.append(kwargs)
        return kwargs

    monkeypatch.setitem(
        sys.modules, "rocke.runtime", SimpleNamespace(KernelLauncher=launcher)
    )

    def read_kernel(library, toc_key, arch):
        fake.reads.append((library, toc_key, arch))
        return _HSACO

    monkeypatch.setattr(rocke_executor, "read_kernel", read_kernel)
    return fake


def test_a_variant_loads_hipdnns_code_object_once(rocke) -> None:
    loaded = {}
    first = load_variant(_kernel(), _PROBLEM, loaded)
    again = load_variant(_kernel(), _PROBLEM, loaded)
    load_variant(_kernel(toc_key="other"), _PROBLEM, loaded)

    assert (first.grid, first.block) == ((2, 32, 1), (512, 1, 1))
    assert first.launcher["hsaco"] == _HSACO
    assert first.launcher["kernel_name"] == "sym"
    # A second row of the same code object reuses its module and load time.
    assert again.launcher is first.launcher and again.load_ms == first.load_ms
    assert [key for _, key, _ in rocke.reads] == ["toc", "other"]
    assert len(rocke.launchers) == 2


def test_a_code_object_with_another_hash_does_not_load(rocke) -> None:
    loaded = {}
    with pytest.raises(ExecutionError, match="SHA-256"):
        load_variant(_kernel(sha256="ab" * 32), _PROBLEM, loaded)
    assert rocke.launchers == [] and loaded == {}


def test_another_argument_list_fails_before_the_read(rocke) -> None:
    with pytest.raises(ExecutionError, match="argument list"):
        load_variant(_kernel(args=_ARGS[:-1]), _PROBLEM, {})
    assert rocke.reads == []


def test_a_grid_other_than_hipdnns_fails_before_the_read(rocke) -> None:
    """rocKE's auto block order for causal problems swaps the grid axes."""
    rocke.grid = (1, 32, 2)
    with pytest.raises(ExecutionError, match=r"grid \(1, 32, 2\).*\(2, 32, 1\)"):
        load_variant(_kernel(), _PROBLEM, {})
    assert rocke.reads == []

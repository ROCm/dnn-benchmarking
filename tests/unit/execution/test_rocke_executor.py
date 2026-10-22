# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""rocKE runtime: zero-copy mapping of packed BSHD tensors onto paged KV."""

import pytest

from dnn_benchmarking.execution.rocke_sdpa import sdpa_problem
from dnn_benchmarking.graph.loader import GraphLoader
from tests.rocke_graphs import sdpa_graph

torch = pytest.importorskip("torch")

from dnn_benchmarking.execution.rocke_executor import paged_views  # noqa: E402


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

# rocKE SDPA Corpus

This collection contains scaled dot-product attention (SDPA) graphs from the
rocKE SDPA Solera performance set.
The source file SHA-256 value is
`948f7fce7bb16bbf8a91f307c44dc1709e91a2277a6e540af692f3e2488d3c79`.

## Contents

The source contains 172 graphs. Eight graphs duplicate executable content.
`files.tar.gz` contains the remaining 164 unique graphs and combines model tags
from duplicate entries.

All graphs are forward inference operations. The collection covers decode,
prefill, chunked prefill, grouped-query attention, sliding-window attention,
and attention-sink shapes.

Twelve graphs have an attention sink: `rocke_sdpa_158` to `rocke_sdpa_165`,
`rocke_sdpa_167` to `rocke_sdpa_169`, and `rocke_sdpa_171`. The source gave
each sink as `[H_q]` in the I/O data type. The sinks are normalized to dims
`[1, H_q, 1, 1]`, strides `[H_q, 1, 1, 1]`, and FLOAT data type for hipDNN's
`sink_token` rule (rocm-libraries #14358). This change also changes the
identity hashes of these graphs.

## Metadata contract

All metadata values are non-empty lists of strings. The tags follow the
convention in [the corpus README](../README.md#metadata-tags). Every graph has
these eleven keys:

- `operation`: `sdpa`;
- `direction`: `fwd`;
- `dtype`: `bfloat16` or `float16`;
- `layout`: `bshd`;
- `model_family`: `llm`, or `synthetic` for test cases;
- `model`: normalized lowercase model identifiers;
- `mode`: `inference`;
- `attention_type`: `gqa` or `mha`;
- `mask`: `causal`, `none`, or `sliding_window`;
- `phase`: `prefill`, `chunked_prefill`, or `decode`;
- `sink`: `true` or `false`.

Model identifiers use lowercase letters, digits, and underscores. A family
name joins its version number. For example, `Llama-3-8B` becomes `llama3_8b`,
and `GPT-3-13B-class` becomes `gpt3_13b_class`. A graph that two models share
lists both models: the `gpt-oss-120b/20b` source label becomes `gpt_oss_120b`
and `gpt_oss_20b`. The `Qwen3-Next` label becomes `qwen3_next_80b_a3b`, the
identifier of the only Qwen3-Next model. `model` also contains the models in
the `shared_by` list of the source rows (`mimo_v2_5` on 2 graphs and `qwen3_8b`
on 2 graphs).

The graph tensors and the SDPA node give `attention_type`, `mask`, `phase`,
and `sink`. The source labels `gfx950_dense_test` (9 graphs) and
`d64_serving_gqa8` (4 graphs) are test-case names, not model names. These
graphs have `model` `synthetic` and `model_family` `synthetic`.

## Identity rule

Identity is the SHA-256 of canonical graph JSON after removal of top-level
`name` and `metadata`. The archive keeps the first filename in lexical order
and combines metadata values from duplicate graphs.

## Archive layout

```text
SDPA/
├── Explanation.md
└── files.tar.gz
    └── sdpa/
        └── *.json
```

Run all graphs:

```bash
dnn-benchmark -g Workloads/corpus/SDPA/files.tar.gz
```

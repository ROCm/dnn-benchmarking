# rocKE SDPA Corpus

This collection contains scaled dot-product attention (SDPA) graphs from the
[rocKE SDPA Solera performance set](https://github.com/AMD-ROCm-Internal/mlse-tools-internal/blob/2e0ee4f0cfd0f385b26a128e1a800bf15a066ce4/solera/rocke_scheduled_attn_runs/attention_shapes_llm.txt).
The source file SHA-256 value is
`948f7fce7bb16bbf8a91f307c44dc1709e91a2277a6e540af692f3e2488d3c79`.

## Contents

The source contains 172 graphs. Eight graphs duplicate executable content.
`files.tar.gz` contains the remaining 164 unique graphs and combines model tags
from duplicate entries.

All graphs are forward inference operations. The collection covers decode,
prefill, chunked prefill, grouped-query attention, sliding-window attention,
and attention-sink shapes.

## Metadata contract

All metadata values are non-empty lists of strings. Every graph has these
seven keys:

- `operation`: `sdpa`;
- `direction`: `fwd`;
- `dtype`: `bfloat16` or `float16`;
- `layout`: `bshd`;
- `model_family`: `llm`;
- `model`: normalized lowercase model identifiers;
- `mode`: `inference`.

Model identifiers use lowercase letters, digits, and underscores. For example,
`Llama-3-8B` becomes `llama3_8b`, and `gpt-oss-120b/20b` becomes
`gpt_oss_120b_20b`.

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
dnn-benchmark -g Workloads/corpus/rocke/SDPA/files.tar.gz
```

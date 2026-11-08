# SDPA Backward Corpus

This collection contains scaled dot-product attention (SDPA) backward graphs.
Each graph is the backward equivalent of an SDPA forward graph from the other
`Workloads/` tarballs. `tools/sdpa_fwd_to_bwd.py` makes the conversion.

## Contents

The source tarballs contain 5,309 forward SDPA graphs:

- 5,094 forward graphs have a backward equivalent in hipDNN.
- 320 of the converted graphs duplicate executable content.
- `files.tar.gz` contains the remaining 4,774 unique backward graphs.
- 215 forward graphs have no backward equivalent and are not converted:
  56 use a paged K/V cache and 159 use attention sinks.

| Axis | Graphs |
|---|---|
| `bfloat16` / `half` / `float` | 2,956 / 1,736 / 82 |
| Causal / no mask / additive bias | 2,318 / 2,454 / 2 |
| Grouped-query attention (fewer K/V heads than Q heads) | 1,957 |
| Variable sequence length (`seq_len_q`, `seq_len_kv`) | 2,249 |
| Decode shapes (one query row) | 319 |

Most graphs come from `microbench/aiter` (3,484 sources). The other sources
are `microbench/pytorch` (575), `microbench/cudnn_attention_inference` (308),
`headline/attn` (175), `microbench/rocke` (160), `corpus/SDPA` (152),
`microbench/cudnn_attention_training` (73), `microbench/aotriton` (48),
`microbench/hipkittens` (42), the `models/` tarballs (76) and
`microbench/cudnn_frontend` (1).

## Conversion rules

For each forward graph, the converter:

1. Keeps the Q, K and V tensors and the scale, additive mask and sequence
   length inputs with their UIDs.
2. Makes the forward output O a backward input.
3. Adds dO with the dims, strides and type of O.
4. Adds the `stats` input: the float32 log-sum-exp, dims `[B, Hq, Sq, 1]`,
   packed strides.
5. Adds dQ, dK and dV with the dims, strides and type of Q, K and V.
6. Copies the mask attributes (`causal_mask`, `causal_mask_bottom_right`,
   `left_bound`, `right_bound`, `diagonal_alignment`), `attn_scale_value`,
   `alibi_mask`, `padding_mask` and `dropout_probability` to `parameters`.
7. Drops the forward-only attributes (`max_seq_len_kv`, `implementation`,
   `mma_core_mode`, `generate_stats`) and the top-level `_comment`.
8. Appends `_bwd` to the graph name.

The converter skips a forward graph that has a paged K/V cache, attention
sinks, a block mask, fp8 scale or descale tensors, or dropout.

dnn-benchmarking generates O and `stats` as the exact float32 forward of the
generated Q, K and V. See [methodology](../../../docs/methodology.md#sdpa-backward-inputs).

## Metadata contract

All metadata values are non-empty lists of strings:

- `operation`: `sdpa`;
- `direction`: `bwd`;
- `dtype`: the graph `io_data_type`;
- `source`: `<tarball>:<member>` of every forward graph with this content;
- `model`: normalized lowercase model identifiers, when a source graph has a
  `model` tag.

## Identity rule

Identity is the SHA-256 of canonical graph JSON after removal of top-level
`name` and `metadata`. The archive keeps the name of the first source in
lexical order and combines the `source` and `model` values of duplicates.

## Known limits

- The variable-length graphs come from `microbench/aiter` (2,244) and
  `microbench/aotriton` (5). dnn-benchmarking sets every sequence length to 1
  for them, as for their forward graphs, so these rows measure a one-token
  problem. It still generates O and `stats` for the dense shape. Variable
  sequence length support is tracked in ALMIOPEN-3139.
- The PyTorch reference does not support variable sequence lengths, so these
  graphs have no PyTorch row and no validation.
- Decode shapes have no training use. They are kept because the collection
  converts every forward graph that has a backward equivalent.

## Regenerate

From the repository root, after `dvc pull`:

```bash
python tools/sdpa_fwd_to_bwd.py -o Workloads/corpus/SDPABackward/files.tar.gz \
  'Workloads/headline/*.tar.gz' 'Workloads/microbench/*.tar.gz' \
  'Workloads/models/*.tar.gz' Workloads/corpus/SDPA/files.tar.gz
```

## Archive layout

```text
SDPABackward/
├── Explanation.md
└── files.tar.gz
    └── sdpa_bwd/
        └── *.json
```

Run all graphs:

```bash
dnn-benchmark -g Workloads/corpus/SDPABackward/files.tar.gz
```

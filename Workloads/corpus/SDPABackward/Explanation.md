# SDPA Backward Corpus

This collection contains scaled dot-product attention (SDPA) backward graphs.
`tools/sdpa_fwd_to_bwd.py` builds it from the other `Workloads/` tarballs.
Each graph has one of three origins (`metadata.origin`):

- `converted_fwd`: the backward equivalent of an SDPA forward graph.
- `captured_bwd`: an SDPA backward graph that the tarballs already contain.
- `model_spec`: a training graph with a feature that no captured workload
  has, at the published shape of a named model.

## Contents

`files.tar.gz` contains 4,790 unique backward graphs:

- 4,774 come from the 5,309 forward SDPA graphs. 5,094 forward graphs have a
  backward equivalent in hipDNN, and 320 of them duplicate other content.
  215 forward graphs have no backward equivalent: 56 use a paged K/V cache
  and 159 use attention sinks.
- 102 come from captured backward graphs (`headline/attn` and
  `microbench/cudnn_attention_training`). 92 of them are identical to a
  converted graph, so only 10 are new.
- 6 come from `model_spec` entries.

| Axis | Graphs |
|---|---|
| `bfloat16` / `half` / `float` | 2,969 / 1,739 / 82 |
| Causal / no mask / additive bias | 2,327 / 2,457 / 6 |
| Dropout / dBias / ALiBi | 5 / 3 / 2 |
| Grouped-query attention (fewer K/V heads than Q heads) | 1,964 |
| Variable sequence length (`seq_len_q`, `seq_len_kv`) | 2,249 |
| `metadata.shape`: `sq_eq_skv` / `sq_lt_skv` / `sq_gt_skv` / `decode` | 1,503 / 2,336 / 632 / 319 |

Most graphs come from `microbench/aiter` (3,484 sources). The other sources
are `microbench/pytorch` (575), `microbench/cudnn_attention_inference` (308),
`headline/attn` (210), `microbench/rocke` (160), `corpus/SDPA` (152),
`microbench/cudnn_attention_training` (141), `microbench/aotriton` (48),
`microbench/hipkittens` (42), the `models/` tarballs (76),
`microbench/cudnn_frontend` (1) and `model_spec` (6).

Training-shaped graphs have `sq_eq_skv`. Most `sq_lt_skv` and all `decode`
graphs come from inference workloads (chunked prefill and KV-cache decode).

## Training graphs with dropout, bias and ALiBi

No captured workload has a backward graph with dropout, a bias gradient or
ALiBi. The collection adds them:

| Graph | Origin | Feature |
|---|---|---|
| `llama-3-8b-sft` (2 graphs), `vit-l-14` | `captured_bwd` | Dropout 0.1 from the capture's `metadata.dropout` |
| `bert-large-uncased`, 16 x 512 tokens | `model_spec` | Padding bias `[B, 1, 1, S]`, dropout 0.1 |
| `t5-large` encoder, 16 x 512 tokens | `model_spec` | Relative position bias `[1, H, S, S]` with dBias, dropout 0.1 |
| `swin-base` stage 1, 4,096 windows of 49 tokens | `model_spec` | Relative position bias with dBias |
| `alphafold2` MSA row attention, 512 rows of 256 residues | `model_spec` | Pair bias with dBias |
| `bloom-176b`, `mpt-7b`, 2,048 tokens | `model_spec` | Causal ALiBi |

Dropout uses cuDNN's Philox form: `dropout_probability` with int64 `seed`
and `offset` tensors of dims `[1, 1, 1, 1]`. The `model_spec` shapes and
their sources are in `_MODEL_SPECS` in `tools/sdpa_fwd_to_bwd.py`.

At rocm-libraries `b279712`, the hipDNN ASM SDPA backward engine declines
dropout, bias, dBias and ALiBi. The PyTorch reference skips dropout graphs,
which include the `t5-large` dBias graph. It computes dBias for the
`swin-base` and `alphafold2` graphs, and runs the ALiBi graphs with its fp32
formula. These graphs are in the collection so that new engines have a
target.

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
   When the graph has both `seq_len_q` and `seq_len_kv`, it sets
   `padding_mask = true`, because cuDNN requires the flag with sequence
   lengths (SEM-008 in hipDNN's `SdpaSemanticsRegister.md`).
7. Drops the forward-only attributes (`max_seq_len_kv`, `implementation`,
   `mma_core_mode`, `generate_stats`) and the top-level `_comment`.
8. Appends `_bwd` to the graph name.

The converter skips a forward graph that has a paged K/V cache, attention
sinks, a block mask, fp8 scale or descale tensors, or dropout.

dnn-benchmarking generates O and `stats` as the exact float32 forward of the
generated Q, K and V, with the bias and ALiBi. It does not apply dropout, so
O of a dropout graph is the forward without dropout. See
[methodology](../../../docs/methodology.md#sdpa-backward-inputs).

## Metadata contract

All metadata values are non-empty lists of strings:

- `operation`: `sdpa`;
- `direction`: `bwd`;
- `dtype`: the graph `io_data_type`;
- `shape`: `decode` (Sq = 1), `sq_lt_skv`, `sq_eq_skv` or `sq_gt_skv`;
- `origin`: `converted_fwd`, `captured_bwd` or `model_spec`, sorted;
- `source`: `<tarball>:<member>` of every source graph with this content, or
  `model_spec:<source of the shape>`;
- `model`: normalized lowercase model identifiers, when a source graph has a
  `model` tag.

## Identity rule

Identity is the SHA-256 of canonical graph JSON after removal of top-level
`name` and `metadata`. The archive keeps the name of the first source in
lexical order and combines the `origin`, `source` and `model` values of
duplicates.

## Known limits

- The variable-length graphs come from `microbench/aiter` (2,244) and
  `microbench/aotriton` (5). Their `seq_len_q` and `seq_len_kv` are int32
  `[B, 1, 1, 1]`, as hipDNN and cuDNN require. dnn-benchmarking sets every
  sequence length to 1 for them, as for their forward graphs, so these rows
  measure a one-token problem. It still generates O and `stats` for the
  dense shape. Variable sequence length support is tracked in ALMIOPEN-3139.
- The PyTorch reference does not support variable sequence lengths, so these
  graphs have no PyTorch row and no validation.
- Decode shapes have no training use. They are kept because the collection
  converts every forward graph that has a backward equivalent. Use
  `metadata.shape` to separate them from training shapes.
- The captured `gpt-oss-120b` sink graph has `metadata.sink`, but hipDNN has
  no backward sink input yet (ALMIOPEN-3020), so the graph has no sink.

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

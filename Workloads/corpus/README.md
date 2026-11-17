# hipDNN Workload Corpus

The `Workloads/corpus/` directory stores curated reference sets of executable
hipDNN graphs.

## Goal

The goal is to represent all hipDNN workloads with the smallest practical
number of graphs. This goal is an admission rule for the folder. It is not a
claim about the current coverage.

Today, the folder contains three collections. `Convolution/` contains
9,471 unique convolution geometries from 31 models. `SDPA/` contains 164
SDPA forward graphs. `SDPABackward/` contains 4,790 SDPA backward graphs.

## Golden standard

In this folder, a golden standard is a curated reference set of hipDNN graphs.
It supplies stable workload inputs for performance tests, correctness tests,
and comparisons between engines and GPU architectures.

The term does not refer to expected output tensors, performance values, or
golden intermediate representation files. Results and engine support depend
on the GPU, the hipDNN providers, and the software configuration.

## Workload directory roles

Each `Workloads/` directory has a different purpose:

| Directory | Purpose |
|---|---|
| `Workloads/headline/` | Small, curated set for routine regression monitoring. |
| `Workloads/microbench/` | Large shape sweeps and collections from one source. |
| `Workloads/models/` | Workloads that preserve the graphs for each model. |
| `Workloads/corpus/` | Reference collections that represent workload classes across models with minimal graph overlap. |

Use `headline/` for a small and stable regression set. Use `models/` to study
one model. Use `microbench/` for broad shape sweeps. Use `corpus/` to measure a
workload class across many representative models.

## Metadata tags

All graphs in the `Workloads/` sets use one tag convention. The tags are in
the top-level `metadata` object of each graph. hipDNN does not read this
object, and `dnn-benchmark` does not read it.

Rules:

1. Each value is a sorted array of unique lowercase strings.
2. A value is a short label. Do not write sentences or notes in a tag.
3. Do not copy a value that the graph already contains, such as a dimension,
   a scale, or an epsilon.
4. Omit a key when the source does not give its value. Do not write a guess.
5. Write the keys in the order of the table below.
6. Do not merge files outside `corpus/` that contain the same executable
   graph. Each file keeps its own tags. Only a `corpus/` collection removes
   duplicate graphs (see [Duplicate policy](#duplicate-policy)).

| Key | Values | Applies to |
|---|---|---|
| `operation` | `convolution`, `sdpa`, `matmul`, `moe_grouped_matmul`, `pointwise`, `batchnorm`, `rmsnorm`, `layernorm`, `block_scale_quantize`, `block_scale_dequantize`, `resample`, `reduction` (one for each node type in the graph) | All graphs |
| `direction` | `fwd`; `dgrad` or `wgrad` for convolution and MoE gradients; `bwd` for other gradients | All graphs |
| `dtype` | `float32`, `float16`, `bfloat16`, `float8_e4m3` (the graph I/O data type) | All graphs |
| `source_dtype` | `float16`, `fp8`, `fp8_block`, `fp6`, `mxfp8`, `mxfp4`, `nvfp4` (the source precision, when the graph uses a different data type) | Graphs whose source precision is not `dtype` |
| `layout` | `nchw` or `nhwc` (convolution, batch normalization, resample); `bshd` or `bhsd` (SDPA) | Graphs with a layout |
| `model_family` | `llm`, `cls`, `detection`, `diffusion`, `segmentation`, `superres`, `video3d`, `med3d`, `audio1d`, `multimodal`, `ssm`, `synthetic` | Graphs with a known model |
| `model` | Model identifiers, such as `llama3_8b` or `resnet50` | Graphs with a known model |
| `mode` | `inference`, `training` | Graphs with a known capture mode |
| `size` | Source workload size label | `corpus/Convolution` |
| `from_transpose` | `true`, `false` | `corpus/Convolution` |
| `convolution_type` | `pointwise`, `standard`, `strided`, `depthwise`, `grouped`, `dilated`, `transposed` | Convolution graphs |
| `attention_type` | `mha`, `gqa`, `mqa`, `mla` | SDPA graphs |
| `mask` | `none`, `causal`, `sliding_window` | SDPA graphs |
| `phase` | `prefill`, `chunked_prefill`, `decode` | SDPA graphs, MoE graphs with a known phase |
| `sink` | `true`, `false` | SDPA graphs |
| `source_feature` | `doc_mask`, `varlen` (an SDPA feature of the source that the graph does not express) | SDPA graphs whose source uses such a feature |
| `activation` | `swiglu`, `swiglu_clamped`, `swiglu_oai`, `situglu` | MoE graphs with a fused gated activation |
| `tensor_parallel` | `1`, `2`, `4`, `8` | `microbench/cudnn_attention_inference` |
| `suite` | `regular`, `extended` | `headline/attn`, `headline/moe`, `microbench/rocke` |
| `priority` | `p0`, `p1`, `p2` | `headline/attn`, `headline/moe`, `microbench/rocke` |
| `purpose` | `acceptance`, `perf_diagnostic`, `frontier`, `coverage_gap`, `ep_scaling` | `headline/attn`, `headline/moe`, `microbench/rocke` |

A model identifier uses lowercase letters, digits, and underscores. A family
name joins its version number: `Llama-3.1-405B` becomes `llama3_1_405b`, and
`GPT-3` becomes `gpt3`. When two or more models use the same graph, `model`
contains all of them. A source label that names a test case and not a model,
such as `gfx950_dense_test`, becomes `model` `synthetic` and `model_family`
`synthetic`. Shapes from library tests and benchmark sweeps that name no
model (for example `microbench/aiter`, `microbench/hipkittens`, and
`microbench/pytorch`) also get `model` and `model_family` `synthetic`.

Tag meanings:

- `mode` is `training` for every gradient graph. A forward normalization
  graph that saves its statistics is also a `training` graph.
- `attention_type` is `mla` when the source names multi-head latent attention,
  when the model uses it (DeepSeek-V4), or when the query-key and value head
  dimensions differ. Otherwise the head counts set it: equal counts give
  `mha`, one key-value head gives `mqa`, and other counts give `gqa`.
- `source_dtype` is present when hipDNN cannot express the source precision,
  or when the graph uses another data type than the source. The graph then
  uses `dtype`, for example `bfloat16`.
- `phase` comes from the source when the source gives it. Otherwise one query
  token gives `decode`, a query shorter than the key-value sequence gives
  `chunked_prefill`, and equal lengths give `prefill`.
- `mask` is `sliding_window` when the SDPA node has a left or right bound.
## Current collections

| Collection | Scope | Documentation |
|---|---|---|
| `Convolution/` | 9,471 unique convolution geometries from 31 models | [Convolution corpus guide](Convolution/Explanation.md) |
| `SDPA/` | 164 unique SDPA forward graphs from the rocKE Solera performance set | [SDPA corpus guide](SDPA/Explanation.md) |
| `SDPABackward/` | 4,790 unique SDPA backward graphs: every dense SDPA forward graph in `Workloads/` converted, the captured backward graphs, and training graphs with dropout, bias, dBias and ALiBi | [SDPA backward corpus guide](SDPABackward/Explanation.md) |

The convolution collection includes forward, backward-data, and
backward-weight graphs. It covers 2-D, 3-D, audio, image, and video workloads.
It does not represent all hipDNN operations.

## Duplicate policy

Duplicate removal occurs inside each collection. No check compares graphs
between different collections.

The convolution collection calculates identity as follows:

1. Remove the top-level `name` and `metadata` keys from the graph JSON.
2. Convert the remaining graph JSON to its canonical form.
3. Calculate the SHA-256 value of the canonical JSON.
4. Keep one file for each SHA-256 value.
5. Combine the source metadata for all sources that use the geometry.

Keep separate graphs when an executable field is different. Executable fields
include tensor dimensions, strides, data types, operation attributes, and
operation direction.

This policy minimizes graph overlap inside the convolution collection. It
also preserves the source data for a shared geometry.

## Get the current collection

Install DVC with S3 support:

```bash
python -m pip install "dvc[s3]"
```

From the repository root, get the convolution corpus:

```bash
dvc pull Workloads/corpus/Convolution/files.tar.gz.dvc
```

Run all convolution graphs:

```bash
dnn-benchmark \
  -g Workloads/corpus/Convolution/files.tar.gz \
  -o corpus-results.json
```

## Add a collection

Add one folder for each workload class:

```text
Workloads/corpus/<Name>Corpus/
├── Explanation.md
├── files.tar.gz.dvc
└── files.tar.gz
```

The DVC command creates `files.tar.gz.dvc`. A fresh clone does not contain
`files.tar.gz` until DVC downloads it.

Complete these steps:

1. Define the workload class and the coverage goal.
2. Collect complete and executable hipDNN graphs.
3. Create one stable identity rule for the collection.
4. Remove duplicate executable graphs inside the collection.
5. Run a duplicate check and record the result in `Explanation.md`.
6. Preserve the source metadata for each retained graph.
7. Document the source, conversion rules, coverage, and known limits.
8. Store `files.tar.gz` with DVC.
9. Add the collection to the table in this file.

Admit a collection only when it adds a missing workload class or a missing
coverage axis. Prefer the smallest graph set that preserves the required
coverage.

# rocKE Convolution Corpus

This collection contains 315 hipDNN convolution graphs converted from two
rocKE Solera shape files. Both files come from the
[pinned rocKE convolution shapes directory](https://github.com/AMD-ROCm-Internal/mlse-tools-internal/tree/f35c898edce1090fbcf3f31f542284eb4be4c117/solera/conv_shapes).

## Sources

| Source file | Commands | SHA-256 |
|---|---:|---|
| `conv_corpus_nhwc.txt` | 283 | `883264b9db163381db8f49a6bc2676d9eed45ab05a0d894689a812eb49590984` |
| `freemont_tracelense.txt` | 32 | `931c3a1e3f444f8db9650ed48e493f8a5a9758f248006e4d471c878a66bb0897` |

The Fremont TraceLens source does not record a model name. The broader
convolution corpus source also does not map commands to individual models.
Graphs without an identity match use provenance-neutral model values.

## Contents

`files.tar.gz` contains both source files in one executable graph collection:

| Direction | Graphs |
|---|---:|
| Forward | 159 |
| Backward-data | 78 |
| Backward-weight | 78 |
| **Total** | **315** |

The archive has 266 `bfloat16`, 47 `float16`, and 2 `float32` graphs.
`convolution_type` can contain more than one value. It covers standard,
pointwise, strided, dilated, grouped, and depthwise convolution.

## Metadata contract

All metadata values are non-empty lists of strings. Every graph has these
keys:

- `operation`: `convolution`;
- `direction`: `fwd`, `dgrad`, or `wgrad`, derived from the node type;
- `dtype`: `bfloat16`, `float16`, or `float32`;
- `layout`: `nhwc`;
- `model_family`: inherited from `ConvolutionModelsCorpus` or `unknown`;
- `model`: inherited source models, `rocke_conv_corpus`, or `freemont_trace`;
- `mode`: inherited or derived as `inference` or `training`;
- `from_transpose`: `false`;
- `convolution_type`: corpus-compatible convolution classes.

Backward commands use `mode: ["training"]`. A forward graph uses `training`
when the same source has a matching gradient geometry. Other forward graphs
use `inference`. For 144 overlapping identities, `model_family`, `model`,
`mode`, `from_transpose`, and `convolution_type` come directly from
`ConvolutionModelsCorpus`.

The converter maps `convfp32` to the hipDNN converter's equivalent `conv`
operation name. This mapping preserves `float32` graph I/O.

Generated filenames include `_bf16`, `_fp16`, or `_fp32`. This suffix prevents
different data types with the same geometry from using the same archive path.

## Identity and overlap

Identity is the SHA-256 of canonical graph JSON after removal of top-level
`name` and `metadata`. The two sources contain no duplicate executable graphs.
All 32 previously packaged Fremont graphs reproduce the same executable
identities after conversion.

The collection has 144 executable identities that also occur in
`ConvolutionModelsCorpus`. Cross-collection overlap is retained because corpus
deduplication applies inside each collection. Their attribution metadata is
identical to the existing corpus records.

## Archive layout

```text
Convolution/
├── Explanation.md
└── files.tar.gz
    └── convolution/
        ├── fwd/
        ├── dgrad/
        └── wgrad/
```

Run all graphs:

```bash
dnn-benchmark -g Workloads/corpus/rocke/Convolution/files.tar.gz
```

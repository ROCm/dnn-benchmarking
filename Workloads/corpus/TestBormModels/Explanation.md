# hipDNN BatchNorm Test Models

This collection contains 24 synthetic BatchNorm graphs. It covers forward
training, backward, and inference with the same eight ResNet-scale shapes.
Use it to compare BatchNorm engines across directions and tensor ranks.

## Contents

- `files.tar.gz`: contains 24 hipDNN BatchNorm graphs.
  - `bnorm/fwd/`: 8 forward-training graphs.
  - `bnorm/bwd/`: 8 backward graphs.
  - `bnorm/inference/`: 8 inference graphs.
  - `bnorm/MANIFEST.md`: records the shapes and graph semantics.

The shapes include 2-D NHWC and 3-D NDHWC tensors. Graph input and output
tensors use `bfloat16`. Computation and statistics use `float32` where the
hipDNN operation supports them.

## Metadata

All metadata values are lists. Each graph has these fields:

| Key | Values |
|---|---|
| `operation` | `batchnorm` |
| `direction` | `fwd` or `bwd` |
| `dtype` | `bfloat16` |
| `layout` | `nhwc` |
| `model_family` | `cls` |
| `model` | `synthetic_resnet_scale` |
| `mode` | `training` or `inference` |

Inference graphs use `direction: ["fwd"]` and `mode: ["inference"]`.

The collection does not use `source` or `use_case`. The `model_family` value
`cls` follows the image-classification convention in the convolution corpus.
The `nhwc` value represents channels-last layout for both 2-D and 3-D graphs.

## Identity and coverage

Each file has a unique executable graph after removal of the top-level `name`
and `metadata` fields. The collection contains:

- 8 forward-training graphs;
- 8 backward graphs;
- 8 inference graphs;
- 24 unique executable graphs.

The inference graphs use the same shapes as the forward graphs. They replace
the batch-statistics calculation with precomputed mean and inverse-variance
inputs.

## Archive layout

```text
TestBormModels/
├── Explanation.md
└── files.tar.gz
    └── bnorm/
        ├── MANIFEST.md
        ├── fwd/
        │   └── *.json
        ├── bwd/
        │   └── *.json
        └── inference/
            └── *.json
```

Run all graphs:

```bash
dnn-benchmark -g Workloads/corpus/TestBormModels/files.tar.gz
```

Operation support depends on the installed hipDNN providers and target GPU.

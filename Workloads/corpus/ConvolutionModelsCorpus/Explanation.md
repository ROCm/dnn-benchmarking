# hipDNN Convolution Models Corpus

The shape table captures convolution workloads from 31 AI models. It includes inference and training workloads across eight model families.

## Contents

- `files.tar.gz`: contains 9,471 unique hipDNN convolution geometries.

`dnn-benchmark` extracts and runs every JSON file in the archive. Extract the
archive before you select a direction, model, or other subset.

## Model coverage

| Family | Models |
|---|---|
| Image classification and backbones | `convnext_base`, `efficientnet_v2_s`, `hrnet_w32`, `inception_v3`, `mobilefacenet`, `mobilenet_v3_large`, `regnet_y_8gf`, `resnet50`, `resnext101_32x8d`, `resnext50_32x4d`, `vit_patch_embed` |
| Detection and instance segmentation | `maskrcnn_resnet50_fpn`, `retinanet_resnet50_fpn`, `ssd300_vgg16`, `yolo11` |
| Diffusion and image generation | `flux_vae`, `sd15_unet`, `sdxl_unet`, `sdxl_vae`, `stylegan2` |
| Semantic segmentation | `deeplabv3_resnet50`, `unet2d` |
| Super-resolution | `realesrgan`, `superres` |
| Video and 3-D vision | `ir_csn_50`, `r2plus1d_18`, `resnext3d_50`, `wan_vae` |
| Medical imaging | `nnunet3d` |
| Speech and audio | `conformer`, `hifigan` |

Audio workloads use 2-D graphs with `H=1`. The corpus also covers pointwise, dense, grouped, depthwise, dilated, strided, transposed, and 3-D geometry.

## Coverage

| Property | Coverage |
|---|---:|
| Unique executable geometries | 9,471 |
| Models | 31 |
| Model families | 8 |
| Forward geometries | 4,546 |
| Backward-data geometries | 2,481 |
| Backward-weight geometries | 2,444 |
| NCHW geometries | 7,280 |
| NHWC geometries | 2,191 |
| Non-3-D geometries | 8,497 |
| 3-D geometries | 974 |

### Data types

| Data type | Geometries |
|---|---:|
| `float16` | 6,295 |
| `bfloat16` | 1,709 |
| `float32` | 1,467 |

### Source convolution classes

| Class | Source rows |
|---|---:|
| Pointwise | 5,160 |
| Standard | 3,546 |
| Strided | 1,216 |
| Depthwise | 568 |
| Grouped | 483 |
| Dilated | 470 |
| Transposed | 254 |

A transposed source operation can map to a backward-data graph. The `from_transpose` metadata field preserves this source information.

## Archive layout

```text
ConvolutionModelsCorpus/
├── Explanation.md
└── files.tar.gz
    └── conv/
        ├── MANIFEST.md
        ├── fwd/
        │   └── *.json
        ├── dgrad/
        │   └── *.json
        └── wgrad/
            └── *.json
```

The directories map to these hipDNN node types:

| Directory | hipDNN node type | MIOpen direction |
|---|---|---:|
| `fwd/` | `ConvolutionFwdAttributes` | `-F 1` |
| `dgrad/` | `ConvolutionBwdAttributes` | `-F 2` |
| `wgrad/` | `ConvolutionWrwAttributes` | `-F 4` |

## File naming and identity

Each file represents one executable geometry. Its graph name and filename use
one existing source `shape_id`. The source data has exactly one unique
`shape_id` for every geometry.

Geometry identity is the SHA-256 of canonical graph JSON after removal of the
top-level `name` and `metadata` keys. Canonical JSON sorts object keys, uses
stable separators, and rejects non-finite numbers.

## Graph format

Each JSON file contains:

- graph-level compute and input/output data types;
- input, weight, and output tensors;
- one hipDNN convolution node;
- a graph name derived from an existing source shape ID;
- explicit performance tags.

Example structure:

```json
{
  "compute_data_type": "float",
  "io_data_type": "bfloat16",
  "intermediate_data_type": "float",
  "tensors": [],
  "nodes": [
    {
      "type": "ConvolutionFwdAttributes"
    }
  ],
  "name": "<source-shape-id>",
  "metadata": {
    "operation": ["convolution"],
    "direction": ["fwd"],
    "dtype": ["bfloat16"],
    "layout": ["nchw"],
    "model_family": ["audio1d"],
    "model": ["conformer"],
    "mode": ["inference", "training"],
    "size": ["200"],
    "from_transpose": ["false"]
  }
}
```

The abbreviated example omits tensor and convolution parameters. The files in
the archive contain complete graph definitions.

## Metadata tags

Every metadata value is a sorted array of unique lowercase strings. The nine
bounded keys are:

- `operation`: `convolution`;
- `direction`: `fwd`, `dgrad`, or `wgrad`;
- `dtype`: `bfloat16`, `float16`, or `float32`;
- `layout`: `nchw` or `nhwc`;
- `model_family`: source workload family;
- `model`: source model name;
- `mode`: inference or training capture mode;
- `size`: source workload size label;
- `from_transpose`: `true` or `false`.

## Run with dnn-benchmark

Run all graphs from the repository root:

```bash
dnn-benchmark -g Workloads/corpus/ConvolutionModelsCorpus/files.tar.gz -o corpus-results.json
```

Run with one engine:

```bash
dnn-benchmark \
  -g Workloads/corpus/ConvolutionModelsCorpus/files.tar.gz \
  -e MIOPEN_ENGINE \
  -o corpus-miopen-results.json
```

Validate engine output against PyTorch:

```bash
dnn-benchmark \
  -g Workloads/corpus/ConvolutionModelsCorpus/files.tar.gz \
  --validate pytorch \
  -o corpus-validation-results.json
```

The archive interface runs all 9,471 geometries. Complete these steps to run only forward graphs:

1. Extract the archive.

   ```bash
   mkdir -p /tmp/hipdnn-conv-corpus
   tar -xzf Workloads/corpus/ConvolutionModelsCorpus/files.tar.gz \
     -C /tmp/hipdnn-conv-corpus
   ```

2. Run the extracted forward graphs.

   ```bash
   dnn-benchmark -g '/tmp/hipdnn-conv-corpus/conv/fwd/*.json' \
     -o fwd-results.json
   ```

## Conversion provenance

Conversion derives layout from executable activation strides when the layout
is observable and uses the source layout only for stride-ambiguous geometries.
The converted graphs are grouped by executable geometry, retain one source
shape ID, and union normalized tag values. The archive uses
timestamp-independent gzip and tar metadata.

## Validation

The generated corpus passed these checks:

- 11,697 source rows resolve to 9,471 unique geometries;
- no two files have the same executable geometry identity;
- every filename and graph name uses an existing source shape ID;
- every graph has all nine metadata keys;
- every metadata value is a sorted array of unique lowercase strings;
- every graph has a nonempty `operation` tag;
- every graph has a valid operation node;
- every non-virtual tensor passes the repository tensor loader;
- every node type matches its archive directory.

Operation support depends on the installed hipDNN providers and target GPU. A valid graph can still have no applicable engine on a specific system.


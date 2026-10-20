# hipDNN Convolution Models Corpus

This archive contains convolution workloads from 31 AI models. It covers
inference and training across eight metadata families and 12 use cases. The
source capture recorded the convolution calls made by model executions. The
archive then converted and deduplicated those calls into executable hipDNN
graphs for performance analysis, competitive comparison, and key performance
indicator (KPI) tracking.

## Contents

- `files.tar.gz`: contains 9,471 unique hipDNN convolution geometries.
  - `conv/MANIFEST.md`: records the archive count, identity rule, and metadata
    contract.

`dnn-benchmark` extracts and runs every JSON file in the archive. Extract the
archive before you select a direction, model, or other subset.

## Model coverage

The `model_family` metadata uses short, bounded values. These values group the
31 models as follows:

| Metadata value | Meaning | Models |
|---|---|---|
| `cls` | Image classification, vision backbones, face recognition, and transformer patch embedding | `convnext_base`, `efficientnet_v2_s`, `hrnet_w32`, `inception_v3`, `mobilefacenet`, `mobilenet_v3_large`, `regnet_y_8gf`, `resnet50`, `resnext101_32x8d`, `resnext50_32x4d`, `vit_patch_embed` |
| `detection` | Object detection and instance segmentation | `maskrcnn_resnet50_fpn`, `retinanet_resnet50_fpn`, `ssd300_vgg16`, `yolo11` |
| `diffusion` | Diffusion, variational autoencoder (VAE), and generative adversarial network (GAN) image generation | `flux_vae`, `sd15_unet`, `sdxl_unet`, `sdxl_vae`, `stylegan2` |
| `segmentation` | 2-D semantic segmentation | `deeplabv3_resnet50`, `unet2d` |
| `superres` | Super-resolution and image restoration | `realesrgan`, `superres` |
| `video3d` | Video generation and action recognition | `ir_csn_50`, `r2plus1d_18`, `resnext3d_50`, `wan_vae` |
| `med3d` | 3-D medical-image segmentation | `nnunet3d` |
| `audio1d` | Speech recognition and speech synthesis | `conformer`, `hifigan` |

### How to interpret model attribution

The archive stores unique executable geometries, not complete models. A model
can use the same geometry many times, and different models can use the same
geometry. Deduplication keeps one file for that geometry and unions its source
tags. Therefore:

- a `model` tag identifies every source model that used the geometry;
- a `size` tag identifies every source size associated with the geometry;
- one file can have multiple `model`, `model_family`, `mode`, or `size` values;
- model tags do not partition the 9,471 files into disjoint totals;
- repeated layers in one model do not create duplicate files.

In this archive, 1,306 files have more than one model tag. One shared geometry
has at most five model tags. These many-to-many tags preserve provenance after
deduplication.

The sizes in the model guide below describe the source capture setup. A file
can contain more size values when another workload uses the same geometry.
Layout and data type also describe the source capture intent. The converter
uses executable tensor strides to determine the graph layout when those
strides distinguish NCHW from NHWC.

## Convolution and model terms

- **Pointwise convolution:** a 1x1 convolution that mixes channels without a
  spatial neighborhood.
- **Dense or standard convolution:** every input channel contributes to every
  output channel.
- **Grouped convolution:** channels are split into independent groups.
- **Depthwise convolution:** one group is used for each input channel.
- **Depthwise-separable convolution:** a depthwise convolution followed by a
  pointwise convolution.
- **Dilated or atrous convolution:** gaps in the kernel increase its receptive
  field without downsampling.
- **Transposed convolution:** a learned upsampling operation. PyTorch executes
  its forward operation as an equivalent convolution backward-data operation.
- **Factorized convolution:** a larger kernel is split into smaller asymmetric
  kernels, such as 1x7 followed by 7x1.
- **(2+1)D convolution:** a 3-D convolution is split into a spatial 1x3x3
  convolution and a temporal 3x1x1 convolution.
- **MBConv:** a mobile inverted bottleneck with pointwise expansion, depthwise
  convolution, and pointwise projection.
- **SE:** squeeze-and-excitation channel attention.
- **ASPP:** atrous spatial pyramid pooling. It runs parallel convolutions with
  different dilation rates.
- **FPN:** a feature pyramid network that combines features at multiple
  resolutions.
- **RRDB:** a residual-in-residual dense block used by Real-ESRGAN.
- **GDConv:** a global depthwise convolution whose kernel spans the complete
  feature map.
- **UNet:** an encoder-decoder network with skip connections.
- **VAE:** variational autoencoder.
- **GAN:** generative adversarial network.

## Model guide

The following tables explain why each model is in the corpus. The layout,
data type, and size values describe the source model runs. `NCHW` is
channels-first. `NHWC` is channels-last. For 3-D models, the layout also
includes the depth or time dimension.

### Image classification and backbones

These models cover common feature extractors and the main 2-D convolution
classes.

| Model | Workload and convolution coverage | Source capture |
|---|---|---|
| `resnet50` | Baseline bottleneck network with 1x1, dense 3x3, strided, and 7x7 stem convolutions. | NCHW, `float16`; 224 and 256 pixel inputs |
| `resnext50_32x4d` | ResNet topology with cardinality-32 grouped 3x3 convolutions. | NCHW, `float16`; 224 and 256 pixel inputs |
| `resnext101_32x8d` | A deeper and wider cardinality-32 grouped-convolution backbone. | NCHW, `float16`; 224 and 256 pixel inputs |
| `convnext_base` | Modern backbone with depthwise 7x7 and pointwise 1x1 convolutions. | NHWC, `bfloat16`; 224 and 256 pixel inputs |
| `efficientnet_v2_s` | Fused MBConv, depthwise-separable 3x3, pointwise 1x1, and SE operations. | NHWC, `float16`; 384 and 480 pixel inputs |
| `mobilenet_v3_large` | Mobile depthwise-separable network with 3x3 and 5x5 depthwise kernels and SE. | NCHW, `float16`; 224 and 256 pixel inputs |
| `hrnet_w32` | Multi-resolution backbone with parallel branches, strided 3x3 downsampling, and 1x1 fusion. | NCHW, `float16`; 224 and 256 pixel inputs |
| `inception_v3` | Parallel branches and factorized 1x7, 7x1, 1x3, and 3x1 convolutions. | NCHW, `float16`; 299 pixel inputs |
| `regnet_y_8gf` | Grouped 3x3 convolutions and SE with RegNet channel widths. | NCHW, `float16`; 224 and 256 pixel inputs |

### Semantic segmentation

| Model | Workload and convolution coverage | Source capture |
|---|---|---|
| `deeplabv3_resnet50` | Dense per-pixel labeling with a ResNet backbone and ASPP. It covers dilated 3x3 convolutions at multiple rates. | NCHW, `float16`; 520 and 1024 pixel inputs |
| `unet2d` | Symmetric encoder-decoder with dense 3x3 blocks, skip fusion, 2x2 stride-2 transposed convolutions, and a 1x1 head. | NCHW, `float16`; 256 and 512 pixel inputs |

### Object detection and instance segmentation

| Model | Workload and convolution coverage | Source capture |
|---|---|---|
| `retinanet_resnet50_fpn` | Single-stage anchor detector with FPN lateral 1x1 operations, 3x3 fusion, and repeated classification and box-regression towers. | NCHW, `float32`; 640 and 800 pixel inputs |
| `maskrcnn_resnet50_fpn` | Two-stage detector with FPN, region proposal, box, and mask heads. Its mask head adds 2x2 transposed convolution. | NCHW, `float32`; 640 and 800 pixel inputs |
| `ssd300_vgg16` | Single-stage detector with a plain VGG 3x3 backbone and multi-scale 3x3 heads. | NCHW, `float32`; fixed 300 pixel input |
| `yolo11` | Modern single-stage detector with cross-stage partial blocks, spatial pyramid pooling, attention, and a depthwise-separable head. | NCHW, `float16`; 512 and 640 pixel inputs |

### Super-resolution and image restoration

| Model | Workload and convolution coverage | Source capture |
|---|---|---|
| `superres` | Lightweight x4 super-resolution network. A convolution expands the channels before PixelShuffle performs sub-pixel upsampling. | NCHW, `float16`; 64 and 128 pixel low-resolution inputs |
| `realesrgan` | x4 RRDB restoration network with dense concatenation and nearest-neighbor resize followed by 3x3 convolution. | NCHW, `float16`; 64 and 128 pixel low-resolution inputs |

The two models use different upsampling methods. `superres` covers sub-pixel
upsampling. `realesrgan` covers resize-convolution and widening RRDB inputs.

### Diffusion and VAE image generation

| Model | Workload and convolution coverage | Source capture |
|---|---|---|
| `sdxl_unet` | Stable Diffusion XL denoiser with dense 3x3, pointwise 1x1, and upsample and downsample convolutions. | NHWC, `bfloat16`; latent heights 96, 128, and 160 |
| `sd15_unet` | Stable Diffusion 1.5 denoiser with the same broad convolution classes and a different channel and resolution distribution. | NHWC, `bfloat16`; latent heights 64, 96, and 128 |
| `sdxl_vae` | Image decoder with dense 3x3 and upsampling convolutions at a four-channel latent boundary. | NHWC, `bfloat16`; latent heights 96 and 128 |
| `flux_vae` | Image decoder with a 16-channel latent boundary used by newer transformer-based diffusion systems. | NHWC, `bfloat16`; latent heights 96 and 128 |

The latent height is one eighth of the decoded image height for these source
runs. For example, a latent height of 128 corresponds to a 1024 pixel image.

### GAN image generation

| Model | Workload and convolution coverage | Source capture |
|---|---|---|
| `stylegan2` | Synthesis network with weight-modulated convolution, transposed-convolution upsampling, and 1x1 RGB output heads. | NCHW, `float16`; 128 and 256 pixel outputs |

The source inference capture used batch 1. At batch 1, grouped-by-batch
modulated convolution reduces to a standard group-1 convolution. The corpus
still covers the transposed upsampling path, but it does not characterize the
batch-groups form used at larger batches.

### Video generation

| Model | Workload and convolution coverage | Source capture |
|---|---|---|
| `wan_vae` | Causal video VAE with dense 3x3x3 and pointwise 1x1x1 convolutions. It uses 8x spatial and 4x temporal compression. | Channels-last 3-D, `bfloat16`; latent `D x H x W` labels 9x40x40, 13x60x104, and 21x60x104 |

The model decodes long clips in temporal chunks. Different clip lengths can
therefore produce the same executable convolution geometry. Source training
capture was limited to the 9x40x40 case by the memory capacity of the capture
system.

### Video understanding and action recognition

| Model | Workload and convolution coverage | Source capture |
|---|---|---|
| `r2plus1d_18` | Factorized 3-D network that splits a spatiotemporal convolution into spatial 1x3x3 and temporal 3x1x1 operations. | NCHW 3-D, `float16`; 16x112x112 and 32x112x112 clips |
| `ir_csn_50` | Channel-separated network with depthwise 3x3x3 convolutions and 1x1x1 channel mixing. | NCHW 3-D, `float16`; 16x112x112 and 32x112x112 clips |
| `resnext3d_50` | 3-D ResNeXt network with cardinality-32 grouped 3x3x3 convolutions and 1x1x1 bottlenecks. | NCHW 3-D, `float16`; 16x112x112 and 32x112x112 clips |

The clip labels use `T x H x W`, where `T` is the frame count.

### Medical imaging

| Model | Workload and convolution coverage | Source capture |
|---|---|---|
| `nnunet3d` | Volumetric segmentation with dense 3x3x3, strided 3-D, and transposed 3-D convolutions. | NCHW 3-D, `float32`; cubic patches of 96, 128, and 160 |

The source used `float32` because its `InstanceNorm3d` path did not support
`float16`.

### Face recognition

| Model | Workload and convolution coverage | Source capture |
|---|---|---|
| `mobilefacenet` | Mobile face-embedding network with inverted residual blocks and GDConv. The final depthwise 7x7 kernel spans the complete feature map. | NCHW, `float16`; 112 pixel face crops |

### Speech and audio

The source models use 1-D convolutions. The converter represents them as 2-D
graphs with `H=1`.

| Model | Workload and convolution coverage | Source capture |
|---|---|---|
| `conformer` | Automatic speech recognition encoder with depthwise and pointwise 1-D convolutions inside an attention model. | NCHW, `bfloat16`; sequence lengths 200 and 400 |
| `hifigan` | Speech vocoder with dense, dilated, and transposed 1-D convolutions. | NCHW, `float16`; 200 and 400 mel-spectrogram frames |

### Transformer convolution probe

| Model | Workload and convolution coverage | Source capture |
|---|---|---|
| `vit_patch_embed` | Vision Transformer patch embedding. One 16x16 convolution with stride 16 converts non-overlapping image patches to 768 channels. | NHWC, `bfloat16`; 224 and 256 pixel inputs |

The corpus does not include complete transformer workloads. Attention and
matrix multiplication dominate those workloads. This model isolates the
convolution that vision transformers, detection transformers, and similar
architectures use to create patch tokens.

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

A transposed source operation can map to a backward-data graph. The
`from_transpose` metadata field preserves this source information.

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
    "from_transpose": ["false"],
    "convolution_type": ["pointwise"]
  }
}
```

The abbreviated example omits tensor and convolution parameters. The files in
the archive contain complete graph definitions.

## Metadata tags

Every metadata value is a sorted array of unique lowercase strings. The ten
bounded keys are:

- `operation`: `convolution`;
- `direction`: `fwd`, `dgrad`, or `wgrad`;
- `dtype`: `bfloat16`, `float16`, or `float32`;
- `convolution_type`: source `conv_type` classification;
- `layout`: `nchw` or `nhwc`;
- `model_family`: source workload family;
- `model`: source model name;
- `mode`: inference or training capture mode;
- `size`: source workload size label;
- `from_transpose`: `true` or `false`.

Deduplication unions `convolution_type` values from every source row for one
geometry. A graph can therefore belong to more than one convolution class.

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

## Source capture provenance and limits

The source collection used a PyTorch dispatch hook to record live
`aten::convolution` calls from model executions. It also recorded the
corresponding MIOpen driver geometry. It did not synthesize convolution
shapes. The converted archive is hardware-independent graph input, but the
source coverage reflects the system that performed the capture.

The source capture ran on a `gfx1151` RDNA 3.5 integrated GPU under Windows
Subsystem for Linux 2 (WSL2). The GPU shared approximately 46 GB of host
memory and used the Windows display-driver watchdog. These constraints caused
the following coverage limits:

- inference used batch 1;
- `retinanet_resnet50_fpn` and `maskrcnn_resnet50_fpn` training used batch 2;
- `ssd300_vgg16` and `yolo11` training used batch 4;
- `r2plus1d_18`, `ir_csn_50`, and `resnext3d_50` training used batch 2;
- `realesrgan` training used batch 2;
- `deeplabv3_resnet50` training at 1024 pixels used batch 4;
- `sd15_unet`, `sdxl_vae`, and `flux_vae` training at size 128 used batch 1;
- `sdxl_unet` and `sd15_unet` training used batch 2 at other captured sizes;
- `sdxl_unet` size 192 was not captured;
- `wan_vae` training was captured only at the 9x40x40 latent size;
- the grouped-by-batch form of `stylegan2` modulated convolution was not
  present at inference batch 1;
- large datacenter batch sizes were not captured;
- `int8` and `float8` convolution were not captured because eager PyTorch on
  this ROCm stack did not route them to MIOpen;
- `hunyuan_vae` had no captured source rows and is not in this archive.

The capture used `MIOPEN_FIND_MODE=fast`. This setting avoided the naive
reference kernel during MIOpen Find because a long reference submission could
trigger the Windows watchdog. The corpus therefore represents production
solver geometry, but it does not cover the naive reference path.

The source chose data types that match the executable model path on the
capture stack. The diffusion and VAE models used `bfloat16` because
`float16` overflowed. `retinanet_resnet50_fpn`,
`maskrcnn_resnet50_fpn`, and `ssd300_vgg16` used `float32` because their
roi_align, NMS, and box-coder operations had no `float16` path. `nnunet3d`
used `float32` because its `InstanceNorm3d` path did not support `float16`.

The archive does not contain model weights, datasets, non-convolution
operators, model accuracy results, or solver performance results. It does not
claim that every installed provider supports every graph. Use the archive to
measure convolution implementations on the target GPU. Do not treat its graph
count as a count of model layers or model invocations.

## Validation

The generated corpus passed these checks:

- 11,697 source rows resolve to 9,471 unique geometries;
- no two files have the same executable geometry identity;
- every filename and graph name uses an existing source shape ID;
- every graph has all ten metadata keys;
- every metadata value is a sorted array of unique lowercase strings;
- every graph has a nonempty `operation` tag;
- every graph has a valid operation node;
- every non-virtual tensor passes the repository tensor loader;
- every node type matches its archive directory.

Operation support depends on the installed hipDNN providers and target GPU. A valid graph can still have no applicable engine on a specific system.

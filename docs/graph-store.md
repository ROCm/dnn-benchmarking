# Graph store

The graph store holds the benchmark graphs of this project: one JSON object
for each graph, selected by the tags in the graph. The public store is:

```text
s3://therock-dvc/dnn-benchmarking-graphs/
https://therock-dvc.s3.amazonaws.com/dnn-benchmarking-graphs/<key>
```

Anyone can read it without credentials. Write access is restricted. The
store keeps the latest state only: there are no versions and no manifest.
To see the graphs, list the prefix and read the objects.

## Run graphs from the store

```bash
# All convolutions
dnn-benchmark -g s3://therock-dvc/dnn-benchmarking-graphs/ --tag operation=convolution

# The headline convolutions in bfloat16 or float16
dnn-benchmark -g s3://therock-dvc/dnn-benchmarking-graphs/ \
  --tag workload=headline --tag operation=convolution \
  --tag dtype=bfloat16 --tag dtype=float16
```

`--tag` terms of different keys must all match (AND). For one key, one
matching term is sufficient (OR). [usage.md](usage.md#graph-store) tells how
the tool keeps a local copy of the store.

## Rules

### 1. Object key

Each object has the key `<sha256>.json`, directly under the prefix:

1. Remove the top-level `name` and `metadata` from the graph.
2. Serialize the rest with
   `json.dumps(graph, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)`.
3. Compute the sha256 hex digest of the UTF-8 bytes.
4. Add `.json`.

`dnn_benchmarking.graph.store.geometry_key` implements the rule. Two graphs
with the same geometry thus have the same key, also when their names or tags
are different. A change to a name or a tag does not change the key.

### 2. Tags

- `metadata.<key>` is a sorted array of unique, lowercase tokens. A token
  matches `[a-z0-9][a-z0-9._+-]{0,63}`.
- `metadata.notes` is an object with free-form provenance. It is not a tag:
  `--tag` does not read it. `notes.sets` names the old `Workloads/` sets of
  a migrated graph.
- `name` is a label. It is not unique in the store.
- Each graph has exactly one `operation`, from its nodes, and one `dtype`,
  from its `io_data_type`.

### 3. Vocabulary

| Key | Values |
|---|---|
| `workload` | `headline`, `microbench`, `corpus`, `model`, `integration` |
| `source` | `bench_cases`, `synthetic`, `hipblaslt`, `cudnn_frontend`, `aiter`, `aotriton`, `conv_sweep`, `cudnn_attention_inference`, `cudnn_attention_training`, `cudnn_bench_moe`, `cudnn_gemm`, `hipkittens`, `pytorch`, `rocke`, `hf_config`, `mad`, `brad_corpus`, `sdpa_fwd_to_bwd`, `integration_tests` |
| `tier` | `quick`, `standard`, `full`, `comprehensive`. Only on `source=integration_tests` graphs. |
| `operation` | `convolution`, `sdpa`, `matmul`, `moe`, `norm`, `batchnorm`, `pointwise`, `reduction`, `resample`, `quantize` |
| `direction` | `fwd`, `bwd`, `dgrad`, `wgrad` |
| `dtype` | `float32`, `float16`, `bfloat16`, `fp8_e4m3` |
| `layout` | `nchw`, `nhwc`, `ncdhw`, `ndhwc`, `bshd`, `bhsd` |
| `mode` | `training`, `inference` |
| `phase` | `prefill`, `chunked_prefill`, `decode` |
| `model_family`, `model`, `use_case`, `capability`, `variant`, `priority`, `purpose`, `suite` | Open lowercase tokens |

`dnn_benchmarking.graph.store` holds the same lists (`CLOSED_TAGS`,
`OPEN_TAGS`). Put other keys in `metadata.notes`.

### 4. Upload

Use [`tools/upload_graphs.py`](../tools/upload_graphs.py) from an environment
with `dnn-benchmarking` installed and a recent AWS CLI v2 on `PATH`. Pass
tagged graph JSON files or directories containing `*.json` files.

```bash
# Preview against the public store without AWS credentials.
python tools/upload_graphs.py /path/to/graphs

# Write with a named AWS profile that has graph-store write access.
python tools/upload_graphs.py /path/to/graphs --profile graph-writer --apply
```

The script computes the object key from each graph, not its filename. It
validates the input and reports `create`, `merge`, or `skip` for each key.
Use `--store s3://BUCKET/PREFIX/` to select another store. A private store
also needs `--profile` for the preview.
The reader needs permission to get objects and list the destination prefix
so S3 can distinguish a missing key from denied access.

The script creates new objects with `If-None-Match: *`. For existing objects,
it keeps the stored name and notes, adds only missing tags, and writes with
`If-Match` using the ETag from the read. If another writer changes an object,
the script stops. Earlier writes remain in place; run the preview again
before you retry. The script does not remove tags or delete objects.

`tools/sdpa_fwd_to_bwd.py` writes tagged graph files into a local folder. Run
`dnn-benchmark -g <folder>` to check those graphs before you upload them.

## Old `Workloads/` sets

The `Workloads/` DVC tarballs became store objects in a one-time migration,
done outside this repository. Duplicate geometries became one object with
the union of their tags. Each old set is the result of one tag query:

| Old set (`Workloads/...`) | Tag query |
|---|---|
| `headline/attn` | `workload=headline operation=sdpa` |
| `headline/bnorm` | `workload=headline operation=batchnorm` |
| `headline/conv` | `workload=headline operation=convolution` |
| `headline/hipblaslt` | `workload=headline source=hipblaslt` |
| `headline/moe` | `workload=headline source=bench_cases operation=moe\|matmul` |
| `headline/norm` | `workload=headline operation=norm` |
| `microbench/<source>` | `workload=microbench source=<source>` |
| `microbench/rocke` | `workload=microbench source=rocke operation=convolution` |
| `models/<set>` | `workload=model source=<source> model=<model>` (table below) |
| `corpus/Convolution` | `workload=corpus operation=convolution` |
| `corpus/SDPA` | `workload=corpus operation=sdpa direction=fwd` |
| `corpus/SDPABackward` | `workload=corpus operation=sdpa direction=bwd` |

`a|b` is two terms of one key: `--tag operation=moe --tag operation=matmul`.

| Old set (`Workloads/models/...`) | `source` | `model` |
|---|---|---|
| `auto_regressive_dit` | `cudnn_attention_training` | `auto_regressive_dit` |
| `dsv3` | `cudnn_attention_training` | `deepseek_v3` |
| `gpt_oss` | `cudnn_attention_training` | `gpt_oss` |
| `kimiK26` | `cudnn_attention_training` | `kimi_k2_6` |
| `llama3.1` | `cudnn_attention_training` | `llama3_1` |
| `ltx2` | `cudnn_attention_training` | `ltx2` |
| `qwen35` | `cudnn_attention_training` | `qwen3_5` |
| `wan22_a14b` | `cudnn_attention_training` | `wan2_2_a14b` |
| `glm_5_2_moe` | `hf_config` | `glm5_2` |
| `qwen3_235b_a22b_moe` | `hf_config` | `qwen3_235b_a22b` |
| `qwen3_30b_a3b_moe` | `hf_config` | `qwen3_30b_a3b` |
| `qwen3_32b` | `hf_config` | `qwen3_32b` |
| `qwen3_8b` | `hf_config` | `qwen3_8b` |
| `mad_llama3_1_8b_train` | `mad` | `llama3_1_8b` |

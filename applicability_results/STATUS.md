# Applicability sweep — graph workloads

There are 34 workload tarballs total under `Workloads/` (`Workloads/headline/*.tar.gz`,
`Workloads/microbench/*.tar.gz`, and `Workloads/models/*.tar.gz`), all DVC-tracked. This ledger records, for every
workload, every graph inside it, and every engine hipDNN discovers as a candidate: did the engine get discovered,
and did the run succeed?

Tested at dnn-benchmarking commit `5c8e838`, `rocm-libraries` submodule pinned at `dcb84f88`.

## CSV schema
`asic,workload,graph_name,engine_id,engine_name,role,status,correctness_match,applicable,error_message,skip_reason`

`applicable=True` means: the hipDNN backend discovered this engine as a candidate for the graph AND the benchmark
run succeeded (status=="success") AND correctness (when checked) didn't fail. A graph with zero discovered engines
gets one row with `engine_name=NONE`, `status=no_engine_discovered`, `applicable=False`.

**Important caveat**: this ledger answers "does the engine run without error?", not "is its output numerically
correct?". Runs used the default (no `--validate pytorch`), so `correctness_match` is blank on every row.

## Layout
- `by_workload/<workload>.csv` — one CSV per workload (same schema).
- `by_workload/INDEX.csv` — per (workload, asic) summary: graph count, applicable/not-applicable row counts, which
  engines are applicable.

## Three fixes verified on both gfx942 and gfx950

**1. 1D depthwise-conv fix — CONFIRMED FIXED.** The `rocm-libraries` upstream fix landed and works. All three
`Conformer-L__bf16_ncl_dw_conv1d_k31_bf16` graphs (fwd, backward-dgrad, backward-wgrad) now pass via
`MIOPEN_ENGINE`/`MIOPEN_ENGINE_DETERMINISTIC` on both ASICs, in `conv.tar.gz`. The remaining 33/179 not-applicable
`conv.tar.gz` graphs are unrelated: 3D video-VAE convolutions (HunyuanVideo-VAE, Cosmos-Tokenizer, WAN-VAE,
Mochi-1 — the asymmetric-padding issue documented below) and SSM/Hyena causal conv1d/FFT ops (Mamba, Mamba2,
Jamba, Hyena) that no engine implements.

**2. Norm fp32-stat-tensor fix — landed, but NOT sufficient for applicability.** `norm.tar.gz` still shows
0/30 applicable on both ASICs. The fix removed the old dtype-mismatch rejection, but that just exposed the real,
deeper gap: no engine on either gfx942 or gfx950 actually implements RMSNorm/LayerNorm forward or backward today.
29/30 graphs now fail with `"No engine configurations available"` (previously they failed with a dtype-mismatch
message instead — same practical outcome, cleaner error). The 30th (GPT3 LayerNorm-backward) still fails a
separate graph-validation constraint (`"mean and scale must both be one-padded or both not"`). The fp32 fix was
necessary but not sufficient — an actual GPU engine implementation is still needed. `norm.tar.gz` is tracked in
`Workloads/headline/` ahead of that engine support landing, not because it has applicability signal today.

**3. Native MoE node fix — landed, failure mode changed, NOT fixed.** `moe.tar.gz`/`cudnn_bench_moe.tar.gz` now
correctly use `MoeGroupedMatmulAttributes` instead of the disconnected two-GEMM approximation — the old
"disconnected components" error (65 graphs) is gone entirely. But no GPU engine (`hipblaslt-provider`,
`hip-kernel-provider`, `miopen-provider`) implements this node yet, so most graphs now fail differently: 89/121
`moe.tar.gz` graphs (and all 26 `cudnn_bench_moe.tar.gz` graphs) fail with `"Failed to create backend graph
descriptor from JSON data"`, and 24/121 fail graph validation (`"grad_out must have at least 3 dimensions"`). Only
8/121 `moe.tar.gz` graphs pass — all router-gate backward GEMMs (dgrad/wgrad for deepseek-r1, qwen3-235b-a22b,
mixtral-8x7b, kimi-k2) that happen to still route through plain `HIPBLASLT_ENGINE` GEMM, unrelated to the new node.
This is the same conclusion as before: the representation is now architecturally correct, but the GPU engine work
is still outstanding.

## Per-workload results

| workload | graphs | applicable engine(s) | gfx942 applicable/not | gfx950 applicable/not |
|---|---|---|---|---|
| moe | 121 | HIPBLASLT_ENGINE | 8 / 113 | 8 / 113 |
| attn | 217 | ASM_SDPA_ENGINE | 5 / 212 | 34 / 183 |
| norm | 30 | none | 0 / 30 | 0 / 30 |
| cudnn_attention_inference | 428 | ASM_SDPA_ENGINE | 2 / 426 | 2 / 426 |
| cudnn_attention_training | 141 | none (gfx942) / ASM_SDPA_ENGINE (gfx950) | 0 / 141 | 20 / 121 |
| cudnn_bench_moe | 26 | none | 0 / 26 | 0 / 26 |
| cudnn_gemm | 5 | HIPBLASLT_ENGINE | 2 / 3 | 2 / 3 |
| conv | 179 | MIOPEN_ENGINE / MIOPEN_ENGINE_DETERMINISTIC | 292 rows applicable / 33 rows not (325 total rows; 146 unique graphs pass x2 engine rows) | same, 146/179 unique graphs pass |
| hipblaslt | 578 | HIPBLASLT_ENGINE | 578 / 0 | 578 / 0 |

Note: `attn` and `cudnn_attention_training` show a real gfx942-vs-gfx950 divergence (34 vs 5, and 20 vs 0
applicable respectively) — worth a follow-up look at which specific shapes ASM_SDPA_ENGINE's kernel catalog
covers on gfx950 but not gfx942, or vice versa.

Bnorm and every other workload not listed above are unchanged from the prior sweep; see `by_workload/*.csv`
and `by_workload/INDEX.csv` for full per-workload/per-ASIC numbers.

## Known issues (current, at the tested commit)
- **Asymmetric-padding 3D convs**: `HunyuanVideo-VAE`/`WAN-VAE` decoder-upsample 3D convs use `pre_padding !=
  post_padding`; MIOpen's C API only accepts symmetric padding (`MiopenConvDescriptor.cpp` hard-rejects the
  mismatch). Real API-level gap, not addressed by the 1D-conv fix.
- **HIP-700 hipblaslt cleanup crash**: running certain workloads back-to-back in the same process can trigger a
  HIP error 700 (illegal memory access) inside hipblaslt's cleanup path at process exit, after the JSON is
  already written. Worked around by isolating each workload in its own subprocess/SLURM job. Standalone
  `hipblaslt.tar.gz` runs cleanly. Not fixed, out of scope for this sweep.
- **ASM_SDPA_ENGINE coverage gaps** (unchanged from before): no fp16 (only bf16/fp8), no `generate_stats` output
  (rejects most training-paired forward graphs), no explicit attn_mask/alibi/padding-mask/paged-KV, and a fixed
  prebuilt-kernel catalog gated on (dtype, head_dim, mask_type) — causal/prefill and head_dim=256 configs commonly
  miss the catalog.
- **No RMSNorm/LayerNorm engine**: see fix #2 above — `norm.tar.gz` is 0/30 applicable on both ASICs.
- **No MoeGroupedMatmulAttributes engine**: see fix #3 above — `moe.tar.gz`/`cudnn_bench_moe.tar.gz` mostly fail
  to build a backend graph.

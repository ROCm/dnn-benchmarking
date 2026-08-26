# Applicability sweep — graph workloads on users/sareeder/collect-brad-david-benchmark-graphs

Branch: `users/sareeder/collect-brad-david-benchmark-graphs` (dnn-benchmarking repo, ROCm/dnn-benchmarking on GitHub).

There are 34 workload tarballs total under `Workloads/` (`Workloads/microbench/*.tar.gz` and
`Workloads/models/*.tar.gz`), all DVC-tracked. This ledger records, for every workload, every graph inside it, and
every engine hipDNN discovers as a candidate: did the engine get discovered, and did the run succeed?

## CSV schema
`asic,workload,graph_name,engine_id,engine_name,role,status,correctness_match,applicable,error_message,skip_reason`

`applicable=True` means: the hipDNN backend discovered this engine as a candidate for the graph AND the benchmark
run succeeded (status=="success") AND correctness (when checked) didn't fail. A graph with zero discovered engines
gets one row with `engine_name=NONE`, `status=no_engine_discovered`, `applicable=False`.

**Important caveat**: this ledger answers "does the engine run without error?", not "is its output numerically
correct?". Runs used the default (no `--validate pytorch`), so `correctness_match` is blank on every row. A
correctness-checked pass is a separate, heavier follow-up (needs a PyTorch reference build) — not done here.

## Layout
- `applicability_combined.csv` — every row, every workload, every ASIC tested so far.
- `by_workload/<workload>.csv` — one CSV per workload (same schema), easier to diff/track per workload.
- `by_workload/INDEX.csv` — per (workload, asic) summary: graph count, applicable/not-applicable row counts, which
  engines are applicable.

## Status: gfx942 (MI300X) — COMPLETE, all 34 workloads
All 34 workload tarballs are benchmarked and in this ledger, including `aiter.tar.gz` (4988 graphs, the largest
tarball — timed out at 3600s on the first attempt, completed in 1912s / ~32 min on a retry with a longer budget).

Setup: container `hipdnn_latest_gfx942.sqsh`; `--reuse-artifacts --rocm-prefix /opt/rocm` failed (no hipDNN CMake
configs baked into the container's `/opt/rocm`); built hipDNN + hipblaslt-provider + hip-kernel-provider + MIOpen
provider from source via `setup_env.py` once, then reused that venv (`.workspace/.venv`) for every subsequent job —
no rebuild needed per workload.

Per-workload results (gfx942):

| workload | graphs | applicable engine(s) | applicable rows | not-applicable rows |
|---|---|---|---|---|
| bench_cases_moe | 121 | HIPBLASLT_ENGINE | 56 | 65 |
| sdpa_rocke | 157 | ASM_SDPA_ENGINE | 33 | 124 |
| cudnn_frontend_full_sdpa | 105 | none | 0 | 105 |
| cudnn_frontend_full_norm | 30 | none | 0 | 30 |
| cudnn_frontend_attention_inference | 428 | ASM_SDPA_ENGINE | 2 | 426 |
| cudnn_frontend_attention_training_v2 | 141 | none | 0 | 141 |
| cudnn_frontend_bench_moe | 12 | HIPBLASLT_ENGINE | 12 | 0 |
| bench_cases | 179 | MIOPEN_ENGINE / MIOPEN_ENGINE_DETERMINISTIC | 270 | 44 |
| hipblaslt | 578 | HIPBLASLT_ENGINE | 578 | 0 |
| aiter | 4988 | HIPBLASLT_ENGINE / ASM_SDPA_ENGINE | 1102 | 3886 |
| aotriton | 48 | none | 0 | 48 |
| bnorm_backward | 8 | HIP_MLOPS_ENGINE / MIOPEN_ENGINE | 16 | 0 |
| bnorm_fwd | 8 | HIP_MLOPS_ENGINE / MIOPEN_ENGINE | 16 | 0 |
| conv_dgrad | 133 | MIOPEN_ENGINE / MIOPEN_ENGINE_DETERMINISTIC | 266 | 0 |
| conv_fwd | 150 | MIOPEN_ENGINE / MIOPEN_ENGINE_DETERMINISTIC | 300 | 0 |
| conv_wgrad | 146 | MIOPEN_ENGINE / MIOPEN_ENGINE_DETERMINISTIC | 292 | 0 |
| cudnn_frontend | 53 | HIPBLASLT_ENGINE | 18 | 35 |
| hipkittens | 73 | ASM_SDPA_ENGINE / HIPBLASLT_ENGINE | 37 | 36 |
| pytorch | 1200 | ASM_SDPA_ENGINE / HIP_MLOPS_ENGINE / HIPBLASLT_ENGINE / MIOPEN_ENGINE / MIOPEN_ENGINE_DETERMINISTIC | 762 | 618 |
| rocke | 32 | none | 0 | 32 |
| auto_regressive_dit | 5 | ASM_SDPA_ENGINE | 5 | 0 |
| dsv3 | 10 | ASM_SDPA_ENGINE | 5 | 5 |
| glm_5_2_moe | 20 | HIPBLASLT_ENGINE | 11 | 9 |
| gpt_oss | 5 | none | 0 | 5 |
| kimiK26 | 10 | ASM_SDPA_ENGINE | 5 | 5 |
| llama3.1 | 10 | ASM_SDPA_ENGINE | 5 | 5 |
| ltx2 | 5 | ASM_SDPA_ENGINE | 5 | 0 |
| mad_llama3_1_8b_train | 8 | HIPBLASLT_ENGINE | 5 | 3 |
| qwen35 | 5 | none | 0 | 5 |
| qwen3_235b_a22b_moe | 15 | HIPBLASLT_ENGINE | 7 | 8 |
| qwen3_30b_a3b_moe | 15 | HIPBLASLT_ENGINE | 7 | 8 |
| qwen3_32b | 14 | HIPBLASLT_ENGINE | 6 | 8 |
| qwen3_8b | 13 | HIPBLASLT_ENGINE | 5 | 8 |
| wan22_a14b | 5 | ASM_SDPA_ENGINE | 5 | 0 |

**Known issue found during the sweep**: running `cudnn_frontend_full_*`/`cudnn_frontend_attention_*` workloads
back-to-back in the same process can trigger a HIP error 700 (illegal memory access) inside hipblaslt's
cleanup/destructor path (`hipblaslt.cpp:187`) at the end of a suite, after its JSON is already written, wedging
the GPU for the next invocation in the same process. Worked around by isolating each workload in its own SLURM
job / subprocess. Standalone `hipblaslt.tar.gz` (578 graphs) did NOT reproduce this crash. Worth a real bug
report against hipDNN/hipblaslt-provider — not fixed here, out of scope for this sweep.

## Status: gfx950 (MI350X/MI355X) — BLOCKED on cluster capacity, all 34 workloads queued in one job
Every gfx950 node cluster-wide (MI350X 1-GPU nodes and MI355X 8-GPU nodes, both `gpu` and `shard` GRES) is 100%
allocated by long-running jobs (7-day/28-day/30-day/365-day reservations). The federation sibling cluster
`alola-blr` has no gfx950 nodes at all.

SLURM job `67845858` (`workspaces/gfx950/run_sweep_gfx950.sh`) is queued on partition `defq`, requesting
`gres/gpu:gfx950-mi350x=1`, reason `Priority`, time limit 10:00:00, submitted 2026-08-26T16:56:48. All 34
workload tarballs are already DVC-pulled into `workspaces/gfx950/Workloads/`. The script runs every `dnn-benchmark`
invocation in its own subprocess (works around the HIP-700 issue) and parses everything directly into
`applicability_gfx950_all34.csv` when it finishes.

**No action needed** — it's a standard `sbatch` submission that runs unattended once SLURM schedules it. When it
completes:
```bash
cd /home/AMD/sareeder/ROCm-workspace/workspaces
{ head -1 applicability_gfx942.csv; tail -n +2 applicability_combined.csv; tail -n +2 applicability_gfx950_all34.csv; } > applicability_combined.csv.new
mv applicability_combined.csv.new applicability_combined.csv
```
(then re-split `by_workload/*.csv` and `INDEX.csv` from the refreshed combined CSV)
- Check status: `squeue -j 67845858` or `sacct -j 67845858`.
- Check output: `workspaces/gfx950/results/*.json` and `workspaces/applicability_gfx950_all34.csv`.

# Applicability sweep — graph workloads on users/sareeder/collect-brad-david-benchmark-graphs

Branch: `users/sareeder/collect-brad-david-benchmark-graphs` (dnn-benchmarking repo, ROCm/dnn-benchmarking on GitHub).
Commits unique to this branch vs `origin/main` (merge-base `6d485c5`): `5a7005b`, `812b1ef`.

7 new workload tarballs added by those commits (all under `Workloads/microbench/`):
- `bench_cases_moe.tar.gz` — 121 graphs, MoE fwd/dgrad/wgrad (router-gate GEMM + one expert FFN approximation)
- `sdpa_rocke.tar.gz` — 157 graphs, rocKE attention shape sweep (14 models)
- `cudnn_frontend_full_sdpa.tar.gz` — 105 graphs, cuDNN-frontend old SDPA training benchmark (8 models)
- `cudnn_frontend_full_norm.tar.gz` — 30 graphs, cuDNN-frontend norm benchmark (RMSNorm/LayerNorm, 11 configs)
- `cudnn_frontend_attention_inference.tar.gz` — 428 graphs, cuDNN-frontend inference-phase SDPA (7 configs)
- `cudnn_frontend_attention_training_v2.tar.gz` — 141 graphs, cuDNN-frontend training SDPA v2 (10 configs)
- `cudnn_frontend_bench_moe.tar.gz` — 12 graphs, cuDNN's own MoE dense-GEMM benchmark

Plus 2 pre-existing workloads added to the sweep on request:
- `bench_cases.tar.gz` — 179 graphs, conv-heavy real-model workload (MIOPEN_ENGINE / MIOPEN_ENGINE_DETERMINISTIC)
- `hipblaslt.tar.gz` — 578 graphs, GEMM-heavy workload (HIPBLASLT_ENGINE)

9 workloads total, 1671 graphs total.

## Workspaces
- `workspaces/gfx942/` — git worktree (detached HEAD at `812b1ef`), used to test on MI300X.
- `workspaces/gfx950/` — git worktree (detached HEAD at `812b1ef`), used to test on MI350X/MI355X.
- `workspaces/parse_applicability.py` — shared JSON→CSV parser (schema documented in its docstring).

## CSV schema
`asic,workload,graph_name,engine_id,engine_name,role,status,correctness_match,applicable,error_message,skip_reason`

`applicable=True` means: the hipDNN backend discovered this engine as a candidate for the graph AND the benchmark run
succeeded (status=="success") AND correctness (when checked) didn't fail. A graph with zero discovered engines gets
one row with `engine_name=NONE`, `status=no_engine_discovered`, `applicable=False`.

## Status: gfx942 — COMPLETE (all 9 workloads)
- CSV: `workspaces/applicability_gfx942.csv` (1886 data rows).
- Raw SuiteResult JSONs: `workspaces/gfx942/results/*.json`.
- Setup: container `hipdnn_latest_gfx942.sqsh`; `--reuse-artifacts --rocm-prefix /opt/rocm` failed (no hipDNN CMake
  configs baked into the container's `/opt/rocm`); fell back to the full `setup_env.py` build (hipDNN + hipblaslt-provider
  + hip-kernel-provider built from the `rocm-libraries` submodule against the bundled rocm-sdk-devel wheel). That
  built venv (`.workspace/.venv`) was reused for the follow-up bench_cases/hipblaslt run — no rebuild needed.
- Result summary (1886 graph×engine rows, 951 applicable / 935 not applicable):

  | workload | graphs | applicable | active engine(s) |
  |---|---|---|---|
  | bench_cases_moe | 121 | 56 | HIPBLASLT_ENGINE |
  | sdpa_rocke | 157 | 33 | ASM_SDPA_ENGINE |
  | cudnn_frontend_full_sdpa | 105 | 0 | none |
  | cudnn_frontend_full_norm | 30 | 0 | none |
  | cudnn_frontend_attention_inference | 428 | 2 | ASM_SDPA_ENGINE |
  | cudnn_frontend_attention_training_v2 | 141 | 0 | none |
  | cudnn_frontend_bench_moe | 12 | 12 | HIPBLASLT_ENGINE |
  | bench_cases | 179 | 135* | MIOPEN_ENGINE / MIOPEN_ENGINE_DETERMINISTIC |
  | hipblaslt | 578 | 578 | HIPBLASLT_ENGINE |

  *bench_cases: 314 graph×engine rows total (most graphs tried against 2 MIOpen engine variants), 270 applicable
  rows / 44 not-applicable rows; the 44 not-applicable rows are Conformer-L depthwise-conv-1d graphs with no
  hipDNN engine configuration available.

- **Known issue found during the sweep**: each `cudnn_frontend_full_*`/`cudnn_frontend_attention_*` workload run
  triggers a HIP error 700 (illegal memory access) inside hipblaslt's cleanup/destructor path
  (`hipblaslt.cpp:187`) at the *end* of the suite, after the JSON is already written. It doesn't corrupt that
  run's JSON, but it leaves the GPU context wedged and hangs the *next* `dnn-benchmark` invocation on the same
  process/node. Worked around by running each workload as its own SLURM job. The standalone `hipblaslt.tar.gz`
  run (578 GEMM graphs, run last, in isolation) did NOT reproduce this crash — it completed cleanly with exit 0 —
  so the crash looks specific to running hipblaslt cleanup back-to-back with cuDNN-frontend SDPA/norm graphs in
  the same process, not to hipblaslt in isolation. Worth a real bug report against hipDNN/hipblaslt-provider —
  not fixed here, out of scope for this sweep.

## Status: gfx950 — BLOCKED on cluster capacity, job queued (all 9 workloads)
- Every gfx950 node cluster-wide (MI350X 1-GPU nodes and MI355X 8-GPU nodes, both `gpu` and `shard` GRES) is
  100% allocated by long-running jobs (7-day/28-day/30-day/365-day reservations). The federation sibling
  cluster `alola-blr` has no gfx950 nodes at all.
- SLURM job `67845561` (`workspaces/gfx950/run_sweep_gfx950.sh`, superseded the earlier `67845319` which was
  cancelled so the script could be updated — SLURM does not re-read a pending job's script from disk) is queued
  on partition `defq`, requesting `gres/gpu:gfx950-mi350x=1`, reason `Priority`, estimated start
  `2026-08-28T21:47:28` (backfill scheduler estimate, ~2.5 days out as of 2026-08-26), time limit 6:00:00.
- The DVC pull for all 9 tarballs (including `bench_cases.tar.gz` and `hipblaslt.tar.gz`) already completed
  inside `workspaces/gfx950/Workloads/microbench/`.
- The job script already encodes the same fast-path/fallback setup strategy validated on gfx942 (try
  `--reuse-artifacts --rocm-prefix /opt/rocm` first, fall back to the full build without `set -e` killing the
  fallback), runs all 9 workloads, and parses directly into `applicability_gfx950.csv`.
- **No action needed** — the job is a standard `sbatch` submission and will run unattended once SLURM schedules
  it, using the same mounted `workspaces/gfx950/` checkout. When it completes:
  1. Check `workspaces/applicability_gfx950.csv` exists.
  2. Regenerate the combined CSV:
     ```bash
     cd /home/AMD/sareeder/ROCm-workspace/workspaces
     { head -1 applicability_gfx942.csv; tail -n +2 applicability_gfx942.csv; tail -n +2 applicability_gfx950.csv; } > applicability_combined.csv
     ```
  - To check status: `squeue -j 67845561` or `sacct -j 67845561`.
  - To check output once it lands: `workspaces/gfx950/results/*.json` and `workspaces/applicability_gfx950.csv`.

## Combined CSV
`workspaces/applicability_combined.csv` — currently gfx942-only (1887 lines incl. header); re-run the merge command
above once gfx950 lands to add its rows.

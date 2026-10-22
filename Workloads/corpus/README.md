# hipDNN Workload Corpus

The `Workloads/corpus/` directory stores curated reference sets of executable
hipDNN graphs.

## Goal

The goal is to represent all hipDNN workloads with the smallest practical
number of graphs. This goal is an admission rule for the folder. It is not a
claim about the current coverage.

Today, the folder contains one collection. `Convolution/` contains
9,471 unique convolution geometries from 31 models.

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

## Current collection

| Collection | Scope | Documentation |
|---|---|---|
| `Convolution/` | 9,471 unique convolution geometries from 31 models | [Convolution corpus guide](Convolution/Explanation.md) |

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

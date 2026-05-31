# Dissertation Canon

This repository measures research-policy misalignment in AI-for-SDG discourse along two separate dimensions:
- `coverage gap`: which SDGs each corpus emphasizes
- `semantic gap`: how differently research and policy discuss the same SDG

The submission surface is intentionally narrow:
- one canonical analysis package under `outputs/`
- one canonical PDF at `outputs/dissertation.pdf`
- one entrypoint at `main.py`

## Main Thesis

Topical overlap is not enough to claim alignment. Research and policy can reference the same SDG while still allocating attention differently and framing the goal differently.

## Canonical Commands

Show repo status only:

```bash
python main.py
```

Rebuild the canonical analysis package and PDF from existing `data/`:

```bash
python main.py --warm-replay --overwrite
```

Build only the canonical PDF from existing canonical tables and figures:

```bash
python main.py --build-pdf --overwrite
```

Run the full active pipeline facade from fetch through PDF:

```bash
python main.py --full-pipeline --overwrite --device cuda --batch-size 256 --local-files-only
```

`--full-pipeline` requires network access, OpenAlex credentials, and the upstream fetch prerequisites expected by the active fetch scripts. It is materially heavier than warm replay.

Important behavior:
- `main.py` with no flags is read-only and prints status
- mutation requires an explicit action flag
- if canonical artifacts already exist, reruns fail closed unless `--overwrite` is supplied
- `--output-dir` can redirect the canonical output root, but the default contract is `outputs/`

## Reproducibility Contract

Warm replay is the primary reproducibility target.

Assumed existing inputs:
- hydrated `data/0_raw/` through `data/3_scored/`
- policy embeddings in `data/2_embedded/`
- research shard embeddings in `data/2_embedded/research_shards/`
- paper score shards in `data/3_scored/paper_scores_shards/`

`data/` and `outputs/` are intentionally not tracked in Git. A stranger can verify pipeline logic from source, but must hydrate `data/` before replaying the canon.

## Canonical Outputs

Root artifacts:
- `outputs/validation_results.json`
- `outputs/confusion_matrix.csv`
- `outputs/centroid_similarity_matrix.csv`
- `outputs/coverage_gap.json`
- `outputs/coverage_gap_raw.json`
- `outputs/semantic_gap.json`
- `outputs/semantic_gap_sensitivity.json`
- `outputs/h25_correlation.json`
- `outputs/h25_scatter.csv`
- `outputs/dissertation.pdf`

Table artifacts:
- `outputs/tables/num_validation.tex`
- `outputs/tables/tab_validation.tex`
- `outputs/tables/num_coverage.tex`
- `outputs/tables/tab_coverage.tex`
- `outputs/tables/num_semantic.tex`
- `outputs/tables/tab_semgap.tex`
- `outputs/tables/num_h25.tex`
- `outputs/tables/tab_h25.tex`

Figure artifacts:
- `outputs/figures/fig1_coverage_profiles.pdf`
- `outputs/figures/fig1_coverage_profiles.png`
- `outputs/figures/fig2_semantic_gap.pdf`
- `outputs/figures/fig2_semantic_gap.png`
- `outputs/figures/fig3_coverage_semantic_scatter.pdf`
- `outputs/figures/fig3_coverage_semantic_scatter.png`

## Active Pipeline

Policy / benchmark side:
1. fetch policy, SDGi, UNGDC, OSDG, and benchmark sources
2. preprocess and merge the policy corpus
3. embed `policy`, `osdg`, and `benchmark`
4. build SDG centroids
5. validate centroids against the benchmark

Research side:
1. fetch OpenAlex works
2. preprocess into resume-safe shards
3. embed paper shards
4. score paper shards against SDG centroids
5. rebuild research centroids from the full scored shard set

Downstream analysis:
1. score the active policy corpus against SDG and research centroids
2. compute coverage gap
3. compute semantic gap with chunk-cap sensitivity checks
4. compute H25/H26 interaction outputs
5. generate figures
6. build the dissertation PDF from canonical tables and figures

## Robustness and Validation

The active canon includes these safeguards:
- centroid validation on the expert-labelled benchmark (`macro-F1` reported before downstream use)
- document-weighted policy coverage, so long policy reports do not dominate by chunk count alone
- semantic-gap chunk-cap sensitivity at 20, 50, and 100 chunks per document
- explicit SDG reliability flags when clusters are too small
- A15 calibration check comparing policy-vs-OSDG and paper-vs-OSDG top scores
- SDG 4 caveat carried into the manuscript where learning vocabulary may inflate education assignments
- hard protection against partial shard runs overwriting canonical research centroids unless explicitly allowed

## Repo Layout

Active source:
- `main.py`
- `code/0_fetch/`
- `code/1_preprocess/`
- `code/2_embed/`
- `code/3_main_analysis/`
- `code/4_visualization/`
- `code/shared_utils.py`
- `writing/dissertation.tex`
- `writing/references.bib`
- `writing/build_pdf.sh`

Working notes kept for the active thesis:
- `notes/ASSUMPTIONS.md`
- `notes/HYPOTHESES.md`
- `notes/LIT_REVIEW_INSIGHTS.md`

## Environment

Recommended setup:

```bash
conda env create -f conda-env-dissertation.yml
conda activate dissertation
pip install -r requirements.txt
```

Lock snapshots retained for auditability:
- `requirements.lock.txt`
- `conda-explicit-dissertation.txt`

## What Was Removed

This cleanup intentionally removed the old multi-run output layout, versioned dissertation builds, legacy manuscript dependencies, and stale prototype notes that contradicted the active pipeline.

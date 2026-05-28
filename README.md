# Dissertation Pipeline (Minimal Core)

This repository is now intentionally simplified to the minimum active workflow:

1. Fetch data
2. Embed / score
3. Analyze and visualize

Legacy/experimental scripts were moved out of the active surface to `archive/legacy/`.

## Active Script Surface

Active scripts live in `code/` and are the only supported path.

### Fetch
- `fetch_openalex.py` (research corpus)
- `fetch_osdg.py` (OSDG centroid source)
- `fetch_sdg_benchmark.py` (benchmark centroid source + SDG17 support)
- `fetch_un_sdg.py`
- `fetch_policy_expanded.py`
- `fetch_policy_v3.py`
- `fetch_sdgi_corpus.py`
- `fetch_ungdc.py`

### Build / Prepare
- `preprocess_papers_streaming.py`
- `preprocess_policy.py`
- `integrate_sdgi.py`
- `filter_ungdc_sdg.py`
- `build_policy_corpus.py`
- `preprocess_osdg.py`
- `preprocess_sdg_benchmark.py`

### Embed / Score
- `embeddings.py` (policy + labelled corpora)
- `sdg_centroids.py`
- `validate_centroids.py`
- `alignment_score.py` (legacy flat-matrix scoring path)
- `embed_paper_shards.py` (full research corpus)
- `score_paper_shards.py` (full research corpus)
- `run_full_corpus_pipeline.py` (orchestration)
- `run_subset_analysis.py` (small fast reruns)

### Analyze / Visualize
- `coverage_gap.py`
- `semantic_gap.py`
- `coverage_semantic_interaction.py`
- `plot_figures.py`
- `revisualize_full_corpus.py` (run-local bridge for full-corpus analysis/plots)

## Quick Start (Canonical)

Install deps:

```bash
pip install -r requirements.txt
```

### A) Build policy + centroid side (if rebuilding from scratch)

```bash
python code/fetch_un_sdg.py
python code/fetch_policy_expanded.py
python code/fetch_policy_v3.py
python code/fetch_sdgi_corpus.py
python code/fetch_ungdc.py

python code/preprocess_policy.py
python code/integrate_sdgi.py
python code/filter_ungdc_sdg.py
python code/build_policy_corpus.py

python code/fetch_osdg.py
python code/fetch_sdg_benchmark.py
python code/preprocess_osdg.py
python code/preprocess_sdg_benchmark.py

python code/embeddings.py
python code/sdg_centroids.py
python code/validate_centroids.py
```

### B) Build full research corpus (resume-safe shard path)

```bash
python code/fetch_openalex.py
python code/run_full_corpus_pipeline.py --device cuda --batch-size 256 --local-files-only
```

### C) Analyze + visualize full corpus (run-local outputs)

```bash
python code/revisualize_full_corpus.py --python /home/manh/miniforge3/envs/dissertation/bin/python
```

Outputs are written under:
- `outputs/runs/full_corpus_viz_*/workspace/data/*`
- `outputs/runs/full_corpus_viz_*/workspace/writing/figures/*`

## Data/Label Interpretation

- OpenAlex SDG tags are used as **retrieval filters**.
- Final analytical SDG assignment is **centroid-based** (nearest-score hard assignment),
  not direct OpenAlex SDG label assignment.

## Legacy Scripts

Anything not required for the core path has been moved to:
- `archive/legacy/code/`

See `archive/legacy/README.md` for the archived inventory.

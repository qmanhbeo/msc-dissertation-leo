# Dissertation Pipeline (Active Core)

This repository implements a reproducible end-to-end methodology for evaluating
alignment between AI-for-SDG research and SDG policy discourse.

## Goal

Measure two different misalignment dimensions:
- **Coverage Gap**: attention mismatch (how much each SDG is emphasized)
- **Semantic Gap**: framing mismatch (how differently each SDG is discussed)

## Pipeline Overview

```mermaid
flowchart TD
  OA[OpenAlex Research Fetch] --> RS[Streaming Clean + Shard]
  OSDG[OSDG + SDG Benchmark] --> CEN[Build + Validate SDG Centroids]
  POL[UN/Policy/SDGi/UNGDC Fetch] --> PB[Policy Build + Dedup]

  RS --> EMBR[Embed Paper Shards]
  EMBR --> SCR[Score Paper Shards vs SDG Centroids]
  CEN --> SCR
  PB --> EMP[Embed Policy Corpus]
  EMP --> AN[Coverage + Semantic + Interaction Analysis]
  SCR --> AN
  AN --> FIG[Final Figures]
```

## Active Scripts

### Fetch
- `fetch_openalex.py`
- `fetch_osdg.py`
- `fetch_sdg_benchmark.py`
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
- `embeddings.py`
- `sdg_centroids.py`
- `validate_centroids.py`
- `alignment_core.py` (shared centroid/scoring helpers)
- `embed_paper_shards.py`
- `score_paper_shards.py`
- `shard_pipeline_utils.py`
- `run_full_corpus_pipeline.py`
- `run_subset_analysis.py`

### Analyze / Visualize
- `coverage_gap.py`
- `semantic_gap.py`
- `coverage_semantic_interaction.py`
- `plot_figures.py`
- `revisualize_full_corpus.py`

### Ops
- `backup_data_snapshot.py`

## Canonical Run Path

Install dependencies:

```bash
pip install -r requirements.txt
```

Build policy + centroid side:

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

Build full research corpus (resume-safe shard flow):

```bash
python code/fetch_openalex.py
python code/run_full_corpus_pipeline.py --device cuda --batch-size 256 --local-files-only
```

Run analysis + figures (run-local outputs):

```bash
python code/revisualize_full_corpus.py --python /home/manh/miniforge3/envs/dissertation/bin/python
```

Outputs:
- `outputs/runs/full_corpus_viz_*/workspace/data/*`
- `outputs/runs/full_corpus_viz_*/workspace/writing/figures/*`

## Policy Corpus Taxonomy (Reviewer-Facing)

| Source Script | Corpus Role | Adds What Existing Sources Lack | Overlap Risk | Where Controlled |
|---|---|---|---|---|
| `fetch_un_sdg.py` | Core policy baseline | Canonical UN SDG/AI docs | Medium | `build_policy_corpus.py` exact-text dedupe |
| `fetch_policy_expanded.py` | Additive policy breadth | Additional multilateral/national strategy texts | Medium | `build_policy_corpus.py` |
| `fetch_policy_v3.py` | Additive long-tail policy docs | Broader document set beyond curated core | High | `build_policy_corpus.py` |
| `fetch_sdgi_corpus.py` + `integrate_sdgi.py` | Government reporting corpus | VNR/VLR implementation language | Low-Medium | `build_policy_corpus.py` |
| `fetch_ungdc.py` + `filter_ungdc_sdg.py` | Diplomatic discourse layer | UN General Debate SDG-relevant passages | Low-Medium | `build_policy_corpus.py` |

Deduplication and merge boundary is explicit: `build_policy_corpus.py` performs
normalized exact-text dedupe and emits the unified policy corpus.

## Method Definitions

Let `R_s` = research share for SDG `s`, `P_s` = policy share for SDG `s`.

- **Coverage Gap (per SDG)**: `|R_s - P_s|`
- **Total Coverage Gap**: `sum_s |R_s - P_s|`
- **Semantic Similarity (per SDG)**: cosine between research and policy sub-centroids within SDG `s`
- **Semantic Gap (per SDG)**: `1 - semantic_similarity_s`

Interpretation:
- Coverage gap answers **“who pays attention where?”**
- Semantic gap answers **“when both attend, do they mean the same thing?”**

## Centroid Validation Transparency

`validate_centroids.py` is the measurement-instrument quality gate.

It reports:
- nearest-centroid classification accuracy on benchmark texts
- macro-F1 (primary uncontaminated SDG1–16 evaluation)
- per-SDG F1
- centroid-to-centroid similarity matrix
- PASS/WARN/FAIL flag based on pre-declared thresholds

This validation section should be cited directly in methodology/results chapters.

## Label Interpretation

- OpenAlex SDG tags are used as **retrieval filters** for candidate papers.
- Final SDG assignment used in analysis is **centroid-score argmax**, not direct OpenAlex tag reuse.

## Legacy Surface

Deprecated/experimental scripts are archived under:
- `archive/legacy/code/`

See `archive/legacy/README.md` for inventory and rationale.

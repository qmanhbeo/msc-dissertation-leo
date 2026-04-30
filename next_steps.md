# Next Steps

Last updated: 2026-04-30

## Current State

- OpenAlex fetch is complete as of 2026-04-29 23:45.
- Raw OpenAlex research universe:
  - `data/openalex/papers.jsonl`
  - 2,763,579 unique papers
  - about 7.1 GiB JSONL
  - all 68 queries done in `data/openalex/progress.json`
- Downstream research artifacts are still from the earlier 6,172-paper snapshot:
  - `data/embeddings/papers.npy`
  - `data/paper_scores.npy`
  - `data/research_centroids.npy`
  - generated tables, figures, and dissertation text values
- Policy-side artifacts can stay fixed unless the policy corpus changes:
  - `data/policy_all/policy_chunks_extended.jsonl`
  - `data/embeddings/policy.npy`
  - `data/policy_scores.npy`
- SDG measurement instrument can stay fixed:
  - `data/sdg_centroids.npy`
  - `data/sdg_centroid_meta.json`
  - `data/validation_results.json`

## Methodology Position To Preserve

- OpenAlex native SDG tags define the candidate research universe, not final SDG labels.
- Final SDG assignment uses the validated centroid instrument:
  - OSDG human-validated labels build SDG 1-16 centroids.
  - SDG Classification Benchmark expert labels validate the instrument.
  - SDG 17 centroid uses Benchmark positives and is not independently validated.
- Nearest-centroid averaging is a deliberate transparent measurement instrument, not an industry-grade operational classifier.
- Macro-F1 0.733 on SDGs 1-16 is good enough for exploratory corpus-level coverage and semantic-gap analysis, with per-SDG caveats.

## Do Not Do

- Do not run current `code/preprocess_papers.py` directly on the 7.1 GiB raw file.
- Do not run current `code/embeddings.py` expecting it to handle the full paper corpus in memory.
- Do not balance papers by SDG, citation count, or OpenAlex relevance rank before coverage analysis.
- Do not rebuild SDG centroids unless OSDG or Benchmark inputs change.

## Rebuild Plan

1. Add streaming OpenAlex cleaning.
   - Read `data/openalex/papers.jsonl` line by line.
   - Apply deterministic eligibility filters:
     - usable title and abstract
     - abstract length threshold
     - year 2018-2025
     - dedupe by OpenAlex ID, with DOI/title fallback only if needed
   - Use SQLite or another disk-backed key store for dedupe.
   - Write clean shards to `data/openalex/clean_shards/part-*.jsonl`.
   - Write a manifest with shard paths, row counts, bytes, checksums, schema version, and per-SDG/year counts.

2. Add paper-shard embedding.
   - Keep `all-MiniLM-L6-v2` and `normalize_embeddings=True`.
   - Embed one clean shard at a time by default.
   - Write:
     - `data/embeddings/papers_shards/part-*.npy`
     - `data/embeddings/papers_shards/part-*_ids.json`
     - `data/embeddings/papers_shards/manifest.json`
   - Add resumable status under `artifacts/job_status/`.
   - Run a 10k-paper smoke test before full embedding.

3. Score paper shards against existing SDG centroids.
   - Input: `data/sdg_centroids.npy`.
   - For each embedding shard, write score shards:
     - `data/paper_scores_shards/part-*.npy`
     - `data/paper_scores_shards/part-*_ids.json`
     - `data/paper_scores_shards/manifest.json`
   - While streaming, accumulate per-SDG paper counts and embedding sums for new research centroids.

4. Rebuild research-dependent outputs.
   - `data/paper_scores.npy` or a manifest-backed replacement
   - `data/paper_scores_ids.json` or shard ID manifest
   - `data/research_centroids.npy`
   - `data/research_centroid_meta.json`
   - `data/policy_scores_vs_research.npy`

5. Rerun downstream analysis.
   - `code/coverage_gap.py`
   - `code/semantic_gap.py`
   - `code/coverage_semantic_interaction.py`
   - `code/kaggle_context.py` if generated context numbers depend on updated gaps
   - `code/fix_sdg4_artefact.py` or an updated full-corpus equivalent
   - `code/plot_figures.py`
   - LaTeX build

6. Update writing after rebuild.
   - Replace provisional 6,172-paper results.
   - Update draft status note from interim OpenAlex snapshot to full OpenAlex corpus snapshot.
   - Keep caveats:
     - SDG 4 ML vocabulary artefact
     - SDG 1/8/10 centroid collinearity
     - SDG 13/17 centroid collinearity
     - A15 policy-language calibration bias
     - no rejection class in centroid validation

## Runtime Estimate

- Current environment sees CPU-only PyTorch for SentenceTransformer.
- Local benchmark: about 39 texts/sec for `all-MiniLM-L6-v2`.
- Full 2,763,579-paper embedding estimate: about 20 hours CPU time.
- End-to-end full rebuild after implementation: roughly 24-36 hours wall-clock.
- If CUDA is made available for the RTX 3050 Laptop GPU, embedding may drop to about 4-8 hours, but 4 GiB VRAM requires conservative batching.

## First Commands After Implementation

```bash
# smoke test only, target around 10k papers
python code/preprocess_papers_streaming.py --limit 10000
python code/embed_paper_shards.py --limit-shards 1
python code/score_paper_shards.py --limit-shards 1

# then inspect manifests, row counts, shapes, and a few sampled IDs
```

Only launch the full job after the smoke test proves resume behavior, row ordering, shapes, and ID alignment.

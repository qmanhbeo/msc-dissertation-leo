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

## Robustness Checks To Consider

These are not blockers for the next engineering step, but they are useful
viva-proofing checks before locking the final dissertation analysis.

### Methodological Checks

- Add a "none of the above" eligibility or distance diagnostic.
  - Current nearest-centroid assignment always picks one SDG.
  - Because the research corpus is pre-filtered by OpenAlex AI+SDG retrieval,
    this is acceptable for relative SDG assignment, but a maximum-distance or
    low-confidence flag would help identify weakly SDG-related papers.
  - Candidate diagnostic: inspect distribution of max centroid similarity and
    flag bottom decile or a fixed threshold chosen from Benchmark negatives.

- Compare mean centroids against medoids.
  - Current SDG centroid = mean of all unit embeddings for that SDG.
  - Robustness check: compute a medoid per SDG, i.e. the actual labelled OSDG
    text closest to the mean, then compare validation macro-F1 and per-SDG F1.
  - Purpose: test whether OSDG outliers pull mean centroids off target.

- Document token handling explicitly.
  - For current analysis, research unit is title + abstract, so truncation risk
    should be small.
  - Policy texts are chunked before embedding, so each chunk should remain under
    sentence-transformer token limits.
  - If full papers are ever embedded, choose and document a strategy:
    head+tail truncation, section-level chunking with mean pooling, or weighted
    pooling across chunks.

- Emphasize baseline context.
  - Clean validation: macro-F1 0.733 on SDGs 1-16.
  - Random 17-way baseline: 0.059.
  - This is about 12.5x random, supporting the efficiency-to-gain argument for
    a simple centroid instrument.

### Semantic And SDG-Specific Checks

- Add top-k SDG diagnostics.
  - Current hard assignment uses top-1 nearest centroid.
  - For cross-cutting SDGs, report top-3 centroid shares or ambiguity rates.
  - Useful cases: SDGs 8, 11, 13, 17 and nexus topics such as energy/transport,
    climate/partnerships, and work/inequality.

- Inspect vocabulary distinctiveness.
  - Strong SDGs such as 4, 6, and 7 may perform better because they have more
    distinctive vocabulary.
  - Weak SDGs such as 8 and 11 may be weaker because their language is broader
    and overlaps with neighbouring goals.
  - Possible check: compare top terms or nearest-neighbour examples per SDG,
    and report this as an interpretability caveat.

- Keep SDG 17 cleanly caveated.
  - Headline validation should remain SDGs 1-16 only.
  - The all-17 result is supplementary because SDG 17 centroid uses Benchmark
    positives and is therefore contaminated.
  - Could mention the contaminated SDG 17 result as a sensitivity illustration
    of training/evaluation overlap, not as independent evidence.

### Future-Ready Extensions

- Vector search libraries are optional, not required for centroid scoring.
  - FAISS or Annoy would be useful for nearest-document retrieval or medoid
    search over millions of vectors.
  - For scoring 2.7M papers against only 17 centroids, brute-force matrix
    multiplication is already cheap; embedding remains the bottleneck.

- Consider weighted centroid variants later.
  - TF-IDF or SIF-style weighting could reduce the effect of generic academic
    language and increase influence from SDG-distinctive terms.
  - Treat this as a future-method extension, not a current requirement.

- Consider multilingual or stronger embedding models later.
  - BGE-M3 or multilingual sentence-transformers would support non-English
    policy documents and a more global policy corpus.
  - This would require rerunning centroid validation before any new claims.

### Validation Snapshot To Remember

- Headline clean performance: macro-F1 0.733 on SDGs 1-16.
- Strongest SDGs:
  - SDG 4: 0.920
  - SDG 7: 0.889
  - SDG 6: 0.869
- Main challenge:
  - Cross-cutting or conceptually broad SDGs, especially SDG 8 and SDG 11.

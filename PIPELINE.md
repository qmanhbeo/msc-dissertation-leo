# FETCH 
 1. Download relevant policy documents manually 

Code: 1_code/0_fetch/fetch_openalex.py --retrieval [keyword | concept]

 2. Fetch OpenAlex for Research corpus (takes days): keyword-based (canon) OR concept-based (robustness test)

Data: 2_data/0_raw/[openalex | openalex_concept]/*

---

Code: 1_code/0_fetch/*.py

 3. Fetch policy documents (web scraper) + fetch UNGDC + convert manually downloaded policy for Policy corpus
 4. Fetch SDGi (for both Policy + Reference corpora) + OSDG + Knowledge Hub + SDG Benchmark + Aurora for Reference corpus

Data: 2_data/0_raw/[source]/*


# PREPROCESS (clean + trim)
CODE: 1_code/1_preprocess/0_preprocess_*.py

 1.  preprocess policy (scraped + manual)
 2.  preprocess ungdc
 3.  preprocess osdg
 4.  preprocess sdg benchmark
 5.  preprocess sdg knowledge hub
 6.  preprocess aurora
 7.  preprocess sdgi

DATA: 2_data/1_preprocessed/individual_sources/[source]/*


# BUILD CORPORA — shared, ONCE
 
 CODE: 1_code/1_preprocess/0_preprocess_papers_streaming.py

 1.  preprocess research shards (canon)
 2.  preprocess concept-based research corpus (robustness)

=> DATA: 2_data/1_preprocessed/[research | research_concept]/*.jsonl

---

CODE: 1_code/1_preprocess/1_build_*.py

 2.  build reference corpus (consolidate + dedup)
 3.  build policy corpus (consolidate + dedup)

=> DATA: 2_data/1_preprocessed/[policy | reference].jsonl. ONE clean jsonl each corpus

# SEGMENT — canonical MPNet segments, shared, ONCE

CODE: 1_code/2_segment/1_segment_corpus.py

 1.  segment reference
 2.  segment policy

DATA: 2_data/2_segmented/[policy|reference].jsonl

---

CODE: 1_code/2_segment/1_segment_corpus.py

 3.  segment canon research
 4.  segment concept-based research (for retrieval-method robustness)

DATA: 2_data/2_segmented/[research|research_concept]/part-*.jsonl

---

CODE: 1_code/2_segment/2_sample_segments.py

 5.  build research 50k subset (for other embedding models' robustness tests)

DATA: 2_data/2_segmented/research_subset/part-00001.jsonl


# PER-MODEL LOOP  ×3  (MPNet → MiniLM → SciBERT) 

for model m in [MPNet, MiniLM, SciBERT]:

## EMBED (m)

CODE: 1_code/3_embed/0_embed_reference_and_policy_corpora.py

1. embed reference
2. embed policy

DATA: 2_data/3_embedded/[model]/[reference|policy].npy

---

CODE: 1_code/3_embed/0_embed_paper_shards.py

3. embed paper shards   (full corpus for MPNet; 50k subset for MiniLM/SciBERT)
4. embed concept-retrieved paper shards   (MPNet only)

DATA: 
- 2_data/3_embedded/[model]/research_shards.npy/part-*.npy (27 shards for MPNet, 1 shard each for MiniLM and SciBERT)
- 2_data/3_embedded/[model]/research_concept/part-*.npy (MPNet only)


## TRAIN (m)

CODE: 1_code/4_supervised_model_train/*

1. prepare training data (what does this do?)
2. Per supervised model [LogReg, MLP]:

    2.1. *LR and MLP grid search with CV done manually. Tested (retrain on train+val, is it?) on held-out test set. Results saved durably. No re-run in replays.*

    2.2. retrain full train+val+test data (currently it's train + val)

5. build SDG reference centroids (from original labelled reference texts)

DATA: 2_data/4_supervised_model_results/[model]/*

## SCORE (m)

CODE: 1_code/5_supervised_model_infer/

1. Per supervised model [LogReg, MLP]:

    1.1. score research shards (use canon + concept-retrieval for MPNet; use subset for MiniLM + SciBERT)

    1.2. score policy corpus

    1.3. check centroid consistency (what does this do?)

2. build centroid similarity matrix (LR only)
3. score research & policy with zeroshot method (nearest-centroid SDG assignment)


# ANALYSIS (run_analysis, in-process)

For MPNet:

1. 0_pca_semantic_landscape
2. 0_coverage_gap: use canon + concept-retrieval + filtered policy corpus (Curated SDGi UNGDC) 
3. 1_semantic_gap: use canon + concept-retrieval + filtered policy corpus (Curated SDGi UNGDC) 
4. 2_coverage_semantic_interaction    [DEBT: runs for MiniLM/SciBERT]

For MiniLM and SciBERT, and any robustness encoders to come: 

1. 0_coverage_gap (use canon + concept-retrieval for MPNet; use subset for MiniLM + SciBERT)
2. 1_semantic_gap (use canon + concept-retrieval for MPNet; use subset for MiniLM + SciBERT)

3. 3_generate_cross_sensitivity_table [WRONG. DEBT: runs per-model, incomplete; overwritten by final step below]

## FIGURES

1. plot figures                      ✓ MPNet-only (already correct)

# Robustness check — after the model loop

1.  generate canonical cross-sensitivity table (with data from all 3 encoders + concept-retrieval + per-policy-corpus filters)

5. appendix a2/a3/b2/c/f/h1 with canon MPNet only[DEBT: runs for MiniLM/SciBERT]

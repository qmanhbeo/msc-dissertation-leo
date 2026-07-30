# FETCH 
 1. Download relevant policy documents manually 

Code: 1_code/0_fetch/fetch_openalex.py --retrieval [keyword | concept]

 2. Fetch OpenAlex for Research corpus (takes days): keyword-based (canon) OR concept-based (robustness test)

Data: 2_data/0_raw/[openalex | openalex_concept]/*

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

CODE: 1_code/1_preprocess/1_build_*.py

 2.  build reference corpus (consolidate + dedup)
 3.  build policy corpus (consolidate + dedup)

=> DATA: 2_data/1_preprocessed/[policy | reference].jsonl. ONE clean jsonl each corpus

# SEGMENT — canonical MPNet segments, shared, ONCE

CODE: 1_code/2_segment/1_segment_corpus.py

 1.  segment reference
 2.  segment policy

DATA: 2_data/2_segmented/[policy|reference].jsonl

CODE: 1_code/2_segment/1_segment_corpus.py

 3.  segment canon research
 4.  segment concept-based research (for retrieval-method robustness)

DATA: 2_data/2_segmented/[research|research_concept]/part-*.jsonl


CODE: 1_code/2_segment/2_sample_segments.py.py

 5.  build research 50k subset (for other embedding models' robustness tests)

DATA: 2_data/2_segmented/research_subset/part-00001.jsonl


# ══ PER-MODEL LOOP  ×3  (MPNet → MiniLM → SciBERT) 

    for model m in [MPNet, MiniLM, SciBERT]:

    -- EMBED (m) --
       embed osdg
       embed benchmark
       embed sdg_knowledge_hub
       embed sdgi
       embed aurora
       embed policy_scrape
       embed policy_manual
       embed ungdc_sdg
       merge policy corpus
       embed paper shards   (full corpus for MPNet; 50k subset for MiniLM/SciBERT)

    -- TRAIN + SCORE (m) --
       prepare training data
       retrain full data (LR)
       build SDG reference centroids
       score research shards (LR)
       score policy corpus (LR)
       retrain MLP
       score MLP
       check centroid consistency
       build centroid similarity matrix

    -- ANALYSIS (run_analysis, in-process) --
       score_zeroshot                     [DEBT: runs for MiniLM/SciBERT]
       0_coverage_gap                     ✓ all models (needed)
       1_semantic_gap                     ✓ all models (needed)
       2_coverage_semantic_interaction    [DEBT: runs for MiniLM/SciBERT]
       3_generate_cross_sensitivity_table [DEBT: runs per-model, incomplete;
                                            overwritten by final step below]
       0_pca_semantic_landscape           ✓ MPNet-only (already correct)
       appendix a2/a3/b2/c/f/h1           [DEBT: runs for MiniLM/SciBERT]

    -- FIGURES --
       plot figures                      ✓ MPNet-only (already correct)

══ FINAL — after the loop ════════════════════════════════════════

18.  regenerate canonical cross-sensitivity table (all 3 encoders)
# SDG classification pipeline — execution order

**Legend**
- `[model]` ∈ {`mpnet`, `minilm`, `scibert`} — encoder slugs (MPNet = `all-mpnet-base-v2`, canonical; MiniLM = `all-MiniLM-L6-v2`; SciBERT = `allenai/scibert_scivocab_uncased`). Every artifact below is namespaced under `[model]`.
- `[source]` — each raw/preprocessed data source.
- **(MPNet only)** — step runs only for the canonical encoder; skipped for MiniLM/SciBERT.
- **Chaining rule:** each stage consumes the previous stage's outputs; script cards show only outputs. External, manual, and shared (fan-out) inputs are annotated inline.
- Segments are canonical (built once with the MPNet segmenter) and shared by every encoder — only the encoder (and its native context window) varies in the per-model loop.
- **Stage mapping:** each section header names the `main.py --stage <stage>` value(s) that run it. There is no `--stage score` or `--stage figures`: scoring spans `--stage infer` + `--stage centroids` (see §6), and figures run inside `--stage analysis` (see §7). Appendix scripts run via `--appendix-*` flags, not `--stage`; the LR/MLP grid-search step is never invoked by main.py.

---

## 1. FETCH — external + manual inputs (`--stage fetch`)

- fetch research corpus (canon)

  Script `1_code/0_fetch/fetch_openalex.py --retrieval keyword` --> Output: `2_data/0_raw/openalex/*`

- fetch concept-based research corpus (retrieval-method robustness)

  Script `1_code/0_fetch/fetch_openalex.py --retrieval concept` --> Output: `2_data/0_raw/openalex_concept/*`

- scrape policy documents

  Script `1_code/0_fetch/fetch_policy.py` --> Output: `2_data/0_raw/policy_scrape/*`

- convert manually downloaded policy documents

  Script `1_code/0_fetch/convert_policy_manual.py` --> Output: `2_data/0_raw/policy_manual/*`

- fetch UNGDC speeches

  Script `1_code/0_fetch/fetch_ungdc.py` --> Output: `2_data/0_raw/ungdc_sdg/*`

- fetch SDGi (dual-role source — feeds both Policy and Reference corpora; labelled in reference, unlabelled policy documents)

  Script `1_code/0_fetch/fetch_sdgi_corpus.py` --> Output: `2_data/0_raw/sdgi/*`

- fetch OSDG

  Script `1_code/0_fetch/fetch_osdg.py` --> Output: `2_data/0_raw/osdg/*`

- fetch SDG Benchmark

  Script `1_code/0_fetch/fetch_sdg_benchmark.py` --> Output: `2_data/0_raw/sdg_benchmark/*`

- fetch SDG Knowledge Hub

  Script `1_code/0_fetch/fetch_sdg_knowledge_hub.py` --> Output: `2_data/0_raw/sdg_knowledge_hub/*`

- fetch Aurora

  Script `1_code/0_fetch/fetch_aurora.py` --> Output: `2_data/0_raw/aurora/*`

Inputs (this stage only): manual policy downloads; `.env` with `OPENALEX_MAILTO` + `OPENALEX_API_KEY`. All fetchers write incrementally with resume — a kill loses nothing.

---

## 2. PREPROCESS — clean + trim, then build consolidated corpora (`--stage preprocess`)

- preprocess policy (scraped + manual)
- preprocess ungdc
- preprocess osdg
- preprocess sdg benchmark
- preprocess sdg knowledge hub
- preprocess aurora
- preprocess sdgi

  Script `1_code/1_preprocess/0_preprocess_*.py` --> Output: `2_data/1_preprocessed/individual_sources/[source]/*`

- preprocess research shards (canon)
- preprocess concept-based research corpus (`--retrieval concept`, retrieval-method robustness; streaming, resume-safe)

  Script `1_code/1_preprocess/0_preprocess_papers_streaming.py` --> Output: `2_data/1_preprocessed/{research, research_concept}/*.jsonl`

- build reference corpus (consolidate + dedup)
- build policy corpus (consolidate + dedup)

  Script `1_code/1_preprocess/1_build_reference_corpus.py` + `1_build_policy_corpus.py` --> Output: `2_data/1_preprocessed/{reference, policy}.jsonl` — ONE clean jsonl per corpus

---

## 3. SEGMENT — once, shared across all encoders (`--stage segment`)

Segments are canonical and shared by every encoder (always `--embed-model all-mpnet-base-v2`). `--stage segment --corpus <all|reference|policy|research|research_concept>` selects which corpora are segmented (default `all` runs the four steps below).

- segment reference
- segment policy

  Script `1_code/2_segment/1_segment_corpus.py` --> Output: `2_data/2_segmented/{reference, policy}.jsonl`

- segment research (canon)
- segment concept-based research (retrieval-method robustness)

  Script `1_code/2_segment/1_segment_corpus.py` --> Output: `2_data/2_segmented/{research, research_concept}/part-*.jsonl`

- build research 100k subset (used by MiniLM/SciBERT to keep embedding feasible)

  Script `1_code/2_segment/2_sample_segments.py` --> Output: `2_data/2_segmented/research_subset/part-00001.jsonl`

---

## 4.–6. PER-MODEL LOOP — repeat for each m ∈ {mpnet, minilm, scibert}

All artifacts below are namespaced under `[model]`. Steps marked (MPNet only) are skipped otherwise.

### 4. EMBED (`--stage embed`)

- embed reference corpus
- embed policy corpus

  Script `1_code/3_embed/0_embed_reference_and_policy_corpora.py` --> Output: `2_data/3_embedded/[model]/{reference, policy}.npy` (+ metadata ids)

- embed research shards (full corpus for MPNet — 27 shards; `--corpus research_subset` for MiniLM/SciBERT — 1 shard)
- embed concept-retrieved paper shards (`--corpus research_concept`; MPNet only)

  Script `1_code/3_embed/0_embed_paper_shards.py` --> Output: `2_data/3_embedded/[model]/research_shards/part-*.npy` · `research_concept/part-*.npy`

Invariant: research-corpus text = `"{title}. {abstract}"` (set in `0_preprocess_papers_streaming.py`) — any subset must embed the same string or the LR scores a different representation.

### 5. TRAIN (`--stage train`)

`--stage train` runs exactly: prepare training data + retrain LR champion + retrain MLP champion. The SDG reference centroids (previously listed here) are built by `--stage centroids` — see §6.

- prepare training data (build one-hot 17-D labels from labelled reference texts — single-label records only; per-source stratified, document-grouped 85/15 train/test split)

  Script `1_code/4_supervised_model_train/0_prepare_data.py` --> Output: `2_data/4_supervised_model_results/[model]/{embeddings, labels, sources, source_docs}.npy` · `indices/{train, test}.npy` · `split_report.txt`

- LR grid search (GroupKFold(5) CV on the train pool: C/l1_ratio/class_weight), champion selected by CV macro-F1
- MLP grid search (GroupKFold(5) CV on the train pool: depth/width/lr), champion selected by CV macro-F1
- results saved durably; NOT invoked by main.py (provenance-guarded); consumed by `d1_export_model_selection_nums.py` → Appendix D; no re-run in replays

  Script `1_code/4_supervised_model_train/1_grid_search.py` --> Output: `2_data/4_supervised_model_results/[model]/model/{lr_cv_results.json, mlp_cv_results.json, lr_grid_search_log.json, mlp_grid_search_log.json}` (shared logic in `train_models_utils.py`)

- retrain LR champion on full train pool (C=10/l2/lbfgs; single-use held-out test evaluation) → `sdg_classifier_retrained.joblib` · test macro-F1 0.8231 → `sdg_retrain_results.json`
- retrain MLP champion on full train pool (`--classifier-type mlp`) → `mlp_retrained.joblib` · `mlp_retrain_results.json`
- test set NOT reabsorbed: the model is the measurement instrument; reported metrics describe the exact scoring artifact

  Script `1_code/4_supervised_model_train/2_retrain_full_data.py` --> Output: `2_data/4_supervised_model_results/[model]/model/*.joblib` · `*_results.json` · `4_outputs/[model]/data/4_1_confusion_matrix.csv` (LR)

Exploratory (not adopted): `2_retrain_full_data.py --cv-full-data` — manual GroupKFold CV on 100% of labelled data with champion hyperparameters; writes `cv_full_data_results.json`, touches no artifacts. Measured +0.002 CV mean (0.8107 → 0.8127) for absorbing the test set — negligible, so the held-out test (0.8231) remains the headline. See Open items.

### 6. SCORE — split across `--stage centroids` + `--stage infer`

No single `--stage score` exists. The `centroids` stage builds the SDG reference centroids and runs the sanity gates; the `infer` stage performs SDG assignment. Cross-stage dependency: the zero-shot step needs `sdg_centroids.npy` (so `centroids` must run before `infer`), while the consistency gate reads the LR/MLP scores produced by `infer` — a piecemeal stage sequence is not a clean DAG, and only warm/cold replay guarantee the full interleaving.

**`--stage centroids`** (internal order: build → consistency → similarity)

- build SDG reference centroids (from original labelled reference texts)

  Script `1_code/6_calculate_centroids/0_build_sdg_reference_centroids.py` --> Output: `2_data/5_supervised_scored/[model]/sdg_centroids.npy`

- check centroid consistency (runtime SANITY GATE, not a producer: compare MLP-assigned SDG vs nearest-centroid SDG per corpus — per-SDG agreement + confusion; also saves policy centroids — diagnostic-only, read by nothing downstream)

  Script `1_code/6_calculate_centroids/0_check_centroid_consistency.py` --> Output: `2_data/5_supervised_scored/[model]/{policy_centroids.npy, metadata/centroid_consistency.json}`

- build centroid similarity matrix (LR only; reads sdg_centroids.npy)

  Script `1_code/6_calculate_centroids/1_build_centroid_similarity_matrix.py` --> Output: `4_outputs/[model]/data/4_1_centroid_similarity_matrix.csv`

**`--stage infer`** (internal order: LR → MLP → zero-shot → concept)

- score research shards (`--classifier lr --corpus research`) → research_centroids.npy (supervised, PRIMARY)
- score policy corpus (`--classifier lr --corpus policy`) → policy_scores.npy
- score MLP (`--classifier mlp`) → mlp_scores/{mlp_summary.json, mlp_policy_scores.npy}
- score research & policy with zeroshot method (nearest-centroid SDG assignment)

  Script `1_code/5_supervised_model_infer/score_supervised.py` + `1_code/6_calculate_centroids/score_zeroshot.py` --> Output: `2_data/5_supervised_scored/[model]/{research_centroids.npy, policy_scores.npy, paper_scores_shards/, mlp_scores/, zeroshot/{research, policy}_centroids.npy}` · `4_outputs/[model]/data/semantic_gap_distances.json`

- score concept-retrieval variant with the concept manifest (MPNet only): LR with `--embedding-manifest research_concept/metadata/manifest.json --out-dir paper_scores_shards_concept --research-centroids-out research_concept_centroids.npy`; MLP with `--corpus research_concept` → mlp_scores_concept/; zero-shot with `--embedding-manifest … --out-dir zeroshot_concept --data-dir data/concept`

  Script `1_code/5_supervised_model_infer/score_supervised.py` + `1_code/6_calculate_centroids/score_zeroshot.py` --> Output: `2_data/5_supervised_scored/[model]/{research_concept_centroids.npy, mlp_scores_concept/, zeroshot_concept/}` · `4_outputs/[model]/data/concept/semantic_gap_distances.json` (zero-shot)

---

### REGISTER ADJUSTMENT (core stage, between SCORE and ANALYSIS)

Register adjustment is a **core main-pipeline stage**, not a sensitivity check. It runs `register_adjust.py` to materialise the INLP projection matrix `G` (stored in `2_data/`, never `4_outputs/`), then the decomposition + convergence diagnostic is generated by `g_register_decomposition.py` (canon, MPNet; runs for every encoder) which also emits the iterative convergence tables consumed by Appendix E.

The linear flow is strictly: **SCORE → COVERAGE GAP → REGISTER ADJUSTMENT → SEMANTIC GAP (raw then adjusted) → PCA before/after → CORRELATION + ROBUSTNESS**. `register_adjust` must run before the adjusted semantic-gap pass (it supplies `G`); coverage gap is computed first because it is adjustment-invariant. The appendix `f_register_adjustment.py` script was folded into this canon flow and deleted.

- INLP register removal (build G)

  Script `1_code/7_main_analysis/0_shared/register_adjust.py` (`--stage register_adjust`) --> Output: `2_data/3_embedded/[model]/register/{track}/G.npy` · `checkpoint.json`

- register-topic decomposition + iterative convergence diagnostic (canon)

  Script `1_code/7_main_analysis/0_shared/g_register_decomposition.py` --> Output: `4_outputs/[model]/data/register_decomposition.json` · `4_outputs/[model]/tables/{tab_register_decomposition, tab_iterative_register_check, num_iterative_register_check}.tex`

---

## 7. ANALYSIS — in-process (run_analysis), shard-native mmap (`--stage analysis`)

All main-text analyses run in a single process per encoder; each reads the research embedding/score shards directly (no consolidated array). Outputs → `4_outputs/[model]/data/*.json` · `tables/*.tex`. `--stage analysis` composes the full linear analysis for `--embed-model` only (appendix analyses included when that is the default model), then calls `_run_analysis_poststeps` to regenerate the canonical cross-sensitivity table + figures. The cross-sensitivity table still needs all three encoders' main-text outputs present (existence-skipped if already built by a prior run); cold replay is the only path that builds all three encoders in one invocation.

| script | runs on | notes |
|---|---|---|
| `0_pca_semantic_landscape.py` | MPNet only | fixed figures paths |
| `0_coverage_gap.py` | all encoders | canon + concept (MPNet) / subset (MiniLM, SciBERT); filtered policy corpus (Curated SDGi UNGDC) |
| `1_semantic_gap.py` | all encoders | canon + concept (MPNet) / subset (MiniLM, SciBERT); filtered policy corpus (Curated SDGi UNGDC) |
| `2_coverage_semantic_interaction.py` | all encoders | reads 4_2 + 4_3; namespaced per-encoder |
| `3_generate_cross_sensitivity_table.py` | default only, via `_run_analysis_poststeps` (not inside `_run_main_analysis_steps`) | canonical tables: policy source × segment cap × retrieval (LR/MLP/ZS) + encoder axis (all 3 encoders) |
| `g_distributional_gap.py` | opt-in | MAIN-RESULT Table; NOT run by warm replay or `--appendix-all`; run before `--build-pdf` |

**Concept pass (MPNet only, after the in-process analyses above, before `3_generate_cross_sensitivity_table.py`):** re-invoke `0_coverage_gap.py` (`--paper-scores-manifest paper_scores_shards_concept/metadata/manifest.json --out-data-dir data/concept`) and `1_semantic_gap.py` (`--research-centroids research_concept_centroids.npy --research-centroid-meta metadata/research_concept_centroid_meta.json --out-data-dir data/concept`) → `4_outputs/[model]/data/concept/{4_2_*, 4_3_*}`. Must precede the cross-sensitivity table, whose Retrieval column reads these files plus `mlp_scores_concept/` and `zeroshot_concept/`.

### Figures (MPNet only) — part of `--stage analysis`

There is no standalone `--stage figures`. `plot_figures.py` runs inside `--stage analysis` (post-step, once all three encoders' analyses exist) and as step 9 of the warm/cold per-model loop — always for the canonical encoder only.

- plot manuscript figures (PCA landscape, coverage profiles, semantic gap, coverage×semantic scatter, centroid-similarity heatmap)

  Script `1_code/8_visualization/plot_figures.py` --> Output: `4_outputs/[model]/figures/{fig1, fig3, fig4, fig5}*.{pdf, png}` · `4_outputs/appendix/[model]/a4_centroid_similarity/figures/*`

---

## 8. ROBUSTNESS + APPENDIX — cross-encoder, after the model loop (via `--appendix-*` flags, not `--stage`)

**Regenerate canonical cross-sensitivity table** — run once AFTER the per-model loop, now that all three encoders' outputs exist:
- `3_generate_cross_sensitivity_table.py` → `4_outputs/[model]/tables/{tab_cross_sensitivity_robustness.tex, tab_cross_sensitivity_coverage.tex, tab_encoder_sensitivity_semantic.tex, tab_encoder_sensitivity_coverage.tex, num_*.tex}` — the PDF-consumed tables (all 3 encoders + concept-retrieval + per-policy-corpus filters).

**Appendix scripts** — run for the canonical encoder in replays (cold replay: `include_appendix` only for MPNet); standalone via `--appendix-*` for any `--embed-model` (outputs namespaced under `4_outputs/appendix/[model]/`, not consumed by the paper for non-default models):

| script | appendix | output |
|---|---|---|
| `a2_policy_source_family_sensitivity.py` | A.2 | `appendix/[model]/a2_source_family_sensitivity/` |
| `a3_sdg4_lexical_audit.py` | A.3 | `appendix/[model]/a3_sdg4_audit/` |
| `b2_semantic_gap_text_interpretability.py` | B.2 | `appendix/[model]/b2_semantic_gap_interpretability/` |
| `c_sample_stability.py` | C | `appendix/[model]/c_sample_stability/` |
| `c0_export_corpus_split_sizes.py` | C.0 | `main/tables/num_reference_split.tex` |
| `d1_export_model_selection_nums.py` | D.1 | `main/tables/num_model_selection.tex` |
| `h1_cross_method_gap_values.py` | H.1 | `appendix/[model]/h1_cross_method_gap_values/` |

---

## Open items

- **MLP champion config discrepancy:** grid search + dissertation text cite lr=3e-4, but the retrained artifact (`mlp_retrained.joblib` / `model_config.json`) and the script's argparse default use lr=1e-3. A replay reproduces the artifact, not the cited champion. Decide: change the script default to 3e-4 or update the text.
- **`--cv-full-data` exploratory route:** exists in `2_retrain_full_data.py` (documented as exploratory, not invoked by main.py). Decision: not adopted — measured +0.002 CV mean for absorbing the test set; held-out test retained as headline.

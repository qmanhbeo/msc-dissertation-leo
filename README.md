# MSc AI and Sustainable Development — Dissertation

**University of Birmingham | School of Government**
**Research topic:** Semantic alignment between AI-for-sustainability research and SDG policy frameworks
**Final submission:** 1 September 2026, 12:00 pm

---

## Hard Rule: Rebuild from Scratch

**Every process in this repository must be fully reproducible from this document alone.**

This README is the single authoritative rebuild guide. If the machine is lost or data is corrupted, follow the pipeline below from Step 0 and every output can be recreated. When any script or data file changes, this README must be updated in the same session.

---

## Repository Layout

```
dissertation/
├── code/               # All scripts — fetch, preprocess, embed, analyse, backup
├── data/               # All data (raw + processed + embeddings + outputs)
│   ├── embeddings/     # .npy embedding matrices + _ids.json metadata
│   ├── osdg/           # OSDG Community Dataset (SDG-labelled texts, SDGs 1–16)
│   ├── sdg_benchmark/  # SDG Classification Benchmark (expert-labelled, SDGs 1–17)
│   ├── openalex/       # AI-for-sustainability research abstracts
│   ├── policy_all/     # Final merged policy corpus (all sources)
│   ├── policy_v3/      # Policy PDF batch 1 (raw + extracted text)
│   ├── policy_expanded/# Policy PDF batch 0 (raw + extracted text)
│   ├── un_sdg/         # UN SDG + curated policy docs (raw + chunks)
│   ├── sdgi_corpus/    # SDGi VNR/VLR national reports (Hugging Face)
│   ├── ungdc/          # UN General Debate Corpus (raw speeches)
│   ├── ungdc_sdg/      # SDG-relevant UNGDC passages (filtered)
│   ├── sdgindex/       # Kaggle SDG Index (country performance 2000–2022)
│   └── kaggle/         # Kaggle SDG Index processed output
├── notes/              # Research design documents
│   ├── HYPOTHESES.md       # 36 pre-registered hypotheses
│   ├── ASSUMPTIONS.md      # 26 documented assumptions with risk levels
│   ├── METHODOLOGY_DECISIONS.md
│   └── LIT_REVIEW_INSIGHTS.md
├── writing/            # Draft chapters
├── literature/         # Reference papers
├── CLAUDE.md           # Collaboration guidelines + hard rules
├── requirements.txt    # Python dependencies
└── README.md           # This file — the rebuild guide
```

---

## Full Pipeline: Rebuild from Scratch

### Step 0 — Environment setup

```bash
# Python 3.10+ required
pip install -r requirements.txt

# rclone for backup (already configured on this machine)
# To configure on a new machine:
#   rclone config   # follow prompts to add Google Drive remote
#   Remote name must be: stocks-ecosystem-data-snapshots
#   (or set DISSERTATION_SNAPSHOT_REMOTE_ROOT env var to override)
```

**`requirements.txt` installs:**
`requests`, `tqdm`, `pdfplumber`, `kaggle`, `beautifulsoup4`, `pandas`, `openpyxl`,
`sentence-transformers`, `scikit-learn`, `datasets`, `numpy`, `zstandard`

---

### Step 1 — Fetch raw data

These scripts are **independent and can run in parallel**. Run from project root.

| Script | Command | Output | Notes |
|--------|---------|--------|-------|
| `fetch_osdg.py` | `python code/fetch_osdg.py` | `data/osdg/osdg_dataset.csv` | Downloads from Zenodo; ~43k rows |
| `fetch_sdg_benchmark.py` | `python code/fetch_sdg_benchmark.py` | `data/sdg_benchmark/benchmark.csv` | Downloads from GitHub; 1,251 rows |
| `fetch_openalex.py` | `python code/fetch_openalex.py` | `data/openalex/papers_sdg01.jsonl` … `papers_sdg17.jsonl` | Queries OpenAlex API; 17 SDG × AI term queries; long-running (~hours) |
| `fetch_un_sdg.py` | `python code/fetch_un_sdg.py` | `data/un_sdg/pdfs/`, `data/un_sdg/texts/` | Downloads curated UN+AI policy PDFs; extracts text with pdfplumber |
| `fetch_policy_expanded.py` | `python code/fetch_policy_expanded.py` | `data/policy_expanded/pdfs/`, `data/policy_expanded/texts/` | 11 additional policy PDFs (IPCC, national AI strategies) |
| `fetch_policy_v3.py` | `python code/fetch_policy_v3.py` | `data/policy_v3/pdfs/`, `data/policy_v3/texts/` | 52 targeted PDFs; ~16 succeed; rest blocked or 404 |
| `fetch_policy_v3b.py` | `python code/fetch_policy_v3b.py` | `data/policy_v3/pdfs/`, `data/policy_v3/texts/` | Retry script for v3 failures; ~2/29 succeed from WSL |
| `fetch_ungdc.py` | `python code/fetch_ungdc.py` | `data/ungdc/TXT/**/*.txt` | Harvard Dataverse; 11,141 UN General Debate speeches |
| `fetch_sdgi_corpus.py` | `python code/fetch_sdgi_corpus.py` | `data/sdgi_corpus/sdgi_corpus.parquet` | Hugging Face; VNR/VLR national reports |
| `fetch_sdgindex.py` | `python code/fetch_sdgindex.py` | `data/sdgindex/sdr2025_data.xlsx` | Kaggle API; SDG Index 2025 |

**Known fetch failures (WSL network restrictions):**
- Most `fetch_policy_v3b.py` targets return 403/404/HTML redirect from WSL due to TLS
  fingerprinting and Bot-protection on WHO, EUR-Lex, OECD iLibrary, World Bank
- **Workaround used (2026-04-10):** OECD AI Principles and Addis Ababa Action Agenda
  were downloaded manually on a browser and placed in `data/policy_v3/pdfs/`
- If rebuilding: re-attempt these from a non-WSL environment, or manually download the
  18 documents listed in the Policy Corpus section below

---

### Step 2 — Preprocess raw data

Run **in this order** (some depend on prior outputs).

```bash
python code/preprocess_osdg.py
# Input:  data/osdg/osdg_dataset.csv
# Output: data/osdg/osdg_clean.jsonl (30,534 rows, agreement ≥ 0.5, ≥ 20 words)
# Filter: drops rows with agreement < 0.5 or word_count < 20; SDG 17 absent from OSDG

python code/preprocess_sdg_benchmark.py
# Input:  data/sdg_benchmark/benchmark.csv
# Output: data/sdg_benchmark/benchmark_clean.jsonl (616 rows)
# Filter: expert-positive labels only; all 17 SDGs present (SDG 17: 31 rows)

python code/preprocess_papers.py
# Input:  data/openalex/papers_sdg01.jsonl ... papers_sdg17.jsonl
# Output: data/openalex/papers_clean.jsonl (6,172 papers after dedup)
# Note:   combines title + abstract into combined_text; deduplicates by openalex_id

python code/preprocess_policy.py
# Input:  data/un_sdg/texts/*.txt, data/policy_expanded/texts/*.txt,
#         data/policy_v3/texts/*.txt
# Output: data/un_sdg/policy_chunks.jsonl (8,592 chunks, ~150–300 words each)
# Note:   dynamic chunking on sentence boundaries

python code/integrate_sdgi.py
# Input:  data/sdgi_corpus/sdgi_corpus.parquet
# Output: data/sdgi_corpus/sdgi_chunks.jsonl (31,941 chunks)
# Note:   rechunks long entries to ~150 words; adds source_doc metadata

python code/filter_ungdc_sdg.py
# Input:  data/ungdc/TXT/Session*/*.txt (sessions 70–80, 2015–2024)
# Output: data/ungdc_sdg/ungdc_sdg_chunks.jsonl (6,472 chunks)
# Note:   keeps only paragraphs containing SDG-related keywords

python code/build_policy_corpus.py
# Input:  data/un_sdg/policy_chunks.jsonl
#         data/sdgi_corpus/sdgi_chunks.jsonl
#         data/ungdc_sdg/ungdc_sdg_chunks.jsonl
# Output: data/policy_all/policy_chunks_extended.jsonl (47,005 chunks)
# Note:   deduplicates by exact text match after normalisation;
#         assigns merged chunk IDs (merged_NNNNNN)

python code/preprocess_sdgindex.py
# Input:  data/sdgindex/sdr2025_data.xlsx
# Output: data/sdgindex/overview.csv, data/sdgindex/summary.json
```

---

### Step 3 — Generate embeddings

```bash
python code/embeddings.py
# Model:  all-MiniLM-L6-v2 (384-dim, ~5× faster than mpnet; CPU/WSL-friendly)
# Output: data/embeddings/papers.npy      (6172, 384)  float32, L2-normalised
#         data/embeddings/papers_ids.json
#         data/embeddings/policy.npy      (47005, 384) float32, L2-normalised
#         data/embeddings/policy_ids.json
#         data/embeddings/osdg.npy        (30534, 384) float32, L2-normalised
#         data/embeddings/osdg_ids.json
#         data/embeddings/benchmark.npy   (616, 384)   float32, L2-normalised
#         data/embeddings/benchmark_ids.json
# Note:   idempotent — skips any corpus whose .npy already exists
# IMPORTANT: normalize_embeddings=True is required. Downstream scripts assume unit
#            vectors (dot product = cosine similarity). Changing this breaks everything.
```

---

### Step 4 — Build and validate the SDG measurement instrument

```bash
python code/sdg_centroids.py
# Input:  data/embeddings/osdg.npy + osdg_ids.json
#         data/embeddings/benchmark.npy + benchmark_ids.json (SDG 17 only)
# Output: data/sdg_centroids.npy      (17, 384) float32, unit-normalised
#         data/sdg_centroid_meta.json  per-SDG diagnostics
# Convention: centroids[i] = centroid for SDG (i+1); row 0 = SDG 1, row 16 = SDG 17
# Note:   SDGs 1–16 from OSDG; SDG 17 from benchmark (no OSDG labels for SDG 17)
# Assumption A-SDG17: the 31 SDG-17 benchmark texts are used to BUILD the centroid,
#           so validating it on the same texts inflates SDG-17 accuracy (see below)

python code/validate_centroids.py
# Input:  data/sdg_centroids.npy, data/sdg_centroid_meta.json
#         data/embeddings/benchmark.npy + benchmark_ids.json
# Output: data/validation_results.json      primary: macro-F1 on SDGs 1–16 (n=585)
#         data/confusion_matrix.csv          17×17 confusion matrix
#         data/centroid_similarity_matrix.csv 17×17 pairwise centroid cosine sim
# Result (2026-04-10): macro-F1 = 0.733 → PASS
# Note:   SDG-17 evaluation is contaminated (same data as centroid); primary metric
#         excludes SDG 17 — this is by design, not an error
```

---

### Step 5 — Alignment scoring

```bash
python code/alignment_score.py
# Input:  data/sdg_centroids.npy
#         data/embeddings/papers.npy + papers_ids.json
#         data/embeddings/policy.npy + policy_ids.json
#         data/policy_all/policy_chunks_extended.jsonl  (source_doc metadata)
# Output: data/paper_scores.npy              (6172, 17)  cosine sim per paper × SDG
#         data/paper_scores_ids.json         list of {id}
#         data/policy_scores.npy             (47005, 17) cosine sim per chunk × SDG
#         data/policy_scores_ids.json        list of {id, source_doc}
#         data/research_centroids.npy        (17, 384) per-SDG mean of research papers (H26)
#         data/research_centroid_meta.json   per-SDG diagnostics for research centroids
#         data/policy_scores_vs_research.npy (47005, 17) policy vs research centroids (H26)
# Note:   bidirectional scoring built in for H26 (research vs OSDG centroids +
#         policy vs research centroids); A15 circularity diagnostic run at end
# Result (2026-04-10): A15 FLAG — policy top scores (0.544) exceed paper top scores (0.353)
#         by 0.191 > 0.10 threshold. Flag in methodology limitations.
#         H26 preview: policy engages research framing more (0.472) than research engages
#         policy framing (0.353) — supports asymmetry hypothesis.
```

---

### Step 6 — Coverage gap

```bash
python code/coverage_gap.py
# Input:  data/paper_scores.npy + paper_scores_ids.json
#         data/policy_scores.npy + policy_scores_ids.json
# Output: data/coverage_gap.json          doc-weighted profiles + gap (canonical)
#         data/coverage_gap_raw.json       chunk-level profiles (diagnostic, biased)
# Note:   document-weighted policy profiles (A19) — each of 2,392 source_docs weighted
#         equally regardless of chunk count
# Result (2026-04-10): Largest gaps: SDG 13 (policy 36% vs research 5%, gap 0.310),
#         SDG 17 (policy 35% vs research 4%, gap 0.308), SDG 4 (research 22% vs policy
#         0.2%, gap 0.219), SDG 9 (research 17% vs policy 0.8%, gap 0.165), SDG 16
#         (policy 20% vs research 8%, gap 0.121).
#         CAVEAT: SDG 4 research dominance (22%) may reflect ML "learning" terminology
#         conflating with the Education centroid — flag in Limitations.
```

---

### Step 7 — Semantic gap

```bash
python code/semantic_gap.py
# Input:  data/paper_scores.npy + paper_scores_ids.json
#         data/policy_scores.npy + policy_scores_ids.json
#         data/embeddings/papers.npy, data/embeddings/policy.npy
# Output: data/semantic_gap.json           primary (chunk_cap=50)
#         data/semantic_gap_sensitivity.json  caps 20 + 100 for robustness check
# Note:   chunk cap (50 per source_doc per SDG) prevents SDSN/SDGi dominance (A19).
#         Rankings stable across caps 20/50/100.
# Result (2026-04-10): Largest gaps: SDG 8 (0.292), SDG 3 (0.285), SDG 16 (0.284).
#         Smallest: SDG 17 (0.182), SDG 9 (0.201) — confirms H10 prediction.
```

---

### Step 8 — Coverage × semantic interaction (headline finding)

```bash
python code/coverage_semantic_interaction.py
# Input:  data/coverage_gap.json, data/semantic_gap.json
#         data/paper_scores.npy, data/policy_scores_vs_research.npy
# Output: data/h25_correlation.json   Pearson + Spearman correlations (H25 + H26)
#         data/h25_scatter.csv        per-SDG (research%, policy%, coverage_gap, semantic_gap)
# Result (2026-04-10):
#   H25: NOT SUPPORTED — r=0.145 (p=0.578); coverage and semantic gaps are independent.
#        This IS a finding: two orthogonal dimensions of misalignment.
#   H26: SUPPORTED — policy engages research framing more (top sim 0.472) than research
#        engages policy framing (0.353); asymmetry gap=0.120. Caveat: A15 flag means
#        some of this gap may reflect calibration bias (A15 gap=0.191).
```

---

### Step 9 — Contextual SDG Index analysis

```bash
python code/kaggle_context.py
# Input:  data/h25_correlation.json
#         data/kaggle/sdg_index_2000-2022.csv  (goal_N_score columns, year=2022)
# Output: data/sdg_context.json   gap scores joined with SDG Index global means
#         data/sdg_context.csv    per-SDG summary table for plotting
# Result (2026-04-10):
#   H21: NOT SUPPORTED — r=-0.040 (p=0.882); research coverage ≠ SDG performance
#   H22: NOT SUPPORTED — r=0.004 (p=0.988); semantic gap ≠ SDG performance
#   H23: CONFIRMED — SDG 13 scores highest in index (82.98 vs mean ~68), supporting
#        the claim that commitment indicators inflate climate scores in the SDG Index
```

---

### Step 10 — OSDG circularity diagnostic (A15)

Run **after** alignment_score.py. No dedicated script — add to validate_centroids.py or run inline:

```python
# Check if policy chunks score systematically higher than research papers
# against OSDG-derived centroids (would indicate calibration bias, not genuine alignment)
mean_policy_top_score = policy_scores.max(axis=1).mean()
mean_paper_top_score  = paper_scores.max(axis=1).mean()
# If mean_policy_top_score > mean_paper_top_score + 0.10: flag A15 in methodology
```

---

### Step 11 — Backup

```bash
python code/backup_data_snapshot.py
# Backs up data/ to Google Drive via rclone
# Remote: stocks-ecosystem-data-snapshots:dissertation-backup/data-snapshots/
# Keeps 7 most recent snapshots locally + on Drive
# Archive: dissertation-data-snapshot-YYYY-MM-DD-HHMMSS.tar.zst + .sha256
# Override remote: --remote-root gdrive:some/other/path
#                  or set DISSERTATION_SNAPSHOT_REMOTE_ROOT env var
```

---

## Current State (last updated 2026-04-10, analysis scripts complete)

### Data files — final

| File | Shape / Size | Notes |
|------|-------------|-------|
| `data/osdg/osdg_clean.jsonl` | 30,534 rows | SDGs 1–16; agreement ≥ 0.5 |
| `data/sdg_benchmark/benchmark_clean.jsonl` | 616 rows | SDGs 1–17; expert-labelled |
| `data/openalex/papers_clean.jsonl` | 6,172 papers | OpenAlex; 2018–2025 |
| `data/policy_all/policy_chunks_extended.jsonl` | 47,005 chunks | 3 sources merged + deduped |
| `data/sdgindex/overview.csv` | 4,140 rows | Country × SDG × year |
| `data/embeddings/papers.npy` | (6172, 384) float32 | L2-normalised |
| `data/embeddings/policy.npy` | (47005, 384) float32 | L2-normalised |
| `data/embeddings/osdg.npy` | (30534, 384) float32 | L2-normalised |
| `data/embeddings/benchmark.npy` | (616, 384) float32 | L2-normalised |
| `data/sdg_centroids.npy` | (17, 384) float32 | Unit-normalised; row i = SDG i+1 |
| `data/paper_scores.npy` | (6172, 17) float32 | Cosine sim per paper × SDG; col j = SDG j+1 |
| `data/paper_scores_ids.json` | 6,172 entries | `{id}` per row |
| `data/policy_scores.npy` | (47005, 17) float32 | Cosine sim per chunk × SDG |
| `data/policy_scores_ids.json` | 47,005 entries | `{id, source_doc}` per row |
| `data/research_centroids.npy` | (17, 384) float32 | Per-SDG mean of research papers; H26 |
| `data/research_centroid_meta.json` | 17 entries | n_papers_assigned, cohesion, zero_flag per SDG |
| `data/policy_scores_vs_research.npy` | (47005, 17) float32 | Policy chunks vs research centroids; H26 |
| `data/coverage_gap.json` | JSON | Doc-weighted policy + research profiles, gap per SDG |
| `data/coverage_gap_raw.json` | JSON | Chunk-level (unweighted, biased) profiles; diagnostic |
| `data/semantic_gap.json` | JSON | Per-SDG semantic gap (1 - cosine_sim), chunk_cap=50 |
| `data/semantic_gap_sensitivity.json` | JSON | Same with chunk_cap=20 and 100 for robustness |
| `data/h25_correlation.json` | JSON | H25 + H26 correlation results; per-SDG table |
| `data/h25_scatter.csv` | CSV | Per-SDG: research%, policy%, coverage_gap, semantic_gap |
| `data/sdg_context.json` | JSON | Gap scores joined with SDG Index 2022 global means |
| `data/sdg_context.csv` | CSV | Per-SDG summary for plotting (SDGs 1–16) |

### Policy corpus — sources

**`data/policy_all/policy_chunks_extended.jsonl` — 47,005 chunks:**

| Source | Chunks | What |
|--------|--------|------|
| Curated AI/SDG policy docs (`data/un_sdg/`) | 8,592 | 31 docs: UN, IPCC, national AI strategies, EU AI Act, SDSN, OECD |
| SDGi VNR/VLR corpus (`data/sdgi_corpus/`) | 31,941 | National govt reports from 40+ countries (UNDP) |
| UNGDC speeches (`data/ungdc_sdg/`) | 6,472 | UN General Debate, sessions 70–80 (2015–2024) |

**Key individual documents in the curated corpus:**

| Document | Institution | Year |
|----------|-------------|------|
| UN 2030 Agenda for Sustainable Development | United Nations | 2015 |
| Paris Agreement | UNFCCC | 2015 |
| Addis Ababa Action Agenda | UN (Financing for Development) | 2015 |
| UN SDG Progress Reports | UN Statistics Division | 2017–2022, 2024 (7 reports) |
| OECD AI Principles | OECD | 2019 |
| UK National AI Strategy | UK Government | 2021 |
| UNESCO Ethics of AI | UNESCO | 2021 |
| UN Secretary-General Roadmap for Digital Cooperation | UN | 2020 |
| IPCC AR6 WG2 + WG3 Summaries for Policymakers | IPCC | 2022 |
| EU AI Act | European Union | 2024 |
| UN AI Advisory Body Report | United Nations | 2024 |
| SDSN Sustainable Development Reports | SDSN | 2024, 2025 |
| African Union Continental AI Strategy | African Union | 2024 |

### Analysis scripts — status

| Script | Status | Output |
|--------|--------|--------|
| `sdg_centroids.py` | ✅ Done | `data/sdg_centroids.npy`, `data/sdg_centroid_meta.json` |
| `validate_centroids.py` | ✅ Done (macro-F1=0.733, PASS) | `data/validation_results.json`, `data/confusion_matrix.csv`, `data/centroid_similarity_matrix.csv` |
| `alignment_score.py` | ✅ Done (A15 FLAG: policy top sim 0.544 vs paper 0.353, gap=0.191) | `data/paper_scores.npy`, `data/policy_scores.npy`, `data/research_centroids.npy`, `data/policy_scores_vs_research.npy` |
| `coverage_gap.py` | ✅ Done | `data/coverage_gap.json`, `data/coverage_gap_raw.json` |
| `semantic_gap.py` | ✅ Done | `data/semantic_gap.json`, `data/semantic_gap_sensitivity.json` |
| `coverage_semantic_interaction.py` | ✅ Done (H25 null result; H26 supported with caveats) | `data/h25_correlation.json`, `data/h25_scatter.csv` |
| `kaggle_context.py` | ✅ Done (H21/H22 null; H23 confirmed) | `data/sdg_context.json`, `data/sdg_context.csv` |
| `topic_model.py` | ❌ Not written (optional) | interpretive clusters |

### Notes and design documents

| File | Contents |
|------|----------|
| `notes/HYPOTHESES.md` | 36 pre-registered hypotheses (H1–H36); updated 2026-04-10 |
| `notes/ASSUMPTIONS.md` | 26 documented assumptions with risk levels; updated 2026-04-10 |
| `notes/METHODOLOGY_DECISIONS.md` | Pipeline design rationale, gap type definitions |
| `notes/LIT_REVIEW_INSIGHTS.md` | ~1,684 papers synthesised via SciSpace |

---

## Research Overview

**Research question:** To what extent does academic AI-for-sustainability research show topical overlap with SDG policy priorities — and where are the gaps?

**Method:** Nearest-centroid SDG classification using Sentence-BERT embeddings. Centroids are built from the OSDG Community Dataset (SDGs 1–16) and the SDG Classification Benchmark (SDG 17). Coverage and semantic gaps are computed by comparing per-SDG score profiles between the research corpus (6,172 abstracts) and the policy corpus (47,005 chunks).

**Key design decisions** (see `notes/METHODOLOGY_DECISIONS.md` for full rationale):
- All claims framed as "topical overlap," never "alignment" without qualification (A16)
- Policy scores are document-weighted to counter SDSN/SDGi dominance (A19)
- SDGs 1, 8, 10 reported as a macro-cluster due to centroid collinearity (A26)
- SDG 17 centroid built from benchmark texts — contamination noted throughout (A-SDG17)
- H25 (coverage × semantic gap correlation) is the headline hypothesis

---

## Assessment

| Component | Weight | Format | Deadline |
|-----------|--------|--------|----------|
| Written Research Report | 75% | 8,000 words | 1 Sep 2026, 12 pm |
| Project Presentation | 25% | 10-min recorded video (max 15 slides) | 1 Sep 2026, 12 pm |
| Supervisor feedback draft deadline | — | Draft chapters | 1 Aug 2026 |

**Grade bands:** Distinction ≥72% · Merit ≥62% · Pass ≥52% · Fail <52%

**Programme:** MSc AI and Sustainable Development, University of Birmingham, School of Government

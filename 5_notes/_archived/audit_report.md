# Pipeline Implementation Audit

**Branch:** `supervised-reference` (HEAD: `0d0af3a`, working tree clean)  
**Date:** 2026-07-22  
**Scope:** Read-only audit of the SDG segmentation, embedding, scoring, and semantic-gap pipeline against the manuscript claims in `3_writing/dissertation.tex`

---

## 1. Segmentation Implementation Inventory

### 1A. `0_preprocess_policy.py` (policy_scrape + policy_manual)

**File:** `1_code/1_preprocess/policy/0_preprocess_policy.py`

**Algorithm (pseudocode):**
```
for each .txt file in input_dir:
    raw = read_file()
    text = clean_document(raw)     # unicode NFKC, strip page-break markers, OCR dup removal
    paragraphs = split_paragraphs(text)  # split on \n\n, discard any <5 words
    # Pre-expand overlong paragraphs:
    expanded = []
    for para in paragraphs:
        if len(para.words) > MAX_WORDS (300):
            expanded += split_long_paragraph(para, MAX_WORDS)
        else:
            expanded.append(para)
    # Greedy merge:
    segments = []
    for item in expanded:
        if current_count + item.word_count > TARGET_WORDS (150) and current non-empty:
            finalize current segment, start new
        else:
            append item to current
    finalize any remaining current
    # No post-hoc discard filter
```

**Merge unit:** Paragraph. Sentence-boundary splitting only fires when a **single paragraph exceeds 300 words** — that paragraph is split via `re.split(r"(?<=[.!?])\s+", text)` (line 105).

**Parameters:** `TARGET_WORDS = 150` (line 47), `MAX_WORDS = 300` (line 48). No overlap/sliding window.

**Min-word discard:** `split_paragraphs` discards paragraphs <5 words (line 101). No post-segment discard in this script.

**Actual segments below thresholds (from disk data):**

| Source | Raw segs | <10 words | <20 words |
|---|---|---|---|
| policy_scrape | 9,190 | 176 | 477 |
| policy_manual | 7,129 | 82 | 203 |

These sub-20-word segments were later discarded by `1_build_policy_corpus.py` (MIN_WORD_COUNT=20, line 37). So the <20 filter is NOT in the segmenter itself but in the merge step.

**Severity: Cosmetic.** The <20-word discard is described as happening at segmentation time but actually happens at merge time.

---

### 1B. `0_integrate_sdgi.py` (SDGi policy-side segmentation)

**File:** `1_code/1_preprocess/policy/0_integrate_sdgi.py`

**Algorithm (pseudocode):**
```
for each row in sdgi_corpus.parquet:
    if metadata.language != "en": skip              # language filter
    if len(text) < 80 chars: skip                   # length filter
    labels = parse_labels(row)
    if len(text.words) > MAX_WORDS (300):
        sub_texts = split_long_text(text)           # sentence-boundary split
    else:
        sub_texts = [text]
    
    for sub_text in sub_texts:
        emit segment with labels

def split_long_text(text):
    sentences = re.split(r"(?<=[.!?])\s+", text)
    for sent in sentences:
        accumulate; if count >= TARGET_WORDS (150): finalize
        hard cap: if next sent would push > MAX_WORDS (300): finalize before adding
    finalize remaining
    return [s for s in segments if len(s.words) >= 10]   # post-hoc discard <10 words
```

**Merge unit:** Sentence (genuine sentence-level, not paragraph). Sentence-boundary splitting is the *primary* merge strategy, not a fallback.

**Parameters:** `TARGET_WORDS = 150` (line 32), `MAX_WORDS = 300` (line 33). Note a subtle difference: the SDGi segmenter has an *early-finalize* rule when `count >= TARGET_WORDS` (line 49), while `0_preprocess_policy.py` only finalizes when the *next* paragraph would exceed TARGET_WORDS (lazy merge). So SDGi segments can be exactly 150–300 words, while policy_scrape/manual segments are "at least 150, at most 300" but often larger.

**Post-segment discard:** Segments <10 words discarded in `split_long_text` return (line 54). From disk: 0 segments <10 words, 174 <20 words (these were discarded by `1_build_policy_corpus.py`).

**Severity: Should fix before submission.** The early-finalize difference means SDGi segments are more tightly bounded (150–300) than policy_scrape/manual segments (which can be 150+ with no upper bound enforcement for paragraphs between 150–300 words). The manuscript describes a single "sentence-boundary-aware dynamic segmentation" for all sources, but the implementations differ.

---

### 1C. `0_filter_ungdc_sdg.py` (UNGDC)

**File:** `1_code/1_preprocess/policy/0_filter_ungdc_sdg.py`

**Algorithm (pseudocode):**
```
for each speech file in sessions 70-80:
    paragraphs = split_paragraphs(text)        # split on \n\n or \n, discard <20 words
    relevant = [p for p in paragraphs if SDG_PATTERNS.search(p)]
    if no relevant paragraphs: skip speech
    merged = merge_segments(relevant, TARGET=150, MAX=300)
    
    for segment_text in merged:
        if len(segment_text.words) < 20: skip   # inline discard
        emit segment

def split_paragraphs(text):
    if "\n\n" in text: split on double-newline
    else: split on single-newline (UNGDC format)
    discard paras < 20 words

def merge_segments(paragraphs, target, max_words):
    for para in paragraphs:
        same as 0_preprocess_policy: lazy merge, finalize when next para hits max_words
        early finalize when current >= target
```

**Merge unit:** Paragraph. **No sentence-boundary splitting at all.** Overlong paragraphs are kept whole — there is no `split_long_paragraph` equivalent. UNGDC speeches are almost entirely under 300 words per paragraph, so this rarely matters, but the claim of "sentence-boundary-aware dynamic segmentation" is false for this source.

**Parameters:** `MIN_PARA_WORDS = 20` (line 36), `TARGET_SEGMENT_WORDS = 150` (line 38), `MAX_SEGMENT_WORDS = 300` (line 40).

**Post-segment discard:** Segments <20 words discarded inline (line 207). From disk: 0 segments <10 words, 0 <20 words.

**Severity: Should fix before submission.** Sentence-boundary awareness is absent for this source, contradicting the manuscript claim.

---

### 1D. Regex quality — abbreviation/decimal handling

The boundary regex `re.split(r"(?<=[.!?])\s+", text)` is naive:

- **Abbreviations** like "U.S.", "Dr.", "e.g.", "i.e.", "Mr." will falsely trigger sentence splits
- **Decimals** like "1.5 million", "97.29%" will also trigger splits mid-number
- **Ellipses** "..." after a word will also match

**Quantified impact on SDGi raw texts** (where sentence-splitting is used most):
- 259,880 total "sentences" detected by the regex across all raw SDGi texts
- 15,141 (5.8%) are under 5 words (likely false positives from abbreviation/numeral splits)
- 722 (0.3%) start with a lowercase letter (direct evidence of mid-sentence splits at abbreviations)

**Per-source segments starting with lowercase** (indicates bad splits were persisted to final segments):

| Source | Segments | Lowercase-starting | % |
|---|---|---|---|
| policy_scrape | 9,190 | 957 | 10.4% |
| policy_manual | 7,129 | 821 | 11.5% |
| sdgi_policy | 32,145 | 427 | 1.3% |
| ungdc | 6,472 | 174 | 2.7% |

policy_scrape and policy_manual have high rates (10-11%) because their paragraph-level merge preserves the split boundary artifacts. SDGi (1.3%) and UNGDC (2.7%) are better because short false-positive "sentences" are either re-merged (SDGi's greedy sentence accumulation) or discarded (UNGDC's 20-word filter).

**Severity: Should fix before submission.** The regex is a known weak point. Recommend NLTK `sent_tokenize` or spaCy for sentence boundary detection.

---

## 2. Truncation Check — Reference Corpora (Whole Texts)

### Max sequence lengths

| Model | `max_seq_length` | Source |
|---|---|---|
| `all-mpnet-base-v2` | 384 tokens | `sentence_transformers` model attribute |
| `all-MiniLM-L6-v2` | 256 tokens | same |

**Embedder behavior:** `SentenceTransformer.encode()` silently truncates input to `max_seq_length` via the tokenizer's default `truncation=True`. There is no warning, error, or diagnostic. Verified by encoding a 1,002-token text: the embedding was computed from the first 384 tokens only, with no indication.

### Truncation rates per corpus

| Corpus | Total docs | MPNet (384) truncated | % | MiniLM (256) truncated | % |
|---|---|---|---|---|---|
| OSDG | 30,534 | 0 | **0.0%** | 43 | **0.1%** |
| Benchmark | 616 | 0 | **0.0%** | 1 | **0.2%** |
| Knowledge Hub | 2,221 | 2,057 | **92.6%** | 2,206 | **99.3%** |
| SDGi (reference) | 5,233 | 4,499 | **86.0%** | 4,767 | **91.1%** |
| Aurora | 4,022 | 779 | **19.4%** | 2,098 | **52.2%** |
| Research papers (5k sample) | 5,000 | 1,041 | **20.8%** | 2,845 | **56.9%** |

**Key findings:**
- OSDG and Benchmark suffer negligible truncation (0-0.2%) — these dominate the centroid
- **Knowledge Hub (92.6%) and SDGi reference (86.0%) are massively truncated** on the canonical MPNet model. Their centroid contributions are based on the first ~384 tokens of very long documents, not the full text.
- Aurora (19.4% MPNet) and Research (20.8% MPNet) have moderate truncation
- MiniLM is worse across the board (52–99% for everything except OSDG/Benchmark)

**Implications:**
- The "whole text" claim for reference corpora is false for KH, SDGi, Aurora, and research. These texts are silently truncated, effectively creating an invisible segmentation by token position.
- The centroid for each SDG is computed from truncated text fragments (for KH, SDGi), but the manuscript describes the reference texts as whole, unsplit documents. This is a disconnect between the description and the actual computation.
- The within-SDG semantic gap compares policy segment embeddings (full 150-300 word windows) with research paper embeddings (truncated to 384 tokens ≈ ~300 words). These are actually comparable in length after truncation, but the manuscript describes them as "whole texts."

**Severity: Could undermine a result if uncorrected.** The silent truncation is a hidden architectural choice. If truncation discards SDG-relevant content (e.g., the SDG-specific framing is in the second half of an SDGi VNR document), the centroid is biased toward opening passages. The manuscript should disclose truncation rates and discuss the bias.

---

## 3. Research Corpus (Paper Shards) Handling

### Preprocessing

**File:** `1_code/1_preprocess/preprocess_papers_streaming.py`

Research papers are preprocessed as:
1. Clean title + abstract (unicode, boilerplate removal)
2. `combined_text = f"{title}. {abstract}"` (line 76)
3. Written to shards as complete records (no chunking, no segmentation)

### Embedding

**File:** `1_code/2_embed/0_embed_paper_shards.py`

The embedder reads `combined_text` from each shard and passes it to `model.encode(texts)` (line 176-183). There is no chunking or document splitting. But as shown in Section 2, the tokenizer silently truncates at 384 (MPNet) or 256 (MiniLM) tokens. **20.8% of research papers are truncated on MPNet; 56.9% on MiniLM.**

### Research vs Policy asymmetry

| Property | Research | Policy |
|---|---|---|
| Text unit | Whole abstract (title + abstract) | 150–300 word segments |
| Effective length after truncation | ~300 words (384 tokens MPNet) | ~150–300 words |
| Segmentation | None (truncation only) | Sentence/paragraph merge |
| Truncation rate (MPNet) | 20.8% | N/A (segments already short) |

**The research and policy sides are NOT processed symmetrically.** The metric compares:
- **Research:** A single embedding per paper (potentially truncated) — representing the full abstract's opening portion
- **Policy:** Multiple embeddings per document (each from a dedicated segment) — no truncation needed since segments are short

**Threat to validity:** The semantic gap measures `cosine(research_sub_centroid, policy_sub_centroid)`. If research papers are truncated and policy segments are not, and if SDG-relevant content tends to appear in different positions in each corpus (e.g., policy documents front-load SDG keywords; research abstracts front-load methods), then the gap reflects positional bias as much as semantic divergence.

**Severity: Could undermine a result if uncorrected.** The asymmetry is disclosed in the manuscript (L294-296: "Research papers are single abstract-length texts (mean ~200 words); policy items are 150–300-word segments") but the *truncation* asymmetry is not discussed. Recommend adding truncation rates to the limitations section.

---

## 4. SDGi Dual-Processing — Full Reconciliation

### Source overlap

All SDGi data comes from the same source: `2_data/0_raw/sdgi_corpus/sdgi_corpus.parquet` (5,880 rows).

| Processing path | Output | Records | Selection criteria |
|---|---|---|---|
| Reference (`preprocess_sdgi_corpus.py`) | `sdgi_clean.jsonl` | 5,233 | MIN_WORDS=20, no language filter |
| Policy (`0_integrate_sdgi.py`) | `sdgi_segments.jsonl` | 32,145 | language=en, MIN_TEXT_LEN=80 chars, then segmented |

**Document-level overlap:**
- Reference uses parquet row indices 0–5,232 (5,233 records)
- Policy uses parquet row indices from **all** 5,880 rows (4,225 unique indices → 32,145 segments)
- **Overlap: 3,743 of 5,233 reference rows (71.5%)**
- 1,490 reference-only rows: these are non-English (Spanish, French, etc.) — the policy path filters to English only
- 482 policy-only rows: these come from parquet rows 5,233–5,879 (indices >5,232) that the reference path never processes

**Different cleaning/filtering applied:**

| Step | Reference path | Policy path |
|---|---|---|
| Language filter | None (keeps all languages) | `metadata.language == "en"` |
| Min text length | 20 words (after cleaning) | 80 characters (before cleaning) |
| Cleaning | unicode NFKC, boilerplate, multi-space | **None** (no cleaning at all) |
| Segmentation | None (whole text) | 150-300 word sentences |
| SDG label format | `sdg: int` (single label, **not `sdgs`**) | `sdg_labels: [int]` in output |
| Label per segment | N/A | Full parent list inherited |
| Final count | 5,233 texts | 32,145 segments |

The reference path has `sdg: int` (single integer) while the policy path stores `sdg_labels: list[int]`. However, the raw parquet contains multi-label data. The reference path's `active = sorted(int(l) for l in labels)` captures the multi-label list but writes it to `sdgs` in the code — but the file on disk still has `sdg: int` due to stale data (see below).

**Data staleness issue:** The preprocess script at HEAD writes `"sdgs": active` (list), but the file on disk (`sdgi_clean.jsonl`, last modified June 30) has `"sdg": 1` (int). The code was changed in commit `ae53c98` (July 22) but the data was never regenerated. This means:
- The embedding metadata for SDGi reference has `sdg: int` (not `sdgs`)
- The embedder tries to read `"sdgs"`, gets `None`, and stores `sdgs: None` in the metadata
- The metadata file actually has `sdg: int` as a leftover key from the original record
- `build_centroid_matrix` in `9_loo_sdgi_circularity.py` (line 126) handles this with fallback: `sdg in (r.get("sdgs") or [r.get("sdg")])`, so the circularity check still works

**Severity: Should fix before submission.** The dual-processing paths differ beyond just segmentation — different cleaning, different filtering, different label formats. Any comparison between SDGi-as-reference and SDGi-as-policy is comparing differently preprocessed data.

---

## 5. Label Inheritance Under Segmentation

### Policy sources

All three segmenters copy the full parent label set to each segment:

| Source | Parent label field | Stored as | Inheritance mechanism |
|---|---|---|---|
| `0_preprocess_policy.py` | None (policy docs have no SDG labels) | N/A | N/A |
| `0_integrate_sdgi.py` | `labels` (from parquet, `np.ndarray`) | `sdg_labels: [int]` | Full list copied to every segment (line 117) |
| `0_filter_ungdc_sdg.py` | None | N/A | N/A — uses keyword filtering, no label assignment |

No per-segment relabeling is attempted. When a multi-label SDGi document (e.g., SDGs [1, 10, 11]) is split into 20 segments, all 20 segments inherit `sdg_labels: [1, 10, 11]`.

**Consistency:** Consistent across all segmenters — labels are inherited without modification.

**Severity: Cosmetic** (for current design). If the pipeline were refactored to train on segments, the label inheritance would need review: a 150-word segment of a multi-SDG document may only discuss one SDG, but carries all parent labels as noise.

---

## 6. Cross-Corpus Deduplication

### Manuscript claim

Line 219: "After deduplication by exact text match, \NPolicySegments{} unique segments remained."

### Actual behavior

**`1_build_policy_corpus.py`** (the merge script) performs **no deduplication**. It concatenates the 4 source JSONLs, applies a MIN_WORD_COUNT=20 filter, and writes the merged output. Lines 70-88: load, filter short, append to list. No hash set, no dedup step.

**Duplicate counts:**

| Source | Internal duplicates | Total segments |
|---|---|---|
| policy_scrape | 505 | 9,190 |
| policy_manual | 71 | 7,129 |
| sdgi_policy | 28 | 32,145 |
| ungdc | 0 | 6,472 |

**Merged file:** 54,082 segments, 53,744 unique texts → **338 duplicate records** (exact text match) present in the merged file.

**Cross-source duplicates:** 0. All duplicates are within the same source family (e.g., two segments from different policy_scrape documents that happen to share a paragraph). No duplicates across policy_scrape/policy_manual/sdgi/ungdc.

**Severity: Should fix before submission.** The manuscript explicitly claims dedup happened but it did not. 338 duplicates (0.6% of the corpus) is small but the claim is false, and 338 duplicated segments will double-count those texts in segment-level statistics.

---

## 7. Segment-Cap Sampling (K=50) — Implementation vs. Description

### Implementation

**Files:** `1_code/7_main_analysis/0_shared/semantic_gap_shared.py` (lines 75–98), `1_code/7_main_analysis/1_canonical/1_semantic_gap.py`

Parameters in `semantic_gap_shared.py`:
```python
SEGMENT_CAP_PRIMARY = 50       # line 32
SEGMENT_CAP_SENS_LO = 20       # line 33
SEGMENT_CAP_SENS_HI = 100      # line 34 (defined but never used in 1_semantic_gap.py)
SEGMENT_CAP_SENS_NONE = 10_000_000  # line 35
RANDOM_SEED = 42               # line 37
```

**Algorithm (`cap_policy_indices_per_doc`, lines 75–98):**
1. Group policy segment indices by `source_doc`
2. For each doc: if len <= cap, keep all; else `rng.choice(len, size=cap, replace=False)`
3. `rng = np.random.default_rng(RANDOM_SEED)` (created fresh for each cap run)

**Seed reproducibility:** Yes — `RANDOM_SEED = 42` is a hardcoded constant. The rng is created fresh with `np.random.default_rng(RANDOM_SEED)` for each cap configuration (lines 169, 190). Reproducible.

**Cap is applied per `source_doc` identity:** Yes, using `policy_ids[i]["source_doc"]` field. This is per document, not per source corpus.

**"Per-SDG" cap:** The cap is applied to segments already assigned to a specific SDG (by argmax). So a document with 100 segments assigned to SDG 13 and 50 assigned to SDG 7 could contribute up to 50 segments to each SDG's cluster.

**"median ~14 segments per document" claim (manuscript line 283):** **False.** Actual median is **3 segments per document** across the 2,456 source documents in the merged corpus.

| Stat | Actual |
|---|---|
| Min | 1 |
| Median | **3** |
| Mean | 22.0 |
| Max | 2,887 |
| Docs with 1-10 segs | 2,064 (84.0%) |
| Docs with 11-50 segs | 95 (3.9%) |
| Docs with 51-100 segs | 117 (4.8%) |
| Docs with 101-500 segs | 176 (7.2%) |
| Docs with 500+ segs | 4 (0.2%) |

The "roughly 14" median is wrong by a factor of ~5. The 14 figure may have come from a pre-filtered subset or an older pipeline run.

**Severity: Should fix before submission.** The segment-count descriptive statistic is wrong in the manuscript. The cap of 50 is still conservative (only 4.8% of docs exceed it), but the rationale ("well above the median of roughly 14") is based on a wrong number.

---

## 8. Other Manuscript Claims vs. Code

### 8A. Aurora text count

**Manuscript (line 236):** "Aurora dataset (5,619 expert-validated texts)"  
**Actual:** 4,022 texts after preprocessing (`aurora_texts.jsonl`, confirmed by `wc -l`).  
The preprocessor drops ~1,596 short texts (<20 words). The manuscript uses the raw count, not the post-filter count.  
**Severity: Cosmetic.** The number should reflect the post-filter count actually used.

### 8B. SDGi corpus item count in policy corpus list

**Manuscript (line 213):** "SDGi: 31,971 segments"  
**Actual:** 32,145 segments in `sdgi_segments.jsonl`.  
The manuscript number is 174 segments lower, which is close to the 174 SDGi segments <20 words (discarded by `1_build_policy_corpus.py` but present in the raw segmenter output). The manuscript may be reporting the post-merge-discard count rather than the segmenter output.  
**Severity: Cosmetic.** Close but should be reconciled.

### 8C. SDGi reference metadata format (`sdg` vs `sdgs`)

**Code claim (preprocess_sdgi_corpus.py line 97):** `"sdgs": active` (list of ints)  
**Data on disk:** `"sdg": 1` (single int)  
Embedding metadata files (`sdgi_ids.json`) have `sdg: int` with no `sdgs` field. The embedder (line 159) reads `sdgs`, gets None, and stores `sdgs: None`. Downstream scripts (`build_centroid_matrix`, `9_loo_sdgi_circularity.py` line 126) handle the fallback but this is fragile.  
**Severity: Should fix before submission.** Stale data — data was generated before the code was changed. Re-running the preprocessor would fix it, but then all embeddings, centroids, and scores depend on SDGi reference, so they'd also need regeneration.

### 8D. No dedup performed (cross-reference Section 6)

**Manuscript (line 219):** "After deduplication by exact text match, \NPolicySegments{} unique segments remained."  
**Actual:** No dedup step exists in the merge pipeline. 338 duplicate texts are present.  
**Severity: Should fix before submission.**

### 8E. Sentence-boundary awareness overclaimed (cross-reference Section 1)

**Manuscript (line 218):** "Each source document was segmented into 150–300 word units using sentence-boundary-aware dynamic segmentation"  
**Actual:** Only SDGi uses genuine sentence-level segmentation. Policy_scrape/manual use sentence splitting only as a fallback for paragraphs >300 words. UNGDC has no sentence-boundary awareness.  
**Severity: Should fix before submission.**

### 8F. Silent truncation not disclosed (cross-reference Section 2)

The manuscript discusses "whole texts" (L294: "Research papers are single abstract-length texts") and "150–300-word segments" for policy. But for Knowledge Hub (92.6% truncated), SDGi reference (86.0% truncated), Aurora (19.4%), and research papers (20.8%), the "whole text" is silently truncated to the first ~384 tokens. This is not mentioned anywhere in the methods.  
**Severity: Could undermine a result if uncorrected.**

### 8G. Segment cap sensitivity test seed reuse

All three cap configurations (cap_20, cap_50, cap_none) use `np.random.default_rng(42)` (lines 169, 190, 204 of `1_semantic_gap.py`). Since cap_20 and cap_50 start from the same seed, the first 20 draws in each are identical. This means cap_20 is a strict subset of cap_50 for each document, making the sensitivity test less informative than it appears: it tests "what if we randomly take fewer segments (from the same starting subset)" rather than "what if we take a different random subset."  
**Severity: Cosmetic.** The findings are robust regardless, but the design is less independent than it appears.

---

## Summary of Severity Counts

| Severity | Count |
|---|---|
| Could undermine a result if uncorrected | 2 (truncation, research/policy embedding asymmetry) |
| Should fix before submission | 6 (sentence-boundary overclaim, UNGDC no sentence split, dedup claim false, SDGi label format stale, median segments wrong, regex quality) |
| Cosmetic | 4 (Aurora count, SDGi segment count off by 174, duplicate count small, cap seed reuse) |

---

## Items Not Fully Verifiable

- **Research shard total count:** The manifest was only partially read (timed out on large file). Total research paper count was taken from manuscript (\NResearchPapers{}) rather than independently verified. The truncation check used a 5,000-paper sample.
- **MLP training truncation:** The training pipeline (`2b_supervised_training_singlelabel/`) was not audited. It may tokenize and truncate the reference texts again during training, creating a second truncation event. This should be checked.
- **Score computation (policy_scores.npy):** The scoring step after embedding was not audited. If the scoring process also tokenizes/truncates, the effective text length for scoring may differ from the embedded text length.

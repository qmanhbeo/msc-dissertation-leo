# Segmentation Scoping Report

**Branch:** `supervised-reference` (HEAD: `0d0af3a`)  
**Date:** 2026-07-22  
**Models:** `all-mpnet-base-v2` (canonical, 384 tokens), `all-MiniLM-L6-v2` (robustness, 256 tokens)

---

## 1. Segment-Length Justification Data

### Current whole-text token-count distributions

| Corpus | N | WC median | WC mean | T(MPNet) median | T(MPNet) mean | T(MiniLM) median | T(MiniLM) mean | Trunc@384 (MPNet) | Trunc@256 (MiniLM) |
|---|---|---|---|---|---|---|---|---|---|
| OSDG | 30,534 | 89 | 94 | 113 | 120 | 113 | 120 | **0.0%** | **0.1%** |
| Benchmark | 616 | 92 | 97 | 117 | 123 | 117 | 123 | **0.0%** | **0.2%** |
| Knowledge Hub | 2,221 | 529 | 626 | 693 | 820 | 693 | 820 | **92.6%** | **99.3%** |
| SDGi-ref | 5,233 | 1,111 | 1,371 | 1,539 | 2,071 | 1,539 | 2,071 | **86.0%** | **91.1%** |
| Aurora | 4,022 | 202 | 209 | 264 | 285 | 264 | 285 | **19.4%** | **52.2%** |
| Research (sample 100k) | 100,000 | 204 | 214 | 282 | 315 | 282 | 315 | **23.2%** | **58.2%** |
| Research (all ~2.5M) | ~2,543,698 | — | — | — | — | — | — | **~21%** | **~57%** |

Note: OSDG and Benchmark are already fine (0 truncation on MPNet). They need no change.

### Token/word ratio per corpus

| Corpus | Mean | Median | p95 | p99 |
|---|---|---|---|---|
| OSDG | 1.282 | 1.260 | 1.483 | 1.707 |
| Benchmark | 1.265 | 1.254 | 1.450 | 1.563 |
| Knowledge Hub | 1.314 | 1.310 | 1.417 | 1.495 |
| SDGi-ref | 1.533 | 1.310 | 2.084 | 2.289 |
| Aurora | 1.369 | 1.297 | 1.737 | 2.342 |
| Research | 1.483 | 1.418 | 1.927 | 2.444 |

**Key finding:** Token/word ratio varies widely both across and within corpora. SDGi-ref has the highest p95 (2.08) due to non-ASCII characters in VNR/VLR reports (Unicode normalization doesn't help — these are genuine characters with multi-byte token encodings). Research has high p95/p99 (1.93/2.44) due to concatenated title+abstract with colons, special characters.

**Implication for segmentation:** A fixed word-count threshold cannot guarantee all segments stay under the token limit. For example, at 150 words:
- 95% of OSDG texts map to ≤222 tokens (safe even for MiniLM 256)
- But at p99 ratio 1.707, even 150 words maps to 256 tokens — borderline for MiniLM
- For SDGi-ref, 150 words × p95 ratio 2.08 = 312 tokens — still fits MPNet 384 but barely

### Word-count targets that keep 99%+ under each limit with 64-token margin

**MPNet (limit 384, margin target = 320 tokens):**
For the individual document (before segmentation), we estimate what word-count threshold would produce segments staying under 320 tokens:

| Corpus | What % of records are ≤200 words | Est. safe segment size |
|---|---|---|
| OSDG | 100.0% at ≤200w | OSDG needs NO segmentation (already 100% <320T) |
| Benchmark | 100.0% at ≤200w | Benchmark needs NO segmentation |
| Knowledge Hub | 0.9% at ≤200w | Needs aggressive segmentation. For 99% of segments under 320T: **~100 words** (mean T/W ratio 1.31 → ~130 tokens; p99 T/W ratio 1.50 → ~150 tokens) |
| SDGi-ref | 10.9% at ≤200w | Needs aggressive segmentation. For 99% under 320T: **~100 words** (p95 T/W ratio 2.08 → ~208 tokens; p99 2.29 → ~229 tokens. Margin allows up to ~150 words) |
| Aurora | 49.4% at ≤200w | Borderline. At 150 words (current policy target): 72.7% of records would still truncate. Recommend **~120 words** (p99 T/W ratio 2.34 → ~280 tokens) |
| Research | 48.0% at ≤200w | At 150 words: 75.6% would still truncate. Recommend **~120 words** for safety |

**MiniLM (limit 256, margin target = 192 tokens):**
| Corpus | Est. safe segment size |
|---|---|
| OSDG | Still OK — max tokens 336, but 95% under 192T. Only 0.1% truncate. Within 192T margin → **~150 words** |
| Benchmark | OK — max 269, 0.2% truncate. **~150 words** |
| Knowledge Hub | Needs very aggressive: **~80 words** (p95 T/W 1.42 → ~114T; but p99 1.50 → 120T. At 80 words × 1.31 → ~105T — safe) |
| SDGi-ref | Even with 100 words, p95 ratio 2.08 → 208T > 192. **~90 words** |
| Aurora | At 120 words: p99 ratio 2.34 → 280T. **~80 words** for safety |
| Research | At 120 words: p99 ratio 2.44 → 293T. **~80 words** for safety |

**Recommendation:** Since different corpora have different T/W profiles, the segmentation should be **token-count-aware** (use the actual model tokenizer), not word-count-based. The policy side's current 150-word target is too aggressive for the reference corpora. A safer approach: segment at the token level with `model.tokenizer.encode()` and split when the cumulative token count approaches `max_seq_length - margin` (e.g., 320 for MPNet, 192 for MiniLM), respecting sentence boundaries.

### Bloat estimate under segmentation

If each document is split into segments not exceeding 320 tokens (MPNet) or 192 tokens (MiniLM), the segment count multipliers are:

| Corpus | Median words → Est. segments (MPNet 320T) | Est. segments (MiniLM 192T) |
|---|---|---|
| OSDG | 89w → 1 segment | 89w → 1 segment |
| Benchmark | 92w → 1 segment | 92w → 1 segment |
| KH | 529w → 3–4 segments | 529w → **5–7 segments** |
| SDGi-ref | 1,111w → 7–9 segments | 1,111w → **12–16 segments** |
| Aurora | 202w → 1–2 segments | 202w → 2–3 segments |
| Research | 204w → 1–2 segments | 204w → 2–3 segments |

MiniLM roughly doubles the segment count for long documents.

### Training corpus size after segmentation

| Corpus | Current | After MPNet-seg | After MiniLM-seg |
|---|---|---|---|
| OSDG | 30,534 | 30,534 | 30,534 |
| Benchmark | 616 | 616 | 616 |
| KH | 2,221 | ~7,500 | ~12,500 |
| SDGi-ref | 5,233 | ~40,000 | ~70,000 |
| Aurora | 4,022 | ~5,500 | ~9,000 |
| **Training total** | **42,626** | **~84,000** | **~123,000** |

Research papers are not in training (unlabeled). They only affect scoring + centroid computation.

### Research corpus after segmentation

| Model | Segments (est.) |
|---|---|
| MPNet (320T cap) | ~3.3M (from 2.5M documents, ~1.3× multiplier) |
| MiniLM (192T cap) | ~5.0M (from 2.5M documents, ~2× multiplier) |

---

## 2. Scripts Requiring Changes

### Preprocessors (need token-count-aware segmentation added)

| Corpus | Current script | Current behavior | Change needed |
|---|---|---|---|
| **Knowledge Hub** | `preprocess_sdg_knowledge_hub.py` | Whole-text JSONL (field: `text`). No segmentation. | Add segmentation stage. Add `segment_id` and `source_doc` fields. Keep `sdgs: [int]` unchanged. |
| **SDGi-reference** | `preprocess_sdgi_corpus.py` | Whole-text JSONL (field: `text`). `sdg` (stale) or `sdgs` (after re-run). | Same as KH. |
| **Aurora** | `preprocess_aurora.py` | Whole-text JSONL (field: `text`). `sdgs: [int]`. | Same as KH. |
| **Research papers** | `preprocess_papers_streaming.py` | Whole-record JSONL per shard (field: `combined_text`). No SDG labels. | Add segmentation stage. Add `segment_id` and `source_doc` fields. |

### Embedders (already generic, possibly no changes)

| Script | Current behavior | Impact of segmentation |
|---|---|---|
| `0_embed_reference_corpora.py` | Reads any JSONL, extracts `text_field`, embeds, saves `.npy` + `_ids.json`. Currently processes all 5 reference corpora + policy. | **No change needed.** The script is already generic. If the preprocessed files are just longer (N rows per document instead of 1), the embedder produces N embeddings per document — exactly what we want. The only subtlety: each segment needs its own `id` field (e.g., `${source_doc}_seg_${i}`) for the metadata. |
| `0_embed_paper_shards.py` | Reads `combined_text` from each shard JSONL, embeds, saves per-shard `.npy`. | **Would need change.** If research papers are segmented, the input format changes from 1 record = 1 combined_text to N records per paper with `combined_text` being shorter. The embedder is currently tied to the `combined_text` field. Would need to either: (a) accept a different JSONL format with `text` + `segment_id` + `source_doc`, or (b) keep the segmentation at embedding time rather than preprocess time. |

### Training data preparation (`0_prepare_data.py`)

Currently reads whole-text embeddings + JSONL for 5 corpora (OSDG, Benchmark, KH, SDGi, Aurora). Expects 1:1 correspondence between embeddings and JSONL records.

**Change needed:** After segmentation, each document produces multiple embeddings. The script would naturally handle this because it loads all embeddings and all JSONL records in order. **No logic change needed** — adding more records (segments) to the input would just mean more training data. The stratified split by SDG still works.

**But:** The split is currently record-level, not document-level. If two segments from the same parent document end up in different splits (train vs test), that's **sibling-segment leakage**. Fixing this requires grouping by `source_doc` before the split — a non-trivial change to `0_prepare_data.py`.

### Downstream analysis scripts

| Script | Current research handling | Research segment-aware? | Change needed |
|---|---|---|---|
| `alignment_core.py` — `build_research_centroids` | Mean-pools all research embeddings per SDG (by argmax of MLP score). | No — treats each embedding independently. | If research becomes multi-segment, this function still works (each segment contributes independently to the centroid). If per-paper pooling is desired (mean-pool segments per paper first), needs new logic. |
| `1_semantic_gap.py` | Loads pre-computed `research_centroids.npy` (17, dim). Caps policy segments per doc. | Research side: no. Policy side: yes (via `cap_policy_indices_per_doc`). | If research centroids still pre-computed and just passed in: **no change**. If research needs per-paper capping: needs update to compute or load per-paper centroids. |
| `semantic_gap_shared.py` — `compute_sdg_semantic_gaps()` | Takes research centroids + counts + cohesions as pre-computed arrays. Caps policy segments. | No. | **No change needed** — function is agnostic to how research centroids were built. |
| `semantic_gap_shared.py` — `cap_policy_indices_per_doc()` | Caps by `source_doc`. | N/A — policy only. | **Generic function** — reusable for research if research segments also have `source_doc` field. |
| `0_coverage_gap.py` | Aggregates research scores per shard, no segmentation concern. | No. | **No change** — operates on scores, not embeddings. |
| `9_loo_sdgi_circularity.py` | Loads `paper_scores_shards/` (not research embeddings). Groups by segment prefix. | Research side: no. Policy side: yes. | **No change** — LOO only groups policy segments. Research papers are always whole-text here. |
| `6_sample_stability.py` | `build_research_centroids()` uses per-paper accumulated `vector_sums`. | No. | If research is multi-segment, the per-paper accumulation approach needs rethinking, or segments are treated independently. |

### Required changes summary (minimal set)

1. **4 preprocessors** — add token-count-aware segmentation, add `segment_id` + `source_doc`
2. **`0_embed_paper_shards.py`** — adapt to accept segmented research records (input format change)
3. **`0_prepare_data.py`** — add document-level grouping for the train/test split to prevent sibling leakage
4. **`alignment_core.py` or consumer** — decide pooling strategy for research segments

---

## 3. Document-Level Grouping Requirements

### Which of the 4 corpora participate in train/val/test splits?

| Corpus | In training? | In evaluation split? | Notes |
|---|---|---|---|
| Knowledge Hub | **Yes** — 0_prepare_data.py, stratified 85/15 per-source | Test set (15%) used by 1_train_models_MLP.py as hidden eval (not loaded — only train indices loaded) | **Sibling leakage risk** if segments from the same parent document cross the 85/15 split boundary |
| SDGi-reference | **Yes** — same as above | Same | **Sibling leakage risk** |
| Aurora | **Yes** — same as above | Same | **Sibling leakage risk** |
| Research papers | **No** — never in training, only in scoring + centroid building | No split — all papers used for centroid | **No leakage risk** — research only affects centroids, not training |

### Current split logic

In `0_prepare_data.py` (lines 159-186):
```python
all_idx = np.arange(len(embeddings))
for src in np.unique(sources):
    mask = sources == src
    src_idx = all_idx[mask]
    src_y = labels[mask]
    y_int = src_y.argmax(axis=1)
    s_train, s_test = train_test_split(
        src_idx, test_size=0.15, random_state=42, stratify=y_int,
    )
```

The split is at the **row index level** with stratification by argmax SDG label. There is no grouping or awareness that multiple rows might belong to the same parent document.

In `1_train_models_MLP.py` (lines 96-102), validation split within training:
```python
n_val = max(1, int(len(X) * 0.1))
perm = torch.randperm(len(X_t), ...)
val_idx = perm[:n_val]
train_idx = perm[n_val:]
```

Also **row-level only**, no document awareness.

### Fix required for safe segmentation

If KH, SDGi-ref, or Aurora are segmented, `0_prepare_data.py` must be changed to:

1. Group records by `source_doc` (a new field that the segmenter would add)
2. Keep all segments of the same parent document in the same split
3. Perform the stratified split at the document level, then expand to segment-level indices

This is the same `cap_policy_indices_per_doc` pattern but for "all or nothing" grouping rather than capping. The existing function groups by `source_doc` but caps — a new variant would do "all or none" placement.

### Loop groups that would also need grouping

- **LOO (`9_loo_sdgi_circularity.py`):** Currently groups policy segments by `segment_id` prefix `"sdgi_"`. If SDGi-reference is segmented for the centroid, the LOO would need to exclude the SDGi-reference centroid contributions from documents that also appear in policy (the 71.5% overlap). This is already complex logic — adding segmentation would require tracking `source_doc` across the train/eval boundary.

- **Sample stability (`6_sample_stability.py`):** Bootstraps research papers per SDG. If research is multi-segment, the bootstrapping needs to sample documents (not segments) to avoid inflating representation of multi-segment documents.

---

## 4. Research-Paper Pooling Decision (Flag, Not Decide)

### The problem

If research papers are segmented (each paper → N ≤ 320-token segments), there are two ways to compute the per-SDG research centroid:

**Option A — Independent segments:** Each segment is treated as an independent "paper." The centroid for SDG j is the mean of all segment embeddings assigned to SDG j. A paper with 3 segments contributes 3 times to the centroid.

**Option B — Per-paper pooled:** Each paper contributes exactly one embedding to the centroid, computed by mean-pooling its N segment embeddings. The current centroid logic (one paper = one vector) is preserved.

### Impact on files

| File | Option A (independent segments) | Option B (per-paper pooled) |
|---|---|---|
| `alignment_core.py` — `build_research_centroids` | **No change.** Already treats each embedding independently. | **Needs change.** Must group by `source_doc`, mean-pool per paper, then build centroid. |
| `semantic_gap_shared.py` — `compute_sdg_semantic_gaps()` | **No change.** Research centroids are pre-computed. | **No change.** Still pre-computed (now per-paper pooled internally). |
| `semantic_gap_shared.py` — `cap_policy_indices_per_doc()` | Not relevant to research. | Could be **reused** with `"source_doc"` key for pre-centroid per-paper capping. |
| `1_semantic_gap.py` | **No change.** | **Minor change** — would need to cap research segments per paper before centroid building, analogous to policy capping. |
| `6_sample_stability.py` — `build_research_centroids()` | Draw accumulation still works per segment. | **Needs change** — draw must accumulate per paper, not per segment. |
| `0_pca_semantic_landscape.py` | **No change.** Loads pre-computed centroids. | **No change.** |

### Reusability of existing capped-sampling code

The function `cap_policy_indices_per_doc` in `semantic_gap_shared.py` (lines 75–98) is **fully generic**:

```python
def cap_policy_indices_per_doc(
    policy_idxs: list[int],
    policy_ids: list[dict],
    segment_cap: int,
    rng: np.random.Generator,
) -> list[int]:
```

It groups indices by `policy_ids[i]["source_doc"]` and caps per document. To reuse for research:
1. Research segments need a `"source_doc"` field in their metadata dicts
2. Call with `policy_idxs` = research segment indices and `policy_ids` = research segment metadata
3. Use `segment_cap=1` for "one segment per document" (Option B equivalent)
4. Or use `segment_cap=SOME_K` for "K segments per document" (intermediate between A and B)

**Do not pick a strategy here.** Both are defensible. Option A is simpler and matches the policy side's treatment (segments independent). Option B avoids weighting multi-segment papers more heavily in the centroid.

---

## 5. Re-Embedding / Retraining Blast Radius

### Which corpora feed MLP training?

| Corpus | In training? | Retrain needed if re-segmented? |
|---|---|---|
| OSDG | Yes | **No** (0% truncated already, segmentation would not change it) |
| Benchmark | Yes | **No** (already fine) |
| Knowledge Hub | **Yes** | **Yes** — segmentation changes both embeddings and labels structure |
| SDGi-reference | **Yes** | **Yes** — segmentation changes everything |
| Aurora | **Yes** | **Yes** — segmentation changes training records |
| Research | **No** (unlabeled) | **No retrain needed** — only rescore + rebuild centroids |

### Embedding generation throughput

Measured on this machine (CPU, WSL):

| Model | Throughput | Notes |
|---|---|---|
| MiniLM (batch 128) | ~287 texts/s | Reference embedder, batch_size=128 |
| MPNet (batch 128) | ~81 texts/s | Reference embedder, batch_size=128 |
| MiniLM (batch 256) | ~330 texts/s (est.) | Paper shard embedder, batch_size=256 |
| MPNet (batch 256) | ~95 texts/s (est.) | Paper shard embedder, batch_size=256 |

### Embedding time estimates

**Training corpora (re-segment + re-embed + retrain):**

| Corpus | Current N | N after MPNet-seg | MiniLM embed time | MPNet embed time |
|---|---|---|---|---|
| OSDG | 30,534 | 30,534 | 1.8 min | 6.3 min |
| Benchmark | 616 | 616 | ~2s | ~8s |
| KH | 2,221 | ~7,500 | ~26s | ~1.5 min |
| SDGi-ref | 5,233 | ~40,000 | ~2.3 min | ~8.2 min |
| Aurora | 4,022 | ~5,500 | ~19s | ~1.1 min |
| **Training total** | **42,626** | **~84,000** | **~5 min** | **~17 min** |

**Research corpus (re-segment + re-embed + rescore):**

| Model | Est. segments | Embed time |
|---|---|---|
| MiniLM (256T cap) | ~3.3M segments | ~190 min (~3.2 hours) |
| MPNet (320T cap) | ~3.3M segments | ~680 min (~11.3 hours) |
| MiniLM (192T cap for robustness) | ~5.0M segments | ~290 min (~4.8 hours) |

Note: Research embedding is the dominant cost by far. The paper shard embedder uses batch_size=256 and streams through shards, which is already efficient.

### MLP training time

| Training set size | Est. time (1 combo × 5 folds) |
|---|---|
| Current 42,626 records | ~35s (observed in prior run) |
| After MPNet-seg ~84,000 records | ~70s |
| After MiniLM-seg ~123,000 records | ~100s |

Training time is negligible compared to embedding.

### Scoring (MLP inference) time

| Corpus | N after MPNet-seg | Time (MPNet MLP inference) |
|---|---|---|
| Policy segments | 54,082 | ~5s |
| Research (segmented) | ~3.3M | ~15s per minibatch, ~3 min total |

Scoring is fast (simple matrix multiplication through 4-layer MLP).

### LOO circularity re-check time

The LOO currently requires re-running `build_centroid_matrix()` which loads all reference corpora + policy segments + research shards. With 2.5M research papers (even segmented), this would be I/O-bound. Estimated: ~30 min (mostly loading shards from disk).

### Full cycle time estimate

#### Scenario A — MPNet only (canonical model)

| Step | Est. time | Notes |
|---|---|---|
| Re-segment 4 corpora (preprocess) | ~10 min | Disk I/O bound |
| Re-embed training corpora (MPNet) | ~17 min | 84k records |
| Re-embed research (MPNet) | ~11.3 hours | 3.3M segments — **dominant cost** |
| Retrain MLP (MPNet) | ~1 min | Negligible |
| Rescore research | ~3 min | Fast |
| Rescore policy | ~5s | Already scored |
| Recompute centroids | ~30 min | I/O bound for research shards |
| Re-run semantic gap | ~2 min | Fast |
| Re-run LOO circularity | ~30 min | I/O bound |
| Re-run sample stability | ~30 min | Research bootstrap |

**Total: ~13–14 hours**, of which ~11 hours is MPNet embedding of research papers.

#### Scenario B — MiniLM only (for robustness tests)

| Step | Est. time |
|---|---|
| Re-segment 4 corpora | ~10 min |
| Re-embed training (MiniLM) | ~5 min |
| Re-embed research (MiniLM) | ~3.2 hours |
| Retrain MLP | ~2 min |
| Rescore + centroids + gap + LOO | ~1 hour |

**Total: ~4.5 hours.**

#### Scenario C — Both models (canonical + robustness)

Research dominates: ~11.3h (MPNet) + ~3.2h (MiniLM) + ~1h overlap = **~15 hours total**.

### Biggest uncertainty

**Research paper embedding on MPNet** is by far the largest uncertainty:
- 2.5M papers → ~3.3M segments after token-count-aware segmentation
- At 81 texts/s (batch 128), ~11.3 hours
- Actual throughput could vary ±30% depending on text length distribution, memory pressure, disk I/O for shard reads
- The batching in `0_embed_paper_shards.py` uses batch_size=256, which may improve throughput beyond the measured 81/s at batch=128
- Running out of memory and triggering swap could dramatically slow the process

**Mitigations:**
- Can parallelize: run MiniLM and MPNet embedding simultaneously (different output directories) — they don't conflict
- Can use batch_size=256 to improve MPNet throughput (est. 95 texts/s → ~9.6 hours)
- The research embedder is already shard-based and resumable — can restart from the last completed shard if interrupted

---

## 6. SDGi Consolidation Side-Effect

### How unification collapses dual paths

Currently, SDGi flows through two completely independent paths:

| | Reference path | Policy path |
|---|---|---|
| Preprocessor | `preprocess_sdgi_corpus.py` | `0_integrate_sdgi.py` |
| Output | `sdgi_clean.jsonl` (5,233 records) | `sdgi_segments.jsonl` (32,145 segments) |
| Language filter | None | `metadata.language == "en"` only |
| Min length | 20 words | 80 characters |
| Cleaning | `clean_text()` (unicode + boilerplate) | **None** |
| Segmentation | None (whole text) | 150-300w sentence-level |
| Label format | `sdg: int` (stale) → should be `sdgs: [int]` | `sdg_labels: [int]` (same as `sdgs`) |
| Labels inherited | N/A (per-record) | Per segment: full parent list |

After unification, SDGi would have a **single preprocessing path**:
1. One preprocessor with: language filter (Y/N?), cleaning (Y/N?), segmentation (token-count-aware to 320/192 tokens), label inheritance
2. One embedded file: `sdgi_segments.npy` (replaces both `sdgi.npy` and the policy-side SDGi segments embedded within `policy.npy`)
3. One set of metadata: with `source_doc` for grouping

### Remaining differences that need explicit reconciliation

Even after segmentation is unified, these differences must be resolved:

**1. Language filter** (cosmetic)
- Currently: reference keeps all languages; policy keeps only English
- Question: should the unified SDGi corpus keep non-English texts for training? Or filter to English only?
- If kept, non-English texts contribute to the English-model centroid with no English content (semantic noise)
- If filtered, ~1,490 non-English texts are removed (28% of reference)

**2. Cleaning** (should fix)
- Currently: reference path applies `clean_text()` (unicode NFKC, strip boilerplate); policy path does NO cleaning
- Question: should the unified preprocessor clean or not?
- Cleaning is beneficial (removes URLs, emails, copyright footers that carry no SDG signal)
- Recommendation: clean (consistency with reference path, which is the primary source)

**3. Multi-label handling for training** (architectural)
- Currently: reference path stores single label per text (`sdg: int`, first label of multi-label row); policy path stores full list
- After segmentation: each segment inherits the full parent label list
- `0_prepare_data.py` filters for single-label records (`len(sdgs) == 1`)
- A multi-label document with 10 segments, all carrying SDGs [1, 3], would produce 10 single-label filter passes — each would be filtered OUT because `len([1, 3]) != 1`
- This means **multi-label documents contribute zero training records** after segmentation, even though each individual segment contains relevant content
- This is a pre-existing issue (multi-label texts are already dropped) but it's costlier now (a 10-segment multi-label document drops 10 potential training records instead of 1)

**4. Segment-level vs document-level labels for training** (architectural)
- Even for single-label documents: a 10-segment document carries one label (e.g., SDG 13), but some segments may discuss other topics
- Currently, each segment gets the full parent label — meaning segment 5 of 10 gets SDG 13 label even if it discusses budgeting
- This is labeling noise that affects training quality
- One could argue this is acceptable (same as policy semantic gap analysis), but it introduces label noise not present in the current whole-text approach

**5. LOO circularity after consolidation**
- Currently, LOO checks whether SDGi-as-policy is circular by removing SDGi reference from the centroid and re-scoring SDGi policy segments against the SDGi-excluded centroid
- After consolidation: SDGi segments are no longer split across two paths. The LOO would check whether unified SDGi segments are circular with the (SDGi-excluded) centroid
- This is **cleaner** — the LOO now operates on a single SDGi corpus with consistent segmentation
- However, any non-English SDGi texts in the centroid would not be comparable to English-only tests

### Migration path recommendation

1. Write a single `preprocess_sdgi_unified.py` that replaces both `preprocess_sdgi_corpus.py` and `0_integrate_sdgi.py`
2. Apply: English filter, `clean_text()`, token-count-aware segmentation (320T for MPNet), full label inheritance
3. Remove SDGi from policy corpus merge (`1_build_policy_corpus.py`)
4. Remove SDGi from reference embedder corpora list in `0_embed_reference_corpora.py`
5. Add unified SDGi as one corpus in the training data prep

---

## Total Time Estimate Summary

| Scenario | Total wall-clock | Notes |
|---|---|---|
| **A — MPNet only** (canonical) | **13–14 hours** | Research MPNet embed dominates (~11h) |
| **B — MiniLM only** (robustness) | **4–5 hours** | Research MiniLM embed (~3.2h) |
| **C — Both models** | **15–17 hours** | Can parallelize MPNet + MiniLM research embeds |
| **C' — Both, no research segmentation** | **3–4 hours** | If research papers remain whole-text (truncated), only training corpora are segmented |

### Largest uncertainty

**Research paper embedding on MPNet (~11 hours):**
- Throughput variance ±30% depending on actual text lengths, memory pressure
- 2.5M papers × ~1.3 segments per paper = ~3.3M segments at 81 texts/s
- If the machine runs out of RAM (16GB with 2.5M records loaded), swap could multiply time by 5–10×
- The shard-based embedder mitigates this (loads one shard at a time), but each shard is 100k records
- **Recommendation:** Run research embedding overnight; use the MiniLM version as a fast first-pass to validate results

### Quickest win (not requested, but worth noting)

If the goal is just to fix truncation for the **canonical results** without touching research papers:

- Segment only KH, SDGi-ref, Aurora, OSDG, Benchmark (training corpora)
- Leave research papers whole-text (accept their 21% truncation rate as a known limitation)
- Re-embed training corpora (MPNet): ~17 min
- Retrain: ~1 min
- Rescore kown research (no change): 0
- **Total: ~20 min of work + overnight for verification**

This eliminates truncation from the training side and reduces the main threat to validity. The 21% research truncation is less severe than 86-93% KH/SDGi truncation and could be disclosed as a limitation.

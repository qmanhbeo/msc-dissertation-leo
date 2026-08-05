# Handoff — research-corpus paper-level weighting ("Plan C")

**Date:** 2026-08-05
**Branch/HEAD at session start:** `ed0e2f4` (clean tree)
**State (updated 2026-08-05, resumed session):** Core weighting is WIRED and the **MPNet trial is COMPLETE and GREEN** (see §8 Progress Log). Code committed; MPNet `4_outputs` regenerated and verified against a `/tmp` snapshot — headline findings preserved, the 3.1M-abstract bug fixed. Remaining: downstream label fixes (Step 6), S1 subset rebuild (Step 7/9), full 3-encoder replay + PDF (Step 10/11), prose (Step 11).

---

## 1. Context — where we are

### The reported issue
The manuscript claims (`3_writing/dissertation.tex:90`):

> "…over `\NResearchPapers{}` AI-for-sustainability abstracts…"  → renders as **3,105,144 abstracts**

That is wrong. **3,105,144 is the number of SEGMENTS.** The corpus contains **2,536,771 abstracts**. The macro `\NResearchPapers` is populated from `research["n_rows"]` — a row count of the score shards, i.e. segments.

### Why it is not just a wording bug
Investigating the mislabel surfaced a real methodological asymmetry:

- The **policy** side is *document-weighted* (Assumption A19): each source document's segment score vectors are averaged, then hard-assigned. One document = one unit.
- The **research** side was *segment-weighted*: each segment was hard-assigned independently. A 33-segment abstract counted 33×, a 1-segment abstract counted 1×.

The manuscript asserts the opposite in two places, both factually false today:
- `:206` — "research abstracts (mean ~200 words, well below the 374-token budget) pass through as single segments, so the research-side text unit is the full abstract"
- `:232` — "The research corpus uses the same logic at the paper level, where each of the `\NResearchPapers{}` abstract texts is already a single unit"

### The chosen fix — "Plan C" (user-approved)
Make the research side genuinely document-weighted, symmetric with policy, and relabel every unit in code, outputs, figures and prose. Three sub-decisions were put to the user and answered:

| Q | Decision |
|---|---|
| **Q1** MiniLM/SciBERT 100k sensitivity subset (currently 100,000 *segments* → partial papers) | **S1**: redefine as 100,000 *papers* → all their segments (~122k), re-embed both encoders, re-snapshot |
| **Q2** Which abstract count is canonical in prose | **2,536,771** (scored corpus), footnote the 6,927 dropped at segmentation |
| **Q3** INLP register direction `G` | **Keep segment-level**; record the choice in run provenance |

---

## 2. Key facts — so you don't have to re-derive them

### 2.1 Ground-truth numbers (all independently verified this session)

| Quantity | Value | How verified |
|---|---|---|
| Preprocessed research records | 2,543,698 | `2_data/1_preprocessed/research/metadata/manifest.json` → `totals.rows` |
| Dropped at segmentation | 6,927 (0.27%) | derived; cause explained in §2.2 |
| **Abstracts in the scored corpus** | **2,536,771** | (a) `grep -c '"segment_index": 0,'` across all 26 segment shards; (b) independently, the new `aggregate_research_scores(unit="document")` — **both agree exactly** |
| **Segments** | **3,105,144** | `2_data/2_segmented/research/metadata/manifest.json` → `totals.rows` |
| Segments per abstract | 1.2241 | derived |
| Abstracts with >1 segment | 435,905 = **17.18%** | measured (NOT the ~23% you may see quoted from the truncation report — that is the % of texts exceeding 384 tokens; some overflow tails are <20 words and get dropped rather than becoming a second segment) |
| Max segments in one abstract | **33** | measured |
| Policy segments / documents | 40,597 / 6,367 | `4_outputs/mpnet/tables/num2_coverage_gap.tex` |
| research:policy in **segments** | 76:1 | 3,105,144 / 40,597 — this is what the manuscript's "76:1" actually is |
| research:policy in **units** | **398:1** | 2,536,771 / 6,367 — the correct ratio once coverage is paper-weighted |

### 2.2 Why preprocessed (2,543,698) ≠ segmented parents (2,536,771)

`1_preprocessed/` was never the final corpus. **The 20-word minimum lives in the segment stage, not preprocess:**

- Preprocess filter: `--min-abstract-chars 30` — **characters**, on the abstract alone (`1_code/1_preprocess/0_preprocess_papers_streaming.py:75,246`).
- Segment filter: `MIN_SEGMENT_WORDS = 20` — **words**, on the whole `combined_text` (`1_code/2_segment/1_segment_corpus.py:182`) and again per emitted segment (`segment_utils.py:99-118`).

Empirically confirmed on shard 1 (100,000 preprocessed rows → 99,697 parents, 303 lost):
- **292** have `combined_text` < 20 words. Example: `"Predictability of drug-induced liver injury by machine learning. This article was reviewed by Maciej Kandula and Paweł P. Labaj."` (19 words) — a 30-char abstract passes preprocess; a 20-word text does not pass segmentation.
- **11** are CJK-body abstracts (English title + Chinese/Japanese abstract). NLTK cannot emit a ≥20-whitespace-word segment → zero segments. **Side finding worth a footnote: the preprocess English filter lets mixed-language records through.**

This is a filter-boundary artefact, not a bug — but `dissertation.tex:293` misattributes the 20-word minimum to preprocessing and must be corrected.

### 2.3 Measured effect of paper weighting (MPNet, full corpus)

| SDG | segment % | document % | Δ |
|---|---|---|---|
| 3 | 27.675 | 25.717 | **−1.959** |
| 9 | 18.609 | 20.104 | **+1.495** |
| 4 | 15.136 | 15.597 | +0.460 |
| 11 | 4.718 | 5.141 | +0.423 |
| 15 | 2.515 | 2.194 | −0.321 |
| 16 | 10.863 | 11.168 | +0.305 |
| 13 | 3.915 | 3.705 | −0.210 |
| 7 | 3.847 | 4.054 | +0.207 |
| *(others)* | | | \|Δ\| < 0.15 |

`mean_top_overall`: 0.635680 (segment) → 0.627009 (document).

**The headline top-3 ordering (SDG 3 > 9 > 4) is PRESERVED.** The derived macro `\ResearchSdgFourPlusSdgNinePct` changes 33.7 → 35.70.

### 2.4 Architecture facts that make this cheap

- **`openalex_id` is already present in every embedding-shard and score-shard `*_ids.jsonl`.** Paper grouping needs **no re-segmentation and no new pipeline stage**.
- **Segments of one abstract are contiguous within a shard**, and **no abstract spans a shard boundary** (segmentation is per preprocessed shard). Both are now asserted in code, fail-closed.
- `np.add.reduceat` over contiguous runs makes the whole 3.1M-row grouped aggregation take **8.5 s**.
- Warm replay **starts at scoring** (`main.py:801` `run_main_text` → `_run_main_analysis_steps`); embedding is *not* part of it. The S1 re-embed is therefore an explicit out-of-band step plus a re-snapshot.

### 2.5 Paths you will need

```
2_data/2_segmented/research/                        26 shards, 3,105,144 segments (~18 GB)
2_data/2_segmented/research/metadata/manifest.json
2_data/2_segmented/research_subset/                 the 100k SEGMENT sample (to be replaced by S1)
2_data/3_embedded/{minilm,scibert}/research_shards/ subset embeddings ONLY (MPNet's full corpus is elsewhere)
2_data/5_supervised_scored/mpnet/paper_scores_shards/{part-*.npy, metadata/part-*_ids.jsonl, metadata/manifest.json}
2_data/5_supervised_scored/{model}/research_centroids.npy
2_data/5_supervised_scored/{model}/metadata/research_centroid_meta.json
2_data/3b_register/{mpnet/canon, minilm/subset, scibert/subset}/G.npy
4_outputs/{mpnet,minilm,scibert}/tables/num2_coverage_gap.tex   ← where \NResearchPapers is emitted
4_outputs/mpnet/data/concept/tables/num2_coverage_gap.tex
```

Current `\NResearchPapers` values: mpnet `3,105,144`; minilm `100,000`; scibert `100,000`; concept `111,541`. **All four are segment counts mislabelled as papers.**

---

## 3. What was done this session, and why

### 3.1 Files changed (2 files, uncommitted)

#### `1_code/7_main_analysis/0_shared/model_utils.py`
Added the named constants that replace the magic behaviour, each with a documented rationale:
- `RESEARCH_SUBSET_PAPERS = 100_000` — **renamed from `RESEARCH_SUBSET_SIZE`** because the draw becomes paper-based (S1).
- `RESEARCH_WEIGHTING_UNIT = "document"` — selects the canonical research unit.
- `RENORMALISE_DOC_VECTORS = True` — a paper vector is the mean of its unit-norm segment vectors, renormalised to unit length so every paper contributes exactly one unit of mass, and so the `raw_centroid_norm == mean_cos_to_centroid` identity in the centroid metadata still holds.
- `INLP_RESEARCH_UNIT = "segment"` — Q3 decision, recorded so it is auditable from output provenance, not only from source.

#### `1_code/7_main_analysis/0_shared/research_score_shards.py` (substantially rewritten)
New public surface:
- `read_shard_paper_ids(ids_path)` — per-row `openalex_id`; raises if any row lacks one (no silent fallback).
- `paper_run_starts(paper_ids)` — start offsets of runs of equal consecutive ids.
- `assert_papers_contiguous(...)` — fails closed if `#runs != #distinct ids`.
- `group_rows_by_paper(scores, starts)` — `np.add.reduceat` mean per paper + segment counts.
- `_ProfileAccumulator` — running hard/soft coverage aggregates.
- `aggregate_research_scores(manifest_path, scored_dir, unit=RESEARCH_WEIGHTING_UNIT)` — one streaming pass returning **both** granularities, plus a cross-shard boundary assertion.

**Changed return contract (this is what breaks callers):**
- removed `n_rows`
- added `unit`, `n_segments`, `n_papers`, `papers_multi_segment`, `segments_per_paper_max`
- canonical profile under unprefixed keys (`hard_counts`, `hard_profile`, `soft_profile`, `mean_top_overall`, `mean_top_per_sdg`) — now **paper-level**
- segment-level diagnostic under `segment_*`-prefixed keys

### 3.2 Verification performed (all passed)

1. **Brute-force groupby agreement**, shard 1: 125,466 rows → 99,697 papers. `np.add.reduceat` means vs an `OrderedDict` reference: **max abs diff 0.0**, counts identical, argmax identical.
2. **Contiguity + no cross-shard papers**: asserted across all 26 shards, no violation.
3. **Cross-validated abstract count**: pipeline grouping returns 2,536,771 — **exactly** the independent `grep` count. Two unrelated methods agree.
4. **Totals**: `n_segments == 3,105,144` ✓.
5. **Runtime**: 8.5 s for the full grouped aggregation.

Scratch artefacts (outside the repo, per AGENTS): `/tmp/opencode/pfcheck/{shard1.py,full.py,full.log,doc_profile.npy,seg_profile.npy}`, `/tmp/opencode/segcount.txt`.

### 3.3 A plan error that was caught and corrected — do not repeat it

An earlier draft of the plan said to run `python main.py --stage segment --overwrite` and `--stage embed --overwrite` for the S1 subset rebuild. **That is destructive.** `--overwrite` is forwarded to `1_segment_corpus.py` (bypasses the per-shard existence guard → re-segments all 3.1M texts) and to `0_embed_paper_shards.py` for **all three** encoders (re-embeds MPNet's full corpus). The correct approach is delete-and-refill **without** `--overwrite` — see §6 Step 6.

---

## 4. Interrupted work — exactly where the cut happened

The session was stopped **mid-way through step 3 of the plan** (paper-level centroids in `score_supervised.py`).

**The last edit applied** was to `research_score_shards.py`: the import block was extended to
```python
from model_utils import (
    N_SDG,
    RENORMALISE_DOC_VECTORS,
    RESEARCH_WEIGHTING_UNIT,
    ZERO_NORM_EPS,
)
```
`RENORMALISE_DOC_VECTORS` and `ZERO_NORM_EPS` are **imported but not yet used** — they were imported in preparation for the function that was about to be written.

**The next action, not yet started**, was to add this helper to `research_score_shards.py`:

```python
def paper_units_from_shard(
    emb: np.ndarray,
    scores: np.ndarray,
    paper_ids: list[str],
    shard_name: str,
    *,
    prev_last_paper_id: str | None = None,
    renormalise: bool = RENORMALISE_DOC_VECTORS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Return (paper_emb, paper_assigned, seg_counts, last_paper_id) for one shard.

    - runs assert_papers_contiguous + the cross-shard boundary check
    - paper score vector  = mean of its segment score vectors -> argmax = paper SDG
    - paper embedding     = mean of its segment embeddings, L2-renormalised
      (fail closed if any paper vector norm < ZERO_NORM_EPS)
    """
```

and then to wire it into **both** centroid-accumulation sites in `1_code/5_supervised_model_infer/score_supervised.py`:

- **LR branch**, `run_research_lr`, the loop at **lines 358-366**:
  ```python
  for sdg_idx in range(N_SDG):
      mask = assigned == sdg_idx          # segment-level today
      counts[sdg_idx] += int(mask.sum())
      sums[sdg_idx] += emb[mask].sum(axis=0)
  ```
  ⚠️ **Non-obvious trap:** in the cached-skip path (**line 311-314**) `scores` is *not* loaded — only `assigned` is reconstructed from the ids JSONL. Paper-level assignment needs the actual score matrix, so that branch must additionally `np.load(score_path)`.
  Also update the meta emitted at **lines 372-399** so `n_papers_assigned` is genuinely papers, and add `n_segments_in_papers_assigned` alongside it.

- **MLP branch**, `run_mlp`, the identical loop at **lines 585-590**. Its shard ids come from the *embedding* manifest: `resolve_manifest_path(shard["ids_path"], allowed_dirs=(embed_root,))`.

Nothing in `score_supervised.py` has been touched yet — it is exactly as at HEAD.

---

## 5. Concerns to emphasise

### 5.1 ⚠️ The repo is currently broken — fix before anything else
Two known breakages introduced by the changes in §3.1:

1. **`RESEARCH_SUBSET_SIZE` was renamed but its only consumer was not updated.**
   `1_code/2_segment/2_sample_segments.py:37,88,90,94` still imports/uses `RESEARCH_SUBSET_SIZE` → **`ImportError` on run**. This is intentional (the rename forces the S1 rewrite of that file) but must not be left dangling.

2. **`aggregate_research_scores` dropped the `n_rows` key.** Callers still reading it:
   - `1_code/7_main_analysis/1_main_text/0_coverage_gap.py:244, 321, 367, 417` → **`KeyError`**
   - `1_code/7_main_analysis/2_appendix/c_sample_stability.py:861-862` reads `mean_top_overall`, which still exists but is now **paper-level**; confirm that is the intended semantics for `mean_paper_top_vs_osdg` (the name suggests yes) and that the per-draw comparison is like-for-like.

A `git stash` / revert of these two files returns the repo to a working HEAD state if you need to abandon.

### 5.2 Step 8 is a hard gate — do not silently rewrite Results
Paper weighting moves SDG 3 by −1.96 pp and SDG 9 by +1.50 pp. Top-3 order survives, but the semantic-gap ranks have **not** been recomputed yet (they depend on the new centroids, which don't exist yet). If any headline ordering moves — the top/bottom adjusted-gap SDG, or the H2.5 null — **stop and report to the user** rather than editing prose interpretation.

### 5.3 S1 breaks offline warm replay until the snapshot is rebuilt
Between changing the subset definition and running `python main.py --backup-data-snapshot embedded`, warm replay is not reproducible from the published snapshot. The re-snapshot is **mandatory**, not optional.

### 5.4 Stale minilm/scibert scores will crash (fail-closed, not corrupt)
Their score-shard manifests record shard 1 = 100,000 rows. After the subset grows to ~122k, a *non*-`--overwrite` `score_supervised.py` run reuses a 100k `assigned` array against a 122k `emb` → boolean-mask length error. Warm replay passes `--overwrite`, so this only bites on piecemeal runs. Deleting those dirs (§6 Step 6) removes the trap.

### 5.5 Operational rules (from AGENTS.md) that bit this session
- **Long jobs must run under `tmux`**, never `setsid`/`disown`; the harness kills anything over ~120 s.
- **`tmux new-session` needs `-c /home/manh/dissertation`** — the persistent tmux server does *not* inherit the caller's cwd, and manifest paths are resolved relative to CWD (`shard_pipeline_utils.resolve_manifest_path` requires **relative** `allowed_dirs`, e.g. `Path("2_data/5_supervised_scored/mpnet")`, not absolute).
- Standalone checks write to `/tmp/opencode/` or `5_notes/scratch/` only — never `2_data/` or `4_outputs/`.
- A backup dir `4_outputs_backup_before_model_namespace/` already pollutes the repo; put the pre-change output snapshot in `/tmp`, not in the repo.

### 5.6 Pre-existing smell noticed, not fixed
`c_sample_stability.build_doc_id_map()` (line ~127) silently falls back to row-level permutations with only a `log.warning` if an ids file is missing. That violates fail-closed. Worth tightening during the §6 Step 5 pass.

---

## 6. The comprehensive plan (steps 1-3 done, 4 onward remain)

### Step 1 ✅ — Named constants
`model_utils.py`: `RESEARCH_WEIGHTING_UNIT`, `RENORMALISE_DOC_VECTORS`, `INLP_RESEARCH_UNIT`, `RESEARCH_SUBSET_PAPERS`. All written into run provenance, not just source.

### Step 2 ✅ — Paper-level aggregation core
`research_score_shards.py` as described in §3.1.

### Step 3 ✅ — Pre-flight verification
Brute-force agreement, contiguity, cross-shard, cross-validated counts. All green (§3.2).

### Step 4 ⬜ — Paper-level research centroids (**INTERRUPTED HERE — see §4**)
`score_supervised.py` LR (lines 358-366, meta 372-399) and MLP (lines 585-590) branches.

### Step 5 ⬜ — Coverage profile + macros
`0_coverage_gap.py`:
- consume the new return contract; canonical research profile = paper-level; the `coverage_diagnostic_unweighted.json` research profile = `segment_*` keys.
- JSON: replace `n_research_papers` with `n_research_segments` + `n_research_abstracts`.
- **Delete `\NResearchPapers` outright — no alias**, so any stale use is a hard LaTeX error.
- Emit: `\NResearchSegments`, `\NResearchAbstracts`, `\NResearchSegPerAbstract`, `\NResearchMultiSegPct`, `\NResearchPreprocessed`, `\NResearchDroppedAtSegmentation`, `\ResearchPolicyUnitRatio`, `\ResearchPolicySegmentRatio`.
  (`\NResearchPreprocessed` comes from `2_data/1_preprocessed/research/metadata/manifest.json` → `totals.rows`; the dropped count is the difference.)

### Step 6 ⬜ — Downstream label + unit fixes
Values or labels: `1_semantic_gap.py` (provenance records the unit), `g_distributional_gap.py:600-634,742` (subsample **papers** to |policy|), zero-shot coverage path.
Labels/consistency only: `2_coverage_semantic_interaction.py`, `a1_register_validation.py`, `a2_policy_source_family_sensitivity.py:166,565`, `a3_sdg4_lexical_audit.py`, `b2_semantic_gap_text_interpretability.py`, `c_sample_stability.py` (fix `full_corpus_rows`; make per-draw centroids paper-weighted — its **draws are already paper-level**), `c1_subset_balanced_stability.py:6`, `h1`, `j1`, `k1`, `i1_assignment_method_comparison.py:10,352`, `plot_figures.py:202`+legend, `0_check_centroid_consistency.py`, `register_adjust.py:13-14` (docstring mislabels the subset "100k papers") + `INLP_RESEARCH_UNIT` provenance.
Also: `1_code/8_visualization/fig_pipeline_flowchart.tex` hardcodes five counts → convert to `\input` of the generated num file so it cannot drift.

### Step 7 ⬜ — S1 paper-based subset
Rewrite `2_sample_segments.py`:
1. Pass 1 over canonical shards → ordered unique-`openalex_id` list, cached to `2_data/2_segmented/research_subset/metadata/paper_index.jsonl` (atomic, resume-safe, `--overwrite`-gated).
2. Seed-42 draw of `RESEARCH_SUBSET_PAPERS` paper ordinals.
3. Pass 2 → write **all** segments of drawn papers (~122k rows).
4. Manifest records `sample_method: "uniform_paper_ordinals"`, seed, `n_papers`, `n_segments`.

### Step 8 ⬜ — MPNet-only trial + **STOP AND REPORT**
```
cp -r 4_outputs /tmp/opencode/4_outputs_pre_paperweighting   # NOT into the repo
python main.py --stage infer      --overwrite
python main.py --stage centroids  --overwrite
python main.py --stage analysis   --overwrite
```
Diff coverage shares and semantic-gap ranks against the /tmp copy. **Report to the user before any prose edit** (see §5.2).

### Step 9 ⬜ — S1 delete-and-refill (**no `--overwrite`**)
Delete exactly:
```
2_data/2_segmented/research_subset/
2_data/3_embedded/minilm/research_shards/
2_data/3_embedded/scibert/research_shards/
2_data/5_supervised_scored/minilm/          2_data/5_supervised_scored/scibert/
2_data/3b_register/minilm/subset/           2_data/3b_register/scibert/subset/
```
(`policy.npy` / `reference.npy` live at the model root, *outside* `research_shards/`, so they survive. All embed resume state lives inside the deleted dir — `status_dir = metadata_dir`, `0_embed_paper_shards.py:142` — so there is no orphaned "complete" checkpoint.)

Then, **without** `--overwrite`:
```
python main.py --stage segment --corpus research    # all 26 shards existence-skip; ~5-10 min of
                                                    # re-read + re-sha256 only. NOT re-segmentation.
python main.py --stage embed                        # only MiniLM+SciBERT subset actually computes
                                                    # (~5 min + ~20 min); everything else cache-hits
```
Leaner equivalent that avoids the 18 GB re-hash — call the scripts directly:
```
python 1_code/2_segment/2_sample_segments.py
python 1_code/3_embed/0_embed_paper_shards.py --embed-model all-MiniLM-L6-v2 \
  --corpus research_subset --device cuda --local-files-only --precision fp16 \
  --normalize-embeddings --batch-size 128
# repeat for allenai/scibert_scivocab_uncased
```
Then **mandatory**: `python main.py --backup-data-snapshot embedded`.

### Step 10 ⬜ — Full replay + PDF
```
python main.py --warm-replay-with-appendix --overwrite    # tmux
python main.py --build-pdf --overwrite
```

### Step 11 ⬜ — Manuscript prose (`3_writing/dissertation.tex`)
| Line | Action |
|---|---|
| `:90` (abstract) | `\NResearchSegments` segments from `\NResearchAbstracts` abstracts |
| `:178` | abstracts count |
| `:182` | "…(each treated as one segment)" → abstracts → segments |
| `:206` | **rewrite** — `\NResearchMultiSegPct`% of abstracts exceed the 374-token budget and split |
| `:232` | **rewrite** — research is *now* genuinely document-weighted, symmetric with policy |
| `:293` | move the 20-word minimum from *preprocessing* to *segmentation* |
| `:359`, `:633` | `76:1` → `\ResearchPolicyUnitRatio` (coverage) / `\ResearchPolicySegmentRatio` (semantic gap); tiers relabelled "abstracts" |
| `:800` | concept corpus: 100,000 papers → `\NResearchSegments` segments |
| `:857` | "(paper level)" — now correct, keep |
| new | supplementary paragraph on the 6,927-record drop, both causes (§2.2) |

Also `5_notes/walkthrough.md:102,217,429,811` (claims 2,543,698 papers were *scored* — 3,105,144 segments were).

### Step 12 ⬜ — Notes
Update `AGENTS.md` (weighting unit + the new subset semantics) and delete this `handoff.md` when the work lands.

### Commit sequence (one concern each)
1. core paper-level weighting (Steps 1-2, 4-5)
2. downstream label/unit fixes (Step 6)
3. paper-based sensitivity subset — S1 (Step 7)
4. regenerated `4_outputs` (Steps 8-10)
5. manuscript prose (Step 11)
6. AGENTS/notes (Step 12)

---

## 7. Immediate next action for the fresh agent

1. Read §4 and §5.1.
2. Write `paper_units_from_shard()` in `1_code/7_main_analysis/0_shared/research_score_shards.py` (signature and semantics given in §4).
3. Wire it into `score_supervised.py` at **lines 358-366** (LR — remember the cached-skip `scores` trap at 311-314) and **lines 585-590** (MLP).
4. Immediately unbreak the two callers in §5.1 (`0_coverage_gap.py`, `2_sample_segments.py`) so the tree runs again.
5. Re-run the pre-flight script `/tmp/opencode/pfcheck/full.py` (from `/home/manh/dissertation`) — it must still print `n_papers 2536771` / `n_segments 3105144`.

---

## 8. Progress Log (resumed session, 2026-08-05)

**DONE — Step 4 (paper-level centroids) + Step 5 (coverage macros) + S1 caller unbreak + MPNet trial.**

- `research_score_shards.py`: added `paper_units_from_shard()` (per-shard collapse to paper vectors, L2-renormalised, cross-shard boundary guard).
- `score_supervised.py`: LR branch (cached-skip now also loads `scores` for paper argmax) and MLP branch both group segments→papers and accumulate unit-normalised paper vectors. Meta gains `n_segments_assigned`; `n_papers_assigned` is now genuinely papers. Verified: LR `n_papers_assigned` sums to **2,536,771**, `n_segments_assigned` to **3,105,144**; centroids unit-norm; `raw_centroid_norm == mean_cos_to_centroid` identity holds.
- `0_coverage_gap.py`: consumes `aggregate_research_scores(unit="document")`; `n_rows`→`n_segments`/`n_papers`; `n_research_papers` retained (=abstracts) for back-compat, plus `n_research_abstracts/segments/preprocessed/dropped`; `\NResearchPapers` deleted, replaced by `\NResearchSegments{3,105,144}`, `\NResearchAbstracts{2,536,771}`, `\NResearchSegPerAbstract{1.224}`, `\NResearchMultiSegPct{17.18}`, `\NResearchPreprocessed{2,543,698}`, `\NResearchDroppedAtSegmentation{6,927}`, `\ResearchPolicyUnitRatio{398}`, `\ResearchPolicySegmentRatio{76.5}`.
- `2_sample_segments.py`: rewritten to draw **100k papers** (seed-42 global ordinals) and emit all their segments (Pass 1 cached `paper_index.jsonl`, Pass 2 copies runs). Manifest records `sample_method: uniform_paper_ordinals`, `n_papers`, `n_segments`, `drawn_ordinals`.

**Trial result (MPNet, vs `/tmp/opencode/4_outputs_pre_paperweighting` snapshot):**
- Coverage top-3 order preserved: SDG3 27.68%→25.72%, SDG9 18.61%→20.10%, SDG4 15.14%→15.60%. Coverage-gap total essentially unchanged (0.952924→0.952857; rounds to 0.953).
- Semantic-gap **largest-gap ranking IDENTICAL**: SDG3 > 13 > 11 > 7 > 1. Smallest-gap set same; only SDG4/SDG5 swap by ~0.01pp. All 17 SDGs reliable in both. **No headline finding moves** → safe to proceed past the Step-8 gate (reported, not silent).
- `\ResearchSdgFourPlusSdgNinePct` 33.7→35.7 (expected).

**Committed:** code + regenerated `4_outputs/mpnet` (coverage + LR/MLP semantic gaps). Snapshot of pre-change outputs at `/tmp/opencode/4_outputs_pre_paperweighting`.

**NEXT (back to plan):** Step 6 downstream label/unit fixes (c_sample_stability `full_corpus_rows`, a2 keys, plot_figures legend, fig_pipeline_flowchart `\input`, i1 header, etc.); then Step 7/9 S1 delete+refill MiniLM/SciBERT + re-snapshot; Step 10 full 3-encoder replay; Step 11 prose; Step 12 commit+notes.

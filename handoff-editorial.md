# Editorial Session Handoff — 2026-08-06

## 1. Context

**Repo:** `/home/manh/dissertation`, branch `main`, remote `https://github.com/qmanhbeo/dissertation-bham.git`

**Goal:** Trim the main text of `3_writing/dissertation.tex` from ~12,433 words to ≤8,800 words (the cap). This is a hard limit for the dissertation.

**Current state:** 10,147 words. Still **1,347 over the cap**.

**Session started:** After the previous session completed the Discussion rewrite, Conclusion rewrite, table split, and rank-vs-magnitude clarification. This session focused on trimming.

**Build system:** `python main.py --build-pdf --overwrite` in tmux session `buildpdf`; poll `/tmp/buildpdf.DONE` (~15s); verify `grep -icE "Citation.*undefined" 3_writing/artifact/dissertation.log` → 0.

**Word counter:** `python 5_notes/word_count.py` (strips LaTeX, counts main text only, stops at `\section{Supplementary Methodology}`).

---

## 2. Key Known Facts

- **Cap:** 8,800 words for main text (everything before `\section{Supplementary Methodology}`)
- **Current:** 10,147 words — need to cut ~1,347 more
- **Per-section counts:**

| Section | Words |
|---|---|
| Introduction | 452 |
| Literature Review | 2,056 |
| Methodology | 3,288 |
| Results | 2,155 |
| Discussion | 1,794 |
| Conclusion | 402 |
| **Total** | **10,147** |

- Methodology (3,288) is by far the largest section — 32% of main text
- Literature Review (2,056) is second largest — 20%
- Results (2,155) and Discussion (1,794) are moderate
- Introduction (452) and Conclusion (402) are already lean

- **Tables 3 and 14 (now split):** Table 3 = Spearman ρ on rank vectors (ranks 1–17); Table 14 = Pearson r on raw magnitudes. Both correlate coverage predictors (raw %) against semantic gaps (cosine distances). The rank-vs-magnitude distinction is now explicit in captions and notes.

- **Table 22 (Pearson table):** Was overflowing vertically due to `\resizebox{\textwidth}{!}{...}`. Fixed by removing resizebox, matching Table 3's format.

- **Table split:** Tables 13 and 14 (paneled) were split into separate files (tab6a/tab6b, tab7a/tab7b). `shared_utils.py` MANUSCRIPT_TABLE_FILES updated to match.

---

## 3. Actions/Decisions Made This Session

### A. Structural rewrites (previous session, committed)
- `8cf839c` — Bridge paragraph inserted at end of Section 2.5, connecting theory to hypotheses (Caplan→H1a, Haas→H1b-d, Gibbons→diagnostic framing)
- `6c24d4f` — Full Discussion rewrite (6 subsections)
- `26932f4` — New Conclusion (3 paragraphs)

### B. Table fixes (this session)
- `bf5ec1e` — Split paneled tables 13 and 14 into separate tables (tab6a/tab6b, tab7a/tab7b)
- `59c9f01` — Updated `shared_utils.py` MANUSCRIPT_TABLE_FILES for new filenames
- `7f33e7a` — Removed `\resizebox{\textwidth}{!}{...}` from Pearson table (Table 14), matching Table 3's format

### C. Rank-vs-magnitude clarification (this session)
- `5a601bd` — Made rank vs. magnitude distinction explicit in:
  - Table 3 caption: "rank correlations (Spearman ρ)"
  - Table 3 notes: "Each correlation is between two rank vectors (ranks 1–17 across the 17 SDGs); Spearman ρ is the Pearson correlation of these ranks, testing monotonic association."
  - Table 14 caption: adds "(not ranked)"
  - Prose line 409: "rank correlations (Spearman ρ)"

### D. Duplicate removal pass (this session)
- `194e0a3` — Removed 5 verified duplicate passages (~234 words):
  1. Method "Geometric effect" paragraph (forward ref to Results)
  2. Results "signal validation" summary (restated 3 preceding paragraphs)
  3. Discussion SDG 9/17 example (verbatim from Results)
  4. Discussion H1b/H1c restatement (verbatim from Results)
  5. Discussion "composite signal" restatement (restated Method)

### E. Sentence-level compression (this session)
- `a05fd0f` — 41 shortenings across Methodology, Results, Discussion (~2,052 words):
  - Ethics boilerplate (3 sentences → 1)
  - Verbose corpus descriptions compressed
  - Redundant cross-references shortened
  - Hedging clauses removed
  - Dense paragraphs compressed (H1a, synthesis, register decomposition, SDG 17, implications, boundary orgs)
  - Two Caplan quotes → one (second was tangential)
  - MLP comparison + summary compressed
  - Per-SDG F1 discussion compressed
  - Table descriptions shortened
  - Robustness sections compressed

---

## 4. What Remains and Why

### The core problem: 1,347 words still over cap

All easy cuts (duplicates, verbose sentences, hedging) are exhausted. The remaining 1,347 words require **structural cuts** — removing or condensing entire paragraphs/sections. This requires judgment calls about what content can be moved to appendix or cut entirely.

### Remaining trimming opportunities (by section)

#### Methodology (3,288 words — target: ~2,800)
The methodology is the most over-long section. Candidates for deeper cuts:

1. **INLP identification argument (lines ~283–285):** The paragraph at `\paragraph{Identification argument.}` is ~150 words. The core argument ("stratification ensures no linear direction encodes SDG topic") is stated twice in the same paragraph (lines 284 and 285). Could merge into ~80 words. **~70 words.**

2. **INLP objective paragraph (line ~267):** The mathematical notation ($\mathbf{x}_i$, $z_i$, $g$, $P$, $W_k$, $N(W_k)$) is ~100 words of formal notation that's already in the Ravfogel citation. Could compress to one sentence + reference. **~60 words.**

3. **Coverage gap procedure (line ~246):** The document-weighted procedure explanation is ~120 words. The 3-step procedure is now compressed but the surrounding justification ("The policy corpus is not a flat set of documents...") could be tightened further. **~30 words.**

4. **Semantic gap method (line ~252):** The K=50 segment cap explanation is ~80 words. The sensitivity test is already in Table 8. Could compress to one sentence. **~40 words.**

5. **Methodology Summary (line ~310):** The summary paragraph is ~100 words. The flowchart figure does this job visually. Could cut to one sentence. **~80 words.**

6. **Classifier description (line ~234-236):** The training/test split description is ~100 words. Could compress. **~30 words.**

#### Literature Review (2,056 words — target: ~1,800)

7. **Semantic Methods subsection (line ~163):** The Biber Dimension 1 example ("Dimension 1 contrasts involved, interactive discourse...") is ~55 words of background that's not essential. Already shortened in pass 2 but could cut the example entirely. **~40 words.**

8. **Research Gap subsection (line ~167):** The "These two contributions are motivated by..." paragraph is ~120 words restating the theoretical motivation that was just reviewed. Could compress. **~40 words.**

9. **AI and SDGs subsection (line ~124):** Four corpus sizes cited (25k, 20.5k, 108, 1,288). Could combine or drop the smaller studies. **~15 words.**

#### Results (2,155 words — target: ~1,900)

10. **Coverage gap description (lines ~357-361):** The policy and research corpus coverage descriptions are ~200 words. The policy corpus description repeats what the reader can see in Figure 2. Could compress. **~50 words.**

11. **Semantic gap figure description (line ~378):** The corpus asymmetry subsampling paragraph is ~100 words. Already shortened in pass 2 but could compress further. **~30 words.**

#### Discussion (1,794 words — target: ~1,600)

12. **Headline paragraph (line ~460):** Already compressed but still restates the raw correlation statistics ($\rho = \SpearmanCovRaw$, etc.) that were just quoted in Results. Could reference Results instead. **~20 words.**

13. **Register decomposition (line ~484):** "This is the Biber register effect operationalised: research abstracts and policy documents differ in situation of use, audience, and communicative purpose, and language models inherently encode these register differences." — restates what's already in Methodology. Could cut. **~25 words.**

14. **Implications paragraph (line ~494):** Already compressed but still verbose. Could compress the Cash et al. framework mention. **~15 words.**

### Estimated total remaining cuts: ~555 words from structural cuts

This would bring the total to ~9,592 — still 792 over cap. To close the remaining gap, you'd need to either:

1. **Move the entire Methodology Summary (line 310-314) to appendix** (~100 words)
2. **Condense the INLP formal notation** (lines 267-269) to prose only (~60 words)
3. **Merge the Research Gap and Hypotheses subsections** (~100 words)
4. **Cut the Biber Dimension 1 example entirely** (~40 words)
5. **Compress the coverage gap description** (~50 words)
6. **Compress the semantic gap subsampling paragraph** (~30 words)
7. **Cut the register decomposition Biber restatement** (~25 words)
8. **Various other sentence-level compression** (~150 words)

This gets close to the target but requires careful judgment about what to cut vs. keep.

---

## 5. Concerns

1. **The 8.8k cap may not be achievable without losing substantive content.** The easy cuts are done. The remaining ~1,347 words require structural decisions that could weaken the methodology's reproducibility or the argument's clarity.

2. **The Literature Review at 2,056 words is already lean for a dissertation.** Further cuts here risk losing theoretical grounding.

3. **The Methodology at 3,288 words is the most over-long section.** It contains the INLP formal notation, the classifier training details, and the coverage/semantic gap definitions — all of which are needed for reproducibility. Moving too much to appendix weakens the self-containedness of the main text.

4. **The Pearson table (Table 14) was fixed but the `\resizebox` removal may cause it to be slightly wider than other tables.** This is cosmetic and acceptable.

5. **Word count tool:** `python 5_notes/word_count.py` strips LaTeX commands and counts whitespace-separated tokens. It's an approximation but consistent.

---

## 6. Comprehensive Plan for Remaining Cuts

### Priority 1: Structural cuts (highest word savings)

| # | Location | Action | ~Words |
|---|---|---|---|
| 1 | Method §310-314 (Summary) | Cut entirely; flowchart figure suffices | ~100 |
| 2 | Method §267-269 (INLP Objective) | Compress formal notation to prose + reference | ~60 |
| 3 | Method §283-285 (Identification) | Merge two restatements into one | ~70 |
| 4 | Method §252 (K=50 cap) | Compress to one sentence | ~40 |
| 5 | LitReview §165 (Biber Dim 1) | Cut the Dimension 1 example | ~40 |
| 6 | LitReview §167-171 (Research Gap) | Compress motivation paragraph | ~40 |

**Subtotal: ~350 words**

### Priority 2: Sentence-level compression (medium savings)

| # | Location | Action | ~Words |
|---|---|---|---|
| 7 | Method §246 (doc-weighted) | Tighten further | ~30 |
| 8 | Method §234-236 (training/test) | Compress | ~30 |
| 9 | Results §357-361 (coverage desc) | Compress | ~50 |
| 10 | Results §378 (subsampling) | Compress further | ~30 |
| 11 | Discussion §460 (headline) | Reference Results instead of restating stats | ~20 |
| 12 | Discussion §484 (Biber restatement) | Cut | ~25 |
| 13 | Discussion §494 (implications) | Compress Cash framework | ~15 |

**Subtotal: ~200 words**

### Priority 3: Deeper structural decisions (requires judgment)

| # | Location | Action | ~Words |
|---|---|---|---|
| 14 | LitReview §149-157 (RP Alignment) | Compress theory paragraphs | ~50 |
| 15 | Method §238 (argmax traceability) | Already compressed; could cut "traceable to 768 signed coefficients" | ~10 |
| 16 | Results §448 (encoder sensitivity) | Already compressed; could trim raw gap ordering | ~15 |
| 17 | Various | Micro-cuts throughout (~15 words each × 10) | ~150 |

**Subtotal: ~225 words**

### Grand total estimated remaining: ~775 words

This would bring the total to ~9,372 — still 572 over cap. The remaining gap would require moving content to appendix or cutting substantive material.

---

## 7. What Was Interrupted

No interruption. The session completed all planned work. The handoff is for continuity if a fresh agent picks up the trimming task.

---

## Files Changed This Session

| File | Changes |
|---|---|
| `3_writing/dissertation.tex` | 51 insertions, 58 deletions across all sections |
| `4_outputs/dissertation.pdf` | Regenerated (1,969,206 bytes) |
| `1_code/7_main_analysis/0_shared/shared_utils.py` | Updated MANUSCRIPT_TABLE_FILES for split table filenames |
| `1_code/7_main_analysis/1_main_text/3_generate_cross_sensitivity_table.py` | Split paneled tables into separate files |
| `4_outputs/mpnet/tables/tab6a_cross_sensitivity.tex` | New (adjusted panel) |
| `4_outputs/mpnet/tables/tab6b_cross_sensitivity.tex` | New (raw panel) |
| `4_outputs/mpnet/tables/tab7a_encoder_sensitivity.tex` | New (adjusted panel) |
| `4_outputs/mpnet/tables/tab7b_encoder_sensitivity.tex` | New (raw panel) |

## Commits This Session

| Hash | Message |
|---|---|
| `a05fd0f` | trim: sentence-level compression across Methodology, Results, Discussion (~2,052 words) |
| `194e0a3` | trim: remove 5 verified duplicate passages (~234 words) |
| `5a601bd` | docs: make rank vs. magnitude distinction explicit in table captions and notes |
| `7f33e7a` | fix: remove resizebox from Pearson table to fix vertical overflow |
| `59c9f01` | fix: update MANUSCRIPT_TABLE_FILES for split table filenames |
| `bf5ec1e` | fix: split paneled tables 13 and 14 into separate tables to fix vertical overflow |
| `26932f4` | writing: rewrite Conclusion to close the paper's opening thread |
| `6c24d4f` | writing: full rewrite of Discussion chapter |
| `8cf839c` | writing: add theory-to-hypothesis bridge in Section 2.5 |

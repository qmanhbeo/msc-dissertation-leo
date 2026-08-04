# Hand-off: Dissertation manuscript cleanup (structure + stale content)

**Purpose of this file:** a self-sufficient brief for a fresh agent to pick up the
manuscript-editing work without re-reading the whole repo. It records where we are,
the established facts, what was already built, what remains, our concerns, and the
full plan for remaining items.

---

## 1. Context — where we are

The dissertation (`3_writing/dissertation.tex`) has been through several rounds of
structural and content cleanup in this session. The goal was to remove stale caveats,
fix noisy captions, promote key method steps to proper subsections, and correct stale
numerical references. All changes are committed and pushed. The PDF compiles clean.

The manuscript is at a mature stage. Most remaining work is polish or future-work
items, not structural.

**Status:** All session work is COMMITTED and PUSHED. Nothing is outstanding.

---

## 2. Key known facts (read these instead of the whole repo)

### Manuscript structure (current Methodology section, after all edits)

```
§3.1  Research Corpus
§3.2  Policy Corpus
§3.3  Embedding Model and Normalisation
§3.4  Token-Aware Segmentation          ← NEW in this session (promoted from appendix)
§3.5  Supervised Reference Classifier   ← MERGED (absorbed §3.6 Scoring + §3.7 Calibration)
§3.6  Coverage Gap Analysis
§3.7  Semantic Gap Analysis
§3.8  Register Adjustment via INLP
§3.9  Coverage–Semantic Interaction
§3.10 Methodology Summary
```

### Deleted sections (this session)
- **Key Assumptions and Mitigations** (`sec:assumptions-stub`) — deleted entirely.
  The SDG 4 caveat was an interpretation issue, not a methodology choice. All
  cross-references now point to the Appendix (`app:sdg4-lexical-audit`).
- **Scoring and Assignment** (`sec:scoring`) — merged into §3.5 (3 sentences, not
  worth its own subsection).
- **Classifier Calibration Across Discourse Types** (`sec:assumptions`) — merged
  into §3.5 (diagnostic property of the classifier, not a separate step).

### Dead labels (fully retired, zero references remaining)
- `sec:assumptions-stub` — was the Key Assumptions subsection
- `sec:sdg4artefact` — was the SDG 4 caveat label (replaced by `app:sdg4-lexical-audit`)
- `sec:scoring` — was the Scoring and Assignment subsection
- `sec:assumptions` — was the Classifier Calibration subsection

### Active labels (used in cross-references)
- `sec:segmentation` — new Token-Aware Segmentation subsection
- `sec:supervised-classifier` — now contains scoring + calibration as paragraphs
- `app:sdg4-lexical-audit` — Appendix SDG 4 lexical audit (replaces old `sec:sdg4artefact`)
- `app:supp-segmentation` — Appendix segmentation mechanics
- `app:supp-truncation` — Appendix truncation fix history

### Concept corpus size
- The concept-retrieved corpus (OECD.AI-style field-of-study method) is **100,000 papers**,
  not 50,000. Code: `MAX_PAPERS_CONCEPT_CORPUS = 100000` (`0_fetch/fetch_openalex.py:134`).
- Variable name in text: `MAX_PAPERS_CONCEPT_CORPUS` (not `MAX_PAPERS_PER_CONCEPT`).

### Encoder subset size
- MPNet (canonical) uses the full ~3.1M corpus.
- MiniLM and SciBERT use a **deterministic 100,000-paper subset** (seed 42).
  Code: `RESEARCH_SUBSET_SIZE = 100_000` (`7_main_analysis/0_shared/model_utils.py:50`).
- This is now noted in §7.3 (Encoder Sensitivity).

### Caption cleanup
- 4 figure captions trimmed (pipeline, PCA, semantic-gap bar, scatter).
- 6 table captions cleaned (column defs, significance stars moved to Note blocks).
- 12 captions kept as-is (already concise or already had Notes).
- Total caption count: 22 (unchanged).

### Build conventions
- PDF build: `python main.py --build-pdf --overwrite`
- **Always use tmux** for the build — never `setsid`/`disown`.
- Build takes ~5-10 seconds, not minutes. Do not poll for 120s.
- Build log: `/tmp/buildpdf.log`; completion flag: `/tmp/buildpdf.DONE`
- `amssymb` is in the preamble (needed for `\mathbb{R}` in INLP notation).

---

## 3. Actions / decisions made & files changed this session (and why)

### Commit history (this session, oldest → newest)

| Commit | Message | What changed |
|--------|---------|-------------|
| `3930dcb` | Add INLP register-adjustment subsection; fix centroid-similarity figure cross-reference | New §3.8 INLP subsection; centroid-similarity figure reference fixed to cite Appendix |
| `5c1a561` | Trim noisy figure/table captions; move definitions to Notes | 10 caption edits (4 figures, 6 tables) |
| `8af95f8` | Promote token-aware segmentation to methodology subsection; remove stale caveat | New §3.4 Token-Aware Segmentation; removed text-unit caveat from Key Assumptions |
| `ea06962` | Remove Key Assumptions subsection; redirect SDG 4 references to Appendix | Deleted §3.8 entirely; 3 cross-references updated to `app:sdg4-lexical-audit` |
| `5acc222` | Fold Scoring/Assignment and Calibration subsections into Supervised Reference Classifier | Merged §3.6 + §3.7 into §3.5; retired `sec:scoring` and `sec:assumptions` labels |
| `dc5e522` | Fix stale concept-corpus size: 50k → 100k; correct variable name | 3 locations: lines 179, 516, 715; variable name `MAX_PAPERS_CONCEPT_CORPUS` |
| `3a60973` | Note 100k subset size for MiniLM/SciBERT in encoder sensitivity section | One sentence added to §7.3 |

### Detailed changes

**A) Token-Aware Segmentation (§3.4, new subsection)**
- Inserted between Embedding Model and Supervised Reference Classifier.
- Describes: NLTK sentence-boundary detection → greedy accumulation → `max_seq_length - 10` token budget (374 tokens for MPNet).
- Explains consequences: research abstracts pass through as single segments; policy docs are split.
- Distinguishes the token budget from the per-document segment cap ($K=50$).
- Cross-references Appendix §5 for full algorithmic details.

**B) Key Assumptions subsection deleted**
- The text-unit handling caveat was stale (promoted to §3.4).
- The SDG 4 limitation was an interpretation issue, not a methodology choice.
- All 3 cross-references to `sec:sdg4artefact` redirected to `app:sdg4-lexical-audit`.
- No new prose added — each Results site already explains the SDG 4 issue inline.

**C) Classifier subsections merged**
- §3.6 (Scoring and Assignment) was 3 sentences → became a paragraph in §3.5.
- §3.7 (Classifier Calibration) was 2 paragraphs → became paragraphs in §3.5.
- All three described the same instrument; the hierarchy was artificially fine-grained.
- `sec:scoring` label retired (1 cross-reference updated).
- `sec:assumptions` label retired (0 external references — was dead).

**D) Stale concept-corpus size fixed (50k → 100k)**
- Line 179 (Research Corpus): `50,000-paper` → `100,000-paper`.
- Line 516 (Appendix §5): `MAX_PAPERS_PER_CONCEPT = 25,000` → `MAX_PAPERS_CONCEPT_CORPUS = 100{,}000`.
- Line 715 (Appendix §13): `50,000-paper` → `100,000-paper`.
- Sample-stability ladder references at 50k tier left unchanged (different quantity).

**E) Encoder subset size noted**
- Added one sentence to §7.3 (Encoder Sensitivity): "MiniLM and SciBERT are scored
  on a deterministic 100,000-paper subset (seed 42) of the canonical segmented corpus,
  so the architecture comparison isolates encoder choice from corpus scale."

**F) Caption cleanup (10 edits)**
- `fig:pipeline-flowchart` — trimmed to one sentence.
- `fig:pca-register-before-after` — trimmed to one sentence.
- `fig:semantic_gap` — trimmed visual-encoding legend.
- `fig:typology_scatter` — trimmed visual-encoding legend.
- `tab:register-decomposition` — column defs → Note.
- `tab:interaction` — column defs + stars → Note.
- `tab:iterative-register-check` — column defs → Note.
- `tab:concept-coverage` — delta def → Note.
- `tab:app-assignment-method-comparison` — column defs merged into existing Note.
- `tab:raw-value-correlation` — grid ref + stars merged into existing Note.

---

## 4. What remains and why

**Nothing from this session's work remains.** All edits are committed and pushed.

### Potential follow-ups (not requested, documented for completeness)

1. **Dead labels:** A few labels are defined but never `\ref`'d (e.g., `sec:intro`,
   `sec:conclusion`). Harmless but could be cleaned up. Low priority.

2. **Zero-shot methodology in main text:** The zero-shot nearest-centroid method is
   only described in the Appendix (`app:assignment-method-comparison`). Defensible
   because it's scoped as a single sensitivity check, but a reader of Section 3 alone
   cannot understand the zero-shot pipeline. Low priority.

3. **MLP methodology in main text:** Same pattern — grid search and held-out evaluation
   are in Appendix (`app:model-selection`). The Methodology Summary mentions MLP in
   passing but has no dedicated subsection. Defensible.

4. **Preprocess/segment stages:** Now partially addressed (segment → §3.4). Preprocess
   is still inline + appendix. Standard for mechanical steps.

5. **Language polish:** A world-class copy-editing pass could tighten phrasing throughout.
   Not urgent.

---

## 5. Concerns to emphasize

- **PDF build takes ~5-10 seconds.** Do not poll for 120s — a short 8s sleep is
  sufficient. The tmux session finishes quickly.

- **`amssymb` is in the preamble** (added for `\mathbb{R}` in INLP notation). If any
  other package conflicts arise, this is the likely source.

- **Verify, don't trust:** after any future edit that shifts line numbers, re-grep for
  active cross-references to confirm they still resolve.

- **Do NOT commit** unless the user explicitly asks.

- **The concept corpus is 100k, not 50k.** Three locations were fixed. If you see
  "50,000" anywhere, check whether it's stale.

- **The encoder subset is 100k** (MiniLM/SciBERT), not the full corpus. This is now
  noted in §7.3 but was previously undocumented in the text.

- **SDG 4 references now point to the Appendix** (`app:sdg4-lexical-audit`), not a
  methodology subsection. The label `sec:sdg4artefact` is fully retired.

---

## 6. Comprehensive plan (for any future manuscript-structure work)

### No active plan — all session work is complete.

The following is preserved for reference if the user requests further structural
improvements.

### Remaining methodology subsections that could be promoted
- **Preprocess:** brief paragraph (cleaning, English filtering, 20-word minimum).
  Already described inline in Research Corpus and Appendix §4.
- **Zero-shot:** brief paragraph (nearest-centroid assignment). Already described in
  Appendix `app:assignment-method-comparison`. Scoped as a single sensitivity check.

### If the user wants to restructure further
- The Methodology Summary (§3.10) currently lists all steps. It could be deleted if
  every step has its own subsection, but it serves as a useful overview.
- The Appendix sections (§5–§6 Segmentation Mechanics, Truncation Fix) are fine as
  implementation history — they don't need to be moved.

### If the user wants to add more robustness checks
- The cross-sensitivity grid is comprehensive (encoder, policy source, segment cap,
  retrieval strategy). No obvious gaps.
- The balanced-subset stability check (Appendix C.1) is already thorough.

---

## 7. Exactly what was being done when interrupted

The session completed all work. The final action was writing this handoff file.
No edits were interrupted. All changes are committed and pushed.

**Last commit:** `3a60973` (Note 100k subset size for MiniLM/SciBERT in encoder sensitivity section)

**Concrete next steps for the fresh agent:**
1. If the user asks for more structural work: refer to §6 above.
2. If the user asks about INLP: the canonical term is INLP (Iterative Nullspace
   Projection). The original paper is in `0_literature/register_adj/`.
3. If the user asks about stale content: check §2 (Key Known Facts) for current values.
4. Do NOT commit unless asked.

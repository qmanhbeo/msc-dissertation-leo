# Hand-off: INLP "register" interpretation — go/no-go validation check

**Last updated:** 2026-08-05
**Status:** Go/no-go check **COMPLETE** (all of Step 0–3 done); verdict **leaning FOR the register interpretation, GO for a full validation appendix** — with two must-address caveats. Report committed and pushed (`0f96a3f`). **Interrupted:** nothing mid-flight — the user asked to stop after the check; the full appendix was never started (out of scope by design).

> **This file replaces the previous `handoff.md`.** The prior hand-off documented
> the H1a–H1d Concept-row gap-dispatch bug (Appendix J.1 / Table `tab:interaction`).
> That task is **fully resolved and committed** (H1 fix + K.1 + PDF rebuild in
> commits `7cdbb8d`..`bb9df3b`; tree is clean). It was preserved verbatim at
> `5_notes/handoff_h1_concept_rows_2026-08-04.md` (its "nothing committed" prose is
> stale — ignore those sections).

---

## 1. Context — where we are

The dissertation measures research–policy divergence in AI-for-SDG discourse.
A core method step removes a **"register" direction** from sentence embeddings
using **Iterative Nullspace Projection (INLP)** (`register_adjust.py`), on the
theory that raw embedding distance conflates *topic* difference with *register*
difference (academic vs policy prose style). The manuscript's own limitation
section states the register interpretation is **"plausible on design grounds"
but "not validated against independent linguistic markers … left to future
work"** (`3_writing/dissertation.tex:279` and `:477`).

This session ran the **first-pass, cheap, go/no-go empirical check** of that
interpretation before committing to a full validation appendix:

- **Step 0** — located all texts, embeddings, INLP projection matrices (G), and
  SDG labels; confirmed which encoders have complete artifacts.
- **Step 1** — built a stratified sample of **408 segments** (204 research + 204
  policy; 12 per SDG per corpus) and computed 6 cheap Biber-style register
  features + a combined register score.
- **Step 2** — correlated the independent register score with (a) the magnitude
  of the INLP-removed component ‖x−x′‖, (c) distance-to-SDG-centroid in raw vs
  adjusted space, and (d) trained corpus classifiers on register-score / raw /
  adjusted embeddings.
- **Step 3** — 17-way SDG classifier on raw vs adjusted embeddings (the "is INLP
  deleting topic?" selectivity check).

**Deliverable produced:** `5_notes/register_validation_report.md` (committed,
`0f96a3f`). The check script + artifacts are in `5_notes/scratch/` (gitignored —
see §5.6/§6 Phase 0).

**No manuscript source (`3_writing/`, PDF) was touched.** This is diagnostic only.

---

## 2. Key known facts (read this instead of re-deriving)

### 2.1 Repository operating rules (from AGENTS.md — non-negotiable)

- **Single entrypoint** `python main.py`. **Default mode is read-only**; any
  command that writes outputs requires `--overwrite` (fails closed otherwise).
- **Incremental write/resume** is the default for long scripts; **scratch-only
  checks write ONLY to `5_notes/scratch/`, `/tmp`, or `__test_queries/`** — never
  `2_data/` or `4_outputs/`.
- **Long jobs must run under `tmux`** (harness kills at ~120s). Short jobs (PDF
  build, this check) can run directly; poll short first.
- **Verify, don't trust** every prior claim/artifact. Deterministic seed **42**.
- No test/lint/typecheck suite; Conda env `dissertation` (Python 3.11) is the only
  build path. No spaCy — **nltk** was used (punkt + POS tagger; see §2.5).

### 2.2 Pipeline architecture (high level)

```
Preprocess → Segment (canonical, 384-token chunks, ONCE, shared by all encoders)
→ Embed (MPNet full research; MiniLM/SciBERT on 100k-paper subset; policy/reference shared)
→ Train supervised LR (17-way SDG) on labeled reference corpora
→ Score research + policy segments (per-segment 17-dim SDG scores)
→ Build centroids → register_adjust (INLP: materialise G, NOT adjusted embeddings)
→ Semantic-gap analysis (raw + adjusted) + cross-sensitivity tables
```

- **Adjusted embeddings are NEVER materialised.** G is small (~KB) and downstream
  consumers project raw embeddings on the fly via `register_utils.project()`
  (`1_code/7_main_analysis/0_shared/register_utils.py`). Any validation must do
  the same — there is no `adjusted.npy` anywhere.
- `register_adjust.py` (`1_code/7_main_analysis/0_shared/`) runs iterative,
  SDG-stratified binary research-vs-policy logistic regression; each iteration
  appends one orthonormal direction `g_k` to G. **Stop criterion: held-out test
  acc ≤ 0.5** (chance), i.e. no linear classifier can separate corpora in the
  residual space. G rows are the **accumulated orthonormal directions**.
- G matrices (verified): **MPNet canon (62, 768)**; MiniLM subset (29, 384);
  SciBERT subset (71, 768).

### 2.3 Data locations and alignment (all verified positionally)

| Need | Path | Alignment |
|---|---|---|
| Research raw text (embedded) | `2_data/2_segmented/research/part-*.jsonl` — `text` field | line `i` == embedding row `i` |
| Policy raw text (embedded) | `2_data/2_segmented/policy.jsonl` — `text` field | line `i` == embedding row `i` |
| Research raw embeddings | `2_data/3_embedded/mpnet/research_shards/part-*.npy` | row == segment |
| Policy raw embeddings | `2_data/3_embedded/mpnet/policy.npy` (40597×768) | row == segment |
| INLP projection matrix G | `2_data/3b_register/{model}/{track}/G.npy` | MPNet→`canon`, MiniLM/SciBERT→`subset` |
| SDG label, research segments | `2_data/5_supervised_scored/mpnet/paper_scores_shards/metadata/part-*_ids.jsonl` → `assigned_sdg` | row == segment |
| SDG label, policy segments | `2_data/5_supervised_scored/mpnet/policy_scores.npy` → `argmax` (17 dims) | row == segment |
| Policy segment ids/meta | `2_data/3_embedded/mpnet/metadata/policy_ids.json` (id, source_doc, source_family, source) | row == segment |

**The three "Concept encoder" rows in the manuscript tables are NOT a fourth
encoder.** `research_concept/` is the concept-retrieved *research corpus*
embedded with MPNet; it has **no** register artifacts under `3b_register/`.
Cannot be used as an independent encoder check.

### 2.4 Critical conceptual facts (learned this session)

- **"Documents" are actually segments** (~384-token chunks), not papers/docs.
  Research: one paper → multiple segments, and a paper's segments can carry
  **different `assigned_sdg`** (label is per-segment from the LR classifier).
  INLP training and gap analysis operate at segment level. A "document-level"
  validation would need its own aggregation design.
- **All index alignments are positional** (row index). Traps: `policy.jsonl`'s
  `id` field is the **source_doc**, not the segment id — `segment_id` matches
  `policy_ids.json`; join on position or `segment_id`, never on `id`.
- **The embedded text for research = the segment's `text` field** (from
  `combined_text` = title + abstract). Register features must be computed on the
  segment `text` (the exact embedded string), not title/abstract separately.

### 2.5 Environment change made this session (not committed)

- Installed **nltk POS tagger** data (`averaged_perceptron_tagger_eng`) into the
  `dissertation` conda env (needed for the passive-voice feature; punkt was
  already present). A clean rebuild of the env will NOT have this. Record the
  download in the appendix's environment note if it gets promoted.

### 2.6 The check — design and numbers (n=408, MPNet canon, seed 42)

Sample: 12 segments/SDG/corpus, distinct papers / distinct policy source-docs per
SDG, min 20 words. Features (nltk, rates per 1000 words): hedge, deontic modal,
passive (VBN after be-form), first-person pronouns, nominalization (−tion/−ment/−ness),
mean sentence length. **Combined score = PC1 of z-scored features**, oriented
positive→longer sentences (dominant data-driven axis, 22% feature variance).

| # | Result | Numbers |
|---|---|---|
| 2b | register score ~ ‖x−x′‖ | pooled ρ=0.102 (p=0.04); **within-research ρ=0.212 (p=0.002); within-policy ρ=0.191 (p=0.006)** |
| 2b | per-feature vs ‖x−x′‖ | deontic **0.241**, mean_sent_len **0.241** (both p<1e-6), passive **−0.194** (p=8e-5), nominal 0.117 (p=0.018), hedge/first_person ns |
| 2c | register score ~ dist-to-SDG-centroid | **RAW ρ=0.126 (p=0.011) → ADJ ρ=0.247 (p=4e-7)** — correlation gets *stronger* in adjusted space (see §5.1) |
| 2c | robustness | persists with own-corpus centroid (0.245), partialling corpus (0.253), dropping mean_sent_len (0.256), within-corpus splits (research 0.160→0.301, policy 0.100→0.204), opposite-corpus centroid (0.076→0.205) |
| 2c | per-SDG research–policy centroid distance | raw 0.477 → adjusted 0.406 (residual corpus offset remains inside SDGs) |
| 2d | corpus classifier (5-fold CV acc) | register-score-only **0.456**; **raw embeddings 0.909**; **adjusted embeddings 0.505** (≈ chance) |
| 3 | 17-way SDG classifier | raw LR **0.691**/kNN 0.554 → adjusted LR **0.672**/kNN 0.578 (chance 0.059) |

Mean features by corpus (research vs policy): hedge 1.43 vs 0.85 (d=−0.16);
deontic **0.78 vs 3.13 (d=+0.54)**; passive 10.30 vs 8.45 (d=−0.20); mean_sent_len
37.0 vs 63.8 (d=+0.15); first_person 6.90 vs 6.46 (d=−0.04); nominal 36.4 vs 39.0
(d=+0.12).

### 2.7 Verdict (from the report — the go/no-go answer)

**GO — leaning FOR "INLP removes a register-like, not topic-like, component", but
not cleanly.** Strongest, most direct evidence: (1) corpus separability destroyed
by adjustment (0.909→0.505 ≈ chance) while SDG/topic separability is preserved
(0.691→0.672) — the "is it deleting topic?" red flag is **absent**; (2) removed
magnitude tracks policy-coded features (deontic, sentence length) within corpus.
Two must-address caveats before a full appendix: **(3) residual register-like
structure survives within SDGs** (Step 2c: the register–distance correlation gets
stronger in adjusted space); (4) the 6 cheap features barely capture what was
removed (register-score-only classifier ≈0.46 vs 0.909 raw) and **the combined
score's operationalization changes the conclusions** (an a-priori "institutional"
z-sum gives null 2b and reversed 2c).

---

## 3. Actions taken this session

- **Wrote** the diagnostic script `5_notes/scratch/register_validation_check.py`
  (18.9 KB; deterministic seed 42; reads `2_data/` read-only; writes only scratch
  `.npy`/`.npz`/log). Iterated on it twice for performance (original streamed
  multi-GB segmented shards repeatedly and timed out at 10 min; rewrote to read
  paper-ids from the small score-metadata jsonl and stream each segmented shard
  **once** — full run now ~2–3 min). **No repo code was modified.**
- **Verified** all positional alignments (research embedding↔text↔score rows;
  policy embedding↔text↔ids rows) before trusting any number.
- **Computed** Steps 1–3 and ran robustness checks (own-corpus centroids, partial
  Spearman, no-sentence-length score, within-corpus splits, per-feature loadings,
  a-priori z-sum variant, per-feature corpus discriminability).
- **Wrote** the report `5_notes/register_validation_report.md` (committed,
  pushed, `0f96a3f`). Moved a copy there from scratch because `5_notes/scratch/`
  is gitignored.
- **Installed** nltk POS tagger into the conda env (§2.5).
- **Decisions made:** (a) single encoder **MPNet canon** (per task: "prefer
  MPNet"); (b) **segment-level** units (the pipeline's own unit); (c) combined
  score = **PC1** of z-scored features (reported choice, per task), z-sum kept as
  a sensitivity; (d) seed 42; (e) no manuscript/appendix text written (diagnostic
  only); (f) old `handoff.md` preserved at `5_notes/handoff_h1_concept_rows_2026-08-04.md`.

**Files changed/created this session:**
- `5_notes/register_validation_report.md` — the deliverable (committed `0f96a3f`).
- `5_notes/scratch/register_validation_check.py`, `regcheck_arrays.npz`,
  `regcheck_full.log`, `regcheck_removed_norm.npy`, `regcheck_reg_score.npy`,
  `regcheck_X_adj.npy` — script + artifacts (gitignored, in scratch).
- `5_notes/handoff_h1_concept_rows_2026-08-04.md` — archived previous handoff.
- `handoff.md` — this file.

---

## 4. What remains, and why

The check is **complete and self-sufficient**. Remaining work is the *decision* on
the full validation appendix plus its execution (deferred by design) and a few
bookkeeping items:

### 4.1 Decision gate (needs the PI)
- Read `5_notes/register_validation_report.md` and decide whether the go/no-go
  evidence is sufficient to fund a full validation appendix. **Recommendation:
  GO**, with the two caveats in §2.7 mandated as deliverables of the appendix.

### 4.2 Bookkeeping (quick, no research needed)
- **Promote the check script into the repo** if it must be reproducible by a fresh
  agent. It currently lives only in gitignored `5_notes/scratch/`. Options: copy to
  `1_code/7_main_analysis/2_appendix/` as a proper appendix stage (would then need
  registering in `APPENDIX_SPECS` + fingerprint/`--overwrite` conventions per
  AGENTS.md), or commit a copy under `5_notes/` (tracked working-note). Note the
  nltk-tagger env dependency (§2.5).
- **Commit this `handoff.md`** and the archived `5_notes/handoff_h1_concept_rows_2026-08-04.md`
  when the PI confirms content.

### 4.3 Full validation appendix — NOT started (out of scope for a go/no-go)
The dissertation needs the register interpretation validated before the adjusted
gap can be called anything stronger than "primary but unvalidated"
(`dissertation.tex:279,373,392,477`). See §6 for the plan.

---

## 5. Concerns to emphasise

1. **The Step 2c pattern is the real risk.** Under a clean "register fully
   removed" hypothesis, the register-score↔distance-to-SDG-centroid correlation
   should *drop* after adjustment; it instead *rises* (0.126→0.247) and survives
   every robustness variant. This means INLP removes the **global** corpus
   direction but **per-SDG register-like offset remains** (per-SDG research–policy
   centroid distance is still 0.41 after adjustment). Any appendix that only shows
   corpus-classifier collapse (0.909→0.505) will be caught by a reviewer who
   re-derives 2c. This must be confronted head-on, not buried.
2. **Six cheap features are not enough to adjudicate the claim.** The register-
   score-only corpus classifier (≈0.46) is far below raw embeddings (0.909): the
   removed subspace is much richer than these proxies. A full appendix needs a
   substantially larger Biber-style battery (ideally an MD-style dimension score),
   otherwise "the removed subspace is register" remains under-supported.
3. **The combined-score operationalization changes the answer.** PC1 (data-driven)
   shows the effects in §2.6; an a-priori "institutional" z-sum gives 2b null
   (ρ=0.007) and *reversed* 2c (−0.24). A validation appendix must pin down and
   pre-register *which* register operationalization it claims, or the result will
   look cherry-picked.
4. **Passive voice points the "wrong" way** (research > policy, d=−0.20, and
   negatively correlated with removed magnitude). Any feature orientation assuming
   "policy = uniformly more formal/institutional" is mis-specified for this corpus
   pair.
5. **Policy text quality.** Policy `mean_sentence_length` (63.8) is inflated by
   PDF-extraction junk (banner/cover artifacts break sentence splitting). The 2c
   result survives dropping this feature, but the appendix must pre-clean policy
   text or drop the feature.
6. **The check is currently not reproducible from the repo alone** (script is in
   gitignored scratch; nltk tagger is a local env add). If this go/no-go result is
   cited anywhere, promote the script first (§4.2).
7. **Concept rows / zero-shot restriction do not change** (see AGENTS.md): the
   zero-shot axis stays MPNet-group-only in manuscript-facing tables. A register
   *validation* is not a ZS result and is not restricted by that rule, but don't
   let the validation mutate any manuscript table.

---

## 6. The comprehensive plan

### Phase 0 — Bookkeeping (immediate, ~30 min)
1. PI reviews `5_notes/register_validation_report.md`; confirms GO or NO-GO.
2. Commit this `handoff.md` + archived prior handoff.
3. Decide script fate: promote to `1_code/.../2_appendix/` (register in
   `APPENDIX_SPECS` + fingerprint/`--overwrite` + env note for nltk tagger) or keep
   as a tracked `5_notes/` working-note copy. If GO, prefer the proper appendix
   stage so it is replayable.

### Phase 1 — Design freeze (before any code)
4. **Register operationalization:** pre-register the feature set and the combined
   score. Recommended: Biber (1988) MD-style dimension scoring over a battery
   (tense/aspect, modality, passives both types, pronouns, complement clauses,
   WH-relatives, nominalizations, coordinators/subordinators, prepositions,
   contractions, etc.), reported as **per-dimension correlations**, with the
   a-priori institutional score and the data-driven PC as explicit alternatives.
5. **Unit of analysis:** decide segment vs paper/document level (segments match the
   pipeline; paper-level avoids within-paper label conflicts but needs aggregation
   + a strategy for multi-SDG papers). Report both if feasible.
6. **Encoders:** extend from MPNet-only to MiniLM + SciBERT (G already exists for
   both) to test whether the register interpretation is encoder-robust. Note the
   subset (100k-paper) track for these.
7. **Power/sample size:** scale from 408 to ~1–2k segments/corpus for tighter CIs;
   keep the 12/SDG stratification design (or increase to ~60/SDG).

### Phase 2 — Data work
8. Pre-clean policy text (strip cover/banner artifacts) or drop `mean_sentence_length`.
9. Compute the richer feature battery on the larger stratified sample; verify
   feature distributions per corpus before modelling (expect passive to point
   toward research — confirm it is a feature-direction fact, not a bug).
10. Compute adjusted embeddings on-the-fly with `register_utils.project()` (never
    materialise); record ‖x−x′‖ and PC-projected coordinates.

### Phase 3 — Analysis that must appear in the appendix
11. Repeat §2.6 results at scale (2b, 2c, 2d, 3).
12. **Confront Step 2c directly:** regress adjusted distance-to-SDG-centroid on the
    full feature set (within-SDG, controlling corpus); test whether residual
    corpus-per-SDG offset after INLP is predicted by register features, and whether
    SDG-specific (per-SDG) INLP directions would remove it. State clearly whether
    the residual is register or topic.
13. Encoder-robustness: does the corpus classifier collapse (2d) and topic-preservation
    (3) replicate on MiniLM/SciBERT?
14. If feasible: ablate the register score → e.g., train the INLP corpus classifier
    on the text features (does feature space recover G's span?); compare
    ‖projected-onto-G‖ vs each register dimension.

### Phase 4 — Writing up (only after Phase 3 passes review)
15. Write the appendix (register validation) per repo conventions (JSON-out,
    macros via `generate_tex_macros.py`, fingerprint-gated, registered in
    `APPENDIX_SPECS`).
16. Wire into `dissertation.tex`: update the "left to future work" sentences at
    :279 and :477 (and the "primary but unvalidated" notes at :373/:392) to cite
    the validation; keep every change macro-driven.
17. `python main.py --build-pdf --overwrite` (bash/WSL; short job — poll short
    first); verify tables/figures; commit per repo discipline (one concern per
    commit; re-verify affected stage first).

### Phase 5 — Commits (only when PI asks)
18. One concern per commit: (a) appendix code + registration; (b) outputs; (c)
    dissertation.tex prose + macros; (d) rebuilt PDF.

---

## 7. What was interrupted

**Nothing was mid-execution.** The go/no-go task ran to completion:
- All of Step 0, 1, 2, 3 executed successfully (n=408, MPNet canon, seed 42).
- The report was written, committed, and pushed (`0f96a3f`).
- The user then asked to **stop** ("please stop for now") before any full
  appendix work. The full validation appendix was explicitly out of scope for a
  go/no-go check and had not started.

**Uncommitted leftovers (all intentional, none blocking):**
- `5_notes/scratch/` script + artifacts (gitignored by design — the report carries
  the numbers; the script is the only way to re-run them).
- nltk POS tagger data (env-side, not versioned).
- This `handoff.md` + `5_notes/handoff_h1_concept_rows_2026-08-04.md` (written,
  not yet committed — pending PI review of the handoff content).

### Immediate next action for a fresh agent
1. `git log --oneline -3` (expect clean tree, `0f96a3f` on top); `git status` clean.
2. Read `5_notes/register_validation_report.md` (the deliverable; ~120 lines) then
   `5_notes/scratch/register_validation_check.py` if re-running matters.
3. Present the verdict (§2.7) and concerns (§5) to the PI; do NOT start Phase 1
   design or any appendix writing without an explicit GO.
4. If GO and script promotion is approved, follow Phase 0 step 3 then Phase 1.

Re-run command (if the scratch dir still exists, env has the nltk tagger):
```bash
source activate dissertation
python 5_notes/scratch/register_validation_check.py   # ~2-3 min, writes only to 5_notes/scratch/
```

# Hand-off: Register-validation follow-up 2 (sample construction, accuracy CIs, policy other-dist pull)

**Last updated:** 2026-08-05 (after completion)
**Status:** Follow-up 2 is **COMPLETE**. The RNG bug was fixed, the script was re-run under tmux, and the **acceptance gate passed exactly** (2A 0.456/0.909/0.505, 2B 0.544/0.944/0.603, Item-3 pooled -0.088/-0.074, policy other-dist -0.197 all reproduced from `regcheck_followup.log`). The report `5_notes/scratch/register_validation_followup2.md` has been rewritten with the corrected, like-for-like numbers and a blunt updated verdict. **Awaiting human review — nothing committed, nothing scaled up.**

---

## 1. Context — where we are

We are validating the dissertation's INLP "register" interpretation (the removed
subspace of sentence embeddings = academic-vs-policy *register*, not topic).

Timeline:
- **Report 1** (`5_notes/register_validation_report.md`, committed `0f96a3f`):
  go/no-go at n=408 (MPNet canon, seed 42). Verdict **GO** with two caveats:
  (a) residual register-like structure inside SDGs after adjustment (Step-2c red
  flag: reg↔centroid-dist ρ rises 0.13→0.25 after adjustment), (b) the 6-feature
  score operationalization changes the answer.
- **Follow-up 1** (`5_notes/register_validation_followup.md`, committed `c2773a9`;
  script `5_notes/scratch/register_validation_followup.py`, log
  `regcheck_followup.log`): Items 1–3 cleared —
  1. Concept rows in Table 3: **NOT a bug** (same embedder/space, empirically proven).
  2. Original 2b/2c signals were **clustering artifacts** (SDSN/UNDP mega-docs);
     one-per-parent sampling flips 2c negative and nulls 2b.
  3. Step-2c decomposition: no residual-register red flag survives; not a
     renormalization artifact.
  Verdict: **GO with qualification** (one-per-parent as primary design).
- **This session = Follow-up 2** (three mandated items). **COMPLETE** — see
  `5_notes/scratch/register_validation_followup2.md` and §3–4 below.

**Scope discipline (still in force):** diagnostic-only. Do NOT touch `3_writing/`,
the PDF, any existing analysis script/table. Do NOT scale to n=120/SDG. All writes
stay in `5_notes/scratch/`. Deliverable is `5_notes/scratch/register_validation_followup2.md`.

---

## 2. The Follow-up-2 task (verbatim intent)

**Item 1 — Explain the one-per-parent resampling procedure.** Is 2B/3 (a) a strict
subset of the original 408 with duplicates dropped (→ n=390), or (b) a
dropped-then-refilled rebuild restoring 12/SDG/corpus? If (b): report the
replacement procedure precisely (same seed/eligibility? which sources filled the
SDSN/UNDP/WHO slots — are they systematically different in genre/length/origin?).
State plainly whether n=24/SDG in Items 2B/3 is like-for-like with 2A or a
materially different composition, and flag caveats for per-SDG (3a) and per-feature
(3b) results.

**Item 2 — Interrogate the 2d accuracy shift** (raw 0.909→0.944, adj 0.505→0.603,
register-only 0.456→0.544). Compute a CI (bootstrap or binomial — say which) for
adjusted-space 5-fold CV accuracy on BOTH samples. Is 0.505 distinguishable from
chance? Is 0.603? Investigate WHY accuracy rose: (a) fold-leakage direction, (b)
mega-docs are systematically hard-to-classify, (c) sample-size noise. Report a
clear verdict on whether "adjusted-space accuracy ≈ chance" still holds.

**Item 3 — Expand on Item 3c** (policy-only within-SDG pull from the research
centroid, pooled ρ=−0.197 in follow-up 1). (a) Per-SDG breakdown (is it driven by a
few SDGs or spread across 17?). (b) Mega-doc exclusion check (does it persist with
mega-doc segments dropped entirely?). (c) Substantive interpretation (residual
register on the policy side?) or plainly say it is noise.

**Deliverable:** `5_notes/scratch/register_validation_followup2.md`, every result
tagged with sample/n, ending with an updated verdict: is "GO, scale to n=120/SDG"
still right? Does anything change the recommended primary sampling strategy? Be
explicit about which prior follow-up claims need revision (especially the "adj ≈
chance" characterization).

---

## 3. What was done this session

1. **Read all mandated inputs** (both reports + both follow-up scripts + logs).
2. **Fixed the RNG bug** in `register_validation_followup2.py`: the script had
   re-seeded `_rng` inside `build_sample` (wrong samples for 2B/Item-3). Now uses
   ONE module-level `_rng = np.random.default_rng(SEED)`, never reassigned, with
   three successive `build_sample` calls (False, True, True) = draws 1/2/3 of one
   continuous seed-42 stream — exactly the original script's effective behaviour.
   Also refactored `build_sample(global_dedup, rng=None)` so fresh independent
   draws (seeds 43/44/45) can be made for the stability check.
3. **Replaced the broken bootstrap CI** (resample-then-CV leaked train/test via
   duplicate rows) with **valid methods**: pooled 5-fold stratified-CV predictions,
   Wilson 95% CI on the pooled proportion, one-sided binomial test vs 0.5, and a
   prediction-level bootstrap of the accuracy difference.
4. **Added Item-3 draw-stability check** (fresh independent one-per-parent draws at
   seeds 43/44/45) and **mega-policy-exclusion accuracy** (original sample minus
   its 25 mega-policy units).
5. **Ran under tmux** (session `followup2`, log `followup2.log`, marker
   `followup2.DONE`). **Acceptance gate passed exactly** against
   `regcheck_followup.log` — all target numbers reproduced. See §4.
6. **Wrote `5_notes/scratch/followup2_replacements.py`** (cheap, no embeddings) to
   answer Item 1's "which sources replaced the mega-doc slots" question. First
   version was wrong (didn't consume the RNG through the research draws first →
   wrong policy sample); fixed to call `sample_research` then `sample_policy` per
   draw and now reproduces the exact 186-distinct/7-mega-doc original policy
   sample. Output `followup2_replacements.txt`.
7. **Rewrote `5_notes/scratch/register_validation_followup2.md`** from scratch with
   the corrected, like-for-like numbers and the blunt updated verdict.

**Files created this session (all gitignored scratch):**
`register_validation_followup2.py`, `followup2.log`, `followup2.DONE`,
`followup2_replacements.py`, `followup2_replacements.txt`,
`register_validation_followup2.md`. **No repo code, no manuscript, no `2_data/`
`4_outputs/` writes, nothing committed.** Working tree is clean.

---

## 4. Key results (from the acceptance-gated run — READ THIS, not the older draft)

### 4.1 Item 1 — one-per-parent is a REBUILD (dropped-then-refilled), not a subset
- Original (draw 1, per-SDG dedup): n=408, 390 distinct parents, 25 multi-parent
  units (6.1%); policy 204 segs / 186 distinct source_docs; research perfectly
  de-clustered. Per-SDG = 12/12 every SDG.
- One-per-parent (draw 2, global dedup): n=408, 408 distinct parents, 0
  multi-parent. Per-SDG = 12/12 every SDG. **Rebuild**: the global-dedup loop
  skips an already-used parent and continues down the same shuffled per-SDG list,
  drawing replacement segments from other docs — quota always met, no attrition.
- Mega-docs (7 docs / 25 segs): SDSN 2024 (SDGs 1,6,7,14,15,16), SDSN 2025
  (4,9,14,15,16,17), UNDP HDR (3,5,7,10,12), WHO Ethics (3,8), EU AI Act (9,13),
  UN Progress 2020 (6,17), UN Progress 2023 (8,10). Mega segments sat in 15/17
  SDGs (all except 2 and 11).
- **Replacement sources (NEW, from `followup2_replacements.py`):** replacements are
  overwhelmingly `pol_sdgi_*` (national SDG-index reports, ~4,225 docs) and
  `pol_ungdc_*` (country reports, ~2,048 docs), plus a few manual OECD/WHO/UN docs.
  **They ARE systematically different**: whole-corpus mean_sent_len 57.8 (mega-docs)
  vs 30.8 (`pol_sdgi_*`) vs 25.2 (`pol_ungdc_*`); words/seg 330 vs 277 vs 280. So the
  one-per-parent sample swaps **global flagship UN prose for short-sentence national
  monitoring reports** in the mega-dominated SDGs — a real genre/institution shift,
  a new (different) composition caveat, not a like-for-like swap. But it is still
  the recommended primary design (cleaner of clustering; balanced; same n).

### 4.2 Item 2 — adjusted-space CV accuracy: CIs + significance
Method: pooled 5-fold stratified-CV predictions (StratifiedKFold(5, shuffle,
seed=42), LR C=1.0), Wilson 95% CI, one-sided binomial vs 0.5. (fold-mean acc ==
pooled acc here, so numbers match follow-up 1 exactly.)

| Classifier | Original 2A (n=408) pooled acc | Wilson 95% CI | p(vs 0.5) | One-per-parent 2B (n=408) pooled acc | Wilson 95% CI | p(vs 0.5) |
|---|---|---|---|---|---|---|
| Register-only (PC1) | 0.456 (186/408) | [0.408, 0.504] | 0.967 | 0.544 (222/408) | [0.496, 0.592] | 0.042 |
| Raw embeddings | 0.909 (371/408) | [0.878, 0.934] | <1e-70 | 0.944 (385/408) | [0.917, 0.962] | <1e-85 |
| **Adjusted** | **0.505 (206/408)** | **[0.457, 0.553]** | **0.441 (ns)** | **0.603 (246/408)** | **[0.555, 0.649]** | **1.9e-05 (sig)** |

- **Original 0.505: NOT distinguishable from 0.5** (p=0.44).
- **One-per-parent 0.603: IS distinguishable from 0.5** (p=1.9e-05; CI excludes 0.5).
  Follow-up 1's "0.603 not chance" is **CONFIRMED** with a proper CI.
- Rise is real, not noise: prediction-bootstrap diff opp−orig = **+0.098, 95% CI
  [+0.024, +0.169], p(diff>0)=0.994**.
- **Why all three rose:** (1) main driver = mega-docs are atypical/hard policy
  (mean_sent_len 276.9 vs 35.6, passive 3.3 vs 9.8, first_person 19.0 vs 5.9,
  nominal 20.0 vs 38.8). Direct test: dropping the 25 mega-policy units from the
  ORIGINAL sample alone raises adj acc 0.505→0.574 (220/383, Wilson [0.524, 0.623],
  p=0.002) — that is +0.070 of the +0.098. (2) Fold-leakage direction: duplicates
  made classification HARDER (mega-docs straddle folds but are atypical, so
  train-fold copies don't help), consistent with the observed rise; not
  "duplicates inflated accuracy". (3) Residual ~+0.03 = composition noise from the
  national-monitoring replacements.

### 4.3 Item 3 — policy other-dist pull: per-SDG, mega-doc exclusion, draw stability
- **Reproduction gate (exact):** pooled reg~dist RAW −0.088 (p=0.075), ADJ −0.074
  (p=0.134); **policy reg ~ other-dist ADJ pooled −0.197 (p=0.0047)** — matches
  follow-up 1 exactly.
- **Per-SDG (n=12 policy each): 0/17 significant at p<0.05.** 14 negative / 3
  positive (SDG 3, 7, 10). Pooled −0.197 is spread thinly across most SDGs, not
  driven by a few.
- **Mega-doc exclusion (drop ALL mega-policy segments, not just one-per-parent):**
  Item-3 sample policy n=199 → other-dist **−0.213 (p=0.003)** — pull SURVIVES and
  slightly strengthens. Not driven by the mega-docs.
- **Draw stability (fresh independent draws):** policy other-dist ρ = −0.197
  (seed 42), −0.130 (43), −0.004 (44), **+0.126 (45)** — **sign flips, mean ≈
  −0.05. NOT draw-stable.** The seed-42 −0.197 is a sample-specific fluctuation.
- **Verdict: downgrade to noise / sample-specific.** Not a robust signal to carry
  to the appendix. Do not report −0.197 as a finding without the draw-instability
  caveat.

### 4.4 Verdict (blunt)
- **"GO, scale to n=120/SDG" still stands** — no integrity red flag survives.
- **Framing change required:** "adjusted-space accuracy ≈ chance" is WRONG for the
  primary (one-per-parent) sample. Correct headline: **"adjusted-space accuracy is
  reduced from 0.91 to ≈0.60 (95% CI 0.55–0.65), significantly above chance."**
  The "≈chance" characterization holds only for the mega-contaminated original.
- **Sampling strategy:** keep one-per-parent as primary. Mega-doc-exclusion is NOT
  needed as a separate arm (one-per-parent already caps mega-docs at one unit; the
  only remaining signal is draw-unstable noise). For n=120/SDG, report the
  replacement-pool composition per SDG as a sensitivity item and consider a
  mega-doc-FLAGGED parallel analysis (the national-monitoring shift is real).
- **Prior-claim reconciliation:** one-per-parent-as-rebuild: confirmed; 2c-artifact:
  confirmed; 2b-null: confirmed; "adj=0.603 not chance": **CONFIRMED**; "2d adj≈chance
  generally": **REVISED**; policy other-dist −0.197 as a signal: **REVISED to noise**;
  replacement sources non-systematic: **REVISED** (they are systematically national).

---

## 5. Environment / repo / data facts (still true)

- Python: `/home/manh/miniforge3/envs/dissertation/bin/python` (conda env
  `dissertation`; `source activate` is BROKEN on this box — use the absolute path).
- **Long jobs MUST run under tmux** (harness kills process group at ~120 s):
  `tmux new-session -d -s <name> "<cmd> > log 2>&1; touch log.DONE"`, poll
  `tail -F log` / `ls log.DONE`. Never poll the PID.
- Git: branch `main`, remote `https://github.com/qmanhbeo/dissertation-bham.git`.
  Committed history: `c2773a9` (follow-up 1), `26455a5` (handoffs), `0f96a3f` (report 1).
  Working tree clean; nothing committed this session; no commits without an explicit ask.
- Units are **segments** (~384-token chunks); research `assigned_sdg` is per-segment;
  alignments positional by row index. Adjusted embeddings never materialised —
  `register_utils.load_G(MODEL)` + `register_utils.project(X, G)`.
- Data paths: research embeddings
  `2_data/3_embedded/mpnet/research_shards/part-*.npy`; research text
  `2_data/2_segmented/research/part-*.jsonl`; research SDG labels
  `2_data/5_supervised_scored/mpnet/paper_scores_shards/metadata/part-*_ids.jsonl`
  (`assigned_sdg`); policy embeddings `2_data/3_embedded/mpnet/policy.npy`;
  policy text `2_data/2_segmented/policy.jsonl`; policy SDG = argmax
  `policy_scores.npy`; policy source docs in
  `2_data/3_embedded/mpnet/metadata/policy_ids.json` (`source_doc`).

## 6. Scripts/files quick map

- `5_notes/scratch/register_validation_followup2.py` — follow-up-2 script (FIXED:
  single module-level `_rng`, three successive draws; valid pooled-prediction CIs;
  draw-stability seeds 43/44/45). Log `followup2.log` (acceptance-gated).
- `5_notes/scratch/followup2_replacements.py` + `.txt` — Item-1 replacement-source
  audit (which sources filled mega-doc slots; genre/length comparison).
- `5_notes/scratch/register_validation_followup2.md` — **the deliverable report**,
  rewritten with corrected numbers. Stop for human review here.
- `5_notes/scratch/register_validation_followup.py` + `regcheck_followup.log` —
  follow-up-1 ground truth (continuous-stream draws 1/2/3). Do not modify.
- `5_notes/scratch/check_concept_same_space.py` — Item-1 (follow-up 1) same-space proof.
- Reports (committed, untouched): `5_notes/register_validation_report.md`,
  `5_notes/register_validation_followup.md`.

---

## 7. What remains / next actions

1. **Human review of `5_notes/scratch/register_validation_followup2.md`** — the
   report is complete and acceptance-gated; it should be read as-is.
2. **On approval, promote the report** to `5_notes/register_validation_followup2.md`
   and commit (as done for the first two reports). Nothing should be committed until
   the user explicitly asks.
3. **Do NOT** start the n=120/SDG scale-up, appendix writing, or any
   `3_writing/`/analysis/table changes without an explicit go.
4. If deeper per-SDG power is needed later, the REGCHECK_N=60 (n=120/SDG) run is the
   natural next step — but it is NOT approved.

**Potential follow-ups worth mentioning (not started):** the draw-instability of the
policy other-dist statistic could be quantified more formally (null distribution via
many seeds) before deciding whether to ever mention it; and the per-SDG
replacement-pool composition should be tabulated at scale-up time.

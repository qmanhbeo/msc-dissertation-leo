# Handoff — Editorial pass: abstract / methods prose correctness + sample-stability ladder estimator fix

Date: 2026-08-05 (updated after the ladder + zero-shot paper-weighting fixes landed).
Repo root: `/home/manh/dissertation`. Read `AGENTS.md` first; it is authoritative.

---

## 1) Context — where we are

The user is doing an editorial quality pass over the dissertation manuscript
(`3_writing/dissertation.tex`, compiled to `4_outputs/dissertation.pdf` via
`python main.py --build-pdf --overwrite`). Guiding rule: **no speculative, stale, or
mislabeled numbers; every number must be grounded in the committed artifacts and its
unit must be correct.**

Timeline of the pass:

- **Issues 2–5** (abstract streamlining, stale p-values, MDE/power wording, abstract
  grammar) — DONE, committed + pushed earlier (commits `92ac826`, `91cf7ac`, `b006d36`,
  `75b82f3`).
- **Sibling audit (S1–S6)** — ran, verified, planned, approved (S4, S5 explicitly via
  the Question tool). Applied via Commit-2 (`db64e99`) and the ladder prose Commit-1
  Step D (`6d5206d`); see §4. A first handoff (`handoff-editorial.md`) was
  written; `handoff.md` was later removed by the repo owner (commit `dbafc56`).
- **Sample-stability ladder investigation.** The user asked to check whether the ladder
  code/output is current, asserting "it can't be that far (>0.002) off". The check proved
  the user right and found a real bug (see §2.4): the ladder mixed two different
  estimators — segment-weighted tiers vs paper-weighted anchor. Option A approved:
  **paper-weight the tiers** (code fix + re-run). **DONE and pushed as `aff6c28`** —
  tiers now converge to the anchor (2m tier gap 0.33951 vs anchor 0.339543; bias 0.121894
  vs 0.121904); c1 re-derived. Commit-1 Step D prose (§6 Commit-1 Step D: tex lines
  519/619/631 unit labels) was intentionally left out of `aff6c28` and is **now DONE and
  pushed as `6d5206d`**.
- **Zero-shot (ZS) research-weighting sibling bug (handoff-research-weight.md).** Audit
  found `score_zeroshot.py` still accumulated research counts/centroids/cohesion per
  segment row (n_papers summed to 3,105,144 = segments) while LR/MLP rows in the same
  tables are paper-weighted (2,536,771). Approved: paper-weight the ZS producer. **DONE
  and pushed as `4ec340d`** — all 6 ZS routes re-run (MPNet 2,536,771; MiniLM/SciBERT
  100,000; concept 99,836), downstream tables regenerated. **This changed the I.1 rank
  deltas that Commit-2 Step C said to "keep"** — SDG17 Δ=10→**9** (ZS rank 7→8), SDG8
  Δ=9→**8** (ZS rank 15→14); `\ZeroShotSemanticRho` 0.63→0.60. Rank claims in
  `dissertation.tex` were updated in `4ec340d`; agreement rates are unchanged
  (67.3/59.5/81.5/81.4/76.3/79.9). `handoff-research-weight.md` has since been removed
  (its §4.2 cross-referenced this file's Commit-2 items, which remain here).

Current `git status`: clean. HEAD = `db64e99`. The outstanding editorial work described
below is now DONE (Commit-1 Step D prose + Commit 2 both landed and pushed); this handoff
is retained for provenance and to flag the two verified-revision details (the three
persisted i1 policy figures in §2.5, and the SDG8/SDG9 Δ=8 tie in the 868 prose).

Workflow conventions observed: plan-then-approve; one concern per commit; never commit
unless asked; leave unrelated dirty files unstaged.

---

## 2) Key known facts (so a fresh agent does not need to re-derive)

### 2.1 Corpus and classifier numbers (all committed macros; do not re-derive)
| Quantity | Macro | Value |
|---|---|---|
| Research abstracts (papers) | `\NResearchAbstracts` | 2,536,771 |
| Research segments | `\NResearchSegments` | 3,105,144 |
| Policy segments | `\NPolicySegments` | 40,597 |
| Policy source docs | `\NPolicyDocs` | 6,367 |
| Classifier training pool | `\NTrainPool` | 52,835 |
| Classifier full reference pool | `\NReferencePool` | 62,173 |
| Classifier held-out test | `\NTestPool` | 9,338 |
| LR test macro-F1 | `\MacroFOne` | 0.816 |
| MLP test macro-F1 | `\MlpMacroFOne` | 0.826 |

Source-of-truth files: `4_outputs/mpnet/tables/num2_coverage_gap.tex`,
`num1_classifier_performance.tex`, `num17_reference_split.tex` (all committed).

### 2.2 Critical unit fact
**3,105,144 is the research SEGMENT count, not a paper count.** The research corpus is
2,536,771 abstracts segmented into 3,105,144 rows (~1.224 seg/abstract; 17.18% of
abstracts exceed the 374-token window and split). Manuscript passages calling 3.1M
"papers" are wrong (this is the root of S1/S3). Confirmed at
`1_code/7_main_analysis/2_appendix/c1_subset_balanced_stability.py:6`.

The classifier (LR) is trained on the reference pool (`\NTrainPool` = 52,835 SDG-labelled
texts from OSDG/Benchmark/Knowledge Hub/SDGi/Aurora), NOT the research abstracts
(`1_code/4_supervised_model_train/0_prepare_data.py`). Research and policy are unlabeled;
the classifier is applied at inference to score segments.

### 2.3 Headline numbers (MPNet paper-weighted replay, committed)
- raw ρ = −0.012, p = 0.963; adjusted ρ = +0.544, p = 0.024; register ρ = −0.390, p = 0.122.
- p-values are macro-driven (`\SpearmanCovRawP`, `\RhoCovTopicP`, `\RhoCovRegisterP`) from
  `4_outputs/mpnet/tables/num5_register_decomposition.tex`.
- MDE (Issue 4, committed `b006d36`): `\HPrimaryMinDetectableR` = 0.63 =
  `tanh((z_{0.975}+z_{0.80})/√(n−3)) = tanh(2.802/3.742)`, Pearson test n=17
  (`1_code/7_main_analysis/1_main_text/2_coverage_semantic_interaction.py:713-716`).

### 2.4 ★ THE SAMPLE-STABILITY LADDER FINDING (RESOLVED — code+outputs in `aff6c28`)

**Question asked:** is the ladder (`4_outputs/appendix/mpnet/c_sample_stability/`) current?
"It can't be that far (>0.002) off."

**Answer: the output WAS fingerprint-current but internally inconsistent — the tiers and
the anchor row computed DIFFERENT quantities.** The user's intuition was exactly right: at
consistent units the ladder converges within 0.00003; the 0.014 gap was an estimator
mismatch introduced by the paper-weighting refactor. This is now FIXED (`aff6c28`).

**Post-fix committed ladder values (`c_sample_stability_table.csv`; anchor row unchanged):**
| Tier | mean semantic gap | policy-text calibration bias |
|---|---|---|
| 50k | 0.34163 | 0.121794 |
| 200k | 0.340233 | 0.1221 |
| 2m | 0.33951 | 0.121894 |
| **full corpus (anchor)** | **0.339543** | **0.121904** |

Tiers now converge upward into the anchor within ~0.001–0.002; c1 `\SubsetGapRho*` now
0.527 (1k) → 1.000 (2m). **Commit-1 Step D prose (tex 519/619/631) is now DONE and
pushed as `6d5206d`.**

Root cause (historical, for context):
1. **Tier rows** (`1_code/7_main_analysis/2_appendix/c_sample_stability.py`,
   `accumulate_draws` lines 401–441) are **segment-weighted**. They sample *documents*
   (`_sample_by_document`, lines 295–308) but then accumulate **per ROW**:
   `assignments = score.argmax(axis=1)` (segment level), `draw.hard_counts += bincount(...)`
   (counts segments), `draw.vector_sums[sdg] += emb[local[mask]].sum(axis=0)` (sums segment
   embeddings), `top_sum_osdg` (segment top scores), `rows_seen` (segment count). So the
   tier semantic gap converges to the **segment-weighted** full-corpus value ≈ 0.326.
2. **Anchor row** is NOT computed by the ladder. `load_policy_state` pulls
   `full_mean_semantic_gap` from the canonical `semantic_gap_distances_lr.json`
   (`c_sample_stability.py:230-236`), which since the paper-weighting refactor uses
   **paper-weighted** research centroids (`research_centroids.npy` built by
   `1_code/5_supervised_model_infer/score_supervised.py:364-378` — each paper collapsed to
   its L2-renormalised mean segment vector, one unit vector per paper). Anchor = 0.339543.
3. **Git proof:** commit `8718d02` ("appendix: regenerate appendix tables under
   paper-weighted research corpus") changed ONLY the anchor row in
   `c_sample_stability_summary.json`/`per_sdg.json`/`table.csv`/`tab_c_sample_stability.tex`:
   `0.325798 → 0.339543` and `0.113233 → 0.121904`. **The tier rows were NOT regenerated.**
   Before that commit the anchor was **0.325798** — within **0.00003** of the 2m tier
   (0.325768). So the ladder was perfectly consistent until the paper-weighting refactor
   (`5261d31` code → `cf80809` main-output replay → `8718d02` appendix replay) swapped the
   anchor to the paper-weighted canonical value while leaving the segment-weighted tiers in
   place. The draw cache was reused because its signature only fingerprints the score/embed
   manifests (which didn't change — only the centroid *construction* changed).
4. **All three tier metrics are segment-weighted** (coverage `hard_counts/rows_seen`, gap
   via segment centroids, calibration bias via segment top-scores) — only the anchor row is
   paper-weighted. All three will be fixed by Option A.

**"Current" in the pipeline sense:** the draw cache at
`2_data/5_supervised_scored/mpnet/paper_sample_seed_42_141/` has
`cache_signature = 6fffa0cf8ca086bb`, which matches `_compute_cache_signature` over the
current score/embed manifests exactly — so a re-run today would skip (cache reused) and
reproduce the committed (inconsistent) values. Cache `schema_version = 2`.
`SCRIPT_VERSION = "1"` in `c_sample_stability.py:822` (fingerprint gate) and
`c1_subset_balanced_stability.py:55`.

**Downstream consumer:** `c1_subset_balanced_stability.py` (Appendix C.1, balanced-subset
rank stability) reads the ladder's `c_sample_stability_draws.jsonl` and correlates each
draw's per-SDG gap ranking with the full-corpus ranking from the paper-weighted
`semantic_gap_distances_lr.json`. It therefore mixes the same two estimators today (its
`\SubsetGapRho*` macros, cited in prose at `dissertation.tex:359` and `:633`). Fixing the
ladder automatically changes c1's inputs; c1 must be re-run and its macros re-verified.

### 2.5 Appendix I.1 assignment-method values (committed
`4_outputs/appendix/mpnet/i1_assignment_method_comparison/`)
- `data/assignment_method_comparison.json`: `research` overall LR-vs-ZS = **0.6734**
  (n_rows = 3,105,144 → per-segment), `research_mlp_vs_zs` overall = **0.5949**.
- Table `tab_app_assignment_method_comparison.tex` Overall row:
  `Overall & 3,105,144 & 67.3 & 59.5 & 40,597 & 81.5`.
  Per-SDG research agree: SDG 17 = 26.1%, SDG 10 = 23.6%.
- **Rank deltas (UPDATED by the ZS fix `4ec340d`; old values in parens):** SDG 17 rank
  LR=17/ZS=**8** (was 7), Δ=**9** (was 10); SDG 8 rank 6/ZS=**14** (was 15), Δ=**8**
  (was 9). `\ZeroShotSemanticRho` = 0.60 (was 0.63). Prose at `dissertation.tex:868`
  already reflects the new ranks (updated in `4ec340d`) — do NOT revert to Δ=10/Δ=9.
- **Recomputed read-only** from committed npys via `doc_level_assignments`
  (`1_code/7_main_analysis/0_shared/semantic_gap_shared.py`): policy doc-level LR-vs-ZS =
  **81.4%** (n_docs 6,367); policy segment MLP-vs-ZS = **76.3%**; policy doc-level MLP-vs-ZS
  = **79.9%**. These three ARE persisted in the committed
  `data/assignment_method_comparison.json` (`policy_lr_vs_zs.document`,
  `policy_mlp_vs_zs.segment`, `policy_mlp_vs_zs.document`) — cite directly; no re-run
  needed to confirm them. Agreement rates are unchanged by the ZS fix (they are
  per-segment computations); only `gap_rank_*` changed.

### 2.6 Git archaeology explaining I.1 prose staleness (unchanged from prior handoff)
- Appendix I.1 table was regenerated 2026-08-05 in commit `8718d02` — **after** the prose
  numbers (62.3/79.7/80.4/16.8/15.9/76.8/78.1) were written 2026-07-31 in `05c45eb`.
  Hence `dissertation.tex:868` is stale relative to the committed table.
- The Research counts in the I.1 table are per-*segment* (n=3,105,144), despite the header
  "Research (papers)" and prose "3.1M research papers".

---

## 3) Actions / decisions made + files changed this session, and why

### Earlier this pass (already committed + pushed)
| Commit | Issue | What | Why |
|---|---|---|---|
| `92ac826` | 2 | Abstract streamlined: removed duplicated result sentence; merged robustness claims. | Abstract repeated the same findings twice. |
| `91cf7ac` | 3 | p-value macros `\RhoCovTopicP`/`\RhoCovRegisterP`/`\SpearmanCovRawP` in `generate_tex_macros.py`; regenerated `num5_register_decomposition.tex` (3 models); replaced hardcoded p-values at 6 sites; removed Discussion double-equals; reframed abstract as "partial, not complete, cancellation". | Hardcoded p-values went stale; `==` was mostly a symptom. |
| `b006d36` | 4 | Removed MDE clause from abstract; grounded Results bound at 0.63 (Fisher-z Pearson MDE, formula footnote); replaced appendix "≈0.6" with `\HPrimaryMinDetectableR`. | No speculative numbers. |
| `75b82f3` | 5 | Abstract methods sentence rewritten (classifier trained on `\NTrainPool` SDG-labelled texts, applied to score segments). | Fix grammar; state training-pool size. |

### This session (investigation only — NO file changes yet)
1. **Resumed the prior handoff** and, per user instruction, **audited the sample-stability
   ladder first** (`c_sample_stability.py`, committed JSONs, git history, cache state).
2. **Found and verified the estimator mismatch** (full details in §2.4): segment-weighted
   tiers vs paper-weighted anchor, introduced by the paper-weighting refactor
   (`5261d31`/`cf80809`/`8718d02`). Confirmed the user's intuition ("can't be >0.002 off")
   was right — the pre-refactor anchor (0.325798) matched the tier plateau within 0.00003.
3. **Decisions made with the user** (via the Question tool):
   - **Option A approved: paper-weight the tiers** so the ladder converges to the canonical
     paper-weighted anchor (0.3395/0.1219), making the prose convergence claims true again.
     (Rejected: prose-only relabel — masks a real mixed-estimator table; segment-weighted
     anchor — contradicts the main-text `\MeanSemanticGap` = 0.340.)
   - **Sequencing approved:** ladder fix first (commit 1), then the independent handoff
     fixes (S3/S4/S5 + remaining S1 unit labels) as commit 2.

### After the original handoff (landed while the repo was on `dbafc56`)
| Commit | What | Notes |
|---|---|---|
| `aff6c28` | **Ladder paper-weighting fix** (Commit-1 Steps A–C): `c_sample_stability.py` `accumulate_draws` paper-weighted via `paper_units_from_shard`, draw-cache schema v3, `SCRIPT_VERSION` 1→2; ladder + c1 re-run. | Prose Step D (tex 519/619/631) later done in `6d5206d` (§4). |
| `4ec340d` | **ZS research-weighting fix** (from `handoff-research-weight.md`): `score_zeroshot.py` paper-weighted (collapses each shard to paper units); 6 ZS routes re-run; fp fixes in `0_coverage_gap.py` + `h1_cross_method_gap_values.py`; stale docstring fixed; downstream tables (cross-sensitivity, h1, j1, k1, i1) regenerated; `dissertation.tex` rank claims updated; PDF rebuilt. | Changed I.1 rank deltas (SDG17 Δ=10→9, SDG8 Δ=9→8) and `\ZeroShotSemanticRho` 0.63→0.60 — see §2.5. |
| `6d5206d` | **Commit-1 Step D prose**: tex 519/619/631 unit labels (`\NResearchSegments{}-segment`), 50k claim softened to "within ~0.002". | Completes Commit-1 (code already in `aff6c28`). |
| `db64e99` | **Commit-2 I.1/macro-F1 fixes**: i1 generator header+docstring relabel (Research segments), table regenerated (only header changed), 865/868 prose updated (67.3/81.5/81.4/26.1/23.6/76.3/79.9/59.5; SDG8/SDG9 Δ=8 tie note), `macro-F1`→`macro-$F_1$` (8 sites). | Completes the sibling-audit Commit-2. |

**Files changed to date in this pass:** `c_sample_stability.py`, `score_zeroshot.py`,
`0_coverage_gap.py`, `h1_cross_method_gap_values.py`,
`3_generate_cross_sensitivity_table.py`, `i1_assignment_method_comparison.py`,
`dissertation.tex` (rank claims + ladder prose + I.1 prose + `macro-$F_1$`), and the
regenerated `4_outputs/` tables/data (incl. `tab_app_assignment_method_comparison.tex`
header relabel). Working tree clean vs HEAD.

---

## 4) What remains and why

The remaining work was (a) the **Commit-1 Step D ladder prose** (unit labels at tex
519/619/631 — deliberately excluded from `aff6c28`), and (b) the **Commit-2 I.1/`macro-F1`
fixes** from the sibling audit. Both code fixes (ladder `aff6c28`, ZS `4ec340d`) are done;
the prose that quotes their numbers is not. Nothing is blocked.

**Both remaining items are now DONE and pushed** (`6d5206d` Commit-1 Step D prose,
`db64e99` Commit-2 I.1/macro-F1 fixes). Sections 6/7 below are kept verbatim as the
record of what was executed; do not re-run the pipeline stages.

Summary (all now executed — see `6d5206d` and `db64e99`):
1. Prose in `dissertation.tex`: S1 unit labels (519/619/631; 50k claim softened to
   "within ~0.002" since raw 50k-vs-2m diff is 0.00212 > 0.002), S3 stale I.1 numbers
   (865/868 updated to 67.3/81.5/81.4/26.1/23.6/76.3/79.9/59.5 + SDG8/SDG9 Δ=8 tie note),
   S4 I.1 header + generator (i1 script line ~352 + docstring ~10 relabelled to segments),
   S5 `macro-F1` → `macro-$F_1$` (8 sites). S2 (relabel "fully converged") was **obsolete**
   — the estimator fix made the original convergence claims numerically true; the ladder
   prose now states the verified values.
2. PDF rebuilt, verified with `pdftotext`, committed in two commits (Commit-1 Step D
   prose; then Commit 2), pushed.

Why these remained: `aff6c28` was scoped to code + outputs so the estimator fix could be
verified before touching prose; the ZS fix `4ec340d` landed between, updating the rank
claims but not the stale agreement-rate numbers at 865/868. Both are now resolved.

---

## 5) Concerns to emphasize

1. **The ladder fix changed many numbers (already landed in `aff6c28`).** ALL tier rows
   now converge to ~0.339–0.340 (bias ~0.122); c1 `\SubsetGapRho*` = 0.527→1.000. Prose at
   `dissertation.tex:359, 519, 619, 631, 633` quotes these macros — do not hand-edit the
   numbers; rebuild and re-read the new macros, then check the *claims* against the new
   values. **Specific trap at line 519:** the claim "at 50,000 papers … mean semantic gap
   … within 0.002 of the full-corpus estimate" — raw 50k-vs-2m is now **0.00212 > 0.002**
   (50k 0.34163 vs 2m 0.33951), i.e. marginally outside the claimed bound. Either cite
   the 200k tier (0.340233, within 0.002) or reword to "≈0.002 / within ~0.002".
2. **Cache invalidation (DONE in `aff6c28`) — do not re-trigger.** The draw cache
   (`2_data/5_supervised_scored/mpnet/paper_sample_seed_42_141/`) was invalidated to
   schema v3 and rebuilt with paper-weighted aggregates. `SCRIPT_VERSION` is "2". Only
   touch this if the ladder needs another re-run.
3. **Papers must not span shard boundaries.** `paper_units_from_shard` raises if a paper
   crosses a shard boundary (`prev_last_paper_id` threading,
   `1_code/7_main_analysis/0_shared/research_score_shards.py:108-159`). This was handled
   in `aff6c28`; keep the threading if the ladder is ever re-run.
4. **Match the canonical recipe exactly.** Canonical centroids = normalized mean of
   *paper-level unit vectors* (`score_supervised.py:364-378`); paper assignment = argmax
   of the mean segment score vector (`paper_units_from_shard`); paper top score = max of
   the mean score vector (`group_rows_by_paper(scores, starts)[0].max(1)`). Coverage must
   be paper counts (`hard_counts` over papers), `rows_seen` = paper count, `top_sum_osdg`
   = sum of paper top scores. (Verified correct in `aff6c28`; reference only.)
5. **Re-running the ladder is a long job → tmux (AGENTS.md HARD RULE).** 26 shards × 1100
   draws (11 tiers × 100) of accumulation. Launch with
   `tmux new-session -d -s ladder "python main.py --appendix-c-sample-stability --overwrite > /tmp/ladder.log 2>&1; touch /tmp/ladder.log.DONE"`,
   poll `tmux capture-pane` / `tail` / `ls /tmp/ladder.log.DONE`. Never `setsid`/`disown`.
   (Only needed if the ladder must be regenerated — it is current as of `aff6c28`.)
6. **Verify, don't trust — the three policy I.1 figures are now in the committed json.**
   `policy_lr_vs_zs.document` = 81.4%, `policy_mlp_vs_zs.segment` = 76.3%,
   `policy_mlp_vs_zs.document` = 79.9% are all persisted in
   `data/assignment_method_comparison.json` (regenerated by `4ec340d`) and were re-run and
   reconfirmed in Commit-2 Step B (only the header relabel changed). Committed-table values
   (67.3/59.5/81.5; SDG17 26.1; SDG10 23.6) are safe to cite. Agreement rates are
   unchanged by the ZS fix — only the `gap_rank_*` columns changed (to Δ=9/Δ=8, ranks
   8/14; already reflected in tex).
7. **Do NOT "fix" the appendix by changing research agreement to paper-level.** Research
   rows in the scored corpus are per-segment (3,105,144); the I.1 header must become
   "Research (segments)", not the numbers changed to paper counts. Coverage *profiles* in
   the main text are paper-weighted; do not conflate.
8. **`macro-F1` → `macro-$F_1$` replaceAll must match only the literal `macro-F1`**
   (occurs at dissertation.tex lines 310/323/342/655/659/666/677/681, incl. `CV macro-F1`).
   It must NOT touch `micro-F1`, `\MlpMacroFOne`, `\MacroFOne`. A single replaceAll of the
   literal is safe (verified).
9. **Table Notes at line 865 are stale the same way as line 868**: it says research
   agreement is "per paper" and "MLP research per-paper scores are not persisted" — both
   now false (research is per segment; research MLP-vs-ZS is persisted and displayed,
   overall 59.5%). Fix in the same commit.
10. **Unrelated dirty files must stay unstaged** when committing: this handoff,
    `4_outputs/not_in_replay/distributional/mpnet/adjusted/g_distributional_gap_records.jsonl`,
    `4_outputs/conceptual_figs/fig6_pipeline_flowchart.pdf` (regenerated each `--build-pdf`).
    (At the time of writing, `git status` is clean — confirm before committing.)
11. After regenerating the i1 table, `git diff` it — DONE in Commit-2 Step B: the ONLY
    change was the header relabel ("Research (papers)" → "Research (segments)"), confirming
    the data JSON is deterministic (three policy figures byte-identical to the committed
    81.4/76.3/79.9). The `gap_rank_*` columns stayed at their post-ZS-fix values (Δ=9/Δ=8)
    as expected.
12. Do not run `--cold-replay` or other long pipeline stages; all required artifacts exist.
13. `c1` re-run depends on the regenerated draws JSONL; its fingerprint gate
    (`fingerprint_of(full_gap_path, draws_path) + SCRIPT_VERSION`) will re-run it once the
    draws change — run it explicitly with `--overwrite` after the ladder completes.

---

## 6) The whole comprehensive plan

ALL steps of both commits are now DONE and pushed: Commit-1 code `aff6c28`, Commit-1
Step D prose `6d5206d`, ZS fix `4ec340d`, Commit-2 `db64e99`. The plan is retained below
verbatim as the executed record. Do not re-run any stage.

### Commit 1 — Ladder paper-weighting fix (Option A) — **COMPLETE (`aff6c28` code+outputs; `6d5206d` Step D prose)**

**Step A — Edit `1_code/7_main_analysis/2_appendix/c_sample_stability.py`:** DONE in
`aff6c28` (paper-level `accumulate_draws`, cache schema v3, `SCRIPT_VERSION` 1→2,
docstring updated). Keep as-is.

**Step B — Re-run the ladder (long job, tmux) + c1:** DONE in `aff6c28`.

**Step C — Verify:** DONE. Post-fix values (§2.4): tiers converge into the anchor — 50k
0.34163, 200k 0.340233, 2m 0.33951, anchor 0.339543 (bias 0.121794/0.1221/0.121894/
0.121904); c1 `\SubsetGapRho*` 0.527 (1k) → 1.000 (2m). `num_c_sample_stability.tex` and
`num_c1_subset_stability.tex` hold the new macros. **The 50k "within 0.002" claim at line
519 needs a wording check in Step D (raw 50k-vs-2m = 0.00212 > 0.002).**

**Step D — Prose fixes in `3_writing/dissertation.tex` (STILL PENDING — commit-1 scope:
ladder prose):**
- Line 519 (concept-retrieval): unit label —
  `while remaining far smaller than the full 3.1-million-paper corpus` →
  `while remaining far smaller than the full \NResearchSegments{}-segment research corpus`.
  **"within 0.002 of the full-corpus estimate of \SampleMeanSemanticGapTwoM{}" is now
  MARGINALLY FALSE** (50k 0.34163 vs 2m 0.33951 → 0.00212). Fix by citing the 200k tier
  (0.340233, diff 0.00069) or rewording to "within ≈0.002 / within ~0.002". Note
  `\SampleMeanSemanticGapFiftyK` = 0.342 and `\SampleMeanSemanticGapTwoM` = 0.340 (3dp
  macros), so the claim "0.002" is also borderline at macro precision.
- Line 619: `The full \SampleStabilityFullCorpusN{}-paper analysis` →
  `\SampleStabilityFullCorpusN{}-segment analysis`.
- Line 631: `The full \SampleStabilityFullCorpusN{}-paper result` →
  `\SampleStabilityFullCorpusN{}-segment result`. The "already stabilise to within 0.001
  of their full-corpus values" claim should now be TRUE (200k 0.340233 vs anchor 0.339543
  → 0.00069 < 0.001) — keep, after verifying against the new macros. Do NOT relabel to
  "fully converged" (that was the obsolete S2 patch).
- Re-check any other spot quoting `\Sample*` macros for unit/claim consistency (359, 633).

**Step E — Build + verify + commit:** now a smaller commit (prose + PDF only — the
ladder/c1 code+outputs are already committed in `aff6c28`).
```
python main.py --build-pdf --overwrite
pdftotext 4_outputs/dissertation.pdf - | grep -c '3.1-million-paper'   # expect 0
```
Assert: no `3.1-million-paper`, no `-paper analysis/result` in ladder prose; line-519
claim no longer overstates 50k precision; `segments` labels present. Commit only
`3_writing/dissertation.tex` + `4_outputs/dissertation.pdf` (and this handoff if wanted).
Suggested message:
```
fix(writing): correct sample-stability ladder unit labels + 50k precision claim (Commit-1 Step D)
```

### Commit 2 — I.1 fixes + macro-F1 normalization (sibling audit S3/S4/S5) — **COMPLETE (`db64e99`)**

**Step A — Fix the I.1 generator (S4):**
`1_code/7_main_analysis/2_appendix/i1_assignment_method_comparison.py`:
- Line ~352: `Research (papers)` → `Research (segments)` in the table header.
- Line ~10 docstring: `per-paper` → `per-segment`.

**Step B — Regenerate the appendix table:**
```
python main.py --appendix-i1-assignment-method --overwrite
```
- Confirm header `Research (segments)` and Overall row unchanged:
  `Overall & 3,105,144 & 67.3 & 59.5 & 40,597 & 81.5`.
- Reconfirm the three non-persisted figures from the run log /
  `data/assignment_method_comparison.json`: policy doc-level LR-vs-ZS (~81.4%), policy
  MLP-vs-ZS segment (~76.3%), doc-level (~79.9%). Use freshly-generated values in prose.

**Step C — Prose fixes in `dissertation.tex`:**
- Line 865 (table Notes): research "each paper" → "each segment"; rewrite the stale "MLP
  research per-paper scores are not persisted" clause (research MLP-vs-ZS IS computed and
  displayed, overall 59.5%).
- Line 868 (assignment-method discussion): replace stale numbers —
  `62.3% over the 3.1M research papers` → `67.3% over the \NResearchSegments{} research
  segments`; `79.7%` → `81.5%`; `80.4% at document level` → `81.4%` (reconfirmed);
  `16.8% ... 15.9%` → `26.1% ... 23.6%`; `76.8% ... 78.1%` → `76.3% ... 79.9%`
  (reconfirmed); replace the "research MLP not reported" clause with the 59.5% figure.
  **Rank claims are ALREADY at their post-ZS-fix values in tex (`4ec340d`)**: SDG 17 Δ=9
  (ZS rank 8), SDG 8 ranks 6/14 Δ=8 — do NOT change them and do NOT revert to the old
  Δ=10/Δ=9 (that pairing is obsolete). Only the agreement-rate percentages above are
  still stale.
- S5: `replaceAll` literal `macro-F1` → `macro-$F_1$` across the file (lines
  310/323/342/655/659/666/677/681). Optional: line 342 `1/17` → `$1/17$`.

**Step D — Build + verify + commit 2:**
```
python main.py --build-pdf --overwrite
pdftotext 4_outputs/dissertation.pdf - | grep ...
```
Assert ABSENCE of: `3.1-million-paper`, `62.3%`, `79.7%`, `80.4%`, `16.8%`, `15.9%`,
`76.8%`, `78.1%`, `not persisted`, `macro-F1`. Presence of: `67.3%`, `81.5%`, `26.1%`,
`23.6%`, `59.5%`, `segments`, `macro-$F_1$`. Compiled I.1 header reads "Research
(segments)". Commit (only the generator, regenerated i1 table/data, `dissertation.tex`,
`dissertation.pdf`). Suggested message:
```
fix: correct I.1 research units and stale assignment-method numbers; normalize macro-F1

- I.1 table header + docstring: Research (papers) -> Research (segments) (n=3,105,144
  rows are segments, not papers).
- Prose updated to the 2026-08-05 regenerated table: research LR-vs-ZS 62.3->67.3%,
  SDG17 16.8->26.1%, SDG10 15.9->23.6%, policy 79.7->81.5%, doc-level 80.4->81.4%,
  MLP 76.8->76.3% / 78.1->79.9%; drop the now-false "research MLP not persisted" claims
  (research MLP-vs-ZS = 59.5%).
- Normalize macro-F1 -> macro-$F_1$ for consistent typography.
```
Then `git push`.

---

## 7) Exactly what was interrupted

The repo has moved past the original handoff's "investigation only" state. Both code
fixes have landed and been pushed, AND both remaining prose fixes have now landed:

- **Ladder paper-weighting fix (`aff6c28`)** — Commit-1 Steps A–C (code, ladder re-run,
  c1 re-run, verification) DONE. **Commit-1 Step D prose (tex 519/619/631 unit labels +
  the 50k precision claim) is now DONE and pushed as `6d5206d`.**
- **ZS research-weighting fix (`4ec340d`)** — from `handoff-research-weight.md` (now
  removed). DONE: producer + fp fixes + all downstream regenerations + rank-claim prose
  + PDF. It changed the I.1 rank deltas (Δ=10→9, Δ=9→8) and `\ZeroShotSemanticRho`
  (0.63→0.60).
- **Commit 2 (I.1 header/`macro-F1` fixes)** — DONE and pushed as `db64e99`: i1 generator
  header/docstring relabel (segments), regenerated table (only the header changed — the
  three policy figures were confirmed persisted in the committed JSON), 865/868 prose
  (incl. the SDG8/SDG9 Δ=8 tie note), and `macro-$F_1$` normalization.

**To resume / wrap up:** the editorial pass is complete. No pipeline stages need to be
re-run. Sibling-audit findings S1–S6 are all resolved; S2 was superseded by the Option-A
fix (the "fully converged" relabel was obsolete).

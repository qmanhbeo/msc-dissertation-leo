# Handoff — Editorial pass: abstract / methods prose correctness + sample-stability ladder estimator fix

Date: 2026-08-05 (updated). Repo root: `/home/manh/dissertation`. Read `AGENTS.md` first; it is authoritative.

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
  the Question tool) but NOT yet applied. A first handoff (`handoff-editorial.md`) was
  written; `handoff.md` was later removed by the repo owner (commit `dbafc56`).
- **NEW (this session): sample-stability ladder investigation.** The user asked to check
  whether the sample-stability ladder code/output is current, asserting "it can't be that
  far (>0.002) off". **The check proved the user right and found a real bug** (see §2.4):
  the ladder mixes two different estimators — segment-weighted tiers vs paper-weighted
  anchor — so the apparent 0.014 "non-convergence" is an artifact of the paper-weighting
  refactor, not real instability. The user approved **Option A: paper-weight the tiers**
  (code fix + re-run) and the sequencing "ladder fix first, then handoff fixes" (two
  commits). **Nothing has been edited yet.**

Current `git status`: clean except untracked `handoff-editorial.md` (this file).
HEAD = `dbafc56`. No code, output, or prose changes have been made for the ladder fix.

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

### 2.4 ★ THE SAMPLE-STABILITY LADDER FINDING (this session's main deliverable)

**Question asked:** is the ladder (`4_outputs/appendix/mpnet/c_sample_stability/`) current?
"It can't be that far (>0.002) off."

**Answer: the output IS fingerprint-current but internally inconsistent — the tiers and
the anchor row compute DIFFERENT quantities.** The user's intuition was exactly right: at
consistent units the ladder converges within 0.00003; the 0.014 gap is an estimator
mismatch introduced by the paper-weighting refactor.

Committed ladder values (`c_sample_stability_summary.json`):
| Tier | mean semantic gap | policy-text calibration bias |
|---|---|---|
| 50k | 0.327891 | 0.113122 |
| 200k | 0.326426 | 0.113391 |
| 2m | 0.325768 | 0.113221 |
| **full corpus (anchor)** | **0.339543** | **0.121904** |

Root cause:
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
  Per-SDG research agree: SDG 17 = 26.1%, SDG 10 = 23.6% (SDG 17 rank LR=17/ZS=7, Δ=10;
  SDG 8 rank 6/15, Δ=9).
- **Recomputed read-only** from committed npys via `doc_level_assignments`
  (`1_code/7_main_analysis/0_shared/semantic_gap_shared.py`): policy doc-level LR-vs-ZS =
  **81.4%** (n_docs 6,367); policy segment MLP-vs-ZS = **76.3%**; policy doc-level MLP-vs-ZS
  = **79.9%**. These three are NOT persisted in any committed json — re-run the i1 script
  to reconfirm before writing prose (§6 Step D).

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

**Files changed this session:** none (investigation only). Working tree clean vs HEAD
except this handoff. `2_data/` untouched; no long-running stages run.

---

## 4) What remains and why

All remaining work is: (a) the **ladder paper-weighting fix** (code + re-run + prose
verification), and (b) the **I.1/`macro-F1` fixes** from the sibling audit. Nothing is
blocked; execution paused at the "write handoff" gate per the user's request.

Summary:
1. Fix `c_sample_stability.py` to accumulate **paper-level units** per draw (mirroring
   `score_supervised.py` / `paper_units_from_shard`), invalidate the stale draw cache,
   bump script/cache versions.
2. Re-run the ladder (long job → tmux) and re-run c1; verify tiers converge to
   ~0.3395/~0.1219 within ~0.001–0.002.
3. Prose fixes in `dissertation.tex`: S1 unit labels (519/619/631), S3 stale I.1 numbers
   (865/868), S4 I.1 header + generator (i1 script, line ~352 + docstring ~10), S5
   `macro-F1` → `macro-$F_1$`. S2 (relabel "fully converged") is now **mostly obsolete** —
   the estimator fix makes the original convergence claims numerically true; re-verify the
   macro values instead.
4. Rebuild PDF, verify with `pdftotext`, commit in two commits, push.

Why these remain: the user asked for a handoff before execution so a fresh agent can take
over with full context; the ladder finding upgrades S2 from a prose issue to a code fix.

---

## 5) Concerns to emphasize

1. **The ladder fix changes many numbers.** After paper-weighting, ALL tier rows change
   (semantic gap tiers converge to ~0.339–0.340; calibration bias to ~0.122), plus c1's
   `\SubsetGapRho*` values. Prose at `dissertation.tex:359, 519, 619, 631, 633` quotes
   these macros — do not hand-edit the numbers; rebuild and re-read the new macros, then
   check the *claims* ("within 0.002/0.001 of the full-corpus estimate") against the new
   values. If a tier (e.g. 50k) is no longer within the claimed bound, adjust the wording.
2. **Cache invalidation is mandatory and easy to miss.** The draw cache
   (`2_data/5_supervised_scored/mpnet/paper_sample_seed_42_141/`, `schema_version: 2`)
   stores SEGMENT-weighted aggregates. `run()` only clears the cache when
   `cache_signature` (manifest hashes) differs — that will NOT change after a code edit.
   You MUST bump `schema_version` to 3 AND fold it into `_compute_cache_signature`
   (c_sample_stability.py:107-113) (or add an explicit schema check) so the stale `.npz`
   aggregates are rejected. Bump `SCRIPT_VERSION` "1"→"2" (line 822) so the fingerprint
   gate records the change.
3. **Papers must not span shard boundaries.** `paper_units_from_shard` raises if a paper
   crosses a shard boundary (`prev_last_paper_id` threading,
   `1_code/7_main_analysis/0_shared/research_score_shards.py:108-159`). Thread
   `prev_last_paper_id` across the shard loop in `accumulate_draws` and fail closed, as
   `score_supervised.py` does. Do not silently fall back to row-level accumulation.
4. **Match the canonical recipe exactly.** Canonical centroids = normalized mean of
   *paper-level unit vectors* (`score_supervised.py:364-378`); paper assignment = argmax
   of the mean segment score vector (`paper_units_from_shard`); paper top score = max of
   the mean score vector (`group_rows_by_paper(scores, starts)[0].max(1)`). Coverage must
   be paper counts (`hard_counts` over papers), `rows_seen` = paper count, `top_sum_osdg`
   = sum of paper top scores. If any of these stays segment-level, the tiers will not
   converge and the whole exercise is wasted.
5. **Re-run is a long job → tmux (AGENTS.md HARD RULE).** 26 shards × 1100 draws
   (11 tiers × 100) of accumulation. Launch with
   `tmux new-session -d -s ladder "python main.py --appendix-c-sample-stability --overwrite > /tmp/ladder.log 2>&1; touch /tmp/ladder.log.DONE"`,
   poll `tmux capture-pane` / `tail` / `ls /tmp/ladder.log.DONE`. Never `setsid`/`disown`.
   The script is resume-safe per-draw, but killing mid-accumulation loses the pass.
6. **Verify, don't trust — especially the three non-persisted I.1 figures** (policy
   doc-level LR-vs-ZS ≈ 81.4%, policy MLP segment ≈ 76.3%, doc-level ≈ 79.9%). They are
   not in any committed json; re-run the i1 script (Step D) and use freshly-generated
   values in prose. Committed-table values (67.3/59.5/81.5; SDG17 26.1; SDG10 23.6) are
   safe to cite directly.
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
    (At the time of writing, `git status` is clean except this file — confirm before
    committing.)
11. After regenerating the i1 table, `git diff` it and confirm it shows ONLY the header
    change (plus legitimately new numbers). If numbers change for unknown reasons, STOP.
12. Do not run `--cold-replay` or other long pipeline stages; all required artifacts exist.
13. `c1` re-run depends on the regenerated draws JSONL; its fingerprint gate
    (`fingerprint_of(full_gap_path, draws_path) + SCRIPT_VERSION`) will re-run it once the
    draws change — run it explicitly with `--overwrite` after the ladder completes.

---

## 6) The whole comprehensive plan

Two commits, executed in order: **(1) ladder paper-weighting fix**, **(2) I.1 + macro-F1
handoff fixes**. After each, verify before moving on.

### Commit 1 — Ladder paper-weighting fix (Option A)

**Step A — Edit `1_code/7_main_analysis/2_appendix/c_sample_stability.py`:**
- Imports: add `read_shard_paper_ids`, `paper_run_starts`, `group_rows_by_paper`,
  `paper_units_from_shard` from `research_score_shards` (already imported:
  `ResearchShard`, `build_research_shards`, `aggregate_research_scores`).
- Rewrite `accumulate_draws` (lines 401–441) to per-shard **paper-level** accumulation:
  - Per shard: `paper_ids = read_shard_paper_ids(shard.ids_path)`; verify
    `len(paper_ids) == score.shape[0]`; `starts = paper_run_starts(paper_ids)`;
    `paper_emb, paper_assigned, seg_counts, last = paper_units_from_shard(emb, score,
    paper_ids, shard.name, prev_last_paper_id=prev_last)`; `prev_last = last`;
    `paper_scores, _ = group_rows_by_paper(score, starts)`; `paper_top = paper_scores.max(1)`;
    build `row_to_paper = np.searchsorted(starts, np.arange(score.shape[0]), side='right') - 1`.
  - Per draw: `local = draw.global_indices[left:right] - shard.start`;
    `papers = np.unique(row_to_paper[local])`; accumulate
    `draw.hard_counts += np.bincount(paper_assigned[papers], minlength=N_SDG)`,
    `draw.top_sum_osdg += float(paper_top[papers].sum())`, and per-SDG
    `draw.vector_sums[sdg] += paper_emb[papers[mask]].sum(axis=0)` where
    `mask = paper_assigned[papers] == sdg`; `draw.rows_seen += len(papers)`.
  - Keep the post-loop sanity check: `rows_seen == number of sampled papers`.
- **Cache invalidation:** bump `write_cache_manifest` `schema_version` to 3 and include
  the version in `_compute_cache_signature` (e.g. `hasher.update(b"schema_v3")`), so the
  committed segment-level `.npz` aggregates are rejected and rebuilt. (Verify by checking
  the run log shows cache clearing or a new signature.)
- Bump `SCRIPT_VERSION` "1" → "2" (line 822).
- Update the module docstring / `DrawAccumulator` comments to state tiers are
  paper-weighted (matching `RESEARCH_WEIGHTING_UNIT = "document"`).

**Step B — Re-run the ladder (long job, tmux):**
```
python main.py --appendix-c-sample-stability --overwrite
```
Poll to completion. Then re-run c1 (fast):
```
python main.py --appendix-c1-balanced-subset --overwrite
```

**Step C — Verify (before touching prose):**
- `4_outputs/appendix/mpnet/c_sample_stability/data/c_sample_stability_summary.json`:
  tier plateau should now sit at ~0.3395 (gap) / ~0.1219 (bias), within ~0.001–0.002 of
  the full-corpus anchor row (0.339543/0.121904). 50k tier is the loosest (std was 0.0028
  segment-weighted); check whether "within 0.002 at 50k" still holds — if not, adjust the
  claim/wording at line 519.
- `tab_c_sample_stability.tex`: Full corpus row `0.340 / 0.122`; tiers converge upward.
- `num_c_sample_stability.tex`: new `\SampleMeanSemanticGap*`, `\SamplePolicyBias*`,
  `\SampleMacroVariance*` values. Record them for the prose check.
- `c1_subset_balanced_stability.json` + `num_c1_subset_stability.tex`: new
  `\SubsetGapRho*` values; compare against prose at lines 359 and 633 (claims like
  "already ρ = X at 10k", "converging to ρ = Y at the full corpus").
- `git diff` the ladder/c1 outputs: expect ALL tier rows + c1 rows to change; anchor row
  unchanged (0.339543/0.121904).

**Step D — Prose fixes in `3_writing/dissertation.tex` (commit-1 scope: ladder prose):**
- Line 519 (concept-retrieval): unit label —
  `while remaining far smaller than the full 3.1-million-paper corpus` →
  `while remaining far smaller than the full \NResearchSegments{}-segment research corpus`.
  Keep the "within 0.002 of the full-corpus estimate of \SampleMeanSemanticGapTwoM{}"
  claim ONLY if the re-run confirms it; otherwise tighten to the 2M-tier estimate
  (this is now expected to hold since tiers ≈ 0.3395 ≈ anchor).
- Line 619: `The full \SampleStabilityFullCorpusN{}-paper analysis` →
  `\SampleStabilityFullCorpusN{}-segment analysis`.
- Line 631: `The full \SampleStabilityFullCorpusN{}-paper result` →
  `\SampleStabilityFullCorpusN{}-segment result`. The "already stabilise to within 0.001
  of their full-corpus values" claim should now be TRUE — keep, after verifying against
  the new macros. Do NOT relabel to "fully converged" (that was the obsolete S2 patch).
- Re-check any other spot quoting `\Sample*` macros for unit/claim consistency.

**Step E — Build + verify + commit 1:**
```
python main.py --build-pdf --overwrite
pdftotext 4_outputs/dissertation.pdf - | grep -c '0.326'   # expect fewer/none in ladder prose
```
Assert: no `3.1-million-paper`; ladder prose numbers match the new macros; `segments`
labels present. Commit (only: `c_sample_stability.py`,
`4_outputs/appendix/mpnet/c_sample_stability/**`, `4_outputs/appendix/mpnet/c1_subset_balanced_stability/**`,
`3_writing/dissertation.tex`, `4_outputs/dissertation.pdf`). Suggested message:
```
fix(appendix): paper-weight the sample-stability tiers to match the canonical anchor

Tiers accumulated segment-level centroids/counts while the full-corpus anchor row
pulled the paper-weighted canonical semantic gap (0.3395) — a mixed-estimator table
introduced by the paper-weighting refactor (5261d31/cf80809/8718d02). Collapse each
draw's sampled segments to paper units (paper_units_from_shard) for hard counts,
centroid sums and top scores; bump draw-cache schema (v3) + SCRIPT_VERSION (2) so the
stale segment-level aggregates are rebuilt. Tiers now converge to the anchor within
~0.001-0.002; c1 balanced-subset ranks re-derived on the paper-weighted draws.
```

### Commit 2 — I.1 fixes + macro-F1 normalization (sibling audit S3/S4/S5)

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
  Rank claims (SDG 17 Δ=10; SDG 8 ranks 6/15 Δ=9) verified correct — keep.
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

The user asked me to STOP before executing, so a fresh agent receives this handoff.

- **Done before the stop:** resumed the prior handoff; audited the sample-stability ladder
  per the user's request ("first check if the sample stability ladder code/output is
  current"); proved the user's intuition correct and found the estimator mismatch (§2.4);
  presented findings + options via the Question tool; user approved **Option A**
  (paper-weight the tiers) and the sequencing "ladder fix first, then handoff fixes"
  (two commits). This handoff written immediately after — **no edits, regen, build, or
  commit has been made this session.** `git status`: clean except this file.
- **Interrupted before it could run:** §6 Commit-1 Step A (editing `c_sample_stability.py`)
  onward. Nothing of the plan has been applied.
- **To resume:** start at §6 Commit-1 Step A. Do not re-run the audit — the ladder finding
  is verified and recorded here; the sibling-audit findings (S1–S6) are unchanged and
  still valid except S2, which is superseded by the Option-A fix (the "fully converged"
  relabel is obsolete; re-verify the now-true convergence claims instead).

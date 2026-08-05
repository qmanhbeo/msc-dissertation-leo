# Handoff — Sibling bugs of the abstract-weighted research pivot: ZS route audit + approved fix

Date: 2026-08-05. Repo root: `/home/manh/dissertation`. Read `AGENTS.md` first; it is authoritative.
This handoff covers the **research-weighting pivot audit** (second sibling-bug hunt). The earlier editorial-pass
handoff (`handoff-editorial.md`, committed `29ae0c8`) is still valid for its own pending items — see §4.

---

## 1) Context — where we are

The research corpus pivot (Plan C, "document-weighted research") made the canonical research-side unit **one
abstract = the L2-renormalised mean of its segments** (paper-weighted), symmetric with the document-weighted
policy side. The pivot commits: `5261d31` (code: `score_supervised.py` LR+MLP centroids paper-collapsed via
`paper_units_from_shard`, `0_coverage_gap.py` paper profile, `2_sample_segments.py` 100k-paper S1 subset) →
`cf80809` (replay all three encoders' main outputs) → `8718d02` (appendix replay) → `6579ae4` (prose unit fixes).

**Sibling bug #1 (FIXED, committed `aff6c28`):** the sample-stability ladder (`c_sample_stability.py`) tiers were
segment-weighted against a paper-weighted anchor. Fixed by paper-weighting `accumulate_draws`; verified tiers now
converge to 0.3395/0.1219 matching the full-corpus anchor (0.339543/0.121904); c1 re-derived. Done and pushed.

**Sibling bug #2 (FOUND this session, APPROVED, NOT yet fixed):** the **zero-shot nearest-centroid route's
research side is still segment-weighted** — the pivot paper-weighted LR and MLP (`score_supervised.py`) but never
touched `score_zeroshot.py`. ZS research counts/centroids/cohesion accumulate per segment row, so
`semantic_gap_distances_zeroshot.json` `n_papers` sums to **3,105,144 (segments)**, not 2,536,771 (papers), and
`coverage_document_weighted_zeroshot.json`'s research profile is segment-weighted. These feed manuscript-facing
tables (tab7/tab9 encoder sensitivity, h1 cross-method, j1, k1, i1 rank deltas, the `\ZeroShotSemanticRho` macro)
whose own notes claim "Coverage gap is document-weighted (Assumption A19)" — false for the ZS columns.

**Session state:** the audit is complete and verified; the user approved the fix via the Question tool
(**paper-weight the ZS route at the producer**; and confirmed the b2 `shards[:2]` truncation is **desired
behavior** — not a bug). The user then asked to stop: commit pending work and write this handoff. **No code or
output changes for the ZS fix have been made yet.**

Repo state: `HEAD = 29ae0c8` (pushed: ladder fix `aff6c28` + this handoff's editorial counterpart `29ae0c8`).
Working tree clean except the handoff files.

---

## 2) Key known facts (so a fresh agent does not need to re-derive)

### 2.1 Corpus numbers (committed macros; do not re-derive)
| Quantity | Macro | Value |
|---|---|---|
| Research abstracts (papers) | `\NResearchAbstracts` | 2,536,771 |
| Research segments | `\NResearchSegments` | 3,105,144 |
| Policy segments | `\NPolicySegments` | 40,597 |
| Policy source docs | `\NPolicyDocs` | 6,367 |
| S1 subset (MiniLM/SciBERT) | — | 100,000 papers / 122,472 segments |
| Concept-retrieval corpus (MPNet) | — | 111,541 rows (ids present) |

3,105,144 is the **segment** count; 2,536,771 is the **paper** count. The pivot's whole point: research-side
coverage/semantic gaps are computed on **papers** (one L2-renormalised unit vector per abstract).

### 2.2 ★ THE BUG (verified end-to-end this session)

**Producer — `1_code/6_calculate_centroids/score_zeroshot.py`:**
- Lines 141-183: per shard, `scores = embeddings @ centroids.T` (segment rows), then
  `res_counts[sdg] += n` (segment count), `res_sums[sdg] += embeddings[mask].sum(axis=0)` (segment embeddings),
  `res_cohesion_sums[sdg] += scores[mask, sdg].sum()` (segment assignment scores). No `paper_units_from_shard`
  collapse anywhere.
- Line 176 log message says "Research total papers" — it prints the **segment** count (mislabel).
- Line 99-112: existence-skip gate (`expected` npys + json); **no fingerprint** — a re-run REQUIRES `--overwrite`.
- Outputs: `{scored}/zeroshot/research_centroids.npy` (segment-weighted), `policy_centroids.npy` (policy side —
  segment-level, unchanged by this fix), `semantic_gap_distances_zeroshot.json` (via the shared
  `compute_sdg_semantic_gaps` in `semantic_gap_shared.py:247-355`, unit-agnostic — inherits whatever research
  centroids/counts it is given).
- Adjusted mode (lines 155-159): projects segment embeddings through G and **re-assigns on the projected space**
  (intentional, PLAN_register_topic_decomposition §6.1). Paper collapse must happen **after** projection, on the
  projected embeddings + projected-space scores.

**Verified on disk (do not re-verify):**
- `4_outputs/mpnet/data/semantic_gap_distances_zeroshot.json`: `sum(n_papers) = 3,105,144` (segments).
- `4_outputs/mpnet/data/coverage_document_weighted_zeroshot.json`: research profile SDG4 = 0.155879 vs LR
  paper-weighted 0.155967 (same estimator family, ~1e-4 apart — the mixing is real but numerically small).
- The canonical LR artifacts are paper-weighted: `coverage_document_weighted.json`
  (`n_research_papers=2,536,771`), `research_centroids.npy` (`score_supervised.py:364-378` via
  `paper_units_from_shard`), `mlp_summary.json` (`research_total=2,536,771`).

**Manuscript-facing surfaces of the mixing (all MPNet ZS — AGENTS.md gates ZS to MPNet):**
1. `3_generate_cross_sensitivity_table.py` → `4_outputs/mpnet/tables/tab7_encoder_sensitivity.tex` (semantic) +
   `tab9_encoder_sensitivity_coverage.tex`: ZS column segment-weighted vs LR/MLP paper-weighted; the script's own
   note (lines 290-291) falsely asserts "Coverage gap is document-weighted (Assumption A19) for all methods";
   docstring lines 501-503 says ZS counts are "already doc-level" — false.
2. `h1_cross_method_gap_values.py` (lines 88-115) → `tab_app_cross_method_covgap.tex` / `tab_app_cross_method_semgap.tex`:
   ZS coverage-gap column recomputed from segment `n_papers` (`res_total = 3,105,144`) beside paper-weighted LR/MLP columns.
3. `j1_raw_value_correlation.py` (tab_j1_raw_value_correlation.tex): "MPNet ZS" row correlates paper-weighted LR
   coverage (shared axis, by design) with segment-weighted ZS gaps — mixed inside one row.
4. `k1_regression_semantic_gap.py` (L671-674): WLS weights use `n_papers` — segment counts for ZS rows vs paper
   counts for LR/MLP rows (Panel-D `wls` spec only). **Auto-fixed** once the producer writes true paper counts.
5. `i1_assignment_method_comparison.py`: `semantic_gap_rank_deltas` section reads the ZS gaps json → rank deltas
   will change. Current baselines (committed, paper-independent parts unchanged): research LR-vs-ZS agreement
   0.6734 (n=3,105,144 **segments** — stays segment-level by prior handoff decision), policy seg 0.8151 / doc
   0.8136, rank deltas SDG17 Δ=10 (LR rank 17, ZS rank 7), SDG8 Δ=9 (LR 6, ZS 15).
6. Prose in `3_writing/dissertation.tex`: `\ZeroShotSemanticRho` (currently **0.63**, defined in
   `4_outputs/mpnet/tables/num7_encoder_sensitivity.tex:4`), "SDG 17 … rank 17 under LR but rank 7 under
   zero-shot" (lines 427, 478, 828), i1 prose (line 868, Δ=10/Δ=9), and the table note (line 818: "Coverage gap is
   document-weighted (Assumption A19)") which becomes TRUE after the fix.

**Not a bug / not manuscript-facing (do not act on):**
- `tab_h1_register_correlation.tex` — NOT wired anywhere: no committed artifact, no invocation in main.py /
  orchestrator, not referenced by dissertation.tex. `h1_register_correlation_table.py` matters only as a loader
  module for j1. (One subagent over-claimed it as a surface — dismissed.)
- b2 `shards[:2]` hardcoded truncation — **USER-CONFIRMED desired behavior** (interpretability sampling).
- a3 SDG4 lexical audit — segment-level by design; label caveat only (N=470,002 is segments, not abstracts).
- a1 register validation — segment-level by design (register G is segment-level, `INLP_RESEARCH_UNIT="segment"`).
- j1 — otherwise fine (fingerprints `h1_grid_input_paths` which includes the ZS json).

### 2.3 Fingerprint/re-run mechanics (verified)
- `score_zeroshot.py`: NO fingerprint — existence-skip only. Must pass `--overwrite`.
- `0_coverage_gap.py`: fp at line 233 = `PAPER_SCORES_MANIFEST, POLICY_SCORES, POLICY_IDS, policy.npy,
  research_centroids.npy` — **does NOT include `semantic_gap_distances_zeroshot.json` or `mlp_summary.json`**, so
  it will NOT auto-re-run after the ZS json changes; must `--overwrite`. Its zs branch (lines 144-160) reads the
  ZS json's `n_papers` → will produce a paper-weighted `coverage_document_weighted_zeroshot.json` automatically
  once the producer is fixed.
- Auto-re-runners (their fingerprints include the ZS json / grid paths): `3_generate_cross_sensitivity_table.py`
  (fp lines 1225-1249), `h1_cross_method_gap_values.py` (fp lines 268-283), j1 and k1 (fp = `h1_grid_input_paths`,
  `2_coverage_semantic_interaction.py:193-218`), i1 (fp includes `zs_gap_path`).
- `h1_cross_method_gap_values.py` fp is missing `coverage_document_weighted_mlp.json` (read at its line 47) —
  fix this in the same pass.
- `POST_ADJUSTED_STEPS` (`g_register_decomposition`, `g_interaction_extended`, `generate_tex_macros`,
  `0_pca_register_before_after`) are LR-only — unaffected by the ZS change.
- Cross-sensitivity table + figures are emitted by `_run_analysis_poststeps` (main.py:783-798), MPNet only:
  `3_generate_cross_sensitivity_table.py` then `plot_figures.py`. Run both with `--overwrite`.

### 2.4 ZS invocation routes (main.py)
| Route | Command shape |
|---|---|
| Raw, per model (MPNet full; MiniLM/SciBERT S1 subset) | `score_zeroshot.py --embed-model {model} --output-dir 4_outputs` (main.py:621-625) |
| Adjusted, MPNet | same + `--embeddings adjusted` (main.py:746-750) |
| Concept raw, MPNet | `--embedding-manifest 2_data/3_embedded/mpnet/research_concept/metadata/manifest.json --out-dir 2_data/5_supervised_scored/mpnet/zeroshot_concept --data-dir 4_outputs/mpnet/data/concept` (main.py:649-655) |
| Concept adjusted, MPNet | same + `--embeddings adjusted` (main.py:756-764) |

All four corpora verified to carry per-row `openalex_id` ids (`ids_path` in manifest):
`research_shards` (3,105,144 rows), S1 subset (122,472 rows), concept (111,541 rows) → `paper_units_from_shard`
works everywhere. All are `2_data/3_embedded/{model}/…` and gitignored.

### 2.5 Reusable paper-collapse helpers (`1_code/7_main_analysis/0_shared/research_score_shards.py`)
`paper_units_from_shard(emb, scores, paper_ids, shard_name, *, prev_last_paper_id=None, renormalise=True)` →
`(paper_emb (P,d) L2-unit, paper_assigned (P,) argmax of mean score vector, seg_counts, last_paper_id)`; plus
`read_shard_paper_ids` (line 45), `paper_run_starts` (63), `group_rows_by_paper` (95). The exact recipe
`score_supervised.py:364-378` uses for the canonical centroids. Cross-shard boundary papers fail closed via
`prev_last_paper_id` threading — thread it across the shard loop.

---

## 3) Actions / decisions made + files changed this session, and why

**This session (audit only — NO fix code written):**
1. **Audited every research-side consumer/producer** for segment-vs-paper mixing, using 3 parallel explore
   subagents (j1/k1/h1; a3/a1/b2; cross-sensitivity/macros/register-correlation) plus direct reads of
   `score_zeroshot.py`, `0_coverage_gap.py`, `main.py` invocations, manifests, and committed JSON values.
2. **Confirmed the ZS-route bug empirically** (n_papers sum = 3,105,144; ZS coverage SDG4 0.155879 vs LR 0.155967).
3. **Dismissed false positives**: `tab_h1_register_correlation.tex` is not wired anywhere; b2 `shards[:2]`
   confirmed by the user as desired; a3/a1 segment-level by design.
4. **User decisions (Question tool):**
   - **Paper-weight the ZS route at the producer** (`score_zeroshot.py`) — NOT relabel-only.
   - **b2 truncation is desired behavior** — do not touch.
5. **Committed + pushed** the pending editorial handoff `handoff-editorial.md` as `29ae0c8` (it was untracked
   from the prior pass; user asked to commit whatever was uncommitted).

**Files changed this session:** none except the two handoff commits (`29ae0c8`, and this file after it).

Earlier this pass (already done + pushed): ladder fix `aff6c28` — `c_sample_stability.py` `accumulate_draws`
paper-weighted via `paper_units_from_shard`, draw-cache schema v3 (explicit clear check), `SCRIPT_VERSION` 1→2;
regenerated ladder + c1 outputs; verified tiers converge to 0.3395/0.1219 (2m tier 0.33951 vs anchor 0.339543).

---

## 4) What remains and why

### 4.1 Primary (this handoff): fix the ZS route — approved, not started
Everything in §6 (Steps 1-7). Why it remains: the user asked for a stop + handoff before execution so a fresh
agent can take over with full context.

### 4.2 Still pending from the editorial pass (handoff-editorial.md §6 Commit 2 — do NOT lose)
- i1 generator `i1_assignment_method_comparison.py`: header line ~352 `Research (papers)` → `Research (segments)`;
  docstring line ~10 `per-paper` → `per-segment`; then re-run `--appendix-i1-assignment-method --overwrite`.
- Prose: `dissertation.tex:865` table note (research "each paper" → "each segment"; drop the stale "MLP research
  per-paper scores are not persisted" clause — research MLP-vs-ZS IS computed, overall 59.5%); line 868 stale
  numbers (62.3→67.3, 79.7→81.5, 80.4→81.4, 16.8→26.1, 15.9→23.6, 76.8→76.3, 78.1→79.9 — reconfirm the three
  non-persisted policy figures by re-running i1).
- `macro-F1` → `macro-$F_1$` replaceAll (lines 310/323/342/655/659/666/677/681).
- **Sequencing note:** the ZS fix (§6) changes i1's rank-delta numbers, so run the ZS fix BEFORE finalizing the
  i1 prose/table values (rank deltas may change again; agreement rates 67.3/59.5/81.5 and per-SDG rates are
  segment-level i1 computations and do NOT change from the ZS fix — only `gap_rank_*` columns do).

### 4.3 Optional (already fixed by the ZS producer fix)
- k1 WLS weights unit-mixing (segment n_papers for ZS rows) — becomes paper counts automatically.

---

## 5) Concerns to emphasize

1. **`score_zeroshot.py` has NO fingerprint** — existence-skip only (lines 99-112). Any re-run must pass
   `--overwrite` or the script silently returns and you commit nothing.
2. **`0_coverage_gap.py` will NOT auto-re-run** — its fp (line 233) omits the ZS json and `mlp_summary.json`.
   Must run with `--overwrite`, and fix the fp in the same commit so future ZS changes propagate.
3. **Adjusted ZS must collapse AFTER projection** (project segment embeddings via G, compute projected-space
   segment scores, THEN `paper_units_from_shard` on the projected arrays) — preserves the intentional
   "re-assign on projected space" semantics (PLAN_register_topic_decomposition §6.1). Do not collapse raw
   embeddings for the adjusted route.
4. **Cohesion must be paper-level too**: currently `res_cohesion_sums[sdg] += scores[mask, sdg].sum()` (mean of
   the assignment-score column over segments). At paper level use `paper_scores[mask, sdg].sum()` where
   `paper_scores, _ = group_rows_by_paper(scores, starts)` and `mask = paper_assigned == sdg`.
5. **Thread `prev_last_paper_id`** across the shard loop and fail closed on a paper spanning a shard boundary
   (same as `score_supervised.py` / `aggregate_research_scores`). Also fail closed if a shard lacks `ids_path`.
6. **Heavy step**: MPNet raw ZS full-corpus re-run (3.1M rows × 768 dims matmul + ids read + collapse).
   Launch with `tmux` (AGENTS.md HARD RULE — never `setsid`/`disown`):
   `tmux new-session -d -s zsfix "…cmd… > /tmp/zsfix.log 2>&1; touch /tmp/zsfix.log.DONE"`, poll
   `tmux capture-pane`/`tail`/`ls /tmp/zsfix.log.DONE`. Estimate ~10-30 min; all other routes are minutes.
7. **Prose claims must be re-verified, not assumed**: after the fix, check `\ZeroShotSemanticRho` (was 0.63) and
   whether SDG 17 is still rank 7 under ZS ("rank 17 under LR but rank 7 under zero-shot" at lines 427/478/828)
   and the i1 Δ=10/Δ=9 claims (line 868). If a rank shifts, update the wording; if unchanged, keep. The ZS gap
   VALUES change even when ranks don't — the tables/csv/json regenerate regardless.
8. **Do NOT touch the LR/MLP routes** (`score_supervised.py`) — already paper-weighted; no re-run needed there.
9. **MiniLM/SciBERT ZS artifacts will change** (subset paper-collapse: ~122,472 segments → ~100,000 papers) but
   are NOT manuscript-facing (AGENTS.md gates ZS to MPNet). Still regenerate them for repo consistency — they
   are fast (subset).
10. **i1 research agreement stays segment-level** (prior handoff decision, AGENTS.md-consistent): only i1's
    `gap_rank_*` columns change with the ZS fix, not the agreement rates or n_rows.
11. **Verify, don't trust**: after the re-run, confirm `sum(n_papers)` = 2,536,771 (mpnet) / ≈100,000 (subset) /
    concept paper count, and that `coverage_document_weighted_zeroshot.json` research profile now matches the LR
    paper-weighted profile (SDG4 ≈ 0.156). Check `git diff` scope — should be `score_zeroshot.py` + 3 fp/docstring
    files + `4_outputs` (zeroshot jsons, coverage zs json, tab7/tab9, h1/j1/k1/i1 tables + macros).
12. **No magic numbers / record seed**: the fix uses the existing `RANDOM_SEED` and `paper_units_from_shard`
    defaults — do not introduce new thresholds.
13. **`handoff.md` at root is intentionally NOT used** (dbafc56 removed a stale one; user requested this file
    name `handoff-research-weight.md` as an override).

---

## 6) The whole comprehensive plan (approved; execute in order)

### Step 1 — Code fix: `1_code/6_calculate_centroids/score_zeroshot.py`
- Imports: add `paper_units_from_shard`, `read_shard_paper_ids`, `paper_run_starts`, `group_rows_by_paper` from
  `research_score_shards` (add `1_code/7_main_analysis/0_shared` to sys.path if not already imported — check the
  existing sys.path setup at the top of the file; it currently imports from model_utils etc.).
- In the research scoring loop (lines 146-173), per shard:
  1. Resolve `ids_path` from the manifest shard entry (via `resolve_manifest_path`); **fail closed** if absent.
  2. `paper_ids = read_shard_paper_ids(ids_path)`; verify `len(paper_ids) == embeddings.shape[0]`.
  3. `scores = embeddings @ centroids.T` (projected embeddings in adjusted mode — as today).
  4. `paper_emb, paper_assigned, _, last = paper_units_from_shard(embeddings, scores, paper_ids, name, prev_last_paper_id=prev)`; `prev = last`.
  5. `starts = paper_run_starts(paper_ids)`; `paper_scores, _ = group_rows_by_paper(scores, starts)`.
  6. Accumulate: `res_counts += np.bincount(paper_assigned, minlength=N_SDG)`; per sdg with
     `mask = paper_assigned == sdg`: `res_sums[sdg] += paper_emb[mask].sum(axis=0)`;
     `res_cohesion_sums[sdg] += float(paper_scores[mask, sdg].sum())`; `res_cohesion_counts[sdg] += int(mask.sum())`.
  7. `del` arrays per shard (memory).
- Fix log line 176: "Research total papers" → print `res_counts.sum()` with a truthful label ("papers (paper-weighted)").
- Update the module docstring: research side is paper-weighted (one L2-unit vector per abstract), matching
  `score_supervised.py`; policy side unchanged (segment-level with per-doc cap).

### Step 2 — Re-run the 6 ZS routes (all with `--overwrite`; mpnet raw in tmux):
```
python 1_code/6_calculate_centroids/score_zeroshot.py --embed-model mpnet --output-dir 4_outputs --overwrite          # heavy, tmux
python 1_code/6_calculate_centroids/score_zeroshot.py --embed-model minilm --output-dir 4_outputs --overwrite
python 1_code/6_calculate_centroids/score_zeroshot.py --embed-model scibert --output-dir 4_outputs --overwrite
python 1_code/6_calculate_centroids/score_zeroshot.py --embed-model mpnet --output-dir 4_outputs --overwrite --embeddings adjusted
python 1_code/6_calculate_centroids/score_zeroshot.py --embed-model mpnet --output-dir 4_outputs --overwrite \
  --embedding-manifest 2_data/3_embedded/mpnet/research_concept/metadata/manifest.json \
  --out-dir 2_data/5_supervised_scored/mpnet/zeroshot_concept --data-dir 4_outputs/mpnet/data/concept
# same as previous + --embeddings adjusted
```

### Step 3 — Fingerprint/docstring fixes (code):
- `0_coverage_gap.py:233`: add `semantic_gap_distances_zeroshot.json`, `mlp_summary.json`,
  `mlp_policy_scores.npy` to the `fingerprint_of(...)` list (fail-closed propagation).
- `h1_cross_method_gap_values.py`: add `coverage_document_weighted_mlp.json` to its fp list (line ~268-283).
- `3_generate_cross_sensitivity_table.py:501-503`: fix the "already doc-level" docstring to say the ZS counts are
  paper-weighted from the producer.

### Step 4 — Re-run `0_coverage_gap.py` (`--overwrite`):
```
python 1_code/7_main_analysis/1_main_text/0_coverage_gap.py --output-dir 4_outputs --embed-model mpnet --overwrite
python 1_code/7_main_analysis/1_main_text/0_coverage_gap.py --output-dir 4_outputs --embed-model minilm --overwrite
python 1_code/7_main_analysis/1_main_text/0_coverage_gap.py --output-dir 4_outputs --embed-model scibert --overwrite
# + the MPNet concept variant: mirror main.py:663-742 (--paper-scores-manifest …/paper_scores_shards_concept/metadata/manifest.json,
#   --out-data-dir 4_outputs/mpnet/data/concept --out-tables-dir …/concept/tables)
```
This regenerates `coverage_document_weighted_zeroshot.json` (paper-weighted now).

### Step 5 — Re-run downstream consumers (`--overwrite` each):
```
python 1_code/7_main_analysis/1_main_text/3_generate_cross_sensitivity_table.py --output-dir 4_outputs --embed-model mpnet --overwrite
python 1_code/8_visualization/plot_figures.py --output-dir 4_outputs --embed-model mpnet --overwrite
python main.py --appendix-h1-cross-method --overwrite
python main.py --appendix-j1-raw-value --overwrite
python main.py --appendix-k1-regression --overwrite
python main.py --appendix-i1-assignment-method --overwrite
```
(plot_figures: run it; if `git diff` shows no figure changes, leave them — verify before committing.)

### Step 6 — Verify (verify, don't trust):
- `4_outputs/mpnet/data/semantic_gap_distances_zeroshot.json`: `sum(n_papers)` = **2,536,771**; minilm/scibert ≈
  **100,000**; concept = concept paper count.
- `4_outputs/mpnet/data/coverage_document_weighted_zeroshot.json`: research profile now ≈ LR paper-weighted
  (SDG4 ≈ 0.156).
- `4_outputs/mpnet/tables/num7_encoder_sensitivity.tex`: new `\ZeroShotSemanticRho` value (was 0.63).
- Rank claims: SDG 17 ZS rank (was 7), i1 deltas (were Δ=10 / Δ=9) — compare against baselines in §2.2.
- `git diff` scope: `score_zeroshot.py`, `0_coverage_gap.py`, `h1_cross_method_gap_values.py`,
  `3_generate_cross_sensitivity_table.py`, and `4_outputs` (zeroshot jsons, coverage zs json, tab7/tab9, h1/j1/k1/i1
  tables + data + macros). Nothing else.

### Step 7 — Prose (only where a claim breaks), PDF, commit, push:
- If SDG 17's ZS rank or `\ZeroShotSemanticRho` changed enough to invalidate wording at dissertation.tex:427/478/828,
  adjust; same for i1 Δ claims at :868. Line 818's "Coverage gap is document-weighted" note becomes TRUE — keep.
- `python main.py --build-pdf --overwrite`; `pdftotext` spot-check.
- Commit (one concern): `score_zeroshot.py` + fp/docstring files + regenerated `4_outputs` (and prose/PDF only if
  changed). Suggested message:
  ```
  fix(zeroshot): paper-weight the ZS research side to match LR/MLP (Plan C)

  score_zeroshot.py accumulated research counts/centroids/cohesion per segment
  row, so semantic_gap_distances_zeroshot.json n_papers summed to 3,105,144
  (segments) while LR/MLP rows in the same tables are paper-weighted
  (2,536,771). Collapse each shard to paper units via paper_units_from_shard
  (threading prev_last_paper_id; fail closed on missing ids), mirroring
  score_supervised.py. Adjusted route collapses after G projection. Also fix
  fingerprint gaps in 0_coverage_gap.py (+ zs json, mlp_summary) and
  h1_cross_method_gap_values.py (+ coverage_document_weighted_mlp.json) and
  the stale 'already doc-level' docstring. ZS coverage/gaps now paper-weighted
  everywhere; k1 WLS weights unit-consistent.
  ```
- `git push`. Do NOT commit `2_data/` (gitignored) or the handoff files unless asked.

---

## 7) Exactly what was interrupted

- **Done before the stop:** the full sibling-bug audit (read-only): 3 parallel subagent audits + direct
  verification; ZS-route bug confirmed; false positives dismissed (register-correlation table unwired, b2 desired
  behavior per user); user approved **paper-weight the ZS route**; baselines recorded (`\ZeroShotSemanticRho`=0.63,
  i1 Δ=10/Δ=9, ZS n_papers=3,105,144). Pending untracked file `handoff-editorial.md` committed + pushed (`29ae0c8`).
- **Interrupted before it could run:** §6 Step 1 (editing `score_zeroshot.py`) onward. **No code or output change
  for the ZS fix exists.** The fix is approved and fully specified; resume at §6 Step 1.
- **What was NOT interrupted:** the ladder fix (`aff6c28`) is complete, verified, committed, pushed.

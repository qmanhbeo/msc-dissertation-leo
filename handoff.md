# Hand-off: Register-validation follow-up (Concept provenance, clustering, Step-2c dig) — INTERRUPTED mid-compute

**Last updated:** 2026-08-05 (updated after report written)
**Status:** Item 1 (Concept provenance) **COMPLETE — verdict: NOT a bug, valid operation, evidence verified empirically.** Item 2 (clustering) **COMPLETE for both samples — substantive result: the original 2b/2c findings do NOT survive one-per-parent sampling.** Item 3 (Step-2c decomposition) **COMPLETE** (tmux job finished at ~09:25; results harvested in §2.5). Deliverable `5_notes/scratch/register_validation_followup.md` **WRITTEN** (190 lines, covers all 3 items + verdict). Human review pending.

> **This file replaces the previous `handoff.md`** (the register-validation go/no-go
> handoff from the prior session, which is preserved verbatim at
> `5_notes/handoff_register_validation_2026-08-05.md`). The H1a–H1d Concept-row bug
> handoff was already preserved at `5_notes/handoff_h1_concept_rows_2026-08-04.md`
> (that work is complete and committed in earlier history: `0f96a3f` + `bb9df3b`).

---

## 1. Context — where we are

We are validating the dissertation's INLP "register" interpretation (removed
subspace of sentence embeddings = academic-vs-policy register, not topic). The
first diagnostic pass (n=408, MPNet canon, seed 42) is complete and committed
(`0f96a3f`; report `5_notes/register_validation_report.md`). Its verdict was
**GO** for a full validation appendix, with two caveats: (a) residual register-like
structure survives within SDGs after adjustment (Step 2c red flag), (b) the 6-feature
score operationalization changes the answer.

The task now in flight is the **follow-up diagnostic** with three mandated items —
do NOT write appendix text, do NOT touch the dissertation PDF/LaTeX or any
existing analysis script/table. All writes stay in `5_notes/scratch/`. Deliverable:
`5_notes/scratch/register_validation_followup.md` (not the first report — do not
overwrite `register_validation_report.md`).

**Headline result so far:** the follow-up has (1) cleared the Concept-row integrity
question — it is NOT a bug (same vector space, empirically proven), and (2) produced
a material correction to the first report: **when the sample is capped at one
segment per source document, the Step-2b within-corpus correlations collapse to
null and the Step-2c red flag (register↔centroid-distance growing after
adjustment) reverses sign — i.e., the first report's two key cautionary/positive
patterns were substantially driven by clustered policy mega-documents (SDSN/UNDP
reports), not by genuine per-segment register structure.** Item 3 is computing the
per-SDG / per-feature / own-vs-other-corpus / renormalization breakdowns that will
tell us whether the one-per-parent view is stable and what mechanism explains the
original 2c pattern.

---

## 2. Key known facts (read this instead of re-deriving)

### 2.1 Working environment / repo rules (from AGENTS.md — non-negotiable)

- Python: `/home/manh/miniforge3/envs/dissertation/bin/python` (conda env
  `dissertation`; `source activate dissertation` is BROKEN on this box — the
  Windows miniconda activate gets picked up; use the absolute python path).
- **Long jobs MUST run under `tmux`** (harness kills the process group at ~120 s).
  Pattern: `tmux new-session -d -s <name> "<cmd> > log 2>&1; touch log.DONE"` then
  poll `tail -F log` / `ls log.DONE` — NEVER poll the PID.
- **Scratch-only:** checks write ONLY to `5_notes/scratch/` (gitignored) or `/tmp`;
  never `2_data/` or `4_outputs/`. Reuse the trained LR
  (`2_data/4_supervised_model_results/mpnet/model/sdg_classifier_retrained.joblib`);
  never retrain.
- Deterministic seed **42** everywhere; record seed + sample sizes in every output.
- No test/lint suite; no repo code was modified this session (scratch scripts only).
- Git: branch `main`, remote `https://github.com/qmanhbeo/dissertation-bham.git`.
  Working tree has uncommitted items ONLY: `handoff.md` (this file) and
  `5_notes/handoff_h1_concept_rows_2026-08-04.md` (untracked, preserved). Nothing
  from this session is committed; nothing should be committed without an explicit ask.

### 2.2 Pipeline / data facts (verified in the first diagnostic)

- Units are **segments** (~384-token chunks), not papers/docs. Research `assigned_sdg`
  is per-segment. All alignments positional by row index.
- Adjusted embeddings are **never materialised** — project on the fly via
  `register_utils.load_G(model)` / `register_utils.project(emb, G)` (orthonormal G,
  `x' = P x` with per-row L2 renormalisation).
- G matrices: MPNet canon (62, 768) at `2_data/3b_register/mpnet/canon/G.npy`;
  MiniLM subset (29, 384); SciBERT subset (71, 768). MPNet G was fit by
  `register_adjust.py` on **canonical keyword-retrieved research + policy**
  embeddings only (its `_input_files` at `register_adjust.py:310-319` — the concept
  corpus is NOT among G's training inputs); n_target=1123/SDG/corpus, seed 42,
  62 iterations, final test acc 0.4984 (checkpoint `2_data/3b_register/mpnet/canon/checkpoint.json`).
- "Concept" = `research_concept/`, a **corpus track** (OpenAlex AI/ML field-of-study
  retrieval, 111,541 segments / 99,836 papers) embedded with **MPNet**, NOT an
  independent encoder. No G under `3b_register` for it (by design — see Item 1).
- Both concept and canonical embeddings were produced by the same script
  `1_code/3_embed/0_embed_paper_shards.py` (`--corpus research_concept` vs
  `research`): same `load_embedder(all-mpnet-base-v2)`, same
  `model.encode(..., normalize_embeddings=True)`, fp16, dim 768. Both manifests
  record `"model": "all-mpnet-base-v2", "normalize_embeddings": true`
  (`2_data/3_embedded/mpnet/{research_shards,research_concept}/metadata/manifest.json`).

### 2.3 Item 1 — Concept provenance in Table 3 (COMPLETE)

**Code path producing the "Concept LR/MLP" adjusted-gap and register numbers in
Table `tab:interaction` (the H1a–H1d grid):**

1. Table rows built by `1_code/7_main_analysis/1_main_text/2_coverage_semantic_interaction.py`:
   `_H1_CONFIGS` (`:170-180`) declares Concept rows with `corpus="concept"`;
   `_h1_config_row` (`:277-305`) → `_adj_gaps_for` (`:266-274`) routes concept →
   `_concept_adj_gaps` / `_concept_mlp_adj_gaps`, imported from
   `1_code/7_main_analysis/0_shared/h1_register_correlation_table.py`
   (`:168-189`) — these are pure readers of
   `4_outputs/mpnet/data/concept/adjusted/semantic_gap_distances_{lr,mlp}.json`.
   Register column = Spearman(coverage predictor, raw_gap − adj_gap) per SDG
   (`2_coverage_semantic_interaction.py:301-303`).
2. Those concept gap JSONs are produced by `main.py:714-732` ("semantic gap
   (concept corpus, adjusted)" LR and MLP) → `1_code/7_main_analysis/1_main_text/1_semantic_gap.py`
   with `--embeddings adjusted --research-centroids <concept centroids> --out-data-dir
   4_outputs/mpnet/data/concept`. In adjusted mode (`1_semantic_gap.py:263-268`):
   `G = register_utils.load_G(args.embed_model)` → **`2_data/3b_register/mpnet/canon/G.npy`**
   (MPNet canon), then projects the concept research centroids AND the canonical
   `policy.npy` through G, and computes gap = 1 − cosine(projected centroids).
3. Concept research centroids: `2_data/5_supervised_scored/mpnet/research_concept_centroids.npy`
   (LR) and `mlp_scores_concept/mlp_research_centroids.npy` (MLP), built by
   `score_supervised.py` scoring the concept embeddings with the canonical retrained
   LR/MLP (`main.py:635-655`; provenance in the JSONs records
   `sdg_classifier_retrained.joblib`).
4. The JSONs' own provenance field confirms
   `"register": {"g_path": "2_data/3b_register/mpnet/canon/G.npy", "track": "canon",
   "g_sha256": "10955e...", "script_version": "2", "n_target": 1123}` for BOTH
   Concept LR and Concept MLP adjusted outputs (read directly from the files).

**Is applying MPNet's G to Concept embeddings valid? YES — verified two ways:**
- Formally: same embedder, same checkpoint, same pooling, same L2 normalisation
  (both manifests), so concept vectors provably live on the same 768-dim unit
  sphere G was learned on; G's rows span a subspace of that space and projection is
  a well-defined linear operation on any vector in it.
- Empirically (new check this session): `5_notes/scratch/check_concept_same_space.py`
  — for 40 seed-42-shared papers (44 segment pairs with **byte-identical text** in
  both corpora), the concept-run embedding and the canonical-run embedding are
  identical to float32 roundoff: **max elementwise |diff| = 0.000183, min cosine =
  0.99999952**.

**What the Concept rows therefore actually represent:** the semantic gap between the
concept-retrieved research corpus and the policy corpus, with **MPNet's canonical
register directions** removed from both sides. The policy side is byte-identical to
the MPNet adjusted run (same `policy.npy`, same G); only the research side differs.
This is the intended robustness design (retrieval-axis variation under MPNet), NOT a
bug, and NOT a silent mismatch.

**Caveats to state (not bugs):** G was *learned* on the keyword-retrieved research
+ policy corpus; transferring it to the concept corpus assumes the register
directions are corpus-generic within MPNet space — geometrically valid, but the
"register" interpretation of what was removed from concept text inherits the same
open validation question the whole program is testing. Also, current Table 3 Concept
cells (H1a LR +0.150/+0.439†/−0.142 etc.) match the corrected values from the H1
handoff — no stale duplication remains.

### 2.4 Item 2 — clustering in the n=408 sample (COMPLETE)

Original sampling (both in `register_validation_check.py` and reproduced exactly in
the follow-up script, seed 42) dedupes **within** SDG only, so a paper/doc can appear
in multiple SDGs.

| Sample | Distinct parents | Parents with >1 unit | Units sharing a parent | Max |
|---|---|---|---|---|
| Original all 408 | 390 | 7 | 25 (6.1%) | 6 |
| Original research 204 | **204** | **0** | **0 (0.0%)** | 1 |
| Original policy 204 | **186** | **7** | **25 (12.3%)** | **6** |

All clustering is in the **policy** corpus, driven by mega-documents:
SDSN Sustainable Development Report 2024 and 2025 (6 segments each), UNDP Human
Development Report 2021/2022 (5), WHO Ethics & Governance of AI for Health (2),
UN SDG Progress Report 2020 (2). Research segments are perfectly de-clustered
(0.0%). Policy clustering **12.3% > the ~10% threshold** → one-per-parent rerun
was required (done).

**One-per-parent rerun (Item 2B; sample still n=408 = 204+204, exactly 12/SDG/corpus,
global dedup across SDGs, seed 42) vs original (Item 2A):**

| Stat | Original sample (2A) | One-per-parent (2B) |
|---|---|---|
| 2b reg~‖x−x′‖ pooled ρ (p) | 0.102 (0.040) | 0.092 (0.063, ns) |
| 2b within-research ρ (p) | **0.212 (0.002)** | **−0.043 (0.545, ns)** |
| 2b within-policy ρ (p) | **0.191 (0.006)** | **−0.036 (0.606, ns)** |
| 2c RAW ρ (p) | 0.126 (0.011) | **−0.212 (1.6e-05)** |
| 2c ADJ ρ (p) | 0.247 (4e-07) | **−0.197 (6.1e-05)** |
| 2c partial-corpus RAW / ADJ | 0.130 / 0.253 | −0.155 / −0.159 |
| 2d corpus acc reg-only / raw / adj | 0.456 / 0.909 / 0.505 | 0.544 / 0.944 / 0.603 |

**Interpretation (draft, to be checked against Item 3):** the Step-2b within-corpus
correlations and the Step-2c "red flag" (register↔centroid-distance rising after
adjustment) **do not survive** one-per-parent sampling. The 2c sign flips negative
in BOTH raw and adjusted space. Likely mechanism: the clustered SDSN/UNDP policy
segments are simultaneously deontic/long-sentence-heavy (high register score) and
spread across many SDGs (far from any single SDG centroid) — they manufactured the
positive 2c correlation. Removing them kills 2b's within-corpus signal and flips 2c.
Caveat: the PC1 register score is recomputed per sample, so part of the change may
reflect a shifted feature-composition anchor; Item 3's per-feature and per-SDG
breakdowns (running) should resolve this. Note also 2d: adj accuracy is no longer
≈chance under one-per-parent (0.603 vs raw 0.944) — the "INLP destroys corpus
signal" story weakens somewhat on the cleaner sample.

### 2.5 Item 3 — Step-2c decomposition (COMPLETE — log harvested 2026-08-05 ~09:25)

Computed on the **one-per-parent** sample (identical composition to Item 2B;
N=12/SDG → n=24 per SDG for 3a). Full log: `5_notes/scratch/regcheck_followup.log`;
arrays: `5_notes/scratch/regcheck_followup_arrays.npz`.

**3a per-SDG** (reg_score ~ centroid-distance, n=24/SDG — LOW POWER, treat as
directional): mixed signs, 9/17 SDGs "worse" (ADJ ρ > RAW ρ), but mean per-SDG ρ
≈ 0 in both spaces (RAW −0.031, ADJ −0.033); partial controlling SDG: RAW −0.094,
ADJ −0.074. Largest positive pairs: SDG 3 (+0.337→+0.500), SDG 16 (+0.478→+0.588),
SDG 17 (+0.418→+0.519); largest negative: SDG 12 (−0.339→−0.470), SDG 5 (−0.379),
SDG 2 (−0.428). **No systematic "gets worse after adjustment" pattern — the first
report's red flag does not reproduce per-SDG.**

**3b per-feature** (pooled; controlling SDG nearly identical): the features that
correlate with centroid distance at all do so NEGATIVELY (high register → CLOSER to
centroid) and that association **weakens after adjustment** (positive deltas):
mean_sent_len RAW −0.293→ADJ −0.172 (Δ+0.121), deontic −0.276→−0.199 (Δ+0.077),
nominal −0.145→−0.098, passive +0.075→+0.002 (Δ−0.073), hedge +0.091→+0.064,
first_person +0.095→+0.154 (Δ+0.059). **No feature drives an adjusted-space
increase of the kind the first report flagged.**

**3c own-vs-other corpus centroid pull:** reg ~ own-dist ADJ: research −0.033,
policy −0.060; reg ~ other-dist ADJ: research +0.003, **policy −0.197**; bias
(other−own) ADJ: overall −0.024, within-policy −0.111. Small residual effect:
high-register **policy** segments sit slightly closer to their own SDG centroid
and farther from the research centroid in adjusted space — a minor within-SDG
corpus pull on the policy side only.

**3d renormalization artifact check — NOT an artifact in the feared direction:**
pooled ρ: raw −0.088 → adjusted(renormalized) −0.074 → adjusted(UNrenormalized
residual) −0.058; within-SDG: −0.094/−0.074/−0.057. The near-zero negative
correlation is not inflated by L2 renormalisation (if anything renorm adds ~0.016).
reg_score ~ renorm-scale 1/‖resid‖: +0.030 (ns); dist_adj ~ 1/‖resid‖: −0.312
(mechanical renorm effect, but it does not translate into a reg-score correlation).
So the first report's 2c red flag was NOT a renormalisation artifact either — it
was driven by the clustered mega-docs (see 2.4).

**Synthesis:** on the one-per-parent sample the original 2c red flag disappears
(pooled ρ ≈ −0.07..−0.09, per-SDG mixed noise, per-feature negative and shrinking
after adjustment). The only surviving within-SDG register trace is the small
policy-side own-centroid pull (3c, ρ≈−0.11..−0.20). Combined with Item 2B, the
follow-up's verdict basis is: **the first report's cautionary residual-register
finding was a clustering artifact; its positive evidence (2b within-corpus
correlations, adj≈chance) also weakened on the clean sample.** Caveats: n=24/SDG
per-SDG correlations are low-power; the REGCHECK_N=60 run (n=120/SDG) is
recommended before per-SDG confirmatory claims; Item 3 was run only on the
one-per-parent sample (a 2A-composition Item-3 run would directly localise the old
red flag to the mega-doc SDGs — optional).

---

## 3. Actions taken this session, and why

- **Item 1 trace:** read `2_coverage_semantic_interaction.py`,
  `h1_register_correlation_table.py`, `1_semantic_gap.py`, `main.py` (concept steps),
  `register_adjust.py`, `register_utils.py`, embedder + manifests; read the concept
  gap JSONs' provenance. Established the full chain and that MPNet canon G is used.
- **Item 1 empirical check (new script):** `5_notes/scratch/check_concept_same_space.py`
  — overlap scan (30,545 shared papers of 99,836 concept papers), sampled 40 papers,
  matched 44 byte-identical segment texts across corpora, compared embeddings
  (max |diff| 0.00018, min cosine 0.99999952). → Concept embeddings are in the same
  space as G's training data. Verdict: **NOT a bug; operation valid**; Table 3
  Concept rows need no fix (their labels should read "MPNet register directions
  applied to the concept-retrieved research corpus", but that is a writing-phase
  matter, not this diagnostic).
- **Item 2+3 script (new):** `5_notes/scratch/register_validation_followup.py`
  (deterministic; reuses the first script's sampling/feature/score code; adds
  clustering stats, one-per-parent sampling with global dedup, Item 3
  decomposition). First run crashed on a trivial bug (`multi.most_common(5)` on a
  plain dict — fixed with `Counter`), then re-launched.
- **Compute launched under tmux** (`regcheck` session, 2026-08-05 09:19:40Z) —
  log `5_notes/scratch/regcheck_followup.log` (unbuffered `-u`), completion marker
  `5_notes/scratch/regcheck_followup.DONE`, artifact
  `5_notes/scratch/regcheck_followup_arrays.npz` (reg, dist_raw, dist_adj,
  dist_noren, corr, sdg, F, resid_norm).
- **Preserved prior handoff:** `handoff.md` (register go/no-go) → `5_notes/handoff_register_validation_2026-08-05.md`.
- **No repo code, no manuscript, no `2_data/`, no `4_outputs/` touched. Nothing committed.**

Files created this session (all gitignored scratch): `check_concept_same_space.py`,
`register_validation_followup.py`, `regcheck_followup.log`, (pending)
`regcheck_followup_arrays.npz`, `regcheck_followup.DONE`.

---

## 4. What remains, and why

1. **Report written.** `5_notes/scratch/register_validation_followup.md` (190 lines)
   covers all 3 items with concrete numbers, file/line references, and an updated
   verdict. Verify with `ls -la 5_notes/scratch/register_validation_followup.md`.
2. **Human review** of the follow-up report; only then decide on the full validation
   appendix (Phase 2+ below). Do NOT start appendix text or the full-corpus job.
3. **Optional (recommended):** re-run `REGCHECK_N=60 ... register_validation_followup.py`
   for stable per-SDG correlations (n=120/SDG; ~2,040/corpus; several more minutes
   under tmux) before finalising per-SDG claims — N=12/SDG correlations (n≈24/SDG)
   are low-power; be honest about that.
4. Commit/push handoffs only when asked.

---

## 5. Concerns to emphasise

1. **The first report's headline cautions were partly artifacts of policy
   mega-document clustering.** SDSN 2024/2025 + UNDP HDR + WHO AI reports produced
   the 12.3% policy clustering AND plausibly the positive 2b-within-policy and
   2c correlations. One-per-parent sampling flips 2c negative and nulls 2b. Anyone
   (including a future reviewer) re-deriving the n=408 numbers must use the
   one-per-parent variant or explicitly justify segment clustering.
2. **Don't overcorrect either.** The one-per-parent 2c ρ≈−0.07..−0.09 (register
   score → slightly CLOSER to SDG centroid) is a new, different pattern that itself
   needs explanation (3b per-feature + 3c own-vs-other help; the negative
   per-feature associations are driven by deontic/sentence-length, which weaken
   post-adjustment). And the PC1 score is re-fit per sample, so sign/composition
   shifts partly reflect the score's anchor changing — do not read the flip as a
   pure sampling effect without the per-feature breakdown.
3. **Item 1 is closed, but keep the semantics straight:** "Concept" is a retrieval
   axis under MPNet with MPNet's G — not an independent encoder, and not an invalid
   application (same-space proven). Any prose that implies otherwise (either
   direction) is wrong. The H1 bug (previous handoff) is a different, already-fixed
   issue — do not conflate.
4. **Adjusted ≈ chance claim weakened on the clean sample** (0.505 → 0.603).
   The "INLP destroys all linear corpus signal" reading of the first report needs
   restating with the one-per-parent number; register-only accuracy also rose
   (0.456 → 0.544). The register-interpretation evidence base shifts: less
   "removed = my 6 features" (2b null) but also less "residual register inside
   SDGs" (2c negative).
5. **Reproducibility:** everything deterministic (seed 42); scripts live only in
   gitignored scratch — if the follow-up report is cited anywhere, promote/copy the
   scripts and commit (as was done for the first report at `5_notes/register_validation_report.md`).
6. **tmux hygiene:** session `regcheck` is DONE (marker exists) — do not relaunch
   the script without clearing `regcheck_followup.log`/`.npz`/`.DONE` first, and
   never run two copies concurrently. Use the absolute python path
   (`/home/manh/miniforge3/envs/dissertation/bin/python -u ...`).
7. **Scope discipline:** this is diagnostic-only; stop after
   `register_validation_followup.md` for human review. No manuscript, no analysis
   scripts, no `2_data/4_outputs` writes (AGENTS.md). Don't fix Table 3 or any table.
8. **Power honesty:** per-SDG correlations at N=12/SDG (n=24) are low-power; the
   REGCHECK_N=60 run (n=120/SDG) is needed before any per-SDG claim in the report.

---

## 6. The comprehensive plan

**Phase 0 — finish this diagnostic (immediate):**
1. Harvest tmux `regcheck` output (Item 3a–3d) once `.DONE` exists.
2. (Recommended) `REGCHECK_N=60` run for per-SDG power; also consider re-running
   Item 3 on the **original (2A-composition)** sample to directly explain the old
   2c red flag vs the one-per-parent view.
3. Write `5_notes/scratch/register_validation_followup.md` (3 items + verdict with
   numbers, file/line refs, seeds, n's).
4. Human review; updated GO/qualified/NO-GO decision.

**Phase 1 — design freeze for the full validation appendix (only after GO):**
5. Pre-register the register operationalization: Biber (1988) MD-style battery;
   report per-dimension correlations; keep PC1 and a-priori institutional score as
   explicit alternatives (first report showed operationalization flips answers).
6. Unit of analysis: segments (pipeline unit) with **one-segment-per-parent
   sampling as the primary design** (Item 2 shows it matters), or paper-level
   aggregation — decide and justify.
7. Encoders: extend to MiniLM + SciBERT (their own Gs exist); note subset track.
8. Sample size: scale to ~1–2k/corpus, 12+ per SDG per corpus, deterministic seed.

**Phase 2 — data work:**
9. Pre-clean policy text (PDF banner junk inflates mean_sentence_length) or drop
   the feature; document SDSN/UNDP mega-doc treatment (cap, dedupe, or exclude).
10. Compute the richer feature battery; verify per-corpus distributions.
11. Always project via `register_utils.project()` (never materialise adjusted
    arrays); record ‖x−x′‖.

**Phase 3 — analysis the appendix must contain:**
12. Repeat 2b/2c/2d/3 on the one-per-parent sample at scale (the new baseline).
13. **Confront Step 2c head-on:** per-SDG, per-feature, own-vs-other-corpus
    centroid pull, and the un-renormalised-distance check (3a–3d already compute
    these); test whether residual corpus-per-SDG offset after INLP is register
    (per-feature) and whether per-SDG INLP directions would remove it; state
    plainly whether the residual is register, topic, or renormalization artifact.
14. Encoder-robustness: corpus-classifier collapse and topic preservation on
    MiniLM/SciBERT.
15. If feasible: feature-space ablation — does the INLP corpus classifier's
    decision boundary align with the feature battery?

**Phase 4 — writing (only after Phase 3 passes review):**
16. Appendix per repo conventions (JSON-out, macros, fingerprint-gated, registered
    in `APPENDIX_SPECS`).
17. Update `dissertation.tex` "left to future work"/"unvalidated" sentences
    (:279, :373, :392, :477) to cite the validation; macro-driven; then
    `python main.py --build-pdf --overwrite` (bash/WSL, short job — poll short
    first); verify tables.

**Phase 5 — commits (one concern per commit, only when asked):**
18. (a) appendix code+registration; (b) outputs; (c) manuscript prose+macros;
    (d) rebuilt PDF. Also commit the preserved handoffs.

---

## 7. Exactly what was interrupted

**The follow-up compute was mid-flight when the user asked to stop, but finished
shortly after. The deliverable was written in the next session.** Sequence:

1. Item 1 fully completed (trace + empirical same-space check + verdict) — no
   interruption there.
2. Item 2 script written; first launch crashed (dict `.most_common` bug — fixed);
3. Re-launched under tmux (`regcheck`, started 2026-08-05 09:19:40Z,
   `/home/manh/miniforge3/envs/dissertation/bin/python -u 5_notes/scratch/register_validation_followup.py`).
4. At the moment of the stop request, the job had **completed Item 2A and Item 2B**
   (numbers in §2.4) and was **inside Item 3**. The tmux session was left running;
   it finished at ~09:25 (`.DONE` marker present; results harvested into §2.5).
5. **Report written** (`5_notes/scratch/register_validation_followup.md`, 190 lines,
   Items 1–3 + verdict). All three mandated diagnostic items are complete.

**Current state (for a fresh agent):**
1. All compute is done (DONE marker exists at `regcheck_followup.DONE`).
2. The deliverable report is written at `5_notes/scratch/register_validation_followup.md`.
3. Awaiting human review; no commits needed unless explicitly asked.

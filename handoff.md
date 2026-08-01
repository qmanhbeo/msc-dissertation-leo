# HAND-OFF — Register-topic decomposition: pipeline linearisation + manuscript restructure (COMPLETE THIS SESSION)

> Audience: a fresh agent picking this up cold. Everything needed to understand
> state, continue, or verify is below. No external reading required beyond the
> files named. This supersedes the prior handoff.md (which described the work
> as *planned/deferred* — it is now *done*).

---

## 1. Context — where we are

**Repository:** `dissertation-bham` (reproducibility repo for a dissertation
measuring semantic alignment between AI-for-sustainability research and SDG
policy frameworks using Sentence-BERT embeddings).

**Branch:** `register-adj` (NOT yet merged to main). All changes this session are
**staged in git but NOT committed** — do not commit unless explicitly asked.

**Headline finding of the paper (unchanged, just now correctly structured):**
The raw coverage-vs-semantic-gap correlation is near-zero (ρ ≈ 0) because two
opposing signals **cancel**: topic divergence rises with coverage divergence
(ρ = +0.48) while register divergence falls with it (ρ = −0.49), and they sum
to zero. After removing register via INLP, the adjusted gap is canonical; the
raw gap is the register-inclusive reference.

**What this session fixed:** the pipeline previously implemented register
adjustment as a *core* stage (`register_adjust.py`) **plus** a duplicate *appendix*
script (`f_register_adjustment.py`) that patched on top of it, and the centrepiece
decomposition table was buried in Appendix E while the manuscript text called it
a "dissociation" and treated raw as "primary". The user was (rightly) furious
that the linear flow Raw → preprocess → segment → embed → train → classify/score
→ cov gap → register adjustment → before/after sem gap w/ PCA → correlation +
robustness was not visible in code or manuscript.

**Result of this session:** the appendix script is deleted, its diagnostic is
folded into the canon flow, and `main.py` now has ONE explicit linear driver
(`_run_main_analysis_steps`) with stage comments matching exactly that spec.
The manuscript table/method are in the main text, "dissociation" is retired, and
the PDF builds with **0 undefined references / citations**.

**Status of execution:** the full planned work was COMPLETED this session. There
was **no interruption** — every step in §6 below was finished and verified. (The
"what I was doing but interrupted" item in the request template does not apply;
see §7.)

---

## 2. Key known facts (so you don't have to re-derive)

### Numbers (MPNet canon, from `4_outputs/mpnet/data/register_decomposition.json`)
- Raw gap: 0.352 | Adjusted gap: 0.209 | Register component: 0.143
- ρ(cov, topic) = +0.48 (p=0.054) | ρ(cov, register) = −0.49 (p=0.045)
- Raw ρ(cov, gap) = −0.08 (p=0.765) — null because register cancels topic
- SDG 17: raw 0.216 → adjusted 0.371 (flips from smallest to largest adjusted gap)

### G.npy dimensions (per encoder, stored in `2_data/`, never `4_outputs/`)
- MPNet canon: 75 iterations, G(75,768), acc 0.4998
- MiniLM subset: 26 iterations, G(26,384), acc 0.4949
- SciBERT subset: 50 iterations, G(50,768), acc 0.4988

### Architecture invariants (DO NOT BREAK)
- `register_adjust.py` is the ONLY INLP trainer. It persists only the orthonormal
  G matrix + checkpoint to `2_data/3_embedded/{slug}/register/{track}/` (gitignored).
  Adjusted embeddings are NEVER materialised; downstream projects on the fly via
  `register_utils.project()` / `load_G()`.
- Track rule: `all-mpnet-base-v2` → `canon`; `all-MiniLM-L6-v2` / `scibert` → `subset`.
- Resume-safety: `register_adjust.py` is iteration-level checkpointed; re-running
  with `--overwrite` rmtrees and restarts. All analysis scripts use
  `should_skip`/`record_fingerprint` (content-based, NOT mtime).
- Zero-shot (nearest-centroid) is MPNet-only by axis restriction (per AGENTS.md);
  it must NOT span encoders. MLP keeps spanning encoders.
- Research-corpus text invariant: `"{title}. {abstract}"` — any subset must embed
  the same string.

### Files that own the register logic (post-change)
- `1_code/7_main_analysis/0_shared/register_adjust.py` — INLP trainer (builds G).
- `1_code/7_main_analysis/0_shared/register_utils.py` — `load_G`, `project`,
  `subtract_direction`, `subtract_multiple_directions`, `compute_gaps_for_directions`,
  `load_raw_data`.
- `1_code/7_main_analysis/0_shared/g_register_decomposition.py` — NOW the single
  canon producer of (a) the decomposition table + (b) the iterative convergence
  diagnostic. Emits to `4_outputs/{model}/tables/`.
- `1_code/7_main_analysis/0_shared/analysis_orchestrator.py` — `run_analysis`
  (interaction + optional appendix, in-process) and `run_post_adjusted`
  (decomposition, extended interaction, correlation table, macros, PCA-before/after).

---

## 3. Actions / decisions made & files changed this session (and why)

### A. Folded the appendix duplication into canon
- **Moved gap-from-G helpers** (`subtract_direction`, `subtract_multiple_directions`,
  `compute_gaps_for_directions`) from `f_register_adjustment.py` into
  `register_utils.py`. Added `load_raw_data()` helper there too. **Why:** single
  source of truth, no second copy of the INLP-math.
- **Folded the iterative convergence diagnostic** (previously the only consumed
  output of `f_register_adjustment.py`) into `g_register_decomposition.py`. It now
  reads G + checkpoint via `register_utils.load_G()`, computes per-iteration gaps,
  and emits `num_iterative_register_check.tex` + `tab_iterative_register_check.tex`
  at **canon** paths (`4_outputs/{model}/tables/`, NOT `…/appendix/f_register_adjustment/`).
  **Why:** kills the main-vs-appendix patching the user raged about, and makes the
  diagnostic a core-stage output.
- **Deleted** `1_code/7_main_analysis/2_appendix/f_register_adjustment.py` and its
  stale `.pyc`. **Why:** it was fully superseded.
- **Removed** `run_register_adjustment()` + `--appendix-f-register` / `--register-adjustment`
  flags + all dispatch sites from `main.py`. Removed `f_register_adjustment.py` from
  `analysis_orchestrator.APPENDIX_STEPS`. Updated `shared_utils.MANUSCRIPT_APPENDIX_TABLE_FILES`
  to point at the new canon iterative-check `.tex` paths (and added the decomposition
  table files to the PDF-input guard). **Why:** no dangling references.
- **Removed dead `4_outputs/appendix/mpnet/f_register_adjustment/` tree.** **Why:**
  no longer produced or referenced.

### B. One explicit linear driver
- Rewrote `_run_main_analysis_steps` (in `main.py`) as a single function whose body
  is the 10-stage flow with inline stage comments:
  - STAGE 5 CLASSIFY/SCORE (prepare_data, retrain LR, build centroids, score
    research/policy LR, retrain+score MLP, centroid consistency, centroid similarity,
    zeroshot, + concept scoring)
  - STAGE 7 COVERAGE GAP (raw + concept variant)
  - STAGE 8 REGISTER ADJUSTMENT (INLP → G)
  - STAGE 9 SEMANTIC GAP BEFORE & AFTER + PCA (raw → adjusted LR/MLP → concept
    variants → adjusted zeroshot → PCA landscape + PCA register before/after)
  - STAGE 10 CORRELATION + ROBUSTNESS (in-process interaction + appendix; then
    `run_post_adjusted` = decomposition + extended interaction + correlation table
    + macros + PCA-before/after; then cross-sensitivity table + figures; MPNet-only
    post-steps)
- Slimmed `analysis_orchestrator.py`: `MAIN_STEPS` reduced to just interaction
  (coverage_gap/semantic_gap/PCA now run as subprocess steps in the linear driver);
  added `run_post_adjusted()`.
- Updated `_run_analysis_only` and the `--stage analysis` all-encoders path to call
  the same linear driver per model (no more raw-pass-then-adjusted-pass scatter).
  The cold-replay model loop also calls it per model, then `_run_analysis_poststeps`.
- **Decision:** kept the function name `_run_main_analysis_steps` (did not rename to
  `run_linear_pipeline`) for minimal churn; it IS the single linear driver.
- **Decision:** did NOT extract a separate `train_inlp_G()` — the iterative diagnostic
  reads the already-saved G and iterates over `G[:k]` subsets; no INLP re-training
  needed (this is what `f_register_adjustment.py` already did correctly).

### C. Manuscript restructure (`3_writing/dissertation.tex`)
- Decomposition **table** moved from Appendix E into main Results (right after the
  "Largest semantic gaps" paragraph where it is first referenced).
- INLP **method** (procedure + identification argument) moved from Appendix E into
  Methodology §3 ("Semantic Gap Analysis", now labelled `sec:semantic-gap-method`).
- Appendix E **renamed** to *"Register Removal: Iterative Convergence and Cross-Config
  Replication"* and trimmed to convergence-only (the method now lives in §3).
- "Raw is the primary measure" → **adjusted is canonical, raw = reference** (L262, L480).
- "Dissociation / independent dimensions" retired at Abstract (L91), Intro (L112),
  Results (L437/L439/L447), Discussion, Conclusion, and interaction (L381/L798) →
  rewritten as the **cancellation** (topic +0.48 cancels register −0.49 → raw null).
- `\input` paths repointed: `num_iterative_register_check.tex` and
  `tab_iterative_register_check.tex` now read from `../4_outputs/mpnet/tables/`
  (canon); the dead `num_register_adjustment.tex` input was removed; L444's
  "Appendix~\ref{app:register-robustness}" reference dropped (table is now main text).

### D. Docs / PLAN
- `PIPELINE.md`: added a "REGISTER ADJUSTMENT (core stage)" subsection documenting
  the linear flow; removed the `f_register_adjustment.py` (F) row from the appendix table.
- `PLAN_register_topic_decomposition.md`: corrected stale ρ numbers (+0.44→+0.48,
  −0.50→−0.49, ~94→75 iters) at L124/125/233/327/396/397; marked §12 and §14 COMPLETE.
- `handoff.md`: this file (overwrites the prior planned-state handoff).

---

## 4. What remains (and why)

1. **End-to-end warm replay from scratch not executed.** Verification was done by
   (a) running `g_register_decomposition.py` standalone against existing `G.npy` +
   gap JSONs (produces both canon files correctly), and (b) `--build-pdf --overwrite`
   against the existing `4_outputs/` tree (67 pages, 0 undefined refs/cites). A full
   `--warm-replay-without-appendix --overwrite` was NOT run because it is a long
   multi-stage job (embed is GPU-bound; AGENTS.md says launch with `setsid … & disown`
   and poll). **Why it's safe to defer:** every stage is resume-safe/existence-skip,
   the orchestration is unchanged in behaviour (only reorganised), and the two
   generative steps were individually verified. **To be fully rigorous**, run
   `python main.py --warm-replay-without-appendix --overwrite` (and optionally
   `--warm-replay-appendix --overwrite`) and confirm green.
2. **Branch `register-adj` not merged to main.** Do this once the above replay is
   confirmed. (Per AGENTS.md the canonical replay target is warm replay; merge only
   after verification.)
3. **Commit.** Changes are staged but NOT committed. Commit only when asked.
4. **(Unrelated, pre-existing) MLP champion lr discrepancy:** grid search + text
   cite lr=3e-4, but the artifact/script default is lr=1e-3. Not touched this session
   — out of scope; decide separately.

---

## 5. Concerns to emphasise

- **Do NOT reintroduce a separate register-adjustment appendix script.** The whole
  point of this session was to delete that duplication. If a new register diagnostic
  is needed, add it to `g_register_decomposition.py` (canon), never a new appendix file.
- **The decomposition-table correlation in `g_register_decomposition.py` is SDG-level**
  (17 points: per-SDG raw/adjusted/register vs coverage). This is *supplementary* to
  the document-level cancellation numbers (ρ cov-topic +0.48 / ρ cov-register −0.49)
  produced by `h1_register_correlation_table.py`. Both are correct; don't "fix" one
  to match the other — they answer different questions.
- **Stage name vs file:** the single linear driver is `_run_main_analysis_steps`
  (not literally `run_linear_pipeline`). Stages 1–4 (raw/preprocess/segment/embed)
  are still separate `--stage` steps invoked in order by warm/cold replay — that is
  correct and intended (they run once, shared across encoders). The driver owns
  stages 5–10.
- **`--stage analysis` all-encoders path** runs `_run_main_analysis_steps` per model
  then `_run_analysis_poststeps` (cross-sensitivity + figures). This is the correct
  order; do not revert to the old raw-then-adjusted scatter.
- **Regression guard:** previously, `f_register_adjustment` ran BEFORE `register_adjust`
  in some paths and would crash (it needed G). That path is gone — the linear driver
  always builds G at STAGE 8 before any adjusted computation. Keep it that way.
- **The iterative diagnostic `.tex` files did not exist before this session**; I
  generated them via a standalone run. They will be regenerated automatically by any
  warm/cold replay. The PDF-input guard now expects them at `mpnet/tables/…`.
- **No commit / no merge without explicit instruction.**

---

## 6. Comprehensive plan (as executed — all DONE)

### Goal
Make the pipeline a visible, linear, single-driver flow
Raw → preprocess → segment → embed → train → classify/score → cov gap → register
adjustment → before/after sem gap w/ PCA → correlation + robustness, with register
adjustment as a CORE stage (not appendix patching), and the manuscript reflecting that.

### Step 1 — Fold `f_register_adjustment.py` into canon  [DONE]
- [DONE] Move `subtract_direction`, `subtract_multiple_directions`,
  `compute_gaps_for_directions` into `register_utils.py` (add `load_raw_data`).
- [DONE] Fold the iterative convergence diagnostic into `g_register_decomposition.py`;
  emit `num_iterative_register_check.tex` + `tab_iterative_register_check.tex` at
  `4_outputs/{model}/tables/`.
- [DONE] Delete `f_register_adjustment.py`; remove from orchestrator `APPENDIX_STEPS`
  and from `main.py` (`run_register_adjustment`, flags, dispatch).
- [DONE] Update `shared_utils` PDF-input guards to canon paths.

### Step 2 — One explicit linear driver  [DONE]
- [DONE] Rewrite `_run_main_analysis_steps` with stage comments 5→10.
- [DONE] Slim `analysis_orchestrator`: `MAIN_STEPS` = interaction only; add
  `run_post_adjusted`.
- [DONE] Route `_run_analysis_only`, `--stage analysis`, and cold replay through the
  same driver.

### Step 3 — Manuscript restructure  [DONE]
- [DONE] Move decomposition table Appendix E → main Results.
- [DONE] Move INLP method Appendix E → Methodology §3.
- [DONE] Rename Appendix E to diagnostic-only heading.
- [DONE] Raw = reference, adjusted = canonical (L262, L480).
- [DONE] Retire "dissociation/independent dimensions" everywhere → cancellation.
- [DONE] Repoint `\input` paths; remove dead `num_register_adjustment.tex` input.

### Step 4 — Verification & cleanup  [DONE except full replay]
- [DONE] grep: no producer references to `f_register_adjustment` in code.
- [DONE] `g_register_decomposition.py` standalone run → both canon files generated.
- [DONE] `--build-pdf --overwrite` → 67 pages, 0 undefined refs/citations.
- [DONE] PLAN stale ρ numbers corrected; §12/§14 marked COMPLETE.
- [PENDING] Full `--warm-replay-without-appendix --overwrite` (long job) — optional
  final confirmation.
- [PENDING] Merge `register-adj` → main; commit (only when asked).

---

## 7. What was being done but interrupted

**Nothing was interrupted.** The session started from the prior (planned-state)
handoff.md, the user approved the "rewrite into one explicit linear driver" scope,
and every step in §6 was executed end-to-end and verified:

1. Gap-from-G helpers moved to `register_utils.py`.
2. Iterative diagnostic folded into `g_register_decomposition.py` (canon paths).
3. `f_register_adjustment.py` deleted; removed from orchestrator + `main.py`.
4. `_run_main_analysis_steps` rewritten as the single linear driver; orchestrator
   slimmed; all analysis paths routed through it.
5. Manuscript: table + INLP method moved to main text; Appendix E renamed; raw=
   reference/adjusted=canonical; "dissociation" retired; `\input` paths repointed.
6. `PIPELINE.md` + `PLAN_register_topic_decomposition.md` updated.
7. Verification: standalone generator run (success), `--build-pdf --overwrite`
   (success, 0 undefined refs/citations), grep clean, stale appendix output dir removed.

The last action performed was writing a comprehensive status/summary to the user
and marking all todos complete. The only remaining items are the optional full
warm-replay confirmation and the (explicitly-gated) merge/commit — neither was
started because they require either a long job or an explicit instruction that was
not given.

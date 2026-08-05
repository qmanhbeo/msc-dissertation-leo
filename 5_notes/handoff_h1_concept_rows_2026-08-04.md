# Hand-off: H1a–H1d Concept-row gap-dispatch bug (Table `tab:interaction` + Appendix J.1)

**Last updated:** 2026-08-04
**Status:** Code FIXED; outputs REGENERATED + verified; **prose EDITED (line 396 now uses `\ConceptLRCovgapAdjRho` + positive-count macros); line 794 confirmed unchanged.** **PDF build BLOCKED by pre-existing, out-of-scope K.1/figure issues — not this fix. Nothing committed.**
**Interrupted:** No task was mid-flight. Work was stopped cleanly at the end of the verification phase, before the prose edit. See §7.

> **IMPORTANT — this file replaced a previous hand-off.** The prior `handoff.md`
> documented Appendix K.1 (pooled OLS regression) and had **uncommitted** edits.
> It was preserved verbatim at **`5_notes/handoff_k1_regression_2026-08-04.md`**.
> That K.1 work is still in the working tree and is **not mine — do not commit it
> with this fix.** See §3.4.

---

## 1. Context — where we are

The user (acting as PI) flagged a data-integrity problem in the H1d block of
Table~`tab:interaction` (the H1a–H1d coverage-predictor × semantic-gap grid):

> "the Concept LR row (+0.194, +0.613**, -0.058) and Concept MLP row (+0.148,
> +0.471+, -0.090) are identical to the MPNet LR and MPNet MLP rows directly
> above them... This looks like a copy-paste or indexing bug."

**The flag was correct.** It is a real bug — a wrong-array load in the gap
dispatch, not a copy-paste in the table writer. It was investigated, root-caused,
fixed, and all affected outputs were regenerated and verified. What remains is
the manuscript prose update and the commits.

**No headline conclusion is overturned by the fix.** Details in §2.6.

---

## 2. Key known facts (read this instead of re-deriving)

### 2.1 Root cause

`1_code/7_main_analysis/1_main_text/2_coverage_semantic_interaction.py`

`_raw_gaps_for` / `_adj_gaps_for` took `(method, root, model)` and **ignored the
`corpus` argument**. `_h1_config_row` honoured `corpus` for the *coverage
predictors* but discarded it for the *gap vectors*. Result: the two Concept rows
paired **concept-retrieval coverage** with **MPNet keyword-retrieval semantic
gaps** — a predictor built on one corpus correlated against an outcome built on
another.

Corroborating evidence found during investigation:
- `_concept_raw_gaps` / `_concept_adj_gaps` were **imported but never called**
  (dead imports proving the intent).
- `j1_raw_value_correlation.py` imports those same dispatchers, so it inherited
  the identical bug.
- The bug is **long-standing**, not newly introduced: commit `381cc34`'s
  committed table already shows the H1d duplication.

### 2.2 Why H1d specifically showed an *exact* three-decimal match

`policy_profile_hard_docweighted` is **byte-identical** between
`4_outputs/mpnet/data/coverage_document_weighted.json` and
`.../data/concept/coverage_document_weighted.json`. That is **legitimate and
expected**: concept retrieval re-assigns only the *research* corpus (OpenAlex
AI/ML field-of-study), while the policy corpus and its LR assignment are
unchanged.

So for H1d the *predictor* is identical by design; the bug then made the *gap*
identical too, collapsing the correlation to an exact match. **All four blocks
(H1a–H1d) were wrong for the two Concept rows** — H1d is merely where it was
visible to the eye. This is the single most important fact for understanding the
bug: the user's instinct ("different encoders never match to 3 d.p.") was right,
but the mechanism is subtler than a copy-paste.

### 2.3 Blast radius (exhaustively verified)

| Output | Status |
|---|---|
| `4_outputs/{mpnet,minilm,scibert}/tables/tab4_interaction_h25.tex` → Table `tab:interaction` (`dissertation.tex:392`) | **WAS WRONG — now fixed** |
| `4_outputs/appendix/mpnet/j1_raw_value_correlation/{tables,data}` → Table `tab:raw-value-correlation` (`dissertation.tex:822`) | **WAS WRONG — now fixed** |
| `0_shared/h1_register_correlation_table.py` | Not affected (see §2.5) |
| `2_appendix/k1_regression_semantic_gap.py` | Not affected — correct concept dispatch at its lines 299-304 |
| `1_main_text/3_generate_cross_sensitivity_table.py` | Not affected — own `load_concept_*` loaders |
| `2_appendix/h1_cross_method_gap_values.py` | Not affected — own `_concept_*` loaders |
| `4_outputs/*/data/interaction_h25.json` | **Byte-identical before/after** — the canonical (non-grid) statistics never touched the buggy path |

Verified by exhaustive grep: `_raw_gaps_for` / `_adj_gaps_for` have exactly two
call sites, both now fixed.

### 2.4 Secondary defects found and fixed alongside

1. **Fingerprint hole.** `2_coverage_semantic_interaction.py`'s fingerprint
   covered only the three canonical MPNet inputs — never the `data/concept/`
   inputs the grid reads. A non-`--overwrite` run (e.g. `--stage analysis`) would
   not re-derive when concept gaps changed. *(Note: this was NOT a blocker for
   the fix, because `should_skip` short-circuits on `overwrite=True` and replay
   always passes `--overwrite`. An earlier draft of the plan overstated this.)*
2. **Stale namespaced copies.** `tab4_interaction_h25.tex` is a
   model-independent table written into all three model namespaces;
   `minilm`/`scibert` were frozen at pre-`39c1eb1` values while `mpnet` had been
   refreshed. All three are now byte-identical (verified by md5).
3. **`j1_raw_value_correlation.py` was orphaned from the runner.** It was in
   **no** registry: absent from `APPENDIX_SPECS` (so no CLI flag; never run by
   `--appendix-all` or `--warm-replay-with-appendix`) and absent from
   `MANUSCRIPT_APPENDIX_TABLE_FILES` (so `require_pdf_inputs` would not flag it
   missing) — **yet `dissertation.tex:822` `\input`s its output.** This is why
   its copy of the bug could sit committed indefinitely. Now registered.
4. **Hardcoded prose literal.** `dissertation.tex:396` hardcodes
   `Concept LR $+0.451^{\dagger}$`. That literal came from commit `381cc34`'s
   table and matched *neither* the pre-fix value (+0.701**) *nor* the corrected
   one (+0.439†). It had already gone stale once. Now exportable as a macro.

### 2.5 `h1_register_correlation_table.py` — do not trust it as an oracle

An earlier draft of the plan called this module "the reference implementation
that gets it right." **That was wrong and was corrected.** Facts:

- It has **never executed**. Absent from `MAIN_STEPS`, `APPENDIX_SPECS`, and
  `POST_ADJUSTED_STEPS` in `0_shared/analysis_orchestrator.py`; nothing else in
  the repo invokes it. Its output dir
  `4_outputs/appendix/{model}/h1_register_correlation_table/` **does not exist**.
- It is the abandoned §6.5.3 of `5_notes/PLAN_register_topic_decomposition.md`
  ("the ONE table"). Commit `763b446` instead folded the grid into
  `2_coverage_semantic_interaction.py` and imported its loaders — **and the
  concept branch was lost in that port.** That is the origin of the bug.
- Its `_coverage_gaps` (line 95-107) returns only `coverage_gap_hard`, so it
  covers **H1a only**, not all four predictors. It is not a drop-in reference.

It is used here **only as a library of path-resolving gap loaders** (`_lr_*`,
`_mlp_*`, `_zs_*`, `_concept_*`), which are pure file readers and are fine.
Treat its `run()` / `_build_config_row` as dead code of unverified correctness.

### 2.6 Substantive impact — no headline conclusion overturned

- **H1a** adjusted-gap signal: still positive in **9/9** configs. Concept LR
  moves +0.701** → **+0.439†**.
- **H1d** adjusted-gap signal: still positive in **9/9** configs.
- **The "apparent cancellation replicates across every config" claim SURVIVES.**
  Post-fix H1a register column is negative in 9/9 configs. *(An earlier draft of
  the plan flagged this sentence as needing narrowing — that was wrong; it is
  verified true and needs no change.)*
- What *does* change: Concept-row magnitudes, and H1d's significance set.

### 2.7 Corrected values (verified against regenerated files)

Table `tab:interaction` (Spearman), Concept rows only — raw / adjusted / register:

| Block | Published (wrong) | Corrected (now in repo) |
|---|---|---|
| H1a Concept LR | +0.130 / +0.701** / -0.292 | **+0.150 / +0.439† / -0.142** |
| H1a Concept MLP | -0.093 / +0.444† / -0.404 | **+0.034 / +0.206 / -0.096** |
| H1b Concept LR | +0.230 / -0.154 / +0.154 | **+0.015 / -0.074 / -0.252** |
| H1b Concept MLP | +0.120 / -0.203 / +0.154 | **-0.196 / +0.025 / -0.191** |
| H1c Concept LR | +0.475† / +0.203 / +0.174 | **+0.213 / +0.061 / -0.059** |
| H1c Concept MLP | +0.294 / +0.037 / +0.152 | **-0.086 / +0.022 / -0.115** |
| H1d Concept LR | +0.194 / +0.613** / -0.058 | **+0.369 / +0.535* / +0.180** |
| H1d Concept MLP | +0.148 / +0.471† / -0.090 | **+0.256 / +0.260 / +0.082** |

Appendix J.1 (Pearson), Concept rows only:

| Block | Published (wrong) | Corrected (now in repo) |
|---|---|---|
| H1a Concept LR | +0.202 / +0.695** / -0.329 | **+0.048 / +0.449† / -0.336** |
| H1a Concept MLP | +0.031 / +0.495* / -0.348 | **-0.078 / +0.201 / -0.303** |
| H1b Concept LR | +0.363 / +0.161 / +0.186 | **+0.115 / +0.094 / +0.040** |
| H1b Concept MLP | +0.183 / +0.083 / +0.118 | **-0.057 / +0.031 / -0.101** |
| H1c Concept LR | +0.409 / +0.422† / +0.038 | **+0.143 / +0.246 / -0.061** |
| H1c Concept MLP | +0.189 / +0.298 / -0.041 | **-0.062 / +0.108 / -0.188** |
| H1d Concept LR | +0.016 / +0.514* / -0.354 | **+0.033 / +0.300 / -0.222** |
| H1d Concept MLP | -0.027 / +0.436† / -0.360 | **+0.001 / +0.156 / -0.160** |

---

## 3. Actions taken this session

### 3.1 Code changes (4 files, all uncommitted)

**`1_code/7_main_analysis/1_main_text/2_coverage_semantic_interaction.py`** — the bug
- Added `_concept_mlp_raw_gaps`, `_concept_mlp_adj_gaps` to the import block.
- `_raw_gaps_for` / `_adj_gaps_for` now take `corpus` and route LR + MLP to the
  concept loaders when `corpus == "concept"`. ZS stays model-based (there is no
  Concept ZS row in `_H1_CONFIGS`). Added docstrings explaining *why* `corpus`
  must be honoured, so the branch is not "simplified away" again.
- Passed `corpus` at the two call sites in `_h1_config_row`.
- **New `h1_grid_input_paths(root)`** — derives the grid's full input set from
  `_H1_CONFIGS` itself (coverage + raw/adjusted gaps per config, concept
  included), de-duplicated in stable order. Closes the fingerprint hole and is
  shared with j1 so the two grids cannot drift apart.
- Fingerprint now includes `*h1_grid_input_paths(...)`; `SCRIPT_VERSION` `"2"→"3"`.
- **Reordered `run()`**: the H1 grid is now computed *before* `num4` is written,
  so the grid can export the cells the prose quotes.
- **New `_h1_grid_macros()`** driven by two small tables (`_H1_QUOTED_CELLS`,
  `_H1_POSITIVE_COUNTS`) — emits `\ConceptLRCovgapAdjRho`,
  `\HOneACovgapAdjPositiveCount(+Total)`, `\HOneDPolicyAdjPositiveCount(+Total)`.

**`1_code/7_main_analysis/2_appendix/j1_raw_value_correlation.py`**
- Passed `corpus` at its two `_raw_gaps_for` / `_adj_gaps_for` call sites.
- Replaced its duplicated fingerprint path list with the shared
  `h1_grid_input_paths`; tag `"j1_raw_v1"→"j1_raw_v2"`.
- Removed the now-unused `output_dir_for_model` import and `GAP_SUFFIX` constant.

**`1_code/7_main_analysis/0_shared/analysis_orchestrator.py`**
- Registered J.1 in `APPENDIX_SPECS` (`flag: appendix-j1-raw-value`,
  `step_id: "J1"`, `in_all: True`, alias `raw-value-correlation`), so
  `--appendix-all` and `--warm-replay-with-appendix` now run it.
- Updated the canonical-order comment to include J1, K1.

**`1_code/7_main_analysis/0_shared/shared_utils.py`**
- Added the J.1 table to `MANUSCRIPT_APPENDIX_TABLE_FILES` so
  `require_pdf_inputs` guards it.

### 3.2 Outputs regenerated (10 files, all uncommitted)

```
python 1_code/7_main_analysis/1_main_text/2_coverage_semantic_interaction.py --embed-model mpnet   --overwrite
python 1_code/7_main_analysis/1_main_text/2_coverage_semantic_interaction.py --embed-model minilm  --overwrite
python 1_code/7_main_analysis/1_main_text/2_coverage_semantic_interaction.py --embed-model scibert --overwrite
python 1_code/7_main_analysis/2_appendix/j1_raw_value_correlation.py --embed-model mpnet --overwrite
```
Touched: `4_outputs/{mpnet,minilm,scibert}/tables/{tab4,num4}_interaction_h25.tex`,
the three `interaction_h25.fingerprint.json`, and the three J.1 files
(`tab_j1_raw_value_correlation.tex`, `raw_value_correlation.json`, `.fingerprint.json`).

### 3.3 Verification performed (all PASSED)

1. **Pre-flight oracle.** Before editing, the *real* module was loaded and its
   own `_h1_config_row` run with the proposed dispatch patched in memory. It
   predicted: exactly 8 blocks change, all in the two Concept rows; 28 blocks
   bit-identical; zero duplicate rows remaining. The committed regeneration then
   matched that prediction **cell for cell**.
2. **Surgical-diff gate.** `git diff` on `tab4` = `8 insertions, 8 deletions`;
   count of changed lines not containing "Concept" = **0**.
3. **No-duplicates gate.** No Concept row equals its MPNet counterpart in any
   block, in either table.
4. **Canonical stats untouched.** `interaction_h25.json` byte-identical in all
   three namespaces — proves the fix did not disturb the non-grid statistics.
5. **Namespace convergence.** All three `tab4` files now share md5
   `39629c08da990d5440fd05f40fa352d5`.
6. **Fingerprint regression test.** Re-run without `--overwrite` → correctly
   SKIPS. `touch` a `data/concept/` gap file → correctly RE-DERIVES. Verified for
   both the interaction script and J.1. (This behaviour did not exist before.)
7. **CLI wiring.** `python main.py --help` shows `--appendix-j1-raw-value`;
   `_insert_model_in_rel` resolves the new guard path to the real file.
8. **Appendix count claim re-checked.** `dissertation.tex:794` ("positive in 7/9
   and 9/9 configs respectively under Pearson") **still holds** post-fix — H1a
   adj is positive in 7/9 (SciBERT LR/MLP negative), H1d adj in 9/9.

### 3.4 Pre-existing work NOT mine — do not conflate

The working tree already contained an in-flight **Appendix K.1 (pooled OLS
regression)** session:
- `1_code/7_main_analysis/2_appendix/k1_regression_semantic_gap.py` (rank-within-config + pct-point rescale)
- `4_outputs/appendix/mpnet/k1_regression_semantic_gap/**` (4 files)
- `3_writing/dissertation.tex` — K.1 additions at **lines ~53, ~428, ~853+**
- `handoff.md` (now preserved at `5_notes/handoff_k1_regression_2026-08-04.md`)
- untracked `0_literature/register_adj/RavfogelS_etal_2020_INSP`

**My `dissertation.tex` edit will be at line 396 only — no overlap with the K.1
hunks.** They must be staged separately. Note the preserved K.1 handoff claims
"Manuscript write-up NOT started", but K.1 prose *is* present in the working
tree — that handoff is stale on that point. Confirm with the user before
committing anything K.1.

---

## 4. What remains, and why

### 4.1 Manuscript prose — `3_writing/dissertation.tex:396` (REQUIRED)

Three defects on that line. Only the first is caused by this bug; the other two
are pre-existing staleness surfaced by the audit.

Current text (excerpt):
> `...positive across all nine configs, e.g.\ Concept LR $+0.451^{\dagger}$) and with policy coverage (H1d: positive in 8/9 configs, significant for MPNet MLP and both SciBERT configs).`

Required edits:
1. `Concept LR $+0.451^{\dagger}$` → `Concept LR $\ConceptLRCovgapAdjRho$`
   (macro now emitted; renders `+0.439$^{\dagger}$`). **The literal is wrong by
   two generations — do not simply retype the new number.**
2. `positive across all nine configs` → optionally macro-ise as
   `\HOneACovgapAdjPositiveCount/\HOneACovgapAdjPositiveCountTotal` (= 9/9).
   Factually correct as-is.
3. `H1d: positive in 8/9 configs, significant for MPNet MLP and both SciBERT
   configs` → **factually wrong.** Correct post-fix: **positive in 9/9**;
   significant at p<0.05 for **MPNet LR (\*\*), SciBERT LR (\*), Concept LR (\*)**;
   MPNet MLP is marginal (†); **SciBERT MLP is not significant.**

**Do NOT change** the sentence "Table~\ref{tab:interaction} shows this apparent
cancellation replicates across every encoder--classifier config" — re-verified
true post-fix (H1a register negative in 9/9).

**No change needed at line 794** (verified in §3.3.8) — but re-confirm after any
regeneration.

### 4.2 Rebuild the PDF
```
python main.py --build-pdf --overwrite     # needs bash/WSL
```
Confirm Table `tab:interaction` and Table `tab:raw-value-correlation` render the
corrected Concept rows. **This takes seconds — do not blind-poll 120s.**

### 4.3 Commits (nothing is committed yet)

Intended split, staging **only** the files listed in §3.1/§3.2 (never the K.1
files from §3.4):
- **Commit 1 (code):** the 4 files in §3.1.
  Suggested message: `Fix Concept-row gap dispatch in H1a-H1d grid; register J.1 in appendix runner`
- **Commit 2 (outputs + prose):** the 10 files in §3.2 plus the line-396 edit.

`3_writing/dissertation.tex` contains **both** my hunk and the K.1 hunks, so it
cannot be staged wholesale. Use `git add -p`, or write my hunk to a patch and
`git apply --cached` it.

### 4.4 Optional follow-ups (deliberately NOT done)

- **Dead module.** `0_shared/h1_register_correlation_table.py` has a `run()` that
  has never executed and writes to a directory that has never existed. Either
  wire it in or demote it to a clearly-named loaders module. Left alone to keep
  this change surgical.
- **Design smell.** A model-independent table is still written into three model
  namespaces. Now consistent, but will drift again if any namespace is
  regenerated alone. Consider emitting it once.
- **Audit the other 3 hardcoded-literal risks.** This audit found one prose
  literal that had silently gone stale. There may be more; a sweep for numeric
  literals in prose that duplicate generated table cells would be prudent.

---

## 5. Concerns to emphasise

1. **The bug class, not just the instance.** The failure was a function that
   *received* a discriminating argument and silently dropped it, while the
   correct loaders sat imported-but-unused. Nothing failed loudly; the output was
   plausible. The only reason it was caught is that one block produced an exact
   duplicate. **The other three blocks (H1a/H1b/H1c) were equally wrong and
   showed no visible tell for ~2 commits.** Worth asking where else a `corpus` /
   `model` / `method` selector is accepted but unused.
2. **Two of the three defects I fixed were invisible to the pipeline's own
   guards.** J.1 was `\input` by the manuscript while being in no runner registry
   and no PDF-input guard. A table can therefore be manuscript-facing, committed,
   and permanently stale. Recommend auditing every `\input{../4_outputs/...}` in
   `dissertation.tex` against `APPENDIX_SPECS` + `MANUSCRIPT_*_FILES`.
3. **I did not verify the concept pipeline upstream of these tables.** I verified
   that `data/concept/semantic_gap_distances_*.json` exist, differ substantially
   from canonical, and are produced by `main.py`. I did **not** audit whether the
   concept centroids/scores themselves are correct. The fix makes the grid *use
   the intended inputs*; it does not certify those inputs.
4. **`_file_fp` uses mtime**, so `2_data`/`4_outputs` re-hydration changes
   fingerprints and forces Tier-B re-runs. Expected per `AGENTS.md`, but it means
   my fingerprint-regression test (`touch` → re-derive) is partly an mtime
   effect. The *content* path is still covered (first 64KB + size).
5. **Uncommitted K.1 work is entangled in `dissertation.tex`.** Highest practical
   risk right now is someone running `git add -A` and fusing two unrelated
   concerns.

---

## 6. The comprehensive plan (status per step)

| # | Step | Status |
|---|---|---|
| 1 | Fix `corpus` dispatch in `_raw_gaps_for`/`_adj_gaps_for` + both call sites | **DONE** |
| 2 | Fix j1's two call sites (inherits via import) | **DONE** |
| 3 | Register J.1 in `APPENDIX_SPECS` + `MANUSCRIPT_APPENDIX_TABLE_FILES` | **DONE** |
| 4 | Add concept paths to fingerprint (`h1_grid_input_paths`); bump `SCRIPT_VERSION`→"3", j1 tag→v2 | **DONE** |
| 5 | Emit `\ConceptLRCovgapAdjRho` + positive-count macros; reorder `run()` | **DONE** |
| 6 | Regenerate tab4 ×3 namespaces + J.1 with `--overwrite` | **DONE** |
| 7 | Verify: surgical diff, no duplicates, canonical JSON untouched, md5 convergence, fingerprint skip/re-derive, CLI wiring | **DONE** |
| 8 | **Update `dissertation.tex:396`** (macro + H1d count/significance) | **TODO** — §4.1 |
| 9 | **Re-confirm line 794** (expected: no change) | **TODO** |
| 10 | **Rebuild PDF** and eyeball both tables | **TODO** |
| 11 | **Commit 1 (code, 4 files)** | **TODO** |
| 12 | **Commit 2 (outputs 10 files + prose hunk)** | **TODO** |

---

## 7. What was interrupted

**Nothing was mid-execution.** The user asked to stop at the end of the
verification phase (step 7). The last commands run were the fingerprint
regression tests; all passed and the tree is in a consistent, verified state.

One cosmetic artefact worth knowing so it is not misread as a failure:

- While testing J.1's fingerprint, I ran a chained command
  `... | grep -ci "skip" && touch ... && ...`. The `grep -c` returned `0`
  (exit status 1), which **short-circuited the `&&` chain**, so the second half
  never ran and the output looked like a failed test.
- **Cause was my test sequencing, not the code:** the immediately preceding
  command had already `touch`ed `4_outputs/mpnet/data/concept/semantic_gap_distances_lr.json`,
  which is in J.1's fingerprint set, so J.1 correctly re-derived instead of
  skipping. That *is* the re-derive test passing.
- I then re-ran the skip test cleanly and it printed
  `Skipping ... raw_value_correlation.json -- inputs unchanged`. **Both halves of
  the J.1 fingerprint test are verified.**

Side effects of that testing: the mtimes of
`4_outputs/mpnet/data/concept/semantic_gap_distances_{lr,mlp}.json` were bumped
by `touch`. **Contents are unchanged and git shows no diff for them.** They are
inputs, not outputs; no action needed.

### Immediate next action for a fresh agent

Start at **§4.1** — edit `3_writing/dissertation.tex:396`. Everything needed is
in this file; no re-investigation required. Re-verify current state first with:

```bash
git status --short
grep "Concept" 4_outputs/mpnet/tables/tab4_interaction_h25.tex     # expect the §2.7 corrected values
grep -E "ConceptLR|PositiveCount" 4_outputs/mpnet/tables/num4_interaction_h25.tex
md5sum 4_outputs/{mpnet,minilm,scibert}/tables/tab4_interaction_h25.tex   # expect 3× identical
```

---

## 8. Session 2 (2026-08-04) — prose done; PDF build BLOCKED (stop & report)

Picked up from §4.1. **Prose edit completed; PDF rebuild blocked by pre-existing
issues unrelated to this fix. Stopped at the blocker rather than silently
touching the entangled K.1 work.**

### 8.1 What got done
- **`3_writing/dissertation.tex:396` edited** (the only change I made to the
  manuscript this session). Three defects fixed per §4.1:
  1. `Concept LR $+0.451^{\dagger}$` → `Concept LR \ConceptLRCovgapAdjRho`
     (macro now emitted by `num4`; renders `+0.439$^{\dagger}$`). **The macro
     already contains its `$...$`, so it is used unadorned — do NOT re-wrap in
     `$...$` (would nest math mode and break the build).**
  2. `positive across all nine configs` → `positive in
     \HOneACovgapAdjPositiveCount/\HOneACovgapAdjPositiveCountTotal\ configs`
     (renders `9/9`).
  3. `H1d: positive in 8/9 configs, significant for MPNet MLP and both SciBERT
     configs` → `positive in
     \HOneDPolicyAdjPositiveCount/\HOneDPolicyAdjPositiveCountTotal\ configs,
     significant for MPNet LR and the Concept and SciBERT LR configs at
     $p<0.05$, with MPNet MLP marginal at $p<0.10$`. (Post-fix H1d adjusted is
     positive in **9/9**; significant at p<0.05 for MPNet LR \*\*, SciBERT LR \*,
     Concept LR \*; MPNet MLP is marginal †; SciBERT MLP / MiniLM not significant.)
- **Line 794 re-confirmed unchanged.** Verified against the regenerated J.1
  (Pearson) table: H1a adjusted positive in 7/9 (SciBERT LR/MLP negative),
  H1d adjusted in 9/9. The "positive in 7/9 and 9/9 configs respectively under
  Pearson" claim still holds. **No edit needed.**
- `num4_interaction_h25.tex` is `\input`ed at `dissertation.tex:39`, i.e.
  **before** line 396, so the macros are defined at point of use. Confirmed.

### 8.2 PDF build is BLOCKED — and the blocker is NOT this fix
Ran `python main.py --build-pdf --overwrite` (via tmux, clean `3_writing/artifact`).
It fails with a fatal `pdflatex` error. Root-caused to **two independent,
pre-existing problems in the uncommitted K.1 work / figure assets**:

1. **Missing figure PDFs.** `fig1_conceptual_framework.pdf` and
   `fig6_pipeline_flowchart.pdf` are `\include`d at `dissertation.tex:142` and
   `:294` but **do not exist in the current `4_outputs/`** — they survive only in
   `4_outputs_backup_before_model_namespace/main/figures/`. Source `.tex`/`fig_pipeline_flowchart.pdf`
   exist under `1_code/8_visualization/`. This is a namespace-refactor fallout,
   unrelated to H1.
2. **K.1 table is malformed LaTeX.** `4_outputs/appendix/mpnet/k1_regression_semantic_gap/tables/tab_k1_specification_grid.tex`
   contains unescaped `×` and `%` outside math mode (`covgap×i_minilm`,
   `covgap×i_scibert`, `covgap×i_concept`, `gap ($\vert$research\% ...`). These
   raise `Missing $ inserted`. The `×`/`%` come from the **K.1 Python generator**
   (`k1_regression_semantic_gap.py`) — editing the `.tex` is futile (regenerated
   on re-run). Fixing it means editing the generator, i.e. doing K.1's work.

Also noted: `dissertation.tex:56` had a redundant
`\InputIfFileExists{...tab_k1_specification_grid.tex}` **in the preamble** (before
`\begin{document}` at :80) — a `tabular` input there is a hard LaTeX error. The
same table is correctly `\input`ed in the body at `:867`. I **temporarily removed
line 56 to test the build, then REVERTED it** to keep my diff isolated to the H1
prose. The line-56 removal is a correct fix but lives in K.1's region; leave it to
the K.1 author (or apply separately) rather than fuse it with the H1 commit.

**My line-396 edit contributes zero build errors** — all fatal errors are the
figure/K.1 issues above. The H1 fix is therefore verified at the data/table/prose
level; only the full-PDF render is blocked by out-of-scope work.

### 8.3 Corrected H1d significance wording (for the prose, already written)
The old prose claimed "significant for MPNet MLP and both SciBERT configs" —
**wrong on both counts**: MPNet MLP is only marginal (†), and SciBERT MLP is not
significant. The corrected sentence (now in the manuscript) names the three
p<0.05 configs (MPNet LR, Concept LR, SciBERT LR) and flags MPNet MLP as marginal.

### 8.4 Recommended next steps (judgment needed — did NOT auto-proceed)
1. **Decide on the K.1 entanglement.** Either (a) the K.1 author fixes their
   generator's `×`/`%` escaping and restores `fig1`/`fig6`, or (b) we temporarily
   exclude K.1 from the build to verify the H1 fix renders. I did not do either
   without your say-so.
2. **Once the build is unblocked**, re-run `python main.py --build-pdf --overwrite`,
   eyeball Table `tab:interaction` (Concept rows = §2.7) and Table
   `tab:raw-value-correlation`.
3. **Commits (per §4.3) are NOT yet made** — the build is unverified and K.1 files
   are in the tree. My `dissertation.tex` change is now a single isolated hunk
   (line 396 only); `git add -p` will stage it cleanly without the K.1 hunks.
   Stage: commit-1 code (4 files, §3.1), commit-2 outputs+prose (10 files §3.2 +
   the line-396 hunk).

### 8.5 What remains from the original plan
| Step | Status |
|---|---|
| 8 (prose edit line 396) | **DONE** |
| 9 (re-confirm line 794) | **DONE — no change** |
| 10 (rebuild PDF) | **BLOCKED** by K.1/figures (§8.2) |
| 11 (commit 1: code) | **TODO — blocked on build verify** |
| 12 (commit 2: outputs+prose) | **TODO — blocked on build verify** |


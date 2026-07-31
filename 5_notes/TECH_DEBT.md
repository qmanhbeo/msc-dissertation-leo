# Technical Debt Ledger

Running ledger of known technical debt in the dissertation reproducibility
repo. Each entry records what is deferred, why, and the condition that retires
it. This file is a durable memory — do not delete entries without retiring the
debt they describe.

## 1. Register num file hand-synced pending full `--overwrite` re-run

- **Debt**: `4_outputs/appendix/mpnet/f_register_adjustment/tables/num_iterative_register_check.tex`
  was hand-edited (2026-07-31) to append `\RegIterGapSdg*` per-SDG macros so the
  manuscript could reference `\RegIterGapSdgSeventeen{}` instead of a hardcoded
  `0.3878`. The generator `f_register_adjustment.py` was patched to emit these
  macros, and the hand-synced values were verified against the committed
  `tab_register_adjusted_semgap.tex` iterative column (mean 0.2134 matches).
- **Why deferred**: a full re-run of the register stage is heavy (~94 iterative
  LR fits) and nothing else in its outputs would change.
- **Retire by**: running the register stage with `--overwrite` and confirming
  the regenerated num file is byte-identical to the hand-synced version.

## 2. Em dashes in pre-existing Results/Discussion prose

- **Debt**: pre-existing em dashes (`---`) remain in Results/Discussion prose
  that was not rewritten in the compression passes. New text avoids em dashes;
  no document-wide purge was ever performed.
- **Why deferred**: deliberate scope decision — purging the whole manuscript is
  churn with no scientific value; only rewritten sections carry the convention.
- **Retire by**: as each remaining section is touched/rewritten, remove its em
  dashes in the same pass. (`grep -n -- '---' 3_writing/dissertation.tex`)

## 3. Word-count counter is an ad-hoc session script

- **Debt**: the per-section word counter (strip LaTeX commands, bounded section
  slicing) exists only as inline Python snippets run in agent sessions; it is
  not committed anywhere reproducible.
- **Why deferred**: the counter is a review-time aid, not a pipeline stage.
- **Retire by**: committing it under `5_notes/` or `1_code/` if section-level
  counts become a recurring check (e.g. before each supervisor submission).

## 4. Verification gap: register-stage macro emission not exercised by CI

- **Debt**: the appended `\RegIterGapSdg*` emission block in
  `f_register_adjustment.py` has not been executed (only the hand-sync used its
  presumed output). A bug in the emission path would only surface on the next
  `--overwrite` run.
- **Why deferred**: same cost reason as entry 1.
- **Retire by**: the full re-run in entry 1 also retires this.

# Technical Debt Ledger

Running ledger of known technical debt in the dissertation reproducibility
repo. Each entry records what is deferred, why, and the condition that retires
it. This file is a durable memory — do not delete entries without retiring the
debt they describe.

## 1. Register num file hand-synced — RETIRED (2026-08-01)

- **Status**: the `f_register_adjustment.py` appendix script that this entry
  tracked was folded into the canon flow (`g_register_decomposition.py`) and
  deleted. Its iterative diagnostic now emits to `4_outputs/{model}/tables/`
  (`num_iterative_register_check.tex` + `tab_iterative_register_check.tex`), and
  the per-SDG `\RegIterGapSdg*` macros are produced there directly. The
  hand-synced `4_outputs/appendix/mpnet/f_register_adjustment/` tree no longer
  exists and is not referenced. This debt is retired.

## 2. Em dashes in pre-existing Results/Discussion prose

- **RETIRED (2026-07-31)**: em dashes removed from Results/Discussion prose in
  the bug-fix commit (caught three-hyphen `---` only). The Discussion-tightening
  pass (2026-07-31) found 9 *unicode* em dashes (`—`, U+2014) that the earlier
  `---` grep had missed — 3 in Discussion (now rewritten) and 6 in appendix/
  comment lines — and replaced all with ` -- `. `python3 -c "open(...).read().
  count(chr(0x2014))"` now returns 0 and `grep -c -- '---'` is 0. The manuscript
  is fully em-dash-free.

## 3. Word-count counter is an ad-hoc session script

- **RETIRED (2026-07-31)**: the per-section word counter is now committed at
  `5_notes/word_count.py` (strips LaTeX commands, bounded section slicing,
  excludes appendix sections from the main-text total). Reproducible:
  `python 5_notes/word_count.py`. Current main-text total ≈ 9,914 (above the
  8.8k target — a separate content-trimming concern, see open Discussion pass).

## 4. Verification gap: register-stage macro emission — RETIRED (2026-08-01)

- **Status**: superseded by entry 1's retirement. The `\RegIterGapSdg*` macros
  are now emitted by `g_register_decomposition.py` (canon) to
  `4_outputs/{model}/tables/num_iterative_register_check.tex`, exercised on every
  replay. No separate `f_register_adjustment.py` emission path remains. Retired.

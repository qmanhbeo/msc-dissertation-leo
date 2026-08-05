# Hand-off: INLP register-validation appendix promotion (IMPLEMENTATION phase)

**Last updated:** 2026-08-05 (implementation phase; COMPLETE — PDF verified; commits pending user approval)
**Status:** The promoted appendix stage `a1_register_validation` is **written, registered,
gated, and fully verified** (61/61 acceptance gates pass, deterministically, in 4 independent
runs). All manuscript edits are applied. The final engineering step — **`--build-pdf` after the
LaTeX structural fix** — has now been **VERIFIED SUCCESSFUL** (see §2.5 / §4): the build
succeeds, Appendix G renders with both tables, all `\RegVal*` macros resolve, the 707pt
"Not in outer par mode" cascade is gone, and the rewired Methodology/Limitations/Conclusion
sentences read correctly in the PDF. The `3a_warm_replay_texts/` gz fallback path was also
exercised (see §4 item 3) and works for this script's model/shard naming. No commits have been
made. Everything below is ground truth for continuing.

---

## 1. Context — where we are

We are promoting the verified INLP "register-removal" validation diagnostic (scratch scripts
`5_notes/scratch/register_validation_{check,followup,followup2}.py`) into a durable,
reproducible dissertation appendix stage, per `report.md` (`54ffea0`, ground truth for all
numbers) and the prior read-only handoff (the file you are reading now was rewritten).

The promotion is essentially complete:

- **Script:** `1_code/7_main_analysis/2_appendix/a1_register_validation.py` (NEW, ~1050 lines)
  — a consolidated, A3-shaped appendix script that re-implements the whole validated line
  (2b/2c/2d, Step-3, Item-1/2/3, mega-doc analysis, draw stability) deterministically from
  `2_data`, with a built-in **acceptance-gate block (61 checks)** that raises RuntimeError on
  any mismatch with `report.md §2.2`. **All 61 gates PASS** (verified in 4 full runs —
  determinism confirmed).
- **Registration:** `APPENDIX_SPECS` entry (step_id `F2`, `in_all=True`, `requires=None`,
  flag `--appendix-a1-register-validation`), outputs added to `shared_utils.py`
  `MANUSCRIPT_EXTRA_FILES` + `MANUSCRIPT_APPENDIX_TABLE_FILES`, `main.py --appendix-all` help
  text updated, README/PIPELINE/AGENTS inventories updated.
- **Outputs generated** (committed-tracked `4_outputs/`, untracked so far):
  `4_outputs/appendix/mpnet/a1_register_validation/{data/register_validation.json,
  data/register_validation.csv, tables/tab_a1_register_validation.tex,
  tables/tab_a1_register_validation_selectivity.tex, tables/num_a1_register_validation.tex}`
  + fingerprint sidecar.
- **Manuscript:** new appendix section inserted (will become Appendix G), all "unvalidated /
  left to future work" wording rewired to point at it, stale appendix-letter comments fixed,
  preamble macro input added.
- **PDF:** `--build-pdf` FAILS until the just-applied structural fix is verified (see §2.5).

### Status summary table

| Item | Status |
|---|---|
| Script written + py_compile clean | DONE |
| Acceptance gates 61/61 (4 runs: a1rv_run, a1rv2_run, a1final, a1main via main.py) | DONE |
| Determinism (two runs byte-consistent on all numbers) | DONE |
| Registry + build-pdf file lists + main.py help + README/PIPELINE/AGENTS | DONE |
| dissertation.tex: new section + rewires + stale comments + preamble input | DONE |
| LaTeX structural bug (`\end{table}` swallowed section G) | FIXED, **VERIFIED via --build-pdf** |
| `--build-pdf` success + visual check of appendix pages | **DONE** (build ok; text-level check ok, no image rendering) |
| Cleanup of min-test scratch files | **DONE** |
| Commits (a) code+docs, (b) TeX, (c) outputs+PDF | REMAINING (needs user approval) |

---

## 2. Key known facts (pick-up without re-reading)

### 2.1 The promoted script (`a1_register_validation.py`) — what it does

- **Slug** `a1_register_validation`; **step_id** `F2`; **flag** `--appendix-a1-register-validation`
  (alias `--register-validation`); **in_all=True**; **requires=None**. Canonical order docstring
  in `analysis_orchestrator.py:35` updated to `..., H1, I1, F2, G(opt-in), J1, K1`.
- **Canonical-MPNet-only gate**: `if model != DEFAULT_EMBED_MODEL: log + return` (mirrors the
  zero-shot gating precedent).
- **`--seed` pinned to 42** (raises on any other value; draw-instability tested at fixed fresh
  seeds 43/44/45).
- **nltk guard**: fail-closed at runtime if `tokenizers/punkt` or
  `taggers/averaged_perceptron_tagger_eng` missing; actionable message with
  `nltk.download(...)`. (Conservative decision: NOT added to environment.yml — conda-forge
  `nltk_data` package ships the legacy pickle tagger, compatibility with nltk 3.10 unverified.
  Documented in script docstring + README Environment notes.)
- **Fingerprint**: `fingerprint_of(score manifest, research-embed manifest, G.npy) +
  SCRIPT_VERSION("1")`; `should_skip` / `record_fingerprint` per A3 pattern.
- **Sampling is the verified seed-42 single-stream** design: module-level `_rng` created once,
  never reassigned; draw 1 = original per-SDG-dedup (2A), draw 2 = one-per-parent (2B),
  draw 3 = Item-3; research sampled BEFORE policy each draw; fresh generators for seeds
  43/44/45. Do NOT refactor (per-call re-seeding already produced wrong samples once in the
  scratch line).
- **Text loading** uses `resolve_research_text_path(model, f"part-{sid:05d}")` /
  `resolve_policy_text_path(model)` + `open_text` (canonical `2_segmented/` first, `3a_warm_replay_texts/`
  gz fallback). **Note: the gz fallback path has NOT been exercised** (canonical files exist on
  disk); optional verification listed in §4.
- **Runtime ~10 min** (6 full assembles: draws 1/2/3 + seeds 43/44/45, POS-tagging 6×408
  texts). MUST run under tmux (harness kills at ~120 s).
- **Acceptance gate**: 61 checks, all PASS in every run. Key values reproduced exactly:
  2A 0.456/0.909/0.505; 2B 0.544/0.944/0.603; pooled 206/408 and 246/408; bootstrap diff
  +0.0979 CI [0.0245, 0.1691] p(>0)=0.9940; mega-policy exclusion 0.5744 (220/383) CI
  [0.5244, 0.6229] p=0.002079; Item-3 pooled −0.088/−0.074; policy other-dist −0.197
  (p=0.00468); per-SDG 0/17 sig (3 pos / 14 neg); mega-excl draw1/2/3 +0.127/−0.138/−0.213;
  draw 43/44/45 −0.130/−0.004/+0.126; 2b/2c orig & 2B (incl. partials 0.130→0.253,
  −0.155→−0.159); Step-3 raw LR 0.691/kNN 0.554, adj 0.672/0.578; corpus feature means
  (1.427/0.854, 0.780/3.130, 10.303/8.453, 37.035/63.803, 6.901/6.463, 36.367/39.003); mega
  features (276.879/35.637, 3.349/9.771, 18.996/5.879, 20.016/38.838); 2A 390 distinct/25
  multi-parent/7 mega-docs/15 SDGs; 2B 408/0.
- **Outputs**: JSON (all numbers, nested, machine-readable; includes
  `register_score_operationalization` honesty note), CSV (8 corpus-discrimination rows),
  two tab tex files, num tex with **88 `\RegVal*` macros** (namespace distinct from
  `Register*`/`RegIter*`).

### 2.2 LaTeX macro namespace — critical gotcha (already handled, don't reintroduce)

- **TeX control-word names cannot contain digits** (`\RegValDraw43` parses as `\RegValDraw`
  + `43` → "Missing \begin{document}" in the preamble). The script emits spelled-out names:
  `\RegValDrawFortyThree/Four/Five(+P)`, `\RegValMegaExclDrawOne/Two/Three(+P)`,
  `\RegValItemThreePooledRaw/Adj`. `\RegValAccAdjOpp`-style names (no digits) are fine.
- The num file is `\input` ONCE in the preamble (`\InputIfFileExists{...num_a1_register_validation.tex}{}{}`
  after the `num17_reference_split.tex` line), NOT again in the section (would re-define).
- All 64 macros referenced in dissertation.tex verified present in the generated num file
  (regex check done; none missing).

### 2.3 Manuscript changes (all applied; exact locations current as of this handoff)

- **Preamble**: `\InputIfFileExists{../4_outputs/appendix/mpnet/a1_register_validation/tables/num_a1_register_validation.tex}{}{}`
  added after `num17_reference_split.tex` (~line 48).
- **New appendix section** inserted between the register-removal appendix (F) and
  Concept-Retrieval, label `app:register-validation`; title "Register Removal: Validation
  Against Independent Linguistic Register Markers". It becomes **Appendix G** (Concept-Retrieval
  → H, Cross-Method → I, Pooled Regression → J, AI Declaration → K; all refs are label-based,
  nothing else needed). Section contains: Motivation; Register score and samples; Two sample
  constructions; Corpus-discrimination accuracy (+ Table `tab:register-validation-accuracy`,
  `\resizebox{\textwidth}{!}{\input{...tab_a1_register_validation.tex}}`); "An apparent
  residual-register signal was traced to clustering"; "No robust evidence that topic is being
  systematically removed" (+ Table `tab:register-validation-selectivity`); "One apparently
  surviving signal failed the draw-stability check" (explicit honest history: mega-doc
  clustering traced, −0.197 downgraded to noise); Conclusion (only §2.2 reconciled values,
  "substantial reduction, not complete elimination, no robust evidence of topic removal");
  Limitations (PC1-vs-z-sum operationalization undecided + flagged, n=408 / per-SDG n=12 power,
  draw-stability applied only to within-SDG trace, one-per-parent is a rebuild not subset
  (compositional), MPNet-only, and the two distinct −0.197 statistics caution).
- **Rewires** (all "unvalidated / left to future work" language):
  - Methodology identification argument (~line 279): now says the interpretation is evaluated
    in Appendix~\ref{app:register-validation}, finds substantial reduction but not complete
    elimination, no robust evidence of topic removal; limitations in the appendix.
  - Table notes: tab:register-decomposition (~:373) and tab:interaction (~:392):
    "(primary, unvalidated)" → "(primary estimate; evaluated in Appendix~\ref{app:register-validation})".
  - Limitations "Register effects" (~:477): "pending independent validation" →
    evaluated in appendix; "primary but unvalidated estimate" → "primary estimate"; final
    "left to future work" sentence replaced with the partial-support + appendix-limitations
    statement.
  - Conclusion (~:488): "the INLP removal is unvalidated" → "whose register interpretation is
    evaluated against independent linguistic register markers in Appendix~... (limits in
    Section~\ref{sec:limitations})".
  - Stale header comments fixed: "Appendix D: Model Selection" → E; "Appendix E:
    Register-Adjustment" → F.
- **Line 471** "left to future work" (corpus-scope symmetry) is UNRELATED — untouched.

### 2.4 Registration / docs (all applied)

- `analysis_orchestrator.py`: new spec dict inserted BEFORE the opt-in G spec (~line 141);
  docstring canonical order updated.
- `shared_utils.py`: `MANUSCRIPT_EXTRA_FILES` += json + csv (appendix/a1_register_validation/…);
  `MANUSCRIPT_APPENDIX_TABLE_FILES` += the 3 tex files (entries at the TOP of the list).
- `main.py:136`: `--appendix-all` help now "(A2, A3, B2, C, C1, C0, D1, H1, I1, F2, J1, K1)".
- `README.md`: new command row after `--appendix-all` row; nltk-data bullet added to
  "Environment notes" (punkt + averaged_perceptron_tagger_eng, with the download one-liner).
  The :12 platform note was reviewed and needs NO change (does not enumerate stages).
- `PIPELINE.md`: appendix table row `a1_register_validation.py | G (step F2; canonical MPNet
  only) | appendix/[model]/a1_register_validation/`.
- `AGENTS.md`: Tier-B counts 15 → 16 (two places); checkpoint-inventory row comment updated.

### 2.5 The LaTeX bug just found and fixed (MUST verify)

- Symptom: `--build-pdf` failed with `! LaTeX Error: Not in outer par mode` at both
  `\begin{table}` in the new section (also "Float too large for page by ~707pt" cascades).
- Root cause (found by bisection with minimal test docs): the first insertion script replaced
  the anchor `\end{table}\n\clearpage\n\section{Concept-Retrieval Sensitivity}` with
  `draft + "\n" + anchor`, which mechanically moved the register-check table's `\end{table}`
  (line 711) to AFTER the new section — so the ENTIRE new section sat inside the register-check
  `table` environment (inner par mode → floats illegal).
- Fix applied (two edits, structure now balanced 3 begin / 3 end in the F..G region):
  1. `\end{table}` restored immediately after the `\par\smallskip...` Notes line of
     `tab:iterative-register-check` (before the Appendix G comment block).
  2. The stray `\end{table}` before `\clearpage\section{Concept-Retrieval Sensitivity}`
     removed (it was the displaced register-check closing).
- **VERIFIED 2026-08-05 (later session)**: `--build-pdf --overwrite` succeeds; Appendix G
  renders with both tables and all macros; the 707pt "Not in outer par mode" cascade is gone.
- Scratch min-test files to delete: `3_writing/min_test*.tex` (min_test, min_test2..9 — 7
  present on disk) + their `.aux/.log/.out/.pdf` siblings, and `/tmp/opencode/*.log` junk is
  harmless (tmp). Also `5_notes/scratch/` untouched (keep — provenance). **DONE** — all
  min_test files removed.

### 2.6 Ground truth / numbers authority

- `report.md §2.2` = the ONLY numbers for the write-up; §2.3 supersessions honored
  (0.603 sig-above-chance framing replaces "collapse to chance"; 2c red flag = mega-doc
  artifact; −0.197 = draw-unstable noise, reported as null, not finding); §2.4 unresolved
  items deliberately NOT resolved (PC1-vs-z-sum operationalization flagged in appendix
  limitations + JSON; the two −0.197 statistics caution included).
- Ground-truth logs: `5_notes/scratch/{followup2.log, regcheck_followup.log, regcheck_full.log}`
  — the promoted script reproduces every value.

### 2.7 Runs performed this session (all under tmux, all 61/61 PASS)

| tmux session | log | what |
|---|---|---|
| a1rv | /tmp/opencode/a1rv_run.log | first script run (pre macro-name edit) |
| a1rv2 | /tmp/opencode/a1rv2_run.log | re-run after corpus-macro name change (determinism check) |
| a1final | /tmp/opencode/a1final.log | re-run after digit-name fix (FINAL script; current outputs) |
| a1main | /tmp/opencode/a1main.log | `python main.py --appendix-a1-register-validation --overwrite` (registry path E2E) |

All sessions completed (log.DONE markers exist). `4_outputs/.../register_validation.json` is
from a1final (macros dict uses the spelled-out names; CSV/tables identical across runs).

### 2.8 Environment facts

- Python: `/home/manh/miniforge3/envs/dissertation/bin/python` (3.11; `source activate` broken).
- nltk 3.10.0; data at `/home/manh/nltk_data/` (punkt + averaged_perceptron_tagger_eng present).
- LaTeX: `latexmk`/`pdflatex` available; `--build-pdf` needs bash and runs from 3_writing.
- tmux discipline: long runs MUST be `tmux new-session -d -s <name> "<cmd> > log 2>&1; touch log.DONE"`,
  poll the log, never the PID. The script takes ~10 min; `--build-pdf` is seconds.

### 2.9 Git state

- HEAD `54ffea0`; working tree has UNCOMMITTED changes:
  - Modified: `analysis_orchestrator.py`, `shared_utils.py`, `dissertation.tex`, `PIPELINE.md`,
    `README.md`, `handoff.md` (this file), `main.py`
  - Untracked: `1_code/7_main_analysis/2_appendix/a1_register_validation.py`,
    `4_outputs/appendix/mpnet/a1_register_validation/`, `3_writing/min_test*.tex`
- Nothing committed this session. AGENTS.md: commit only on explicit user approval, one
  concern per commit, re-verify the affected stage before committing.

---

## 3. Actions / decisions made this session and why

1. **Wrote the promoted script** `1_code/7_main_analysis/2_appendix/a1_register_validation.py`
   (A3-shaped skeleton; sys.path bootstrap; argparse; logging; canonical-only gate; seed pin;
   nltk fail-closed guard; fingerprint of score manifest + embed manifest + G.npy +
   SCRIPT_VERSION). Ported the verified scratch logic VERBATIM (sampling stream order,
   feature formulas, PC1 orientation, centroid defs, Wilson/binomial/boot, LR C=1.0, Step-3
   LR C=10/kNN 5). Added: corpus-mean features (report1 Step 1), mega-vs-non-mega feature
   contrast, Step-3 selectivity on draw-1, and a 61-check ACCEPTANCE GATE block that raises on
   any mismatch. No new analyses, no n=120 scaling, no §2.4 resolution.
2. **Decisions:** slug `a1_register_validation` + step_id `F2` + in_all=True (handoff
   recommendation); outputs as planned (json/csv/2 tabs/num tex with `\RegVal*` macros);
   nltk handled by docstring+README+runtime guard (NOT environment.yml — unverified conda
   package would risk a clean rebuild); whole-corpus replacement stats (57.8/30.8/25.2) kept
   OUT of the script (in-sample mega contrast computed instead); check_concept_same_space.py
   left in scratch (out of scope per handoff §3).
3. **Registered** the stage (spec dict, MANUSCRIPT_* lists, main.py help) and updated
   README/PIPELINE/AGENTS inventories.
4. **Wrote the appendix TeX** and rewired the manuscript (see §2.3). Wording decisions:
   - "substantial reduction of the corpus-linear signal, but not complete elimination; no
     robust evidence that topic is being systematically removed" — used consistently.
   - The "collapse to chance" reading is reported only as a corrected supersession (0.505 n.s.
     on the contaminated original; 0.603 sig on the clean sample).
   - One overclaim caught and removed: an early draft said mega-doc policy segments are
     "classified almost perfectly after adjustment" — NOT verified (the exclusion re-runs the
     classifier; the followup2 mega-doc acc block never printed). Now says only: excluding
     them raises 0.505→0.574 and closes most of the +0.098 gap.
   - "as the identification argument requires" → "anticipates" (softened).
   - "Stage 4" references in limitations → "draw-stability stage" (prose headers unnumbered).
5. **Fixed two LaTeX issues found by building:**
   a. Digit-containing macro names (see §2.2) → spelled-out names in script + regenerated
      outputs (run a1final) + tex updated.
   b. The displaced `\end{table}` structural bug (see §2.5) — fixed but unverified.
6. **Verified end-to-end**: `main.py --appendix-a1-register-validation --overwrite` runs the
   stage through the registry (subprocess) with 61/61 PASS and writes the fingerprint sidecar.
7. Did NOT commit anything (needs user approval). Did NOT touch scratch provenance.

---

## 4. What remains and why

1. **PDF rebuild — DONE.** `python main.py --build-pdf --overwrite` (tmux session `pdfbuild`)
   succeeded: latexmk completed, `4_outputs/dissertation.pdf` built (72 pp), no fatal errors,
   0 "Not in outer par mode", 0 "Missing \begin{document}", 0 undefined control sequences.
   Text-level inspection (pdftotext) confirms: Appendix G ("Register Removal: Validation
   Against Independent Linguistic Register Markers") present with both tables
   (tab:register-validation-accuracy + selectivity), all `\RegVal*` macros resolved to
   report.md §2.2 values, rewired Methodology/Limitations/Conclusion sentences render, and no
   register-related "unvalidated"/"left to future work" wording remains (line-471 corpus-scope
   sentence legitimately unchanged). Remaining `Float too large` warnings (lines 71/76/593/881)
   are pre-existing floats, not from the new section. No image render check (model cannot view
   images) — geometry inferred from absence of overfull-vbox errors on the G pages.
2. **Cleanup — DONE.** All `3_writing/min_test*.{tex,aux,log,out,pdf,bcf,run.xml,toc,bbl,blg}`
   deleted; `git status` now shows only intended files.
3. **gz fallback — VERIFIED.** Temporarily hid `2_data/2_segmented/{research,policy.jsonl}`
   and confirmed `resolve_research_text_path('all-mpnet-base-v2','part-00001')` /
   `resolve_policy_text_path` resolve to `3a_warm_replay_texts/mpnet/*.jsonl.gz` and
   `open_text` reads them (research line 1546 chars, policy 1238); canonical dirs restored
   and verified intact. The script passes the full model name (`args.embed_model` →
   `DEFAULT_EMBED_MODEL`) and `part-{sid:05d}` shard names, both of which resolve correctly.
4. **Commits** (one concern each, ONLY on explicit user approval):
   (a) code + registration + docs: script, analysis_orchestrator.py, shared_utils.py, main.py,
       README.md, PIPELINE.md, AGENTS.md;
   (b) manuscript TeX: dissertation.tex;
   (c) outputs + PDF: `4_outputs/appendix/mpnet/a1_register_validation/` + dissertation.pdf.
   Re-verify the affected stage before each commit (AGENTS.md).
5. **Leave unresolved (deliberately, per scope):** PC1-vs-z-sum register-score
   operationalization; the two-statistics-both-−0.197 coincidence (now documented as a
   caution in the appendix); n=120/SDG scaling; per-SDG power questions. Do NOT resolve.

---

## 5. Concerns to emphasize

1. **The PDF rebuild gate is now VERIFIED (2026-08-05, later session).** `--build-pdf
   --overwrite` succeeds; the `\end{table}` fix holds; Appendix G renders with both tables and
   all macros. If a future build still fails, re-bisect with the min-test method (see §2.5)
   rather than guessing; do not weaken the tables (e.g., do NOT convert the `table` envs to
   `tabularx` or drop `\resizebox` unless proven necessary).
2. **Reproducibility is the hard gate.** The script must reproduce every report.md §2.2
   number; the gate block is the enforcement. If any gate misses on a future run (e.g., after
   a data snapshot change), STOP — do not fudge; the whole point is the verified numbers.
3. **RNG stream order is a trap.** Single module-level `_rng`; research-then-policy per draw;
   fresh generators for 43/44/45. Do not "refactor" the sampling.
4. **Macro namespace.** `\RegVal*` only; no digits in macro names; Register*/RegIter* are
   taken by g_register_decomposition.py. The num file is input ONLY in the preamble.
5. **nltk data.** Clean env rebuilds lack the POS tagger; the guard + README note are the
   mitigation. Do NOT add the unverified conda `nltk_data` package to environment.yml without
   testing under nltk 3.10.
6. **tmux discipline.** The script is ~10 min; always tmux + log + .DONE marker.
7. **Honesty constraints in the write-up.** Only report.md §2.2 numbers; §2.3 supersessions
   honored; §2.4 items flagged, not resolved. The appendix text already encodes this — when
   reviewing the PDF, do not "improve" the wording.
8. **Don't delete scratch.** `5_notes/scratch/` is the only provenance (scripts + logs);
   `git clean -fdX` would destroy it.
9. **Outputs must match the committed script.** If any script edit is made, the outputs must
   be regenerated (a1final was the last full run; the current outputs match the current script).
10. **Commit hygiene.** Nothing committed yet; 3 commit groups proposed in §4; run
    `git status`/`git diff` review first; never commit secrets; keep min_test junk out.

---

## 6. Comprehensive plan (remaining work)

Phase 4 (resume) — Verify the fix + finish:
1. `python main.py --build-pdf --overwrite` (short; poll briefly first).
   - On success: inspect `4_outputs/dissertation.pdf` — appendix G section (title
     "Register Removal: Validation Against Independent Linguistic Register Markers", label
     app:register-validation), both tables, macros resolved; then grep the PDF text for the
     rewired sentences (Methodology ~p.17, Limitations ~p.33, Conclusion) and confirm no
     "unvalidated"/"left to future work" register wording remains (line 471's corpus-scope
     sentence is legitimately unchanged).
   - On failure: capture the error line; re-run the min-test bisection (head + body up to
     `\section{Concept-Retrieval Sensitivity}`), fix the structural cause, rebuild.
2. Delete `3_writing/min_test*.tex` (+ aux/log/out/pdf siblings); `git status` should then
   show only intended files.
3. Optional: gz-fallback exercise (§4.3) and/or commit ground-truth logs under `5_notes/`.
4. Present the implementation report to the user (files added/modified, appendix identifier
   F2/`a1_register_validation`, outputs produced, manuscript locations updated, unresolved
   items intentionally left).
5. Commits (only on explicit user go-ahead), one concern per commit, in this order:
   (a) script + registration + docs; (b) dissertation.tex; (c) outputs + PDF. Push only if
   asked.

Phase 5 (optional, user decision): retire/keep scratch; decide on committing ground-truth
logs (`5_notes/scratch/followup2.log` etc.) per report.md §3.5.

---

## 6. Comprehensive plan — DONE

Steps 1–4 of Phase 4 are **complete** (see §4). Only item 5 (commits, on user go-ahead)
remains. Phase 5 (retire scratch / commit ground-truth logs) is a user decision.

---

## 7. Exactly what was interrupted

The session was interrupted immediately after fixing the LaTeX structural bug, **before
verifying it**. That verification has since been completed successfully (see §4), along with
the min-test cleanup and the gz-fallback exercise. The implementation is complete; only the
commit sequence (§4 item 4) awaits user approval.

The last completed actions, in order:
1. Inserted the new appendix section into `3_writing/dissertation.tex` (python splice at the
   register-check table's end) and applied all manuscript rewires + stale-comment fixes +
   preamble `num_a1` input.
2. Ran `main.py --appendix-a1-register-validation --overwrite` under tmux (a1main) — 61/61
   PASS via the registry path; fingerprint sidecar written.
3. Ran `--build-pdf --overwrite` → FAILED with "Not in outer par mode" at both `\begin{table}`
   in the new section (plus cascade "Float too large by 707pt").
4. Diagnosed via minimal test docs in `3_writing/` (min_test*.tex + pdflatex bisection:
   section-only clean; F+G fails; A..E+G clean; isolated to the F→G boundary).
5. Root cause: the insertion anchor `\end{table}\n\clearpage\n\section{Concept-Retrieval
   Sensitivity}` was replaced by `draft + "\n" + anchor`, which displaced the register-check
   table's `\end{table}` to after the new section — the whole new section was inside the
   register-check `table` environment.
6. **Applied the fix (two edits, §2.5) and confirmed 3 begin/3 end table balance in the
   F..G region. This is the last action; NO rebuild was run after the fix.**

Also in this session (earlier): macro-name digit bug found via the first failed build
("Missing \begin{document}" from `\RegValDraw43`), fixed by renaming to spelled-out names in
the script, re-running the full script (a1final, 61/61), and updating the tex references.

The very next intended action was: `python main.py --build-pdf --overwrite`, then the PDF
inspection + cleanup + commit sequence of §6.

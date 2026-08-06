# Handoff — Editorial / structure pass: TOC, hypotheses placement, appendix fold, macros, n=17 defense, citation grounding

Date: 2026-08-06. Repo root: `/home/manh/dissertation`. Read `AGENTS.md` first; it is authoritative.
Branch: `main`. Remote: `origin` = `https://github.com/qmanhbeo/dissertation-bham.git`. HEAD = `3e91557` (pushed). Working tree: **only `handoff-editorial.md` is modified/uncommitted** (this file).

---

## 1) Context — where we are

The user acts as a critical reviewer (Nature-editor style) over the dissertation
manuscript `3_writing/dissertation.tex` (compiled to the committed artifact
`4_outputs/dissertation.pdf` via `python main.py --build-pdf --overwrite`; 75 pages).

**Three editorial passes are now complete and pushed:**

1. **First pass** (prior session, commits `92ac826`…`db64e99`): abstract/methods prose
   correctness, stale p-value macros, MDE 0.63 grounding, sample-stability ladder
   paper-weighting fix, zero-shot paper-weighting fix, Appendix I.1 unit labels,
   `macro-$F_1$` normalization. Complete; a prior `handoff-editorial.md` documenting it
   has been superseded by this file.
2. **Structure pass** (commits `c68a3c9`→`c4bdf8e`): hypotheses moved to end of
   Literature Review (new §2.6), §3.9 renamed "Coverage–Semantic Interaction",
   Appendices F+G merged, cross-config convergence macros + `$K\to\infty$` framing,
   H1a–H1d bullet list, Appendix I supplementary-analysis sentence, and the **n=17
   finite-population (census) defense** in §3.9.
3. **Citation-grounding pass** (commits `1d3d32b` + `3e91557`, THIS session's work):
   the n=17 census defense is now backed by two verified citations
   (`BerkWesternWeiss1995`, `AbadieAtheyImbensWooldridge2020`), the dense §3.9 paragraph
   was split into five, a wrong cross-reference was fixed, and §2.6's forward reference
   was aligned with the new §3.9 framing. Details in §3.

**Where we are now:** the manuscript is structurally sound and internally consistent on
the n=17 issue; the open items are the remaining grader eyebrows (§4), all requiring
user decisions rather than mechanical fixes.

---

## 2) Key known facts (so a fresh agent does not re-derive)

### 2.1 Manuscript structure (current line numbers, `dissertation.tex`)
- `\section{Introduction}` L105 · `\section{Literature Review}` L119 ·
  `\section{Methodology}` L183 · `\section{Results}` L323 · `\section{Discussion}` **L456** ·
  `\section{Conclusion}` L509 · `\appendix` L530.
- **§2.6 `\subsection{Research Questions and Hypotheses}`** **L171**, label `sec:hypotheses`.
  H1 → **H1a–H1d bullet list**. L180 states the operationalization is described in §3.9
  "including the $n = 17$ scope conditions" (recently aligned wording; **do not revert to
  "power limitation stated as a scope condition"** — §3.9 no longer frames it that way).
- **§3.9 `\subsection{Coverage–Semantic Interaction}`** **L299**, label
  `sec:coverage-semantic-interaction`. **Now FIVE paragraphs** (split 2026-08-06):
  1. Operationalization (Pearson/Spearman over 17 SDGs; SDG~4 separate);
  2. **Census defense**: fully enumerated population; associations are properties of the
     framework as a whole; "sampling-based power reasoning does not directly apply"
     `\parencite{BerkWesternWeiss1995, AbadieAtheyImbensWooldridge2020}`;
  3. Robustness is the operative concern: **leave-one-out check (Section 4.3, `sec:interaction`)**
     + cross-configuration checks (**Section 4.4, `sec:robustness`**);
  4. Finer-grained units / 169 targets / no target-level labels → future work;
  5. Supplementary pooled regression: **all 24 encoder--retrieval--method--cap configurations**,
     Appendix I (`app:regression`), future-work framing.
- **§4.3 `\subsection{Coverage–Semantic Interaction: A Possible Cancellation}`** **L403**,
  label `sec:interaction`. **Still hedged ("Possible") — open decision (see §4.1).**
  Contains the only leave-one-out check in the document (SDG-4-removed, L426) and the
  MDE 0.63 footnote (L418).
- **§4.4 `\subsection{Robustness of the Gap Rankings}`** **L439**, label `sec:robustness`
  (cross-sensitivity, encoder, assignment-method, distance-functional, synthesis; pooled
  OLS sentence at L453).
- Discussion L456 (population parenthetical at L459); Conclusion L509.

### 2.2 Appendix letters (A→J, post-F+G-fold — never hardcode letters, use `\ref`)
| Letter | Section (label) | Line |
|---|---|---|
| A | Supplementary Methodology (`app:supp-methodology`) | 531 |
| B | Diagnostics for Reference Classifier | 553 |
| C | Supplementary Robustness and Sensitivity Checks | 591 |
| D | Sample-Stability Robustness Check | 633 |
| E | Model Selection: Grid Search Protocol and Rationale | 657 |
| **F** | **Register Removal: Convergence and Validation** (`app:register-removal`) ← merged F+G | 705 |
| G | Concept-Retrieval Sensitivity (`app:concept-retrieval-sensitivity`) | 805 |
| H | Supplementary Cross-Method Data (`app:cross-method-section`) | 820 |
| **I** | **Pooled Regression: Coverage Predictors and the Semantic Gap** (`app:regression`) | 948 |
| J | Declaration of AI Use | 962 |

Preserved labels inside merged Appendix F: `app:register-robustness` (on `\subsection{Convergence}`),
`app:register-inlp` (on "Convergence results." paragraph), `app:register-iterative`,
`app:register-validation` (on Construct Validation section), `sec:regval-*` (seven subsections).
Other key labels: `app:sdg4-lexical-audit` (B.1), `app:balanced-subset-stability` (C.1),
`app:raw-value-correlation`, `app:assignment-method-comparison`, `app:centroid-similarity`,
`app:semantic-gap-text-interpretability`, `app:policy-source-data`, `sec:distributional-robustness`,
`sec:regval-*`, `sec:robustness` (4.4), `sec:interaction` (4.3).

### 2.3 Canonical numbers (all committed macros; do not re-derive)
- Research abstracts = 2,536,771 (`\NResearchAbstracts`); research **segments** = 3,105,144
  (`\NResearchSegments`); policy segments = 40,597; policy source docs = 6,367.
  **3.1M is segments, not papers** — never call it papers. `\NResearchPapers` macro deleted.
- LR test macro-F1 = 0.816 (`\MacroFOne`); reference pool = 62,173 (`\NReferencePool`).
- Headline correlations (MPNet paper-weighted, `num5_register_decomposition.tex`):
  raw ρ = −0.012 (p=0.963); adjusted ρ = +0.544 (p=0.024); register ρ = −0.390 (p=0.122).
- MDE 0.63 (`\HPrimaryMinDetectableR`) at n=17, Pearson, 80% power — still used in Results
  4.3 L418 footnote and Discussion L459; **deliberately left in place** (see §4.3).

### 2.4 The n=17 census defense and its citations (VERIFIED — full texts read)
- §3.9 paragraph 2 claim: the 17 SDGs are the complete SDG framework set (not a sample),
  tests run over a fully enumerated population, associations are properties of the
  framework as a whole, sampling-based power reasoning does not directly apply.
- **`BerkWesternWeiss1995`** — Berk, R. A., Western, B., & Weiss, R. E. (1995). Statistical
  inference for apparent populations. *Sociological Methodology, 25,* 421–458.
  DOI `10.2307/271073`. Verified verbatim (extraction `Berk.txt`): "apparent populations"
  defined as "the data on hand are all the data there are. No additional data could be
  collected, even in principle" (L47–52); examples ~18 OECD countries (L86–92), all 157
  SMSAs (L95–103); fixed-data stance → "conventional statistical inference becomes
  irrelevant ... descriptive statistics are employed" (L134–136). **Caveat:** the paper
  conditions this on a *deterministic* data-generating view (L260–263) and warns the
  super-population stance is the legitimate alternative (L130–143); it recommends a
  Bayesian middle path (L144–149). Our text's hedged "does not directly apply" is the
  right calibration — do not strengthen to "does not apply".
- **`AbadieAtheyImbensWooldridge2020`** — Abadie, A., Athey, S., Imbens, G. W., &
  Wooldridge, J. M. (2020). Sampling-based versus design-based uncertainty in regression
  analysis. *Econometrica, 88*(1), 265–296. DOI `10.3982/ECTA12675`. Verified verbatim
  (extraction `Abadie.txt`): opens "data for all 50 states ... all visits to a website"
  (L19–20); quotes Manski & Pepper (2018): "Random sampling assumptions ... are not
  natural when considering states or counties as units of observation" (L63–65);
  descriptive vs causal estimands (L130–135); entire-population case (L162–170);
  **Comment 12, ρ=1**: "a descriptive perspective would suggest the standard errors
  should be zero" (L952–958); Comment 2: sampling variance → ~0 as sampling rate → 1
  (L342–347). **Caveat:** the SE=0 claim applies to *descriptive* estimands; causal
  estimands still carry design-based uncertainty. Our correlations are descriptive
  associations over the enumerated framework, so the citation is placed on the
  descriptive claim only.
- Full-text PDFs archived (committed): `0_literature/statistical_methods/BerkWesternWeiss1995.pdf`
  (10 pp) and `AbadieAtheyImbensWooldridge2020.pdf` (15 pp). Verification `.txt`
  extractions were created locally then deleted; re-verify with `pdftotext -layout` if needed.
- Bib entries appended to `3_writing/references.bib` in repo style (author/title/journal/
  volume/number/pages/year/doi), after `Beltagy2019`.

### 2.5 INLP register convergence (cross-config macros)
- Checkpoints: `2_data/3b_register/mpnet/canon/checkpoint.json`,
  `2_data/3b_register/minilm/subset/checkpoint.json`,
  `2_data/3b_register/scibert/subset/checkpoint.json` (iterations, n_iters, final_acc,
  stopped_reason).
- Convergence: MPNet 62 iters (first-iter acc 0.978, final 0.498, acc@15 0.685);
  MiniLM 40 iters (0.965 / 0.498 / 0.584); SciBERT 79 iters (0.973 / 0.491 / 0.539).
  All stop at the ≤0.5 threshold (`ITERATIVE_ACC_THRESHOLD`).
- Macro file: `4_outputs/mpnet/tables/num12b_register_cross_config.tex` (12 macros,
  `\InputIfFileExists`'d near `dissertation.tex` L53). Macro names **cannot contain
  digits** (TeX gotcha already hit: `At15` broke the build → `AtFifteen`).
- Methodology states $K \to \infty$ (no fixed bound); `ITERATIVE_MAX_K = 200` in
  `register_adjust.py` is documented as a safety cap only.

### 2.6 Build / verify workflow
- `python main.py --build-pdf --overwrite` (bash required; writes committed
  `4_outputs/dissertation.pdf`). Check `3_writing/artifact/dissertation.log` for
  `Undefined control sequence` / undefined refs; render-check with
  `pdftotext 4_outputs/dissertation.pdf - | grep ...`.
- Regenerate cross-config macros cheaply (no pipeline re-run):
  `python 1_code/7_main_analysis/2_appendix/f2_export_register_cross_config.py --output-dir 4_outputs --overwrite`.
- Warm/cold replay regenerate them automatically (wired in `main.py`).

---

## 3) Actions / decisions made + files changed, and why

### 3.1 Structure pass (commits `c68a3c9`→`c4bdf8e`) — summary for continuity
| Commit | What | Why |
|---|---|---|
| `c68a3c9` | §2.6 "Research Questions and Hypotheses" added at end of LR (after Research Gap, before Methodology); H1/H1a–H1d moved there; former §3.9 renamed "Hypothesis and Analysis Plan: …" → "Coverage–Semantic Interaction" and rewritten as the operationalization with forward link to §2.6. | Grader: hypotheses must precede the methods that test them. |
| `2ed23d8` | **Merged Appendices F+G** into one `\section{Register Removal: Convergence and Validation}`; preserved all labels; dropped over-promising "Cross-Config Replication" subtitle; added true cross-config convergence paragraph. | Grader: two appendices defending one method step. Zero content cut. |
| `4810c25` | **Macros instead of hardcoded values**; new script `f2_export_register_cross_config.py`; wired into `main.py` warm+cold replay; Methodology $K\to\infty$; comment at `ITERATIVE_MAX_K`. | No magic numbers (AGENTS.md). |
| `0c88295` | §2.6 H1a–H1d as bullet list. | User request. |
| `1cd0936` | §3.9 sentence referencing Appendix I (Pooled Regression) as supplementary analysis / future work. **No em dashes** (user preference). | User request. |
| `c4bdf8e` | §3.9: replaced "limited power" caveat with **finite-population (census) defense**; Discussion L459 light parenthetical back-referencing §3.9. **Results 4.3 untouched** (user decision). | User asked whether the n=17 population defense is statistically sound — yes, as a finite-population argument with measurement-error and framework-bound limits. |

### 3.2 Citation-grounding pass (THIS session; commits `1d3d32b`, `3e91557`)
| Commit | What | Why |
|---|---|---|
| `1d3d32b` | **§3.9 split into five single-topic paragraphs.** | User: "split that paragraph up. It's too dense now." |
| `1d3d32b` | **Two citations added** to §3.9 paragraph 2: `\parencite{BerkWesternWeiss1995, AbadieAtheyImbensWooldridge2020}` after "sampling-based power reasoning does not directly apply". Bib entries added to `references.bib`. | Ground the census defense. Both citations verified 100% against full texts (see §2.4): subagents read the pdftotext extractions and confirmed every claim with verbatim quotes + line numbers. |
| `1d3d32b` | **Cross-reference fix**: §3.9 previously pointed "leave-one-out and cross-configuration checks" at Section 4.4 (`sec:robustness`); the only leave-one-out check (SDG-4 removed) lives in **Section 4.3** (`sec:interaction`, L426). Now split: "leave-one-out check (Section 4.3) and the cross-configuration checks (Section 4.4)". | Grounding: reference must point where the check actually is. |
| `1d3d32b` | **Appendix description fix**: §3.9 said "combines all encoder--classifier configurations"; Appendix I (L943) and §4.4 (L453) say **"all 24 encoder--retrieval--method--cap configurations"**. Aligned §3.9 to the appendix's own wording. | Grounding: match the analysis actually reported. |
| `1d3d32b` | **§2.6 L180 wording fix**: "with the $n = 17$ power limitation stated as a scope condition" → "including the $n = 17$ scope conditions". | §2.6 described §3.9's content; §3.9 no longer frames n=17 as a *power limitation* (it argues power reasoning doesn't directly apply). Forward reference now matches. |
| `3e91557` | **Archived the two full-text PDFs** under `0_literature/statistical_methods/`. | Repo convention: literature PDFs live in `0_literature/`; provenance for the citations. |

**Files changed overall this session:** `3_writing/dissertation.tex`, `3_writing/references.bib`,
regenerated `4_outputs/dissertation.pdf`, `0_literature/statistical_methods/` (2 new PDFs).
Plus `handoff-editorial.md` (this file, uncommitted).

---

## 4) What remains and why

1. **§4.3 heading "…: A Possible Cancellation" (L403).** Grader flagged the hedged
   "Possible". Remains because the register interpretation is deliberately bounded
   ("partial, not complete, empirical support"). **Open decision:** keep the hedge, or
   retitle (e.g., "…: An Apparent Cancellation" / drop "Possible" and let Discussion carry
   the bound). Not actioned without user input.
2. **Appendix/main-body scale imbalance (~75 pp appendix vs ~34 pp main).** Flagged by the
   grader; **not actioned**. Merging F+G removed one section but not the size imbalance.
   Candidate follow-ups (each needs a user decision): tighten redundant appendix prose, or
   restructure what is "Supplementary". Do **not** cut validation content.
3. **Results 4.3 MDE/power passage (L418 footnote, L417 text)** still uses classical power
   framing ("can reliably detect only very large correlations … MDE 0.63 at 80% power"),
   technically in tension with the §3.9 census defense. **Deliberately left** per the
   user's decision ("3.9 for the full defense; Discussion only a light phrase"). Revisit
   only if the user changes their mind. §2.6 has been aligned; 4.3 has not.
4. **`handoff-editorial.md` uncommitted** — this file. Commit it (or leave as the user
   prefers) before handing off.
5. **Appendix-letter drift risk.** Post-fold letters A→J (see §2.2). All `\ref`s use
   labels so nothing breaks; any future prose hardcoding "Appendix H/I/J/K" must be
   re-checked.
6. **MiniLM/SciBERT run on the S1 100k subset, MPNet on the full corpus** — by design
   (encoder-sensitivity setup, AGENTS.md). The cross-config macros compare full-corpus
   (MPNet) with subset (MiniLM/SciBERT) convergence; Appendix F states this. Keep it honest.

---

## 5) Concerns to emphasize

1. **The n=17 defense is only sound as a finite-population argument with its limits
   stated.** §3.9 does state them (robustness is the operative concern; claim is
   framework-bound; 169 targets lack labels). Do NOT strengthen "does not directly apply"
   to "does not apply" — Berk et al. explicitly require a deterministic commitment, and
   Abadie et al. preserve design-based uncertainty for causal estimands. Our correlations
   are descriptive; that is why the citation placement is correct.
2. **Berk et al. expose the counter-argument.** Citing them makes the census stance
   visible; a sharp examiner can invoke the super-population reading (data as a realization
   of a larger process → power analysis *would* apply). This is the standard framing choice
   for framework-exhaustive unit sets; do not be surprised if it comes up. The §3.9 text
   already implicitly commits to the census stance.
3. **No em dashes (`---`)** in newly edited prose (user preference). Note §2.6 L180 and
   other pre-existing sentences still contain `---` from earlier passes — only new edits
   must avoid them.
4. **TeX macro names cannot contain digits** — keep `AtFifteen`-style word forms.
5. **Never hardcode the cross-config numbers** (62/40/79, 0.965/0.978, etc.) — they live in
   `num12b_register_cross_config.tex`, regenerated from checkpoints by the export script.
6. **Fingerprint/staleness:** the export script fingerprints the 3 checkpoints; `main.py`
   wiring runs it at the end of warm/cold replay. Do not add mtime-based fingerprinting
   (`2_data/` re-hydration resets mtimes).
7. **Verification is mandatory** (AGENTS.md): after any tex edit, rebuild
   (`python main.py --build-pdf --overwrite`), grep the log for `Undefined control
   sequence` / `Rerun`, sanity-check the PDF with `pdftotext`. A digit-in-macro bug once
   slipped past the first build.
8. **Results 4.3 MDE 0.63 vs §3.9 census defense tension is deliberate and
   user-approved.** If a grader pushes, the agreed resolution is: full defense in §3.9,
   light pointer in Discussion, 4.3 untouched. Do not unilaterally rewrite 4.3.
9. **Appendix I regression is cited from multiple places** (Discussion, §3.9, §4.4
   synthesis, and its own body). Keep the framing consistent: *supplementary / future
   work*, not a primary result. §3.9 now matches the "24 encoder--retrieval--method--cap
   configurations" wording used elsewhere.
10. **`handoff.md` at repo root is stale** (from an earlier pass). This file
    (`handoff-editorial.md`) is the current handoff. Commit/retire `handoff.md` if the
    user wants a clean root.

---

## 6) The whole comprehensive plan

### 6.1 Complete and pushed
1. Hypotheses → end of Literature Review (§2.6) + §3.9 renamed "Coverage–Semantic Interaction" — `c68a3c9`.
2. Appendices F+G folded into one "Register Removal: Convergence and Validation" — `2ed23d8`.
3. Cross-config convergence macros + `$K\to\infty$` framing + `ITERATIVE_MAX_K` comment — `4810c25`.
4. H1a–H1d as bullet list in §2.6 — `0c88295`.
5. §3.9 sentence referring to Appendix I (Pooled Regression) as supplementary/future work — `1cd0936`.
6. §3.9 finite-population n=17 defense + Discussion light pointer (Results 4.3 untouched) — `c4bdf8e`.
7. §3.9 split into five paragraphs; census defense cited with two full-text-verified
   references; leave-one-out cross-ref corrected to §4.3; pooled-regression description
   aligned to 24 configs; §2.6 forward reference aligned — `1d3d32b`.
8. Full-text PDFs archived in `0_literature/statistical_methods/` — `3e91557`.
9. Earlier pass (abstract/methods/ladder/ZS/I.1) — commits `92ac826`…`db64e99` (do not re-run).

### 6.2 Open / future (needs user decisions)
1. §4.3 heading hedge ("Possible Cancellation") — keep vs retitle.
2. Appendix/main scale imbalance — decide whether and how to trim.
3. Results 4.3 vs §3.9 power/population framing — revisit only if the user wants it.
4. Commit this handoff file; optionally retire stale `handoff.md`.

### 6.3 If the pass continues (typical next steps)
- Re-read §3.9 (now 5 paragraphs) and §4.3 side by side; confirm the census-defense
  wording reads naturally and does not oversell.
- Check merged Appendix F TOC: one `\section`, nine `\subsection`s. No orphan labels.
- Rebuild → verify log → commit **one concern per commit**, then push.

---

## 7) Exactly what was interrupted

**Nothing was interrupted.** The last task (citation-grounding pass: §3.9 split, two
verified citations, cross-ref fix, appendix-description fix, §2.6 alignment; commits
`1d3d32b` + `3e91557`) completed cleanly: build exit 0, log clean, render verified,
pushed to `origin/main` (`c4bdf8e..3e91557`). The session stopped because the user
asked for this handoff file, not because of a failure or timeout. Working tree is clean
except for this uncommitted `handoff-editorial.md`. To resume, a fresh agent only needs
this file + `AGENTS.md`; the §4 open items are the natural next decisions.

# Verification Report: INLP "Register Removal" Validation Diagnostic

**Date of audit:** 2026-08-05
**Auditor:** opencode verification pass (read-only; no code/data/manuscript touched)
**Repo:** `/home/manh/dissertation`, branch `main`, HEAD `1a2f97a` (== `origin/main`)

This report is ground truth for the promotion of the register-validation diagnostic
into a durable appendix stage. It inventories what exists, fixes which numbers are
final, verifies reproducibility from disk, quotes the current manuscript wording,
documents the appendix-registration pattern, and states git status precisely.

---

## 1. Inventory of artifacts (register validation / INLP / regcheck)

### 1.1 Committed (tracked in git)

| Artifact | Commit | Date | What it is |
|---|---|---|---|
| `5_notes/register_validation_report.md` | `0f96a3f` | 2026-08-05 08:57 | **Report 1** — go/no-go check of the INLP register interpretation (MPNet canon, n=408, seed 42). Steps 2b/2c/2d + Step 3 (17-way SDG selectivity). Verdict: GO with caveats (2c "red flag", 6-feature operationalization). 123 lines. |
| `5_notes/register_validation_followup.md` | `c2773a9` | 2026-08-05 09:39 | **Follow-up 1** — Items 1–3: (1) Concept-row provenance = NOT a bug (same embedder/space), (2) original 2b/2c signals = clustering artifacts (SDSN/UNDP mega-docs), one-per-parent rerun, (3) Step-2c decomposition (3a per-SDG, 3b per-feature, 3c own/other-dist, 3d renorm check). Verdict: GO with qualification. 190 lines. |
| `5_notes/register_validation_followup2.md` | `1a2f97a` | 2026-08-05 10:34 | **Follow-up 2 — FINAL / most-corrected.** Item 1 sample-construction audit (REBUILD, not subset); Item 2 accuracy CIs (Wilson + binomial + prediction bootstrap; mega-doc exclusion test); Item 3 per-SDG / mega-exclusion / draw-stability of policy other-dist −0.197 → downgraded to noise. Verdict: GO stands with framing change. 175 lines. |
| `5_notes/handoff_register_validation_2026-08-05.md` | `26455a5` | 2026-08-05 09:35 | Handoff for Report 1 + comprehensive plan (Phases 0–5). |
| `5_notes/handoff_register_validation_followup2_2026-08-05.md` | `1a2f97a` | 2026-08-05 10:34 | Handoff for Follow-up 2; documents the RNG fix and acceptance gate. |
| `handoff.md` (repo root) | `1a2f97a` (rewritten) | 2026-08-05 | Working handoff file; content == followup-2 handoff. |

The three report files in `5_notes/` are **byte-identical** to the copies in
`5_notes/scratch/` (verified with `diff` on all three; see §1.2).

### 1.2 Gitignored / untracked (all under `5_notes/scratch/`, ignored by `.gitignore:90`)

Scripts:

| Artifact | Date | What it is |
|---|---|---|
| `register_validation_check.py` (18.9 KB) | 2026-08-05 00:35 | Original go/no-go script (Report 1). Runs in ~2–3 min; writes `regcheck_*.npy/npz`. |
| `register_validation_followup.py` (19.7 KB) | 2026-08-05 09:19 | Follow-up-1 script (Items 2–3). Writes `regcheck_followup_arrays.npz`. |
| `register_validation_followup2.py` (24.4 KB) | 2026-08-05 10:16 | Follow-up-2 script, **RNG-fixed** (single module-level `_rng`, three successive draws; fresh draws at seeds 43/44/45). This is the current/final version (see §3). |
| `followup2_replacements.py` (6.6 KB) | 2026-08-05 10:27 | Item-1 replacement-source audit (cheap, no embeddings). Prints to stdout. |
| `check_concept_same_space.py` (3.8 KB) | 2026-08-05 09:11 | Item-1 same-space proof (Follow-up 1). **Hardcodes `ROOT = Path('/home/manh/dissertation')` at line 11.** |

Logs / data artifacts:

| Artifact | Date | What it is |
|---|---|---|
| `regcheck_full.log` | 2026-08-05 00:39 | Ground-truth log of Report 1 (complete run). |
| `regcheck_arrays.npz` | 2026-08-05 00:39 | Report-1 saved arrays (F, dist_raw/adj, reg_score, etc.). |
| `regcheck_X_adj.npy` / `regcheck_removed_norm.npy` / `regcheck_reg_score.npy` | 2026-08-05 00:39 | Report-1 per-unit vectors. |
| `regcheck_followup.log` | 2026-08-05 09:24 | **Ground-truth log of Follow-up 1** (the "acceptance gate" target). |
| `regcheck_followup_arrays.npz` | 2026-08-05 09:24 | Follow-up-1 arrays (reg, dist_raw/adj/noren, corr, sdg, F, resid_norm). |
| `regcheck_followup.DONE` | 2026-08-05 09:24 | tmux completion marker. |
| `followup2.log` | 2026-08-05 10:24 | Follow-up-2 acceptance-gated log (see §3 — verified to be from the current script). |
| `followup2.DONE` | 2026-08-05 10:24 | tmux completion marker. |
| `followup2_replacements.txt` | 2026-08-05 10:27 | Replacement-source audit output (reproduced; see §3). |
| `register_validation_report.md` / `register_validation_followup.md` / `register_validation_followup2.md` | 00:42 / 09:30 / 10:31 | Scratch copies of the reports — identical to the committed `5_notes/` versions. |
| `__pycache__/` | — | Compiled bytecode. |

Unrelated scratch (not part of this work, do not confuse): `build_3a.log`,
`verify_a3_canonical.log`, `verify_a3_fallback.log`, `verify_b2_fallback.log`,
`embed_stage.log`, `baseline_cmdlists.py/.txt`.

**Git history context (pipeline/manuscript, NOT the diagnostic line):** the
manuscript's INLP register-adjustment stage lives in `1_code/7_main_analysis/0_shared/register_adjust.py`
(+ `register_utils.py`, `g_register_decomposition.py`) and was developed in
commits such as `82a269e`, `3930dcb`, `203395f`, etc. Those are the register
*adjustment* engine the diagnostic validates; the diagnostic scripts themselves
are entirely in scratch.

---

## 2. Source of truth — final numbers per result

### 2.1 Which file is the FINAL, most-corrected version

| Stage | Final file | Superseded by |
|---|---|---|
| Go/no-go (Report 1) | `5_notes/register_validation_report.md` (`0f96a3f`) | Partially superseded by Follow-up 1 (2c red flag) and Follow-up 2 ("adj≈chance"). |
| Follow-up 1 (Items 1–3) | `5_notes/register_validation_followup.md` (`c2773a9`) | Item-3 "surviving trace" reading superseded by Follow-up 2 (downgraded to noise). |
| Follow-up 2 (sample audit, CIs, other-dist) | **`5_notes/register_validation_followup2.md` (`1a2f97a`)** — THE final, most-corrected state of the whole line | — |

The scratch copies are byte-identical to the committed reports (verified).
There is **no other candidate "final" file** — the intermediate RNG-bug draft of
`followup2.md` is **not on disk** (it was overwritten by the rewrite at 10:31 and
is only referenced in the RNG-fix note, `followup2.md:7`).

### 2.2 Reconciled table of final correct numbers

All numbers verified against `regcheck_followup.log` / `followup2.log` / `regcheck_full.log`
and — for the whole Follow-up-2 set — against a fresh end-to-end re-run (see §3).

| Result | Final number | Source (report:line) |
|---|---|---|
| **2b** original 2A, reg~‖x−x′‖: pooled / within-res / within-pol | 0.102 (p=0.040) / 0.212 (p=0.002) / 0.191 (p=0.006) | Report1 §2b; Followup1 2A table (log line 30–32) |
| **2b** one-per-parent 2B: pooled / within-res / within-pol | 0.092 (p=0.063, ns) / −0.043 (p=0.545) / −0.036 (p=0.606) | Followup1 2B table (log line 38–40) |
| **2c** original 2A: RAW→ADJ | 0.126 (p=0.011) → 0.247 (p=4.4e-07); partial 0.130→0.253 | Report1 §2c; Followup1 2A table |
| **2c** one-per-parent 2B: RAW→ADJ | −0.212 (p=1.6e-05) → −0.197 (p=6.1e-05); partial −0.155→−0.159 | Followup1 2B table (log line 41–42) |
| **2c/3d** Item-3 draw-3 pooled: RAW / ADJ | −0.088 (p=0.075) / −0.074 (p=0.134) | Followup1 §3d; Followup2 repro gate (log line 85) |
| **2d** original 2A CV acc: register-only / raw / adj | 0.456 (CI [0.408,0.504], p=0.967) / 0.909 (CI [0.878,0.934]) / **0.505 (CI [0.457,0.553], p=0.441, ns)** | Followup2 Item 2 table (log line 42–44) |
| **2d** one-per-parent 2B: register-only / raw / adj | 0.544 (CI [0.496,0.592], p=0.042) / 0.944 (CI [0.917,0.962]) / **0.603 (CI [0.555,0.649], p=1.9e-05, sig)** | Followup2 Item 2 table (log line 48–50) |
| **2d** adj-accuracy rise 0.505→0.603 | diff **+0.098**, 95% CI [+0.024, +0.169], p(diff>0)=0.994 | Followup2 Item 2 (log line 54) |
| **2d** mega-policy exclusion on original (drop 25 units) | adj acc 0.505 → **0.574** (220/383, CI [0.524,0.623], p=0.002) — +0.070 of the +0.098 rise | Followup2 Item 2 (log line 75–77) |
| **Item 3** policy reg~other-dist ADJ pooled (Item-3 draw 3) | **−0.197 (p=0.0047)** — reproduces exactly | Followup1 §3c (`regcheck_followup.log:96`); Followup2 repro gate (log line 86) |
| **Item 3** per-SDG (n=12 policy each) | **0/17** significant; 14 neg / 3 pos (SDG 3, 7, 10) | Followup2 Item 3 (log line 88–108) |
| **Item 3** mega-doc exclusion (drop ALL mega-policy segs): draw1 / draw2 / draw3 | +0.127 (p=0.091) / −0.138 (p=0.053) / **−0.213 (p=0.003)** | Followup2 Item 3 (log line 111–119) |
| **Item 3** draw stability (fresh draws) | seed 42 −0.197 (p=0.005) / 43 −0.130 (p=0.063) / 44 −0.004 (p=0.949) / **45 +0.126 (p=0.072)** → sign flips, mean ≈ −0.05 | Followup2 Item 3 (log line 122–124) |
| **Item 1** (Followup-1) concept same-space | 30,545 shared papers; 40 sampled; **44 byte-identical pairs; max \|diff\| 0.000183; min cos 0.99999952** → NOT a bug | Followup1 Item 1 (reproduced this audit, §3) |
| **Item 1** (Followup-2) sample construction | 2A n=408, 390 distinct parents, 25 multi-parent units (6.1%), policy 186 docs; 2B n=408, 408 distinct parents, 0 multi-parent, 12/SDG/corpus → **rebuild with refill, not subset** | Followup2 Item 1 (log line 5–31) |
| **Item 1** replacement sources | mega-docs 7 docs (SDSN×2, UNDP, WHO, EU, UN×2), 25 segs in 15/17 SDGs; replacements overwhelmingly `pol_sdgi_*` (~4,225 docs) + `pol_ungdc_*` (~2,048 docs); mean_sent_len 57.8 vs 30.8 vs 25.2 | Followup2 Item 1 + `followup2_replacements.txt` |
| **Step 3** (Report 1, not re-run in followups) | raw LR 0.691 / kNN 0.554 → adj LR 0.672 / kNN 0.578 (chance 0.059) | Report1 §3 (`regcheck_full.log:62–63`) |

### 2.3 Superseded / corrected numbers (old → new)

| Claim | Old value/statement (report) | Final value/statement (report) |
|---|---|---|
| "Adjusted-space corpus accuracy ≈ chance" | 0.505 ≈ chance (Report1 §2d/verdict; also Followup1 §2d "collapse large but not to chance" was about one-per-parent) | **0.505 holds only for the mega-contaminated original (p=0.44, ns). Primary one-per-parent sample: 0.603 (p=1.9e-05), significantly above chance.** General "≈chance" characterization REVISED (Followup2 Item 2 + verdict). |
| Step-2c "red flag" (register→farther from centroid, rises 0.126→0.247) | Real residual-register red flag (Report1 §2c) | **Clustering artifact** from SDSN/UNDP mega-docs; flips negative on the clean one-per-parent sample (−0.212 RAW / −0.197 ADJ on 2B; −0.088/−0.074 pooled on Item-3) (Followup1 Item 2; Followup2 confirms reproduction). |
| Policy other-dist −0.197 | "The only surviving within-SDG register trace … not a cross-corpus red flag" (Followup1 §3c:146) | **Downgraded to noise / sample-specific**: reproduces and survives mega-exclusion (−0.213) but sign-flips across fresh draws 43/44/45; mean ≈ −0.05 (Followup2 Item 3). Do not carry to the appendix. |
| Replacement sources | "non-systematic" (attributed to prior follow-up in Followup2's table; see §2.4 flag 3) | Systematically national `pol_sdgi_*`/`pol_ungdc_*` monitoring reports (short-sentence) vs global flagship mega-docs (Followup2 Item 1). |
| CI method (Followup-2 early draft) | Resample-then-CV bootstrap (leaked train/test via duplicate rows → inflated CIs) | Replaced by pooled 5-fold predictions + Wilson CI + one-sided binomial + prediction-level bootstrap diff (Followup2 Item 2, handoff §3.3). |
| 2B/Item-3 sample numbers (Followup-2 early draft) | Wrong samples due to per-call `_rng` re-seeding (RNG bug) | Single module-level seed-42 stream; acceptance gate reproduced exactly (Followup2 line 7, handoff §3.2). **The intermediate draft's wrong numbers are not on disk and cannot be tabulated.** |

### 2.4 Unresolved / flagged items (do NOT resolve here — human decision)

1. **Register-score operationalization is still undecided.** Report1 caveat #5
   (`register_validation_report.md:112`) states an a-priori "institutional" z-sum
   gives null 2b (ρ=0.007) and reversed 2c (−0.24), vs PC1 used throughout.
   Follow-up 1 and 2 both kept PC1 and never re-tested or reconciled the z-sum
   variant. If the appendix claims a specific operationalization, this needs a
   decision first (pre-registration concern, also raised in handoff concern #3).
2. **Two different statistics both round to −0.197 in Follow-up 1** — potential
   confusion source, not a contradiction:
   - `regcheck_followup.log:41`: **2B pooled** reg~pooled-SDG-centroid **ADJ ρ=−0.197 (p=6.1e-05)** (draw-2 sample);
   - `regcheck_followup.log:96`: **Item-3 policy** reg~other-dist **ADJ ρ=−0.197 (p=0.0047)** (draw-3 sample).
   They are different quantities that coincide at 3 decimals. Follow-up 2's "matches
   prior report" refers to the **second** (Item-3 policy other-dist). The 2B pooled
   2c value is **not** re-printed by `register_validation_followup2.py` (the script
   only prints the Item-3 gate), so its reproduction is implicit (same deterministic
   stream), not explicitly logged by the followup-2 run.
3. **Attribution oddity in Followup2's reconciliation table.** `register_validation_followup2.md:159`
   lists a prior claim "Replacement sources are non-systematic" as REVISED, but no
   literal sentence in `register_validation_followup.md` or `register_validation_report.md`
   makes that claim; Followup1 only said replacements are "drawn from the remainder
   of each SDG's policy pool." Minor, but the "prior claim" appears to be an
   assumption rather than a documented finding.
4. **The RNG-bug intermediate draft is unrecoverable.** No old→new tabulation of
   its numbers is possible; only the fact of the bug and its fix is recorded
   (Followup2 line 7; handoff §3.2). This is by design (draft overwritten), not a risk.

No other contradictions were found: Report1 ↔ Followup1 ↔ Followup2 numbers agree
once the three supersessions in §2.3 (2c artifact, "adj≈chance", other-dist noise)
are applied. Every number in §2.2 was cross-checked against at least one on-disk log.

---

## 3. Reproducibility verification (from what is on disk NOW)

### 3.1 Environment

- Python 3.11.15 at `/home/manh/miniforge3/envs/dissertation/bin/python` (conda env
  `dissertation`; `source activate` is broken on this box — use the absolute path).
- `numpy`, `scipy`, `sklearn`, `nltk` all import OK in that env.
- nltk data present: `tokenizers/punkt` and `taggers/averaged_perceptron_tagger_eng`.
  **A clean env rebuild will NOT have the POS tagger** (`averaged_perceptron_tagger_eng`)
  — it was installed ad hoc this session (handoff §2.5). `punkt` was already present.
  A promoted appendix must add the tagger to `environment.yml` or the run fails on the
  passive-voice feature.
- All data dependencies exist on disk: `2_data/3b_register/mpnet/canon/G.npy`,
  `2_data/2_segmented/research/part-*.jsonl`, `2_data/3_embedded/mpnet/research_shards/part-*.npy`,
  `2_data/5_supervised_scored/mpnet/paper_scores_shards/metadata/part-*_ids.jsonl`,
  `2_data/2_segmented/policy.jsonl`, `2_data/3_embedded/mpnet/policy.npy`,
  `2_data/5_supervised_scored/mpnet/policy_scores.npy`,
  `2_data/3_embedded/mpnet/metadata/policy_ids.json`, and the `research_concept` track
  files used by `check_concept_same_space.py`.

### 3.2 Syntax / import status (all 5 scripts)

`py_compile` passes on all of: `register_validation_check.py`,
`register_validation_followup.py`, `register_validation_followup2.py`,
`followup2_replacements.py`, `check_concept_same_space.py`. `register_utils` imports
correctly via the scripts' `sys.path` insert.

### 3.3 Full runs actually executed this audit

- **`register_validation_followup2.py` — RAN END-TO-END (fresh, under tmux, ~9 min).**
  Output is **byte-identical to `5_notes/scratch/followup2.log`** after trailing
  whitespace normalization. Every acceptance-gate number reproduced exactly:
  2A 0.456/0.909/0.505, 2B 0.544/0.944/0.603, Item-3 pooled −0.088/−0.074, policy
  other-dist −0.197 (p=0.00468), per-SDG table, mega-exclusion rows (−0.213),
  draw-stability seeds 43/44/45, mega-feature table, bootstrap diff +0.098, Wilson CIs.
  **This confirms the seed-42 numbers in the final report are reproducible from the
  CURRENT (RNG-fixed) script on disk.** It also confirms `followup2.log` was produced
  by the current script: script mtime 10:16 < log mtime 10:24, no edits since, and the
  fresh run is identical.
- **`followup2_replacements.py` — RAN.** Matches `followup2_replacements.txt`
  except a cosmetic listing-order difference of two equal-count (2×) mega-docs in the
  "Mega-doc sources" block (Python set-iteration/hash order is process-randomised;
  counts and all per-SDG replacement lists identical). Deterministic content.
- **`check_concept_same_space.py` — RAN.** Reproduces exactly: 30,545 shared papers,
  44 identical-text pairs, max elementwise |diff| = 0.000183, min cosine = 0.99999952.
  **Caveat: line 11 hardcodes `ROOT = Path('/home/manh/dissertation')`** — machine-specific;
  will not run on a fresh checkout at a different path without editing.

### 3.4 Not re-run (relied on on-disk ground-truth logs + code reading)

- **`register_validation_followup.py`** (~2–4 min) and **`register_validation_check.py`**
  (~2–3 min): not re-run; their ground-truth logs (`regcheck_followup.log`,
  `regcheck_full.log`) are on disk and every number in Reports 1/Followup-1 was
  checked against them. Their RNG logic was verified by code reading:
  `register_validation_followup.py:47` defines one module-level `rng`; the re-seeds
  inside `main()` (lines 309/319/345) are **dead locals** because `sample_research`/
  `sample_policy`/`build_with_text` read the module-global `rng` — confirming the
  "three successive draws of one continuous seed-42 stream" behaviour that the
  followup-2 script reproduces (and which the byte-identical re-run empirically confirms).

### 3.5 Scratch-only intermediate dependencies

- **No script READS any scratch-only intermediate file.** All read `2_data/` +
  `register_utils` from `1_code/` only. Scratch writes are each script's own outputs
  (`.npz`/`.npy`/`.log`/`.txt`).
- The real reproducibility risk is the reverse: **the scripts themselves are
  gitignored scratch.** A fresh checkout (or `git clean -fdX`) loses every script and
  every ground-truth log; only the three `.md` reports survive. Promoting the appendix
  therefore requires committing the script(s) (e.g., under
  `1_code/7_main_analysis/2_appendix/`) and either committing the ground-truth logs or
  re-running. Also required for a clean run: the embedded `2_data` snapshot, the nltk
  POS tagger, and de-hardcoding `ROOT` in `check_concept_same_space.py`.

---

## 4. Current manuscript state (exact, unmodified)

All quotes from `3_writing/dissertation.tex` (895 lines; line numbers are current as of
HEAD `1a2f97a`).

- **Line 279** (Methodology, `\subsection{Register Adjustment via Iterative Nullspace
  Projection (INLP)}` starts at line 256, `\label{sec:register-adjustment-inlp}`;
  paragraph "Identification argument." begins at line 276):

  > "This identification argument is structural, not empirical: the stratification design \emph{ensures} that no linear direction can simultaneously encode SDG topic and corpus identity, because every SDG is balanced across the two corpora at every iteration. What remains linearly decodable is therefore the corpus-level register difference. The removed subspace is interpreted as register in the sense of \textcite{biberConrad2009register}: a variety associated with a particular situation of use, including communicative purpose, audience, and production circumstances. It is not validated against independent linguistic markers (e.g.\ hedge-word density, passive-voice frequency, mean sentence length), which would be the natural next step; this study treats the register interpretation as plausible on design grounds and reports the adjusted gap as a main but unvalidated estimate. Independent validation against Biber-style linguistic features \parencite{biber1988variation} would strengthen the register claim and is left to future work."

- **Line 373** (table note under Table `tab:register-decomposition`, caption at 370
  "Register-topic decomposition of the semantic gap."; `\input{...tab5_register_decomposition.tex}` at 372):

  > "Notes: Raw gap is the reference semantic-gap estimate; Adjusted gap is after INLP register removal (primary, unvalidated); Register component = raw $-$ adjusted. Coverage correlations are Spearman $\rho$ across 17 SDGs."

- **Line 392** (table note under Table `tab:interaction`, caption at 389 "H1a–H1d
  coverage-predictor vs semantic-gap Spearman correlations across encoder--classifier
  configs."; `\input{...tab4_interaction_h25.tex}` at 391):

  > "Notes: Raw gap = main semantic-gap estimate; Adj.\ gap = after INLP register removal (main reported estimate, unvalidated); Register = raw $-$ adjusted. Each row is an independent computation over 17 SDGs. $^{***}p<.001$, $^{**}p<.01$, $^{*}p<.05$, $^{\dagger}p<.10$."

- **Line 477** (Limitations, `\label{sec:limitations}` at 461; bold lead "Register effects."):

  > "**Register effects.** The gap may capture register, not topic. A register decomposition using Iterative Nullspace Projection (INLP) separates the raw centroid distance into a topic component (adjusted gap, mean \MeanAdjustedGap{}) and a register component (mean \MeanRegisterComponent{}; Appendix~\ref{app:register-robustness}). After register removal, the adjusted topic gap is the primary comparison (pending independent validation of the INLP removal): it shows a positive association with coverage divergence (Spearman $\rho = \RhoCovTopic$, $p = 0.054$), while the raw gap is near-zero ($\rho = \SpearmanCovRaw$, $p = 0.765$) because the register component ($\rho = \RhoCovRegister$, $p = 0.045$) appears to cancel the topic signal. For all SDGs except SDG~17, the register component is positive (range: \RegCompSdgFifteen{} to \RegCompSdgThirteen{}), confirming that register uniformly inflates the raw gap. The raw gap is retained as a reference measure because it directly corresponds to the observed framing difference, while the adjusted topic gap is the primary but unvalidated estimate. The removed subspace is interpreted as register; in principle, some substantive framing differences between corpora within an SDG could also be linearly decodable from corpus identity and removed alongside register. Independently validating the removed subspace against established linguistic register markers (e.g.\ hedge-word rate, passive-voice frequency, mean sentence length) would strengthen the register interpretation and is left to future work."

- **Related unvalidated mentions:** line 488 (Conclusion, `\section{Conclusion}` at 484):
  "The primary result is the register-adjusted (INLP) topic gap, though the INLP removal
  is unvalidated (Section~\ref{sec:limitations})…"; also line 303 (Results preamble,
  "the adjusted (register-removed) ranking…").

Prior handoffs referenced "lines 279/477" for the "left to future work" language and
"lines 373/392" for "primary but unvalidated" — all four line numbers are still exact.

---

## 5. Appendix-registration mechanics (the pattern to mirror)

### 5.1 The registry

- Single source of truth: **`APPENDIX_SPECS`** in
  `1_code/7_main_analysis/0_shared/analysis_orchestrator.py:37–174`. Each entry:
  `flag`, `aliases`, `script`, `help`, `warn`, `run_label`, `step_id`, `in_all`,
  `requires`. `in_all=True` → runs under `--appendix-all` (`APPENDIX_STEPS`, line 176).
- `main.py:342–363` `run_appendix_spec()` executes each spec as a subprocess:
  `python 1_code/7_main_analysis/<script> --output-dir 4_outputs [--embed-model ...] [--overwrite]`,
  after checking `requires` files exist in `output_dir/{model}/data/`.
- Appendix scripts live under `1_code/7_main_analysis/2_appendix/`.

### 5.2 Fingerprint / `--overwrite` conventions (Tier B)

Pattern in every appendix script (concrete example `a2_policy_source_family_sensitivity.py:426–435,688`):
1. `fp = fingerprint_of(*direct_input_files) + "<script_version>"` (plus a G/`adjusted` component when `--embeddings adjusted`).
2. `if should_skip(OUTPUTS, fp, args.overwrite, PRIMARY): return` — skips iff outputs
   exist, the sidecar fingerprint matches, and `--overwrite` is not set.
3. Write outputs, then `record_fingerprint(OUTPUTS, fp, PRIMARY)` — writes
   `<primary>.fingerprint.json` next to the primary output.

Implementations: `shared_utils.py` (`fingerprint_of` :27, `should_skip` :39,
`record_fingerprint` :54). Fingerprints are **mtime+size+first-64KB content-based**
(SHA-256) — consistent with the AGENTS.md "do not add mtime-based fingerprinting to the
expensive frontier stages" rule; these appendix stages are cheap Tier-B.

### 5.3 Output layout

- Data/JSON/CSV → `4_outputs/appendix/{model}/{slug}/data/`
- Tables/macros (.tex) → `4_outputs/appendix/{model}/{slug}/tables/`
- Model namespacing via `shared_utils._insert_model_in_rel` (:68) /
  `ensure_dissertation_outputs` (:187); e.g. MPNet → `4_outputs/appendix/mpnet/...`.
- For `--build-pdf` to pass, new appendix outputs must also be listed in
  `shared_utils.py`: `MANUSCRIPT_EXTRA_FILES` (:105, JSON/CSV) and/or
  `MANUSCRIPT_APPENDIX_TABLE_FILES` (:161, .tex). `require_pdf_inputs` (:233) checks them.

### 5.4 Macro generation

- Appendix scripts generate their **own** `num_*.tex` / `tab_*.tex` (e.g.
  `a2`: `write_table_h25`/`write_h25_macros` → `num_a2_policy_source_family_h25.tex`).
- Main-text macros are consolidated by `1_code/7_main_analysis/0_shared/generate_tex_macros.py`
  (reads `register_decomposition.json` + `interaction_extended.json` →
  `4_outputs/{model}/tables/num5_register_decomposition.tex`).
- The `.tex` consumes them via `\input{../4_outputs/...}` — see §5.5.

### 5.5 Concrete example (Appendix A2 — mirror this exactly)

| Piece | Path |
|---|---|
| Registration | `analysis_orchestrator.py:38–48` (`flag=appendix-a2-family`, `step_id=A2`, `in_all=True`, `requires=None`) |
| Code | `1_code/7_main_analysis/2_appendix/a2_policy_source_family_sensitivity.py` |
| Output data | `4_outputs/appendix/mpnet/a2_source_family_sensitivity/data/{policy_source_family_summary.csv, policy_source_family_coverage.csv, policy_source_family_semantic_gaps.csv, policy_source_family_h25.csv, policy_source_family_h25.json}` (+ `policy_source_family_summary.csv.fingerprint.json` sidecar) |
| Output tables/macros | `4_outputs/appendix/mpnet/a2_source_family_sensitivity/tables/{tab_a2_policy_source_family_combined.tex, tab_a2_policy_source_family_h25.tex, num_a2_policy_source_family_h25.tex}` |
| `.tex` inclusion | `3_writing/dissertation.tex:841` (`tab_a2_policy_source_family_combined.tex`) and `:851` (`tab_a2_policy_source_family_h25.tex`), both `\resizebox{\textwidth}{!}{\input{../4_outputs/appendix/mpnet/a2_source_family_sensitivity/tables/...}}` |
| Registered for build-pdf | `shared_utils.py:162–164` (in `MANUSCRIPT_APPENDIX_TABLE_FILES`); JSONs at `:106–110` (in `MANUSCRIPT_EXTRA_FILES`) |

Second example (K.1, most recent): registered `analysis_orchestrator.py:164–173`;
outputs `4_outputs/appendix/mpnet/k1_regression_semantic_gap/data/{spec_grid.json,
spec_grid.fingerprint.json, bootstrap_grid.json}` + `tables/tab_k1_specification_grid.tex`;
`\input{...tab_k1_specification_grid.tex}` at `dissertation.tex:866`.

**Environment note for the promotion step:** a register-validation appendix will
need (a) the nltk `averaged_perceptron_tagger_eng` data added to `environment.yml`
(§3.1), (b) `check_concept_same_space.py`'s hardcoded `ROOT` de-hardcoded if that
proof is re-run, and (c) the `2_data` embedded snapshot hydrated. It should follow
the §5.1–§5.4 pattern rather than inventing a new one.

---

## 6. Git status precisely

- **HEAD:** `1a2f97a` (`register validation follow-up 2: sample construction, accuracy
  CIs, policy other-dist pull`), branch `main`.
- **`origin/main` == HEAD** — everything committed is pushed.
- **`git status` is clean:** nothing staged, nothing modified-but-uncommitted
  (verified `git status --short` empty; `2_data/`, `4_outputs/`, `1_code/`,
  `3_writing/` all untouched by this audit).
- **Committed register-validation work:** 3 reports (`0f96a3f`, `c2773a9`, `1a2f97a`),
  2 handoffs (`26455a5`, `1a2f97a`), plus `handoff.md` (`1a2f97a`).
- **Untracked (ignored):** everything in `5_notes/scratch/` — all 5 scripts, all logs
  (`regcheck_full.log`, `regcheck_followup.log`, `followup2.log`, `followup2.DONE`,
  `regcheck_followup.DONE`), all `.npz`/`.npy` artifacts, `followup2_replacements.txt`,
  the three scratch report copies (identical to committed), `__pycache__/`.

**Risk of losing work:**
- A `git reset --hard` (or `git checkout -- .`) **would not lose anything** — the
  tree is clean; scratch is untracked and untouched by reset.
- The **real loss vectors** are (a) `git clean -fdX` (removes ignored files → deletes
  **the entire scratch dir including all scripts and ground-truth logs**), or
  (b) deleting/not-having `5_notes/scratch/` (a fresh clone never has it). In either
  case only the three committed `.md` reports and two handoffs survive; the scripts
  and logs are gone. If the diagnostic is to remain re-runnable, the scripts (and
  ideally the ground-truth logs) must be promoted out of scratch as part of the
  appendix work.
- No uncommitted edits to any tracked file exist right now.

---

## 7. Confidence notes (things I did not verify at 100%)

- I re-ran only the Follow-up-2 line end-to-end plus the two cheap scripts. Follow-up-1
  and Report-1 runs were verified by code-reading + their on-disk ground-truth logs,
  not by a fresh re-run. The byte-identical Follow-up-2 re-run transitively validates
  the shared sampling stream (the acceptance gate), but the per-SDG 3a/3b/3d tables of
  Follow-up-1 rest on `regcheck_followup.log` alone.
- The claim that `followup2.log` came from the current script rests on script-mtime
  ordering (10:16 < 10:24) + the fresh run being byte-identical — strong but not a
  git-attested record.
- The nltk POS-tagger dependency is confirmed present in the current env, but I cannot
  attest that a clean conda rebuild lacks it; that claim comes from AGENTS.md §2.5 /
  the handoff.
- The two distinct −0.197 values (§2.4 flag 2) were confirmed from
  `regcheck_followup.log:41` vs `:96` (different p-values 6.1e-05 vs 0.0047), but the
  followup-2 script does not print the 2B pooled 2c value, so that one number is
  implicitly — not explicitly — re-verified by the Follow-up-2 run.

---

# Implementation Report — INLP Register-Validation Appendix (step F2 / `a1_register_validation`)

**Date:** 2026-08-05 (implementation phase; complete)
**Repo:** `/home/manh/dissertation`, branch `main`
**Scope:** promotion of the verified register-validation diagnostic (Sections 1–4 of this
report) into a durable dissertation appendix stage, per Section 5's conventions. The
verification report above remains ground truth for all numbers; this section records the
promotion.

## 8. Deliverable summary

### 8.1 Files added

- `1_code/7_main_analysis/2_appendix/a1_register_validation.py` — consolidated A3-shaped
  appendix script. Canonical-MPNet-only gate (mirrors the zero-shot precedent); `--seed` pinned
  to 42 (raises on any other value); nltk fail-closed runtime guard (`punkt` +
  `averaged_perceptron_tagger_eng`) with an actionable download message (conservatively NOT
  added to `environment.yml` — conda-forge `nltk_data` compatibility with nltk 3.10 unverified;
  documented in docstring + README). Fingerprint-gated skip via
  `shared_utils.should_skip` / `record_fingerprint` over (score manifest, research-embed
  manifest, `G.npy`, `SCRIPT_VERSION("1")`). Ports the verified scratch logic verbatim
  (single module-level seed-42 `_rng` stream, research-then-policy per draw, draws 1/2/3 +
  fresh seeds 43/44/45; do not refactor the sampling). Contains a **61-check acceptance-gate
  block** that raises `RuntimeError` on any mismatch with report.md §2.2; all 61 gates pass in
  4 independent runs (determinism confirmed).
- `4_outputs/appendix/mpnet/a1_register_validation/`
  - `data/register_validation.json` — nested machine-readable record of every number, including
    the `register_score_operationalization` honesty note.
  - `data/register_validation.csv` — 8 corpus-discrimination rows.
  - `data/register_validation.fingerprint.json` — fingerprint sidecar.
  - `tables/tab_a1_register_validation.tex`, `tables/tab_a1_register_validation_selectivity.tex`
    — the two appendix tables.
  - `tables/num_a1_register_validation.tex` — 88 `\RegVal*` macros. **TeX control-word names
    cannot contain digits** — the script emits spelled-out names (`\RegValDrawFortyThree` etc.),
    namespace distinct from `Register*` / `RegIter*`.

### 8.2 Files modified

- `1_code/7_main_analysis/0_shared/analysis_orchestrator.py` — `APPENDIX_SPECS` entry
  (`flag=--appendix-a1-register-validation`, alias `--register-validation`, `step_id=F2`,
  `in_all=True`, `requires=None`); canonical-order docstring updated to `..., H1, I1, F2,
  G(opt-in), J1, K1`.
- `1_code/7_main_analysis/0_shared/shared_utils.py` — outputs added to
  `MANUSCRIPT_EXTRA_FILES` (JSON/CSV) and `MANUSCRIPT_APPENDIX_TABLE_FILES` (the 3 tex files),
  required by `require_pdf_inputs` for `--build-pdf`.
- `main.py` — `--appendix-all` help now "(A2, A3, B2, C, C1, C0, D1, H1, I1, F2, J1, K1)".
- `README.md` — new appendix command row + nltk-data bullet in Environment notes.
- `PIPELINE.md` — appendix table row for `a1_register_validation.py`.
- `AGENTS.md` — Tier-B counts 15 → 16 (two places); checkpoint-inventory row comment updated.
- `3_writing/dissertation.tex` — see §8.4.
- `4_outputs/dissertation.pdf` — rebuilt; Appendix G renders (see §8.5).
- `handoff.md` — status updated to complete.
- `report.md` — this section.

### 8.3 Appendix identifier

- step_id **F2**; slug **`a1_register_validation`**; flag `--appendix-a1-register-validation`
  (alias `--register-validation`); `in_all=True`; `requires=None`. Runs under
  `--appendix-all`, honors `--overwrite`, supports fingerprints, writes to
  `4_outputs/appendix/{model}/a1_register_validation/` (canonical MPNet only).

### 8.4 Manuscript (dissertation.tex)

- New appendix section, label `app:register-validation`, title "Register Removal: Validation
  Against Independent Linguistic Register Markers"; becomes **Appendix G** (Concept-Retrieval
  → H, Cross-Method → I, Pooled Regression → J, AI Declaration → K; all refs label-based).
  Contains: Motivation; Register score and samples; Two sample constructions;
  Corpus-discrimination accuracy (Table `tab:register-validation-accuracy`);
  "An apparent residual-register signal was traced to clustering"; "No robust evidence that
  topic is being systematically removed" (Table `tab:register-validation-selectivity`);
  "One apparently surviving signal failed the draw-stability check" (honest history: mega-doc
  clustering traced, −0.197 downgraded to noise); Conclusion (only report.md §2.2 reconciled
  values; "substantial reduction, not complete elimination, no robust evidence of topic
  removal"); Limitations (PC1-vs-z-sum operationalization flagged + unresolved, n=408 /
  per-SDG n=12 power, draw-stability applied only to within-SDG trace, one-per-parent is a
  rebuild not subset, MPNet-only, the two −0.197 statistics caution).
- Preamble: `\InputIfFileExists{...num_a1_register_validation.tex}{}{}` after the
  `num17_reference_split.tex` line (input once, not again in the section).
- Rewires of "unvalidated / left to future work" wording: Methodology identification-argument
  (~line 279) now says the interpretation is evaluated in the appendix, finds substantial
  reduction but not complete elimination, no robust evidence of topic removal; Table notes at
  ~373 and ~392 now "(primary estimate; evaluated in Appendix~\ref{app:register-validation})";
  Limitations "Register effects" (~477) and Conclusion (~488) rewired to the appendix. Stale
  header comments ("Appendix D/E") fixed. Line-471 corpus-scope "left to future work" is
  unrelated and untouched.

### 8.5 Verification performed

- **Acceptance gates 61/61**, deterministic across 4 runs (scratch/registry paths).
- **`--build-pdf --overwrite` succeeds.** A LaTeX structural bug found during the build — the
  first insertion displaced the register-check table's `\end{table}`, placing the entire new
  section inside a `table` environment ("Not in outer par mode", 707pt float cascade) — was
  fixed by restoring the `\end{table}` after the `tab:iterative-register-check` notes line and
  removing the stray closing before `\section{Concept-Retrieval Sensitivity}`. After the fix
  the F..G region has balanced 3 begin / 3 end table environments. Rebuild verified: 72 pp, no
  fatal errors, 0 "Not in outer par mode", 0 "Missing \begin{document}", 0 undefined control
  sequences; Appendix G renders with both tables and all `\RegVal*` macros resolve to the
  report.md §2.2 values; rewired Methodology/Limitations/Conclusion sentences read correctly in
  the PDF; no orphaned register-related "unvalidated"/"left to future work" wording remains.
  Remaining `Float too large` warnings (lines 71/76/593/881) are pre-existing floats, not from
  the new section. The PDF was not image-inspected (model cannot view images); geometry was
  checked via absence of overfull-vbox errors on the G pages.
- **gz fallback path exercised** (previously untested for this script): with
  `2_segmented/{research,policy.jsonl}` temporarily hidden, `resolve_research_text_path`
  (`all-mpnet-base-v2`, `part-{sid:05d}`) and `resolve_policy_text_path` resolve to
  `3a_warm_replay_texts/mpnet/*.jsonl.gz` and `open_text` reads them; canonical dirs restored
  intact. The script passes the full model name and `part-{sid:05d}` shard names, which match
  the fallback layout (all 26 shards present in both score metadata and fallback dirs).
- **Cleanup:** all `3_writing/min_test*.{tex,aux,log,out,pdf,bcf,run.xml,toc,bbl,blg}`
  bisection junk deleted; `5_notes/scratch/` untouched (provenance).

### 8.6 Unresolved issues intentionally left (require human scientific judgment)

Per report.md §2.4, none were resolved here:

1. **Register-score operationalization** (PC1 vs. the a-priori "institutional" z-sum, which gave
   null/reversed 2b/2c on the initial screen) — flagged in the appendix Limitations and in the
   output JSON as an open limitation.
2. **The two distinct −0.197 statistics** (2B pooled centroid-distance vs. Item-3 policy
   other-dist) — documented as a non-contradiction caution in the appendix.
3. **n=408 / per-SDG n=12 power** and the n=120/SDG scaling question — noted as limitations,
   not resolved.
4. **Commits** (see §8.7) and the Phase-5 decisions (retire/keep scratch; commit ground-truth
   logs under `5_notes/` per report.md §3.5) remain user decisions.

### 8.7 Git state

Committed in three one-concern commits (per AGENTS.md), then pushed:

- (a) script + registration + docs: `1_code/7_main_analysis/2_appendix/a1_register_validation.py`,
  `analysis_orchestrator.py`, `shared_utils.py`, `main.py`, `README.md`, `PIPELINE.md`,
  `AGENTS.md`, `report.md`, `handoff.md`.
- (b) manuscript: `3_writing/dissertation.tex`.
- (c) outputs + PDF: `4_outputs/appendix/mpnet/a1_register_validation/`,
  `4_outputs/dissertation.pdf`.

Working tree is clean after the push.

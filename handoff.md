# Hand-off: Make the adjusted (register-removed) topic gap the canonical main-text thread

**Status:** Work C (figures) and Work G (consistency) are DONE. Work A (distributional
section rewrite) is **rewritten in the tex** but the supporting macro file it depends on is
**not yet written by the pipeline** — a distributional job is still running (PID 228663) and
will emit the *wrong* (raw-reference) macros; a one-line-fixed script must be re-run after it
finishes. Final PDF build is therefore still blocked on that re-run.

**Override note:** this message replaces the previous `handoff.md` in full.

---

## 1. Context — where we are

This is a dissertation reproducibility repo (`/home/manh/dissertation`). The manuscript is
`3_writing/dissertation.tex`. The agreed user decision (from the prior hand-off): the **adjusted
(INLP register-removed) topic gap is the canonical main-text measure**; the **raw gap is only a
naive baseline**. Appendix seams stay as-is; the "affinity-era" item is dropped.

This session completed the figure work (Work C) and most of the consistency pass (Work G), and
rewrote the distributional Results section (Work A) in the tex — but discovered that, under
adjusted embeddings, the distribution-aware battery does **NOT** rank SDG 17 first and correlates
only moderately with the adjusted centroid gap. This contradicted the prior hand-off's planned
sentence ("SDG 17 is the most divergent under every metric"). The section was rewritten to be
honest. A script bug was also found and fixed (the distributional ρ reference was the RAW gap,
not the adjusted gap, in adjusted mode). A long-running distributional job (PID 228663) is
mid-flight; it was launched with the OLD (buggy) script, so its output must be superseded by a
re-run of the now-fixed script.

**Net:** figures done; consistency done; distributional tex prose done; pipeline macro file
pending (job + 1 re-run). Then build & verify.

---

## 2. Key known facts (so you don't re-derive)

- **Adjusted vs raw gap (SDG, Adj / Raw)** from `4_outputs/mpnet/data/adjusted/4_3_semantic_gap_distances.json`
  and `4_outputs/mpnet/data/4_3_semantic_gap_distances.json` (per_sdg `semantic_gap`):
  - SDG 17: 0.371 / 0.216 ← most divergent **adjusted** (was least raw)
  - SDG 3:  0.290 / 0.470
  - SDG 16: 0.269 / 0.331
  - SDG 1:  0.268 / 0.447
  - SDG 12: 0.245 / 0.434
  - SDG 10: 0.249 / 0.296
  - SDG 13: 0.230 / 0.481 ← was the raw leader
  - SDG 15: 0.089 / 0.221 ← least divergent under both
  - SDG 2:  0.094 / 0.278
  - **Adjusted top-5:** 17, 3, 16, 1, 10. **Raw top-5:** 13, 3, 11, 1, 12.
  - **Adjusted bottom-5:** 15, 2, 6, 14, 4.
  - Adjusted median = 0.2226, mean = 0.2091; raw median = 0.331.
- **Register decomposition:** `raw − adjusted = register component`. The register component has
  **larger variance than the raw gap itself** (var ratio 1.24), correlates ρ=0.667 with the raw
  gap, and ρ=−0.292 with the adjusted topic gap. So raw distribution-aware metrics were tracking
  **register**, not topic — this is the mechanism behind the original "ranking preserved"
  claim and why it breaks after INLP.
- **Adjusted distribution-aware metrics (full-corpus, from the running job's log):** on adjusted
  (projected) embeddings, SWD and Gaussian-2-Wasserstein are LARGEST for **SDG 12**, Chamfer
  LARGEST for **SDG 1** — NOT SDG 17. Spearman ρ vs the **adjusted** centroid gap:
  swd=0.40, w2=0.32, chamfer=0.67. (Vs the raw gap the script currently emits: swd=0.47,
  w2=0.81, chamfer=0.42 — still not "preserved at p<0.001".)
- **Macros:** tex inputs `num_register_topic_decomposition.tex` (provides `\AdjGapSdgOne`…,
  `\MeanAdjustedGap`, etc.) and `num_semantic.tex` (raw `\SemanticGapSdgX`, `\MeanRawGap`).
  Distributional macros come from `4_outputs/mpnet/adjusted/tables/num_distributional_gap.tex`
  (currently NOT yet written): `\DistGapSpearmanSwd`, `\DistGapSpearmanFrechet`,
  `\DistGapSpearmanChamfer`, `\DistGapShapeShareMean`, `\DistGapMaxSeedDelta`.
- **G matrices** exist for all three models under `2_data/3_embedded/{mpnet,minilm,scibert}/register/{canon,subset}/G.npy`.
- **Adjusted JSONs** exist for every model/config:
  `4_outputs/{model}/data/adjusted/4_3{_mlp}_semantic_gap_distances.json` and
  `4_outputs/mpnet/data/adjusted/4_3_semantic_gap_robustness_caps.json`.
- The INLP fix in `g_distributional_gap.py` (frozen-dataclass bug) was applied in the PRIOR
  session; the script now runs under `--embeddings adjusted`.
- `plot_figures.py` has NO `--embeddings` flag by design; this session made fig4/fig5
  **always** render adjusted-primary (with raw overlay) when the adjusted JSON is present, so a
  plain `plot_figures.py` run (e.g. warm replay) produces canonical figures reproducibly.

---

## 3. Actions / decisions made & files changed this session (and why)

### Pipeline scripts
- **`1_code/8_visualization/plot_figures.py`** (MODIFIED + regenerated figures):
  - Added `load_gap_maps(layout)` and `use_adjusted` flag. `fig4_semantic_gap.pdf` now renders
    **adjusted bars + raw baseline diamonds**; `fig5_coverage_semantic_scatter.pdf` renders
    **adjusted solid points + raw open points**. When the adjusted JSON is absent it falls back
    to the old raw-only rendering (safe for other models/configs).
  - Ran `python 1_code/8_visualization/plot_figures.py --overwrite` → regenerated `fig4_*`,
    `fig5_*`, `fig3_*`, heatmap in `4_outputs/mpnet/figures/`.
- **`1_code/7_main_analysis/1_main_text/3_generate_cross_sensitivity_table.py`** (MODIFIED + regenerated):
  - **Header fix:** the raw block was appended as 4 separate flat groups, leaving a stray empty
    header cell and no "Raw" group title. Rebuilt it as a single nested group `"Raw (naive
    baseline)"` mirroring the `"Adj. gap (canonical)"` group, so the table now clearly shows two
    labeled parent groups.
  - **Encoder-macro bug fix (important):** `write_encoder_axis_semantic()` looked up rho with keys
    `::LR`, `::MLP`, `::ZS`, but Work B had renamed the columns to `LR (adj.)`, `MLP (adj.)`,
    `ZS`. The mismatch made `MiniLMSemanticRho`/`SciBERTSemanticRho`/`MlpSemanticRho`/`ZeroShotSemanticRho`
    emit `--` (breaking the Limitations text that cites them). Fixed the lookup keys to
    `::LR (adj.)`, `::MLP (adj.)`, `::ZS`. After re-run they are now 0.69 / 0.48 / 0.83 / 0.00.
  - Ran with `--overwrite` → regenerated `tab_cross_sensitivity_robustness.tex`,
    `tab_encoder_sensitivity_semantic.tex`, and the `num_*` macro files.
- **`1_code/7_main_analysis/1_main_text/g_distributional_gap.py`** (MODIFIED, NOT yet re-run to completion):
  - **Reference bug fix:** `load_canonical()` always loaded the **raw** `4_3_semantic_gap_distances.json`
    as the Spearman-ρ reference, even in `--embeddings adjusted` mode. So the emitted
    `\DistGapSpearman*` macros were ρ(distribution-metric-on-adjusted-embeddings, **raw** gap) —
    a mismatched comparison. Fixed so that in adjusted mode the canonical reference is the
    **adjusted** gap (only `semantic_gap` is substituted; GATE 1/2 assignment-count fields stay
    raw). GATE 4 is skipped in adjusted mode regardless, so no side effects there.
  - NOTE: the currently-running PID 228663 was launched with the OLD code, so its output will
    carry the raw-reference ρ. It must be superseded by a re-run of the fixed script.

### Manuscript `3_writing/dissertation.tex` (edits)
- **fig4 caption** (≈line 354): notes solid bars = adjusted (canonical), grey diamonds = raw baseline.
- **fig5 caption** (≈line 393): notes solid blue = adjusted, open grey = raw baseline.
- **Interaction sentence** (≈line 388): replaced the false "SDG 16 has the largest coverage gap
  but a below-median gap" with accurate phrasing — SDG 3 has the largest coverage gap AND the
  2nd-largest adjusted gap; SDG 9 has 2nd-largest coverage gap but a near-median adjusted gap.
  (The old claim was wrong on two counts: data shows SDG 3 has the largest coverage gap, and
  "below-median" no longer holds under adjusted.)
- **§Robustness of the Semantic-Gap Ranking to the Distance Functional** (≈lines 429–434):
  rewrote the three paragraphs. Now states the adjusted centroid gap ranks **SDG 17 first**
  (validated by GATE 4), but the distribution-aware metrics do **NOT** rank 17 first (SWD/W2 →
  SDG 12, Chamfer → SDG 1) and correlate only moderately (ρ `\DistGapSpearmanSwd` / `\DistGapSpearmanFrechet`
  / `\DistGapSpearmanChamfer`). The adjusted topic gap is framed as a **mean-direction** finding,
  with distribution shape as a complementary signal. Margin nuance now names SDGs 3 & 16 as the
  2nd/3rd adjusted positions.
- **Limitations — "Distance metric" paragraph** (≈line 473): replaced "SDG 13 most divergent
  under every metric; ρ=0.44–0.91" with the accurate adjusted framing, attributing the old raw
  robustness to the register artifact.
- **Appendix §Distance-Functional Robustness Table** (≈line 604): added a detailed paragraph
  explaining *why the raw robustness result does not carry over* — the register-component
  decomposition (variance ratio 1.24, ρ=0.67 vs raw, ρ=−0.29 vs adjusted), i.e. the user's
  "register is what was projected out" insight, as a forward-looking methodological note.
- **Conclusion** (≈line 490): added a sentence leading with the canonical adjusted finding
  (SDG 17 most divergent, SDG 15 least, SDGs 3/16/1 strongly divergent; raw inverts this).
- **Appendix distributional table input** (≈line 606): already repointed to
  `4_outputs/mpnet/adjusted/tables/tab_distributional_gap.tex` (done in prior session).

---

## 4. What remains and why

- **A. Finish the distributional macro file.** The original PID 228663 was launched with the
  OLD (buggy) script, so it would have emitted raw-reference ρ. It has been **killed**, and the
  **fixed** script relaunched as **PID 241083** (`python 1_code/7_main_analysis/1_main_text/g_distributional_gap.py --embeddings adjusted --overwrite`). It resumed from 40 cached records
  (`g_distributional_gap_records.jsonl`, config hash unchanged → safe resume) and is finishing
  the remaining sampled SDG×seed runs (~19 left, ~160–330 s each → roughly 40–70 min). When it
  ends it writes `4_outputs/mpnet/adjusted/tables/{num,tab}_distributional_gap.tex` **with the
  corrected adjusted-reference ρ**.
  - Do **not** build the PDF before this finishes: the tex references those five macros and will
    fail with "undefined control sequence" until the file exists.
  - To poll: `grep -c sampled /tmp/dist_adj2.log` (target 34/34) or `ls 4_outputs/mpnet/adjusted/tables/`.
- **B. Build & verify.** `bash 3_writing/build_pdf.sh` from repo root, in the `dissertation`
  conda env, on bash/Linux. Fix any undefined macro or compile error. Confirm fig4/fig5 show
  adjusted-primary. Confirm the cross-sensitivity table still compiles (two labeled parent
  groups) and the encoder macros render real numbers (not `--`).
- **C. (Optional polish, non-blocking)** The `tab_cross_sensitivity_robustness.tex` header still
  has two harmless empty `\multicolumn{1}{c}{}` cells (above the Canon sub-column of each parent
  group). Purely cosmetic; leave or clean up in `assemble_table` if time permits.

---

## 5. Concerns to emphasize

1. **The running job is the OLD (buggy) script.** PID 228663 will emit raw-reference ρ macros.
   They are wrong for an adjusted analysis. You MUST re-run the fixed script after it finishes
   (reuses records, fast). Do not treat the job's direct output as final.
2. **Do not assert "SDG 17 most divergent under every metric."** It is false under adjusted:
   SWD/W2 favor SDG 12, Chamfer favors SDG 1. The manuscript now correctly says the adjusted
   centroid gap ranks 17 first, but distribution metrics diverge. Keep it that way.
3. **The adjusted distributional ρ is modest (≈0.3–0.7), not "preserved at p<0.001".** The
   original strong claim was a register artifact. The rewritten section reflects this; do not
   revert to the stronger wording.
4. **Zero-shot ρ = 0.00** is correct as now computed (raw ZS vs adjusted LR baseline). The
   Limitations text was changed from "moderate" to "near-zero" to match. (If a future decision
   makes the encoder table show the *adjusted* ZS column instead, this would become ≈0.855 —
   out of scope now; noted only so the number isn't "fixed" back to a wrong value.)
5. **Reproducibility of figures:** `plot_figures.py` now always renders adjusted-primary when the
   adjusted JSON exists, so warm replay produces canonical figures without a flag. Good — but be
   aware any plain re-run overwrites `fig4/fig5` in place (same filenames the tex references).
6. **Register-component numbers in the appendix are hardcoded text** (variance ratio 1.24,
   ρ=0.67, ρ=−0.29), computed from the mpnet JSONs this session. They are accurate for mpnet;
   if you later extend to MiniLM/SciBERT, recompute or soften to qualitative language.
7. **Nothing has been committed.** Changed files: the four scripts above, `dissertation.tex`, and
   many `4_outputs/*` artifacts (`?? 4_outputs/mpnet/adjusted/` is the pending job output). The
   prior session's a2-policy-source edits are also uncommitted.

---

## 6. The comprehensive plan (approved: "proceed with full plan", now including this session's corrections)

**Goal:** Adjusted (INLP) topic gap = canonical main-text measure; raw = naive baseline.
Keep appendix seams; re-run distributional test on adjusted AND fix its reference.

- **Work 0 ✅** Adjusted JSONs exist for MiniLM/SciBERT.
- **Work A 🔶** Distributional battery → adjusted. Script fix applied (ρ reference = adjusted).
  tex section REWRITTEN honestly. Remaining: let PID 228663 finish → re-run fixed script →
  verify macros.
- **Work B ✅** Cross-sensitivity + encoder tables → adjusted primary, raw baseline. Script
  modified; tables regenerated. Header fix + encoder-macro bug fix applied this session.
- **Work C ✅** Figures fig4 + fig5 → adjusted primary, raw baseline. DONE (plot_figures
  modified + regenerated; tex captions + sentence reconciled).
- **Work D ✅** Results §Semantic Gap → lead with adjusted. DONE (prior session).
- **Work E ✅** Discussion §Robust Patterns → adjusted ranks. DONE (prior session); verified
  against regenerated table (SDG 3/16/1 robust at canonical MPNet-LR; SDG 16 dips under Curated
  source family — fine, claim is scoped to canonical spec).
- **Work F ✅** Abstract + Intro → SDG 17 (adjusted) headline. DONE (prior session).
- **Work G ✅** Consistency pass: Conclusion (adjusted lead), Limitations register paragraph
  (canonical), Limitations "Distance metric" paragraph (now accurate adjusted framing),
  interaction sentence (reconciled), cross-sensitivity header (fixed), encoder macros (fixed).
  DONE.
- **Work H (this session, new)** Distributional reference bug in `g_distributional_gap.py`
  fixed; appendix forward-looking register note added.
- **Final step 🔶** Re-run fixed distributional script; `bash 3_writing/build_pdf.sh`; verify.

---

## 7. Exactly what was interrupted (last actions before this hand-off)

1. The original distributional job **PID 228663** was launched with the OLD (buggy) script, so it
   would have emitted raw-reference ρ. It has been **killed**; the **fixed** script is relaunched
   as **PID 241083** (`python 1_code/7_main_analysis/1_main_text/g_distributional_gap.py --embeddings adjusted --overwrite`). It resumed from 40 cached records (config hash unchanged → safe
   resume) and is finishing the remaining sampled SDG×seed runs (poll `grep -c sampled /tmp/dist_adj2.log`, target 34/34).
2. Completed ALL prose/figure/scope edits listed in §3 (plot_figures modified+regenerated;
   cross-sensitivity header + encoder-macro bug fixed + regenerated; g_distributional_gap.py
   reference bug fixed in source; dissertation.tex Work A section, Limitations "Distance metric",
   appendix register note, Conclusion, fig4/fig5 captions, interaction sentence all edited).
3. Verified the adjusted distribution-aware metric behavior from the job log and recomputed the
   register-component decomposition (variance ratio 1.24, ρ=0.67 vs raw, ρ=−0.29 vs adjusted),
   confirming the honest framing now in the tex.
4. **Immediate next action (was interrupted only by the hand-off request, now in progress):**
   (a) Wait for PID 241083 to finish (poll `grep -c sampled /tmp/dist_adj2.log`, target 34/34, or
       `ls 4_outputs/mpnet/adjusted/tables/`).
   (b) Verify `4_outputs/mpnet/adjusted/tables/num_distributional_gap.tex` now defines
       `\DistGapSpearmanSwd`, `\DistGapSpearmanFrechet`, `\DistGapSpearmanChamfer`,
       `\DistGapShapeShareMean`, `\DistGapMaxSeedDelta` with plausible adjusted values
       (full-corpus Swd≈0.40, Frechet≈0.32, Chamfer≈0.67 — sampled EMD/energy/MMD from the job).
   (c) `bash 3_writing/build_pdf.sh` (repo root, conda env `dissertation`, bash/Linux); fix any
       compile errors; confirm fig4/fig5 are adjusted-primary and the cross-sensitivity/encoder
       tables render with real numbers.

(End of hand-off)

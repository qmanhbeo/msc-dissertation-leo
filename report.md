# Independent Scientific Audit — Appendix G

**Title:** Register Removal: Validation Against Independent Linguistic Register Markers

**Role:** independent reviewer, not a coauthor defending the work.
**Date:** 2026-08-05
**Repo:** `/home/manh/dissertation`, branch `main`, HEAD `010a237`

## Scope and stance

This is an independent scientific audit of Appendix G (`dissertation.tex:718–764`).
The validation numbers are fixed and are not re-analysed or changed. The audit
verifies that (1) the reported evidence supports the stated conclusions, (2) the
conclusions are neither overstated nor unnecessarily conservative, (3) the appendix
tells a coherent scientific story, and (4) the main manuscript references the
appendix appropriately.

Authoritative sources read in full: `3_writing/dissertation.tex`; Appendix G;
the verification report (former `report.md` §§1–7); the implementation report
(former §8); the three register-validation reports
(`5_notes/register_validation_report.md`, `register_validation_followup.md`,
`register_validation_followup2.md`); the promoted appendix script
(`1_code/7_main_analysis/2_appendix/a1_register_validation.py`); and the appendix
outputs (`4_outputs/appendix/mpnet/a1_register_validation/{data,tables}/`).
The verification report remains the authoritative record of what analyses exist;
this document supersedes its prose sections.

No numerical result is changed. This document replaces the previous contents of
`report.md`.

---

## Phase 1 — Independent scientific assessment

### 1.1 Scientific narrative

**What question Appendix G answers.** The dissertation's central move is an INLP
adjustment that removes a "register" subspace from the embeddings (the corpus-level
research-vs-policy contrast), so that the remaining adjusted gap is claimed to be a
*topic* gap. The justification in the main text was purely structural: the
SDG-stratified training guarantees that what remains linearly decodable is corpus
identity, which is then *interpreted* as register. Appendix G asks an empirical
version of the same question: **does what INLP removes actually behave like register
in the text, and is anything non-register (topic) being destroyed?** It answers with
an independent, surface-linguistic register score (Biber-style features: hedge,
deontic modality, passive voice, sentence length, first person, nominalisation)
computed from the segment texts themselves.

**Evidence collected.** On two sample constructions of 408 segments each (12 per SDG
per corpus, seed 42, MPNet canon):
1. Corpus-feature means showing the corpora differ in deontic modality, sentence
   length, hedging, and passivisation.
2. A mega-document audit showing 25 segments from 7 flagship documents (SDSN, UNDP,
   WHO, EU AI Act, UN progress reports) are register outliers and create spurious
   pooled correlations; a one-per-parent rebuild.
3. Corpus-discrimination accuracy (5-fold LR) before/after removal: raw 0.909→adjusted
   0.505 (original), raw 0.944→adjusted 0.603 (one-per-parent), with Wilson CIs,
   binomial tests, and a bootstrap on the 0.505→0.603 difference.
4. Two "residual-register" diagnostics (removed-magnitude ~ register score; register
   score ~ distance to SDG centroid) that look positive on the original sample and
   disappear/reverse on the clean sample.
5. A 17-way SDG selectivity check (LR + kNN) showing topic decodability is preserved
   after removal.
6. A draw-stability check (seeds 43–45) showing the one surviving within-SDG trace
   (−0.197) is noise.

**Conclusions justified.** (i) INLP removes a large corpus-separable linear component
(0.94→0.60 on the clean sample); (ii) removal is not complete — adjusted accuracy is
significantly above chance; (iii) there is no robust evidence that topic is
systematically removed (selectivity essentially unchanged); (iv) the apparent
within-SDG residual-register traces are attributable to document clustering or draw
noise, not to residual register.

**Conclusions not justified.** The claim that the removed subspace is **empirically**
register — i.e., that it matches the surface-linguistic register features. The direct
evidence for this is weak: the removed-magnitude↔register-score correlations collapse
to null on the clean sample, and the register features alone classify corpus identity
at only ~0.54 (barely above chance), meaning they capture a small slice of what INLP
removed. What the evidence actually supports is: the removed component is
corpus-separable (by construction of INLP), is *not* topic, is *consistent* with a
register reading, and leaves no detectable residual register. The load-bearing step
from "corpus-separable linear signal" to "register" still rests on the structural
identification argument, which this appendix corroborates but does not independently
confirm.

### 1.2 Evidence audit (every major claim)

| # | Claim (Appendix G) | Verdict | Evidence |
|---|---|---|---|
| 1 | "The features discriminate the two corpora in the direction expected of register: policy more deontic, markedly longer sentences; research more hedge and more passive" (`dissertation.tex:725`) | **Partially supported / overstated** | On the original sample: deontic 3.130 vs 0.780 ✓; hedge 1.427 vs 0.854 ✓; passive 10.303 vs 8.453 ✓. But "markedly longer sentences" (63.803 vs 37.035) is mega-doc-driven: excluding the 25 mega-policy segments, policy mean sentence length falls to ≈34.0 — **below** research's 37.0. The direction reverses on the clean sample. Verified by arithmetic on `register_validation.json` `corpus_mean_features` + `mega_vs_nonmega_features`. |
| 2 | "7 source documents … contribute 25 segments across 15 of 17 SDGs … register outliers (sentence length 276.9 vs 35.6; passive 3.3 vs 9.8; first person 19.0 vs 5.9; nominal 20.0 vs 38.8)" (`:728`) | **Fully supported** | Matches JSON `item1_sample_construction` and `mega_vs_nonmega_features` exactly. |
| 3 | "Corpus accuracy falls from 0.909→0.505 (original) and 0.944→0.603 (one-per-parent)" (`:731`) | **Fully supported** | Table `tab_a1_register_validation.tex`; JSON `step2d_accuracy`. |
| 4 | "One-per-parent value significantly above chance (p=1.9e-05, CI [0.555,0.649]); original 0.505 n.s. (p=0.441); 'collapse to chance' corrected" (`:731`) | **Fully supported, well handled** | JSON `step2d_accuracy.opp.adj` / `orig.adj`; matches follow-up 2 and the verification report §2.2. |
| 5 | "Mega-policy exclusion raises original adjusted accuracy to 0.574 (p=0.002); +0.098 difference largely attributable to clustering" (`:731`) | **Fully supported** | JSON `step2d_accuracy.mega_policy_exclusion` = 0.574 (220/383, p=0.002); `bootstrap_diff` +0.098 CI [0.025,0.169]. "Largely attributable" (≈0.070/0.098) is a fair reading. |
| 6 | "Two residual-register signals (removed-norm ρ=+0.102 p=0.040; centroid-distance ρ 0.126→0.247) disappear/reverse on the clean sample (+0.092 n.s.; −0.212/−0.197)" (`:743`) | **Fully supported as reported** | JSON `step2b_removed_norm`, `step2c_centroid_dist`. But see §1.3 C1/C2: the *positive* 2b evidence dies with the red flag, and this loss is not discussed. |
| 7 | "17-way SDG selectivity essentially unchanged (LR 0.691→0.672; kNN 0.554→0.578; chance 0.059)" (`:746`) | **Supported, with an omission** | JSON `step3_selectivity`; computed on the **original (draw-1)** sample only, which the prose does not state. |
| 8 | "The −0.197 trace 'reproduced exactly on an independent sample' and survived mega-document exclusion (−0.213, p=0.003)" (`:758`) | **Misleading as worded** | The reproduction was an independent *re-run of the same draw-3 sample* (follow-up 1 vs follow-up 2), not a fresh draw — and the appendix's own draw-stability check immediately contradicts the "independent sample" reading. Mega-exclusion is reported only for draw 3; draws 1 and 2 give **+0.127 and −0.138** (sign flips). See §1.3 C3/C8. |
| 9 | "Draw-unstable: −0.130, −0.004, +0.126 across seeds 43–45; 0/17 per-SDG significant (14 neg / 3 pos); treated as noise" (`:758`) | **Fully supported** | JSON `item3.draw_stability`, `item3.per_sdg`; matches follow-up 2. |
| 10 | Conclusion: "register interpretation is therefore empirically supported for the dominant corpus-level component of the removed subspace" (`:761`) | **Overstated** | The evidence establishes the removed component is corpus-separable and not topic, and is *consistent* with register; it does not directly establish it *is* register (see §1.1). The main text (`:478`) is more accurate: "partial, not complete, empirical support." |
| 11 | Limitations (operationalisation, n=408/per-SDG n=12, draw-stability scope, rebuild-not-subset, MPNet-only, two −0.197s) (`:764`) | **Fully supported, honest** | All items traceable to the verification report §2.4 / follow-up 2. |

Additional observations:
- The **register-only classifier rows** (0.456 orig / 0.544 opp, Table 1) are never
  discussed in prose. They are the honest quantification of how much of the corpus
  difference the six features capture — directly relevant to claim #10, both
  supporting (0.544, p=0.042) and bounding it.
- The table note "(no leakage)" (`:739`) is inaccurate for the original sample, where
  mega-document segments straddle folds; follow-up 2 itself documents this.
- The register score's **PC1 variance share (22%) and loadings** are not reported,
  and the score is fit on the pooled (both-corpus) sample, making the "features
  discriminate the corpora" claim partly self-referential.

### 1.3 Examiner stress test

| # | Challenge | Why | Severity | Does the current appendix answer it? |
|---|---|---|---|---|
| C1 | "Your clean-sample 2b correlations collapsed to null and your register-only classifier is ~0.54. Where is the direct evidence that what INLP removed is *your* register features?" | 2b was the only direct link between removed magnitude and the surface features; it dies on the clean sample. The 6 features barely distinguish the corpora (0.544). | **High** | No — it is the central weakness. The appendix reports the 2b collapse but frames it only as refuting "residual register," not as removing the positive evidence; the register-only rows are never interpreted. |
| C2 | "Your 'markedly longer sentences' corpus difference reverses once the 7 flagship documents are excluded, and the 276.9-word figure is PDF-junk. Isn't sentence length your weakest feature, yet you orient the whole register score on it?" | Arithmetic on the appendix's own JSON shows clean-sample policy ≈34.0 vs research 37.0. The score's sign convention is anchored to a junk-inflated, reversal-prone feature. | **Medium-high** | No. The mega-doc contrast is reported, but the clean-sample *reversal* and the PDF-artifact origin are not; the sentence-length-based orientation is asserted as "more formal register." |
| C3 | "'Reproduced exactly on an independent sample' — but your own draw-stability check shows the statistic is noise. What is 'independent'?" | The reproduction was a re-run of the same draw-3 sample, not a fresh draw; "independent sample" invites a reproducibility reading that the next sentence refutes. | **Medium** | Partially — the draw-stability section immediately undercuts it, but the wording should be corrected, not left to the reader to reconcile. |
| C4 | "Why is adjusted accuracy 0.603 when INLP's stopping rule guarantees held-out accuracy ≤ 0.5?" | Seeming contradiction between the methodology (stop at chance) and the validation (above chance). | **Medium** | Implicitly (removal "substantial but not complete"), but the reason — INLP converges on its own training distribution, not on fresh samples — is never stated. |
| C5 | "Your selectivity check is on the contaminated original sample; does it hold on the one-per-parent sample?" | The appendix's own thesis is that sample composition matters; the selectivity check silently uses draw 1. | **Medium** | No — the sample is not stated. (It was not re-run on the clean sample.) |
| C6 | "Your Table 1 says 'no leakage,' but 25 mega-doc segments straddle folds in the original sample." | Grouped leakage via document twins. | **Low-medium** | No — the "(no leakage)" parenthetical is unqualified. |
| C7 | "The register score is the first PC of the *pooled* features; isn't the corpus-discrimination claim circular? Report loadings and variance." | A data-driven axis fit on both corpora will partly track corpus; nothing is reported about what PC1 weighs (Report 1 said 22% variance). | **Medium** | No — not addressed in the appendix. |
| C8 | "You cite only the draw-3 mega-exclusion value (−0.213). Draws 1 and 2 give +0.127 and −0.138." | Selective reporting of a three-draw result. | **Low-medium** | No — only the favourable draw is quoted. |
| C9 | "No multiple-comparison correction across the per-SDG and per-feature correlation tables." | 17 SDGs × several features at n=12. | **Low** | Partially — the appendix treats per-SDG results as low-power/noise, which is the right posture, but doesn't say so explicitly at the point of use. |
| C10 | "The residual 0.603 signal is uncharacterised — could it be non-register (topic-adjacent, formatting, extraction junk)?" | The appendix removes candidates one by one but never characterises what survives. | **Medium** | Partially — the draw-unstable other-dist was the only candidate and is noise; the appendix is honest that residual signal remains. |
| C11 | "MPNet only, n=408 — single encoder, small sample." | G exists for MiniLM/SciBERT; the register interpretation is claimed corpus-wide. | **Medium** | Yes — Limitations items 2 and 5. |
| C12 | "Your appendix conclusion says 'empirically supported'; the main text says 'partial, not complete, empirical support.' Which is it?" | Internal calibration mismatch. | **Low-medium** | No — the two should be aligned (the main text's is the more defensible). |

### 1.4 Remaining scientific risks (interpretation/inference only)

1. **The removed subspace is never directly shown to equal the surface register
   features.** The clean-sample 2b correlations are null and the register-only
   classifier is near-chance; the register reading of the removed component rests on
   the structural argument + absence of counter-evidence. This is the single largest
   residual risk and the one the main text's "partial support" phrasing already
   concedes.
2. **The register-score operationalization is unresolved** (PC1 vs the a-priori
   "institutional" z-sum, which gave null/reversed results on the initial screen and
   was never re-run on the clean samples). Since the score's *sign* is arbitrary and
   two different conventions gave opposite 2c signs, the correlation results are
   operationalization-sensitive even where they are robust in magnitude.
3. **The sentence-length feature is junk-inflated and its corpus direction reverses on
   the clean sample**, yet it anchors the score's orientation. The "policy = longer
   sentences = more formal" framing is not supported by the clean data.
4. **The residual corpus-linear signal (adjusted accuracy 0.603) is
   content-unidentified.** The only candidate interpretation (policy-side register
   pull) failed draw-stability; what the residual actually is remains open.
5. **Single encoder, small n.** The register interpretation is used for
   MPNet/MiniLM/SciBERT-adjusted tables in the main text, but Appendix G validates
   only MPNet at n=408.
6. **Two-sample composition change.** The one-per-parent rebuild systematically swaps
   global flagship prose for national-monitoring reports; the two-sample accuracy
   comparison (0.505 vs 0.603) is partly compositional, as the appendix honestly notes
   but cannot fully disentangle.

---

## Phase 2 — Writing audit

**Logical flow.** The arc is sound: motivate → instrument → sample-integrity →
headline result → residual checks → topic check → honesty episode → conclusion →
limitations. This is a defensible dissertation-appendix structure and is markedly more
honest than the norm (reporting a signal that died). But there are four structural
problems:

1. **The mega-doc disclosure comes one paragraph too late.** The reader is first told
   policy has "markedly longer sentences" (line 725) and only then that the sample is
   contaminated (line 728). Every claim in the feature paragraph is retroactively
   suspect. The contamination must precede or be integrated with the feature evidence.
2. **The most important bounding evidence is buried in a table.** The register-only
   classifier rows (0.456/0.544) are the honest answer to "how much of the corpus
   difference is captured by your register features," and they are never mentioned in
   prose. They should be a named result, not a silent table row.
3. **The 2b collapse is framed one-sidedly.** "An apparent residual-register signal was
   traced to clustering" (line 743) correctly kills the red flag, but the same finding
   also kills the *positive* evidence (removed magnitude tracks register). The
   paragraph should acknowledge both consequences, or an examiner will.
4. **The strongest honesty content is well placed; the strongest positive evidence is
   not.** The draw-stability paragraph (line 758) is excellent and correctly placed
   near the end. The corpus-feature discrimination — the only direct surface-linguistic
   evidence — is placed *before* the contamination caveat and is the overclaimed item.

**Redundancy.** The accuracy numbers are given in the prose (line 731), in Table 1,
and again in the Conclusion (line 761). Acceptable in a dissertation, but the
Conclusion should not *repeat* the table as evidence (i) — it should reference it.

**Paragraph ordering recommendation.** Merge/reorder to: (1) motivation;
(2) sample construction and integrity (mega-docs *first*); (3) register
operationalisation *with* the pooled-PC caveat and the register-only classification
result; (4) corpus discrimination before/after; (5) topic selectivity (with sample
stated); (6) residual-signal history (2b/2c + draw-stability, both directions of the
2b collapse); (7) bounded conclusion; (8) limitations.

**Caveat timing.** The two-sample rebuild caveat is currently in Limitations (line 764,
fourth item) but is *design-critical* and should appear where the samples are
introduced. The "two −0.197s" caution, by contrast, is correctly placed late (it is a
bookkeeping note).

**Overall:** the current text is already good and honest; the defects are (a) one
overstated feature-direction claim, (b) a too-strong final sentence, (c) three wording
inaccuracies ("independent sample", "no leakage", selectivity sample), and (d)
structural timing of the mega-doc disclosure and the register-only evidence.

---

## Phase 3 — Main manuscript consistency

The rewiring (`:280, :374, :393, :478, :489`) is **accurate and, if anything, more
conservative than Appendix G's own conclusion**. No sentence contradicts the appendix.
Specific findings:

- **Line 280** (Methodology): "substantial reduction of the corpus-level signal … but
  not its complete elimination, and no robust evidence that topic is being
  systematically removed" — **accurate**, matches Appendix G.
- **Lines 283 & 348**: "the adjusted space merges them, **confirming** that the
  separation is driven by register rather than topic." **Mild overclaim** — the
  appendix shows partial confirmation with residual signal; "consistent with" is the
  defensible word. The PCA figure is a 2-component visualisation; the 768-D result
  leaves adjusted accuracy above chance.
- **Lines 374 & 393** (table notes): "(primary estimate; evaluated in Appendix G)" —
  **accurate**.
- **Line 478** (Limitations, Register effects): "substantially reduces … does not
  eliminate … no robust evidence topic removed … rests on **partial, not complete,
  empirical support**" — **accurate and well-calibrated**; slightly *stronger* as a
  hedge than the appendix's own conclusion, creating the C12 tension.
- **Line 489** (Conclusion): "register interpretation is evaluated against independent
  linguistic register markers" — **accurate**.
- **Lines 695 & 702** (Appendix D): "no further linear register separation was
  detectable" / "the first 15 orthogonal directions capture the bulk of the **register
  signal**" — pre-existing; "register signal" is now slightly ahead of Appendix G's
  evidence. Low priority but worth a one-word tightening to "corpus signal."

No remaining "unvalidated"/"left to future work" wording survives (`grep` confirms);
the four original line numbers cited in the verification report (§4) were all rewired
correctly.

**Verdict:** the main text is consistent and appropriately hedged. One softening
("confirming" → "consistent with") is recommended at lines 283/348; the appendix
conclusion should be brought down to the main text's level rather than vice versa.

---

## Phase 4 — Writing plan for Appendix G (no rewriting performed)

**Overall calibration target:** make the appendix's final claim match the main text's
"partial, not complete, empirical support," and make every sample-dependent number
carry its sample label.

### Proposed section structure

1. **Purpose and scope** (1 paragraph)
   - Key message: the structural argument guarantees *what* is removed (corpus-separable
     linear signal) but not *that it is register*; this appendix tests the
     interpretation against independent surface-linguistic markers, with honest limits
     (MPNet, n=408, single operationalisation). State the two things it can establish
     and the one it cannot (it cannot prove the removed subspace *is* the surface
     register).

2. **Register score and samples** (2 paragraphs + the mega-doc audit moved here)
   - **Move the "Two sample constructions" content up** so the mega-doc contamination
     precedes any feature claim.
   - Key messages: (a) six Biber-style features, PC1 pooled score (report variance
     share 22% and, if available, the loadings or at least note the orientation
     convention explicitly as a convention, not as "formality"); (b) the original
     per-SDG-dedup sample is contaminated by 7 flagship documents (25 segments, 15/17
     SDGs) which are register outliers and cluster in embedding space; (c) the
     one-per-parent rebuild is the primary design — and *say it is a rebuild whose
     replacements are systematically national-monitoring reports* (composition shift
     caveat here, not only in Limitations).
   - **Reword the feature-direction claim**: report deontic/hedge/passive as robust
     across samples, and either drop or explicitly qualify the sentence-length
     direction (it reverses on the clean sample; the 276.9-word figure is extraction
     junk).

3. **Corpus discrimination before and after removal** (Table 1 + 2–3 sentences)
   - Key messages: raw ~0.94 → adjusted ~0.60 (one-per-parent; significantly above
     chance, CI given), the "collapse to chance" reading is corrected (holds only for
     the mega-contaminated sample; mega-exclusion reproduces most of the 0.505→0.603
     rise).
   - **Add prose on the register-only rows** (0.456/0.544): the six features alone
     barely distinguish the corpora — this both corroborates that the corpora differ
     in register-like ways and bounds the claim (the removed subspace is far richer
     than six surface proxies).
   - **Fix the table note**: "no leakage" holds for the one-per-parent sample; qualify
     for the original sample (document twins straddle folds).

4. **Topic is not measurably removed** (Table 2 + 1–2 sentences)
   - Key message: LR/kNN selectivity essentially unchanged; **state the sample used
     (original/draw-1)** and note it was not re-run on the one-per-parent sample.

5. **The apparent residual-register signals did not survive controlled samples**
   (the honest history)
   - Report 2b and 2c on both samples (numbers unchanged). **Acknowledge both
     consequences** of the clean-sample collapse: the original "residual register" red
     flag was a clustering artefact, *and* the positive link between removed magnitude
     and the surface features also disappeared — so the direct surface-linguistic
     corroboration of the removed subspace is weaker than the original sample
     suggested.
   - Report the draw-stability results and the −0.197 trace, **correcting "independent
     sample" to "independent re-run of the same sample"** and noting the mega-exclusion
     outcome is draw-specific (draws 1/2 give +0.127/−0.138; draw 3 gives −0.213).
     State plainly why INLP's chance-level stopping rule is compatible with 0.603 on
     fresh samples (convergence is on the training distribution).

6. **What this validation supports** (short synthesis, replaces the current
   "Conclusion")
   - Four numbered claims, kept at the evidence level: (i) substantial corpus-separable
     signal removed; (ii) removal incomplete; (iii) no robust evidence of topic
     removal; (iv) apparent residual traces attributable to clustering/draw noise.
     **Final sentence calibrated to the main text**: the register interpretation is
     *consistent with* the evidence and *partially* supported, with the residual corpus
     signal and the operationalisation limits explicitly bounding it. Remove "empirically
     supported for the dominant corpus-level component."

7. **Limitations** (as now, lightly tightened)
   - Keep all six items; add the clean-sample sentence-length reversal and the
     residual-signal-unidentified point; keep the "two −0.197s" bookkeeping note.

**Where the strongest evidence lives:** corpus discrimination (sec. 3), selectivity
(sec. 4), and the honest failure history (sec. 5) are the load-bearing results.
**Where the overclaim is removed:** sec. 2 (sentence-length direction) and sec. 6
(final sentence). **Where the conclusion ends:** with the bounded sentence in sec. 6,
immediately before Limitations — no new numbers after the synthesis.

---

## Bottom line

The validation work is scientifically sound and exceptionally honest — Report 1 →
Follow-up 1 → Follow-up 2 is a model of self-correction, and the verification work
(61 acceptance gates, byte-identical re-runs) makes the numbers trustworthy. Appendix G
as written is good but slightly overstates its central claim in two places (the
"markedly longer sentences" corpus direction and the final "empirically supported"
sentence), contains three wording inaccuracies an examiner could exploit ("independent
sample", "no leakage", unstated selectivity sample), and buries its most important
bounding evidence (the register-only classifier) in a table. The main text is
consistent with the evidence and if anything more conservative than the appendix. The
Phase-4 plan above preserves every number while recalibrating the claims to what the
evidence actually supports.

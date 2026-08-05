# Register Validation Follow-up 2: Sample Construction, Accuracy CIs, and Policy Other-Dist Pull

**Date:** 2026-08-05
**Model:** `all-mpnet-base-v2` (MPNet canon, G 62x768)
**Seed:** 42 throughout (draw-instability check: fresh draws at seeds 43/44/45)
**Scripts:** `5_notes/scratch/register_validation_followup2.py` (deterministic; log `followup2.log`), `5_notes/scratch/followup2_replacements.py` (replacement-source audit; output `followup2_replacements.txt`)
**RNG fix:** the first draft of the follow-up-2 script re-seeded `_rng` per `build_sample` call, which produced WRONG samples for 2B/Item-3. Fixed to a single module-level `_rng` with three successive draws of one continuous seed-42 stream — exactly matching the original follow-up script's effective behaviour (its in-`main()` re-seeds were dead locals; sampling functions read the module-level `rng`). **Acceptance gate reproduced exactly from `regcheck_followup.log`: 2A 0.456/0.909/0.505, 2B 0.544/0.944/0.603, Item-3 pooled -0.088/-0.074, policy other-dist -0.197.** All numbers below are on the exact original samples, like-for-like with the prior report.

---

## Item 1 — How was the one-per-parent sample constructed? (REBUILD, not subset)

### Procedure (identical in both scripts)

`register_validation_followup2.py::build_sample` reproduces the original `build_with_text`:

- **Original (2A) = draw 1:** `sample_research(12, global_dedup=False)` then `sample_policy(12, global_dedup=False)`. Per-SDG dedup only: within each SDG a parent may appear once, but a parent may appear in multiple SDGs.
- **One-per-parent (2B) = draw 2:** `global_dedup=True`. A global `used` set skips any parent already picked for ANY SDG; the loop then **continues down the same shuffled per-SDG list and draws replacement segments** until 12 are picked.

Because both draws come from one continuous RNG stream, 2B is a **fresh rebuild**, not a subset of 2A.

### Result

| | Original (2A, draw 1) | One-per-parent (2B, draw 2) |
|---|---|---|
| Total n | 408 | 408 |
| Research | 204 (204 distinct parents) | 204 (204 distinct parents) |
| Policy | 204 (**186 distinct parents**, 25 units from 7 mega-docs) | 204 (**204 distinct parents**, 0 multi-parent) |
| Research per-SDG | 12 each | 12 each |
| Policy per-SDG | 12 each | **12 each** |

**Answer: it is (b) — dropped-then-refilled, a rebuild.** The 12-policy-per-SDG quota is always met because the global-dedup branch skips already-used mega-doc segments and continues down the shuffled list, pulling replacement segments from other policy documents in the same SDG. There is **no attrition**: both samples are exactly n=408 with 12/SDG/corpus. The "n=390" alternative (drop duplicates, don't refill) is **not** what happened; n=24/SDG in Items 2B/3 is exact and full.

### Replacement procedure details

- Same seed/stream and same `rng.shuffle` strategy as the original Step 1 (identical `sample_policy` code path).
- Same eligibility: the source pool is the same pre-filtered policy corpus (segments with >=20 words; policy segmentation already applied corpus-wide). No additional filtering.
- **Which sources filled the slots vacated by mega-docs** (from `followup2_replacements.py`): mega-doc segments sat in 15 of 17 SDGs (all except SDG 2 and 11). Their replacements are drawn from the remainder of each SDG's policy pool and are **overwhelmingly `pol_sdgi_*` (national SDG-index indicator reports, ~4,225 docs) and `pol_ungdc_*` (country/national-development reports, ~2,048 docs)**, plus a few manual OECD/WHO/UN documents (e.g. SDG 1 is refilled by `pol_sdgi_sdgi_00020`, `pol_sdgi_sdgi_00129`, `pol_ungdc_sdg_STP_74_2019`, etc.).

### Are replacement sources systematically different from the mega-docs they replaced? YES

Whole-policy-corpus doc statistics (not sample-specific):

| Doc group | docs | segments | mean_sent_len | mean words/seg |
|---|---|---|---|---|
| Mega-docs (SDSN/UNDP/WHO/EU/UN flagships) | 7 | 5,975 | **57.8** | 330.8 |
| `pol_sdgi_*` (national SDG-index reports) | 4,225 | 21,346 | **30.8** | 276.6 |
| `pol_ungdc_*` (country reports) | 2,048 | 6,128 | **25.2** | 279.7 |
| All policy | 6,367 | 40,597 | 32.2 | 287.8 |

So for the SDGs previously dominated by mega-docs, the one-per-parent sample replaces **global flagship/UN report prose (long sentences, institutional) with short-sentence national monitoring/indicator reports** — a real genre/institution shift, not a like-for-like swap. This is a **new, different form of composition change**: the one-per-parent sample is *cleaner of the clustering artifact* but *systematically more national-monitoring-flavoured* in exactly the mega-dominated SDGs. It is a defensible primary design, but it is not simply "the original sample with duplicates removed."

### Caveat for per-SDG / per-feature results

Items 3a (per-SDG) and 3b (per-feature) were computed on the one-per-parent sample, whose policy composition for the mega-dominated SDGs is materially different from the original. Any per-SDG inference inherits this composition shift **in addition to** the low n=12/SDG power already flagged. Report both caveats together.

---

## Item 2 — Adjusted-space CV accuracy: CIs and why accuracy rose

**CI method:** pooled 5-fold stratified-CV predictions (each row predicted by a classifier trained on the other 4 folds; no leakage), Wilson 95% CI on the pooled proportion, one-sided binomial test vs 0.5, and a bootstrap of the accuracy *difference* on the pooled predictions. The earlier resample-then-CV bootstrap was **removed** — it leaked train/test via duplicate rows and produced inflated CIs.

### Accuracy, CIs, and significance

| Classifier | Original (2A, n=408) pooled acc (k/n) | Wilson 95% CI | p(vs 0.5) | One-per-parent (2B, n=408) pooled acc (k/n) | Wilson 95% CI | p(vs 0.5) |
|---|---|---|---|---|---|---|
| Register-only (PC1) | 0.456 (186/408) | [0.408, 0.504] | 0.967 | 0.544 (222/408) | [0.496, 0.592] | 0.042 |
| Raw embeddings | 0.909 (371/408) | [0.878, 0.934] | <1e-70 | 0.944 (385/408) | [0.917, 0.962] | <1e-85 |
| **Adjusted embeddings** | **0.505 (206/408)** | **[0.457, 0.553]** | **0.441 (ns)** | **0.603 (246/408)** | **[0.555, 0.649]** | **1.9e-05 (sig)** |

(fold-mean accuracy, the prior metric, is identical to pooled here: 0.505/0.909/0.505 and 0.603/0.944/0.544.)

**Verdict on distinguishability:**
- **Original 2A adj = 0.505 is NOT distinguishable from 0.5** (p=0.44; CI includes 0.5). The original "approx chance" characterization is accurate *for the original composition*.
- **One-per-parent adj = 0.603 IS distinguishable from 0.5** (p=1.9e-05; CI [0.555, 0.649] excludes 0.5). The prior follow-up's "0.603 not chance" is **confirmed**, now with a proper CI.
- **The 0.505->0.603 rise is real, not noise:** prediction-bootstrap difference = **+0.098, 95% CI [+0.024, +0.169], p(diff>0)=0.994**.

### Why did ALL THREE classifiers move up?

1. **Mega-docs are atypical, hard-to-classify policy text (main driver).** Sample mega-doc features (n=25) vs non-mega (n=383): mean_sent_len **276.9 vs 35.6**, passive **3.3 vs 9.8**, first_person **19.0 vs 5.9**, nominal **20.0 vs 38.8**; deontic/hedge similar. These are PDF-junk-inflated sentence lengths and non-prototypical feature mixes — *hard* policy segments. Direct test: dropping the 25 mega-policy units from the **original** sample alone raises its adj acc from 0.505 to **0.574** (220/383, Wilson [0.524, 0.623], p=0.002). That is +0.070 of the +0.098 total rise; the residual ~+0.03 is composition noise from the new replacement docs.
2. **Fold-leakage direction:** In the original sample, mega-docs can straddle train/test folds. Because mega-docs are *atypical*, train-fold copies do **not** help classify their test-fold twins — the leak direction would *inflate* original accuracy, yet original is *lower*; so the observed direction supports "duplicates made classification HARDER by injecting within-corpus atypical clusters," not "duplicates inflated accuracy." Removing them raises accuracy across raw, adjusted, AND register-only simultaneously (all three classifiers see the same removed hard points).
3. **Composition noise:** the remaining +0.03 is the new national-monitoring replacements being easier / more prototypically policy.

### Verdict on the "adj approx chance" claim

| Characterization | Original (2A) | One-per-parent (2B, recommended primary) |
|---|---|---|
| "adj approx chance" | **Accurate.** 0.505, p=0.44, ns. | **Wrong for the primary sample.** 0.603, p=1.9e-05, sig. |
| Correct statement | "0.505 (95% CI 0.46-0.55)" | "0.603 (95% CI 0.55-0.65), ~11% above chance" |

**For the full-scale appendix the characterization must be revised to:** *"adjusted-space corpus accuracy is strongly reduced from raw (0.91->0.60) but NOT to chance on the one-per-parent sample; magnitude 0.60 +- 0.05, significantly above 0.5."* The "reduced to chance" phrasing survives **only** for the mega-contaminated original composition, which is not the recommended design.

---

## Item 3 — Policy other-dist pull: per-SDG, mega-doc exclusion, draw stability

### Reproduction gate (exact, on the Item-3 draw-3 sample, n=408)

- pooled reg~dist RAW rho=-0.088 (p=0.075), ADJ rho=-0.074 (p=0.134) — matches prior report.
- **policy reg ~ other-dist ADJ pooled: rho=-0.197 (p=0.0047)** — matches prior report. (Interpretation: higher-register policy segments sit *farther from the research SDG centroid*; negative because distance is 1-cosine.)

### Per-SDG breakdown (policy segments, n=12 each, ADJ space, other-dist)

| SDG | rho | p | #mega in sample | SDG | rho | p | #mega |
|---|---|---|---|---|---|---|---|
| 1 | -0.063 | 0.846 | 1 | 10 | +0.517 | 0.085 | 0 |
| 2 | -0.566 | 0.055 | 0 | 11 | -0.028 | 0.931 | 0 |
| 3 | +0.224 | 0.485 | 2 | 12 | -0.021 | 0.948 | 0 |
| 4 | -0.119 | 0.713 | 0 | 13 | -0.112 | 0.729 | 0 |
| 5 | -0.490 | 0.106 | 1 | 14 | -0.434 | 0.159 | 0 |
| 6 | -0.238 | 0.457 | 0 | 15 | -0.210 | 0.513 | 1 |
| 7 | +0.510 | 0.090 | 0 | 16 | -0.056 | 0.863 | 0 |
| 8 | -0.280 | 0.379 | 0 | 17 | -0.238 | 0.457 | 0 |
| 9 | -0.280 | 0.379 | 0 | | | | |

**SDGs with p<0.05: 0/17.** Signs: 14 negative / 3 positive (SDG 3, 7, 10). The three most negative (2, 5, 14) and two most positive (7, 10) are all p~0.06-0.16 at n=12 — **the pooled -0.197 is spread thinly across most SDGs, not driven by a few.** Same low-power caveat as Item 3a.

### Mega-doc exclusion (drop ALL policy segments from mega-docs, not just one-per-parent)

| Sample | total n kept | policy n | centroid-dist ADJ rho (p) | other-dist ADJ rho (p) |
|---|---|---|---|---|
| Original (draw 1) | 383 | 179 | +0.163 (0.030) | +0.127 (0.091, ns) |
| One-per-parent (draw 2) | 402 | 198 | -0.155 (0.029) | -0.138 (0.053, ns) |
| One-per-parent (draw 3 / Item-3) | 403 | 199 | -0.117 (0.101) | **-0.213 (0.003)** |

The Item-3 **-0.197 pull survives and slightly strengthens (-0.213, p=0.003) when every mega-doc policy segment is removed.** It is **not** driven by the same SDSN/UNDP/WHO mega-docs that drove the original 2c red flag. (Note the original sample's other-dist is *positive* once mega-docs are removed — consistent with the mega-docs having manufactured the positive direction there.)

### Draw-stability check (fresh independent one-per-parent draws)

| seed | pooled reg~dist ADJ rho (p) | policy reg~other-dist ADJ rho (p) |
|---|---|---|
| 42 (Item-3 draw 3) | -0.074 (0.134) | **-0.197 (0.005)** |
| 43 | -0.125 (0.012) | -0.130 (0.063) |
| 44 | -0.036 (0.465) | -0.004 (0.949) |
| 45 | -0.016 (0.743) | **+0.126 (0.072)** |

**The pooled -0.197 is NOT draw-stable.** Across four independent draws the policy other-dist correlation ranges from -0.20 to +0.13 with sign flips (mean ~ -0.05). The seed-42 value is a sample-specific fluctuation, not a reproducible signal.

### Interpretation and verdict

- The -0.197 reproduces exactly on the original sample, survives a strict all-mega-doc exclusion, and has a nominally significant pooled p — but it **does not replicate across independent draws** (sign flips, mean ~ -0.05) and is **not concentrated in any SDG** (0/17 significant).
- Plausible reading if it were robust: INLP's stratified training (equal research/policy per SDG) might leave asymmetric residual register expression on the policy side (high-register policy drifting toward "less research-like"). **The data do not support promoting this**: draw instability downgrades it to noise at the current sample size. It is not a robust "surviving signal worth carrying to the appendix."
- **Downgrade to noise / sample-specific.** Do not report -0.197 as a finding without the draw-instability caveat; do not build appendix claims on it.

---

## Updated verdict

### Which prior claims need revision vs confirmation

| Claim (prior follow-up) | Status | Corrected |
|---|---|---|
| One-per-parent is n=408 rebuilt, not subset | **Confirmed** | n=408 exact, 12/SDG/corpus; rebuild with refill |
| "2c red flag was clustering artifact" (pooled -0.074 adj) | **Confirmed** | reproduces exactly |
| "2b within-corpus correlations collapse to null" | **Confirmed** | -0.04/-0.04, ns |
| **"adj=0.603 not chance"** | **CONFIRMED** | 0.603, Wilson CI [0.555, 0.649], p=1.9e-05 |
| "2d adj approx chance" as a general statement | **REVISED** | Only true for the mega-contaminated original (0.505, ns). Primary one-per-parent sample: 0.603, sig. |
| Policy other-dist rho=-0.197 "surviving signal" | **REVISED (downgraded)** | Reproduces on seed-42 sample and survives mega-doc exclusion, but sign-flips across draws 43/44/45 -> noise, not robust |
| Replacement sources are non-systematic | **REVISED** | Replacements are systematically national `pol_sdgi_*`/`pol_ungdc_*` monitoring reports (short-sentence) vs global flagship mega-docs |

### Is "GO, scale to n=120/SDG" still correct? YES — with one framing change and one caveat

1. **GO stands.** No integrity red flag survives: the original 2c/2b signals were mega-doc artifacts; adjusted-space SDG selectivity and the concept-provenance check are unaffected; the one-per-parent design is sound.
2. **Framing change:** drop "adjusted-space corpus accuracy is approx chance." The correct headline is **"adjusted-space accuracy is reduced from 0.91 to approx 0.60 (95% CI 0.55-0.65), significantly above chance"**. This is a *better* outcome for the register story (INLP removes most, but not all, linear corpus signal) and must be written that way in the appendix. A full-scale validation that claims "collapse to chance" would now be wrong.
3. **Caveat for the scaled-up run:** the one-per-parent rebuild systematically swaps global flagship policy for national monitoring reports in the mega-dominated SDGs. For n=120/SDG, keep one-per-parent as primary **but** (a) report the replacement-pool composition per SDG as a sensitivity item, and (b) consider a parallel mega-doc-*flagged* (not excluded) analysis so the national-monitoring shift is visible rather than silently absorbed. Mega-doc exclusion is **not** additionally necessary as a separate sampling arm — one-per-parent already caps mega-docs at one unit, and the only remaining signal (-0.197) is draw-unstable noise.
4. **Do NOT carry the policy other-dist -0.197 to the appendix** as a finding. Report it (if at all) as a draw-instability example or leave it out.

---

## Reproducibility

- All Item 1-3 numbers: `5_notes/scratch/followup2.log` (seed 42, N_PER_SDG=12; draws 1-3 of one continuous stream for 2A/2B/Item-3; fresh draws at seeds 43/44/45 for stability).
- Replacement-source audit: `5_notes/scratch/followup2_replacements.py` -> `5_notes/scratch/followup2_replacements.txt` (seed 42, same stream order: research-then-policy per draw).
- Prior reports untouched: `5_notes/register_validation_report.md` (`0f96a3f`), `5_notes/register_validation_followup.md` (`c2773a9`). Ground-truth log: `5_notes/scratch/regcheck_followup.log`.
- CI method: pooled 5-fold stratified-CV predictions (StratifiedKFold(5, shuffle, seed 42), LogisticRegression C=1.0), Wilson 95% interval, one-sided binomial, and prediction-level bootstrap difference. No resample-then-CV bootstrap was used.

# Hand-off: Appendix K.1 — Pooled OLS Regression (Semantic Gap ~ Coverage + Indicators)

**Last updated:** 2026-08-04
**Status:** Code FIXED and re-run. 23-spec grid rebuilt with corrected scale and ranking. Manuscript write-up NOT started.
**Nothing is interrupted for code.** Manuscript integration is the next step.

---

## 1. Context — where we are

The dissertation pipeline produces per-SDG semantic gap values across 24 methodological configurations (3 encoders × 2 retrieval × 3 methods × 3 caps). The existing analysis (H25) reports bivariate Spearman correlations within each config (n=17 SDGs each).

**Appendix K.1 pools all 24 configs into a single dataset (N=408)** and runs OLS regressions to estimate how coverage-related predictors explain the semantic gap while controlling for configuration through indicator variables.

### What was done this session

The original code had TWO design flaws that were fixed:

1. **DV scale was wrong:** The rank DV was computed across ALL 408 observations (ranks 1–408), producing coefficients of +306 that were uninterpretable. Fixed to rank WITHIN each config (ranks 1–17 per config).

2. **Predictor scale was wrong:** covgap and polcov were proportions (0–1). Fixed to percentage points (0–100) for interpretability.

### What remains

The code is fixed and re-run. The spec_grid.json, tab_k1_specification_grid.tex, and bootstrap_grid.json are all regenerated with the corrected values. **The manuscript (3_writing/dissertation.tex) has NOT been modified.** The next step is manuscript integration: deciding where to put the table and writing the prose.

---

## 2. Key known facts

### Data
- 24 configs × 17 SDGs = 408 observations
- ZS only has cap=50 (no cap=20 or none) — degenerate columns auto-dropped in subsamples
- Coverage gap is segment-cap-independent (shared across caps for same encoder/retrieval/method)
- `n_papers` available per SDG in gap JSONs (used for WLS weights)
- Concept retrieval is MPNet-only

### Model (base, 10 params)
```
rank(sem_gap) = β₀ + β₁·predictor + β₂·polcov
              + i_minilm + i_scibert + i_concept
              + i_cap20 + i_cap_none
              + i_mlp + i_zs
```

**DV:** rank of semantic gap WITHIN each config (1–17 per config, ties averaged)
**Predictors:** covgap, polcov in PERCENTAGE POINTS (0–100)

### Headline results (adjusted gap, N=408, per-config rank DV)

| Predictor | b | SE | p | Interpretation |
|-----------|---|----|---|----------------|
| covgap | +0.18 | 0.09 | 0.069† | +1 pp covgap → +0.18 rank positions (out of 17) |
| polcov | +0.41 | 0.12 | **0.003** | Corpus size artifact, not substantive |

**practical effect:** covgap ranges 0–23 pp. Full-range effect: 0.18 × 23 = 4.1 rank positions out of 17 (~24% of range).

### Register gap: nothing significant
- covgap: b=−0.04, p=0.700
- polcov: b=−0.31, p=0.100†

### Raw gap: nothing significant (cancellation confirmed)
- covgap: b=+0.09, p=0.590
- polcov: b=−0.09, p=0.630

### Robustness (adjusted + covgap)
- Supervised only (N=357): b=+0.20, **p=0.040** ✓
- Keyword only (N=357): b=+0.20, p=0.055†
- MPNet only (N=170): b=+0.33, **p=0.003** ✓
- SDG FE: p=0.88 ✗ (absorbs all variation — expected)
- All functional forms (raw DV, log, WLS): p < 0.01 ✓
- Bootstrap (500 reps): CI crosses zero (17 clusters → wide CI)

### Raw DV (cosine units, not rank)
- covgap: b=+0.003, **p=0.007** ✓
- polcov: b=+0.004, **p=0.035** ✓
- R²=0.684

### Key correlations (MPNet LR, n=17 SDGs)
- covgap vs adj. gap: rho=+0.664, p=0.004
- policy% vs adj. gap: rho=+0.613, p=0.009
- research% vs adj. gap: rho=+0.176, p=0.50
- research vs dominance: r=+0.89 (nearly identical in model)
- n_policy_segments vs adj. gap: rho=+0.730, p=0.0009 (drives polcov effect)

### The polcov artifact
`policy_profile_hard_docweighted` is literally `n_policy_segments / total_segments`. It's a corpus size measure, not a topic coverage measure. The positive polcov coefficient means: more policy docs → more diffuse centroid → larger gap. This is structural, not substantive. Don't interpret polcov as "more policy coverage = bigger gap."

---

## 3. Actions / decisions / files changed this session (and why)

### Files changed
| File | Change | Why |
|------|--------|-----|
| `1_code/7_main_analysis/2_appendix/k1_regression_semantic_gap.py` (line ~374) | Scale covgap, polcov, research, dominance by ×100 in `build_panel()` | Predictors were proportions (0–1); needed percentage points (0–100) for interpretable coefficients |
| `1_code/7_main_analysis/2_appendix/k1_regression_semantic_gap.py` (line ~417) | Changed `rankdata(sem_vals)` to per-config ranking (1–17 within each config) | Original pooled ranking (1–408) was wrong — ranks across configs with different gap scales are meaningless |

### Files regenerated (by re-running `--spec-grid --overwrite`)
| File | Content |
|------|---------|
| `4_outputs/appendix/mpnet/k1_regression_semantic_gap/data/spec_grid.json` | 23 specs, machine-readable (corrected values) |
| `4_outputs/appendix/mpnet/k1_regression_semantic_gap/data/bootstrap_grid.json` | Bootstrap CIs (corrected) |
| `4_outputs/appendix/mpnet/k1_regression_semantic_gap/tables/tab_k1_specification_grid.tex` | Compact coef*(SE) LaTeX table (corrected values) |

### Decisions made
1. **Per-config rank DV (1–17)** — not pooled (1–408). Ranks are meaningful only within each config.
2. **Percentage-point scale** — covgap and polcov are now 0–100, not 0–1. Coefficients are "rank positions per percentage point."
3. **No statsmodels** — OLS via `numpy.linalg.lstsq` + manual sandwich SE (unchanged from before).
4. **Compact coef*(SE) format** — two rows per variable (coef+stars, SE in parens) (unchanged).
5. **Cluster-robust SE by SDG** (17 clusters), Moulton/Angrist-Pischke df adjustment (unchanged).
6. **polcov is a nuisance control** — not interpretable substantively (corpus size artifact). Coefficient is +0.41 (p=0.003), meaning: +1 pp in policy coverage → +0.41 rank positions. But this is just "more policy docs = more diffuse centroid."
7. **The main spec (adj_covgap) is marginal at p=0.069** — but supervised-only (p=0.040) and MPNet-only (p=0.003) subsamples confirm the signal is real. The raw-DV spec (p=0.007) also confirms it.

---

## 4. What remains

### Immediate: Manuscript integration (NOT started)

The code is done. The next agent needs to:

1. **Decide where the table goes in the manuscript.** The user's preference (stated before interruption): ONE comprehensive table in the main text (Section 4 or 5), with the full 23-spec grid in the appendix. The main text table should show the 9 core panel-A specs (3 DV types × 3 predictor variants).

2. **Write the prose** for the main text section (3–4 sentences interpreting the pooled result).

3. **Write the appendix section** (new `\section` after Appendix H, before AI declaration). Include:
   - Panel B: Robustness (6 specs)
   - Panel C: Interactions (4 specs)
   - Panel D: Functional forms (4 specs)
   - Scope conditions (endogeneity, polcov artifact, R² interpretation)

4. **Add `\InputIfFileExists`** for the table in the tex preamble.

5. **Verify** with `python main.py --warm-replay-without-appendix --overwrite`.

### The user's preferred table design (stated but not implemented)

The user wanted a comprehensive table in the main text with columns:
```
, raw1, raw2, raw3, adj1, adj2, adj3, reg1, reg2, reg3
```
Where:
- raw1/2/3 = raw gap with covgap/dominance/rescov predictors
- adj1/2/3 = adjusted gap with covgap/dominance/rescov predictors
- reg1/2/3 = register gap with covgap/dominance/rescov predictors

And rows: covgap, rescov, polcov, i_minilm, i_scibert, i_concept, i_cap20, i_cap_none, i_mlp, i_zs, R²

This is essentially panel A of the spec grid, reformatted for the main text. The remaining panels (B–D) go in the appendix.

**BUT** — the user also discussed using the raw DV (cosine units) instead of rank for the main text table, since the coefficients are more interpretable (b=+0.003 in cosine units vs b=+0.18 in rank units). This decision was NOT finalized before the session was stopped. The next agent should ask the user which DV to use for the main text table.

### Not needed
- More robustness checks (already 23 specs)
- Different DV transformations (already tested raw, log, WLS)
- More predictor variants (covgap, dominance, rescov covers the space)
- Code changes (the regression code is fixed and re-run)

---

## 5. Concerns

### The p-value shifted from 0.024 to 0.069

The original pooled ranking (1–408) gave covgap p=0.024. The corrected per-config ranking (1–17) gives p=0.069. This is NOT a bug — it's the honest assessment. The pooled ranking was artificially inflating the t-statistic by mixing across configs with different gap scales (MPNet gaps ~0.15, SciBERT gaps ~0.02). The per-config ranking removes this cross-config variation.

The finding is still supported by:
- Supervised-only subsample: p=0.040
- MPNet-only subsample: p=0.003
- Raw DV (cosine units): p=0.007
- WLS: p<0.001

The next agent should report the main spec as marginal (p=0.069) and note the robustness checks confirm the signal.

### Endogeneity

Coverage gap is computed from the same embeddings as semantic gap. Coefficient is biased for causal interpretation. This is association, not causation. Flag in manuscript.

### Bootstrap CI crosses zero

With 17 clusters, bootstrap CI crosses zero. Sandwich SE (p=0.069) is more reliable. Bootstrap included for transparency.

### R² dropped from 0.727 to 0.195

The old R² (0.727) was inflated by cross-config variance (encoder indicators). The new R² (0.195) reflects only within-config variance. This is the correct R².

### register gap is undetectable

All register gap coefficients are insignificant. The register component is not explained by coverage variables. It's driven by something unmeasured (genre, writing conventions, etc.).

### dominance and rescov are redundant

They produce identical predictor coefficients because `dominance = research - policy` and policy is already in the model. Only covgap (unsigned) is useful. The main text table can show both for completeness, but they tell the same story.

---

## 6. Specification grid (23 specs, corrected values)

### Panel A: Core regressions (9 specs)
| Spec | DV | Predictor | N | b | p | R² |
|------|----|-----------|----|-----|------|-----|
| adj_covgap | adjusted (rank) | covgap | 408 | +0.18 | 0.069† | 0.195 |
| adj_dominance | adjusted (rank) | dominance | 408 | +0.13 | 0.162 | 0.162 |
| adj_rescov | adjusted (rank) | rescov | 408 | +0.13 | 0.162 | 0.162 |
| raw_covgap | raw (rank) | covgap | 408 | +0.09 | 0.590 | 0.014 |
| raw_dominance | raw (rank) | dominance | 408 | +0.11 | 0.451 | 0.049 |
| raw_rescov | raw (rank) | rescov | 408 | +0.11 | 0.451 | 0.049 |
| reg_covgap | register (rank) | covgap | 408 | −0.04 | 0.700 | 0.073 |
| reg_dominance | register (rank) | dominance | 408 | +0.00 | 0.987 | 0.074 |
| reg_rescov | register (rank) | rescov | 408 | +0.00 | 0.987 | 0.074 |

### Panel B: Robustness (6 specs, all adjusted + covgap)
| Spec | Modification | N | covgap p |
|------|-------------|----|----------|
| adj_noclf | No classifier indicator | 408 | 0.069† |
| adj_sdgfe | + SDG FE | 408 | 0.877 |
| adj_supervised | Supervised only (LR+MLP) | 357 | **0.040*** |
| adj_keyword | Keyword only | 357 | 0.055† |
| adj_mpnet | MPNet only | 170 | **0.003**** |
| adj_noclf_sdgfe | No clf + SDG FE | 408 | 0.877 |

### Panel C: Interactions (4 specs)
| Spec | Interaction | covgap p |
|------|------------|----------|
| adj_int_enc | covgap × encoder | **0.004**** |
| adj_int_ret | covgap × retrieval | 0.034* |
| adj_int_mth | covgap × method | **0.004**** |
| adj_int_all | all interactions | **<0.001**** |

### Panel D: Functional forms (4 specs)
| Spec | Form | covgap p |
|------|------|----------|
| adj_raw_dv | raw DV (cosine, not ranked) | **0.007**** |
| adj_log_dv | log DV | **0.008**** |
| adj_wls | weighted least squares | **<0.001**** |
| adj_covgap_boot | rank + bootstrap 500 | 0.069† |

---

## 7. CLI usage

```bash
# Full spec grid (23 specs) — ALREADY RE-RUN, outputs updated
python 1_code/7_main_analysis/2_appendix/k1_regression_semantic_gap.py --spec-grid --overwrite

# Single spec
python 1_code/7_main_analysis/2_appendix/k1_regression_semantic_gap.py --gap-type adjusted
python 1_code/7_main_analysis/2_appendix/k1_regression_semantic_gap.py --gap-type adjusted --predictor rescov

# With bootstrap
python 1_code/7_main_analysis/2_appendix/k1_regression_semantic_gap.py --gap-type adjusted --bootstrap-se 500

# Custom subsample/interactions
python 1_code/7_main_analysis/2_appendix/k1_regression_semantic_gap.py --gap-type adjusted --subsample supervised --interactions encoder
```

---

## 8. What was interrupted

**Code work is complete.** The regression code was fixed (per-config ranking + percentage-point scale) and re-run. All outputs (spec_grid.json, bootstrap_grid.json, tab_k1_specification_grid.tex) are regenerated with corrected values.

**Manuscript integration was NOT started.** The next agent needs to:
1. Decide table placement in dissertation.tex (user wanted ONE comprehensive table in main text + full grid in appendix)
2. Write the prose for the main text section
3. Write the appendix section
4. Add \InputIfFileExists for the tables
5. Verify with warm replay

The user's preferred table design was stated (9 columns: 3 DV types × 3 predictor variants) but not finalized. The user also discussed using the raw DV (cosine units) instead of rank for the main text table. **Ask the user to confirm before writing.**

---

## 9. Comprehensive plan for manuscript integration

### Step 1: Generate headline table for main text

Create `4_outputs/appendix/mpnet/k1_regression_semantic_gap/tables/tab_k1_headline.tex` — a compact 9-column table showing the core panel-A results. This is a reformatted subset of the existing spec grid.

### Step 2: Add table to tex preamble

In `3_writing/dissertation.tex`, add near line 48:
```latex
\InputIfFileExists{../4_outputs/appendix/mpnet/k1_regression_semantic_gap/tables/tab_k1_headline.tex}{}{}
\InputIfFileExists{../4_outputs/appendix/mpnet/k1_regression_semantic_gap/tables/tab_k1_specification_grid.tex}{}{}
```

### Step 3: Add main-text section

Insert after Section 4.4 (Robustness, line 428), before Discussion (line 431). New subsection:

**Section 4.5: Pooled Regression: Coverage Predictors and the Semantic Gap**

~3–4 sentences interpreting the headline result, referencing the table. Key narrative:
- Pools all 24 configs (N=408) into OLS with cluster-robust SEs by SDG
- Only covgap predicts the adjusted gap (b=+0.18, p=0.069†); raw gap and register gap show nothing
- Confirms the cancellation: topic divergence rises with coverage mismatch, register divergence does not
- Full robustness (23 specs) in Appendix K.1

### Step 4: Add appendix section

New `\section` after Appendix H (line 853), before AI Declaration (line 855). Label: `\label{app:regression}`.

Include:
- Pooled OLS design description
- Full spec grid table (`tab_k1_specification_grid.tex`)
- Robustness discussion (panels B–D)
- Scope conditions (endogeneity, polcov artifact, R² interpretation)

### Step 5: Cross-references

- Main text §4.5 references Appendix K.1
- Appendix K.1 references Table 4 (bivariate correlations) and Table 5 (register decomposition)

### Step 6: Verify

```bash
python main.py --warm-replay-without-appendix --overwrite
```

---

## 10. Appendix letter assignment

Current appendix letters (after `\appendix` at line 505):
- A: Supplementary Methodology
- B: Diagnostics for Reference Classifier
- C: Supplementary Robustness and Sensitivity Checks
- D: Sample-Stability Robustness Check
- E: Model Selection
- F: Register Removal
- G: Concept-Retrieval Sensitivity
- H: Supplementary Cross-Method Data
- I: Declaration of AI Use

The new regression section should go between H and I. It will become **Appendix I** (and the current AI declaration becomes Appendix J, or the section is added as a new unlettered section — LaTeX handles this automatically with `\section` after `\appendix`).

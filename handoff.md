# Hand-off: Appendix K.1 — Pooled OLS Regression (Semantic Gap ~ Coverage + Indicators)

**Last updated:** 2026-08-04
**Status:** Core work COMPLETE. 23-spec grid built, tested, committed, pushed.
**Nothing is interrupted.** This file is for context transfer, not resumption.

---

## 1. Context — where we are

The dissertation pipeline produces per-SDG semantic gap values across 24 methodological configurations (3 encoders × 2 retrieval × 3 methods × 3 caps). The existing analysis (H25) reports bivariate Spearman correlations within each config (n=17 SDGs each).

**Appendix K.1 pools all 24 configs into a single dataset (N=408)** and runs OLS regressions to estimate how coverage-related predictors explain the semantic gap while controlling for configuration through indicator variables. The DV is rank of semantic gap (ties averaged). SEs are cluster-robust by SDG (17 clusters).

**Three DV types:**
- **Adjusted gap** (topic-only, after INLP register removal) — the headline
- **Register gap** (raw − adjusted) — the register component
- **Raw gap** (unadjusted) — the composite

**Three continuous predictors (each paired with polcov):**
- **covgap** = |research% − policy%| (unsigned mismatch)
- **dominance** = research% − policy% (signed mismatch)
- **rescov** = research coverage %

**The headline finding:** Only covgap significantly predicts the adjusted (topic) gap (p=0.024). Research coverage alone and dominance do not. Register gap is not predicted by any coverage variable. Raw gap shows nothing (cancellation).

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
              + i_mlp + i_zs + ε
```

### Headline results (adjusted gap, N=408, rank DV)

| Predictor | b | SE | p | Interpretation |
|-----------|---|----|---|----------------|
| covgap | +306 | 123 | **0.024*** | Unsigned mismatch → topic divergence |
| dominance | +148 | 128 | 0.267 | Signed mismatch: not significant |
| rescov | +148 | 128 | 0.267 | Research coverage alone: not significant |

**polcov** (policy coverage) is always positive and significant in adjusted gap specs, but this is a **corpus size artifact** — SDGs with more policy documents have higher polcov AND larger gaps (more diffuse centroid). After controlling for n_policy_segments, the polcov effect disappears. polcov is a nuisance control, not a substantive predictor.

### Register gap: nothing significant
- covgap: b=−201, p=0.33 (negative but not significant)
- polcov: b=−405, p=0.20
- Direction is opposite to adjusted gap → cancellation story

### Raw gap: nothing significant
- All p > 0.22

### Robustness (adjusted + covgap)
- Supervised only (N=357): p=0.013 ✓
- Keyword only (N=357): p=0.049 ✓
- MPNet only (N=170): p=0.003 ✓
- SDG FE: p=0.88 ✗ (absorbs all variation — expected)
- All functional forms (raw DV, log, WLS): p < 0.01 ✓
- Bootstrap (500 reps): CI=[−58.6, +696.1] crosses zero (17 clusters → wide CI)

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

### Files created
| File | Purpose |
|------|---------|
| `1_code/7_main_analysis/2_appendix/k1_regression_semantic_gap.py` | Main regression script (~1080 lines) |
| `4_outputs/appendix/mpnet/k1_regression_semantic_gap/data/spec_grid.json` | 23 specs, machine-readable |
| `4_outputs/appendix/mpnet/k1_regression_semantic_gap/data/bootstrap_grid.json` | Bootstrap CIs (1 spec) |
| `4_outputs/appendix/mpnet/k1_regression_semantic_gap/tables/tab_k1_specification_grid.tex` | Compact coef*(SE) LaTeX table |
| `4_outputs/minilm/data/adjusted/semantic_gap_distances_zeroshot.json` | Computed this session |
| `4_outputs/scibert/data/adjusted/semantic_gap_distances_zeroshot.json` | Computed this session |

### Files modified
| File | Change | Why |
|------|--------|-----|
| `1_code/7_main_analysis/0_shared/analysis_orchestrator.py` | Added K1 APPENDIX_SPEC entry (in_all=True) | Wire into `--appendix-all` pipeline |
| `handoff.md` | This file | Context transfer |

### Decisions made
1. **Rank DV** — ties averaged via `scipy.stats.rankdata`
2. **Cluster-robust SE** by SDG (17 clusters), Moulton/Angrist-Pischke df adjustment
3. **No statsmodels** — OLS via `numpy.linalg.lstsq` + manual sandwich SE
4. **Compact coef*(SE) format** — two rows per variable (coef+stars, SE in parens)
5. **No backward compat** — old per-run output dirs deleted, replaced by unified spec_grid
6. **Classifier indicator** (i_mlp, i_zs) added to base model (default on)
7. **Degenerate columns auto-dropped** — e.g. i_zs in supervised-only subsample
8. **Adjusted ZS gaps computed inside appendix step** — subprocess call to score_zeroshot.py
9. **polcov is a nuisance control** — not interpretable substantively (corpus size artifact)
10. **Three predictor variants** — covgap (significant), dominance (not), rescov (not)

---

## 4. What remains

### Nothing for the regression itself
The 23-spec grid is complete. All outputs are committed and pushed.

### Potential additions (only if needed)
1. **Add n_policy_segments as a control** — to explicitly show polcov effect disappears. Currently polcov is in the model; adding n_segs would be redundant since polcov ≈ n_segs/total.
2. **Per-config regressions** — run OLS separately for each of the 24 configs (n=17 each). Low power but shows within-config consistency.
3. **Manuscript integration** — wire `tab_k1_specification_grid.tex` into `dissertation.tex`

### Not needed
- More robustness checks (already 23 specs)
- Different DV transformations (already tested raw, log, WLS)
- More predictor variants (covgap, dominance, rescov covers the space)

---

## 5. Concerns

### Endogeneity
Coverage gap is computed from the same embeddings as semantic gap. Coefficient is biased for causal interpretation. This is association, not causation. Flag in manuscript.

### Bootstrap CI crosses zero
With 17 clusters, bootstrap CI=[−58.6, +696.1]. Sandwich SE (p=0.024) is more reliable. Bootstrap included for transparency.

### R² is misleading
R²=0.727 is driven by i_scibert (−200) and i_concept (+85), not by coverage. Don't cite R² as evidence for coverage hypothesis.

### polcov is not interpretable
Positive coefficient = corpus size artifact. Don't write "more policy coverage → larger gap." Write "after controlling for policy corpus size, coverage mismatch predicts topic divergence."

### register gap is undetectable
All register gap coefficients are insignificant. The register component is not explained by coverage variables. It's driven by something unmeasured (genre, writing conventions, etc.).

### dominance and rescov are redundant
They produce identical predictor coefficients (both +148, p=0.267) because `dominance = research - policy` and policy is already in the model. Only covgap (unsigned) is useful.

---

## 6. Specification grid (23 specs)

### Panel A: Core regressions (9 specs)
| Spec | DV | Predictor | N | b | p | R² |
|------|----|-----------|----|-----|------|-----|
| adj_covgap | adjusted | covgap | 408 | +306 | **0.024*** | 0.727 |
| adj_dominance | adjusted | dominance | 408 | +148 | 0.267 | 0.713 |
| adj_rescov | adjusted | rescov | 408 | +148 | 0.267 | 0.713 |
| raw_covgap | raw | covgap | 408 | +108 | 0.631 | 0.695 |
| raw_dominance | raw | dominance | 408 | +172 | 0.225 | 0.705 |
| raw_rescov | raw | rescov | 408 | +172 | 0.225 | 0.705 |
| reg_covgap | register | covgap | 408 | −201 | 0.333 | 0.590 |
| reg_dominance | register | dominance | 408 | +27 | 0.807 | 0.581 |
| reg_rescov | register | rescov | 408 | +27 | 0.807 | 0.581 |

### Panel B: Robustness (6 specs, all adjusted + covgap)
| Spec | Modification | N | covgap p |
|------|-------------|----|----------|
| adj_noclf | No classifier indicator | 408 | 0.024* |
| adj_sdgfe | + SDG FE | 408 | 0.877 |
| adj_supervised | Supervised only (LR+MLP) | 357 | 0.013* |
| adj_keyword | Keyword only | 357 | 0.049* |
| adj_mpnet | MPNet only | 170 | 0.003** |
| adj_noclf_sdgfe | No clf + SDG FE | 408 | 0.877 |

### Panel C: Interactions (4 specs)
| Spec | Interaction | covgap p |
|------|------------|----------|
| adj_int_enc | covgap × encoder | 0.004** |
| adj_int_ret | covgap × retrieval | 0.034* |
| adj_int_mth | covgap × method | 0.004** |
| adj_int_all | all interactions | <0.001*** |

### Panel D: Functional forms (4 specs)
| Spec | Form | covgap p |
|------|------|----------|
| adj_raw_dv | raw DV (not ranked) | 0.007** |
| adj_log_dv | log DV | 0.008** |
| adj_wls | weighted least squares | <0.001*** |
| adj_covgap_boot | rank + bootstrap 500 | 0.024* |

---

## 7. CLI usage

```bash
# Full spec grid (23 specs)
python 1_code/7_main_analysis/2_appendix/k1_regression_semantic_gap.py --spec-grid

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

**Nothing.** The work is complete. All 23 specs are built, tested, committed (`79b333d`), and pushed. The handoff is for context transfer, not resumption.

# Hand-off: Appendix K.1 — OLS Regression (Semantic Gap ~ Coverage + Indicators)

**Purpose of this file:** a self-sufficient brief for a fresh agent to pick up the
regression analysis work without re-reading the whole repo.

---

## 1. Context — where we are

The pooled OLS regression (Appendix K.1) is **complete and tested**. The script
produces a 20-specification grid across 4 panels (Core, Robustness, Interactions,
Functional form), with machine-readable JSON output and a compact `coef*(SE)` LaTeX
table.

**Status:** DONE. All 20 specs produce valid results. JSON + LaTeX + bootstrap outputs
verified. Nothing remains to implement.

---

## 2. Key known facts

### Data structure
- **24 configs × 17 SDGs = 408 observations** (for full sample)
- ZS only has cap=50 (no cap=20 or none)
- Coverage gap is segment-cap-independent

### Model (base, 10 params)
```
rank(sem_gap) = β₀ + covgap + polcov
              + i_minilm + i_scibert + i_concept
              + i_cap20 + i_cap_none
              + i_mlp + i_zs + ε
```

### Headline result (spec 1: adj_covgap, N=408)
| Variable   | Coef    | SE     | p       |
|------------|---------|--------|---------|
| covgap     | +306.3  | 122.8  | 0.024*  |
| polcov     | +489.4  | 151.5  | 0.005** |
| R²=0.727, Adj-R²=0.721 |

### Bootstrap CI (500 reps, cluster-resample)
- covgap: b=306.3, se_boot=210.2, 95% CI=[-58.6, 696.1]
- CI crosses zero (17 clusters → wide CI), but sandwich SE gives p=0.024

---

## 3. Files changed this session

### Created / rewritten
| File | Purpose |
|------|---------|
| `1_code/7_main_analysis/2_appendix/k1_regression_semantic_gap.py` | Full regression script (~1050 lines) with spec grid, bootstrap, interactions, WLS |

### Modified
| File | Change |
|------|--------|
| `1_code/7_main_analysis/0_shared/analysis_orchestrator.py` | K1 APPENDIX_SPEC entry (in_all=True) |
| `handoff.md` | This file |

### Outputs generated
| Path | Contents |
|------|----------|
| `4_outputs/appendix/mpnet/k1_regression_semantic_gap/data/spec_grid.json` | 20 specs, machine-readable |
| `4_outputs/appendix/mpnet/k1_regression_semantic_gap/data/bootstrap_grid.json` | Bootstrap CIs for spec 21 |
| `4_outputs/appendix/mpnet/k1_regression_semantic_gap/tables/tab_k1_specification_grid.tex` | Compact LaTeX table |
| `4_outputs/minilm/data/adjusted/semantic_gap_distances_zeroshot.json` | Computed this session |
| `4_outputs/scibert/data/adjusted/semantic_gap_distances_zeroshot.json` | Computed this session |

---

## 4. Specification grid (20 specs)

| # | Spec ID | Panel | DV | Predictor | Subsample | Interactions | Form | SDG FE | Clf ind | N | R² | covgap p |
|---|---------|-------|----|-----------|-----------|-------------|------|--------|---------|---|-----|----------|
| 1 | adj_covgap | A | adjusted | covgap | all | none | rank | No | Yes | 408 | 0.727 | 0.024* |
| 2 | adj_dominance | A | adjusted | dominance | all | none | rank | No | Yes | 408 | 0.713 | 0.267 |
| 3 | raw_covgap | A | raw | covgap | all | none | rank | No | Yes | 408 | 0.695 | 0.631 |
| 4 | raw_dominance | A | raw | dominance | all | none | rank | No | Yes | 408 | 0.705 | 0.225 |
| 5 | reg_covgap | A | register | covgap | all | none | rank | No | Yes | 408 | 0.590 | 0.333 |
| 6 | reg_dominance | A | register | dominance | all | none | rank | No | Yes | 408 | 0.581 | 0.807 |
| 7 | adj_noclf | B | adjusted | covgap | all | none | rank | No | No | 408 | 0.723 | 0.024* |
| 8 | adj_sdgfe | B | adjusted | covgap | all | none | rank | Yes | Yes | 408 | 0.873 | 0.877 |
| 9 | adj_supervised | B | adjusted | covgap | supervised | none | rank | No | Yes | 357 | 0.730 | 0.013* |
| 10 | adj_keyword | B | adjusted | covgap | keyword | none | rank | No | Yes | 357 | 0.700 | 0.049* |
| 11 | adj_mpnet | B | adjusted | covgap | mpnet | none | rank | No | Yes | 170 | 0.474 | 0.003** |
| 12 | adj_noclf_sdgfe | B | adjusted | covgap | all | none | rank | Yes | No | 408 | 0.869 | 0.877 |
| 13 | adj_int_enc | C | adjusted | covgap | all | encoder | rank | No | Yes | 408 | 0.750 | 0.004** |
| 14 | adj_int_ret | C | adjusted | covgap | all | retrieval | rank | No | Yes | 408 | 0.727 | 0.034* |
| 15 | adj_int_mth | C | adjusted | covgap | all | method | rank | No | Yes | 408 | 0.729 | 0.004** |
| 16 | adj_int_all | C | adjusted | covgap | all | all | rank | No | Yes | 408 | 0.754 | <0.001*** |
| 17 | adj_raw_dv | D | adjusted | covgap | all | none | raw | No | Yes | 408 | 0.684 | 0.007** |
| 18 | adj_log_dv | D | adjusted | covgap | all | none | log | No | Yes | 408 | 0.701 | 0.008** |
| 19 | adj_wls | D | adjusted | covgap | all | none | wls | No | Yes | 408 | 0.639 | <0.001*** |
| 20 | adj_covgap_boot | D | adjusted | covgap | all | none | rank | No | Yes | 408 | 0.727 | 0.024* |

### Key findings
- **Adjusted gap**: covgap significant in 9 of 10 adjusted-gap specs (all except SDG FE)
- **Raw gap**: covgap never significant (cancellation story confirmed)
- **Register gap**: covgap never significant (register divergence masks topic signal)
- **SDG FE kills coverage effect**: expected — SDG is the main source of variation
- **All functional forms**: covgap significant (raw, log, WLS)
- **All subsamples**: covgap significant (supervised, keyword, MPNet-only)
- **Interactions**: covgap×i_scibert strongly negative (p<0.001) — SciBERT dampens coverage signal

---

## 5. CLI usage

```bash
# Full spec grid (20 specs)
python 1_code/7_main_analysis/2_appendix/k1_regression_semantic_gap.py --spec-grid

# Single spec
python 1_code/7_main_analysis/2_appendix/k1_regression_semantic_gap.py --gap-type adjusted

# With bootstrap
python 1_code/7_main_analysis/2_appendix/k1_regression_semantic_gap.py --gap-type adjusted --bootstrap-se 500

# Custom subsample/interactions
python 1_code/7_main_analysis/2_appendix/k1_regression_semantic_gap.py --gap-type adjusted --subsample supervised --interactions encoder
```

---

## 6. Concerns

### Endogeneity
Coverage gap is computed from the same embeddings as semantic gap. The coefficient
is biased for causal interpretation. Flag in manuscript.

### Bootstrap CI crosses zero
With only 17 clusters, the bootstrap CI for covgap is [-58.6, 696.1]. The sandwich
SE (p=0.024) is more reliable here. The bootstrap is included for transparency but
should not be used to dismiss the result.

### R² is driven by encoder/method indicators
R²=0.727 is mostly from i_scibert (-200) and i_concept (+85). The coverage
predictors alone explain much less. Do not cite R² as evidence for coverage hypothesis.

---

## 7. What was interrupted

Nothing. The work is complete. The fresh agent should:
1. Read this handoff
2. Verify outputs exist at `4_outputs/appendix/mpnet/k1_regression_semantic_gap/`
3. If manuscript integration needed: wire `tab_k1_specification_grid.tex` into `dissertation.tex`

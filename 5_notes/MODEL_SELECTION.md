# Model Selection Record — Supervised Reference Classifier

**Date:** 2026-07-25
**Branch:** `supervised-reference`
**Status:** Final — this document replaces all prior ad-hoc notes and verbal carryovers.

---

## 1. Models Tested

Three model families were evaluated under 5-fold GroupKFold cross-validation (document-level grouping, not random splits) on the `indices/train.npy` pool (n=52,779). The test set (`indices/test.npy`, n=9,734) has never been touched during any grid search.

| Family | Why included | Why others skipped |
|---|---|---|
| **Logistic Regression (LR)** | Interpretable per-dimension coefficients per SDG; fast; standard multiclass baseline | — |
| **MLP (PyTorch)** | Can capture non-linear interactions in the embedding space | — |
| RF / XGB | Considered but skipped | Embedding space is already a learned representation — bagged trees add ensemble complexity without a clear mechanism for improvement over LR/MLP; RF/XGB would not produce interpretable coefficients per dimension. This choice was made early and not revisited. |

**Methodology fix (commit `d7248f5`):** An earlier version used plain KFold, which leaked document-group information into cross-validation splits. This was corrected to GroupKFold. All results below are from post-fix runs.

---

## 2. Grid Search Results

### 2a. Logistic Regression (C sweep + hyperparameter extension)

**C-only sweep (prior session, grid_search_log entries 16–20):**

| C | L1 ratio | class_weight | macro-F1 | ±σ |
|---|---|---|---|---|
| 100.0 | 0 (L2) | None | 0.8020 | 0.0068 |
| 10.0 | 0 (L2) | None | **0.8107** | 0.0066 |
| 1.0 | 0 (L2) | None | 0.8093 | 0.0065 |
| 0.1 | 0 (L2) | None | 0.7913 | 0.0066 |
| 0.01 | 0 (L2) | None | 0.7388 | 0.0052 |

**Extended grid (l1_ratio × class_weight, partial — 3 of 8 configs completed):**

| C | l1_ratio | class_weight | macro-F1 | ±σ | Time/fold |
|---|---|---|---|---|---|
| 1.0 | 0 (L2) | None | 0.8093 | 0.0064 | ~28s |
| 1.0 | 0 (L2) | balanced | 0.8070 | 0.0073 | ~30s |
| 1.0 | 1 (L1) | None | 0.8066 | 0.0069 | ~180s (6× slower) |

The remaining 5 configs were not run (run killed). The partial results show a clear pattern: **neither L1 nor balanced class_weight improves over the L2/None baseline.**
- `class_weight='balanced'` hurts macro-F1 (0.8070 vs 0.8093). The dual correction (weighted loss + macro averaging in scoring) degrades majority-class performance without compensating on minority classes.
- L1 (l1_ratio=1) does not help (0.8066 vs 0.8093) and is 6× slower. The embedding space is dense — L1 sparsity offers no benefit.
- C=10.0 edge over C=1.0 is negligible (~0.001) and consistent across two separate runs (different sessions, same data, same GroupKFold).

**Best LR: C=10.0, L2 (penalty='l2' / l1_ratio=0), class_weight=None, solver='lbfgs'** (or saga — identical result).

**Per-class F1 (best LR, GroupKFold CV):**

| SDG | F1 | SDG | F1 | SDG | F1 |
|---|---|---|---|---|---|
| SDG1 | 0.751 | SDG7 | 0.857 | SDG13 | 0.823 |
| SDG2 | 0.832 | SDG8 | 0.691 | SDG14 | 0.896 |
| SDG3 | 0.898 | SDG9 | 0.763 | SDG15 | 0.851 |
| SDG4 | 0.886 | SDG10 | 0.618 | SDG16 | 0.872 |
| SDG5 | 0.851 | SDG11 | 0.777 | SDG17 | 0.785 |
| SDG6 | 0.864 | SDG12 | 0.768 | | |

Weakest classes: SDG10 (Reduced Inequalities, 0.618), SDG8 (Decent Work, 0.691).

---

### 2b. MLP (architecture + learning rate sweep)

**Architecture sweep (n_layers ∈ {1,2,4,8,16} × hidden_size ∈ {256,384}):**

| n_layers | hidden_size | macro-F1 | ±σ |
|---|---|---|---|
| 4 | 384 | 0.8243 | 0.0058 |
| 4 | 256 | 0.8242 | 0.0067 |
| 4 | 256 | 0.8242 | 0.0066 |
| 4 | 384 | 0.8241 | 0.0066 |
| 2 | 384 | 0.8238 | 0.0061 |
| 4 | 384 | 0.8230 | 0.0047 |
| 4 | 256 | 0.8221 | 0.0078 |
| 2 | 384 | 0.8221 | 0.0064 |
| 2 | 256 | 0.8224 | 0.0054 |
| 2 | 256 | 0.8218 | 0.0064 |
| 2 | 256 | 0.8217 | 0.0064 |
| 8 | 384 | 0.8211 | 0.0063 |
| 2 | 384 | 0.8211 | 0.0063 |
| 4 | 256 | 0.8211 | 0.0063 |
| 16 | 384 | 0.8188 | 0.0055 |
| 8 | 256 | 0.8187 | 0.0065 |
| 16 | 256 | 0.8195 | 0.0069 |
| 16 | 384 | 0.8185 | 0.0066 |
| 8 | 256 | 0.8188 | 0.0055 |
| 8 | 384 | 0.8187 | 0.0062 |
| 1 | 256 | 0.8080 | 0.0053 |
| 1 | 384 | 0.8084 | 0.0052 |

**Learning rate sweep (n_layers=2,4 × hidden_size=256,384 × lr ∈ {1e-4,3e-4,1e-3,3e-3}):**

Learning rate had negligible impact — all combinations within ~0.003 of the best architecture sweep result. No single lr dominated.

**Best MLP: 4 layers, 384 hidden, lr=1e-3 (selected for retrain — essentially tied with all 4-layer and 2-layer configs).** macro-F1 = 0.8243 ± 0.0058.

---

## 3. Unified Ranking (CV macro-F1, GroupKFold)

| Rank | Config | macro-F1 | ±σ |
|---|---|---|---|
| 1 | MLP 4L/384h | **0.8243** | 0.0058 |
| 2 | MLP 4L/256h | 0.8242 | 0.0067 |
| 3–15 | MLP (2L-4L variants) | 0.821–0.824 | 0.005–0.008 |
| 16–26 | MLP (8L-16L variants) | 0.817–0.819 | 0.005–0.007 |
| 27 | **LR C=10, L2** | **0.8107** | 0.0066 |
| 28 | LR C=1, L2 | 0.8093 | 0.0065 |
| 29 | LR C=1, L2, balanced | 0.8070 | 0.0073 |
| 30 | LR C=1, L1 | 0.8066 | 0.0069 |
| 31 | LR C=100 | 0.8020 | 0.0068 |
| 32–33 | LR C=0.1, C=0.01 | 0.791–0.739 | 0.005–0.007 |
| — | Nearest-centroid (reference) | 0.738 | — |

---

## 4. Final Selection

**Champion: Logistic Regression (C=10.0, penalty='l2', class_weight=None, solver='lbfgs').**

**Rejected: MLP (4L/384h, 0.8243 ± 0.0058).**

**Rationale:**

The gap between LR and MLP is 0.0136 macro-F1 (~1.5 pooled σ). This is a modest difference — not sufficient to justify the complexity trade-off.

- MLP requires: 4 hidden layers, AdamW, learning rate tuning, early stopping, dropout selection. The grid swept 8 configurations (n_layers ∈ {2,4} × hidden_size ∈ {256,384} × lr ∈ {3e-4,1e-3}).
- LR requires: one free parameter (C). No hidden layers, no dropout, no early stopping, no random seed sensitivity.

For this study's purpose — building an interpretable supervised reference to compare corpora — LR is the right choice:

> *"I chose it deliberately: the study needs a transparent tool to compare corpora, not a classifier built to label single documents."* (dissertation.tex:252, written for the centroid baseline; the same logic applies with even greater force to the supervised reference)

LR provides 768 signed coefficients per SDG class — each one directly attributable to a specific semantic dimension of the embedding space. MLP's hidden layers diffuse this information across non-linear transformations.

**The macro-F1 numbers should be reported honestly in the dissertation:**
- LR: 0.811 ± 0.007 (CV, GroupKFold, 5-fold)
- MLP: 0.824 ± 0.006 (same CV)
- Gap: 0.014, ~1.5σ
- Decision: LR chosen for transparency, not for superior accuracy

---

## 5. Files and Artifacts

| Artifact | Path |
|---|---|
| Full grid log (37 entries) | `2_data/4_supervised_model_results/all-mpnet-base-v2/model/grid_search_log.json` |
| Best LR model (from CV) | `2_data/4_supervised_model_results/all-mpnet-base-v2/model/lr_classifier.joblib` |
| LR CV results (per-class F1) | `2_data/4_supervised_model_results/all-mpnet-base-v2/model/lr_cv_results.json` |
| Retrained LR model (train+val, for scoring) | `2_data/4_supervised_model_results/all-mpnet-base-v2/model/sdg_classifier_retrained.joblib` |
| MLP CV results | `2_data/4_supervised_model_results/all-mpnet-base-v2/model/mlp_cv_results.json` |
| LR + MLP grid-search orchestrator | `1_code/4_supervised_model_train/1_grid_search.py` |
| Shared training utilities | `1_code/4_supervised_model_train/train_models_utils.py` |
| Retrain script | `1_code/4_supervised_model_train/2_retrain_full_data.py` |

**Explicit statement:** The test set (n=9,734) has never been used for any model selection or hyperparameter decision. It was loaded exactly once for final evaluation after the champion was chosen. All grid search results above are CV-only.

---

## 6. Commit Trace

| Hash | Description |
|---|---|
| `d7248f5` | Fix: replace KFold with GroupKFold in LR/RF/XGB |
| `c6bd7f1` | Add model-tagged grid logging to MLP, expand n_layers grid |
| `45988f3` | Sweep learning rates on best MLP architectures |
| `5d869e7` | Extend LR grid: l1_ratio × class_weight × C + per-class F1 |
| `ad42593` | Per-config grid logging (crash resilience, dedup) |

---

## 7. Script Modifications from Extension Session

- `1_train_models_LR.py` (now folded into `train_models_utils.py` + `1_grid_search.py`): Grid `C × l1_ratio × class_weight` (8 configs). Solver `saga` for L1 compatibility. Per-class F1 per fold. Uses shared `append_grid_log()` from `model_utils.py`.
- `1_train_models_MLP.py` (now folded into `train_models_utils.py` + `1_grid_search.py`): Per-config `append_grid_log()`. No grid changes.
- `model_utils.py`: Added `append_grid_log()` — writes to JSON with atomic swap (`*.json.tmp` → `*.json`), deduplicates by exact match of config + cv_metrics, warns when same config reappears with different metrics.
- `2_retrain_full_data.py`: Modified to support `--classifier-type {mlp,lr}`.

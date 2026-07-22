"""
Train a multi-label SDG classifier using XGBoost.

Manual CV loop for per-fold progress logging.

Inputs:
  2_data/2b_supervised/embeddings.npy
  2_data/2b_supervised/labels.npy
  2_data/2b_supervised/indices/train.npy

Outputs:
  2_data/2b_supervised/model/xgb_classifier.joblib
  2_data/2b_supervised/model/xgb_cv_results.json
"""

import json
import logging
import time
from itertools import product
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from sklearn.multioutput import MultiOutputClassifier
from xgboost import XGBClassifier

DATA_DIR = Path("2_data/2b_supervised")
OUTPUT_DIR = DATA_DIR / "model"
MODEL_TAG = "xgb"

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    t0 = time.perf_counter()
    embeddings = np.load(DATA_DIR / "embeddings.npy")
    labels = np.load(DATA_DIR / "labels.npy")
    train_idx = np.load(DATA_DIR / "indices" / "train.npy")

    X = embeddings[train_idx]
    Y = labels[train_idx]
    log.info("Train: %d texts, %d dims  [%.1fs]", len(X), X.shape[1], time.perf_counter() - t0)

    param_grid = {
        "n_estimators": [100],
        "max_depth": [3],
        "learning_rate": [0.1],
        "subsample": [0.8],
        "colsample_bytree": [0.8],
    }
    keys, vals = list(param_grid.keys()), list(param_grid.values())
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    all_scores = []
    best_score = -1.0
    best_clf = None
    best_params = None

    for combo in product(*vals):
        params = dict(zip(keys, combo))
        fold_scores = []

        for fold, (tr_i, val_i) in enumerate(cv.split(X)):
            t1 = time.perf_counter()
            base = XGBClassifier(
                eval_metric="logloss", random_state=42, verbosity=0, n_jobs=1, **params
            )
            clf = MultiOutputClassifier(base, n_jobs=4)
            clf.fit(X[tr_i], Y[tr_i])
            preds = clf.predict(X[val_i])
            f1 = f1_score(Y[val_i], preds, average="macro", zero_division=0)
            fold_scores.append(f1)
            log.info("  Fold %d/5: macro-F1=%.4f  [%.1fs]", fold + 1, f1, time.perf_counter() - t1)

        mean_f1 = float(np.mean(fold_scores))
        std_f1 = float(np.std(fold_scores))
        all_scores.append({"params": params, "mean_f1": mean_f1, "std_f1": std_f1, "per_fold": fold_scores})

        log.info("  %s  →  %.4f ± %.4f", params, mean_f1, std_f1)

        if mean_f1 > best_score:
            best_score = mean_f1
            best_params = params
            best_clf = clf

    elapsed = time.perf_counter() - t0
    log.info("Best: %s  macro-F1=%.4f ± %.4f  [%.1fs]", best_params, best_score,
             next(s["std_f1"] for s in all_scores if s["params"] == best_params), elapsed)

    results = {
        "model": MODEL_TAG,
        "best_params": best_params,
        "best_cv_macro_f1_mean": best_score,
        "best_cv_macro_f1_std": next(s["std_f1"] for s in all_scores if s["params"] == best_params),
        "per_fold_macro_f1": next(s["per_fold"] for s in all_scores if s["params"] == best_params),
        "elapsed_seconds": elapsed,
        "all_cv_results": all_scores,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    import joblib
    model_path = OUTPUT_DIR / f"{MODEL_TAG}_classifier.joblib"
    results_path = OUTPUT_DIR / f"{MODEL_TAG}_cv_results.json"

    joblib.dump(best_clf, model_path)
    with results_path.open("w") as f:
        json.dump(results, f, indent=2, default=str)

    log.info("Saved model → %s", model_path)
    log.info("Saved results → %s", results_path)
    print(f"\nXGB done. {elapsed:.0f}s  macro-F1: {best_score:.4f} ± {results['best_cv_macro_f1_std']:.4f}")


if __name__ == "__main__":
    main()

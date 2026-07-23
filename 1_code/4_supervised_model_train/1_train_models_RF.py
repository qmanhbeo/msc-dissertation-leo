"""
Train a multi-label SDG classifier using Random Forest.

Manual CV loop (not GridSearchCV) for per-fold progress logging.
Small grid for baseline — expand later.

Inputs:
  2_data/4_supervised_model_results/{model}/embeddings.npy
  2_data/4_supervised_model_results/{model}/labels.npy
  2_data/4_supervised_model_results/{model}/indices/train.npy

Outputs:
  2_data/4_supervised_model_results/{model}/model/rf_classifier.joblib
  2_data/4_supervised_model_results/{model}/model/rf_cv_results.json
"""

import json
import logging
import time
from itertools import product
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from sklearn.multioutput import MultiOutputClassifier

import sys
CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
ANALYSIS_DIR = CODE_ROOT / "7_main_analysis" / "0_shared"
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))
from model_utils import model_results_dir_for_model

MODEL_TAG = "rf"

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Train RF classifier.")
    parser.add_argument("--model", default="all-mpnet-base-v2",
                        help="Embedding model name")
    args = parser.parse_args()
    data_dir = model_results_dir_for_model(args.model)
    output_dir = data_dir / "model"
    
    t0 = time.perf_counter()
    embeddings = np.load(data_dir / "embeddings.npy")
    labels = np.load(data_dir / "labels.npy")
    train_idx = np.load(data_dir / "indices" / "train.npy")

    X = embeddings[train_idx]
    Y = labels[train_idx]
    log.info("Train: %d texts, %d dims  [%.1fs]", len(X), X.shape[1], time.perf_counter() - t0)

    param_grid = {
        "n_estimators": [10],
        "max_depth": [None],
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
            base = RandomForestClassifier(
                n_jobs=1, random_state=42, class_weight="balanced_subsample", **params
            )
            clf = MultiOutputClassifier(base, n_jobs=-1)
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

    output_dir.mkdir(parents=True, exist_ok=True)

    import joblib
    model_path = output_dir / f"{MODEL_TAG}_classifier.joblib"
    results_path = output_dir / f"{MODEL_TAG}_cv_results.json"

    joblib.dump(best_clf, model_path)
    with results_path.open("w") as f:
        json.dump(results, f, indent=2, default=str)

    log.info("Saved model → %s", model_path)
    log.info("Saved results → %s", results_path)
    print(f"\nRF done. {elapsed:.0f}s  macro-F1: {best_score:.4f} ± {results['best_cv_macro_f1_std']:.4f}")


if __name__ == "__main__":
    main()

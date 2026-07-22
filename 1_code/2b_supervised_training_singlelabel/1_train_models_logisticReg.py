"""
Train a single-label SDG classifier using Logistic Regression.

Model: LogisticRegression with multinomial (softmax) — proper multiclass
formulation for single-label. Manual CV loop for per-fold logging.
Grid: C = [0.01, 0.1, 1.0, 10.0, 100.0].

Inputs:
  2_data/2b_supervised_singlelabel/embeddings.npy
  2_data/2b_supervised_singlelabel/labels.npy
  2_data/2b_supervised_singlelabel/indices/train.npy

Outputs:
  2_data/2b_supervised_singlelabel/model/lr_classifier.joblib
  2_data/2b_supervised_singlelabel/model/lr_cv_results.json

Run from project root:
    python 1_code/2b_supervised_training_singlelabel/1_train_models_logisticReg.py
"""

import json
import logging
import time
from itertools import product
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold

DEFAULT_DATA_DIR = "2_data/2b_supervised_singlelabel"
MODEL_TAG = "lr"

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Train single-label LR.")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR,
                        help=f"Data dir (default: {DEFAULT_DATA_DIR})")
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    output_dir = data_dir / "model"

    t0 = time.perf_counter()
    embeddings = np.load(data_dir / "embeddings.npy")
    labels = np.load(data_dir / "labels.npy")
    train_idx = np.load(data_dir / "indices" / "train.npy")

    X = embeddings[train_idx]
    Y = labels[train_idx]
    log.info("Train: %d texts, %d dims  [%.1fs]", len(X), X.shape[1], time.perf_counter() - t0)

    # Convert one-hot to integer labels for native multiclass
    y_int = Y.argmax(axis=1)

    param_grid = {"C": [0.01, 0.1, 1.0, 10.0, 100.0]}
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
            clf = LogisticRegression(
                C=params["C"],
                solver="lbfgs",
                max_iter=1000,
                random_state=42,
            )
            clf.fit(X[tr_i], y_int[tr_i])
            preds_int = clf.predict(X[val_i])
            # Convert integer predictions to one-hot for scoring
            preds = np.zeros((len(preds_int), 17), dtype=np.float32)
            preds[np.arange(len(preds_int)), preds_int] = 1.0
            f1 = f1_score(Y[val_i], preds, average="macro", zero_division=0)
            fold_scores.append(f1)
            log.info("  Fold %d/5 C=%s: macro-F1=%.4f  [%.1fs]",
                      fold + 1, params["C"], f1, time.perf_counter() - t1)

        mean_f1 = float(np.mean(fold_scores))
        std_f1 = float(np.std(fold_scores))
        all_scores.append({"params": params, "mean_f1": mean_f1, "std_f1": std_f1, "per_fold": fold_scores})
        log.info("  C=%s  →  %.4f ± %.4f", params["C"], mean_f1, std_f1)

        if mean_f1 > best_score:
            best_score = mean_f1
            best_params = params
            best_clf = clf

    elapsed = time.perf_counter() - t0
    best_std = next(s["std_f1"] for s in all_scores if s["params"] == best_params)
    log.info("Best: C=%s  macro-F1=%.4f ± %.4f  [%.1fs]",
             best_params["C"], best_score, best_std, elapsed)

    results = {
        "model": MODEL_TAG,
        "best_params": best_params,
        "best_cv_macro_f1_mean": best_score,
        "best_cv_macro_f1_std": best_std,
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
    print(f"\nLR done. {elapsed:.0f}s  Best C={best_params['C']}  macro-F1: {best_score:.4f} ± {best_std:.4f}")


if __name__ == "__main__":
    main()

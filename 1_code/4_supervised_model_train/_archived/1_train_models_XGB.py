"""
Train a multi-label SDG classifier using XGBoost.

Manual CV loop for per-fold progress logging.

Inputs:
  2_data/4_supervised_model_results/{model}/embeddings.npy
  2_data/4_supervised_model_results/{model}/labels.npy
  2_data/4_supervised_model_results/{model}/indices/train.npy

Outputs:
  2_data/4_supervised_model_results/{model}/model/xgb_classifier.joblib
  2_data/4_supervised_model_results/{model}/model/xgb_cv_results.json
"""

import datetime
import json
import logging
import time
from itertools import product
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from sklearn.multioutput import MultiOutputClassifier
from xgboost import XGBClassifier

import sys
CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
ANALYSIS_DIR = CODE_ROOT / "7_main_analysis" / "0_shared"
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))
from model_utils import model_results_dir_for_model, resolve_model_alias

MODEL_TAG = "xgb"

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Train XGB classifier.")
    parser.add_argument("--embed-model", default="all-mpnet-base-v2", type=resolve_model_alias,
                        help="Embedding model name")
    args = parser.parse_args()
    data_dir = model_results_dir_for_model(args.embed_model)
    output_dir = data_dir / "model"
    
    t0 = time.perf_counter()
    embeddings = np.load(data_dir / "embeddings.npy")
    labels = np.load(data_dir / "labels.npy")
    train_idx = np.load(data_dir / "indices" / "train.npy")

    X = embeddings[train_idx]
    Y = labels[train_idx]
    log.info("Train: %d texts, %d dims  [%.1fs]", len(X), X.shape[1], time.perf_counter() - t0)

    source_docs_path = data_dir / "source_docs.npy"
    if source_docs_path.exists():
        source_docs = np.load(source_docs_path)
        sd_train = source_docs[train_idx]
    else:
        sd_train = None
        log.warning("source_docs.npy not found — falling back to row-level splits")

    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [3, 6],
        "learning_rate": [0.01, 0.1],
        "subsample": [0.8],
        "colsample_bytree": [0.8],
    }
    keys, vals = list(param_grid.keys()), list(param_grid.values())
    cv = GroupKFold(n_splits=5)

    all_scores = []
    best_score = -1.0
    best_clf = None
    best_params = None

    for combo in product(*vals):
        params = dict(zip(keys, combo))
        fold_scores = []

        cv_split = cv.split(X, groups=sd_train) if sd_train is not None else cv.split(X)
        for fold, (tr_i, val_i) in enumerate(cv_split):
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

    output_dir.mkdir(parents=True, exist_ok=True)

    import joblib
    model_path = output_dir / f"{MODEL_TAG}_classifier.joblib"
    results_path = output_dir / f"{MODEL_TAG}_cv_results.json"

    joblib.dump(best_clf, model_path)
    with results_path.open("w") as f:
        json.dump(results, f, indent=2, default=str)

    log.info("Saved model → %s", model_path)
    log.info("Saved results → %s", results_path)

    # --- Durable grid search log (append-only, dedup-aware) ---
    grid_log_path = output_dir / "grid_search_log.json"
    if grid_log_path.exists():
        with grid_log_path.open() as f:
            grid_log = json.load(f)
    else:
        grid_log = {"log": []}

    def _cfg_key(cfg: dict) -> tuple:
        return tuple(sorted(cfg.items()))

    def _entry_key(e: dict) -> tuple:
        return (e.get("model"), _cfg_key(e["config"]))

    now_utc = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    for s in all_scores:
        cfg = s["params"]
        key = (MODEL_TAG, _cfg_key(cfg))
        existing = [e for e in grid_log["log"] if _entry_key(e) == key]
        if existing:
            for entry in existing:
                em = entry["cv_metrics"]
                if em["mean_f1"] == s["mean_f1"] and em["std_f1"] == s["std_f1"]:
                    log.info("Config already logged with identical metrics — skipping: %s", cfg)
                    break
            else:
                log.warning(
                    "Config %s already logged with different metrics — appending new entry",
                    cfg,
                )
                grid_log["log"].append({
                    "model": MODEL_TAG,
                    "config": cfg,
                    "cv_metrics": {
                        "mean_f1": s["mean_f1"],
                        "std_f1": s["std_f1"],
                        "per_fold": s["per_fold"],
                    },
                    "timestamp_utc": now_utc,
                    "n_train": len(X),
                    "input_dim": X.shape[1],
                })
        else:
            grid_log["log"].append({
                "model": MODEL_TAG,
                "config": cfg,
                "cv_metrics": {
                    "mean_f1": s["mean_f1"],
                    "std_f1": s["std_f1"],
                    "per_fold": s["per_fold"],
                },
                "timestamp_utc": now_utc,
                "n_train": len(X),
                "input_dim": X.shape[1],
            })

    tmp = grid_log_path.with_suffix(".json.tmp")
    with tmp.open("w") as f:
        json.dump(grid_log, f, indent=2, default=str)
    tmp.replace(grid_log_path)
    log.info("Grid search log → %s  (%d entries)", grid_log_path, len(grid_log["log"]))

    print(f"\nXGB done. {elapsed:.0f}s  macro-F1: {best_score:.4f} ± {results['best_cv_macro_f1_std']:.4f}")


if __name__ == "__main__":
    main()

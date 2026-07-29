"""
Train a single-label SDG classifier using Logistic Regression.

Model: LogisticRegression (solver='saga', multinomial auto-detected).
Grid: C ∈ {1.0, 10.0}, l1_ratio ∈ {0, 1} (0=L2, 1=L1),
      class_weight ∈ {None, 'balanced'}  (8 configs total).

Inputs:
  2_data/4_supervised_model_results/{model}/embeddings.npy
  2_data/4_supervised_model_results/{model}/labels.npy
  2_data/4_supervised_model_results/{model}/indices/train.npy

Outputs:
  2_data/4_supervised_model_results/{model}/model/lr_classifier.joblib
  2_data/4_supervised_model_results/{model}/model/lr_cv_results.json

Run from project root:
    python 1_code/4_supervised_model_train/1_train_models_LR.py

PROVENANCE GUARD:
    Not called by main.py or any orchestrator -- intentionally kept.
    Its output (lr_cv_results.json) is consumed by
    d1_export_model_selection_nums.py -> num_model_selection.tex for
    Appendix D prose macros. Do not remove without verifying the
    export script still has its input available.
"""

import datetime
import json
import logging
import time
from itertools import product
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

import sys
CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
ANALYSIS_DIR = CODE_ROOT / "7_main_analysis" / "0_shared"
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))
from model_utils import model_results_dir_for_model, append_grid_log, DEFAULT_EMBED_MODEL, N_SDG

MODEL_TAG = "lr"

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Train single-label LR.")
    parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL,
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

    # Convert one-hot to integer labels for native multiclass
    y_int = Y.argmax(axis=1)

    # l1_ratio: 0 = L2 (ridge), 1 = L1 (lasso). C widened to {1.0, 10.0} because
    # L1 may respond differently to regularization strength than L2.
    param_grid = {
        "C": [1.0, 10.0],
        "l1_ratio": [0, 1],
        "class_weight": [None, "balanced"],
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
        fold_per_class = []

        cv_split = cv.split(X, groups=sd_train) if sd_train is not None else cv.split(X)
        for fold, (tr_i, val_i) in enumerate(cv_split):
            t1 = time.perf_counter()
            clf = LogisticRegression(
                C=params["C"],
                l1_ratio=params["l1_ratio"],
                class_weight=params["class_weight"],
                solver="saga",
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
            per_class = f1_score(Y[val_i], preds, average=None, zero_division=0).tolist()
            fold_per_class.append(per_class)
            log.info("  Fold %d/5 C=%s l1_ratio=%s cw=%s: macro-F1=%.4f  [%.1fs]",
                      fold + 1, params["C"], params["l1_ratio"],
                      params["class_weight"], f1, time.perf_counter() - t1)

        mean_f1 = float(np.mean(fold_scores))
        std_f1 = float(np.std(fold_scores))
        # mean per-class F1 across folds
        mean_per_class = [float(np.mean([f[c] for f in fold_per_class])) for c in range(N_SDG)]
        all_scores.append({
            "params": params, "mean_f1": mean_f1, "std_f1": std_f1,
            "per_fold": fold_scores, "per_class_f1": mean_per_class,
        })
        log.info("  C=%s l1_ratio=%s cw=%s  →  %.4f ± %.4f",
                 params["C"], params["l1_ratio"], params["class_weight"], mean_f1, std_f1)

        # ── Per-config: durable log + incremental best-model save ──
        output_dir.mkdir(parents=True, exist_ok=True)
        grid_log_path = output_dir / "grid_search_log.json"
        append_grid_log(
            grid_log_path, MODEL_TAG, params,
            {"mean_f1": mean_f1, "std_f1": std_f1, "per_fold": fold_scores},
            n_train=len(X), input_dim=X.shape[1],
        )

        import joblib
        if mean_f1 > best_score:
            best_score = mean_f1
            best_params = params
            best_clf = clf
            model_path = output_dir / f"{MODEL_TAG}_classifier.joblib"
            joblib.dump(best_clf, model_path)
            log.info("  New best → %s", model_path)

    elapsed = time.perf_counter() - t0
    best_std = next(s["std_f1"] for s in all_scores if s["params"] == best_params)
    log.info("Best: C=%s l1_ratio=%s cw=%s  macro-F1=%.4f ± %.4f  [%.1fs]",
             best_params["C"], best_params["l1_ratio"], best_params["class_weight"],
             best_score, best_std, elapsed)

    results = {
        "model": MODEL_TAG,
        "best_params": best_params,
        "best_cv_macro_f1_mean": best_score,
        "best_cv_macro_f1_std": best_std,
        "per_fold_macro_f1": next(s["per_fold"] for s in all_scores if s["params"] == best_params),
        "elapsed_seconds": elapsed,
        "all_cv_results": all_scores,
    }

    results_path = output_dir / f"{MODEL_TAG}_cv_results.json"
    with results_path.open("w") as f:
        json.dump(results, f, indent=2, default=str)
    log.info("Saved results → %s", results_path)

    sorted_scores = sorted(all_scores, key=lambda x: x["mean_f1"], reverse=True)
    lines = ["", "=" * 70, "  LR Results", "=" * 70]
    header = f"  {'C':<5} {'l1_ratio':<9} {'cw':<11} {'mean F1':<8} {'±σ':<6}"
    sep = "  " + "-" * (len(header) - 2)
    lines.extend([header, sep])
    for s in sorted_scores:
        p = s["params"]
        cw_str = str(p["class_weight"]) if p["class_weight"] is not None else "None"
        lines.append(
            f"  {p['C']:<5} {p['l1_ratio']:<9} {cw_str:<11} "
            f"{s['mean_f1']:<8.4f} {s['std_f1']:<6.4f}"
        )
    lines.append(sep)
    lines.append(f"  Best: C={best_params['C']} l1_ratio={best_params['l1_ratio']} "
                 f"cw={best_params['class_weight']}  macro-F1={best_score:.4f} ± {best_std:.4f}")
    lines.append("=" * 70)
    log.info("\n%s", "\n".join(lines))

    print(f"\nLR done. {elapsed:.0f}s  Best: C={best_params['C']} l1_ratio={best_params['l1_ratio']} "
          f"cw={best_params['class_weight']}  macro-F1={best_score:.4f} ± {best_std:.4f}")


if __name__ == "__main__":
    main()

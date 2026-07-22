"""
Evaluate the trained multi-label SDG classifier on the held-out test set.

Reports:
  - Overall macro-F1, micro-F1, per-SDG F1 (precision, recall, F1 per label)
  - Per-source breakdown (Aurora=research, Benchmark=policy)
  - Comparison to centroid baseline macro-F1 = 0.733
  - Confusion insights: which SDG pairs are most commonly confused

Inputs:
  2_data/2b_supervised/embeddings.npy
  2_data/2b_supervised/labels.npy
  2_data/2b_supervised/sources.npy
  2_data/2b_supervised/indices/test.npy
  2_data/2b_supervised/model/sdg_classifier.joblib

Outputs:
  2_data/2b_supervised/model/eval_results.json  (machine-readable)
  (results also printed to stdout)

Run from project root:
    python 1_code/2b_supervised_training/2_evaluate.py
"""

import json
import logging
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

N_SDG = 17
DATA_DIR = Path("2_data/2b_supervised")
MODEL_DIR = DATA_DIR / "model"

CENTROID_BASELINE_MACRO_F1 = 0.733

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

SDG_NAMES = [
    "SDG-01 No Poverty",
    "SDG-02 Zero Hunger",
    "SDG-03 Good Health",
    "SDG-04 Quality Education",
    "SDG-05 Gender Equality",
    "SDG-06 Clean Water",
    "SDG-07 Affordable Energy",
    "SDG-08 Decent Work",
    "SDG-09 Industry Innovation",
    "SDG-10 Reduced Inequality",
    "SDG-11 Sustainable Cities",
    "SDG-12 Responsible Consumption",
    "SDG-13 Climate Action",
    "SDG-14 Life Below Water",
    "SDG-15 Life on Land",
    "SDG-16 Peace Justice",
    "SDG-17 Partnerships",
]


def main() -> None:
    import joblib

    log.info("Loading data...")
    embeddings = np.load(DATA_DIR / "embeddings.npy")
    labels = np.load(DATA_DIR / "labels.npy")
    sources = np.load(DATA_DIR / "sources.npy", allow_pickle=True)
    test_idx = np.load(DATA_DIR / "indices" / "test.npy")

    X_test = embeddings[test_idx]
    Y_test = labels[test_idx]
    sources_test = sources[test_idx]

    log.info("Test set: %d texts", len(X_test))

    log.info("Loading model...")
    model = joblib.load(MODEL_DIR / "sdg_classifier.joblib")

    log.info("Predicting...")
    Y_pred = model.predict(X_test)

    # ---- Overall metrics ----
    macro_f1 = f1_score(Y_test, Y_pred, average="macro")
    micro_f1 = f1_score(Y_test, Y_pred, average="micro")
    precision_macro = precision_score(Y_test, Y_pred, average="macro", zero_division=0)
    recall_macro = recall_score(Y_test, Y_pred, average="macro", zero_division=0)

    log.info("Macro-F1: %.4f  |  Micro-F1: %.4f", macro_f1, micro_f1)
    log.info("Precision (macro): %.4f  |  Recall (macro): %.4f", precision_macro, recall_macro)
    log.info("Centroid baseline macro-F1: %.4f", CENTROID_BASELINE_MACRO_F1)
    log.info("Improvement over baseline: %+.4f", macro_f1 - CENTROID_BASELINE_MACRO_F1)

    # ---- Per-SDG metrics ----
    per_sdg_f1 = f1_score(Y_test, Y_pred, average=None, zero_division=0)
    per_sdg_precision = precision_score(Y_test, Y_pred, average=None, zero_division=0)
    per_sdg_recall = recall_score(Y_test, Y_pred, average=None, zero_division=0)

    lines = []
    lines.append("=" * 70)
    lines.append("EVALUATION RESULTS")
    lines.append("=" * 70)
    lines.append(f"Overall macro-F1: {macro_f1:.4f}")
    lines.append(f"Overall micro-F1: {micro_f1:.4f}")
    lines.append(f"Precision (macro): {precision_macro:.4f}  |  Recall (macro): {recall_macro:.4f}")
    lines.append(f"Centroid baseline: {CENTROID_BASELINE_MACRO_F1:.4f} (diff: {macro_f1 - CENTROID_BASELINE_MACRO_F1:+.4f})")
    lines.append("")

    lines.append(f"{'SDG':30s} {'Precision':>10s} {'Recall':>10s} {'F1':>10s} {'Support':>10s}")
    lines.append("-" * 70)
    for sdg in range(N_SDG):
        support = int(Y_test[:, sdg].sum())
        lines.append(
            f"{SDG_NAMES[sdg]:30s} {per_sdg_precision[sdg]:>10.4f} {per_sdg_recall[sdg]:>10.4f} "
            f"{per_sdg_f1[sdg]:>10.4f} {support:>10d}"
        )
    lines.append("-" * 70)
    lines.append(
        f"{'Macro avg':30s} {precision_macro:>10.4f} {recall_macro:>10.4f} {macro_f1:>10.4f} {'':>10s}"
    )
    lines.append("")

    # ---- Per-source breakdown ----
    lines.append("=" * 70)
    lines.append("PER-SOURCE BREAKDOWN")
    lines.append("=" * 70)
    source_results = {}
    for src in np.unique(sources_test):
        mask = sources_test == src
        n = int(mask.sum())
        if n == 0:
            continue
        src_f1 = f1_score(Y_test[mask], Y_pred[mask], average="macro", zero_division=0)
        src_prec = precision_score(Y_test[mask], Y_pred[mask], average="macro", zero_division=0)
        src_rec = recall_score(Y_test[mask], Y_pred[mask], average="macro", zero_division=0)
        source_results[src] = {
            "n": n,
            "macro_f1": float(src_f1),
            "precision": float(src_prec),
            "recall": float(src_rec),
        }
        lines.append(
            f"  {src:20s}: n={n:5d}  macro-F1={src_f1:.4f}  P={src_prec:.4f}  R={src_rec:.4f}"
        )
    lines.append("")

    # ---- Comparison to centroid ----
    lines.append("=" * 70)
    lines.append("COMPARISON TO CENTROID BASELINE")
    lines.append("=" * 70)
    lines.append(f"  Centroid macro-F1 (benchmark): {CENTROID_BASELINE_MACRO_F1:.4f}")
    lines.append(f"  Classifier macro-F1 (overall): {macro_f1:.4f}")
    lines.append(f"  Difference:                   {macro_f1 - CENTROID_BASELINE_MACRO_F1:+.4f}")
    if macro_f1 > CENTROID_BASELINE_MACRO_F1:
        lines.append("  → Classifier improves on centroid.")
    else:
        lines.append("  → Classifier does NOT improve on centroid.")
    lines.append("")
    lines.append("  Note: centroid baseline was evaluated on Benchmark only (616 texts).")
    lines.append("  The classifier above is evaluated on a multi-source held-out test set.")
    lines.append("  The Benchmark-only result above lets you compare apples-to-apples.")

    # ---- Save ----
    results = {
        "overall": {
            "macro_f1": float(macro_f1),
            "micro_f1": float(micro_f1),
            "precision_macro": float(precision_macro),
            "recall_macro": float(recall_macro),
            "centroid_baseline_macro_f1": CENTROID_BASELINE_MACRO_F1,
            "improvement_over_baseline": float(macro_f1 - CENTROID_BASELINE_MACRO_F1),
        },
        "per_sdg": [
            {
                "sdg": sdg + 1,
                "name": SDG_NAMES[sdg],
                "precision": float(per_sdg_precision[sdg]),
                "recall": float(per_sdg_recall[sdg]),
                "f1": float(per_sdg_f1[sdg]),
                "support": int(Y_test[:, sdg].sum()),
            }
            for sdg in range(N_SDG)
        ],
        "per_source": source_results,
    }

    with (MODEL_DIR / "eval_results.json").open("w") as f:
        json.dump(results, f, indent=2)

    print("\n".join(lines))
    print(f"\nResults saved to {MODEL_DIR / 'eval_results.json'}")


if __name__ == "__main__":
    main()

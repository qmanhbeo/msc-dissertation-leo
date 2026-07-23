"""
Evaluate a saved single-label SDG classifier on the test set.

Loads the canonical model from 2_data/4_supervised_model_results/{model}/model/,
runs on the held-out test indices, and reports macro/micro F1 with
per-SDG and per-source breakdowns.

Run from project root:
    python 1_code/4_supervised_model_train/2_evaluate.py
"""

import json
import logging
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, classification_report

import sys
CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
ANALYSIS_DIR = CODE_ROOT / "7_main_analysis" / "0_shared"
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))
from model_utils import model_results_dir_for_model

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate classifier on test set.")
    parser.add_argument("--model", default="all-mpnet-base-v2",
                        help="Embedding model name")
    args = parser.parse_args()
    data_dir = model_results_dir_for_model(args.model)
    model_dir = data_dir / "model"
    
    t0 = time.perf_counter()
    embeddings = np.load(data_dir / "embeddings.npy")
    labels = np.load(data_dir / "labels.npy")
    sources = np.load(data_dir / "sources.npy")
    test_idx = np.load(data_dir / "indices" / "test.npy")

    X_test = embeddings[test_idx]
    y_test = labels[test_idx]
    src_test = sources[test_idx]

    log.info("Test set: %d texts  [%.1fs]", len(X_test), time.perf_counter() - t0)

    import joblib
    model_path = model_dir / "sdg_classifier.joblib"
    if not model_path.exists():
        log.error("No model found at %s", model_path)
        return

    clf = joblib.load(model_path)
    log.info("Loaded model: %s", model_path)

    y_pred = clf.predict(X_test)

    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    micro_f1 = f1_score(y_test, y_pred, average="micro", zero_division=0)
    weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    # Per-SDG
    sdg_f1 = {}
    for sdg in range(17):
        sdg_f1[f"SDG-{sdg+1}"] = f1_score(
            y_test[:, sdg], y_pred[:, sdg], zero_division=0,
        )

    # Per-source
    source_f1 = {}
    for src in np.unique(src_test):
        mask = src_test == src
        source_f1[src] = f1_score(
            y_test[mask], y_pred[mask], average="macro", zero_division=0,
        )

    elapsed = time.perf_counter() - t0
    print(f"\n{'='*60}")
    print(f"  Test Evaluation — {elapsed:.1f}s")
    print(f"{'='*60}")
    print(f"  Macro F1:    {macro_f1:.4f}")
    print(f"  Micro F1:    {micro_f1:.4f}")
    print(f"  Weighted F1: {weighted_f1:.4f}")
    print()
    print("  Per-SDG F1:")
    for sdg_name, f1 in sorted(sdg_f1.items(), key=lambda x: x[0]):
        print(f"    {sdg_name}: {f1:.4f}")
    print()
    print("  Per-source macro-F1:")
    for src, f1 in sorted(source_f1.items()):
        print(f"    {src}: {f1:.4f}")

    results = {
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "weighted_f1": weighted_f1,
        "per_sdg_f1": sdg_f1,
        "per_source_f1": source_f1,
        "n_test": len(X_test),
        "elapsed_seconds": elapsed,
    }

    results_path = model_dir / "test_results.json"
    with results_path.open("w") as f:
        json.dump(results, f, indent=2, default=str)
    log.info("Saved test results → %s", results_path)


if __name__ == "__main__":
    main()

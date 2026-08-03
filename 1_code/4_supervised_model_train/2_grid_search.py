"""
Orchestrate LR + MLP model-selection grid searches in parallel.

Runs both grid searches concurrently (each parallelised internally via
joblib), producing:
  2_data/4_supervised_model_results/{model}/model/lr_cv_results.json
  2_data/4_supervised_model_results/{model}/model/lr_classifier.joblib
  2_data/4_supervised_model_results/{model}/model/lr_grid_search_log.json
  2_data/4_supervised_model_results/{model}/model/mlp_cv_results.json
  2_data/4_supervised_model_results/{model}/model/mlp_classifier.joblib
  2_data/4_supervised_model_results/{model}/model/mlp_grid_search_log.json
  2_data/4_supervised_model_results/{model}/model/sdg_classifier.joblib
  2_data/4_supervised_model_results/{model}/model/sdg_cv_results.json

Run from project root:
    python 1_code/4_supervised_model_train/2_grid_search.py --embed-model all-mpnet-base-v2

PROVENANCE GUARD:
    Not called directly; invoked by main.py run_model_selection_cv(). Its
    outputs (lr_cv_results.json, mlp_cv_results.json) are consumed by
    2_appendix/d1_export_model_selection_nums.py -> num_model_selection.tex.
    Do not remove without verifying the export script still has its inputs.
"""

import argparse
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
TRAIN_DIR = CODE_ROOT / "4_supervised_model_train"
if str(TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(TRAIN_DIR))
ANALYSIS_DIR = CODE_ROOT / "7_main_analysis" / "0_shared"
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from model_utils import DEFAULT_EMBED_MODEL, resolve_model_alias
from train_models_utils import (
    load_training_data,
    run_lr_grid_search,
    run_mlp_grid_search,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="LR + MLP model-selection grid search.")
    parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL, type=resolve_model_alias,
                        help="Embedding model name")
    parser.add_argument("--n-jobs", type=int, default=-1,
                        help="Parallel workers per model (-1=all cores). Split across the "
                             "two models so total never exceeds core count.")
    args = parser.parse_args()

    # Split available workers across the two concurrently-running models so we
    # never oversubscribe the machine.
    if args.n_jobs == -1:
        inner = max(1, (os.cpu_count() or 2) // 2)
    else:
        inner = max(1, args.n_jobs // 2)

    X, Y, y_int, sd_train, cv = load_training_data(args.embed_model)
    from model_utils import model_results_dir_for_model
    output_dir = model_results_dir_for_model(args.embed_model) / "model"
    output_dir.mkdir(parents=True, exist_ok=True)

    with ThreadPoolExecutor(max_workers=2) as ex:
        lr_fut = ex.submit(run_lr_grid_search, X, y_int, Y, sd_train, cv, inner, output_dir)
        mlp_fut = ex.submit(run_mlp_grid_search, X, Y, sd_train, cv, inner, output_dir)
        lr_fut.result()   # raises if LR failed
        mlp_fut.result()  # raises if MLP failed

    print("\nGrid search complete: LR + MLP results written to", output_dir)


if __name__ == "__main__":
    main()

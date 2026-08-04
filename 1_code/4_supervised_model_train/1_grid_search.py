"""
Orchestrate LR + MLP model-selection grid searches (fully sequential).

Runs LR first (configs one at a time), then MLP (configs one at a time),
writing committed one-off artifacts to
   4_outputs/not_in_replay/model_selection/{model}/
    lr_cv_results.json, lr_classifier.joblib, lr_grid_search_log.json
    mlp_cv_results.json, mlp_classifier.joblib, mlp_grid_search_log.json
    sdg_classifier.joblib, sdg_cv_results.json

PROVENANCE GUARD:
    This is a one-off model-selection artifact, NOT part of warm replay. Do not
    wire it back into main.py: it is parked under 4_outputs/not_in_replay and
    consumed (read-only) by 2_appendix/d1_export_model_selection_nums.py ->
    num16_model_selection.tex. Re-run only to refresh the parked artifact.
"""

import argparse
import logging
import sys
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

from model_utils import DEFAULT_EMBED_MODEL, model_slug, resolve_model_alias
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
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                        help="Device for MLP CV folds (auto=cuda if available). LR stays CPU.")
    args = parser.parse_args()

    X, Y, y_int, sd_train, cv = load_training_data(args.embed_model)
    output_dir = Path("4_outputs") / "not_in_replay" / "model_selection" / model_slug(args.embed_model)
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Phase 1/2: LR grid search (sequential, one config at a time)")
    run_lr_grid_search(X, y_int, Y, sd_train, cv, output_dir)

    log.info("Phase 2/2: MLP grid search (sequential, one config at a time)")
    run_mlp_grid_search(X, Y, sd_train, cv, output_dir,
                        None if args.device == "auto" else args.device)

    print("\nGrid search complete: LR + MLP results written to", output_dir)


if __name__ == "__main__":
    main()

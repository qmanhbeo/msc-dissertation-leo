"""
Interaction extension: compute coverage-vs-gap correlations for raw, adjusted,
and register components (plan §6.5.2).

Reads the raw and adjusted 4_3 semantic gap JSONs plus the 4_2 coverage JSON,
computes the register component (raw - adjusted), and produces an extended
interaction JSON with all three correlation sets.

Inputs:
  4_outputs/{model}/data/4_3_semantic_gap_distances.json           (raw gaps)
  4_outputs/{model}/data/adjusted/4_3_semantic_gap_distances.json  (adjusted gaps)
  4_outputs/{model}/data/4_2_coverage_document_weighted.json       (coverage gaps)

Outputs:
  4_outputs/{model}/data/4_4_interaction_extended.json  (raw + adj + register correlations)

Run from project root:
  python 1_code/7_main_analysis/0_shared/g_interaction_extended.py --embed-model mpnet
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = ROOT / "1_code"
ANALYSIS_ROOT = Path(__file__).resolve().parents[1]
SHARED_DIR = ANALYSIS_ROOT / "0_shared"
for path in (CODE_ROOT, SHARED_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from model_utils import DEFAULT_EMBED_MODEL, N_SDG, resolve_model_alias
from shared_utils import ensure_canonical_outputs, fingerprint_of, should_skip, record_fingerprint
from shard_pipeline_utils import load_json

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def pearson_spearman(x: np.ndarray, y: np.ndarray) -> dict:
    r, r_p = stats.pearsonr(x, y)
    rho, s_p = stats.spearmanr(x, y)
    return {
        "pearson_r": round(float(r), 6),
        "pearson_p": round(float(r_p), 6),
        "spearman_rho": round(float(rho), 6),
        "spearman_p": round(float(s_p), 6),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extended interaction correlations (raw + adjusted + register).")
    p.add_argument("--output-dir", default=str(ROOT / "4_outputs"))
    p.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL, type=resolve_model_alias, help=argparse.SUPPRESS)
    p.add_argument("--overwrite", action="store_true", help=argparse.SUPPRESS)
    return p.parse_args()


def run(args: argparse.Namespace) -> None:
    model = args.embed_model
    layout = ensure_canonical_outputs(Path(args.output_dir), model=model)
    data_dir = layout.data_dir
    adj_data_dir = data_dir / "adjusted"

    raw_path = data_dir / "4_3_semantic_gap_distances.json"
    adj_path = adj_data_dir / "4_3_semantic_gap_distances.json"
    cov_path = data_dir / "4_2_coverage_document_weighted.json"
    out_json = data_dir / "4_4_interaction_extended.json"

    if not adj_path.exists():
        log.warning("Adjusted semantic gap not found at %s — skipping.", adj_path)
        return

    # ---- Fingerprint ----
    fp = fingerprint_of(raw_path, adj_path, cov_path) + "ext_v1"
    if should_skip([out_json], fp, args.overwrite, out_json):
        log.info("Skipping extended interaction — inputs unchanged")
        return

    # ---- Load data ----
    cov_data = load_json(cov_path)
    raw_data = load_json(raw_path)
    adj_data = load_json(adj_path)

    res_hard = np.array([cov_data["research_profile_hard"][f"SDG{i}"] for i in range(1, N_SDG + 1)])
    pol_dw = np.array([cov_data["policy_profile_hard_docweighted"][f"SDG{i}"] for i in range(1, N_SDG + 1)])
    cov_gap = np.array([cov_data["coverage_gap_hard"][f"SDG{i}"] for i in range(1, N_SDG + 1)])
    dominance = res_hard - pol_dw

    raw_map = {r["sdg"]: r for r in raw_data["per_sdg"]}
    adj_map = {r["sdg"]: r for r in adj_data["per_sdg"]}

    raw_gap = np.array([float(raw_map[i]["semantic_gap"]) if raw_map[i]["semantic_gap"] is not None else np.nan for i in range(1, N_SDG + 1)])
    adj_gap = np.array([float(adj_map[i]["semantic_gap"]) if adj_map[i]["semantic_gap"] is not None else np.nan for i in range(1, N_SDG + 1)])
    reg_gap = raw_gap - adj_gap  # register component

    mask = np.isfinite(raw_gap) & np.isfinite(adj_gap)
    n_valid = int(mask.sum())
    log.info("Valid SDGs for correlation: %d / %d", n_valid, N_SDG)

    # ---- Compute correlations for each gap type ----
    def _correlate(y: np.ndarray, label: str) -> dict:
        return {
            "research_vs_" + label: pearson_spearman(res_hard[mask], y[mask]),
            "policy_vs_" + label: pearson_spearman(pol_dw[mask], y[mask]),
            "coverage_gap_vs_" + label: pearson_spearman(cov_gap[mask], y[mask]),
            "dominance_vs_" + label: pearson_spearman(dominance[mask], y[mask]),
        }

    results = {}
    results.update(_correlate(raw_gap, "raw_gap"))
    results.update(_correlate(adj_gap, "adjusted_gap"))
    results.update(_correlate(reg_gap, "register_component"))

    # ---- Headline numbers ----
    headline = {
        "coverage_vs_raw_gap": pearson_spearman(cov_gap[mask], raw_gap[mask]),
        "coverage_vs_adjusted_gap": pearson_spearman(cov_gap[mask], adj_gap[mask]),
        "coverage_vs_register_component": pearson_spearman(cov_gap[mask], reg_gap[mask]),
    }

    output = {
        "embedding_model": model,
        "n_valid_sdgs": n_valid,
        "headline": headline,
        "per_predictor": results,
        "per_sdg": [
            {
                "sdg": i + 1,
                "raw_gap": round(float(raw_gap[i]), 6) if np.isfinite(raw_gap[i]) else None,
                "adjusted_gap": round(float(adj_gap[i]), 6) if np.isfinite(adj_gap[i]) else None,
                "register_component": round(float(reg_gap[i]), 6) if np.isfinite(reg_gap[i]) else None,
                "coverage_gap": round(float(cov_gap[i]), 6),
            }
            for i in range(N_SDG)
        ],
    }

    with out_json.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    log.info("Saved: %s", out_json)

    # ---- Print headline ----
    h = headline
    log.info("")
    log.info("HEADLINE CORRELATIONS (Spearman):")
    log.info("  Coverage vs raw gap:              rho=%+.4f (p=%.4f)", h["coverage_vs_raw_gap"]["spearman_rho"], h["coverage_vs_raw_gap"]["spearman_p"])
    log.info("  Coverage vs adjusted gap (topic):  rho=%+.4f (p=%.4f)", h["coverage_vs_adjusted_gap"]["spearman_rho"], h["coverage_vs_adjusted_gap"]["spearman_p"])
    log.info("  Coverage vs register component:    rho=%+.4f (p=%.4f)", h["coverage_vs_register_component"]["spearman_rho"], h["coverage_vs_register_component"]["spearman_p"])

    record_fingerprint([out_json], fp, out_json)


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()

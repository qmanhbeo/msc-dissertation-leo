"""
Compare MiniLM (canonical) and MPNet (sensitivity) analysis outputs.

Computes Spearman correlations between the two models' outputs for:
  1. Coverage profiles (per-SDG research proportions, document-weighted)
  2. Semantic gap rankings (per-SDG semantic gap values)
  3. Semantic similarity values (per-SDG sim between research and policy sub-centroids)

Outputs:
  4_outputs/appendix/e_model_sensitivity/data/sensitivity_comparison.json
  4_outputs/appendix/e_model_sensitivity/tables/tab_model_sensitivity.tex

Run from project root:
    python 1_code/3_main_analysis/3_appendix/e_model_sensitivity/comparison.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, pearsonr

ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = ROOT / "1_code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))


logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

CANONICAL_OUTPUT = ROOT / "4_outputs" / "main" / "data"

REQUIRED_CANONICAL_FILES = [
    "4_2_coverage_document_weighted.json",
    "4_3_semantic_gap_distances.json",
    "4_4_interaction_correlation_asymmetry.json",
]

REQUIRED_COMPARISON_FILES = [
    "4_2_coverage_document_weighted.json",
    "4_3_semantic_gap_distances.json",
    "4_4_interaction_correlation_asymmetry.json",
]

N_SDG = 17
SDG_LABELS = list(range(1, N_SDG + 1))


def load_json(path: Path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def extract_coverage_profile(data: dict) -> np.ndarray:
    """Extract document-weighted research proportions per SDG (SDG order 1-17)."""
    raw = data.get("research_proportions", data.get("research_hard_proportions"))
    if raw is None:
        raise KeyError(f"Could not find research proportions in {list(data.keys())}")
    return np.array([raw[str(s)] for s in SDG_LABELS], dtype=np.float64)


def extract_semantic_gap_values(data: dict) -> np.ndarray:
    """Extract per-SDG semantic gap (SDG order 1-17)."""
    per_sdg = data.get("per_sdg")
    if per_sdg is None:
        raise KeyError("Missing 'per_sdg' in semantic gap data")
    gap_map = {r["sdg"]: r["semantic_gap"] for r in per_sdg}
    gaps = []
    for s in SDG_LABELS:
        g = gap_map.get(s)
        gaps.append(float(g) if g is not None else np.nan)
    return np.array(gaps, dtype=np.float64)


def extract_semantic_similarity_values(data: dict) -> np.ndarray:
    """Extract per-SDG semantic similarity (SDG order 1-17)."""
    per_sdg = data.get("per_sdg")
    if per_sdg is None:
        raise KeyError("Missing 'per_sdg' in semantic gap data")
    sim_map = {r["sdg"]: r["semantic_similarity"] for r in per_sdg}
    sims = []
    for s in SDG_LABELS:
        v = sim_map.get(s)
        sims.append(float(v) if v is not None else np.nan)
    return np.array(sims, dtype=np.float64)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default=str(ROOT / "4_outputs"))
    args_inner = p.parse_args()

    mpnet_root = Path(args_inner.output_dir) / "appendix" / "e_model_sensitivity"
    comparison_output = mpnet_root / "main" / "data"

    log.info("Canonical output dir: %s", CANONICAL_OUTPUT)
    log.info("Comparison output dir: %s", comparison_output)

    # Verify required files exist
    for fname in REQUIRED_CANONICAL_FILES:
        p = CANONICAL_OUTPUT / fname
        if not p.exists():
            raise FileNotFoundError(f"Canonical output missing: {p}")
    for fname in REQUIRED_COMPARISON_FILES:
        p = comparison_output / fname
        if not p.exists():
            raise FileNotFoundError(f"Comparison output missing: {p}")

    # Load canonical (MiniLM) data
    canonical_coverage = load_json(CANONICAL_OUTPUT / "4_2_coverage_document_weighted.json")
    canonical_gap = load_json(CANONICAL_OUTPUT / "4_3_semantic_gap_distances.json")
    canonical_interaction = load_json(CANONICAL_OUTPUT / "4_4_interaction_correlation_asymmetry.json")

    # Load comparison (MPNet) data
    comparison_coverage = load_json(comparison_output / "4_2_coverage_document_weighted.json")
    comparison_gap = load_json(comparison_output / "4_3_semantic_gap_distances.json")
    comparison_interaction = load_json(comparison_output / "4_4_interaction_correlation_asymmetry.json")

    # ---- Extract profiles ----
    res_prop_canonical = extract_coverage_profile(canonical_coverage)
    res_prop_mpnet = extract_coverage_profile(comparison_coverage)

    gaps_canonical = extract_semantic_gap_values(canonical_gap)
    gaps_mpnet = extract_semantic_gap_values(comparison_gap)

    sims_canonical = extract_semantic_similarity_values(canonical_gap)
    sims_mpnet = extract_semantic_similarity_values(comparison_gap)

    # ---- Coverage correlation ----
    valid_cov = ~(np.isnan(res_prop_canonical) | np.isnan(res_prop_mpnet))
    if valid_cov.sum() < 3:
        log.warning("Too few valid coverage values for correlation")
        cov_spearman_r, cov_spearman_p = np.nan, np.nan
        cov_pearson_r, cov_pearson_p = np.nan, np.nan
    else:
        cov_spearman_r, cov_spearman_p = spearmanr(res_prop_canonical[valid_cov], res_prop_mpnet[valid_cov])
        cov_pearson_r, cov_pearson_p = pearsonr(res_prop_canonical[valid_cov], res_prop_mpnet[valid_cov])

    log.info("Coverage profile comparison:")
    log.info("  Spearman ρ = %.4f (p=%.4f)", cov_spearman_r, cov_spearman_p)
    log.info("  Pearson  r = %.4f (p=%.4f)", cov_pearson_r, cov_pearson_p)

    # ---- Semantic gap correlation ----
    valid_gap = ~(np.isnan(gaps_canonical) | np.isnan(gaps_mpnet))
    if valid_gap.sum() < 3:
        log.warning("Too few valid gap values for correlation")
        gap_spearman_r, gap_spearman_p = np.nan, np.nan
        gap_pearson_r, gap_pearson_p = np.nan, np.nan
    else:
        gap_spearman_r, gap_spearman_p = spearmanr(gaps_canonical[valid_gap], gaps_mpnet[valid_gap])
        gap_pearson_r, gap_pearson_p = pearsonr(gaps_canonical[valid_gap], gaps_mpnet[valid_gap])

    log.info("Semantic gap comparison:")
    log.info("  Spearman ρ = %.4f (p=%.4f)", gap_spearman_r, gap_spearman_p)
    log.info("  Pearson  r = %.4f (p=%.4f)", gap_pearson_r, gap_pearson_p)

    # ---- Semantic similarity correlation ----
    valid_sim = ~(np.isnan(sims_canonical) | np.isnan(sims_mpnet))
    if valid_sim.sum() < 3:
        log.warning("Too few valid similarity values for correlation")
        sim_spearman_r, sim_spearman_p = np.nan, np.nan
        sim_pearson_r, sim_pearson_p = np.nan, np.nan
    else:
        sim_spearman_r, sim_spearman_p = spearmanr(sims_canonical[valid_sim], sims_mpnet[valid_sim])
        sim_pearson_r, sim_pearson_p = pearsonr(sims_canonical[valid_sim], sims_mpnet[valid_sim])

    log.info("Semantic similarity comparison:")
    log.info("  Spearman ρ = %.4f (p=%.4f)", sim_spearman_r, sim_spearman_p)
    log.info("  Pearson  r = %.4f (p=%.4f)", sim_pearson_r, sim_pearson_p)

    # ---- Per-SDG table data ----
    per_sdg_rows = []
    for i, s in enumerate(SDG_LABELS):
        per_sdg_rows.append({
            "sdg": s,
            "res_proportion_minilm": float(res_prop_canonical[i]) if not np.isnan(res_prop_canonical[i]) else None,
            "res_proportion_mpnet": float(res_prop_mpnet[i]) if not np.isnan(res_prop_mpnet[i]) else None,
            "semantic_gap_minilm": float(gaps_canonical[i]) if not np.isnan(gaps_canonical[i]) else None,
            "semantic_gap_mpnet": float(gaps_mpnet[i]) if not np.isnan(gaps_mpnet[i]) else None,
            "semantic_sim_minilm": float(sims_canonical[i]) if not np.isnan(sims_canonical[i]) else None,
            "semantic_sim_mpnet": float(sims_mpnet[i]) if not np.isnan(sims_mpnet[i]) else None,
        })

    # ---- Interaction comparison (where applicable) ----
    # Compare H25 correlation coefficients across models
    interaction_comparison = {}
    for key in ("coverage_gap_vs_semantic_gap", "research_proportion_vs_semantic_gap",
                 "research_dominance_vs_semantic_gap"):
        c_val = canonical_interaction.get(key)
        m_val = comparison_interaction.get(key)
        if c_val is not None and m_val is not None:
            interaction_comparison[key] = {
                "minilm_spearman_r": c_val.get("spearman_r"),
                "minilm_pearson_r": c_val.get("pearson_r"),
                "mpnet_spearman_r": m_val.get("spearman_r"),
                "mpnet_pearson_r": m_val.get("pearson_r"),
            }

    # ---- Build output ----
    results = {
        "model_sensitivity": {
            "canonical_model": "all-MiniLM-L6-v2",
            "comparison_model": "all-mpnet-base-v2",
            "canonical_slug": "minilm",
            "comparison_slug": "mpnet",
        },
        "coverage_profile": {
            "spearman_r": round(float(cov_spearman_r), 6) if not np.isnan(cov_spearman_r) else None,
            "spearman_p": round(float(cov_spearman_p), 6) if not np.isnan(cov_spearman_p) else None,
            "pearson_r": round(float(cov_pearson_r), 6) if not np.isnan(cov_pearson_r) else None,
            "pearson_p": round(float(cov_pearson_p), 6) if not np.isnan(cov_pearson_p) else None,
            "n_sdg_valid": int(valid_cov.sum()),
        },
        "semantic_gap": {
            "spearman_r": round(float(gap_spearman_r), 6) if not np.isnan(gap_spearman_r) else None,
            "spearman_p": round(float(gap_spearman_p), 6) if not np.isnan(gap_spearman_p) else None,
            "pearson_r": round(float(gap_pearson_r), 6) if not np.isnan(gap_pearson_r) else None,
            "pearson_p": round(float(gap_pearson_p), 6) if not np.isnan(gap_pearson_p) else None,
            "n_sdg_valid": int(valid_gap.sum()),
        },
        "semantic_similarity": {
            "spearman_r": round(float(sim_spearman_r), 6) if not np.isnan(sim_spearman_r) else None,
            "spearman_p": round(float(sim_spearman_p), 6) if not np.isnan(sim_spearman_p) else None,
            "pearson_r": round(float(sim_pearson_r), 6) if not np.isnan(sim_pearson_r) else None,
            "pearson_p": round(float(sim_pearson_p), 6) if not np.isnan(sim_pearson_p) else None,
            "n_sdg_valid": int(valid_sim.sum()),
        },
        "per_sdg": per_sdg_rows,
        "interaction_comparison": interaction_comparison,
    }

    # ---- Save ----
    model_sens_data_dir = mpnet_root / "data"
    model_sens_tables_dir = mpnet_root / "tables"
    model_sens_data_dir.mkdir(parents=True, exist_ok=True)
    model_sens_tables_dir.mkdir(parents=True, exist_ok=True)

    out_json = model_sens_data_dir / "sensitivity_comparison.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    log.info("Saved: %s", out_json)

    # ---- LaTeX table ----
    tex_lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"& \multicolumn{2}{c}{Coverage proportion} & \multicolumn{2}{c}{Semantic gap} & \multicolumn{2}{c}{Semantic sim.} \\",
        r"\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7}",
        r"SDG & MiniLM & MPNet & MiniLM & MPNet & MiniLM & MPNet \\",
        r"\midrule",
    ]
    for row in per_sdg_rows:
        s = row["sdg"]
        cov_c = f"{row['res_proportion_minilm']:.4f}" if row['res_proportion_minilm'] is not None else "---"
        cov_m = f"{row['res_proportion_mpnet']:.4f}" if row['res_proportion_mpnet'] is not None else "---"
        gap_c = f"{row['semantic_gap_minilm']:.4f}" if row['semantic_gap_minilm'] is not None else "---"
        gap_m = f"{row['semantic_gap_mpnet']:.4f}" if row['semantic_gap_mpnet'] is not None else "---"
        sim_c = f"{row['semantic_sim_minilm']:.4f}" if row['semantic_sim_minilm'] is not None else "---"
        sim_m = f"{row['semantic_sim_mpnet']:.4f}" if row['semantic_sim_mpnet'] is not None else "---"
        tex_lines.append(f"SDG {s:2d} & {cov_c} & {cov_m} & {gap_c} & {gap_m} & {sim_c} & {sim_m} \\\\")

    tex_lines.extend([
        r"\midrule",
    ])

    if cov_spearman_r is not None and not np.isnan(cov_spearman_r):
        tex_lines.append(
            rf"Spearman $\rho$ & \multicolumn{{2}}{{c}}{{{cov_spearman_r:.4f}}}"
            rf" & \multicolumn{{2}}{{c}}{{{gap_spearman_r:.4f}}}"
            rf" & \multicolumn{{2}}{{c}}{{{sim_spearman_r:.4f}}} \\"
        )
    tex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
    ])

    out_tex = model_sens_tables_dir / "tab_model_sensitivity.tex"
    out_tex.write_text("\n".join(tex_lines) + "\n", encoding="utf-8")
    log.info("Saved: %s", out_tex)

    log.info("")
    log.info("Model sensitivity comparison complete.")
    log.info("Coverage Spearman ρ = %.4f, Gap Spearman ρ = %.4f, Similarity Spearman ρ = %.4f",
             cov_spearman_r, gap_spearman_r, sim_spearman_r)


if __name__ == "__main__":
    main()

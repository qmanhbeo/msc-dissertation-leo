"""
Compare MiniLM (canonical) and MPNet (sensitivity) analysis outputs.

Computes Spearman correlations between the two models' outputs for:
  1. Coverage profiles (per-SDG research proportions, document-weighted)
  2. Validation F1 scores (per-SDG centroid-validation F1)
  3. Semantic gap rankings (per-SDG semantic gap values)


 
Outputs:
  4_outputs/appendix/d_model_sensitivity/data/sensitivity_comparison.json
  4_outputs/appendix/d_model_sensitivity/tables/tab_model_sensitivity.tex

Run from project root:
    python 1_code/3_main_analysis/3_appendix/d_model_sensitivity/comparison.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, pearsonr, rankdata

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from model_utils import N_SDG

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

CANONICAL_OUTPUT = ROOT / "4_outputs" / "main" / "data"

REQUIRED_CANONICAL_FILES = [
    "4_1_validation_results.json",
    "4_2_coverage_document_weighted.json",
    "4_3_semantic_gap_distances.json",
    "4_4_interaction_correlation_asymmetry.json",
]

REQUIRED_COMPARISON_FILES = [
    "4_1_validation_results.json",
    "4_2_coverage_document_weighted.json",
    "4_3_semantic_gap_distances.json",
    "4_4_interaction_correlation_asymmetry.json",
]

SDG_LABELS = list(range(1, N_SDG + 1))


def load_json(path: Path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def extract_coverage_profile(data: dict) -> np.ndarray:
    """Extract document-weighted research proportions per SDG (SDG order 1-17)."""
    raw = data["research_profile_hard"]
    return np.array([raw[f"SDG{s}"] for s in SDG_LABELS], dtype=np.float64)


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


def extract_f1_per_sdg(data: dict) -> np.ndarray:
    """Extract per-SDG validation F1 (SDG order 1-17)."""
    raw = data["per_sdg_f1"]
    return np.array([float(raw[str(s)]) for s in SDG_LABELS], dtype=np.float64)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default=str(PROJECT_ROOT / "4_outputs"))
    args_inner = p.parse_args()

    mpnet_root = Path(args_inner.output_dir) / "appendix" / "d_model_sensitivity"
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
    canonical_validation = load_json(CANONICAL_OUTPUT / "4_1_validation_results.json")
    canonical_coverage = load_json(CANONICAL_OUTPUT / "4_2_coverage_document_weighted.json")
    canonical_gap = load_json(CANONICAL_OUTPUT / "4_3_semantic_gap_distances.json")
    canonical_interaction = load_json(CANONICAL_OUTPUT / "4_4_interaction_correlation_asymmetry.json")

    # Load comparison (MPNet) data
    comparison_validation = load_json(comparison_output / "4_1_validation_results.json")
    comparison_coverage = load_json(comparison_output / "4_2_coverage_document_weighted.json")
    comparison_gap = load_json(comparison_output / "4_3_semantic_gap_distances.json")
    comparison_interaction = load_json(comparison_output / "4_4_interaction_correlation_asymmetry.json")

    # ---- Extract profiles ----
    f1_canonical = extract_f1_per_sdg(canonical_validation)
    f1_mpnet = extract_f1_per_sdg(comparison_validation)

    res_prop_canonical = extract_coverage_profile(canonical_coverage)
    res_prop_mpnet = extract_coverage_profile(comparison_coverage)

    gaps_canonical = extract_semantic_gap_values(canonical_gap)
    gaps_mpnet = extract_semantic_gap_values(comparison_gap)

    # ---- Compute ranks (1 = highest) ----
    f1_canonical_rank = rankdata(-f1_canonical, method="min")
    f1_mpnet_rank = rankdata(-f1_mpnet, method="min")
    cov_canonical_rank = rankdata(-res_prop_canonical, method="min")
    cov_mpnet_rank = rankdata(-res_prop_mpnet, method="min")
    gap_canonical_rank = rankdata(-gaps_canonical, method="min")   # 1 = largest gap
    gap_mpnet_rank = rankdata(-gaps_mpnet, method="min")

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

    # ---- Validation F1 correlation ----
    valid_f1 = ~(np.isnan(f1_canonical) | np.isnan(f1_mpnet))
    if valid_f1.sum() < 3:
        log.warning("Too few valid F1 values for correlation")
        f1_spearman_r, f1_spearman_p = np.nan, np.nan
        f1_pearson_r, f1_pearson_p = np.nan, np.nan
    else:
        f1_spearman_r, f1_spearman_p = spearmanr(f1_canonical[valid_f1], f1_mpnet[valid_f1])
        f1_pearson_r, f1_pearson_p = pearsonr(f1_canonical[valid_f1], f1_mpnet[valid_f1])

    log.info("Validation F1 comparison:")
    log.info("  Spearman ρ = %.4f (p=%.4f)", f1_spearman_r, f1_spearman_p)
    log.info("  Pearson  r = %.4f (p=%.4f)", f1_pearson_r, f1_pearson_p)

    # ---- Per-SDG table data ----
    per_sdg_rows = []
    for i, s in enumerate(SDG_LABELS):
        per_sdg_rows.append({
            "sdg": s,
            "f1_minilm": float(f1_canonical[i]) if not np.isnan(f1_canonical[i]) else None,
            "f1_mpnet": float(f1_mpnet[i]) if not np.isnan(f1_mpnet[i]) else None,
            "f1_minilm_rank": int(f1_canonical_rank[i]) if not np.isnan(f1_canonical[i]) else None,
            "f1_mpnet_rank": int(f1_mpnet_rank[i]) if not np.isnan(f1_mpnet[i]) else None,
            "res_proportion_minilm": float(res_prop_canonical[i]) if not np.isnan(res_prop_canonical[i]) else None,
            "res_proportion_mpnet": float(res_prop_mpnet[i]) if not np.isnan(res_prop_mpnet[i]) else None,
            "res_proportion_minilm_rank": int(cov_canonical_rank[i]) if not np.isnan(res_prop_canonical[i]) else None,
            "res_proportion_mpnet_rank": int(cov_mpnet_rank[i]) if not np.isnan(res_prop_mpnet[i]) else None,
            "semantic_gap_minilm": float(gaps_canonical[i]) if not np.isnan(gaps_canonical[i]) else None,
            "semantic_gap_mpnet": float(gaps_mpnet[i]) if not np.isnan(gaps_mpnet[i]) else None,
            "semantic_gap_minilm_rank": int(gap_canonical_rank[i]) if not np.isnan(gaps_canonical[i]) else None,
            "semantic_gap_mpnet_rank": int(gap_mpnet_rank[i]) if not np.isnan(gaps_mpnet[i]) else None,
        })

    # ---- Interaction comparison (where applicable) ----
    # Compare H25 correlation coefficients across models
    interaction_comparison = {}
    key_map = {
        "a_res_prop_vs_sem_gap": "research_proportion_vs_semantic_gap",
        "b_cov_gap_abs_vs_sem_gap": "coverage_gap_vs_semantic_gap",
        "c_res_dominance_vs_sem_gap": "research_dominance_vs_semantic_gap",
    }
    canon_corr = canonical_interaction.get("correlation", {}).get("correlations_primary_observed", {})
    compar_corr = comparison_interaction.get("correlation", {}).get("correlations_primary_observed", {})
    for short_key, long_key in key_map.items():
        c_val = canon_corr.get(short_key)
        m_val = compar_corr.get(short_key)
        if c_val is not None and m_val is not None:
            interaction_comparison[long_key] = {
                "minilm_spearman_r": c_val.get("spearman_rho"),
                "minilm_pearson_r": c_val.get("pearson_r"),
                "mpnet_spearman_r": m_val.get("spearman_rho"),
                "mpnet_pearson_r": m_val.get("pearson_r"),
            }

    # ---- Build output ----
    results = {
        "model_sensitivity": {
            "canonical_model": "all-MiniLM-L6-v2",
            "comparison_model": "all-mpnet-base-v2",
            "canonical_label": "minilm",
            "comparison_label": "mpnet",
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
        "validation_f1": {
            "spearman_r": round(float(f1_spearman_r), 6) if not np.isnan(f1_spearman_r) else None,
            "spearman_p": round(float(f1_spearman_p), 6) if not np.isnan(f1_spearman_p) else None,
            "pearson_r": round(float(f1_pearson_r), 6) if not np.isnan(f1_pearson_r) else None,
            "pearson_p": round(float(f1_pearson_p), 6) if not np.isnan(f1_pearson_p) else None,
            "n_sdg_valid": int(valid_f1.sum()),
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
        r"\small",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"& \multicolumn{2}{c}{Validation F1} & \multicolumn{2}{c}{Coverage gap} & \multicolumn{2}{c}{Semantic gap} \\",
        r"\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7}",
        r"SDG & MiniLM & MPNet & MiniLM & MPNet & MiniLM & MPNet \\",
        r"\midrule",
    ]
    for row in per_sdg_rows:
        s = row["sdg"]
        f1_c = f"{row['f1_minilm']:.3f}" if row['f1_minilm'] is not None else "---"
        f1_c_r = f"({row['f1_minilm_rank']:2d})" if row['f1_minilm_rank'] is not None else ""
        f1_m = f"{row['f1_mpnet']:.3f}" if row['f1_mpnet'] is not None else "---"
        f1_m_r = f"({row['f1_mpnet_rank']:2d})" if row['f1_mpnet_rank'] is not None else ""
        cov_c = f"{row['res_proportion_minilm']:.4f}" if row['res_proportion_minilm'] is not None else "---"
        cov_c_r = f"({row['res_proportion_minilm_rank']:2d})" if row['res_proportion_minilm_rank'] is not None else ""
        cov_m = f"{row['res_proportion_mpnet']:.4f}" if row['res_proportion_mpnet'] is not None else "---"
        cov_m_r = f"({row['res_proportion_mpnet_rank']:2d})" if row['res_proportion_mpnet_rank'] is not None else ""
        gap_c = f"{row['semantic_gap_minilm']:.3f}" if row['semantic_gap_minilm'] is not None else "---"
        gap_c_r = f"({row['semantic_gap_minilm_rank']:2d})" if row['semantic_gap_minilm_rank'] is not None else ""
        gap_m = f"{row['semantic_gap_mpnet']:.3f}" if row['semantic_gap_mpnet'] is not None else "---"
        gap_m_r = f"({row['semantic_gap_mpnet_rank']:2d})" if row['semantic_gap_mpnet_rank'] is not None else ""
        tex_lines.append(f"SDG {s:2d} & {f1_c}\,{f1_c_r} & {f1_m}\,{f1_m_r} & {cov_c}\,{cov_c_r} & {cov_m}\,{cov_m_r} & {gap_c}\,{gap_c_r} & {gap_m}\,{gap_m_r} \\\\")

    tex_lines.extend([
        r"\midrule",
    ])

    if cov_spearman_r is not None and not np.isnan(cov_spearman_r):
        tex_lines.append(
            rf"Spearman $\rho$ & \multicolumn{{2}}{{c}}{{{f1_spearman_r:.4f}}}"
            rf" & \multicolumn{{2}}{{c}}{{{cov_spearman_r:.4f}}}"
            rf" & \multicolumn{{2}}{{c}}{{{gap_spearman_r:.4f}}} \\"
        )
    tex_lines.extend([
        r"\multicolumn{7}{l}{\footnotesize Note. Rank 1 = highest F1 / highest coverage proportion / largest semantic gap per model.}\\",
        r"\bottomrule",
        r"\end{tabular}",
    ])

    out_tex = model_sens_tables_dir / "tab_model_sensitivity.tex"
    out_tex.write_text("\n".join(tex_lines) + "\n", encoding="utf-8")
    log.info("Saved: %s", out_tex)

    log.info("")
    log.info("Model sensitivity comparison complete.")
    log.info("F1 Spearman ρ = %.4f, Coverage Spearman ρ = %.4f, Gap Spearman ρ = %.4f",
             f1_spearman_r, cov_spearman_r, gap_spearman_r)


if __name__ == "__main__":
    main()

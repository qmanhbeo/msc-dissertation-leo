"""
Test H25: is there a correlation between coverage gap and semantic gap across SDGs?

H25 (headline hypothesis):
  SDGs with the highest research attention will show the largest within-SDG semantic gaps —
  i.e. the SDGs where research engages most are precisely where research and policy diverge
  most strongly in framing within that SDG.

  Two operationalisations are tested:
    (a) Correlation between research_proportion (SDG coverage in research corpus) and
        semantic_gap (within-SDG semantic divergence).
        A POSITIVE correlation = H25 supported: more research attention → more divergence.
        The hypothesis labels this a "negative correlation" between coverage and semantic
        *similarity* — equivalent to a positive correlation with semantic *gap*.

    (b) Correlation between coverage_gap_abs (|research% - policy%|) and semantic_gap.
        This tests whether SDGs that are unbalanced between corpora are also more divergent.

    (c) Correlation between research_dominance (research% - policy%) and semantic_gap.
        Signed version: are SDGs where research dominates more divergent than SDGs where
        policy dominates?

  All three operationalisations are reported. The primary test for H25 is (a).

H26 asymmetry diagnostic:
  Computed from active score artifacts. The mean top-SDG score when papers are scored
  against OSDG centroids is compared to the mean top-SDG score when policy segments are scored
  against research centroids. This is treated as an appendix-style directional diagnostic,
  not as a headline result.

Statistics:
  Pearson r (parametric, assumes linear relationship) and Spearman ρ (rank correlation,
  non-parametric) are both reported. With only 17 data points (one per SDG), both tests
  have low power. We report p-values but interpret them cautiously — with n=17, even
  strong trends may not reach p < 0.05. The qualitative pattern (top-5 SDGs in scatter)
  is the primary evidence.

  ASSUMPTION (A-STAT): With 17 SDGs, correlation statistics have limited power. A nominally
  non-significant result does not rule out a real pattern; the direction and magnitude of
  the correlation are the primary evidence, not the p-value.

  ASSUMPTION (A-SDG4): SDG 4's 22% research proportion may be inflated due to ML "learning"
  terminology aligning with the Education centroid (not genuine SDG 4 research engagement).
  Correlation results are reported with and without SDG 4 to test sensitivity.

Inputs:
  4_outputs/main/data/4_2_coverage_document_weighted.json            per-SDG research + policy profiles (doc-weighted)
  4_outputs/main/data/4_3_semantic_gap_distances.json                per-SDG semantic gap (segment_cap=50)

  4_outputs/main/data/4_4_interaction_correlation_asymmetry.json     H25 correlation results + H26 asymmetry
  4_outputs/main/data/4_4_interaction_scatter_data.csv               per-SDG data table for plotting (SDG, research%, policy%,
                                  coverage_gap, semantic_gap, semantic_similarity)
  4_outputs/main/tables/*.tex            generated LaTeX macros/tables

Run from project root (after the canonical coverage and semantic outputs exist):
    python 1_code/3_main_analysis/1_canonical/2_coverage_semantic_interaction.py
"""

import csv
import json
import logging
import math
import numpy as np
import argparse
import sys
from pathlib import Path
from scipy import stats

ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = ROOT / "1_code"
ANALYSIS_ROOT = Path(__file__).resolve().parents[1]
SHARED_DIR = ANALYSIS_ROOT / "0_shared"
for path in (CODE_ROOT, SHARED_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from research_score_shards import aggregate_research_scores
from shared_utils import ensure_canonical_outputs, require_output_files
from model_slug_utils import scored_dir_for_model, DEFAULT_EMBED_MODEL

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEFAULT_OUTPUT_ROOT = Path("4_outputs")

N_SDG = 17

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def load_json(path: Path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def pearson_and_spearman(x: np.ndarray, y: np.ndarray, label: str) -> dict:
    """Compute Pearson r and Spearman ρ between x and y, return as dict."""
    assert len(x) == len(y), f"Length mismatch: {len(x)} vs {len(y)}"
    if len(x) < 3:
        raise ValueError(f"{label}: need at least 3 observations, got {len(x)}")
    r, r_p   = stats.pearsonr(x, y)
    rho, s_p = stats.spearmanr(x, y)
    result = {
        "n": len(x),
        "pearson_r": round(float(r), 6),
        "pearson_p": round(float(r_p), 6),
        "spearman_rho": round(float(rho), 6),
        "spearman_p": round(float(s_p), 6),
    }
    log.info(
        "  %-55s  Pearson r=%.3f (p=%.3f)  Spearman ρ=%.3f (p=%.3f)",
        label, r, r_p, rho, s_p
    )
    return result


def correlation_or_skip(
    x: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    label: str,
) -> dict:
    sdgs = [i + 1 for i, keep in enumerate(mask) if keep]
    if int(mask.sum()) < 3:
        log.warning("  %-55s  skipped (n=%d)", label, int(mask.sum()))
        return {
            "n": int(mask.sum()),
            "sdgs": sdgs,
            "skipped": True,
            "reason": "fewer_than_3_valid_sdgs",
        }
    result = pearson_and_spearman(x[mask], y[mask], label)
    result["sdgs"] = sdgs
    result["skipped"] = False
    return result


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute correlation and asymmetry outputs into the canonical output folder.")
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_ROOT))
    p.add_argument("--model", default=DEFAULT_EMBED_MODEL, help=argparse.SUPPRESS)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    model = args.model
    scored_dir = scored_dir_for_model(model)
    paper_scores_manifest = scored_dir / "paper_scores_shards" / "metadata" / "manifest.json"
    pol_vs_res_path = scored_dir / "policy_scores_vs_research.npy"
    policy_scores_path = scored_dir / "policy_scores.npy"

    layout = ensure_canonical_outputs(Path(args.output_dir))
    require_output_files(layout.data_dir, ["4_2_coverage_document_weighted.json", "4_3_semantic_gap_distances.json"])

    coverage_gap_path = layout.data_dir / "4_2_coverage_document_weighted.json"
    semantic_gap_path = layout.data_dir / "4_3_semantic_gap_distances.json"
    out_corr = layout.data_dir / "4_4_interaction_correlation_asymmetry.json"
    out_scatter = layout.data_dir / "4_4_interaction_scatter_data.csv"
    tables_dir = layout.tables_dir
    log.info("Canonical output dir: %s", layout.data_dir)

    # ---- Load coverage data ----
    log.info("Loading coverage gap: %s", coverage_gap_path)
    cov_data = load_json(coverage_gap_path)

    # Extract per-SDG arrays (1-indexed labels, so SDG{i} = SDG i, row index i-1)
    res_hard    = np.array([cov_data["research_profile_hard"][f"SDG{i}"] for i in range(1, 18)])
    pol_dw_hard = np.array([cov_data["policy_profile_hard_docweighted"][f"SDG{i}"] for i in range(1, 18)])
    cov_gap_abs = np.array([cov_data["coverage_gap_hard"][f"SDG{i}"] for i in range(1, 18)])

    # Signed research dominance: positive = research > policy, negative = policy > research.
    res_dominance = res_hard - pol_dw_hard

    # ---- Load semantic gap data ----
    log.info("Loading semantic gap: %s", semantic_gap_path)
    sem_data = load_json(semantic_gap_path)

    per_sdg = {r["sdg"]: r for r in sem_data["per_sdg"]}
    sem_gap = np.array(
        [
            np.nan if per_sdg[i]["semantic_gap"] is None else float(per_sdg[i]["semantic_gap"])
            for i in range(1, 18)
        ],
        dtype=float,
    )
    sem_sim = np.array(
        [
            np.nan if per_sdg[i]["semantic_similarity"] is None else float(per_sdg[i]["semantic_similarity"])
            for i in range(1, 18)
        ],
        dtype=float,
    )
    unreliable = np.array([bool(per_sdg[i]["unreliable"]) for i in range(1, 18)], dtype=bool)

    # Only SDGs with finite semantic values are eligible for correlation or plotting.
    available_mask = np.isfinite(sem_gap) & np.isfinite(sem_sim)
    available_sdgs = [i + 1 for i, keep in enumerate(available_mask) if keep]
    reliable_mask = available_mask & ~unreliable
    reliable_sdgs = [i + 1 for i, keep in enumerate(reliable_mask) if keep]
    missing_sdgs = [i + 1 for i, keep in enumerate(available_mask) if not keep]
    log.info("Semantic-gap SDGs available for correlation: %s  (n=%d)", available_sdgs, len(available_sdgs))
    if missing_sdgs:
        log.warning("Excluded from correlation due to missing semantic gap: %s", missing_sdgs)
    log.info("Reliable SDGs for correlation: %s  (n=%d)", reliable_sdgs, len(reliable_sdgs))

    # ---- Build per-SDG table ----
    log.info("")
    log.info("Per-SDG data table:")
    log.info("  %-6s %-12s %-12s %-12s %-12s %-10s", "SDG", "res%", "pol%", "cov_gap", "sem_gap", "reliable")
    for i in range(N_SDG):
        sem_gap_display = "N/A" if not np.isfinite(sem_gap[i]) else f"{sem_gap[i]:.4f}"
        log.info(
            "  SDG %2d  %10.2f%%  %10.2f%%  %10.4f  %10s  %s",
            i + 1,
            res_hard[i] * 100,
            pol_dw_hard[i] * 100,
            cov_gap_abs[i],
            sem_gap_display,
            "✓" if reliable_mask[i] else "✗",
        )

    # ---- Correlation tests ----
    # Use all SDGs with finite semantic gaps first, then re-check with only reliable ones.
    log.info("")
    log.info("=" * 70)
    log.info("CORRELATION TESTS")
    log.info("=" * 70)
    log.info("")
    log.info("OBSERVED SDGs WITH FINITE SEMANTIC GAP (n=%d):", int(available_mask.sum()))
    corr_primary = {
        "a_res_prop_vs_sem_gap": correlation_or_skip(
            res_hard, sem_gap, available_mask, "(a) research_proportion vs semantic_gap"
        ),
        "b_cov_gap_abs_vs_sem_gap": correlation_or_skip(
            cov_gap_abs, sem_gap, available_mask, "(b) coverage_gap_abs vs semantic_gap"
        ),
        "c_res_dominance_vs_sem_gap": correlation_or_skip(
            res_dominance, sem_gap, available_mask, "(c) research_dominance (res%-pol%) vs semantic_gap"
        ),
    }

    # Reliable SDGs only (excludes SDG 10 if flagged).
    if reliable_mask.sum() < available_mask.sum():
        log.info("")
        log.info("RELIABLE SDGs ONLY (n=%d):", reliable_mask.sum())
        corr_reliable = {
            "a_res_prop_vs_sem_gap": correlation_or_skip(
                res_hard, sem_gap, reliable_mask, "(a) research_proportion vs semantic_gap [reliable only]"
            ),
            "b_cov_gap_abs_vs_sem_gap": correlation_or_skip(
                cov_gap_abs, sem_gap, reliable_mask, "(b) coverage_gap_abs vs semantic_gap [reliable only]"
            ),
            "c_res_dominance_vs_sem_gap": correlation_or_skip(
                res_dominance, sem_gap, reliable_mask, "(c) research_dominance vs semantic_gap [reliable only]"
            ),
        }
    else:
        corr_reliable = None

    # Sensitivity: exclude SDG 4 (suspected ML terminology artefact in research SDG 4 = 22%).
    # If SDG 4's high research proportion is genuine → correlation should be robust to exclusion.
    # If SDG 4 is an artefact inflating the research proportion → excluding it tests robustness.
    excl4_mask = available_mask.copy()
    excl4_mask[3] = False   # SDG 4 is index 3
    log.info("")
    log.info("SENSITIVITY — EXCLUDING SDG 4 (suspected ML 'learning' terminology artefact):")
    corr_excl4 = {
        "a_res_prop_vs_sem_gap": correlation_or_skip(
            res_hard, sem_gap, excl4_mask, "(a) research_proportion vs semantic_gap [excl SDG4]"
        ),
        "b_cov_gap_abs_vs_sem_gap": correlation_or_skip(
            cov_gap_abs, sem_gap, excl4_mask, "(b) coverage_gap_abs vs semantic_gap [excl SDG4]"
        ),
    }

    # ---- Correlation interpretation ----
    # Primary test: correlation (a) research_proportion vs semantic_gap.
    primary_stats = corr_primary["a_res_prop_vs_sem_gap"]
    if primary_stats["skipped"]:
        raise RuntimeError("Primary H25 correlation could not be computed: fewer than 3 valid SDGs.")
    r_primary = primary_stats["pearson_r"]
    rho_primary = primary_stats["spearman_rho"]
    p_primary = primary_stats["pearson_p"]

    log.info("")
    log.info("=" * 70)
    log.info("CORRELATION INTERPRETATION (PRIMARY TEST: research_proportion vs semantic_gap)")
    log.info("=" * 70)
    if r_primary > 0.3:
        correlation_direction = "SUPPORTED"
        correlation_story = (
            "Positive correlation: SDGs with higher research attention show greater within-SDG "
            "semantic divergence from policy."
        )
    elif r_primary < -0.3:
        correlation_direction = "CONTRADICTED"
        correlation_story = (
            "Negative correlation: SDGs with more research attention are MORE semantically aligned "
            "with policy — suggests research reduces semantic divergence over time."
        )
    else:
        correlation_direction = "NOT SUPPORTED (WEAK CORRELATION)"
        correlation_story = (
            "Near-zero correlation: research attention does not predict semantic gap direction. "
            "Coverage and semantic divergence are largely independent dimensions."
        )
    log.info("  Correlation direction: %s", correlation_direction)
    log.info("  Pearson r=%.3f  p=%.3f  Spearman ρ=%.3f", r_primary, p_primary, rho_primary)
    log.info("  %s", correlation_story)

    # Top-3 outliers (SDGs furthest from the trend line).
    log.info("")
    log.info("NOTABLE SDGS (highest research % + gap relationship):")
    pairs = sorted(
        [(i + 1, res_hard[i], sem_gap[i]) for i in range(N_SDG) if np.isfinite(sem_gap[i])],
        key=lambda x: x[1], reverse=True
    )[:5]
    for sdg, rp, sg in pairs:
        log.info("  SDG %2d  res=%.1f%%  sem_gap=%.3f  (%s)",
                 sdg, rp*100, sg,
                 "high_coverage_high_gap" if sg > 0.25 else "high_coverage_low_gap")

    # ---- Asymmetry diagnostic ----
    log.info("")
    log.info("=" * 70)
    log.info("DIRECTIONAL ASYMMETRY DIAGNOSTIC")
    log.info("=" * 70)
    research = aggregate_research_scores(paper_scores_manifest, scored_dir)
    pol_vs_research = np.load(pol_vs_res_path)
    policy_scores = np.load(policy_scores_path)

    mean_paper_top    = float(research["mean_top_overall"])
    mean_pol_vs_res   = float(pol_vs_research.max(axis=1).mean())
    a15_policy_top    = float(policy_scores.max(axis=1).mean())
    a15_gap           = a15_policy_top - mean_paper_top

    # Per-SDG: mean score of policy segments for their top research centroid.
    pol_assignments = pol_vs_research.argmax(axis=1)
    mean_pol_per_sdg = np.array([
        float(pol_vs_research[pol_assignments == j, j].mean()) if (pol_assignments == j).sum() > 0 else 0.0
        for j in range(N_SDG)
    ])

    log.info("  Research papers vs OSDG centroids — mean top sim: %.4f", mean_paper_top)
    log.info("  Policy segments vs research centroids — mean top sim: %.4f", mean_pol_vs_res)
    asym_supported = mean_pol_vs_res > mean_paper_top
    asym_gap = mean_pol_vs_res - mean_paper_top
    log.info("  Asymmetry gap (policy - research): %.4f  → Asymmetry direction %s",
             asym_gap, "OBSERVED" if asym_supported else "NOT OBSERVED")
    if asym_supported:
        log.info(
            "  Diagnostic reading: policy-facing texts score closer to research-derived "
            "centroids than papers score to OSDG-derived centroids."
        )
    else:
        log.info(
            "  Diagnostic reading: the observed direction does not favour the asymmetry claim."
        )

    # ---- Save outputs ----
    results = {
        "correlation": {
            "hypothesis": (
                "SDGs with the highest research attention show the largest within-SDG semantic "
                "gaps (research and policy talk past each other at points of engagement)."
            ),
            "primary_test": (
                "research_proportion vs semantic_gap "
                f"(Pearson r, Spearman rho, n={primary_stats['n']})"
            ),
            "direction_found": correlation_direction,
            "story": correlation_story,
            "caveats": [
                f"n={primary_stats['n']} SDGs in the primary test gives low statistical power.",
                "SDG 4 research proportion (22%) may be inflated by ML 'learning' terminology.",
                "Hard assignment creates zero-sum profiles — SDGs with overlapping centroids "
                "(e.g. SDG 1/8/10 cluster) may trade assignments artificially.",
            ],
            "available_semantic_gap_sdgs": available_sdgs,
            "missing_semantic_gap_sdgs": missing_sdgs,
            "correlations_primary_observed": corr_primary,
            "correlations_reliable_only": corr_reliable,
            "correlations_excl_sdg4": corr_excl4,
        },
        "asymmetry": {
            "hypothesis": (
                "Policy-facing texts may score closer to research-derived centroids than papers score "
                "to OSDG-derived centroids, but this is treated only as a directional diagnostic."
            ),
            "mean_paper_top_vs_osdg": round(mean_paper_top, 6),
            "mean_policy_top_vs_research": round(mean_pol_vs_res, 6),
            "asymmetry_gap": round(asym_gap, 6),
            "supported": asym_supported,
            "caveats": [
                f"A15 FLAG: policy scores against OSDG centroids are inflated by {a15_gap:.3f} relative "
                "to paper scores. The asymmetry diagnostic may partly reflect this calibration bias, "
                "not genuine directional asymmetry.",
                "Research centroids are built via hard assignment from OSDG centroids — a "
                "circularity that may reduce apparent research-centroid distance.",
            ],
        },
        "per_sdg_table": [
            {
                "sdg": i + 1,
                "research_proportion": round(float(res_hard[i]), 6),
                "policy_proportion_docweighted": round(float(pol_dw_hard[i]), 6),
                "coverage_gap_abs": round(float(cov_gap_abs[i]), 6),
                "research_dominance": round(float(res_dominance[i]), 6),
                "semantic_gap": None if not np.isfinite(sem_gap[i]) else round(float(sem_gap[i]), 6),
                "semantic_similarity": None if not np.isfinite(sem_sim[i]) else round(float(sem_sim[i]), 6),
                "reliable": bool(reliable_mask[i]),
            }
            for i in range(N_SDG)
        ],
    }

    with out_corr.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    log.info("Saved: %s", out_corr)

    # CSV scatter table for plotting.
    with out_scatter.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "sdg", "research_pct", "policy_pct_docweighted",
            "coverage_gap_abs", "research_dominance",
            "semantic_gap", "semantic_similarity", "reliable"
        ])
        writer.writeheader()
        for i in range(N_SDG):
            writer.writerow({
                "sdg": i + 1,
                "research_pct": round(float(res_hard[i]) * 100, 4),
                "policy_pct_docweighted": round(float(pol_dw_hard[i]) * 100, 4),
                "coverage_gap_abs": round(float(cov_gap_abs[i]), 6),
                "research_dominance": round(float(res_dominance[i]), 6),
                "semantic_gap": "" if not np.isfinite(sem_gap[i]) else round(float(sem_gap[i]), 6),
                "semantic_similarity": "" if not np.isfinite(sem_sim[i]) else round(float(sem_sim[i]), 6),
                "reliable": int(reliable_mask[i]),
            })
    log.info("Saved: %s", out_scatter)

    log.info("")
    log.info("Next step: python 1_code/4_visualization/plot_figures.py")

    # ---- Write LaTeX generated outputs ----
    gen_dir = tables_dir

    # Median research% across 17 SDGs.
    median_res_pct = float(np.median(res_hard * 100.0))

    # Excl-SDG4 correlation values
    excl4 = corr_excl4["a_res_prop_vs_sem_gap"]
    primary = corr_primary["a_res_prop_vs_sem_gap"]
    n_primary = int(primary["n"])
    z_primary = math.atanh(r_primary)
    z_se = 1 / math.sqrt(n_primary - 3)
    r_ci_lo = math.tanh(z_primary - 1.96 * z_se)
    r_ci_hi = math.tanh(z_primary + 1.96 * z_se)

    def _fmt(v):
        """Format float for LaTeX: negative values get a minus sign, 3 d.p."""
        s = f"{abs(v):.3f}"
        return f"-{s}" if v < 0 else s

    def _fmt2(v):
        """Format float for LaTeX with 2 d.p."""
        s = f"{abs(v):.2f}"
        return f"-{s}" if v < 0 else s

    # num_interaction.tex — macro definitions
    num_lines = [
        "% Auto-generated by 1_code/3_main_analysis/1_canonical/2_coverage_semantic_interaction.py — do not edit manually",
        rf"\newcommand{{\HPrimaryN}}{{{n_primary}}}",
        rf"\newcommand{{\HPrimaryPearsonR}}{{{_fmt(r_primary)}}}",
        rf"\newcommand{{\HPrimaryPearsonP}}{{{primary['pearson_p']:.3f}}}",
        rf"\newcommand{{\HPrimaryPearsonCiLower}}{{{_fmt2(r_ci_lo)}}}",
        rf"\newcommand{{\HPrimaryPearsonCiUpper}}{{{_fmt2(r_ci_hi)}}}",
        rf"\newcommand{{\HPrimarySpearmanRho}}{{{_fmt(rho_primary)}}}",
        rf"\newcommand{{\HPrimarySpearmanP}}{{{primary['spearman_p']:.3f}}}",
        rf"\newcommand{{\HExclSdgFourPearsonR}}{{{_fmt(excl4['pearson_r'])}}}",
        rf"\newcommand{{\HExclSdgFourPearsonP}}{{{excl4['pearson_p']:.3f}}}",
        rf"\newcommand{{\HExclSdgFourSpearmanRho}}{{{_fmt(excl4['spearman_rho'])}}}",
        rf"\newcommand{{\HExclSdgFourSpearmanP}}{{{excl4['spearman_p']:.3f}}}",
        rf"\newcommand{{\HAsymPolicyScore}}{{{mean_pol_vs_res:.3f}}}",
        rf"\newcommand{{\HAsymResearchScore}}{{{mean_paper_top:.3f}}}",
        rf"\newcommand{{\HAsymGap}}{{{_fmt(asym_gap)}}}",
        rf"\newcommand{{\MedianResearchPct}}{{{median_res_pct:.2f}}}",
    ]
    (gen_dir / "num_interaction.tex").write_text("\n".join(num_lines) + "\n", encoding="utf-8")
    log.info("Saved: %s", gen_dir / "num_interaction.tex")

    # tab_interaction.tex — full tabular block
    tab_lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Test & Pearson $r$ & $p$ & Spearman $\rho$ & $p$ \\",
        r"\midrule",
        rf"Primary observed SDGs ($n=\HPrimaryN$) & \HPrimaryPearsonR & \HPrimaryPearsonP"
        rf" & \HPrimarySpearmanRho & \HPrimarySpearmanP \\",
        rf"Excluding SDG 4 & \HExclSdgFourPearsonR & \HExclSdgFourPearsonP"
        rf" & \HExclSdgFourSpearmanRho & \HExclSdgFourSpearmanP \\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    (gen_dir / "tab_interaction.tex").write_text("\n".join(tab_lines) + "\n", encoding="utf-8")
    log.info("Saved: %s", gen_dir / "tab_interaction.tex")


if __name__ == "__main__":
    main()

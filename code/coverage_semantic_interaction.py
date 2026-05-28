"""
Test H25: is there a correlation between coverage gap and semantic gap across SDGs?

H25 (headline hypothesis):
  SDGs with the highest research attention will show the largest within-SDG semantic gaps —
  i.e. the SDGs where research engages most are precisely where research and policy talk
  past each other (high coverage → high divergence within that SDG).

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

H26 asymmetry:
  Computed from active score artifacts. The mean top-SDG score when papers are scored
  against OSDG centroids is compared to the mean top-SDG score when policy chunks are scored
  against research centroids. An asymmetry where policy engages research framing more than
  research engages policy framing supports H26.

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
  data/coverage_gap.json        per-SDG research + policy profiles (doc-weighted)
  data/semantic_gap.json        per-SDG semantic gap (chunk_cap=50)
  data/paper_scores.npy         (6172, 17) — for H26 paper top-scores
  data/policy_scores_vs_research.npy  (47005, 17) — for H26 policy vs research centroids

Outputs:
  data/h25_correlation.json     H25 correlation results + H26 asymmetry
  data/h25_scatter.csv          per-SDG data table for plotting (SDG, research%, policy%,
                                coverage_gap, semantic_gap, semantic_similarity)

Run from project root (after semantic_gap.py):
    python code/coverage_semantic_interaction.py
"""

import csv
import json
import logging
import math
import numpy as np
from pathlib import Path
from scipy import stats

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR = Path("data")

COVERAGE_GAP_PATH   = DATA_DIR / "coverage_gap.json"
SEMANTIC_GAP_PATH   = DATA_DIR / "semantic_gap.json"
PAPER_SCORES_PATH   = DATA_DIR / "paper_scores.npy"
POL_VS_RES_PATH     = DATA_DIR / "policy_scores_vs_research.npy"

OUT_CORR   = DATA_DIR / "h25_correlation.json"
OUT_SCATTER = DATA_DIR / "h25_scatter.csv"

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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    # ---- Load coverage data ----
    log.info("Loading coverage gap: %s", COVERAGE_GAP_PATH)
    cov_data = load_json(COVERAGE_GAP_PATH)

    # Extract per-SDG arrays (1-indexed labels, so SDG{i} = SDG i, row index i-1)
    res_hard    = np.array([cov_data["research_profile_hard"][f"SDG{i}"] for i in range(1, 18)])
    pol_dw_hard = np.array([cov_data["policy_profile_hard_docweighted"][f"SDG{i}"] for i in range(1, 18)])
    cov_gap_abs = np.array([cov_data["coverage_gap_hard"][f"SDG{i}"] for i in range(1, 18)])

    # Signed research dominance: positive = research > policy, negative = policy > research.
    res_dominance = res_hard - pol_dw_hard

    # ---- Load semantic gap data ----
    log.info("Loading semantic gap: %s", SEMANTIC_GAP_PATH)
    sem_data = load_json(SEMANTIC_GAP_PATH)

    per_sdg = {r["sdg"]: r for r in sem_data["per_sdg"]}
    sem_gap  = np.array([per_sdg[i]["semantic_gap"]         for i in range(1, 18)], dtype=float)
    sem_sim  = np.array([per_sdg[i]["semantic_similarity"]  for i in range(1, 18)], dtype=float)
    unreliable = np.array([per_sdg[i]["unreliable"] for i in range(1, 18)])

    # Reliability flags from semantic gap (SDG 10 has n_papers=20, borderline).
    reliable_mask = ~unreliable   # (17,) bool
    reliable_sdgs = [i+1 for i, r in enumerate(reliable_mask) if r]
    log.info("Reliable SDGs for correlation: %s  (n=%d)", reliable_sdgs, len(reliable_sdgs))

    # ---- Build per-SDG table ----
    log.info("")
    log.info("Per-SDG data table:")
    log.info("  %-6s %-12s %-12s %-12s %-12s %-10s", "SDG", "res%", "pol%", "cov_gap", "sem_gap", "reliable")
    for i in range(N_SDG):
        log.info("  SDG %2d  %10.2f%%  %10.2f%%  %10.4f  %10.4f  %s",
                 i+1, res_hard[i]*100, pol_dw_hard[i]*100,
                 cov_gap_abs[i], sem_gap[i], "✓" if reliable_mask[i] else "✗")

    # ---- H25 Correlations ----
    # Use all 17 SDGs first, then re-check with only reliable ones.
    log.info("")
    log.info("=" * 70)
    log.info("H25 CORRELATION TESTS")
    log.info("=" * 70)
    log.info("")
    log.info("ALL 17 SDGs:")
    corr_all = {
        "a_res_prop_vs_sem_gap": pearson_and_spearman(
            res_hard, sem_gap,
            "(a) research_proportion vs semantic_gap"
        ),
        "b_cov_gap_abs_vs_sem_gap": pearson_and_spearman(
            cov_gap_abs, sem_gap,
            "(b) coverage_gap_abs vs semantic_gap"
        ),
        "c_res_dominance_vs_sem_gap": pearson_and_spearman(
            res_dominance, sem_gap,
            "(c) research_dominance (res%-pol%) vs semantic_gap"
        ),
    }

    # Reliable SDGs only (excludes SDG 10 if flagged).
    if reliable_mask.sum() < N_SDG:
        log.info("")
        log.info("RELIABLE SDGs ONLY (n=%d):", reliable_mask.sum())
        corr_reliable = {
            "a_res_prop_vs_sem_gap": pearson_and_spearman(
                res_hard[reliable_mask], sem_gap[reliable_mask],
                "(a) research_proportion vs semantic_gap [reliable only]"
            ),
            "b_cov_gap_abs_vs_sem_gap": pearson_and_spearman(
                cov_gap_abs[reliable_mask], sem_gap[reliable_mask],
                "(b) coverage_gap_abs vs semantic_gap [reliable only]"
            ),
            "c_res_dominance_vs_sem_gap": pearson_and_spearman(
                res_dominance[reliable_mask], sem_gap[reliable_mask],
                "(c) research_dominance vs semantic_gap [reliable only]"
            ),
        }
    else:
        corr_reliable = None

    # Sensitivity: exclude SDG 4 (suspected ML terminology artefact in research SDG 4 = 22%).
    # If SDG 4's high research proportion is genuine → correlation should be robust to exclusion.
    # If SDG 4 is an artefact inflating the research proportion → excluding it tests robustness.
    excl4_mask = np.ones(N_SDG, dtype=bool)
    excl4_mask[3] = False   # SDG 4 is index 3
    log.info("")
    log.info("SENSITIVITY — EXCLUDING SDG 4 (suspected ML 'learning' terminology artefact):")
    corr_excl4 = {
        "a_res_prop_vs_sem_gap": pearson_and_spearman(
            res_hard[excl4_mask], sem_gap[excl4_mask],
            "(a) research_proportion vs semantic_gap [excl SDG4]"
        ),
        "b_cov_gap_abs_vs_sem_gap": pearson_and_spearman(
            cov_gap_abs[excl4_mask], sem_gap[excl4_mask],
            "(b) coverage_gap_abs vs semantic_gap [excl SDG4]"
        ),
    }

    # ---- H25 interpretation ----
    # Primary test: correlation (a) research_proportion vs semantic_gap.
    r_primary = corr_all["a_res_prop_vs_sem_gap"]["pearson_r"]
    rho_primary = corr_all["a_res_prop_vs_sem_gap"]["spearman_rho"]
    p_primary   = corr_all["a_res_prop_vs_sem_gap"]["pearson_p"]

    log.info("")
    log.info("=" * 70)
    log.info("H25 INTERPRETATION (PRIMARY TEST: research_proportion vs semantic_gap)")
    log.info("=" * 70)
    if r_primary > 0.3:
        h25_direction = "SUPPORTED"
        h25_story = (
            "Positive correlation: SDGs with higher research attention show greater within-SDG "
            "semantic divergence from policy — the 'talking past each other' story."
        )
    elif r_primary < -0.3:
        h25_direction = "CONTRADICTED"
        h25_story = (
            "Negative correlation: SDGs with more research attention are MORE semantically aligned "
            "with policy — suggests research reduces semantic divergence over time."
        )
    else:
        h25_direction = "NOT SUPPORTED (WEAK CORRELATION)"
        h25_story = (
            "Near-zero correlation: research attention does not predict semantic gap direction. "
            "Coverage and semantic divergence are largely independent dimensions."
        )
    log.info("  H25 direction: %s", h25_direction)
    log.info("  Pearson r=%.3f  p=%.3f  Spearman ρ=%.3f", r_primary, p_primary, rho_primary)
    log.info("  %s", h25_story)

    # Top-3 outliers (SDGs furthest from the trend line).
    log.info("")
    log.info("NOTABLE SDGS (highest research % + gap relationship):")
    pairs = sorted(
        [(i+1, res_hard[i], sem_gap[i]) for i in range(N_SDG)],
        key=lambda x: x[1], reverse=True
    )[:5]
    for sdg, rp, sg in pairs:
        log.info("  SDG %2d  res=%.1f%%  sem_gap=%.3f  (%s)",
                 sdg, rp*100, sg,
                 "high_coverage_high_gap" if sg > 0.25 else "high_coverage_low_gap")

    # ---- H26 asymmetry ----
    log.info("")
    log.info("=" * 70)
    log.info("H26 DIRECTIONAL ASYMMETRY")
    log.info("=" * 70)
    paper_scores    = np.load(PAPER_SCORES_PATH)           # (6172, 17)
    pol_vs_research = np.load(POL_VS_RES_PATH)             # (47005, 17)

    mean_paper_top    = float(paper_scores.max(axis=1).mean())
    mean_pol_vs_res   = float(pol_vs_research.max(axis=1).mean())

    # Per-SDG: mean score of research papers for their top OSDG centroid.
    paper_assignments = paper_scores.argmax(axis=1)
    mean_paper_per_sdg = np.array([
        float(paper_scores[paper_assignments == j, j].mean()) if (paper_assignments == j).sum() > 0 else 0.0
        for j in range(N_SDG)
    ])

    # Per-SDG: mean score of policy chunks for their top research centroid.
    pol_assignments = pol_vs_research.argmax(axis=1)
    mean_pol_per_sdg = np.array([
        float(pol_vs_research[pol_assignments == j, j].mean()) if (pol_assignments == j).sum() > 0 else 0.0
        for j in range(N_SDG)
    ])

    log.info("  Research papers vs OSDG centroids — mean top sim: %.4f", mean_paper_top)
    log.info("  Policy chunks vs research centroids — mean top sim: %.4f", mean_pol_vs_res)
    h26_supported = mean_pol_vs_res > mean_paper_top
    h26_gap = mean_pol_vs_res - mean_paper_top
    log.info("  Asymmetry gap (policy - research): %.4f  → H26 %s",
             h26_gap, "SUPPORTED" if h26_supported else "NOT SUPPORTED")
    if h26_supported:
        log.info(
            "  Interpretation: policy engages research framing more than research engages "
            "policy framing. Research ignores policy language more than policy ignores research."
        )
    else:
        log.info(
            "  Interpretation: research engages policy framing more than policy engages "
            "research framing — AGAINST H26 direction."
        )

    # ---- Save outputs ----
    results = {
        "h25": {
            "hypothesis": (
                "SDGs with the highest research attention show the largest within-SDG semantic "
                "gaps (research and policy talk past each other at points of engagement)."
            ),
            "primary_test": "research_proportion vs semantic_gap (Pearson r, Spearman rho, n=17)",
            "direction_found": h25_direction,
            "story": h25_story,
            "caveats": [
                "n=17 SDGs gives low statistical power — p-values are indicative only.",
                "SDG 4 research proportion (22%) may be inflated by ML 'learning' terminology.",
                "Hard assignment creates zero-sum profiles — SDGs with overlapping centroids "
                "(e.g. SDG 1/8/10 cluster) may trade assignments artificially.",
            ],
            "correlations_all17": corr_all,
            "correlations_reliable_only": corr_reliable,
            "correlations_excl_sdg4": corr_excl4,
        },
        "h26": {
            "hypothesis": (
                "Research papers score lower against OSDG centroids than policy chunks score "
                "against research centroids — research ignores policy framing more than vice versa."
            ),
            "mean_paper_top_vs_osdg": round(mean_paper_top, 6),
            "mean_policy_top_vs_research": round(mean_pol_vs_res, 6),
            "asymmetry_gap": round(h26_gap, 6),
            "supported": h26_supported,
            "caveats": [
                "A15 FLAG: policy scores against OSDG centroids are inflated by 0.191 relative "
                "to paper scores. The H26 asymmetry may partly reflect this calibration bias, "
                "not genuine research-policy framing asymmetry.",
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
                "semantic_gap": round(float(sem_gap[i]), 6),
                "semantic_similarity": round(float(sem_sim[i]), 6),
                "reliable": bool(reliable_mask[i]),
            }
            for i in range(N_SDG)
        ],
    }

    with OUT_CORR.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    log.info("Saved: %s", OUT_CORR)

    # CSV scatter table for plotting.
    with OUT_SCATTER.open("w", newline="", encoding="utf-8") as f:
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
                "semantic_gap": round(float(sem_gap[i]), 6),
                "semantic_similarity": round(float(sem_sim[i]), 6),
                "reliable": int(reliable_mask[i]),
            })
    log.info("Saved: %s", OUT_SCATTER)

    log.info("")
    log.info("Next step: python code/kaggle_context.py")

    # ---- Write LaTeX generated outputs ----
    gen_dir = DATA_DIR / "generated"
    gen_dir.mkdir(parents=True, exist_ok=True)

    # Median research% (midpoint of 8th and 9th values of 17 sorted values)
    sorted_res = sorted(float(v) * 100 for v in res_hard)
    median_res_pct = (sorted_res[7] + sorted_res[8]) / 2

    # Excl-SDG4 correlation values
    excl4 = corr_excl4["a_res_prop_vs_sem_gap"]
    primary = corr_all["a_res_prop_vs_sem_gap"]
    n_primary = len(res_hard)
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

    # num_h25.tex — macro definitions
    num_lines = [
        "% Auto-generated by code/coverage_semantic_interaction.py — do not edit manually",
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
        rf"\newcommand{{\HAsymGap}}{{{_fmt(h26_gap)}}}",
        rf"\newcommand{{\MedianResearchPct}}{{{median_res_pct:.2f}}}",
    ]
    (gen_dir / "num_h25.tex").write_text("\n".join(num_lines) + "\n", encoding="utf-8")
    log.info("Saved: %s", gen_dir / "num_h25.tex")

    # tab_h25.tex — full tabular block
    tab_lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Test & Pearson $r$ & $p$ & Spearman $\rho$ & $p$ \\",
        r"\midrule",
        rf"All 17 SDGs & \HPrimaryPearsonR & \HPrimaryPearsonP"
        rf" & \HPrimarySpearmanRho & \HPrimarySpearmanP \\",
        rf"Excluding SDG 4 & \HExclSdgFourPearsonR & \HExclSdgFourPearsonP"
        rf" & \HExclSdgFourSpearmanRho & \HExclSdgFourSpearmanP \\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    (gen_dir / "tab_h25.tex").write_text("\n".join(tab_lines) + "\n", encoding="utf-8")
    log.info("Saved: %s", gen_dir / "tab_h25.tex")


if __name__ == "__main__":
    main()

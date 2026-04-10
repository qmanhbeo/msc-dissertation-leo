"""
Contextualise SDG gaps against real-world SDG performance (Kaggle SDG Index).

This script joins the gap scores (coverage gap, semantic gap) with the Kaggle SDG Index
to test hypotheses H21–H24: do the SDGs with the largest coverage/semantic gaps correspond
to the SDGs with the worst real-world progress?

This is the final empirical analysis step. It answers the question: does the research-policy
misalignment concentrated on the most urgent real-world problems?

H21 — Neglected SDGs will have worse real-world progress scores.
H22 — The semantic gap will be largest for SDGs with the lowest SDG Index scores.
H23 — SDG 13 score inflation: the SDG Index may methodologically inflate SDG 13 (Climate).
H24 — Recent progress (2018–2022) will be weakest for the most neglected SDGs.

SDG Index data:
  Kaggle SDG Index (data/sdgindex/overview.csv) provides country-level SDG scores
  (0–100 scale) for 166+ countries from 2000–2022. We compute the global (unweighted)
  mean score per SDG in the most recent year available (2022 or latest) as the
  "real-world SDG performance" baseline.

  LIMITATION: The SDG Index scores aggregate multiple indicators per SDG and are designed
  to be comparable across countries, not corpora. High SDG Index score = good performance.
  A low SDG Index score = poor real-world progress = urgent need for research + policy.

  ASSUMPTION (A-SDGINDEX): The SDG Index global mean is dominated by wealthy countries
  (more data available, larger UN representation). This biases the index toward developed-
  country progress on SDGs that are easy to achieve at high income levels (e.g. SDG 4, SDG 9).
  Interpret correlations with awareness of this bias.

  ASSUMPTION (A-SDG17-INDEX): SDG 17 is not scored in the standard SDG Index (it covers
  "Partnerships" which is not measured as country outcomes). We exclude SDG 17 from the
  SDG Index correlation analysis (n=16 for index-joined tests).

Inputs:
  data/h25_correlation.json      per-SDG coverage + semantic gap table
  data/sdgindex/overview.csv     country-SDG-year performance scores

Outputs:
  data/sdg_context.json          gap scores joined with SDG Index means; H21–H24 correlations
  data/sdg_context.csv           per-SDG summary table for plotting

Run from project root (after coverage_semantic_interaction.py):
    python code/kaggle_context.py
"""

import csv
import json
import logging
import numpy as np
from pathlib import Path
from scipy import stats

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR      = Path("data")
SDG_INDEX_CSV = DATA_DIR / "kaggle/sdg_index_2000-2022.csv"
H25_PATH      = DATA_DIR / "h25_correlation.json"

OUT_CONTEXT     = DATA_DIR / "sdg_context.json"
OUT_CONTEXT_CSV = DATA_DIR / "sdg_context.csv"

# The SDG Index reports scores for SDGs 1–17, but SDG 17 is typically absent or NaN.
# We restrict to SDGs 1–16 for correlation analyses involving the SDG Index.
SDG_INDEX_SDGS = list(range(1, 17))   # SDGs 1–16

# Year to use for SDG performance baseline. Use latest available year.
# The SDG Index runs 2000–2022 in our data. 2022 = most recent.
TARGET_YEAR = 2022

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


def pearson_and_spearman(x: np.ndarray, y: np.ndarray, label: str, n_label: int = None) -> dict:
    """Compute Pearson r and Spearman ρ, return as dict."""
    n = len(x)
    r, r_p   = stats.pearsonr(x, y)
    rho, s_p = stats.spearmanr(x, y)
    n_str = f"n={n_label}" if n_label else f"n={n}"
    log.info(
        "  %-55s  r=%.3f (p=%.3f)  ρ=%.3f (p=%.3f)  [%s]",
        label, r, r_p, rho, s_p, n_str
    )
    return {
        "n": n,
        "pearson_r":    round(float(r), 6),
        "pearson_p":    round(float(r_p), 6),
        "spearman_rho": round(float(rho), 6),
        "spearman_p":   round(float(s_p), 6),
    }


# ---------------------------------------------------------------------------
# Load SDG Index scores
# ---------------------------------------------------------------------------
def load_sdg_index_means(csv_path: Path, target_year: int) -> dict[int, float]:
    """
    Load SDG Index data and return mean global score per SDG for the target year.

    Expected CSV columns (based on preprocess_sdgindex.py output):
      country, year, sdg_number, score
    OR a wide-format with sdg1..sdg16 column names.

    ASSUMPTION (A-SDG-INDEX-COL): We inspect the CSV header to handle both formats.
    The global mean is computed as an unweighted average across all countries.
    """
    means: dict[int, float] = {}

    try:
        with csv_path.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames
            log.info("SDG Index columns: %s", headers[:10])

            # Collect all rows for the target year.
            rows = [r for r in reader if r.get("year", "") == str(target_year)]
            if not rows:
                # Try without year filter (some preprocessed files have one row per country
                # with all SDG scores).
                f.seek(0)
                reader = csv.DictReader(f)
                rows = list(reader)
                log.warning(
                    "No rows for year=%d — using all rows (n=%d)", target_year, len(rows)
                )

            log.info("Rows for year=%d: %d", target_year, len(rows))
            if not rows:
                raise ValueError(f"No data found for year={target_year}")

            # Detect format: long (country, year, sdg_number, score) vs
            # wide (country, year, sdg1, sdg2, ..., sdg16).
            if "sdg_number" in (headers or []) and "score" in (headers or []):
                # Long format.
                sdg_values: dict[int, list[float]] = {s: [] for s in SDG_INDEX_SDGS}
                for row in rows:
                    try:
                        sdg = int(row["sdg_number"])
                        score = float(row["score"])
                        if sdg in sdg_values and not np.isnan(score):
                            sdg_values[sdg].append(score)
                    except (ValueError, KeyError):
                        continue
                for sdg, vals in sdg_values.items():
                    if vals:
                        means[sdg] = float(np.mean(vals))
            else:
                # Wide format: look for columns named sdg1, sdg2, ... or goal1, goal2, ...
                # The preprocess_sdgindex.py output likely has a different column scheme.
                # Try common naming patterns.
                sdg_cols = {}
                for h in (headers or []):
                    h_lower = h.lower().strip().lstrip("\ufeff")
                    for sdg in SDG_INDEX_SDGS:
                        # Match patterns: sdgN, goal_N, sdg_N, goal_N_score, sdg_N_score
                        candidates = (
                            f"sdg{sdg}", f"goal{sdg}", f"sdg_{sdg}", f"goal_{sdg}",
                            f"goal_{sdg}_score", f"sdg_{sdg}_score",
                        )
                        if h_lower in candidates:
                            sdg_cols[sdg] = h

                log.info("Detected SDG score columns: %s", list(sdg_cols.keys())[:10])
                for sdg, col in sdg_cols.items():
                    vals = []
                    for row in rows:
                        try:
                            v = float(row[col])
                            if not np.isnan(v) and v > 0:
                                vals.append(v)
                        except (ValueError, KeyError):
                            continue
                    if vals:
                        means[sdg] = float(np.mean(vals))

    except FileNotFoundError:
        log.error("SDG Index file not found: %s", csv_path)
        return {}

    log.info("Loaded SDG Index means for %d SDGs:", len(means))
    for sdg, m in sorted(means.items()):
        log.info("  SDG %2d: mean score=%.2f", sdg, m)

    return means


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    if OUT_CONTEXT.exists() and OUT_CONTEXT_CSV.exists():
        log.info("Outputs already exist. Delete to re-run.")
        return

    # ---- Load gap data ----
    log.info("Loading H25 per-SDG table: %s", H25_PATH)
    h25_data = load_json(H25_PATH)
    per_sdg_table = {r["sdg"]: r for r in h25_data["per_sdg_table"]}

    # ---- Load SDG Index ----
    log.info("Loading SDG Index: %s", SDG_INDEX_CSV)
    sdg_means = load_sdg_index_means(SDG_INDEX_CSV, TARGET_YEAR)

    if not sdg_means:
        log.error("Could not load SDG Index data — check file format.")
        # Save empty outputs so pipeline does not fail.
        with OUT_CONTEXT.open("w") as f:
            json.dump({"error": "SDG Index data unavailable", "h21_h24": None}, f, indent=2)
        return

    # ---- Join gap data with SDG Index ----
    # Build aligned arrays for SDGs 1–16 (SDG 17 excluded from index analysis).
    joined = []
    for sdg in SDG_INDEX_SDGS:
        g = per_sdg_table.get(sdg, {})
        idx_score = sdg_means.get(sdg)
        if idx_score is None:
            log.warning("SDG %d: no SDG Index score available", sdg)
            continue
        joined.append({
            "sdg": sdg,
            "research_proportion": g.get("research_proportion", np.nan),
            "policy_proportion": g.get("policy_proportion_docweighted", np.nan),
            "coverage_gap_abs": g.get("coverage_gap_abs", np.nan),
            "research_dominance": g.get("research_dominance", np.nan),
            "semantic_gap": g.get("semantic_gap", np.nan),
            "sdg_index_score": round(idx_score, 4),
        })

    log.info("")
    log.info("Joined data for %d SDGs:", len(joined))
    log.info("  %-6s  %-12s  %-12s  %-12s  %-12s  %-10s",
             "SDG", "res%", "sem_gap", "cov_gap", "res_dom", "idx_score")
    for r in joined:
        log.info("  SDG %2d  %10.2f%%  %10.4f  %10.4f  %10.4f  %10.2f",
                 r["sdg"], r["research_proportion"]*100, r["semantic_gap"],
                 r["coverage_gap_abs"], r["research_dominance"], r["sdg_index_score"])

    # ---- H21–H24 Correlations ----
    idx_scores   = np.array([r["sdg_index_score"]   for r in joined])
    sem_gaps     = np.array([r["semantic_gap"]       for r in joined])
    cov_gaps     = np.array([r["coverage_gap_abs"]   for r in joined])
    res_props    = np.array([r["research_proportion"] for r in joined])
    res_dom      = np.array([r["research_dominance"]  for r in joined])

    # Note: lower SDG Index score = worse real-world progress.
    # If neglected SDGs have worse performance, we expect negative correlation:
    # (high research/coverage proportion → higher SDG Index score)
    # or: (coverage gap → lower SDG Index score)

    log.info("")
    log.info("=" * 70)
    log.info("H21–H24 CORRELATIONS (SDGs 1–16 with SDG Index data)")
    log.info("=" * 70)
    log.info("Note: SDG Index score = real-world progress (higher = better).")
    log.info("H21: Neglected SDGs (low research coverage) → worse SDG performance.")
    log.info("H22: Larger semantic gap → worse SDG performance.")
    log.info("")

    h21_corr = pearson_and_spearman(
        res_props, idx_scores,
        "H21: research_proportion vs SDG_Index_score (positive=H21supported)",
        n_label=len(joined)
    )
    h21_cov_corr = pearson_and_spearman(
        cov_gaps, idx_scores,
        "H21b: coverage_gap_abs vs SDG_Index_score",
        n_label=len(joined)
    )
    h22_corr = pearson_and_spearman(
        sem_gaps, idx_scores,
        "H22: semantic_gap vs SDG_Index_score (negative=H22supported)",
        n_label=len(joined)
    )
    h22_res_dom = pearson_and_spearman(
        res_dom, idx_scores,
        "H22b: research_dominance vs SDG_Index_score",
        n_label=len(joined)
    )

    # H23 note: SDG 13 inflation check.
    # The SDG Index 2022 methodology note that SDG 13 scores are inflated because
    # commitment indicators (climate pledges) are scored positively, not outcomes.
    sdg13_score = sdg_means.get(13)
    sdg_scores_list = sorted(sdg_means.items(), key=lambda x: x[1], reverse=True)
    log.info("")
    log.info("H23 SDG 13 SCORE CHECK (index inflation diagnostic):")
    log.info("  SDG 13 mean global score: %.2f", sdg13_score if sdg13_score else -1)
    log.info("  SDG scores ranked (high to low):")
    for sdg, score in sdg_scores_list[:5]:
        log.info("    SDG %2d: %.2f", sdg, score)
    if sdg13_score and sdg13_score > 65:
        log.info("  H23 indicator: SDG 13 score appears elevated — possible inflation by commitment indicators.")

    # ---- Build output ----
    context_out = {
        "note": (
            "Gap scores joined with Kaggle SDG Index 2022 global mean scores. "
            "SDG Index score = composite real-world progress (0–100, higher=better). "
            "SDG 17 excluded (not scored in SDG Index). "
            "See Assumptions A-SDGINDEX and A-SDG17-INDEX."
        ),
        "target_year": TARGET_YEAR,
        "n_sdgs_joined": len(joined),
        "h21_research_vs_index": {
            "hypothesis": "SDGs with less research attention have worse real-world performance.",
            "expected_direction": "positive (high research → high SDG score)",
            "result": h21_corr,
            "interpretation": (
                "SUPPORTED" if h21_corr["pearson_r"] > 0.2
                else "NOT SUPPORTED" if h21_corr["pearson_r"] < -0.2
                else "WEAK / AMBIGUOUS"
            ),
        },
        "h22_semantic_vs_index": {
            "hypothesis": "Larger semantic gap corresponds to worse real-world SDG progress.",
            "expected_direction": "negative (high semantic gap → low SDG score)",
            "result": h22_corr,
            "interpretation": (
                "SUPPORTED" if h22_corr["pearson_r"] < -0.2
                else "NOT SUPPORTED" if h22_corr["pearson_r"] > 0.2
                else "WEAK / AMBIGUOUS"
            ),
        },
        "h21b_cov_gap_vs_index": h21_cov_corr,
        "h22b_res_dom_vs_index": h22_res_dom,
        "h23_sdg13_note": {
            "sdg13_mean_score": round(sdg13_score, 2) if sdg13_score else None,
            "note": (
                "SDG 13 score may be inflated by policy commitment indicators in the SDG Index. "
                "If SDG 13 ranks high despite large coverage/semantic gaps, this is H23 evidence."
            ),
        },
        "per_sdg": joined,
    }

    with OUT_CONTEXT.open("w", encoding="utf-8") as f:
        json.dump(context_out, f, indent=2)
    log.info("Saved: %s", OUT_CONTEXT)

    with OUT_CONTEXT_CSV.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "sdg", "research_proportion_pct", "policy_proportion_pct",
            "coverage_gap_abs", "research_dominance", "semantic_gap", "sdg_index_score"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in joined:
            writer.writerow({
                "sdg": r["sdg"],
                "research_proportion_pct": round(r["research_proportion"] * 100, 4),
                "policy_proportion_pct": round(r["policy_proportion"] * 100, 4),
                "coverage_gap_abs": round(r["coverage_gap_abs"], 6),
                "research_dominance": round(r["research_dominance"], 6),
                "semantic_gap": round(r["semantic_gap"], 6),
                "sdg_index_score": r["sdg_index_score"],
            })
    log.info("Saved: %s", OUT_CONTEXT_CSV)

    log.info("")
    log.info("Analysis pipeline complete.")
    log.info("All outputs saved to data/. See README.md for full pipeline description.")


if __name__ == "__main__":
    main()

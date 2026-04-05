"""
Preprocess SDR 2025 Overview data for gap analysis context.

Input:  data/sdgindex/sdr2025_data.xlsx  (sheet: "Overview")
Output: data/sdgindex/sdr2025_overview.csv      — per-country SDG scores and dashboard colors
        data/sdgindex/sdr2025_summary.json       — aggregate statistics per SDG

This data provides contextual grounding: after computing research-policy alignment
gaps per SDG, we join with real-world SDG performance to test whether gaps are
largest where global progress is weakest (H3).

Usage:
    python code/preprocess_sdgindex.py

Requires: pandas, openpyxl
"""

import json
import logging
from collections import defaultdict
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
INPUT_FILE = Path("data/sdgindex/sdr2025_data.xlsx")
OUTPUT_CSV = Path("data/sdgindex/sdr2025_overview.csv")
OUTPUT_JSON = Path("data/sdgindex/sdr2025_summary.json")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SDG metadata
# ---------------------------------------------------------------------------
SDG_NAMES = {
    1: "No Poverty",
    2: "Zero Hunger",
    3: "Good Health and Well-Being",
    4: "Quality Education",
    5: "Gender Equality",
    6: "Clean Water and Sanitation",
    7: "Affordable and Clean Energy",
    8: "Decent Work and Economic Growth",
    9: "Industry, Innovation and Infrastructure",
    10: "Reduced Inequalities",
    11: "Sustainable Cities and Communities",
    12: "Responsible Consumption and Production",
    13: "Climate Action",
    14: "Life Below Water",
    15: "Life on Land",
    16: "Peace, Justice and Strong Institutions",
    17: "Partnerships for the Goals",
}

DASHBOARD_COLORS = {"green": 4, "yellow": 3, "orange": 2, "red": 1}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    log.info("Loading %s", INPUT_FILE)

    try:
        # Overview sheet has colors/trends; SDR2025 Data has actual scores
        df = pd.read_excel(INPUT_FILE, sheet_name="SDR2025 Data")
    except Exception as e:
        log.error("Failed to load Excel file: %s", e)
        return

    log.info("Loaded %d rows, %d columns", df.shape[0], df.shape[1])

    # SDR2025 Data columns: Country, Goal 1 Score, Goal 1 Dash, Goal 1 Trend, ...
    # Score columns: "Goal N Score"
    # Color columns: "Goal N Dash" (green/yellow/orange/red)
    # Trend columns: "Goal N Trend" (arrows)

    # Build output dataframe
    records = []
    for _, row in df.iterrows():
        country = row.get("Country", "")
        if pd.isna(country) or country == "":
            continue

        record = {
            "country": country,
            "country_iso3": row.get("Country Code ISO3", ""),
            "region": row.get("Regions used for the SDR", ""),
            "sdg_index_score": pd.to_numeric(row.get("2025 SDG Index Score"), errors="coerce"),
            "sdg_index_rank": pd.to_numeric(row.get("2025 SDG Index Rank"), errors="coerce"),
        }

        for sdg_num in range(1, 18):
            score_col = f"Goal {sdg_num} Score"
            dash_col = f"Goal {sdg_num} Dash"
            trend_col = f"Goal {sdg_num} Trend"

            score_val = row.get(score_col, None)
            record[f"sdg{sdg_num}_score"] = float(score_val) if pd.notna(score_val) else None
            record[f"sdg{sdg_num}_dash"] = str(row.get(dash_col, "")) if pd.notna(row.get(dash_col)) else ""
            record[f"sdg{sdg_num}_trend"] = str(row.get(trend_col, "")) if pd.notna(row.get(trend_col)) else ""

        records.append(record)

    # Convert to DataFrame and save CSV
    out_df = pd.DataFrame(records)
    out_df.to_csv(OUTPUT_CSV, index=False)
    log.info("Saved CSV with %d countries → %s", len(out_df), OUTPUT_CSV)

    # Compute summary statistics per SDG
    summary = {
        "n_countries": len(out_df),
        "overall": {
            "mean_sdg_index": round(float(pd.to_numeric(out_df["sdg_index_score"], errors="coerce").mean()), 2),
            "median_sdg_index": round(float(pd.to_numeric(out_df["sdg_index_score"], errors="coerce").median()), 2),
            "std_sdg_index": round(float(pd.to_numeric(out_df["sdg_index_score"], errors="coerce").std()), 2),
        },
        "by_sdg": {},
    }

    for sdg_num in range(1, 18):
        col = f"sdg{sdg_num}_score"
        if col in out_df.columns:
            scores = pd.to_numeric(out_df[col], errors="coerce").dropna()
            summary["by_sdg"][sdg_num] = {
                "name": SDG_NAMES.get(sdg_num, f"SDG {sdg_num}"),
                "n_available": int(len(scores)),
                "mean_score": round(float(scores.mean()), 2) if len(scores) > 0 else None,
                "median_score": round(float(scores.median()), 2) if len(scores) > 0 else None,
                "min_score": round(float(scores.min()), 2) if len(scores) > 0 else None,
                "max_score": round(float(scores.max()), 2) if len(scores) > 0 else None,
            }

    with OUTPUT_JSON.open("w") as f:
        json.dump(summary, f, indent=2)
    log.info("Saved summary → %s", OUTPUT_JSON)

    # Print key stats
    print(f"\n{'='*60}")
    print("SDR 2025 Overview Summary")
    print(f"{'='*60}")
    print(f"Countries: {summary['n_countries']}")
    print(f"Overall SDG Index: mean={summary['overall']['mean_sdg_index']}, "
          f"median={summary['overall']['median_sdg_index']}")
    print(f"\nPer-SDG mean scores (sorted by score):")

    sdg_means = [(num, data["mean_score"]) for num, data in summary["by_sdg"].items()
                 if data["mean_score"] is not None]
    sdg_means.sort(key=lambda x: x[1] if x[1] is not None else 0)

    for sdg_num, mean_score in sdg_means:
        name = SDG_NAMES.get(sdg_num, f"SDG {sdg_num}")
        bar_len = int((mean_score or 0) / 5)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        print(f"  SDG {sdg_num:2d} ({name[:30]:30s}) {mean_score:5.1f} {bar}")

    print(f"{'='*60}\n")
    print(f"Outputs:")
    print(f"  - {OUTPUT_CSV} ({len(out_df)} countries)")
    print(f"  - {OUTPUT_JSON} (summary statistics)")
    print(f"\nUse this data to contextualize alignment gaps:")
    print(f"  - Lower SDG scores = more off-track globally")
    print(f"  - Compare gap magnitude with SDG performance to test H3")


if __name__ == "__main__":
    main()

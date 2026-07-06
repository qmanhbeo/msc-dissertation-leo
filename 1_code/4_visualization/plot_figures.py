"""
plot_figures.py — Generate dissertation figures from analysis outputs.

Produces three publication-quality figures for the dissertation:

    Figure 1 — Coverage profiles: research vs policy (horizontal grouped bar chart)
    Figure 2 — Semantic gap by SDG (horizontal bar chart, sorted descending)
    Figure 3 — Coverage vs semantic gap scatter (diagnostic map)

Inputs:
     4_outputs/main/data/4_4_interaction_scatter_data.csv      — per-SDG metrics table from coverage_semantic_interaction.py
     4_outputs/main/data/4_2_coverage_document_weighted.json  — corpus-level n counts for legend labels

Outputs:
    4_outputs/main/figures/fig1_coverage_profiles.pdf
    4_outputs/main/figures/fig2_semantic_gap.pdf
    4_outputs/main/figures/fig3_coverage_semantic_scatter.pdf

Run:
    python 1_code/4_visualization/plot_figures.py
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-dissertation")

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for WSL
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = ROOT / "1_code"
SHARED_DIR = ROOT / "1_code" / "3_main_analysis" / "0_shared"
for path in (CODE_ROOT, SHARED_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
from model_utils import DEFAULT_OUTPUT_ROOT
from shared_utils import ensure_canonical_outputs, require_output_files


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate figures from the canonical output folder.")
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_ROOT))
    return p.parse_args()


# ---------------------------------------------------------------------------
# Shared style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 150,
})

RESEARCH_COLOR = "#2166AC"   # blue
POLICY_COLOR   = "#D6604D"   # red-orange

SDG_SHORT = {
    1:  "SDG 1\nNo Poverty",
    2:  "SDG 2\nZero Hunger",
    3:  "SDG 3\nGood Health",
    4:  "SDG 4\nEducation†",
    5:  "SDG 5\nGender Equality",
    6:  "SDG 6\nClean Water",
    7:  "SDG 7\nClean Energy",
    8:  "SDG 8\nDecent Work",
    9:  "SDG 9\nInnovation",
    10: "SDG 10\nReduced Ineq.",
    11: "SDG 11\nSust. Cities",
    12: "SDG 12\nConsumption",
    13: "SDG 13\nClimate‡",
    14: "SDG 14\nLife Below Water",
    15: "SDG 15\nLife on Land",
    16: "SDG 16\nPeace & Justice",
    17: "SDG 17\nPartnerships‡",
}


def main() -> None:
    args = parse_args()
    layout = ensure_canonical_outputs(Path(args.output_dir))
    require_output_files(layout.data_dir, ["4_4_interaction_scatter_data.csv", "4_2_coverage_document_weighted.json"])
    figures_dir = layout.figures_dir

    print(f"Canonical output dir: {layout.data_dir}")

    # -----------------------------------------------------------------------
    # Load data
    # -----------------------------------------------------------------------
    df = pd.read_csv(layout.data_dir / "4_4_interaction_scatter_data.csv")
    df = df.sort_values("sdg").reset_index(drop=True)
    df["semantic_gap"] = pd.to_numeric(df["semantic_gap"], errors="coerce")
    df["semantic_similarity"] = pd.to_numeric(df["semantic_similarity"], errors="coerce")
    df_sem_valid = df[df["semantic_gap"].notna()].copy()
    if df_sem_valid.empty:
        raise RuntimeError("No finite semantic-gap rows available for figure generation.")

    # Medians for the diagonal thresholds used in the diagnostic map
    median_research_pct = df["research_pct"].median()
    median_semantic_gap = df_sem_valid["semantic_gap"].median()
    mean_semantic_gap = df_sem_valid["semantic_gap"].mean()

    # Load corpus-level counts for legend labels
    with open(layout.data_dir / "4_2_coverage_document_weighted.json") as f:
        _cov_counts = json.load(f)
    N_RESEARCH_PAPERS = _cov_counts["n_research_papers"]
    N_POLICY_DOCS = _cov_counts["n_policy_documents"]

    # -----------------------------------------------------------------------
    # Figure 1 — Coverage profiles comparison
    # -----------------------------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(8.5, 6))

    df_sorted = df.sort_values("sdg", ascending=True).reset_index(drop=True)
    y = np.arange(len(df_sorted))
    height = 0.38

    ax1.barh(
        y - height / 2,
        df_sorted["policy_pct_docweighted"],
        height=height,
        color=POLICY_COLOR,
        alpha=0.88,
        label=f"Policy (document-weighted %, n = {N_POLICY_DOCS})",
    )
    ax1.barh(
        y + height / 2,
        df_sorted["research_pct"],
        height=height,
        color=RESEARCH_COLOR,
        alpha=0.88,
        label=f"Research (%, n = {N_RESEARCH_PAPERS})",
    )

    # Raw count annotations at bar ends
    research_counts = (df_sorted["research_pct"] / 100 * N_RESEARCH_PAPERS).astype(int)
    policy_counts = (df_sorted["policy_pct_docweighted"] / 100 * N_POLICY_DOCS).astype(int)

    def _fmt(n):
        return f"{n/1_000:.0f}k" if n >= 1_000 else str(n)

    for i in range(len(df_sorted)):
        ax1.text(
            df_sorted["policy_pct_docweighted"].iloc[i] + 0.3,
            i - height / 2,
            _fmt(policy_counts.iloc[i]),
            va="center", ha="left", fontsize=5.5,
        )
        ax1.text(
            df_sorted["research_pct"].iloc[i] + 0.3,
            i + height / 2,
            _fmt(research_counts.iloc[i]),
            va="center", ha="left", fontsize=5.5,
        )

    labels = [SDG_SHORT[int(row["sdg"])].replace("\n", " ") for _, row in df_sorted.iterrows()]
    ax1.set_yticks(y)
    ax1.set_yticklabels(labels, fontsize=7.5)
    ax1.set_xlabel("Proportion of corpus assigned to SDG (%)")
    # ax1.set_title("Coverage profiles by SDG", fontsize=8.5, loc="left")
    ax1.legend(loc="upper right")
    ax1.axvline(0, color="black", linewidth=0.5)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.invert_yaxis()

    fig1.tight_layout()
    fig1.savefig(figures_dir / "fig1_coverage_profiles.pdf", bbox_inches="tight")
    fig1.savefig(figures_dir / "fig1_coverage_profiles.png", bbox_inches="tight", dpi=150)
    plt.close(fig1)
    print("Saved: fig1_coverage_profiles.pdf")

    # -----------------------------------------------------------------------
    # Figure 2 — Semantic gap by SDG
    # -----------------------------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(7, 5.5))

    df_sem = df_sem_valid.sort_values("semantic_gap", ascending=False).reset_index(drop=True)
    y = np.arange(len(df_sem))
    colors = [RESEARCH_COLOR if v > median_semantic_gap else "#92C5DE" for v in df_sem["semantic_gap"]]

    ax2.barh(y, df_sem["semantic_gap"], color=colors, alpha=0.88)
    for i, val in enumerate(df_sem["semantic_gap"]):
        ax2.text(val + 0.008, i, f"{val:.3f}", va="center", ha="left", fontsize=6)
    ax2.axvline(median_semantic_gap, color="grey", linestyle="--", linewidth=1,
                label=f"Median ({median_semantic_gap:.3f})")
    ax2.axvline(mean_semantic_gap, color="black", linestyle=":", linewidth=1,
                label=f"Mean ({mean_semantic_gap:.3f})")
    ax2.set_yticks(y)
    ax2.set_yticklabels([SDG_SHORT[int(r["sdg"])].replace("\n", " ") for _, r in df_sem.iterrows()], fontsize=7.5)
    ax2.set_xlabel("Semantic gap (1 − cosine similarity between research and policy sub-centroids)")
    # ax2.set_title("Within-SDG semantic gap by SDG\nHigher values indicate greater research-policy semantic divergence", fontsize=8.5, loc="left")

    high_patch = mpatches.Patch(color=RESEARCH_COLOR, alpha=0.88, label="Above median gap")
    low_patch = mpatches.Patch(color="#92C5DE", alpha=0.88, label="Below median gap")
    ax2.legend(handles=[
        high_patch, low_patch,
        plt.Line2D([0], [0], color="grey", linestyle="--", label=f"Median ({median_semantic_gap:.3f})"),
        plt.Line2D([0], [0], color="black", linestyle=":", label=f"Mean ({mean_semantic_gap:.3f})"),
    ], fontsize=7.5)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.invert_yaxis()

    fig2.tight_layout()
    fig2.savefig(figures_dir / "fig2_semantic_gap.pdf", bbox_inches="tight")
    fig2.savefig(figures_dir / "fig2_semantic_gap.png", bbox_inches="tight", dpi=150)
    plt.close(fig2)
    print("Saved: fig2_semantic_gap.pdf")

    # -----------------------------------------------------------------------
    # Figure 3 — Coverage vs semantic gap scatter (diagnostic map)
    # -----------------------------------------------------------------------
    fig3, ax3 = plt.subplots(figsize=(7.5, 6))

    ax3.axvline(median_research_pct, color="grey", linestyle="--", linewidth=1, alpha=0.7)
    ax3.axhline(median_semantic_gap, color="grey", linestyle="--", linewidth=1, alpha=0.7)

    xlim = (0, df["research_pct"].max() * 1.12)
    ylim = (max(0.0, df_sem_valid["semantic_gap"].min() * 0.92), df_sem_valid["semantic_gap"].max() * 1.08)

    for _, row in df_sem_valid.iterrows():
        sdg = int(row["sdg"])
        x = row["research_pct"]
        y = row["semantic_gap"]
        ax3.scatter(x, y, s=55, color=RESEARCH_COLOR, zorder=5, alpha=0.85)
        offsets = {4: (0.3, 0.003), 9: (-0.5, 0.003), 3: (0.3, -0.004),
                   16: (0.3, 0.003), 8: (0.2, 0.003), 17: (0.3, -0.005),
                   13: (0.3, -0.004), 12: (0.3, 0.003)}
        dx, dy = offsets.get(sdg, (0.25, 0.002))
        ax3.annotate(f"SDG {sdg}", (x, y), xytext=(x + dx, y + dy),
                     fontsize=7.5, color="black",
                     arrowprops=dict(arrowstyle="-", color="grey", lw=0.5) if sdg in offsets else None)

    ax3.set_xlim(xlim)
    ax3.set_ylim(ylim)
    ax3.set_xlabel("Research corpus SDG coverage (%)")
    ax3.set_ylabel("Within-SDG semantic gap (1 − cosine similarity)")
    ax3.set_title(
        "Coverage vs semantic gap: diagnostic map",
        fontsize=8.5,
        loc="left",
    )
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)

    fig3.tight_layout()
    fig3.savefig(figures_dir / "fig3_coverage_semantic_scatter.pdf", bbox_inches="tight")
    fig3.savefig(figures_dir / "fig3_coverage_semantic_scatter.png", bbox_inches="tight", dpi=150)
    plt.close(fig3)
    print("Saved: fig3_coverage_semantic_scatter.pdf")
    print(f"\\nAll figures saved to {figures_dir}")


if __name__ == "__main__":
    main()

"""
plot_figures.py — Generate dissertation figures from analysis outputs.

Produces three publication-quality figures for the dissertation:

    Figure 1 — Coverage profiles: research vs policy (horizontal grouped bar chart)
    Figure 2 — Semantic gap by SDG (horizontal bar chart, sorted descending)
    Figure 3 — Coverage vs semantic gap scatter (2×2 typology visualisation)

Inputs:
    outputs/<run_name>/h25_scatter.csv        — per-SDG metrics table from coverage_semantic_interaction.py

Outputs:
    outputs/<run_name>/figures/fig1_coverage_profiles.pdf
    outputs/<run_name>/figures/fig2_semantic_gap.pdf
    outputs/<run_name>/figures/fig3_coverage_semantic_scatter.pdf

Run:
    python code/visualization/plot_figures.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for WSL
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
from shared.output_runs import latest_run_dir, require_run_with_files, resolve_run_dir

DEFAULT_OUTPUT_ROOT = Path("outputs")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate figures from a run-scoped output folder.")
    p.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    p.add_argument("--input-run", default=None, help="Run folder containing h25_scatter.csv")
    p.add_argument("--run-name", default=None, help="Run folder to write figures to. Defaults to input run.")
    p.add_argument("--run-label", default="analysis")
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
    output_root = Path(args.output_root)

    if args.input_run:
        input_dir = require_run_with_files(output_root, args.input_run, ["h25_scatter.csv"])
    else:
        latest = latest_run_dir(output_root, ["h25_scatter.csv"])
        if latest is None:
            raise FileNotFoundError(
                "No compatible output run found containing h25_scatter.csv. "
                "Run coverage_semantic_interaction.py first, or pass --input-run."
            )
        input_dir = latest

    if args.run_name:
        output_dir = resolve_run_dir(
            output_root=output_root,
            run_name=args.run_name,
            run_label=args.run_label,
            prefer_latest_if_missing=False,
        ).run_dir
    else:
        output_dir = input_dir
        output_dir.mkdir(parents=True, exist_ok=True)

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    print(f"Input run:  {input_dir}")
    print(f"Output run: {output_dir}")

    # -----------------------------------------------------------------------
    # Load data
    # -----------------------------------------------------------------------
    df = pd.read_csv(input_dir / "h25_scatter.csv")
    df = df.sort_values("sdg").reset_index(drop=True)

    # Medians for 2×2 boundary
    median_research_pct = df["research_pct"].median()
    median_semantic_gap = df["semantic_gap"].median()

    # -----------------------------------------------------------------------
    # Figure 1 — Coverage profiles comparison
    # -----------------------------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(8.5, 6))

    df_sorted = df.sort_values("policy_pct_docweighted", ascending=False).reset_index(drop=True)
    y = np.arange(len(df_sorted))
    height = 0.38

    ax1.barh(
        y - height / 2,
        df_sorted["policy_pct_docweighted"],
        height=height,
        color=POLICY_COLOR,
        alpha=0.88,
        label="Policy (document-weighted %)",
    )
    ax1.barh(
        y + height / 2,
        df_sorted["research_pct"],
        height=height,
        color=RESEARCH_COLOR,
        alpha=0.88,
        label="Research (%)",
    )

    labels = [SDG_SHORT[int(row["sdg"])].replace("\n", " ") for _, row in df_sorted.iterrows()]
    ax1.set_yticks(y)
    ax1.set_yticklabels(labels, fontsize=7.5)
    ax1.set_xlabel("Proportion of corpus assigned to SDG (%)")
    ax1.set_title(
        "Figure 1. Coverage profiles: research vs policy by SDG\n"
        "†SDG 4 research share likely inflated by ML vocabulary artefact  "
        "‡SDG 13 & 17 centroids highly collinear (cos = 0.860); combined share = 71%",
        fontsize=8.5,
        loc="left",
    )
    ax1.legend(loc="lower right")
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

    df_sem = df.sort_values("semantic_gap", ascending=False).reset_index(drop=True)
    y = np.arange(len(df_sem))
    colors = [RESEARCH_COLOR if v > median_semantic_gap else "#92C5DE" for v in df_sem["semantic_gap"]]

    ax2.barh(y, df_sem["semantic_gap"], color=colors, alpha=0.88)
    ax2.axvline(median_semantic_gap, color="grey", linestyle="--", linewidth=1,
                label=f"Median ({median_semantic_gap:.3f})")
    ax2.set_yticks(y)
    ax2.set_yticklabels([SDG_SHORT[int(r["sdg"])].replace("\n", " ") for _, r in df_sem.iterrows()], fontsize=7.5)
    ax2.set_xlabel("Semantic gap (1 − cosine similarity between research and policy sub-centroids)")
    ax2.set_title(
        "Figure 2. Within-SDG semantic gap (chunk cap = 50)\n"
        "Higher gap = research and policy discuss this SDG more differently",
        fontsize=8.5,
        loc="left",
    )

    high_patch = mpatches.Patch(color=RESEARCH_COLOR, alpha=0.88, label="Above median gap")
    low_patch = mpatches.Patch(color="#92C5DE", alpha=0.88, label="Below median gap")
    ax2.legend(handles=[high_patch, low_patch, plt.Line2D([0], [0], color="grey", linestyle="--",
               label=f"Median ({median_semantic_gap:.3f})")], fontsize=7.5)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.invert_yaxis()

    fig2.tight_layout()
    fig2.savefig(figures_dir / "fig2_semantic_gap.pdf", bbox_inches="tight")
    fig2.savefig(figures_dir / "fig2_semantic_gap.png", bbox_inches="tight", dpi=150)
    plt.close(fig2)
    print("Saved: fig2_semantic_gap.pdf")

    # -----------------------------------------------------------------------
    # Figure 3 — Coverage vs semantic gap scatter (2×2 typology)
    # -----------------------------------------------------------------------
    fig3, ax3 = plt.subplots(figsize=(7.5, 6))

    ax3.axvline(median_research_pct, color="grey", linestyle="--", linewidth=1, alpha=0.7)
    ax3.axhline(median_semantic_gap, color="grey", linestyle="--", linewidth=1, alpha=0.7)

    xlim = (0, df["research_pct"].max() * 1.12)
    ylim = (0.16, df["semantic_gap"].max() * 1.08)

    ax3.fill_betweenx([median_semantic_gap, ylim[1]], xlim[0], median_research_pct,
                      color="#FDDBC7", alpha=0.35, zorder=0)
    ax3.fill_betweenx([median_semantic_gap, ylim[1]], median_research_pct, xlim[1],
                      color="#D1E5F0", alpha=0.35, zorder=0)
    ax3.fill_betweenx([ylim[0], median_semantic_gap], xlim[0], median_research_pct,
                      color="#F7F7F7", alpha=0.55, zorder=0)
    ax3.fill_betweenx([ylim[0], median_semantic_gap], median_research_pct, xlim[1],
                      color="#E0F3DB", alpha=0.35, zorder=0)

    ax3.text(0.01, 0.98, "Low coverage\nHigh semantic gap\n(double neglect)", transform=ax3.transAxes,
             fontsize=7, ha="left", va="top", color="#8B0000", style="italic")
    ax3.text(0.99, 0.98, "High coverage\nHigh semantic gap\n(problematic misalignment)", transform=ax3.transAxes,
             fontsize=7, ha="right", va="top", color="#1A237E", style="italic")
    ax3.text(0.01, 0.02, "Low coverage\nLow semantic gap\n(aligned but limited)", transform=ax3.transAxes,
             fontsize=7, ha="left", va="bottom", color="#555", style="italic")
    ax3.text(0.99, 0.02, "High coverage\nLow semantic gap\n(apparent alignment)", transform=ax3.transAxes,
             fontsize=7, ha="right", va="bottom", color="#2e7d32", style="italic")

    for _, row in df.iterrows():
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
        "Figure 3. Coverage vs semantic gap: 2×2 misalignment typology\n"
        "Dashed lines = median thresholds (research: 4.25%, semantic gap: 0.211)",
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

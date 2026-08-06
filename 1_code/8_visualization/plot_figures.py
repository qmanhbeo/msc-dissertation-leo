"""
plot_figures.py — Generate dissertation figures from analysis outputs.

Produces four publication-quality figures for the dissertation:

    Figure 1 — Centroid pairwise similarity heatmap (lower triangle, all-MiniLM-L6-v2)
    Figure 2 — Coverage profiles: research vs policy (horizontal grouped bar chart)
    Figure 3 — Semantic gap by SDG (horizontal bar chart, sorted descending)
    Figure 4 — Coverage vs semantic gap scatter (diagnostic map)

Inputs:
      4_outputs/mpnet/data/centroid_similarity_matrix.csv     — 17x17 pairwise centroid cosine similarity
      4_outputs/mpnet/data/interaction_scatter_data.csv       — per-SDG metrics table from coverage_semantic_interaction.py
      4_outputs/mpnet/data/coverage_document_weighted.json   — corpus-level n counts for legend labels

Outputs:
      4_outputs/appendix/mpnet/a4_centroid_similarity/figures/fig8_centroid_similarity_heatmap.pdf
      4_outputs/mpnet/figures/fig2_coverage_profiles.pdf
      4_outputs/mpnet/figures/fig4_semantic_gap.pdf
      4_outputs/mpnet/figures/fig5_coverage_semantic_scatter.pdf

Run:
    python 1_code/8_visualization/plot_figures.py
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib-dissertation"))

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for WSL
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = ROOT / "1_code"
SHARED_DIR = ROOT / "1_code" / "7_main_analysis" / "0_shared"
for path in (CODE_ROOT, SHARED_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
from model_utils import DEFAULT_EMBED_MODEL, DEFAULT_OUTPUT_ROOT, model_slug, resolve_model_alias
from shared_utils import ensure_canonical_outputs, require_output_files


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate figures from the canonical output folder.")
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_ROOT))
    p.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL, type=resolve_model_alias, help=argparse.SUPPRESS)
    p.add_argument("--overwrite", action="store_true",
                   help="Regenerate figures even if they already exist.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Shared style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "serif"],
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

RESEARCH_COLOR = "#0077BB"   # blue (Tol palette, colorblind-safe)
POLICY_COLOR   = "#EE7733"   # orange (Tol palette, colorblind-safe)

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


def plot_centroid_similarity_heatmap(layout, model: str) -> None:
    """Render an annotated 17x17 heatmap of pairwise canonical SDG-centroid cosine similarity."""
    csv_path = layout.data_dir / "centroid_similarity_matrix.csv"
    mat = pd.read_csv(csv_path, index_col=0).apply(pd.to_numeric, errors="coerce")
    M = mat.to_numpy(dtype=float)
    n = M.shape[0]
    ticks = [int(str(c).replace("SDG", "")) for c in mat.columns]

    # The matrix is symmetric; show only the lower triangle + diagonal.
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    M_masked = np.ma.masked_where(mask, M)

    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    vmin = float(np.nanmin(M))
    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad("white")
    im = ax.imshow(M_masked, cmap=cmap, vmin=vmin, vmax=1.0, aspect="equal")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(ticks, fontsize=8)
    ax.set_yticklabels(ticks, fontsize=8)
    ax.set_xlabel("SDG centroid")
    ax.set_ylabel("SDG centroid")
    for i in range(n):
        for j in range(n):
            if mask[i, j]:
                continue
            v = M[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=6.5, color="white" if v > 0.72 else "black")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Cosine similarity")
    fig.tight_layout()
    out_dir = layout.root.parent / "appendix" / model_slug(model) / "a4_centroid_similarity" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "fig8_centroid_similarity_heatmap.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "fig8_centroid_similarity_heatmap.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("Saved: fig8_centroid_similarity_heatmap.pdf")


def load_gap_maps(layout) -> tuple[dict[int, float], dict[int, float]]:
    """Load per-SDG semantic gaps. Adjusted = after INLP register removal (canonical);
    raw = the naive baseline. Falls back to empty maps if files are absent so the
    script never crashes (raw-only rendering is used when no adjusted data exists)."""
    adj_path = layout.data_dir / "adjusted" / "semantic_gap_distances_lr.json"
    raw_path = layout.data_dir / "semantic_gap_distances_lr.json"
    adj_map: dict[int, float] = {}
    raw_map: dict[int, float] = {}
    if adj_path.exists():
        with open(adj_path) as f:
            for r in json.load(f).get("per_sdg", []):
                adj_map[int(r["sdg"])] = float(r["semantic_gap"])
    if raw_path.exists():
        with open(raw_path) as f:
            for r in json.load(f).get("per_sdg", []):
                raw_map[int(r["sdg"])] = float(r["semantic_gap"])
    return adj_map, raw_map


def main() -> None:
    args = parse_args()
    layout = ensure_canonical_outputs(Path(args.output_dir), model=args.embed_model)
    require_output_files(layout.data_dir, ["interaction_scatter_data.csv", "coverage_document_weighted.json"])
    figures_dir = layout.figures_dir

    print(f"Canonical output dir: {layout.data_dir}")

    if not args.overwrite:
        expected = [
            figures_dir / "fig2_coverage_profiles.pdf",
            figures_dir / "fig4_semantic_gap.pdf",
            figures_dir / "fig5_coverage_semantic_scatter.pdf",
        ]
        if all(p.exists() for p in expected):
            print(f"Figures already exist at {figures_dir} — skip. Use --overwrite to regenerate.")
            return

    # -----------------------------------------------------------------------
    # Load data
    # -----------------------------------------------------------------------
    df = pd.read_csv(layout.data_dir / "interaction_scatter_data.csv")
    df = df.sort_values("sdg").reset_index(drop=True)
    df["semantic_gap"] = pd.to_numeric(df["semantic_gap"], errors="coerce")
    df["semantic_similarity"] = pd.to_numeric(df["semantic_similarity"], errors="coerce")
    df_sem_valid = df[df["semantic_gap"].notna()].copy()
    if df_sem_valid.empty:
        raise RuntimeError("No finite semantic-gap rows available for figure generation.")

    # Medians for the diagonal thresholds used in the diagnostic map
    median_research_pct = df["research_pct"].median()
    median_coverage_gap = df["coverage_gap_abs"].median()
    median_semantic_gap = df_sem_valid["semantic_gap"].median()
    mean_semantic_gap = df_sem_valid["semantic_gap"].mean()

    # Load corpus-level counts for legend labels
    with open(layout.data_dir / "coverage_document_weighted.json") as f:
        _cov_counts = json.load(f)
    N_RESEARCH_PAPERS = _cov_counts["n_research_papers"]
    N_POLICY_DOCS = _cov_counts["n_policy_documents"]

    # Adjusted (register-removed, canonical) + raw (baseline) semantic gaps.
    adj_map, raw_map = load_gap_maps(layout)
    use_adjusted = bool(adj_map)

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
            va="center", ha="left", fontsize=7,
        )
        ax1.text(
            df_sorted["research_pct"].iloc[i] + 0.3,
            i + height / 2,
            _fmt(research_counts.iloc[i]),
            va="center", ha="left", fontsize=7,
        )

    labels = [SDG_SHORT[int(row["sdg"])].replace("\n", " ") for _, row in df_sorted.iterrows()]
    ax1.set_yticks(y)
    ax1.set_yticklabels(labels, fontsize=8.5)
    ax1.set_xlabel("Proportion of corpus assigned to SDG (%)")
    # ax1.set_title("Coverage profiles by SDG", fontsize=8.5, loc="left")
    ax1.legend(loc="upper right")
    ax1.axvline(0, color="black", linewidth=0.5)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.invert_yaxis()

    fig1.tight_layout()
    fig1.savefig(figures_dir / "fig2_coverage_profiles.pdf", bbox_inches="tight")
    fig1.savefig(figures_dir / "fig2_coverage_profiles.png", bbox_inches="tight", dpi=300)
    plt.close(fig1)
    print("Saved: fig2_coverage_profiles.pdf")

    # -----------------------------------------------------------------------
    # Figure 2 — Semantic gap by SDG (adjusted canonical, raw baseline)
    # -----------------------------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(7, 5.5))

    if use_adjusted:
        fig2_df = pd.DataFrame({"sdg": list(adj_map.keys()), "gap": list(adj_map.values())})
        fig2_df["raw"] = fig2_df["sdg"].map(raw_map)
        fig2_df = fig2_df.sort_values("gap", ascending=False).reset_index(drop=True)
        y = np.arange(len(fig2_df))
        _med = float(np.median(fig2_df["gap"].to_numpy()))
        _mean = float(np.mean(fig2_df["gap"].to_numpy()))
        colors = [RESEARCH_COLOR if v > _med else "#BBBBBB" for v in fig2_df["gap"]]
        ax2.barh(y, fig2_df["gap"], color=colors, alpha=0.88, label="Adjusted gap (canonical)")
        for i, (_, row) in enumerate(fig2_df.iterrows()):
            if pd.notna(row["raw"]):
                ax2.plot(row["raw"], i, "D", color="#555555", markersize=5, alpha=0.85,
                         label="Semantic gap (baseline)" if i == 0 else None)
        for i, val in enumerate(fig2_df["gap"]):
            ax2.text(val + 0.008, i, f"{val:.3f}", va="center", ha="left", fontsize=7.5)
        ax2.axvline(_med, color="grey", linestyle="--", linewidth=1,
                    label=f"Median (adj, {_med:.3f})")
        ax2.axvline(_mean, color="black", linestyle=":", linewidth=1,
                    label=f"Mean (adj, {_mean:.3f})")
        ax2.set_yticks(y)
        ax2.set_yticklabels([SDG_SHORT[int(r["sdg"])].replace("\n", " ") for _, r in fig2_df.iterrows()], fontsize=8.5)
        ax2.set_xlabel("Semantic gap (1 − cosine similarity; adjusted = after INLP register removal)")
        ax2.legend(handles=[
            mpatches.Patch(color=RESEARCH_COLOR, alpha=0.88, label="Adjusted gap (canonical)"),
            plt.Line2D([0], [0], marker="D", color="#555555", linestyle="None", label="Semantic gap (baseline)"),
            plt.Line2D([0], [0], color="grey", linestyle="--", label=f"Median (adj, {median_semantic_gap:.3f})"),
            plt.Line2D([0], [0], color="black", linestyle=":", label=f"Mean (adj, {mean_semantic_gap:.3f})"),
        ], fontsize=7.5)
    else:
        df_sem = df_sem_valid.sort_values("semantic_gap", ascending=False).reset_index(drop=True)
        y = np.arange(len(df_sem))
        colors = [RESEARCH_COLOR if v > median_semantic_gap else "#BBBBBB" for v in df_sem["semantic_gap"]]
        ax2.barh(y, df_sem["semantic_gap"], color=colors, alpha=0.88)
        for i, val in enumerate(df_sem["semantic_gap"]):
            ax2.text(val + 0.008, i, f"{val:.3f}", va="center", ha="left", fontsize=7.5)
        ax2.axvline(median_semantic_gap, color="grey", linestyle="--", linewidth=1,
                    label=f"Median ({median_semantic_gap:.3f})")
        ax2.axvline(mean_semantic_gap, color="black", linestyle=":", linewidth=1,
                    label=f"Mean ({mean_semantic_gap:.3f})")
        ax2.set_yticks(y)
        ax2.set_yticklabels([SDG_SHORT[int(r["sdg"])].replace("\n", " ") for _, r in df_sem.iterrows()], fontsize=8.5)
        ax2.set_xlabel("Semantic gap (1 − cosine similarity between research and policy sub-centroids)")
        high_patch = mpatches.Patch(color=RESEARCH_COLOR, alpha=0.88, label="Above median gap")
        low_patch = mpatches.Patch(color="#BBBBBB", alpha=0.88, label="Below median gap")
        ax2.legend(handles=[
            high_patch, low_patch,
            plt.Line2D([0], [0], color="grey", linestyle="--", label=f"Median ({median_semantic_gap:.3f})"),
            plt.Line2D([0], [0], color="black", linestyle=":", label=f"Mean ({mean_semantic_gap:.3f})"),
        ], fontsize=7.5)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.invert_yaxis()

    fig2.tight_layout()
    fig2.savefig(figures_dir / "fig4_semantic_gap.pdf", bbox_inches="tight")
    fig2.savefig(figures_dir / "fig4_semantic_gap.png", bbox_inches="tight", dpi=300)
    plt.close(fig2)
    print("Saved: fig4_semantic_gap.pdf")

    # -----------------------------------------------------------------------
    # Figure 3 — Coverage vs semantic gap scatter (diagnostic map)
    # Adjusted (canonical) gap is the solid blue series; raw gap is the open
    # grey baseline so the reader can see both on the same coverage axis.
    # -----------------------------------------------------------------------
    fig3, ax3 = plt.subplots(figsize=(7.5, 6))

    if use_adjusted:
        _ymed = float(np.median(list(adj_map.values())))
        ax3.axvline(median_coverage_gap, color="grey", linestyle="--", linewidth=1, alpha=0.7)
        ax3.axhline(_ymed, color="grey", linestyle="--", linewidth=1, alpha=0.7)
        xlim = (0, df["coverage_gap_abs"].max() * 1.12)
        _adj_vals = list(adj_map.values())
        _raw_vals = [raw_map.get(s, np.nan) for s in adj_map]
        ylim = (max(0.0, min(_adj_vals) * 0.92), max(_adj_vals) * 1.08)
        for sdg in adj_map:
            x = df_sem_valid.loc[df_sem_valid["sdg"] == sdg, "coverage_gap_abs"]
            if x.empty:
                continue
            x = float(x.iloc[0])
            ax3.scatter(x, raw_map.get(sdg, np.nan), s=45, facecolors="none",
                        edgecolors="#888888", zorder=4, alpha=0.7)
            ax3.scatter(x, adj_map[sdg], s=55, color=RESEARCH_COLOR, zorder=5, alpha=0.9)
            offsets = {4: (0.02, 0.003), 9: (-0.03, 0.003), 3: (0.02, -0.004),
                       16: (0.02, 0.003), 8: (0.015, 0.003), 17: (0.02, -0.005),
                       13: (0.02, -0.004), 12: (0.02, 0.003)}
            dx, dy = offsets.get(sdg, (0.015, 0.002))
            ax3.annotate(f"SDG {sdg}", (x, adj_map[sdg]), xytext=(x + dx, adj_map[sdg] + dy),
                         fontsize=8, color="black",
                         arrowprops=dict(arrowstyle="-", color="grey", lw=0.5) if sdg in offsets else None)
        ax3.legend(handles=[
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=RESEARCH_COLOR,
                       markersize=7, label="Adjusted gap (canonical)"),
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="none",
                       markeredgecolor="#888888", markersize=7, label="Semantic gap (baseline)"),
        ], fontsize=8, loc="upper right")
    else:
        ax3.axvline(median_coverage_gap, color="grey", linestyle="--", linewidth=1, alpha=0.7)
        ax3.axhline(median_semantic_gap, color="grey", linestyle="--", linewidth=1, alpha=0.7)
        xlim = (0, df["coverage_gap_abs"].max() * 1.12)
        ylim = (max(0.0, df_sem_valid["semantic_gap"].min() * 0.92), df_sem_valid["semantic_gap"].max() * 1.08)
        for _, row in df_sem_valid.iterrows():
            sdg = int(row["sdg"])
            x = row["coverage_gap_abs"]
            y = row["semantic_gap"]
            ax3.scatter(x, y, s=55, color=RESEARCH_COLOR, zorder=5, alpha=0.85)
            offsets = {4: (0.02, 0.003), 9: (-0.03, 0.003), 3: (0.02, -0.004),
                       16: (0.02, 0.003), 8: (0.015, 0.003), 17: (0.02, -0.005),
                       13: (0.02, -0.004), 12: (0.02, 0.003)}
            dx, dy = offsets.get(sdg, (0.015, 0.002))
            ax3.annotate(f"SDG {sdg}", (x, y), xytext=(x + dx, y + dy),
                         fontsize=8, color="black",
                         arrowprops=dict(arrowstyle="-", color="grey", lw=0.5) if sdg in offsets else None)

    ax3.set_xlim(xlim)
    ax3.set_ylim(ylim)
    ax3.set_xlabel("Absolute research–policy coverage gap (H1a predictor)")
    ax3.set_ylabel("Within-SDG semantic gap (adjusted = after INLP register removal)")
    ax3.set_title(
        "",
        fontsize=8.5,
        loc="left",
    )
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    ax3.grid(True, alpha=0.3, linestyle=":", linewidth=0.5)

    fig3.tight_layout()
    fig3.savefig(figures_dir / "fig5_coverage_semantic_scatter.pdf", bbox_inches="tight")
    fig3.savefig(figures_dir / "fig5_coverage_semantic_scatter.png", bbox_inches="tight", dpi=300)
    plt.close(fig3)
    print("Saved: fig5_coverage_semantic_scatter.pdf")

    # -----------------------------------------------------------------------
    # Centroid pairwise similarity heatmap
    # -----------------------------------------------------------------------
    plot_centroid_similarity_heatmap(layout, args.embed_model)

    print(f"\\nAll figures saved to {figures_dir}")


if __name__ == "__main__":
    main()

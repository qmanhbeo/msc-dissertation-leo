"""
plot_figures.py — Generate dissertation figures from analysis outputs.

Produces the figure set (PDF + PNG) from main-text analysis outputs:

    fig2 — Coverage profiles: research vs policy (horizontal grouped bar chart)
    fig4 — Semantic gap by SDG (horizontal bar chart, sorted descending)
    fig9 — H1a–H1d coverage-predictor vs semantic-gap scatter grid (one
           combined 2x2 image with embedded panel titles and a shared legend)
    fig8 — Centroid pairwise similarity heatmap (lower triangle), written under
           4_outputs/appendix/{model}/a4_centroid_similarity/figures/

Inputs:
      4_outputs/{model}/data/interaction_scatter_data.csv    — per-SDG metrics table
      4_outputs/{model}/data/coverage_document_weighted.json — corpus-level n counts
      4_outputs/{model}/data/centroid_similarity_matrix.csv  — pairwise cosine similarities

Outputs:
      4_outputs/mpnet/figures/fig2_coverage_profiles.{pdf,png}
      4_outputs/mpnet/figures/fig4_semantic_gap.{pdf,png}
      4_outputs/mpnet/figures/fig9_h1_grid.{pdf,png}
      4_outputs/appendix/mpnet/a4_centroid_similarity/figures/fig8_centroid_similarity_heatmap.{pdf,png}

Run:
    python 1_code/8_visualization/plot_figures.py --output-dir 4_outputs/mpnet
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
from model_utils import DEFAULT_EMBED_MODEL, DEFAULT_OUTPUT_ROOT, SDG_SHORT_NAMES, model_slug, resolve_model_alias
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
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
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

SDG_SHORT = {sdg: f"SDG {sdg}\n{SDG_SHORT_NAMES[sdg]}" for sdg in range(1, 18)}


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
            ax.text(j, i, f"{v:.3f}", ha="center", va="center",
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


def plot_h1_scatter_grid(layout, model: str) -> None:
    """H1a--H1d coverage-predictor vs semantic-gap scatter grid (single image).

    One combined PNG/PDF: a 2x2 grid of panels, each plotting a different
    coverage predictor on x against the within-SDG semantic gap on y. The
    y-axis is the same semantic-gap variable for every panel (solid blue =
    adjusted / canonical, open grey = raw baseline), so the panels share a
    common y-limit. Panel titles "(a)"-"(d)" are embedded in the image (the
    manuscript text references "panel a" directly), and a single shared
    legend replaces the former per-panel legends.

    Predictors (from interaction_scatter_data.csv):
        H1a -> coverage_gap_abs        (absolute research--policy coverage gap, pp)
        H1b -> research_dominance      (signed dominance = research% - policy%, pp)
        H1c -> research_pct            (research coverage %)
        H1d -> policy_pct_docweighted  (policy coverage %)
    """
    figures_dir = layout.figures_dir
    df = pd.read_csv(layout.data_dir / "interaction_scatter_data.csv")
    df = df.sort_values("sdg").reset_index(drop=True)
    df["semantic_gap"] = pd.to_numeric(df["semantic_gap"], errors="coerce")
    df_sem_valid = df[df["semantic_gap"].notna()].copy()
    if df_sem_valid.empty:
        raise RuntimeError("No finite semantic-gap rows available for figure generation.")

    adj_map, raw_map = load_gap_maps(layout)
    use_adjusted = bool(adj_map)

    # Shared y-limits across all four panels (identical semantic-gap quantity).
    if use_adjusted:
        _adj_vals = list(adj_map.values())
        ymin = max(0.0, min(_adj_vals) * 0.92)
        ymax = max(_adj_vals) * 1.08
        ymed = float(np.median(_adj_vals))
    else:
        ymin = max(0.0, df_sem_valid["semantic_gap"].min() * 0.92)
        ymax = df_sem_valid["semantic_gap"].max() * 1.08
        ymed = float(df_sem_valid["semantic_gap"].median())

    # (embedded panel title, dataframe column, x-axis label, x-scale factor)
    panels = [
        ("(a) H1a", "coverage_gap_abs",
         "Absolute research–policy coverage gap (%)", 100),
        ("(b) H1b", "research_dominance",
         "Signed dominance (research% − policy%, %)", 100),
        ("(c) H1c", "research_pct",
         "Research coverage (%)", 1),
        ("(d) H1d", "policy_pct_docweighted",
         "Policy coverage (%)", 1),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(6.6, 7.0))
    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=RESEARCH_COLOR,
                   markersize=6, label="Adjusted Semantic Gap"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="none",
                   markeredgecolor="#888888", markersize=6, label="Raw Semantic Gap"),
    ] if use_adjusted else None

    for (title, col, xlabel, xscale), ax in zip(panels, axes.ravel()):
        ax.axhline(ymed, color="grey", linestyle="--", linewidth=1, alpha=0.7)

        signed = col == "research_dominance"
        xvals = df[col] * xscale
        xmin = float(xvals.min()) if signed else 0.0
        xmax = float(xvals.max()) * 1.12
        if signed:
            xmin = float(xvals.min()) * 1.12

        if use_adjusted:
            for sdg in adj_map:
                x = df_sem_valid.loc[df_sem_valid["sdg"] == sdg, col]
                if x.empty:
                    continue
                x = float(x.iloc[0]) * xscale
                ax.scatter(x, raw_map.get(sdg, np.nan), s=22, facecolors="none",
                           edgecolors="#888888", zorder=4, alpha=0.7)
                ax.scatter(x, adj_map[sdg], s=28, color=RESEARCH_COLOR, zorder=5, alpha=0.9)
                ax.annotate(f"{sdg}", (x, raw_map.get(sdg, np.nan)),
                            fontsize=4, color="black", fontweight="bold",
                            ha="center", va="center", zorder=6)
                ax.annotate(f"{sdg}", (x, adj_map[sdg]),
                            fontsize=4, color="white", fontweight="bold",
                            ha="center", va="center", zorder=6)
        else:
            for _, row in df_sem_valid.iterrows():
                sdg = int(row["sdg"])
                x = row[col] * xscale
                y = row["semantic_gap"]
                ax.scatter(x, y, s=28, color=RESEARCH_COLOR, zorder=5, alpha=0.85)
                ax.annotate(f"{sdg}", (x, y),
                            fontsize=4, color="white", fontweight="bold",
                            ha="center", va="center", zorder=6)

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax * 1.05)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, alpha=0.3, linestyle=":", linewidth=0.5)
        ax.tick_params(labelsize=7.5)

    # Shared y-label on the left column only; one shared legend above the grid.
    for ax in axes[:, 0]:
        ax.set_ylabel("Within-SDG semantic gap", fontsize=8)
    if legend_handles is not None:
        fig.legend(handles=legend_handles, loc="upper center", ncol=2,
                   frameon=False, fontsize=8, columnspacing=1.6, handletextpad=0.4)
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    out_base = figures_dir / "fig9_h1_grid"
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("Saved: fig9_h1_grid.png")


def main() -> None:
    args = parse_args()
    layout = ensure_canonical_outputs(Path(args.output_dir), model=args.embed_model)
    require_output_files(layout.data_dir, ["interaction_scatter_data.csv", "coverage_document_weighted.json"])
    figures_dir = layout.figures_dir

    print(f"Canonical output dir: {layout.data_dir}")

    if not args.overwrite:
        # Existence-skip covers fig2/fig4/fig9a-d ONLY. fig8 (the appendix
        # centroid heatmap, rendered further below) is checked nowhere, so a
        # deleted fig8 is NOT regenerated by a plain re-run — it resurfaces
        # later as a --build-pdf missing-input error. Keep this list in sync
        # with the figures rendered below AND shared_utils.MANUSCRIPT_FIGURE_FILES.
        expected = [
            figures_dir / "fig2_coverage_profiles.pdf",
            figures_dir / "fig4_semantic_gap.pdf",
            figures_dir / "fig9_h1_grid.pdf",
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
        label=f"Research (abstract-weighted %, n = {N_RESEARCH_PAPERS})",
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
            plt.Line2D([0], [0], color="grey", linestyle="--", label=f"Median (adj, {_med:.3f})"),
            plt.Line2D([0], [0], color="black", linestyle=":", label=f"Mean (adj, {_mean:.3f})"),
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
    # H1a--H1d coverage-predictor vs semantic-gap scatter grid (2x2, four
    # separate PNG/PDF files; grouped as a subfigure grid in the manuscript).
    # Replaces the standalone fig5 single-panel scatter.
    # -----------------------------------------------------------------------
    plot_h1_scatter_grid(layout, args.embed_model)

    # -----------------------------------------------------------------------
    # Centroid pairwise similarity heatmap
    # -----------------------------------------------------------------------
    plot_centroid_similarity_heatmap(layout, args.embed_model)

    print(f"\\nAll figures saved to {figures_dir}")


if __name__ == "__main__":
    main()

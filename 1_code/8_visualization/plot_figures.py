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

Appendix table-to-figure renders (MPNet canonical track only; each renders the
same frozen pipeline artifacts as the appendix table it replaces in the
manuscript):
      4_outputs/mpnet/figures/fig10_concept_coverage.{pdf,png}
      4_outputs/mpnet/figures/fig12_register_convergence.{pdf,png}
      4_outputs/appendix/mpnet/c_sample_stability/figures/fig11_sample_stability.{pdf,png}
      4_outputs/appendix/mpnet/h1_cross_method_gap_values/figures/fig13_cross_method_heatmap.{pdf,png}

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
    "axes.labelsize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

RESEARCH_COLOR = "#0077BB"   # blue (Tol palette, colorblind-safe)
POLICY_COLOR   = "#EE7733"   # orange (Tol palette, colorblind-safe)

# ---------------------------------------------------------------------------
# Semantic color roles (ratified 2026-08-28) — follow these for every new
# figure; do not introduce new hues without updating this block.
#
#   1. One primary contrast per figure: the two things a figure compares get
#      the blue/orange pair; everything else in the figure is grey.
#   2. POLICY_COLOR is RESERVED for the policy corpus. Never use orange for
#      any other series (retrieval strategies, encoders, metrics, ...).
#   3. RESEARCH_COLOR (blue) = the canonical object: the research corpus in
#      corpus figures, the adjusted (post-INLP) state in method figures.
#   4. BASELINE_GREY = baselines and deviations: raw (pre-INLP) series, the
#      reference pool, matched samples. Open/hollow markers reinforce it.
#   5. Derive variants within a family instead of adding hues: the
#      non-canonical member of a pair is a hollow/dashed/lighter variant of
#      the family colour (e.g. concept retrieval = hollow blue, keyword =
#      filled blue).
#
#   Axis cheat-sheet:
#     corpus      research=blue solid | policy=orange solid | reference=grey
#     state       raw=grey (open)     | adjusted=blue
#     encoder     MPNet=blue | MiniLM=purple | SciBERT=teal   (TRACK_COLORS)
#     retrieval   keyword=filled blue | concept=hollow blue
#     classifier  NEVER coloured: linestyle/marker only (LR solid, MLP
#                 dashed, ZS dotted) or column labels in heatmaps
#
#   Grayscale note: blue/orange/teal differ in luminance, and hollow vs
#   filled + linestyle survive B/W printing — check any new figure there.
# ---------------------------------------------------------------------------
BASELINE_GREY = "#555555"   # raw / baseline series (raw gap, reference pool)
MINILM_COLOR  = "#AA3377"   # Tol purple — encoder family; keeps orange policy-only
SCIBERT_COLOR = "#009988"   # Tol teal — encoder family

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
        plt.Line2D([0], [0], marker="o", linestyle="None", color=RESEARCH_COLOR,
                   markerfacecolor=RESEARCH_COLOR, markeredgecolor=RESEARCH_COLOR,
                   markeredgewidth=0, markersize=5, label="Adjusted Gap"),
        plt.Line2D([0], [0], marker="o", linestyle="None", color=BASELINE_GREY,
                   markerfacecolor=BASELINE_GREY, markeredgecolor=BASELINE_GREY,
                   markeredgewidth=0, markersize=5, label="Raw Gap"),
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
                ax.scatter(x, raw_map.get(sdg, np.nan), s=22, color=BASELINE_GREY,
                           edgecolors=BASELINE_GREY, zorder=4, alpha=0.9)
                ax.scatter(x, adj_map[sdg], s=28, color=RESEARCH_COLOR, zorder=5, alpha=0.9)
                ax.annotate(f"{sdg}", (x, raw_map.get(sdg, np.nan)),
                            fontsize=4, color="white", fontweight="bold",
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
        fig.legend(handles=legend_handles, loc="lower center", ncol=2,
                   frameon=False, fontsize=8, columnspacing=1.6, handletextpad=0.4)
    fig.tight_layout(rect=(0, 0.06, 1, 1))

    out_base = figures_dir / "fig9_h1_grid"
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("Saved: fig9_h1_grid.png")


# ---------------------------------------------------------------------------
# Appendix table-to-figure renders (MPNet canonical track only)
#
# Each figure below replaces an appendix table in dissertation.tex and is
# rendered from the SAME frozen pipeline artifacts the published table was
# generated from. Where the published table itself is the only persisted form
# of the values (tab12 convergence paths, the cross-method value grids), the
# figure parses that generated .tex strictly and fails closed on any shape
# mismatch — the figure can therefore never silently diverge from the table
# it replaced. House style (Tol palette, PDF+PNG 300 dpi) matches the rest of
# this module.
# ---------------------------------------------------------------------------

# Encoder family (palette block): MPNet stays blue (canonical encoder);
# MiniLM/SciBERT get their own hues so POLICY_COLOR stays policy-only.
TRACK_COLORS = {"MPNet": RESEARCH_COLOR, "MiniLM": MINILM_COLOR, "SciBERT": SCIBERT_COLOR}
_EM_DASH_CELLS = {"\u2014", "--", "---"}


def _read_tex_table_rows(path: Path) -> list[list[str]]:
    """Return '&'-split cell rows of the first tabular block of a generated table.

    Strips comments, booktabs rules and \\multicolumn scaffolding rows (header
    groups / summary lines); rows whose first cell is not an integer (headers
    like "It." / "SDG") are dropped by the callers via int(cells[0]).
    """
    rows = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("%"):
            continue
        if line.startswith(("\\begin{tabular", "\\end{tabular", "\\toprule",
                            "\\midrule", "\\bottomrule", "\\cmidrule")):
            continue
        if "\\multicolumn" in line:
            continue
        if "&" not in line:
            continue
        cells = [c.strip() for c in line.removesuffix("\\\\").split("&")]
        rows.append(cells)
    return rows


def _parse_concept_coverage_tex(path: Path) -> dict[int, tuple[float, float, float]]:
    """Parse tab10_concept_coverage.tex -> {sdg: (keyword %, concept %, delta pp)}."""
    out: dict[int, tuple[float, float, float]] = {}
    for cells in _read_tex_table_rows(path):
        try:
            sdg = int(cells[0])
        except ValueError:
            continue
        if len(cells) != 4:
            raise RuntimeError(f"tab10 parse: row {cells!r} does not have 4 cells")
        kw, con, delta = (float(c) for c in cells[1:4])
        out[sdg] = (kw, con, delta)
    if sorted(out) != list(range(1, 18)):
        raise RuntimeError(f"tab10 parse: expected 17 SDG rows, got {sorted(out)}")
    return out


def _parse_tab12_register_cross(path: Path) -> dict[str, list[tuple[int, float | None, float | None, float | None]]]:
    """Parse tab12_register_cross.tex -> {track: [(k, acc, gap, rho_vs_raw), ...]}.

    Em-dash cells (iterations not shown for a track) become None.
    """
    tracks = ["MPNet", "MiniLM", "SciBERT"]
    out: dict[str, list[tuple[int, float | None, float | None, float | None]]] = {t: [] for t in tracks}
    for cells in _read_tex_table_rows(path):
        try:
            k = int(cells[0])
        except ValueError:
            continue
        if len(cells) != 10:
            raise RuntimeError(f"tab12 parse: row k={k} has {len(cells)} cells, expected 10")
        vals = [None if c in _EM_DASH_CELLS else float(c) for c in cells[1:]]
        for t, group in zip(tracks, (vals[0:3], vals[3:6], vals[6:9])):
            out[t].append((k, group[0], group[1], group[2]))
    for t in tracks:
        if len(out[t]) < 8:
            raise RuntimeError(f"tab12 parse: track {t} has only {len(out[t])} rows")
        ks = [r[0] for r in out[t]]
        if not all(b > a for a, b in zip(ks, ks[1:])):
            raise RuntimeError(f"tab12 parse: non-monotonic iteration k for {t}")
    return out


def _parse_cross_method_values_tex(path: Path) -> dict[int, list[float]]:
    """Parse a tab_app_cross_method_* table -> {sdg: [9 method values]}."""
    out: dict[int, list[float]] = {}
    for cells in _read_tex_table_rows(path):
        try:
            sdg = int(cells[0])
        except ValueError:
            continue
        if len(cells) != 10:
            raise RuntimeError(f"cross-method parse: row {cells!r} does not have 10 cells")
        out[sdg] = [float(c) for c in cells[1:]]
    if sorted(out) != list(range(1, 18)):
        raise RuntimeError(f"cross-method parse: expected 17 SDG rows, got {sorted(out)}")
    return out


def _appendix_fig_dir(layout, model: str, module: str) -> Path:
    """figures/ under 4_outputs/appendix/{model}/{module}/ (fig8 convention)."""
    out_dir = layout.root.parent / "appendix" / model_slug(model) / module / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _save_fig(fig, out_dir: Path, name: str) -> None:
    fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved: {name}.pdf")


def plot_concept_coverage_figure(layout) -> None:
    """fig10 — keyword vs concept retrieval coverage shares per SDG (dumbbell).

    Replaces tab10_concept_coverage.tex. Shares come from the two frozen
    coverage_document_weighted.json profiles; the generated tex table is
    cross-checked (fail-closed) so the figure cannot diverge from the table
    it replaced.
    """
    kw_json = layout.data_dir / "coverage_document_weighted.json"
    con_json = layout.data_dir / "concept" / "coverage_document_weighted.json"
    tex = layout.tables_dir / "tab10_concept_coverage.tex"
    missing = [str(p) for p in (kw_json, con_json, tex) if not p.exists()]
    if missing:
        print(f"Skip fig10 (missing inputs: {', '.join(missing)})")
        return

    shares: dict[str, dict[int, float]] = {}
    for label, path in (("keyword", kw_json), ("concept", con_json)):
        with open(path) as f:
            prof = json.load(f)["research_profile_hard"]
        shares[label] = {int(k[3:]): 100.0 * v for k, v in prof.items()}
        if sorted(shares[label]) != list(range(1, 18)):
            raise RuntimeError(f"fig10: {label} profile does not cover SDGs 1-17")

    tex_rows = _parse_concept_coverage_tex(tex)
    for sdg in range(1, 18):
        kw_tex, con_tex, delta_tex = tex_rows[sdg]
        kw, con = shares["keyword"][sdg], shares["concept"][sdg]
        # tex values are rounded to 1 dp -> |unrounded - printed| <= 0.05.
        if abs(kw - kw_tex) > 0.0501 or abs(con - con_tex) > 0.0501:
            raise RuntimeError(
                f"fig10: SDG{sdg} shares (kw={kw:.3f}, con={con:.3f}) diverge "
                f"from published table ({kw_tex}, {con_tex})"
            )
        # Delta may have been rounded from unrounded shares (|diff - round| <= 0.05)
        # or from the printed 1-dp shares (|diff - (round-con - round-kw)| <= 0.1).
        if abs((con - kw) - delta_tex) > 0.101:
            raise RuntimeError(
                f"fig10: SDG{sdg} delta ({con - kw:+.3f}) diverges from "
                f"published table ({delta_tex:+.1f})"
            )

    sdgs = list(range(1, 18))
    fig, ax = plt.subplots(figsize=(7.5, 6.8))
    y = np.arange(len(sdgs))
    for i, sdg in enumerate(sdgs):
        kw, con = shares["keyword"][sdg], shares["concept"][sdg]
        ax.plot([kw, con], [i, i], color="#BBBBBB", lw=1.4, zorder=1)
        ax.plot(kw, i, "o", color=RESEARCH_COLOR, ms=5.5, zorder=2)
        # Concept = non-canonical retrieval -> hollow blue variant (palette
        # rule 5); orange stays reserved for the policy corpus (rule 2).
        ax.plot(con, i, "o", markerfacecolor="white", markeredgecolor=RESEARCH_COLOR,
                markeredgewidth=1.4, ms=5.5, zorder=2)
        # Share labels: keyword above its dot, concept below its dot; gap
        # right of the righter-most dot — all in the axis's % unit.
        ax.text(kw, i - 0.30, f"{kw:.1f}", ha="center", va="center", fontsize=6.5,
                color="#444444")
        ax.text(con, i + 0.30, f"{con:.1f}", ha="center", va="center", fontsize=6.5,
                color="#444444")
        d = con - kw
        ax.text(max(kw, con) + 0.6, i, f"{d:+.1f}%", ha="left", va="center",
                fontsize=7, fontweight="bold" if sdg in (4, 9) else "normal",
                color="#444444")
    for sdg in (4, 9):  # the two deviations the manuscript prose discusses
        ax.axhspan(sdg - 1 - 0.42, sdg - 1 + 0.42, color=RESEARCH_COLOR, alpha=0.07, lw=0)
    ax.set_yticks(y)
    ax.set_yticklabels([SDG_SHORT[s].replace("\n", " ") for s in sdgs], fontsize=8)
    ax.axvline(0, color="black", lw=0.5)
    ax.set_xlim(0, 31)
    # y already inverted by the (top, bottom) ordering of set_ylim below.
    ax.set_ylim(len(sdgs) - 0.5, -1.4)
    ax.set_xlabel("Share of research corpus assigned to SDG (%)")
    ax.legend(handles=[
        plt.Line2D([0], [0], marker="o", ls="None", color=RESEARCH_COLOR, ms=5.5,
                   label="Keyword-retrieved (canonical)"),
        plt.Line2D([0], [0], marker="o", ls="None", markerfacecolor="white",
                   markeredgecolor=RESEARCH_COLOR, markeredgewidth=1.4, ms=5.5,
                   label="Concept-retrieved (AI/ML field-of-study)"),
    ], loc="lower right", fontsize=8, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    _save_fig(fig, layout.figures_dir, "fig10_concept_coverage")


def plot_sample_stability_figure(layout, model: str) -> None:
    """fig11 — sample-stability ladder (replaces tab_c_sample_stability.tex).

    Three stacked panels (coverage gap; semantic gap raw; semantic gap
    adjusted) against research-corpus size on a log axis, mean line with a
    +/-1 SD draw-to-draw band, and the deterministic full-corpus value as a
    star. Data: c_sample_stability_table.csv (mean and SD over 100 draws per
    tier; full-corpus row deterministic).
    """
    module_dir = layout.root.parent / "appendix" / model_slug(model) / "c_sample_stability"
    csv_path = module_dir / "data" / "c_sample_stability_table.csv"
    if not csv_path.exists():
        print(f"Skip fig11 (missing input: {csv_path})")
        return

    df = pd.read_csv(csv_path)
    needed = {"sample_size", "deterministic", "coverage_gap", "std_coverage_gap",
              "mean_semantic_gap", "std_semantic_gap",
              "mean_semantic_gap_adjusted", "std_semantic_gap_adjusted"}
    if not needed.issubset(df.columns) or len(df) != 12:
        raise RuntimeError(f"fig11: unexpected c_sample_stability_table.csv shape ({len(df)} rows)")
    if int(df.loc[~df["deterministic"].astype(bool), "n_draws"].nunique()) != 1:
        raise RuntimeError("fig11: sampled tiers do not share a single draw count")

    # Palette: raw = baseline grey, adjusted = canonical blue (state axis);
    # coverage gap is not a pipeline state, so it takes the spare encoder-
    # family hue (teal) rather than policy orange.
    specs = [
        ("coverage_gap", "std_coverage_gap", "Coverage gap (pp)", SCIBERT_COLOR),
        ("mean_semantic_gap", "std_semantic_gap", "Semantic gap (raw)", BASELINE_GREY),
        ("mean_semantic_gap_adjusted", "std_semantic_gap_adjusted", "Semantic gap (adjusted)", RESEARCH_COLOR),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(7.5, 7.6), sharex=True)
    x = df["sample_size"].to_numpy(float)
    det = df[df["deterministic"].astype(bool)]
    for ax, (m, s, lbl, color) in zip(axes, specs):
        mean = df[m].to_numpy(float)
        sd = df[s].to_numpy(float)
        ax.plot(x, mean, "-o", color=color, ms=3.5, lw=1.4, zorder=2)
        ax.fill_between(x, mean - sd, mean + sd, color=color, alpha=0.18, lw=0,
                        label="\u00b11 SD (draw-to-draw)", zorder=1)
        ax.plot(det["sample_size"], det[m], "*", color="black", ms=11, zorder=3,
                label="Full corpus (deterministic)")
        ax.set_ylabel(lbl, fontsize=9)
        ax.set_xscale("log")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, alpha=0.3, linestyle=":", linewidth=0.5, which="both")
    axes[0].legend(loc="upper right", fontsize=7.5, frameon=False)
    axes[2].axvline(2e5, color="grey", ls="--", lw=0.9)
    axes[2].text(2e5 * 1.15, 0.02, "stabilised by \u2248200k", fontsize=7.5,
                 color="grey", ha="left", va="bottom", rotation=90,
                 transform=axes[2].get_xaxis_transform())  # x=data, y=axes fraction
    axes[2].set_xlabel("Research-corpus size (papers, log scale)")
    axes[2].set_xticks([1e3, 1e4, 1e5, 1e6])
    axes[2].set_xticklabels(["1k", "10k", "100k", "1m"])
    axes[2].set_xlim(8e2, 5e6)
    fig.tight_layout()
    _save_fig(fig, _appendix_fig_dir(layout, model, "c_sample_stability"), "fig11_sample_stability")


def plot_register_convergence_figure(layout) -> None:
    """fig12 — INLP register-removal convergence (replaces tab12_register_cross.tex).

    Three stacked panels (held-out register-classification accuracy; mean
    within-SDG semantic gap; Spearman rho of the per-SDG gap vector vs. the
    raw gap) against the number of removed directions k, one line per encoder
    track. Data: the generated tab12_register_cross.tex (values at the shown
    iterations only, parsed strictly).
    """
    tex = layout.tables_dir / "tab12_register_cross.tex"
    if not tex.exists():
        print(f"Skip fig12 (missing input: {tex})")
        return
    data = _parse_tab12_register_cross(tex)

    panels = [
        (1, "Held-out register-classification accuracy", (0.44, 1.02)),
        (2, "Mean within-SDG semantic gap", None),
        (3, "Spearman $\\rho$ of gap vector vs. raw", (-0.02, 1.06)),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(7.5, 7.8), sharex=True)
    for ax, (idx, lbl, ylim) in zip(axes, panels):
        for track, color in TRACK_COLORS.items():
            xs, ys = [], []
            for k, acc, gap, rho in data[track]:
                v = (acc, gap, rho)[idx - 1]
                if v is not None:
                    xs.append(k)
                    ys.append(v)
            ax.plot(xs, ys, "-o", color=color, ms=3.5, lw=1.4, label=track)
        if idx == 1:
            ax.axhline(0.5, color="grey", ls="--", lw=0.9)
            ax.text(79 * 0.92, 0.507, "0.5 (majority-class baseline)", fontsize=7,
                    color="grey", ha="right", va="bottom")
        if ylim:
            ax.set_ylim(*ylim)
        ax.set_ylabel(lbl, fontsize=9)
        ax.set_xscale("log")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, alpha=0.3, linestyle=":", linewidth=0.5, which="both")
    axes[2].set_xlabel("INLP directions removed ($k$, log scale)")
    axes[2].set_xticks([1, 2, 5, 10, 20, 50, 79])
    axes[2].set_xticklabels(["1", "2", "5", "10", "20", "50", "79"])
    axes[2].set_xlim(0.9, 95)
    axes[0].legend(loc="upper right", fontsize=8, frameon=False, ncol=3)
    fig.tight_layout()
    _save_fig(fig, layout.figures_dir, "fig12_register_convergence")


def plot_cross_method_heatmap_figure(layout, model: str) -> None:
    """fig13 — cross-method coverage/semantic gap values (replaces BOTH
    tab_app_cross_method_covgap.tex and tab_app_cross_method_semgap.tex).

    Two annotated heatmaps (17 SDGs x 9 method configurations): coverage gap
    in pp on top, semantic gap in cosine units below. Values are parsed
    strictly from the two generated tables the figure replaces.
    """
    tables_dir = layout.root.parent / "appendix" / model_slug(model) / "h1_cross_method_gap_values" / "tables"
    cov_tex = tables_dir / "tab_app_cross_method_covgap.tex"
    sem_tex = tables_dir / "tab_app_cross_method_semgap.tex"
    missing = [str(p) for p in (cov_tex, sem_tex) if not p.exists()]
    if missing:
        print(f"Skip fig13 (missing inputs: {', '.join(missing)})")
        return

    cov = _parse_cross_method_values_tex(cov_tex)
    sem = _parse_cross_method_values_tex(sem_tex)
    A_cov = np.array([[cov[s][j] for j in range(9)] for s in range(1, 18)])
    A_sem = np.array([[sem[s][j] for j in range(9)] for s in range(1, 18)])

    method_cols = ["LR", "MLP", "ZS", "LR", "MLP", "LR", "MLP", "LR", "MLP"]
    groups = [("MPNet", 0, 2), ("MiniLM", 3, 4), ("SciBERT", 5, 6), ("Concept", 7, 8)]
    fig, axes = plt.subplots(2, 1, figsize=(8.8, 8.4))
    for ax, (A, title, fmt) in zip(axes, [
        (A_cov, "(a) Coverage gap (%)", "{:.1f}"),
        (A_sem, "(b) Semantic gap (cosine)", "{:.3f}"),
    ]):
        vmax = float(A.max())
        im = ax.imshow(A, cmap="Blues", vmin=0, vmax=vmax, aspect="auto")
        ax.set_yticks(range(17))
        ax.set_yticklabels([f"SDG {s}" for s in range(1, 18)], fontsize=7)
        ax.set_xticks(range(9))
        ax.set_xticklabels(method_cols, fontsize=7.5)
        # Panel title inside the axes (above the SDG 1 row) so it cannot
        # collide with the encoder group labels above the top panel.
        ax.text(-0.45, -0.72, title, fontsize=9.5, fontweight="bold", ha="left", va="bottom")
        for i in range(17):
            for j in range(9):
                v = A[i, j]
                ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=5.4,
                        color="white" if v > 0.62 * vmax else "black")
        for xb in (2.5, 4.5, 6.5):
            ax.axvline(xb, color="white", lw=1.6)
        cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.015)
        cbar.ax.tick_params(labelsize=7)
    axes[1].set_xlabel("Embedding--classifier configuration")
    for name, j0, j1 in groups:
        axes[0].annotate(name, xy=((j0 + j1) / 2, 1.09),
                         xycoords=axes[0].get_xaxis_transform(),
                         ha="center", fontsize=8.5, fontweight="bold")
    fig.tight_layout()
    _save_fig(fig, _appendix_fig_dir(layout, model, "h1_cross_method_gap_values"), "fig13_cross_method_heatmap")


def plot_coverage_dumbbell(df: pd.DataFrame, figures_dir: Path,
                           n_research: int, n_policy: int) -> None:
    """fig2 — research vs policy coverage shares per SDG (dumbbell).

    Redesign of the former grouped bar chart:     one row per SDG with research
    and policy dots on a common % axis. The research share is labelled above
    its dot (black), the policy share below its dot (black), and the coverage
    gap as a signed dominance value (research − policy, %, bold) right of the
    righter-most dot of the row. Palette: this is the one pair that earns the
    full blue/orange contrast (research vs policy).
    """
    sdgs = list(range(1, 18))
    res = {int(r["sdg"]): float(r["research_pct"]) for _, r in df.iterrows()}
    pol = {int(r["sdg"]): float(r["policy_pct_docweighted"]) for _, r in df.iterrows()}
    fig, ax = plt.subplots(figsize=(8.0, 6.8))
    y = np.arange(len(sdgs))
    for i, sdg in enumerate(sdgs):
        xr, xp = res[sdg], pol[sdg]
        ax.plot([xr, xp], [i, i], color="#BBBBBB", lw=1.4, zorder=1)
        ax.plot(xr, i, "o", color=RESEARCH_COLOR, ms=5.5, zorder=2)
        ax.plot(xp, i, "o", color=POLICY_COLOR, ms=5.5, zorder=2)
        research_label_x = xr + (0.2 if xr < 0.5 else 0.0)
        policy_label_x = xp + (0.2 if xp < 0.5 else 0.0)
        ax.text(research_label_x, i - 0.30, f"{xr:.1f}", ha="center", va="center", fontsize=6.5,
                color="black")
        ax.text(policy_label_x, i + 0.30, f"{xp:.1f}", ha="center", va="center", fontsize=6.5,
                color="black")
        gap = xr - xp
        gap_x_offset = 1.4 if max(xr, xp) < 0.5 else 0.6
        ax.text(max(xr, xp) + gap_x_offset, i, f"{gap:+.1f}%", ha="left", va="center",
                fontsize=7, color="#222222")
    ax.set_yticks(y)
    ax.set_yticklabels([SDG_SHORT[s].replace("\n", " ") for s in sdgs], fontsize=8)
    ax.axvline(0, color="black", lw=0.5)
    ax.set_xlim(0, 31)
    # y inverted by the (bottom, top) ordering of set_ylim: SDG 1 at the top.
    ax.set_ylim(len(sdgs) - 0.5, -1.2)
    ax.set_xlabel("Share of corpus assigned to SDG (%)")
    ax.legend(handles=[
        plt.Line2D([0], [0], marker="o", ls="None", color=RESEARCH_COLOR, ms=5.5,
                   label=f"Research (abstract-weighted, n = {n_research})"),
        plt.Line2D([0], [0], marker="o", ls="None", color=POLICY_COLOR, ms=5.5,
                   label=f"Policy (document-weighted, n = {n_policy})"),
    ], loc="lower right", fontsize=8, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    _save_fig(fig, figures_dir, "fig2_coverage_profiles")


def plot_semantic_dumbbell(adj_map: dict[int, float], raw_map: dict[int, float],
                           figures_dir: Path) -> None:
    """fig4 — within-SDG semantic gap: adjusted (topic) vs raw baseline dumbbell.

    One row per SDG, sorted by the adjusted (post-INLP) gap descending. The
    adjusted gap is the blue canonical dot; the raw baseline is the grey dot;
    the connector between them IS the register component (raw − adjusted,
    signed — SDG 17 inverts, raw < adjusted). The register component is labelled
    right of the righter-most dot in cosine units. Palette: blue = canonical
    adjusted object, grey = raw/baseline (ratified scheme).
    """
    rows = [(s, adj_map[s], raw_map[s]) for s in range(1, 18)
            if s in adj_map and s in raw_map]
    rows.sort(key=lambda r: r[0])
    fig, ax = plt.subplots(figsize=(8.0, 6.8))
    y = np.arange(len(rows))
    for i, (sdg, adj, raw) in enumerate(rows):
        ax.plot([adj, raw], [i, i], color="#BBBBBB", lw=1.4, zorder=1)
        ax.plot(adj, i, "o", color=RESEARCH_COLOR, ms=5.5, zorder=2)
        ax.plot(raw, i, "o", color=BASELINE_GREY, ms=5.5, zorder=2)
        ax.text(adj, i - 0.30, f"{adj:.3f}", ha="center", va="center", fontsize=6.5,
                color="black")
        ax.text(raw, i + 0.30, f"{raw:.3f}", ha="center", va="center", fontsize=6.5,
                color="black")
        reg = raw - adj
        ax.text(max(adj, raw) + 0.006, i, f"{reg:+.3f}", ha="left", va="center",
                fontsize=7, color="#222222")
    ax.set_yticks(y)
    ax.set_yticklabels([SDG_SHORT[s].replace("\n", " ") for s, _, _ in rows], fontsize=8)
    ax.axvline(0, color="black", lw=0.5)
    ax.set_xlim(0, max(raw_map.values()) + 0.05)
    # y inverted: SDG 1 at the top, matching the coverage figure.
    ax.set_ylim(len(rows) - 0.5, -1.2)
    ax.set_xlabel("Semantic gap (1 − cosine similarity; adjusted = after INLP register removal)")
    ax.legend(handles=[
        plt.Line2D([0], [0], marker="o", ls="None", color=RESEARCH_COLOR, ms=5.5,
                   label="Adjusted Gap"),
        plt.Line2D([0], [0], marker="o", ls="None", color=BASELINE_GREY, ms=5.5,
                   label="Raw Gap"),
    ], loc="lower right", fontsize=8, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    _save_fig(fig, figures_dir, "fig4_semantic_gap")

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
        if model_slug(args.embed_model) == "mpnet":
            # Appendix table-to-figure renders (canonical MPNet track only).
            appendix_root = layout.root.parent / "appendix" / "mpnet"
            expected += [
                figures_dir / "fig10_concept_coverage.pdf",
                figures_dir / "fig12_register_convergence.pdf",
                appendix_root / "c_sample_stability" / "figures" / "fig11_sample_stability.pdf",
                appendix_root / "h1_cross_method_gap_values" / "figures" / "fig13_cross_method_heatmap.pdf",
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
    # Figure 1 — Coverage profiles: research vs policy dumbbell per SDG
    # (replaces the former grouped bar chart; value labels above/below the
    # dots, coverage gap right of the righter-most dot, all in %)
    # -----------------------------------------------------------------------
    df_sorted = df.sort_values("sdg", ascending=True).reset_index(drop=True)
    plot_coverage_dumbbell(
        df_sorted, figures_dir, n_research=N_RESEARCH_PAPERS, n_policy=N_POLICY_DOCS,
    )

    # -----------------------------------------------------------------------
    # Figure 2 — Semantic gap by SDG (adjusted canonical vs raw baseline)
    # rendered as a dumbbell; the connector IS the register component.
    # -----------------------------------------------------------------------
    if use_adjusted and adj_map and raw_map:
        plot_semantic_dumbbell(adj_map, raw_map, figures_dir)
    else:
        # Fallback: raw-only horizontal bars when no adjusted data exists.
        df_sem = df_sem_valid.sort_values("semantic_gap", ascending=False).reset_index(drop=True)
        fig2, ax2 = plt.subplots(figsize=(7, 5.5))
        y = np.arange(len(df_sem))
        colors = [RESEARCH_COLOR if v > median_semantic_gap else BASELINE_GREY for v in df_sem["semantic_gap"]]
        ax2.barh(y, df_sem["semantic_gap"], color=colors, alpha=0.88)
        for i, val in enumerate(df_sem["semantic_gap"]):
            ax2.text(val + 0.008, i, f"{val:.3f}", va="center", ha="left", fontsize=7.5)
        ax2.axvline(median_semantic_gap, color="grey", linestyle="--", linewidth=1,
                    label=f"Median ({median_semantic_gap:.3f})")
        ax2.axvline(mean_semantic_gap, color="black", linestyle=":", linewidth=1,
                    label=f"Mean ({mean_semantic_gap:.3f})")
        ax2.set_yticks(y)
        ax2.set_yticklabels([SDG_SHORT[int(r["sdg"])].replace("\n", " ") for _, r in df_sem.iterrows()], fontsize=8)
        ax2.set_xlabel("Semantic gap (1 − cosine similarity between research and policy sub-centroids)")
        ax2.legend(handles=[
            mpatches.Patch(color=RESEARCH_COLOR, alpha=0.88, label="Above median gap"),
            mpatches.Patch(color=BASELINE_GREY, alpha=0.88, label="Below median gap"),
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

    # -----------------------------------------------------------------------
    # Appendix table-to-figure renders (MPNet canonical track only)
    # -----------------------------------------------------------------------
    if model_slug(args.embed_model) == "mpnet":
        plot_concept_coverage_figure(layout)
        plot_sample_stability_figure(layout, args.embed_model)
        plot_register_convergence_figure(layout)
        plot_cross_method_heatmap_figure(layout, args.embed_model)

    print(f"\\nAll figures saved to {figures_dir}")


if __name__ == "__main__":
    main()

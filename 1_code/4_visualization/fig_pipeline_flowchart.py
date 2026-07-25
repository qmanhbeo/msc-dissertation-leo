"""
Figure 6 — Methodology pipeline flowchart.

Single-column flow with source-specific annotations, showing the
end-to-end pipeline from raw data through preprocessing, segmentation,
embedding, classifier training, inference, and gap analysis.

All n values are verified against data artifacts, not manuscript prose.

Outputs:
  4_outputs/main/figures/fig6_pipeline_flowchart.pdf
  4_outputs/main/figures/fig6_pipeline_flowchart.png
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-dissertation")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = ROOT / "1_code"
SHARED_DIR = ROOT / "1_code" / "7_main_analysis" / "0_shared"
for path in (CODE_ROOT, SHARED_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
from model_utils import DEFAULT_OUTPUT_ROOT

OUTPUT_DIR = ROOT / "4_outputs" / "main" / "figures"

# ---------------------------------------------------------------------------
# Colour palette (matches plot_figures.py / Tol colourblind-safe)
# ---------------------------------------------------------------------------
RESEARCH_CLR  = "#0077BB"
POLICY_CLR    = "#EE7733"
REFERENCE_CLR = "#228833"
STAGE_CLR     = "#DDDDDD"
ARROW_CLR     = "#666666"
BYPASS_CLR    = "#CCBB44"
TEXT_CLR      = "#222222"

# ---------------------------------------------------------------------------
# Shared style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "figure.dpi": 300,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})


def rounded_box(ax, x, y, w, h, text, *, subtext="",
                color=STAGE_CLR, fs=8.5, sfs=7.5,
                bold=False, tc=TEXT_CLR, lw=0.8):
    patch = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.1",
        facecolor=color, edgecolor=TEXT_CLR,
        linewidth=lw, zorder=2,
    )
    ax.add_patch(patch)
    ax.text(x, y + (0.1 * h if subtext else 0), text,
            ha="center", va="center",
            fontsize=fs, fontweight="bold" if bold else "normal",
            color=tc, zorder=3)
    if subtext:
        ax.text(x, y - 0.16 * h, subtext,
                ha="center", va="top",
                fontsize=sfs, color=tc, zorder=3)


def side_anno(ax, x, y, lines, color="#555555"):
    """Small annotation box on the right."""
    n = len(lines)
    h = 0.15 * n + 0.1
    patch = FancyBboxPatch(
        (x - 0.02, y - h / 2), 3.6, h,
        boxstyle="round,pad=0.05",
        facecolor=color, edgecolor="none",
        alpha=0.08, zorder=1,
    )
    ax.add_patch(patch)
    for i, line in enumerate(lines):
        ax.text(x + 0.08, y + h / 2 - 0.12 - i * 0.16, line,
                ha="left", va="top",
                fontsize=6.5, color=TEXT_CLR, zorder=3)


def arrow(ax, x1, y1, x2, y2, label="", style="->"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=ARROW_CLR, lw=0.8),
                zorder=1)
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx + 0.05, my, label,
                ha="left", va="center",
                fontsize=6.5, color=TEXT_CLR, fontstyle="italic", zorder=3)


def dashed_arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=BYPASS_CLR,
                                lw=0.6, linestyle="dashed"),
                zorder=1)


def main() -> None:
    # -----------------------------------------------------------------------
    # Geometry  (centre column x=3.0, annotations on right at x=5.2)
    # -----------------------------------------------------------------------
    CX = 3.0    # centre of main pipeline boxes
    AX = 5.3    # left edge of annotation boxes
    BW = 3.4    # main box width
    BH = 0.45   # main box height
    SEP = 0.65  # vertical separation between stages

    # Y positions (bottom to top)
    Y_ANALYSIS  = 8.8
    Y_INFER     = 7.4
    Y_TRAIN     = 5.9
    Y_EMBED     = 4.3
    Y_SEGMENT   = 2.5
    Y_PREPROC   = 0.7
    Y_RAW       = -1.0

    fig, ax = plt.subplots(figsize=(9.5, 10.5))
    ax.set_xlim(0, 9.0)
    ax.set_ylim(-1.5, 10.0)
    ax.axis("off")

    # ===================================================================
    # Title
    # ===================================================================
    ax.text(CX, 9.7, "Methodology Pipeline Overview",
            ha="center", va="center",
            fontsize=12, fontweight="bold", color=TEXT_CLR)

    # ===================================================================
    # Stage 0 — Raw Data Sources (3 input streams)
    # ===================================================================
    # Three input boxes arranged side by side above the main flow
    IN_W = 1.8
    IN_H = 0.5
    in_y = Y_RAW
    for x, label, sub, clr in [
        (0.6, "OpenAlex", "Research corpus", RESEARCH_CLR),
        (CX, "4 policy sources", "Scrape / Manual / UNGDC / SDGi", POLICY_CLR),
        (5.4, "5 labeled corpora", "OSDG / Benchmark / KH / SDGi / Aurora", REFERENCE_CLR),
    ]:
        rounded_box(ax, x, in_y, IN_W, IN_H, label, subtext=sub,
                    color=clr, tc="white", fs=8, sfs=6.5)

    # Arrows from 3 sources → Preprocess
    arrow(ax, 0.6, in_y - 0.3, CX, Y_PREPROC + 0.25)
    arrow(ax, CX, in_y - 0.3, CX, Y_PREPROC + 0.25)
    arrow(ax, 5.4, in_y - 0.3, CX, Y_PREPROC + 0.25)

    # ===================================================================
    # Stage 1 — Preprocess
    # ===================================================================
    rounded_box(ax, CX, Y_PREPROC, BW, BH,
                "Preprocess — clean text, English filter, min 20 words",
                subtext="Research: 2,543,698 papers  |  Policy: ~6,400 docs  |  Reference: ~48,000 texts",
                color="#333333", tc="white", fs=9, sfs=7.5)

    side_anno(ax, AX, Y_PREPROC, [
        "Research: OpenAlex abstracts",
        "  AI + SDG boolean query (68 terms)",
        "Policy: deduplicate, strip boilerplate",
        "  UNGDC: paragraph-level SDG keyword filter",
        "Reference: agreement ≥ 0.5 (OSDG),",
        "  label=True (Benchmark), English (SDGi)",
    ])

    arrow(ax, CX, Y_PREPROC - 0.25, CX, Y_SEGMENT + 0.25)

    # ===================================================================
    # Stage 2 — Segmentation
    # ===================================================================
    rounded_box(ax, CX, Y_SEGMENT, BW, BH,
                "Segment — token-aware (margin=10, min 20 words)",
                subtext="Research: 3,105,144  |  Policy: 40,547 (4 sources merged)",
                color="#333333", tc="white", fs=9, sfs=7.5)

    side_anno(ax, AX, Y_SEGMENT, [
        "All corpora use segment_text() with",
        "  identical parameters (margin=10,",
        "  min_words=20). OSDG and Benchmark",
        "  bypass segmentation (already short",
        "  single-label texts).",
        "Policy merge order: scrape → manual",
        "  → UNGDC → SDGi (SDGi-first dedup).",
    ])

    # Dashed bypass arrow from reference preprocess to embed
    # (OSDG + Benchmark skip segmentation)
    dashed_arrow(ax, AX + 1.5, Y_SEGMENT - 0.1, AX + 1.5, Y_EMBED + 0.3)

    arrow(ax, CX, Y_SEGMENT - 0.25, CX, Y_EMBED + 0.25)

    # ===================================================================
    # Stage 3 — Embedding
    # ===================================================================
    rounded_box(ax, CX, Y_EMBED, BW, BH,
                "Embed — all-mpnet-base-v2 (768d, L2-normalised)",
                subtext="3,208,204 total vectors",
                color="#333333", tc="white", fs=9, sfs=7.5)

    side_anno(ax, AX, Y_EMBED, [
        "Research:  3,105,144 vectors",
        "Policy:      40,547 vectors",
        "Reference:   62,513 vectors",
        "Each source embedded separately,",
        "  then policy sub-sources merged.",
    ])

    arrow(ax, CX, Y_EMBED - 0.25, CX, Y_TRAIN + 0.25)

    # ===================================================================
    # Stage 4 — Training
    # ===================================================================
    tr_box_h = 0.65
    patch = FancyBboxPatch(
        (CX - BW / 2, Y_TRAIN - tr_box_h / 2), BW, tr_box_h,
        boxstyle="round,pad=0.1",
        facecolor=REFERENCE_CLR, edgecolor=TEXT_CLR,
        linewidth=0.8, zorder=2,
    )
    ax.add_patch(patch)
    ax.text(CX, Y_TRAIN + 0.08,
            "Train LR Classifier (C=10, L2, lbfgs)",
            ha="center", va="center",
            fontsize=9, fontweight="bold", color="white", zorder=3)
    ax.text(CX, Y_TRAIN - 0.16,
            "Pool 5 reference corpora → single-label filter (62,513) → 52,779 train / 9,734 test  |  macro-F₁ = 0.823",
            ha="center", va="top",
            fontsize=7, color="white", zorder=3)

    side_anno(ax, AX, Y_TRAIN, [
        "Training pool breakdown:",
        "  OSDG:          30,534",
        "  SDGi (labels): 20,267",
        "  Knowledge Hub:  6,125",
        "  Aurora:         4,971",
        "  Benchmark:        616",
        "Per-source stratified GroupKFold.",
    ])

    # Branch: trained model → Inference (both research + policy)
    arrow(ax, CX - 0.5, Y_TRAIN - tr_box_h / 2 - 0.05,
          CX - 0.5, Y_INFER + 0.25,
          label="Score with trained LR")

    arrow(ax, CX + 0.5, Y_TRAIN - tr_box_h / 2 - 0.05,
          CX + 0.5, Y_INFER + 0.25)

    # ===================================================================
    # Stage 5 — Inference (two sub-boxes side by side)
    # ===================================================================
    in_w = 1.5
    in_h = 0.5
    rounded_box(ax, CX - 0.95, Y_INFER, in_w, in_h,
                "Score Research",
                subtext="3,105,144 argmax assignments",
                color=RESEARCH_CLR, tc="white", fs=8.5, sfs=7)
    rounded_box(ax, CX + 0.95, Y_INFER, in_w, in_h,
                "Score Policy",
                subtext="40,547 argmax assignments",
                color=POLICY_CLR, tc="white", fs=8.5, sfs=7)

    side_anno(ax, AX, Y_INFER, [
        "Research: per-paper SDG assignment",
        "  via LR argmax over 17 classes.",
        "Policy: per-segment SDG assignment.",
        "Both use the same trained classifier.",
        "Document-weighting applied post-hoc.",
    ])

    arrow(ax, CX - 0.95, Y_INFER - 0.3, CX, Y_ANALYSIS + 0.25)
    arrow(ax, CX + 0.95, Y_INFER - 0.3, CX, Y_ANALYSIS + 0.25)

    # ===================================================================
    # Stage 6 — Analysis
    # ===================================================================
    an_w = 4.0
    an_h = 0.55
    patch = FancyBboxPatch(
        (CX - an_w / 2, Y_ANALYSIS - an_h / 2), an_w, an_h,
        boxstyle="round,pad=0.1",
        facecolor="#444444", edgecolor=TEXT_CLR,
        linewidth=0.8, zorder=2,
    )
    ax.add_patch(patch)
    ax.text(CX, Y_ANALYSIS,
            "Coverage Gap  ↔  Semantic Gap  →  Coverage–Semantic Interaction  →  Figures 3–5",
            ha="center", va="center",
            fontsize=8.5, color="white", zorder=3)

    side_anno(ax, AX, Y_ANALYSIS, [
        "Coverage: document-weighted SDG",
        "  profile comparison (Fig 3).",
        "Semantic: within-SDG centroid",
        "  distance (Fig 4).",
        "Interaction: scatter + correlation",
        "  tests (Fig 5, Table 2).",
    ])

    # ===================================================================
    # Stage labels on left
    # ===================================================================
    stages = [
        (Y_RAW + 0.05,     "1. Fetch"),
        (Y_PREPROC, "2. Preprocess"),
        (Y_SEGMENT, "3. Segment"),
        (Y_EMBED,   "4. Embed"),
        (Y_TRAIN,   "5. Train"),
        (Y_INFER,   "6. Infer"),
        (Y_ANALYSIS,"7. Analysis"),
    ]
    for y, label in stages:
        ax.text(0.12, y, label, ha="left", va="center",
                fontsize=7.5, color="#999999",
                fontweight="bold", zorder=3)

    # ===================================================================
    # Legend
    # ===================================================================
    leg_y = -1.3
    for x, clr, label in [
        (0.4,  RESEARCH_CLR, "Research pipeline"),
        (2.6, POLICY_CLR,   "Policy pipeline"),
        (4.8, REFERENCE_CLR,"Reference / Training"),
        (7.0, "#333333",    "Shared stage"),
    ]:
        rect = FancyBboxPatch(
            (x, leg_y), 0.3, 0.16,
            boxstyle="round,pad=0.04",
            facecolor=clr, edgecolor=TEXT_CLR,
            linewidth=0.5, zorder=2,
        )
        ax.add_patch(rect)
        ax.text(x + 0.38, leg_y + 0.08, label,
                ha="left", va="center",
                fontsize=7, color=TEXT_CLR, zorder=3)

    # ===================================================================
    # Save
    # ===================================================================
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for fmt, ext in [("pdf", ".pdf"), ("png", ".png")]:
        path = OUTPUT_DIR / f"fig6_pipeline_flowchart{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight", format=fmt)
        print(f"Saved → {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()

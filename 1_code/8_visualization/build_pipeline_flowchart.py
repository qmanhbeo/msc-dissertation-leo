#!/usr/bin/env python3
"""Regenerate the manuscript's pipeline-flowchart figure (Fig. 6) with matplotlib.

This supersedes the hand-authored TikZ diagram ``fig_pipeline_flowchart.tex``.
The figure is model-independent, so it is emitted once into
``4_outputs/conceptual_figs/`` rather than a model-namespaced directory.

It is deterministic (no randomness) and idempotent: skipped when the PDF + PNG
already exist and ``--overwrite`` is not passed.
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib-dissertation"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "4_outputs" / "conceptual_figs"
JOBNAME = "fig6_pipeline_flowchart"


# ---------- palette ----------
C_RESEARCH = "#AED6F1"
C_RESEARCH_EDGE = "#1F618D"
C_POLICY = "#FAD7A0"
C_POLICY_EDGE = "#B9770E"
C_REFERENCE = "#A9DFBF"
C_REF_EDGE = "#1E8449"
C_SHARED = "#EAECEE"
C_SHARED_EDGE = "#4D5656"
C_SECONDARY_FILL = "#EAF2F8"
C_NOTE = "#F5F5F5"
C_NOTE_EDGE = "#7F8C8D"


def _needs_build(out_pdf: Path, out_png: Path, overwrite: bool) -> bool:
    if overwrite:
        return True
    return not (out_pdf.exists() and out_png.exists())


def build(overwrite: bool = False) -> int:
    out_pdf = OUT_DIR / f"{JOBNAME}.pdf"
    out_png = OUT_DIR / f"{JOBNAME}.png"
    if not _needs_build(out_pdf, out_png, overwrite):
        print(f"[pipeline-flowchart] UP-TO-DATE {JOBNAME}.pdf + .png")
        return 0

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(13.5, 16.8))
    ax.axis("off")

    LINE_H = 0.475   # data-units per text line (must match actual rendered line height)
    PAD = 0.42       # top+bottom internal padding (tightened from 0.62)

    def n_lines(t):
        return t.count("\n") + 1

    def row_height(texts, min_h=0.0):
        return max(min_h, PAD + LINE_H * max(n_lines(t) for t in texts))

    def box(x, y_top, w, h, text, fc, ec, fontsize=8.0, dashed=False, lw=1.5, zorder=2):
        y = y_top - h
        style = "round,pad=0.06,rounding_size=0.10"
        b = FancyBboxPatch((x, y), w, h, boxstyle=style, facecolor=fc, edgecolor=ec,
                           linewidth=lw, linestyle="--" if dashed else "-", zorder=zorder)
        ax.add_patch(b)
        cx, cy = x + w / 2, y + h / 2
        ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize, zorder=zorder + 1,
                linespacing=1.32, color="#1B2631")
        return dict(cx=cx, top=y_top, bottom=y, left=x, right=x + w, y_mid=cy)

    def arrow(p_from, p_to, color="#4D5656", lw=1.4, dashed=False, connectionstyle="arc3,rad=0.0", mutation_scale=14):
        a = FancyArrowPatch(p_from, p_to, arrowstyle="-|>", mutation_scale=mutation_scale,
                            linewidth=lw, color=color, linestyle="--" if dashed else "-",
                            connectionstyle=connectionstyle, zorder=1)
        ax.add_patch(a)

    def elbow_fan(source_xy, branches, color="#4D5656", lw=1.3):
        """source_xy: (x, y) — left-center of the source box.
        branches: list of (x_target, y_target_top, dashed_bool).
        Routes as ONE horizontal run at the source's own height, all the way
        across to each branch's x-position, then a single straight drop down
        into that branch — left first, down only once directly above the target."""
        x0, y0 = source_xy
        xs = [b[0] for b in branches]
        ax.plot([min(xs), x0], [y0, y0], color=color, lw=lw, zorder=1, solid_capstyle="butt")
        for tx, ty, dashed in branches:
            arrow((tx, y0), (tx, ty), color=color, lw=lw, dashed=dashed)

    # =========================================================
    # Columns
    # =========================================================
    col_w = 4.15
    gap = 0.35
    x1 = 0.4
    x2 = x1 + col_w + gap
    x3 = x2 + col_w + gap
    x4 = x3 + col_w + gap
    cols = [x1, x2, x3, x4]

    cursor = 24.0
    TOP = cursor

    # =========================================================
    # TIER 1 — Corpus Construction
    # =========================================================
    ax.text(0.2, cursor, "Tier 1 — Corpus Construction (four parallel intake tracks)",
            fontsize=13.5, fontweight="bold", ha="left", color="#1B2631")
    cursor -= 0.75

    t_srcA = "OpenAlex Concept Retrieval\nAI/ML field-of-study tags\n(robustness track, Sec. 4.4 / App. A)"
    t_srcB = "OpenAlex API\n68 Boolean queries (17 SDG × 4 AI terms)\n2018–2025"
    t_srcC = "4 Policy Sources\nCurated AI+SDG (94) / SDGi VNR-VLR /\nUNGDC / scrape + manual"
    t_srcD = "5 Reference Corpora\nOSDG / Benchmark / IISD Knowledge Hub /\nSDGi-labels / Aurora"
    h_src = row_height([t_srcA, t_srcB, t_srcC, t_srcD])
    srcA = box(x1, cursor, col_w, h_src, t_srcA, C_SECONDARY_FILL, C_RESEARCH_EDGE, dashed=True)
    srcB = box(x2, cursor, col_w, h_src, t_srcB, C_RESEARCH, C_RESEARCH_EDGE)
    srcC = box(x3, cursor, col_w, h_src, t_srcC, C_POLICY, C_POLICY_EDGE)
    srcD = box(x4, cursor, col_w, h_src, t_srcD, C_REFERENCE, C_REF_EDGE)
    cursor -= (h_src + 0.35)

    t_ppA = "Preprocess\nSDG filter: NONE (AI/ML only)\nLanguage filter: none\nMin. 20 words\nCapped at 100,000 papers"
    t_ppB = "Preprocess\nSDG filter: yes (OpenAlex native)\nLanguage filter: none (99.5% English)\nMin. 20 words"
    t_ppC = "Preprocess\nSDG filter: n/a (institutional SDG corpora)\nLanguage filter: none\nMin. 20 words"
    t_ppD = "Preprocess\nLanguage filter: none\nMin. 20 words\nSingle-label filtering applied"
    h_pp = row_height([t_ppA, t_ppB, t_ppC, t_ppD])
    ppA = box(x1, cursor, col_w, h_pp, t_ppA, C_SECONDARY_FILL, C_RESEARCH_EDGE, dashed=True)
    ppB = box(x2, cursor, col_w, h_pp, t_ppB, C_RESEARCH, C_RESEARCH_EDGE)
    ppC = box(x3, cursor, col_w, h_pp, t_ppC, C_POLICY, C_POLICY_EDGE)
    ppD = box(x4, cursor, col_w, h_pp, t_ppD, C_REFERENCE, C_REF_EDGE)
    cursor -= (h_pp + 0.35)

    t_sgA = "Token-Aware Segmentation\n~100,000 papers\n(SDG label: classifier-assigned, post hoc)"
    t_sgB = "Token-Aware Segmentation\n2,536,771 abstracts →\n3,105,144 segments"
    t_sgC = "Segment + Merge\n6,367 documents →\n40,597 segments"
    t_sgD = "Segment + Bypass\n62,173 single-label segments\n(52,835 train / 9,338 test)"
    h_seg = row_height([t_sgA, t_sgB, t_sgC, t_sgD])
    sgA = box(x1, cursor, col_w, h_seg, t_sgA, C_SECONDARY_FILL, C_RESEARCH_EDGE, dashed=True)
    sgB = box(x2, cursor, col_w, h_seg, t_sgB, C_RESEARCH, C_RESEARCH_EDGE)
    sgC = box(x3, cursor, col_w, h_seg, t_sgC, C_POLICY, C_POLICY_EDGE)
    sgD = box(x4, cursor, col_w, h_seg, t_sgD, C_REFERENCE, C_REF_EDGE)

    for s, p in [(srcA, ppA), (srcB, ppB), (srcC, ppC), (srcD, ppD)]:
        arrow((s["cx"], s["bottom"]), (p["cx"], p["top"]))
    for p, sg, d in [(ppA, sgA, True), (ppB, sgB, False), (ppC, sgC, False), (ppD, sgD, False)]:
        arrow((p["cx"], p["bottom"]), (sg["cx"], sg["top"]), dashed=d)

    cursor -= (h_seg + 0.42)

    # =========================================================
    # Shared Embedding + Encoder-Robustness side note
    # =========================================================
    emb_w = (x4 + col_w) - x1
    t_emb = (
        "Shared Embedding — all-mpnet-base-v2 (768d, L2-normalised, frozen)\n"
        "Concept ~100,000 (side track) | Research 3,105,144 | Policy 40,597 | Reference 62,173\n\n"
        "Encoder robustness (Sec. 3.2 / 4.4): a separate, independently random-seeded 100,000-paper sample drawn from the full 2,536,771-paper\n"
        "primary research corpus — distinct from the concept-retrieved sample above — plus the full Policy and Reference corpora, all re-embedded\n"
        "with all-MiniLM-L6-v2 (384d) and SciBERT (768d); classifier retrained per encoder and gap rankings recomputed for comparison."
    )
    h_emb = row_height([t_emb])
    emb = box(x1, cursor, emb_w, h_emb, t_emb, C_SHARED, C_SHARED_EDGE, fontsize=8.4)

    for s, sg in [(x1, sgA), (x2, sgB), (x3, sgC), (x4, sgD)]:
        dashed = (s == x1)
        arrow((sg["cx"], sg["bottom"]), (sg["cx"], emb["top"]), dashed=dashed)

    cursor -= (h_emb + 0.5)

    # =========================================================
    # TIER 2 — Classification & Decomposition
    # =========================================================
    ax.text(0.2, cursor, "Tier 2 — Classification & Decomposition", fontsize=13.5,
            fontweight="bold", ha="left", color="#1B2631")
    cursor -= 0.75

    # --- Row: Classifier trained FIRST ---
    t_clf = "① Train LR Classifier (C=3.0, L2)\nReference pool → 52,835 train / 9,338 test\nMacro-F1 = 0.816 (Table 1)"
    h_clf = row_height([t_clf])
    clf = box(x4, cursor, col_w, h_clf, t_clf, C_REFERENCE, C_REF_EDGE)
    clf_source_x = x4 + col_w / 2
    arrow((clf_source_x, emb["bottom"]), (clf["cx"], clf["top"]))

    cursor -= (h_clf + 0.4)

    # --- Row: Scoring happens only AFTER the classifier is trained ---
    t_scC = "② Score Research (concept)\nsame trained classifier\n(robustness track)"
    t_scR = "② Score Research (primary)\nLR argmax, 17 SDGs\n3,105,144 segment assignments"
    t_scP = "② Score Policy\nLR argmax, 17 SDGs\n40,597 segment assignments"
    h_sc = row_height([t_scC, t_scR, t_scP])
    scC = box(x1, cursor, col_w, h_sc, t_scC, C_SECONDARY_FILL, C_RESEARCH_EDGE, dashed=True)
    scR = box(x2, cursor, col_w, h_sc, t_scR, C_RESEARCH, C_RESEARCH_EDGE)
    scP = box(x3, cursor, col_w, h_sc, t_scP, C_POLICY, C_POLICY_EDGE)

    for s, target in [(x1, scC), (x2, scR), (x3, scP)]:
        dashed = (s == x1)
        arrow((s + col_w / 2, emb["bottom"]), (target["cx"], target["top"]), dashed=dashed)

    elbow_fan(
        source_xy=(clf["left"], clf["y_mid"]),
        branches=[
            (scP["cx"] - 0.4, scP["top"], False),
            (scR["cx"] - 0.4, scR["top"], False),
            (scC["cx"] - 0.4, scC["top"], True),
        ],
    )

    cursor -= (h_sc + 1.5)   # extra vertical room so cov/sem arrows are not squeezed under boxes

    # --- Robustness terminus for concept track ---
    t_rob = "Robustness check\n(Sec. 4.4, Appendix A)\nReuses main register-\nadjusted space (untested\nassumption, Sec. 1.6)"
    t_cov = "Coverage Gap Analysis (Sec. 3.6)\nDocument-weighted SDG shares\nCoverageGapⱼ = |Researchⱼ − Policyⱼ|"
    t_sem = "Semantic Gap Analysis (Sec. 3.7)\nSDG-level research vs. policy centroids\nRaw Gap = 1 − (cᵢꞋ · cₚꞌ)"
    h_row3 = row_height([t_rob, t_cov, t_sem])
    rob = box(x1, cursor, col_w, h_row3, t_rob, C_NOTE, C_NOTE_EDGE, fontsize=7.8, dashed=True)
    cov = box(x2, cursor, col_w, h_row3, t_cov, C_SHARED, C_SHARED_EDGE)
    sem = box(x3, cursor, col_w, h_row3, t_sem, C_SHARED, C_SHARED_EDGE)

    arrow((scC["cx"], scC["bottom"]), (rob["cx"], rob["top"]), dashed=True)
    arrow((scR["cx"], scR["bottom"]), (cov["cx"] - 1.0, cov["top"]))
    arrow((scP["cx"], scP["bottom"]), (cov["cx"] + 1.0, cov["top"]), connectionstyle="arc3,rad=0.28")
    arrow((scP["cx"], scP["bottom"]), (sem["cx"] + 0.7, sem["top"]))
    arrow((scR["cx"], scR["bottom"]), (sem["cx"] - 1.1, sem["top"]), connectionstyle="arc3,rad=-0.32")

    cursor -= (h_row3 + 0.42)

    # --- INLP — same width as Semantic Gap box directly above it ---
    t_inlp = ("SDG-Stratified INLP\n(Sec. 3.8, Ravfogel et al. 2020)\n34-class strata, n_target = 1,123/class\n"
              "40–79 iterations\n\nRaw Gap → [INLP] →\nAdjusted Gap (topic) + Register Component")
    h_inlp = row_height([t_inlp])
    inlp = box(x3, cursor, col_w, h_inlp, t_inlp, "#FDEBD0", "#B9770E", fontsize=8.0)
    arrow((sem["cx"], sem["bottom"]), (inlp["cx"], inlp["top"]))

    cursor -= (h_inlp + 0.45)

    # =========================================================
    # Final convergence
    # =========================================================
    fin_w = (x4 + col_w) - x2
    t_fin = ("Coverage–Semantic Interaction (Sec. 3.9)\n"
             "H1a–H1d: Pearson r / Spearman ρ between coverage predictors and\n"
             "{raw, adjusted, register} semantic gap across 17 SDGs "
             "(Monte Carlo permutation, 100,000 resamples)")
    h_fin = row_height([t_fin])
    fin = box(x2, cursor, fin_w, h_fin, t_fin, C_SHARED, C_SHARED_EDGE, fontsize=8.6)
    arrow((cov["cx"], cov["bottom"]), (fin["cx"] - 2.7, fin["top"]))
    arrow((inlp["cx"], inlp["bottom"]), (fin["cx"] + 2.4, fin["top"]))

    cursor -= (h_fin + 0.55)

    # =========================================================
    # Legend
    # =========================================================
    sw, sh = 0.44, 0.30
    leg_items = [
        (C_SECONDARY_FILL, C_RESEARCH_EDGE, "Research (concept-retrieved, robustness track)"),
        (C_RESEARCH, C_RESEARCH_EDGE, "Research (keyword-retrieved, primary)"),
        (C_POLICY, C_POLICY_EDGE, "Policy"),
        (C_REFERENCE, C_REF_EDGE, "Reference"),
        (C_SHARED, C_SHARED_EDGE, "Shared / downstream analysis"),
    ]

    def legend_row(items, y, x_start, item_gap):
        x = x_start
        for fc, ec, label in items:
            ax.add_patch(mpatches.Rectangle((x, y), sw, sh, facecolor=fc, edgecolor=ec, linewidth=1.1))
            ax.text(x + sw + 0.16, y + sh / 2, label, ha="left", va="center", fontsize=8.8, color="#1B2631")
            x += item_gap

    legend_row(leg_items[:3], cursor, 0.6, item_gap=6.5)
    legend_row(leg_items[3:], cursor - 0.6, 0.6, item_gap=6.5)
    ax.add_line(Line2D([0.6, 1.04], [cursor - 1.2, cursor - 1.2], color="#4D5656", lw=1.4, linestyle="--"))
    ax.text(0.6 + sw + 0.16, cursor - 1.2, "Concept-retrieved robustness track (dashed)",
            ha="left", va="center", fontsize=8.8, color="#1B2631")

    cursor -= 1.9

    ax.set_xlim(0, x4 + col_w + 0.4)
    ax.set_ylim(cursor, TOP + 0.4)

    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[pipeline-flowchart] wrote {out_pdf} and {out_png}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--overwrite", action="store_true",
                    help="Regenerate even if the PDF and PNG already exist.")
    args = ap.parse_args()
    return build(overwrite=args.overwrite)


if __name__ == "__main__":
    raise SystemExit(main())

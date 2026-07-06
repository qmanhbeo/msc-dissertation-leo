"""
Build a descriptive PCA semantic landscape for the research-policy embedding space.

This figure is exploratory only. PCA is used to compress the shared embedding
Sentence-BERT space into two dimensions for visualisation. All substantive alignment
metrics in the dissertation remain computed in the original embedding space.

Fitting logic:
  1. Load all usable policy embeddings.
  2. Sample exactly the same number of research embeddings from the full research
     embedding universe using a fixed seed.
  3. Fit PCA on the balanced combined sample only.
  4. Project the balanced background clouds plus SDG reference centroids and the
     full-corpus research/policy SDG centroids.

Outputs:
  4_outputs/appendix/b1_pca_semantic_landscape/figures/fig_b1_pca_semantic_landscape.pdf
  4_outputs/appendix/b1_pca_semantic_landscape/figures/fig_b1_pca_semantic_landscape.png
  4_outputs/appendix/b1_pca_semantic_landscape/data/b1_pca_landscape_metadata.json
  4_outputs/appendix/b1_pca_semantic_landscape/tables/num_b1_pca_landscape.tex

Run from project root:
    python 1_code/3_main_analysis/3_appendix/0_pca_semantic_landscape.py
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-dissertation")
import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = ROOT / "1_code"
ANALYSIS_ROOT = Path(__file__).resolve().parents[1]
SHARED_DIR = ANALYSIS_ROOT / "0_shared"
for path in (CODE_ROOT, SHARED_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


from alignment_core import verify_unit_norms
from model_utils import DEFAULT_EMBED_MODEL, DEFAULT_OUTPUT_ROOT, embed_dir_for_model, scored_dir_for_model
from research_embedding_shards import load_sampled_research_embeddings, total_research_embedding_rows
import semantic_gap_shared
from semantic_gap_shared import (
    SEGMENT_CAP_PRIMARY,
    N_SDG,
    RANDOM_SEED,
    build_sub_centroid,
    cap_policy_indices_per_doc,
    get_cluster_assignments,
    load_json,
)
PCA_METADATA_JSON = "b1_pca_landscape_metadata.json"
PCA_NUM_TEX = "num_b1_pca_landscape.tex"

RESEARCH_COLOR = "#2166AC"
POLICY_COLOR = "#D6604D"
REFERENCE_COLOR = "#1B7837"
GAP_LINE_COLOR = "#666666"

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate the descriptive PCA semantic landscape figure.")
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_ROOT))
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    p.add_argument(
        "--policy-fit-cap",
        type=int,
        default=0,
        help="Optional cap on the number of policy segments used for PCA fitting. Default: 0 (use all policy segments).",
    )
    p.add_argument("--model", default=DEFAULT_EMBED_MODEL, help=argparse.SUPPRESS)
    return p.parse_args()


def fit_policy_sample_indices(n_policy_total: int, fit_cap: int, rng: np.random.Generator) -> np.ndarray:
    if fit_cap <= 0 or fit_cap >= n_policy_total:
        return np.arange(n_policy_total, dtype=np.int64)
    idx = np.sort(rng.choice(n_policy_total, size=fit_cap, replace=False).astype(np.int64))
    return idx


def build_policy_centroids(
    policy_emb: np.ndarray,
    policy_scores: np.ndarray,
    policy_ids: list[dict],
    segment_cap: int,
    rng_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    assignments = get_cluster_assignments(policy_scores)
    centroids = np.zeros((N_SDG, policy_emb.shape[1]), dtype=np.float32)
    available = np.zeros(N_SDG, dtype=bool)
    for sdg_idx in range(N_SDG):
        raw_idxs = np.flatnonzero(assignments == sdg_idx).tolist()
        capped = cap_policy_indices_per_doc(
            raw_idxs,
            policy_ids,
            segment_cap,
            np.random.default_rng(rng_seed + sdg_idx),
        )
        centroid, _ = build_sub_centroid(policy_emb, capped)
        if centroid is None:
            log.warning("SDG %2d: could not build policy centroid for PCA overlay", sdg_idx + 1)
            continue
        centroids[sdg_idx] = centroid
        available[sdg_idx] = True
    return centroids, available


def draw_marker_numbers(
    ax: plt.Axes,
    coords_2d: np.ndarray,
    available: np.ndarray | None,
    *,
    text_color: str,
    fontsize: float,
    stroke_color: str,
    stroke_width: float,
    zorder: int,
) -> None:
    for sdg_idx in range(N_SDG):
        if available is not None and not bool(available[sdg_idx]):
            continue
        x, y = coords_2d[sdg_idx]
        ax.text(
            float(x),
            float(y),
            str(sdg_idx + 1),
            ha="center",
            va="center",
            color=text_color,
            fontsize=fontsize,
            fontweight="bold",
            zorder=zorder,
            path_effects=[pe.withStroke(linewidth=stroke_width, foreground=stroke_color)],
        )


def write_metadata_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_num_tex(path: Path, payload: dict) -> None:
    def latex_int(v: int) -> str:
        return f"{v:,}".replace(",", "{,}")

    lines = [
        "% Auto-generated by 1_code/3_main_analysis/3_appendix/0_pca_semantic_landscape.py — do not edit manually",
        rf"\newcommand{{\PcaLandscapeTotalResearch}}{{{latex_int(int(payload['total_research_embeddings_available']))}}}",
        rf"\newcommand{{\PcaLandscapeTotalPolicy}}{{{latex_int(int(payload['total_policy_embeddings_available']))}}}",
        rf"\newcommand{{\PcaLandscapeResearchFitN}}{{{latex_int(int(payload['n_research_used_for_pca_fit']))}}}",
        rf"\newcommand{{\PcaLandscapePolicyFitN}}{{{latex_int(int(payload['n_policy_used_for_pca_fit']))}}}",
        rf"\newcommand{{\PcaLandscapePcOnePct}}{{{payload['pc1_explained_variance_ratio'] * 100:.1f}}}",
        rf"\newcommand{{\PcaLandscapePcTwoPct}}{{{payload['pc2_explained_variance_ratio'] * 100:.1f}}}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    _POLICY_EMB = semantic_gap_shared.get_policy_emb(args.model)
    _POLICY_IDS = semantic_gap_shared.get_policy_ids(args.model)
    _POLICY_SCORES = semantic_gap_shared.get_policy_scores(args.model)
    _RESEARCH_CENTROIDS = semantic_gap_shared.get_research_centroids(args.model)
    _RESEARCH_CENTROID_META = semantic_gap_shared.get_research_centroid_meta(args.model)
    out_root = Path(args.output_dir) / "appendix" / "b1_pca_semantic_landscape"
    data_dir = out_root / "data"
    tables_dir = out_root / "tables"
    figures_dir = out_root / "figures"
    for d in (data_dir, tables_dir, figures_dir):
        d.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    log.info("Output dir: %s", out_root)
    log.info("Loading policy embeddings: %s", _POLICY_EMB)
    policy_emb = np.load(_POLICY_EMB).astype(np.float32)
    verify_unit_norms(policy_emb, "policy embeddings")
    policy_scores = np.load(_POLICY_SCORES).astype(np.float32)
    policy_ids = load_json(_POLICY_IDS)
    if policy_emb.shape[0] != policy_scores.shape[0] or policy_emb.shape[0] != len(policy_ids):
        raise RuntimeError(
            f"Policy alignment mismatch for PCA stage: emb={policy_emb.shape[0]} "
            f"scores={policy_scores.shape[0]} ids={len(policy_ids)}"
        )

    n_policy_total = int(policy_emb.shape[0])
    policy_fit_indices = fit_policy_sample_indices(n_policy_total, args.policy_fit_cap, rng)
    policy_fit_emb = policy_emb[policy_fit_indices]

    embed_dir = embed_dir_for_model(args.model)
    scored_dir = scored_dir_for_model(args.model)
    sdg_centroids_path = scored_dir / "sdg_centroids.npy"
    research_manifest = embed_dir / "research_shards" / "metadata" / "manifest.json"
    total_research = total_research_embedding_rows(research_manifest, embed_dir)
    n_research_fit = int(policy_fit_emb.shape[0])
    if total_research < n_research_fit:
        raise RuntimeError(
            f"Research embedding universe is smaller than policy fit sample: {total_research} < {n_research_fit}"
        )

    research_sample_indices = np.sort(
        rng.choice(total_research, size=n_research_fit, replace=False).astype(np.int64)
    )
    log.info(
        "Balanced PCA fit sample: policy=%d, research=%d (total research available=%d)",
        policy_fit_emb.shape[0],
        research_sample_indices.size,
        total_research,
    )
    research_fit_emb = load_sampled_research_embeddings(research_manifest, research_sample_indices, embed_dir)
    verify_unit_norms(research_fit_emb, "sampled research embeddings")

    fit_matrix = np.concatenate([policy_fit_emb, research_fit_emb], axis=0).astype(np.float32)
    expected_rows = int(policy_fit_emb.shape[0] + research_fit_emb.shape[0])
    if fit_matrix.shape[0] != expected_rows:
        raise RuntimeError(f"PCA fit matrix row mismatch: expected {expected_rows}, got {fit_matrix.shape[0]}")
    log.info("PCA fit matrix shape: %s", fit_matrix.shape)

    pca = PCA(n_components=2, random_state=args.seed)
    pca.fit(fit_matrix)
    evr = pca.explained_variance_ratio_.astype(float)
    log.info("PCA explained variance ratio: PC1=%.4f, PC2=%.4f", evr[0], evr[1])

    policy_bg_2d = pca.transform(policy_fit_emb)
    research_bg_2d = pca.transform(research_fit_emb)

    sdg_centroids = np.load(sdg_centroids_path).astype(np.float32)
    research_centroids = np.load(_RESEARCH_CENTROIDS).astype(np.float32)
    research_meta = load_json(_RESEARCH_CENTROID_META)
    verify_unit_norms(sdg_centroids, "sdg centroids", n_sample=17)
    verify_unit_norms(research_centroids, "research centroids", n_sample=17)
    if sdg_centroids.shape != research_centroids.shape:
        raise RuntimeError(
            f"SDG/reference centroid shape mismatch: sdg_centroids={sdg_centroids.shape}, research={research_centroids.shape}"
        )
    if len(research_meta) != N_SDG:
        raise RuntimeError(f"Expected {N_SDG} research centroid meta rows, got {len(research_meta)}")

    policy_centroids, policy_centroid_available = build_policy_centroids(
        policy_emb=policy_emb,
        policy_scores=policy_scores,
        policy_ids=policy_ids,
        segment_cap=SEGMENT_CAP_PRIMARY,
        rng_seed=args.seed,
    )
    research_centroid_available = np.array(
        [not bool(row.get("zero_flag", False)) for row in research_meta],
        dtype=bool,
    )

    ref_2d = pca.transform(sdg_centroids)
    research_centroids_2d = pca.transform(research_centroids)
    policy_centroids_2d = pca.transform(policy_centroids)

    if ref_2d.shape != (N_SDG, 2):
        raise RuntimeError(f"Expected projected SDG centroids shape (17, 2), got {ref_2d.shape}")

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.dpi": 150,
        }
    )
    fig, ax = plt.subplots(figsize=(8.6, 6.5))

    ax.scatter(
        research_bg_2d[:, 0],
        research_bg_2d[:, 1],
        s=4,
        c=RESEARCH_COLOR,
        alpha=0.08,
        linewidths=0,
        rasterized=True,
        label="Research sample",
    )
    ax.scatter(
        policy_bg_2d[:, 0],
        policy_bg_2d[:, 1],
        s=4,
        c=POLICY_COLOR,
        alpha=0.08,
        linewidths=0,
        rasterized=True,
        label="Policy segments",
    )

    for sdg_idx in range(N_SDG):
        sdg = sdg_idx + 1
        if research_centroid_available[sdg_idx] and policy_centroid_available[sdg_idx]:
            ax.plot(
                [research_centroids_2d[sdg_idx, 0], policy_centroids_2d[sdg_idx, 0]],
                [research_centroids_2d[sdg_idx, 1], policy_centroids_2d[sdg_idx, 1]],
                color=GAP_LINE_COLOR,
                linewidth=0.8,
                alpha=0.75,
                zorder=4,
            )
        else:
            log.warning("SDG %2d: skipping PCA gap line because one corpus centroid is unavailable", sdg)

    ax.scatter(
        ref_2d[:, 0],
        ref_2d[:, 1],
        s=150,
        c=REFERENCE_COLOR,
        marker="D",
        edgecolors="white",
        linewidths=0.8,
        zorder=6,
        label="SDG reference centroid",
    )
    ax.scatter(
        research_centroids_2d[research_centroid_available, 0],
        research_centroids_2d[research_centroid_available, 1],
        s=95,
        c=RESEARCH_COLOR,
        marker="o",
        edgecolors="white",
        linewidths=0.6,
        zorder=7,
        label="Research SDG centroid",
    )
    ax.scatter(
        policy_centroids_2d[policy_centroid_available, 0],
        policy_centroids_2d[policy_centroid_available, 1],
        s=95,
        c=POLICY_COLOR,
        marker="s",
        edgecolors="white",
        linewidths=0.6,
        zorder=7,
        label="Policy SDG centroid",
    )
    draw_marker_numbers(
        ax,
        ref_2d,
        None,
        text_color="white",
        fontsize=7.2,
        stroke_color="#0A2E15",
        stroke_width=1.1,
        zorder=8,
    )
    draw_marker_numbers(
        ax,
        research_centroids_2d,
        research_centroid_available,
        text_color="white",
        fontsize=6.4,
        stroke_color="#0B223F",
        stroke_width=1.0,
        zorder=9,
    )
    draw_marker_numbers(
        ax,
        policy_centroids_2d,
        policy_centroid_available,
        text_color="white",
        fontsize=6.4,
        stroke_color="#5A2016",
        stroke_width=1.0,
        zorder=9,
    )

    ax.set_xlabel(f"PC1 ({evr[0] * 100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({evr[1] * 100:.1f}% variance)")
    ax.set_title(
        "PCA semantic landscape of research and policy embeddings\n"
        "Descriptive projection only; formal distances remain in the original embedding space",
        loc="left",
        fontsize=9,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.axhline(0.0, color="#cccccc", lw=0.6, zorder=1)
    ax.axvline(0.0, color="#cccccc", lw=0.6, zorder=1)

    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=RESEARCH_COLOR, alpha=0.55, markersize=5, label="Research sample"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=POLICY_COLOR, alpha=0.55, markersize=5, label="Policy segments"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor=REFERENCE_COLOR, markeredgecolor="white", markersize=7, label="SDG reference centroid"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=RESEARCH_COLOR, markeredgecolor="white", markersize=6, label="Research SDG centroid"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=POLICY_COLOR, markeredgecolor="white", markersize=6, label="Policy SDG centroid"),
        Line2D([0], [0], color=GAP_LINE_COLOR, lw=1.0, label="Research-policy gap line"),
    ]
    ax.legend(handles=legend_handles, loc="best", frameon=False)

    note = (
        "PCA is fitted on a balanced research-policy sample for visual interpretability. "
        "The projection is descriptive only; formal semantic distances are computed in the original embedding space."
    )
    fig.text(0.01, 0.01, note, ha="left", va="bottom", fontsize=7)
    fig.tight_layout(rect=(0, 0.04, 1, 1))

    pdf_path = figures_dir / "fig_b1_pca_semantic_landscape.pdf"
    png_path = figures_dir / "fig_b1_pca_semantic_landscape.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    log.info("Saved: %s", pdf_path)
    log.info("Saved: %s", png_path)

    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "random_seed": args.seed,
        "balanced_sampling_used": True,
        "total_research_embeddings_available": int(total_research),
        "total_policy_embeddings_available": int(n_policy_total),
        "n_research_used_for_pca_fit": int(research_fit_emb.shape[0]),
        "n_policy_used_for_pca_fit": int(policy_fit_emb.shape[0]),
        "n_research_projected_background": int(research_bg_2d.shape[0]),
        "n_policy_projected_background": int(policy_bg_2d.shape[0]),
        "policy_capped": bool(args.policy_fit_cap > 0 and args.policy_fit_cap < n_policy_total),
        "policy_fit_cap": int(args.policy_fit_cap),
        "research_trimmed_to_policy_count": True,
        "pc1_explained_variance_ratio": float(evr[0]),
        "pc2_explained_variance_ratio": float(evr[1]),
        "segment_cap_for_policy_gap_overlay": int(SEGMENT_CAP_PRIMARY),
        "note": (
            "PCA is visual-only. Main coverage and semantic-gap results remain computed in the original embedding space."
        ),
    }
    metadata_path = data_dir / PCA_METADATA_JSON
    write_metadata_json(metadata_path, metadata)
    write_num_tex(tables_dir / PCA_NUM_TEX, metadata)
    log.info("Saved: %s", metadata_path)
    log.info("Saved: %s", tables_dir / PCA_NUM_TEX)


if __name__ == "__main__":
    main()

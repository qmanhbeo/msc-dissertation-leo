"""
PCA before/after register-removal figure (two-panel).

Shows the geometric effect of INLP register removal on the MPNet embedding
space: the raw space (left) separates research and policy clouds, while the
adjusted space (right) merges them, confirming that the separation is driven
by register rather than topic.

Fitting logic:
  1. Load all usable policy embeddings (raw).
  2. Sample exactly the same number of research embeddings (same seed as
     the existing PCA landscape script).
  3. Fit PCA on the BALANCED RAW combined sample for panel (a).
  4. Project raw embeddings into raw PCA space.
  5. Project adjusted embeddings through G, then fit a SEPARATE PCA on
     the balanced adjusted sample for panel (b).
  6. Overlay SDG centroids and research/policy corpus centroids in each
     panel's own PCA space.

Outputs:
  4_outputs/{model}/figures/fig3_pca_register_before_after.pdf
  4_outputs/{model}/figures/fig3_pca_register_before_after.png
  4_outputs/{model}/data/pca_register_before_after_metadata.json
  4_outputs/{model}/tables/num15_pca_register_before_after.tex

Run from project root:
    python 1_code/7_main_analysis/1_main_text/0_pca_register_before_after.py
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib-dissertation"))
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
from model_utils import (
    DEFAULT_EMBED_MODEL,
    DEFAULT_OUTPUT_ROOT,
    embed_dir_for_model,
    embed_research_dir_for_model,
    output_dir_for_model,
    scored_dir_for_model,
    resolve_model_alias,
)
import register_utils
from register_utils import load_G, project
from shared_utils import fingerprint_of, should_skip, record_fingerprint
from research_embedding_shards import load_sampled_research_embeddings, total_research_embedding_rows
import semantic_gap_shared
from semantic_gap_shared import (
    SEGMENT_CAP_PRIMARY,
    N_SDG,
    RANDOM_SEED,
    build_sub_centroid,
    cap_policy_indices_per_doc,
    get_cluster_assignments,
    latex_int,
    load_json,
)

SCRIPT_VERSION = "2"
METADATA_JSON = "pca_register_before_after_metadata.json"
NUM_TEX = "num15_pca_register_before_after.tex"
FIG_NAME = "fig3_pca_register_before_after"

RESEARCH_COLOR = "#2166AC"
POLICY_COLOR = "#D6604D"
REFERENCE_COLOR = "#555555"
GAP_LINE_COLOR = "#666666"

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate the two-panel PCA before/after register-removal figure."
    )
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_ROOT))
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    p.add_argument(
        "--policy-fit-cap",
        type=int,
        default=0,
        help="Optional cap on the number of policy segments used for PCA fitting. Default: 0 (use all).",
    )
    p.add_argument(
        "--embed-model",
        default=DEFAULT_EMBED_MODEL,
        type=resolve_model_alias,
        help=argparse.SUPPRESS,
    )
    p.add_argument(
        "--n-components",
        type=int,
        default=2,
        help="Number of PCA components (default: %(default)s)",
    )
    p.add_argument("--overwrite", action="store_true", help=argparse.SUPPRESS)
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
            log.warning("SDG %2d: could not build policy centroid", sdg_idx + 1)
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


def _scatter_cloud(ax: plt.Axes, coords_2d: np.ndarray, color: str, label: str) -> None:
    ax.scatter(
        coords_2d[:, 0],
        coords_2d[:, 1],
        s=4,
        c=color,
        alpha=0.08,
        linewidths=0,
        rasterized=True,
        label=label,
    )


def _scatter_centroids(
    ax: plt.Axes,
    ref_2d: np.ndarray,
    research_centroids_2d: np.ndarray,
    policy_centroids_2d: np.ndarray,
    research_available: np.ndarray,
    policy_available: np.ndarray,
    *,
    draw_numbers: bool = True,
) -> None:
    for sdg_idx in range(N_SDG):
        if research_available[sdg_idx] and policy_available[sdg_idx]:
            ax.plot(
                [research_centroids_2d[sdg_idx, 0], policy_centroids_2d[sdg_idx, 0]],
                [research_centroids_2d[sdg_idx, 1], policy_centroids_2d[sdg_idx, 1]],
                color=GAP_LINE_COLOR,
                linewidth=0.8,
                alpha=0.75,
                zorder=4,
            )

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
        research_centroids_2d[research_available, 0],
        research_centroids_2d[research_available, 1],
        s=95,
        c=RESEARCH_COLOR,
        marker="o",
        edgecolors="white",
        linewidths=0.6,
        zorder=7,
        label="Research SDG centroid",
    )
    ax.scatter(
        policy_centroids_2d[policy_available, 0],
        policy_centroids_2d[policy_available, 1],
        s=95,
        c=POLICY_COLOR,
        marker="s",
        edgecolors="white",
        linewidths=0.6,
        zorder=7,
        label="Policy SDG centroid",
    )
    if draw_numbers:
        draw_marker_numbers(
            ax, ref_2d, None,
            text_color="white", fontsize=7.2,
            stroke_color="#0A2E15", stroke_width=1.1, zorder=8,
        )
        draw_marker_numbers(
            ax, research_centroids_2d, research_available,
            text_color="white", fontsize=6.4,
            stroke_color="#0B223F", stroke_width=1.0, zorder=9,
        )
        draw_marker_numbers(
            ax, policy_centroids_2d, policy_available,
            text_color="white", fontsize=6.4,
            stroke_color="#5A2016", stroke_width=1.0, zorder=9,
        )


def write_metadata_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_num_tex(path: Path, payload: dict) -> None:
    lines = [
        "% Auto-generated by 1_code/7_main_analysis/1_main_text/0_pca_register_before_after.py — do not edit manually",
        rf"\newcommand{{\PcaRegBefTotalResearch}}{{{latex_int(int(payload['total_research_embeddings_available']))}}}",
        rf"\newcommand{{\PcaRegBefTotalPolicy}}{{{latex_int(int(payload['total_policy_embeddings_available']))}}}",
        rf"\newcommand{{\PcaRegBefResearchFitN}}{{{latex_int(int(payload['n_research_used_for_pca_fit']))}}}",
        rf"\newcommand{{\PcaRegBefPolicyFitN}}{{{latex_int(int(payload['n_policy_used_for_pca_fit']))}}}",
        rf"\newcommand{{\PcaRegBefPcOnePct}}{{{payload['pc1_explained_variance_ratio'] * 100:.1f}}}",
        rf"\newcommand{{\PcaRegBefPcTwoPct}}{{{payload['pc2_explained_variance_ratio'] * 100:.1f}}}",
        rf"\newcommand{{\PcaRegBefNGDirs}}{{{payload['n_inlp_directions']}}}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    _POLICY_EMB = semantic_gap_shared.get_policy_emb(args.embed_model)
    _POLICY_IDS = semantic_gap_shared.get_policy_ids(args.embed_model)
    _POLICY_SCORES = semantic_gap_shared.get_policy_scores(args.embed_model)
    _RESEARCH_CENTROIDS = semantic_gap_shared.get_research_centroids(args.embed_model)
    _RESEARCH_CENTROID_META = semantic_gap_shared.get_research_centroid_meta(args.embed_model)

    out_root = output_dir_for_model(args.embed_model, root=Path(args.output_dir))
    data_dir = out_root / "data"
    tables_dir = out_root / "tables"
    figures_dir = out_root / "figures"
    for d in (data_dir, tables_dir, figures_dir):
        d.mkdir(parents=True, exist_ok=True)

    PRIMARY = data_dir / METADATA_JSON
    OUTPUTS = [
        PRIMARY,
        figures_dir / f"{FIG_NAME}.pdf",
        figures_dir / f"{FIG_NAME}.png",
        tables_dir / NUM_TEX,
    ]

    G = load_G(args.embed_model)
    fp = (
        fingerprint_of(
            _POLICY_EMB, _POLICY_IDS, _POLICY_SCORES,
            _RESEARCH_CENTROIDS, _RESEARCH_CENTROID_META,
            embed_dir_for_model(args.embed_model) / "research_shards" / "metadata" / "manifest.json",
            scored_dir_for_model(args.embed_model) / "sdg_centroids.npy",
            register_utils.register_dir(args.embed_model) / "G.npy",
        )
        + SCRIPT_VERSION
    )
    if should_skip(OUTPUTS, fp, args.overwrite, PRIMARY):
        log.info("Skipping %s — inputs unchanged", PRIMARY)
        return

    rng = np.random.default_rng(args.seed)

    log.info("Output dir: %s", out_root)
    log.info("Loading policy embeddings: %s", _POLICY_EMB)
    policy_emb = np.load(_POLICY_EMB).astype(np.float32)
    verify_unit_norms(policy_emb, "policy embeddings")
    policy_scores = np.load(_POLICY_SCORES).astype(np.float32)
    policy_ids = load_json(_POLICY_IDS)
    if policy_emb.shape[0] != policy_scores.shape[0] or policy_emb.shape[0] != len(policy_ids):
        raise RuntimeError(
            f"Policy alignment mismatch: emb={policy_emb.shape[0]} "
            f"scores={policy_scores.shape[0]} ids={len(policy_ids)}"
        )

    # --- Load research sample ---
    n_policy_total = int(policy_emb.shape[0])
    policy_fit_indices = fit_policy_sample_indices(n_policy_total, args.policy_fit_cap, rng)
    policy_fit_emb = policy_emb[policy_fit_indices]

    embed_dir = embed_dir_for_model(args.embed_model)
    scored_dir = scored_dir_for_model(args.embed_model)
    sdg_centroids_path = scored_dir / "sdg_centroids.npy"
    research_manifest = embed_research_dir_for_model(args.embed_model) / "metadata" / "manifest.json"
    total_research = total_research_embedding_rows(research_manifest, embed_dir)
    n_research_fit = int(policy_fit_emb.shape[0])
    if total_research < n_research_fit:
        raise RuntimeError(
            f"Research embedding universe smaller than policy fit sample: {total_research} < {n_research_fit}"
        )

    research_sample_indices = np.sort(
        rng.choice(total_research, size=n_research_fit, replace=False).astype(np.int64)
    )
    log.info(
        "Balanced fit sample: policy=%d, research=%d (total research=%d)",
        policy_fit_emb.shape[0],
        research_sample_indices.size,
        total_research,
    )
    research_fit_emb = load_sampled_research_embeddings(research_manifest, research_sample_indices, embed_dir)
    verify_unit_norms(research_fit_emb, "sampled research embeddings")

    # --- Project adjusted copies ---
    log.info("Projecting adjusted embeddings through G (K=%d)", G.shape[0])
    policy_fit_adj = project(policy_fit_emb, G)
    research_fit_adj = project(research_fit_emb, G)
    policy_bg_adj = project(policy_emb, G)

    # --- Fit PCA on RAW data (panel a) ---
    fit_matrix = np.concatenate([policy_fit_emb, research_fit_emb], axis=0).astype(np.float32)
    log.info("PCA fit matrix shape: %s", fit_matrix.shape)
    pca = PCA(n_components=args.n_components, random_state=args.seed)
    pca.fit(fit_matrix)
    evr = pca.explained_variance_ratio_.astype(float)
    log.info("PCA explained variance (raw): PC1=%.4f, PC2=%.4f", evr[0], evr[1])

    # --- Fit PCA on ADJUSTED data (panel b) ---
    fit_adj_matrix = np.concatenate([policy_fit_adj, research_fit_adj], axis=0).astype(np.float32)
    log.info("Adjusted PCA fit matrix shape: %s", fit_adj_matrix.shape)
    pca_adj = PCA(n_components=args.n_components, random_state=args.seed)
    pca_adj.fit(fit_adj_matrix)
    evr_adj = pca_adj.explained_variance_ratio_.astype(float)
    log.info("PCA explained variance (adjusted): PC1=%.4f, PC2=%.4f", evr_adj[0], evr_adj[1])

    # --- Project raw clouds into raw PCA ---
    policy_bg_raw = pca.transform(policy_fit_emb)
    research_bg_raw = pca.transform(research_fit_emb)

    # --- Project adjusted clouds into adjusted PCA ---
    policy_bg_adj_2d = pca_adj.transform(policy_bg_adj)
    research_bg_adj_2d = pca_adj.transform(research_fit_adj)

    # --- Centroids: raw and adjusted ---
    sdg_centroids = np.load(sdg_centroids_path).astype(np.float32)
    research_centroids = np.load(_RESEARCH_CENTROIDS).astype(np.float32)
    research_meta = load_json(_RESEARCH_CENTROID_META)
    verify_unit_norms(sdg_centroids, "sdg centroids", n_sample=17)
    verify_unit_norms(research_centroids, "research centroids", n_sample=17)

    policy_centroids_raw, policy_centroid_available = build_policy_centroids(
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

    # Adjusted centroids: project through G then compute centroids in adjusted space.
    # NOTE: build_policy_centroids reseeds per SDG (rng_seed + sdg_idx), so
    # passing the SAME args.seed as the raw call above means raw and adjusted
    # policy centroids are built from the IDENTICAL capped segment sets — the
    # raw-vs-adjusted movement in the figure is purely the G projection, not
    # sampling noise. Keep the two seeds in sync.
    policy_emb_adj_full = project(policy_emb, G)
    policy_centroids_adj, _ = build_policy_centroids(
        policy_emb=policy_emb_adj_full,
        policy_scores=policy_scores,
        policy_ids=policy_ids,
        segment_cap=SEGMENT_CAP_PRIMARY,
        rng_seed=args.seed,
    )
    research_centroids_adj = project(research_centroids, G)
    ref_adj = project(sdg_centroids, G)

    # Project all centroids into PCA space
    ref_2d = pca.transform(sdg_centroids)
    ref_adj_2d = pca_adj.transform(ref_adj)
    research_centroids_raw_2d = pca.transform(research_centroids)
    research_centroids_adj_2d = pca_adj.transform(research_centroids_adj)
    policy_centroids_raw_2d = pca.transform(policy_centroids_raw)
    policy_centroids_adj_2d = pca_adj.transform(policy_centroids_adj)

    # --- Draw two-panel figure ---
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "serif"],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 7.5,
            "figure.dpi": 150,
        }
    )
    fig, (ax_raw, ax_adj) = plt.subplots(1, 2, figsize=(14, 5.5))

    # --- Left panel: RAW ---
    _scatter_cloud(ax_raw, research_bg_raw, RESEARCH_COLOR, "Research sample")
    _scatter_cloud(ax_raw, policy_bg_raw, POLICY_COLOR, "Policy segments")
    _scatter_centroids(
        ax_raw, ref_2d, research_centroids_raw_2d, policy_centroids_raw_2d,
        research_centroid_available, policy_centroid_available,
    )
    ax_raw.set_title("(a) Raw embeddings", fontweight="bold")
    ax_raw.set_xlabel(f"PC1 ({evr[0] * 100:.1f}% variance)")
    ax_raw.set_ylabel(f"PC2 ({evr[1] * 100:.1f}% variance)")
    ax_raw.spines["top"].set_visible(False)
    ax_raw.spines["right"].set_visible(False)
    ax_raw.axhline(0.0, color="#cccccc", lw=0.6, zorder=1)
    ax_raw.axvline(0.0, color="#cccccc", lw=0.6, zorder=1)

    # --- Right panel: ADJUSTED ---
    _scatter_cloud(ax_adj, research_bg_adj_2d, RESEARCH_COLOR, "Research sample")
    _scatter_cloud(ax_adj, policy_bg_adj_2d, POLICY_COLOR, "Policy segments")
    _scatter_centroids(
        ax_adj, ref_adj_2d, research_centroids_adj_2d, policy_centroids_adj_2d,
        research_centroid_available, policy_centroid_available,
    )
    ax_adj.set_title("(b) Adjusted embeddings (register removed)", fontweight="bold")
    ax_adj.set_xlabel(f"PC1 ({evr_adj[0] * 100:.1f}% variance)")
    ax_adj.set_ylabel(f"PC2 ({evr_adj[1] * 100:.1f}% variance)")
    ax_adj.spines["top"].set_visible(False)
    ax_adj.spines["right"].set_visible(False)
    ax_adj.axhline(0.0, color="#cccccc", lw=0.6, zorder=1)
    ax_adj.axvline(0.0, color="#cccccc", lw=0.6, zorder=1)

    # Independent axis limits per panel
    for ax, x_data, y_data in [
        (ax_raw, np.concatenate([policy_bg_raw[:, 0], research_bg_raw[:, 0]]),
                np.concatenate([policy_bg_raw[:, 1], research_bg_raw[:, 1]])),
        (ax_adj, np.concatenate([policy_bg_adj_2d[:, 0], research_bg_adj_2d[:, 0]]),
                 np.concatenate([policy_bg_adj_2d[:, 1], research_bg_adj_2d[:, 1]])),
    ]:
        x_m = (x_data.max() - x_data.min()) * 0.05
        y_m = (y_data.max() - y_data.min()) * 0.05
        ax.set_xlim(x_data.min() - x_m, x_data.max() + x_m)
        ax.set_ylim(y_data.min() - y_m, y_data.max() + y_m)

    # Shared legend (right panel only to save space)
    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=RESEARCH_COLOR, alpha=0.55, markersize=5, label="Research sample"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=POLICY_COLOR, alpha=0.55, markersize=5, label="Policy segments"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor=REFERENCE_COLOR, markeredgecolor="white", markersize=7, label="SDG reference centroid"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=RESEARCH_COLOR, markeredgecolor="white", markersize=6, label="Research SDG centroid"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=POLICY_COLOR, markeredgecolor="white", markersize=6, label="Policy SDG centroid"),
        Line2D([0], [0], color=GAP_LINE_COLOR, lw=1.0, label="Research-policy gap line"),
    ]
    ax_adj.legend(handles=legend_handles, loc="best", frameon=False)

    fig.tight_layout(rect=(0, 0, 1, 1))

    pdf_path = figures_dir / f"{FIG_NAME}.pdf"
    png_path = figures_dir / f"{FIG_NAME}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    log.info("Saved: %s", pdf_path)
    log.info("Saved: %s", png_path)

    # --- Metadata + macros ---
    metadata = {
        "random_seed": args.seed,
        "balanced_sampling_used": True,
        "total_research_embeddings_available": int(total_research),
        "total_policy_embeddings_available": int(n_policy_total),
        "n_research_used_for_pca_fit": int(research_fit_emb.shape[0]),
        "n_policy_used_for_pca_fit": int(policy_fit_emb.shape[0]),
        "n_research_projected_background": int(research_bg_raw.shape[0]),
        "n_policy_projected_background": int(policy_bg_raw.shape[0]),
        "policy_capped": bool(args.policy_fit_cap > 0 and args.policy_fit_cap < n_policy_total),
        "policy_fit_cap": int(args.policy_fit_cap),
        "pc1_explained_variance_ratio": float(evr[0]),
        "pc2_explained_variance_ratio": float(evr[1]),
        "pc1_explained_variance_ratio_adj": float(evr_adj[0]),
        "pc2_explained_variance_ratio_adj": float(evr_adj[1]),
        "n_inlp_directions": int(G.shape[0]),
        "pca_fitted_on": "independent: raw PCA for panel (a), adjusted PCA for panel (b)",
        "segment_cap_for_policy_gap_overlay": int(SEGMENT_CAP_PRIMARY),
        "note": (
            "PCA is visual-only. Main coverage and semantic-gap results remain "
            "computed in the original embedding space."
        ),
    }
    metadata_path = data_dir / METADATA_JSON
    write_metadata_json(metadata_path, metadata)
    write_num_tex(tables_dir / NUM_TEX, metadata)
    log.info("Saved: %s", metadata_path)
    log.info("Saved: %s", tables_dir / NUM_TEX)
    record_fingerprint(OUTPUTS, fp, PRIMARY)


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()

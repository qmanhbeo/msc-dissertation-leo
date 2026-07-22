"""
Diagnose within-corpus SDG centroid structure for the research and policy corpora.

This stage is descriptive only. It assesses whether the current hard-assignment and
centroid method produces coherent within-corpus SDG structure, without altering any
main alignment metrics.

Outputs:
  4_outputs/appendix/b1_within_corpus_centroid/figures/fig_b1_research_sdg_pca.pdf
  4_outputs/appendix/b1_within_corpus_centroid/figures/fig_b1_research_sdg_pca.png
  4_outputs/appendix/b1_within_corpus_centroid/figures/fig_b1_policy_sdg_pca.pdf
  4_outputs/appendix/b1_within_corpus_centroid/figures/fig_b1_policy_sdg_pca.png
  4_outputs/appendix/b1_within_corpus_centroid/data/b1_within_corpus_metrics.csv
  4_outputs/appendix/b1_within_corpus_centroid/data/b1_within_corpus_summary.json
  4_outputs/appendix/b1_within_corpus_centroid/tables/num_b1_within_corpus_centroid.tex

Run from project root:
    python 1_code/3_main_analysis/3_appendix/1_within_corpus_centroid_structure.py
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-dissertation")
import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import IncrementalPCA, PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.metrics.cluster import contingency_matrix

ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = ROOT / "1_code"
ANALYSIS_ROOT = Path(__file__).resolve().parents[1]
SHARED_DIR = ANALYSIS_ROOT / "0_shared"
for path in (CODE_ROOT, SHARED_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


from alignment_core import verify_unit_norms
from model_utils import DEFAULT_EMBED_MODEL, DEFAULT_OUTPUT_ROOT, embed_dir_for_model, scored_dir_for_model
from research_embedding_shards import (
    iter_research_embedding_shards,
    load_json as load_embedding_manifest_json,
    load_sampled_research_embeddings,
    total_research_embedding_rows,
)
from research_score_shards import load_json as load_score_manifest_json
from research_score_shards import resolve_from_manifest as resolve_score_manifest_path
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

RESEARCH_FIG_PDF = "fig_b1_research_sdg_pca.pdf"
RESEARCH_FIG_PNG = "fig_b1_research_sdg_pca.png"
POLICY_FIG_PDF = "fig_b1_policy_sdg_pca.pdf"
POLICY_FIG_PNG = "fig_b1_policy_sdg_pca.png"
METRICS_CSV = "b1_within_corpus_metrics.csv"
SUMMARY_JSON = "b1_within_corpus_summary.json"
NUM_TEX = "num_b1_within_corpus_centroid.tex"

RESEARCH_COLOR = "#2166AC"
POLICY_COLOR = "#D6604D"
BACKGROUND_GREY = "#7F7F7F"

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ScoredResearchShard:
    shard_id: int
    name: str
    rows: int
    ids_path: Path


@dataclass
class SdqMetricAccumulator:
    own_segments: list[np.ndarray] = field(default_factory=list)
    margin_segments: list[np.ndarray] = field(default_factory=list)
    second_best_sum: float = 0.0
    competitor_counts: np.ndarray = field(default_factory=lambda: np.zeros(N_SDG, dtype=np.int64))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate within-corpus SDG centroid structure diagnostics.")
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_ROOT))
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    p.add_argument("--research-plot-sample-size", type=int, default=100_000)
    p.add_argument("--policy-plot-sample-size", type=int, default=0, help="0 means use all policy segments.")
    p.add_argument("--silhouette-sample-size", type=int, default=10_000)
    p.add_argument("--research-kmeans-sample-size", type=int, default=100_000)
    p.add_argument("--policy-kmeans-sample-size", type=int, default=0, help="0 means use all policy segments.")
    p.add_argument("--research-pca-batch-size", type=int, default=16_384)
    p.add_argument("--model", default=DEFAULT_EMBED_MODEL, help=argparse.SUPPRESS)
    return p.parse_args()


def validate_research_centroid_order(meta_path: Path, centroids: np.ndarray) -> list[dict]:
    meta = load_json(meta_path)
    if len(meta) != N_SDG or centroids.shape[0] != N_SDG:
        raise RuntimeError(
            f"Expected {N_SDG} research centroid rows and metadata entries, got {centroids.shape[0]} and {len(meta)}"
        )
    for idx, row in enumerate(meta):
        if int(row["sdg"]) != idx + 1:
            raise RuntimeError(f"Research centroid metadata out of order at row {idx}: {row}")
    return meta


def load_scored_research_shards(scored_dir: Path) -> dict[int, ScoredResearchShard]:
    manifest_path = scored_dir / "paper_scores_shards" / "metadata" / "manifest.json"
    manifest = load_score_manifest_json(manifest_path)
    out: dict[int, ScoredResearchShard] = {}
    for shard in manifest.get("shards", []):
        shard_id = int(shard["shard_id"])
        out[shard_id] = ScoredResearchShard(
            shard_id=shard_id,
            name=str(shard["name"]),
            rows=int(shard["rows"]),
            ids_path=resolve_score_manifest_path(manifest_path, shard["ids_path"], scored_dir),
        )
    return out


def load_assigned_sdg_array(ids_path: Path, expected_rows: int) -> np.ndarray:
    labels = np.empty(expected_rows, dtype=np.int16)
    count = 0
    with ids_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            labels[count] = int(row["assigned_sdg"]) - 1
            count += 1
    if count != expected_rows:
        raise RuntimeError(f"Assigned-SDG row mismatch for {ids_path}: expected {expected_rows}, got {count}")
    return labels


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
            log.warning("Policy SDG %2d: centroid unavailable after segment cap", sdg_idx + 1)
            continue
        centroids[sdg_idx] = centroid
        available[sdg_idx] = True
    return centroids, available


def sample_indices(total_rows: int, sample_size: int, seed: int) -> np.ndarray:
    n = total_rows if sample_size <= 0 else min(total_rows, sample_size)
    if n <= 0:
        raise RuntimeError(f"Cannot sample from empty population: total_rows={total_rows}")
    rng = np.random.default_rng(seed)
    if n == total_rows:
        return np.arange(total_rows, dtype=np.int64)
    return np.sort(rng.choice(total_rows, size=n, replace=False).astype(np.int64))


def load_sampled_research_assignments(
    sampled_global_indices: np.ndarray,
    score_shards: dict[int, ScoredResearchShard],
    embed_dir: Path,
) -> np.ndarray:
    if sampled_global_indices.ndim != 1 or sampled_global_indices.size == 0:
        raise ValueError("sampled_global_indices must be a non-empty 1D array")
    manifest_path = embed_dir / "research_shards" / "metadata" / "manifest.json"
    parts: list[np.ndarray] = []
    for shard in iter_research_embedding_shards(manifest_path, embed_dir):
        left = int(np.searchsorted(sampled_global_indices, shard.start, side="left"))
        right = int(np.searchsorted(sampled_global_indices, shard.stop, side="left"))
        if right <= left:
            continue
        score_shard = score_shards.get(shard.shard_id)
        if score_shard is None:
            raise RuntimeError(f"Missing scored-shard metadata for research shard {shard.shard_id}")
        labels = load_assigned_sdg_array(score_shard.ids_path, shard.rows)
        local_indices = sampled_global_indices[left:right] - shard.start
        parts.append(labels[local_indices])
    result = np.concatenate(parts, axis=0)
    if result.shape[0] != sampled_global_indices.size:
        raise RuntimeError(
            f"Sampled research assignment mismatch: expected {sampled_global_indices.size}, got {result.shape[0]}"
        )
    return result


def fit_incremental_research_pca(manifest_path: Path, batch_size: int, embed_dir: Path) -> IncrementalPCA:
    pca = IncrementalPCA(n_components=2, batch_size=batch_size)
    total_rows = 0
    for shard in iter_research_embedding_shards(manifest_path, embed_dir):
        log.info("Research PCA partial_fit on shard %s (%d rows)", shard.name, shard.rows)
        emb = np.load(shard.embedding_path, mmap_mode="r")
        for start in range(0, shard.rows, batch_size):
            stop = min(start + batch_size, shard.rows)
            batch = np.asarray(emb[start:stop], dtype=np.float32)
            pca.partial_fit(batch)
            total_rows += batch.shape[0]
    if total_rows <= 0:
        raise RuntimeError("Incremental research PCA received zero rows.")
    return pca


def init_metric_accumulators() -> list[SdqMetricAccumulator]:
    return [SdqMetricAccumulator() for _ in range(N_SDG)]


def update_metric_accumulators(
    accumulators: list[SdqMetricAccumulator],
    emb_batch: np.ndarray,
    assigned_batch: np.ndarray,
    centroids: np.ndarray,
    centroid_available: np.ndarray,
    *,
    corpus_name: str,
) -> None:
    valid_mask = centroid_available[assigned_batch]
    if not np.all(valid_mask):
        n_invalid = int((~valid_mask).sum())
        log.warning("%s: skipping %d rows assigned to unavailable centroids", corpus_name, n_invalid)
        emb_batch = emb_batch[valid_mask]
        assigned_batch = assigned_batch[valid_mask]
        if emb_batch.shape[0] == 0:
            return

    sims = emb_batch @ centroids.T
    unavailable_cols = ~centroid_available
    if np.any(unavailable_cols):
        sims[:, unavailable_cols] = -np.inf
    row_idx = np.arange(emb_batch.shape[0])
    own = sims[row_idx, assigned_batch].astype(np.float32)
    sims[row_idx, assigned_batch] = -np.inf
    second_best_idx = sims.argmax(axis=1).astype(np.int16)
    second_best = sims[row_idx, second_best_idx].astype(np.float32)
    margin = (own - second_best).astype(np.float32)

    for sdg_idx in range(N_SDG):
        mask = assigned_batch == sdg_idx
        if not np.any(mask):
            continue
        acc = accumulators[sdg_idx]
        acc.own_segments.append(own[mask])
        acc.margin_segments.append(margin[mask])
        acc.second_best_sum += float(second_best[mask].sum())
        acc.competitor_counts += np.bincount(second_best_idx[mask], minlength=N_SDG)


def summarise_metric_rows(
    corpus_name: str,
    accumulators: list[SdqMetricAccumulator],
    counts: np.ndarray,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for sdg_idx in range(N_SDG):
        n = int(counts[sdg_idx])
        if n == 0:
            rows.append(
                {
                    "corpus": corpus_name,
                    "sdg": sdg_idx + 1,
                    "n_assigned": 0,
                    "mean_cosine_to_assigned_centroid": None,
                    "median_cosine_to_assigned_centroid": None,
                    "std_cosine_to_assigned_centroid": None,
                    "mean_assignment_margin": None,
                    "median_assignment_margin": None,
                    "share_margin_gt_0_01": None,
                    "share_margin_gt_0_05": None,
                    "nearest_competing_sdg_centroid": None,
                    "mean_similarity_to_nearest_competing_centroid": None,
                    "separation_gap": None,
                }
            )
            continue

        acc = accumulators[sdg_idx]
        own = np.concatenate(acc.own_segments) if acc.own_segments else np.empty(0, dtype=np.float32)
        margin = np.concatenate(acc.margin_segments) if acc.margin_segments else np.empty(0, dtype=np.float32)
        if own.size != n or margin.size != n:
            raise RuntimeError(
                f"{corpus_name} SDG {sdg_idx + 1}: metric accumulation mismatch own={own.size} margin={margin.size} n={n}"
            )
        nearest_comp = int(acc.competitor_counts.argmax()) + 1
        mean_second_best = float(acc.second_best_sum / max(n, 1))
        mean_own = float(own.mean())
        rows.append(
            {
                "corpus": corpus_name,
                "sdg": sdg_idx + 1,
                "n_assigned": n,
                "mean_cosine_to_assigned_centroid": round(mean_own, 6),
                "median_cosine_to_assigned_centroid": round(float(np.median(own)), 6),
                "std_cosine_to_assigned_centroid": round(float(own.std(ddof=0)), 6),
                "mean_assignment_margin": round(float(margin.mean()), 6),
                "median_assignment_margin": round(float(np.median(margin)), 6),
                "share_margin_gt_0_01": round(float((margin > 0.01).mean()), 6),
                "share_margin_gt_0_05": round(float((margin > 0.05).mean()), 6),
                "nearest_competing_sdg_centroid": nearest_comp,
                "mean_similarity_to_nearest_competing_centroid": round(mean_second_best, 6),
                "separation_gap": round(mean_own - mean_second_best, 6),
            }
        )
    return rows


def compute_clustering_metrics(
    emb_for_silhouette: np.ndarray,
    labels_for_silhouette: np.ndarray,
    pca_points_for_dbch: np.ndarray,
    labels_for_dbch: np.ndarray,
    emb_for_kmeans: np.ndarray,
    labels_for_kmeans: np.ndarray,
    *,
    random_seed: int,
) -> dict[str, object]:
    metrics: dict[str, object] = {}

    if np.unique(labels_for_silhouette).size >= 2:
        metrics["silhouette_cosine"] = float(
            silhouette_score(emb_for_silhouette, labels_for_silhouette, metric="cosine")
        )
    else:
        metrics["silhouette_cosine"] = None

    if np.unique(labels_for_dbch).size >= 2:
        metrics["davies_bouldin_pca2d"] = float(davies_bouldin_score(pca_points_for_dbch, labels_for_dbch))
        metrics["calinski_harabasz_pca2d"] = float(calinski_harabasz_score(pca_points_for_dbch, labels_for_dbch))
    else:
        metrics["davies_bouldin_pca2d"] = None
        metrics["calinski_harabasz_pca2d"] = None

    kmeans = MiniBatchKMeans(
        n_clusters=N_SDG,
        random_state=random_seed,
        batch_size=min(4096, max(256, emb_for_kmeans.shape[0])),
        n_init=10,
    )
    cluster_labels = kmeans.fit_predict(emb_for_kmeans)
    metrics["kmeans_adjusted_rand_index"] = float(adjusted_rand_score(labels_for_kmeans, cluster_labels))
    metrics["kmeans_normalized_mutual_info"] = float(
        normalized_mutual_info_score(labels_for_kmeans, cluster_labels)
    )
    contingency = contingency_matrix(labels_for_kmeans, cluster_labels)
    metrics["kmeans_cluster_purity"] = float(contingency.max(axis=0).sum() / contingency.sum())
    metrics["kmeans_contingency"] = contingency.astype(int).tolist()
    return metrics


def plot_within_corpus_pca(
    points_2d: np.ndarray,
    centroids_2d: np.ndarray,
    centroid_available: np.ndarray,
    *,
    point_label: str,
    point_color: str,
    centroid_marker: str,
    centroid_label: str,
    centroid_color: str,
    title: str,
    subtitle_note: str,
    evr: np.ndarray,
    output_pdf: Path,
    output_png: Path,
) -> None:
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
    fig, ax = plt.subplots(figsize=(7.8, 6.2))

    ax.scatter(
        points_2d[:, 0],
        points_2d[:, 1],
        s=4,
        c=BACKGROUND_GREY,
        alpha=0.08,
        linewidths=0,
        rasterized=True,
        label=point_label,
    )
    ax.scatter(
        centroids_2d[centroid_available, 0],
        centroids_2d[centroid_available, 1],
        s=130,
        c=centroid_color,
        marker=centroid_marker,
        edgecolors="white",
        linewidths=0.8,
        zorder=5,
        label=centroid_label,
    )
    for sdg_idx in range(N_SDG):
        if not bool(centroid_available[sdg_idx]):
            continue
        x, y = centroids_2d[sdg_idx]
        ax.text(
            float(x),
            float(y),
            str(sdg_idx + 1),
            ha="center",
            va="center",
            color="white",
            fontsize=7.0,
            fontweight="bold",
            zorder=6,
            path_effects=[pe.withStroke(linewidth=1.0, foreground=point_color)],
        )

    ax.set_xlabel(f"PC1 ({evr[0] * 100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({evr[1] * 100:.1f}% variance)")
    ax.set_title(title + "\n" + subtitle_note, loc="left", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.axhline(0.0, color="#cccccc", lw=0.6, zorder=1)
    ax.axvline(0.0, color="#cccccc", lw=0.6, zorder=1)
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=BACKGROUND_GREY, alpha=0.55, markersize=5, label=point_label),
        Line2D([0], [0], marker=centroid_marker, color="none", markerfacecolor=centroid_color, markeredgecolor="white", markersize=8, label=centroid_label),
    ]
    ax.legend(handles=handles, loc="best", frameon=False)
    fig.text(
        0.01,
        0.01,
        "PCA is fitted separately within this corpus and is used only to visualise internal SDG structure.",
        ha="left",
        va="bottom",
        fontsize=7,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(output_pdf, bbox_inches="tight")
    fig.savefig(output_png, bbox_inches="tight", dpi=150)
    plt.close(fig)


def write_metrics_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "corpus",
        "sdg",
        "n_assigned",
        "mean_cosine_to_assigned_centroid",
        "median_cosine_to_assigned_centroid",
        "std_cosine_to_assigned_centroid",
        "mean_assignment_margin",
        "median_assignment_margin",
        "share_margin_gt_0_01",
        "share_margin_gt_0_05",
        "nearest_competing_sdg_centroid",
        "mean_similarity_to_nearest_competing_centroid",
        "separation_gap",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_num_tex(path: Path, summary: dict[str, object]) -> None:
    research = summary["research"]
    policy = summary["policy"]
    def latex_int(v: int) -> str:
        return f"{v:,}".replace(",", "{,}")
    lines = [
        "% Auto-generated by 1_code/3_main_analysis/3_appendix/1_within_corpus_centroid_structure.py — do not edit manually",
        rf"\newcommand{{\WithinCorpusResearchPlotN}}{{{latex_int(int(research['plotted_rows']))}}}",
        rf"\newcommand{{\WithinCorpusPolicyPlotN}}{{{latex_int(int(policy['plotted_rows']))}}}",
        rf"\newcommand{{\WithinCorpusResearchSilhouette}}{{{research['global_metrics']['silhouette_cosine_sample']:.3f}}}",
        rf"\newcommand{{\WithinCorpusPolicySilhouette}}{{{policy['global_metrics']['silhouette_cosine_sample']:.3f}}}",
        rf"\newcommand{{\WithinCorpusResearchPcOnePct}}{{{research['pc1_explained_variance_ratio'] * 100:.1f}}}",
        rf"\newcommand{{\WithinCorpusResearchPcTwoPct}}{{{research['pc2_explained_variance_ratio'] * 100:.1f}}}",
        rf"\newcommand{{\WithinCorpusPolicyPcOnePct}}{{{policy['pc1_explained_variance_ratio'] * 100:.1f}}}",
        rf"\newcommand{{\WithinCorpusPolicyPcTwoPct}}{{{policy['pc2_explained_variance_ratio'] * 100:.1f}}}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    _POLICY_EMB = semantic_gap_shared.get_policy_emb(args.model)
    _POLICY_IDS = semantic_gap_shared.get_policy_ids(args.model)
    _POLICY_SCORES = semantic_gap_shared.get_policy_scores(args.model)
    _RESEARCH_CENTROIDS = semantic_gap_shared.get_research_centroids(args.model)
    _RESEARCH_CENTROID_META = semantic_gap_shared.get_research_centroid_meta(args.model)
    out_root = Path(args.output_dir) / "appendix" / "b1_within_corpus_centroid"
    data_dir = out_root / "data"
    tables_dir = out_root / "tables"
    figures_dir = out_root / "figures"
    for d in (data_dir, tables_dir, figures_dir):
        d.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    embed_dir = embed_dir_for_model(args.model)
    scored_dir = scored_dir_for_model(args.model)
    research_manifest = embed_dir / "research_shards" / "metadata" / "manifest.json"

    research_centroids = np.load(_RESEARCH_CENTROIDS).astype(np.float32)
    verify_unit_norms(research_centroids, "research centroids")
    research_meta = validate_research_centroid_order(_RESEARCH_CENTROID_META, research_centroids)
    research_centroid_available = np.array(
        [not bool(row.get("zero_flag", False)) for row in research_meta],
        dtype=bool,
    )

    total_research = total_research_embedding_rows(research_manifest, embed_dir)
    scored_research_shards = load_scored_research_shards(scored_dir)

    log.info("Fitting research-only IncrementalPCA on %d rows", total_research)
    research_pca = fit_incremental_research_pca(research_manifest, args.research_pca_batch_size, embed_dir)
    research_evr = np.asarray(research_pca.explained_variance_ratio_, dtype=float)
    log.info("Research PCA explained variance ratio: PC1=%.4f, PC2=%.4f", research_evr[0], research_evr[1])

    research_plot_indices = sample_indices(total_research, args.research_plot_sample_size, args.seed)
    research_plot_emb = load_sampled_research_embeddings(research_manifest, research_plot_indices, embed_dir)
    verify_unit_norms(research_plot_emb, "research plot sample embeddings")
    research_plot_labels = load_sampled_research_assignments(research_plot_indices, scored_research_shards, embed_dir)
    research_plot_2d = research_pca.transform(research_plot_emb)
    research_centroids_2d = research_pca.transform(research_centroids)

    research_accumulators = init_metric_accumulators()
    research_counts = np.zeros(N_SDG, dtype=np.int64)
    for shard in iter_research_embedding_shards(research_manifest, embed_dir):
        score_shard = scored_research_shards.get(shard.shard_id)
        if score_shard is None:
            raise RuntimeError(f"Missing scored research shard for shard_id={shard.shard_id}")
        if score_shard.rows != shard.rows:
            raise RuntimeError(
                f"Research shard row mismatch for shard {shard.name}: emb={shard.rows} scores={score_shard.rows}"
            )
        labels = load_assigned_sdg_array(score_shard.ids_path, shard.rows)
        research_counts += np.bincount(labels, minlength=N_SDG)
        emb = np.load(shard.embedding_path, mmap_mode="r")
        for start in range(0, shard.rows, args.research_pca_batch_size):
            stop = min(start + args.research_pca_batch_size, shard.rows)
            batch_emb = np.asarray(emb[start:stop], dtype=np.float32)
            batch_labels = labels[start:stop]
            update_metric_accumulators(
                research_accumulators,
                batch_emb,
                batch_labels,
                research_centroids,
                research_centroid_available,
                corpus_name="research",
            )
    research_metric_rows = summarise_metric_rows("research", research_accumulators, research_counts)

    research_silhouette_indices = sample_indices(
        total_research,
        min(args.silhouette_sample_size, total_research),
        args.seed + 1,
    )
    research_silhouette_emb = load_sampled_research_embeddings(research_manifest, research_silhouette_indices, embed_dir)
    research_silhouette_labels = load_sampled_research_assignments(research_silhouette_indices, scored_research_shards, embed_dir)

    research_kmeans_indices = sample_indices(
        total_research,
        min(args.research_kmeans_sample_size, total_research),
        args.seed + 2,
    )
    research_kmeans_emb = load_sampled_research_embeddings(research_manifest, research_kmeans_indices, embed_dir)
    research_kmeans_labels = load_sampled_research_assignments(research_kmeans_indices, scored_research_shards, embed_dir)
    research_global_metrics = compute_clustering_metrics(
        research_silhouette_emb,
        research_silhouette_labels,
        research_plot_2d,
        research_plot_labels,
        research_kmeans_emb,
        research_kmeans_labels,
        random_seed=args.seed,
    )
    research_global_metrics["silhouette_cosine_sample"] = research_global_metrics.pop("silhouette_cosine")
    research_global_metrics["silhouette_sample_size"] = int(research_silhouette_emb.shape[0])
    research_global_metrics["kmeans_sample_size"] = int(research_kmeans_emb.shape[0])

    policy_emb = np.load(_POLICY_EMB).astype(np.float32)
    verify_unit_norms(policy_emb, "policy embeddings")
    policy_scores = np.load(_POLICY_SCORES).astype(np.float32)
    policy_ids = load_json(_POLICY_IDS)
    policy_assignments = get_cluster_assignments(policy_scores)
    policy_counts = np.bincount(policy_assignments, minlength=N_SDG).astype(np.int64)
    policy_centroids, policy_centroid_available = build_policy_centroids(
        policy_emb,
        policy_scores,
        policy_ids,
        SEGMENT_CAP_PRIMARY,
        args.seed,
    )
    verify_unit_norms(policy_centroids[policy_centroid_available], "policy centroids")

    log.info("Fitting policy-only PCA on %d rows", policy_emb.shape[0])
    policy_pca = PCA(n_components=2, random_state=args.seed)
    policy_pca.fit(policy_emb)
    policy_evr = np.asarray(policy_pca.explained_variance_ratio_, dtype=float)
    log.info("Policy PCA explained variance ratio: PC1=%.4f, PC2=%.4f", policy_evr[0], policy_evr[1])

    policy_plot_indices = sample_indices(policy_emb.shape[0], args.policy_plot_sample_size, args.seed + 3)
    policy_plot_emb = policy_emb[policy_plot_indices]
    policy_plot_labels = policy_assignments[policy_plot_indices]
    policy_plot_2d = policy_pca.transform(policy_plot_emb)
    policy_centroids_2d = policy_pca.transform(policy_centroids)

    policy_accumulators = init_metric_accumulators()
    update_metric_accumulators(
        policy_accumulators,
        policy_emb,
        policy_assignments,
        policy_centroids,
        policy_centroid_available,
        corpus_name="policy",
    )
    policy_metric_rows = summarise_metric_rows("policy", policy_accumulators, policy_counts)

    policy_silhouette_indices = sample_indices(
        policy_emb.shape[0],
        min(args.silhouette_sample_size, policy_emb.shape[0]),
        args.seed + 4,
    )
    policy_silhouette_emb = policy_emb[policy_silhouette_indices]
    policy_silhouette_labels = policy_assignments[policy_silhouette_indices]

    policy_kmeans_indices = sample_indices(
        policy_emb.shape[0],
        args.policy_kmeans_sample_size,
        args.seed + 5,
    )
    policy_kmeans_emb = policy_emb[policy_kmeans_indices]
    policy_kmeans_labels = policy_assignments[policy_kmeans_indices]
    policy_global_metrics = compute_clustering_metrics(
        policy_silhouette_emb,
        policy_silhouette_labels,
        policy_plot_2d,
        policy_plot_labels,
        policy_kmeans_emb,
        policy_kmeans_labels,
        random_seed=args.seed,
    )
    policy_global_metrics["silhouette_cosine_sample"] = policy_global_metrics.pop("silhouette_cosine")
    policy_global_metrics["silhouette_sample_size"] = int(policy_silhouette_emb.shape[0])
    policy_global_metrics["kmeans_sample_size"] = int(policy_kmeans_emb.shape[0])

    plot_within_corpus_pca(
        research_plot_2d,
        research_centroids_2d,
        research_centroid_available,
        point_label="Research sample",
        point_color=RESEARCH_COLOR,
        centroid_marker="o",
        centroid_label="Research SDG centroid",
        centroid_color=RESEARCH_COLOR,
        title="Within-corpus PCA structure of research SDG assignments",
        subtitle_note="PCA fitted on research embeddings only; diagnostic visualisation only",
        evr=research_evr,
        output_pdf=figures_dir / RESEARCH_FIG_PDF,
        output_png=figures_dir / RESEARCH_FIG_PNG,
    )
    plot_within_corpus_pca(
        policy_plot_2d,
        policy_centroids_2d,
        policy_centroid_available,
        point_label="Policy segments",
        point_color=POLICY_COLOR,
        centroid_marker="s",
        centroid_label="Policy SDG centroid",
        centroid_color=POLICY_COLOR,
        title="Within-corpus PCA structure of policy SDG assignments",
        subtitle_note="PCA fitted on policy embeddings only; diagnostic visualisation only",
        evr=policy_evr,
        output_pdf=figures_dir / POLICY_FIG_PDF,
        output_png=figures_dir / POLICY_FIG_PNG,
    )

    metrics_rows = research_metric_rows + policy_metric_rows
    metrics_path = data_dir / METRICS_CSV
    write_metrics_csv(metrics_path, metrics_rows)

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "random_seed": int(args.seed),
        "note": (
            "Diagnostic only. PCA is fitted separately within each corpus and the two panels do not share a common coordinate system."
        ),
        "research": {
            "total_embeddings_available": int(total_research),
            "pca_fitting_mode": "incremental_full",
            "pca_fit_rows": int(total_research),
            "plotted_rows": int(research_plot_emb.shape[0]),
            "pc1_explained_variance_ratio": float(research_evr[0]),
            "pc2_explained_variance_ratio": float(research_evr[1]),
            "zero_assignment_sdgs": [int(i + 1) for i, n in enumerate(research_counts) if int(n) == 0],
            "counts_by_sdg": {str(i + 1): int(research_counts[i]) for i in range(N_SDG)},
            "global_metrics": research_global_metrics,
        },
        "policy": {
            "total_embeddings_available": int(policy_emb.shape[0]),
            "pca_fitting_mode": "full",
            "pca_fit_rows": int(policy_emb.shape[0]),
            "plotted_rows": int(policy_plot_emb.shape[0]),
            "pc1_explained_variance_ratio": float(policy_evr[0]),
            "pc2_explained_variance_ratio": float(policy_evr[1]),
            "zero_assignment_sdgs": [int(i + 1) for i, n in enumerate(policy_counts) if int(n) == 0],
            "zero_centroid_sdgs_after_cap": [int(i + 1) for i, ok in enumerate(policy_centroid_available) if not bool(ok)],
            "counts_by_sdg": {str(i + 1): int(policy_counts[i]) for i in range(N_SDG)},
            "global_metrics": policy_global_metrics,
        },
    }
    summary_path = data_dir / SUMMARY_JSON
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_num_tex(tables_dir / NUM_TEX, summary)

    log.info("Saved: %s", figures_dir / RESEARCH_FIG_PDF)
    log.info("Saved: %s", figures_dir / RESEARCH_FIG_PNG)
    log.info("Saved: %s", figures_dir / POLICY_FIG_PDF)
    log.info("Saved: %s", figures_dir / POLICY_FIG_PNG)
    log.info("Saved: %s", metrics_path)
    log.info("Saved: %s", summary_path)
    log.info("Saved: %s", tables_dir / NUM_TEX)


if __name__ == "__main__":
    main()

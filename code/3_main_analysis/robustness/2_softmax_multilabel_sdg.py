"""
Softmax-weighted multi-label SDG membership robustness stage.

This is an alternative-specification robustness check only. It does not replace the
canonical hard-assignment coverage and semantic-gap results.

Run from project root:
    python code/3_main_analysis/robustness/2_softmax_multilabel_sdg.py
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-dissertation")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = ROOT / "code"
ANALYSIS_ROOT = Path(__file__).resolve().parents[1]
SHARED_DIR = ANALYSIS_ROOT / "shared"
for path in (CODE_ROOT, SHARED_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from research_embedding_shards import iter_research_embedding_shards
from research_score_shards import (
    load_json as load_score_manifest_json,
    resolve_from_manifest as resolve_score_manifest_path,
)
from semantic_gap_shared import (
    CHUNK_CAP_PRIMARY,
    N_SDG,
    POLICY_EMB,
    POLICY_IDS,
    POLICY_SCORES,
    RANDOM_SEED,
    build_sub_centroid,
    cap_policy_indices_per_doc,
    load_json,
)


DEFAULT_OUTPUT_ROOT = Path("outputs")
SCORED_DIR = Path("data/3_scored")
RESEARCH_SCORE_MANIFEST = SCORED_DIR / "paper_scores_shards" / "metadata" / "manifest.json"
RESEARCH_EMBED_MANIFEST = Path("data/2_embedded/research_shards/metadata/manifest.json")
SDG_CENTROIDS = SCORED_DIR / "sdg_centroids.npy"

DEFAULT_TEMPERATURES = [0.03, 0.05, 0.10, 0.20]
VARIANTS = ("raw_softmax", "corpus_calibrated_softmax")
EPS = 1e-8

COVERAGE_CSV = "softmax_multilabel_coverage.csv"
SEMANTIC_CSV = "softmax_multilabel_semantic_gaps.csv"
SUMMARY_CSV = "softmax_multilabel_comparison_summary.csv"
METADATA_JSON = "softmax_multilabel_metadata.json"

COVERAGE_SCATTER_PDF = "fig_softmax_vs_hard_coverage_gap.pdf"
COVERAGE_SCATTER_PNG = "fig_softmax_vs_hard_coverage_gap.png"
SEMANTIC_SCATTER_PDF = "fig_softmax_vs_hard_semantic_gap.pdf"
SEMANTIC_SCATTER_PNG = "fig_softmax_vs_hard_semantic_gap.png"
TEMP_SENS_PDF = "fig_softmax_temperature_sensitivity.pdf"
TEMP_SENS_PNG = "fig_softmax_temperature_sensitivity.png"

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ResearchScoreShard:
    shard_id: int
    name: str
    rows: int
    score_path: Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run softmax-weighted multi-label SDG robustness.")
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_ROOT))
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    p.add_argument("--temperatures", nargs="*", type=float, default=DEFAULT_TEMPERATURES)
    return p.parse_args()


def stable_softmax(scores: np.ndarray, temperature: float) -> np.ndarray:
    scaled = scores / float(temperature)
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    weights = np.exp(scaled, dtype=np.float64)
    denom = weights.sum(axis=1, keepdims=True)
    weights = weights / np.maximum(denom, EPS)
    return weights.astype(np.float32)


def normalize_rowsum(weights: np.ndarray, label: str) -> None:
    row_sums = weights.sum(axis=1)
    max_err = float(np.max(np.abs(row_sums - 1.0)))
    if not np.isfinite(max_err) or max_err > 1e-4:
        raise RuntimeError(f"{label}: softmax rows do not sum to 1 within tolerance (max err {max_err})")


def normalize_centroid(raw_vec: np.ndarray) -> np.ndarray | None:
    norm = float(np.linalg.norm(raw_vec))
    if norm < 1e-8:
        return None
    return (raw_vec / norm).astype(np.float32)


def load_research_score_shards(manifest_path: Path) -> dict[int, ResearchScoreShard]:
    manifest = load_score_manifest_json(manifest_path)
    out: dict[int, ResearchScoreShard] = {}
    for shard in manifest.get("shards", []):
        shard_id = int(shard["shard_id"])
        out[shard_id] = ResearchScoreShard(
            shard_id=shard_id,
            name=str(shard["name"]),
            rows=int(shard["rows"]),
            score_path=resolve_score_manifest_path(manifest_path, shard["score_path"]),
        )
    return out


def validate_sdg_centroids(path: Path) -> np.ndarray:
    centroids = np.load(path).astype(np.float32)
    if centroids.shape != (N_SDG, centroids.shape[1]):
        raise RuntimeError(f"Expected 17 SDG centroids, got shape {centroids.shape}")
    norms = np.linalg.norm(centroids, axis=1)
    if np.max(np.abs(norms - 1.0)) > 1e-4:
        raise RuntimeError("SDG centroids are expected to be unit-normalized.")
    return centroids


def compute_column_stats_streaming(score_paths: list[Path]) -> tuple[np.ndarray, np.ndarray, int]:
    total_rows = 0
    sum_scores = np.zeros(N_SDG, dtype=np.float64)
    sum_sq_scores = np.zeros(N_SDG, dtype=np.float64)
    for path in score_paths:
        scores = np.load(path, mmap_mode="r")
        total_rows += int(scores.shape[0])
        sum_scores += np.asarray(scores.sum(axis=0), dtype=np.float64)
        sum_sq_scores += np.asarray((scores.astype(np.float64) ** 2).sum(axis=0), dtype=np.float64)
    if total_rows <= 0:
        raise RuntimeError("No research score rows found for softmax robustness.")
    means = sum_scores / float(total_rows)
    variances = np.maximum(sum_sq_scores / float(total_rows) - means**2, EPS)
    stds = np.sqrt(variances)
    return means.astype(np.float32), stds.astype(np.float32), total_rows


def compute_column_stats_in_memory(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    means = scores.mean(axis=0).astype(np.float32)
    stds = np.maximum(scores.std(axis=0).astype(np.float32), EPS)
    return means, stds, int(scores.shape[0])


def document_weighted_policy_soft_profile(weights: np.ndarray, policy_ids: list[dict]) -> tuple[np.ndarray, dict]:
    doc_to_rows: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(policy_ids):
        doc_to_rows[row["source_doc"]].append(i)
    n_docs = len(doc_to_rows)
    doc_vectors = np.zeros((n_docs, N_SDG), dtype=np.float32)
    meta: dict[str, dict] = {}
    for d_idx, (source_doc, idxs) in enumerate(doc_to_rows.items()):
        doc_vec = weights[idxs].mean(axis=0)
        doc_vectors[d_idx] = doc_vec
        meta[source_doc] = {"n_chunks": len(idxs)}
    profile = doc_vectors.mean(axis=0).astype(np.float64)
    return profile, meta


def load_hard_baselines(output_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coverage = json.loads((output_dir / "sdg_attention_distribution_document_weighted.json").read_text())
    semantic = json.loads((output_dir / "sdg_conceptual_alignment_cosine_distances.json").read_text())
    def dict_profile_to_array(payload: dict[str, float]) -> np.ndarray:
        return np.asarray([float(payload[f"SDG{i}"]) for i in range(1, N_SDG + 1)], dtype=np.float64)

    hard_research = dict_profile_to_array(coverage["research_profile_hard"])
    hard_policy = dict_profile_to_array(coverage["policy_profile_hard_docweighted"])
    hard_coverage_gap = dict_profile_to_array(coverage["coverage_gap_hard"])
    hard_semantic_gap = np.zeros(N_SDG, dtype=np.float64)
    for row in semantic["per_sdg"]:
        hard_semantic_gap[int(row["sdg"]) - 1] = float(row["semantic_gap"])
    return hard_research, hard_policy, hard_coverage_gap, hard_semantic_gap


def capped_policy_indices_for_soft_semantics(policy_ids: list[dict], seed: int) -> dict[int, list[int]]:
    all_indices = list(range(len(policy_ids)))
    capped: dict[int, list[int]] = {}
    for sdg_idx in range(N_SDG):
        capped[sdg_idx] = cap_policy_indices_per_doc(
            all_indices,
            policy_ids,
            CHUNK_CAP_PRIMARY,
            np.random.default_rng(seed + sdg_idx),
        )
    return capped


def pearson_spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    pearson = float(stats.pearsonr(x, y).statistic)
    spearman = float(stats.spearmanr(x, y).statistic)
    return pearson, spearman


def top_k_overlap(a: np.ndarray, b: np.ndarray, k: int) -> int:
    a_top = set(np.argsort(a)[-k:])
    b_top = set(np.argsort(b)[-k:])
    return len(a_top & b_top)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_scatter_compare(
    out_pdf: Path,
    out_png: Path,
    title: str,
    x_label: str,
    y_label: str,
    hard_values: np.ndarray,
    results: dict[tuple[str, float], np.ndarray],
) -> None:
    variants = list(VARIANTS)
    fig, axes = plt.subplots(1, len(variants), figsize=(10.5, 4.5), sharex=True, sharey=True)
    if len(variants) == 1:
        axes = [axes]
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len({temp for _, temp in results})))
    temp_to_color = {temp: colors[i] for i, temp in enumerate(sorted({temp for _, temp in results}))}
    for ax, variant in zip(axes, variants):
        min_v = float(min(hard_values.min(), *(results[(variant, t)].min() for t in temp_to_color)))
        max_v = float(max(hard_values.max(), *(results[(variant, t)].max() for t in temp_to_color)))
        pad = 0.02 * (max_v - min_v if max_v > min_v else 1.0)
        ax.plot([min_v - pad, max_v + pad], [min_v - pad, max_v + pad], color="#888888", lw=0.9)
        for temp, color in temp_to_color.items():
            vals = results[(variant, temp)]
            ax.scatter(hard_values, vals, s=28, alpha=0.8, color=color, label=f"T={temp:.2f}")
        ax.set_title(variant.replace("_", " "))
        ax.set_xlabel(x_label)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel(y_label)
    axes[-1].legend(frameon=False, loc="best")
    fig.suptitle(title, x=0.01, ha="left", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=150)
    plt.close(fig)


def plot_temperature_sensitivity(
    out_pdf: Path,
    out_png: Path,
    summary_rows: list[dict],
) -> None:
    variants = list(VARIANTS)
    temps = sorted({float(r["temperature"]) for r in summary_rows})
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2), sharex=True)
    metrics = [
        ("mean_abs_coverage_difference", "Mean abs. coverage-gap diff."),
        ("mean_abs_semantic_gap_difference", "Mean abs. semantic-gap diff."),
    ]
    for ax, (metric, ylabel) in zip(axes, metrics):
        for variant in variants:
            ys = [
                float(
                    next(
                        row["value"]
                        for row in summary_rows
                        if row["variant"] == variant and float(row["temperature"]) == temp and row["metric"] == metric
                    )
                )
                for temp in temps
            ]
            ax.plot(temps, ys, marker="o", linewidth=1.2, label=variant.replace("_", " "))
        ax.set_xlabel("Temperature")
        ax.set_ylabel(ylabel)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[-1].legend(frameon=False, loc="best")
    fig.suptitle("Softmax temperature sensitivity (robustness only)", x=0.01, ha="left", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    temps = [float(t) for t in args.temperatures]
    if not temps:
        raise RuntimeError("At least one temperature is required.")

    sdg_centroids = validate_sdg_centroids(SDG_CENTROIDS)
    hard_research, hard_policy, hard_coverage_gap, hard_semantic_gap = load_hard_baselines(output_dir)

    research_score_shards = load_research_score_shards(RESEARCH_SCORE_MANIFEST)
    research_score_paths = [research_score_shards[k].score_path for k in sorted(research_score_shards)]
    research_means, research_stds, total_research_rows = compute_column_stats_streaming(research_score_paths)

    policy_scores = np.load(POLICY_SCORES).astype(np.float32)
    policy_emb = np.load(POLICY_EMB).astype(np.float32)
    policy_ids = load_json(POLICY_IDS)
    if policy_scores.shape[0] != policy_emb.shape[0] or policy_scores.shape[0] != len(policy_ids):
        raise RuntimeError("Policy score/embedding/id row mismatch.")
    policy_means, policy_stds, total_policy_rows = compute_column_stats_in_memory(policy_scores)

    research_acc: dict[tuple[str, float], dict[str, np.ndarray]] = {}
    for variant in VARIANTS:
        for temp in temps:
            research_acc[(variant, temp)] = {
                "weight_sums": np.zeros(N_SDG, dtype=np.float64),
                "weighted_embedding_sums": np.zeros((N_SDG, policy_emb.shape[1]), dtype=np.float64),
            }

    log.info("Streaming research corpus for softmax robustness: rows=%d", total_research_rows)
    for emb_shard in iter_research_embedding_shards(RESEARCH_EMBED_MANIFEST):
        score_shard = research_score_shards.get(emb_shard.shard_id)
        if score_shard is None:
            raise RuntimeError(f"Missing research score shard for embedding shard {emb_shard.shard_id}")
        scores = np.load(score_shard.score_path, mmap_mode="r").astype(np.float32)
        emb = np.load(emb_shard.embedding_path, mmap_mode="r").astype(np.float32)
        if scores.shape[0] != emb.shape[0]:
            raise RuntimeError(f"Row mismatch in research shard {emb_shard.name}: scores={scores.shape[0]} emb={emb.shape[0]}")
        raw_variant = scores
        calibrated_variant = (scores - research_means) / np.maximum(research_stds, EPS)
        for variant, variant_scores in (
            ("raw_softmax", raw_variant),
            ("corpus_calibrated_softmax", calibrated_variant),
        ):
            for temp in temps:
                weights = stable_softmax(variant_scores, temp)
                normalize_rowsum(weights, f"research {variant} T={temp}")
                acc = research_acc[(variant, temp)]
                acc["weight_sums"] += weights.sum(axis=0)
                acc["weighted_embedding_sums"] += weights.T @ emb

    policy_capped_indices = capped_policy_indices_for_soft_semantics(policy_ids, args.seed)

    coverage_rows: list[dict] = []
    semantic_rows: list[dict] = []
    summary_rows: list[dict] = []
    metadata_variants: dict[str, dict] = {}

    soft_coverage_gap_results: dict[tuple[str, float], np.ndarray] = {}
    soft_semantic_gap_results: dict[tuple[str, float], np.ndarray] = {}

    for variant in VARIANTS:
        if variant == "raw_softmax":
            policy_variant_scores = policy_scores
            policy_calibration = {
                "means": None,
                "stds": None,
            }
        else:
            policy_variant_scores = (policy_scores - policy_means) / np.maximum(policy_stds, EPS)
            policy_calibration = {
                "means": policy_means.tolist(),
                "stds": policy_stds.tolist(),
            }

        metadata_variants[variant] = {
            "research_similarity_means": research_means.tolist() if variant == "corpus_calibrated_softmax" else None,
            "research_similarity_stds": research_stds.tolist() if variant == "corpus_calibrated_softmax" else None,
            "policy_similarity_means": policy_calibration["means"],
            "policy_similarity_stds": policy_calibration["stds"],
        }

        for temp in temps:
            weights_policy = stable_softmax(policy_variant_scores, temp)
            normalize_rowsum(weights_policy, f"policy {variant} T={temp}")
            research_weight_sums = research_acc[(variant, temp)]["weight_sums"]
            research_weighted_sums = research_acc[(variant, temp)]["weighted_embedding_sums"]
            research_share = research_weight_sums / float(total_research_rows)

            policy_docweighted_share, policy_doc_meta = document_weighted_policy_soft_profile(weights_policy, policy_ids)
            coverage_gap = np.abs(research_share - policy_docweighted_share)
            soft_coverage_gap_results[(variant, temp)] = coverage_gap

            research_centroids_weighted = np.zeros((N_SDG, sdg_centroids.shape[1]), dtype=np.float32)
            policy_centroids_weighted = np.zeros((N_SDG, sdg_centroids.shape[1]), dtype=np.float32)
            semantic_gap = np.zeros(N_SDG, dtype=np.float64)

            for sdg_idx in range(N_SDG):
                research_mass = float(research_weight_sums[sdg_idx])
                if research_mass <= 0:
                    raise RuntimeError(f"Research total soft mass is zero for SDG {sdg_idx + 1}")
                research_centroid = normalize_centroid(research_weighted_sums[sdg_idx] / research_mass)
                if research_centroid is None:
                    raise RuntimeError(f"Research weighted centroid vanished for SDG {sdg_idx + 1}")
                research_centroids_weighted[sdg_idx] = research_centroid

                capped_idxs = policy_capped_indices[sdg_idx]
                policy_mass = float(weights_policy[capped_idxs, sdg_idx].sum())
                if policy_mass <= 0:
                    raise RuntimeError(f"Policy total soft mass is zero for SDG {sdg_idx + 1} after capping")
                raw_policy = np.average(policy_emb[capped_idxs], axis=0, weights=weights_policy[capped_idxs, sdg_idx])
                policy_centroid = normalize_centroid(raw_policy)
                if policy_centroid is None:
                    raise RuntimeError(f"Policy weighted centroid vanished for SDG {sdg_idx + 1}")
                policy_centroids_weighted[sdg_idx] = policy_centroid

                gap = 1.0 - float(np.dot(research_centroid, policy_centroid))
                semantic_gap[sdg_idx] = gap

                semantic_rows.append(
                    {
                        "variant": variant,
                        "temperature": temp,
                        "sdg": sdg_idx + 1,
                        "research_weighted_mass": round(research_mass, 6),
                        "policy_weighted_mass": round(policy_mass, 6),
                        "semantic_gap_cosine_distance_384d": round(gap, 6),
                        "hard_semantic_gap_if_available": round(float(hard_semantic_gap[sdg_idx]), 6),
                        "difference_from_hard_if_available": round(float(gap - hard_semantic_gap[sdg_idx]), 6),
                    }
                )

            soft_semantic_gap_results[(variant, temp)] = semantic_gap

            for sdg_idx in range(N_SDG):
                coverage_rows.extend(
                    [
                        {
                            "variant": variant,
                            "temperature": temp,
                            "corpus": "research",
                            "sdg": sdg_idx + 1,
                            "weighted_attention": round(float(research_weight_sums[sdg_idx]), 6),
                            "weighted_attention_share": round(float(research_share[sdg_idx]), 6),
                            "hard_attention_share_if_available": round(float(hard_research[sdg_idx]), 6),
                            "difference_from_hard_if_available": round(float(research_share[sdg_idx] - hard_research[sdg_idx]), 6),
                        },
                        {
                            "variant": variant,
                            "temperature": temp,
                            "corpus": "policy",
                            "sdg": sdg_idx + 1,
                            "weighted_attention": round(float(policy_docweighted_share[sdg_idx] * len(policy_doc_meta)), 6),
                            "weighted_attention_share": round(float(policy_docweighted_share[sdg_idx]), 6),
                            "hard_attention_share_if_available": round(float(hard_policy[sdg_idx]), 6),
                            "difference_from_hard_if_available": round(float(policy_docweighted_share[sdg_idx] - hard_policy[sdg_idx]), 6),
                        },
                    ]
                )

            cov_pearson, cov_spearman = pearson_spearman(hard_coverage_gap, coverage_gap)
            sem_pearson, sem_spearman = pearson_spearman(hard_semantic_gap, semantic_gap)
            summary_metrics = {
                "spearman_hard_vs_soft_coverage_gap": cov_spearman,
                "pearson_hard_vs_soft_coverage_gap": cov_pearson,
                "spearman_hard_vs_soft_semantic_gap": sem_spearman,
                "pearson_hard_vs_soft_semantic_gap": sem_pearson,
                "mean_abs_coverage_difference": float(np.mean(np.abs(coverage_gap - hard_coverage_gap))),
                "mean_abs_semantic_gap_difference": float(np.mean(np.abs(semantic_gap - hard_semantic_gap))),
                "top3_overlap_coverage_gap": float(top_k_overlap(hard_coverage_gap, coverage_gap, 3)),
                "top3_overlap_semantic_gap": float(top_k_overlap(hard_semantic_gap, semantic_gap, 3)),
            }
            for metric, value in summary_metrics.items():
                summary_rows.append(
                    {
                        "variant": variant,
                        "temperature": temp,
                        "metric": metric,
                        "value": round(float(value), 6),
                    }
                )

    for key, arr in soft_coverage_gap_results.items():
        if not np.isfinite(arr).all():
            raise RuntimeError(f"Non-finite coverage gap values for {key}")
    for key, arr in soft_semantic_gap_results.items():
        if not np.isfinite(arr).all():
            raise RuntimeError(f"Non-finite semantic gap values for {key}")

    coverage_path = tables_dir / COVERAGE_CSV
    semantic_path = tables_dir / SEMANTIC_CSV
    summary_path = tables_dir / SUMMARY_CSV
    metadata_path = tables_dir / METADATA_JSON

    write_csv(
        coverage_path,
        [
            "variant",
            "temperature",
            "corpus",
            "sdg",
            "weighted_attention",
            "weighted_attention_share",
            "hard_attention_share_if_available",
            "difference_from_hard_if_available",
        ],
        coverage_rows,
    )
    write_csv(
        semantic_path,
        [
            "variant",
            "temperature",
            "sdg",
            "research_weighted_mass",
            "policy_weighted_mass",
            "semantic_gap_cosine_distance_384d",
            "hard_semantic_gap_if_available",
            "difference_from_hard_if_available",
        ],
        semantic_rows,
    )
    write_csv(
        summary_path,
        ["variant", "temperature", "metric", "value"],
        summary_rows,
    )

    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "temperatures_used": temps,
        "variants_used": list(VARIANTS),
        "total_research_embeddings_processed": int(total_research_rows),
        "total_policy_embeddings_processed": int(total_policy_rows),
        "policy_coverage_weighting_method": "document_weighted_mean_of_chunk_weight_vectors",
        "policy_semantic_weighting_method": "per_sdg_weighted_chunk_centroid_with_per_document_chunk_cap",
        "policy_chunk_cap_used": int(CHUNK_CAP_PRIMARY),
        "sdg_centroid_file": str(SDG_CENTROIDS),
        "sdg_indexing_confirmation": (
            "No separate SDG mapping file exists in the active pipeline. Canonical convention is row 0 -> SDG 1 ... row 16 -> SDG 17."
        ),
        "embedding_normalization_convention": "all embeddings and SDG centroids are unit-normalized; semantic gaps remain computed in original 384D space.",
        "research_similarity_means": research_means.tolist(),
        "research_similarity_stds": research_stds.tolist(),
        "policy_similarity_means": policy_means.tolist(),
        "policy_similarity_stds": policy_stds.tolist(),
        "variant_specific_calibration": metadata_variants,
        "note": (
            "Robustness/alternative specification only. Softmax weighting treats SDGs as overlapping anchors rather than mutually exclusive clusters."
        ),
        "calibrated_variant_note": (
            "Corpus-calibrated softmax standardizes SDG similarities within each corpus before applying softmax. "
            "This may reduce sensitivity to absolute centroid proximity but does not eliminate benchmark/register bias."
        ),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    plot_scatter_compare(
        figures_dir / COVERAGE_SCATTER_PDF,
        figures_dir / COVERAGE_SCATTER_PNG,
        "Softmax vs hard coverage-gap specification",
        "Hard coverage gap",
        "Softmax coverage gap",
        hard_coverage_gap,
        soft_coverage_gap_results,
    )
    plot_scatter_compare(
        figures_dir / SEMANTIC_SCATTER_PDF,
        figures_dir / SEMANTIC_SCATTER_PNG,
        "Softmax vs hard semantic-gap specification",
        "Hard semantic gap",
        "Softmax semantic gap",
        hard_semantic_gap,
        soft_semantic_gap_results,
    )
    plot_temperature_sensitivity(
        figures_dir / TEMP_SENS_PDF,
        figures_dir / TEMP_SENS_PNG,
        summary_rows,
    )

    log.info("Saved: %s", coverage_path)
    log.info("Saved: %s", semantic_path)
    log.info("Saved: %s", summary_path)
    log.info("Saved: %s", metadata_path)
    log.info("Saved: %s", figures_dir / COVERAGE_SCATTER_PDF)
    log.info("Saved: %s", figures_dir / SEMANTIC_SCATTER_PDF)
    log.info("Saved: %s", figures_dir / TEMP_SENS_PDF)


if __name__ == "__main__":
    main()

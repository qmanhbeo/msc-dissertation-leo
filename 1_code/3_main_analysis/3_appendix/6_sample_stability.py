"""
Measure sample-size stability for the research-side downstream analysis.

This stage reuses existing score shards and embedding shards. It does not re-embed
research papers and does not overwrite any canonical scored artifacts.

For each sampled research tier, we draw multiple random subsets without replacement,
recompute the research-side coverage profile, rebuild sampled research SDG centroids,
and compare those sampled results against the fixed canonical policy-side quantities.

Outputs:
  4_outputs/appendix/c_sample_stability/data/4_5_sample_stability_summary.json
  4_outputs/appendix/c_sample_stability/data/4_5_sample_stability_draws.jsonl
  4_outputs/appendix/c_sample_stability/data/4_5_sample_stability_per_sdg.json
  4_outputs/appendix/c_sample_stability/data/4_5_sample_stability_table.csv
  4_outputs/appendix/c_sample_stability/tables/num_sample_stability.tex
  4_outputs/appendix/c_sample_stability/tables/tab_sample_stability.tex
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = ROOT / "1_code"
ANALYSIS_ROOT = Path(__file__).resolve().parents[1]
SHARED_DIR = ANALYSIS_ROOT / "0_shared"
for path in (CODE_ROOT, SHARED_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from model_utils import embed_dir_for_model, scored_dir_for_model
import semantic_gap_shared
from shared_utils import ensure_dissertation_outputs, require_output_files
from semantic_gap_shared import (
    MIN_CLUSTER_SIZE,
    RANDOM_SEED as POLICY_SEGMENT_CAP_SEED,
    SEGMENT_CAP_PRIMARY,
    build_sub_centroid,
    cap_policy_indices_per_doc,
)


DEFAULT_OUTPUT_ROOT = Path("4_outputs")
CANONICAL_COVERAGE_JSON = "4_2_coverage_document_weighted.json"
CANONICAL_SEMANTIC_JSON = "4_3_semantic_gap_distances.json"
CANONICAL_INTERACTION_JSON = "4_4_interaction_correlation_asymmetry.json"

N_SDG = 17
DRAW_SEEDS = tuple(range(42, 142))
DRAWS_PER_TIER = len(DRAW_SEEDS)
TIER_SPECS: list[tuple[str, int]] = [
    ("1k", 1_000),
    ("2k", 2_000),
    ("5k", 5_000),
    ("10k", 10_000),
    ("20k", 20_000),
    ("50k", 50_000),
    ("100k", 100_000),
    ("200k", 200_000),
    ("500k", 500_000),
    ("1m", 1_000_000),
    ("2m", 2_000_000),
]

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ResearchShard:
    shard_id: int
    name: str
    rows: int
    start: int
    stop: int
    score_path: Path
    emb_path: Path


@dataclass
class DrawAccumulator:
    tier_label: str
    sample_size: int
    draw_index: int
    seed: int
    global_indices: np.ndarray
    hard_counts: np.ndarray
    vector_sums: np.ndarray
    top_sum_osdg: float = 0.0
    rows_seen: int = 0


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def resolve_manifest_path(stored_path: str, embed_dir: Path, scored_dir: Path) -> Path:
    raw = Path(stored_path)
    if raw.is_absolute():
        if raw.exists():
            return raw
        raise FileNotFoundError(f"Absolute path from manifest does not exist: {raw}")
    posix = raw.as_posix()
    allowed = (str(embed_dir) + "/", str(scored_dir) + "/", "2_data/1_preprocessed/")
    if not any(posix.startswith(p) for p in allowed):
        raise RuntimeError(
            f"Hard pivot violation: expected path under {allowed}, got: {stored_path}"
        )
    resolved = Path.cwd() / raw
    if resolved.exists():
        return resolved
    raise FileNotFoundError(f"Manifest path does not exist: {stored_path} (resolved: {resolved})")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the sample-stability robustness stage.")
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_ROOT))
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--model", default="all-MiniLM-L6-v2", help=argparse.SUPPRESS)
    return p.parse_args()


def build_research_shards(embed_dir: Path, scored_dir: Path) -> tuple[list[ResearchShard], int]:
    score_manifest = load_json(scored_dir / "paper_scores_shards" / "metadata" / "manifest.json")
    emb_manifest = load_json(embed_dir / "research_shards" / "metadata" / "manifest.json")
    score_shards = sorted(score_manifest.get("shards", []), key=lambda x: int(x["shard_id"]))
    emb_shards = sorted(emb_manifest.get("shards", []), key=lambda x: int(x["shard_id"]))
    if len(score_shards) != len(emb_shards):
        raise RuntimeError(
            f"Shard count mismatch: score={len(score_shards)} embedding={len(emb_shards)}"
        )

    shards: list[ResearchShard] = []
    offset = 0
    for score_shard, emb_shard in zip(score_shards, emb_shards):
        score_id = int(score_shard["shard_id"])
        emb_id = int(emb_shard["shard_id"])
        if score_id != emb_id or score_shard["name"] != emb_shard["name"]:
            raise RuntimeError(
                "Research score/embedding manifests are not aligned on shard_id/name: "
                f"score=({score_id}, {score_shard['name']}) "
                f"embedding=({emb_id}, {emb_shard['name']})"
            )
        rows = int(score_shard["rows"])
        if rows != int(emb_shard["rows"]):
            raise RuntimeError(
                f"Row mismatch for shard {score_shard['name']}: score={rows} embedding={emb_shard['rows']}"
            )
        shards.append(
            ResearchShard(
                shard_id=score_id,
                name=score_shard["name"],
                rows=rows,
                start=offset,
                stop=offset + rows,
                score_path=resolve_manifest_path(score_shard["score_path"], embed_dir, scored_dir),
                emb_path=resolve_manifest_path(emb_shard["embedding_path"], embed_dir, scored_dir),
            )
        )
        offset += rows

    return shards, offset


def draw_seed(draw_index: int) -> int:
    if draw_index < 1 or draw_index > DRAWS_PER_TIER:
        raise ValueError(f"draw_index out of range: {draw_index}")
    return DRAW_SEEDS[draw_index - 1]


def format_sample_label(label: str, sample_size: int, is_full: bool = False) -> str:
    if is_full:
        return "Full corpus"
    return f"{sample_size:,}"


def format_pm(mean_value: float | None, std_value: float | None, *, precision: int = 3) -> str:
    if mean_value is None:
        return "--"
    if std_value is None:
        return f"${mean_value:.{precision}f}$"
    return f"${mean_value:.{precision}f} \\pm {std_value:.{precision}f}$"


def format_variance(value: float | None, *, precision: int = 3) -> str:
    if value is None:
        return "--"
    return f"${value:.{precision}f}$"


def latex_num(value: int) -> str:
    return f"{value:,}".replace(",", "{,}")


def normalize_centroid(raw: np.ndarray) -> tuple[np.ndarray, float]:
    norm = float(np.linalg.norm(raw))
    if norm < 1e-8:
        return np.zeros_like(raw, dtype=np.float32), 0.0
    return (raw / norm).astype(np.float32), norm


def load_policy_state(canonical_data_dir: Path, policy_emb_path: Path, policy_ids_path: Path, policy_scores_path: Path) -> dict[str, Any]:
    require_output_files(
        canonical_data_dir,
        [CANONICAL_COVERAGE_JSON, CANONICAL_SEMANTIC_JSON, CANONICAL_INTERACTION_JSON],
    )

    coverage_out = load_json(canonical_data_dir / CANONICAL_COVERAGE_JSON)
    semantic_out = load_json(canonical_data_dir / CANONICAL_SEMANTIC_JSON)
    interaction_data = load_json(canonical_data_dir / CANONICAL_INTERACTION_JSON)

    policy_scores = np.load(policy_scores_path).astype(np.float32)
    policy_ids = load_json(policy_ids_path)
    policy_emb = np.load(policy_emb_path).astype(np.float32)
    if policy_scores.shape[0] != len(policy_ids) or policy_emb.shape[0] != len(policy_ids):
        raise RuntimeError(
            "Policy score/embedding/id row mismatch: "
            f"scores={policy_scores.shape[0]} emb={policy_emb.shape[0]} ids={len(policy_ids)}"
        )

    policy_profile_hard = np.array(
        [
            float(coverage_out["policy_profile_hard_docweighted"][f"SDG{i}"])
            for i in range(1, N_SDG + 1)
        ],
        dtype=np.float64,
    )
    policy_top_vs_osdg = float(policy_scores.max(axis=1).mean())
    policy_assignments = policy_scores.argmax(axis=1)

    rng = np.random.default_rng(POLICY_SEGMENT_CAP_SEED)
    dim = int(policy_emb.shape[1])
    policy_centroids = np.zeros((N_SDG, dim), dtype=np.float32)
    policy_counts_raw = np.zeros(N_SDG, dtype=np.int64)
    policy_counts_capped = np.zeros(N_SDG, dtype=np.int64)
    policy_doc_counts_capped = np.zeros(N_SDG, dtype=np.int64)
    policy_cohesions = np.zeros(N_SDG, dtype=np.float32)
    policy_centroid_available = np.zeros(N_SDG, dtype=bool)

    for sdg_idx in range(N_SDG):
        policy_idxs = np.flatnonzero(policy_assignments == sdg_idx).tolist()
        policy_counts_raw[sdg_idx] = len(policy_idxs)
        capped = cap_policy_indices_per_doc(policy_idxs, policy_ids, SEGMENT_CAP_PRIMARY, rng)
        policy_counts_capped[sdg_idx] = len(capped)
        policy_doc_counts_capped[sdg_idx] = len({policy_ids[i]["source_doc"] for i in capped})
        centroid, cohesion = build_sub_centroid(policy_emb, capped)
        if centroid is not None:
            policy_centroids[sdg_idx] = centroid
            policy_cohesions[sdg_idx] = float(cohesion)
            policy_centroid_available[sdg_idx] = True

    per_sdg_semantic = {int(row["sdg"]): row for row in semantic_out["per_sdg"]}
    full_semantic_gaps = [
        float(per_sdg_semantic[sdg]["semantic_gap"])
        for sdg in range(1, N_SDG + 1)
        if per_sdg_semantic[sdg]["semantic_gap"] is not None
    ]
    full_mean_semantic_gap = float(np.mean(full_semantic_gaps))
    full_mean_paper_top_vs_osdg = float(interaction_data["asymmetry"]["mean_paper_top_vs_osdg"])
    full_asym_gap = float(interaction_data["asymmetry"]["asymmetry_gap"])

    return {
        "policy_profile_hard_docweighted": policy_profile_hard,
        "policy_top_vs_osdg": policy_top_vs_osdg,
        "policy_embeddings": policy_emb,
        "policy_counts_raw": policy_counts_raw,
        "policy_counts_capped": policy_counts_capped,
        "policy_doc_counts_capped": policy_doc_counts_capped,
        "policy_centroids": policy_centroids,
        "policy_cohesions": policy_cohesions,
        "policy_centroid_available": policy_centroid_available,
        "full_mean_semantic_gap": full_mean_semantic_gap,
        "full_mean_paper_top_vs_osdg": full_mean_paper_top_vs_osdg,
        "full_asym_gap": full_asym_gap,
    }

def cache_dir_for_tier(cache_root: Path, tier_label: str) -> Path:
    return cache_root / tier_label


def draw_indices_cache_path(cache_root: Path, tier_label: str, draw_index: int) -> Path:
    return cache_dir_for_tier(cache_root, tier_label) / f"draw_{draw_index:02d}_indices.npy"


def draw_aggregate_cache_path(cache_root: Path, tier_label: str, draw_index: int) -> Path:
    return cache_dir_for_tier(cache_root, tier_label) / f"draw_{draw_index:02d}_aggregate.npz"


def load_cached_draw(cache_root: Path, tier_label: str, draw_index: int) -> dict[str, Any] | None:
    agg_path = draw_aggregate_cache_path(cache_root, tier_label, draw_index)
    if not agg_path.exists():
        return None
    cached = np.load(agg_path)
    return {
        "sample_size": int(cached["sample_size"]),
        "seed": int(cached["seed"]),
        "hard_counts": cached["hard_counts"].astype(np.int64),
        "vector_sums": cached["vector_sums"].astype(np.float64),
        "top_sum_osdg": float(cached["top_sum_osdg"]),
        "rows_seen": int(cached["rows_seen"]),
    }


def write_cache_manifest(cache_root: Path, total_rows: int) -> None:
    cache_root.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_root / "manifest.json"
    payload = {
        "schema_version": 1,
        "draw_seed_start": DRAW_SEEDS[0],
        "draw_seed_end": DRAW_SEEDS[-1],
        "draw_seeds": list(DRAW_SEEDS),
        "draws_per_tier": DRAWS_PER_TIER,
        "sampled_tiers": [{"tier_label": label, "sample_size": size} for label, size in TIER_SPECS],
        "research_rows_available": total_rows,
    }
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def build_draw_accumulators(cache_root: Path, total_rows: int, dim: int) -> tuple[list[DrawAccumulator], list[DrawAccumulator]]:
    draws: list[DrawAccumulator] = []
    pending: list[DrawAccumulator] = []
    for tier_label, sample_size in TIER_SPECS:
        if sample_size >= total_rows:
            raise RuntimeError(
                f"Tier {tier_label} requires {sample_size:,} rows, but only {total_rows:,} exist."
            )
        for draw_index in range(1, DRAWS_PER_TIER + 1):
            seed = draw_seed(draw_index)
            cached = load_cached_draw(cache_root, tier_label, draw_index)
            if cached is not None:
                if cached["sample_size"] != sample_size or cached["seed"] != seed:
                    raise RuntimeError(
                        f"Cached draw mismatch for {tier_label} draw {draw_index}: "
                        f"expected size={sample_size} seed={seed}, "
                        f"got size={cached['sample_size']} seed={cached['seed']}"
                    )
                draw = DrawAccumulator(
                    tier_label=tier_label,
                    sample_size=sample_size,
                    draw_index=draw_index,
                    seed=seed,
                    global_indices=np.empty(0, dtype=np.int32),
                    hard_counts=cached["hard_counts"],
                    vector_sums=cached["vector_sums"],
                    top_sum_osdg=cached["top_sum_osdg"],
                    rows_seen=cached["rows_seen"],
                )
                draws.append(draw)
                continue

            indices_path = draw_indices_cache_path(cache_root, tier_label, draw_index)
            if indices_path.exists():
                global_indices = np.load(indices_path).astype(np.int32)
            else:
                rng = np.random.default_rng(seed)
                global_indices = np.sort(
                    rng.choice(total_rows, size=sample_size, replace=False).astype(np.int32)
                )
                indices_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(indices_path, global_indices)

            draw = DrawAccumulator(
                tier_label=tier_label,
                sample_size=sample_size,
                draw_index=draw_index,
                seed=seed,
                global_indices=global_indices,
                hard_counts=np.zeros(N_SDG, dtype=np.int64),
                vector_sums=np.zeros((N_SDG, dim), dtype=np.float64),
            )
            draws.append(draw)
            pending.append(draw)
    return draws, pending


def write_draw_caches(cache_root: Path, draws: list[DrawAccumulator]) -> None:
    for draw in draws:
        tier_dir = cache_dir_for_tier(cache_root, draw.tier_label)
        tier_dir.mkdir(parents=True, exist_ok=True)
        agg_path = draw_aggregate_cache_path(cache_root, draw.tier_label, draw.draw_index)
        np.savez_compressed(
            agg_path,
            sample_size=np.array(draw.sample_size, dtype=np.int64),
            seed=np.array(draw.seed, dtype=np.int64),
            hard_counts=draw.hard_counts.astype(np.int64),
            vector_sums=draw.vector_sums.astype(np.float64),
            top_sum_osdg=np.array(draw.top_sum_osdg, dtype=np.float64),
            rows_seen=np.array(draw.rows_seen, dtype=np.int64),
        )


def accumulate_draws(shards: list[ResearchShard], draws: list[DrawAccumulator]) -> None:
    for shard in shards:
        log.info("Processing research shard %s (%d rows)", shard.name, shard.rows)
        score = np.load(shard.score_path).astype(np.float32)
        emb = np.load(shard.emb_path).astype(np.float32)
        if score.shape[0] != emb.shape[0]:
            raise RuntimeError(
                f"Score/embedding row mismatch for shard {shard.name}: "
                f"score={score.shape[0]} emb={emb.shape[0]}"
            )
        assignments = score.argmax(axis=1)
        top_vals = score[np.arange(score.shape[0]), assignments]

        for draw in draws:
            left = int(np.searchsorted(draw.global_indices, shard.start, side="left"))
            right = int(np.searchsorted(draw.global_indices, shard.stop, side="left"))
            if left >= right:
                continue
            local = draw.global_indices[left:right].astype(np.int64) - shard.start
            local_assignments = assignments[local]
            draw.rows_seen += int(local.shape[0])
            draw.hard_counts += np.bincount(local_assignments, minlength=N_SDG)
            draw.top_sum_osdg += float(top_vals[local].sum())
            for sdg_idx in np.unique(local_assignments):
                mask = local_assignments == sdg_idx
                draw.vector_sums[sdg_idx] += emb[local[mask]].sum(axis=0)

        del score
        del emb

    for draw in draws:
        if draw.rows_seen != draw.sample_size:
            raise RuntimeError(
                f"Sample size mismatch for {draw.tier_label} draw {draw.draw_index}: "
                f"expected {draw.sample_size}, saw {draw.rows_seen}"
            )


def build_research_centroids(draw: DrawAccumulator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dim = draw.vector_sums.shape[1]
    centroids = np.zeros((N_SDG, dim), dtype=np.float32)
    counts = draw.hard_counts.astype(np.int64, copy=True)
    cohesions = np.zeros(N_SDG, dtype=np.float32)
    for sdg_idx in range(N_SDG):
        n = int(counts[sdg_idx])
        if n == 0:
            continue
        raw = draw.vector_sums[sdg_idx] / float(n)
        centroid, cohesion = normalize_centroid(raw)
        centroids[sdg_idx] = centroid
        cohesions[sdg_idx] = float(cohesion)
    return centroids, counts, cohesions


def to_sdg_dict(values: np.ndarray, *, scale: float = 1.0, round_digits: int = 6) -> dict[str, float]:
    return {
        f"SDG{i + 1}": round(float(values[i] * scale), round_digits)
        for i in range(N_SDG)
    }


def compute_draw_metrics(draw: DrawAccumulator, policy_state: dict[str, Any]) -> dict[str, Any]:
    research_centroids, research_counts, research_cohesions = build_research_centroids(draw)
    coverage_profile = draw.hard_counts.astype(np.float64) / float(draw.sample_size)

    semantic_gap_by_sdg: dict[str, float | None] = {}
    semantic_reliable_by_sdg: dict[str, bool] = {}
    research_counts_by_sdg: dict[str, int] = {}
    semantic_values_all: list[float] = []
    semantic_values_reliable: list[float] = []

    for sdg_idx in range(N_SDG):
        sdg = sdg_idx + 1
        n_papers = int(research_counts[sdg_idx])
        n_policy = int(policy_state["policy_counts_capped"][sdg_idx])
        research_counts_by_sdg[f"SDG{sdg}"] = n_papers

        if (
            n_papers == 0
            or not policy_state["policy_centroid_available"][sdg_idx]
            or float(np.linalg.norm(research_centroids[sdg_idx])) < 1e-8
        ):
            semantic_gap_by_sdg[f"SDG{sdg}"] = None
            semantic_reliable_by_sdg[f"SDG{sdg}"] = False
            continue

        similarity = float(np.dot(research_centroids[sdg_idx], policy_state["policy_centroids"][sdg_idx]))
        gap = 1.0 - similarity
        unreliable = n_papers < MIN_CLUSTER_SIZE or n_policy < MIN_CLUSTER_SIZE
        semantic_gap_by_sdg[f"SDG{sdg}"] = round(gap, 6)
        semantic_reliable_by_sdg[f"SDG{sdg}"] = not unreliable
        semantic_values_all.append(gap)
        if not unreliable:
            semantic_values_reliable.append(gap)

    mean_paper_top_vs_osdg = draw.top_sum_osdg / float(draw.sample_size)
    policy_vs_sample_research = (
        policy_state["policy_embeddings"] @ research_centroids.T
    ).max(axis=1).mean()
    asym_gap = float(policy_vs_sample_research - mean_paper_top_vs_osdg)
    a15_gap = float(policy_state["policy_top_vs_osdg"] - mean_paper_top_vs_osdg)

    mean_semantic_gap = float(np.mean(semantic_values_all)) if semantic_values_all else None
    mean_semantic_gap_reliable = (
        float(np.mean(semantic_values_reliable)) if semantic_values_reliable else None
    )

    return {
        "tier_label": draw.tier_label,
        "sample_size": draw.sample_size,
        "draw_index": draw.draw_index,
        "seed": draw.seed,
        "coverage_profile_hard": to_sdg_dict(coverage_profile),
        "mean_paper_top_vs_osdg": round(mean_paper_top_vs_osdg, 6),
        "policy_top_vs_sample_research": round(float(policy_vs_sample_research), 6),
        "asymmetry_gap": round(asym_gap, 6),
        "a15_calibration_bias": round(a15_gap, 6),
        "mean_semantic_gap": None if mean_semantic_gap is None else round(mean_semantic_gap, 6),
        "mean_semantic_gap_reliable_only": (
            None if mean_semantic_gap_reliable is None else round(mean_semantic_gap_reliable, 6)
        ),
        "n_observed_semantic_sdgs": len(semantic_values_all),
        "n_reliable_semantic_sdgs": len(semantic_values_reliable),
        "semantic_gap_by_sdg": semantic_gap_by_sdg,
        "semantic_reliable_by_sdg": semantic_reliable_by_sdg,
        "research_counts_by_sdg": research_counts_by_sdg,
        "coverage_gap_vs_policy_abs_mean": round(
            float(
                np.abs(
                    coverage_profile - policy_state["policy_profile_hard_docweighted"]
                ).mean()
            ),
            6,
        ),
        "research_cohesion_by_sdg": {
            f"SDG{i + 1}": round(float(research_cohesions[i]), 6) for i in range(N_SDG)
        },
    }


def summarize_tiers(
    draw_results: list[dict[str, Any]],
    *,
    total_rows: int,
    full_mean_semantic_gap: float,
    full_asym_gap: float,
    full_a15_gap: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    summary_rows: list[dict[str, Any]] = []
    per_sdg_rows: list[dict[str, Any]] = []

    for tier_label, sample_size in TIER_SPECS:
        tier_draws = [row for row in draw_results if row["tier_label"] == tier_label]
        coverage_matrix = np.array(
            [
                [row["coverage_profile_hard"][f"SDG{i}"] for i in range(1, N_SDG + 1)]
                for row in tier_draws
            ],
            dtype=np.float64,
        )
        semantic_matrix = np.array(
            [
                [
                    np.nan
                    if row["semantic_gap_by_sdg"][f"SDG{i}"] is None
                    else row["semantic_gap_by_sdg"][f"SDG{i}"]
                    for i in range(1, N_SDG + 1)
                ]
                for row in tier_draws
            ],
            dtype=np.float64,
        )
        reliable_matrix = np.array(
            [
                [bool(row["semantic_reliable_by_sdg"][f"SDG{i}"]) for i in range(1, N_SDG + 1)]
                for row in tier_draws
            ],
            dtype=bool,
        )
        macro_coverage_sd_by_sdg = coverage_matrix.std(axis=0)
        macro_coverage_variance = float(macro_coverage_sd_by_sdg.mean())

        semantic_means = np.array([row["mean_semantic_gap"] for row in tier_draws], dtype=np.float64)
        h26_values = np.array([row["asymmetry_gap"] for row in tier_draws], dtype=np.float64)
        a15_values = np.array([row["a15_calibration_bias"] for row in tier_draws], dtype=np.float64)

        summary_rows.append(
            {
                "tier_label": tier_label,
                "sample_size": sample_size,
                "n_draws": DRAWS_PER_TIER,
                "deterministic": False,
                "macro_coverage_variance": round(macro_coverage_variance, 6),
                "mean_semantic_gap": round(float(semantic_means.mean()), 6),
                "std_semantic_gap": round(float(semantic_means.std()), 6),
                "mean_asymmetry_gap": round(float(h26_values.mean()), 6),
                "std_asymmetry_gap": round(float(h26_values.std()), 6),
                "mean_a15_calibration_bias": round(float(a15_values.mean()), 6),
                "std_a15_calibration_bias": round(float(a15_values.std()), 6),
                "mean_observed_semantic_sdgs": round(
                    float(np.mean([row["n_observed_semantic_sdgs"] for row in tier_draws])), 3
                ),
                "mean_reliable_semantic_sdgs": round(
                    float(np.mean([row["n_reliable_semantic_sdgs"] for row in tier_draws])), 3
                ),
            }
        )

        per_sdg_entry = {
            "tier_label": tier_label,
            "sample_size": sample_size,
            "n_draws": DRAWS_PER_TIER,
            "coverage_share": {},
            "semantic_gap": {},
        }
        for sdg_idx in range(N_SDG):
            sdg_key = f"SDG{sdg_idx + 1}"
            per_sdg_entry["coverage_share"][sdg_key] = {
                "mean": round(float(coverage_matrix[:, sdg_idx].mean()), 6),
                "std": round(float(macro_coverage_sd_by_sdg[sdg_idx]), 6),
            }
            observed = int(np.isfinite(semantic_matrix[:, sdg_idx]).sum())
            reliable = int(reliable_matrix[:, sdg_idx].sum())
            per_sdg_entry["semantic_gap"][sdg_key] = {
                "mean": (
                    None
                    if observed == 0
                    else round(float(np.nanmean(semantic_matrix[:, sdg_idx])), 6)
                ),
                "std": (
                    None
                    if observed == 0
                    else round(float(np.nanstd(semantic_matrix[:, sdg_idx])), 6)
                ),
                "observed_draws": observed,
                "reliable_draws": reliable,
            }
        per_sdg_rows.append(per_sdg_entry)

    summary_rows.append(
        {
            "tier_label": "full corpus",
            "sample_size": total_rows,
            "n_draws": 1,
            "deterministic": True,
            "macro_coverage_variance": None,
            "mean_semantic_gap": round(full_mean_semantic_gap, 6),
            "std_semantic_gap": None,
            "mean_asymmetry_gap": round(full_asym_gap, 6),
            "std_asymmetry_gap": None,
            "mean_a15_calibration_bias": round(full_a15_gap, 6),
            "std_a15_calibration_bias": None,
            "mean_observed_semantic_sdgs": None,
            "mean_reliable_semantic_sdgs": None,
        }
    )

    return summary_rows, {
        "method": "sample_stability_random_subsampling",
        "draws_per_tier": DRAWS_PER_TIER,
        "sampled_tiers": [label for label, _ in TIER_SPECS],
        "sampled_tier_sizes": {label: size for label, size in TIER_SPECS},
        "per_tier": per_sdg_rows,
        "full_corpus": {
            "sample_size": total_rows,
            "mean_semantic_gap": round(full_mean_semantic_gap, 6),
            "mean_asymmetry_gap": round(full_asym_gap, 6),
            "mean_a15_calibration_bias": round(full_a15_gap, 6),
        },
    }


def write_outputs(
    output_root: Path,
    tables_dir: Path,
    *,
    summary_rows: list[dict[str, Any]],
    draw_results: list[dict[str, Any]],
    per_sdg_payload: dict[str, Any],
    total_rows: int,
) -> None:
    summary_path = output_root / "4_5_sample_stability_summary.json"
    draws_path = output_root / "4_5_sample_stability_draws.jsonl"
    per_sdg_path = output_root / "4_5_sample_stability_per_sdg.json"
    table_csv_path = output_root / "4_5_sample_stability_table.csv"

    summary_payload = {
        "method": "sample_stability_random_subsampling",
        "draws_per_tier": DRAWS_PER_TIER,
        "draw_seed_start": DRAW_SEEDS[0],
        "draw_seed_end": DRAW_SEEDS[-1],
        "draw_seeds": list(DRAW_SEEDS),
        "segment_cap": SEGMENT_CAP_PRIMARY,
        "min_cluster_size": MIN_CLUSTER_SIZE,
        "sampled_tiers": [
            {"tier_label": label, "sample_size": size} for label, size in TIER_SPECS
        ],
        "full_corpus_rows": total_rows,
        "summary_rows": summary_rows,
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2) + "\n", encoding="utf-8")

    with draws_path.open("w", encoding="utf-8") as f:
        for row in draw_results:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")

    per_sdg_path.write_text(json.dumps(per_sdg_payload, indent=2) + "\n", encoding="utf-8")

    with table_csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "tier_label",
                "sample_size",
                "n_draws",
                "deterministic",
                "macro_coverage_variance",
                "mean_semantic_gap",
                "std_semantic_gap",
                "mean_asymmetry_gap",
                "std_asymmetry_gap",
                "mean_a15_calibration_bias",
                "std_a15_calibration_bias",
                "mean_observed_semantic_sdgs",
                "mean_reliable_semantic_sdgs",
            ],
        )
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    num_lines = [
        "% Auto-generated by 1_code/3_main_analysis/3_appendix/6_sample_stability.py — do not edit manually",
        rf"\newcommand{{\SampleStabilityDraws}}{{{DRAWS_PER_TIER}}}",
        rf"\newcommand{{\SampleStabilitySampledTierCount}}{{{len(TIER_SPECS)}}}",
        rf"\newcommand{{\SampleStabilityTierCount}}{{{len(TIER_SPECS) + 1}}}",
        rf"\newcommand{{\SampleStabilityFullCorpusN}}{{{latex_num(total_rows)}}}",
    ]
    for row in summary_rows:
        if row["tier_label"] == "full corpus":
            continue
        word = {
            "1k": "OneK",
            "2k": "TwoK",
            "5k": "FiveK",
            "10k": "TenK",
            "20k": "TwentyK",
            "50k": "FiftyK",
            "100k": "HundredK",
            "200k": "TwoHundredK",
            "500k": "FiveHundredK",
            "1m": "OneM",
            "2m": "TwoM",
        }[row["tier_label"]]
        num_lines.append(
            rf"\newcommand{{\SampleMacroVariance{word}}}{{{row['macro_coverage_variance']:.3f}}}"
        )
        num_lines.append(
            rf"\newcommand{{\SampleMeanSemanticGap{word}}}{{{row['mean_semantic_gap']:.3f}}}"
        )
        num_lines.append(
            rf"\newcommand{{\SampleStdSemanticGap{word}}}{{{row['std_semantic_gap']:.3f}}}"
        )
    (tables_dir / "num_sample_stability.tex").write_text(
        "\n".join(num_lines) + "\n", encoding="utf-8"
    )

    tab_lines = [
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Sample Size & \shortstack{Mean SD of SDG\\coverage shares} & \shortstack{Mean within-SDG\\semantic gap} & \shortstack{Policy-to-research\\asymmetry} & \shortstack{Policy-text\\calibration bias} \\",
        r"\midrule",
    ]
    for row in summary_rows:
        label = format_sample_label(
            row["tier_label"],
            int(row["sample_size"]),
            is_full=bool(row["deterministic"]),
        )
        tab_lines.append(
            " & ".join(
                [
                    label,
                    format_variance(row["macro_coverage_variance"]),
                    format_pm(row["mean_semantic_gap"], row["std_semantic_gap"]),
                    format_pm(row["mean_asymmetry_gap"], row["std_asymmetry_gap"]),
                    format_pm(row["mean_a15_calibration_bias"], row["std_a15_calibration_bias"]),
                ]
            )
            + r" \\"
        )
    tab_lines.extend([r"\bottomrule", r"\end{tabular}"])
    (tables_dir / "tab_sample_stability.tex").write_text(
        "\n".join(tab_lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    embed_dir = embed_dir_for_model(args.model)
    scored_dir = scored_dir_for_model(args.model)
    _POLICY_EMB = semantic_gap_shared.get_policy_emb(args.model)
    _POLICY_IDS = semantic_gap_shared.get_policy_ids(args.model)
    _POLICY_SCORES = semantic_gap_shared.get_policy_scores(args.model)
    layout = ensure_dissertation_outputs(Path(args.output_dir), subdir="appendix/c_sample_stability")
    cache_root = Path(args.cache_dir) if args.cache_dir is not None else scored_dir / f"paper_sample_seed_{DRAW_SEEDS[0]}_{DRAW_SEEDS[-1]}"
    log.info("Canonical output dir: %s", layout.root)
    log.info("Sample-stability cache dir: %s", cache_root)

    policy_state = load_policy_state(Path(args.output_dir) / "main" / "data", _POLICY_EMB, _POLICY_IDS, _POLICY_SCORES)
    shards, total_rows = build_research_shards(embed_dir, scored_dir)
    dim = int(policy_state["policy_embeddings"].shape[1])
    log.info("Research corpus rows available for sampling: %d", total_rows)
    log.info("Sampling %d tiers x %d draws each", len(TIER_SPECS), DRAWS_PER_TIER)

    write_cache_manifest(cache_root, total_rows)
    draws, pending_draws = build_draw_accumulators(cache_root, total_rows, dim)
    log.info(
        "Sample-stability draw cache: %d reused, %d to build",
        len(draws) - len(pending_draws),
        len(pending_draws),
    )
    if pending_draws:
        accumulate_draws(shards, pending_draws)
        write_draw_caches(cache_root, pending_draws)

    LOG_INTERVAL = 100
    draw_results = []
    for idx, draw in enumerate(draws, start=1):
        if idx == 1 or idx % LOG_INTERVAL == 0:
            log.info(
                "Scoring sampled draw %d/%d: tier=%s draw=%d n=%d",
                idx,
                len(draws),
                draw.tier_label,
                draw.draw_index,
                draw.sample_size,
            )
        draw_results.append(compute_draw_metrics(draw, policy_state))

    full_a15_gap = float(
        policy_state["policy_top_vs_osdg"] - policy_state["full_mean_paper_top_vs_osdg"]
    )

    summary_rows, per_sdg_payload = summarize_tiers(
        draw_results,
        total_rows=total_rows,
        full_mean_semantic_gap=policy_state["full_mean_semantic_gap"],
        full_asym_gap=policy_state["full_asym_gap"],
        full_a15_gap=full_a15_gap,
    )
    write_outputs(
        layout.data_dir,
        layout.tables_dir,
        summary_rows=summary_rows,
        draw_results=draw_results,
        per_sdg_payload=per_sdg_payload,
        total_rows=total_rows,
    )
    log.info("Saved sample-stability outputs into %s", layout.data_dir)


if __name__ == "__main__":
    main()

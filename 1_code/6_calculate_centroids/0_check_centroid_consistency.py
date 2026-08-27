"""
Check centroid consistency for the LR-supervised classification.

For each corpus (research, policy):
  - Compute which SDG centroid each text is nearest to (cosine sim).
  - Compare with the LR-assigned SDG.
  - Report per-SDG agreement rates and confusion matrices.

Also saves policy centroids persistently (research centroids already exist).

NOTE: this is a runtime SANITY GATE, not a producer. Its three written
outputs (policy_centroids.npy, policy_centroid_meta.json,
centroid_consistency.json) are diagnostic-only and are NOT read by any
downstream script — its value is failing loudly if research/policy
centroids or scores are inconsistent, not the artifacts it writes.
(Its inputs became the LR classifier's products when score_supervised.py
absorbed the former per-classifier scoring scripts; older copies of this
docstring still said MLP.)

Inputs:
   2_data/3_embedded/{model}/research_shards/metadata/manifest.json
   2_data/3_embedded/{model}/policy.npy
   2_data/5_supervised_scored/{model}/research_centroids.npy
   2_data/5_supervised_scored/{model}/paper_scores_shards/
   2_data/5_supervised_scored/{model}/policy_scores.npy

Outputs:
   2_data/5_supervised_scored/{model}/policy_centroids.npy
   2_data/5_supervised_scored/{model}/metadata/policy_centroid_meta.json
   2_data/5_supervised_scored/{model}/metadata/centroid_consistency.json

Run from project root:
    python 1_code/6_calculate_centroids/0_check_centroid_consistency.py --model all-mpnet-base-v2
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
ANALYSIS_DIR = CODE_ROOT / "7_main_analysis" / "0_shared"
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from model_utils import N_SDG, embed_dir_for_model, embed_research_dir_for_model, scored_dir_for_model, DEFAULT_EMBED_MODEL, ZERO_NORM_EPS, resolve_model_alias
from shard_pipeline_utils import atomic_write_npy, ensure_dir, now_iso, read_json

log = logging.getLogger(__name__)


def resolve_embedding_path(manifest_path: Path, stored_path: str, embed_dir: Path) -> Path:
    del manifest_path
    raw = Path(stored_path)
    if raw.is_absolute():
        if raw.exists():
            return raw
        raise FileNotFoundError(f"Absolute path from manifest does not exist: {raw}")
    expected_prefix = embed_dir.as_posix() + "/"
    if not raw.as_posix().startswith(expected_prefix):
        raise RuntimeError(
            f"Hard pivot violation: expected data path under {expected_prefix}, got: {stored_path}"
        )
    resolved = Path.cwd() / raw
    if resolved.exists():
        return resolved
    raise FileNotFoundError(f"Manifest path does not exist: {stored_path} (resolved: {resolved})")


def compute_centroid_consistency(
    embeddings: np.ndarray,
    scores: np.ndarray,
    centroids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    assigned = scores.argmax(axis=1)
    sims = embeddings @ centroids.T
    nearest = sims.argmax(axis=1)

    n_per_sdg = np.zeros(N_SDG, dtype=np.int64)
    n_matched = np.zeros(N_SDG, dtype=np.int64)
    confusion = np.zeros((N_SDG, N_SDG), dtype=np.int64)

    for sdg_idx in range(N_SDG):
        mask = assigned == sdg_idx
        n = int(mask.sum())
        n_per_sdg[sdg_idx] = n
        if n > 0:
            n_matched[sdg_idx] = int((nearest[mask] == sdg_idx).sum())
            for j in range(N_SDG):
                confusion[sdg_idx, j] = int((nearest[mask] == j).sum())

    return n_per_sdg, n_matched, confusion


def build_policy_centroids(
    embeddings: np.ndarray,
    scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    assigned = scores.argmax(axis=1)
    centroids = np.zeros((N_SDG, embeddings.shape[1]), dtype=np.float32)
    meta: list[dict[str, Any]] = []

    for sdg_idx in range(N_SDG):
        mask = assigned == sdg_idx
        n = int(mask.sum())
        sdg = sdg_idx + 1
        if n == 0:
            centroids[sdg_idx] = 0.0
            meta.append({
                "sdg": sdg,
                "n_segments_assigned": 0,
                "raw_centroid_norm": 0.0,
                "zero_flag": True,
            })
            continue

        raw = embeddings[mask].mean(axis=0).astype(np.float32)
        norm = float(np.linalg.norm(raw))
        if norm < ZERO_NORM_EPS:
            centroids[sdg_idx] = 0.0
            meta.append({
                "sdg": sdg,
                "n_segments_assigned": n,
                "raw_centroid_norm": 0.0,
                "zero_flag": True,
            })
            continue

        unit = (raw / norm).astype(np.float32)
        centroids[sdg_idx] = unit
        meta.append({
            "sdg": sdg,
            "n_segments_assigned": n,
            "raw_centroid_norm": round(norm, 6),
            "zero_flag": False,
        })

    return centroids, np.array([not m["zero_flag"] for m in meta], dtype=bool), meta


def check_research(manifest_path: Path, research_centroids: np.ndarray, research_scores_dir: Path) -> dict[str, Any]:
    log.info("=== Research corpus ===")

    emb_manifest = read_json(manifest_path)
    if not emb_manifest or "shards" not in emb_manifest:
        raise RuntimeError(f"Invalid research embedding manifest: {manifest_path}")

    embed_dir = manifest_path.parent.parent
    shards = emb_manifest["shards"]

    total_n = np.zeros(N_SDG, dtype=np.int64)
    total_matched = np.zeros(N_SDG, dtype=np.int64)
    total_confusion = np.zeros((N_SDG, N_SDG), dtype=np.int64)
    total_rows = 0

    for shard in shards:
        shard_id = shard["shard_id"]
        shard_name = shard["name"]

        emb_path = resolve_embedding_path(manifest_path, shard["embedding_path"], embed_dir)
        score_path = research_scores_dir / f"{shard_name}.npy"

        emb = np.load(emb_path).astype(np.float32)
        scores = np.load(score_path).astype(np.float32)

        if emb.shape[0] != scores.shape[0]:
            raise RuntimeError(f"Shard {shard_name}: emb rows {emb.shape[0]} != scores rows {scores.shape[0]}")

        n_per_sdg, n_matched, confusion = compute_centroid_consistency(emb, scores, research_centroids)
        total_n += n_per_sdg
        total_matched += n_matched
        total_confusion += confusion
        total_rows += emb.shape[0]

        log.info("  Shard %s (%d rows): done", shard_name, emb.shape[0])

    overall_agreement = float(total_matched.sum()) / float(max(total_n.sum(), 1))
    per_sdg: list[dict[str, Any]] = []
    for sdg_idx in range(N_SDG):
        n = int(total_n[sdg_idx])
        matched = int(total_matched[sdg_idx])
        rate = float(matched) / float(max(n, 1))
        per_sdg.append({
            "sdg": sdg_idx + 1,
            "n_assigned": n,
            "n_nearest_own_centroid": matched,
            "agreement_rate": round(rate, 6),
        })

    return {
        "overall_agreement_rate": round(overall_agreement, 6),
        "total_rows": total_rows,
        "per_sdg": per_sdg,
        "confusion_matrix": total_confusion.tolist(),
    }


def check_policy(
    policy_emb: np.ndarray,
    policy_scores: np.ndarray,
    policy_centroids: np.ndarray,
) -> dict[str, Any]:
    log.info("=== Policy corpus ===")

    total_n, total_matched, total_confusion = compute_centroid_consistency(
        policy_emb, policy_scores, policy_centroids
    )

    overall_agreement = float(total_matched.sum()) / float(max(total_n.sum(), 1))
    per_sdg: list[dict[str, Any]] = []
    for sdg_idx in range(N_SDG):
        n = int(total_n[sdg_idx])
        matched = int(total_matched[sdg_idx])
        rate = float(matched) / float(max(n, 1))
        per_sdg.append({
            "sdg": sdg_idx + 1,
            "n_assigned": n,
            "n_nearest_own_centroid": matched,
            "agreement_rate": round(rate, 6),
        })

    return {
        "overall_agreement_rate": round(overall_agreement, 6),
        "total_rows": int(policy_emb.shape[0]),
        "per_sdg": per_sdg,
        "confusion_matrix": total_confusion.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check centroid consistency for LR-supervised classification."
    )
    parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL, type=resolve_model_alias,
                        help="Embed model (default: %(default)s)")
    parser.add_argument("--output-dir", default=None,
                        help="Ignored (compatibility with main.py pipeline)")
    parser.add_argument("--research-centroids", default=None)
    parser.add_argument("--policy-emb", default=None)
    parser.add_argument("--policy-scores", default=None)
    parser.add_argument("--research-manifest", default=None)
    parser.add_argument("--policy-centroids-out", default=None)
    parser.add_argument("--policy-centroid-meta-out", default=None)
    parser.add_argument("--consistency-out", default=None)
    parser.add_argument("--overwrite", action="store_true",
                        help="Recompute centroid consistency even if outputs exist")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    scored_root = scored_dir_for_model(args.embed_model)
    embed_root = embed_dir_for_model(args.embed_model)

    research_centroids_path = Path(args.research_centroids) if args.research_centroids else scored_root / "research_centroids.npy"
    policy_emb_path = Path(args.policy_emb) if args.policy_emb else embed_root / "policy.npy"
    policy_scores_path = Path(args.policy_scores) if args.policy_scores else scored_root / "policy_scores.npy"
    research_manifest_path = Path(args.research_manifest) if args.research_manifest else embed_research_dir_for_model(args.embed_model) / "metadata" / "manifest.json"
    policy_centroids_out = Path(args.policy_centroids_out) if args.policy_centroids_out else scored_root / "policy_centroids.npy"
    policy_centroid_meta_out = Path(args.policy_centroid_meta_out) if args.policy_centroid_meta_out else scored_root / "metadata" / "policy_centroid_meta.json"
    consistency_out = Path(args.consistency_out) if args.consistency_out else scored_root / "metadata" / "centroid_consistency.json"

    if not args.overwrite and policy_centroids_out.exists() and policy_centroid_meta_out.exists() and consistency_out.exists():
        log.info("Skip — centroid consistency outputs already exist at %s", consistency_out)
        return

    # --- Load research centroids ---
    log.info("Loading research centroids: %s", research_centroids_path)
    research_centroids = np.load(research_centroids_path).astype(np.float32)
    log.info("  Shape: %s", research_centroids.shape)

    # --- Research consistency check ---
    research_result = check_research(research_manifest_path, research_centroids, scored_root / "paper_scores_shards")
    log.info(
        "Research overall agreement: %.4f",
        research_result["overall_agreement_rate"],
    )

    # --- Load policy data ---
    log.info("Loading policy embeddings: %s", policy_emb_path)
    policy_emb = np.load(policy_emb_path).astype(np.float32)
    log.info("  Shape: %s", policy_emb.shape)

    log.info("Loading policy scores: %s", policy_scores_path)
    policy_scores = np.load(policy_scores_path).astype(np.float32)
    log.info("  Shape: %s", policy_scores.shape)

    # --- Compute and save policy centroids ---
    log.info("Computing policy centroids from LR-assigned segments")
    policy_centroids, policy_centroid_available, policy_centroid_meta = build_policy_centroids(
        policy_emb, policy_scores
    )
    n_available = int(policy_centroid_available.sum())
    log.info("  %d / %d SDGs have non-zero policy centroids", n_available, N_SDG)

    atomic_write_npy(policy_centroids_out, policy_centroids)
    if not policy_centroids_out.exists():
        raise RuntimeError(f"Failed to write {policy_centroids_out}")
    log.info("Saved: %s", policy_centroids_out)

    ensure_dir(policy_centroid_meta_out.parent)
    with policy_centroid_meta_out.open("w", encoding="utf-8") as f:
        json.dump(policy_centroid_meta, f, indent=2, ensure_ascii=False)
    log.info("Saved: %s", policy_centroid_meta_out)

    # --- Policy consistency check ---
    policy_result = check_policy(policy_emb, policy_scores, policy_centroids)
    log.info(
        "Policy overall agreement: %.4f",
        policy_result["overall_agreement_rate"],
    )

    # --- Write combined output ---
    output = {
        "created_at_utc": now_iso(),
        "research_centroids": str(research_centroids_path),
        "policy_centroids": str(policy_centroids_out),
        "research": research_result,
        "policy": policy_result,
    }

    ensure_dir(consistency_out.parent)
    with consistency_out.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    log.info("Saved: %s", consistency_out)

    log.info("=== Summary ===")
    log.info("Research overall agreement: %.4f", research_result["overall_agreement_rate"])
    log.info("Policy  overall agreement: %.4f", policy_result["overall_agreement_rate"])


if __name__ == "__main__":
    main()

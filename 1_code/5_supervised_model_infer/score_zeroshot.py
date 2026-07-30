"""
Zero-shot nearest-centroid scoring for the assignment-method sensitivity axis.

Computes per-SDG semantic gaps under zero-shot nearest-centroid assignment
using the same post-fix reference centroids (sdg_centroids.npy) that the
LR classifier uses as its class means.

Outputs: {output_dir}/main/{model}/zeroshot/semantic_gap_distances.json  (per-SDG gaps)
          {output_dir}/main/{model}/zeroshot/research_centroids.npy      (per-SDG mean of zs-assigned papers)
          {output_dir}/main/{model}/zeroshot/policy_centroids.npy        (per-SDG mean of zs-assigned segments)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = ROOT / "1_code"
ANALYSIS_ROOT = CODE_ROOT / "7_main_analysis"
SHARED_DIR = ANALYSIS_ROOT / "0_shared"
for path in (CODE_ROOT, SHARED_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from model_utils import DEFAULT_EMBED_MODEL, N_SDG, RANDOM_SEED, ZERO_NORM_EPS, MIN_CENTROID_NORM, embed_dir_for_model, embed_research_dir_for_model, output_main_dir_for_model, scored_dir_for_model, preprocessed_dir, resolve_model_alias
from shard_pipeline_utils import load_json, resolve_manifest_path

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def centroid_from_sumcount(sums: np.ndarray, counts: np.ndarray) -> np.ndarray:
    out = np.zeros_like(sums)
    for i in range(sums.shape[0]):
        if counts[i] > 0:
            raw = sums[i] / counts[i]
            norm = float(np.linalg.norm(raw))
            if norm > ZERO_NORM_EPS:
                out[i] = (raw / norm).astype(np.float32)
    return out


def cap_indices_per_doc(policy_ids: list[dict], segment_cap: int, rng: np.random.Generator) -> list[int]:
    doc_to_indices: dict[str, list[int]] = {}
    for i, row in enumerate(policy_ids):
        doc_to_indices.setdefault(row["source_doc"], []).append(i)
    selected: list[int] = []
    for indices in doc_to_indices.values():
        if len(indices) <= segment_cap:
            selected.extend(indices)
        else:
            selected.extend(rng.choice(indices, size=segment_cap, replace=False).tolist())
    return sorted(selected)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL, type=resolve_model_alias)
    p.add_argument("--segment-cap", type=int, default=50,
                  help="Max segments sampled per source_doc per SDG (default: %(default)s)")
    p.add_argument("--min-centroid-norm", type=float, default=MIN_CENTROID_NORM,
                  help="Centroids with L2 norm below this are treated as degenerate and "
                       "excluded from semantic-gap (default: %(default)s)")
    p.add_argument("--output-dir", default="4_outputs")
    p.add_argument("--overwrite", action="store_true",
                   help="Recompute zero-shot centroids even if outputs already exist.")
    return p.parse_args()


def run(args: argparse.Namespace) -> None:
    model = args.embed_model

    out_root = output_main_dir_for_model(model, root=Path(args.output_dir)) / "zeroshot"
    out_root.mkdir(parents=True, exist_ok=True)

    if not args.overwrite:
        expected = [
            out_root / "research_centroids.npy",
            out_root / "policy_centroids.npy",
            out_root / "semantic_gap_distances.json",
        ]
        if all(p.exists() for p in expected):
            log.info("Zero-shot outputs already exist at %s — skip. Use --overwrite to recompute.", out_root)
            return

    # 1. Load reference centroids (same ones LR uses)
    centroids_path = scored_dir_for_model(model) / "sdg_centroids.npy"
    log.info("Loading reference centroids: %s", centroids_path)
    centroids = np.load(centroids_path).astype(np.float32)
    assert centroids.shape[0] == N_SDG
    embed_dim = centroids.shape[1]
    norms = np.linalg.norm(centroids, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5), f"Centroids not unit: {norms}"

    embed_root = embed_dir_for_model(model)

    # 2. Score research papers — accumulate per-SDG sums and counts.
    # Score research papers — load each shard's embedding directly (shard-native,
    # mmap). This is byte-identical to the former consolidated-array slice.
    manifest_path = embed_root / "research_shards" / "metadata" / "manifest.json"
    manifest = load_json(manifest_path)
    shards = sorted(manifest["shards"], key=lambda x: int(x["shard_id"]))
    log.info("Scoring %d research shards (zero-shot)...", len(shards))

    res_sums = np.zeros((N_SDG, embed_dim), dtype=np.float64)
    res_counts = np.zeros(N_SDG, dtype=np.int64)

    for shard in shards:
        emb = np.load(
            resolve_manifest_path(
                shard["embedding_path"],
                allowed_dirs=(embed_research_dir_for_model(model), scored_dir_for_model(model), preprocessed_dir()),
            ),
            mmap_mode="r",
        )
        embeddings = np.asarray(emb).astype(np.float32)
        scores = embeddings @ centroids.T
        assignments = scores.argmax(axis=1)
        for sdg_idx in range(N_SDG):
            mask = assignments == sdg_idx
            n = int(mask.sum())
            if n > 0:
                res_sums[sdg_idx] += embeddings[mask].sum(axis=0).astype(np.float64)
                res_counts[sdg_idx] += n
        log.info("  Shard %s done", shard.get("name", shard["shard_id"]))
        del embeddings, scores, assignments

    log.info("Research per-SDG counts: %s", res_counts.tolist())
    log.info("Research total papers: %d", res_counts.sum())

    research_centroids = centroid_from_sumcount(res_sums, res_counts)
    np.save(out_root / "research_centroids.npy", research_centroids)
    log.info("Saved research centroids: %s", out_root / "research_centroids.npy")

    # 3. Score policy corpus
    log.info("Scoring policy corpus (zero-shot)...")
    policy_emb = np.load(embed_root / "policy.npy").astype(np.float32)
    policy_ids = load_json(embed_root / "metadata" / "policy_ids.json")

    policy_scores = policy_emb @ centroids.T
    policy_assignments = policy_scores.argmax(axis=1)

    # Apply segment cap
    rng = np.random.Generator(np.random.PCG64(RANDOM_SEED))
    capped_idxs = cap_indices_per_doc(policy_ids, args.segment_cap, rng)
    log.info("Policy: %d total segments, %d capped", len(policy_ids), len(capped_idxs))

    pol_sums = np.zeros((N_SDG, embed_dim), dtype=np.float64)
    pol_counts = np.zeros(N_SDG, dtype=np.int64)
    for i in capped_idxs:
        sdg_idx = policy_assignments[i]
        pol_sums[sdg_idx] += policy_emb[i].astype(np.float64)
        pol_counts[sdg_idx] += 1

    policy_centroids = centroid_from_sumcount(pol_sums, pol_counts)
    np.save(out_root / "policy_centroids.npy", policy_centroids)
    log.info("Saved policy centroids: %s", out_root / "policy_centroids.npy")

    # 4. Compute semantic gaps
    per_sdg = []
    for sdg_idx in range(N_SDG):
        r = research_centroids[sdg_idx]
        p = policy_centroids[sdg_idx]
        rn = float(np.linalg.norm(r))
        pn = float(np.linalg.norm(p))
        if rn < args.min_centroid_norm or pn < args.min_centroid_norm:
            sim = None
            gap = None
        else:
            sim = float(r @ p)
            gap = 1.0 - sim
        per_sdg.append({
            "sdg": sdg_idx + 1,
            "n_papers": int(res_counts[sdg_idx]),
            "n_policy_capped": int(pol_counts[sdg_idx]),
            "semantic_similarity": sim,
            "semantic_gap": gap,
        })
        log.info("  SDG %2d  gap=%s  n_res=%d  n_pol=%d",
                 sdg_idx + 1, f"{gap:.4f}" if gap is not None else "N/A",
                 res_counts[sdg_idx], pol_counts[sdg_idx])

    out_data = {
        "method": "zeroshot_nearest_centroid",
        "segment_cap": args.segment_cap,
        "embedding_model": model,
        "per_sdg": per_sdg,
    }
    gap_path = out_root / "semantic_gap_distances.json"
    with gap_path.open("w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2)
    log.info("Saved: %s", gap_path)
    log.info("Zero-shot scoring complete.")


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()

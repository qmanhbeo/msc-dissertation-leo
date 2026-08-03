"""
Zero-shot nearest-centroid scoring for the assignment-method sensitivity axis.

Computes per-SDG semantic gaps under zero-shot nearest-centroid assignment
using the same post-fix reference centroids (sdg_centroids.npy) that the
LR classifier uses as its class means.

Outputs: {data_dir}/semantic_gap_distances_zeroshot.json  (per-SDG gaps, default {output_dir}/{model}/data/)
          {out_dir}/research_centroids.npy       (per-SDG mean of zs-assigned papers, default 2_data/.../zeroshot/)
          {out_dir}/policy_centroids.npy         (per-SDG mean of zs-assigned segments, default 2_data/.../zeroshot/)
"""

from __future__ import annotations

import argparse
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

from model_utils import DEFAULT_EMBED_MODEL, N_SDG, RANDOM_SEED, ZERO_NORM_EPS, MIN_CENTROID_NORM, embed_dir_for_model, output_dir_for_model, scored_dir_for_model, preprocessed_dir, resolve_model_alias
from semantic_gap_shared import compute_sdg_semantic_gaps
from shard_pipeline_utils import atomic_write_json, atomic_write_npy, load_json, resolve_manifest_path

# register_utils may not be on path when run standalone; add it.
import sys as _sys
_ANALYSIS_ROOT = Path(__file__).resolve().parents[1] / "7_main_analysis" / "0_shared"
if str(_ANALYSIS_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ANALYSIS_ROOT))
import register_utils

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
    p.add_argument("--embedding-manifest", default=None,
                   help="Override research embedding manifest (default: canonical "
                        "research_shards/metadata/manifest.json). Used for the "
                        "concept-retrieval variant.")
    p.add_argument("--out-dir", default=None,
                   help="Override zero-shot .npy output dir (default: canonical "
                        "2_data/5_supervised_scored/{model}/zeroshot/). "
                        "Concept variant writes to .../zeroshot_concept/.")
    p.add_argument("--data-dir", default=None,
                   help="Override semantic_gap_distances_zeroshot.json output dir (default: "
                        "canonical 4_outputs/{model}/data/). "
                        "Concept variant writes to .../data/concept/.")
    p.add_argument("--embeddings", choices=["raw", "adjusted"], default="raw",
                   help="Use raw (default) or register-adjusted embeddings (project via G).")
    return p.parse_args()


def run(args: argparse.Namespace) -> None:
    model = args.embed_model
    output_root = Path(args.output_dir)

    # .npy files → 2_data/5_supervised_scored/{model}/zeroshot/ (or custom)
    npy_root = Path(args.out_dir) if args.out_dir else scored_dir_for_model(model) / "zeroshot"
    npy_root.mkdir(parents=True, exist_ok=True)

    # semantic_gap_distances_zeroshot.json → 4_outputs/{model}/data/ (or custom)
    data_root = Path(args.data_dir) if args.data_dir else output_dir_for_model(model, root=output_root) / "data"
    data_root.mkdir(parents=True, exist_ok=True)

    # ---- Adjusted mode: load G ----
    is_adjusted = args.embeddings == "adjusted"

    if not args.overwrite:
        expected_json = (
            data_root / "adjusted" / "semantic_gap_distances_zeroshot.json"
            if is_adjusted
            else data_root / "semantic_gap_distances_zeroshot.json"
        )
        expected = [
            npy_root / "research_centroids.npy",
            npy_root / "policy_centroids.npy",
            expected_json,
        ]
        if all(p.exists() for p in expected):
            log.info("Zero-shot outputs already exist — skip. Use --overwrite to recompute.")
            return

    G = None
    if is_adjusted:
        G = register_utils.load_G(model)
        log.info("Adjusted mode: G loaded (%d directions)", G.shape[0])

    # 1. Load reference centroids (same ones LR uses)
    centroids_path = scored_dir_for_model(model) / "sdg_centroids.npy"
    log.info("Loading reference centroids: %s", centroids_path)
    centroids = np.load(centroids_path).astype(np.float32)
    if is_adjusted:
        log.info("Projecting reference centroids through G...")
        centroids = register_utils.project(centroids, G)
    assert centroids.shape[0] == N_SDG
    embed_dim = centroids.shape[1]
    norms = np.linalg.norm(centroids, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5), f"Centroids not unit: {norms}"

    embed_root = embed_dir_for_model(model)

    # 2. Score research papers — accumulate per-SDG sums and counts.
    # Score research papers — load each shard's embedding directly (shard-native,
    # mmap). This is byte-identical to the former consolidated-array slice.
    manifest_path = Path(args.embedding_manifest) if args.embedding_manifest else embed_root / "research_shards" / "metadata" / "manifest.json"
    manifest = load_json(manifest_path)
    shards = sorted(manifest["shards"], key=lambda x: int(x["shard_id"]))
    log.info("Scoring %d research shards (zero-shot)...", len(shards))

    res_sums = np.zeros((N_SDG, embed_dim), dtype=np.float64)
    res_counts = np.zeros(N_SDG, dtype=np.int64)
    res_cohesion_sums = np.zeros(N_SDG, dtype=np.float64)
    res_cohesion_counts = np.zeros(N_SDG, dtype=np.int64)

    for shard in shards:
        emb = np.load(
            resolve_manifest_path(
                shard["embedding_path"],
                allowed_dirs=(embed_dir_for_model(model), scored_dir_for_model(model), preprocessed_dir()),
            ),
            mmap_mode="r",
        )
        embeddings = np.asarray(emb).astype(np.float32)
        if is_adjusted:
            # Adjusted ZS: project research embeddings through G and RE-ASSIGN on
            # the projected space (intentional; PLAN_register_topic_decomposition
            # §6.1). LR/MLP keep raw-space clusters and only project vectors.
            embeddings = register_utils.project(embeddings, G)
        scores = embeddings @ centroids.T
        assignments = scores.argmax(axis=1)
        for sdg_idx in range(N_SDG):
            mask = assignments == sdg_idx
            n = int(mask.sum())
            if n > 0:
                res_sums[sdg_idx] += embeddings[mask].sum(axis=0).astype(np.float64)
                res_counts[sdg_idx] += n
                # Mean cosine of assigned research vectors to their reference
                # centroid = mean of the assignment score column for that SDG.
                res_cohesion_sums[sdg_idx] += float(scores[mask, sdg_idx].sum())
                res_cohesion_counts[sdg_idx] += n
        log.info("  Shard %s done", shard.get("name", shard["shard_id"]))
        del embeddings, scores, assignments

    log.info("Research per-SDG counts: %s", res_counts.tolist())
    log.info("Research total papers: %d", res_counts.sum())

    research_centroids = centroid_from_sumcount(res_sums, res_counts)
    research_cohesions = np.zeros(N_SDG, dtype=np.float32)
    for sdg_idx in range(N_SDG):
        n = int(res_cohesion_counts[sdg_idx])
        research_cohesions[sdg_idx] = float(res_cohesion_sums[sdg_idx] / n) if n > 0 else 0.0
    atomic_write_npy(npy_root / "research_centroids.npy", research_centroids)
    log.info("Saved research centroids: %s", npy_root / "research_centroids.npy")

    # 3. Score policy corpus
    log.info("Scoring policy corpus (zero-shot)...")
    policy_emb = np.load(embed_root / "policy.npy").astype(np.float32)
    if is_adjusted:
        policy_emb = register_utils.project(policy_emb, G)
    policy_ids = load_json(embed_root / "metadata" / "policy_ids.json")
    if policy_emb.shape[0] != len(policy_ids):
        raise ValueError(
            f"policy_emb has {policy_emb.shape[0]} rows but {len(policy_ids)} policy_ids"
        )

    policy_scores = policy_emb @ centroids.T
    policy_assignments = policy_scores.argmax(axis=1)

    # 4. Compute semantic gaps via the shared LR/MLP route so all three scoring
    # methods use one gap implementation (single source of truth). The shared
    # function applies the per-(SDG, doc) segment cap (matching the supervised
    # routes), MIN_CLUSTER_SIZE and --min-centroid-norm reliability rules, and
    # returns the per-SDG policy sub-centroid built from that same capped set.
    rng = np.random.Generator(np.random.PCG64(RANDOM_SEED))
    per_sdg, policy_centroids = compute_sdg_semantic_gaps(
        research_centroids=research_centroids,
        research_counts=res_counts,
        research_cohesions=research_cohesions,
        policy_emb=policy_emb,
        policy_assignments=policy_assignments,
        policy_ids=policy_ids,
        segment_cap=args.segment_cap,
        rng=rng,
        min_centroid_norm=args.min_centroid_norm,
    )
    atomic_write_npy(npy_root / "policy_centroids.npy", policy_centroids)
    log.info("Saved policy centroids: %s", npy_root / "policy_centroids.npy")
    for row in per_sdg:
        sdg_idx = row["sdg"] - 1
        gap = row["semantic_gap"]
        n_papers = row["n_papers"]
        n_policy_capped = row["n_policy_segments_capped"]
        log.info("  SDG %2d  gap=%s  n_res=%d  n_pol=%d%s",
                 row["sdg"], f"{gap:.4f}" if gap is not None else "N/A",
                 n_papers, n_policy_capped,
                 "  [unreliable]" if row["unreliable"] else "")

    out_data = {
        "method": "zeroshot_nearest_centroid",
        "embedding_model": model,
        "embeddings": args.embeddings,
        "segment_cap": args.segment_cap,
        # Documented assignment-space rule (PLAN_register_topic_decomposition.md §6.1):
        # raw ZS assigns on raw embeddings, while adjusted ZS projects research
        # texts and policy centroids through G and RE-ASSIGNS on the projected
        # embeddings (clusters in projected space). This is intentional design,
        # not a bug; only LR/MLP keep raw-space clusters and merely project the
        # vectors for the gap.
        "note": (
            "Zero-shot assigns in the space of the embeddings used: raw gaps assign on "
            "raw embeddings; adjusted gaps project research texts and policy centroids "
            "through G and re-assign on projected embeddings (PLAN_register_topic_"
            "decomposition.md §6.1). Policy segments are capped at --segment-cap per "
            "(source_doc, SDG), matching the LR/MLP routes."
        ),
        "per_sdg": per_sdg,
    }
    if is_adjusted:
        gap_dir = data_root / "adjusted"
        gap_dir.mkdir(parents=True, exist_ok=True)
        gap_path = gap_dir / "semantic_gap_distances_zeroshot.json"
    else:
        gap_path = data_root / "semantic_gap_distances_zeroshot.json"
    atomic_write_json(gap_path, out_data)
    log.info("Saved: %s", gap_path)
    log.info("Zero-shot scoring complete.")


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()

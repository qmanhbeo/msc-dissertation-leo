"""
Build per-SDG reference centroids from the pooled labelled corpus.

Reads the shared embeddings/labels/split files that the supervised classifier
pipeline already uses (2_data/4_supervised_model_results/{model}/), pools all
5 labelled sources (OSDG, Benchmark, Knowledge Hub, SDGi, Aurora), and
optionally restricts to the training split to avoid test-set leakage.

Row ordering convention (critical for ALL downstream scripts):
  centroids[i] = centroid for SDG (i + 1)

Two variants:
  train_only (default, reportable): centroids from train split only (52,779 texts)
  full_pool  (diagnostic only):     centroids from all 62,513 texts
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
ANALYSIS_DIR = CODE_ROOT / "7_main_analysis" / "0_shared"
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from model_utils import DEFAULT_EMBED_MODEL, N_SDG, ZERO_NORM_EPS, model_results_dir_for_model, scored_dir_for_model
from shard_pipeline_utils import atomic_write_npy

COHESION_WARN_THRESHOLD = 0.50

log = logging.getLogger(__name__)

SOURCE_NAMES = ["osdg", "benchmark", "sdg_knowledge_hub", "sdgi", "aurora"]


def build_centroid(emb: np.ndarray, n: int, sdg: int) -> tuple[np.ndarray, dict]:
    raw = emb.mean(axis=0)
    norm = float(np.linalg.norm(raw))
    if norm < ZERO_NORM_EPS:
        raise ValueError(f"SDG {sdg}: near-zero centroid norm")
    unit = (raw / norm).astype(np.float32)
    mean_cos = float((emb @ unit).mean()) if n > 0 else 0.0
    high_variance = mean_cos < COHESION_WARN_THRESHOLD
    meta = {
        "sdg": sdg,
        "n": n,
        "raw_centroid_norm": round(norm, 6),
        "mean_cos_to_centroid": round(mean_cos, 6),
        "high_variance_flag": high_variance,
    }
    return unit, meta


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build SDG reference centroids from pooled labelled corpus."
    )
    parser.add_argument(
        "--embed-model", default=DEFAULT_EMBED_MODEL,
        help=f"Embed model (default: {DEFAULT_EMBED_MODEL})",
    )
    parser.add_argument(
        "--variant", choices=["train_only", "full_pool"], default="train_only",
        help="train_only (reportable, uses train split, no test leakage) or full_pool (diagnostic only, uses all texts)",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing centroids")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    results_root = model_results_dir_for_model(args.embed_model)
    scored_root = scored_dir_for_model(args.embed_model)

    if args.variant == "train_only":
        centroids_out = scored_root / "sdg_centroids.npy"
    else:
        centroids_out = scored_root / "sdg_centroids_fullpool_diagnostic.npy"
    meta_out = scored_root / "metadata" / f"sdg_centroid_meta_{args.variant}.json"
    meta_out.parent.mkdir(parents=True, exist_ok=True)

    if centroids_out.exists() and not args.overwrite:
        log.info("Reference centroids already exist: %s (use --overwrite to rebuild)", centroids_out)
        return

    emb_path = results_root / "embeddings.npy"
    labels_path = results_root / "labels.npy"
    sources_path = results_root / "sources.npy"

    for p in [emb_path, labels_path, sources_path]:
        if not p.exists():
            raise FileNotFoundError(f"Required input not found: {p}")

    log.info("Loading embeddings: %s", emb_path)
    embeddings = np.load(emb_path).astype(np.float32)
    log.info("  shape=%s  dtype=%s", embeddings.shape, embeddings.dtype)

    log.info("Loading labels: %s", labels_path)
    labels = np.load(labels_path).astype(np.float32)
    log.info("  shape=%s", labels.shape)

    log.info("Loading sources: %s", sources_path)
    sources = np.load(sources_path)
    log.info("  shape=%s  dtype=%s", sources.shape, sources.dtype)

    if not (embeddings.shape[0] == labels.shape[0] == sources.shape[0]):
        raise RuntimeError(
            f"Row count mismatch: emb={embeddings.shape[0]} labels={labels.shape[0]} sources={sources.shape[0]}"
        )

    split_desc = "all texts (62,513)"
    if args.variant == "train_only":
        train_path = results_root / "indices" / "train.npy"
        if not train_path.exists():
            raise FileNotFoundError(f"Train split not found: {train_path}")
        train_indices = np.load(train_path)
        log.info("Loaded train indices: %d", len(train_indices))
        embeddings = embeddings[train_indices]
        labels = labels[train_indices]
        sources = sources[train_indices]
        split_desc = f"train split only ({len(train_indices)} texts)"

    total_n = embeddings.shape[0]
    log.info("Using %s", split_desc)

    sample_norms = np.linalg.norm(embeddings[:20], axis=1)
    if not np.allclose(sample_norms, 1.0, atol=1e-4):
        log.warning("Embeddings may not be L2-normalised (norms: %s)", sample_norms[:5])
    else:
        log.info("Embedding norms verified ~= 1.0 (L2-normalised)")

    log.info("")
    log.info("Building centroids...")
    log.info("%-8s %-7s %-12s %-10s %s", "SDG", "n", "source_counts", "raw_norm", "cohesion")
    log.info("-" * 65)

    centroid_vectors: list[np.ndarray] = []
    centroid_meta: list[dict] = []

    for sdg in range(1, N_SDG + 1):
        mask = labels[:, sdg - 1] == 1.0
        idxs = np.flatnonzero(mask)
        n = len(idxs)
        if n == 0:
            raise RuntimeError(
                f"SDG {sdg}: no texts found in "
                f"{'train split' if args.variant == 'train_only' else 'full pool'}"
            )

        emb_sdg = embeddings[idxs]

        source_counts: dict[str, int] = {}
        for src in SOURCE_NAMES:
            source_counts[src] = int((sources[idxs] == src).sum())

        vec, meta = build_centroid(emb_sdg, n, sdg)
        meta["source_counts"] = source_counts
        centroid_vectors.append(vec)
        centroid_meta.append(meta)

        src_str = str(source_counts)
        level = logging.WARNING if meta["high_variance_flag"] else logging.INFO
        log.log(
            level,
            "SDG %2d | n=%5d | %s | norm=%.4f | cohesion=%.4f%s",
            sdg, meta["n"], src_str, meta["raw_centroid_norm"],
            meta["mean_cos_to_centroid"],
            " [HIGH VARIANCE]" if meta["high_variance_flag"] else "",
        )

    centroids = np.stack(centroid_vectors, axis=0)
    assert centroids.shape == (N_SDG, centroids.shape[1]), (
        f"Unexpected centroid shape: {centroids.shape}"
    )

    norms = np.linalg.norm(centroids, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-5):
        log.warning("Post-normalisation centroid norms not all ~= 1.0: %s", norms)
    else:
        log.info("All %d centroid norms ~= 1.0", N_SDG)

    atomic_write_npy(centroids_out, centroids)
    log.info("Saved: %s  shape=%s", centroids_out, centroids.shape)

    global_meta = {
        "variant": args.variant,
        "embedding_model": args.embed_model,
        "sources_pooled": list(SOURCE_NAMES),
        "total_texts": int(total_n),
        "split_source": "indices/train.npy" if args.variant == "train_only" else "none (all texts)",
        "test_held_out": args.variant == "train_only",
        "note": (
            "Reportable only for train_only variant. "
            "full_pool is diagnostic only — contains test-set leakage."
        ),
        "row_ordering": "centroids[i] = SDG (i + 1)",
        "embedding_dim": int(centroids.shape[1]),
        "per_sdg": centroid_meta,
    }
    with open(meta_out, "w") as f:
        json.dump(global_meta, f, indent=2)
    log.info("Saved metadata: %s", meta_out)

    high_var = [m["sdg"] for m in centroid_meta if m["high_variance_flag"]]
    if high_var:
        log.warning("High-variance SDGs (cohesion < %.2f): %s", COHESION_WARN_THRESHOLD, high_var)

    log.info("Row ordering: centroids[i] = SDG (i+1)")


if __name__ == "__main__":
    main()

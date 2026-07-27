"""
Build per-SDG reference centroids from labelled corpora (OSDG + Benchmark).

SDG centroids are the core measurement instrument. Every downstream analysis
scores texts against these centroids via cosine similarity.

Sources:
  SDGs 1-16 -> OSDG Community Dataset (agreement >= 0.5)
  SDG 17    -> SDG Classification Benchmark (expert-labelled)

Row ordering convention (critical for ALL downstream scripts):
  centroids[i] = centroid for SDG (i + 1)
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

from model_utils import DEFAULT_EMBED_MODEL, N_SDG, embed_dir_for_model, preprocessed_dir, scored_dir_for_model

COHESION_WARN_THRESHOLD = 0.50

log = logging.getLogger(__name__)


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def build_centroid(emb: np.ndarray, idxs: list[int], sdg: int, source: str) -> tuple[np.ndarray, dict]:
    vecs = emb[idxs]
    raw = vecs.mean(axis=0)
    norm = float(np.linalg.norm(raw))
    if norm < 1e-8:
        raise ValueError(f"SDG {sdg}: near-zero centroid norm")
    unit = (raw / norm).astype(np.float32)
    mean_cos = float((vecs @ unit).mean())
    high_variance = mean_cos < COHESION_WARN_THRESHOLD
    meta = {
        "sdg": sdg,
        "n": len(idxs),
        "source": source,
        "raw_centroid_norm": round(norm, 6),
        "mean_cos_to_centroid": round(mean_cos, 6),
        "high_variance_flag": high_variance,
    }
    return unit, meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Build SDG reference centroids from OSDG + Benchmark.")
    parser.add_argument("--model", default=DEFAULT_EMBED_MODEL, help=f"Embed model (default: {DEFAULT_EMBED_MODEL})")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing centroids")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    embed_root = embed_dir_for_model(args.model)
    scored_root = scored_dir_for_model(args.model)

    centroids_out = scored_root / "sdg_centroids.npy"
    meta_out = scored_root / "metadata" / "sdg_centroid_meta.json"
    meta_out.parent.mkdir(parents=True, exist_ok=True)

    if centroids_out.exists() and not args.overwrite:
        log.info("Reference centroids already exist: %s (use --overwrite to rebuild)", centroids_out)
        log.info("Metadata: %s", meta_out)
        return

    osdg_emb_path = embed_root / "osdg.npy"
    bench_emb_path = embed_root / "benchmark.npy"
    osdg_source_path = preprocessed_dir() / "osdg" / "osdg_clean.jsonl"
    bench_source_path = preprocessed_dir() / "sdg_benchmark" / "benchmark_clean.jsonl"

    for p in [osdg_emb_path, bench_emb_path, osdg_source_path, bench_source_path]:
        if not p.exists():
            raise FileNotFoundError(f"Required input not found: {p}")

    log.info("Loading OSDG embeddings: %s", osdg_emb_path)
    osdg_emb = np.load(osdg_emb_path)
    log.info("  shape=%s  dtype=%s", osdg_emb.shape, osdg_emb.dtype)

    log.info("Loading OSDG labels from source: %s", osdg_source_path)
    osdg_records = load_jsonl(osdg_source_path)
    log.info("  records=%d", len(osdg_records))

    log.info("Loading benchmark embeddings: %s", bench_emb_path)
    bench_emb = np.load(bench_emb_path)
    log.info("  shape=%s  dtype=%s", bench_emb.shape, bench_emb.dtype)

    log.info("Loading benchmark labels from source: %s", bench_source_path)
    bench_records = load_jsonl(bench_source_path)
    log.info("  records=%d", len(bench_records))

    sample_norms = np.linalg.norm(osdg_emb[:20], axis=1)
    if not np.allclose(sample_norms, 1.0, atol=1e-4):
        log.warning("OSDG embeddings may not be L2-normalised (norms: %s)", sample_norms[:5])
    else:
        log.info("Embedding norms verified ~= 1.0 (L2-normalised)")

    osdg_by_sdg: dict[int, list[int]] = {}
    for i, r in enumerate(osdg_records):
        sdg_label = r.get("sdgs")
        if sdg_label is None:
            sdg_label = r.get("sdg")
        if sdg_label is None:
            continue
        if isinstance(sdg_label, list):
            sdg_label = sdg_label[0]
        osdg_by_sdg.setdefault(int(sdg_label), []).append(i)

    bench_by_sdg: dict[int, list[int]] = {}
    for i, r in enumerate(bench_records):
        sdg_label = r.get("sdgs")
        if sdg_label is None:
            continue
        if isinstance(sdg_label, list):
            sdg_label = sdg_label[0]
        bench_by_sdg.setdefault(int(sdg_label), []).append(i)

    osdg_sdgs = sorted(osdg_by_sdg.keys())
    if osdg_sdgs != list(range(1, 17)):
        log.warning("Unexpected OSDG SDG labels: %s", osdg_sdgs)
    else:
        log.info("OSDG SDG coverage confirmed: 1-16")

    if 17 not in bench_by_sdg:
        raise RuntimeError("No SDG-17 texts found in benchmark")

    log.info("")
    log.info("Building centroids...")
    log.info("%-8s %-7s %-7s %-10s %-10s %s", "SDG", "n", "source", "raw_norm", "cohesion", "variance_flag")
    log.info("-" * 60)

    centroid_vectors: list[np.ndarray] = []
    centroid_meta: list[dict] = []

    for sdg in range(1, 17):
        idxs = osdg_by_sdg[sdg]
        vec, meta = build_centroid(osdg_emb, idxs, sdg, source="osdg")
        centroid_vectors.append(vec)
        centroid_meta.append(meta)
        level = logging.WARNING if meta["high_variance_flag"] else logging.INFO
        log.log(level, "SDG %2d | n=%5d | osdg      | norm=%.4f | cohesion=%.4f%s",
                sdg, meta["n"], meta["raw_centroid_norm"], meta["mean_cos_to_centroid"],
                " [HIGH VARIANCE]" if meta["high_variance_flag"] else "")

    sdg17_idxs = bench_by_sdg[17]
    vec17, meta17 = build_centroid(bench_emb, sdg17_idxs, 17, source="benchmark")
    centroid_vectors.append(vec17)
    centroid_meta.append(meta17)
    log.log(logging.WARNING if meta17["high_variance_flag"] else logging.INFO,
            "SDG 17 | n=%5d | benchmark | norm=%.4f | cohesion=%.4f%s",
            meta17["n"], meta17["raw_centroid_norm"], meta17["mean_cos_to_centroid"],
            " [HIGH VARIANCE]" if meta17["high_variance_flag"] else "")

    centroids = np.stack(centroid_vectors, axis=0)
    assert centroids.shape == (N_SDG, centroids.shape[1]), f"Unexpected centroid shape: {centroids.shape}"

    norms = np.linalg.norm(centroids, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-5):
        log.warning("Post-normalisation centroid norms not all ~= 1.0: %s", norms)
    else:
        log.info("All 17 centroid norms ~= 1.0")

    np.save(centroids_out, centroids)
    log.info("Saved: %s  shape=%s", centroids_out, centroids.shape)

    with open(meta_out, "w") as f:
        json.dump(centroid_meta, f, indent=2)
    log.info("Saved: %s", meta_out)

    high_var = [m["sdg"] for m in centroid_meta if m["high_variance_flag"]]
    if high_var:
        log.warning("High-variance SDGs (cohesion < %.2f): %s", COHESION_WARN_THRESHOLD, high_var)

    log.info("Row ordering: centroids[i] = SDG (i+1)")


if __name__ == "__main__":
    main()

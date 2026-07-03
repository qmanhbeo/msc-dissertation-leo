"""
Embed Aurora corpus and build per-SDG centroids.

Aurora centroids are built from research-domain expert-validated SDG labels
(Vanderfeesten et al., 2020). Unlike OSDG (UN-adjacent policy sources),
Aurora labels come from domain researchers validating research papers.

This script:
  1. Loads and embeds the Aurora corpus using all-MiniLM-L6-v2
  2. Builds per-SDG centroids (same method as canonical centroids)
  3. Validates centroids against the SDG Classification Benchmark
  4. Saves outputs for downstream gap comparison

Outputs:
  2_data/2_embedded/aurora.npy                     — (5619, 384) float32
  2_data/2_embedded/metadata/aurora_ids.json        — list of {doi, sdg, title}
  2_data/3_scored/aurora_centroids.npy              — (17, 384) float32 unit vectors
  2_data/3_scored/metadata/aurora_centroid_meta.json
  4_outputs/appendix/d1_aurora_centroids/          — tables + data
"""

import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 128

AURORA_INPUT = Path("2_data/1_raw/aurora/aurora_texts.jsonl")
EMBED_OUTPUT_DIR = Path("2_data/2_embedded")
EMBED_METADATA_DIR = EMBED_OUTPUT_DIR / "metadata"
CENTROID_DIR = Path("2_data/3_scored")
CENTROID_METADATA_DIR = CENTROID_DIR / "metadata"
OUT_NPY = EMBED_OUTPUT_DIR / "aurora.npy"
OUT_IDS = EMBED_METADATA_DIR / "aurora_ids.json"
OUT_CENTROIDS = CENTROID_DIR / "aurora_centroids.npy"
OUT_CENTROID_META = CENTROID_METADATA_DIR / "aurora_centroid_meta.json"

APPENDIX_DIR = Path("4_outputs/appendix/d1_aurora_centroids")
APPENDIX_TABLES = APPENDIX_DIR / "tables"
APPENDIX_DATA = APPENDIX_DIR / "data"

# Same validation benchmark used by the canonical instrument
BENCH_EMB = EMBED_OUTPUT_DIR / "benchmark.npy"
BENCH_IDS = EMBED_METADATA_DIR / "benchmark_ids.json"

MIN_TEXTS_PER_SDG = 100
COHESION_WARN_THRESHOLD = 0.50


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_centroid(emb: np.ndarray, idxs: list[int], sdg: int, source: str) -> tuple:
    """L2-normalised mean centroid (same as canonical build_centroid)."""
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


def reconstruct_abstract(inverted_index: dict | None) -> str | None:
    """Convert OpenAlex abstract_inverted_index to plain text."""
    if inverted_index is None:
        return None
    words = []
    for word, positions in inverted_index.items():
        for pos in positions:
            words.append((pos, word))
    words.sort()
    return " ".join(w for _, w in words)


def main():
    APPENDIX_TABLES.mkdir(parents=True, exist_ok=True)
    APPENDIX_DATA.mkdir(parents=True, exist_ok=True)
    EMBED_METADATA_DIR.mkdir(parents=True, exist_ok=True)
    CENTROID_METADATA_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Step 1: Load Aurora corpus ----
    log.info("Loading Aurora corpus: %s", AURORA_INPUT)
    records = load_jsonl(AURORA_INPUT)
    log.info("  %d records loaded", len(records))

    texts = [r["text"] for r in records]
    # We'll build sdg array from the records
    # Validate per-SDG counts
    per_sdg = defaultdict(int)
    for r in records:
        per_sdg[r["sdg"]] += 1

    log.info("  Per-SDG counts (Aurora accepted, OpenAlex-matched):")
    for sdg in sorted(per_sdg):
        flag = " ⚠️ too few" if per_sdg[sdg] < MIN_TEXTS_PER_SDG else ""
        log.info("    SDG %2d: %d texts%s", sdg, per_sdg[sdg], flag)

    # ---- Step 2: Embed ----
    log.info("Loading model: %s", MODEL_NAME)
    model = SentenceTransformer(MODEL_NAME)

    log.info("Embedding %d texts...", len(texts))
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    embeddings = np.asarray(embeddings, dtype=np.float32)
    log.info("  Embeddings shape: %s", embeddings.shape)

    # Save embeddings
    np.save(OUT_NPY, embeddings)
    log.info("Saved: %s", OUT_NPY)

    ids_out = [{"doi": r["doi"], "sdg": r["sdg"], "title": r.get("title", "")} for r in records]
    with open(OUT_IDS, "w", encoding="utf-8") as f:
        json.dump(ids_out, f, indent=2)
    log.info("Saved: %s", OUT_IDS)

    # ---- Step 3: Build centroids ----
    log.info("Building Aurora centroids...")
    by_sdg: dict[int, list[int]] = defaultdict(list)
    for i, r in enumerate(records):
        by_sdg[r["sdg"]].append(i)

    centroid_vectors = []
    centroid_meta = []

    for sdg in range(1, 18):
        idxs = by_sdg.get(sdg, [])
        n = len(idxs)
        if n < MIN_TEXTS_PER_SDG:
            log.warning("SDG %2d: only %d texts — centroid will be unstable (min=%d)", sdg, n, MIN_TEXTS_PER_SDG)

        if n == 0:
            log.warning("SDG %2d: no texts — storing zero vector with unreliable flag", sdg)
            vec = np.zeros(384, dtype=np.float32)
            meta = {
                "sdg": sdg,
                "n": 0,
                "source": "aurora",
                "raw_centroid_norm": 0.0,
                "mean_cos_to_centroid": None,
                "high_variance_flag": True,
                "unreliable": True,
                "unreliable_reason": "no_texts",
            }
            centroid_vectors.append(vec)
            centroid_meta.append(meta)
            continue

        vec, meta = build_centroid(embeddings, idxs, sdg, source=f"aurora({n})")
        meta["unreliable"] = n < MIN_TEXTS_PER_SDG
        if n < MIN_TEXTS_PER_SDG:
            meta["unreliable_reason"] = f"only_{n}_texts_below_min_{MIN_TEXTS_PER_SDG}"
        centroid_vectors.append(vec)
        centroid_meta.append(meta)

        flag = " [UNRELIABLE]" if meta["unreliable"] else ""
        log.log(logging.WARNING if meta["unreliable"] else logging.INFO,
                "SDG %2d | n=%4d | cohesion=%.4f%s", sdg, n, meta["mean_cos_to_centroid"] or 0, flag)

    centroids = np.stack(centroid_vectors, axis=0)
    assert centroids.shape == (17, 384)
    np.save(OUT_CENTROIDS, centroids)
    log.info("Saved: %s", OUT_CENTROIDS)

    with open(OUT_CENTROID_META, "w", encoding="utf-8") as f:
        json.dump(centroid_meta, f, indent=2)
    log.info("Saved: %s", OUT_CENTROID_META)

    # ---- Step 4: Validate against benchmark ----
    log.info("Validating Aurora centroids against SDG Classification Benchmark...")
    if BENCH_EMB.exists() and BENCH_IDS.exists():
        bench_emb = np.load(BENCH_EMB)
        bench_ids = load_json(BENCH_IDS)

        bench_sdg_labels = [r["sdg"] for r in bench_ids]
        bench_sdg_indices = np.array([s - 1 for s in bench_sdg_labels], dtype=np.int64)

        # Score each benchmark text against all Aurora centroids
        bench_scores = bench_emb @ centroids.T  # (N_bench, 17)
        pred_indices = bench_scores.argmax(axis=1)
        pred_sdgs = pred_indices + 1

        # Per-SDG F1
        from sklearn.metrics import f1_score
        macro_f1 = f1_score(bench_sdg_labels, pred_sdgs, average="macro")
        per_sdg_f1 = f1_score(bench_sdg_labels, pred_sdgs, average=None, labels=list(range(1, 18)))

        log.info("  Macro-F1: %.4f (canonical: see validation table for comparison)", macro_f1)
        for i, f1 in enumerate(per_sdg_f1):
            sdg = i + 1
            if f1 > 0:
                log.info("    SDG %2d F1: %.4f", sdg, f1)
            else:
                log.info("    SDG %2d F1: %.4f (unreliable centroid)", sdg, f1)

        # Write validation to appendix
        val_lines = [
            "% Auto-generated by 3_build_aurora_centroids.py",
            rf"\newcommand{{\AuroraMacroFOne}}{{{macro_f1:.3f}}}",
        ]
        sdg_num_words = {
            1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five",
            6: "Six", 7: "Seven", 8: "Eight", 9: "Nine", 10: "Ten",
            11: "Eleven", 12: "Twelve", 13: "Thirteen", 14: "Fourteen",
            15: "Fifteen", 16: "Sixteen", 17: "Seventeen",
        }
        for i, f1 in enumerate(per_sdg_f1):
            word = sdg_num_words[i + 1]
            val_lines.append(rf"\newcommand{{\AuroraFOneSdg{word}}}{{{f1:.3f}}}")

        (APPENDIX_TABLES / "num_aurora_validation.tex").write_text(
            "\n".join(val_lines) + "\n", encoding="utf-8")
        log.info("Saved: %s", APPENDIX_TABLES / "num_aurora_validation.tex")
    else:
        log.warning("Benchmark not found — skipping validation (run 0_embed_reference_corpora.py first)")

    # ---- Step 5: Save per-SDG counts to appendix ----
    sdg_counts = {str(sdg): {"total": per_sdg.get(sdg, 0)} for sdg in range(1, 18)}
    with open(APPENDIX_DATA / "aurora_sdg_counts.json", "w") as f:
        json.dump(sdg_counts, f, indent=2)

    log.info("Done. Next step: d2_aurora_semantic_gap.py")


def load_json(path: Path) -> list:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    main()

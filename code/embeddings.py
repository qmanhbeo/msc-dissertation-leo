"""
Generate Sentence-BERT embeddings for all four corpora.

Model: all-MiniLM-L6-v2 (384-dim, 5x faster than mpnet — practical choice for CPU/WSL)
       Change MODEL_NAME to "all-mpnet-base-v2" for 768-dim higher-quality embeddings (GPU recommended).

Inputs:
  data/openalex/papers_clean.jsonl       (94 texts,    field: combined_text)
  data/un_sdg/policy_chunks.jsonl        (253 texts,   field: text)
  data/osdg/osdg_clean.jsonl             (30,534 texts, field: text)
  data/sdg_benchmark/benchmark_clean.jsonl (616 texts, field: text)

Outputs per corpus (saved to data/embeddings/):
  <name>.npy       float32 matrix (n_texts, 768)
  <name>_ids.json  list of dicts with id, sdg (where available), text_field

Idempotent: skips a corpus if its .npy already exists (delete to re-embed).

Run from project root:
    python code/embeddings.py
"""

import json
import logging
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 128
OUTPUT_DIR = Path("data/embeddings")

CORPORA = [
    {
        "name": "papers",
        "input": Path("data/openalex/papers_clean.jsonl"),
        "text_field": "combined_text",
        "id_field": "openalex_id",
        "sdg_field": None,
    },
    {
        "name": "policy",
        "input": Path("data/policy_all/policy_chunks_extended.jsonl"),
        "text_field": "text",
        "id_field": "chunk_id",
        "sdg_field": None,
    },
    {
        "name": "osdg",
        "input": Path("data/osdg/osdg_clean.jsonl"),
        "text_field": "text",
        "id_field": "text_id",
        "sdg_field": "sdg",
    },
    {
        "name": "benchmark",
        "input": Path("data/sdg_benchmark/benchmark_clean.jsonl"),
        "text_field": "text",
        "id_field": "id",
        "sdg_field": "sdg",
    },
]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def embed_corpus(corpus: dict, model: SentenceTransformer) -> None:
    name = corpus["name"]
    emb_path = OUTPUT_DIR / f"{name}.npy"
    ids_path = OUTPUT_DIR / f"{name}_ids.json"

    if emb_path.exists():
        log.info("Skipping %s — %s already exists", name, emb_path)
        return

    log.info("Embedding corpus: %s (%s)", name, corpus["input"])
    records = load_jsonl(corpus["input"])
    texts = [r[corpus["text_field"]] for r in records]

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,   # L2-normalised → cosine sim = dot product
    )
    embeddings = embeddings.astype(np.float32)

    # Build ID metadata
    ids_meta = []
    for r in records:
        entry = {"id": r.get(corpus["id_field"], ""), "text": r[corpus["text_field"]]}
        if corpus["sdg_field"]:
            entry["sdg"] = r.get(corpus["sdg_field"])
        ids_meta.append(entry)

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(emb_path, embeddings)
    with ids_path.open("w") as f:
        json.dump(ids_meta, f)

    log.info("Saved %s → shape %s", emb_path, embeddings.shape)


def sanity_check() -> None:
    """Verify that same-SDG OSDG pairs score higher than different-SDG pairs."""
    log.info("Running sanity check on OSDG embeddings...")
    emb = np.load(OUTPUT_DIR / "osdg.npy")
    with open(OUTPUT_DIR / "osdg_ids.json") as f:
        ids = json.load(f)

    # Find two texts for SDG 13 and two for SDG 1
    sdg13 = [i for i, r in enumerate(ids) if r.get("sdg") == 13][:2]
    sdg1  = [i for i, r in enumerate(ids) if r.get("sdg") == 1][:2]

    if len(sdg13) < 2 or len(sdg1) < 2:
        log.warning("Not enough examples for sanity check")
        return

    same_sim = float(cosine_similarity(emb[sdg13[:1]], emb[sdg13[1:2]])[0][0])
    diff_sim = float(cosine_similarity(emb[sdg13[:1]], emb[sdg1[:1]])[0][0])

    log.info("Sanity check — same-SDG sim: %.3f  |  diff-SDG sim: %.3f", same_sim, diff_sim)
    if same_sim > diff_sim:
        log.info("PASS: same-SDG similarity > different-SDG similarity")
    else:
        log.warning("FAIL: embeddings may not separate SDGs well — check model choice")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    log.info("Loading model: %s", MODEL_NAME)
    model = SentenceTransformer(MODEL_NAME)
    log.info("Embedding dimension: %d", model.get_sentence_embedding_dimension())

    for corpus in CORPORA:
        embed_corpus(corpus, model)

    sanity_check()

    # Final summary
    print("\nEmbedding complete:")
    for corpus in CORPORA:
        path = OUTPUT_DIR / f"{corpus['name']}.npy"
        if path.exists():
            shape = np.load(path).shape
            print(f"  {corpus['name']:12s} {str(shape):15s} → {path}")


if __name__ == "__main__":
    main()

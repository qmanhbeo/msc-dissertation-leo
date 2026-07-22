"""
Generate Sentence-BERT embeddings for the active non-sharded corpora.

Model: all-MiniLM-L6-v2 (384-dim for MiniLM; MPNet is 768-dim — practical choice for CPU/WSL)
       Change MODEL_NAME to "all-mpnet-base-v2" for 768-dim higher-quality embeddings (GPU recommended).

Inputs:
  2_data/1_preprocessed/policy_all/policy_segments_all.jsonl         (policy segments, field: text)
  2_data/1_preprocessed/osdg/osdg_clean.jsonl              (30,534 texts, field: text)
  2_data/1_preprocessed/sdg_benchmark/benchmark_clean.jsonl  (616 texts, field: text)
  2_data/1_preprocessed/sdg_knowledge_hub/sdg_knowledge_hub_clean.jsonl  (616 SDG-17 texts, journalism)
   2_data/1_preprocessed/sdgi_corpus/sdgi_clean.jsonl  (5,233 texts, all 17 SDGs, policy VNR/VLR)

Outputs per corpus:
  <name>.npy       float32 matrix (n_texts, embedding_dim)
  2_data/2_embedded/metadata/<name>_ids.json  list of dicts with id, sdg (where available), text_field

Idempotent by default: skips a corpus if its .npy already exists.
Use --overwrite to replace selected corpus artifacts.

Run from project root:
    python 1_code/2_embed/reference/0_embed_reference_corpora.py
    python 1_code/2_embed/reference/0_embed_reference_corpora.py --corpora policy --overwrite
    python 1_code/2_embed/reference/0_embed_reference_corpora.py --corpora policy --overwrite --local-files-only
"""

import argparse
import json
import logging
import sys
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
ANALYSIS_DIR = CODE_ROOT / "3_main_analysis" / "0_shared"
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from model_utils import DEFAULT_EMBED_MODEL, embed_dir_for_model

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEFAULT_MODEL = DEFAULT_EMBED_MODEL

CORPORA = [
    {
        "name": "policy",
        "input": Path("2_data/1_preprocessed/policy_all/policy_segments_all.jsonl"),
        "text_field": "text",
        "id_field": "segment_id",
        "sdg_field": None,
    },
    {
        "name": "osdg",
        "input": Path("2_data/1_preprocessed/osdg/osdg_clean.jsonl"),
        "text_field": "text",
        "id_field": "text_id",
        "sdg_field": "sdg",
    },
    {
        "name": "benchmark",
        "input": Path("2_data/1_preprocessed/sdg_benchmark/benchmark_clean.jsonl"),
        "text_field": "text",
        "id_field": "id",
        "sdg_field": "sdg",
    },
    {
        "name": "sdg_knowledge_hub",
        "input": Path("2_data/1_preprocessed/sdg_knowledge_hub/sdg_knowledge_hub_clean.jsonl"),
        "text_field": "text",
        "id_field": "id",
        "sdg_field": "sdg",
    },
    {
        "name": "sdgi",
        "input": Path("2_data/1_preprocessed/sdgi_corpus/sdgi_clean.jsonl"),
        "text_field": "text",
        "id_field": "id",
        "sdg_field": "sdg",
    },
    {
        "name": "aurora",
        "input": Path("2_data/1_preprocessed/aurora/aurora_texts.jsonl"),
        "text_field": "text",
        "id_field": "doi",
        "sdg_field": "sdg",
    },
]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI / Helpers
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Embed the active non-sharded corpora.")
    parser.add_argument(
        "--corpora",
        nargs="+",
        choices=[corpus["name"] for corpus in CORPORA],
        help="Optional subset of corpora to embed. Default: all corpora.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing embedding and metadata files for the selected corpora.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load the sentence-transformer model from the local Hugging Face cache only.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Sentence-transformer model name (default: %(default)s).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for embedding (default: %(default)s). Reduce to 32–64 for MPNet on 4GB GPUs.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def embed_corpus(corpus: dict, model: SentenceTransformer, *, overwrite: bool, output_dir: Path, batch_size: int) -> None:
    name = corpus["name"]
    metadata_dir = output_dir / "metadata"
    emb_path = output_dir / f"{name}.npy"
    ids_path = metadata_dir / f"{name}_ids.json"

    if emb_path.exists() and not overwrite:
        log.info("Skipping %s — %s already exists", name, emb_path)
        return

    log.info("Embedding corpus: %s (%s)", name, corpus["input"])
    records = load_jsonl(corpus["input"])
    texts = [r[corpus["text_field"]] for r in records]

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
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
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    np.save(emb_path, embeddings)
    with ids_path.open("w") as f:
        json.dump(ids_meta, f)

    log.info("Saved %s → shape %s", emb_path, embeddings.shape)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    selected_names = list(args.corpora or [corpus["name"] for corpus in CORPORA])
    selected_corpora = [corpus for corpus in CORPORA if corpus["name"] in selected_names]

    output_dir = embed_dir_for_model(args.model)

    log.info("Loading model: %s", args.model)
    model = SentenceTransformer(args.model, local_files_only=args.local_files_only)
    log.info("Embedding dimension: %d", model.get_sentence_embedding_dimension())

    for corpus in selected_corpora:
        embed_corpus(corpus, model, overwrite=args.overwrite, output_dir=output_dir, batch_size=args.batch_size)

    # Final summary
    print("\nEmbedding complete:")
    for corpus in selected_corpora:
        path = output_dir / f"{corpus['name']}.npy"
        if path.exists():
            shape = np.load(path).shape
            print(f"  {corpus['name']:12s} {str(shape):15s} → {path}")


if __name__ == "__main__":
    main()

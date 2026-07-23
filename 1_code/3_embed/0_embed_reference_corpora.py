"""
Generate Sentence-BERT embeddings for the active non-sharded corpora.

Inputs (resolved via model_utils helpers):
   Segmented corpora: segmented_dir_for_model(model) / {corpus}.jsonl
   Pass-through:       preprocessed_dir() / subdir / {filename}

Outputs per corpus:
   <name>.npy       float32 matrix (n_texts, embedding_dim)
   embed_dir_for_model(model) / metadata / <name>_ids.json

Idempotent by default: skips a corpus if its .npy already exists.
Use --overwrite to replace selected corpus artifacts.

Run from project root:
    python 1_code/3_embed/0_embed_reference_corpora.py
    python 1_code/3_embed/0_embed_reference_corpora.py --corpora aurora --overwrite
    python 1_code/3_embed/0_embed_reference_corpora.py --corpora aurora --overwrite --model all-mpnet-base-v2
"""

import argparse
import json
import logging
import sys
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
ANALYSIS_DIR = CODE_ROOT / "7_main_analysis" / "0_shared"
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from model_utils import DEFAULT_EMBED_MODEL, embed_dir_for_model, preprocessed_dir, segmented_dir_for_model

CORPORA = [
    {
        "name": "policy",
        "text_field": "text",
        "id_field": "segment_id",
        "sdg_field": None,
        "segmented": True,
    },
    {
        "name": "osdg",
        "text_field": "text",
        "id_field": "text_id",
        "sdg_field": "sdgs",
        "segmented": False,
        "preprocessed_subdir": "osdg",
        "preprocessed_filename": "osdg_clean.jsonl",
    },
    {
        "name": "benchmark",
        "text_field": "text",
        "id_field": "id",
        "sdg_field": "sdgs",
        "segmented": False,
        "preprocessed_subdir": "sdg_benchmark",
        "preprocessed_filename": "benchmark_clean.jsonl",
    },
    {
        "name": "sdg_knowledge_hub",
        "text_field": "text",
        "id_field": "id",
        "sdg_field": "sdgs",
        "segmented": True,
    },
    {
        "name": "sdgi",
        "text_field": "text",
        "id_field": "segment_id",
        "sdg_field": "sdgs",
        "segmented": True,
    },
    {
        "name": "aurora",
        "text_field": "text",
        "id_field": "doi",
        "sdg_field": "sdgs",
        "segmented": True,
    },
]

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


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
        default=DEFAULT_EMBED_MODEL,
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


def _resolve_input_path(corpus: dict, model_name: str) -> Path:
    if corpus.get("segmented"):
        return segmented_dir_for_model(model_name) / f"{corpus['name']}.jsonl"
    return preprocessed_dir() / corpus["preprocessed_subdir"] / corpus["preprocessed_filename"]


def embed_corpus(corpus: dict, model: SentenceTransformer, *, overwrite: bool, output_dir: Path, batch_size: int, model_name: str) -> None:
    name = corpus["name"]
    metadata_dir = output_dir / "metadata"
    emb_path = output_dir / f"{name}.npy"
    ids_path = metadata_dir / f"{name}_ids.json"

    if emb_path.exists() and not overwrite:
        log.info("Skipping %s — %s already exists", name, emb_path)
        return

    input_path = _resolve_input_path(corpus, model_name)
    log.info("Embedding corpus: %s (%s)", name, input_path)
    records = load_jsonl(input_path)
    texts = [r[corpus["text_field"]] for r in records]

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    embeddings = embeddings.astype(np.float32)

    ids_meta = []
    for r in records:
        entry = {"id": r.get(corpus["id_field"], ""), "text": r[corpus["text_field"]]}
        if corpus["sdg_field"]:
            sdg_val = r.get(corpus["sdg_field"])
            entry["sdgs"] = sdg_val if isinstance(sdg_val, list) else [sdg_val] if sdg_val is not None else None
        if "source_doc" in r:
            entry["source_doc"] = r["source_doc"]
        ids_meta.append(entry)

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    np.save(emb_path, embeddings)
    with ids_path.open("w") as f:
        json.dump(ids_meta, f)

    log.info("Saved %s → shape %s", emb_path, embeddings.shape)


def main() -> None:
    args = parse_args()
    selected_names = list(args.corpora or [corpus["name"] for corpus in CORPORA])
    selected_corpora = [corpus for corpus in CORPORA if corpus["name"] in selected_names]

    output_dir = embed_dir_for_model(args.model)

    log.info("Loading model: %s", args.model)
    model = SentenceTransformer(args.model, local_files_only=args.local_files_only)
    log.info("Embedding dimension: %d", model.get_sentence_embedding_dimension())

    for corpus in selected_corpora:
        embed_corpus(corpus, model, overwrite=args.overwrite, output_dir=output_dir, batch_size=args.batch_size, model_name=args.model)

    print("\nEmbedding complete:")
    for corpus in selected_corpora:
        path = output_dir / f"{corpus['name']}.npy"
        if path.exists():
            shape = np.load(path).shape
            print(f"  {corpus['name']:12s} {str(shape):15s} → {path}")


if __name__ == "__main__":
    main()

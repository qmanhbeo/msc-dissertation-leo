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
import shutil
import sys
from datetime import datetime

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


def _write_manifest(path: Path, corpus_name: str, total_rows: int, dim: int,
                    completed_batches: list[int], rows_completed: int, status: str) -> None:
    manifest = {
        "corpus": corpus_name,
        "total_rows": total_rows,
        "dim": dim,
        "completed_batches": completed_batches,
        "rows_completed": rows_completed,
        "status": status,
        "last_updated_utc": datetime.utcnow().isoformat(),
    }
    path.write_text(json.dumps(manifest, indent=2))


def _concatenate_batches(tmp_dir: Path, emb_path: Path, n: int, dim: int,
                         ids_meta: list[dict], ids_path: Path) -> None:
    manifest_path = tmp_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest["status"] = "concatenating"
    manifest_path.write_text(json.dumps(manifest))

    batch_files = sorted(tmp_dir.glob("batch_*.npy"),
                         key=lambda p: int(p.stem.split("_")[1]))
    log.info("Concatenating %d batch files \u2192 %s", len(batch_files), emb_path)
    all_embs = np.concatenate([np.load(f) for f in batch_files], axis=0)
    if all_embs.shape != (n, dim):
        raise RuntimeError(f"Shape mismatch after concatenation: {all_embs.shape} != ({n}, {dim})")

    tmp_emb = emb_path.with_suffix(".npy.tmp")
    np.save(tmp_emb, all_embs)
    tmp_emb.replace(emb_path)

    shutil.rmtree(tmp_dir)

    with ids_path.open("w") as f:
        json.dump(ids_meta, f)

    log.info("Saved %s \u2192 shape %s", emb_path, all_embs.shape)


def embed_corpus(corpus: dict, model: SentenceTransformer, *, overwrite: bool, output_dir: Path, batch_size: int, model_name: str) -> None:
    name = corpus["name"]
    metadata_dir = output_dir / "metadata"
    emb_path = output_dir / f"{name}.npy"
    ids_path = metadata_dir / f"{name}_ids.json"

    if emb_path.exists() and not overwrite:
        log.info("Skipping %s \u2014 %s already exists", name, emb_path)
        return

    input_path = _resolve_input_path(corpus, model_name)
    log.info("Embedding corpus: %s (%s)", name, input_path)
    records = load_jsonl(input_path)
    texts = [r[corpus["text_field"]] for r in records]
    n = len(texts)

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    ids_meta = []
    for r in records:
        entry = {"id": r.get(corpus["id_field"], ""), "text": r[corpus["text_field"]]}
        if corpus["sdg_field"]:
            sdg_val = r.get(corpus["sdg_field"])
            entry["sdgs"] = sdg_val if isinstance(sdg_val, list) else [sdg_val] if sdg_val is not None else None
        if "source_doc" in r:
            entry["source_doc"] = r["source_doc"]
        ids_meta.append(entry)

    dim = model.get_embedding_dimension()
    log.info("Dim=%d  Total texts=%d  Batch size=%d", dim, n, batch_size)

    # --- Per-batch checkpointing ---
    tmp_dir = output_dir / f"{name}_batches"
    manifest_path = tmp_dir / "manifest.json"

    if overwrite and tmp_dir.exists():
        shutil.rmtree(tmp_dir)

    completed_batches: set[int] = set()
    rows_completed: int = 0

    if tmp_dir.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("status") == "concatenating":
            log.info("Resuming %s \u2014 final concatenation step (all batches done)", name)
            _concatenate_batches(tmp_dir, emb_path, n, dim, ids_meta, ids_path)
            return
        completed_batches = set(manifest.get("completed_batches", []))
        rows_completed = manifest.get("rows_completed", 0)
        log.info("Resuming %s from batch %d (had %d batches, %d rows done)",
                 name, len(completed_batches), len(completed_batches), rows_completed)
    elif tmp_dir.exists():
        log.warning("Found %s but no manifest \u2014 starting fresh", tmp_dir)
        shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)
    else:
        tmp_dir.mkdir(parents=True, exist_ok=True)

    batch_starts = list(range(0, n, batch_size))
    n_batches = len(batch_starts)

    for batch_i, start in enumerate(batch_starts):
        if batch_i in completed_batches:
            continue

        end = min(start + batch_size, n)
        batch_texts = texts[start:end]

        batch_emb = model.encode(
            batch_texts,
            batch_size=len(batch_texts),
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)

        batch_path = tmp_dir / f"batch_{batch_i:05d}.npy"
        tmp_batch = batch_path.with_suffix(".npy.tmp")
        with tmp_batch.open("wb") as f:
            np.save(f, batch_emb)
        tmp_batch.replace(batch_path)

        completed_batches.add(batch_i)
        rows_completed += len(batch_emb)
        _write_manifest(manifest_path, name, n, dim, sorted(completed_batches), rows_completed, "in_progress")

        pct = 100.0 * rows_completed / n
        log.info("  batch %4d/%d (%5d\u2013%5d, %5d docs)  %5.1f%%  \u2192 wrote %s",
                 batch_i + 1, n_batches, start, end - 1, end - start, pct, batch_path)

    _concatenate_batches(tmp_dir, emb_path, n, dim, ids_meta, ids_path)


def main() -> None:
    args = parse_args()
    selected_names = list(args.corpora or [corpus["name"] for corpus in CORPORA])
    selected_corpora = [corpus for corpus in CORPORA if corpus["name"] in selected_names]

    output_dir = embed_dir_for_model(args.model)

    log.info("Loading model: %s", args.model)
    model = SentenceTransformer(args.model, local_files_only=args.local_files_only)
    log.info("Embedding dimension: %d", model.get_embedding_dimension())

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

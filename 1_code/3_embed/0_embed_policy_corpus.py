"""
Embed the policy corpus with per-batch incremental checkpointing.

Input:  segmented_dir_for_model(model) / policy.jsonl
Output: embed_dir_for_model(model) / policy.npy
        embed_dir_for_model(model) / metadata / policy_ids.json

The policy corpus is architecturally distinct from the 5 labeled reference
corpora — it is unlabeled, composed of 4 merged sources, and used only for
scoring/centroids, never for training.

Run from project root:
    python 1_code/3_embed/0_embed_policy_corpus.py
    python 1_code/3_embed/0_embed_policy_corpus.py --overwrite --batch-size 64
"""

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
ANALYSIS_DIR = CODE_ROOT / "7_main_analysis" / "0_shared"
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from embed_utils import concatenate_batches, load_jsonl, write_batch_manifest
from model_utils import DEFAULT_EMBED_MODEL, embed_dir_for_model, segmented_dir_for_model

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Embed the policy corpus.")
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing policy.npy and metadata.",
    )
    parser.add_argument(
        "--local-files-only", action="store_true",
        help="Load the model from the local Hugging Face cache only.",
    )
    parser.add_argument(
        "--model", default=DEFAULT_EMBED_MODEL,
        help="Sentence-transformer model name (default: %(default)s).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=128,
        help="Batch size (default: %(default)s). Reduce to 32–64 for MPNet on 4GB GPUs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_name = args.model
    output_dir = embed_dir_for_model(model_name)
    emb_path = output_dir / "policy.npy"
    metadata_dir = output_dir / "metadata"
    ids_path = metadata_dir / "policy_ids.json"

    if emb_path.exists() and not args.overwrite:
        log.info("Skipping policy \u2014 %s already exists", emb_path)
        return

    input_path = segmented_dir_for_model(model_name) / "policy.jsonl"
    log.info("Embedding policy corpus: %s", input_path)
    records = load_jsonl(input_path)
    texts = [r["text"] for r in records]
    n = len(texts)

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    ids_meta = []
    for r in records:
        ids_meta.append({
            "id": r.get("segment_id", ""),
            "text": r["text"],
            "source_doc": r.get("source_doc", ""),
        })

    log.info("Loading model: %s", model_name)
    model = SentenceTransformer(model_name, local_files_only=args.local_files_only)
    dim = model.get_embedding_dimension()
    log.info("Dim=%d  Total texts=%d  Batch size=%d", dim, n, args.batch_size)

    # --- Per-batch checkpointing ---
    tmp_dir = output_dir / "policy_batches"
    manifest_path = tmp_dir / "manifest.json"

    if args.overwrite and tmp_dir.exists():
        shutil.rmtree(tmp_dir)

    completed_batches: set[int] = set()
    rows_completed: int = 0

    if tmp_dir.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("status") == "concatenating":
            log.info("Resuming policy \u2014 final concatenation step (all batches done)")
            concatenate_batches(tmp_dir, emb_path, n, dim, ids_meta=ids_meta, ids_path=ids_path)
            return
        completed_batches = set(manifest.get("completed_batches", []))
        rows_completed = manifest.get("rows_completed", 0)
        log.info("Resuming policy from batch %d (had %d batches, %d rows done)",
                 len(completed_batches), len(completed_batches), rows_completed)
    elif tmp_dir.exists():
        log.warning("Found %s but no manifest \u2014 starting fresh", tmp_dir)
        shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)
    else:
        tmp_dir.mkdir(parents=True, exist_ok=True)

    batch_starts = list(range(0, n, args.batch_size))
    n_batches = len(batch_starts)

    for batch_i, start in enumerate(batch_starts):
        if batch_i in completed_batches:
            continue

        end = min(start + args.batch_size, n)
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
        write_batch_manifest(
            manifest_path,
            corpus_name="policy",
            total_rows=n,
            dim=dim,
            completed_batches=sorted(completed_batches),
            rows_completed=rows_completed,
            status="in_progress",
        )

        pct = 100.0 * rows_completed / n
        log.info("  batch %4d/%d (%5d\u2013%5d, %5d docs)  %5.1f%%  \u2192 wrote %s",
                 batch_i + 1, n_batches, start, end - 1, end - start, pct, batch_path)

    concatenate_batches(tmp_dir, emb_path, n, dim, ids_meta=ids_meta, ids_path=ids_path)


if __name__ == "__main__":
    main()

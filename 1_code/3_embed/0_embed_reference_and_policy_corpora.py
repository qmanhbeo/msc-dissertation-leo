"""
Embed a single reference or policy corpus with per-batch incremental checkpointing.

Parameterized by corpus name — call once per source file.

Usage:
    python 1_code/3_embed/0_embed_reference_and_policy_corpora.py --corpus sdgi --batch-size 64
    python 1_code/3_embed/0_embed_reference_and_policy_corpora.py --corpus policy_scrape --overwrite
    python 1_code/3_embed/0_embed_reference_and_policy_corpora.py --corpus aurora --model all-MiniLM-L6-v2

Input paths resolved via model_utils helpers.
Outputs per corpus:
    embed_dir/{model}/{corpus}.npy          float32 (n_texts, embed_dim)
    embed_dir/{model}/metadata/{corpus}_ids.json
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
from embed_loader import load_embedder
from model_utils import DEFAULT_EMBED_MODEL, embed_dir_for_model, preprocessed_dir, segmented_dir_for_model, resolve_model_alias

CORPUS_CONFIG = {
    "osdg": {
        "text_field": "text",
        "id_field": "text_id",
        "sdg_field": "sdgs",
        "input_path": lambda model: preprocessed_dir() / "osdg" / "osdg_clean.jsonl",
    },
    "benchmark": {
        "text_field": "text",
        "id_field": "id",
        "sdg_field": "sdgs",
        "input_path": lambda model: preprocessed_dir() / "sdg_benchmark" / "benchmark_clean.jsonl",
    },
    "sdg_knowledge_hub": {
        "text_field": "text",
        "id_field": "id",
        "sdg_field": "sdgs",
        "input_path": lambda model: segmented_dir_for_model(model) / "sdg_knowledge_hub.jsonl",
    },
    "sdgi": {
        "text_field": "text",
        "id_field": "segment_id",
        "sdg_field": "sdgs",
        "input_path": lambda model: segmented_dir_for_model(model) / "sdgi.jsonl",
    },
    "aurora": {
        "text_field": "text",
        "id_field": "doi",
        "sdg_field": "sdgs",
        "input_path": lambda model: segmented_dir_for_model(model) / "aurora.jsonl",
    },
    "policy_scrape": {
        "text_field": "text",
        "id_field": "segment_id",
        "sdg_field": None,
        "input_path": lambda model: segmented_dir_for_model(model) / "policy_scrape.jsonl",
    },
    "policy_manual": {
        "text_field": "text",
        "id_field": "segment_id",
        "sdg_field": None,
        "input_path": lambda model: segmented_dir_for_model(model) / "policy_manual.jsonl",
    },
    "ungdc_sdg": {
        "text_field": "text",
        "id_field": "segment_id",
        "sdg_field": None,
        "input_path": lambda model: segmented_dir_for_model(model) / "ungdc_sdg.jsonl",
    },
}

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Embed one reference or policy corpus by name."
    )
    parser.add_argument(
        "--corpus",
        required=True,
        choices=sorted(CORPUS_CONFIG),
        help="Corpus name to embed.",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing .npy and metadata for this corpus.",
    )
    parser.add_argument(
        "--local-files-only", action="store_true",
        help="Load model from local Hugging Face cache only.",
    )
    parser.add_argument(
        "--embed-model", default=DEFAULT_EMBED_MODEL, type=resolve_model_alias,
        help="Sentence-transformer model (default: %(default)s).",
    )
    parser.add_argument(
        "--device", choices=["auto", "cuda", "cpu"], default="auto",
        help="Device for embedding (default: %(default)s).",
    )
    parser.add_argument(
        "--seg-model", default=None,
        help="Read segmented texts from this model's dir (default: this model). "
             "Use 'all-mpnet-base-v2' so a domain encoder (e.g. SciBERT) embeds "
             "the identical canonical inputs as the baseline encoder.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=128,
        help="Batch size (default: %(default)s). Reduce to 64 for MPNet on 4GB GPUs.",
    )
    parser.add_argument(
        "--precision", choices=["fp32", "fp16"], default=None,
        help="Compute + storage precision for embeddings (fp16 ≈ 2x faster on Ampere GPUs). "
             "Default: fp16 for all-MiniLM-L6-v2, fp32 otherwise.",
    )
    parser.add_argument("--normalize-embeddings", action="store_true", default=True,
                        help="L2-normalise embeddings so cosine similarity equals dot product (default: %(default)s)")
    return parser.parse_args()


def default_precision(model: str) -> str:
    """MPNet stays fp32; MiniLM and SciBERT default to fp16.

    Both MiniLM and SciBERT are encoder-sensitivity checks, and fp16 halves
    their embedding footprint.
    """
    return "fp16" if model in ("all-MiniLM-L6-v2", "allenai/scibert_scivocab_uncased") else "fp32"


def embed_corpus(
    corpus_name: str,
    config: dict,
    model: SentenceTransformer,
    *,
    overwrite: bool,
    output_dir: Path,
    batch_size: int,
    precision: str,
    model_name: str,
    seg_model: str | None = None,
    normalize_embeddings: bool = True,
) -> None:
    metadata_dir = output_dir / "metadata"
    emb_path = output_dir / f"{corpus_name}.npy"
    ids_path = metadata_dir / f"{corpus_name}_ids.json"

    if emb_path.exists() and not overwrite:
        # Resume-safe: a completed embedding is reused when --overwrite is not
        # passed, so re-runs never redo finished corpora.
        log.info("Skipping %s — %s already exists", corpus_name, emb_path)
        return
    if overwrite and emb_path.exists():
        # --overwrite forces a clean re-embed: drop the stale .npy/.json so a
        # changed segmentation/encoder produces fresh embeddings.
        log.info("Overwrite requested — removing existing %s", emb_path)
        emb_path.unlink()
        if ids_path.exists():
            ids_path.unlink()

    input_path = config["input_path"](seg_model or model_name)
    log.info("Embedding %s (%s)", corpus_name, input_path)
    records = load_jsonl(input_path)
    texts = [r[config["text_field"]] for r in records]
    n = len(texts)

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    ids_meta = []
    for r in records:
        entry = {
            "id": r.get(config["id_field"], ""),
        }
        if config["sdg_field"]:
            sdg_val = r.get(config["sdg_field"])
            entry["sdgs"] = (
                sdg_val
                if isinstance(sdg_val, list)
                else [sdg_val] if sdg_val is not None else None
            )
        source_doc = r.get("source_doc")
        if source_doc is None:
            source_doc = r.get(config["id_field"], "")
        entry["source_doc"] = source_doc
        ids_meta.append(entry)

    dim = model.get_embedding_dimension()
    log.info("Dim=%d  Total texts=%d  Batch size=%d", dim, n, batch_size)

    # Per-batch checkpointing
    tmp_dir = output_dir / f"{corpus_name}_batches"
    manifest_path = tmp_dir / "manifest.json"

    if overwrite and tmp_dir.exists() and not manifest_path.exists():
        # Resume-safe: keep a batch checkpoint dir that still has a valid manifest
        # so an interrupted embed continues from the last completed batch instead
        # of restarting from zero.
        shutil.rmtree(tmp_dir)

    completed_batches: set[int] = set()
    rows_completed: int = 0

    if tmp_dir.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("status") == "concatenating":
            log.info("Resuming %s — final concatenation step (all batches done)", corpus_name)
            concatenate_batches(tmp_dir, emb_path, n, dim, ids_meta=ids_meta, ids_path=ids_path)
            return
        completed_batches = set(manifest.get("completed_batches", []))
        rows_completed = manifest.get("rows_completed", 0)
        log.info("Resuming %s from batch %d (had %d batches, %d rows done)",
                 corpus_name, len(completed_batches), len(completed_batches), rows_completed)
    elif tmp_dir.exists():
        log.warning("Found %s but no manifest — starting fresh", tmp_dir)
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
            normalize_embeddings=normalize_embeddings,
        ).astype(np.float16 if precision == "fp16" else np.float32)

        batch_path = tmp_dir / f"batch_{batch_i:05d}.npy"
        tmp_batch = batch_path.with_suffix(".npy.tmp")
        with tmp_batch.open("wb") as f:
            np.save(f, batch_emb)
        tmp_batch.replace(batch_path)

        completed_batches.add(batch_i)
        rows_completed += len(batch_emb)
        write_batch_manifest(
            manifest_path,
            corpus_name=corpus_name,
            total_rows=n,
            dim=dim,
            completed_batches=sorted(completed_batches),
            rows_completed=rows_completed,
            status="in_progress",
        )

        pct = 100.0 * rows_completed / n
        log.info("  batch %4d/%d (%5d–%5d, %5d docs)  %5.1f%%  → wrote %s",
                 batch_i + 1, n_batches, start, end - 1, end - start, pct, batch_path)

    concatenate_batches(tmp_dir, emb_path, n, dim, ids_meta=ids_meta, ids_path=ids_path)


def main() -> None:
    args = parse_args()
    precision = args.precision or default_precision(args.embed_model)
    output_dir = embed_dir_for_model(args.embed_model)
    config = CORPUS_CONFIG[args.corpus]

    log.info("Loading model: %s", args.embed_model)
    model = load_embedder(args.embed_model, device=args.device, local_files_only=args.local_files_only)
    if precision == "fp16":
        model = model.half()
    log.info("Embedding dimension: %d", model.get_embedding_dimension())

    embed_corpus(
        args.corpus, config, model,
        overwrite=args.overwrite,
        output_dir=output_dir,
        batch_size=args.batch_size,
        precision=precision,
        model_name=args.embed_model,
        seg_model=args.seg_model,
        normalize_embeddings=args.normalize_embeddings,
    )

    path = output_dir / f"{args.corpus}.npy"
    if path.exists():
        shape = np.load(path).shape
        print(f"\nDone: {args.corpus}  {shape}  → {path}")


if __name__ == "__main__":
    main()

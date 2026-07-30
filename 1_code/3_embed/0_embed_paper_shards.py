"""
Embed segmented research shards into reusable embedding shards.

Input manifest (from segmentation stage):
    2_data/2_segmented/{model}/research/metadata/manifest.json

Outputs:
    2_data/3_embedded/{model}/research_shards/part-00001.npy
    2_data/3_embedded/{model}/research_shards/metadata/part-00001_ids.jsonl
    2_data/3_embedded/{model}/research_shards/metadata/manifest.json

Shard-level resume: completed shards (tracked in output manifest) are skipped.

Run from project root:
    python 1_code/3_embed/0_embed_paper_shards.py
    python 1_code/3_embed/0_embed_paper_shards.py --corpus research_concept
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
ANALYSIS_DIR = CODE_ROOT / "7_main_analysis" / "0_shared"
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from embed_utils import concatenate_batches, write_batch_manifest
from embed_loader import load_embedder
from shard_pipeline_utils import atomic_write_json, ensure_dir, now_iso, read_json, sha256_file, update_stage_status
from model_utils import DEFAULT_EMBED_MODEL, embed_dir_for_model, embed_research_dir_for_model, research_concept_segmented_dir_for_model, research_subset_dir, segmented_dir_for_model, resolve_model_alias


log = logging.getLogger(__name__)
STATUS_STAGE = "openalex_clean_shards_to_embeddings"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input-manifest", default=None)
    p.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL, type=resolve_model_alias,
                   help="Embedding model. One of: all-mpnet-base-v2 (default), all-MiniLM-L6-v2, allenai/scibert_scivocab_uncased. Short aliases: mpnet, minilm, scibert.")
    p.add_argument("--batch-size", type=int, default=256,
                   help="Internal batch size passed to model.encode (GPU occupancy).")
    p.add_argument("--precision", choices=["fp32", "fp16"], default=None,
                   help="Compute + storage precision for embeddings. Default: fp16.")
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--normalize-embeddings", action="store_true", default=True,
                   help="L2-normalise embeddings so cosine similarity equals dot product (default: %(default)s)")
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--limit-shards", type=int, default=0)
    p.add_argument("--corpus", choices=["research", "research_concept", "research_subset"], default="research",
                   help="Corpus to embed (default: %(default)s). Auto-derives input manifest and output dir.")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-embed existing shards even if already complete")
    return p.parse_args()


def resolve_device(name: str) -> str:
    if name == "cpu":
        return "cpu"
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"


def default_precision(model: str) -> str:
    """Default to fp16 for all models — safe for cosine-similarity-based analysis."""
    return "fp16"


def load_texts(path: Path) -> list[str]:
    texts: list[str] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            texts.append(row["text"])
    return texts


def generate_ids(data_path: Path, ids_out: Path) -> None:
    tmp = ids_out.with_suffix(ids_out.suffix + ".tmp")
    with data_path.open(encoding="utf-8") as src, tmp.open("w", encoding="utf-8") as dst:
        for row_in_shard, line in enumerate(src):
            if not line.strip():
                continue
            row = json.loads(line)
            out = {
                "openalex_id": row.get("openalex_id", ""),
                "source_doc": row.get("source_doc", ""),
                "segment_id": row.get("segment_id", ""),
                "row_in_shard": row_in_shard,
            }
            dst.write(json.dumps(out, ensure_ascii=False) + "\n")
    tmp.replace(ids_out)


def main() -> None:
    args = parse_args()
    precision = args.precision or default_precision(args.embed_model)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    if args.input_manifest:
        seg_root = Path(args.input_manifest).resolve().parent.parent
    elif args.corpus == "research_concept":
        seg_root = research_concept_segmented_dir_for_model(args.embed_model)
    elif args.corpus == "research_subset":
        seg_root = research_subset_dir()
    else:
        seg_root = segmented_dir_for_model(args.embed_model) / "research"

    input_manifest = Path(args.input_manifest) if args.input_manifest else seg_root / "metadata" / "manifest.json"

    if args.corpus == "research_concept":
        out_dir = embed_dir_for_model(args.embed_model) / "research_concept"
    else:
        out_dir = embed_research_dir_for_model(args.embed_model)

    metadata_dir = out_dir / "metadata"
    status_dir = metadata_dir
    ensure_dir(out_dir)
    ensure_dir(metadata_dir)
    ensure_dir(status_dir)

    data = read_json(input_manifest)
    if not data or "shards" not in data:
        raise RuntimeError(f"Invalid or missing input manifest: {input_manifest}")

    device = resolve_device(args.device)
    try:
        model = load_embedder(args.embed_model, device=device, local_files_only=args.local_files_only)
    except Exception as exc:
        hint = "Model load failed. If the environment has no internet, ensure model cache exists and use --local-files-only."
        raise RuntimeError(f"{hint} Original error: {exc}") from exc
    if precision == "fp16":
        if device == "cpu":
            raise RuntimeError("fp16 precision requires a CUDA device.")
        model = model.half()
    emb_dim = int(model.get_sentence_embedding_dimension())
    log.info("Model=%s device=%s dim=%d", args.embed_model, device, emb_dim)

    out_manifest_path = metadata_dir / "manifest.json"
    out_manifest = read_json(out_manifest_path, default=None)
    if out_manifest is None:
        out_manifest = {
            "stage": STATUS_STAGE,
            "schema_version": 1,
            "created_at_utc": now_iso(),
            "model": args.embed_model,
            "device": device,
            "embedding_dim": emb_dim,
            "normalize_embeddings": args.normalize_embeddings,
            "input_manifest": str(input_manifest),
            "shards": [],
            "totals": {"rows": 0, "shards": 0},
        }

    completed = {int(s["shard_id"]): s for s in out_manifest.get("shards", [])}
    if args.overwrite:
        completed = {}
        out_manifest["shards"] = []
        out_manifest["totals"]["rows"] = 0
        out_manifest["totals"]["shards"] = 0
        out_manifest["created_at_utc"] = now_iso()
    shards = data["shards"][: args.limit_shards] if args.limit_shards > 0 else data["shards"]

    update_stage_status(
        status_dir,
        STATUS_STAGE,
        "running",
        {"model": args.embed_model, "device": device, "input_manifest": str(input_manifest)},
    )

    for shard in shards:
        shard_id = int(shard["shard_id"])
        shard_name = shard["name"]
        out_emb = out_dir / f"{shard_name}.npy"
        out_ids = metadata_dir / f"{shard_name}_ids.jsonl"
        in_data = seg_root / f"{shard_name}.jsonl"

        if not in_data.exists():
            log.error("Segmented shard missing at %s — run 2_segment/ first", in_data)
            raise FileNotFoundError(f"Segmented research shard not found: {in_data}")

        if not args.overwrite and shard_id in completed and out_emb.exists() and out_ids.exists():
            log.info("Skip shard %s (already embedded)", shard_name)
            continue

        t_start = time.time()
        log.info("Embedding shard %s", shard_name)
        texts = load_texts(in_data)
        n = len(texts)
        dim = int(model.get_embedding_dimension())

        batches_dir = out_dir / f"{shard_name}_batches"
        manifest_path = batches_dir / "manifest.json"

        completed_batches: set[int] = set()
        rows_completed: int = 0

        if batches_dir.exists() and manifest_path.exists():
            m = json.loads(manifest_path.read_text())
            if m.get("status") == "concatenating":
                log.info("Resuming %s — final concatenation", shard_name)
                concatenate_batches(batches_dir, out_emb, n, dim)
                generate_ids(in_data, out_ids)
            else:
                completed_batches = set(m.get("completed_batches", []))
                rows_completed = m.get("rows_completed", 0)
                log.info("Resuming %s from batch %d (%d/%d rows)",
                         shard_name, len(completed_batches), rows_completed, n)
        elif batches_dir.exists():
            shutil.rmtree(batches_dir)
            batches_dir.mkdir(parents=True, exist_ok=True)
        else:
            batches_dir.mkdir(parents=True, exist_ok=True)

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
                normalize_embeddings=args.normalize_embeddings,
            ).astype(np.float16 if precision == "fp16" else np.float32)

            batch_path = batches_dir / f"batch_{batch_i:05d}.npy"
            tmp_batch = batch_path.with_suffix(".npy.tmp")
            with tmp_batch.open("wb") as f:
                np.save(f, batch_emb)
            tmp_batch.replace(batch_path)

            completed_batches.add(batch_i)
            rows_completed += len(batch_emb)
            write_batch_manifest(
                manifest_path,
                corpus_name=shard_name,
                total_rows=n,
                dim=dim,
                completed_batches=sorted(completed_batches),
                rows_completed=rows_completed,
                status="in_progress",
            )

            pct = 100.0 * rows_completed / n
            log.info("  batch %4d/%d (%5d–%5d, %5d docs)  %5.1f%%  → wrote %s",
                     batch_i + 1, n_batches, start, end - 1, end - start, pct, batch_path)

        concatenate_batches(batches_dir, out_emb, n, dim)
        generate_ids(in_data, out_ids)

        out_record = {
            "shard_id": shard_id,
            "name": shard_name,
            "embedding_path": str(out_emb),
            "ids_path": str(out_ids),
            "rows": n,
            "dim": dim,
            "bytes": out_emb.stat().st_size,
            "sha256": sha256_file(out_emb),
            "ids_sha256": sha256_file(out_ids),
        }

        elapsed = time.time() - t_start
        texts_per_s = n / elapsed if elapsed > 0 else 0.0
        log.info("Shard %s done: %d rows in %.1fs (%.1f texts/s)", shard_name, n, elapsed, texts_per_s)

        out_manifest["shards"] = [s for s in out_manifest["shards"] if int(s["shard_id"]) != shard_id]
        out_manifest["shards"].append(out_record)
        out_manifest["shards"].sort(key=lambda x: int(x["shard_id"]))
        out_manifest["totals"]["rows"] = int(sum(int(s["rows"]) for s in out_manifest["shards"]))
        out_manifest["totals"]["shards"] = int(len(out_manifest["shards"]))
        atomic_write_json(out_manifest_path, out_manifest)
        update_stage_status(
            status_dir,
            STATUS_STAGE,
            "running",
            {"last_completed_shard": shard_id, "rows_done": out_manifest["totals"]["rows"]},
        )

    update_stage_status(
        status_dir,
        STATUS_STAGE,
        "completed",
        {"manifest_path": str(out_manifest_path), "rows_done": out_manifest["totals"]["rows"]},
    )
    log.info("Embedding complete. manifest=%s rows=%s", out_manifest_path, out_manifest["totals"]["rows"])


if __name__ == "__main__":
    main()

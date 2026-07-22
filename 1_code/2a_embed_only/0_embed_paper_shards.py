"""
Embed cleaned OpenAlex shards into reusable embedding shards.

Input manifest:
  2_data/1_preprocessed/research_corpus/metadata/manifest.json

Outputs:
  2_data/2_embedded/research_shards/part-00001.npy
  2_data/2_embedded/research_shards/metadata/part-00001_ids.jsonl
  2_data/2_embedded/research_shards/metadata/manifest.json

Run from project root:
    python 1_code/2_embed/research/0_embed_paper_shards.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
ANALYSIS_DIR = CODE_ROOT / "3_appendix_centroid" / "0_shared"
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from shard_pipeline_utils import atomic_write_json, ensure_dir, now_iso, read_json, sha256_file, update_stage_status
from model_utils import DEFAULT_EMBED_MODEL, embed_dir_for_model


log = logging.getLogger(__name__)
STATUS_STAGE = "openalex_clean_shards_to_embeddings"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input-manifest", default="2_data/1_preprocessed/research_corpus/metadata/manifest.json")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--status-dir", default=None)
    p.add_argument("--metadata-dir", default="")
    p.add_argument("--model", default=DEFAULT_EMBED_MODEL)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--limit-shards", type=int, default=0)
    return p.parse_args()


def resolve_device(name: str) -> str:
    if name == "cpu":
        return "cpu"
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_texts(path: Path) -> list[str]:
    texts: list[str] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            texts.append(row["combined_text"])
    return texts


def copy_ids(ids_in: Path, ids_out: Path) -> None:
    tmp = ids_out.with_suffix(ids_out.suffix + ".tmp")
    with ids_in.open(encoding="utf-8") as src, tmp.open("w", encoding="utf-8") as dst:
        for line in src:
            if line.strip():
                dst.write(line)
    tmp.replace(ids_out)


def resolve_from_manifest(manifest_path: Path, stored_path: str) -> Path:
    del manifest_path  # hard pivot: no location fallback based on manifest placement
    raw = Path(stored_path)
    if raw.is_absolute():
        if raw.exists():
            return raw
        raise FileNotFoundError(f"Absolute path from manifest does not exist: {raw}")
    prefix = raw.as_posix()
    if not prefix.startswith("2_data/1_preprocessed/"):
        raise RuntimeError(
            f"Hard pivot violation: expected data path under 2_data/1_preprocessed/, got: {stored_path}"
        )
    resolved = Path.cwd() / raw
    if resolved.exists():
        return resolved
    raise FileNotFoundError(f"Manifest path does not exist: {stored_path} (resolved: {resolved})")


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    embed_root = embed_dir_for_model(args.model)

    input_manifest = Path(args.input_manifest)
    out_dir = Path(args.out_dir) if args.out_dir else embed_root / "research_shards"
    metadata_dir = Path(args.metadata_dir) if args.metadata_dir else out_dir / "metadata"
    status_dir = Path(args.status_dir) if args.status_dir else embed_root / "research_shards" / "metadata"
    ensure_dir(out_dir)
    ensure_dir(metadata_dir)
    ensure_dir(status_dir)

    data = read_json(input_manifest)
    if not data or "shards" not in data:
        raise RuntimeError(f"Invalid or missing input manifest: {input_manifest}")

    device = resolve_device(args.device)
    try:
        model = SentenceTransformer(args.model, device=device, local_files_only=args.local_files_only)
    except Exception as exc:
        hint = (
            "Model load failed. If the environment has no internet, ensure model cache exists and use "
            "--local-files-only."
        )
        raise RuntimeError(f"{hint} Original error: {exc}") from exc
    emb_dim = int(model.get_sentence_embedding_dimension())
    log.info("Model=%s device=%s dim=%d", args.model, device, emb_dim)

    out_manifest_path = metadata_dir / "manifest.json"
    out_manifest = read_json(out_manifest_path, default=None)
    if out_manifest is None:
        out_manifest = {
            "stage": STATUS_STAGE,
            "schema_version": 1,
            "created_at_utc": now_iso(),
            "model": args.model,
            "device": device,
            "embedding_dim": emb_dim,
            "normalize_embeddings": True,
            "input_manifest": str(input_manifest),
            "shards": [],
            "totals": {"rows": 0, "shards": 0},
        }

    completed = {int(s["shard_id"]): s for s in out_manifest.get("shards", [])}
    shards = data["shards"][: args.limit_shards] if args.limit_shards > 0 else data["shards"]

    update_stage_status(
        status_dir,
        STATUS_STAGE,
        "running",
        {"model": args.model, "device": device, "input_manifest": str(input_manifest)},
    )

    for shard in shards:
        shard_id = int(shard["shard_id"])
        shard_name = shard["name"]
        out_emb = out_dir / f"{shard_name}.npy"
        out_ids = metadata_dir / f"{shard_name}_ids.jsonl"
        in_data = resolve_from_manifest(input_manifest, shard["data_path"])
        in_ids = resolve_from_manifest(input_manifest, shard["ids_path"])

        if shard_id in completed and out_emb.exists() and out_ids.exists():
            log.info("Skip shard %s (already embedded)", shard_name)
            continue

        log.info("Embedding shard %s", shard_name)
        texts = load_texts(in_data)
        emb = model.encode(
            texts,
            batch_size=args.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)

        tmp_emb = out_emb.with_suffix(".npy.tmp")
        with tmp_emb.open("wb") as f:
            np.save(f, emb)
        tmp_emb.replace(out_emb)
        copy_ids(in_ids, out_ids)

        out_record = {
            "shard_id": shard_id,
            "name": shard_name,
            "embedding_path": str(out_emb),
            "ids_path": str(out_ids),
            "rows": int(emb.shape[0]),
            "dim": int(emb.shape[1]),
            "bytes": out_emb.stat().st_size,
            "sha256": sha256_file(out_emb),
            "ids_sha256": sha256_file(out_ids),
        }

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

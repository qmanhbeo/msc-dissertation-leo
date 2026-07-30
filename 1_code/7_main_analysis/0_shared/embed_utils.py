"""
Shared per-batch checkpointing helpers for non-sharded corpus embedders.

`0_embed_reference_and_policy_corpora.py` uses these to write embeddings
incrementally — each batch lands on disk immediately and killed runs resume
from the last completed batch rather than re-encoding.

The research-paper shard embedder (`0_embed_paper_shards.py`) uses a different
per-shard checkpointing model and is NOT unified onto this module yet.
"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import os

from shard_pipeline_utils import atomic_write_json

log = logging.getLogger(__name__)


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_batch_manifest(
    path: Path,
    *,
    corpus_name: str,
    total_rows: int,
    dim: int,
    completed_batches: list[int],
    rows_completed: int,
    status: str,
) -> None:
    manifest = {
        "corpus": corpus_name,
        "total_rows": total_rows,
        "dim": dim,
        "completed_batches": completed_batches,
        "rows_completed": rows_completed,
        "status": status,
        "last_updated_utc": datetime.utcnow().isoformat(),
    }
    atomic_write_json(path, manifest)


def concatenate_batches(
    tmp_dir: Path,
    emb_path: Path,
    n: int,
    dim: int,
    *,
    ids_meta: list[dict] | None = None,
    ids_path: Path | None = None,
) -> None:
    manifest_path = tmp_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest["status"] = "concatenating"
    atomic_write_json(manifest_path, manifest)

    batch_files = sorted(tmp_dir.glob("batch_*.npy"),
                         key=lambda p: int(p.stem.split("_")[1]))
    log.info("Concatenating %d batch files \u2192 %s", len(batch_files), emb_path)
    all_embs = np.concatenate([np.load(f) for f in batch_files], axis=0)
    if all_embs.shape != (n, dim):
        raise RuntimeError(f"Shape mismatch after concatenation: {all_embs.shape} != ({n}, {dim})")

    tmp_emb = emb_path.with_suffix(".npy.tmp")
    with tmp_emb.open("wb") as f:
        np.save(f, all_embs)
        f.flush()
    if not tmp_emb.exists():
        raise RuntimeError(f"Failed to write {tmp_emb}")
    tmp_emb.replace(emb_path)

    shutil.rmtree(tmp_dir)

    if ids_meta is not None and ids_path is not None:
        with ids_path.open("w") as f:
            json.dump(ids_meta, f)

    log.info("Saved %s \u2192 shape %s", emb_path, all_embs.shape)

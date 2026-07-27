"""
Consolidate the 27 research embedding shards and 27 supervised score shards
into single, memmap-friendly .npy files.

Precision is preserved per model: the consolidated file uses the *native*
dtype of the source shards (float32 for all-mpnet-base-v2, float16 for
all-MiniLM-L6-v2). Consolidation is NOT a precision-reduction step — it only
concatenates shards that already exist on disk into one array.

This is a DERIVED cache. Discipline (mirrors the rest of the pipeline):
  - skip if the output exists and the recorded input-shard checksums still
    match the current embedding/score manifests (unless --overwrite);
  - on stale input or --overwrite, regenerate atomically (write .tmp then
    replace) so a crash never leaves a half-written file;
  - MUST be re-run after a re-embed (embeddings) or re-score (scores), since
    the source shards changed underneath it.

Outputs:
    2_data/3_embedded/{model}/research_shards/research_embeddings.npy
    2_data/5_supervised_scored/{model}/research_scores.npy

Run from project root:
    python 1_code/7_main_analysis/0_shared/consolidate_research_artifacts.py \
        --embed-model all-mpnet-base-v2 [--overwrite] [--kind both|embeddings|scores]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np

from model_utils import (
    DEFAULT_EMBED_MODEL,
    embed_research_dir_for_model,
    scored_dir_for_model,
)
from shard_pipeline_utils import (
    atomic_write_json,
    load_json,
    resolve_manifest_path,
)

log = logging.getLogger(__name__)


def _shard_signature(manifest: dict[str, Any]) -> list[dict[str, str]]:
    """Return [(shard_id, sha256), ...] in sorted shard_id order for staleness checks."""
    shards = sorted(manifest.get("shards", []), key=lambda x: int(x["shard_id"]))
    return [
        {"shard_id": int(s["shard_id"]), "sha256": s.get("sha256", "")}
        for s in shards
    ]


def _signature_matches(sidecar: dict[str, Any], manifest: dict[str, Any]) -> bool:
    return _shard_signature(sidecar) == _shard_signature(manifest)


def consolidate_embeddings(model: str, overwrite: bool = False) -> Path:
    """Concatenate the 27 embedding shards into one native-dtype array.

    Returns the consolidated file path. Preserves the source dtype (fp32/fp16).
    """
    emb_research_dir = embed_research_dir_for_model(model)
    manifest_path = emb_research_dir / "metadata" / "manifest.json"
    out = emb_research_dir / "research_embeddings.npy"
    sidecar = emb_research_dir / "research_embeddings_consolidated.json"

    manifest = load_json(manifest_path)
    shards = sorted(manifest.get("shards", []), key=lambda x: int(x["shard_id"]))
    if not shards:
        raise RuntimeError(f"No embedding shards in manifest: {manifest_path}")

    if out.exists() and not overwrite:
        if sidecar.exists() and _signature_matches(load_json(sidecar), manifest):
            log.info("Embeddings already consolidated and up to date: %s", out)
            return out
        log.warning("Embeddings consolidation is STALE (input shards changed); regenerating.")

    total = int(sum(int(s["rows"]) for s in shards))
    first_path = resolve_manifest_path(shards[0]["embedding_path"], allowed_dirs=(emb_research_dir,))
    first = np.load(first_path)
    dtype = first.dtype
    dim = int(first.shape[1])
    log.info(
        "Consolidating %d embedding shards -> %s (dtype=%s, rows=%d, dim=%d)",
        len(shards), out, dtype, total, dim,
    )

    tmp = out.with_suffix(".npy.tmp")
    mm = np.lib.format.open_memmap(tmp, mode="w+", dtype=dtype, shape=(total, dim))
    offset = 0
    for s in shards:
        arr = np.load(resolve_manifest_path(s["embedding_path"], allowed_dirs=(emb_research_dir,)))
        if arr.shape[1] != dim:
            raise RuntimeError(f"Shard {s['shard_id']} dim {arr.shape[1]} != {dim}")
        mm[offset:offset + arr.shape[0]] = arr
        offset += arr.shape[0]
    mm.flush()
    tmp.replace(out)

    atomic_write_json(
        sidecar,
        {
            "input_manifest": str(manifest_path),
            "shards": _shard_signature(manifest),
            "dtype": str(dtype),
            "shape": [total, dim],
        },
    )
    log.info("Embeddings consolidated: %s", out)
    return out


def consolidate_scores(model: str, overwrite: bool = False) -> Path:
    """Concatenate the 27 supervised score shards into one float32 array."""
    scored_dir = scored_dir_for_model(model)
    manifest_path = scored_dir / "paper_scores_shards" / "metadata" / "manifest.json"
    out = scored_dir / "research_scores.npy"
    sidecar = scored_dir / "research_scores_consolidated.json"

    manifest = load_json(manifest_path)
    shards = sorted(manifest.get("shards", []), key=lambda x: int(x["shard_id"]))
    if not shards:
        raise RuntimeError(f"No score shards in manifest: {manifest_path}")

    if out.exists() and not overwrite:
        if sidecar.exists() and _signature_matches(load_json(sidecar), manifest):
            log.info("Scores already consolidated and up to date: %s", out)
            return out
        log.warning("Scores consolidation is STALE (input shards changed); regenerating.")

    total = int(sum(int(s["rows"]) for s in shards))
    first_path = resolve_manifest_path(shards[0]["score_path"], allowed_dirs=(scored_dir,))
    first = np.load(first_path)
    dim = int(first.shape[1])
    log.info(
        "Consolidating %d score shards -> %s (rows=%d, dim=%d)",
        len(shards), out, total, dim,
    )

    tmp = out.with_suffix(".npy.tmp")
    mm = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float32, shape=(total, dim))
    offset = 0
    for s in shards:
        arr = np.load(resolve_manifest_path(s["score_path"], allowed_dirs=(scored_dir,)))
        if arr.shape[1] != dim:
            raise RuntimeError(f"Score shard {s['shard_id']} dim {arr.shape[1]} != {dim}")
        mm[offset:offset + arr.shape[0]] = arr
        offset += arr.shape[0]
    mm.flush()
    tmp.replace(out)

    atomic_write_json(
        sidecar,
        {
            "input_manifest": str(manifest_path),
            "shards": _shard_signature(manifest),
            "dtype": "float32",
            "shape": [total, dim],
        },
    )
    log.info("Scores consolidated: %s", out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Consolidate research shards into single arrays.")
    parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL,
                        help="Embed model (default: %(default)s)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Force regeneration even if up to date.")
    parser.add_argument("--kind", choices=["both", "embeddings", "scores"], default="both",
                        help="Which artifact to consolidate (default: both).")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    if args.kind in ("both", "embeddings"):
        consolidate_embeddings(args.embed_model, overwrite=args.overwrite)
    if args.kind in ("both", "scores"):
        consolidate_scores(args.embed_model, overwrite=args.overwrite)


if __name__ == "__main__":
    main()

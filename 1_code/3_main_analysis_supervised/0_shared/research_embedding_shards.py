from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np


@dataclass(frozen=True)
class ResearchEmbeddingShard:
    shard_id: int
    name: str
    start: int
    stop: int
    rows: int
    embedding_path: Path
    ids_path: Path


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def resolve_from_manifest(manifest_path: Path, stored_path: str, embed_dir: Path) -> Path:
    """Resolve a model-aware hard-pivot path recorded in a research-embedding manifest."""
    del manifest_path  # hard pivot: no location fallback based on manifest placement
    raw = Path(stored_path)
    if raw.is_absolute():
        if raw.exists():
            return raw
        raise FileNotFoundError(f"Absolute path from manifest does not exist: {raw}")
    expected_prefix = embed_dir.as_posix() + "/"
    if not raw.as_posix().startswith(expected_prefix):
        raise RuntimeError(
            f"Hard pivot violation: expected data path under {expected_prefix}, got: {stored_path}"
        )
    resolved = Path.cwd() / raw
    if resolved.exists():
        return resolved
    raise FileNotFoundError(f"Manifest path does not exist: {stored_path} (resolved: {resolved})")


def iter_research_embedding_shards(manifest_path: Path, embed_dir: Path) -> Iterator[ResearchEmbeddingShard]:
    manifest = load_json(manifest_path)
    shards = sorted(manifest.get("shards", []), key=lambda x: int(x["shard_id"]))
    offset = 0
    for shard in shards:
        rows = int(shard["rows"])
        start = offset
        stop = offset + rows
        yield ResearchEmbeddingShard(
            shard_id=int(shard["shard_id"]),
            name=str(shard["name"]),
            start=start,
            stop=stop,
            rows=rows,
            embedding_path=resolve_from_manifest(manifest_path, shard["embedding_path"], embed_dir),
            ids_path=resolve_from_manifest(manifest_path, shard["ids_path"], embed_dir),
        )
        offset = stop


def total_research_embedding_rows(manifest_path: Path, embed_dir: Path) -> int:
    total = 0
    for shard in iter_research_embedding_shards(manifest_path, embed_dir):
        total += shard.rows
    return total


def load_sampled_research_embeddings(
    manifest_path: Path,
    sampled_global_indices: np.ndarray,
    embed_dir: Path,
) -> np.ndarray:
    """
    Load only the requested global row indices from research embedding shards.

    `sampled_global_indices` must be 1D, unique, and sorted in ascending order.
    """
    if sampled_global_indices.ndim != 1:
        raise ValueError("sampled_global_indices must be 1D")
    if sampled_global_indices.size == 0:
        raise ValueError("sampled_global_indices must not be empty")
    if np.any(sampled_global_indices[1:] <= sampled_global_indices[:-1]):
        raise ValueError("sampled_global_indices must be strictly increasing")

    parts: list[np.ndarray] = []
    cursor = 0
    n_total = int(sampled_global_indices.size)

    for shard in iter_research_embedding_shards(manifest_path, embed_dir):
        if cursor >= n_total:
            break
        left = int(np.searchsorted(sampled_global_indices, shard.start, side="left"))
        right = int(np.searchsorted(sampled_global_indices, shard.stop, side="left"))
        if right <= left:
            continue
        local_indices = sampled_global_indices[left:right] - shard.start
        emb = np.load(shard.embedding_path, mmap_mode="r")
        parts.append(np.asarray(emb[local_indices], dtype=np.float32))
        cursor = right

    if not parts:
        raise RuntimeError("No research embedding rows were loaded for the requested sample.")

    result = np.concatenate(parts, axis=0)
    if result.shape[0] != sampled_global_indices.size:
        raise RuntimeError(
            f"Sampled research embedding row mismatch: expected {sampled_global_indices.size}, got {result.shape[0]}"
        )
    return result

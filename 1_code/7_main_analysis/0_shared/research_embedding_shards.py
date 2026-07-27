from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from model_utils import embed_research_dir_for_model, scored_dir_for_model
from shard_pipeline_utils import load_json, resolve_manifest_path


def load_consolidated_embeddings(model: str, mmap: bool = True) -> np.ndarray:
    """Load the single consolidated research-embedding array for `model`.

    Returns a memmap (default) or a fully materialised array. The array is in
    sorted shard_id order, row-aligned with `load_consolidated_scores`. Raises
    a clear error if the consolidation has not been built yet.

    Precision is preserved: MPNet returns float32, MiniLM float16. Callers that
    need float32 must upcast slices locally — never materialise the whole array.
    """
    path = embed_research_dir_for_model(model) / "research_embeddings.npy"
    if not path.exists():
        raise FileNotFoundError(
            f"Consolidated embeddings missing for {model}: {path}. "
            f"Run: python 1_code/7_main_analysis/0_shared/consolidate_research_artifacts.py "
            f"--embed-model {model}"
        )
    return np.load(path, mmap_mode="r" if mmap else None)


def load_consolidated_scores(model: str, mmap: bool = True) -> np.ndarray:
    """Load the single consolidated research-score array for `model`.

    Returns a memmap (default) or a fully materialised float32 array, in sorted
    shard_id order, row-aligned with `load_consolidated_embeddings`.
    """
    path = scored_dir_for_model(model) / "research_scores.npy"
    if not path.exists():
        raise FileNotFoundError(
            f"Consolidated scores missing for {model}: {path}. "
            f"Run: python 1_code/7_main_analysis/0_shared/consolidate_research_artifacts.py "
            f"--embed-model {model} --kind scores"
        )
    return np.load(path, mmap_mode="r" if mmap else None)


@dataclass(frozen=True)
class ResearchEmbeddingShard:
    shard_id: int
    name: str
    start: int
    stop: int
    rows: int
    embedding_path: Path
    ids_path: Path



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
            embedding_path=resolve_manifest_path(shard["embedding_path"], allowed_dirs=(embed_dir,)),
            ids_path=resolve_manifest_path(shard["ids_path"], allowed_dirs=(embed_dir,)),
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

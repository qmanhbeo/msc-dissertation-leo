from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from model_utils import preprocessed_dir
from shard_pipeline_utils import load_json, resolve_manifest_path


@dataclass(frozen=True)
class ResearchShard:
    """A research shard with aligned score + embedding artifacts.

    Moved verbatim from 2_appendix/c_sample_stability.py so that every
    consumer of the score/embedding shard alignment shares one implementation.
    """

    shard_id: int
    name: str
    rows: int
    start: int
    stop: int
    score_path: Path
    emb_path: Path
    ids_path: Path | None = None


def build_research_shards(embed_dir: Path, scored_dir: Path) -> tuple[list[ResearchShard], int]:
    score_manifest = load_json(scored_dir / "paper_scores_shards" / "metadata" / "manifest.json")
    emb_manifest = load_json(embed_dir / "metadata" / "manifest.json")
    score_shards = sorted(score_manifest.get("shards", []), key=lambda x: int(x["shard_id"]))
    emb_shards = sorted(emb_manifest.get("shards", []), key=lambda x: int(x["shard_id"]))
    if len(score_shards) != len(emb_shards):
        raise RuntimeError(
            f"Shard count mismatch: score={len(score_shards)} embedding={len(emb_shards)}"
        )

    shards: list[ResearchShard] = []
    offset = 0
    for score_shard, emb_shard in zip(score_shards, emb_shards):
        score_id = int(score_shard["shard_id"])
        emb_id = int(emb_shard["shard_id"])
        if score_id != emb_id or score_shard["name"] != emb_shard["name"]:
            raise RuntimeError(
                "Research score/embedding manifests are not aligned on shard_id/name: "
                f"score=({score_id}, {score_shard['name']}) "
                f"embedding=({emb_id}, {emb_shard['name']})"
            )
        rows = int(score_shard["rows"])
        if rows != int(emb_shard["rows"]):
            raise RuntimeError(
                f"Row mismatch for shard {score_shard['name']}: score={rows} embedding={emb_shard['rows']}"
            )

        ids_stored = emb_shard.get("ids_path", "")
        ids_path = resolve_manifest_path(ids_stored, allowed_dirs=(embed_dir, scored_dir, preprocessed_dir())) if ids_stored else None

        shards.append(
            ResearchShard(
                shard_id=score_id,
                name=score_shard["name"],
                rows=rows,
                start=offset,
                stop=offset + rows,
                score_path=resolve_manifest_path(score_shard["score_path"], allowed_dirs=(embed_dir, scored_dir, preprocessed_dir())),
                emb_path=resolve_manifest_path(emb_shard["embedding_path"], allowed_dirs=(embed_dir, scored_dir, preprocessed_dir())),
                ids_path=ids_path,
            )
        )
        offset += rows

    return shards, offset


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

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

import numpy as np


N_SDG = 17


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def resolve_from_manifest(manifest_path: Path, stored_path: str) -> Path:
    """Resolve a canonical hard-pivot path recorded in a manifest."""
    del manifest_path  # hard pivot: no location fallback based on manifest placement
    raw = Path(stored_path)
    if raw.is_absolute():
        if raw.exists():
            return raw
        raise FileNotFoundError(f"Absolute path from manifest does not exist: {raw}")
    if not str(raw).startswith("data/3_scored/"):
        raise RuntimeError(
            f"Hard pivot violation: expected data path under data/3_scored/, got: {stored_path}"
        )
    resolved = Path.cwd() / raw
    if resolved.exists():
        return resolved
    raise FileNotFoundError(f"Manifest path does not exist: {stored_path} (resolved: {resolved})")


def iter_research_score_shards(manifest_path: Path) -> Iterator[tuple[int, np.ndarray]]:
    manifest = load_json(manifest_path)
    shards = sorted(manifest.get("shards", []), key=lambda x: int(x["shard_id"]))
    for shard in shards:
        shard_id = int(shard["shard_id"])
        score_path = resolve_from_manifest(manifest_path, shard["score_path"])
        yield shard_id, np.load(score_path).astype(np.float32)


def aggregate_research_scores(manifest_path: Path) -> dict[str, Any]:
    """Streaming aggregates from paper score shards needed by downstream analysis."""
    row_count = 0
    hard_counts = np.zeros(N_SDG, dtype=np.int64)
    soft_sums = np.zeros(N_SDG, dtype=np.float64)
    top_sum = 0.0
    top_sum_per_sdg = np.zeros(N_SDG, dtype=np.float64)

    for _, scores in iter_research_score_shards(manifest_path):
        if scores.ndim != 2 or scores.shape[1] != N_SDG:
            raise RuntimeError(f"Expected score shard shape (?, {N_SDG}), got {scores.shape}")
        n = int(scores.shape[0])
        if n == 0:
            continue
        row_count += n
        assignments = scores.argmax(axis=1)
        hard_counts += np.bincount(assignments, minlength=N_SDG)
        soft_sums += scores.sum(axis=0)
        top_vals = scores[np.arange(n), assignments]
        top_sum += float(top_vals.sum())
        top_sum_per_sdg += np.bincount(assignments, weights=top_vals, minlength=N_SDG)

    if row_count == 0:
        raise RuntimeError(f"No rows found in score shards manifest: {manifest_path}")

    hard_profile = hard_counts.astype(np.float64) / float(row_count)
    soft_profile = soft_sums / float(row_count)

    mean_top_per_sdg = np.zeros(N_SDG, dtype=np.float64)
    nonzero = hard_counts > 0
    mean_top_per_sdg[nonzero] = top_sum_per_sdg[nonzero] / hard_counts[nonzero]

    return {
        "n_rows": row_count,
        "hard_counts": hard_counts,
        "hard_profile": hard_profile,
        "soft_profile": soft_profile,
        "mean_top_overall": float(top_sum / float(row_count)),
        "mean_top_per_sdg": mean_top_per_sdg,
    }

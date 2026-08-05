"""Streaming aggregation over the sharded research score matrices.

The research corpus is segmented with the same token-aware segmenter as the
policy corpus, so one abstract can produce more than one row. Everything in
this module therefore distinguishes two units explicitly:

  segment  — one row of a score shard (one text chunk the encoder saw)
  document — one ABSTRACT (one unique ``openalex_id``), i.e. the mean of its
             segment vectors

``RESEARCH_WEIGHTING_UNIT`` (model_utils) selects which of the two is canonical
for downstream analysis; both are always returned so the diagnostic
(segment-level) profile stays available without a second pass.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from model_utils import (
    N_SDG,
    RENORMALISE_DOC_VECTORS,
    RESEARCH_WEIGHTING_UNIT,
    ZERO_NORM_EPS,
)
from shard_pipeline_utils import load_json, resolve_manifest_path


VALID_UNITS = ("segment", "document")


def iter_research_score_shards(manifest_path: Path, scored_dir: Path) -> Iterator[tuple[int, np.ndarray]]:
    manifest = load_json(manifest_path)
    shards = sorted(manifest.get("shards", []), key=lambda x: int(x["shard_id"]))
    for shard in shards:
        shard_id = int(shard["shard_id"])
        score_path = resolve_manifest_path(shard["score_path"], allowed_dirs=(scored_dir,))
        yield shard_id, np.load(score_path).astype(np.float32)


def read_shard_paper_ids(ids_path: Path) -> list[str]:
    """Read the per-row paper identifier (``openalex_id``) of one score shard."""
    out: list[str] = []
    with ids_path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            paper_id = row.get("openalex_id")
            if not paper_id:
                raise RuntimeError(
                    f"{ids_path}:{line_no} has no 'openalex_id'; paper-level "
                    "aggregation cannot group this row."
                )
            out.append(str(paper_id))
    return out


def paper_run_starts(paper_ids: list[str]) -> np.ndarray:
    """Start offsets of each run of equal consecutive paper ids.

    Segments of one abstract are emitted consecutively by
    1_code/2_segment/1_segment_corpus.py, so a run == a paper. The caller must
    verify that assumption with :func:`assert_papers_contiguous`.
    """
    if not paper_ids:
        return np.zeros(0, dtype=np.int64)
    starts = [0]
    for i in range(1, len(paper_ids)):
        if paper_ids[i] != paper_ids[i - 1]:
            starts.append(i)
    return np.asarray(starts, dtype=np.int64)


def assert_papers_contiguous(paper_ids: list[str], starts: np.ndarray, shard_name: str) -> None:
    """Fail closed if a paper's segments are not contiguous within the shard.

    Run-based grouping is only valid when each paper occupies exactly one
    contiguous span. If a paper id reappeared after an intervening paper the
    number of runs would exceed the number of distinct ids.
    """
    n_distinct = len(set(paper_ids))
    if int(starts.shape[0]) != n_distinct:
        raise RuntimeError(
            f"Shard {shard_name}: {starts.shape[0]} contiguous runs but "
            f"{n_distinct} distinct papers — segments of a paper are not "
            "contiguous, so run-based grouping would mis-assign rows."
        )


def group_rows_by_paper(scores: np.ndarray, starts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean score vector per paper, plus the segment count of each paper.

    Uses ``np.add.reduceat`` over the contiguous runs, which is O(n) and avoids
    materialising a per-paper index.
    """
    n_rows = int(scores.shape[0])
    sums = np.add.reduceat(scores.astype(np.float64), starts, axis=0)
    seg_counts = np.diff(np.append(starts, n_rows)).astype(np.int64)
    means = sums / seg_counts[:, None]
    return means, seg_counts


def paper_units_from_shard(
    emb: np.ndarray,
    scores: np.ndarray,
    paper_ids: list[str],
    shard_name: str,
    *,
    prev_last_paper_id: str | None = None,
    renormalise: bool = RENORMALISE_DOC_VECTORS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Collapse one embedding/score shard to per-abstract (document-level) units.

    Returns
    -------
    paper_emb : (P, d) float64
        Mean of each paper's segment embeddings, L2-renormalised to unit length
        when ``renormalise`` is True (fail closed if any paper vector is
        degenerate).
    paper_assigned : (P,) int64
        Paper-level SDG assignment = argmax of the mean segment score vector
        (NOT a majority vote of per-segment assignments).
    seg_counts : (P,) int64
        Number of segments in each paper.
    last_paper_id : str
        The paper id of the shard's final row, threaded into the next shard for
        the cross-shard boundary check.

    ``emb`` and ``scores`` must be aligned row-for-row with ``paper_ids``.
    """
    n_rows = int(emb.shape[0])
    if int(scores.shape[0]) != n_rows:
        raise RuntimeError(f"Shard {shard_name}: {n_rows} emb rows but {scores.shape[0]} score rows")
    if len(paper_ids) != n_rows:
        raise RuntimeError(f"Shard {shard_name}: {n_rows} emb rows but {len(paper_ids)} id rows")
    starts = paper_run_starts(paper_ids)
    assert_papers_contiguous(paper_ids, starts, shard_name)
    if prev_last_paper_id is not None and paper_ids[0] == prev_last_paper_id:
        raise RuntimeError(
            f"Paper {paper_ids[0]} spans the boundary into shard {shard_name}; "
            "per-shard grouping would split one abstract into two units."
        )
    paper_scores, seg_counts = group_rows_by_paper(scores, starts)
    paper_assigned = paper_scores.argmax(axis=1).astype(np.int64)
    paper_emb, _ = group_rows_by_paper(emb, starts)
    if renormalise:
        norms = np.linalg.norm(paper_emb, axis=1, keepdims=True)
        if np.any(norms < ZERO_NORM_EPS):
            raise RuntimeError(
                f"Shard {shard_name}: a paper embedding has norm < {ZERO_NORM_EPS}; "
                "refusing to normalise a degenerate vector."
            )
        paper_emb = (paper_emb / norms).astype(np.float64)
    return paper_emb, paper_assigned, seg_counts, paper_ids[-1]


class _ProfileAccumulator:
    """Running hard/soft coverage aggregates over a stream of score vectors."""

    def __init__(self) -> None:
        self.n = 0
        self.hard_counts = np.zeros(N_SDG, dtype=np.int64)
        self.soft_sums = np.zeros(N_SDG, dtype=np.float64)
        self.top_sum = 0.0
        self.top_sum_per_sdg = np.zeros(N_SDG, dtype=np.float64)

    def add(self, scores: np.ndarray) -> None:
        n = int(scores.shape[0])
        if n == 0:
            return
        self.n += n
        assignments = scores.argmax(axis=1)
        self.hard_counts += np.bincount(assignments, minlength=N_SDG)
        self.soft_sums += scores.sum(axis=0)
        top_vals = scores[np.arange(n), assignments]
        self.top_sum += float(top_vals.sum())
        self.top_sum_per_sdg += np.bincount(assignments, weights=top_vals, minlength=N_SDG)

    def finish(self, prefix: str) -> dict[str, Any]:
        total = float(self.n)
        mean_top_per_sdg = np.zeros(N_SDG, dtype=np.float64)
        nonzero = self.hard_counts > 0
        mean_top_per_sdg[nonzero] = self.top_sum_per_sdg[nonzero] / self.hard_counts[nonzero]
        return {
            f"{prefix}hard_counts": self.hard_counts,
            f"{prefix}hard_profile": self.hard_counts.astype(np.float64) / total,
            f"{prefix}soft_profile": self.soft_sums / total,
            f"{prefix}mean_top_overall": float(self.top_sum / total),
            f"{prefix}mean_top_per_sdg": mean_top_per_sdg,
        }


def aggregate_research_scores(
    manifest_path: Path,
    scored_dir: Path,
    unit: str = RESEARCH_WEIGHTING_UNIT,
) -> dict[str, Any]:
    """Streaming aggregates from the research score shards.

    Returns the canonical profile (at ``unit`` granularity) under the unprefixed
    keys, and always returns the segment-level profile under ``segment_*`` keys
    for the unweighted diagnostic. ``n_papers`` is None when ``unit`` is
    ``"segment"`` (no grouping pass is performed).
    """
    if unit not in VALID_UNITS:
        raise ValueError(f"unit must be one of {VALID_UNITS}, got {unit!r}")

    manifest = load_json(manifest_path)
    shards = sorted(manifest.get("shards", []), key=lambda x: int(x["shard_id"]))

    seg_acc = _ProfileAccumulator()
    doc_acc = _ProfileAccumulator()
    n_papers = 0
    segments_per_paper_max = 0
    papers_multi_segment = 0
    prev_last_paper_id: str | None = None

    for shard in shards:
        shard_name = str(shard["name"])
        score_path = resolve_manifest_path(shard["score_path"], allowed_dirs=(scored_dir,))
        scores = np.load(score_path).astype(np.float32)
        if scores.ndim != 2 or scores.shape[1] != N_SDG:
            raise RuntimeError(f"Expected score shard shape (?, {N_SDG}), got {scores.shape}")
        if scores.shape[0] == 0:
            continue

        seg_acc.add(scores)

        if unit == "document":
            ids_path = resolve_manifest_path(shard["ids_path"], allowed_dirs=(scored_dir,))
            paper_ids = read_shard_paper_ids(ids_path)
            if len(paper_ids) != scores.shape[0]:
                raise RuntimeError(
                    f"Shard {shard_name}: {scores.shape[0]} score rows but "
                    f"{len(paper_ids)} id rows."
                )
            starts = paper_run_starts(paper_ids)
            assert_papers_contiguous(paper_ids, starts, shard_name)
            if prev_last_paper_id is not None and paper_ids[0] == prev_last_paper_id:
                raise RuntimeError(
                    f"Paper {paper_ids[0]} spans the boundary into shard "
                    f"{shard_name}; per-shard grouping would split one abstract "
                    "into two units."
                )
            prev_last_paper_id = paper_ids[-1]

            means, seg_counts = group_rows_by_paper(scores, starts)
            doc_acc.add(means)
            n_papers += int(means.shape[0])
            segments_per_paper_max = max(segments_per_paper_max, int(seg_counts.max()))
            papers_multi_segment += int((seg_counts > 1).sum())

    if seg_acc.n == 0:
        raise RuntimeError(f"No rows found in score shards manifest: {manifest_path}")

    out: dict[str, Any] = {
        "unit": unit,
        "n_segments": int(seg_acc.n),
        "n_papers": int(n_papers) if unit == "document" else None,
        "papers_multi_segment": int(papers_multi_segment) if unit == "document" else None,
        "segments_per_paper_max": int(segments_per_paper_max) if unit == "document" else None,
    }
    out.update(seg_acc.finish("segment_"))
    canonical = doc_acc if unit == "document" else seg_acc
    out.update(canonical.finish(""))
    return out

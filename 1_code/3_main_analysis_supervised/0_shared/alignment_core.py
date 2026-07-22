"""
Shared alignment helpers for active pipelines.

This module holds reusable primitives used by the active analysis pipeline:
- unit-norm verification for embedding matrices
- research centroid construction from paper embeddings and SDG scores
"""

from __future__ import annotations

import logging

import numpy as np


log = logging.getLogger(__name__)


def verify_unit_norms(emb: np.ndarray, name: str, n_sample: int = 50) -> None:
    """Sample-check that embedding rows are L2-normalised unit vectors."""
    sample = emb[:n_sample]
    norms = np.linalg.norm(sample, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-4):
        log.warning(
            "%s: embeddings may not be L2-normalised (sample norms min=%.4f max=%.4f). "
            "Dot product may differ from cosine similarity.",
            name,
            norms.min(),
            norms.max(),
        )
    else:
        log.info("%s: embedding norms verified ≈ 1.0 ✓", name)


def build_research_centroids(
    paper_emb: np.ndarray,
    paper_scores: np.ndarray,
    n_sdg: int = 17,
) -> tuple[np.ndarray, list[dict]]:
    """
    Build per-SDG research centroids from paper embeddings using hard SDG assignment.

    Each paper is assigned to its top-scoring SDG (argmax over paper_scores). The centroid
    for SDG j is the L2-normalised mean of all papers assigned to SDG j.
    """
    d_emb = paper_emb.shape[1]
    assignments = paper_scores.argmax(axis=1)  # (N,) int in 0..n_sdg-1

    centroids = np.zeros((n_sdg, d_emb), dtype=np.float32)
    meta: list[dict] = []

    for sdg_idx in range(n_sdg):
        sdg = sdg_idx + 1
        mask = assignments == sdg_idx
        n = int(mask.sum())

        if n == 0:
            log.warning(
                "SDG %2d: no papers assigned — research centroid is zero vector "
                "(downstream asymmetry estimates for this SDG are unreliable)",
                sdg,
            )
            meta.append(
                {
                    "sdg": sdg,
                    "n_papers_assigned": 0,
                    "raw_centroid_norm": 0.0,
                    "mean_cos_to_centroid": 0.0,
                    "zero_flag": True,
                }
            )
            continue

        vecs = paper_emb[mask]
        raw = vecs.mean(axis=0)
        norm = float(np.linalg.norm(raw))

        if norm < 1e-8:
            log.warning(
                "SDG %2d: near-zero centroid norm despite n=%d papers — data may be corrupt",
                sdg,
                n,
            )
            meta.append(
                {
                    "sdg": sdg,
                    "n_papers_assigned": n,
                    "raw_centroid_norm": 0.0,
                    "mean_cos_to_centroid": 0.0,
                    "zero_flag": True,
                }
            )
            continue

        unit = (raw / norm).astype(np.float32)
        centroids[sdg_idx] = unit
        mean_cos = float((vecs @ unit).mean())

        meta.append(
            {
                "sdg": sdg,
                "n_papers_assigned": n,
                "raw_centroid_norm": round(norm, 6),
                "mean_cos_to_centroid": round(mean_cos, 6),
                "zero_flag": False,
            }
        )
        log.info("SDG %2d | n_papers=%4d | norm=%.4f | cohesion=%.4f", sdg, n, norm, mean_cos)

    return centroids, meta

"""
Shared alignment helpers for active pipelines.

This module holds reusable primitives used by the active analysis pipeline:
- unit-norm verification for embedding matrices
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



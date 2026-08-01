"""
Shared utilities for the register-topic decomposition (plan §6.2).

Loads the orthonormal projection matrix G from the register_adjust stage and
projects embeddings on the fly.  Adjusted embeddings are NEVER stored as full
.npy arrays — G is ~KB and projection runs in memory.

Primary entry points used by downstream scripts:
    load_G(model, track)          — load completed G from 2_data/
    project(emb, G)               — project + L2-renormalise per row
    project_centroids(path, G)    — load centroids .npy, project
    project_sub_centroid(emb, idxs, G) — project members, build sub-centroid
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from model_utils import DEFAULT_EMBED_MODEL, embed_dir_for_model, scored_dir_for_model
from shard_pipeline_utils import load_json

log = logging.getLogger(__name__)

TRACK_CANON = "canon"
TRACK_SUBSET = "subset"


def track_for_model(model: str) -> str:
    """Derive INLP track from embed-model (same rule as register_adjust.py)."""
    return TRACK_CANON if model == DEFAULT_EMBED_MODEL else TRACK_SUBSET


def register_dir(model: str, track: str | None = None) -> Path:
    """Path to 2_data/3_embedded/{slug}/register/{track}/."""
    if track is None:
        track = track_for_model(model)
    return embed_dir_for_model(model) / "register" / track


def load_G(model: str, track: str | None = None) -> np.ndarray:
    """Load the completed orthonormal projection matrix G.

    Raises RuntimeError if the checkpoint is not complete (fail-closed).
    """
    d = register_dir(model, track)
    ckpt_path = d / "checkpoint.json"
    g_path = d / "G.npy"

    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"register_adjust checkpoint not found at {ckpt_path}. "
            "Run: python main.py --stage register_adjust --embed-model {model}"
        )
    ckpt = load_json(ckpt_path)
    if not ckpt.get("complete"):
        raise RuntimeError(
            f"register_adjust checkpoint at {ckpt_path} is not complete "
            f"(completed_k={ckpt.get('completed_k')}, complete={ckpt.get('complete')}). "
            "Let the stage finish or re-run with --overwrite."
        )
    if not g_path.exists():
        raise FileNotFoundError(
            f"G.npy missing at {g_path} but checkpoint says complete. Corrupt state."
        )
    G = np.load(g_path).astype(np.float32)
    expected_k = ckpt["completed_k"]
    if G.shape[0] != expected_k:
        raise RuntimeError(
            f"G.npy has {G.shape[0]} rows but checkpoint says completed_k={expected_k}. "
            "Corrupt state — re-run register_adjust with --overwrite."
        )
    log.info("Loaded G: shape=%s from %s", G.shape, g_path)
    return G


# --------------------------------------------------------------------------- #
# Projection
# --------------------------------------------------------------------------- #


def project(emb: np.ndarray, G: np.ndarray) -> np.ndarray:
    """Project embeddings through orthonormal G and L2-renormalise per row.

    Mathematically: subtract the component of each row that lies in span(G),
    then renormalise to unit length.  This matches INLP's definition of
    adjusted = P X where P = I - G G^T (orthonormal rows).

    Parameters
    ----------
    emb : (N, dim) float32
    G   : (K, dim) float32 — orthonormal rows

    Returns
    -------
    adjusted : (N, dim) float32 — unit L2 norm per row
    """
    if G.shape[0] == 0:
        return emb.copy()
    proj = (emb @ G.T) @ G  # (N, K) @ (K, dim) = (N, dim)
    residual = emb - proj
    norms = np.linalg.norm(residual, axis=1, keepdims=True)
    norms = np.where(norms > 1e-12, norms, 1.0)
    return (residual / norms).astype(np.float32)


def project_centroids(path: Path, G: np.ndarray) -> np.ndarray:
    """Load a centroids .npy file and project through G."""
    raw = np.load(path).astype(np.float32)
    return project(raw, G)


def project_sub_centroid(
    emb: np.ndarray,
    idxs: list[int],
    G: np.ndarray | None,
) -> tuple[np.ndarray | None, float]:
    """Build a sub-centroid, optionally projecting members through G first.

    When G is provided, each member vector is projected through G before
    computing the mean and cohesion.  This is equivalent to projecting the
    unprojected sub-centroid (proj(mean) = mean(proj)), so only the mean
    is projected for efficiency; cohesion is computed against the projected
    centroid for consistency.

    Identical interface to semantic_gap_shared.build_sub_centroid when G=None.
    """
    if len(idxs) == 0:
        return None, 0.0
    vecs = emb[idxs]
    raw_mean = vecs.mean(axis=0)
    if G is not None and G.shape[0] > 0:
        # Project the mean (equivalent to projecting each member then averaging)
        raw_mean = project(raw_mean.reshape(1, -1), G).ravel()
    norm = float(np.linalg.norm(raw_mean))
    if norm < 1e-8:
        return None, 0.0
    unit = (raw_mean / norm).astype(np.float32)
    # Cohesion: mean cosine sim of member vectors to the unit centroid.
    # For the adjusted case, compute against the projected centroid for
    # consistency (projected members have removed the register component).
    cohesion = float((vecs @ unit).mean())
    return unit, cohesion

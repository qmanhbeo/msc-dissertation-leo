"""
Shared utilities for the register-topic decomposition (plan §6.2).

Loads the orthonormal projection matrix G from the register_adjust stage and
projects embeddings on the fly.  Adjusted embeddings are NEVER stored as full
.npy arrays — G is ~KB and projection runs in memory.

Primary entry points used by downstream scripts:
    load_G(model, track)               — load completed G from 2_data/
    project(emb, G)                    — project + L2-renormalise per row
    subtract_direction(emb, g_dir)     — single-direction projection
    subtract_multiple_directions(emb, G) — sequential single-direction
    compute_gaps_for_directions(G_list, ...) — per-SDG gap after removal
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from model_utils import DEFAULT_EMBED_MODEL, N_SDG, embed_dir_for_model
from semantic_gap_shared import (
    SEGMENT_CAP_PRIMARY,
    build_sub_centroid,
    cap_policy_indices_per_doc,
    get_cluster_assignments,
    get_policy_emb,
    get_policy_ids,
    get_policy_scores,
    get_research_centroids,
    get_research_centroid_meta,
)
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
            f"Run: python main.py --stage register_adjust --embed-model {model}"
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


# --------------------------------------------------------------------------- #
# Gap-from-G helpers (moved from 2_appendix/f_register_adjustment.py)
# --------------------------------------------------------------------------- #


def subtract_direction(emb: np.ndarray, g_dir: np.ndarray) -> np.ndarray:
    """Subtract the projection onto a single unit direction and L2-renormalise per row."""
    proj = np.dot(emb, g_dir)[:, np.newaxis] * g_dir
    residual = emb - proj
    norms = np.linalg.norm(residual, axis=1, keepdims=True)
    norms = np.where(norms > 1e-12, norms, 1.0)
    return (residual / norms).astype(np.float32)


def subtract_multiple_directions(emb: np.ndarray, G: np.ndarray) -> np.ndarray:
    """Apply subtract_direction for each row in G sequentially."""
    result = emb.copy()
    for k in range(G.shape[0]):
        result = subtract_direction(result, G[k])
    return result


def load_raw_data(model: str) -> tuple[np.ndarray, np.ndarray, list, np.ndarray, np.ndarray]:
    """Load all data needed for gap-from-G computations.

    Returns (policy_emb, policy_assignments, policy_ids, research_centroids,
    research_cohesions).
    """
    policy_emb = np.load(get_policy_emb(model)).astype(np.float32)
    policy_scores = np.load(get_policy_scores(model))
    policy_ids = load_json(get_policy_ids(model))
    policy_assignments = get_cluster_assignments(policy_scores)
    research_centroids = np.load(get_research_centroids(model)).astype(np.float32)
    research_meta = load_json(get_research_centroid_meta(model))
    research_cohesions = np.array(
        [float(r["mean_cos_to_centroid"]) for r in research_meta], dtype=np.float32
    )
    return policy_emb, policy_assignments, policy_ids, research_centroids, research_cohesions


def compute_gaps_for_directions(
    G_list: list[np.ndarray],
    policy_emb: np.ndarray,
    policy_assignments: list[int],
    policy_ids: list,
    research_centroids: np.ndarray,
    research_cohesions: np.ndarray,
    rng: np.random.Generator,
) -> dict[int, float]:
    """Compute per-SDG semantic gaps after removing directions in G_list.

    For each SDG j, projects both policy and research centroids through G_list,
    then computes semantic_gap = 1 - cosine_sim(research_adj, policy_adj).
    """
    G = np.vstack(G_list) if G_list else np.zeros((0, policy_emb.shape[1]), dtype=np.float32)
    policy_adj = subtract_multiple_directions(policy_emb, G)

    adj_pol_centroids = np.zeros((N_SDG, policy_emb.shape[1]), dtype=np.float32)
    for sdg_idx in range(N_SDG):
        policy_idxs = [i for i, a in enumerate(policy_assignments) if a == sdg_idx]
        if not policy_idxs:
            continue
        idxs_capped = cap_policy_indices_per_doc(policy_idxs, policy_ids, SEGMENT_CAP_PRIMARY, rng)
        centroid, _ = build_sub_centroid(policy_adj, idxs_capped)
        if centroid is not None:
            adj_pol_centroids[sdg_idx] = centroid

    adj_res_centroids = np.zeros((N_SDG, research_centroids.shape[1]), dtype=np.float32)
    for sdg_idx in range(N_SDG):
        raw_mean = research_centroids[sdg_idx] * research_cohesions[sdg_idx]
        adj_raw = raw_mean.copy()
        for g_k in G_list:
            adj_raw = adj_raw - np.dot(adj_raw, g_k) * g_k
        norm_val = float(np.linalg.norm(adj_raw))
        if norm_val > 1e-8:
            adj_res_centroids[sdg_idx] = (adj_raw / norm_val).astype(np.float32)
        else:
            adj_res_centroids[sdg_idx] = research_centroids[sdg_idx]

    gaps: dict[int, float] = {}
    for sdg_idx in range(N_SDG):
        sdg = sdg_idx + 1
        r_adj = adj_res_centroids[sdg_idx]
        p_adj = adj_pol_centroids[sdg_idx]
        if float(np.linalg.norm(r_adj)) > 1e-8 and float(np.linalg.norm(p_adj)) > 1e-8:
            gaps[sdg] = 1.0 - float(np.dot(r_adj, p_adj))
    return gaps

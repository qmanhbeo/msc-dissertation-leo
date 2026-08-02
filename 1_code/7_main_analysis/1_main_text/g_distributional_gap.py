"""
Main-result robustness: distributional distance metrics for the research-policy semantic gap.

The canonical semantic gap (1_main_text/1_semantic_gap.py) is the cosine distance
between the per-SDG research centroid (mean of ALL LR-assigned paper embeddings)
and the per-SDG policy sub-centroid (mean of cap-50 capped policy segments).
A single mean discards the shape of the two point clouds. This stage recomputes
the research-policy divergence per SDG under a battery of distribution-aware
metrics on the SAME point clouds, changing only the distance functional:

  Full corpus, exact (no sampling):
    - sliced Wasserstein distance (512 seeded projections, complete clouds)
    - Chamfer distance (both directions + symmetric mean, cosine ground metric)
    - modified/max Hausdorff diagnostics
    - Gaussian 2-Wasserstein (Frechet) with mean-term/shape-term decomposition
    - Grassmann distance between top-10 principal subspaces
    - linear-MMD anchor (== raw-mean distance; consistency bridge to the canon)
    - full-corpus centroid gap (must reproduce the canonical gap -- GATE 4)

  Sampled at 50,000 research rows per SDG (Appendix C tier: sampled centroid
  gap 0.3582 +/- 0.0032 vs 0.3563 full corpus, <0.5% bias; SDGs with fewer
  papers use ALL rows, flagged "exhaustive"):
    - exact EMD / 1-Wasserstein (Hungarian assignment, equal-size subsamples)
    - debiased Sinkhorn divergence (entropic cross-check of exact EMD)
    - energy distance (unbiased U-statistic)
    - RBF-MMD (unbiased; median-heuristic bandwidth, recorded)
    - classifier two-sample test (LR, 5-fold held-out ROC-AUC)
    - sampled centroid gap (control column for sampling bias)

RNG inventory (NO undocumented randomness; every generator below is seeded by
a named constant and echoed into the output config):
  1. Policy per-document segment cap: np.random.default_rng(RANDOM_SEED), SDGs
     iterated 0..16 in order -- byte-identical consumption to the canonical run
     (verified by GATE 1 against 4_3_semantic_gap_distances.json).
  2. Research row sampling: np.random.default_rng([seed, sdg, STREAM_SAMPLE])
     for seed in SAMPLE_SEEDS.
  3. EMD equal-size subsample: np.random.default_rng([seed, sdg, STREAM_EMD]).
  4. C2ST equal-size subsample: np.random.default_rng([seed, sdg, STREAM_C2ST]).
  5. SWD projection directions: np.random.default_rng(SWD_PROJECTION_SEED),
     one fixed 768x512 matrix shared by every SDG.
  6. C2ST fold shuffling / LR: random_state=seed (the record's sample seed).

Fail-closed gates (the script halts rather than emit wrong-but-plausible output):
  GATE 1: capped policy cluster sizes == canonical n_policy_segments_capped.
  GATE 2: per-SDG research assignment counts == canonical n_papers.
  GATE 3: streamed research means reproduce committed research_centroids.npy.
  GATE 4: full-corpus centroid gap reproduces the canonical semantic gap.

Outputs (4_outputs/main/{model}/ ; tables in tables/, data in data/):
  data/g_distributional_gap_records.jsonl   incremental, resume-safe records
  data/g_distributional_gap_summary.json    method -> SDG -> gap (+diagnostics)
  tables/num_distributional_gap.tex         LaTeX numeric macros
  tables/tab_distributional_gap.tex         per-SDG comparison table (Part 1 + Part 2)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterator

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = ROOT / "1_code"
ANALYSIS_ROOT = Path(__file__).resolve().parents[1]
SHARED_DIR = ANALYSIS_ROOT / "0_shared"
for path in (CODE_ROOT, SHARED_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import scipy
import sklearn
from scipy.linalg import eigh, subspace_angles
from scipy.optimize import linear_sum_assignment
from scipy.special import logsumexp
from scipy.stats import spearmanr, wasserstein_distance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import semantic_gap_shared
import register_utils
from model_utils import (
    DEFAULT_EMBED_MODEL,
    DEFAULT_OUTPUT_ROOT,
    N_SDG,
    embed_dir_for_model,
    embed_research_dir_for_model,
    output_dir_for_model,
    scored_dir_for_model,
    resolve_model_alias,
)
from shared_utils import fingerprint_of, should_skip, record_fingerprint
from research_embedding_shards import (
    ResearchShard,
    build_research_shards,
    load_sampled_research_embeddings,
)
from semantic_gap_shared import (
    MIN_CLUSTER_SIZE,
    RANDOM_SEED,
    SEGMENT_CAP_PRIMARY,
    build_sub_centroid,
    cap_policy_indices_per_doc,
)
from shard_pipeline_utils import load_json
from shared_utils import ensure_dissertation_outputs, require_output_files

SCHEMA_VERSION = 1
CANONICAL_SEMANTIC_JSON = "4_3_semantic_gap_distances.json"

# Research sample size for the quadratic/assignment metric family. 50k is the
# Appendix C tier where the sampled centroid gap (0.3582 +/- 0.0032) sits within
# 0.5% of the full-corpus anchor (0.3563) -- reusing an already-validated tier
# rather than introducing a new constant. SDGs with fewer assigned papers
# (SDG 1: 16,547; SDG 17: 13,393) use their COMPLETE population ("exhaustive").
RESEARCH_SAMPLE_SIZE = 50_000
# 42 = repo-wide RANDOM_SEED convention; 43 = adjacent replicate seed used only
# to bound sampling noise (skipped for exhaustive SDGs, where a second draw
# would be identical by construction).
SAMPLE_SEEDS = (42, 43)
# Sub-stream tags for per-record generators (rng = default_rng([seed, sdg, tag]))
# so each consumer has an independent, order-insensitive stream.
STREAM_SAMPLE = 0
STREAM_EMD = 1
STREAM_C2ST = 2

# Sliced Wasserstein: number of random 1-D projections. Per-projection SD is
# reported so Monte-Carlo adequacy is auditable from the output itself.
N_SWD_PROJECTIONS = 512
SWD_PROJECTION_SEED = 42

# Entropic OT: eps = SINKHORN_EPS_SCALE * median(cost) (Cuturi 2013 convention);
# fails closed if the marginal violation has not converged.
SINKHORN_EPS_SCALE = 0.1
SINKHORN_MAX_ITER = 2_000
SINKHORN_MARGINAL_TOL = 1e-6

# Grassmann comparison: top-K principal directions per cloud; the explained
# variance captured by each subspace is recorded next to the angle so the
# subspace's representativeness is visible.
PCA_SUBSPACE_K = 10

# Ridge robustness check for the Gaussian-W2 shape term: policy clusters have
# as few as ~1,100 points in 768-d, so their covariances are near rank-deficient.
# The diagnostic recomputes W2 with Sigma + lambda*I, lambda = scale*tr(Sigma)/d.
COV_RIDGE_SCALE = 1e-3

# Classifier two-sample test.
C2ST_FOLDS = 5
C2ST_LR_MAX_ITER = 1_000

# Streaming chunk sizes. STREAM_CHUNK_ROWS matches the embed stage --chunk-size
# precedent; PAIRWISE_CHUNK_ROWS keeps a 50k-column f32 block at ~200 MB.
STREAM_CHUNK_ROWS = 8_192
PAIRWISE_CHUNK_ROWS = 1_024

ROUND_DIGITS = 6
# Tolerance for GATE 3/4: canonical accumulations sum float32 rows in a
# different order than this script's float64 chunked sums.
GATE_ATOL = 1e-4

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the distributional semantic-gap appendix stage.")
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_ROOT))
    p.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL, type=resolve_model_alias, help=argparse.SUPPRESS)
    # Dev-only smoke flag: comma-separated SDG numbers (e.g. "17"). The summary
    # is marked partial when set; never used in canonical runs.
    p.add_argument("--limit-sdgs", default=None, help=argparse.SUPPRESS)
    p.add_argument("--embeddings", choices=["raw", "adjusted"], default="raw",
                   help="Use raw (default) or register-adjusted embeddings (project via G).")
    p.add_argument("--overwrite", action="store_true", help=argparse.SUPPRESS)
    return p.parse_args()


def config_payload(model: str, sample_seeds: tuple[int, ...] = SAMPLE_SEEDS) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "embed_model": model,
        "research_sample_size": RESEARCH_SAMPLE_SIZE,
        "sample_seeds": list(sample_seeds),
        "stream_tags": {"sample": STREAM_SAMPLE, "emd": STREAM_EMD, "c2st": STREAM_C2ST},
        "n_swd_projections": N_SWD_PROJECTIONS,
        "swd_projection_seed": SWD_PROJECTION_SEED,
        "policy_segment_cap": SEGMENT_CAP_PRIMARY,
        "policy_cap_seed": RANDOM_SEED,
        "sinkhorn_eps_scale": SINKHORN_EPS_SCALE,
        "sinkhorn_max_iter": SINKHORN_MAX_ITER,
        "sinkhorn_marginal_tol": SINKHORN_MARGINAL_TOL,
        "pca_subspace_k": PCA_SUBSPACE_K,
        "cov_ridge_scale": COV_RIDGE_SCALE,
        "c2st_folds": C2ST_FOLDS,
        "c2st_lr_max_iter": C2ST_LR_MAX_ITER,
        "min_cluster_size": MIN_CLUSTER_SIZE,
        "round_digits": ROUND_DIGITS,
        "gate_atol": GATE_ATOL,
        "versions": {
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
        },
    }


def compute_config_hash(cfg: dict[str, Any], scored_dir: Path, embed_dir: Path) -> str:
    hasher = hashlib.sha256()
    hasher.update(json.dumps(cfg, sort_keys=True).encode("utf-8"))
    for path in (
        scored_dir / "paper_scores_shards" / "metadata" / "manifest.json",
        embed_dir / "metadata" / "manifest.json",
    ):
        hasher.update(path.read_bytes())
    return hasher.hexdigest()[:16]


def load_canonical(model: str) -> dict[int, dict[str, Any]]:
    """Read the canonical semantic-gap table (a fixed committed reference in
    4_outputs, independent of this run's --output-dir) used by GATES 1-4."""
    data_dir = output_dir_for_model(model) / "data"
    require_output_files(data_dir, [CANONICAL_SEMANTIC_JSON])
    payload = load_json(data_dir / CANONICAL_SEMANTIC_JSON)
    return {int(row["sdg"]): row for row in payload["per_sdg"]}


def load_policy_side(model: str, canonical: dict[int, dict[str, Any]]) -> dict[str, Any]:
    """Load policy embeddings and rebuild the canonical capped per-SDG clusters.

    GATE 1: the capped cluster sizes must equal the canonical
    n_policy_segments_capped exactly, proving the rng consumption order
    (one generator, SDGs 0..16) matches the canonical run byte-for-byte.
    """
    # NOTE: policy.npy is fp16 (embedded with --precision fp16). Do NOT cast to
    # float32 here: 1_semantic_gap.py consumes the raw fp16 rows, and its
    # sub-centroid (float16 mean -> float32 unit) is the canonical measurement
    # object. A float32 cast drifts the GATE 4 gap by ~1.5e-4 and false-fails.
    policy_emb = np.load(semantic_gap_shared.get_policy_emb(model))
    policy_scores = np.load(semantic_gap_shared.get_policy_scores(model)).astype(np.float32)
    policy_ids = load_json(semantic_gap_shared.get_policy_ids(model))
    if policy_scores.shape[0] != len(policy_ids) or policy_emb.shape[0] != len(policy_ids):
        raise RuntimeError(
            "Policy score/embedding/id row mismatch: "
            f"scores={policy_scores.shape[0]} emb={policy_emb.shape[0]} ids={len(policy_ids)}"
        )
    assignments = policy_scores.argmax(axis=1)

    rng = np.random.default_rng(RANDOM_SEED)
    clouds: list[np.ndarray] = []
    capped_counts: list[int] = []
    for sdg_idx in range(N_SDG):
        idxs = np.flatnonzero(assignments == sdg_idx).tolist()
        capped = cap_policy_indices_per_doc(idxs, policy_ids, SEGMENT_CAP_PRIMARY, rng)
        expected = int(canonical[sdg_idx + 1]["n_policy_segments_capped"])
        if len(capped) != expected:
            raise RuntimeError(
                f"GATE 1 FAILED for SDG {sdg_idx + 1}: capped policy cluster has "
                f"{len(capped)} segments, canonical run had {expected}. "
                "The rng consumption order no longer matches the canonical run."
            )
        clouds.append(policy_emb[np.sort(np.asarray(capped, dtype=np.int64))])
        capped_counts.append(len(capped))
    log.info("GATE 1 passed: all 17 capped policy cluster sizes match the canonical run")
    return {"clouds": clouds, "capped_counts": capped_counts, "dim": int(policy_emb.shape[1])}


def build_sdg_row_index(
    shards: list[ResearchShard], canonical: dict[int, dict[str, Any]]
) -> list[np.ndarray]:
    """One pass over the 27 score shards -> sorted global row indices per SDG.

    GATE 2: per-SDG counts must equal the canonical n_papers exactly.
    """
    per_sdg: list[list[np.ndarray]] = [[] for _ in range(N_SDG)]
    for shard in shards:
        scores = np.load(shard.score_path)
        if scores.shape[0] != shard.rows:
            raise RuntimeError(
                f"Score shard {shard.name}: rows {scores.shape[0]} != manifest {shard.rows}"
            )
        assignments = scores.argmax(axis=1)
        for sdg_idx in range(N_SDG):
            local = np.flatnonzero(assignments == sdg_idx)
            if local.size:
                per_sdg[sdg_idx].append(local.astype(np.int64) + shard.start)
    sdg_rows: list[np.ndarray] = []
    for sdg_idx in range(N_SDG):
        rows = (
            np.concatenate(per_sdg[sdg_idx])
            if per_sdg[sdg_idx]
            else np.empty(0, dtype=np.int64)
        )
        rows.sort()
        expected = int(canonical[sdg_idx + 1]["n_papers"])
        if rows.size != expected:
            raise RuntimeError(
                f"GATE 2 FAILED for SDG {sdg_idx + 1}: {rows.size} assigned papers, "
                f"canonical run had {expected}. Assignment definition has drifted."
            )
        sdg_rows.append(rows)
    log.info("GATE 2 passed: all 17 research assignment counts match the canonical run")
    return sdg_rows


def iter_sdg_chunks(
    shards: list[ResearchShard], rows: np.ndarray, chunk_rows: int,
    G: np.ndarray | None = None,
) -> Iterator[np.ndarray]:
    """Yield float32 embedding chunks for the given sorted global row indices."""
    for shard in shards:
        left = int(np.searchsorted(rows, shard.start, side="left"))
        right = int(np.searchsorted(rows, shard.stop, side="left"))
        if right <= left:
            continue
        local = rows[left:right] - shard.start
        emb = np.load(shard.emb_path, mmap_mode="r")
        for lo in range(0, local.size, chunk_rows):
            sel = local[lo : lo + chunk_rows]
            chunk = np.asarray(emb[sel], dtype=np.float32)
            if G is not None and G.shape[0] > 0:
                chunk = register_utils.project(chunk, G)
            yield chunk


def normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < 1e-8:
        raise RuntimeError("Degenerate (near-zero) mean vector; cannot normalise.")
    return vec / norm


def stream_moments_and_chamfer(
    shards: list[ResearchShard],
    sdg_rows: list[np.ndarray],
    policy_clouds: list[np.ndarray],
    active_sdgs: list[int],
    dim: int,
    G: np.ndarray | None = None,
) -> dict[int, dict[str, Any]]:
    """Single shard-major pass over ALL research rows of the active SDGs.

    Accumulates, per SDG:
      - first moment (sum_x) and second moment (sum_xxT), float64
      - Chamfer/Hausdorff statistics against the capped policy cloud
        (cosine ground distance, d = 1 - similarity).
    """
    acc: dict[int, dict[str, Any]] = {}
    for sdg_idx in active_sdgs:
        m = policy_clouds[sdg_idx].shape[0]
        acc[sdg_idx] = {
            "sum_x": np.zeros(dim, dtype=np.float64),
            "sum_xxT": np.zeros((dim, dim), dtype=np.float64),
            "n": 0,
            "sum_min_rp": 0.0,   # sum over research rows of min_j d(r, p_j)
            "max_min_rp": 0.0,   # directed Hausdorff research->policy
            "policy_max_sim": np.full(m, -np.inf, dtype=np.float32),
        }

    for shard in shards:
        emb = None
        for sdg_idx in active_sdgs:
            rows = sdg_rows[sdg_idx]
            left = int(np.searchsorted(rows, shard.start, side="left"))
            right = int(np.searchsorted(rows, shard.stop, side="left"))
            if right <= left:
                continue
            if emb is None:
                emb = np.load(shard.emb_path, mmap_mode="r")
            local = rows[left:right] - shard.start
            a = acc[sdg_idx]
            pol = policy_clouds[sdg_idx]
            for lo in range(0, local.size, STREAM_CHUNK_ROWS):
                chunk32 = np.asarray(emb[local[lo : lo + STREAM_CHUNK_ROWS]], dtype=np.float32)
                if G is not None and G.shape[0] > 0:
                    chunk32 = register_utils.project(chunk32, G)
                chunk64 = chunk32.astype(np.float64)
                a["sum_x"] += chunk64.sum(axis=0)
                a["sum_xxT"] += chunk64.T @ chunk64
                a["n"] += chunk32.shape[0]
                sims = chunk32 @ pol.T
                row_max = sims.max(axis=1)
                min_rp = 1.0 - row_max.astype(np.float64)
                a["sum_min_rp"] += float(min_rp.sum())
                a["max_min_rp"] = max(a["max_min_rp"], float(min_rp.max()))
                np.maximum(a["policy_max_sim"], sims.max(axis=0), out=a["policy_max_sim"])
        del emb
        log.info("moments/chamfer: finished shard %s", shard.name)

    out: dict[int, dict[str, Any]] = {}
    for sdg_idx in active_sdgs:
        a = acc[sdg_idx]
        n = a["n"]
        if n != sdg_rows[sdg_idx].size:
            raise RuntimeError(
                f"SDG {sdg_idx + 1}: streamed {n} rows, expected {sdg_rows[sdg_idx].size}"
            )
        min_pr = 1.0 - a["policy_max_sim"].astype(np.float64)
        chamfer_rp = a["sum_min_rp"] / float(n)
        chamfer_pr = float(min_pr.mean())
        out[sdg_idx] = {
            "n": n,
            "mu": a["sum_x"] / float(n),
            "sum_xxT": a["sum_xxT"],
            "chamfer_research_to_policy": chamfer_rp,
            "chamfer_policy_to_research": chamfer_pr,
            "chamfer_symmetric": 0.5 * (chamfer_rp + chamfer_pr),
            "hausdorff_modified": max(chamfer_rp, chamfer_pr),
            "hausdorff_directed_rp": a["max_min_rp"],
            "hausdorff_directed_pr": float(min_pr.max()),
        }
    return out


def covariance_from_moments(mu: np.ndarray, sum_xxT: np.ndarray, n: int) -> np.ndarray:
    if n < 2:
        raise RuntimeError("Need n >= 2 for a covariance estimate.")
    return (sum_xxT - float(n) * np.outer(mu, mu)) / float(n - 1)


def moments_of_cloud(cloud: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    x = cloud.astype(np.float64)
    n = x.shape[0]
    mu = x.mean(axis=0)
    sum_xxT = x.T @ x
    return mu, covariance_from_moments(mu, sum_xxT, n), n


def swd_direction_matrix(dim: int) -> np.ndarray:
    """Fixed 768 x N_SWD_PROJECTIONS matrix of unit directions (seeded)."""
    rng = np.random.default_rng(SWD_PROJECTION_SEED)
    directions = rng.standard_normal((dim, N_SWD_PROJECTIONS))
    directions /= np.linalg.norm(directions, axis=0, keepdims=True)
    return directions.astype(np.float32)


def compute_swd_full(
    shards: list[ResearchShard],
    rows: np.ndarray,
    policy_cloud: np.ndarray,
    directions: np.ndarray,
    G: np.ndarray | None = None,
) -> dict[str, float]:
    """Sliced Wasserstein over the COMPLETE research cloud vs capped policy cloud.

    Exact per projection (scipy handles unequal sample sizes); no subsampling.
    Peak memory is the projection buffer: n_rows x 512 float32 (SDG 3: ~1.6 GB).
    """
    proj = np.empty((rows.size, N_SWD_PROJECTIONS), dtype=np.float32)
    cursor = 0
    for chunk in iter_sdg_chunks(shards, rows, STREAM_CHUNK_ROWS, G=G):
        proj[cursor : cursor + chunk.shape[0]] = chunk @ directions
        cursor += chunk.shape[0]
    if cursor != rows.size:
        raise RuntimeError(f"SWD projection filled {cursor} rows, expected {rows.size}")
    pol_proj = policy_cloud @ directions
    dists = np.empty(N_SWD_PROJECTIONS, dtype=np.float64)
    for j in range(N_SWD_PROJECTIONS):
        dists[j] = wasserstein_distance(proj[:, j], pol_proj[:, j])
    return {
        "swd_mean": float(dists.mean()),
        "swd_projection_sd": float(dists.std()),
        "swd_mc_standard_error": float(dists.std() / np.sqrt(N_SWD_PROJECTIONS)),
    }


def _psd_sqrt(matrix: np.ndarray) -> np.ndarray:
    """Symmetric PSD square root via eigh with eigenvalue clipping at zero
    (avoids scipy.linalg.sqrtm complex drift on near-singular covariances)."""
    vals, vecs = eigh(matrix)
    vals = np.clip(vals, 0.0, None)
    return (vecs * np.sqrt(vals)) @ vecs.T


def gaussian_w2_terms(
    mu_r: np.ndarray, sig_r: np.ndarray, mu_p: np.ndarray, sig_p: np.ndarray
) -> dict[str, float]:
    """Gaussian 2-Wasserstein squared, decomposed:
        W2^2 = ||mu_r - mu_p||^2  +  Tr(Sig_r + Sig_p - 2 (Sig_r^1/2 Sig_p Sig_r^1/2)^1/2)
    The first term is exactly what the centroid method sees (up to
    normalisation); the second is the covariance-shape mismatch it discards.
    """
    diff = mu_r - mu_p
    mean_term = float(diff @ diff)
    sqrt_r = _psd_sqrt(sig_r)
    cross = sqrt_r @ sig_p @ sqrt_r
    cross_vals = np.clip(eigh(cross, eigvals_only=True), 0.0, None)
    shape_term = float(np.trace(sig_r) + np.trace(sig_p) - 2.0 * np.sqrt(cross_vals).sum())
    shape_term = max(shape_term, 0.0)
    w2_sq = mean_term + shape_term
    return {
        "w2": float(np.sqrt(w2_sq)),
        "w2_squared": w2_sq,
        "mean_term": mean_term,
        "shape_term": shape_term,
        "shape_share": shape_term / w2_sq if w2_sq > 0 else 0.0,
    }


def gaussian_w2_ridged(
    mu_r: np.ndarray, sig_r: np.ndarray, mu_p: np.ndarray, sig_p: np.ndarray
) -> float:
    """Ridge-regularised W2 diagnostic (Sig + lambda I, lambda = scale*tr/d):
    checks that near-singular policy covariances do not drive the shape term."""
    d = sig_r.shape[0]
    rr = sig_r + (COV_RIDGE_SCALE * np.trace(sig_r) / d) * np.eye(d)
    rp = sig_p + (COV_RIDGE_SCALE * np.trace(sig_p) / d) * np.eye(d)
    return gaussian_w2_terms(mu_r, rr, mu_p, rp)["w2"]


def top_k_eig(sig: np.ndarray, k: int) -> tuple[np.ndarray, float]:
    """Top-k eigenvectors (columns) and their explained-variance share."""
    vals, vecs = eigh(sig)
    order = np.argsort(vals)[::-1][:k]
    total = float(np.clip(vals, 0.0, None).sum())
    share = float(np.clip(vals[order], 0.0, None).sum() / total) if total > 0 else 0.0
    return vecs[:, order], share


def grassmann_metrics(sig_r: np.ndarray, sig_p: np.ndarray) -> dict[str, float]:
    basis_r, evr_r = top_k_eig(sig_r, PCA_SUBSPACE_K)
    basis_p, evr_p = top_k_eig(sig_p, PCA_SUBSPACE_K)
    angles = subspace_angles(basis_r, basis_p)
    return {
        "grassmann_distance": float(np.sqrt((angles**2).sum())),
        "mean_principal_angle_deg": float(np.degrees(angles.mean())),
        "max_principal_angle_deg": float(np.degrees(angles.max())),
        "explained_var_share_research": evr_r,
        "explained_var_share_policy": evr_p,
    }


def rms_within_distance(mu: np.ndarray) -> float:
    """RMS pairwise distance within a cloud of UNIT vectors, exactly
    sqrt(E||x-x'||^2) = sqrt(2 - 2 ||mu||^2) -- free from the first moment."""
    return float(np.sqrt(max(2.0 - 2.0 * float(mu @ mu), 0.0)))


def check_research_centroid_gate(
    mu_by_sdg: dict[int, np.ndarray], model: str, active_sdgs: list[int],
    is_adjusted: bool = False,
) -> None:
    """GATE 3: streamed research means must reproduce research_centroids.npy."""
    if is_adjusted:
        log.info("GATE 3 skipped (adjusted mode)")
        return
    committed = np.load(scored_dir_for_model(model) / "research_centroids.npy")
    for sdg_idx in active_sdgs:
        ours = normalize(mu_by_sdg[sdg_idx])
        max_diff = float(np.abs(ours - committed[sdg_idx].astype(np.float64)).max())
        if max_diff > GATE_ATOL:
            raise RuntimeError(
                f"GATE 3 FAILED for SDG {sdg_idx + 1}: streamed research centroid "
                f"deviates from committed research_centroids.npy (max abs diff "
                f"{max_diff:.2e} > {GATE_ATOL}). Moment accumulation is wrong."
            )
    log.info("GATE 3 passed: streamed research means reproduce research_centroids.npy")


def centroid_gap_and_gate4(
    research_centroid: np.ndarray, policy_cloud: np.ndarray, canonical_row: dict[str, Any],
    is_adjusted: bool = False,
) -> float:
    """Full-corpus centroid gap; GATE 4: must reproduce the canonical gap.

    The canonical measurement object is
        sim = research_centroids[sdg] . policy_sub_centroid
    where policy_sub_centroid is build_sub_centroid's float16-mean -> float32
    unit centroid of the capped policy cloud (policy.npy is fp16). Replicating
    exactly what 1_semantic_gap.py computes is what makes the gate meaningful;
    a float64 mean of the same cloud drifts by ~1.5e-4 and would false-fail.
    """
    unit_p, _ = build_sub_centroid(policy_cloud, list(range(policy_cloud.shape[0])))
    if unit_p is None:
        raise RuntimeError(f"GATE 4 FAILED for SDG {canonical_row['sdg']}: empty policy cloud")
    gap = 1.0 - float(np.dot(research_centroid, unit_p))
    if is_adjusted:
        return gap
    expected = canonical_row["semantic_gap"]
    if expected is not None and abs(gap - float(expected)) > GATE_ATOL:
        raise RuntimeError(
            f"GATE 4 FAILED for SDG {canonical_row['sdg']}: full-corpus centroid gap "
            f"{gap:.6f} vs canonical {expected} (tol {GATE_ATOL}). The clouds are "
            "not the canonical measurement objects."
        )
    return gap


def sample_research_cloud(
    manifest_path: Path,
    embed_dir: Path,
    rows: np.ndarray,
    sdg_idx: int,
    seed: int,
    G: np.ndarray | None = None,
) -> tuple[np.ndarray, bool]:
    """Seeded research sample (documented stream: [seed, sdg, STREAM_SAMPLE]).

    Returns (cloud, exhaustive). Exhaustive = the SDG has <= RESEARCH_SAMPLE_SIZE
    papers, so the COMPLETE population is used and no rng is consumed.
    """
    if rows.size <= RESEARCH_SAMPLE_SIZE:
        picked = rows
        exhaustive = True
    else:
        rng = np.random.default_rng([seed, sdg_idx, STREAM_SAMPLE])
        picked = np.sort(rng.choice(rows, size=RESEARCH_SAMPLE_SIZE, replace=False))
        exhaustive = False
    cloud = load_sampled_research_embeddings(manifest_path, picked, embed_dir)
    if G is not None and G.shape[0] > 0:
        cloud = register_utils.project(cloud, G)
    return cloud, exhaustive


def cosine_cost(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine ground cost 1 - a@b.T, float64, clipped to [0, 2]."""
    return np.clip(1.0 - a.astype(np.float64) @ b.astype(np.float64).T, 0.0, 2.0)


def exact_emd(research: np.ndarray, policy: np.ndarray, sdg_idx: int, seed: int) -> dict[str, Any]:
    """Exact 1-Wasserstein via equal-size Hungarian assignment.

    Research side is subsampled to m = |policy| rows (documented stream
    [seed, sdg, STREAM_EMD]); with equal sizes and uniform weights the optimal
    transport plan is a permutation, so linear_sum_assignment is exact OT.
    """
    m = policy.shape[0]
    if research.shape[0] < m:
        raise RuntimeError(
            f"SDG {sdg_idx + 1}: research cloud ({research.shape[0]}) smaller than "
            f"policy cloud ({m}); equal-size EMD impossible."
        )
    rng = np.random.default_rng([seed, sdg_idx, STREAM_EMD])
    sub = research[np.sort(rng.choice(research.shape[0], size=m, replace=False))]
    cost = cosine_cost(sub, policy)
    row_ind, col_ind = linear_sum_assignment(cost)
    return {
        "emd": float(cost[row_ind, col_ind].mean()),
        "n_per_side": int(m),
        "median_cross_cost": float(np.median(cost)),
        "_cost_rr": cosine_cost(sub, sub),
        "_cost_pp": cosine_cost(policy, policy),
        "_cost_rp": cost,
    }


def _sinkhorn_ot(cost: np.ndarray, eps: float) -> float:
    """Log-domain Sinkhorn <P, C> for uniform marginals; fails closed."""
    n, m = cost.shape
    log_a = -np.log(n)
    log_b = -np.log(m)
    neg_c = -cost / eps
    f = np.zeros(n)
    g = np.zeros(m)
    for _ in range(SINKHORN_MAX_ITER):
        f = eps * (log_a - logsumexp((neg_c + g[None, :] / eps), axis=1))
        g = eps * (log_b - logsumexp((neg_c + f[:, None] / eps), axis=0))
        log_plan = neg_c + f[:, None] / eps + g[None, :] / eps
        marginal_err = float(np.abs(np.exp(logsumexp(log_plan, axis=1)) - 1.0 / n).sum())
        if marginal_err < SINKHORN_MARGINAL_TOL:
            plan = np.exp(log_plan)
            return float((plan * cost).sum())
    raise RuntimeError(
        f"Sinkhorn did not converge in {SINKHORN_MAX_ITER} iterations "
        f"(marginal error {marginal_err:.2e} > {SINKHORN_MARGINAL_TOL})."
    )


def sinkhorn_divergence(cost_rp: np.ndarray, cost_rr: np.ndarray, cost_pp: np.ndarray) -> dict[str, float]:
    """Debiased Sinkhorn divergence S = OT(r,p) - (OT(r,r) + OT(p,p)) / 2,
    on the same equal-size subsample as exact EMD (entropic cross-check)."""
    eps = SINKHORN_EPS_SCALE * float(np.median(cost_rp))
    ot_rp = _sinkhorn_ot(cost_rp, eps)
    ot_rr = _sinkhorn_ot(cost_rr, eps)
    ot_pp = _sinkhorn_ot(cost_pp, eps)
    return {
        "sinkhorn_divergence": float(max(ot_rp - 0.5 * (ot_rr + ot_pp), 0.0)),
        "sinkhorn_ot_rp": ot_rp,
        "sinkhorn_epsilon": eps,
    }


def _pair_sums(a: np.ndarray, b: np.ndarray, bandwidth_sq: float, same: bool) -> tuple[float, float, int]:
    """Chunked sums of Euclidean distance and RBF kernel over all pairs.

    For same=True, diagonal pairs are excluded (unbiased U-statistics).
    Unit vectors: ||x-y||^2 = 2 - 2 x.y.
    """
    sum_dist = 0.0
    sum_kern = 0.0
    b64 = b.astype(np.float64)
    for lo in range(0, a.shape[0], PAIRWISE_CHUNK_ROWS):
        block = a[lo : lo + PAIRWISE_CHUNK_ROWS].astype(np.float64) @ b64.T
        d_sq = np.clip(2.0 - 2.0 * block, 0.0, None)
        if same:
            idx = np.arange(block.shape[0])
            d_sq[idx, lo + idx] = np.nan
            sum_dist += float(np.nansum(np.sqrt(d_sq)))
            sum_kern += float(np.nansum(np.exp(-d_sq / (2.0 * bandwidth_sq))))
        else:
            sum_dist += float(np.sqrt(d_sq).sum())
            sum_kern += float(np.exp(-d_sq / (2.0 * bandwidth_sq)).sum())
    n_pairs = a.shape[0] * b.shape[0] - (a.shape[0] if same else 0)
    return sum_dist, sum_kern, n_pairs


def energy_and_rbf_mmd(research: np.ndarray, policy: np.ndarray, bandwidth: float) -> dict[str, float]:
    """Energy distance and unbiased RBF-MMD^2 in one chunked pairwise sweep.

    bandwidth = median cross-distance on the EMD subsample (median heuristic;
    deterministic given the documented STREAM_EMD sample, and recorded)."""
    bw_sq = bandwidth * bandwidth
    d_rp, k_rp, np_rp = _pair_sums(research, policy, bw_sq, same=False)
    d_rr, k_rr, np_rr = _pair_sums(research, research, bw_sq, same=True)
    d_pp, k_pp, np_pp = _pair_sums(policy, policy, bw_sq, same=True)
    e_rp, e_rr, e_pp = d_rp / np_rp, d_rr / np_rr, d_pp / np_pp
    m_rp, m_rr, m_pp = k_rp / np_rp, k_rr / np_rr, k_pp / np_pp
    return {
        "energy_distance": float(max(2.0 * e_rp - e_rr - e_pp, 0.0)),
        "rbf_mmd_squared": float(m_rr + m_pp - 2.0 * m_rp),
        "rbf_bandwidth": float(bandwidth),
        "mean_cross_distance": float(e_rp),
        "mean_within_research_distance": float(e_rr),
        "mean_within_policy_distance": float(e_pp),
    }


def c2st_auc(research: np.ndarray, policy: np.ndarray, sdg_idx: int, seed: int) -> dict[str, float]:
    """Classifier two-sample test: balanced LR, held-out ROC-AUC.

    Research is subsampled to |policy| rows (documented stream
    [seed, sdg, STREAM_C2ST]); power is bounded by the policy side anyway."""
    m = policy.shape[0]
    rng = np.random.default_rng([seed, sdg_idx, STREAM_C2ST])
    sub = research[np.sort(rng.choice(research.shape[0], size=m, replace=False))]
    x = np.vstack([sub, policy]).astype(np.float64)
    y = np.concatenate([np.zeros(m, dtype=np.int64), np.ones(m, dtype=np.int64)])
    skf = StratifiedKFold(n_splits=C2ST_FOLDS, shuffle=True, random_state=seed)
    aucs = []
    for train_idx, test_idx in skf.split(x, y):
        clf = LogisticRegression(max_iter=C2ST_LR_MAX_ITER, random_state=seed)
        clf.fit(x[train_idx], y[train_idx])
        aucs.append(roc_auc_score(y[test_idx], clf.predict_proba(x[test_idx])[:, 1]))
    return {"c2st_auc": float(np.mean(aucs)), "c2st_auc_sd": float(np.std(aucs))}


def sampled_centroid_gap(research: np.ndarray, policy: np.ndarray) -> float:
    """Canonical metric on the sampled cloud (sampling-bias control column)."""
    return 1.0 - float(
        normalize(research.astype(np.float64).mean(axis=0))
        @ normalize(policy.astype(np.float64).mean(axis=0))
    )


def record_key(record: dict[str, Any]) -> tuple:
    return (record["kind"], record["sdg"], record.get("seed"))


def load_existing_records(path: Path, cfg_hash: str) -> dict[tuple, dict[str, Any]]:
    """Resume support: keep records whose config_hash matches; else start fresh."""
    if not path.exists():
        return {}
    records: dict[tuple, dict[str, Any]] = {}
    stale = False
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                # Only possible for a truncated trailing line left by a kill
                # during append_record. A completed record is always valid JSON,
                # so skipping it is safe and idempotent on resume.
                log.warning("Skipping malformed trailing record line (interrupted write?)")
                continue
            if row.get("config_hash") != cfg_hash:
                stale = True
                break
            records[record_key(row)] = row
    if stale:
        log.info("Existing records have a different config hash — starting fresh")
        path.unlink()
        return {}
    log.info("Resuming: %d completed records found", len(records))
    return records


def append_record(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _round(value: float) -> float:
    return round(float(value), ROUND_DIGITS)


def build_full_record(
    sdg_idx: int,
    cfg_hash: str,
    stream_out: dict[str, Any],
    swd_out: dict[str, float],
    policy_cloud: np.ndarray,
    canonical_row: dict[str, Any],
    research_centroid: np.ndarray,
    is_adjusted: bool = False,
) -> dict[str, Any]:
    mu_r = stream_out["mu"]
    sig_r = covariance_from_moments(mu_r, stream_out["sum_xxT"], stream_out["n"])
    mu_p, sig_p, n_p = moments_of_cloud(policy_cloud)
    w2 = gaussian_w2_terms(mu_r, sig_r, mu_p, sig_p)
    gap_full = centroid_gap_and_gate4(research_centroid, policy_cloud, canonical_row, is_adjusted=is_adjusted)
    diff = mu_r - mu_p
    record = {
        "config_hash": cfg_hash,
        "kind": "full",
        "sdg": sdg_idx + 1,
        "n_research": int(stream_out["n"]),
        "n_policy_capped": int(n_p),
        "unreliable": bool(
            stream_out["n"] < MIN_CLUSTER_SIZE or n_p < MIN_CLUSTER_SIZE
        ),
        "centroid_gap_full_corpus": _round(gap_full),
        "canonical_gap": canonical_row["semantic_gap"],
        "swd": _round(swd_out["swd_mean"]),
        "swd_projection_sd": _round(swd_out["swd_projection_sd"]),
        "swd_mc_standard_error": _round(swd_out["swd_mc_standard_error"]),
        "chamfer_symmetric": _round(stream_out["chamfer_symmetric"]),
        "chamfer_research_to_policy": _round(stream_out["chamfer_research_to_policy"]),
        "chamfer_policy_to_research": _round(stream_out["chamfer_policy_to_research"]),
        "hausdorff_modified": _round(stream_out["hausdorff_modified"]),
        "hausdorff_directed_rp": _round(stream_out["hausdorff_directed_rp"]),
        "hausdorff_directed_pr": _round(stream_out["hausdorff_directed_pr"]),
        "gaussian_w2": _round(w2["w2"]),
        "gaussian_w2_mean_term": _round(w2["mean_term"]),
        "gaussian_w2_shape_term": _round(w2["shape_term"]),
        "gaussian_w2_shape_share": _round(w2["shape_share"]),
        "gaussian_w2_ridged": _round(gaussian_w2_ridged(mu_r, sig_r, mu_p, sig_p)),
        "linear_mmd_squared": _round(float(diff @ diff)),
        "rms_within_research": _round(rms_within_distance(mu_r)),
        "rms_within_policy": _round(rms_within_distance(mu_p)),
        "n_over_dim_policy": _round(n_p / policy_cloud.shape[1]),
    }
    record.update({f"grassmann_{k}": _round(v) for k, v in grassmann_metrics(sig_r, sig_p).items()})
    return record


def build_sampled_record(
    sdg_idx: int,
    seed: int,
    cfg_hash: str,
    research_cloud: np.ndarray,
    exhaustive: bool,
    policy_cloud: np.ndarray,
) -> dict[str, Any]:
    t0 = time.time()
    emd_out = exact_emd(research_cloud, policy_cloud, sdg_idx, seed)
    sink = sinkhorn_divergence(emd_out["_cost_rp"], emd_out["_cost_rr"], emd_out["_cost_pp"])
    bandwidth = float(np.sqrt(np.clip(2.0 * emd_out["median_cross_cost"], 1e-12, None)))
    e_mmd = energy_and_rbf_mmd(research_cloud, policy_cloud, bandwidth)
    c2st = c2st_auc(research_cloud, policy_cloud, sdg_idx, seed)
    record = {
        "config_hash": cfg_hash,
        "kind": "sampled",
        "sdg": sdg_idx + 1,
        "seed": seed,
        "n_research_sampled": int(research_cloud.shape[0]),
        "exhaustive": exhaustive,
        "n_policy_capped": int(policy_cloud.shape[0]),
        "centroid_gap_sampled": _round(sampled_centroid_gap(research_cloud, policy_cloud)),
        "exact_emd": _round(emd_out["emd"]),
        "emd_n_per_side": emd_out["n_per_side"],
        "sinkhorn_divergence": _round(sink["sinkhorn_divergence"]),
        "sinkhorn_epsilon": _round(sink["sinkhorn_epsilon"]),
        "energy_distance": _round(e_mmd["energy_distance"]),
        "rbf_mmd_squared": _round(e_mmd["rbf_mmd_squared"]),
        "rbf_bandwidth": _round(e_mmd["rbf_bandwidth"]),
        "mean_cross_distance": _round(e_mmd["mean_cross_distance"]),
        "mean_within_research_distance": _round(e_mmd["mean_within_research_distance"]),
        "mean_within_policy_distance": _round(e_mmd["mean_within_policy_distance"]),
        "c2st_auc": _round(c2st["c2st_auc"]),
        "c2st_auc_sd": _round(c2st["c2st_auc_sd"]),
        "runtime_seconds": round(time.time() - t0, 1),
    }
    return record


# method name -> (record kind, gap field, [diagnostic fields copied per SDG])
METHOD_SPECS: dict[str, tuple[str, str, list[str]]] = {
    "sliced_wasserstein": ("full", "swd", ["swd_projection_sd", "swd_mc_standard_error"]),
    "chamfer_symmetric": (
        "full",
        "chamfer_symmetric",
        ["chamfer_research_to_policy", "chamfer_policy_to_research"],
    ),
    "hausdorff_modified": (
        "full",
        "hausdorff_modified",
        ["hausdorff_directed_rp", "hausdorff_directed_pr"],
    ),
    "gaussian_w2": (
        "full",
        "gaussian_w2",
        [
            "gaussian_w2_mean_term",
            "gaussian_w2_shape_term",
            "gaussian_w2_shape_share",
            "gaussian_w2_ridged",
            "n_over_dim_policy",
        ],
    ),
    "grassmann": (
        "full",
        "grassmann_grassmann_distance",
        [
            "grassmann_mean_principal_angle_deg",
            "grassmann_max_principal_angle_deg",
            "grassmann_explained_var_share_research",
            "grassmann_explained_var_share_policy",
        ],
    ),
    "linear_mmd": ("full", "linear_mmd_squared", []),
    "centroid_gap_full_corpus": ("full", "centroid_gap_full_corpus", []),
    "exact_emd": ("sampled", "exact_emd", ["emd_n_per_side"]),
    "sinkhorn": ("sampled", "sinkhorn_divergence", ["sinkhorn_epsilon"]),
    "energy_distance": (
        "sampled",
        "energy_distance",
        [
            "mean_cross_distance",
            "mean_within_research_distance",
            "mean_within_policy_distance",
        ],
    ),
    "rbf_mmd": ("sampled", "rbf_mmd_squared", ["rbf_bandwidth"]),
    "c2st": ("sampled", "c2st_auc", ["c2st_auc_sd"]),
    "centroid_gap_sampled": ("sampled", "centroid_gap_sampled", ["exhaustive"]),
}


def ranks_from_gaps(gaps: dict[str, float]) -> dict[str, int]:
    """Rank 1 = largest gap (most divergent), matching the manuscript convention."""
    ordered = sorted(gaps, key=lambda k: gaps[k], reverse=True)
    return {key: rank + 1 for rank, key in enumerate(ordered)}


def top3_from_gaps(gaps: dict[str, float]) -> list[int]:
    ordered = sorted(gaps, key=lambda k: gaps[k], reverse=True)
    return [int(k.replace("SDG", "")) for k in ordered[:3]]


def build_summary(
    records: dict[tuple, dict[str, Any]],
    canonical: dict[int, dict[str, Any]],
    cfg: dict[str, Any],
    cfg_hash: str,
    active_sdgs: list[int],
    partial: bool,
) -> dict[str, Any]:
    sdg_keys = [f"SDG{s + 1}" for s in active_sdgs]
    canonical_gaps = {
        f"SDG{s + 1}": float(canonical[s + 1]["semantic_gap"]) for s in active_sdgs
    }
    canonical_vector = [canonical_gaps[k] for k in sdg_keys]

    methods: dict[str, Any] = {}
    for method, (kind, field, diag_fields) in METHOD_SPECS.items():
        gaps: dict[str, float] = {}
        replicate: dict[str, float] = {}
        diagnostics: dict[str, dict[str, Any]] = {}
        for s in active_sdgs:
            sdg_key = f"SDG{s + 1}"
            primary = records.get((kind, s + 1, SAMPLE_SEEDS[0] if kind == "sampled" else None))
            if primary is None:
                continue
            gaps[sdg_key] = float(primary[field])
            diagnostics[sdg_key] = {d: primary.get(d) for d in diag_fields}
            if kind == "sampled":
                alt = records.get((kind, s + 1, SAMPLE_SEEDS[1]))
                if alt is not None:
                    replicate[sdg_key] = float(alt[field])
        if not gaps:
            continue
        entry: dict[str, Any] = {
            "gap_by_sdg": {k: _round(v) for k, v in gaps.items()},
            "rank_by_sdg": ranks_from_gaps(gaps),
            "top3_sdgs": top3_from_gaps(gaps),
            "diagnostics": diagnostics,
        }
        if len(gaps) == len(sdg_keys) and len(sdg_keys) >= 3:
            rho, pval = spearmanr(canonical_vector, [gaps[k] for k in sdg_keys])
            entry["spearman_vs_canonical"] = {"rho": _round(rho), "p": _round(pval)}
        if replicate:
            primary_ranks = entry["rank_by_sdg"]
            alt_ranks = ranks_from_gaps(
                {k: replicate.get(k, gaps[k]) for k in gaps}
            )
            entry["seed_replicate"] = {
                "seed": SAMPLE_SEEDS[1],
                "max_abs_gap_delta": _round(
                    max(abs(replicate[k] - gaps[k]) for k in replicate)
                ),
                "max_abs_rank_shift": max(
                    abs(alt_ranks[k] - primary_ranks[k]) for k in gaps
                ),
                "n_sdgs_replicated": len(replicate),
            }
        methods[method] = entry

    methods["centroid_gap_canonical"] = {
        "gap_by_sdg": {k: _round(v) for k, v in canonical_gaps.items()},
        "rank_by_sdg": ranks_from_gaps(canonical_gaps),
        "top3_sdgs": top3_from_gaps(canonical_gaps),
        "diagnostics": {},
    }

    per_sdg_context = {}
    for s in active_sdgs:
        full = records.get(("full", s + 1, None), {})
        sampled = records.get(("sampled", s + 1, SAMPLE_SEEDS[0]), {})
        per_sdg_context[f"SDG{s + 1}"] = {
            "n_research": full.get("n_research"),
            "n_research_sampled": sampled.get("n_research_sampled"),
            "exhaustive_sample": sampled.get("exhaustive"),
            "n_policy_capped": full.get("n_policy_capped"),
            "unreliable": full.get("unreliable"),
            "rms_within_research": full.get("rms_within_research"),
            "rms_within_policy": full.get("rms_within_policy"),
        }

    max_seed_replicate_gap_delta = None
    for entry in methods.values():
        rep = entry.get("seed_replicate")
        if rep and rep.get("max_abs_gap_delta") is not None:
            delta = rep["max_abs_gap_delta"]
            max_seed_replicate_gap_delta = (
                delta if max_seed_replicate_gap_delta is None else max(max_seed_replicate_gap_delta, delta)
            )

    return {
        "method": "distributional_semantic_gap_battery",
        "partial": partial,
        "config": cfg,
        "config_hash": cfg_hash,
        "max_seed_replicate_gap_delta": max_seed_replicate_gap_delta,
        "methods": methods,
        "per_sdg_context": per_sdg_context,
    }


AUTOGEN_HEADER = (
    "% Auto-generated by 1_code/7_main_analysis/1_main_text/g_distributional_gap.py"
    " — do not edit manually"
)
# Table columns, partitioned into two 7-column tables (canonical + 6 methods
# each) so all 13 distance functionals are shown without overflowing the page.
# centroid_gap_full_corpus is omitted: GATE 4 proves it equals the canonical
# gap to 1e-4, so it is a validation control, not an independent method.
TABLE_PART1 = [
    ("centroid_gap_canonical", "Canonical"),
    ("sliced_wasserstein", "SWD"),
    ("chamfer_symmetric", "Chamfer"),
    ("hausdorff_modified", "Hausdorff"),
    ("gaussian_w2", "Gauss.\\ $W_2$"),
    ("grassmann", "Grassmann"),
    ("linear_mmd", "linear MMD"),
]
TABLE_PART2 = [
    ("centroid_gap_canonical", "Canonical"),
    ("exact_emd", "EMD"),
    ("sinkhorn", "Sinkhorn"),
    ("energy_distance", "Energy"),
    ("rbf_mmd", r"MMD$^2$"),
    ("c2st", "C2ST AUC"),
    ("centroid_gap_sampled", "Centroid (samp.)"),
]
# Macro-name fragments (LaTeX macro names cannot contain digits).
MACRO_WORDS = {
    "exact_emd": "Emd",
    "sliced_wasserstein": "Swd",
    "rbf_mmd": "Mmd",
    "energy_distance": "Energy",
    "gaussian_w2": "Frechet",
    "chamfer_symmetric": "Chamfer",
    "c2st": "Ctst",
    "sinkhorn": "Sinkhorn",
    "hausdorff_modified": "Hausdorff",
    "grassmann": "Grassmann",
}


def _render_part(
    summary: dict[str, Any],
    part: list[tuple[str, str]],
    label: str,
    caption: str,
) -> list[str]:
    methods = summary["methods"]
    active_cols = [(m, h) for m, h in part if m in methods]
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\resizebox{\textwidth}{!}{",
        r"\begin{tabular}{l" + "c" * len(active_cols) + "}",
        r"\toprule",
        "SDG & " + " & ".join(h for _, h in active_cols) + r" \\",
        r"\midrule",
    ]
    sdg_keys = sorted(
        methods["centroid_gap_canonical"]["gap_by_sdg"],
        key=lambda k: int(k.replace("SDG", "")),
    )
    for sdg_key in sdg_keys:
        cells = [sdg_key.replace("SDG", "")]
        for method, _ in active_cols:
            value = methods[method]["gap_by_sdg"].get(sdg_key)
            cells.append("--" if value is None else f"{value:.3f}")
        lines.append(" & ".join(cells) + r" \\")
    lines.append(r"\midrule")
    rho_cells = [r"$\rho$ vs.\ canonical"]
    for method, _ in active_cols:
        rho = methods[method].get("spearman_vs_canonical", {}).get("rho")
        rho_cells.append("--" if rho is None else f"{rho:.3f}")
    lines.extend([
        " & ".join(rho_cells) + r" \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        r"\end{table}",
    ])
    return lines


def write_tables(tables_dir: Path, summary: dict[str, Any]) -> None:
    methods = summary["methods"]
    num_lines = [AUTOGEN_HEADER]
    for method, word in MACRO_WORDS.items():
        rho = methods.get(method, {}).get("spearman_vs_canonical", {}).get("rho")
        if rho is not None:
            num_lines.append(rf"\newcommand{{\DistGapSpearman{word}}}{{{rho:.3f}}}")
    canonical_top3 = set(methods["centroid_gap_canonical"]["top3_sdgs"])
    overlaps = [
        len(canonical_top3 & set(methods[m]["top3_sdgs"]))
        for part in (TABLE_PART1, TABLE_PART2)
        for m, _ in part
        if m != "centroid_gap_canonical" and m in methods
    ]
    if overlaps:
        num_lines.append(
            rf"\newcommand{{\DistGapTopThreeOverlapMin}}{{{min(overlaps)}}}"
        )
    shape_shares = [
        row.get("gaussian_w2_shape_share")
        for row in methods.get("gaussian_w2", {}).get("diagnostics", {}).values()
        if row.get("gaussian_w2_shape_share") is not None
    ]
    if shape_shares:
        num_lines.append(
            rf"\newcommand{{\DistGapShapeShareMean}}{{{np.mean(shape_shares) * 100:.0f}}}"
        )
    max_delta = summary.get("max_seed_replicate_gap_delta")
    if max_delta is not None:
        num_lines.append(
            rf"\newcommand{{\DistGapMaxSeedDelta}}{{{max_delta:.3f}}}"
        )
    (tables_dir / "num_distributional_gap.tex").write_text(
        "\n".join(num_lines) + "\n", encoding="utf-8"
    )

    part1 = _render_part(
        summary, TABLE_PART1, "tab:distributional-gap",
        "Per-SDG research--policy gap under distribution-aware metrics versus the "
        "canonical centroid gap (Part 1 of 2: full-corpus exact metrics). SWD = "
        "sliced Wasserstein; MMD$^2$ = squared RBF maximum mean discrepancy; "
        "Gauss.\\ $W_2$ = Gaussian 2-Wasserstein; C2ST = classifier two-sample test "
        "AUC. The final row reports Spearman $\\rho$ between each metric's ranking "
        "and the canonical ranking across the 17 SDGs.",
    )
    part2 = _render_part(
        summary, TABLE_PART2, "tab:distributional-gap-cont",
        "Per-SDG research--policy gap under distribution-aware metrics versus the "
        "canonical centroid gap (Part 2 of 2: sampled metrics; continuation of "
        "Table~\\ref{tab:distributional-gap}). EMD = exact 1-Wasserstein; MMD$^2$ = "
        "squared RBF-MMD; Centroid (samp.) = sampled centroid-gap control column.",
    )
    (tables_dir / "tab_distributional_gap.tex").write_text(
        "\n".join(part1 + [""] + part2) + "\n", encoding="utf-8"
    )


def parse_limit_sdgs(raw: str | None) -> list[int]:
    """Return active 0-based SDG indices; full range unless --limit-sdgs (dev)."""
    if raw is None:
        return list(range(N_SDG))
    picked = sorted({int(tok) for tok in raw.split(",") if tok.strip()})
    if any(s < 1 or s > N_SDG for s in picked):
        raise ValueError(f"--limit-sdgs out of range 1..{N_SDG}: {raw}")
    return [s - 1 for s in picked]


def run(args: argparse.Namespace) -> None:
    model = args.embed_model
    output_dir = Path(args.output_dir)
    active_sdgs = parse_limit_sdgs(args.limit_sdgs)
    partial = len(active_sdgs) != N_SDG
    is_adjusted = args.embeddings == "adjusted"
    if partial:
        log.warning("DEV MODE: --limit-sdgs=%s — summary will be marked partial", args.limit_sdgs)

    embed_dir = embed_research_dir_for_model(model)
    scored_dir = scored_dir_for_model(model)
    manifest_path = embed_dir / "metadata" / "manifest.json"
    layout = ensure_dissertation_outputs(
        output_dir, subdir="main", model=model
    )
    if is_adjusted:
        import dataclasses
        adj_root = layout.root / "adjusted"
        adj_root.mkdir(parents=True, exist_ok=True)
        adj_data = adj_root / "data"
        adj_tables = adj_root / "tables"
        adj_data.mkdir(parents=True, exist_ok=True)
        adj_tables.mkdir(parents=True, exist_ok=True)
        layout = dataclasses.replace(
            layout,
            root=adj_root,
            data_dir=adj_data,
            tables_dir=adj_tables,
        )
    records_path = layout.data_dir / "g_distributional_gap_records.jsonl"
    summary_path = layout.data_dir / "g_distributional_gap_summary.json"

    SCRIPT_VERSION = "1"
    PRIMARY = summary_path
    OUTPUTS = [PRIMARY, records_path]
    OUTPUTS += [
        layout.tables_dir / "num_distributional_gap.tex",
        layout.tables_dir / "tab_distributional_gap.tex",
    ]
    fp = fingerprint_of(
        manifest_path,
        scored_dir / "paper_scores_shards" / "metadata" / "manifest.json",
        embed_dir_for_model(model) / "policy.npy",
        scored_dir / "policy_scores.npy",
    ) + SCRIPT_VERSION
    if is_adjusted:
        g_path = register_utils.register_dir(model) / "G.npy"
        fp += f"_adjusted_{register_utils.track_for_model(model)}"
        fp += fingerprint_of(g_path)
    if should_skip(OUTPUTS, fp, args.overwrite, PRIMARY):
        log.info("Skipping %s \u2014 inputs unchanged", PRIMARY)
        return

    seeds = SAMPLE_SEEDS
    cfg = config_payload(model)
    cfg["embeddings"] = args.embeddings
    cfg_hash = compute_config_hash(cfg, scored_dir, embed_dir)
    log.info("Config hash: %s | output: %s", cfg_hash, layout.root)

    canonical = load_canonical(model)
    # In adjusted mode the canonical reference for the rank-correlation must be the
    # register-removed (adjusted) centroid gap, not the raw gap. Under raw embeddings
    # the distribution-aware metrics track the register component; comparing them to
    # the raw gap would conflate topic and register. So we substitute the adjusted
    # semantic_gap as the reference ranking. (Other canonical fields used by GATE 1/2
    # — n_policy_segments_capped, n_papers — are assignment counts and stay raw.)
    if is_adjusted:
        adj_canonical_path = output_dir_for_model(model) / "data" / "adjusted" / CANONICAL_SEMANTIC_JSON
        if adj_canonical_path.exists():
            adj_payload = load_json(adj_canonical_path)
            adj_map = {int(r["sdg"]): r for r in adj_payload["per_sdg"]}
            for s in canonical:
                if s in adj_map and adj_map[s].get("semantic_gap") is not None:
                    canonical[s]["semantic_gap"] = float(adj_map[s]["semantic_gap"])
        else:
            log.warning("Adjusted canonical %s not found; ρ reference falls back to raw gap", adj_canonical_path)
    research_centroids = np.load(scored_dir_for_model(model) / "research_centroids.npy").astype(np.float64)
    policy_state = load_policy_side(model, canonical)

    G = register_utils.load_G(model) if is_adjusted else None
    if is_adjusted:
        log.info("Projecting policy embeddings and research centroids through G...")
        research_centroids = register_utils.project(research_centroids.astype(np.float32), G).astype(np.float64)
        for sdg_idx in range(N_SDG):
            policy_state["clouds"][sdg_idx] = register_utils.project(
                policy_state["clouds"][sdg_idx].astype(np.float32), G
            ).astype(np.float64)

    shards, total_rows = build_research_shards(embed_dir, scored_dir)
    log.info("Research corpus: %d rows across %d shards", total_rows, len(shards))
    sdg_rows = build_sdg_row_index(shards, canonical)

    records = load_existing_records(records_path, cfg_hash)

    # ── Full-corpus records (moments + chamfer in ONE shard-major pass) ──
    missing_full = [s for s in active_sdgs if ("full", s + 1, None) not in records]
    if missing_full:
        log.info("Full-corpus pass for %d SDGs: %s", len(missing_full), [s + 1 for s in missing_full])
        stream_out = stream_moments_and_chamfer(
            shards, sdg_rows, policy_state["clouds"], missing_full, policy_state["dim"],
            G=G,
        )
        check_research_centroid_gate(
            {s: stream_out[s]["mu"] for s in missing_full}, model, missing_full,
            is_adjusted=is_adjusted,
        )
        directions = swd_direction_matrix(policy_state["dim"])
        for s in missing_full:
            t0 = time.time()
            swd_out = compute_swd_full(
                shards, sdg_rows[s], policy_state["clouds"][s], directions, G=G
            )
            record = build_full_record(
                s, cfg_hash, stream_out[s], swd_out, policy_state["clouds"][s], canonical[s + 1],
                research_centroids[s], is_adjusted=is_adjusted,
            )
            record["runtime_seconds"] = round(time.time() - t0, 1)
            append_record(records_path, record)
            records[record_key(record)] = record
            log.info(
                "SDG %2d full: gap=%.4f swd=%.4f w2=%.4f shape_share=%.2f chamfer=%.4f (%.0fs)",
                s + 1,
                record["centroid_gap_full_corpus"],
                record["swd"],
                record["gaussian_w2"],
                record["gaussian_w2_shape_share"],
                record["chamfer_symmetric"],
                record["runtime_seconds"],
            )
    else:
        log.info("All full-corpus records already present — GATES 3/4 previously verified")

    # ── Sampled-family records ──
    for s in active_sdgs:
        for seed in seeds:
            if ("sampled", s + 1, seed) in records:
                continue
            if seed != SAMPLE_SEEDS[0] and sdg_rows[s].size <= RESEARCH_SAMPLE_SIZE:
                continue  # exhaustive: replicate draw would be identical
            cloud, exhaustive = sample_research_cloud(
                manifest_path, embed_dir, sdg_rows[s], s, seed, G=G
            )
            record = build_sampled_record(
                s, seed, cfg_hash, cloud, exhaustive, policy_state["clouds"][s]
            )
            append_record(records_path, record)
            records[record_key(record)] = record
            log.info(
                "SDG %2d seed %d sampled%s: emd=%.4f energy=%.4f mmd2=%.5f c2st=%.4f (%.0fs)",
                s + 1,
                seed,
                " [exhaustive]" if exhaustive else "",
                record["exact_emd"],
                record["energy_distance"],
                record["rbf_mmd_squared"],
                record["c2st_auc"],
                record["runtime_seconds"],
            )
            del cloud

    summary = build_summary(records, canonical, cfg, cfg_hash, active_sdgs, partial)
    summary_path = layout.data_dir / "g_distributional_gap_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_tables(layout.tables_dir, summary)
    log.info("Saved distributional-gap outputs into %s", layout.root)
    record_fingerprint(OUTPUTS, fp, PRIMARY)


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()

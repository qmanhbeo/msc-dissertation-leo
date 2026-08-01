from __future__ import annotations

import csv
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np

from model_utils import DEFAULT_EMBED_MODEL, N_SDG, embed_dir_for_model, scored_dir_for_model
from shard_pipeline_utils import load_json


def get_policy_emb(model: str = DEFAULT_EMBED_MODEL) -> Path:
    return embed_dir_for_model(model) / "policy.npy"


def get_policy_ids(model: str = DEFAULT_EMBED_MODEL) -> Path:
    return scored_dir_for_model(model) / "metadata" / "policy_scores_ids.json"


def get_policy_scores(model: str = DEFAULT_EMBED_MODEL) -> Path:
    return scored_dir_for_model(model) / "policy_scores.npy"


def get_research_centroids(model: str = DEFAULT_EMBED_MODEL) -> Path:
    return scored_dir_for_model(model) / "research_centroids.npy"


def get_research_centroid_meta(model: str = DEFAULT_EMBED_MODEL) -> Path:
    return scored_dir_for_model(model) / "metadata" / "research_centroid_meta.json"


# ---------------------------------------------------------------------------
# MLP classifier paths (parallel to LR paths above)
# ---------------------------------------------------------------------------

def get_mlp_research_centroids(model: str = DEFAULT_EMBED_MODEL) -> Path:
    return scored_dir_for_model(model) / "mlp_scores" / "mlp_research_centroids.npy"


def get_mlp_research_centroid_meta(model: str = DEFAULT_EMBED_MODEL) -> Path:
    return scored_dir_for_model(model) / "mlp_scores" / "mlp_research_centroid_meta.json"


def get_mlp_policy_scores(model: str = DEFAULT_EMBED_MODEL) -> Path:
    return scored_dir_for_model(model) / "mlp_scores" / "mlp_policy_scores.npy"


def build_mlp_centroid_meta(model: str = DEFAULT_EMBED_MODEL) -> list[dict]:
    """Build centroid metadata for MLP from mlp_summary.json.

    MLP centroids don't have a pre-built metadata JSON like LR, so we
    synthesize one from mlp_summary.json (research_coverage) and compute
    mean_cos_to_centroid from the centroids themselves.
    """
    summary_path = scored_dir_for_model(model) / "mlp_scores" / "mlp_summary.json"
    centroids_path = get_mlp_research_centroids(model)
    summary = load_json(summary_path)
    centroids = np.load(centroids_path).astype(np.float32)
    research_coverage = summary["research_coverage"]

    meta = []
    for sdg_idx in range(N_SDG):
        sdg = sdg_idx + 1
        n_papers = int(research_coverage.get(str(sdg), 0))
        centroid = centroids[sdg_idx]
        norm = float(np.linalg.norm(centroid))
        meta.append({
            "sdg": sdg,
            "n_papers_assigned": n_papers,
            "raw_centroid_norm": round(norm, 6),
            "mean_cos_to_centroid": round(norm, 6),
            "zero_flag": norm < 1e-8,
        })
    return meta

SEGMENT_CAP_PRIMARY = 50
SEGMENT_CAP_SENS_LO = 20
SEGMENT_CAP_SENS_HI = 100
SEGMENT_CAP_SENS_NONE = 10_000_000
MIN_CLUSTER_SIZE = 10
RANDOM_SEED = 42

log = logging.getLogger(__name__)


def get_cluster_assignments(scores: np.ndarray) -> np.ndarray:
    """Return hard SDG assignment (0..16) for each item."""
    return scores.argmax(axis=1)


def build_sub_centroid(emb: np.ndarray, idxs: list[int]) -> tuple[np.ndarray | None, float]:
    """
    Compute L2-normalised sub-centroid for a set of row indices into `emb`.

    Returns (unit_centroid, cohesion) or (None, 0.0) if idxs is empty or near-zero norm.
    cohesion = mean cosine sim of member vectors to the unit centroid = raw centroid norm
               (mathematically equivalent for unit input vectors).
    """
    if len(idxs) == 0:
        return None, 0.0

    vecs = emb[idxs]
    raw = vecs.mean(axis=0)
    norm = float(np.linalg.norm(raw))

    if norm < 1e-8:
        return None, 0.0

    unit = (raw / norm).astype(np.float32)
    cohesion = float((vecs @ unit).mean())
    return unit, cohesion


def cap_policy_indices_per_doc(
    policy_idxs: list[int],
    policy_ids: list[dict],
    segment_cap: int,
    rng: np.random.Generator,
) -> list[int]:
    """
    Apply per-document segment cap to a list of policy segment indices.

    Groups indices by source_doc and samples at most `segment_cap` per document.
    """
    doc_to_idxs: dict[str, list[int]] = defaultdict(list)
    for i in policy_idxs:
        doc_to_idxs[policy_ids[i]["source_doc"]].append(i)

    result = []
    for doc_idxs in doc_to_idxs.values():
        if len(doc_idxs) <= segment_cap:
            result.extend(doc_idxs)
        else:
            sampled = rng.choice(doc_idxs, size=segment_cap, replace=False).tolist()
            result.extend(sampled)

    return result


def compute_sdg_semantic_gaps(
    research_centroids: np.ndarray,
    research_counts: np.ndarray,
    research_cohesions: np.ndarray,
    policy_emb: np.ndarray,
    policy_assignments: np.ndarray,
    policy_ids: list[dict],
    segment_cap: int,
    rng: np.random.Generator,
) -> list[dict]:
    """
    Compute semantic gap for each SDG using research and policy sub-centroids.
    """
    results = []

    for sdg_idx in range(N_SDG):
        sdg = sdg_idx + 1
        policy_idxs = [i for i, a in enumerate(policy_assignments) if a == sdg_idx]

        n_papers = int(research_counts[sdg_idx])
        n_segments = len(policy_idxs)

        policy_idxs_capped = cap_policy_indices_per_doc(policy_idxs, policy_ids, segment_cap, rng)
        n_segments_capped = len(policy_idxs_capped)

        policy_docs_raw = {policy_ids[i]["source_doc"] for i in policy_idxs}
        policy_docs_capped = {policy_ids[i]["source_doc"] for i in policy_idxs_capped}

        unreliable_paper = n_papers < MIN_CLUSTER_SIZE
        unreliable_policy = n_segments_capped < MIN_CLUSTER_SIZE
        unreliable = unreliable_paper or unreliable_policy

        if unreliable:
            log.warning(
                "SDG %2d: unreliable gap estimate — n_papers=%d, n_segments_capped=%d "
                "(min=%d required for both)",
                sdg,
                n_papers,
                n_segments_capped,
                MIN_CLUSTER_SIZE,
            )

        res_centroid = research_centroids[sdg_idx]
        res_cohesion = float(research_cohesions[sdg_idx])
        pol_centroid, pol_cohesion = build_sub_centroid(policy_emb, policy_idxs_capped)

        if pol_centroid is None or float(np.linalg.norm(res_centroid)) < 1e-8:
            log.warning("SDG %2d: could not build sub-centroid (empty cluster)", sdg)
            results.append(
                {
                    "sdg": sdg,
                    "n_papers": n_papers,
                    "n_policy_segments": n_segments,
                    "n_policy_segments_capped": n_segments_capped,
                    "n_policy_docs": len(policy_docs_raw),
                    "n_policy_docs_capped": len(policy_docs_capped),
                    "segment_cap": segment_cap,
                    "semantic_similarity": None,
                    "semantic_gap": None,
                    "research_cohesion": None,
                    "policy_cohesion": None,
                    "unreliable": True,
                    "unreliable_reason": "empty_cluster",
                }
            )
            continue

        sim = float(np.dot(res_centroid, pol_centroid))
        gap = 1.0 - sim

        results.append(
            {
                "sdg": sdg,
                "n_papers": n_papers,
                "n_policy_segments": n_segments,
                "n_policy_segments_capped": n_segments_capped,
                "n_policy_docs": len(policy_docs_raw),
                "n_policy_docs_capped": len(policy_docs_capped),
                "segment_cap": segment_cap,
                "semantic_similarity": round(sim, 6),
                "semantic_gap": round(gap, 6),
                "research_cohesion": round(res_cohesion, 6),
                "policy_cohesion": round(pol_cohesion, 6),
                "unreliable": unreliable,
                "unreliable_reason": (
                    "n_papers_too_small"
                    if unreliable_paper
                    else "n_policy_segments_too_small"
                    if unreliable_policy
                    else None
                ),
            }
        )

        level = logging.WARNING if unreliable else logging.INFO
        log.log(
            level,
            "SDG %2d | n_papers=%4d | n_segments=%5d→%4d (cap=%d) | "
            "n_docs=%4d | sim=%.4f | gap=%.4f%s",
            sdg,
            n_papers,
            n_segments,
            n_segments_capped,
            segment_cap,
            len(policy_docs_capped),
            sim,
            gap,
            " [UNRELIABLE]" if unreliable else "",
        )

    return results


def latex_escape(text: str) -> str:
    """Escape LaTeX special characters in text."""
    return (
        text.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("$", r"\$")
        .replace("#", r"\#")
        .replace("_", r"\_")
        .replace("{", r"\{")
        .replace("}", r"\}")
    )


def latex_int(value: int) -> str:
    """Format integer with LaTeX-safe thousands separator ({,})."""
    return f"{value:,}".replace(",", "{,}")


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    """Write rows to a CSV file with given fieldnames."""
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_source_family_map(model: str = DEFAULT_EMBED_MODEL) -> dict[str, str]:
    """Build mapping from source_doc to source_family from policy_ids.json."""
    ids_path = embed_dir_for_model(model) / "metadata" / "policy_ids.json"
    policy_ids = load_json(ids_path)
    source_family: dict[str, str] = {}
    for row in policy_ids:
        source_doc = str(row["source_doc"])
        family = row.get("source_family")
        if family is None:
            continue
        existing = source_family.get(source_doc)
        if existing is not None and existing != family:
            raise RuntimeError(
                f"source_doc '{source_doc}' appears in multiple families: {existing} vs {family}"
            )
        source_family[source_doc] = family
    if not source_family:
        raise RuntimeError("No source-family assignments found in policy_ids.json.")
    return source_family

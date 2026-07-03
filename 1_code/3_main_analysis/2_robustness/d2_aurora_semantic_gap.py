"""
Aurora-robustness semantic gap analysis.

Replicates the canonical research-policy semantic gap computation (as in
1_semantic_gap.py) but uses Aurora-derived centroids instead of the default
OSDG-based centroids. This provides an independent robustness check against
the "OSDG centroids are policy-calibrated" vulnerability.

Method (identical to canonical):
  1. Score all research paper embeddings against Aurora centroids → Aurora-based
     research sub-centroids per SDG
  2. Score all policy segment embeddings against Aurora centroids → Aurora-based
     policy sub-centroids per SDG (with per-document segment cap)
  3. Semantic gap_j = 1 - cosine(research_subcentroid_j, policy_subcentroid_j)
  4. Compare SDG gap rankings (Aurora vs canonical) via Spearman correlation

Outputs:
  4_outputs/appendix/d1_aurora_centroids/data/aurora_gap_distances.json
  4_outputs/appendix/d1_aurora_centroids/tables/
"""

import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RESEARCH_SHARDS_DIR = Path("2_data/2_embedded/research_shards")
RESEARCH_META_DIR = RESEARCH_SHARDS_DIR / "metadata"

POLICY_EMB = Path("2_data/2_embedded/policy.npy")
POLICY_IDS = Path("2_data/3_scored/metadata/policy_scores_ids.json")

AURORA_CENTROIDS = Path("2_data/3_scored/aurora_centroids.npy")
CANONICAL_GAP = Path("4_outputs/main/data/4_3_semantic_gap_distances.json")

APPENDIX_DATA = Path("4_outputs/appendix/d1_aurora_centroids/data")
APPENDIX_TABLES = Path("4_outputs/appendix/d1_aurora_centroids/tables")

SEGMENT_CAP = 50          # same default as canonical
MIN_CLUSTER_SIZE = 10     # same as canonical


def load_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def cap_policy_segments_per_doc(assignments: dict[int, list[int]],
                                pol_ids: list[dict]) -> dict[int, list[int]]:
    """Cap at SEGMENT_CAP segments per source_doc (same as canonical)."""
    capped: dict[int, list[int]] = {}
    for sdg, indices in assignments.items():
        doc_groups: dict[str, list[int]] = defaultdict(list)
        for idx in indices:
            doc = pol_ids[idx].get("source_doc", "unknown")
            doc_groups[doc].append(idx)
        selected: list[int] = []
        for doc, segs in doc_groups.items():
            rng = np.random.default_rng(42)  # reproducible
            selected.extend(segs if len(segs) <= SEGMENT_CAP
                            else rng.choice(segs, size=SEGMENT_CAP, replace=False).tolist())
        capped[sdg] = selected
    return capped


def build_sub_centroid(emb: np.ndarray, indices: list[int]) -> tuple[np.ndarray, float]:
    """L2-normalised mean (same as canonical)."""
    if len(indices) == 0:
        return np.zeros(emb.shape[1], dtype=np.float32), 0.0
    raw = emb[indices].mean(axis=0)
    norm = float(np.linalg.norm(raw))
    if norm < 1e-8:
        return np.zeros(emb.shape[1], dtype=np.float32), 0.0
    unit = (raw / norm).astype(np.float32)
    mean_cos = float((emb[indices] @ unit).mean())
    return unit, mean_cos


def main():
    APPENDIX_DATA.mkdir(parents=True, exist_ok=True)
    APPENDIX_TABLES.mkdir(parents=True, exist_ok=True)

    # ---- Load Aurora centroids ----
    log.info("Loading Aurora centroids: %s", AURORA_CENTROIDS)
    aurora_centroids = np.load(AURORA_CENTROIDS).astype(np.float32)
    log.info("  Shape: %s", aurora_centroids.shape)

    # ---- Step 1: Score research papers against Aurora centroids (shard-by-shard) ----
    manifest = load_json(RESEARCH_META_DIR / "manifest.json")
    shards = manifest["shards"]

    sdg_sums: dict[int, np.ndarray] = {s: np.zeros(384, dtype=np.float64) for s in range(1, 18)}
    sdg_counts: dict[int, int] = {s: 0 for s in range(1, 18)}

    log.info("Scoring research papers against Aurora centroids (%d shards)...", len(shards))
    for shard_info in shards:
        sid = shard_info["shard_id"]
        emb_path = Path(shard_info["embedding_path"])
        log.info("  Shard %2d/%d: %s", sid, len(shards), emb_path)

        emb = np.load(emb_path.resolve()).astype(np.float32)
        scores = emb @ aurora_centroids.T   # (N, 17)
        assignments = scores.argmax(axis=1)  # 0-indexed SDG

        for i, sdg in enumerate(assignments + 1):
            sdg_sums[sdg] += emb[i].astype(np.float64)
            sdg_counts[sdg] += 1

    # Build research sub-centroids
    research_centroids = np.zeros((17, 384), dtype=np.float32)
    research_meta = []
    log.info("Building Aurora-based research sub-centroids...")
    for sdg in range(1, 18):
        n = sdg_counts[sdg]
        log.info("  SDG %2d: %d papers assigned", sdg, n)
        if n == 0:
            research_meta.append({"sdg": sdg, "n_papers": 0, "mean_cos": 0, "unreliable": True})
            continue
        raw = sdg_sums[sdg].astype(np.float64) / n
        norm = float(np.linalg.norm(raw))
        if norm < 1e-8:
            research_meta.append({"sdg": sdg, "n_papers": n, "mean_cos": 0, "unreliable": True})
            continue
        unit = (raw / norm).astype(np.float32)
        research_centroids[sdg - 1] = unit
        mean_cos = float((research_centroids[sdg - 1: sdg] @ unit).item())  # identity, but for consistency
        unreliable = n < MIN_CLUSTER_SIZE
        research_meta.append({"sdg": sdg, "n_papers": n, "mean_cos": mean_cos, "unreliable": unreliable})
        log.info("         centroid built (n=%d, unreliable=%s)", n, unreliable)

    # ---- Step 2: Score policy segments against Aurora centroids ----
    log.info("Loading policy embeddings: %s", POLICY_EMB)
    policy_emb = np.load(POLICY_EMB).astype(np.float32)
    policy_ids = load_json(POLICY_IDS)
    log.info("  %d policy segments", policy_emb.shape[0])

    policy_scores = policy_emb @ aurora_centroids.T  # (N_pol, 17)
    pol_assignments_raw: dict[int, list[int]] = defaultdict(list)
    for i, sdg in enumerate(policy_scores.argmax(axis=1) + 1):
        pol_assignments_raw[sdg].append(i)

    log.info("Policy segments per SDG (raw):")
    for sdg in sorted(pol_assignments_raw):
        log.info("  SDG %2d: %d segments", sdg, len(pol_assignments_raw[sdg]))

    # Cap per document (same as canonical)
    pol_assignments_capped = cap_policy_segments_per_doc(pol_assignments_raw, policy_ids)

    log.info("Policy segments per SDG (capped at %d/doc):", SEGMENT_CAP)
    for sdg in sorted(pol_assignments_capped):
        log.info("  SDG %2d: %d segments", sdg, len(pol_assignments_capped[sdg]))

    # Build policy sub-centroids
    policy_centroids = np.zeros((17, 384), dtype=np.float32)
    policy_meta = []
    log.info("Building Aurora-based policy sub-centroids...")
    for sdg in range(1, 18):
        idxs = pol_assignments_capped.get(sdg, [])
        n = len(idxs)
        if n == 0:
            policy_meta.append({"sdg": sdg, "n_policy_segments": 0, "unreliable": True})
            continue
        raw = policy_emb[idxs].mean(axis=0).astype(np.float64)
        norm = float(np.linalg.norm(raw))
        if norm < 1e-8:
            policy_meta.append({"sdg": sdg, "n_policy_segments": n, "unreliable": True})
            continue
        unit = (raw / norm).astype(np.float32)
        policy_centroids[sdg - 1] = unit
        unreliable = n < MIN_CLUSTER_SIZE
        policy_meta.append({"sdg": sdg, "n_policy_segments": n, "unreliable": unreliable})

    # ---- Step 3: Compute semantic gap ----
    results = {"method": "aurora_centroid_gap", "centroid_source": "Aurora (Vanderfeesten et al., 2020)"}
    per_sdg = []

    for sdg in range(1, 18):
        res_cent = research_centroids[sdg - 1]
        pol_cent = policy_centroids[sdg - 1]

        res_n = research_meta[sdg - 1]["n_papers"]
        pol_n = policy_meta[sdg - 1]["n_policy_segments"]

        res_unreliable = research_meta[sdg - 1].get("unreliable", False)
        pol_unreliable = policy_meta[sdg - 1].get("unreliable", False)
        unreliable = res_unreliable or pol_unreliable

        res_norm = float(np.linalg.norm(res_cent))
        pol_norm = float(np.linalg.norm(pol_cent))

        if res_norm < 1e-8 or pol_norm < 1e-8:
            sim = 0.0
            gap = 1.0
            unreliable = True
            reason = "zero_centroid"
        else:
            sim = float(np.dot(res_cent, pol_cent))
            gap = 1.0 - sim
            reason = "ok" if not unreliable else "small_cluster"

        entry = {
            "sdg": sdg,
            "semantic_gap": round(gap, 6),
            "semantic_similarity": round(sim, 6),
            "n_papers": res_n,
            "n_policy_segments": pol_n,
            "unreliable": unreliable or (res_n < MIN_CLUSTER_SIZE or pol_n < MIN_CLUSTER_SIZE),
            "unreliable_reason": reason,
            "research_cohesion": research_meta[sdg - 1].get("mean_cos", None),
        }
        per_sdg.append(entry)

        log.info("SDG %2d | gap=%.4f | sim=%.4f | n_papers=%d | n_pol=%d%s",
                 sdg, gap, sim, res_n, pol_n, " [UNRELIABLE]" if unreliable else "")

    # Compute ranking comparison with canonical
    if CANONICAL_GAP.exists():
        canonical = load_json(CANONICAL_GAP)
        canon_gaps = {}
        for entry in canonical.get("per_sdg", []):
            sdg = entry["sdg"]
            canon_gaps[sdg] = entry["semantic_gap"]

        # Only compare SDGs that are reliable in both methods
        aurora_gaps = {e["sdg"]: e["semantic_gap"] for e in per_sdg if not e["unreliable"]}
        common_sdgs = sorted(set(canon_gaps.keys()) & set(aurora_gaps.keys()))

        aurora_ranked = sorted(common_sdgs, key=lambda s: aurora_gaps[s], reverse=True)
        canon_ranked = sorted(common_sdgs, key=lambda s: canon_gaps[s], reverse=True)

        aurora_ranks = {s: i for i, s in enumerate(aurora_ranked)}
        canon_ranks = {s: i for i, s in enumerate(canon_ranked)}

        rank_data = []
        for sdg in common_sdgs:
            rank_data.append({
                "sdg": sdg,
                "aurora_gap": aurora_gaps[sdg],
                "canonical_gap": canon_gaps[sdg],
                "aurora_rank": aurora_ranks[sdg],
                "canonical_rank": canon_ranks[sdg],
                "rank_diff": aurora_ranks[sdg] - canon_ranks[sdg],
            })

        aurora_vals = [aurora_gaps[s] for s in common_sdgs]
        canon_vals = [canon_gaps[s] for s in common_sdgs]
        corr, p_value = spearmanr(aurora_vals, canon_vals)

        log.info("=" * 60)
        log.info("RANKING COMPARISON: Aurora vs Canonical centroids")
        log.info("Common SDGs (reliable in both): %s", common_sdgs)
        log.info("Spearman ρ = %.4f (p=%.6f)", corr, p_value)
        log.info("SDG rankings:")
        for r in sorted(rank_data, key=lambda x: x["aurora_rank"]):
            diff_str = f"+{r['rank_diff']}" if r['rank_diff'] > 0 else str(r['rank_diff'])
            log.info("  SDG %2d | Aurora rank=%2d | Canon rank=%2d | Δ=%s | Aurora gap=%.4f | Canon gap=%.4f",
                     r['sdg'], r['aurora_rank'], r['canonical_rank'], diff_str,
                     r['aurora_gap'], r['canonical_gap'])

        ranking_comparison = {
            "spearman_rho": round(corr, 6),
            "spearman_p_value": round(p_value, 6),
            "n_common_sdgs": len(common_sdgs),
            "common_sdgs": common_sdgs,
            "per_sdg_rankings": rank_data,
        }
    else:
        log.warning("Canonical gap not found: %s", CANONICAL_GAP)
        ranking_comparison = None

    # ---- Save outputs ----
    output = {
        "method": results["method"],
        "centroid_source": results["centroid_source"],
        "segment_cap": SEGMENT_CAP,
        "min_cluster_size": MIN_CLUSTER_SIZE,
        "per_sdg": per_sdg,
        "reliable_sdgs": [e["sdg"] for e in per_sdg if not e["unreliable"]],
        "unreliable_sdgs": [e["sdg"] for e in per_sdg if e["unreliable"]],
    }
    if ranking_comparison:
        output["ranking_comparison"] = ranking_comparison

    out_path = APPENDIX_DATA / "aurora_gap_distances.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    log.info("Saved: %s", out_path)

    # ---- Write LaTeX macros ----
    gap_vals = [e["semantic_gap"] for e in per_sdg]
    mean_gap = float(np.mean(gap_vals))
    median_gap = float(np.median(gap_vals))

    sdg_num_words = {
        1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five",
        6: "Six", 7: "Seven", 8: "Eight", 9: "Nine", 10: "Ten",
        11: "Eleven", 12: "Twelve", 13: "Thirteen", 14: "Fourteen",
        15: "Fifteen", 16: "Sixteen", 17: "Seventeen",
    }

    lines = [
        "% Auto-generated by d2_aurora_semantic_gap.py",
        r"\newcommand{\AuroraMeanGap}{" + f"{mean_gap:.4f}" + "}",
        r"\newcommand{\AuroraMedianGap}{" + f"{median_gap:.4f}" + "}",
    ]
    if ranking_comparison:
        lines.append(r"\newcommand{\AuroraCanonicalSpearmanRho}{" + f"{ranking_comparison['spearman_rho']:.4f}" + "}")
        lines.append(r"\newcommand{\AuroraCanonicalSpearmanP}{" + f"{ranking_comparison['spearman_p_value']:.4f}" + "}")

    for e in per_sdg:
        word = sdg_num_words[e["sdg"]]
        lines.append(r"\newcommand{\AuroraGapSdg" + word + "}{" + f"{e['semantic_gap']:.4f}" + "}")

    tex_path = APPENDIX_TABLES / "num_aurora_gap.tex"
    tex_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("Saved: %s", tex_path)

    log.info("=" * 60)
    log.info("DONE — Aurora gap re-analysis complete")
    if ranking_comparison:
        log.info("Spearman ρ(Aurora, Canonical) = %.4f", ranking_comparison['spearman_rho'])


if __name__ == "__main__":
    main()

"""
Compute intra-SDG semantic gap between research and policy corpora.

The semantic gap measures whether research and policy texts assigned to the *same* SDG are
semantically similar to each other. A large semantic gap on SDG j means that even though both
corpora discuss SDG j, they do so in different semantic registers — the research and policy
communities are "talking past each other" within that SDG.

Method:
  For each SDG j:
    1. Research cluster j  = paper embeddings assigned to SDG j (all papers; no cap needed since
                             papers are independently authored, not dominated by one document).
    2. Policy cluster j    = policy chunk embeddings assigned to SDG j, with per-document chunk cap.
    3. Research sub-centroid j = L2-normalised mean of research cluster j embeddings.
    4. Policy sub-centroid j   = L2-normalised mean of policy cluster j embeddings (chunk-capped).
    5. semantic_similarity[j]  = cosine_sim(research_sub_centroid_j, policy_sub_centroid_j)
                               = dot product (both are unit vectors after normalisation)
    6. semantic_gap[j]         = 1 - semantic_similarity[j]

  Interpretation:
    semantic_gap = 0.0 → perfect semantic overlap; both corpora discuss SDG j identically
    semantic_gap = 1.0 → orthogonal; the corpora discuss SDG j in completely unrelated ways
    Typical range: 0.1–0.8 (for real-world policy/research text in SBERT space)

Per-document chunk cap (Assumption A-CHUNKCAT):
  Without capping, SDSN 2024 (~3,179 chunks) and SDGi VNR/VLR reports (31,941 total chunks)
  would dominate the policy cluster centroids for whichever SDG they are assigned to.
  We cap at CHUNK_CAP chunks per source_doc per SDG. Random sampling is seeded for
  reproducibility.

  CHUNK_CAP = 50 was chosen as a round number that:
    - Prevents any single document from contributing more than 50 chunks to a policy cluster
    - Still allows documents to contribute substantively (a 50-chunk sample = ~7,500 words)
    - Is conservative relative to median document size (~14 chunks/document in the corpus)
  This is Assumption A-CHUNKCAT. Results with CHUNK_CAP = 20 and CHUNK_CAP = 100 are
  included as sensitivity checks.

Minimum cluster size:
  SDGs with fewer than MIN_CLUSTER_SIZE items in the research OR policy cluster are flagged
  as unreliable. The semantic gap estimate for these SDGs should not be reported as a finding.
  MIN_CLUSTER_SIZE = 10 is a conservative lower bound; sub-centroids built from < 10 items
  are dominated by noise.

  Note on coverage gap interaction:
  SDGs with very small research clusters (SDG 1: 43 papers, SDG 10: 20 papers) are precisely
  the SDGs that appear "neglected" in coverage gap analysis. Their semantic gap estimates may
  be noisy. This is acknowledged in Assumption A-SPARSE.

Inputs:
  data/3_scored/research_centroids.npy       (17, 384)    float32
  data/3_scored/metadata/research_centroid_meta.json  list of 17 SDG centroid metadata rows
  data/3_scored/policy_scores.npy            (47005, 17)  float32
  data/3_scored/metadata/policy_scores_ids.json       list of {id, source_doc}
  data/2_embedded/policy.npy        (47005, 384) float32, L2-normalised

Outputs:
  outputs/semantic_gap.json              primary: semantic gap per SDG (CHUNK_CAP=50)
  outputs/semantic_gap_sensitivity.json  sensitivity analysis at CHUNK_CAP=20 and CHUNK_CAP=100
  outputs/tables/*.tex                   generated LaTeX macros/tables

Run from project root (after coverage_gap.py):
    python code/3_main_analysis/semantic_gap.py
"""

import json
import logging
import numpy as np
import argparse
import sys
from collections import defaultdict
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
from shared_utils import ensure_canonical_outputs

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SCORED_DIR     = Path("data/3_scored")
EMBEDDINGS_DIR = Path("data/2_embedded")
DEFAULT_OUTPUT_ROOT = Path("outputs")

POLICY_SCORES = SCORED_DIR / "policy_scores.npy"
POLICY_IDS    = SCORED_DIR / "metadata" / "policy_scores_ids.json"
POLICY_EMB    = EMBEDDINGS_DIR / "policy.npy"
RESEARCH_CENTROIDS = SCORED_DIR / "research_centroids.npy"
RESEARCH_CENTROID_META = SCORED_DIR / "metadata" / "research_centroid_meta.json"

N_SDG = 17

# Per-document chunk cap for policy clusters. See docstring.
CHUNK_CAP_PRIMARY  = 50
CHUNK_CAP_SENS_LO  = 20
CHUNK_CAP_SENS_HI  = 100

# Minimum items in research OR policy cluster for a reliable gap estimate.
# SDGs below this threshold in either corpus are flagged as unreliable.
MIN_CLUSTER_SIZE = 10

# Random seed for chunk cap sampling (ensures reproducibility).
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def load_json(path: Path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


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

    vecs = emb[idxs]   # (n, D) — L2-normalised inputs
    raw = vecs.mean(axis=0)
    norm = float(np.linalg.norm(raw))

    if norm < 1e-8:
        return None, 0.0

    unit = (raw / norm).astype(np.float32)
    cohesion = float((vecs @ unit).mean())   # = norm for unit inputs
    return unit, cohesion


def cap_policy_indices_per_doc(
    policy_idxs: list[int],
    policy_ids: list[dict],
    chunk_cap: int,
    rng: np.random.Generator,
) -> list[int]:
    """
    Apply per-document chunk cap to a list of policy chunk indices.

    Groups indices by source_doc and samples at most `chunk_cap` per document.

    Args:
        policy_idxs: row indices into policy_emb for chunks assigned to this SDG.
        policy_ids:  full policy IDs list (length = total policy corpus size).
        chunk_cap:   maximum chunks per source_doc.
        rng:         seeded numpy random generator for reproducibility.

    Returns:
        Filtered list of indices with at most `chunk_cap` per source_doc.

    ASSUMPTION (A-CHUNKCAT-SAMPLE): Random sampling without replacement is used when a
    document exceeds the cap. All chunks in a document are equally informative (no ordering
    preference). This may undersample the most substantive body text if a document's
    introductory chunks (lower indices) are less informative than its body. The alternative
    — max-marginal relevance sampling — would be more principled but adds complexity. For a
    sub-centroid computation, random sampling is sufficient.
    """
    doc_to_idxs: dict[str, list[int]] = defaultdict(list)
    for i in policy_idxs:
        doc_to_idxs[policy_ids[i]["source_doc"]].append(i)

    result = []
    for doc_idxs in doc_to_idxs.values():
        if len(doc_idxs) <= chunk_cap:
            result.extend(doc_idxs)
        else:
            # Sample without replacement.
            sampled = rng.choice(doc_idxs, size=chunk_cap, replace=False).tolist()
            result.extend(sampled)

    return result


def compute_sdg_semantic_gaps(
    research_centroids: np.ndarray,
    research_counts: np.ndarray,
    research_cohesions: np.ndarray,
    policy_emb: np.ndarray,
    policy_assignments: np.ndarray,
    policy_ids: list[dict],
    chunk_cap: int,
    rng: np.random.Generator,
) -> list[dict]:
    """
    Compute semantic gap for each SDG (centroid-to-centroid method).

    Returns list of 17 dicts with per-SDG gap metrics.
    """
    results = []

    for sdg_idx in range(N_SDG):
        sdg = sdg_idx + 1

        # Gather cluster indices.
        policy_idxs = [i for i, a in enumerate(policy_assignments) if a == sdg_idx]

        n_papers  = int(research_counts[sdg_idx])
        n_chunks  = len(policy_idxs)

        # Apply per-document chunk cap to policy side.
        policy_idxs_capped = cap_policy_indices_per_doc(policy_idxs, policy_ids, chunk_cap, rng)
        n_chunks_capped = len(policy_idxs_capped)

        # Identify unique source_docs in policy cluster (before and after capping).
        policy_docs_raw    = {policy_ids[i]["source_doc"] for i in policy_idxs}
        policy_docs_capped = {policy_ids[i]["source_doc"] for i in policy_idxs_capped}

        # Flag unreliable estimates.
        unreliable_paper  = n_papers < MIN_CLUSTER_SIZE
        unreliable_policy = n_chunks_capped < MIN_CLUSTER_SIZE
        unreliable = unreliable_paper or unreliable_policy

        if unreliable:
            log.warning(
                "SDG %2d: unreliable gap estimate — n_papers=%d, n_chunks_capped=%d "
                "(min=%d required for both)",
                sdg, n_papers, n_chunks_capped, MIN_CLUSTER_SIZE
            )

        # Build policy sub-centroid and read research centroid pre-computed at scoring stage.
        res_centroid = research_centroids[sdg_idx]
        res_cohesion = float(research_cohesions[sdg_idx])
        pol_centroid, pol_cohesion = build_sub_centroid(policy_emb, policy_idxs_capped)

        if pol_centroid is None or float(np.linalg.norm(res_centroid)) < 1e-8:
            # Cannot compute gap — one or both clusters are empty or near-zero.
            log.warning("SDG %2d: could not build sub-centroid (empty cluster)", sdg)
            results.append({
                "sdg": sdg,
                "n_papers": n_papers,
                "n_policy_chunks": n_chunks,
                "n_policy_chunks_capped": n_chunks_capped,
                "n_policy_docs": len(policy_docs_raw),
                "n_policy_docs_capped": len(policy_docs_capped),
                "chunk_cap": chunk_cap,
                "semantic_similarity": None,
                "semantic_gap": None,
                "research_cohesion": None,
                "policy_cohesion": None,
                "unreliable": True,
                "unreliable_reason": "empty_cluster",
            })
            continue

        # Semantic similarity = cosine sim between research and policy sub-centroids.
        # Both are unit vectors → dot product = cosine similarity.
        sim = float(np.dot(res_centroid, pol_centroid))
        gap = 1.0 - sim

        results.append({
            "sdg": sdg,
            "n_papers": n_papers,
            "n_policy_chunks": n_chunks,
            "n_policy_chunks_capped": n_chunks_capped,
            "n_policy_docs": len(policy_docs_raw),
            "n_policy_docs_capped": len(policy_docs_capped),
            "chunk_cap": chunk_cap,
            "semantic_similarity": round(sim, 6),
            "semantic_gap": round(gap, 6),
            "research_cohesion": round(res_cohesion, 6),
            "policy_cohesion": round(pol_cohesion, 6),
            "unreliable": unreliable,
            "unreliable_reason": (
                "n_papers_too_small" if unreliable_paper
                else "n_policy_chunks_too_small" if unreliable_policy
                else None
            ),
        })

        level = logging.WARNING if unreliable else logging.INFO
        log.log(level,
            "SDG %2d | n_papers=%4d | n_chunks=%5d→%4d (cap=%d) | "
            "n_docs=%4d | sim=%.4f | gap=%.4f%s",
            sdg, n_papers, n_chunks, n_chunks_capped, chunk_cap,
            len(policy_docs_capped), sim, gap,
            " [UNRELIABLE]" if unreliable else ""
        )

    return results


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute semantic gap outputs into the canonical output folder.")
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_ROOT))
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    layout = ensure_canonical_outputs(Path(args.output_dir))
    out_sem_gap = layout.root / "semantic_gap.json"
    out_sem_sens = layout.root / "semantic_gap_sensitivity.json"
    tables_dir = layout.tables_dir
    log.info("Canonical output dir: %s", layout.root)

    # ---- Load research centroids/meta ----
    log.info("Loading research centroids: %s", RESEARCH_CENTROIDS)
    research_centroids = np.load(RESEARCH_CENTROIDS).astype(np.float32)
    research_meta = load_json(RESEARCH_CENTROID_META)
    if research_centroids.shape[0] != N_SDG:
        raise RuntimeError(f"Expected research centroids shape ({N_SDG}, d), got {research_centroids.shape}")
    if len(research_meta) != N_SDG:
        raise RuntimeError(f"Expected {N_SDG} research centroid meta rows, got {len(research_meta)}")
    research_counts = np.array([int(r["n_papers_assigned"]) for r in research_meta], dtype=np.int64)
    research_cohesions = np.array([float(r["mean_cos_to_centroid"]) for r in research_meta], dtype=np.float32)

    log.info("Loading policy embeddings: %s", POLICY_EMB)
    policy_emb = np.load(POLICY_EMB)   # (47005, 384)
    policy_ids = load_json(POLICY_IDS)

    # ---- Load score matrices for cluster assignments ----
    log.info("Loading score matrices...")
    policy_scores = np.load(POLICY_SCORES)

    # Hard assignment (0-indexed SDG index).
    policy_assignments = get_cluster_assignments(policy_scores)

    log.info("Paper cluster sizes by SDG:")
    for sdg_idx in range(N_SDG):
        n = int(research_counts[sdg_idx])
        log.info("  SDG %2d: %d papers", sdg_idx + 1, n)

    log.info("Policy cluster sizes by SDG (raw chunks):")
    for sdg_idx in range(N_SDG):
        n = int((policy_assignments == sdg_idx).sum())
        log.info("  SDG %2d: %d chunks", sdg_idx + 1, n)

    # ---- Primary analysis (CHUNK_CAP = 50) ----
    log.info("")
    log.info("=" * 60)
    log.info("PRIMARY SEMANTIC GAP (chunk cap = %d)", CHUNK_CAP_PRIMARY)
    log.info("=" * 60)
    rng_primary = np.random.default_rng(RANDOM_SEED)
    primary_results = compute_sdg_semantic_gaps(
        research_centroids, research_counts, research_cohesions,
        policy_emb, policy_assignments,
        policy_ids, CHUNK_CAP_PRIMARY, rng_primary
    )

    # Summary: sort by semantic gap (largest first).
    reliable = [r for r in primary_results if not r["unreliable"] and r["semantic_gap"] is not None]
    log.info("")
    log.info("Sorted by semantic gap (reliable SDGs only, cap=%d):", CHUNK_CAP_PRIMARY)
    for r in sorted(reliable, key=lambda x: x["semantic_gap"], reverse=True):
        log.info("  SDG %2d | gap=%.4f | sim=%.4f | n_papers=%4d | n_policy_docs=%4d",
                 r["sdg"], r["semantic_gap"], r["semantic_similarity"],
                 r["n_papers"], r["n_policy_docs_capped"])

    # ---- Sensitivity analyses ----
    log.info("")
    log.info("=" * 60)
    log.info("SENSITIVITY: chunk cap = %d", CHUNK_CAP_SENS_LO)
    log.info("=" * 60)
    rng_lo = np.random.default_rng(RANDOM_SEED)
    sens_lo = compute_sdg_semantic_gaps(
        research_centroids, research_counts, research_cohesions,
        policy_emb, policy_assignments,
        policy_ids, CHUNK_CAP_SENS_LO, rng_lo
    )

    log.info("")
    log.info("=" * 60)
    log.info("SENSITIVITY: chunk cap = %d", CHUNK_CAP_SENS_HI)
    log.info("=" * 60)
    rng_hi = np.random.default_rng(RANDOM_SEED)
    sens_hi = compute_sdg_semantic_gaps(
        research_centroids, research_counts, research_cohesions,
        policy_emb, policy_assignments,
        policy_ids, CHUNK_CAP_SENS_HI, rng_hi
    )

    # Check sensitivity: do rankings change substantially across caps?
    # A finding is robust if its gap rank is stable across all three caps.
    log.info("")
    log.info("SENSITIVITY CHECK — gap rank stability across chunk caps:")
    log.info("  %-6s  %-12s  %-12s  %-12s", "SDG", "cap20", "cap50", "cap100")
    log.info("  " + "-" * 50)
    for i in range(N_SDG):
        sdg = i + 1
        g20   = sens_lo[i]["semantic_gap"]
        g50   = primary_results[i]["semantic_gap"]
        g100  = sens_hi[i]["semantic_gap"]
        if g20 is None or g50 is None or g100 is None:
            log.info("  SDG %2d  %-12s  %-12s  %-12s", sdg, "N/A", "N/A", "N/A")
        else:
            log.info("  SDG %2d  %.4f       %.4f       %.4f", sdg, g20, g50, g100)

    # ---- Build output JSON ----
    primary_out = {
        "method": "centroid_to_centroid",
        "chunk_cap": CHUNK_CAP_PRIMARY,
        "min_cluster_size": MIN_CLUSTER_SIZE,
        "random_seed": RANDOM_SEED,
        "note": (
            "semantic_gap[j] = 1 - cosine_sim(research_sub_centroid_j, policy_sub_centroid_j). "
            "Both sub-centroids are L2-normalised means of cluster embeddings. "
            "Policy clusters are chunk-capped per source_doc to avoid SDSN/SDGi dominance (A19). "
            "SDGs flagged unreliable have fewer than MIN_CLUSTER_SIZE items in research or policy."
        ),
        "per_sdg": primary_results,
        "reliable_sdgs": [r["sdg"] for r in primary_results if not r["unreliable"]],
        "unreliable_sdgs": [r["sdg"] for r in primary_results if r["unreliable"]],
    }

    sensitivity_out = {
        "method": "centroid_to_centroid",
        "random_seed": RANDOM_SEED,
        "note": (
            "Sensitivity analysis: same computation as semantic_gap.json but with different "
            "per-document chunk caps (20 and 100). Use to verify finding robustness. "
            "Rankings should be broadly stable if findings are robust."
        ),
        f"cap_{CHUNK_CAP_SENS_LO}": sens_lo,
        f"cap_{CHUNK_CAP_SENS_HI}": sens_hi,
    }

    with out_sem_gap.open("w", encoding="utf-8") as f:
        json.dump(primary_out, f, indent=2)
    log.info("Saved: %s", out_sem_gap)

    with out_sem_sens.open("w", encoding="utf-8") as f:
        json.dump(sensitivity_out, f, indent=2)
    log.info("Saved: %s", out_sem_sens)

    log.info("")
    log.info("Next step: python code/3_main_analysis/coverage_semantic_interaction.py")

    # ---- Write LaTeX generated outputs ----
    _sdg_names_17 = {
        1: "No Poverty", 2: "Zero Hunger", 3: "Good Health and Well-Being",
        4: "Quality Education", 5: "Gender Equality",
        6: "Clean Water and Sanitation", 7: "Affordable and Clean Energy",
        8: "Decent Work and Economic Growth",
        9: "Industry, Innovation and Infrastructure",
        10: "Reduced Inequalities", 11: "Sustainable Cities and Communities",
        12: "Responsible Consumption and Production", 13: "Climate Action",
        14: "Life Below Water", 15: "Life on Land",
        16: "Peace, Justice and Strong Institutions",
        17: "Partnerships for the Goals",
    }
    _sdg_num_words = {
        1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five",
        6: "Six", 7: "Seven", 8: "Eight", 9: "Nine", 10: "Ten",
        11: "Eleven", 12: "Twelve", 13: "Thirteen", 14: "Fourteen",
        15: "Fifteen", 16: "Sixteen", 17: "Seventeen",
    }

    gen_dir = tables_dir

    # Extract per-SDG values from primary_results (SDG order 1–17)
    per_sdg_map = {r["sdg"]: r for r in primary_results}
    gaps = [per_sdg_map[s]["semantic_gap"] for s in range(1, 18)]
    valid_gaps = [g for g in gaps if g is not None]
    mean_gap = float(np.mean(valid_gaps))
    sorted_gaps = sorted(valid_gaps)
    n = len(sorted_gaps)
    median_gap = float(
        sorted_gaps[n // 2] if n % 2 == 1
        else (sorted_gaps[n // 2 - 1] + sorted_gaps[n // 2]) / 2
    )

    def _ltx_num(v: int) -> str:
        return f"{v:,}".replace(",", "{,}")

    # num_semantic.tex — macro definitions
    num_lines = [
        "% Auto-generated by code/semantic_gap.py — do not edit manually",
        rf"\newcommand{{\MeanSemanticGap}}{{{mean_gap:.3f}}}",
        rf"\newcommand{{\MedianSemanticGap}}{{{median_gap:.3f}}}",
        rf"\newcommand{{\SemanticGapRange}}{{{max(valid_gaps) - min(valid_gaps):.3f}}}",
    ]
    for sdg_num, word in _sdg_num_words.items():
        row = per_sdg_map[sdg_num]
        g = row["semantic_gap"]
        if g is not None:
            num_lines.append(rf"\newcommand{{\SemanticGapSdg{word}}}{{{g:.3f}}}")
        num_lines.append(
            rf"\newcommand{{\NPapersSdg{word}}}{{{_ltx_num(int(row['n_papers']))}}}"
        )
        num_lines.append(
            rf"\newcommand{{\NPolicyDocsSdg{word}}}{{{_ltx_num(int(row['n_policy_docs_capped']))}}}"
        )
    (gen_dir / "num_semantic.tex").write_text("\n".join(num_lines) + "\n", encoding="utf-8")
    log.info("Saved: %s", gen_dir / "num_semantic.tex")

    # tab_semgap.tex — full tabular block
    sorted_results = sorted(
        [r for r in primary_results if r["semantic_gap"] is not None],
        key=lambda x: x["semantic_gap"],
        reverse=True,
    )
    tab_lines = [
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"SDG & Description & Sem. Gap & n$_{\text{res}}$ & n$_{\text{pol docs}}$ \\",
        r"\midrule",
    ]
    for r in sorted_results:
        sdg = r["sdg"]
        name = _sdg_names_17[sdg]
        g = r["semantic_gap"]
        n_res = r["n_papers"]
        n_pol = r["n_policy_docs_capped"]
        tab_lines.append(
            rf"SDG {sdg:2d} & {name} & {g:.3f} & {n_res:,} & {n_pol:,} \\"
        )
    tab_lines.extend([
        r"\midrule",
        r"\multicolumn{2}{l}{Mean semantic gap} & \MeanSemanticGap & & \\",
        r"\bottomrule",
        r"\end{tabular}",
    ])
    (gen_dir / "tab_semgap.tex").write_text("\n".join(tab_lines) + "\n", encoding="utf-8")
    log.info("Saved: %s", gen_dir / "tab_semgap.tex")


if __name__ == "__main__":
    main()

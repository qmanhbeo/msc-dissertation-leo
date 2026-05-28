"""
Score each paper and each policy chunk against the 17 SDG centroids.

This script produces the primary data products for all downstream analyses:
  - paper_scores.npy / policy_scores.npy    cosine sim per item × SDG (main direction)
  - research_centroids.npy                  per-SDG mean of research embeddings (H26)
  - policy_scores_vs_research.npy           policy scored against research centroids (H26)

Scoring method:
  All embeddings are L2-normalised (unit vectors, from embeddings.py). The SDG centroids are
  also unit-normalised (from sdg_centroids.py, verified in validate_centroids.py). For unit
  vectors, dot product equals cosine similarity. The matrix product E @ C.T computes all
  pairwise cosine similarities in one operation — O(N × 17 × 384).

  ASSUMPTION (A-UNIT): embeddings.py saved all embeddings with normalize_embeddings=True.
  If this is violated, the dot product below is NOT cosine similarity (it is biased inner
  product). This would corrupt all downstream coverage/semantic/interaction results silently.
  We verify L2-norms at runtime and log a warning if they deviate from 1.0.

Bidirectional scoring for H26:
  H26 tests whether research-policy alignment is asymmetric: does research ignore policy more
  than policy ignores research? To test this, we need TWO alignment directions:
    Direction A (main): research papers scored against OSDG-derived SDG centroids.
                        → Does research engage with policy-defined SDG priorities?
    Direction B (H26):  policy chunks scored against research-derived SDG centroids.
                        → Does policy cover the same SDG space that research addresses?

  Research centroids (Direction B) are built by:
    1. Hard-assigning each paper to its top-scoring SDG (argmax of paper_scores).
    2. Computing the mean embedding of all papers assigned to each SDG.
    3. L2-normalising each mean vector.
  ASSUMPTION (A-H26-HARD): Hard assignment (argmax) may produce misleading research centroids
  for SDGs with few assigned papers or where the paper corpus is biased toward certain SDGs.
  Papers are assigned based on OSDG centroids — a circularity, since we use the OSDG centroid
  direction to infer which papers "belong" to each SDG before building the research centroid.
  There is no ground-truth SDG label for research papers, so this is unavoidable. The research
  centroid should be interpreted as "the average embedding of papers the OSDG instrument calls
  SDG X" — not "papers whose authors self-identified as SDG X research."

OSDG circularity diagnostic (A15):
  After scoring, we check whether policy chunks score systematically higher against OSDG
  centroids than research papers do. If so, this would suggest the OSDG-derived centroids are
  calibrated to policy-style language (since OSDG texts are policy-adjacent), inflating policy
  scores relative to research scores. A gap > 0.10 in mean top scores triggers a flag.
  See: notes/ASSUMPTIONS.md, assumption A15.

SDG score interpretation:
  scores[i, j] = cosine similarity of item i to the centroid for SDG (j+1).
  Higher score = higher topical overlap with that SDG's semantic direction.
  These are NOT probabilities and do NOT sum to 1.
  The scores can be negative (if an item is anti-correlated with a centroid direction), but
  this is rare for real-world texts; most scores fall in [0.1, 0.7].

Row ordering convention (inherited from sdg_centroids.py):
  centroids[i] = SDG (i+1)   →   scores[:, 0] = SDG 1 scores, ..., scores[:, 16] = SDG 17 scores
  This convention MUST be maintained across all downstream scripts that index scores by SDG.

Inputs:
  data/sdg_centroids.npy                   (17, 384) float32, unit-normalised
  data/embeddings/papers.npy               (6172, 384) float32, L2-normalised
  data/embeddings/papers_ids.json          list of {id, text}
  data/embeddings/policy.npy               (47005, 384) float32, L2-normalised
  data/embeddings/policy_ids.json          list of {id, text}
  data/policy_all/policy_chunks_extended.jsonl  source_doc metadata for each policy chunk

Outputs:
  data/paper_scores.npy                    (6172, 17)  float32  papers × SDG cosine sim
  data/paper_scores_ids.json               list of {id} — row i matches paper_scores[i]
  data/policy_scores.npy                   (47005, 17) float32  chunks × SDG cosine sim
  data/policy_scores_ids.json              list of {id, source_doc} — row i = policy_scores[i]
  data/research_centroids.npy             (17, 384) float32  per-SDG mean of research papers
  data/research_centroid_meta.json        per-SDG diagnostics for research centroids
  data/policy_scores_vs_research.npy      (47005, 17) float32  policy × research-centroid sim

Run from project root (after validate_centroids.py):
    python code/alignment_score.py
"""

import json
import logging
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
EMBEDDINGS_DIR = Path("data/embeddings")
DATA_DIR = Path("data")
POLICY_CORPUS = Path("data/policy_all/policy_chunks_extended.jsonl")

CENTROIDS_PATH  = DATA_DIR / "sdg_centroids.npy"
PAPERS_EMB      = EMBEDDINGS_DIR / "papers.npy"
PAPERS_IDS      = EMBEDDINGS_DIR / "papers_ids.json"
POLICY_EMB      = EMBEDDINGS_DIR / "policy.npy"
POLICY_IDS      = EMBEDDINGS_DIR / "policy_ids.json"

OUT_PAPER_SCORES    = DATA_DIR / "paper_scores.npy"
OUT_PAPER_IDS       = DATA_DIR / "paper_scores_ids.json"
OUT_POLICY_SCORES   = DATA_DIR / "policy_scores.npy"
OUT_POLICY_IDS      = DATA_DIR / "policy_scores_ids.json"
OUT_RES_CENTROIDS   = DATA_DIR / "research_centroids.npy"
OUT_RES_CENTROID_META = DATA_DIR / "research_centroid_meta.json"
OUT_POLICY_VS_RES   = DATA_DIR / "policy_scores_vs_research.npy"

# A15 circularity threshold: flag if policy mean top-score exceeds paper mean top-score
# by this margin. 0.10 was set in the methodology plan before running analysis.
A15_CIRCULARITY_THRESHOLD = 0.10

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


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def verify_unit_norms(emb: np.ndarray, name: str, n_sample: int = 50) -> None:
    """Sample-check that embedding rows are L2-normalised unit vectors."""
    # Full check would be O(N), sample is sufficient to catch a systematic failure.
    sample = emb[:n_sample]
    norms = np.linalg.norm(sample, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-4):
        log.warning(
            "%s: embeddings may not be L2-normalised (sample norms min=%.4f max=%.4f). "
            "Dot product ≠ cosine similarity — downstream scores will be wrong.",
            name, norms.min(), norms.max()
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

    Args:
        paper_emb:    (N, D) L2-normalised paper embeddings.
        paper_scores: (N, 17) cosine sim of each paper against OSDG centroids.
        n_sdg:        number of SDGs (17).

    Returns:
        (research_centroids, meta_list)
        research_centroids: (17, D) float32, unit-normalised.
        meta_list: list of 17 dicts with n, cohesion, zero_flag per SDG.

    ASSUMPTION (A-H26-HARD): Hard assignment (argmax) is used rather than soft assignment
    (weighted mean). This means each paper contributes to exactly one SDG centroid.
    Papers with ambiguous SDG affiliation (high scores on multiple SDGs) are arbitrarily
    assigned to one SDG. The alternative — soft/weighted assignment — would let each paper
    contribute to multiple centroids proportionally, but would make centroids more similar
    to each other (and to the OSDG centroids, since every paper influences every centroid).
    Hard assignment was chosen for conceptual clarity and to match how OSDG texts are labelled.
    """
    D = paper_emb.shape[1]
    # Hard assignment: paper i → SDG (argmax + 1), i.e., row index 0..16
    assignments = paper_scores.argmax(axis=1)   # (N,) int in 0..16

    centroids = np.zeros((n_sdg, D), dtype=np.float32)
    meta = []

    for sdg_idx in range(n_sdg):
        sdg = sdg_idx + 1
        mask = (assignments == sdg_idx)
        n = int(mask.sum())

        if n == 0:
            # No papers assigned to this SDG — centroid is undefined.
            # ASSUMPTION (A-H26-ZERO): A zero vector is stored and flagged. Downstream H26
            # comparisons involving this SDG's research centroid are unreliable and should
            # be excluded from H26 analysis. This can happen for very low-coverage SDGs
            # (e.g. SDG 2, SDG 14, SDG 16) if the OSDG centroid is rarely the argmax.
            log.warning("SDG %2d: no papers assigned — research centroid is zero vector (H26 unreliable for this SDG)", sdg)
            meta.append({
                "sdg": sdg,
                "n_papers_assigned": 0,
                "raw_centroid_norm": 0.0,
                "mean_cos_to_centroid": 0.0,
                "zero_flag": True,
            })
            continue

        vecs = paper_emb[mask]   # (n, D) L2-normalised
        raw = vecs.mean(axis=0)   # (D,) — mean of unit vectors; not itself a unit vector
        norm = float(np.linalg.norm(raw))

        if norm < 1e-8:
            log.warning("SDG %2d: near-zero centroid norm despite n=%d papers — data may be corrupt", sdg, n)
            meta.append({
                "sdg": sdg,
                "n_papers_assigned": n,
                "raw_centroid_norm": 0.0,
                "mean_cos_to_centroid": 0.0,
                "zero_flag": True,
            })
            continue

        unit = (raw / norm).astype(np.float32)
        centroids[sdg_idx] = unit

        # Cohesion: mean cosine sim of assigned papers to their research centroid.
        # For unit inputs, this equals raw_centroid_norm (derived in sdg_centroids.py).
        mean_cos = float((vecs @ unit).mean())

        meta.append({
            "sdg": sdg,
            "n_papers_assigned": n,
            "raw_centroid_norm": round(norm, 6),
            "mean_cos_to_centroid": round(mean_cos, 6),
            "zero_flag": False,
        })
        log.info(
            "SDG %2d | n_papers=%4d | norm=%.4f | cohesion=%.4f",
            sdg, n, norm, mean_cos
        )

    return centroids, meta


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    # ---- Check if outputs already exist (idempotency) ----
    all_outputs = [
        OUT_PAPER_SCORES, OUT_PAPER_IDS,
        OUT_POLICY_SCORES, OUT_POLICY_IDS,
        OUT_RES_CENTROIDS, OUT_RES_CENTROID_META,
        OUT_POLICY_VS_RES,
    ]
    existing = [p for p in all_outputs if p.exists()]
    if existing:
        log.info("Existing outputs found: %s", [str(p) for p in existing])
        log.info("Delete these files to re-run alignment scoring.")
        return

    # ---- Load SDG centroids ----
    log.info("Loading SDG centroids: %s", CENTROIDS_PATH)
    centroids = np.load(CENTROIDS_PATH)   # (17, 384) float32, unit-normalised
    log.info("  shape=%s  dtype=%s", centroids.shape, centroids.dtype)

    # Verify centroid normalisation — the dot product convention depends on unit vectors.
    centroid_norms = np.linalg.norm(centroids, axis=1)
    if not np.allclose(centroid_norms, 1.0, atol=1e-4):
        log.warning("Centroid norms not all ≈ 1.0 — dot product ≠ cosine sim. Aborting.")
        raise RuntimeError("Centroid normalisation check failed — re-run sdg_centroids.py")
    log.info("  Centroid norms verified ≈ 1.0 ✓")

    # ---- Load paper embeddings ----
    log.info("Loading paper embeddings: %s", PAPERS_EMB)
    paper_emb = np.load(PAPERS_EMB)   # (6172, 384) float32
    paper_ids = load_json(PAPERS_IDS)  # list of {id, text}
    log.info("  shape=%s  n_ids=%d", paper_emb.shape, len(paper_ids))
    assert paper_emb.shape[0] == len(paper_ids), (
        f"Shape mismatch: paper_emb has {paper_emb.shape[0]} rows but paper_ids has {len(paper_ids)} entries"
    )
    verify_unit_norms(paper_emb, "papers")

    # ---- Load policy embeddings + source_doc metadata ----
    log.info("Loading policy embeddings: %s", POLICY_EMB)
    policy_emb = np.load(POLICY_EMB)    # (47005, 384) float32
    policy_ids = load_json(POLICY_IDS)  # list of {id, text}
    log.info("  shape=%s  n_ids=%d", policy_emb.shape, len(policy_ids))
    assert policy_emb.shape[0] == len(policy_ids), (
        f"Shape mismatch: policy_emb has {policy_emb.shape[0]} rows but policy_ids has {len(policy_ids)} entries"
    )
    verify_unit_norms(policy_emb, "policy")

    # Load source_doc per policy chunk. policy_chunks_extended.jsonl is ordered identically
    # to policy_ids.json (verified 2026-04-10 by checking IDs at random indices).
    # source_doc is needed for document-weighted coverage scores in coverage_gap.py (A19).
    log.info("Loading policy source_doc metadata: %s", POLICY_CORPUS)
    policy_chunks = load_jsonl(POLICY_CORPUS)
    assert len(policy_chunks) == len(policy_ids), (
        f"policy_chunks_extended.jsonl ({len(policy_chunks)}) and policy_ids.json "
        f"({len(policy_ids)}) have different lengths — cannot safely join source_doc"
    )
    # Quick spot-check: first and last IDs must match.
    if policy_chunks[0]["chunk_id"] != policy_ids[0]["id"]:
        raise RuntimeError(
            f"Policy corpus ordering mismatch: "
            f"chunks[0]={policy_chunks[0]['chunk_id']} vs ids[0]={policy_ids[0]['id']}"
        )
    if policy_chunks[-1]["chunk_id"] != policy_ids[-1]["id"]:
        raise RuntimeError(
            f"Policy corpus ordering mismatch at last row: "
            f"chunks[-1]={policy_chunks[-1]['chunk_id']} vs ids[-1]={policy_ids[-1]['id']}"
        )
    log.info("  Policy corpus ordering verified ✓")

    # ---- Direction A: Score papers against OSDG SDG centroids ----
    # paper_scores[i, j] = cosine similarity of paper i to centroid for SDG (j+1).
    # ASSUMPTION (A-UNIT): both paper_emb and centroids are unit vectors (verified above).
    # For unit vectors: dot(a, b) = cosine similarity.
    log.info("")
    log.info("Scoring %d papers against 17 SDG centroids...", paper_emb.shape[0])
    paper_scores = paper_emb @ centroids.T    # (6172, 17) float32
    log.info("  paper_scores shape=%s  dtype=%s", paper_scores.shape, paper_scores.dtype)
    log.info(
        "  paper scores — mean=%.4f  std=%.4f  min=%.4f  max=%.4f",
        paper_scores.mean(), paper_scores.std(), paper_scores.min(), paper_scores.max()
    )

    # Per-SDG mean score across all papers (coverage profile preview).
    log.info("")
    log.info("Paper SDG mean scores (SDG 1–17):")
    per_sdg_paper_mean = paper_scores.mean(axis=0)
    for i, v in enumerate(per_sdg_paper_mean):
        log.info("  SDG %2d: %.4f", i + 1, v)

    # ---- Direction A: Score policy chunks against OSDG SDG centroids ----
    # policy_scores[i, j] = cosine similarity of policy chunk i to centroid for SDG (j+1).
    # Same convention and assumption as paper_scores.
    log.info("")
    log.info("Scoring %d policy chunks against 17 SDG centroids...", policy_emb.shape[0])
    policy_scores = policy_emb @ centroids.T  # (47005, 17) float32
    log.info("  policy_scores shape=%s  dtype=%s", policy_scores.shape, policy_scores.dtype)
    log.info(
        "  policy scores — mean=%.4f  std=%.4f  min=%.4f  max=%.4f",
        policy_scores.mean(), policy_scores.std(), policy_scores.min(), policy_scores.max()
    )

    per_sdg_policy_mean = policy_scores.mean(axis=0)
    log.info("")
    log.info("Policy SDG mean scores (SDG 1–17):")
    for i, v in enumerate(per_sdg_policy_mean):
        log.info("  SDG %2d: %.4f", i + 1, v)

    # ---- OSDG circularity diagnostic (A15) ----
    # If policy chunks score systematically higher than research papers against OSDG-derived
    # centroids, it would suggest the centroids are calibrated to policy-style vocabulary
    # (since OSDG texts are themselves policy-adjacent UN/NGO documents), inflating all
    # policy alignment estimates relative to research papers.
    # A gap > A15_CIRCULARITY_THRESHOLD (0.10) triggers a flag in the methodology.
    # This is a diagnostic, not a correctness check — some gap is expected because policy
    # chunks are structurally more similar to OSDG texts than academic abstracts.
    mean_paper_top  = float(paper_scores.max(axis=1).mean())
    mean_policy_top = float(policy_scores.max(axis=1).mean())
    a15_gap = mean_policy_top - mean_paper_top

    log.info("")
    log.info("A15 CIRCULARITY DIAGNOSTIC:")
    log.info("  Mean top score — papers: %.4f  policy: %.4f  gap: %.4f",
             mean_paper_top, mean_policy_top, a15_gap)
    if a15_gap > A15_CIRCULARITY_THRESHOLD:
        log.warning(
            "  A15 FLAG: policy top scores exceed paper top scores by %.4f > %.2f threshold.",
            a15_gap, A15_CIRCULARITY_THRESHOLD
        )
        log.warning(
            "  Interpretation: OSDG-derived centroids may be calibrated to policy vocabulary.")
        log.warning(
            "  Action: Flag A15 in the Limitations section of the methodology chapter.")
        log.warning(
            "  Coverage and semantic gap results still valid but policy scores may be inflated.")
    else:
        log.info(
            "  A15 PASS: gap=%.4f ≤ %.2f — no strong evidence of OSDG circularity bias.",
            a15_gap, A15_CIRCULARITY_THRESHOLD
        )

    # ---- Direction B: Build research centroids (H26) ----
    # ASSUMPTION (A-H26-HARD): papers are hard-assigned to their top OSDG centroid match.
    # See build_research_centroids() docstring for full rationale.
    log.info("")
    log.info("Building research centroids for H26 bidirectional analysis...")
    research_centroids, res_centroid_meta = build_research_centroids(paper_emb, paper_scores)
    log.info("  research_centroids shape=%s", research_centroids.shape)

    # ---- Direction B: Score policy against research centroids (H26) ----
    # policy_scores_vs_research[i, j] = cosine sim of policy chunk i to research centroid j.
    # Zero-vector research centroids (SDGs with no assigned papers) will produce zero scores
    # for that SDG column. These columns are flagged in research_centroid_meta and should be
    # excluded from H26 asymmetry comparisons.
    log.info("")
    log.info("Scoring policy chunks against research centroids (H26)...")
    policy_vs_research = policy_emb @ research_centroids.T  # (47005, 17) float32
    log.info("  policy_vs_research shape=%s", policy_vs_research.shape)
    log.info(
        "  policy vs research scores — mean=%.4f  std=%.4f  min=%.4f  max=%.4f",
        policy_vs_research.mean(), policy_vs_research.std(),
        policy_vs_research.min(), policy_vs_research.max()
    )

    # Brief asymmetry preview for H26 (full analysis deferred to coverage_semantic_interaction.py).
    # Asymmetry = research scores (against OSDG centroids) vs policy scores (against research
    # centroids). If papers score lower against OSDG centroids than policy scores against
    # research centroids → research is more "foreign" to the policy framing than vice versa.
    mean_paper_vs_osdg = float(paper_scores.max(axis=1).mean())
    mean_policy_vs_res = float(policy_vs_research.max(axis=1).mean())
    log.info("")
    log.info("H26 ASYMMETRY PREVIEW:")
    log.info("  Research papers scored vs OSDG centroids — mean top sim: %.4f", mean_paper_vs_osdg)
    log.info("  Policy chunks scored vs research centroids — mean top sim: %.4f", mean_policy_vs_res)
    if mean_policy_vs_res > mean_paper_vs_osdg:
        log.info(
            "  Direction: policy engages research framing more than research engages policy framing "
            "(supports H26 — research ignores policy more than policy ignores research)"
        )
    else:
        log.info(
            "  Direction: research engages policy framing more than policy engages research framing "
            "(against H26 direction)"
        )
    log.info("  NOTE: This preview uses top-SDG scores. Full H26 analysis in coverage_semantic_interaction.py.")

    # ---- Build output IDs ----
    # paper_scores_ids.json: one entry per row in paper_scores.npy, preserving paper ID.
    paper_scores_ids = [{"id": r["id"]} for r in paper_ids]

    # policy_scores_ids.json: one entry per row in policy_scores.npy, with source_doc.
    # source_doc is needed for document-weighted policy scores in coverage_gap.py (A19).
    policy_scores_ids = [
        {"id": c["chunk_id"], "source_doc": c["source_doc"]}
        for c in policy_chunks
    ]

    # ---- Save outputs ----
    np.save(OUT_PAPER_SCORES, paper_scores)
    log.info("Saved: %s  shape=%s", OUT_PAPER_SCORES, paper_scores.shape)

    with OUT_PAPER_IDS.open("w", encoding="utf-8") as f:
        json.dump(paper_scores_ids, f)
    log.info("Saved: %s  n=%d", OUT_PAPER_IDS, len(paper_scores_ids))

    np.save(OUT_POLICY_SCORES, policy_scores)
    log.info("Saved: %s  shape=%s", OUT_POLICY_SCORES, policy_scores.shape)

    with OUT_POLICY_IDS.open("w", encoding="utf-8") as f:
        json.dump(policy_scores_ids, f)
    log.info("Saved: %s  n=%d", OUT_POLICY_IDS, len(policy_scores_ids))

    np.save(OUT_RES_CENTROIDS, research_centroids)
    log.info("Saved: %s  shape=%s", OUT_RES_CENTROIDS, research_centroids.shape)

    with OUT_RES_CENTROID_META.open("w", encoding="utf-8") as f:
        json.dump(res_centroid_meta, f, indent=2)
    log.info("Saved: %s", OUT_RES_CENTROID_META)

    np.save(OUT_POLICY_VS_RES, policy_vs_research)
    log.info("Saved: %s  shape=%s", OUT_POLICY_VS_RES, policy_vs_research.shape)

    log.info("")
    log.info("=" * 60)
    log.info("alignment_score.py complete")
    log.info("=" * 60)
    log.info("Summary:")
    log.info("  Papers scored:          %d × 17 SDGs", paper_emb.shape[0])
    log.info("  Policy chunks scored:   %d × 17 SDGs", policy_emb.shape[0])
    log.info("  A15 gap:                %.4f (%s)",
             a15_gap, "FLAG" if a15_gap > A15_CIRCULARITY_THRESHOLD else "PASS")
    zero_sdgs = [m["sdg"] for m in res_centroid_meta if m["zero_flag"]]
    if zero_sdgs:
        log.warning("  Research centroids with zero vectors (H26 unreliable): SDGs %s", zero_sdgs)
    log.info("")
    log.info("Next step: python code/coverage_gap.py")


if __name__ == "__main__":
    main()

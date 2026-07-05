"""
Compute per-SDG coverage profiles and coverage gap between research and policy corpora.

Coverage gap = the difference in how much each corpus emphasises each SDG, measured as
the absolute difference in SDG proportion between research and policy.

Two coverage profile methods are computed:
  1. RAW (segment-level):       each paper / each policy segment contributes equally.
  2. DOCUMENT-WEIGHTED:       each *document* contributes equally to the policy profile,
                              regardless of how many segments it contains.

The document-weighted method is required for valid analysis (Assumption A19). Without it,
SDGi VNR/VLR national reports (31,941 segments) and SDSN (5,591 segments) dominate the policy
profile by segment count alone, drowning out the curated UN/AI policy documents (8,592 segments)
and UNGDC speeches (6,472 segments). A single large report's SDG emphasis would overwhelm the
full range of policy voices in the corpus.

SDG assignment method:
  Each paper / segment is assigned to its top-scoring SDG (argmax). The coverage proportion for
  SDG j in a corpus is the fraction of items assigned to SDG j. This is the "hard assignment"
  coverage profile. For the document-weighted version, each document's SDG assignment is the
  argmax of its *mean segment score vector* (so large documents are averaged before assignment).

  ASSUMPTION (A-COV-HARD): Hard assignment creates a zero-sum profile — every paper/document
  contributes to exactly one SDG. This understates "breadth" items (e.g. a paper covering
  multiple SDGs) and overstates the dominance of the top SDG. The alternative — fractional
  assignment proportional to scores — would give smoother profiles but make SDGs with similar
  centroids (SDG 1 ↔ SDG 10, sim=0.887) hard to separate. Hard assignment is used for
  consistency with downstream semantic gap calculations, where per-SDG clusters need clean
  membership. Reported alongside mean-score profiles for transparency.

  ADDITIONAL PROFILE (mean-score): We also report the mean cosine similarity to each SDG
  centroid, averaged across all papers (or documents). This is a soft/continuous coverage
  proxy. Both hard-assignment and mean-score profiles are saved for downstream use.

Coverage gap per SDG:
  coverage_gap[sdg] = |research_proportion[sdg] - policy_proportion[sdg]|
  (using document-weighted policy proportions as the canonical comparison)

Inputs:
  2_data/3_scored/paper_scores_shards/metadata/manifest.json
  2_data/3_scored/policy_scores.npy          float32 matrix with one row per policy segment
  2_data/3_scored/metadata/policy_scores_ids.json     list of {id, source_doc}

Outputs:
  4_outputs/main/data/4_2_coverage_document_weighted.json              per-SDG coverage profiles + gap (canonical analysis)
  4_outputs/main/data/4_2_coverage_diagnostic_unweighted.json          segment-level (unweighted) profiles + gap (diagnostic)
  4_outputs/main/tables/*.tex              generated LaTeX macros/tables

Run from project root (after score materialization for the target run context):
    python 1_code/3_main_analysis/1_canonical/0_coverage_gap.py
"""

import json
import logging
import numpy as np
import argparse
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = ROOT / "1_code"
ANALYSIS_ROOT = Path(__file__).resolve().parents[1]
SHARED_DIR = ANALYSIS_ROOT / "0_shared"
for path in (CODE_ROOT, SHARED_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from research_score_shards import aggregate_research_scores
from shared_utils import ensure_canonical_outputs
from model_slug_utils import scored_dir_for_model

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEFAULT_OUTPUT_ROOT = Path("4_outputs")

N_SDG = 17

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


def hard_assignment_profile(scores: np.ndarray) -> np.ndarray:
    """
    Compute per-SDG proportion via hard (argmax) assignment.

    Returns an array of shape (N_SDG,) summing to 1.0, where element j is the fraction
    of items assigned to SDG (j+1).
    """
    assignments = scores.argmax(axis=1)   # (N,) int in 0..16
    counts = np.bincount(assignments, minlength=N_SDG).astype(float)
    return counts / counts.sum()


def mean_score_profile(scores: np.ndarray) -> np.ndarray:
    """
    Compute per-SDG mean cosine similarity across all items.

    Returns an array of shape (N_SDG,) where element j is the mean cosine sim to SDG (j+1).
    This is a soft/continuous coverage proxy — does not sum to 1.
    """
    return scores.mean(axis=0)


def document_weighted_policy_profile(
    policy_scores: np.ndarray,
    policy_ids: list[dict],
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Compute document-weighted per-SDG profiles for the policy corpus.

    Each *document* (unique source_doc) contributes equally, regardless of how many segments
    it contains. This counteracts SDSN/SDGi dominance (Assumption A19).

    Method:
      1. For each document, average all its segment score vectors → document score vector (17,).
      2. Hard-assign the document to its top SDG (argmax of document score vector).
      3. Compute the proportion of documents assigned to each SDG.
      4. Also compute the mean of document score vectors as a soft profile.

    Returns:
      (hard_profile, soft_profile, doc_meta)
      hard_profile: (N_SDG,) proportion of *documents* assigned to each SDG.
      soft_profile: (N_SDG,) mean document-level cosine sim to each SDG centroid.
      doc_meta: {source_doc: {n_segments, sdg_assignment, score_vector}} for diagnostics.
    """
    # Group row indices by source_doc.
    doc_to_rows: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(policy_ids):
        doc_to_rows[r["source_doc"]].append(i)

    n_docs = len(doc_to_rows)
    log.info("  Document-weighted: %d unique source_docs", n_docs)

    doc_vectors = np.zeros((n_docs, N_SDG), dtype=np.float32)
    doc_meta = {}

    for d_idx, (source_doc, row_idxs) in enumerate(doc_to_rows.items()):
        # Average segment scores for this document.
        # This is the document's SDG score profile — its "topical footprint" in SDG space.
        # ASSUMPTION (A-DOC-MEAN): Averaging segment scores assumes all segments of a document
        # are equally representative. Long introductions and appendices contribute the same as
        # substantive body text. A weighted average by segment word count would be more precise
        # but requires loading word_count from policy_segments_all.jsonl. The current
        # approach is conservative and avoids introducing another assumption about weights.
        doc_vec = policy_scores[row_idxs].mean(axis=0)   # (17,)
        doc_vectors[d_idx] = doc_vec

        top_sdg = int(doc_vec.argmax()) + 1   # 1-indexed
        doc_meta[source_doc] = {
            "n_segments": len(row_idxs),
            "sdg_assignment": top_sdg,
            "top_score": round(float(doc_vec.max()), 6),
        }

    # Hard-assignment profile over documents.
    doc_assignments = doc_vectors.argmax(axis=1)   # (n_docs,) int in 0..16
    counts = np.bincount(doc_assignments, minlength=N_SDG).astype(float)
    hard_profile = counts / counts.sum()

    # Soft profile: mean of document-level score vectors.
    soft_profile = doc_vectors.mean(axis=0)

    return hard_profile, soft_profile, doc_meta


def compute_coverage_gap(research_profile: np.ndarray, policy_profile: np.ndarray) -> np.ndarray:
    """
    Compute absolute coverage gap per SDG.

    coverage_gap[j] = |research_profile[j] - policy_profile[j]|

    Returns (N_SDG,) float array.
    """
    return np.abs(research_profile - policy_profile)


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute coverage gap outputs into the canonical output folder.")
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_ROOT))
    p.add_argument("--model", default="all-MiniLM-L6-v2", help=argparse.SUPPRESS)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    model = args.model
    scored_dir = scored_dir_for_model(model)
    PAPER_SCORES_MANIFEST = scored_dir / "paper_scores_shards" / "metadata" / "manifest.json"
    POLICY_SCORES   = scored_dir / "policy_scores.npy"
    POLICY_IDS      = scored_dir / "metadata" / "policy_scores_ids.json"
    layout = ensure_canonical_outputs(Path(args.output_dir))
    out_cov_gap = layout.data_dir / "4_2_coverage_document_weighted.json"
    out_cov_gap_raw = layout.data_dir / "4_2_coverage_diagnostic_unweighted.json"
    tables_dir = layout.tables_dir
    log.info("Canonical output dir: %s", layout.data_dir)

    # ---- Load scores ----
    log.info("Loading paper score shards: %s", PAPER_SCORES_MANIFEST)
    research = aggregate_research_scores(PAPER_SCORES_MANIFEST)
    log.info("  rows=%d", research["n_rows"])

    log.info("Loading policy scores: %s", POLICY_SCORES)
    policy_scores = np.load(POLICY_SCORES)
    policy_ids    = load_json(POLICY_IDS)
    log.info("  shape=%s  n_ids=%d", policy_scores.shape, len(policy_ids))

    # ---- Research coverage profiles ----
    # Papers = 1 abstract = 1 vector. No document-weighting needed for research corpus.
    # Each paper is independently authored; treating them as equal is appropriate.
    log.info("")
    log.info("Computing research coverage profiles...")
    res_hard = research["hard_profile"].astype(np.float64)
    res_soft = research["soft_profile"].astype(np.float64)

    log.info("  Research hard-assignment profile (proportion per SDG):")
    for i, v in enumerate(res_hard):
        log.info("    SDG %2d: %.4f", i + 1, v)

    # ---- Policy coverage profiles — RAW (segment-level) ----
    # Segment-level profile: each of 47,005 segments contributes equally.
    # This is biased by SDSN/SDGi document length — saved as a diagnostic only.
    log.info("")
    log.info("Computing raw (segment-level) policy coverage profiles...")
    pol_raw_hard = hard_assignment_profile(policy_scores)
    pol_raw_soft = mean_score_profile(policy_scores)

    # ---- Policy coverage profiles — DOCUMENT-WEIGHTED (canonical) ----
    # Each document contributes equally. This is the primary analysis profile.
    log.info("")
    log.info("Computing document-weighted policy coverage profiles...")
    pol_dw_hard, pol_dw_soft, doc_meta = document_weighted_policy_profile(
        policy_scores, policy_ids
    )

    log.info("  Document-weighted hard-assignment profile (proportion per SDG):")
    for i, v in enumerate(pol_dw_hard):
        log.info("    SDG %2d: %.4f", i + 1, v)

    # ---- Coverage gaps ----
    gap_dw   = compute_coverage_gap(res_hard, pol_dw_hard)    # canonical
    gap_raw  = compute_coverage_gap(res_hard, pol_raw_hard)   # diagnostic

    log.info("")
    log.info("=" * 70)
    log.info("COVERAGE GAP (document-weighted policy vs research, hard assignment)")
    log.info("=" * 70)
    log.info("  %-6s  %-12s  %-12s  %-12s  %-12s", "SDG", "Research%", "Policy%", "Gap", "Direction")
    log.info("  " + "-" * 65)
    for i in range(N_SDG):
        sdg = i + 1
        r = res_hard[i]
        p = pol_dw_hard[i]
        g = gap_dw[i]
        direction = "RESEARCH>" if r > p else "POLICY>  "
        log.info("  SDG %2d  %10.2f%%  %10.2f%%  %10.4f  %s",
                 sdg, r * 100, p * 100, g, direction)

    log.info("")
    log.info("Total coverage gap (sum of absolute differences): %.4f", gap_dw.sum())
    log.info("Mean coverage gap per SDG:                        %.4f", gap_dw.mean())

    # ---- Build output dicts ----
    sdg_labels = [f"SDG{i+1}" for i in range(N_SDG)]

    def make_sdg_dict(arr: np.ndarray) -> dict:
        return {sdg_labels[i]: round(float(v), 6) for i, v in enumerate(arr)}

    # Canonical output (document-weighted)
    a15_paper_top = float(research["mean_top_overall"])
    a15_policy_top = float(policy_scores.max(axis=1).mean())
    a15_diff = a15_policy_top - a15_paper_top
    coverage_gap_out = {
        "method": "document_weighted",
        "note": (
            "Policy profile uses document-weighted assignment (each source_doc weighted equally). "
            "Research profile uses hard assignment (each paper weighted equally). "
            "Coverage gap = |research_proportion - policy_proportion| per SDG. "
            "See Assumption A19 for document-weighting rationale."
        ),
        "n_research_papers": int(research["n_rows"]),
        "n_policy_segments": int(policy_scores.shape[0]),
        "n_policy_documents": len(doc_meta),
        "a15_note": (
            f"A15 FLAG: policy top scores exceed paper top scores by {a15_diff:.3f} "
            "(threshold 0.10). OSDG centroids may be calibrated to policy vocabulary. "
            "Policy proportions may be inflated relative to research. "
            "Coverage gap results valid but directional comparison should be framed carefully."
        ),
        "research_profile_hard": make_sdg_dict(res_hard),
        "research_profile_soft": make_sdg_dict(res_soft),
        "policy_profile_hard_docweighted": make_sdg_dict(pol_dw_hard),
        "policy_profile_soft_docweighted": make_sdg_dict(pol_dw_soft),
        "coverage_gap_hard": make_sdg_dict(gap_dw),
        "coverage_gap_total": round(float(gap_dw.sum()), 6),
        "coverage_gap_mean": round(float(gap_dw.mean()), 6),
        "top5_largest_gaps": sorted(
            [(sdg_labels[i], round(float(gap_dw[i]), 6)) for i in range(N_SDG)],
            key=lambda x: x[1], reverse=True
        )[:5],
        "top5_research_dominant": sorted(
            [(sdg_labels[i], round(float(res_hard[i] - pol_dw_hard[i]), 6))
             for i in range(N_SDG) if res_hard[i] > pol_dw_hard[i]],
            key=lambda x: x[1], reverse=True
        )[:5],
        "top5_policy_dominant": sorted(
            [(sdg_labels[i], round(float(pol_dw_hard[i] - res_hard[i]), 6))
             for i in range(N_SDG) if pol_dw_hard[i] > res_hard[i]],
            key=lambda x: x[1], reverse=True
        )[:5],
        "per_document_assignments": doc_meta,
    }

    # Diagnostic output (raw/segment-level)
    coverage_gap_raw_out = {
        "method": "segment_level_raw",
        "note": (
            "Each policy segment weighted equally — BIASED by document length. "
            "SDGi VNR/VLR (31,941 segments) and SDSN (5,591 segments) dominate. "
            "Use 4_2_coverage_document_weighted.json (document-weighted) for primary analysis."
        ),
        "n_research_papers": int(research["n_rows"]),
        "n_policy_segments": int(policy_scores.shape[0]),
        "research_profile_hard": make_sdg_dict(res_hard),
        "policy_profile_hard_raw": make_sdg_dict(pol_raw_hard),
        "policy_profile_soft_raw": make_sdg_dict(pol_raw_soft),
        "coverage_gap_hard": make_sdg_dict(gap_raw),
        "coverage_gap_total": round(float(gap_raw.sum()), 6),
        "coverage_gap_mean": round(float(gap_raw.mean()), 6),
    }

    # ---- Save ----
    with out_cov_gap.open("w", encoding="utf-8") as f:
        json.dump(coverage_gap_out, f, indent=2)
    log.info("Saved: %s", out_cov_gap)

    with out_cov_gap_raw.open("w", encoding="utf-8") as f:
        json.dump(coverage_gap_raw_out, f, indent=2)
    log.info("Saved: %s", out_cov_gap_raw)

    log.info("")
    log.info("Next step: python 1_code/3_main_analysis/1_canonical/1_semantic_gap.py")

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

    # A15 calibration bias values
    # Derived combined coverage shares for prose macros
    pol13 = float(pol_dw_hard[12]) * 100
    pol17 = float(pol_dw_hard[16]) * 100
    pol16 = float(pol_dw_hard[15]) * 100
    res4  = float(res_hard[3])  * 100
    res9  = float(res_hard[8])  * 100
    def _ltx_num(n: int) -> str:
        """Format integer with LaTeX-safe thousands separator ({,})."""
        return f"{n:,}".replace(",", "{,}")

    # num_coverage.tex — macro definitions
    num_lines = [
        "% Auto-generated by 1_code/3_main_analysis/1_canonical/0_coverage_gap.py — do not edit manually",
        rf"\newcommand{{\NResearchPapers}}{{{_ltx_num(int(research['n_rows']))}}}",
        rf"\newcommand{{\NPolicySegments}}{{{_ltx_num(policy_scores.shape[0])}}}",
        rf"\newcommand{{\NPolicyDocs}}{{{_ltx_num(len(doc_meta))}}}",
        rf"\newcommand{{\PolicyPctSdgThreePlusSeventeen}}{{{pol13 + pol17:.1f}}}",
        rf"\newcommand{{\PolicyTopThreePct}}{{{pol13 + pol16 + pol17:.1f}}}",
        rf"\newcommand{{\ResearchSdgFourPlusSdgNinePct}}{{{res4 + res9:.1f}}}",
        rf"\newcommand{{\TotalCoverageGap}}{{{float(gap_dw.sum()):.3f}}}",
        rf"\newcommand{{\MeanCoverageGap}}{{{float(gap_dw.mean()):.3f}}}",
        rf"\newcommand{{\AffFifteenPaperScore}}{{{a15_paper_top:.3f}}}",
        rf"\newcommand{{\AffFifteenPolicyScore}}{{{a15_policy_top:.3f}}}",
        rf"\newcommand{{\AffFifteenDiff}}{{{a15_diff:.3f}}}",
    ]
    for i in range(N_SDG):
        sdg = i + 1
        word = _sdg_num_words[sdg]
        r_pct = float(res_hard[i]) * 100
        p_pct = float(pol_dw_hard[i]) * 100
        g = float(gap_dw[i])
        num_lines.extend([
            rf"\newcommand{{\ResearchPctSdg{word}}}{{{r_pct:.1f}}}",
            rf"\newcommand{{\PolicyPctSdg{word}}}{{{p_pct:.1f}}}",
            rf"\newcommand{{\CoverageGapSdg{word}}}{{{g:.3f}}}",
        ])
    (gen_dir / "num_coverage.tex").write_text("\n".join(num_lines) + "\n", encoding="utf-8")
    log.info("Saved: %s", gen_dir / "num_coverage.tex")

    # tab_coverage.tex — full tabular block
    tab_lines = [
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"SDG & Description & Research \% & Policy \% & Gap \\",
        r"\midrule",
    ]
    for i in range(N_SDG):
        sdg = i + 1
        name = _sdg_names_17[sdg]
        footnote = r"$^\dagger$" if sdg == 4 else ""
        r_pct = float(res_hard[i]) * 100
        p_pct = float(pol_dw_hard[i]) * 100
        g     = float(gap_dw[i])
        tab_lines.append(
            rf"SDG {sdg:2d} & {name}{footnote} & {r_pct:.1f} & {p_pct:.1f} & {g:.3f} \\"
        )
    tab_lines.extend([
        r"\midrule",
        r"\multicolumn{4}{l}{Total coverage gap} & \TotalCoverageGap \\",
        r"\multicolumn{4}{l}{Mean per-SDG gap} & \MeanCoverageGap \\",
        r"\bottomrule",
        r"\end{tabular}",
    ])
    (gen_dir / "tab_coverage.tex").write_text("\n".join(tab_lines) + "\n", encoding="utf-8")
    log.info("Saved: %s", gen_dir / "tab_coverage.tex")


if __name__ == "__main__":
    main()

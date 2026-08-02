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
  2_data/5_supervised_scored/{model}/paper_scores_shards/metadata/manifest.json
  2_data/5_supervised_scored/{model}/policy_scores.npy          float32 matrix with one row per policy segment
  2_data/5_supervised_scored/{model}/metadata/policy_scores_ids.json     list of {id, source_doc}

Outputs:
  4_outputs/main/data/4_2_coverage_document_weighted.json              per-SDG coverage profiles + gap (canonical analysis)
  4_outputs/main/data/4_2_coverage_diagnostic_unweighted.json          segment-level (unweighted) profiles + gap (diagnostic)
  4_outputs/main/tables/*.tex              generated LaTeX macros/tables

Run from project root (after score materialization for the target run context):
    python 1_code/7_main_analysis/1_main_text/0_coverage_gap.py
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
from shared_utils import ensure_canonical_outputs, fingerprint_of, should_skip, record_fingerprint
from model_utils import DEFAULT_EMBED_MODEL, DEFAULT_OUTPUT_ROOT, N_SDG, SDG_NAMES, SDG_NUM_WORDS, embed_dir_for_model, output_dir_for_model, scored_dir_for_model, resolve_model_alias
from shard_pipeline_utils import load_json
from semantic_gap_shared import document_weighted_policy_profile, latex_int

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
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


def compute_coverage_gap(research_profile: np.ndarray, policy_profile: np.ndarray) -> np.ndarray:
    """
    Compute absolute coverage gap per SDG.

    coverage_gap[j] = |research_profile[j] - policy_profile[j]|

    Returns (N_SDG,) float array.
    """
    return np.abs(research_profile - policy_profile)


def _route_coverage_payload(model: str, route: str) -> dict | None:
    """Build the coverage-gap payload for route 'mlp' or 'zs'.

    Research proportions come from the route's own assignment counts; the policy
    profile uses the shared A19 document-weighting (document_weighted_policy_profile).
    Returns None if the route's inputs are absent (e.g. a model without that route).
    """
    if route == "mlp":
        scored_dir = scored_dir_for_model(model)
        summary_path = scored_dir / "mlp_scores" / "mlp_summary.json"
        scores_path = scored_dir / "mlp_scores" / "mlp_policy_scores.npy"
        ids_path = scored_dir / "metadata" / "policy_scores_ids.json"
        if not (summary_path.exists() and scores_path.exists() and ids_path.exists()):
            return None
        with open(summary_path) as f:
            summary = json.load(f)
        res_counts = {int(k): v for k, v in summary["research_coverage"].items()}
        res_total = summary["research_total"]
        policy_scores = np.load(scores_path)
        with open(ids_path) as f:
            policy_ids = json.load(f)
    else:  # zs
        gap_path = output_dir_for_model(model) / "data" / "semantic_gap_distances.json"
        embed_dir = embed_dir_for_model(model)
        emb_path = embed_dir / "policy.npy"
        ids_path = embed_dir / "metadata" / "policy_ids.json"
        centroids_path = scored_dir_for_model(model) / "sdg_centroids.npy"
        if not (gap_path.exists() and emb_path.exists() and ids_path.exists() and centroids_path.exists()):
            return None
        with open(gap_path) as f:
            data = json.load(f)
        res_counts = {r["sdg"]: r["n_papers"] for r in data["per_sdg"]}
        res_total = sum(res_counts.values())
        policy_emb = np.load(emb_path).astype(np.float32)
        with open(ids_path) as f:
            policy_ids = json.load(f)
        centroids = np.load(centroids_path).astype(np.float32)
        policy_scores = policy_emb @ centroids.T

    pol_profile, _, _ = document_weighted_policy_profile(policy_scores, policy_ids)

    sdg_labels = [f"SDG{i+1}" for i in range(N_SDG)]

    def make_sdg_dict(arr: np.ndarray) -> dict:
        return {sdg_labels[i]: round(float(v), 6) for i, v in enumerate(arr)}

    res_profile = np.zeros(N_SDG, dtype=np.float64)
    for sdg, c in res_counts.items():
        res_profile[sdg - 1] = c / res_total
    gap = np.abs(res_profile - pol_profile)

    return {
        "method": "document_weighted",
        "route": route,
        "note": (
            f"Document-weighted coverage gap for the {route.upper()} assignment route. "
            "Policy profile uses document-weighted assignment (Assumption A19); "
            "research profile uses the route's own hard-assignment counts. "
            "Coverage gap = |research_proportion - policy_proportion| per SDG."
        ),
        "n_policy_documents": int(pol_profile.shape[0]),
        "research_profile_hard": make_sdg_dict(res_profile),
        "policy_profile_hard_docweighted": make_sdg_dict(pol_profile),
        "coverage_gap_hard": make_sdg_dict(gap),
        "coverage_gap_total": round(float(gap.sum()), 6),
        "coverage_gap_mean": round(float(gap.mean()), 6),
    }


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute coverage gap outputs into the canonical output folder.")
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_ROOT))
    p.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL, type=resolve_model_alias, help=argparse.SUPPRESS)
    p.add_argument("--paper-scores-manifest", default=None,
                   help="Override paper score shards manifest (default: canonical paper_scores_shards/metadata/manifest.json). Used for the concept-retrieval variant.")
    p.add_argument("--out-data-dir", default=None,
                   help="Override output data directory (default: canonical layout data_dir). Concept variant writes here.")
    p.add_argument("--out-tables-dir", default=None,
                   help="Override output tables directory (default: canonical layout tables_dir). Concept variant writes here.")
    p.add_argument("--overwrite", action="store_true", help=argparse.SUPPRESS)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run(args: argparse.Namespace) -> None:
    model = args.embed_model
    scored_dir = scored_dir_for_model(model)
    PAPER_SCORES_MANIFEST = Path(args.paper_scores_manifest) if args.paper_scores_manifest else scored_dir / "paper_scores_shards" / "metadata" / "manifest.json"
    POLICY_SCORES   = scored_dir / "policy_scores.npy"
    POLICY_IDS      = scored_dir / "metadata" / "policy_scores_ids.json"
    layout = ensure_canonical_outputs(Path(args.output_dir), model=model)
    if args.out_data_dir:
        Path(args.out_data_dir).mkdir(parents=True, exist_ok=True)
    if args.out_tables_dir:
        Path(args.out_tables_dir).mkdir(parents=True, exist_ok=True)
    out_cov_gap = Path(args.out_data_dir).joinpath("4_2_coverage_document_weighted.json") if args.out_data_dir else layout.data_dir / "4_2_coverage_document_weighted.json"
    out_cov_gap_raw = Path(args.out_data_dir).joinpath("4_2_coverage_diagnostic_unweighted.json") if args.out_data_dir else layout.data_dir / "4_2_coverage_diagnostic_unweighted.json"
    out_cov_gap_mlp = layout.data_dir / "mlp_coverage_document_weighted.json"
    out_cov_gap_zs = layout.data_dir / "zs_coverage_document_weighted.json"
    tables_dir = Path(args.out_tables_dir) if args.out_tables_dir else layout.tables_dir
    log.info("Canonical output dir: %s", layout.data_dir)

    SCRIPT_VERSION = "1"
    PRIMARY = out_cov_gap
    OUTPUTS = [out_cov_gap, out_cov_gap_raw, out_cov_gap_mlp, out_cov_gap_zs]
    fp = fingerprint_of(PAPER_SCORES_MANIFEST, POLICY_SCORES, POLICY_IDS,
                        embed_dir_for_model(model) / "policy.npy",
                        scored_dir / "research_centroids.npy")
    fp += SCRIPT_VERSION
    if should_skip(OUTPUTS, fp, args.overwrite, PRIMARY):
        log.info("Skipping %s — inputs unchanged", PRIMARY)
        return

    # ---- Load scores ----
    log.info("Loading paper score shards: %s", PAPER_SCORES_MANIFEST)
    research = aggregate_research_scores(PAPER_SCORES_MANIFEST, scored_dir)
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

    # Per-SDG policy–research "top score" gap (uniformity check for the A15 asymmetry claim).
    # Policy: mean of each segment's max centroid similarity, averaged within the SDG it tops.
    # Research: mean_top_per_sdg already computed by aggregate_research_scores (mean top score
    # per SDG over papers assigned to that SDG).
    pol_top = policy_scores.max(axis=1)
    pol_arg = policy_scores.argmax(axis=1)
    pol_cnt = np.bincount(pol_arg, minlength=N_SDG).astype(float)
    pol_top_per_sdg = np.bincount(pol_arg, weights=pol_top, minlength=N_SDG).astype(float)
    valid = pol_cnt > 0
    pol_top_per_sdg[valid] /= pol_cnt[valid]
    res_top_per_sdg = np.asarray(research["mean_top_per_sdg"], dtype=float)
    per_sdg_top_gap = np.full(N_SDG, np.nan)
    per_sdg_top_gap[valid] = pol_top_per_sdg[valid] - res_top_per_sdg[valid]
    finite = ~np.isnan(per_sdg_top_gap)
    gap_stats = {
        "min": float(per_sdg_top_gap[finite].min()),
        "max": float(per_sdg_top_gap[finite].max()),
        "std": float(per_sdg_top_gap[finite].std()),
        "mean": float(per_sdg_top_gap[finite].mean()),
        "n_valid": int(finite.sum()),
    }
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
        "per_sdg_top_score_gap": {
            f"SDG{i+1}": (None if np.isnan(v) else round(float(v), 6))
            for i, v in enumerate(per_sdg_top_gap)
        },
        "per_sdg_top_score_gap_stats": gap_stats,
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

    # ---- Persist MLP/ZS route coverage gaps (single source of truth) ----
    # Skipped under the concept override (--out-data-dir): the route JSONs are only
    # meaningful for the canonical main-text run.
    if args.out_data_dir is None:
        for route, out_path in (("mlp", out_cov_gap_mlp), ("zs", out_cov_gap_zs)):
            payload = _route_coverage_payload(model, route)
            if payload is None:
                log.info("Skipping %s coverage JSON (route inputs absent for model %s)",
                         route.upper(), model)
                continue
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            log.info("Saved: %s", out_path)

    log.info("")
    log.info("Next step: python 1_code/7_main_analysis/1_main_text/1_semantic_gap.py")

    # ---- Write LaTeX generated outputs ----
    gen_dir = tables_dir

    # A15 calibration bias values
    # Derived combined coverage shares for prose macros
    pol13 = float(pol_dw_hard[12]) * 100
    pol17 = float(pol_dw_hard[16]) * 100
    pol16 = float(pol_dw_hard[15]) * 100
    res4  = float(res_hard[3])  * 100
    res9  = float(res_hard[8])  * 100

    # num_coverage.tex — macro definitions
    num_lines = [
        "% Auto-generated by 1_code/7_main_analysis/1_main_text/0_coverage_gap.py — do not edit manually",
        rf"\newcommand{{\NResearchPapers}}{{{latex_int(int(research['n_rows']))}}}",
        rf"\newcommand{{\NPolicySegments}}{{{latex_int(policy_scores.shape[0])}}}",
        rf"\newcommand{{\NPolicyDocs}}{{{latex_int(len(doc_meta))}}}",
        rf"\newcommand{{\PolicyPctSdgThreePlusSeventeen}}{{{pol13 + pol17:.1f}}}",
        rf"\newcommand{{\PolicyTopThreePct}}{{{pol13 + pol16 + pol17:.1f}}}",
        rf"\newcommand{{\ResearchSdgFourPlusSdgNinePct}}{{{res4 + res9:.1f}}}",
        rf"\newcommand{{\TotalCoverageGap}}{{{float(gap_dw.sum()):.3f}}}",
        rf"\newcommand{{\MeanCoverageGap}}{{{float(gap_dw.mean()):.3f}}}",
        rf"\newcommand{{\AffFifteenPaperScore}}{{{a15_paper_top:.3f}}}",
        rf"\newcommand{{\AffFifteenPolicyScore}}{{{a15_policy_top:.3f}}}",
        rf"\newcommand{{\AffFifteenDiff}}{{{a15_diff:.3f}}}",
        rf"\newcommand{{\TopGapMin}}{{{gap_stats['min']:.3f}}}",
        rf"\newcommand{{\TopGapMax}}{{{gap_stats['max']:.3f}}}",
        rf"\newcommand{{\TopGapStd}}{{{gap_stats['std']:.3f}}}",
        rf"\newcommand{{\TopGapMean}}{{{gap_stats['mean']:.3f}}}",
    ]
    for i in range(N_SDG):
        sdg = i + 1
        word = SDG_NUM_WORDS[sdg]
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
        "% Auto-generated by 1_code/7_main_analysis/1_main_text/0_coverage_gap.py — do not edit manually",
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"SDG & Description & Research \% & Policy \% & Gap \\",
        r"\midrule",
    ]
    for i in range(N_SDG):
        sdg = i + 1
        name = SDG_NAMES[sdg]
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
        r"\par\smallskip\footnotesize\emph{Notes:} $^\dagger$SDG 4 research share is treated as artefact-affected: ML papers share core vocabulary with the OSDG education corpus (see Section~\ref{sec:sdg4artefact}).",
    ])
    (gen_dir / "tab_coverage.tex").write_text("\n".join(tab_lines) + "\n", encoding="utf-8")
    log.info("Saved: %s", gen_dir / "tab_coverage.tex")
    record_fingerprint(OUTPUTS, fp, PRIMARY)


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()

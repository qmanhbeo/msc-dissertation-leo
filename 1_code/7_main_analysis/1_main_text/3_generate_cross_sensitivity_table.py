"""
Generate main-text tables for the supervised LR classifier.

Outputs:
  tab1_classifier_performance.tex                     — single-column LR test F1 (canonical)
  tab6_cross_sensitivity.tex   — 3-axis gap-rank sensitivity table
  num6_cross_sensitivity.tex             — segment-cap stability macro
  num1_classifier_performance.tex                    — per-SDG F1 macros for use in text
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = ROOT / "1_code"
ANALYSIS_ROOT = Path(__file__).resolve().parents[1]
SHARED_DIR = ANALYSIS_ROOT / "0_shared"
for path in (CODE_ROOT, SHARED_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from model_utils import DEFAULT_EMBED_MODEL, DEFAULT_OUTPUT_ROOT, N_SDG, embed_dir_for_model, model_results_dir_for_model, model_slug, output_dir_for_model, scored_dir_for_model, resolve_model_alias
from shared_utils import fingerprint_of, should_skip, record_fingerprint
from semantic_gap_shared import document_weighted_policy_profile, load_route_coverage_gap

SDG_NAMES = {
    1: "No Poverty", 2: "Zero Hunger", 3: "Good Health", 4: "Quality Education",
    5: "Gender Equality", 6: "Clean Water", 7: "Clean Energy",
    8: "Decent Work", 9: "Industry \\& Infra.", 10: "Reduced Inequalities",
    11: "Sustainable Cities", 12: "Responsible Cons.", 13: "Climate Action",
    14: "Life Below Water", 15: "Life on Land", 16: "Peace \\& Justice",
    17: "Partnerships",
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL, type=resolve_model_alias, help=argparse.SUPPRESS)
    parser.add_argument("--overwrite", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()

# ---------------------------------------------------------------------------
# 2/3/4. Load LR / zero-shot / MLP semantic gaps for an ARBITRARY encoder.
#    Paths are derived from `root` (set in run()) so the same loader
#    serves both the canonical encoder and the encoder-sensitivity partner.
# ---------------------------------------------------------------------------
def load_lr_gaps(m):
    p = output_dir_for_model(m, root=root) / "data" / "semantic_gap_distances_lr.json"
    if not p.exists():
        return None
    with open(p) as f:
        data = json.load(f)
    return {row["sdg"]: row["semantic_gap"] for row in data["per_sdg"] if row["semantic_gap"] is not None}


def load_zs_gaps(m):
    p = output_dir_for_model(m, root=root) / "data" / "semantic_gap_distances_zeroshot.json"
    if not p.exists():
        return None
    with open(p) as f:
        data = json.load(f)
    return {row["sdg"]: row["semantic_gap"] for row in data["per_sdg"] if row["semantic_gap"] is not None}


def load_mlp_gaps(m):
    # Capped, single-source MLP gap (mirrors load_lr_gaps). Replaces the
    # uncapped mlp_summary.json["semantic_gaps"] value, which was divergent.
    p = output_dir_for_model(m, root=root) / "data" / "semantic_gap_distances_mlp.json"
    if not p.exists():
        return None
    with open(p) as f:
        data = json.load(f)
    return {row["sdg"]: row["semantic_gap"] for row in data["per_sdg"] if row["semantic_gap"] is not None}


def load_concept_mlp_gaps(m):
    p = output_dir_for_model(m, root=root) / "data" / "concept" / "semantic_gap_distances_mlp.json"
    if not p.exists():
        return None
    with open(p) as f:
        data = json.load(f)
    return {row["sdg"]: row["semantic_gap"] for row in data["per_sdg"] if row["semantic_gap"] is not None}


# ---------------------------------------------------------------------------
# Adjusted (register-removed) gap loaders — mirror the raw loaders but read the
# `adjusted/` subdirectory. These feed the canonical (adjusted) ranking columns.
# ---------------------------------------------------------------------------
def _load_gap_json(path):
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    return {row["sdg"]: row["semantic_gap"] for row in data["per_sdg"] if row["semantic_gap"] is not None}


def load_lr_gaps_adj(m):
    return _load_gap_json(output_dir_for_model(m, root=root) / "data" / "adjusted" / "semantic_gap_distances_lr.json")


def load_mlp_gaps_adj(m):
    return _load_gap_json(output_dir_for_model(m, root=root) / "data" / "adjusted" / "semantic_gap_distances_mlp.json")


def load_zs_gaps_adj(m):
    return _load_gap_json(output_dir_for_model(m, root=root) / "data" / "adjusted" / "semantic_gap_distances_zeroshot.json")


def load_concept_lr_gaps_adj(m):
    return _load_gap_json(output_dir_for_model(m, root=root) / "data" / "concept" / "adjusted" / "semantic_gap_distances_lr.json")


def load_concept_mlp_gaps_adj(m):
    return _load_gap_json(output_dir_for_model(m, root=root) / "data" / "concept" / "adjusted" / "semantic_gap_distances_mlp.json")


def load_cap_gaps_adj():
    p = output_dir_for_model(DEFAULT_EMBED_MODEL, root=root) / "data" / "adjusted" / "semantic_gap_robustness_caps_lr.json"
    if not p.exists():
        return None, None
    with open(p) as f:
        data = json.load(f)
    cap_20 = {row["sdg"]: row["semantic_gap"] for row in data.get("cap_20", []) if row.get("semantic_gap") is not None}
    cap_none = {row["sdg"]: row["semantic_gap"] for row in data.get("cap_none", []) if row.get("semantic_gap") is not None}
    return cap_20, cap_none


def _parse_policy_source_gaps_from(tex_path):
    """Shared parser: extract per-family semantic gaps from an a2 combined tex table."""
    text = tex_path.read_text(encoding="utf-8")
    in_header = True
    families = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("%") or line.startswith(r"\toprule") or line.startswith(r"\midrule") or line.startswith(r"\bottomrule") or line.startswith(r"\end") or line.startswith(r"\cmidrule"):
            continue
        if in_header and "SDG" in line and "&" in line:
            in_header = False
            continue
        if in_header:
            continue
        m = re.match(r"(\d+)", line)
        if not m:
            continue
        sdg = int(m.group(1))
        parts = [p.strip() for p in line.rstrip("\\").split("&")]
        # Format: num & cov.(n) & sem.(n) & cov.(n) & sem.(n) & cov.(n) & sem.(n) & cov.(n) & sem.(n)
        # sem.(n) cells at indices 2, 4, 6, 8 — cell format "0.435 (1,634)"
        gap_indices = [2, 4, 6, 8]
        labels = ["full", "curated", "sdgi", "ungdc"]
        for label, gi in zip(labels, gap_indices):
            if gi < len(parts):
                try:
                    val = float(parts[gi].split()[0])
                    families.setdefault(label, {})[sdg] = val
                except ValueError:
                    pass
    return families


def parse_policy_source_gaps():
    """Return {family_label: {sdg: gap}} parsing the (raw) appendix tex table."""
    if not POLICY_SOURCE_FAMILY_TEX.exists():
        return {}
    return _parse_policy_source_gaps_from(POLICY_SOURCE_FAMILY_TEX)


def parse_policy_source_gaps_adj():
    """Adjusted variant of parse_policy_source_gaps: read the adjusted tex table."""
    adj_tex = (root / "appendix" / model_slug(DEFAULT_EMBED_MODEL) / "a2_source_family_sensitivity"
               / "tables" / "adjusted" / "tab_a2_policy_source_family_combined.tex")
    if not adj_tex.exists():
        return {}
    return _parse_policy_source_gaps_from(adj_tex)


def _spearman(x, y):
    """Spearman rho via Pearson of the rank vectors (scipy-free).

    Each column is a permutation of SDG gap ranks, so the Pearson
    correlation of the two rank vectors equals Spearman. Used for the
    Rank-Corr row vs the canonical MPNet-LR baseline.
    """
    n = len(x)
    if n < 2:
        return float("nan")
    mx = sum(x) / n
    my = sum(y) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(x, y))
    vx = sum((a - mx) ** 2 for a in x)
    vy = sum((b - my) ** 2 for b in y)
    if vx == 0 or vy == 0:
        return float("nan")
    return cov / (vx ** 0.5 * vy ** 0.5)

# ---------------------------------------------------------------------------
# 4. Load segment-cap robustness gaps
# ---------------------------------------------------------------------------
def load_cap_gaps():
    if not CAP_PATH.exists():
        return None, None
    with open(CAP_PATH) as f:
        data = json.load(f)
    cap_20 = {row["sdg"]: row["semantic_gap"] for row in data.get("cap_20", []) if row.get("semantic_gap") is not None}
    cap_none = {row["sdg"]: row["semantic_gap"] for row in data.get("cap_none", []) if row.get("semantic_gap") is not None}
    return cap_20, cap_none

# ---------------------------------------------------------------------------
# Concept-retrieval variant: coverage/semantic gap loaders + Kendall tau
# ---------------------------------------------------------------------------
SEMANTIC_CAPTION = "Cross-sensitivity robustness of within-SDG semantic-gap rankings across policy source, segment cap, and retrieval strategy."
SEMANTIC_NOTES = (
    "Each cell reports the within-SDG semantic-gap rank "
    "($1 = \\text{largest gap}$, $17 = \\text{smallest gap}$). "
    "Panel~(a) reports the register-removed (adjusted) ranking, which is the canonical measure of this study; "
    "Panel~(b) reports the naive baseline (un-adjusted, raw) ranking. "
    "Within each panel, the Canon column is the canonical MPNet---LR ranking. "
    "Policy-source columns compare the keyword-retrieved research profile against each policy-source family. "
    "Segment-cap columns compare cap~20 and no cap. "
    "The Retrieval column replaces keyword retrieval with concept-based (AI/ML field-of-study) retrieval. "
    "Rank Corr ($\\rho$) is the Spearman correlation of each column's SDG gap ranks against that panel's Canon column."
)
COVERAGE_CAPTION = "Cross-sensitivity robustness of within-SDG coverage-gap rankings across policy source and retrieval strategy."
COVERAGE_NOTES = (
    "Each cell reports the within-SDG coverage-gap rank "
    "($1 = \\text{largest gap}$, $17 = \\text{smallest gap}$). "
    "The Canon column is the canonical MPNet---LR ranking. "
    "Coverage gap = $|\\text{research proportion} - \\text{policy proportion}|$ "
    "per SDG, using document-weighted policy proportions (Assumption A19). "
    "Policy-source columns compare the keyword-retrieved research profile against each policy-source family. "
    "The Retrieval column replaces keyword retrieval with concept-based (AI/ML field-of-study) retrieval. "
    "The Segment-cap axis is omitted: coverage gap is segment-cap-independent. "
    "Rank Corr ($\\rho$) is the Spearman correlation of each column's SDG coverage-gap ranks against the canonical column."
)


# ---------------------------------------------------------------------------
# Domain-encoder sensitivity (same-dimension scientific encoder, SciBERT).
# Added for the encoder-sensitivity robustness check: MPNet (768-d) vs
# SciBERT (768-d) isolates architecture/domain from the dimensionality drop
# that confounds the MPNet--MiniLM (384-d) pair.
# ---------------------------------------------------------------------------
ENC_AXIS_ENCODERS = [
    ("all-mpnet-base-v2", "MPNet (768-d)"),
    ("all-MiniLM-L6-v2", "MiniLM (384-d)"),
    ("allenai/scibert_scivocab_uncased", "SciBERT (768-d)"),
]
ENC_AXIS_SEMANTIC_CAPTION = (
    "Domain-encoder sensitivity of within-SDG semantic-gap rankings across embedding "
    "architectures of differing domain specialisation but matched dimensionality."
)
ENC_AXIS_SEMANTIC_NOTES = (
    "Each cell reports the within-SDG semantic-gap rank ($1 = \\text{largest gap}$, "
    "$17 = \\text{smallest gap}$) under that encoder and assignment method. "
    "Panel~(a) reports the register-removed (adjusted, canonical) ranking; "
    "Panel~(b) reports the naive baseline (un-adjusted, raw) ranking. "
    "MPNet (768-d) is the canonical general-purpose encoder; MiniLM (384-d) is a smaller "
    "general-purpose encoder; SciBERT (768-d) is a scientific-domain encoder. "
    "MPNet and SciBERT share dimensionality (768-d), isolating architecture/domain from "
    "the dimensionality drop that confounds the MPNet--MiniLM pair "
    "(Section~\\ref{sec:encoder-sensitivity}). "
    "LR = logistic regression; MLP = 4-layer/384-hidden network; "
    "Zero-shot = nearest-centroid on SDG reference centroids, "
    "reported for the canonical MPNet encoder only (scoped to a single supervised-vs-"
    "nearest-centroid comparison, Appendix~\\ref{app:assignment-method-comparison}). "
    "Rank Corr ($\\rho$) is the Spearman correlation of each column's SDG gap ranks against that "
    "panel's canonical adjusted (Panel~a) or raw (Panel~b) MPNet-LR column."
)
ENC_AXIS_COVERAGE_CAPTION = (
    "Domain-encoder sensitivity of within-SDG coverage-gap rankings across embedding "
    "architectures of differing domain specialisation but matched dimensionality."
)
ENC_AXIS_COVERAGE_NOTES = (
    "Each cell reports the within-SDG coverage-gap rank ($1 = \\text{largest gap}$, "
    "$17 = \\text{smallest gap}$) under that encoder and assignment method. "
    "MPNet (768-d) is the canonical general-purpose encoder; MiniLM (384-d) is a smaller "
    "general-purpose encoder; SciBERT (768-d) is a scientific-domain encoder. "
    "MPNet and SciBERT share dimensionality (768-d), isolating architecture/domain from "
    "the dimensionality drop that confounds the MPNet--MiniLM pair "
    "(Section~\\ref{sec:encoder-sensitivity}). "
    "LR = canonical supervised logistic regression; MLP = 4-layer/384-hidden network; "
    "Zero-shot = nearest-centroid on SDG reference centroids, "
    "reported for the canonical MPNet encoder only. Coverage gap is "
    "document-weighted (Assumption A19) for all methods. "
    "Rank Corr ($\\rho$) is the Spearman correlation of each column's SDG coverage-gap ranks "
    "against the canonical MPNet-LR column."
)


def load_mean_gap_and_cohesion(m):
    """Mean semantic gap, 1-gap (cosine similarity), and mean per-SDG cohesion.

    Reads the committed 4_3 semantic-gap distances JSON for encoder ``m`` and
    returns (mean_gap, mean_cos, mean_research_cohesion, mean_policy_cohesion).
    Returns None if the artifact is missing.
    """
    p = output_dir_for_model(m, root=root) / "data" / "semantic_gap_distances_lr.json"
    if not p.exists():
        return None
    with open(p) as f:
        data = json.load(f)
    rows = [r for r in data["per_sdg"] if r.get("semantic_gap") is not None]
    if not rows:
        return None
    gaps = [r["semantic_gap"] for r in rows]
    mean_gap = float(np.mean(gaps))
    mean_cos = 1.0 - mean_gap
    res_co = [r["research_cohesion"] for r in rows if r.get("research_cohesion") is not None]
    pol_co = [r["policy_cohesion"] for r in rows if r.get("policy_cohesion") is not None]
    mean_res_co = float(np.mean(res_co)) if res_co else None
    mean_pol_co = float(np.mean(pol_co)) if pol_co else None
    return mean_gap, mean_cos, mean_res_co, mean_pol_co


def write_encoder_axis_semantic():
    enc_subgroups_adj = []
    enc_subgroups_raw = []
    for m, sublabel in ENC_AXIS_ENCODERS:
        lr = load_lr_gaps(m)
        mlp = load_mlp_gaps(m)
        zs = load_zs_gaps(m) if m == DEFAULT_EMBED_MODEL else None
        lr_adj = load_lr_gaps_adj(m)
        mlp_adj = load_mlp_gaps_adj(m)
        zs_adj = load_zs_gaps_adj(m) if m == DEFAULT_EMBED_MODEL else None
        cols_adj = []
        cols_raw = []
        if lr_adj:
            cols_adj.append(("LR", compute_ranks(lr_adj),
                             "LR — register-removed (adjusted, canonical) — policy segments capped at 50/doc/SDG"))
        if lr:
            cols_raw.append(("LR", compute_ranks(lr),
                             "LR (canonical supervised, raw naive baseline) — policy segments capped at 50/doc/SDG"))
        if mlp_adj:
            cols_adj.append(("MLP", compute_ranks(mlp_adj),
                             "MLP — register-removed (adjusted, canonical) — policy segments capped at 50/doc/SDG"))
        if mlp:
            cols_raw.append(("MLP", compute_ranks(mlp),
                             "MLP (4-layer/384-hidden, raw naive baseline) — policy segments capped at 50/doc/SDG"))
        if zs_adj:
            cols_adj.append(("ZS", compute_ranks(zs_adj),
                             "Zero-shot nearest-centroid (register-removed, adjusted) on SDG reference centroids (canonical encoder only)"))
        if zs:
            cols_raw.append(("ZS", compute_ranks(zs),
                             "Zero-shot nearest-centroid (raw naive baseline) on SDG reference centroids (canonical encoder only)"))
        if cols_adj:
            enc_subgroups_adj.append((sublabel, cols_adj))
        if cols_raw:
            enc_subgroups_raw.append((sublabel, cols_raw))
    if not enc_subgroups_adj and not enc_subgroups_raw:
        print("WARNING: no encoder data for encoder-axis semantic table, skipping")
        return
    rho_a, rho_b = assemble_paneled(
        adj_groups=[("Encoder (embedding architecture)", enc_subgroups_adj)],
        raw_groups=[("Encoder (embedding architecture)", enc_subgroups_raw)],
        out_path=OUT_MAIN / "tab7_encoder_sensitivity.tex",
        caption=ENC_AXIS_SEMANTIC_CAPTION, notes=ENC_AXIS_SEMANTIC_NOTES, label="tab:encoder-sensitivity-semantic",
        panel_a_note="Adjusted (register-removed, canonical).",
        panel_b_note="Raw (naive baseline, un-adjusted).",
    )
    scibert_key = "Encoder (embedding architecture)::SciBERT (768-d)::LR"
    scibert_rho = rho_a.get(scibert_key, float("nan"))
    val = f"{scibert_rho:.2f}" if not np.isnan(scibert_rho) else "--"
    mlp_key = "Encoder (embedding architecture)::MPNet (768-d)::MLP"
    mlp_rho = rho_a.get(mlp_key, float("nan"))
    mlp_val = f"{mlp_rho:.2f}" if not np.isnan(mlp_rho) else "--"
    zs_key = "Encoder (embedding architecture)::MPNet (768-d)::ZS"
    zs_rho = rho_a.get(zs_key, float("nan"))
    zs_val = f"{zs_rho:.2f}" if not np.isnan(zs_rho) else "--"
    minilm_key = "Encoder (embedding architecture)::MiniLM (384-d)::LR"
    minilm_rho = rho_a.get(minilm_key, float("nan"))
    minilm_val = f"{minilm_rho:.2f}" if not np.isnan(minilm_rho) else "--"
    lines = [
        "% Auto-generated by 1_code/7_main_analysis/1_main_text/3_generate_cross_sensitivity_table.py — do not edit manually",
        rf"\newcommand{{\SciBERTSemanticRho}}{{{val}}}",
        rf"\newcommand{{\MlpSemanticRho}}{{{mlp_val}}}",
        rf"\newcommand{{\ZeroShotSemanticRho}}{{{zs_val}}}",
        rf"\newcommand{{\MiniLMSemanticRho}}{{{minilm_val}}}",
    ]
    scibert_stats = load_mean_gap_and_cohesion("allenai/scibert_scivocab_uncased")
    if scibert_stats:
        sg, sc, src, spc = scibert_stats
        lines.append(rf"\newcommand{{\SciBERTMeanGap}}{{{sg:.2f}}}")
        lines.append(rf"\newcommand{{\SciBERTMeanCos}}{{{sc:.2f}}}")
        if src is not None:
            lines.append(rf"\newcommand{{\SciBERTResCohesion}}{{{src:.2f}}}")
        if spc is not None:
            lines.append(rf"\newcommand{{\SciBERTPolCohesion}}{{{spc:.2f}}}")
    mpnet_stats = load_mean_gap_and_cohesion("all-mpnet-base-v2")
    if mpnet_stats:
        mg, mc, mrc, mpc = mpnet_stats
        lines.append(rf"\newcommand{{\MPNetMeanGap}}{{{mg:.2f}}}")
        lines.append(rf"\newcommand{{\MPNetMeanCos}}{{{mc:.2f}}}")
        if mrc is not None:
            lines.append(rf"\newcommand{{\MPNetResCohesion}}{{{mrc:.2f}}}")
        if mpc is not None:
            lines.append(rf"\newcommand{{\MPNetPolCohesion}}{{{mpc:.2f}}}")
    (OUT_MAIN / "num7_encoder_sensitivity.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    print(f"Written num7_encoder_sensitivity.tex  scibert_lr_rho={val} mlp_rho={mlp_val} zs_rho={zs_val} minilm_rho={minilm_val}")


def write_encoder_axis_coverage():
    enc_subgroups = []
    for m, sublabel in ENC_AXIS_ENCODERS:
        lr = load_lr_covgaps(m)
        mlp = load_mlp_covgaps(m)
        zs = load_zs_covgaps(m) if m == DEFAULT_EMBED_MODEL else None
        cols = []
        if lr:
            cols.append(("LR", compute_ranks(lr),
                         "LR (canonical supervised) — coverage gap vs full policy corpus"))
        if mlp:
            cols.append(("MLP", compute_ranks(mlp),
                         "MLP (4-layer/384-hidden) — coverage gap vs full policy corpus"))
        if zs:
            cols.append(("ZS", compute_ranks(zs),
                         "Zero-shot nearest-centroid — coverage gap vs full policy corpus (canonical encoder only)"))
        if cols:
            enc_subgroups.append((sublabel, cols))
    if not enc_subgroups:
        print("WARNING: no encoder data for encoder-axis coverage table, skipping")
        return
    col_groups = [("Encoder (embedding architecture)", enc_subgroups)]
    rho_by_col = assemble_table(
        col_groups, OUT_MAIN / "tab9_encoder_sensitivity_coverage.tex",
        ENC_AXIS_COVERAGE_CAPTION, ENC_AXIS_COVERAGE_NOTES, "tab:encoder-sensitivity-coverage",
    )
    scibert_key = "Encoder (embedding architecture)::SciBERT (768-d)::LR"
    scibert_rho = rho_by_col.get(scibert_key, float("nan"))
    val = f"{scibert_rho:.2f}" if not np.isnan(scibert_rho) else "--"
    minilm_key = "Encoder (embedding architecture)::MiniLM (384-d)::LR"
    minilm_rho = rho_by_col.get(minilm_key, float("nan"))
    minilm_val = f"{minilm_rho:.2f}" if not np.isnan(minilm_rho) else "--"
    (OUT_MAIN / "num9_encoder_sensitivity_coverage.tex").write_text(
        "\n".join([
            "% Auto-generated by 1_code/7_main_analysis/1_main_text/3_generate_cross_sensitivity_table.py — do not edit manually",
            rf"\newcommand{{\SciBERTCoverageRho}}{{{val}}}",
            rf"\newcommand{{\MiniLMCoverageRho}}{{{minilm_val}}}",
        ]) + "\n", encoding="utf-8")
    print(f"Written num9_encoder_sensitivity_coverage.tex  scibert_lr_rho={val}  minilm_lr_rho={minilm_val}")


def load_lr_covgaps(m):
    p = output_dir_for_model(m, root=root) / "data" / "coverage_document_weighted.json"
    if not p.exists():
        return None
    with open(p) as f:
        data = json.load(f)
    cg = data.get("coverage_gap_hard")
    if not cg:
        return None
    return {int(k[3:]): v for k, v in cg.items()}


def load_mlp_covgaps(m):
    """Document-weighted coverage gap for MLP assignment.

    Research counts from mlp_summary.json (already doc-level: 1 paper = 1 doc).
    Policy profile uses the shared A19 document-weighting (single source of truth),
    preferring the persisted coverage_document_weighted_mlp.json when available.

    Returns {sdg: |res% - pol%|} or None if data missing.
    """
    cached = load_route_coverage_gap(m, "mlp")
    if cached is not None:
        return cached

    scored_dir = scored_dir_for_model(m)
    summary_path = scored_dir / "mlp_scores" / "mlp_summary.json"
    if not summary_path.exists():
        return None
    with open(summary_path) as f:
        summary = json.load(f)
    res_counts = {int(k): v for k, v in summary["research_coverage"].items()}
    res_total = summary["research_total"]

    scores_path = scored_dir / "mlp_scores" / "mlp_policy_scores.npy"
    ids_path = scored_dir / "metadata" / "policy_scores_ids.json"
    if not scores_path.exists() or not ids_path.exists():
        return None
    policy_scores = np.load(scores_path)
    with open(ids_path) as f:
        policy_ids = json.load(f)

    pol_profile, _, _ = document_weighted_policy_profile(policy_scores, policy_ids)

    gaps = {}
    for i in range(N_SDG):
        sdg = i + 1
        res_share = res_counts.get(sdg, 0) / res_total
        gaps[sdg] = float(abs(res_share - pol_profile[i]))
    return gaps


def load_zs_covgaps(m, concept: bool = False):
    """Document-weighted coverage gap for zero-shot nearest-centroid assignment.

    Research counts from ZS semantic_gap_distances_zeroshot.json (paper-weighted
    per-SDG n_papers, matching the LR/MLP routes; the producer collapses each
    shard to one unit per abstract). Policy profile uses the shared A19
    document-weighting (single source of truth), preferring the persisted
    coverage_document_weighted_zeroshot.json when available.

    When `concept=True`, loads the concept-retrieved variant
    (data/concept/semantic_gap_distances_zeroshot.json + zeroshot_concept/research_centroids.npy)
    instead of the canonical keyword route.

    Returns {sdg: |res% - pol%|} or None if data missing.
    """
    cached = load_route_coverage_gap(m, "zs")
    if cached is not None and not concept:
        return cached

    gap_path = (
        output_dir_for_model(m, root=root) / "data" / "concept" / "semantic_gap_distances_zeroshot.json"
        if concept
        else output_dir_for_model(m, root=root) / "data" / "semantic_gap_distances_zeroshot.json"
    )
    if not gap_path.exists():
        return None
    with open(gap_path) as f:
        data = json.load(f)
    res_counts = {r["sdg"]: r["n_papers"] for r in data["per_sdg"]}
    res_total = sum(res_counts.values())

    embed_dir = embed_dir_for_model(m)
    emb_path = embed_dir / "policy.npy"
    ids_path = embed_dir / "metadata" / "policy_ids.json"
    centroids_path = (
        scored_dir_for_model(m) / "zeroshot_concept" / "research_centroids.npy"
        if concept
        else scored_dir_for_model(m) / "sdg_centroids.npy"
    )
    if not (emb_path.exists() and ids_path.exists() and centroids_path.exists()):
        return None

    policy_emb = np.load(emb_path).astype(np.float32)
    with open(ids_path) as f:
        policy_ids = json.load(f)
    centroids = np.load(centroids_path).astype(np.float32)

    policy_scores = policy_emb @ centroids.T

    pol_profile, _, _ = document_weighted_policy_profile(policy_scores, policy_ids)

    gaps = {}
    for i in range(N_SDG):
        sdg = i + 1
        res_share = res_counts.get(sdg, 0) / res_total
        gaps[sdg] = float(abs(res_share - pol_profile[i]))
    return gaps


def load_concept_mlp_covgaps(m):
    scored_dir = scored_dir_for_model(m)
    summary_path = scored_dir / "mlp_scores_concept" / "mlp_summary.json"
    if not summary_path.exists():
        return None
    with open(summary_path) as f:
        summary = json.load(f)
    res_counts = {int(k): v for k, v in summary["research_coverage"].items()}
    res_total = summary["research_total"]

    scores_path = scored_dir / "mlp_scores_concept" / "mlp_policy_scores.npy"
    ids_path = scored_dir / "metadata" / "policy_scores_ids.json"
    if not scores_path.exists() or not ids_path.exists():
        return None
    policy_scores = np.load(scores_path)
    with open(ids_path) as f:
        policy_ids = json.load(f)

    pol_profile, _, _ = document_weighted_policy_profile(policy_scores, policy_ids)

    gaps = {}
    for i in range(N_SDG):
        sdg = i + 1
        res_share = res_counts.get(sdg, 0) / res_total
        gaps[sdg] = float(abs(res_share - pol_profile[i]))
    return gaps


def load_concept_zs_covgaps(m):
    """Concept-retrieved zero-shot coverage gap (mirrors load_concept_mlp_covgaps).

    Unused by current tables but kept available so a concept-ZS coverage column
    resolves the correct concept centroids rather than the keyword sdg_centroids.npy.
    """
    return load_zs_covgaps(m, concept=True)


def load_concept_lr_gaps(m):
    p = output_dir_for_model(m, root=root) / "data" / "concept" / "semantic_gap_distances_lr.json"
    if not p.exists():
        return None
    with open(p) as f:
        data = json.load(f)
    return {row["sdg"]: row["semantic_gap"] for row in data["per_sdg"] if row.get("semantic_gap") is not None}


def load_concept_covgaps(m):
    p = output_dir_for_model(m, root=root) / "data" / "concept" / "coverage_document_weighted.json"
    if not p.exists():
        return None
    with open(p) as f:
        data = json.load(f)
    cg = data.get("coverage_gap_hard")
    if not cg:
        return None
    return {int(k[3:]): v for k, v in cg.items()}


def load_canonical_research_profile():
    p = output_dir_for_model(model, root=root) / "data" / "coverage_document_weighted.json"
    with open(p) as f:
        data = json.load(f)
    return {int(k[3:]): v for k, v in data["research_profile_hard"].items()}


def parse_policy_source_covgaps(research_profile=None):
    if not POLICY_SOURCE_FAMILY_TEX.exists():
        return {}
    if research_profile is None:
        research_profile = load_canonical_research_profile()
    research = research_profile
    text = POLICY_SOURCE_FAMILY_TEX.read_text(encoding="utf-8")
    fam_share = {"full": {}, "curated": {}, "sdgi": {}, "ungdc": {}}
    in_header = True
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("%") or line.startswith(r"\toprule") or line.startswith(r"\midrule") or line.startswith(r"\bottomrule") or line.startswith(r"\end") or line.startswith(r"\cmidrule"):
            continue
        if in_header and "SDG" in line and "&" in line:
            in_header = False
            continue
        if in_header:
            continue
        m = re.match(r"(\d+)", line)
        if not m:
            continue
        sdg = int(m.group(1))
        parts = [p.strip() for p in line.rstrip("\\").split("&")]
        # cov.(n) cells at indices 1, 3, 5, 7 — cell format "2.0 (1,646)"
        for label, gi in (("full", 1), ("curated", 3), ("sdgi", 5), ("ungdc", 7)):
            if gi < len(parts):
                try:
                    fam_share[label][sdg] = float(parts[gi].split()[0]) / 100.0
                except ValueError:
                    pass
    out = {}
    for label in ("curated", "sdgi", "ungdc"):
        out[label] = {sdg: abs(research[sdg] - fam_share[label][sdg]) for sdg in fam_share[label]}
    return out


def _kendall(x, y):
    """Scipy-free Kendall tau (concordant - discordant) / sqrt((n0-nx)(n0-ny))."""
    n = len(x)
    if n < 2:
        return float("nan")
    concordant = discordant = 0
    ties_x = ties_y = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            if dx == 0 and dy == 0:
                ties_x += 1
                ties_y += 1
            elif dx == 0:
                ties_x += 1
            elif dy == 0:
                ties_y += 1
            elif (dx > 0) == (dy > 0):
                concordant += 1
            else:
                discordant += 1
    denom = ((n * (n - 1) / 2 - ties_x) * (n * (n - 1) / 2 - ties_y)) ** 0.5
    if denom == 0:
        return float("nan")
    return (concordant - discordant) / denom


# ---------------------------------------------------------------------------
# Rank computation (1 = largest gap)
# ---------------------------------------------------------------------------
def compute_ranks(gap_dict):
    items = [(sdg, gap) for sdg, gap in gap_dict.items()]
    items.sort(key=lambda x: x[1], reverse=True)
    return {sdg: rank + 1 for rank, (sdg, _) in enumerate(items)}

def macro_f1(values_dict):
    vals = list(values_dict.values())
    return sum(vals) / len(vals) if vals else 0.0

# ---------------------------------------------------------------------------
# Write num1_classifier_performance.tex (LR F1 macros)
# ---------------------------------------------------------------------------
def write_num_validation():
    lines = [
        f"% Auto-generated by 1_code/7_main_analysis/1_main_text/3_generate_cross_sensitivity_table.py — do not edit manually",
        rf"\newcommand{{\MacroFOne}}{{{lr_macro:.3f}}}",
        rf"\newcommand{{\ValidationAccuracy}}{{{retrain['test_results']['micro_f1']:.4f}}}",
        rf"\newcommand{{\RandomBaselineSeventeenClass}}{{{1/17:.3f}}}",
    ]
    for sdg in range(1, N_SDG + 1):
        f1 = lr_per_sdg[sdg]
        name = {1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five", 6: "Six", 7: "Seven",
                8: "Eight", 9: "Nine", 10: "Ten", 11: "Eleven", 12: "Twelve", 13: "Thirteen",
                14: "Fourteen", 15: "Fifteen", 16: "Sixteen", 17: "Seventeen"}[sdg]
        lines.append(rf"\newcommand{{\FiSdg{name}}}{{{f1:.3f}}}")
    # MLP validation macro (used in cross-sensitivity / Appendix D)
    MLP_RETRAIN_PATH = model_results_dir_for_model(model) / "model" / "mlp_retrain_results.json"
    if MLP_RETRAIN_PATH.exists():
        with open(MLP_RETRAIN_PATH) as f:
            mlp_data = json.load(f)
        mlp_macro = mlp_data["test_results"]["macro_f1"]
        lines.append(rf"\newcommand{{\MlpMacroFOne}}{{{mlp_macro:.3f}}}")
    path = OUT_MAIN / "num1_classifier_performance.tex"
    path.write_text("\n".join(lines) + "\n")
    print(f"Written {path}")

# ---------------------------------------------------------------------------
# Write tab1_classifier_performance.tex (single LR F1 column)
# ---------------------------------------------------------------------------
def write_validation_table():
    lines = [
        "% Auto-generated by 1_code/7_main_analysis/1_main_text/3_generate_cross_sensitivity_table.py — do not edit manually",
        r"\begin{tabular}{lc}",
        r"\toprule",
        r"SDG & LR test F1 \\",
        r"\midrule",
    ]
    for sdg in range(1, N_SDG + 1):
        f1 = lr_per_sdg[sdg]
        lines.append(f"SDG {sdg} ({SDG_NAMES[sdg]}) & {f1:.3f} \\\\")
    lines.append(r"\midrule")
    lines.append(f"Macro-F1 (SDGs 1--17) & \\textbf{{{lr_macro:.3f}}} \\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path = OUT_MAIN / "tab1_classifier_performance.tex"
    path.write_text("\n".join(lines) + "\n")
    print(f"Written {path}")

# ---------------------------------------------------------------------------
# Write tab6_cross_sensitivity.tex
# ---------------------------------------------------------------------------
def write_cross_sensitivity():
    cap_20, cap_none = load_cap_gaps()
    policy_families = parse_policy_source_gaps()
    canon_gaps = load_lr_gaps("all-mpnet-base-v2")

    # --- Adjusted (canonical) group: register-removed rankings ------------
    adj_cap_20, adj_cap_none = load_cap_gaps_adj()
    adj_families = parse_policy_source_gaps_adj()
    adj_canon = load_lr_gaps_adj("all-mpnet-base-v2")
    adj_concept_lr = load_concept_lr_gaps_adj(model)
    adj_concept_mlp = load_concept_mlp_gaps_adj(model)
    adj_group = []
    if adj_canon:
        adj_group.append(("", [("Canon", compute_ranks(adj_canon),
                               "Canonical MPNet-LR ranking (register-removed, adjusted)")]))
    adj_pcols = []
    family_labels = {"curated": "Curated", "sdgi": "SDGi", "ungdc": "UNGDC", "full": "Full"}
    for key, label in family_labels.items():
        if key in adj_families:
            adj_pcols.append((label, compute_ranks(adj_families[key]),
                             f"Policy source: {label} (register-removed)"))
    if adj_pcols:
        adj_group.append(("Policy source", adj_pcols))
    adj_cap_cols = []
    if adj_cap_20:
        adj_cap_cols.append(("20", compute_ranks(adj_cap_20), "Segment cap 20 (register-removed)"))
    if adj_cap_none:
        adj_cap_cols.append(("None", compute_ranks(adj_cap_none), "No segment cap (register-removed)"))
    if adj_cap_cols:
        adj_group.append(("Segment cap", adj_cap_cols))
    adj_concept_cols = []
    if adj_concept_lr:
        adj_concept_cols.append(("LR", compute_ranks(adj_concept_lr),
                                 "LR centroids (concept retrieval, register-removed)"))
    if adj_concept_mlp:
        adj_concept_cols.append(("MLP", compute_ranks(adj_concept_mlp),
                                 "MLP centroids (concept retrieval, register-removed)"))
    if adj_concept_cols:
        adj_group.append(("Retrieval", adj_concept_cols))

    # --- Raw (naive baseline) group --------------------------------------
    # Mirrors the adjusted group's structure so the two panels share an
    # identical column layout (Canon / Policy source / Segment cap / Retrieval).
    raw_group = []
    if canon_gaps:
        raw_group.append(("", [("Canon", compute_ranks(canon_gaps),
                                 "Canonical MPNet-LR ranking (raw, naive baseline)")]))
    pcols = []
    for key, label in family_labels.items():
        if key in policy_families:
            pcols.append((label, compute_ranks(policy_families[key]), f"Policy source: {label} (raw)"))
    if pcols:
        raw_group.append(("Policy source", pcols))

    cap_cols = []
    if cap_20:
        cap_cols.append(("20", compute_ranks(cap_20), "Segment cap 20 (raw)"))
    if cap_none:
        cap_cols.append(("None", compute_ranks(cap_none), "No segment cap (raw)"))
    if cap_cols:
        raw_group.append(("Segment cap", cap_cols))

    concept_lr = load_concept_lr_gaps(model)
    concept_mlp = load_concept_mlp_gaps(model)
    concept_sub_cols = []
    if concept_lr:
        concept_sub_cols.append(("LR", compute_ranks(concept_lr),
                                 "LR centroids (concept-based AI/ML retrieval, raw)"))
    if concept_mlp:
        concept_sub_cols.append(("MLP", compute_ranks(concept_mlp),
                                 "MLP centroids (concept-based AI/ML retrieval, raw)"))
    if concept_sub_cols:
        raw_group.append(("Retrieval", concept_sub_cols))

    if not adj_group and not raw_group:
        print("WARNING: no data available for cross-sensitivity table, skipping")
        return

    rho_a, rho_b = assemble_paneled(
        adj_groups=adj_group, raw_groups=raw_group,
        out_path=OUT_MAIN / "tab6_cross_sensitivity.tex",
        caption=SEMANTIC_CAPTION, notes=SEMANTIC_NOTES, label="tab:cross-sensitivity-robustness",
        panel_a_note="Adjusted (register-removed, canonical).",
        panel_b_note="Raw (naive baseline, un-adjusted).",
    )

    concept_rho_val = rho_b.get("Retrieval::LR", float("nan"))
    concept_sem_rho = f"{concept_rho_val:.2f}" if not np.isnan(concept_rho_val) else "--"
    (OUT_MAIN / "num11_concept_semantic.tex").write_text(
        "\n".join([
            "% Auto-generated by 1_code/7_main_analysis/1_main_text/3_generate_cross_sensitivity_table.py — do not edit manually",
            rf"\newcommand{{\ConceptSemanticGapRho}}{{{concept_sem_rho}}}",
        ]) + "\n",
        encoding="utf-8",
    )
    print(f"Written num11_concept_semantic.tex  concept_sem_rho={concept_sem_rho}")


def is_nested(g):
    """A col_group is nested when its body is a list of (sublabel, cols) tuples."""
    _, body = g
    return bool(body) and isinstance(body[0], tuple) and len(body[0]) == 2 \
        and isinstance(body[0][1], list)


def group_total(g):
    _, body = g
    if is_nested(g):
        return sum(len(cols) for _, cols in body)
    return len(body)


def flat_cols(g):
    _, body = g
    if is_nested(g):
        out = []
        for _, cols in body:
            out.extend(cols)
        return out
    return list(body)


def build_tabular(col_groups):
    """Build a single (resizebox-wrapped) tabular block for the given col_groups.

    Returns (tabular_lines, rho_by_col). The block includes the
    ``\\resizebox`` wrapper and the ``\\begin{tabular}``/``\\end{tabular}``
    pair but NOT the surrounding ``\\begin{table}``/``\\end{table}`` so that
    callers may compose several panels into one float.
    """
    all_cols = []
    all_col_keys = []
    for glabel, body in col_groups:
        if is_nested((glabel, body)):
            for sublabel, cols in body:
                for col_label, ranks, note in cols:
                    all_cols.append((col_label, ranks, note))
                    all_col_keys.append(f"{glabel}::{sublabel}::{col_label}")
        else:
            for col_label, ranks, note in body:
                all_cols.append((col_label, ranks, note))
                all_col_keys.append(f"{glabel}::{col_label}")
    n_cols = sum(group_total(g) for g in col_groups)
    has_nested = any(is_nested(g) for g in col_groups)

    n_header_rows = 2 if not has_nested else 3
    rowA = [fr"\multirow{{{n_header_rows}}}{{*}}{{SDG}}"]
    midrules = []
    col_idx = 2
    for glabel, body in col_groups:
        total = group_total((glabel, body))
        if total == 0:
            continue
        rowA.append(r"\multicolumn{" + str(total) + r"}{c}{" + glabel + "}")
        if is_nested((glabel, body)):
            s = col_idx
            for _, cols in body:
                n = len(cols)
                midrules.append((s, s + n - 1))
                s += n
        elif total > 1:
            midrules.append((col_idx, col_idx + total - 1))
        col_idx += total

    rowB = [""]
    for glabel, body in col_groups:
        if is_nested((glabel, body)):
            for sublabel, cols in body:
                rowB.append(r"\multicolumn{" + str(len(cols)) + r"}{c}{" + sublabel + "}")
        else:
            for col_label, _, _ in body:
                rowB.append(col_label)

    if has_nested:
        rowC = [""]
        for glabel, body in col_groups:
            if is_nested((glabel, body)):
                for _, cols in body:
                    for col_label, _, _ in cols:
                        rowC.append(col_label)
    else:
        rowC = None

    tex = [
        r"\resizebox{\textwidth}{!}{%",
        rf"\begin{{tabular}}{{l{'c' * n_cols}}}",
        r"\toprule",
    ]
    tex.append(" & ".join(rowA) + r" \\")
    for s, e in midrules:
        tex.append(r"\cmidrule(lr){" + f"{s}-{e}" + "}")
    tex.append(" & ".join(rowB) + r" \\")
    if rowC is not None:
        tex.append(r"\cmidrule(lr){2-" + str(n_cols + 1) + "}")
        tex.append(" & ".join(rowC) + r" \\")
    tex.append(r"\midrule")

    all_sdgs = set()
    for _, ranks, _ in all_cols:
        all_sdgs.update(ranks.keys())
    all_sdgs = sorted(all_sdgs)

    STABLE_RANK_DELTA = 1
    SENSITIVE_RANK_DELTA = 4
    mpnet_lr = minilm_lr = None
    for glabel, body in col_groups:
        if glabel.startswith("Encoder"):
            for sublabel, cols in body:
                for label, ranks, _ in cols:
                    if label == "LR" and "mpnet" in sublabel.lower():
                        mpnet_lr = ranks
                    if label == "LR" and "minilm" in sublabel.lower():
                        minilm_lr = ranks

    def _highlight(sdg):
        if mpnet_lr is None or minilm_lr is None:
            return ""
        a = mpnet_lr.get(sdg)
        b = minilm_lr.get(sdg)
        if a is None or b is None:
            return ""
        d = abs(a - b)
        if d <= STABLE_RANK_DELTA:
            return "b"
        if d >= SENSITIVE_RANK_DELTA:
            return "i"
        return ""

    for sdg in all_sdgs:
        hl = _highlight(sdg)
        cells = [f"SDG {sdg}"]
        for _, ranks, _ in all_cols:
            v = ranks.get(sdg, "--")
            if hl == "b":
                cells.append(r"\textbf{" + str(v) + "}")
            elif hl == "i":
                cells.append(r"\textit{" + str(v) + "}")
            else:
                cells.append(str(v))
        tex.append(" & ".join(cells) + r" \\")

    baseline = all_cols[0][1] if all_cols else None
    rho_cells = [r"Rank Corr ($\rho$)"]
    rho_by_col = {}
    if baseline:
        common = [s for s in all_sdgs if s in baseline]
        bv = [baseline[s] for s in common]
        for (col_label, ranks, _), key in zip(all_cols, all_col_keys):
            cv = [ranks[s] for s in common if s in ranks]
            if len(cv) >= 2 and len(cv) == len(bv):
                rho = _spearman(bv, cv)
            else:
                rho = float("nan")
            rho_by_col[key] = rho
            rho_cells.append(f"{rho:.2f}" if not np.isnan(rho) else "--")
    else:
        for _ in all_cols:
            rho_cells.append("--")
    tex.append(r"\midrule")
    tex.append(" & ".join(rho_cells) + r" \\")

    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")
    tex.append(r"}")
    return tex, rho_by_col


def assemble_table(col_groups, out_path, caption, notes, label):
    tabular, rho_by_col = build_tabular(col_groups)
    tex = [
        "% Auto-generated by 1_code/7_main_analysis/1_main_text/3_generate_cross_sensitivity_table.py — do not edit manually",
        r"\begin{table}[ht]",
        r"\centering",
        r"\footnotesize",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
    ]
    tex.extend(tabular)
    tex.append(r"\par\smallskip\footnotesize\emph{Notes:} " + notes + r"\par")
    tex.append(r"\end{table}")
    out_path.write_text("\n".join(tex) + "\n")
    print(f"Written {out_path}  columns={sum(group_total(g) for g in col_groups)}")
    return rho_by_col


def assemble_paneled(adj_groups, raw_groups, out_path, caption, notes, label,
                    panel_a_note, panel_b_note):
    """Write one table split into two panels: (a) adjusted, (b) raw.

    ``adj_groups`` / ``raw_groups`` are col_groups lists in the same format
    accepted by :func:`build_tabular`. Each panel keeps the canonical
    (Encoder -> Method or Policy source -> Segment cap -> Retrieval) nesting;
    the raw/adjusted split is promoted to the panel level so the two
    robustness tables share one presentation order.
    """
    tex = [
        "% Auto-generated by 1_code/7_main_analysis/1_main_text/3_generate_cross_sensitivity_table.py — do not edit manually",
        r"\begin{table}[ht]",
        r"\centering",
        r"\footnotesize",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
    ]
    tab_a, rho_a = build_tabular(adj_groups)
    tex.extend(tab_a)
    tex.append(r"\par\smallskip\footnotesize\emph{Panel (a):} " + panel_a_note + r"\par")
    tab_b, rho_b = build_tabular(raw_groups)
    tex.extend(tab_b)
    tex.append(r"\par\smallskip\footnotesize\emph{Panel (b):} " + panel_b_note + r"\par")
    tex.append(r"\par\smallskip\footnotesize\emph{Notes:} " + notes + r"\par")
    tex.append(r"\end{table}")
    out_path.write_text("\n".join(tex) + "\n")
    n_a = sum(group_total(g) for g in adj_groups)
    n_b = sum(group_total(g) for g in raw_groups)
    print(f"Written {out_path}  adj_columns={n_a} raw_columns={n_b}")
    return rho_a, rho_b


def write_coverage_table():
    # Load the canonical coverage JSON once and reuse for both the Canon
    # column (coverage_gap_hard) and the policy-source profile (research_profile_hard).
    canon_cov_path = output_dir_for_model(model, root=root) / "data" / "coverage_document_weighted.json"
    with open(canon_cov_path) as f:
        _canon = json.load(f)
    mpnet_lr = {int(k[3:]): v for k, v in _canon["coverage_gap_hard"].items()} if _canon.get("coverage_gap_hard") else None
    research_profile = {int(k[3:]): v for k, v in _canon["research_profile_hard"].items()}

    # Coverage gap is segment-cap-independent: no Segment-cap group.
    policy_families = parse_policy_source_covgaps(research_profile=research_profile)

    col_groups = []
    if mpnet_lr:
        col_groups.append(("", [("Canon", compute_ranks(mpnet_lr),
                                 "Canonical MPNet-LR coverage ranking")]))

    pcols = []
    family_labels = {"curated": "Curated", "sdgi": "SDGi", "ungdc": "UNGDC"}
    for key, label in family_labels.items():
        if key in policy_families:
            pcols.append((label, compute_ranks(policy_families[key]), f"Policy source: {label}"))
    if pcols:
        col_groups.append(("Policy source", pcols))

    concept_lr = load_concept_covgaps(model)
    concept_mlp = load_concept_mlp_covgaps(model)
    concept_sub_cols = []
    if concept_lr:
        concept_sub_cols.append(("LR", compute_ranks(concept_lr),
                                 "LR centroid assignment (concept-based AI/ML retrieval)"))
    if concept_mlp:
        concept_sub_cols.append(("MLP", compute_ranks(concept_mlp),
                                 "MLP centroid assignment (concept-based AI/ML retrieval)"))
    if concept_sub_cols:
        col_groups.append(("Retrieval", concept_sub_cols))

    if not col_groups:
        print("WARNING: no data available for coverage cross-sensitivity table, skipping")
        return

    rho_by_col = assemble_table(col_groups, OUT_MAIN / "tab8_coverage_sensitivity.tex",
                                COVERAGE_CAPTION, COVERAGE_NOTES, "tab:cross-sensitivity-coverage")

    concept_rho_val = rho_by_col.get("Retrieval::LR", float("nan"))
    concept_rho = f"{concept_rho_val:.2f}" if not np.isnan(concept_rho_val) else "--"
    lines = [
        f"% Auto-generated by 1_code/7_main_analysis/1_main_text/3_generate_cross_sensitivity_table.py — do not edit manually",
        rf"\newcommand{{\ConceptCoverageGapRho}}{{{concept_rho}}}",
    ]
    (OUT_MAIN / "num8_coverage_sensitivity.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Written num8_coverage_sensitivity.tex  concept_rho={concept_rho}")


def write_concept_coverage():
    canon_p = output_dir_for_model(model, root=root) / "data" / "coverage_document_weighted.json"
    concept_p = output_dir_for_model(model, root=root) / "data" / "concept" / "coverage_document_weighted.json"
    if not (canon_p.exists() and concept_p.exists()):
        print("WARNING: concept coverage json missing, skipping concept coverage table")
        return
    with open(canon_p) as f:
        canon = json.load(f)
    with open(concept_p) as f:
        concept = json.load(f)
    cr = {int(k[3:]): v for k, v in canon["research_profile_hard"].items()}
    cc = {int(k[3:]): v for k, v in concept["research_profile_hard"].items()}

    sdgs = list(range(1, N_SDG + 1))
    bv = [cr[s] for s in sdgs]
    cv = [cc[s] for s in sdgs]
    rho = _spearman(bv, cv)
    tau = _kendall(bv, cv)
    max_abs_delta = max(abs(cr[s] - cc[s]) for s in sdgs)

    tab_lines = [
        "% Auto-generated by 1_code/7_main_analysis/1_main_text/3_generate_cross_sensitivity_table.py — do not edit manually",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"SDG & Keyword \% & Concept \% & $\Delta$ \\",
        r"\midrule",
    ]
    for s in sdgs:
        delta = cc[s] - cr[s]
        tab_lines.append(f"SDG {s:2d} & {cr[s]*100:.1f} & {cc[s]*100:.1f} & {delta*100:+.1f} \\\\")
    tab_lines.extend([
        r"\midrule",
        rf"\multicolumn{{4}}{{l}}{{Spearman $\rho$ = {rho:.3f}; Kendall $\tau$ = {tau:.3f}}} \\",
        r"\bottomrule",
        r"\end{tabular}",
    ])
    (OUT_MAIN / "tab10_concept_coverage.tex").write_text("\n".join(tab_lines) + "\n", encoding="utf-8")
    print(f"Written tab10_concept_coverage.tex  rho={rho:.3f} tau={tau:.3f}")

    num_lines = [
        "% Auto-generated by 1_code/7_main_analysis/1_main_text/3_generate_cross_sensitivity_table.py — do not edit manually",
        rf"\newcommand{{\ConceptCoverageSpearman}}{{{rho:.3f}}}",
        rf"\newcommand{{\ConceptCoverageKendall}}{{{tau:.3f}}}",
        rf"\newcommand{{\ConceptCoverageMaxAbsDelta}}{{{max_abs_delta*100:.1f}}}",
    ]
    (OUT_MAIN / "num10_concept_coverage.tex").write_text("\n".join(num_lines) + "\n", encoding="utf-8")
    print(f"Written num10_concept_coverage.tex")

# ---------------------------------------------------------------------------
# Write num6_cross_sensitivity.tex
# ---------------------------------------------------------------------------
def write_num_cross_sensitivity():
    cap_20, cap_none = load_cap_gaps()
    rho_val = "--"
    if cap_20 and cap_none:
        vec_20 = [cap_20.get(sdg) for sdg in range(1, N_SDG + 1)]
        vec_none = [cap_none.get(sdg) for sdg in range(1, N_SDG + 1)]
        if all(v is not None for v in vec_20) and all(v is not None for v in vec_none):
            r = np.corrcoef(vec_20, vec_none)[0, 1]
            rho_val = f"{r:.3f}"
    lines = [
        f"% Auto-generated by 1_code/7_main_analysis/1_main_text/3_generate_cross_sensitivity_table.py — do not edit manually",
        rf"\newcommand{{\CapStabilityRho}}{{{rho_val}}}",
    ]
    path = OUT_MAIN / "num6_cross_sensitivity.tex"
    path.write_text("\n".join(lines) + "\n")
    print(f"Written {path}  rho={rho_val}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run(args: argparse.Namespace) -> None:
    global model, root, OUT_MAIN, RETRAIN_JSON, retrain, lr_per_sdg, lr_macro
    global LR_GAP_PATH, ZS_GAP_PATH, CAP_PATH, POLICY_SOURCE_FAMILY_TEX

    model = args.embed_model

    root = Path(args.output_dir)
    OUT_MAIN = output_dir_for_model(model, root=root) / "tables"
    OUT_MAIN.mkdir(parents=True, exist_ok=True)

    # 1. Load LR test F1 from retrain results
    RETRAIN_JSON = model_results_dir_for_model(model) / "model" / "sdg_retrain_results.json"
    with open(RETRAIN_JSON) as f:
        retrain = json.load(f)
    lr_per_sdg = {}
    for k, v in retrain["test_results"]["per_sdg_f1"].items():
        sdg_num = int(k.split("_")[1])
        lr_per_sdg[sdg_num] = v
    lr_macro = retrain["test_results"]["macro_f1"]

    SCRIPT_VERSION = "2"
    PRIMARY = OUT_MAIN / "tab6_cross_sensitivity.tex"
    OUTPUTS = [
        OUT_MAIN / "num1_classifier_performance.tex",
        OUT_MAIN / "tab1_classifier_performance.tex",
        PRIMARY,
        OUT_MAIN / "tab8_coverage_sensitivity.tex",
        OUT_MAIN / "tab10_concept_coverage.tex",
        OUT_MAIN / "num6_cross_sensitivity.tex",
        OUT_MAIN / "num11_concept_semantic.tex",
        OUT_MAIN / "num10_concept_coverage.tex",
        OUT_MAIN / "num8_coverage_sensitivity.tex",
        OUT_MAIN / "num7_encoder_sensitivity.tex",
        OUT_MAIN / "num9_encoder_sensitivity_coverage.tex",
    ]
    fp_paths = [RETRAIN_JSON]
    for m, _ in ENC_AXIS_ENCODERS:
        fp_paths += [
            output_dir_for_model(m, root=root) / "data" / "semantic_gap_distances_lr.json",
            output_dir_for_model(m, root=root) / "data" / "semantic_gap_robustness_caps_lr.json",
            output_dir_for_model(m, root=root) / "data" / "coverage_document_weighted.json",
        ]
    # Zero-shot gaps are a direct input to the ZS column (canonical encoder only).
    fp_paths.append(
        output_dir_for_model(DEFAULT_EMBED_MODEL, root=root) / "data" / "semantic_gap_distances_zeroshot.json"
    )
    # MLP capped raw gaps (per encoder) + concept (MPNet only) — single source.
    for m, _ in ENC_AXIS_ENCODERS:
        fp_paths.append(
            output_dir_for_model(m, root=root) / "data" / "semantic_gap_distances_mlp.json"
        )
    fp_paths.append(
        output_dir_for_model(DEFAULT_EMBED_MODEL, root=root) / "data" / "concept" / "semantic_gap_distances_mlp.json"
    )
    # Concept LR raw gaps also feed \ConceptSemanticGapRho (load_concept_lr_gaps) —
    # include so the table re-derives when they change.
    fp_paths.append(
        output_dir_for_model(DEFAULT_EMBED_MODEL, root=root) / "data" / "concept" / "semantic_gap_distances_lr.json"
    )
    fp = fingerprint_of(*fp_paths) + SCRIPT_VERSION
    if should_skip(OUTPUTS, fp, args.overwrite, PRIMARY):
        print(f"Skipping {PRIMARY} \u2014 inputs unchanged")
        return

    LR_GAP_PATH = output_dir_for_model(model, root=root) / "data" / "semantic_gap_distances_lr.json"
    ZS_GAP_PATH = output_dir_for_model(model, root=root) / "data" / "semantic_gap_distances_zeroshot.json"
    CAP_PATH = output_dir_for_model(model, root=root) / "data" / "semantic_gap_robustness_caps_lr.json"
    POLICY_SOURCE_FAMILY_TEX = root / "appendix" / model_slug(model) / "a2_source_family_sensitivity" / "tables" / "tab_a2_policy_source_family_combined.tex"

    write_num_validation()
    write_validation_table()
    write_cross_sensitivity()
    write_coverage_table()
    write_concept_coverage()
    write_num_cross_sensitivity()
    write_encoder_axis_semantic()
    write_encoder_axis_coverage()
    print("Cross-sensitivity table generation complete.")
    record_fingerprint(OUTPUTS, fp, PRIMARY)


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()

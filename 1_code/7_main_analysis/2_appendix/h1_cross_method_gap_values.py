"""
Generate appendix table for cross-method coverage gap and semantic gap raw values.

Outputs:
  tab_app_cross_method_gap_values.tex   — combined table (incl. concept retrieval)
"""

from __future__ import annotations

import argparse
import json
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

from model_utils import DEFAULT_EMBED_MODEL, DEFAULT_OUTPUT_ROOT, N_SDG, embed_dir_for_model, model_slug, output_dir_for_model, scored_dir_for_model, resolve_model_alias
from shared_utils import fingerprint_of, should_skip, record_fingerprint
from semantic_gap_shared import document_weighted_policy_profile, load_route_coverage_gap

# ---------------------------------------------------------------------------
# Coverage gap loaders
# ---------------------------------------------------------------------------

def _lr_covgaps(root, m):
    p = output_dir_for_model(m, root=root) / "data" / "coverage_document_weighted.json"
    if not p.exists():
        return None
    with open(p) as f:
        data = json.load(f)
    cg = data.get("coverage_gap_hard")
    if not cg:
        return None
    return {int(k[3:]): v for k, v in cg.items()}


def _mlp_covgaps(m):
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


def _zs_covgaps(root, m, concept: bool = False):
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

# ---------------------------------------------------------------------------
# Semantic gap loaders
# ---------------------------------------------------------------------------

def _lr_gaps(root, m):
    p = output_dir_for_model(m, root=root) / "data" / "semantic_gap_distances_lr.json"
    if not p.exists():
        return None
    with open(p) as f:
        data = json.load(f)
    return {row["sdg"]: row["semantic_gap"] for row in data["per_sdg"] if row["semantic_gap"] is not None}


def _mlp_gaps(root, m):
    # Capped, single-source MLP gap (mirrors _lr_gaps). Replaces the uncapped
    # mlp_summary.json["semantic_gaps"] value, which was divergent.
    p = output_dir_for_model(m, root=root) / "data" / "semantic_gap_distances_mlp.json"
    if not p.exists():
        return None
    with open(p) as f:
        data = json.load(f)
    return {row["sdg"]: row["semantic_gap"] for row in data["per_sdg"] if row["semantic_gap"] is not None}


def _zs_gaps(root, m):
    p = output_dir_for_model(m, root=root) / "data" / "semantic_gap_distances_zeroshot.json"
    if not p.exists():
        return None
    with open(p) as f:
        data = json.load(f)
    return {row["sdg"]: row["semantic_gap"] for row in data["per_sdg"] if row["semantic_gap"] is not None}


def _concept_covgaps(root, m):
    p = output_dir_for_model(m, root=root) / "data" / "concept" / "coverage_document_weighted.json"
    if not p.exists():
        return None
    with open(p) as f:
        data = json.load(f)
    cg = data.get("coverage_gap_hard")
    if not cg:
        return None
    return {int(k[3:]): v for k, v in cg.items()}


def _concept_gaps(root, m):
    p = output_dir_for_model(m, root=root) / "data" / "concept" / "semantic_gap_distances_lr.json"
    if not p.exists():
        return None
    with open(p) as f:
        data = json.load(f)
    return {row["sdg"]: row["semantic_gap"] for row in data["per_sdg"] if row["semantic_gap"] is not None}


def _concept_zs_covgaps(root, m):
    """Concept-retrieved zero-shot coverage gap (mirrors _concept_gaps).

    Unused by current tables but kept available so a concept-ZS coverage column
    resolves the correct concept centroids rather than the keyword sdg_centroids.npy.
    """
    return _zs_covgaps(root, m, concept=True)


# ---------------------------------------------------------------------------
# Write table helpers
# ---------------------------------------------------------------------------

ENCODERS = [
    ("MPNet", "all-mpnet-base-v2", "768d"),
    ("MiniLM", "all-MiniLM-L6-v2", "384d"),
    ("SciBERT", "allenai/scibert_scivocab_uncased", "768d"),
]

# Zero-shot is scoped to the canonical encoder only (see AGENTS.md
# "Manuscript scope decisions"): MiniLM/SciBERT carry LR+MLP columns only.
ENC_METHODS = {
    "MPNet": ["LR", "MLP", "ZS"],
    "MiniLM": ["LR", "MLP"],
    "SciBERT": ["LR", "MLP"],
}


def _write_table(path, rows, fmt, has_concept, label):
    n_data = 1 + sum(len(methods) for _, methods in ENC_METHODS.items()) + (1 if has_concept else 0)

    lines = [
        "% Auto-generated by 1_code/7_main_analysis/2_appendix/h1_cross_method_gap_values.py",
        f"\\begin{{tabular}}{{{'l' + 'r' * (n_data - 1)}}}",
        r"\toprule",
    ]

    h1 = ["SDG", f"\\multicolumn{{{n_data - 1}}}{{c}}{{{label}}}"]
    lines.append(" & ".join(h1) + r" \\")

    h2 = [""]
    for enc_label, _, _ in ENCODERS:
        n = len(ENC_METHODS[enc_label])
        h2.append(f"\\multicolumn{{{n}}}{{c}}{{{enc_label}}}")
    if has_concept:
        h2.append("Concept")
    lines.append(" & ".join(h2) + r" \\")

    cmidrules = []
    start = 2
    for enc_label, _, _ in ENCODERS:
        n = len(ENC_METHODS[enc_label])
        cmidrules.append(f"\\cmidrule(lr){{{start}-{start + n - 1}}}")
        start += n
    lines.append(" ".join(cmidrules))

    h3 = [""]
    for enc_label, _, _ in ENCODERS:
        h3 += ENC_METHODS[enc_label]
    if has_concept:
        h3.append("")
    lines.append(" & ".join(h3) + r" \\")

    lines.append(r"\midrule")

    for entry in rows:
        if has_concept:
            sdg, vals, cc = entry
        else:
            sdg, vals = entry
            cc = None
        cells = [str(sdg)]
        for v in vals:
            cells.append(fmt.format(v) if v is not None else "--")
        if has_concept:
            cells.append(fmt.format(cc) if cc is not None else "--")
        lines.append(" & ".join(cells) + r" \\")

    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Written {path}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(args):
    model = args.embed_model
    root = Path(args.output_dir)

    out_dir = root / "appendix" / model_slug(model) / "h1_cross_method_gap_values" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    SCRIPT_VERSION = "2"
    PRIMARY = out_dir / "tab_app_cross_method_semgap.tex"
    OUTPUTS = [PRIMARY, out_dir / "tab_app_cross_method_covgap.tex"]
    fp_paths = []
    for _, m, _ in ENCODERS:
        fp_paths += [
            output_dir_for_model(m, root=root) / "data" / "coverage_document_weighted.json",
            output_dir_for_model(m, root=root) / "data" / "semantic_gap_distances_lr.json",
            output_dir_for_model(m, root=root) / "data" / "semantic_gap_distances_zeroshot.json",
            output_dir_for_model(m, root=root) / "data" / "semantic_gap_distances_mlp.json",
            # _mlp_covgaps reads the cached MLP coverage-gap payload.
            output_dir_for_model(m, root=root) / "data" / "coverage_document_weighted_mlp.json",
        ]
    # Concept-retrieval track (MPNet only) feeds the appendix cross-method table.
    concept_root = output_dir_for_model(DEFAULT_EMBED_MODEL, root=root) / "data" / "concept"
    fp_paths += [
        concept_root / "coverage_document_weighted.json",
        concept_root / "semantic_gap_distances_lr.json",
        concept_root / "adjusted" / "semantic_gap_distances_lr.json",
    ]
    fp = fingerprint_of(*fp_paths) + SCRIPT_VERSION
    if should_skip(OUTPUTS, fp, args.overwrite, PRIMARY):
        print(f"Skipping {PRIMARY} \u2014 inputs unchanged")
        return

    # Coverage and semantic gaps
    cov_cols = {}
    for label, m, _ in ENCODERS:
        d = _lr_covgaps(root, m)
        if d:
            cov_cols[f"{label} LR"] = d
    for label, m, _ in ENCODERS:
        d = _mlp_covgaps(m)
        if d:
            cov_cols[f"{label} MLP"] = d
    # Zero-shot is scoped to the canonical encoder only (AGENTS.md).
    for label, m, _ in ENCODERS:
        if label != "MPNet":
            continue
        d = _zs_covgaps(root, m)
        if d:
            cov_cols[f"{label} ZS"] = d

    gap_cols = {}
    for label, m, _ in ENCODERS:
        d = _lr_gaps(root, m)
        if d:
            gap_cols[f"{label} LR"] = d
    for label, m, _ in ENCODERS:
        d = _mlp_gaps(root, m)
        if d:
            gap_cols[f"{label} MLP"] = d
    # Zero-shot is scoped to the canonical encoder only (AGENTS.md).
    for label, m, _ in ENCODERS:
        if label != "MPNet":
            continue
        d = _zs_gaps(root, m)
        if d:
            gap_cols[f"{label} ZS"] = d

    # Concept retrieval (LR only, default encoder)
    concept_cov = _concept_covgaps(root, "all-mpnet-base-v2")
    concept_gap = _concept_gaps(root, "all-mpnet-base-v2")

    col_order = [
        "MPNet LR", "MPNet MLP", "MPNet ZS",
        "MiniLM LR", "MiniLM MLP",
        "SciBERT LR", "SciBERT MLP",
    ]
    avail_cov = [c for c in col_order if c in cov_cols]
    avail_gap = [c for c in col_order if c in gap_cols]
    if avail_cov and avail_gap:
        all_sdgs = sorted(
            set().union(*[cov_cols[c].keys() for c in avail_cov])
            | set().union(*[gap_cols[c].keys() for c in avail_gap])
        )
        cov_rows = []
        sem_rows = []
        for sdg in all_sdgs:
            cov_vals = []
            for c in avail_cov:
                v = cov_cols[c].get(sdg)
                cov_vals.append(v * 100.0 if v is not None else None)
            gap_vals = []
            for c in avail_gap:
                v = gap_cols[c].get(sdg)
                gap_vals.append(v if v is not None else None)
            cc = concept_cov.get(sdg) if concept_cov else None
            cg = concept_gap.get(sdg) if concept_gap else None
            cov_rows.append((sdg, cov_vals, cc * 100.0 if cc is not None else None))
            sem_rows.append((sdg, gap_vals, cg))
        n_enc = len(ENCODERS)
        has_concept = concept_cov is not None
        _write_table(
            out_dir / "tab_app_cross_method_covgap.tex",
            cov_rows, "{:.1f}", has_concept,
            "Coverage gap (\\%)",
        )
        _write_table(
            out_dir / "tab_app_cross_method_semgap.tex",
            sem_rows, "{:.3f}", has_concept,
            "Semantic gap (cosine)",
        )
        record_fingerprint(OUTPUTS, fp, PRIMARY)
    else:
        print("WARNING: coverage or semantic gap data missing, skipping combined table")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL, type=resolve_model_alias, help=argparse.SUPPRESS)
    parser.add_argument("--overwrite", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()

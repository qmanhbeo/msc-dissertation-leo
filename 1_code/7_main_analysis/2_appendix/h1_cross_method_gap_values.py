"""
Generate appendix table for cross-method coverage gap and semantic gap raw values.

Outputs:
  tab_app_cross_method_gap_values.tex   — 15-column combined table (incl. concept retrieval)
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = ROOT / "1_code"
ANALYSIS_ROOT = Path(__file__).resolve().parents[1]
SHARED_DIR = ANALYSIS_ROOT / "0_shared"
for path in (CODE_ROOT, SHARED_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from model_utils import DEFAULT_EMBED_MODEL, DEFAULT_OUTPUT_ROOT, N_SDG, embed_dir_for_model, scored_dir_for_model

# ---------------------------------------------------------------------------
# Coverage gap loaders
# ---------------------------------------------------------------------------

def _lr_covgaps(root, m):
    p = root / "main" / m / "data" / "4_2_coverage_document_weighted.json"
    if not p.exists():
        return None
    with open(p) as f:
        data = json.load(f)
    cg = data.get("coverage_gap_hard")
    if not cg:
        return None
    return {int(k[3:]): v for k, v in cg.items()}


def _mlp_covgaps(m):
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

    doc_to_rows = defaultdict(list)
    for i, r in enumerate(policy_ids):
        doc_to_rows[r["source_doc"]].append(i)

    n_docs = len(doc_to_rows)
    doc_assignments = np.empty(n_docs, dtype=np.int32)
    for d_idx, (_, row_idxs) in enumerate(doc_to_rows.items()):
        doc_vec = policy_scores[row_idxs].mean(axis=0)
        doc_assignments[d_idx] = doc_vec.argmax()

    pol_counts = np.bincount(doc_assignments, minlength=N_SDG).astype(float)
    pol_profile = pol_counts / pol_counts.sum()

    gaps = {}
    for i in range(N_SDG):
        sdg = i + 1
        res_share = res_counts.get(sdg, 0) / res_total
        gaps[sdg] = float(abs(res_share - pol_profile[i]))
    return gaps


def _zs_covgaps(root, m):
    gap_path = root / "main" / m / "zeroshot" / "semantic_gap_distances.json"
    if not gap_path.exists():
        return None
    with open(gap_path) as f:
        data = json.load(f)
    res_counts = {r["sdg"]: r["n_papers"] for r in data["per_sdg"]}
    res_total = sum(res_counts.values())

    embed_dir = embed_dir_for_model(m)
    emb_path = embed_dir / "policy.npy"
    ids_path = embed_dir / "metadata" / "policy_ids.json"
    centroids_path = scored_dir_for_model(m) / "sdg_centroids.npy"
    if not (emb_path.exists() and ids_path.exists() and centroids_path.exists()):
        return None

    policy_emb = np.load(emb_path).astype(np.float32)
    with open(ids_path) as f:
        policy_ids = json.load(f)
    centroids = np.load(centroids_path).astype(np.float32)

    policy_scores = policy_emb @ centroids.T

    doc_to_rows = defaultdict(list)
    for i, r in enumerate(policy_ids):
        doc_to_rows[r["source_doc"]].append(i)

    n_docs = len(doc_to_rows)
    doc_assignments = np.empty(n_docs, dtype=np.int32)
    for d_idx, (_, row_idxs) in enumerate(doc_to_rows.items()):
        doc_vec = policy_scores[row_idxs].mean(axis=0)
        doc_assignments[d_idx] = doc_vec.argmax()

    pol_counts = np.bincount(doc_assignments, minlength=N_SDG).astype(float)
    pol_profile = pol_counts / pol_counts.sum()

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
    p = root / "main" / m / "data" / "4_3_semantic_gap_distances.json"
    if not p.exists():
        return None
    with open(p) as f:
        data = json.load(f)
    return {row["sdg"]: row["semantic_gap"] for row in data["per_sdg"] if row["semantic_gap"] is not None}


def _mlp_gaps(m):
    p = scored_dir_for_model(m) / "mlp_scores" / "mlp_summary.json"
    if not p.exists():
        return None
    with open(p) as f:
        data = json.load(f)
    return {int(k): v for k, v in data["semantic_gaps"].items()}


def _zs_gaps(root, m):
    p = root / "main" / m / "zeroshot" / "semantic_gap_distances.json"
    if not p.exists():
        return None
    with open(p) as f:
        data = json.load(f)
    return {row["sdg"]: row["semantic_gap"] for row in data["per_sdg"] if row["semantic_gap"] is not None}


def _concept_covgaps(root, m):
    p = root / "main" / m / "concept" / "data" / "4_2_coverage_document_weighted.json"
    if not p.exists():
        return None
    with open(p) as f:
        data = json.load(f)
    cg = data.get("coverage_gap_hard")
    if not cg:
        return None
    return {int(k[3:]): v for k, v in cg.items()}


def _concept_gaps(root, m):
    p = root / "main" / m / "concept" / "data" / "4_3_semantic_gap_distances.json"
    if not p.exists():
        return None
    with open(p) as f:
        data = json.load(f)
    return {row["sdg"]: row["semantic_gap"] for row in data["per_sdg"] if row["semantic_gap"] is not None}


# ---------------------------------------------------------------------------
# Write table helpers
# ---------------------------------------------------------------------------

COMBINED_NOTES = (
    "Coverage gap = $|\\text{research proportion} - \\text{policy proportion}|$ "
    "per SDG in percentage points, using document-weighted policy proportions "
    "(Assumption A19) for all methods. "
    "Semantic gap = $1 - \\cos(\\mathbf{r}_{\\text{SDG}}, \\mathbf{p}_{\\text{SDG}})$ "
    "where $\\mathbf{r}_{\\text{SDG}}$ and $\\mathbf{p}_{\\text{SDG}}$ are the "
    "research and policy centroids; policy centroids use a 50-segment-per-document "
    "cap. All values are cosine units."
)


def _write_combined_table(path, rows, cov_fmt, gap_fmt, concept_cov=None, concept_gap=None):
    has_concept = concept_cov is not None
    n_data = 13 + (2 if has_concept else 0)

    lines = [
        "% Auto-generated by 1_code/7_main_analysis/2_appendix/h1_cross_method_gap_values.py",
        f"\\begin{{tabular}}{{{'l' + 'r' * (n_data - 1)}}}",
        r"\toprule",
    ]

    h1_g = r"\multicolumn{7}{c}" if has_concept else r"\multicolumn{6}{c}"
    h1 = ["SDG",
          f"{h1_g}{{Coverage gap (\\%)}}",
          f"{h1_g}{{Semantic gap (cosine)}}"]
    lines.append(" & ".join(h1) + r" \\")

    if has_concept:
        cm1 = [r"\cmidrule(lr){2-8}", r"\cmidrule(lr){9-15}"]
    else:
        cm1 = [r"\cmidrule(lr){2-7}", r"\cmidrule(lr){8-13}"]
    lines.append(" ".join(cm1))

    h2 = [""]
    for _ in range(2):
        h2.append(r"\multicolumn{3}{c}{MPNet (768d)}")
        h2.append(r"\multicolumn{3}{c}{MiniLM (384d)}")
        if has_concept:
            h2.append("Retrieval")
    lines.append(" & ".join(h2) + r" \\")

    if has_concept:
        cm2 = [r"\cmidrule(lr){2-4}", r"\cmidrule(lr){5-7}",
               r"\cmidrule(lr){9-11}", r"\cmidrule(lr){12-14}"]
    else:
        cm2 = [r"\cmidrule(lr){2-4}", r"\cmidrule(lr){5-7}",
               r"\cmidrule(lr){8-10}", r"\cmidrule(lr){11-13}"]
    lines.append(" ".join(cm2))

    h3 = [""]
    for _ in range(2):
        h3 += ["LR", "MLP", "ZS", "LR", "MLP", "ZS"]
        if has_concept:
            h3.append("Concept")
    lines.append(" & ".join(h3) + r" \\")

    lines.append(r"\midrule")

    for entry in rows:
        if has_concept:
            sdg, cov_vals, gap_vals, cc, cg = entry
        else:
            sdg, cov_vals, gap_vals = entry
            cc = cg = None
        cells = [str(sdg)]
        for v in cov_vals[:3]:
            cells.append(cov_fmt.format(v) if v is not None else "--")
        for v in cov_vals[3:]:
            cells.append(cov_fmt.format(v) if v is not None else "--")
        if has_concept:
            cells.append(cov_fmt.format(cc * 100.0) if cc is not None else "--")
        for v in gap_vals[:3]:
            cells.append(gap_fmt.format(v) if v is not None else "--")
        for v in gap_vals[3:]:
            cells.append(gap_fmt.format(v) if v is not None else "--")
        if has_concept:
            cells.append(gap_fmt.format(cg) if cg is not None else "--")
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

    out_dir = root / "appendix" / model / "h1_cross_method_gap_values" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    ENCODERS = [
        ("MPNet", "all-mpnet-base-v2"),
        ("MiniLM", "all-MiniLM-L6-v2"),
    ]

    # Coverage and semantic gaps
    cov_cols = {}
    for label, m in ENCODERS:
        d = _lr_covgaps(root, m)
        if d:
            cov_cols[f"{label} LR"] = d
    for label, m in ENCODERS:
        d = _mlp_covgaps(m)
        if d:
            cov_cols[f"{label} MLP"] = d
    for label, m in ENCODERS:
        d = _zs_covgaps(root, m)
        if d:
            cov_cols[f"{label} ZS"] = d

    gap_cols = {}
    for label, m in ENCODERS:
        d = _lr_gaps(root, m)
        if d:
            gap_cols[f"{label} LR"] = d
    for label, m in ENCODERS:
        d = _mlp_gaps(m)
        if d:
            gap_cols[f"{label} MLP"] = d
    for label, m in ENCODERS:
        d = _zs_gaps(root, m)
        if d:
            gap_cols[f"{label} ZS"] = d

    # Concept retrieval (LR only, default encoder)
    concept_cov = _concept_covgaps(root, "all-mpnet-base-v2")
    concept_gap = _concept_gaps(root, "all-mpnet-base-v2")

    col_order = ["MPNet LR", "MPNet MLP", "MPNet ZS", "MiniLM LR", "MiniLM MLP", "MiniLM ZS"]
    avail_cov = [c for c in col_order if c in cov_cols]
    avail_gap = [c for c in col_order if c in gap_cols]
    if avail_cov and avail_gap:
        all_sdgs = sorted(
            set().union(*[cov_cols[c].keys() for c in avail_cov])
            | set().union(*[gap_cols[c].keys() for c in avail_gap])
        )
        rows = []
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
            rows.append((sdg, cov_vals, gap_vals, cc, cg))
        _write_combined_table(
            out_dir / "tab_app_cross_method_gap_values.tex",
            rows,
            "{:.1f}",
            "{:.3f}",
            concept_cov=concept_cov,
            concept_gap=concept_gap,
        )
    else:
        print("WARNING: coverage or semantic gap data missing, skipping combined table")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()

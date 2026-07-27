"""
Generate main-text tables for the supervised LR classifier.

Outputs:
  tab_validation.tex                     — single-column LR test F1 (canonical)
  tab_cross_sensitivity_robustness.tex   — 3-axis gap-rank sensitivity table
  num_cross_sensitivity.tex             — segment-cap stability macro
  num_validation.tex                    — per-SDG F1 macros for use in text
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

from model_utils import DEFAULT_OUTPUT_ROOT, N_SDG

SDG_NAMES = {
    1: "No Poverty", 2: "Zero Hunger", 3: "Good Health", 4: "Quality Education",
    5: "Gender Equality", 6: "Clean Water", 7: "Clean Energy",
    8: "Decent Work", 9: "Industry \\& Infra.", 10: "Reduced Inequalities",
    11: "Sustainable Cities", 12: "Responsible Cons.", 13: "Climate Action",
    14: "Life Below Water", 15: "Life on Land", 16: "Peace \\& Justice",
    17: "Partnerships",
}

parser = argparse.ArgumentParser()
parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_ROOT))
args = parser.parse_args()

root = Path(args.output_dir)
OUT_MAIN = root / "main" / "tables"
OUT_MAIN.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Load LR test F1 from retrain results
# ---------------------------------------------------------------------------
RETRAIN_JSON = ROOT / "2_data/4_supervised_model_results/all-mpnet-base-v2/model/sdg_retrain_results.json"
with open(RETRAIN_JSON) as f:
    retrain = json.load(f)
lr_per_sdg = {}
for k, v in retrain["test_results"]["per_sdg_f1"].items():
    sdg_num = int(k.split("_")[1])
    lr_per_sdg[sdg_num] = v
lr_macro = retrain["test_results"]["macro_f1"]

# ---------------------------------------------------------------------------
# 2. Load LR semantic gaps (canonical assignment method)
# ---------------------------------------------------------------------------
LR_GAP_PATH = root / "main" / "data" / "4_3_semantic_gap_distances.json"
def load_lr_gaps():
    if not LR_GAP_PATH.exists():
        return None
    with open(LR_GAP_PATH) as f:
        data = json.load(f)
    return {row["sdg"]: row["semantic_gap"] for row in data["per_sdg"] if row["semantic_gap"] is not None}

# ---------------------------------------------------------------------------
# 3. Load zero-shot semantic gaps
# ---------------------------------------------------------------------------
ZS_GAP_PATH = root / "zeroshot" / "semantic_gap_distances.json"
def load_zs_gaps():
    if not ZS_GAP_PATH.exists():
        return None
    with open(ZS_GAP_PATH) as f:
        data = json.load(f)
    return {row["sdg"]: row["semantic_gap"] for row in data["per_sdg"] if row["semantic_gap"] is not None}

# ---------------------------------------------------------------------------
# 4. Load segment-cap robustness gaps
# ---------------------------------------------------------------------------
CAP_PATH = root / "main" / "data" / "4_3_semantic_gap_robustness_caps.json"
def load_cap_gaps():
    if not CAP_PATH.exists():
        return None, None
    with open(CAP_PATH) as f:
        data = json.load(f)
    cap_20 = {row["sdg"]: row["semantic_gap"] for row in data.get("cap_20", []) if row.get("semantic_gap") is not None}
    cap_none = {row["sdg"]: row["semantic_gap"] for row in data.get("cap_none", []) if row.get("semantic_gap") is not None}
    return cap_20, cap_none

# ---------------------------------------------------------------------------
# 5. Load policy source-family gaps from appendix table
# ---------------------------------------------------------------------------
POLICY_GAP_TEX = root / "appendix" / "a2_source_family_sensitivity" / "tables" / "tab_a2_policy_source_family_gap.tex"
def parse_policy_source_gaps():
    """Return {family_label: {sdg: gap}} parsing the appendix tex table."""
    if not POLICY_GAP_TEX.exists():
        return {}
    text = POLICY_GAP_TEX.read_text(encoding="utf-8")
    col_labels = []  # e.g. "Full Corpus", "Curated AI/SDG", "SDGi VNR/VLR", "UNGDC"
    in_header = True
    families = {}
    for line in text.splitlines():
        line = line.strip()
        # Skip non-data lines
        if not line or line.startswith("%") or line.startswith(r"\toprule") or line.startswith(r"\midrule") or line.startswith(r"\bottomrule") or line.startswith(r"\end") or line.startswith(r"\cmidrule"):
            continue
        # Parse header
        if in_header and "SDG" in line and "&" in line:
            parts = [p.strip() for p in line.rstrip("\\").split("&")]
            for i in range(1, len(parts)):
                col_labels.append(parts[i])
            in_header = False
            continue
        if in_header:
            continue
        # Data row
        m = re.match(r"SDG\s+(\d+)", line)
        if not m:
            continue
        sdg = int(m.group(1))
        parts = [p.strip() for p in line.rstrip("\\").split("&")]
        # Format: SDG & Full n & Full gap & Curated n & Curated gap & SDGi n & SDGi gap & UNGDC n & UNGDC gap
        # parts[0] = "SDG 1", parts[1] = n, parts[2] = gap, parts[3] = n, parts[4] = gap, ...
        # So gap values are at indices 2, 4, 6, 8
        gap_indices = [2, 4, 6, 8]
        labels = ["full", "curated", "sdgi", "ungdc"]
        for label, gi in zip(labels, gap_indices):
            if gi < len(parts):
                try:
                    val = float(parts[gi])
                    families.setdefault(label, {})[sdg] = val
                except ValueError:
                    pass
    return families

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
# Write num_validation.tex (LR F1 macros)
# ---------------------------------------------------------------------------
def write_num_validation():
    lines = [
        f"% Auto-generated by 1_code/7_main_analysis/1_canonical/3_generate_cross_sensitivity_table.py — do not edit manually",
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
    MLP_RETRAIN_PATH = ROOT / "2_data/4_supervised_model_results/all-mpnet-base-v2/model/mlp_retrain_results.json"
    if MLP_RETRAIN_PATH.exists():
        with open(MLP_RETRAIN_PATH) as f:
            mlp_data = json.load(f)
        mlp_macro = mlp_data["test_results"]["macro_f1"]
        lines.append(rf"\newcommand{{\MlpMacroFOne}}{{{mlp_macro:.3f}}}")
    path = OUT_MAIN / "num_validation.tex"
    path.write_text("\n".join(lines) + "\n")
    print(f"Written {path}")

# ---------------------------------------------------------------------------
# Write tab_validation.tex (single LR F1 column)
# ---------------------------------------------------------------------------
def write_validation_table():
    lines = [
        "% Auto-generated by 1_code/7_main_analysis/1_canonical/3_generate_cross_sensitivity_table.py — do not edit manually",
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
    path = OUT_MAIN / "tab_validation.tex"
    path.write_text("\n".join(lines) + "\n")
    print(f"Written {path}")

# ---------------------------------------------------------------------------
# Write tab_cross_sensitivity_robustness.tex
# ---------------------------------------------------------------------------
def write_cross_sensitivity():
    lr_gaps = load_lr_gaps()
    zs_gaps = load_zs_gaps()
    cap_20, cap_none = load_cap_gaps()
    policy_families = parse_policy_source_gaps()

    # Determine which column groups are available
    col_groups = []

    # Load MLP gaps
    MLP_SUMMARY = ROOT / "2_data/5_supervised_scored/all-mpnet-base-v2/mlp_scores/mlp_summary.json"
    def load_mlp_gaps():
        if not MLP_SUMMARY.exists():
            return None
        with open(MLP_SUMMARY) as f:
            data = json.load(f)
        return {int(k): v for k, v in data["semantic_gaps"].items()}
    mlp_gaps = load_mlp_gaps()

    # Group 1: Assignment method
    if lr_gaps:
        cols = [("LR", compute_ranks(lr_gaps), "LR (canonical)")]
        if zs_gaps:
            cols.append(("Zero-shot", compute_ranks(zs_gaps), "Zero-shot centroid"))
        if mlp_gaps:
            cols.append(("MLP", compute_ranks(mlp_gaps), "MLP"))
        col_groups.append(("Assignment method", cols))

    # Group 2: Policy source family
    pcols = []
    family_labels = {"full": "Full", "curated": "Curated", "sdgi": "SDGi", "ungdc": "UNGDC"}
    for key, label in family_labels.items():
        if key in policy_families:
            pcols.append((label, compute_ranks(policy_families[key]), label))
    if pcols:
        col_groups.append(("Policy source", pcols))

    # Group 3: Segment cap
    cap_cols = []
    if cap_20:
        cap_cols.append(("20", compute_ranks(cap_20), "Cap 20"))
    if cap_none:
        cap_cols.append(("None", compute_ranks(cap_none), "No cap"))
    if cap_cols:
        col_groups.append(("Segment cap", cap_cols))

    if not col_groups:
        print("WARNING: no data available for cross-sensitivity table, skipping")
        return

    # Build column header from available groups
    n_cols = sum(len(cols) for _, cols in col_groups)

    # LaTeX table
    tex = [
        "% Auto-generated by 1_code/7_main_analysis/1_canonical/3_generate_cross_sensitivity_table.py — do not edit manually",
        r"\begin{table}[ht]",
        r"\centering",
        r"\footnotesize",
        r"\caption{Cross-sensitivity robustness of within-SDG semantic gap rankings.}",
        r"\label{tab:cross-sensitivity-robustness}",
        r"\resizebox{\textwidth}{!}{%",
        rf"\begin{{tabular}}{{l{'c' * n_cols}}}",  # adjust to n_cols
        r"\toprule",
    ]

    # Build multi-column headers
    header_parts = ["SDG"]
    header_midrules = []
    col_idx = 2  # column index start (1 = "SDG")
    for group_label, columns in col_groups:
        n = len(columns)
        if n == 0:
            continue
        span_start = col_idx
        span_end = col_idx + n - 1
        if n == 1:
            header_parts.append(r"\multicolumn{1}{c}{" + columns[0][2] + "}")
        else:
            header_parts.append(r"\multicolumn{" + str(n) + r"}{c}{" + group_label + "}")
            span_str = f"{span_start}-{span_end}"
            header_midrules.append((span_start, span_end))
        col_idx += n

    tex.append(" & ".join(header_parts) + r" \\")

    # Midrule for multi-column groups
    for start, end in header_midrules:
        tex.append(r"\cmidrule(lr){" + f"{start}-{end}" + "}")

    # Column sub-headers (individual column names)
    sub_headers = ["SDG"]
    for _, columns in col_groups:
        for label, _, _ in columns:
            sub_headers.append(label)
    tex.append(" & ".join(sub_headers) + r" \\")
    tex.append(r"\midrule")

    # Data rows
    all_sdgs = set()
    for _, columns in col_groups:
        for _, ranks, _ in columns:
            all_sdgs.update(ranks.keys())
    all_sdgs = sorted(all_sdgs)

    for sdg in all_sdgs:
        cells = [f"SDG {sdg}"]
        for _, columns in col_groups:
            for _, ranks, _ in columns:
                cells.append(str(ranks.get(sdg, "--")))
        tex.append(" & ".join(cells) + r" \\")

    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")
    tex.append(r"}")
    tex.append(r"\par\smallskip\footnotesize\emph{Notes:} Each cell reports the gap rank (1 = largest gap, 17 = smallest gap) under each measurement configuration. The LR column is the canonical assignment method (cap~50 segment cap, full policy corpus); the Zero-shot column uses nearest-centroid assignment; the MLP column uses the 4-layer/384-hidden champion retrained on the full training pool. Segment-cap and policy-source columns are LR-based unless noted.\par")
    tex.append(r"\end{table}")

    path = OUT_MAIN / "tab_cross_sensitivity_robustness.tex"
    path.write_text("\n".join(tex) + "\n")
    print(f"Written {path}  columns={n_cols}")

# ---------------------------------------------------------------------------
# Write num_cross_sensitivity.tex
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
        f"% Auto-generated by 1_code/7_main_analysis/1_canonical/3_generate_cross_sensitivity_table.py — do not edit manually",
        rf"\newcommand{{\CapStabilityRho}}{{{rho_val}}}",
    ]
    path = OUT_MAIN / "num_cross_sensitivity.tex"
    path.write_text("\n".join(lines) + "\n")
    print(f"Written {path}  rho={rho_val}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    write_num_validation()
    write_validation_table()
    write_cross_sensitivity()
    write_num_cross_sensitivity()
    print("Cross-sensitivity table generation complete.")

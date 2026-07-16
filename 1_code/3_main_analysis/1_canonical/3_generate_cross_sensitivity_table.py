"""Generate cross-sensitivity tables for the main text.

Outputs two tables:
  1. 4_outputs/main/tables/tab_cross_sensitivity_robustness.tex  (gap ranks)
  2. 4_outputs/main/tables/tab_validation.tex                    (F1 validation)
"""

import argparse
import json
import re
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--output-dir", default="4_outputs", help="Root output directory (default: 4_outputs)")
args = parser.parse_args()

root = Path(args.output_dir)
APPENDIX = root / "appendix"
OUT_TABLES = root / "tables"
OUT_TABLES.mkdir(parents=True, exist_ok=True)
OUT_MAIN = root / "main" / "tables"
OUT_MAIN.mkdir(parents=True, exist_ok=True)

MODEL_FILE = APPENDIX / "d_model_sensitivity" / "tables" / "tab_model_sensitivity.tex"
REF_FILE = APPENDIX / "a1_sdg_source_comparison" / "tables" / "tab_a1_source_comparison_covgap.tex"
POLICY_FILE = APPENDIX / "a2_source_family_sensitivity" / "tables" / "tab_a2_policy_source_family_gap.tex"
F1_REF_FILE = APPENDIX / "a1_sdg_source_comparison" / "tables" / "tab_a1_source_comparison_f1cos.tex"

CI_FILE = root / "main" / "data" / "4_1_validation_bootstrap_ci.json"
CAP_FILE = root / "main" / "data" / "4_3_semantic_gap_robustness_caps.json"

import numpy as np


def load_cap_ranks():
    """Return {cap_key: {sdg: rank}} for cap_20, cap_50 (canon), cap_none.

    cap_50 baseline is taken from the canonical (minilm) model column, which is the
    same configuration (MiniLM + combined centroids + full corpus + segment cap 50).
    """
    can_ranks = compute_ranks(parse_model_table(MODEL_FILE), "minilm")
    data = {"cap_50": can_ranks}
    if not CAP_FILE.exists():
        return data
    raw = json.loads(CAP_FILE.read_text(encoding="utf-8"))
    for cap_key in ("cap_20", "cap_none"):
        if cap_key not in raw:
            continue
        rows = {row["sdg"]: row["semantic_gap"] for row in raw[cap_key] if row["semantic_gap"] is not None}
        ranks = {sdg: rank + 1 for rank, (sdg, _) in enumerate(
            sorted(rows.items(), key=lambda x: x[1], reverse=True))}
        data[cap_key] = ranks
    return data


def load_ci():
    """Return bootstrap CI dict, or None if the JSON is absent."""
    if not CI_FILE.exists():
        return None
    return json.loads(CI_FILE.read_text(encoding="utf-8"))

# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def parse_model_table(path):
    """Return {sdg: {col: gap_val}} for col in ('minilm', 'mpnet')."""
    data = {}
    text = path.read_text()
    for line in text.splitlines():
        line = line.strip()
        m = re.match(r"SDG\s+(\d+)", line)
        if not m:
            continue
        sdg = int(m.group(1))
        parts = [p.strip() for p in line.rstrip("\\").split("&")]
        # parts[5] = "0.327\,(13)", parts[6] = "0.420\,( 7)"
        minilm_val = float(re.match(r"([\d.]+)", parts[5]).group(1))
        mpnet_val = float(re.match(r"([\d.]+)", parts[6]).group(1))
        data[sdg] = {"minilm": minilm_val, "mpnet": mpnet_val}
    return data


def parse_ref_table(path):
    """Return {sdg: {col: gap_val or None}}.

    Columns in the file after SDG label:
       Combined-n, Combined-gap, OSDG-n, OSDG-gap, SDGi-n, SDGi-gap,
       KH-n, KH-gap, Aurora-n, Aurora-gap
    """
    col_names = ["combined", "osdg", "sdgi", "kh", "aurora"]
    data = {}
    text = path.read_text()
    for line in text.splitlines():
        line = line.strip()
        m = re.match(r"SDG\s+(\d+)", line)
        if not m:
            continue
        sdg = int(m.group(1))
        parts = [p.strip() for p in line.rstrip("\\").split("&")]
        vals = {}
        for i, name in enumerate(col_names):
            raw = parts[2 + 2 * i]  # gap column: index 2, 4, 6, 8, 10
            if raw == "--":
                vals[name] = None
            else:
                vals[name] = float(raw.replace(",", ""))
        data[sdg] = vals
    return data


def parse_policy_table(path):
    """Return {sdg: {col: gap_val}} for col in ('full', 'ai_sdg', 'sdgi_vnr', 'ungdc').

    Columns: Full-n, Full-gap, AI/SDG-n, AI/SDG-gap, SDGi VNR/VLR-n, SDGi VNR/VLR-gap, UNGDC-n, UNGDC-gap
    """
    col_names = ["full", "ai_sdg", "sdgi_vnr", "ungdc"]
    data = {}
    text = path.read_text()
    for line in text.splitlines():
        line = line.strip()
        m = re.match(r"SDG\s+(\d+)", line)
        if not m:
            continue
        sdg = int(m.group(1))
        parts = [p.strip() for p in line.rstrip("\\").split("&")]
        vals = {}
        for i, name in enumerate(col_names):
            raw = parts[2 + 2 * i]
            vals[name] = float(raw.replace(",", ""))
        data[sdg] = vals
    return data


def parse_model_f1_table(path):
    """Return {sdg: {'minilm': f1, 'mpnet': f1}}."""
    data = {}
    text = path.read_text()
    for line in text.splitlines():
        line = line.strip()
        m = re.match(r"SDG\s+(\d+)", line)
        if not m:
            continue
        sdg = int(m.group(1))
        parts = [p.strip() for p in line.rstrip("\\").split("&")]
        data[sdg] = {
            "minilm": float(re.match(r"([\d.]+)", parts[1]).group(1)),
            "mpnet": float(re.match(r"([\d.]+)", parts[2]).group(1)),
        }
    return data


def parse_f1_ref_table(path):
    """Return {sdg: {col: f1 or None}} for col in ('osdg', 'sdgi', 'kh', 'aurora').

    Columns: SDG, Combined-f1, Combined-cos, OSDG-f1, OSDG-cos,
             SDGi-f1, SDGi-cos, KH-f1, KH-cos, Aurora-f1, Aurora-cos
    """
    col_names = ["osdg", "sdgi", "kh", "aurora"]
    data = {}
    text = path.read_text()
    for line in text.splitlines():
        line = line.strip()
        m = re.match(r"SDG\s+(\d+)", line)
        if not m:
            continue
        sdg = int(m.group(1))
        parts = [p.strip() for p in line.rstrip("\\").split("&")]
        vals = {}
        for i, name in enumerate(col_names):
            raw = parts[3 + 2 * i]  # F1 at indices 3, 5, 7, 9
            if raw == "--":
                vals[name] = None
            else:
                vals[name] = float(raw)
        data[sdg] = vals
    return data


def macro_f1(values_dict, key):
    """Unweighted mean F1 over SDGs where value is not None."""
    vals = [row[key] for row in values_dict.values() if row[key] is not None]
    return sum(vals) / len(vals) if vals else 0.0


SDG_NAMES = {
    1: "No Poverty", 2: "Zero Hunger", 3: "Good Health", 4: "Quality Education",
    5: "Gender Equality", 6: "Clean Water", 7: "Clean Energy",
    8: "Decent Work", 9: "Industry \\& Infra.", 10: "Reduced Inequalities",
    11: "Sustainable Cities", 12: "Responsible Cons.", 13: "Climate Action",
    14: "Life Below Water", 15: "Life on Land", 16: "Peace \\& Justice",
    17: "Partnerships",
}


def write_f1_table(model_data, f1_ref_data, ci_data=None):
    """Write the cross-condition F1 validation table.

    If ci_data is provided (bootstrap JSON), the canonical (Canon) column cells
    are suffixed with their 95% confidence interval as (low; high).
    """
    rows = []
    macro_vals = {}

    # Column definitions for macro computation
    col_defs = [
        ("can", r"Canon (95\% bootstrap CI)", "minilm"),
        ("mpnet", "MPNet", "mpnet"),
        ("osdg", "OSDG", "osdg"),
        ("sdgi", "SDGi", "sdgi"),
        ("kh", "KH", "kh"),
        ("aurora", "Aurora", "aurora"),
    ]

    per_sdg_ci = ci_data.get("per_sdg_f1_ci") if ci_data else None
    macro_ci = ci_data.get("macro_f1_ci") if ci_data else None

    # Compute per-SDG rows
    for sdg in range(1, 18):
        cells = [f"SDG {sdg} ({SDG_NAMES[sdg]})"]
        for key, _, model_key in col_defs:
            if key in ("can", "mpnet"):
                val = model_data.get(sdg, {}).get(model_key)
            else:
                val = f1_ref_data.get(sdg, {}).get(model_key)
            if val is not None:
                cell = f"{val:.3f}"
                if key == "can" and per_sdg_ci is not None:
                    ci = per_sdg_ci.get(str(sdg))
                    if ci is not None:
                        cell += f" ({ci['ci_low']:.3f}; {ci['ci_high']:.3f})"
                cells.append(cell)
            else:
                cells.append("--")
        rows.append(cells)

    # Compute macro-F1 per column
    for key, label, model_key in col_defs:
        if key in ("can", "mpnet"):
            vals = [row[model_key] for row in model_data.values() if row.get(model_key) is not None]
        else:
            vals = [row[model_key] for row in f1_ref_data.values() if row.get(model_key) is not None]
        macro_vals[key] = sum(vals) / len(vals) if vals else 0.0

    lines = [
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"& \multicolumn{2}{c}{Model} & \multicolumn{4}{c}{Reference source} \\",
        r"\cmidrule(r){2-3} \cmidrule(l){4-7}",
        r"SDG & " + " & ".join(label for _, label, _ in col_defs) + r" \\",
        r"\midrule",
    ]

    for cells in rows:
        lines.append(" & ".join(cells) + r" \\")

    # Macro-F1 row
    macro_cells = ["Macro-F1 (SDGs 1--17)"]
    for key, _, _ in col_defs:
        cell = f"\\textbf{{{macro_vals[key]:.3f}}}"
        if key == "can" and macro_ci is not None:
            cell += f" ({macro_ci['ci_low']:.3f}; {macro_ci['ci_high']:.3f})"
        macro_cells.append(cell)
    lines.append(" & ".join(macro_cells) + r" \\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
    ])

    path = OUT_MAIN / "tab_validation.tex"
    path.write_text("\n".join(lines) + "\n")
    print(f"Written {path}")


# ---------------------------------------------------------------------------
# Rank computation
# ---------------------------------------------------------------------------

def compute_ranks(data_dict, key):
    """Return {sdg: rank} where rank 1 = highest gap.

    Only includes SDGs where val is not None.
    """
    items = [(sdg, row[key]) for sdg, row in data_dict.items() if row[key] is not None]
    items.sort(key=lambda x: x[1], reverse=True)  # highest gap first
    return {sdg: rank + 1 for rank, (sdg, _) in enumerate(items)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    model_data = parse_model_table(MODEL_FILE)
    ref_data = parse_ref_table(REF_FILE)
    policy_data = parse_policy_table(POLICY_FILE)
    model_f1_data = parse_model_f1_table(MODEL_FILE)
    f1_ref_data = parse_f1_ref_table(F1_REF_FILE)

    # Compute ranks per column
    canon_ranks = compute_ranks(model_data, "minilm")
    mpnet_ranks = compute_ranks(model_data, "mpnet")

    ref_cols = ["osdg", "sdgi", "kh", "aurora"]
    ref_labels = ["OSDG-only", "SDGi-only", "KH-only", "Aurora"]
    ref_ranks = {col: compute_ranks(ref_data, col) for col in ref_cols}

    policy_cols = ["ai_sdg", "sdgi_vnr", "ungdc"]
    policy_labels = ["Man.", "SDGi", "UNGDC"]
    policy_ranks = {col: compute_ranks(policy_data, col) for col in policy_cols}

    # Segment-cap sensitivity ranks (cap_50 = canonical baseline, not shown separately)
    cap_ranks = load_cap_ranks()
    cap_keys = ["cap_20", "cap_none"]
    cap_labels = ["20", "no"]

    # Rank-correlation stability across the cap family (most conservative = min pairwise)
    cap_rank_vectors = []
    for key in cap_keys:
        if key not in cap_ranks:
            continue
        vec = [cap_ranks[key].get(sdg) for sdg in range(1, 18)]
        if all(v is not None for v in vec):
            cap_rank_vectors.append(vec)
    cap_stability_rho = "--"
    if len(cap_rank_vectors) >= 2:
        rho_vals = []
        for i in range(len(cap_rank_vectors)):
            for j in range(i + 1, len(cap_rank_vectors)):
                r = np.corrcoef(cap_rank_vectors[i], cap_rank_vectors[j])[0, 1]
                rho_vals.append(float(r))
        cap_stability_rho = f"{min(rho_vals):.3f}"

    # Build table rows (SDGs in numerical order)
    rows = []
    for sdg in range(1, 18):
        cells = [f"SDG {sdg}"]

        # Model group
        cells.append(str(canon_ranks.get(sdg, "--")))
        cells.append(str(mpnet_ranks.get(sdg, "--")))

        # Reference group
        for col in ref_cols:
            r = ref_ranks[col].get(sdg)
            cells.append(str(r) if r is not None else "--")

        # Policy group
        for col in policy_cols:
            r = policy_ranks[col].get(sdg)
            cells.append(str(r) if r is not None else "--")

        # Segment-cap group
        for key in cap_keys:
            r = cap_ranks.get(key, {}).get(sdg)
            cells.append(str(r) if r is not None else "--")

        rows.append(cells)

    # Write LaTeX table
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\footnotesize",
        r"\caption{Cross-sensitivity robustness of within-SDG semantic gap rankings.}",
        r"\label{tab:cross-sensitivity-robustness}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lccccccccccc}",
        r"\toprule",
        r"& \multicolumn{2}{c}{Model}",
        r"& \multicolumn{4}{c}{Reference source}",
        r"& \multicolumn{3}{c}{Policy source}",
        r"& \multicolumn{2}{c}{Segment cap} \\",
        r"\cmidrule(r){2-3} \cmidrule(lr){4-7} \cmidrule(lr){8-10} \cmidrule(l){11-12}",
        r"SDG & Canon & MPNet & OSDG & SDGi & KH & Aur & Man. & SDGi & UNGDC & 20 & no \\",
        r"\midrule",
    ]

    for cells in rows:
        lines.append(" & ".join(cells) + r" \\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        r"\par\smallskip\footnotesize\emph{Notes:} Each cell reports the gap rank (1 = largest gap, 17 = smallest gap) under each measurement configuration. ``Canon'' is the cap-50 baseline (MiniLM + combined centroids + full policy corpus); the Segment-cap group isolates the per-document segment-cap parameter (encoder, centroids, and corpus held fixed), with an alternative cap of 20 segments per document and an uncapped (no) run. A ``--'' indicates that the source does not cover that SDG.\par",
        r"\end{table}",
    ])

    out_path = OUT_MAIN / "tab_cross_sensitivity_robustness.tex"
    out_path.write_text("\n".join(lines) + "\n")
    print(f"Written {out_path}")
    print(f"Table rows: {len(rows)}")
    print(f"Cap stability rho (min pairwise): {cap_stability_rho}")

    # Emit numeric macro for segment-cap stability
    num_path = OUT_MAIN / "num_cross_sensitivity.tex"
    num_path.write_text(
        f"% Auto-generated by 3_generate_cross_sensitivity_table.py — do not edit by hand.\n"
        f"\\newcommand{{\\CapStabilityRho}}{{{cap_stability_rho}}}\n"
    )
    print(f"Written {num_path}")

    write_f1_table(model_f1_data, f1_ref_data, load_ci())


if __name__ == "__main__":
    main()

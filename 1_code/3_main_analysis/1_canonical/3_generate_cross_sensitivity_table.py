"""Generate cross-sensitivity robustness table for the main text.

Reads three appendix .tex tables (model sensitivity, reference-source
comparison, policy source-family sensitivity), extracts gap values
per SDG, computes ranks (1 = largest gap), and outputs a unified
LaTeX table to 4_outputs/tables/tab_cross_sensitivity_robustness.tex.
"""

import re
from pathlib import Path

BASE = Path("../../../4_outputs/appendix")
OUT = Path("../../../4_outputs/tables")
OUT.mkdir(parents=True, exist_ok=True)

MODEL_FILE = BASE / "d_model_sensitivity" / "tables" / "tab_model_sensitivity.tex"
REF_FILE = BASE / "a1_sdg_source_comparison" / "tables" / "tab_a1_source_comparison_covgap.tex"
POLICY_FILE = BASE / "a2_source_family_sensitivity" / "tables" / "tab_a2_policy_source_family_gap.tex"

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

    # Compute ranks per column
    can_ranks = compute_ranks(model_data, "minilm")
    mpnet_ranks = compute_ranks(model_data, "mpnet")

    ref_cols = ["combined", "osdg", "sdgi", "kh", "aurora"]
    ref_labels = ["Combined", "OSDG-only", "SDGi-only", "KH-only", "Aurora"]
    ref_ranks = {col: compute_ranks(ref_data, col) for col in ref_cols}

    policy_cols = ["full", "ai_sdg", "sdgi_vnr", "ungdc"]
    policy_labels = ["Full", "AI/SDG", "SDGi-VNR", "UNGDC"]
    policy_ranks = {col: compute_ranks(policy_data, col) for col in policy_cols}

    # Build table rows (SDGs in numerical order)
    rows = []
    for sdg in range(1, 18):
        cells = [f"SDG {sdg}"]

        # Model group
        cells.append(str(can_ranks.get(sdg, "--")))
        cells.append(str(mpnet_ranks.get(sdg, "--")))

        # Reference group — the first column (Combined) is the baseline
        for col in ref_cols:
            r = ref_ranks[col].get(sdg)
            cells.append(str(r) if r is not None else "--")

        # Policy group — the first column (Full) is the baseline
        for col in policy_cols:
            r = policy_ranks[col].get(sdg)
            cells.append(str(r) if r is not None else "--")

        rows.append(cells)

    # Write LaTeX table
    lines = [
        r"\begin{landscape}",
        r"\begin{table}[ht]",
        r"\centering",
        r"\small",
        r"\caption{Cross-sensitivity robustness of within-SDG semantic gap rankings.}",
        r"\label{tab:cross-sensitivity-robustness}",
        r"\begin{tabular}{l|c|c|c|ccccc|c|cccc}",
        r"\toprule",
        r"& \multicolumn{2}{c}{Model}",
        r"& \multicolumn{5}{c}{Reference source}",
        r"& \multicolumn{4}{c}{Policy source} \\",
        r"\cmidrule(lr){2-3} \cmidrule(lr){4-8} \cmidrule(lr){9-12}",
        r"SDG & Can & MPNet & Combined & OSDG & SDGi & KH & Aur & Full & AI/SDG & SDGi-VNR & UNGDC \\",
        r"\midrule",
    ]

    for cells in rows:
        lines.append(" & ".join(cells) + r" \\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        r"\end{landscape}",
    ])

    out_path = OUT / "tab_cross_sensitivity_robustness.tex"
    out_path.write_text("\n".join(lines) + "\n")
    print(f"Written {out_path}")
    print(f"Table rows: {len(rows)}")


if __name__ == "__main__":
    main()

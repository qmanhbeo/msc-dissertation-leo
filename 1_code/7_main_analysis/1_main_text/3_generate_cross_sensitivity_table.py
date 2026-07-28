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

from model_utils import DEFAULT_EMBED_MODEL, DEFAULT_OUTPUT_ROOT, N_SDG, model_results_dir_for_model, scored_dir_for_model

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
    parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL, help=argparse.SUPPRESS)
    return parser.parse_args()

# ---------------------------------------------------------------------------
# 2/3/4. Load LR / zero-shot / MLP semantic gaps for an ARBITRARY encoder.
#    Paths are derived from `root` (set in run()) so the same loader
#    serves both the canonical encoder and the encoder-sensitivity partner.
# ---------------------------------------------------------------------------
def load_lr_gaps(m):
    p = root / "main" / m / "data" / "4_3_semantic_gap_distances.json"
    if not p.exists():
        return None
    with open(p) as f:
        data = json.load(f)
    return {row["sdg"]: row["semantic_gap"] for row in data["per_sdg"] if row["semantic_gap"] is not None}


def load_zs_gaps(m):
    p = root / "main" / m / "zeroshot" / "semantic_gap_distances.json"
    if not p.exists():
        return None
    with open(p) as f:
        data = json.load(f)
    return {row["sdg"]: row["semantic_gap"] for row in data["per_sdg"] if row["semantic_gap"] is not None}


def load_mlp_gaps(m):
    p = scored_dir_for_model(m) / "mlp_scores" / "mlp_summary.json"
    if not p.exists():
        return None
    with open(p) as f:
        data = json.load(f)
    return {int(k): v for k, v in data["semantic_gaps"].items()}


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
# 5. Load policy source-family gaps from appendix table
# ---------------------------------------------------------------------------
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
    path = OUT_MAIN / "num_validation.tex"
    path.write_text("\n".join(lines) + "\n")
    print(f"Written {path}")

# ---------------------------------------------------------------------------
# Write tab_validation.tex (single LR F1 column)
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
    path = OUT_MAIN / "tab_validation.tex"
    path.write_text("\n".join(lines) + "\n")
    print(f"Written {path}")

# ---------------------------------------------------------------------------
# Write tab_cross_sensitivity_robustness.tex
# ---------------------------------------------------------------------------
def write_cross_sensitivity():
    # --- Encoder-sensitivity axis (the headline robustness check) ------
    # Fixed encoder pair: canonical base model + the alternative encoder.
    # The table is always canonical-primary, so the manuscript's MPNet
    # column stays the reference and MiniLM is the sensitivity partner.
    ENCODER_CANONICAL = "all-mpnet-base-v2"
    ENCODER_PARTNER = "all-MiniLM-L6-v2"
    ENCODER_DIM = {"all-mpnet-base-v2": "768d", "all-MiniLM-L6-v2": "384d"}

    cap_20, cap_none = load_cap_gaps()
    policy_families = parse_policy_source_gaps()

    def _enc_sub(m, sublabel):
        lr = load_lr_gaps(m)
        mlp = load_mlp_gaps(m)
        zs = load_zs_gaps(m)
        cols = []
        if lr:
            cols.append(("LR", compute_ranks(lr),
                         "LR (canonical supervised) — policy segments capped at 50/doc/SDG"))
        if mlp:
            cols.append(("MLP", compute_ranks(mlp),
                         "MLP (4-layer/384-hidden) — policy segments capped at 50/doc/SDG"))
        if zs:
            cols.append(("ZS", compute_ranks(zs),
                         "Zero-shot nearest-centroid on SDG reference centroids"))
        return sublabel, cols

    enc_subgroups = []
    c = _enc_sub(ENCODER_CANONICAL,
                 f"{ENCODER_CANONICAL.split('-')[1]} ({ENCODER_DIM[ENCODER_CANONICAL]})")
    if c[1]:
        enc_subgroups.append(c)
    p = _enc_sub(ENCODER_PARTNER,
                 f"{ENCODER_PARTNER.split('-')[1]} ({ENCODER_DIM[ENCODER_PARTNER]})")
    if p[1]:
        enc_subgroups.append(p)

    col_groups = []
    if enc_subgroups:
        col_groups.append(("Encoder (embedding architecture)", enc_subgroups))

    # --- Policy source family (LR-based, primary model) -----------------
    # "Full" (full policy corpus) is intentionally omitted: it is
    # identical to the canonical MPNet-LR column (rho = 1.00 in the
    # Rank-Corr row), so keeping it would only duplicate that
    # baseline and widen an already-wide table.
    pcols = []
    family_labels = {"curated": "Curated", "sdgi": "SDGi", "ungdc": "UNGDC"}
    for key, label in family_labels.items():
        if key in policy_families:
            pcols.append((label, compute_ranks(policy_families[key]), f"Policy source: {label}"))
    if pcols:
        col_groups.append(("Policy source", pcols))

    # --- Segment cap (LR-based, primary model) ------------------------
    cap_cols = []
    if cap_20:
        cap_cols.append(("20", compute_ranks(cap_20), "Segment cap 20"))
    if cap_none:
        cap_cols.append(("None", compute_ranks(cap_none), "No segment cap"))
    if cap_cols:
        col_groups.append(("Segment cap", cap_cols))

    if not col_groups:
        print("WARNING: no data available for cross-sensitivity table, skipping")
        return

    # --- group helpers (support nested encoder subgroups) --------------
    def is_nested(g):
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

    all_cols = []
    for g in col_groups:
        all_cols.extend(flat_cols(g))
    n_cols = sum(group_total(g) for g in col_groups)
    has_nested = any(is_nested(g) for g in col_groups)

    # --- Level-1 / Level-2 / (Level-3) headers ----------------------
    # SDG spans all three header rows via \multirow; rows B/C leave the
    # first cell blank so it is not repeated. Flat (non-nested)
    # groups print their column labels ONCE (in row B); row C
    # prints only the method labels for the nested encoder group
    # and leaves flat groups blank (no duplicated policy/segment
    # labels across two stacked rows).
    rowA = [r"\multirow{3}{*}{SDG}"]
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
            for label, _, _ in body:
                rowB.append(label)

    if has_nested:
        rowC = [""]
        for glabel, body in col_groups:
            if is_nested((glabel, body)):
                for _, cols in body:
                    for label, _, _ in cols:
                        rowC.append(label)
            # flat groups: labels already shown in row B -> leave blank
    else:
        rowC = None

    tex = [
        "% Auto-generated by 1_code/7_main_analysis/1_main_text/3_generate_cross_sensitivity_table.py — do not edit manually",
        r"\begin{table}[ht]",
        r"\centering",
        r"\footnotesize",
        r"\caption{Cross-sensitivity robustness of within-SDG semantic gap rankings across embedding architectures and assignment methods.}",
        r"\label{tab:cross-sensitivity-robustness}",
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

    # --- SDG-level highlight (encoder-sensitivity axis) -----------------
    # Bold  = encoding-invariant top gap (|Δrank| <= STABLE_RANK_DELTA
    #         between MPNet-LR and MiniLM-LR).
    # Italic = encoder-sensitive (|Δrank| >= SENSITIVE_RANK_DELTA).
    # Thresholds are NAMED + documented; derived mechanically, not by eye.
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

    # --- Data rows -----------------------------------------------------
    all_sdgs = set()
    for _, ranks, _ in all_cols:
        all_sdgs.update(ranks.keys())
    all_sdgs = sorted(all_sdgs)

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

    # --- Rank-Corr (ρ) summary row ------------------------------------
    # Spearman of each column's SDG gap ranks vs the canonical MPNet-LR
    # column. This IS the MPNet<->MiniLM sanity check, surfaced in the
    # table itself so a reviewer can audit the encoder-sensitivity claim.
    baseline = mpnet_lr
    rho_cells = [r"Rank Corr ($\rho$)"]
    if baseline:
        common = [s for s in all_sdgs if s in baseline]
        bv = [baseline[s] for s in common]
        for _, ranks, _ in all_cols:
            cv = [ranks[s] for s in common if s in ranks]
            if len(cv) >= 2 and len(cv) == len(bv):
                rho = _spearman(bv, cv)
            else:
                rho = float("nan")
            rho_cells.append(f"{rho:.2f}" if not np.isnan(rho) else "--")
    else:
        for _ in all_cols:
            rho_cells.append("--")
    tex.append(r"\midrule")
    tex.append(" & ".join(rho_cells) + r" \\")

    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")
    tex.append(r"}")
    tex.append(
        r"\par\smallskip\footnotesize\emph{Notes:} Each cell reports the within-SDG semantic gap rank "
        r"($1 = \text{largest gap}$, $17 = \text{smallest gap}$) under that encoder and assignment method. "
        r"The base encoder is \texttt{all-mpnet-base-v2} (768-d); the alternative encoder is "
        r"\texttt{all-MiniLM-L6-v2} (384-d). LR = canonical supervised logistic-regression classifier; "
        r"policy segments are capped at 50 per source document per SDG (Assumption A-CHUNKCAT). "
        r"Zero-shot = nearest-centroid assignment on the SDG reference centroids. "
        r"MLP = 4-layer/384-hidden network retrained on the full training pool. "
        r"Policy-source and segment-cap columns are LR-based. "
        r"\textbf{Bold} = encoding-invariant (rank difference $\le$1 between MPNet and MiniLM for the same assignment method); "
        r"\textit{italic} = encoder-sensitive (rank difference $\ge$4). "
        r"Rank Corr ($\rho$) is the Spearman correlation of each column's SDG gap ranks against the "
        r"canonical MPNet-LR column.\par"
    )
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
        f"% Auto-generated by 1_code/7_main_analysis/1_main_text/3_generate_cross_sensitivity_table.py — do not edit manually",
        rf"\newcommand{{\CapStabilityRho}}{{{rho_val}}}",
    ]
    path = OUT_MAIN / "num_cross_sensitivity.tex"
    path.write_text("\n".join(lines) + "\n")
    print(f"Written {path}  rho={rho_val}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run(args: argparse.Namespace) -> None:
    global model, root, OUT_MAIN, RETRAIN_JSON, retrain, lr_per_sdg, lr_macro
    global LR_GAP_PATH, ZS_GAP_PATH, CAP_PATH, POLICY_GAP_TEX

    model = args.embed_model

    root = Path(args.output_dir)
    OUT_MAIN = root / "main" / model / "tables"
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

    LR_GAP_PATH = root / "main" / model / "data" / "4_3_semantic_gap_distances.json"
    ZS_GAP_PATH = root / "main" / model / "zeroshot" / "semantic_gap_distances.json"
    CAP_PATH = root / "main" / model / "data" / "4_3_semantic_gap_robustness_caps.json"
    POLICY_GAP_TEX = root / "appendix" / model / "a2_source_family_sensitivity" / "tables" / "tab_a2_policy_source_family_gap.tex"

    write_num_validation()
    write_validation_table()
    write_cross_sensitivity()
    write_num_cross_sensitivity()
    print("Cross-sensitivity table generation complete.")


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()

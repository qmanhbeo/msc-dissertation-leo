"""
Generate the register-topic decomposition table (plan §6.5.1).

Per-SDG decomposition: raw gap | adjusted gap | register component (raw - adj)
| coverage gap.  This is the paper's new centrepiece table.

Inputs:
  4_outputs/{model}/data/4_3_semantic_gap_distances.json           (raw gaps)
  4_outputs/{model}/data/adjusted/4_3_semantic_gap_distances.json  (adjusted gaps)
  4_outputs/{model}/data/4_2_coverage_document_weighted.json       (coverage gaps)

Outputs:
  4_outputs/{model}/data/register_decomposition.json               (JSON)
  4_outputs/{model}/tables/tab_register_decomposition.tex          (LaTeX table)
  4_outputs/{model}/tables/num_register_decomposition.tex          (LaTeX macros)

Run from project root:
  python 1_code/7_main_analysis/0_shared/g_register_decomposition.py --embed-model mpnet
"""

from __future__ import annotations

import argparse
import json
import logging
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

from model_utils import DEFAULT_EMBED_MODEL, N_SDG, SDG_NAMES, SDG_NUM_WORDS, resolve_model_alias
from shared_utils import ensure_canonical_outputs, fingerprint_of, should_skip, record_fingerprint
from semantic_gap_shared import latex_int
from shard_pipeline_utils import load_json

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate register-topic decomposition table.")
    p.add_argument("--output-dir", default=str(ROOT / "4_outputs"))
    p.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL, type=resolve_model_alias, help=argparse.SUPPRESS)
    p.add_argument("--overwrite", action="store_true", help=argparse.SUPPRESS)
    return p.parse_args()


def run(args: argparse.Namespace) -> None:
    model = args.embed_model
    layout = ensure_canonical_outputs(Path(args.output_dir), model=model)
    data_dir = layout.data_dir
    adj_data_dir = data_dir / "adjusted"
    tables_dir = layout.tables_dir

    raw_path = data_dir / "4_3_semantic_gap_distances.json"
    adj_path = adj_data_dir / "4_3_semantic_gap_distances.json"
    cov_path = data_dir / "4_2_coverage_document_weighted.json"
    out_json = data_dir / "register_decomposition.json"
    out_tex = tables_dir / "tab_register_decomposition.tex"
    out_num = tables_dir / "num_register_decomposition.tex"

    if not adj_path.exists():
        log.warning("Adjusted semantic gap not found at %s — skipping decomposition.", adj_path)
        return

    # ---- Fingerprints ----
    fp = fingerprint_of(raw_path, adj_path, cov_path) + "v1"
    if should_skip([out_json], fp, args.overwrite, out_json):
        log.info("Skipping decomposition — inputs unchanged")
        return

    # ---- Load data ----
    raw_data = load_json(raw_path)
    adj_data = load_json(adj_path)
    cov_data = load_json(cov_path)

    raw_map = {r["sdg"]: r for r in raw_data["per_sdg"]}
    adj_map = {r["sdg"]: r for r in adj_data["per_sdg"]}
    cov_gap_abs = {f"SDG{i}": cov_data["coverage_gap_hard"][f"SDG{i}"] for i in range(1, N_SDG + 1)}

    # ---- Build decomposition table ----
    per_sdg = []
    for sdg in range(1, N_SDG + 1):
        raw_gap = raw_map[sdg].get("semantic_gap")
        adj_gap = adj_map[sdg].get("semantic_gap")
        cov_gap = cov_gap_abs.get(f"SDG{sdg}")

        raw_val = float(raw_gap) if raw_gap is not None else None
        adj_val = float(adj_gap) if adj_gap is not None else None
        cov_val = float(cov_gap) if cov_gap is not None else None

        register_component = None
        if raw_val is not None and adj_val is not None:
            register_component = round(raw_val - adj_val, 6)

        per_sdg.append({
            "sdg": sdg,
            "raw_gap": raw_val,
            "adjusted_gap": adj_val,
            "register_component": register_component,
            "coverage_gap": cov_val,
            "name": SDG_NAMES[sdg],
        })

    # ---- Summary statistics ----
    valid = [r for r in per_sdg if r["raw_gap"] is not None and r["adjusted_gap"] is not None]
    mean_raw = float(np.mean([r["raw_gap"] for r in valid]))
    mean_adj = float(np.mean([r["adjusted_gap"] for r in valid]))
    mean_reg = float(np.mean([r["register_component"] for r in valid]))
    mean_cov = float(np.mean([r["coverage_gap"] for r in valid if r["coverage_gap"] is not None]))

    # ---- Correlation: coverage vs adjusted (topic), coverage vs register ----
    from scipy import stats
    valid_corr = [r for r in per_sdg if all(v is not None for v in [r["raw_gap"], r["adjusted_gap"], r["coverage_gap"], r["register_component"]])]
    if len(valid_corr) >= 3:
        cov_arr = np.array([r["coverage_gap"] for r in valid_corr])
        adj_arr = np.array([r["adjusted_gap"] for r in valid_corr])
        reg_arr = np.array([r["register_component"] for r in valid_corr])
        rho_cov_adj, p_cov_adj = stats.spearmanr(cov_arr, adj_arr)
        rho_cov_reg, p_cov_reg = stats.spearmanr(cov_arr, reg_arr)
    else:
        rho_cov_adj = p_cov_adj = rho_cov_reg = p_cov_reg = None

    # ---- Write JSON ----
    output = {
        "embedding_model": model,
        "note": "register_component = raw_gap - adjusted_gap. Positive means register divergence; negative means register similarity masking topic divergence.",
        "correlations": {
            "coverage_vs_adjusted": {"rho": round(rho_cov_adj, 4) if rho_cov_adj is not None else None, "p": round(p_cov_adj, 4) if p_cov_adj is not None else None},
            "coverage_vs_register": {"rho": round(rho_cov_reg, 4) if rho_cov_reg is not None else None, "p": round(p_cov_reg, 4) if p_cov_reg is not None else None},
        },
        "summary": {
            "mean_raw_gap": round(mean_raw, 4),
            "mean_adjusted_gap": round(mean_adj, 4),
            "mean_register_component": round(mean_reg, 4),
            "mean_coverage_gap": round(mean_cov, 4),
        },
        "per_sdg": per_sdg,
    }
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    log.info("Saved: %s", out_json)

    # ---- LaTeX table ----
    tex_lines = [
        "% Auto-generated by g_register_decomposition.py — do not edit manually",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"SDG & Description & Raw Gap & Adj. Gap & Reg. Comp. & Cov. Gap \\",
        r"\midrule",
    ]
    for r in per_sdg:
        sdg = r["sdg"]
        name = SDG_NAMES[sdg]
        raw_s = f"{r['raw_gap']:.3f}" if r['raw_gap'] is not None else "N/A"
        adj_s = f"{r['adjusted_gap']:.3f}" if r['adjusted_gap'] is not None else "N/A"
        reg_s = f"{r['register_component']:+.3f}" if r['register_component'] is not None else "N/A"
        cov_s = f"{r['coverage_gap']:.4f}" if r['coverage_gap'] is not None else "N/A"
        tex_lines.append(f"SDG {sdg:2d} & {name} & {raw_s} & {adj_s} & {reg_s} & {cov_s} \\\\")
    tex_lines.extend([
        r"\midrule",
        r"\bottomrule",
        r"\end{tabular}",
    ])
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    out_tex.write_text("\n".join(tex_lines) + "\n", encoding="utf-8")
    log.info("Saved: %s", out_tex)

    # ---- LaTeX macros ----
    num_lines = [
        "% Auto-generated by g_register_decomposition.py — do not edit manually",
        rf"\newcommand{{\MeanRawGap}}{{{mean_raw:.3f}}}",
        rf"\newcommand{{\MeanAdjustedGap}}{{{mean_adj:.3f}}}",
        rf"\newcommand{{\MeanRegisterComponent}}{{{mean_reg:.3f}}}",
    ]
    if rho_cov_adj is not None:
        num_lines.append(rf"\newcommand{{\RhoCovTopic}}{{{rho_cov_adj:.3f}}}")
    if rho_cov_reg is not None:
        num_lines.append(rf"\newcommand{{\RhoCovRegister}}{{{rho_cov_reg:.3f}}}")
    out_num.parent.mkdir(parents=True, exist_ok=True)
    out_num.write_text("\n".join(num_lines) + "\n", encoding="utf-8")
    log.info("Saved: %s", out_num)

    record_fingerprint([out_json], fp, out_json)
    log.info("Decomposition table complete.")


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()

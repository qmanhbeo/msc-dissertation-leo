"""
Register-topic decomposition + iterative convergence diagnostic (plan §6.5.1).

Part 1 — Decomposition table (centrepiece):
  Per-SDG decomposition: raw gap | adjusted gap | register component (raw - adj)
  | coverage gap.

Part 2 — Iterative register check (Appendix E diagnostic):
  Reads canonical G + checkpoint, computes per-SDG gaps at selected iteration
  counts, and emits convergence tables/macros.  No re-training — uses the G
  produced by register_adjust.py.

Inputs (part 1):
  4_outputs/{model}/data/4_3_semantic_gap_distances.json           (raw gaps)
  4_outputs/{model}/data/adjusted/4_3_semantic_gap_distances.json  (adjusted gaps)
  4_outputs/{model}/data/4_2_coverage_document_weighted.json       (coverage gaps)

Inputs (part 2):
  2_data/3b_register/{slug}/{track}/G.npy                          (INLP G)
  2_data/3b_register/{slug}/{track}/checkpoint.json                (iteration data)
  + policy emb, scores, IDs, research centroids, centroid meta

Outputs:
  4_outputs/{model}/data/register_decomposition.json               (JSON)
  4_outputs/{model}/tables/tab_register_decomposition.tex          (decomposition table)
  4_outputs/{model}/tables/tab_iterative_register_check.tex        (convergence table)
  4_outputs/{model}/tables/num_iterative_register_check.tex        (convergence macros)

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

from model_utils import DEFAULT_EMBED_MODEL, N_SDG, SDG_NAMES, resolve_model_alias
from register_utils import (
    compute_gaps_for_directions,
    load_G,
    load_raw_data,
    register_dir,
)
from shared_utils import ensure_canonical_outputs, fingerprint_of, should_skip, record_fingerprint
from shard_pipeline_utils import atomic_write_json, load_json

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

ITERATIVE_N_PER_SDG = 1000


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate register-topic decomposition + iterative diagnostic.")
    p.add_argument("--output-dir", default=str(ROOT / "4_outputs"))
    p.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL, type=resolve_model_alias, help=argparse.SUPPRESS)
    p.add_argument("--overwrite", action="store_true", help=argparse.SUPPRESS)
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Part 1: Decomposition table
# --------------------------------------------------------------------------- #


def _generate_decomposition(
    model: str,
    layout,
    overwrite: bool,
) -> None:
    data_dir = layout.data_dir
    adj_data_dir = data_dir / "adjusted"
    tables_dir = layout.tables_dir

    raw_path = data_dir / "4_3_semantic_gap_distances.json"
    adj_path = adj_data_dir / "4_3_semantic_gap_distances.json"
    cov_path = data_dir / "4_2_coverage_document_weighted.json"
    out_json = data_dir / "register_decomposition.json"
    out_tex = tables_dir / "tab_register_decomposition.tex"

    if not adj_path.exists():
        log.warning("Adjusted semantic gap not found at %s — skipping decomposition.", adj_path)
        return

    fp = fingerprint_of(raw_path, adj_path, cov_path) + "v1"
    if should_skip([out_json], fp, overwrite, out_json):
        log.info("Skipping decomposition — inputs unchanged")
        return

    raw_data = load_json(raw_path)
    adj_data = load_json(adj_path)
    cov_data = load_json(cov_path)

    raw_map = {r["sdg"]: r for r in raw_data["per_sdg"]}
    adj_map = {r["sdg"]: r for r in adj_data["per_sdg"]}
    cov_gap_abs = {f"SDG{i}": cov_data["coverage_gap_hard"][f"SDG{i}"] for i in range(1, N_SDG + 1)}

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

    valid = [r for r in per_sdg if r["raw_gap"] is not None and r["adjusted_gap"] is not None]
    mean_raw = float(np.mean([r["raw_gap"] for r in valid]))
    mean_adj = float(np.mean([r["adjusted_gap"] for r in valid]))
    mean_reg = float(np.mean([r["register_component"] for r in valid]))
    mean_cov = float(np.mean([r["coverage_gap"] for r in valid if r["coverage_gap"] is not None]))

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
    atomic_write_json(out_json, output)
    log.info("Saved: %s", out_json)

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

    record_fingerprint([out_json, out_tex], fp, out_json)
    log.info("Decomposition table complete.")


# --------------------------------------------------------------------------- #
# Part 2: Iterative register convergence diagnostic
# --------------------------------------------------------------------------- #


def _generate_iterative_diagnostic(
    model: str,
    layout,
    overwrite: bool,
) -> None:
    """Read canonical G + checkpoint and compute iterative gap diagnostics.

    No re-training — reads G.npy rows and checkpoint iteration data from
    register_adjust.py, then computes per-SDG gaps at selected iteration counts.
    Folded from 2_appendix/f_register_adjustment.py to canon.
    """
    from scipy.stats import spearmanr

    tables_dir = layout.tables_dir
    out_tex = tables_dir / "tab_iterative_register_check.tex"
    out_num = tables_dir / "num_iterative_register_check.tex"

    g_path = register_dir(model) / "G.npy"
    ckpt_path = register_dir(model) / "checkpoint.json"
    if not g_path.exists() or not ckpt_path.exists():
        log.warning("Register adjust outputs not found at %s — skipping iterative diagnostic.", g_path)
        return

    fp = fingerprint_of(g_path, ckpt_path) + "iter_v1"
    if should_skip([out_tex], fp, overwrite, out_tex):
        log.info("Skipping iterative diagnostic — inputs unchanged")
        return

    G_full = load_G(model)
    ckpt = load_json(ckpt_path)
    iterations_data = ckpt["iterations"]
    n_iters = ckpt["completed_k"]
    log.info("Loaded canonical G: %d iterations, final acc %.4f", n_iters, ckpt.get("final_acc", 0))

    policy_emb, policy_assignments, policy_ids, research_centroids, research_cohesions = (
        load_raw_data(model)
    )
    rng = np.random.default_rng(42)

    iteration_results: list[dict] = []
    for item in iterations_data:
        iteration_results.append({"k": item["k"], "test_acc": item["test_acc"]})

    show_ks = {1, 2, 3, 4, 5, 10, 15, 20, 30, 40, 50, len(iteration_results)}

    for r in iteration_results:
        k = r["k"]
        if k in show_ks:
            k_gaps = compute_gaps_for_directions(
                [np.asarray(G_full[i]) for i in range(k)],
                policy_emb, policy_assignments, policy_ids,
                research_centroids, research_cohesions, rng,
            )
            r["mean_gap"] = round(float(np.mean(list(k_gaps.values()))), 4) if k_gaps else 0.0
            if k == 1:
                r["rho_vs_iter1"] = 1.0
                _iter1_gaps = k_gaps
            else:
                sdgs_common = sorted(set(_iter1_gaps.keys()) & set(k_gaps.keys()))
                rho_k = 0.0
                if len(sdgs_common) >= 3:
                    rho_k, _ = spearmanr(
                        [_iter1_gaps[s] for s in sdgs_common],
                        [k_gaps[s] for s in sdgs_common],
                    )
                r["rho_vs_iter1"] = round(rho_k, 4)
        else:
            r["mean_gap"] = None
            r["rho_vs_iter1"] = None

    rho_iter1_final = iteration_results[-1].get("rho_vs_iter1", 0.0)

    # ---- LaTeX convergence table ----
    tab_lines = [
        "% Auto-generated by g_register_decomposition.py iterative diagnostic — do not edit",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Iteration & Test acc. & Mean gap & Spearman $\rho$ vs iter\,1 \\",
        r"\midrule",
    ]
    for r in iteration_results:
        if r["k"] not in show_ks:
            continue
        mg = f"{r['mean_gap']:.3f}" if r.get("mean_gap") is not None else "—"
        sp = f"{r['rho_vs_iter1']:.3f}" if r.get("rho_vs_iter1") is not None else "—"
        tab_lines.append(f"{r['k']} & {r['test_acc']:.3f} & {mg} & {sp} \\\\")
    tab_lines.extend([r"\bottomrule", r"\end{tabular}"])
    tables_dir.mkdir(parents=True, exist_ok=True)
    out_tex.write_text("\n".join(tab_lines) + "\n", encoding="utf-8")
    log.info("Saved: %s", out_tex)

    # ---- Convergence macros ----
    first_mean_gap = iteration_results[0].get("mean_gap", 0)
    final_mean_gap = iteration_results[-1].get("mean_gap", 0)
    reduction_pct = (first_mean_gap - final_mean_gap) / first_mean_gap * 100 if first_mean_gap > 0 else 0
    rho_at_15 = next((r.get("rho_vs_iter1", 0) for r in iteration_results if r["k"] == 15), 0)
    displayed = [r for r in iteration_results if r.get("rho_vs_iter1") is not None]
    plateau_rho = float(np.mean([r["rho_vs_iter1"] for r in displayed[-5:]])) if len(displayed) >= 5 else rho_iter1_final
    num_lines = [
        "% Auto-generated by g_register_decomposition.py iterative diagnostic",
        rf"\newcommand{{\RegisterIterNPerSdg}}{{{ITERATIVE_N_PER_SDG}}}",
        rf"\newcommand{{\RegisterFirstAcc}}{{{iteration_results[0]['test_acc']:.3f}}}",
        rf"\newcommand{{\RegisterFinalAcc}}{{{iteration_results[-1]['test_acc']:.3f}}}",
        rf"\newcommand{{\RegisterIterFinalK}}{{{len(iteration_results)}}}",
        rf"\newcommand{{\RegisterIterSpearmanRho}}{{{rho_iter1_final:.3f}}}",
        rf"\newcommand{{\RegisterIterMeanGapFirst}}{{{first_mean_gap:.3f}}}",
        rf"\newcommand{{\RegisterIterMeanGapFinal}}{{{final_mean_gap:.3f}}}",
        rf"\newcommand{{\RegisterIterReductionPct}}{{{reduction_pct:.0f}}}",
        rf"\newcommand{{\RegisterIterRhoAtFifteen}}{{{rho_at_15:.3f}}}",
        rf"\newcommand{{\RegisterIterPlateauRho}}{{{plateau_rho:.3f}}}",
    ]
    out_num.write_text("\n".join(num_lines) + "\n", encoding="utf-8")
    log.info("Saved: %s", out_num)

    # ---- Per-SDG final gaps ----
    sdg_words = ["One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight",
                 "Nine", "Ten", "Eleven", "Twelve", "Thirteen", "Fourteen",
                 "Fifteen", "Sixteen", "Seventeen"]
    final_gaps = compute_gaps_for_directions(
        [np.asarray(G_full[i]) for i in range(G_full.shape[0])],
        policy_emb, policy_assignments, policy_ids,
        research_centroids, research_cohesions, rng,
    )
    per_sdg_lines = []
    for sdg_idx in range(N_SDG):
        sdg = sdg_idx + 1
        gap = final_gaps.get(sdg)
        if gap is not None:
            per_sdg_lines.append(
                rf"\newcommand{{\RegIterGapSdg{sdg_words[sdg_idx]}}}{{{gap:.4f}}}"
            )
    if per_sdg_lines:
        existing = out_num.read_text().splitlines() if out_num.exists() else []
        out_num.write_text("\n".join(existing + per_sdg_lines) + "\n", encoding="utf-8")
        log.info("Saved (appended per-SDG macros): %s", out_num)

    record_fingerprint([out_tex, out_num], fp, out_tex)
    log.info("Iterative register diagnostic complete: %d iterations, Spearman rho=%.4f", len(iteration_results), rho_iter1_final)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def run(args: argparse.Namespace) -> None:
    model = args.embed_model
    layout = ensure_canonical_outputs(Path(args.output_dir), model=model)
    _generate_decomposition(model, layout, args.overwrite)
    _generate_iterative_diagnostic(model, layout, args.overwrite)


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()

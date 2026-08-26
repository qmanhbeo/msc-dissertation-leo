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
  4_outputs/{model}/data/semantic_gap_distances_lr.json           (raw gaps)
  4_outputs/{model}/data/adjusted/semantic_gap_distances_lr.json  (adjusted gaps)
  4_outputs/{model}/data/coverage_document_weighted.json       (coverage gaps)

Inputs (part 2):
  2_data/3b_register/{slug}/{track}/G.npy                          (INLP G)
  2_data/3b_register/{slug}/{track}/checkpoint.json                (iteration data)
  + policy emb, scores, IDs, research centroids, centroid meta

Outputs:
  4_outputs/{model}/data/register_decomposition.json               (JSON)
  4_outputs/{model}/tables/tab5_register_decomposition.tex          (decomposition table)
  4_outputs/{model}/tables/tab12_register_check.tex        (convergence table)
  4_outputs/{model}/tables/num12_register_check.tex        (convergence macros)

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

from model_utils import DEFAULT_EMBED_MODEL, N_SDG, SDG_SHORT_NAMES, resolve_model_alias
from register_utils import (
    compute_gaps_for_directions,
    load_G,
    load_raw_data,
    register_dir,
)
from shared_utils import (PERMUTATION_N_RESAMPLES, PERMUTATION_SEED,
                          ensure_canonical_outputs, fingerprint_of, permutation_p,
                          should_skip, record_fingerprint)
from shard_pipeline_utils import atomic_write_json, load_json

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate register-topic decomposition + iterative diagnostic.")
    p.add_argument("--output-dir", default=str(ROOT / "4_outputs"))
    p.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL, type=resolve_model_alias, help=argparse.SUPPRESS)
    p.add_argument("--corpus", default="research", choices=["research", "concept"],
                   help="Research corpus variant. 'concept' reads 4_outputs/{model}/data/concept.")
    p.add_argument("--overwrite", action="store_true", help=argparse.SUPPRESS)
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Part 1: Decomposition table
# --------------------------------------------------------------------------- #


def _generate_decomposition(
    model: str,
    layout,
    overwrite: bool,
    corpus: str = "research",
) -> None:
    data_dir = layout.data_dir / "concept" if corpus == "concept" else layout.data_dir
    adj_data_dir = data_dir / "adjusted"
    tables_dir = layout.tables_dir

    raw_path = data_dir / "semantic_gap_distances_lr.json"
    adj_path = adj_data_dir / "semantic_gap_distances_lr.json"
    cov_path = data_dir / "coverage_document_weighted.json"

    if corpus == "concept":
        out_tex = tables_dir / "tab_concept_reference.tex"
    elif "minilm" in model.lower():
        out_tex = tables_dir / "tab_minilm_reference.tex"
    elif "scibert" in model.lower():
        out_tex = tables_dir / "tab_scibert_reference.tex"
    else:
        out_tex = tables_dir / "tab5_register_decomposition.tex"
    out_json = data_dir / "register_decomposition.json"

    if not adj_path.exists():
        log.warning("Adjusted semantic gap not found at %s — skipping decomposition.", adj_path)
        return

    fp = fingerprint_of(raw_path, adj_path, cov_path) + "v3"
    if should_skip([out_json, out_tex], fp, overwrite, out_json):
        log.info("Skipping decomposition — inputs unchanged")
        return

    raw_data = load_json(raw_path)
    adj_data = load_json(adj_path)
    cov_data = load_json(cov_path)

    raw_map = {r["sdg"]: r for r in raw_data["per_sdg"]}
    adj_map = {r["sdg"]: r for r in adj_data["per_sdg"]}
    cov_gap_abs = {f"SDG{i}": cov_data["coverage_gap_hard"][f"SDG{i}"] for i in range(1, N_SDG + 1)}
    res_profile = cov_data.get("research_profile_hard", {})
    pol_profile = cov_data.get("policy_profile_hard_docweighted", {})

    per_sdg = []
    for sdg in range(1, N_SDG + 1):
        raw_gap = raw_map[sdg].get("semantic_gap")
        adj_gap = adj_map[sdg].get("semantic_gap")
        cov_gap = cov_gap_abs.get(f"SDG{sdg}")
        res_share = res_profile.get(f"SDG{sdg}")
        pol_share = pol_profile.get(f"SDG{sdg}")

        raw_val = float(raw_gap) if raw_gap is not None else None
        adj_val = float(adj_gap) if adj_gap is not None else None
        cov_val = float(cov_gap) if cov_gap is not None else None
        res_pct = float(res_share) * 100.0 if res_share is not None else None
        pol_pct = float(pol_share) * 100.0 if pol_share is not None else None
        signed_dom = (res_pct - pol_pct) if (res_pct is not None and pol_pct is not None) else None

        register_component = None
        if raw_val is not None and adj_val is not None:
            register_component = round(raw_val - adj_val, 6)

        per_sdg.append({
            "sdg": sdg,
            "raw_gap": raw_val,
            "adjusted_gap": adj_val,
            "register_component": register_component,
            "coverage_gap": cov_val,
            "research_pct": res_pct,
            "policy_pct": pol_pct,
            "signed_dom": signed_dom,
            "name": SDG_SHORT_NAMES[sdg],
        })

    valid = [r for r in per_sdg if r["raw_gap"] is not None and r["adjusted_gap"] is not None]
    mean_raw = float(np.mean([r["raw_gap"] for r in valid]))
    mean_adj = float(np.mean([r["adjusted_gap"] for r in valid]))
    mean_reg = float(np.mean([r["register_component"] for r in valid]))
    mean_cov = float(np.mean([r["coverage_gap"] for r in valid if r["coverage_gap"] is not None]))
    mean_res = float(np.mean([r["research_pct"] for r in valid if r["research_pct"] is not None]))
    mean_pol = float(np.mean([r["policy_pct"] for r in valid if r["policy_pct"] is not None]))
    mean_sdom = float(np.mean([r["signed_dom"] for r in valid if r["signed_dom"] is not None]))
    if abs(mean_sdom) < 1e-9:
        mean_sdom = 0.0

    from scipy import stats  # noqa: F401  (kept for parity with prior versions)
    valid_corr = [r for r in per_sdg if all(v is not None for v in [r["raw_gap"], r["adjusted_gap"], r["coverage_gap"], r["register_component"]])]
    if len(valid_corr) >= 3:
        cov_arr = np.array([r["coverage_gap"] for r in valid_corr])
        adj_arr = np.array([r["adjusted_gap"] for r in valid_corr])
        reg_arr = np.array([r["register_component"] for r in valid_corr])
        rho_cov_adj, p_cov_adj = permutation_p(cov_arr, adj_arr, kind="spearman")
        rho_cov_reg, p_cov_reg = permutation_p(cov_arr, reg_arr, kind="spearman")
    else:
        rho_cov_adj = p_cov_adj = rho_cov_reg = p_cov_reg = None

    output = {
        "embedding_model": model,
        "corpus": corpus,
        "note": "register_component = raw_gap - adjusted_gap. Positive means register divergence; negative means register similarity masking topic divergence. signed_dom = research share - policy share (percentage points).",
        "p_value": {
            "method": "monte_carlo_permutation",
            "n_resamples": PERMUTATION_N_RESAMPLES,
            "seed": PERMUTATION_SEED,
        },
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
        r"\begin{tabular}{llrrrrrrr}",
        r"\toprule",
        r"SDG & Description & \multicolumn{3}{c}{Semantic gaps} & \multicolumn{4}{c}{Coverage predictors} \\",
        r"\cmidrule(lr){3-5} \cmidrule(lr){6-9}",
        r"& & Raw & Adj. & Reg. & Cov. & SDom & Res \% & Pol \% \\",
        r"\midrule",
    ]
    for r in per_sdg:
        sdg = r["sdg"]
        name = SDG_SHORT_NAMES[sdg].replace("&", r"\&")
        raw_s = f"{r['raw_gap']:.3f}" if r['raw_gap'] is not None else "N/A"
        adj_s = f"{r['adjusted_gap']:.3f}" if r['adjusted_gap'] is not None else "N/A"
        reg_s = f"{r['register_component']:+.3f}" if r['register_component'] is not None else "N/A"
        cov_s = f"{r['coverage_gap'] * 100.0:.1f}" if r['coverage_gap'] is not None else "N/A"
        sdom_s = f"{r['signed_dom']:+.1f}" if r['signed_dom'] is not None else "N/A"
        res_s = f"{r['research_pct']:.1f}" if r['research_pct'] is not None else "N/A"
        pol_s = f"{r['policy_pct']:.1f}" if r['policy_pct'] is not None else "N/A"
        tex_lines.append(f"{sdg:2d} & {name} & {raw_s} & {adj_s} & {reg_s} & {cov_s} & {sdom_s} & {res_s} & {pol_s} \\\\")
    tex_lines.extend([
        r"\midrule",
        rf"Mean & --- & {mean_raw:.3f} & {mean_adj:.3f} & {mean_reg:+.3f} & {mean_cov * 100.0:.1f} & {mean_sdom:+.1f} & {mean_res:.1f} & {mean_pol:.1f} \\",
        r"\bottomrule",
        r"\end{tabular}",
    ])
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    out_tex.write_text("\n".join(tex_lines) + "\n", encoding="utf-8")
    log.info("Saved: %s", out_tex)

    record_fingerprint([out_json, out_tex], fp, out_json)
    log.info("Decomposition table complete (%s/%s).", model, corpus)


# --------------------------------------------------------------------------- #
# Part 2: Iterative register convergence diagnostic
# --------------------------------------------------------------------------- #


SHOW_KS_BASE = {1, 2, 3, 4, 5, 10, 15, 20, 30, 40, 50, 60, 70}


def _compute_iterative_rows(model: str, track=None):
    """Compute per-iteration diagnostic rows for one INLP track.

    Returns ``(iteration_results, summary, G_full, raw_data)``.  Each result dict
    has keys ``k``, ``test_acc``, ``mean_gap``, ``rho_vs_raw``, ``shown``.
    ``rho_vs_raw`` is the Spearman rank correlation of the per-SDG gap vector at
    iteration ``k`` versus the *raw* (iteration 0, un-projected) gap vector — not
    versus iteration 1.  ``summary`` holds the final / plateau / iteration-1
    correlations and the gap-reduction percentage.
    """
    from scipy.stats import spearmanr

    g_dir = register_dir(model, track)
    g_path = g_dir / "G.npy"
    ckpt_path = g_dir / "checkpoint.json"
    if not g_path.exists() or not ckpt_path.exists():
        raise FileNotFoundError(f"Register adjust outputs not found at {g_dir}")
    G_full = load_G(model, track)
    ckpt = load_json(ckpt_path)
    iterations_data = ckpt["iterations"]
    n_iters = ckpt["completed_k"]
    policy_emb, policy_assignments, policy_ids, research_centroids, research_cohesions = load_raw_data(model)
    rng = np.random.default_rng(PERMUTATION_SEED)

    iteration_results = [{"k": it["k"], "test_acc": it["test_acc"]} for it in iterations_data]
    show_ks = set(SHOW_KS_BASE) | {len(iteration_results)}
    raw_gaps = compute_gaps_for_directions(
        [], policy_emb, policy_assignments, policy_ids, research_centroids, research_cohesions, rng
    )
    for r in iteration_results:
        k = r["k"]
        if k in show_ks:
            k_gaps = compute_gaps_for_directions(
                [np.asarray(G_full[i]) for i in range(k)],
                policy_emb, policy_assignments, policy_ids, research_centroids, research_cohesions, rng,
            )
            r["mean_gap"] = round(float(np.mean(list(k_gaps.values()))), 4) if k_gaps else 0.0
            sdgs_common = sorted(set(raw_gaps.keys()) & set(k_gaps.keys()))
            rho_k = 0.0
            if len(sdgs_common) >= 3:
                rho_k, _ = spearmanr([raw_gaps[s] for s in sdgs_common], [k_gaps[s] for s in sdgs_common])
            r["rho_vs_raw"] = round(float(rho_k), 4)
            r["shown"] = True
        else:
            r["mean_gap"] = None
            r["rho_vs_raw"] = None
            r["shown"] = False

    displayed = [r for r in iteration_results if r["rho_vs_raw"] is not None]
    final_rho = displayed[-1]["rho_vs_raw"]
    rho_at_15 = next((r["rho_vs_raw"] for r in displayed if r["k"] == 15), final_rho)
    plateau_rho = float(np.mean([r["rho_vs_raw"] for r in displayed[-5:]])) if len(displayed) >= 5 else final_rho
    iter1_rho = next((r["rho_vs_raw"] for r in displayed if r["k"] == 1), 1.0)
    first_mean_gap = iteration_results[0]["mean_gap"]
    final_mean_gap = iteration_results[-1]["mean_gap"]
    reduction_pct = (first_mean_gap - final_mean_gap) / first_mean_gap * 100 if first_mean_gap else 0
    summary = {
        "n_iters": n_iters, "final_rho": final_rho, "rho_at_15": rho_at_15,
        "plateau_rho": plateau_rho, "iter1_rho": iter1_rho,
        "first_mean_gap": first_mean_gap, "final_mean_gap": final_mean_gap,
        "reduction_pct": reduction_pct,
    }
    raw_data = (policy_emb, policy_assignments, policy_ids, research_centroids, research_cohesions)
    return iteration_results, summary, G_full, raw_data


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
    tables_dir = layout.tables_dir
    out_tex = tables_dir / "tab12_register_check.tex"
    out_num = tables_dir / "num12_register_check.tex"

    try:
        iteration_results, summary, G_full, raw_data = _compute_iterative_rows(model)
    except (FileNotFoundError, RuntimeError, ValueError) as e:
        log.warning("Register adjust outputs unavailable for %s — skipping iterative diagnostic: %s", model, e)
        return

    g_dir = register_dir(model)
    fp = fingerprint_of(g_dir / "G.npy", g_dir / "checkpoint.json") + "iter_v2"
    if should_skip([out_tex], fp, overwrite, out_tex):
        log.info("Skipping iterative diagnostic — inputs unchanged")
        return

    policy_emb, policy_assignments, policy_ids, research_centroids, research_cohesions = raw_data
    rng = np.random.default_rng(PERMUTATION_SEED)

    # ---- LaTeX convergence table ----
    tab_lines = [
        "% Auto-generated by g_register_decomposition.py iterative diagnostic — do not edit",
        "% seed: 42 (PERMUTATION_SEED) — policy per-document cap sampling",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"It. & Test acc. & Mean gap & p vs raw \\",
        r"\midrule",
    ]
    for r in iteration_results:
        if not r.get("shown"):
            continue
        mg = f"{r['mean_gap']:.3f}" if r.get("mean_gap") is not None else "—"
        sp = f"{r['rho_vs_raw']:.3f}" if r.get("rho_vs_raw") is not None else "—"
        tab_lines.append(f"{r['k']} & {r['test_acc']:.3f} & {mg} & {sp} \\\\")
    tab_lines.extend([r"\bottomrule", r"\end{tabular}"])
    tables_dir.mkdir(parents=True, exist_ok=True)
    out_tex.write_text("\n".join(tab_lines) + "\n", encoding="utf-8")
    log.info("Saved: %s", out_tex)

    # ---- Convergence macros ----
    num_lines = [
        "% Auto-generated by g_register_decomposition.py iterative diagnostic",
        "% seed: 42 (PERMUTATION_SEED) — policy per-document cap sampling",
        rf"\newcommand{{\RegisterFirstAcc}}{{{iteration_results[0]['test_acc']:.3f}}}",
        rf"\newcommand{{\RegisterFinalAcc}}{{{iteration_results[-1]['test_acc']:.3f}}}",
        rf"\newcommand{{\RegisterIterFinalK}}{{{len(iteration_results)}}}",
        rf"\newcommand{{\RegisterIterSpearmanRho}}{{{summary['final_rho']:.3f}}}",
        rf"\newcommand{{\RegisterIterRhoAtOne}}{{{summary['iter1_rho']:.3f}}}",
        rf"\newcommand{{\RegisterIterRhoAtFifteen}}{{{summary['rho_at_15']:.3f}}}",
        rf"\newcommand{{\RegisterIterPlateauRho}}{{{summary['plateau_rho']:.3f}}}",
        rf"\newcommand{{\RegisterIterMeanGapFirst}}{{{summary['first_mean_gap']:.3f}}}",
        rf"\newcommand{{\RegisterIterMeanGapFinal}}{{{summary['final_mean_gap']:.3f}}}",
        rf"\newcommand{{\RegisterIterReductionPct}}{{{summary['reduction_pct']:.0f}}}",
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
    log.info("Iterative register diagnostic complete: %d iterations, Spearman rho vs raw=%.4f", len(iteration_results), summary['final_rho'])


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def run(args: argparse.Namespace) -> None:
    model = args.embed_model
    corpus = getattr(args, "corpus", "research")
    layout = ensure_canonical_outputs(Path(args.output_dir), model=model)
    _generate_decomposition(model, layout, args.overwrite, corpus)
    if corpus != "concept":
        _generate_iterative_diagnostic(model, layout, args.overwrite)


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()

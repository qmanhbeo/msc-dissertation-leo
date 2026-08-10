"""
Distributional semantic-gap companion to the H1a--H1d correlation grid.

Re-runs the H1a--H1d coverage-predictor vs semantic-gap correlation tests with
each distribution-aware semantic gap (from g_distributional_gap.py) in place of
the centroid gap. Same test as the canonical grid: Spearman rho with two-sided
Monte Carlo permutation p-values (100,000 resamples, seed 42, shared_utils
permutation_p). It answers whether the coverage--framing association survives
the mean-direction estimate, and exposes where distribution shape diverges from
it (e.g. research coverage on shape-aware metrics, Sinkhorn's negative H1d).

Inputs:
  - coverage predictors: 4_outputs/{model}/data/coverage_document_weighted.json
    (via 2_coverage_semantic_interaction._load_coverage_predictors)
  - canonical adjusted centroid gaps: 4_outputs/{model}/data/adjusted/
    semantic_gap_distances_lr.json (via _adj_gaps_for)
  - distributional gaps: 4_outputs/not_in_replay/distributional/{model}/adjusted/
    g_distributional_gap_summary.json (committed artifact of the opt-in G step)

Outputs:
  4_outputs/{model}/data/distributional_h1_interaction.json
  4_outputs/{model}/adjusted/tables/tab14_distributional_h1.tex
  4_outputs/{model}/adjusted/tables/num18_distributional_h1.tex

Run from project root:
  python main.py --appendix-g-distributional-h1 --overwrite
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = ROOT / "1_code"
ANALYSIS_ROOT = Path(__file__).resolve().parents[1]
MAIN_TEXT = ANALYSIS_ROOT / "1_main_text"
SHARED_DIR = ANALYSIS_ROOT / "0_shared"
for path in (CODE_ROOT, MAIN_TEXT, SHARED_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from model_utils import (
    DEFAULT_EMBED_MODEL,
    DEFAULT_OUTPUT_ROOT,
    model_slug,
    resolve_model_alias,
)
from shared_utils import (
    PERMUTATION_N_RESAMPLES,
    PERMUTATION_SEED,
    ensure_dissertation_outputs,
    fingerprint_of,
    permutation_p,
    record_fingerprint,
    should_skip,
)

# Reuse the canonical H1 data loaders so the predictors and the canonical
# adjusted gaps are byte-identical to the main grid (same pattern as
# j1_raw_value_correlation.py).
_cov_inter_spec = importlib.util.spec_from_file_location(
    "cov_inter_main", MAIN_TEXT / "2_coverage_semantic_interaction.py"
)
cov_inter = importlib.util.module_from_spec(_cov_inter_spec)
_cov_inter_spec.loader.exec_module(cov_inter)

_load_coverage_predictors = cov_inter._load_coverage_predictors
_adj_gaps_for = cov_inter._adj_gaps_for
h1_grid_input_paths = cov_inter.h1_grid_input_paths

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

SCRIPT_VERSION = "g_dist_h1_v1"

# Regression tripwires from the committed manuscript macros
# (num5_register_decomposition.tex: \RhoCovTopic 0.544 / \RhoCovTopicP 0.025;
# limitations prose: leave-one-out SDG4 rho = 0.597, p = 0.016). If the canonical
# inputs ever change, the GATE below fails loudly instead of silently feeding a
# stale distributional table.
GATE_CANON_N17 = {"rho": 0.544, "p": 0.025}
GATE_CANON_N16 = {"rho": 0.597, "p": 0.016}
GATE_TOL = 0.001
CENTROID_RANK_MIN = 0.999

# Methods for the manuscript table, in the order they appear.
TABLE_METHODS = [
    ("sliced_wasserstein", "Sliced Wasserstein"),
    ("linear_mmd", "Linear MMD"),
    ("energy_distance", "Energy distance"),
    ("rbf_mmd", "Squared RBF MMD"),
    ("exact_emd", "Exact EMD"),
    ("chamfer_symmetric", "Chamfer"),
    ("hausdorff_modified", "Modified Hausdorff"),
    ("gaussian_w2", "Gaussian-2-Wasserstein"),
    ("sinkhorn", "Sinkhorn"),
    ("c2st", "C2ST AUC"),
    ("grassmann", "Grassmann"),
]

# Centroid references: validation rows in the JSON only, not the table.
CENTROID_REFS = ["centroid_gap_canonical", "centroid_gap_sampled", "centroid_gap_full_corpus"]

HYPOTHESES = ["covgap", "dominance", "research", "policy"]
SDGS = list(range(1, 18))
MASK_N16 = np.array([s != 4 for s in SDGS], dtype=bool)


def _rho_dict(x: np.ndarray, y: np.ndarray, mask: np.ndarray) -> dict | None:
    if x[mask].size < 3 or y[mask].size < 3:
        return None
    rho, p = permutation_p(x[mask], y[mask], kind="spearman")
    return {"rho": round(float(rho), 6), "p": round(float(p), 6)}


def _sig_stars(p: float) -> str:
    if p < 0.001:
        return "$^{***}$"
    if p < 0.01:
        return "$^{**}$"
    if p < 0.05:
        return "$^{*}$"
    if p < 0.10:
        return "$^{\\dagger}$"
    return ""


def _fmt_rho(d: dict | None) -> str:
    if d is None:
        return "--"
    return f"{d['rho']:+.3f}{_sig_stars(d['p'])}"


def _gate_canonical(x: np.ndarray, adj: np.ndarray) -> None:
    """The canonical centroid grid must reproduce the committed manuscript values."""
    for n_label, mask, target in (
        ("n=17", np.ones(17, dtype=bool), GATE_CANON_N17),
        ("n=16 (SDG 4 excluded)", MASK_N16, GATE_CANON_N16),
    ):
        rho, p = permutation_p(x[mask], adj[mask], kind="spearman")
        if abs(rho - target["rho"]) > GATE_TOL or abs(p - target["p"]) > GATE_TOL:
            raise RuntimeError(
                f"GATE FAILED: canonical covgap vs adjusted centroid gap {n_label} = "
                f"rho {rho:.3f}, p {p:.3f}; expected rho {target['rho']:.3f}, p "
                f"{target['p']:.3f}. The distributional H1 table would be stale; "
                "refuse to emit."
            )
    log.info("GATE passed: canonical grid reproduces committed values (n=17 and n=16).")


def _gate_centroid_rank(canon_gaps: np.ndarray, canonical_adj: np.ndarray) -> None:
    """The summary's canonical centroid reference must rank-match the canonical gaps."""
    rho = stats.spearmanr(canon_gaps, canonical_adj)[0]
    if rho < CENTROID_RANK_MIN:
        raise RuntimeError(
            f"GATE FAILED: summary centroid_gap_canonical rank agreement with the "
            f"canonical adjusted gaps = {rho:.4f} (< {CENTROID_RANK_MIN}). The summary "
            "belongs to a different input generation; refuse to emit."
        )
    log.info("GATE passed: summary canonical centroid ranks match canonical gaps (rho=%.4f).", rho)


def write_tex(path: Path, rows: list[dict]) -> None:
    """Self-contained table (same style as tab13_distributional_gap.tex)."""
    lines = [
        "% Auto-generated by 1_code/7_main_analysis/1_main_text/g_distributional_h1_correlation.py — do not edit manually",
        "% Spearman rho; p-values: two-sided Monte Carlo permutation (100,000 resamples, seed 42), n = 17 SDGs.",
        r"\begin{table}[ht]",
        r"\centering",
        r"\small",
        r"\caption{H1a--H1d coverage-predictor vs semantic-gap Spearman correlations under distribution-aware semantic gaps (adjusted embeddings). Distributional gaps replace the centroid gap as the outcome vector; predictors and test are identical to Table~\ref{tab:interaction}.}",
        r"\label{tab:distributional-h1}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r" & H1a & H1b & H1c & H1d \\",
        r" & Cov.\ gap & Dominance & Research cov.\ & Policy cov.\ \\",
        r"\midrule",
    ]
    for key, name in TABLE_METHODS:
        row = next(r for r in rows if r["key"] == key)
        cells = [_fmt_rho(row["hypotheses"][h]["n17"]) for h in HYPOTHESES]
        lines.append(f"{name} & " + " & ".join(cells) + r" \\")
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\par\smallskip\footnotesize\emph{Notes:} Spearman $\rho$ with two-sided "
        r"Monte Carlo permutation $p$-values (100{,}000 resamples, seed 42), $n = 17$ "
        r"SDGs; $^{***}p<0.001$, $^{**}p<0.01$, $^{*}p<0.05$, $^{\dagger}p<0.10$. "
        r"Excluding SDG~4 ($n = 16$) leaves every conclusion unchanged; the full "
        r"$n = 16$ grid is in the analysis JSON (distributional\_h1\_interaction.json).",
        r"\end{table}",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_num_macros(path: Path, rows: list[dict]) -> None:
    """Headline values quoted in Appendix D prose.

    LaTeX cannot tokenise control words containing digits (\\DistH1aRhoSwd is read
    as \\DistH + text), so numbers are spelled out per repo convention (HOneA,
    matching \\HOneACovgapAdjPositiveCount etc.).
    """
    by_key = {r["key"]: r for r in rows}
    macros = {
        "DistHOneARhoSwd": ("sliced_wasserstein", "covgap", "n17", "rho"),
        "DistHOneAPSwd": ("sliced_wasserstein", "covgap", "n17", "p"),
        "DistHOneARhoLmmd": ("linear_mmd", "covgap", "n17", "rho"),
        "DistHOneAPLmmd": ("linear_mmd", "covgap", "n17", "p"),
        "DistHOneARhoEnergy": ("energy_distance", "covgap", "n17", "rho"),
        "DistHOneAPEnergy": ("energy_distance", "covgap", "n17", "p"),
        "DistHOneARhoRbf": ("rbf_mmd", "covgap", "n17", "rho"),
        "DistHOneAPRbf": ("rbf_mmd", "covgap", "n17", "p"),
        "DistHOneCRhoSwd": ("sliced_wasserstein", "research", "n17", "rho"),
        "DistHOneCPSwd": ("sliced_wasserstein", "research", "n17", "p"),
        "DistHOneCRhoEnergy": ("energy_distance", "research", "n17", "rho"),
        "DistHOneCPEnergy": ("energy_distance", "research", "n17", "p"),
        "DistHOneCRhoRbf": ("rbf_mmd", "research", "n17", "rho"),
        "DistHOneCPRbf": ("rbf_mmd", "research", "n17", "p"),
        "DistHOneDRhoSinkhorn": ("sinkhorn", "policy", "n17", "rho"),
        "DistHOneDPSinkhorn": ("sinkhorn", "policy", "n17", "p"),
    }
    lines = [
        "% Auto-generated by g_distributional_h1_correlation.py — do not edit manually",
        "% Values quoted in Appendix D prose (two-sided MC permutation p, 100k, seed 42, n=17).",
    ]
    for name, (key, hyp, scope, field) in macros.items():
        val = by_key[key]["hypotheses"][hyp][scope][field]
        lines.append(rf"\newcommand{{\{name}}}{{{val:.3f}}}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="H1a--H1d grid under distribution-aware semantic gaps.")
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_ROOT))
    p.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL, type=resolve_model_alias, help=argparse.SUPPRESS)
    p.add_argument("--overwrite", action="store_true", help=argparse.SUPPRESS)
    return p.parse_args()


def run(args: argparse.Namespace) -> None:
    root = Path(args.output_dir)
    model = args.embed_model

    summary_path = (
        ROOT / "4_outputs" / "not_in_replay" / "distributional" / model_slug(model)
        / "adjusted" / "g_distributional_gap_summary.json"
    )
    if not summary_path.exists():
        raise FileNotFoundError(
            f"Distributional summary not found: {summary_path}. Run the opt-in "
            "G step first: python main.py --appendix-g-distributional --embeddings adjusted --overwrite"
        )

    layout = ensure_dissertation_outputs(root, subdir="main", model=model)
    import dataclasses
    adj_tables = layout.root / "adjusted" / "tables"
    adj_tables.mkdir(parents=True, exist_ok=True)
    layout = dataclasses.replace(layout, tables_dir=adj_tables)

    out_json = layout.data_dir / "distributional_h1_interaction.json"
    out_tex = adj_tables / "tab14_distributional_h1.tex"
    out_num = adj_tables / "num18_distributional_h1.tex"
    outputs = [out_json, out_tex, out_num]

    fp = fingerprint_of(*h1_grid_input_paths(root), summary_path) + SCRIPT_VERSION
    if should_skip(outputs, fp, args.overwrite, out_json):
        log.info("Skipping %s -- inputs unchanged", out_json)
        return

    cov = _load_coverage_predictors(root, model, "canon")
    if cov is None:
        raise RuntimeError("Missing canonical coverage predictors (coverage_document_weighted.json)")
    adj = _adj_gaps_for("LR", root, model, "canon")
    if adj is None:
        raise RuntimeError("Missing canonical adjusted gaps (adjusted/semantic_gap_distances_lr.json)")

    x = {h: np.array([cov[h][s] for s in SDGS], dtype=float) for h in HYPOTHESES}
    adj_arr = np.array([adj[s] for s in SDGS], dtype=float)

    _gate_canonical(x["covgap"], adj_arr)

    summary = load_summary(summary_path)
    methods = summary["methods"]

    def gaps_of(key: str) -> np.ndarray:
        g = methods[key]["gap_by_sdg"]
        return np.array([g[f"SDG{s}"] for s in SDGS], dtype=float)

    _gate_centroid_rank(gaps_of("centroid_gap_canonical"), adj_arr)

    rows = []
    for key, name in TABLE_METHODS + [(k, k) for k in CENTROID_REFS]:
        g = gaps_of(key)
        hyp_rows = {}
        for h in HYPOTHESES:
            hyp_rows[h] = {
                "n17": _rho_dict(x[h], g, np.ones(17, dtype=bool)),
                "n16": _rho_dict(x[h], g, MASK_N16),
            }
        rows.append({
            "key": key,
            "name": name,
            "is_reference": key in CENTROID_REFS,
            "rank_vs_canonical": round(float(stats.spearmanr(g, adj_arr)[0]), 6),
            "hypotheses": hyp_rows,
        })

    payload = {
        "embedding_model": model,
        "note": "H1a--H1d correlation grid re-run with distribution-aware semantic gaps (adjusted embeddings) in place of the centroid gap.",
        "test": {
            "method": "spearman",
            "p_value": "two-sided Monte Carlo permutation",
            "n_resamples": PERMUTATION_N_RESAMPLES,
            "seed": PERMUTATION_SEED,
        },
        "gates": {
            "canonical_n17": GATE_CANON_N17,
            "canonical_n16": GATE_CANON_N16,
            "centroid_rank_min": CENTROID_RANK_MIN,
        },
        "n17_all_sdgs": True,
        "n16_sdg4_excluded": True,
        "methods": rows,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log.info("Saved: %s", out_json)

    write_tex(out_tex, rows)
    log.info("Saved: %s", out_tex)
    write_num_macros(out_num, rows)
    log.info("Saved: %s", out_num)

    record_fingerprint(outputs, fp, out_json)

    print(f"\nDistributional H1 grid: {len(TABLE_METHODS)} methods + {len(CENTROID_REFS)} centroid refs")
    for r in rows:
        h = r["hypotheses"]["covgap"]["n17"]
        print(f"  {r['key']:24s} H1a rho={h['rho']:+.4f} p={h['p']:.4f}  rank~canon={r['rank_vs_canonical']:+.4f}")


def load_summary(path: Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()

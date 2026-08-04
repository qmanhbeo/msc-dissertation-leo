"""
Test H25: do coverage measures predict the within-SDG semantic gap across SDGs?

H25 (headline hypothesis):
  SDGs with the highest research attention will show the largest within-SDG semantic gaps —
  i.e. the SDGs where research engages most are precisely where research and policy diverge
  most strongly in framing within that SDG.

  Four coverage predictors are each correlated with semantic_gap (within-SDG semantic
  divergence), so that the paper's claimed "coverage gap" test is actually surfaced rather
  than only the research-proportion test:
    (a) research coverage  — research_proportion (SDG coverage in research corpus)
    (b) policy coverage    — policy_proportion (SDG coverage in policy corpus)
    (c) coverage gap       — coverage_gap_abs (|research% - policy%|)
    (d) dominance          — research_dominance (research% - policy%, signed)

  A POSITIVE correlation for (a) = H25 supported: more research attention → more divergence.
  The hypothesis labels this a "negative correlation" between coverage and semantic
  *similarity* — equivalent to a positive correlation with semantic *gap*.

   All four predictors are reported, each with a canonical result (all observed SDGs),
   a leave-one-out sensitivity test excluding SDG 4 (reported on BOTH the raw gap and the
   register-adjusted topic gap, and wired into the interaction section), plus replications
   across encoders and assignment methods via the H1 config grid. The primary test for H25 is (a).

Statistics:
  Pearson r (parametric, assumes linear relationship) and Spearman ρ (rank correlation,
  non-parametric) are both reported. With only 17 data points (one per SDG), both tests
  have low power. We report p-values but interpret them cautiously — with n=17, even
  strong trends may not reach p < 0.05. The qualitative pattern (top-5 SDGs in scatter)
  is the primary evidence.

  ASSUMPTION (A-STAT): With 17 SDGs, correlation statistics have limited power. A nominally
  non-significant result does not rule out a real pattern; the direction and magnitude of
  the correlation are the primary evidence, not the p-value.

   ASSUMPTION (A-SDG4): SDG 4's research proportion may be inflated due to ML "learning"
   terminology aligning with the Education centroid (not genuine SDG 4 research engagement).
   A leave-one-out correlation check (SDG 4 removed) is reported on the raw gap (null
   hypothesis) and on the register-adjusted topic gap, and is cited in the interaction section.

Inputs:
   4_outputs/main/data/coverage_document_weighted.json            per-SDG research + policy profiles (doc-weighted)
   4_outputs/main/data/semantic_gap_distances_lr.json                per-SDG semantic gap (segment_cap=50)



    4_outputs/main/data/interaction_h25.json     H25 correlation results
   4_outputs/main/data/interaction_scatter_data.csv               per-SDG data table for plotting (SDG, research%, policy%,
                                   coverage_gap, semantic_gap, semantic_similarity)
   4_outputs/main/tables/*.tex            generated LaTeX macros/tables

Run from project root (after the canonical coverage and semantic outputs exist):
    python 1_code/7_main_analysis/1_main_text/2_coverage_semantic_interaction.py
"""

import csv
import json
import logging
import math
import numpy as np
import argparse
import sys
from pathlib import Path
from scipy import stats

ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = ROOT / "1_code"
ANALYSIS_ROOT = Path(__file__).resolve().parents[1]
SHARED_DIR = ANALYSIS_ROOT / "0_shared"
for path in (CODE_ROOT, SHARED_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from shared_utils import ensure_canonical_outputs, fingerprint_of, require_output_files, should_skip, record_fingerprint
from model_utils import DEFAULT_EMBED_MODEL, DEFAULT_OUTPUT_ROOT, N_SDG, output_dir_for_model, scored_dir_for_model, resolve_model_alias
from shard_pipeline_utils import load_json

# Gap loaders for the cross-config H1 grid (reuse the consolidated register-correlation
# table's per-config raw/adjusted gap readers so the grid stays consistent with it).
from h1_register_correlation_table import (
    _lr_raw_gaps, _lr_adj_gaps,
    _zs_raw_gaps, _zs_adj_gaps,
    _mlp_raw_gaps, _mlp_adj_gaps,
    _concept_raw_gaps, _concept_adj_gaps,
    _concept_mlp_raw_gaps, _concept_mlp_adj_gaps,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def pearson_and_spearman(x: np.ndarray, y: np.ndarray, label: str) -> dict:
    """Compute Pearson r and Spearman ρ between x and y, return as dict."""
    assert len(x) == len(y), f"Length mismatch: {len(x)} vs {len(y)}"
    if len(x) < 3:
        raise ValueError(f"{label}: need at least 3 observations, got {len(x)}")
    r, r_p   = stats.pearsonr(x, y)
    rho, s_p = stats.spearmanr(x, y)
    result = {
        "n": len(x),
        "pearson_r": round(float(r), 6),
        "pearson_p": round(float(r_p), 6),
        "spearman_rho": round(float(rho), 6),
        "spearman_p": round(float(s_p), 6),
    }
    log.info(
        "  %-55s  Pearson r=%.3f (p=%.3f)  Spearman ρ=%.3f (p=%.3f)",
        label, r, r_p, rho, s_p
    )
    return result


def correlation_or_skip(
    x: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    label: str,
) -> dict:
    sdgs = [i + 1 for i, keep in enumerate(mask) if keep]
    if int(mask.sum()) < 3:
        log.warning("  %-55s  skipped (n=%d)", label, int(mask.sum()))
        return {
            "n": int(mask.sum()),
            "sdgs": sdgs,
            "skipped": True,
            "reason": "fewer_than_3_valid_sdgs",
        }
    result = pearson_and_spearman(x[mask], y[mask], label)
    result["sdgs"] = sdgs
    result["skipped"] = False
    return result


def compute_four_tests(
    x_research: np.ndarray,
    x_policy: np.ndarray,
    x_covgap: np.ndarray,
    x_dominance: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
) -> dict:
    """Correlate each of four coverage predictors with the semantic gap.

    Returns a dict with keys: research, policy, covgap, dominance.
    """
    return {
        "research": correlation_or_skip(x_research, y, mask, "research coverage vs semantic gap"),
        "policy": correlation_or_skip(x_policy, y, mask, "policy coverage vs semantic gap"),
        "covgap": correlation_or_skip(x_covgap, y, mask, "coverage gap vs semantic gap"),
        "dominance": correlation_or_skip(x_dominance, y, mask, "policy-research dominance vs semantic gap"),
    }


# ---------------------------------------------------------------------------
# Cross-config H1 register-correlation grid
# ---------------------------------------------------------------------------
# Each row = one encoder--classifier config; each hypothesis (H1a--H1d) is a
# block of 9 config rows. Columns = Spearman rho of the predictor with the raw
# gap, the adjusted (topic) gap, and the register component (raw - adjusted).
# Coverage predictors are read from the per-config coverage JSON (LR coverage is
# reused for MLP/ZS rows, matching h1_register_correlation_table.py); gap vectors
# use the same per-config raw/adjusted readers imported above.
_H1_CONFIGS = [
    ("MPNet LR", "all-mpnet-base-v2", "LR", "canon"),
    ("MPNet MLP", "all-mpnet-base-v2", "MLP", "canon"),
    ("MPNet ZS", "all-mpnet-base-v2", "ZS", "canon"),
    ("MiniLM LR", "all-MiniLM-L6-v2", "LR", "subset"),
    ("MiniLM MLP", "all-MiniLM-L6-v2", "MLP", "subset"),
    ("SciBERT LR", "allenai/scibert_scivocab_uncased", "LR", "subset"),
    ("SciBERT MLP", "allenai/scibert_scivocab_uncased", "MLP", "subset"),
    ("Concept LR", "all-mpnet-base-v2", "LR", "concept"),
    ("Concept MLP", "all-mpnet-base-v2", "MLP", "concept"),
]

_H1_GROUPS = [
    ("H1a. Coverage gap $\\leftrightarrow$ Semantic gap", "covgap"),
    ("H1b. Policy--research dominance $\\leftrightarrow$ Semantic gap", "dominance"),
    ("H1c. Research coverage $\\leftrightarrow$ Semantic gap", "research"),
    ("H1d. Policy coverage $\\leftrightarrow$ Semantic gap", "policy"),
]


_H1_GAP_SUFFIX = {"LR": "lr", "MLP": "mlp", "ZS": "zeroshot"}


def h1_grid_input_paths(root: Path) -> list[Path]:
    """Every file the H1 config grid reads, derived from _H1_CONFIGS itself.

    The canonical fingerprint in run() covers only the MPNet keyword inputs, so
    without this the grid would not re-derive when a per-config coverage or gap
    input changes -- in particular the data/concept/ inputs behind the Concept
    rows. Shared with j1_raw_value_correlation.py so both grids fingerprint the
    same set.
    """
    paths: list[Path] = []
    for _label, model, method, corpus in _H1_CONFIGS:
        base = output_dir_for_model(model, root=root) / "data"
        if corpus == "concept":
            base = base / "concept"
        suffix = _H1_GAP_SUFFIX[method]
        paths.append(base / "coverage_document_weighted.json")
        paths.append(base / f"semantic_gap_distances_{suffix}.json")
        paths.append(base / "adjusted" / f"semantic_gap_distances_{suffix}.json")
    # De-duplicate, preserving _H1_CONFIGS order so the fingerprint is stable.
    seen: set[Path] = set()
    unique: list[Path] = []
    for p in paths:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return unique


def _spearman_dict(x: np.ndarray, y: np.ndarray) -> dict | None:
    if x.size < 3 or y.size < 3:
        return None
    rho, p = stats.spearmanr(x, y)
    return {"rho": round(float(rho), 6), "p": round(float(p), 6)}


def _load_coverage_predictors(root: Path, model: str, corpus: str) -> dict | None:
    """Return {covgap, dominance, research, policy} dicts keyed by SDG number."""
    if corpus == "concept":
        p = output_dir_for_model(model, root=root) / "data" / "concept" / "coverage_document_weighted.json"
    else:
        p = output_dir_for_model(model, root=root) / "data" / "coverage_document_weighted.json"
    if not p.exists():
        return None
    data = load_json(p)

    def _arr(key: str) -> dict[int, float]:
        return {i: float(data[key][f"SDG{i}"]) for i in range(1, N_SDG + 1)}

    research = _arr("research_profile_hard")
    policy = _arr("policy_profile_hard_docweighted")
    covgap = _arr("coverage_gap_hard")
    dominance = {i: research[i] - policy[i] for i in range(1, N_SDG + 1)}
    return {"research": research, "policy": policy, "covgap": covgap, "dominance": dominance}


def _raw_gaps_for(method: str, root: Path, model: str, corpus: str) -> dict | None:
    """Raw within-SDG gap vector for one config.

    `corpus` MUST be honoured: the concept rows retrieve a different research
    corpus (AI/ML field-of-study instead of keyword), so both their coverage
    predictors AND their semantic gaps come from data/concept/. Falling back to
    the keyword gaps here correlates a predictor built on one corpus against an
    outcome built on another. There is no concept ZS row in _H1_CONFIGS.
    """
    if method == "LR":
        return _concept_raw_gaps(root) if corpus == "concept" else _lr_raw_gaps(root, model)
    if method == "ZS":
        return _zs_raw_gaps(root, model)
    if method == "MLP":
        return _concept_mlp_raw_gaps(root) if corpus == "concept" else _mlp_raw_gaps(root, model)
    return None


def _adj_gaps_for(method: str, root: Path, model: str, corpus: str) -> dict | None:
    """Register-adjusted (topic) gap vector for one config. See _raw_gaps_for on `corpus`."""
    if method == "LR":
        return _concept_adj_gaps(root) if corpus == "concept" else _lr_adj_gaps(root, model)
    if method == "ZS":
        return _zs_adj_gaps(root, model)
    if method == "MLP":
        return _concept_mlp_adj_gaps(root) if corpus == "concept" else _mlp_adj_gaps(root, model)
    return None


def _h1_config_row(label: str, model: str, method: str, corpus: str, root: Path) -> dict | None:
    cov = _load_coverage_predictors(root, model, corpus)
    if cov is None:
        return None
    raw = _raw_gaps_for(method, root, model, corpus)
    if raw is None:
        return None
    adj = _adj_gaps_for(method, root, model, corpus)
    out = {"label": label, "predictors": {}}
    for pname in ("covgap", "dominance", "research", "policy"):
        pred = cov[pname]
        if adj is not None:
            common = sorted(set(pred) & set(raw) & set(adj))
        else:
            common = sorted(set(pred) & set(raw))
        if len(common) < 3:
            out["predictors"][pname] = None
            continue
        x = np.array([pred[s] for s in common], dtype=float)
        raw_arr = np.array([raw[s] for s in common], dtype=float)
        rho_raw = _spearman_dict(x, raw_arr)
        rho_adj = rho_reg = None
        if adj is not None:
            adj_arr = np.array([adj[s] for s in common], dtype=float)
            reg_arr = raw_arr - adj_arr
            rho_adj = _spearman_dict(x, adj_arr)
            rho_reg = _spearman_dict(x, reg_arr)
        out["predictors"][pname] = {"raw": rho_raw, "adj": rho_adj, "reg": rho_reg}
    return out


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


def _fmt_rho(v: dict | None) -> str:
    if v is None:
        return "--"
    return f"{v['rho']:+.3f}{_sig_stars(v['p'])}"


# Grid cells the interaction section quotes inline. Exported as macros so prose
# and Table tab:interaction cannot disagree: (macro stem, row label, predictor).
_H1_QUOTED_CELLS = [
    ("ConceptLRCovgapAdj", "Concept LR", "covgap"),
]

# Blocks whose "positive in N/9 configs" count the prose reports.
_H1_POSITIVE_COUNTS = [
    ("HOneACovgapAdjPositiveCount", "covgap"),
    ("HOneDPolicyAdjPositiveCount", "policy"),
]


def _h1_grid_macros(rows: list[dict]) -> list[str]:
    """Export the grid cells and config counts that the interaction prose cites."""
    by_label = {r["label"]: r for r in rows}
    lines: list[str] = []
    for stem, label, predictor in _H1_QUOTED_CELLS:
        pred = by_label.get(label, {}).get("predictors", {}).get(predictor)
        cell = pred["adj"] if pred else None
        lines.append(rf"\newcommand{{\{stem}Rho}}{{{_fmt_rho(cell)}}}")
    for stem, predictor in _H1_POSITIVE_COUNTS:
        adj = [
            r["predictors"].get(predictor, {}).get("adj") if r["predictors"].get(predictor) else None
            for r in rows
        ]
        n_pos = sum(1 for v in adj if v is not None and v["rho"] > 0)
        lines.append(rf"\newcommand{{\{stem}}}{{{n_pos}}}")
        lines.append(rf"\newcommand{{\{stem}Total}}{{{len(rows)}}}")
    return lines


def write_h1_grid_tex(path: Path, rows: list[dict]) -> None:
    lines = [
        "% Auto-generated by 1_code/7_main_analysis/1_main_text/2_coverage_semantic_interaction.py — do not edit manually",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r" & Raw gap & Adj.\ gap & Register \\",
        r"\midrule",
    ]
    for title, key in _H1_GROUPS:
        lines.append(rf"\multicolumn{{4}}{{l}}{{\textbf{{{title}}}}} \\")
        for r in rows:
            pr = r["predictors"].get(key)
            cells = [
                _fmt_rho(pr["raw"] if pr else None),
                _fmt_rho(pr["adj"] if pr else None),
                _fmt_rho(pr["reg"] if pr else None),
            ]
            lines.append(f"{r['label']} & " + " & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute H25 correlation outputs into the canonical output folder.")
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_ROOT))
    p.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL, type=resolve_model_alias, help=argparse.SUPPRESS)
    p.add_argument("--overwrite", action="store_true", help=argparse.SUPPRESS)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run(args: argparse.Namespace) -> None:
    model = args.embed_model
    scored_dir = scored_dir_for_model(model)
    paper_scores_manifest = scored_dir / "paper_scores_shards" / "metadata" / "manifest.json"

    layout = ensure_canonical_outputs(Path(args.output_dir), model=model)
    require_output_files(layout.data_dir, ["coverage_document_weighted.json", "semantic_gap_distances_lr.json", "adjusted/semantic_gap_distances_lr.json"])

    coverage_gap_path = layout.data_dir / "coverage_document_weighted.json"
    semantic_gap_path = layout.data_dir / "semantic_gap_distances_lr.json"
    adj_gap_path = layout.data_dir / "adjusted" / "semantic_gap_distances_lr.json"
    out_corr = layout.data_dir / "interaction_h25.json"
    out_scatter = layout.data_dir / "interaction_scatter_data.csv"
    tables_dir = layout.tables_dir
    log.info("Canonical output dir: %s", layout.data_dir)

    # Bumped to "3": the H1 grid's Concept rows previously read the MPNet keyword
    # gaps instead of the concept-corpus gaps, so every committed tab4 is stale.
    SCRIPT_VERSION = "3"
    PRIMARY = out_corr
    OUTPUTS = [out_corr, out_scatter, tables_dir / "tab4_interaction_h25.tex"]
    fp = fingerprint_of(
        coverage_gap_path, semantic_gap_path, adj_gap_path,
        *h1_grid_input_paths(Path(args.output_dir)),
    ) + SCRIPT_VERSION
    if should_skip(OUTPUTS, fp, args.overwrite, PRIMARY):
        log.info("Skipping %s \u2014 inputs unchanged", PRIMARY)
        return

    # ---- Load coverage data ----
    log.info("Loading coverage gap: %s", coverage_gap_path)
    cov_data = load_json(coverage_gap_path)

    # Extract per-SDG arrays (1-indexed labels, so SDG{i} = SDG i, row index i-1)
    res_hard    = np.array([cov_data["research_profile_hard"][f"SDG{i}"] for i in range(1, N_SDG + 1)])
    pol_dw_hard = np.array([cov_data["policy_profile_hard_docweighted"][f"SDG{i}"] for i in range(1, N_SDG + 1)])
    cov_gap_abs = np.array([cov_data["coverage_gap_hard"][f"SDG{i}"] for i in range(1, N_SDG + 1)])

    # Signed research dominance: positive = research > policy, negative = policy > research.
    res_dominance = res_hard - pol_dw_hard

    # ---- Load semantic gap data ----
    log.info("Loading semantic gap: %s", semantic_gap_path)
    sem_data = load_json(semantic_gap_path)

    per_sdg = {r["sdg"]: r for r in sem_data["per_sdg"]}
    sem_gap = np.array(
        [
            np.nan if per_sdg[i]["semantic_gap"] is None else float(per_sdg[i]["semantic_gap"])
            for i in range(1, N_SDG + 1)
        ],
        dtype=float,
    )
    sem_sim = np.array(
        [
            np.nan if per_sdg[i]["semantic_similarity"] is None else float(per_sdg[i]["semantic_similarity"])
            for i in range(1, N_SDG + 1)
        ],
        dtype=float,
    )
    unreliable = np.array([bool(per_sdg[i]["unreliable"]) for i in range(1, N_SDG + 1)], dtype=bool)

    # Only SDGs with finite semantic values are eligible for correlation or plotting.
    available_mask = np.isfinite(sem_gap) & np.isfinite(sem_sim)
    available_sdgs = [i + 1 for i, keep in enumerate(available_mask) if keep]
    reliable_mask = available_mask & ~unreliable
    reliable_sdgs = [i + 1 for i, keep in enumerate(reliable_mask) if keep]
    missing_sdgs = [i + 1 for i, keep in enumerate(available_mask) if not keep]
    log.info("Semantic-gap SDGs available for correlation: %s  (n=%d)", available_sdgs, len(available_sdgs))
    if missing_sdgs:
        log.warning("Excluded from correlation due to missing semantic gap: %s", missing_sdgs)
    log.info("Reliable SDGs for correlation: %s  (n=%d)", reliable_sdgs, len(reliable_sdgs))

    # ---- Load register-adjusted (topic) semantic gap (mirror of raw above) ----
    log.info("Loading adjusted (register-removed) semantic gap: %s", adj_gap_path)
    sem_data_adj = load_json(adj_gap_path)
    per_sdg_adj = {r["sdg"]: r for r in sem_data_adj["per_sdg"]}
    sem_gap_adj = np.array(
        [
            np.nan if per_sdg_adj[i].get("semantic_gap") is None else float(per_sdg_adj[i]["semantic_gap"])
            for i in range(1, N_SDG + 1)
        ],
        dtype=float,
    )
    sem_sim_adj = np.array(
        [
            np.nan if per_sdg_adj[i].get("semantic_similarity") is None else float(per_sdg_adj[i]["semantic_similarity"])
            for i in range(1, N_SDG + 1)
        ],
        dtype=float,
    )
    unreliable_adj = np.array(
        [bool(per_sdg_adj[i].get("unreliable", False)) for i in range(1, N_SDG + 1)],
        dtype=bool,
    )
    available_mask_adj = np.isfinite(sem_gap_adj) & np.isfinite(sem_sim_adj)
    reliable_mask_adj = available_mask_adj & ~unreliable_adj
    log.info(
        "Adjusted-gap SDGs available for correlation: %s  (n=%d)",
        [i + 1 for i, keep in enumerate(available_mask_adj) if keep],
        int(available_mask_adj.sum()),
    )

    # ---- Correlation tests ----
    # Use all SDGs with finite semantic gaps first, then re-check with only reliable ones.
    log.info("")
    log.info("=" * 70)
    log.info("CORRELATION TESTS")
    log.info("=" * 70)
    log.info("")
    log.info("OBSERVED SDGs WITH FINITE SEMANTIC GAP (n=%d):", int(available_mask.sum()))
    tests_primary = compute_four_tests(res_hard, pol_dw_hard, cov_gap_abs, res_dominance, sem_gap, available_mask)

    # Reliable SDGs only (excludes any SDG flagged unreliable).
    if reliable_mask.sum() < available_mask.sum():
        log.info("")
        log.info("RELIABLE SDGs ONLY (n=%d):", reliable_mask.sum())
        tests_reliable = compute_four_tests(res_hard, pol_dw_hard, cov_gap_abs, res_dominance, sem_gap, reliable_mask)
    else:
        tests_reliable = None

    # Sensitivity: exclude SDG 4 (suspected ML terminology artefact). Reported on BOTH the raw
    # gap (null hypothesis) and the register-adjusted (topic) gap, so the reader can see the
    # SDG 4 artefact does not drive either the raw null or the adjusted positive topic signal.
    # SDG 17 was retired as a leave-one-out: its method sensitivity is instead covered by the
    # encoder / assignment-method sensitivity tables.
    excl4_mask = available_mask.copy()
    excl4_mask[3] = False   # SDG 4 is index 3
    log.info("")
    log.info("SENSITIVITY — EXCLUDING SDG 4 (suspected ML 'learning' terminology artefact), RAW GAP:")
    tests_excl4 = compute_four_tests(res_hard, pol_dw_hard, cov_gap_abs, res_dominance, sem_gap, excl4_mask)

    excl4_mask_adj = available_mask_adj.copy()
    excl4_mask_adj[3] = False
    log.info("")
    log.info("SENSITIVITY — EXCLUDING SDG 4 (suspected ML 'learning' terminology artefact), ADJUSTED (TOPIC) GAP:")
    tests_excl4_adj = compute_four_tests(res_hard, pol_dw_hard, cov_gap_abs, res_dominance, sem_gap_adj, excl4_mask_adj)

    # ---- Correlation interpretation ----
    # Primary headline test: research coverage vs semantic gap (predictor "research").
    primary_stats = tests_primary["research"]
    if primary_stats["skipped"]:
        raise RuntimeError("Primary H25 correlation could not be computed: fewer than 3 valid SDGs.")
    r_primary = primary_stats["pearson_r"]
    rho_primary = primary_stats["spearman_rho"]
    p_primary = primary_stats["pearson_p"]

    log.info("")
    log.info("=" * 70)
    log.info("CORRELATION INTERPRETATION (FOUR PREDICTORS vs SEMANTIC GAP)")
    log.info("=" * 70)
    for key, label in [
        ("research", "research coverage"),
        ("policy", "policy coverage"),
        ("covgap", "coverage gap (abs)"),
        ("dominance", "policy-research dominance (signed)"),
    ]:
        st = tests_primary[key]
        if st.get("skipped"):
            log.info("  %-30s skipped (n=%s)", label, st.get("n"))
            continue
        log.info(
            "  %-30s Pearson r=%.3f (p=%.3f)  Spearman rho=%.3f",
            label, st["pearson_r"], st["pearson_p"], st["spearman_rho"],
        )

    # Directional reading for the research predictor (matches prior reporting).
    if r_primary > 0.3:
        correlation_direction = "SUPPORTED"
        correlation_story = (
            "Positive correlation: SDGs with higher research attention show greater within-SDG "
            "semantic divergence from policy."
        )
    elif r_primary < -0.3:
        correlation_direction = "CONTRADICTED"
        correlation_story = (
            "Negative correlation: SDGs with more research attention are MORE semantically aligned "
            "with policy — suggests research reduces semantic divergence over time."
        )
    else:
        correlation_direction = "NOT SUPPORTED (WEAK CORRELATION)"
        correlation_story = (
            "Near-zero raw correlation: research attention does not predict semantic gap direction, "
            "but coverage and framing are not independent — a positive topic signal is cancelled by a "
            "negative register signal (adjusted gap is positively associated with coverage divergence)."
        )
    log.info("  Research-coverage direction: %s", correlation_direction)
    log.info("  Pearson r=%.3f  p=%.3f  Spearman ρ=%.3f", r_primary, p_primary, rho_primary)
    log.info("  %s", correlation_story)

    # Top-3 outliers (SDGs furthest from the trend line).
    log.info("")
    log.info("NOTABLE SDGS (highest research % + gap relationship):")
    pairs = sorted(
        [(i + 1, res_hard[i], sem_gap[i]) for i in range(N_SDG) if np.isfinite(sem_gap[i])],
        key=lambda x: x[1], reverse=True
    )[:5]
    for sdg, rp, sg in pairs:
        log.info("  SDG %2d  res=%.1f%%  sem_gap=%.3f  (%s)",
                 sdg, rp*100, sg,
                 "high_coverage_high_gap" if sg > 0.25 else "high_coverage_low_gap")

    # ---- Save outputs ----
    results = {
        "correlation": {
            "hypothesis": (
                "SDGs with the highest research attention show the largest within-SDG semantic "
                "gaps (research and policy talk past each other at points of engagement)."
            ),
            "primary_test": (
                "research_proportion vs semantic_gap "
                f"(Pearson r, Spearman rho, n={primary_stats['n']})"
            ),
            "direction_found": correlation_direction,
            "story": correlation_story,
            "caveats": [
                f"n={primary_stats['n']} SDGs in the primary test gives low statistical power.",
                "SDG 4 research proportion (22%) may be inflated by ML 'learning' terminology.",
                "Hard assignment creates zero-sum profiles — SDGs with overlapping centroids "
                "(e.g. SDG 1/8/10 cluster) may trade assignments artificially.",
            ],
            "available_semantic_gap_sdgs": available_sdgs,
            "missing_semantic_gap_sdgs": missing_sdgs,
            "correlations_primary_observed": tests_primary,
            "correlations_reliable_only": tests_reliable,
            "correlations_excl_sdg4": {"raw": tests_excl4, "adjusted": tests_excl4_adj},
        },
        "per_sdg_table": [
            {
                "sdg": i + 1,
                "research_proportion": round(float(res_hard[i]), 6),
                "policy_proportion_docweighted": round(float(pol_dw_hard[i]), 6),
                "coverage_gap_abs": round(float(cov_gap_abs[i]), 6),
                "research_dominance": round(float(res_dominance[i]), 6),
                "semantic_gap": None if not np.isfinite(sem_gap[i]) else round(float(sem_gap[i]), 6),
                "semantic_similarity": None if not np.isfinite(sem_sim[i]) else round(float(sem_sim[i]), 6),
                "reliable": bool(reliable_mask[i]),
            }
            for i in range(N_SDG)
        ],
        "provenance": {
            "semantic_gap_raw": sem_data.get("provenance"),
            "coverage_gap_path": str(Path(coverage_gap_path).relative_to(Path(args.output_dir))),
        },
    }

    with out_corr.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    log.info("Saved: %s", out_corr)

    # CSV scatter table for plotting.
    with out_scatter.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "sdg", "research_pct", "policy_pct_docweighted",
            "coverage_gap_abs", "research_dominance",
            "semantic_gap", "semantic_similarity", "reliable"
        ])
        writer.writeheader()
        for i in range(N_SDG):
            writer.writerow({
                "sdg": i + 1,
                "research_pct": round(float(res_hard[i]) * 100, 4),
                "policy_pct_docweighted": round(float(pol_dw_hard[i]) * 100, 4),
                "coverage_gap_abs": round(float(cov_gap_abs[i]), 6),
                "research_dominance": round(float(res_dominance[i]), 6),
                "semantic_gap": "" if not np.isfinite(sem_gap[i]) else round(float(sem_gap[i]), 6),
                "semantic_similarity": "" if not np.isfinite(sem_sim[i]) else round(float(sem_sim[i]), 6),
                "reliable": int(reliable_mask[i]),
            })
    log.info("Saved: %s", out_scatter)

    log.info("")
    log.info("Next step: python 1_code/8_visualization/plot_figures.py")

    # ---- Write LaTeX generated outputs ----
    gen_dir = tables_dir

    # Median research% across 17 SDGs.
    median_res_pct = float(np.median(res_hard * 100.0))

    # Primary predictor handle (N and directional reading); CI/MDE computed below after _fmt2.
    primary = tests_primary["research"]
    n_primary = int(primary["n"])

    def _fmt(v):
        """Format float for LaTeX: negative values get a minus sign, 3 d.p."""
        s = f"{abs(v):.3f}"
        return f"-{s}" if v < 0 else s

    def _fmt2(v):
        """Format float for LaTeX with 2 d.p."""
        s = f"{abs(v):.2f}"
        return f"-{s}" if v < 0 else s

    # ---- 95% CIs (Fisher z) for all four primary predictors, plus minimal detectable effect ----
    def _fisher_ci(r, n):
        if n < 4 or not np.isfinite(r):
            return float("nan"), float("nan")
        z = math.atanh(max(-0.999, min(0.999, r)))
        se = 1.0 / math.sqrt(n - 3)
        return math.tanh(z - 1.96 * se), math.tanh(z + 1.96 * se)

    _pred_ci = {
        "research": "HPrimary",        # H1c
        "policy": "HPrimaryPolicy",    # H1d
        "covgap": "HPrimaryCovgap",    # H1a (primary headline)
        "dominance": "HPrimaryDominance",  # H1b
    }
    _ci_lines = []
    for _k, _m in _pred_ci.items():
        _s = tests_primary.get(_k, {})
        if _s.get("skipped") or not np.isfinite(_s.get("pearson_r", float("nan"))):
            _ci_lines.append(rf"\newcommand{{\{_m}CiLower}}{{-}}")
            _ci_lines.append(rf"\newcommand{{\{_m}CiUpper}}{{-}}")
        else:
            _lo, _hi = _fisher_ci(_s["pearson_r"], int(_s["n"]))
            _ci_lines.append(rf"\newcommand{{\{_m}CiLower}}{{{_fmt2(_lo)}}}")
            _ci_lines.append(rf"\newcommand{{\{_m}CiUpper}}{{{_fmt2(_hi)}}}")

    # Minimal detectable Pearson r at 80% power, two-sided alpha = 0.05, n = n_primary.
    _z_a = stats.norm.ppf(1 - 0.05 / 2)
    _z_b = stats.norm.ppf(0.80)
    _r_mde = math.tanh((_z_a + _z_b) / math.sqrt(n_primary - 3))

    def _macro(name, cdict, key):
        """Emit \namePearsonR / PearsonP / SpearmanRho / SpearmanP from a correlation dict."""
        c = cdict.get(key, {})
        if c.get("skipped"):
            r_s, p_s, rho_s, pr_s = "--", "--", "--", "--"
        else:
            r_s, p_s, rho_s, pr_s = (
                _fmt(c["pearson_r"]), f"{c['pearson_p']:.3f}",
                _fmt(c["spearman_rho"]), f"{c['spearman_p']:.3f}",
            )
        return [
            rf"\newcommand{{\{name}PearsonR}}{{{r_s}}}",
            rf"\newcommand{{\{name}PearsonP}}{{{p_s}}}",
            rf"\newcommand{{\{name}SpearmanRho}}{{{rho_s}}}",
            rf"\newcommand{{\{name}SpearmanP}}{{{pr_s}}}",
        ]

    # num4_interaction_h25.tex — macro definitions
    num_lines = [
        "% Auto-generated by 1_code/7_main_analysis/1_main_text/2_coverage_semantic_interaction.py — do not edit manually",
        rf"\newcommand{{\HPrimaryN}}{{{n_primary}}}",
    ]
    num_lines += _ci_lines
    num_lines.append(rf"\newcommand{{\HPrimaryMinDetectableR}}{{{_fmt2(_r_mde)}}}")
    num_lines += _macro("HPrimary", tests_primary, "research")
    num_lines += _macro("HPrimaryPolicy", tests_primary, "policy")
    num_lines += _macro("HPrimaryCovgap", tests_primary, "covgap")
    num_lines += _macro("HPrimaryDominance", tests_primary, "dominance")
    # SDG 4 leave-one-out sensitivity (raw gap): wired into the interaction section.
    num_lines += _macro("HExclFourResearch", tests_excl4, "research")
    num_lines += _macro("HExclFourPolicy", tests_excl4, "policy")
    num_lines += _macro("HExclFourCovgap", tests_excl4, "covgap")
    num_lines += _macro("HExclFourDominance", tests_excl4, "dominance")
    # SDG 4 leave-one-out sensitivity (register-adjusted topic gap).
    num_lines += _macro("HExclFourResearchAdj", tests_excl4_adj, "research")
    num_lines += _macro("HExclFourPolicyAdj", tests_excl4_adj, "policy")
    num_lines += _macro("HExclFourCovgapAdj", tests_excl4_adj, "covgap")
    num_lines += _macro("HExclFourDominanceAdj", tests_excl4_adj, "dominance")
    num_lines += [
        rf"\newcommand{{\MedianResearchPct}}{{{median_res_pct:.2f}}}",
    ]

    # ---- H1 x config register-correlation grid (replaces the stale exclude-SDG rows) ----
    # Computed before num4 is written so the grid can also export the cells the
    # prose quotes -- the interaction section previously hard-coded them, and the
    # literals silently went stale when the grid was regenerated.
    h1_grid: list[dict] = []
    for _label, _m, _method, _corpus in _H1_CONFIGS:
        _row = _h1_config_row(_label, _m, _method, _corpus, Path(args.output_dir))
        if _row is not None:
            h1_grid.append(_row)
        else:
            log.warning("WARNING: missing data for %s -- skipping row", _label)

    num_lines += _h1_grid_macros(h1_grid)
    (gen_dir / "num4_interaction_h25.tex").write_text("\n".join(num_lines) + "\n", encoding="utf-8")
    log.info("Saved: %s", gen_dir / "num4_interaction_h25.tex")

    write_h1_grid_tex(gen_dir / "tab4_interaction_h25.tex", h1_grid)
    log.info("Saved: %s", gen_dir / "tab4_interaction_h25.tex")
    record_fingerprint(OUTPUTS, fp, PRIMARY)


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()

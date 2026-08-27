"""
Appendix K.1: OLS regression — semantic gap ~ coverage + configuration indicators.

Pools across encoder (MPNet / MiniLM / SciBERT), retrieval (keyword / concept),
assignment method (LR / MLP / ZS), and segment cap (20 / 50 / none) to estimate
how coverage gap and policy coverage predict the within-SDG semantic gap, while
controlling for configuration through indicator variables.

Model (base):
  rank(sem_gap) = β₀ + β₁·covgap + β₂·polcov
                + β₃·i_minilm + β₄·i_scibert
                + β₅·i_concept
                + β₆·i_cap20 + β₇·i_cap_none
                + β₈·i_mlp + β₉·i_zs + ε

With classifier indicator (default), subsamples, interactions, functional forms,
and bootstrap SE available via flags.

Run from project root:
  python 1_code/7_main_analysis/2_appendix/k1_regression_semantic_gap.py --spec-grid
  python 1_code/7_main_analysis/2_appendix/k1_regression_semantic_gap.py --gap-type adjusted
  python 1_code/7_main_analysis/2_appendix/k1_regression_semantic_gap.py --gap-type adjusted --bootstrap-se 500
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, t as t_dist

ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = ROOT / "1_code"
ANALYSIS_ROOT = Path(__file__).resolve().parents[1]
SHARED_DIR = ANALYSIS_ROOT / "0_shared"
MAIN_TEXT = ANALYSIS_ROOT / "1_main_text"
for path in (CODE_ROOT, SHARED_DIR, MAIN_TEXT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from model_utils import (
    DEFAULT_EMBED_MODEL,
    DEFAULT_OUTPUT_ROOT,
    N_SDG,
    output_dir_for_model,
    resolve_model_alias,
    model_slug,
)
from shared_utils import (
    ensure_dissertation_outputs,
    fingerprint_of,
    record_fingerprint,
    should_skip,
)
from shard_pipeline_utils import load_json, atomic_write_json

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config grid: (label, encoder_slug, method, corpus, cap)
# Coverage gap is segment-cap-independent, so we share it across caps.
# ---------------------------------------------------------------------------
ENCODERS = {
    "mpnet": "all-mpnet-base-v2",
    "minilm": "all-MiniLM-L6-v2",
    "scibert": "allenai/scibert_scivocab_uncased",
}

CONFIGS: list[tuple[str, str, str, str, int]] = [
    # Keyword × LR × 3 caps × 3 encoders
    ("MPNet LR",      "all-mpnet-base-v2",              "LR",  "keyword", 50),
    ("MPNet LR c20",  "all-mpnet-base-v2",              "LR",  "keyword", 20),
    ("MPNet LR c∞",   "all-mpnet-base-v2",              "LR",  "keyword", 0),
    ("MiniLM LR",     "all-MiniLM-L6-v2",               "LR",  "keyword", 50),
    ("MiniLM LR c20", "all-MiniLM-L6-v2",               "LR",  "keyword", 20),
    ("MiniLM LR c∞",  "all-MiniLM-L6-v2",               "LR",  "keyword", 0),
    ("SciBERT LR",    "allenai/scibert_scivocab_uncased", "LR",  "keyword", 50),
    ("SciBERT LR c20","allenai/scibert_scivocab_uncased", "LR",  "keyword", 20),
    ("SciBERT LR c∞", "allenai/scibert_scivocab_uncased", "LR",  "keyword", 0),
    # Keyword × MLP × 3 caps × 3 encoders
    ("MPNet MLP",      "all-mpnet-base-v2",              "MLP", "keyword", 50),
    ("MPNet MLP c20",  "all-mpnet-base-v2",              "MLP", "keyword", 20),
    ("MPNet MLP c∞",   "all-mpnet-base-v2",              "MLP", "keyword", 0),
    ("MiniLM MLP",     "all-MiniLM-L6-v2",               "MLP", "keyword", 50),
    ("MiniLM MLP c20", "all-MiniLM-L6-v2",               "MLP", "keyword", 20),
    ("MiniLM MLP c∞",  "all-MiniLM-L6-v2",               "MLP", "keyword", 0),
    ("SciBERT MLP",    "allenai/scibert_scivocab_uncased", "MLP", "keyword", 50),
    ("SciBERT MLP c20","allenai/scibert_scivocab_uncased", "MLP", "keyword", 20),
    ("SciBERT MLP c∞", "allenai/scibert_scivocab_uncased", "MLP", "keyword", 0),
    # Keyword × ZS × cap=50 only × 3 encoders
    ("MPNet ZS",       "all-mpnet-base-v2",              "ZS",  "keyword", 50),
    ("MiniLM ZS",      "all-MiniLM-L6-v2",               "ZS",  "keyword", 50),
    ("SciBERT ZS",     "allenai/scibert_scivocab_uncased", "ZS",  "keyword", 50),
    # Concept × LR × 3 caps (MPNet only)
    ("Concept LR",      "all-mpnet-base-v2",              "LR",  "concept", 50),
    ("Concept LR c20",  "all-mpnet-base-v2",              "LR",  "concept", 20),
    ("Concept LR c∞",   "all-mpnet-base-v2",              "LR",  "concept", 0),
]

# ---------------------------------------------------------------------------
# Specification grid (21 specs)
# ---------------------------------------------------------------------------
SPEC_GRID = [
    # Panel A: Core regressions
    {"spec_id": "adj_covgap",     "panel": "A", "label": "Adj. gap ~ covgap",
     "gap_type": "adjusted",  "predictor": "covgap",     "subsample": "all", "interactions": "none", "form": "rank", "sdg_fe": False, "classifier_ind": True},
    {"spec_id": "adj_dominance",  "panel": "A", "label": "Adj. gap ~ dominance",
     "gap_type": "adjusted",  "predictor": "dominance",  "subsample": "all", "interactions": "none", "form": "rank", "sdg_fe": False, "classifier_ind": True},
    {"spec_id": "raw_covgap",     "panel": "A", "label": "Semantic gap ~ covgap",
     "gap_type": "raw",       "predictor": "covgap",     "subsample": "all", "interactions": "none", "form": "rank", "sdg_fe": False, "classifier_ind": True},
    {"spec_id": "raw_dominance",  "panel": "A", "label": "Semantic gap ~ dominance",
     "gap_type": "raw",       "predictor": "dominance",  "subsample": "all", "interactions": "none", "form": "rank", "sdg_fe": False, "classifier_ind": True},
    {"spec_id": "reg_covgap",     "panel": "A", "label": "Reg. comp. ~ covgap",
     "gap_type": "register",  "predictor": "covgap",     "subsample": "all", "interactions": "none", "form": "rank", "sdg_fe": False, "classifier_ind": True},
    {"spec_id": "reg_dominance",  "panel": "A", "label": "Reg. comp. ~ dominance",
     "gap_type": "register",  "predictor": "dominance",  "subsample": "all", "interactions": "none", "form": "rank", "sdg_fe": False, "classifier_ind": True},
    # Panel A: Predictor decomposition (rescov)
    {"spec_id": "adj_rescov",     "panel": "A", "label": "Adj. gap ~ rescov",
     "gap_type": "adjusted",  "predictor": "rescov",     "subsample": "all", "interactions": "none", "form": "rank", "sdg_fe": False, "classifier_ind": True},
    {"spec_id": "raw_rescov",     "panel": "A", "label": "Semantic gap ~ rescov",
     "gap_type": "raw",       "predictor": "rescov",     "subsample": "all", "interactions": "none", "form": "rank", "sdg_fe": False, "classifier_ind": True},
    {"spec_id": "reg_rescov",     "panel": "A", "label": "Reg. comp. ~ rescov",
     "gap_type": "register",  "predictor": "rescov",     "subsample": "all", "interactions": "none", "form": "rank", "sdg_fe": False, "classifier_ind": True},
    # Panel B: Robustness (all adjusted + covgap)
    {"spec_id": "adj_noclf",      "panel": "B", "label": "No classifier ind.",
     "gap_type": "adjusted",  "predictor": "covgap",     "subsample": "all", "interactions": "none", "form": "rank", "sdg_fe": False, "classifier_ind": False},
    {"spec_id": "adj_sdgfe",      "panel": "B", "label": "+ SDG FE",
     "gap_type": "adjusted",  "predictor": "covgap",     "subsample": "all", "interactions": "none", "form": "rank", "sdg_fe": True,  "classifier_ind": True},
    {"spec_id": "adj_supervised", "panel": "B", "label": "Supervised only",
     "gap_type": "adjusted",  "predictor": "covgap",     "subsample": "supervised", "interactions": "none", "form": "rank", "sdg_fe": False, "classifier_ind": True},
    {"spec_id": "adj_keyword",    "panel": "B", "label": "Keyword only",
     "gap_type": "adjusted",  "predictor": "covgap",     "subsample": "keyword", "interactions": "none", "form": "rank", "sdg_fe": False, "classifier_ind": True},
    {"spec_id": "adj_mpnet",      "panel": "B", "label": "MPNet only",
     "gap_type": "adjusted",  "predictor": "covgap",     "subsample": "mpnet", "interactions": "none", "form": "rank", "sdg_fe": False, "classifier_ind": True},
    {"spec_id": "adj_noclf_sdgfe","panel": "B", "label": "No clf + SDG FE",
     "gap_type": "adjusted",  "predictor": "covgap",     "subsample": "all", "interactions": "none", "form": "rank", "sdg_fe": True,  "classifier_ind": False},
    # Panel C: Interactions (adjusted + covgap)
    {"spec_id": "adj_int_enc",    "panel": "C", "label": "+ covgap$\\times$enc",
     "gap_type": "adjusted",  "predictor": "covgap",     "subsample": "all", "interactions": "encoder", "form": "rank", "sdg_fe": False, "classifier_ind": True},
    {"spec_id": "adj_int_ret",    "panel": "C", "label": "+ covgap$\\times$ret",
     "gap_type": "adjusted",  "predictor": "covgap",     "subsample": "all", "interactions": "retrieval", "form": "rank", "sdg_fe": False, "classifier_ind": True},
    {"spec_id": "adj_int_mth",    "panel": "C", "label": "+ covgap$\\times$mth",
     "gap_type": "adjusted",  "predictor": "covgap",     "subsample": "all", "interactions": "method", "form": "rank", "sdg_fe": False, "classifier_ind": True},
    {"spec_id": "adj_int_all",    "panel": "C", "label": "+ all interactions",
     "gap_type": "adjusted",  "predictor": "covgap",     "subsample": "all", "interactions": "all", "form": "rank", "sdg_fe": False, "classifier_ind": True},
    # Panel D: Functional forms (adjusted + covgap)
    {"spec_id": "adj_raw_dv",     "panel": "D", "label": "Raw DV (not ranked)",
     "gap_type": "adjusted",  "predictor": "covgap",     "subsample": "all", "interactions": "none", "form": "raw", "sdg_fe": False, "classifier_ind": True},
    {"spec_id": "adj_log_dv",     "panel": "D", "label": "Log DV",
     "gap_type": "adjusted",  "predictor": "covgap",     "subsample": "all", "interactions": "none", "form": "log", "sdg_fe": False, "classifier_ind": True},
    {"spec_id": "adj_wls",        "panel": "D", "label": "WLS",
     "gap_type": "adjusted",  "predictor": "covgap",     "subsample": "all", "interactions": "none", "form": "wls", "sdg_fe": False, "classifier_ind": True},
    {"spec_id": "adj_covgap_boot","panel": "D", "label": "Rank + bootstrap",
     "gap_type": "adjusted",  "predictor": "covgap",     "subsample": "all", "interactions": "none", "form": "rank", "sdg_fe": False, "classifier_ind": True, "bootstrap": 500},
]

# Variable label map (LaTeX)
VAR_LABELS = {
    "covgap":     r"Coverage gap ($|\text{research}\% - \text{policy}\%|)$",
    "dominance":  r"Domination (research\% $-$ policy\%)",
    "rescov":     r"Research coverage (\%)",
    "polcov":     "Policy coverage",
    "i_minilm":   r"MiniLM (vs.\ MPNet)",
    "i_scibert":  r"SciBERT (vs.\ MPNet)",
    "i_concept":  r"Concept (vs.\ keyword)",
    "i_cap20":    r"Cap 20 (vs.\ 50)",
    "i_cap_none": r"No cap (vs.\ 50)",
    "i_mlp":      r"MLP (vs.\ LR)",
    "i_zs":       r"ZS (vs.\ LR)",
    # Interaction-term variable labels (Panel C) -- wrapped in math mode so the
    # subscript underscore and the \times symbol are valid LaTeX.
    "covgap×i_minilm":  r"covgap$\times i_{\text{minilm}}$",
    "covgap×i_scibert": r"covgap$\times i_{\text{scibert}}$",
    "covgap×i_concept": r"covgap$\times i_{\text{concept}}$",
    "covgap×i_mlp":     r"covgap$\times i_{\text{mlp}}$",
    "covgap×i_zs":      r"covgap$\times i_{\text{zs}}$",
}

# Canonical variable ordering for the table
CANONICAL_VARS = [
    "covgap", "dominance", "polcov",
    "i_minilm", "i_scibert", "i_concept",
    "i_cap20", "i_cap_none", "i_mlp", "i_zs",
]

# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return load_json(path)


def _coverage_gaps(root: Path, model: str, corpus: str) -> dict[int, dict] | None:
    """Load coverage_gap_hard + policy_profile keyed by SDG number."""
    if corpus == "concept":
        p = output_dir_for_model(model, root=root) / "data" / "concept" / "coverage_document_weighted.json"
    else:
        p = output_dir_for_model(model, root=root) / "data" / "coverage_document_weighted.json"
    data = _load_json(p)
    if data is None:
        return None
    cg = data.get("coverage_gap_hard")
    pp = data.get("policy_profile_hard_docweighted")
    if not cg or not pp:
        return None
    return {
        i: {
            "covgap": float(cg[f"SDG{i}"]),
            "polcov": float(pp[f"SDG{i}"]),
            "research": float(data["research_profile_hard"][f"SDG{i}"]),
        }
        for i in range(1, N_SDG + 1)
    }


def _lr_raw_gaps(root: Path, model: str) -> dict[int, dict] | None:
    p = output_dir_for_model(model, root=root) / "data" / "semantic_gap_distances_lr.json"
    data = _load_json(p)
    if data is None:
        return None
    return {row["sdg"]: {"gap": row["semantic_gap"], "n_papers": row.get("n_papers", 0)}
            for row in data["per_sdg"] if row["semantic_gap"] is not None}


def _lr_adj_gaps(root: Path, model: str) -> dict[int, dict] | None:
    p = output_dir_for_model(model, root=root) / "data" / "adjusted" / "semantic_gap_distances_lr.json"
    data = _load_json(p)
    if data is None:
        return None
    return {row["sdg"]: {"gap": row["semantic_gap"], "n_papers": row.get("n_papers", 0)}
            for row in data["per_sdg"] if row["semantic_gap"] is not None}


def _mlp_raw_gaps(root: Path, model: str) -> dict[int, dict] | None:
    p = output_dir_for_model(model, root=root) / "data" / "semantic_gap_distances_mlp.json"
    data = _load_json(p)
    if data is None:
        return None
    return {row["sdg"]: {"gap": row["semantic_gap"], "n_papers": row.get("n_papers", 0)}
            for row in data["per_sdg"] if row["semantic_gap"] is not None}


def _mlp_adj_gaps(root: Path, model: str) -> dict[int, dict] | None:
    p = output_dir_for_model(model, root=root) / "data" / "adjusted" / "semantic_gap_distances_mlp.json"
    data = _load_json(p)
    if data is None:
        return None
    return {row["sdg"]: {"gap": row["semantic_gap"], "n_papers": row.get("n_papers", 0)}
            for row in data["per_sdg"] if row["semantic_gap"] is not None}


def _zs_raw_gaps(root: Path, model: str) -> dict[int, dict] | None:
    p = output_dir_for_model(model, root=root) / "data" / "semantic_gap_distances_zeroshot.json"
    data = _load_json(p)
    if data is None:
        return None
    return {row["sdg"]: {"gap": row["semantic_gap"], "n_papers": row.get("n_papers", 0)}
            for row in data["per_sdg"] if row["semantic_gap"] is not None}


def _zs_adj_gaps(root: Path, model: str) -> dict[int, dict] | None:
    p = output_dir_for_model(model, root=root) / "data" / "adjusted" / "semantic_gap_distances_zeroshot.json"
    data = _load_json(p)
    if data is None:
        return None
    return {row["sdg"]: {"gap": row["semantic_gap"], "n_papers": row.get("n_papers", 0)}
            for row in data["per_sdg"] if row["semantic_gap"] is not None}


def _concept_raw_gaps(root: Path) -> dict[int, dict] | None:
    p = output_dir_for_model("all-mpnet-base-v2", root=root) / "data" / "concept" / "semantic_gap_distances_lr.json"
    data = _load_json(p)
    if data is None:
        return None
    return {row["sdg"]: {"gap": row["semantic_gap"], "n_papers": row.get("n_papers", 0)}
            for row in data["per_sdg"] if row["semantic_gap"] is not None}


def _concept_adj_gaps(root: Path) -> dict[int, dict] | None:
    p = output_dir_for_model("all-mpnet-base-v2", root=root) / "data" / "concept" / "adjusted" / "semantic_gap_distances_lr.json"
    data = _load_json(p)
    if data is None:
        return None
    return {row["sdg"]: {"gap": row["semantic_gap"], "n_papers": row.get("n_papers", 0)}
            for row in data["per_sdg"] if row["semantic_gap"] is not None}


def _cap_gaps(root: Path, model: str, method: str, cap: int, corpus: str,
              gap_type: str = "raw") -> dict[int, dict] | None:
    """Load semantic gap for a specific segment cap.

    cap=50 -> base file; cap=20 or cap=0 (none) -> robustness_caps file.
    gap_type: 'raw' or 'adj'.
    Returns dict[sdg] = {"gap": float, "n_papers": int}.
    """
    if cap == 50:
        if corpus == "concept":
            if method == "LR":
                return _concept_raw_gaps(root) if gap_type == "raw" else _concept_adj_gaps(root)
            return None
        if method == "LR":
            return _lr_raw_gaps(root, model) if gap_type == "raw" else _lr_adj_gaps(root, model)
        if method == "MLP":
            return _mlp_raw_gaps(root, model) if gap_type == "raw" else _mlp_adj_gaps(root, model)
        if method == "ZS":
            return _zs_raw_gaps(root, model) if gap_type == "raw" else _zs_adj_gaps(root, model)
        return None

    # cap=20 or cap=0 (none): load from robustness caps
    suffix = "lr" if method == "LR" else "mlp"
    adj_prefix = "" if gap_type == "raw" else "adjusted/"
    if corpus == "concept":
        p = output_dir_for_model("all-mpnet-base-v2", root=root) / "data" / "concept" / adj_prefix / f"semantic_gap_robustness_caps_{suffix}.json"
    else:
        p = output_dir_for_model(model, root=root) / "data" / adj_prefix / f"semantic_gap_robustness_caps_{suffix}.json"
    data = _load_json(p)
    if data is None:
        return None
    key = "cap_20" if cap == 20 else "cap_none"
    arr = data.get(key)
    if arr is None:
        return None
    return {row["sdg"]: {"gap": row["semantic_gap"], "n_papers": row.get("n_papers", 0)}
            for row in arr if row["semantic_gap"] is not None}


# ---------------------------------------------------------------------------
# Panel builder
# ---------------------------------------------------------------------------

def _encoder_label(slug: str) -> str:
    for short, s in ENCODERS.items():
        if s == slug:
            return short.capitalize()
    return slug


def build_panel(root: Path, gap_type: str) -> list[dict]:
    """Build the pooled panel: one row per (SDG, config).

    gap_type: 'raw', 'adjusted', or 'register' (= raw - adjusted).
    """
    rows = []
    for label, model, method, corpus, cap in CONFIGS:
        cov = _coverage_gaps(root, model, corpus)
        if cov is None:
            log.warning("  Coverage missing for %s -- skipping config", label)
            continue

        if gap_type == "register":
            raw_g = _cap_gaps(root, model, method, cap, corpus, gap_type="raw")
            adj_g = _cap_gaps(root, model, method, cap, corpus, gap_type="adj")
            if raw_g is not None and adj_g is not None:
                common_g = sorted(set(raw_g) & set(adj_g))
                sem = {sdg: {"gap": raw_g[sdg]["gap"] - adj_g[sdg]["gap"],
                              "n_papers": raw_g[sdg]["n_papers"]}
                       for sdg in common_g}
            else:
                sem = None
        else:
            sem = _cap_gaps(root, model, method, cap, corpus, gap_type=gap_type)
        if sem is None:
            log.warning("  Semantic gap missing for %s (cap=%d) -- skipping config", label, cap)
            continue

        common = sorted(set(cov) & set(sem))
        if len(common) < N_SDG:
            log.warning("  %s: only %d/%d common SDGs", label, len(common), N_SDG)

        enc_short = _encoder_label(model)
        for sdg in common:
            # Scale to percentage points (0-100) for interpretability
            covgap_pct = cov[sdg]["covgap"] * 100.0
            polcov_pct = cov[sdg]["polcov"] * 100.0
            research_pct = cov[sdg]["research"] * 100.0
            dominance_pct = research_pct - polcov_pct
            rows.append({
                "sdg": sdg,
                "encoder": enc_short,
                "retrieval": corpus,
                "method": method,
                "cap": cap if cap > 0 else "none",
                "sem_gap": sem[sdg]["gap"],
                "n_papers": sem[sdg]["n_papers"],
                "covgap": covgap_pct,
                "polcov": polcov_pct,
                "research": research_pct,
                "dominance": dominance_pct,
            })
    return rows


def filter_panel(panel: list[dict], subsample: str) -> list[dict]:
    """Filter panel by subsample constraint."""
    if subsample == "all":
        return panel
    if subsample == "supervised":
        return [r for r in panel if r["method"] in ("LR", "MLP")]
    if subsample == "keyword":
        return [r for r in panel if r["retrieval"] == "keyword"]
    if subsample == "mpnet":
        return [r for r in panel if r["encoder"] == "Mpnet"]
    raise ValueError(f"Unknown subsample: {subsample}")


# ---------------------------------------------------------------------------
# OLS helpers (no statsmodels dependency)
# ---------------------------------------------------------------------------

def _build_design(
    panel: list[dict], predictor: str, with_sdg_fe: bool,
    classifier_ind: bool, interactions: str, form: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Build Y, X, cluster_ids, var_names from panel."""
    n = len(panel)
    sem_vals = np.array([r["sem_gap"] for r in panel], dtype=float)

    # DV transformation (wls uses rank DV with weighted OLS)
    if form in ("rank", "wls"):
        # Rank WITHIN each config (1-17 per config), not pooled (1-N).
        # Each config has 17 SDGs; rank them by semantic gap.
        Y = np.empty(n, dtype=float)
        config_keys = {}
        for idx, r in enumerate(panel):
            key = (r["encoder"], r["method"], r["retrieval"], r["cap"])
            config_keys.setdefault(key, []).append(idx)
        for key, indices in config_keys.items():
            vals = sem_vals[indices]
            Y[indices] = rankdata(vals, method="average")
    elif form == "raw":
        Y = sem_vals
    elif form == "log":
        # log(1 + |gap|) * sign(gap) — preserves sign, handles zeros
        Y = np.sign(sem_vals) * np.log1p(np.abs(sem_vals))
    else:
        raise ValueError(f"Unknown form: {form}")

    # Build design matrix
    var_names = ["intercept"]
    cols = [np.ones(n)]

    # Continuous predictor
    pred_key = {"covgap": "covgap", "dominance": "dominance", "rescov": "research"}[predictor]
    cols.append(np.array([r[pred_key] for r in panel], dtype=float))
    var_names.append(predictor)

    # Policy coverage
    cols.append(np.array([r["polcov"] for r in panel], dtype=float))
    var_names.append("polcov")

    # Encoder indicators (base = MPNet)
    i_minilm = np.array([1.0 if r["encoder"] == "Minilm" else 0.0 for r in panel])
    i_scibert = np.array([1.0 if r["encoder"] == "Scibert" else 0.0 for r in panel])
    cols.append(i_minilm)
    var_names.append("i_minilm")
    cols.append(i_scibert)
    var_names.append("i_scibert")

    # Retrieval indicator (base = keyword)
    i_concept = np.array([1.0 if r["retrieval"] == "concept" else 0.0 for r in panel])
    cols.append(i_concept)
    var_names.append("i_concept")

    # Segment cap indicators (base = 50)
    i_cap20 = np.array([1.0 if r["cap"] == 20 else 0.0 for r in panel])
    i_cap_none = np.array([1.0 if r["cap"] == "none" else 0.0 for r in panel])
    cols.append(i_cap20)
    var_names.append("i_cap20")
    cols.append(i_cap_none)
    var_names.append("i_cap_none")

    # Classifier indicators (base = LR)
    if classifier_ind:
        i_mlp = np.array([1.0 if r["method"] == "MLP" else 0.0 for r in panel])
        i_zs = np.array([1.0 if r["method"] == "ZS" else 0.0 for r in panel])
        cols.append(i_mlp)
        var_names.append("i_mlp")
        cols.append(i_zs)
        var_names.append("i_zs")

    # Interaction terms (multiplicative with continuous predictor)
    pred_vals = np.array([r[pred_key] for r in panel], dtype=float)
    if interactions in ("encoder", "all"):
        cols.append(pred_vals * i_minilm)
        var_names.append(f"{predictor}×i_minilm")
        cols.append(pred_vals * i_scibert)
        var_names.append(f"{predictor}×i_scibert")
    if interactions in ("retrieval", "all"):
        cols.append(pred_vals * i_concept)
        var_names.append(f"{predictor}×i_concept")
    if interactions in ("method", "all") and classifier_ind:
        cols.append(pred_vals * i_mlp)
        var_names.append(f"{predictor}×i_mlp")
        cols.append(pred_vals * i_zs)
        var_names.append(f"{predictor}×i_zs")

    # SDG fixed effects
    if with_sdg_fe:
        sdg_set = sorted({r["sdg"] for r in panel})
        for sdg in sdg_set[:-1]:
            d = np.array([1.0 if r["sdg"] == sdg else 0.0 for r in panel])
            cols.append(d)
            var_names.append(f"sdg{sdg}")

    X = np.column_stack(cols)
    cluster_ids = np.array([r["sdg"] for r in panel], dtype=int)

    # Drop degenerate columns (zero variance) to avoid singular matrix.
    # E.g. i_zs is all zeros in supervised-only subsample.
    keep = [0]  # always keep intercept
    for j in range(1, X.shape[1]):
        if np.std(X[:, j]) > 1e-10:
            keep.append(j)
    if len(keep) < X.shape[1]:
        dropped = [var_names[j] for j in range(X.shape[1]) if j not in keep]
        log.info("  Dropped degenerate columns: %s", dropped)
    X = X[:, keep]
    var_names = [var_names[j] for j in keep]

    return Y, X, cluster_ids, var_names


def ols_cluster_robust(
    Y: np.ndarray, X: np.ndarray, cluster_ids: np.ndarray,
) -> dict:
    """OLS with cluster-robust (sandwich) standard errors.

    Returns dict with beta, se, t, p, r2, adj_r2, n, n_clusters.
    """
    n, k = X.shape
    beta, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
    beta = beta.flatten()

    y_hat = X @ beta
    resid = Y - y_hat

    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((Y - Y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / (n - k) if n > k else 0.0

    unique_clusters = np.unique(cluster_ids)
    n_clusters = len(unique_clusters)

    bread = np.linalg.inv(X.T @ X)
    meat = np.zeros((k, k))
    for c in unique_clusters:
        mask = cluster_ids == c
        X_c = X[mask]
        e_c = resid[mask].reshape(-1, 1)
        meat += (X_c.T @ e_c) @ (e_c.T @ X_c)

    df_adj = n_clusters / (n_clusters - 1)
    V_robust = bread @ meat @ bread * df_adj

    se = np.sqrt(np.diag(V_robust))
    t = beta / se
    p = 2.0 * t_dist.sf(np.abs(t), df=n_clusters - 1)

    return {
        "beta": [round(float(b), 6) for b in beta],
        "se": [round(float(s), 6) for s in se],
        "t": [round(float(t_), 4) for t_ in t],
        "p": [round(float(p_), 6) for p_ in p],
        "r2": round(float(r2), 6),
        "adj_r2": round(float(adj_r2), 6),
        "n": int(n),
        "n_clusters": int(n_clusters),
    }


def ols_wls(
    Y: np.ndarray, X: np.ndarray, weights: np.ndarray, cluster_ids: np.ndarray,
) -> dict:
    """Weighted least squares with cluster-robust SE."""
    n, k = X.shape
    W = np.diag(weights)
    XtW = X.T @ W
    beta = np.linalg.solve(XtW @ X, XtW @ Y)

    resid = Y - X @ beta
    ss_res = float(np.sum(weights * resid ** 2))
    ss_tot = float(np.sum(weights * (Y - np.average(Y, weights=weights)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / (n - k) if n > k else 0.0

    # Cluster-robust sandwich: (X'WX)^-1 (Σ_g X_g'W_g ε_g ε_g' W_g X_g) (X'WX)^-1
    bread = np.linalg.inv(XtW @ X)
    unique_clusters = np.unique(cluster_ids)
    n_clusters = len(unique_clusters)
    meat = np.zeros((k, k))
    for c in unique_clusters:
        mask = cluster_ids == c
        X_c = X[mask]
        w_c = weights[mask]
        e_c = resid[mask]
        Xw = X_c * w_c[:, None]
        meat += Xw.T @ (e_c[:, None] * e_c[None, :] @ Xw)
    df_adj = n_clusters / (n_clusters - 1)
    V = bread @ meat @ bread * df_adj
    se = np.sqrt(np.diag(V))
    t = beta / se
    p = 2.0 * t_dist.sf(np.abs(t), df=n_clusters - 1)

    return {
        "beta": [round(float(b), 6) for b in beta],
        "se": [round(float(s), 6) for s in se],
        "t": [round(float(t_), 4) for t_ in t],
        "p": [round(float(p_), 6) for p_ in p],
        "r2": round(float(r2), 6),
        "adj_r2": round(float(adj_r2), 6),
        "n": int(n),
        "n_clusters": int(n_clusters),
    }


def ols_bootstrap(
    Y: np.ndarray, X: np.ndarray, cluster_ids: np.ndarray,
    n_boot: int = 500, seed: int = 42,
) -> dict:
    """Bootstrap SE by resampling SDG clusters."""
    rng = np.random.default_rng(seed)
    unique = np.unique(cluster_ids)
    betas = []
    for _ in range(n_boot):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        idx = np.isin(cluster_ids, sampled)
        b, _, _, _ = np.linalg.lstsq(X[idx], Y[idx], rcond=None)
        betas.append(b.flatten())
    betas = np.array(betas)
    return {
        "se_boot": [round(float(s), 6) for s in np.std(betas, axis=0)],
        "ci_lo": [round(float(c), 6) for c in np.percentile(betas, 2.5, axis=0)],
        "ci_hi": [round(float(c), 6) for c in np.percentile(betas, 97.5, axis=0)],
    }


# ---------------------------------------------------------------------------
# Run one spec
# ---------------------------------------------------------------------------

def run_spec(
    root: Path, spec: dict, all_panel: list[dict], do_bootstrap: bool = False,
) -> dict:
    """Run a single specification and return structured result."""
    spec_id = spec["spec_id"]
    log.info("  Running spec: %s", spec_id)

    panel = filter_panel(all_panel, spec["subsample"])
    if not panel:
        log.error("  Empty panel for spec %s after subsample=%s", spec_id, spec["subsample"])
        return {"spec_id": spec_id, "error": "empty_panel"}

    n_configs = len({(r["encoder"], r["retrieval"], r["method"], r["cap"]) for r in panel})
    n_sdgs = len({r["sdg"] for r in panel})

    Y, X, cluster_ids, var_names = _build_design(
        panel, spec["predictor"], spec["sdg_fe"],
        spec["classifier_ind"], spec["interactions"], spec["form"],
    )

    if spec["form"] == "wls":
        weights = np.array([r["n_papers"] for r in panel], dtype=float)
        weights = weights / weights.sum() * len(weights)  # normalize
        ols = ols_wls(Y, X, weights, cluster_ids)
    else:
        ols = ols_cluster_robust(Y, X, cluster_ids)

    result = {
        "spec_id": spec_id,
        "panel": spec["panel"],
        "label": spec["label"],
        "gap_type": spec["gap_type"],
        "predictor": spec["predictor"],
        "subsample": spec["subsample"],
        "interactions": spec["interactions"],
        "form": spec["form"],
        "sdg_fe": spec["sdg_fe"],
        "classifier_ind": spec["classifier_ind"],
        "n_configs": n_configs,
        "n_sdgs": n_sdgs,
        "n": ols["n"],
        "n_clusters": ols["n_clusters"],
        "r2": ols["r2"],
        "adj_r2": ols["adj_r2"],
        "var_names": var_names,
        "coef": {},
    }

    for i, name in enumerate(var_names):
        if name == "intercept":
            continue
        result["coef"][name] = {
            "b": ols["beta"][i],
            "se": ols["se"][i],
            "p": ols["p"][i],
        }

    if do_bootstrap or spec.get("bootstrap"):
        n_boot = spec.get("bootstrap", 500)
        log.info("    Bootstrap (%d reps, seed=42) ...", n_boot)
        boot = ols_bootstrap(Y, X, cluster_ids, n_boot=n_boot, seed=42)
        result["bootstrap"] = {
            "n_boot": n_boot,
            "seed": 42,
            "coef_ci": {},
        }
        for i, name in enumerate(var_names):
            if name == "intercept":
                continue
            result["bootstrap"]["coef_ci"][name] = {
                "b": ols["beta"][i],
                "se_boot": boot["se_boot"][i],
                "ci_lo": boot["ci_lo"][i],
                "ci_hi": boot["ci_hi"][i],
            }

    return result


# ---------------------------------------------------------------------------
# LaTeX writer — compact coef*(SE) format
# ---------------------------------------------------------------------------

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


def _fmt_coef_se(b: float, se: float, p: float, in_log: bool = False) -> tuple[str, str]:
    """Return (coef_line, se_line) for one cell."""
    stars = _sig_stars(p)
    if in_log:
        coef_str = f"{b:+.3f}{stars}"
    else:
        coef_str = f"{b:+.3f}{stars}"
    se_str = f"({se:.3f})"
    return coef_str, se_str


def write_spec_grid_tex(path: Path, results: list[dict], bootstrap_results: list[dict]) -> None:
    """Write compact specification grid table.

    Format: two rows per variable — coef+stars on top, SE in parentheses below.
    """
    n_specs = len(results)
    boot_map = {b["spec_id"]: b for b in bootstrap_results}

    # Determine which variables appear across all specs
    all_vars = []
    seen = set()
    for r in results:
        for v in r["var_names"]:
            if v != "intercept" and v not in seen:
                all_vars.append(v)
                seen.add(v)

    # Determine panel boundaries
    panels = []
    current_panel = None
    for r in results:
        if r["panel"] != current_panel:
            current_panel = r["panel"]
            panels.append({"label": current_panel, "start": len(panels), "specs": []})
        panels[-1]["specs"].append(r["spec_id"])

    panel_names = {"A": "Core", "B": "Robustness", "C": "Interactions", "D": "Functional form"}

    lines = [
        "% Auto-generated by k1_regression_semantic_gap.py",
        r"\small",
        f"\\begin{{tabular}}{{l*{{{n_specs}}}{{c}}}}",
        r"\toprule",
    ]

    # Panel header row
    panel_headers = []
    for p in panels:
        n = len(p["specs"])
        pname = panel_names.get(p["label"], p["label"])
        panel_headers.append(f"\\multicolumn{{{n}}}{{c}}{{\\textbf{{{pname}}}}}")
    lines.append(" & ".join(panel_headers) + r" \\")
    # cmidrules
    cmidrules = []
    col = 2
    for p in panels:
        n = len(p["specs"])
        cmidrules.append(f"\\cmidrule(lr){{{col}-{col + n - 1}}}")
        col += n
    lines.append(" ".join(cmidrules))

    # Spec numbers
    spec_nums = [f"({i+1})" for i in range(n_specs)]
    lines.append(" & ".join([""] + spec_nums) + r" \\")

    # Spec labels (short)
    labels = [r["label"] for r in results]
    lines.append(" & ".join([""] + labels) + r" \\")
    lines.append(r"\midrule")

    # Variables — two rows each
    for var in all_vars:
        label = VAR_LABELS.get(var, var)
        coef_cells = []
        se_cells = []
        has_any = False
        for r in results:
            c = r.get("coef", {}).get(var)
            if c is not None:
                has_any = True
                coef_str, se_str = _fmt_coef_se(c["b"], c["se"], c["p"])
                coef_cells.append(coef_str)
                se_cells.append(se_str)
            else:
                coef_cells.append("--")
                se_cells.append("")

        if has_any:
            lines.append(f"{label} & " + " & ".join(coef_cells) + r" \\")
            lines.append(" & " + " & ".join(se_cells) + r" \\")

    # Fit stats
    lines.append(r"\midrule")
    n_row = ["$N$"] + [str(r["n"]) for r in results]
    lines.append(" & ".join(n_row) + r" \\")
    r2_row = ["$R^2$"] + [f"{r['r2']:.3f}" for r in results]
    lines.append(" & ".join(r2_row) + r" \\")
    adjr2_row = ["Adj. $R^2$"] + [f"{r['adj_r2']:.3f}" for r in results]
    lines.append(" & ".join(adjr2_row) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OLS regression: semantic gap ~ coverage + indicators.")
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_ROOT))
    p.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL, type=resolve_model_alias, help=argparse.SUPPRESS)
    p.add_argument("--predictor", choices=["covgap", "dominance", "rescov"], default="covgap")
    p.add_argument("--gap-type", choices=["raw", "adjusted", "register"], default="raw")
    p.add_argument("--with-sdg-fe", action="store_true")
    p.add_argument("--no-classifier-indicator", action="store_true")
    p.add_argument("--subsample", choices=["all", "supervised", "keyword", "mpnet"], default="all")
    p.add_argument("--interactions", choices=["none", "encoder", "retrieval", "method", "all"], default="none")
    p.add_argument("--functional-form", choices=["rank", "raw", "log", "wls"], default="rank")
    p.add_argument("--bootstrap-se", type=int, default=0, metavar="N")
    p.add_argument("--spec-grid", action="store_true", help="Run all 21 specs in SPEC_GRID")
    p.add_argument("--overwrite", action="store_true", help=argparse.SUPPRESS)
    return p.parse_args()


def run(args: argparse.Namespace) -> None:
    root = Path(args.output_dir)
    model = args.embed_model

    # Ensure adjusted ZS gaps exist for MiniLM/SciBERT
    for zs_model in ("minilm", "scibert"):
        zs_cmd = [sys.executable, "1_code/6_calculate_centroids/score_zeroshot.py",
                  "--embed-model", zs_model, "--embeddings", "adjusted",
                  "--output-dir", str(root)]
        if args.overwrite:
            zs_cmd.append("--overwrite")
        subprocess.run(zs_cmd, check=True)

    if args.spec_grid:
        _run_spec_grid(root, model, args)
    else:
        _run_single(root, model, args)


def _run_single(root: Path, model: str, args: argparse.Namespace) -> None:
    """Run a single specification (backward-compatible mode)."""
    fe_tag = "_sdgfe" if args.with_sdg_fe else ""
    clf_tag = "_noclf" if args.no_classifier_indicator else ""
    sub_tag = f"_{args.subsample}" if args.subsample != "all" else ""
    int_tag = f"_int{args.interactions}" if args.interactions != "none" else ""
    form_tag = f"_{args.functional_form}" if args.functional_form != "rank" else ""
    boot_tag = f"_boot{args.bootstrap_se}" if args.bootstrap_se else ""
    tag = f"{args.gap_type}_{args.predictor}{fe_tag}{clf_tag}{sub_tag}{int_tag}{form_tag}{boot_tag}"

    subdir = f"appendix/k1_regression_semantic_gap/{tag}"
    layout = ensure_dissertation_outputs(root, subdir=subdir, model=model)
    out_json = layout.data_dir / "regression_results.json"
    out_tex = layout.tables_dir / "tab_k1_regression.tex"
    outputs = [out_json, out_tex]

    # Fingerprint
    fp_paths: list[Path] = []
    for _, m, method, corpus, _ in CONFIGS:
        base = output_dir_for_model(m, root=root) / "data"
        if corpus == "concept":
            base = base / "concept"
        fp_paths.append(base / "coverage_document_weighted.json")
        for suffix in ("lr", "mlp", "zeroshot"):
            fp_paths.append(base / f"semantic_gap_distances_{suffix}.json")
            fp_paths.append(base / "adjusted" / f"semantic_gap_distances_{suffix}.json")
            fp_paths.append(base / f"semantic_gap_robustness_caps_{suffix}.json")
            fp_paths.append(base / "adjusted" / f"semantic_gap_robustness_caps_{suffix}.json")

    fp = fingerprint_of(*fp_paths) + f"k1_reg_v2_{tag}"
    if should_skip(outputs, fp, args.overwrite, out_json):
        log.info("Skipping %s -- inputs unchanged", out_json)
        return

    log.info("Building panel (gap_type=%s, predictor=%s) ...", args.gap_type, args.predictor)
    all_panel = build_panel(root, args.gap_type)
    if not all_panel:
        log.error("Empty panel")
        return

    spec = {
        "spec_id": tag, "panel": "X", "label": tag,
        "gap_type": args.gap_type, "predictor": args.predictor,
        "subsample": args.subsample, "interactions": args.interactions,
        "form": args.functional_form, "sdg_fe": args.with_sdg_fe,
        "classifier_ind": not args.no_classifier_indicator,
        "bootstrap": args.bootstrap_se if args.bootstrap_se else None,
    }
    result = run_spec(root, spec, all_panel, do_bootstrap=args.bootstrap_se > 0)
    atomic_write_json(out_json, result)

    # Compact tex (single-spec version)
    _write_single_tex(out_tex, result)

    record_fingerprint(outputs, fp, out_json)
    _print_single(result)


def _run_spec_grid(root: Path, model: str, args: argparse.Namespace) -> None:
    """Run all 21 specs and write grid outputs."""
    subdir = "appendix/k1_regression_semantic_gap"
    layout = ensure_dissertation_outputs(root, subdir=subdir, model=model)
    out_grid = layout.data_dir / "spec_grid.json"
    out_boot = layout.data_dir / "bootstrap_grid.json"
    out_tex = layout.tables_dir / "tab_k1_specification_grid.tex"
    outputs = [out_grid, out_tex]

    # Fingerprint
    fp_paths: list[Path] = []
    for _, m, method, corpus, _ in CONFIGS:
        base = output_dir_for_model(m, root=root) / "data"
        if corpus == "concept":
            base = base / "concept"
        fp_paths.append(base / "coverage_document_weighted.json")
        for suffix in ("lr", "mlp", "zeroshot"):
            fp_paths.append(base / f"semantic_gap_distances_{suffix}.json")
            fp_paths.append(base / "adjusted" / f"semantic_gap_distances_{suffix}.json")
            fp_paths.append(base / f"semantic_gap_robustness_caps_{suffix}.json")
            fp_paths.append(base / "adjusted" / f"semantic_gap_robustness_caps_{suffix}.json")

    fp = fingerprint_of(*fp_paths) + f"k1_specgrid_v1_{len(SPEC_GRID)}specs"
    if should_skip(outputs, fp, args.overwrite, out_grid):
        log.info("Skipping spec grid -- inputs unchanged")
        return

    # Build panel once (gap_type=adjusted for all grid specs, but build per gap_type)
    log.info("Building panels for spec grid ...")
    panels_by_gap = {}
    for gt in ("adjusted", "raw", "register"):
        panels_by_gap[gt] = build_panel(root, gt)
        log.info("  %s: %d obs", gt, len(panels_by_gap[gt]))

    # Run all specs
    all_results = []
    boot_results = []
    for spec in SPEC_GRID:
        panel = panels_by_gap[spec["gap_type"]]
        result = run_spec(root, spec, panel, do_bootstrap=spec.get("bootstrap", 0) > 0)
        all_results.append(result)
        if "bootstrap" in result:
            boot_results.append({
                "spec_id": result["spec_id"],
                **result["bootstrap"],
            })

    # Write outputs
    atomic_write_json(out_grid, all_results)
    log.info("Saved: %s", out_grid)

    if boot_results:
        atomic_write_json(out_boot, boot_results)
        log.info("Saved: %s", out_boot)
        outputs.append(out_boot)

    write_spec_grid_tex(out_tex, all_results, boot_results)
    log.info("Saved: %s", out_tex)

    record_fingerprint(outputs, fp, out_grid)

    # Console summary
    _print_grid(all_results)


def _write_single_tex(path: Path, result: dict) -> None:
    """Write compact single-spec table."""
    var_names = result["var_names"]
    coef = result["coef"]

    lines = [
        "% Auto-generated by k1_regression_semantic_gap.py",
        r"\small",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r" & Coef. & SE \\",
        r"\midrule",
    ]

    for name in var_names:
        if name == "intercept":
            lines.append(r"\midrule")
        label = VAR_LABELS.get(name, name)
        c = coef.get(name)
        if c is not None:
            stars = _sig_stars(c["p"])
            lines.append(f"{label} & {c['b']:+.1f}{stars} & ({c['se']:.1f}) \\\\")
        else:
            lines.append(f"{label} & -- & -- \\\\")

    lines.extend([
        r"\midrule",
        f"$N$ & \\multicolumn{{2}}{{l}}{{{result['n']}}} \\\\",
        f"$R^2$ & \\multicolumn{{2}}{{l}}{{{result['r2']:.3f}}} \\\\",
        f"Adj. $R^2$ & \\multicolumn{{2}}{{l}}{{{result['adj_r2']:.3f}}} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
    ])

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _print_single(result: dict) -> None:
    """Console print for single spec."""
    if "error" in result:
        print(f"ERROR: {result['error']}")
        return
    print(f"\n{'='*50}")
    print(f"Spec: {result['spec_id']}  |  N={result['n']}  R²={result['r2']:.3f}")
    print(f"{'='*50}")
    print(f"{'Variable':<25s} {'Coef':>9s} {'SE':>9s} {'p':>8s}")
    print("-" * 50)
    for name in result["var_names"]:
        if name == "intercept":
            continue
        c = result["coef"].get(name)
        if c:
            stars = _sig_stars(c["p"])
            print(f"{name:<25s} {c['b']:>+9.1f} {c['se']:>9.1f} {c['p']:>8.4f}{stars}")
    print()


def _print_grid(results: list[dict]) -> None:
    """Console print for spec grid."""
    print(f"\n{'='*70}")
    print(f"SPECIFICATION GRID — {len(results)} specs")
    print(f"{'='*70}")
    print(f"{'Spec':<25s} {'N':>5s} {'R²':>6s} {'AdjR²':>6s}  Coefs")
    print("-" * 70)
    for r in results:
        if "error" in r:
            print(f"{r['spec_id']:<25s} ERROR: {r['error']}")
            continue
        # Show key coefs
        cov = r["coef"].get("covgap") or r["coef"].get("dominance")
        cov_str = f"cov={cov['b']:+.1f}({cov['se']:.1f}){ _sig_stars(cov['p'])}" if cov else ""
        pol = r["coef"].get("polcov")
        pol_str = f" pol={pol['b']:+.1f}({pol['se']:.1f}){_sig_stars(pol['p'])}" if pol else ""
        print(f"{r['spec_id']:<25s} {r['n']:>5d} {r['r2']:>6.3f} {r['adj_r2']:>6.3f}  {cov_str}{pol_str}")
    print()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()

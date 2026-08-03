"""
Consolidated register-correlation table (plan §6.5.3).

Per config x {rho(cov,raw), rho(cov,adj), rho(cov,register)}.
Reads per-config coverage + raw/adj semantic-gap JSONs.

Inputs (per encoder):
  4_outputs/{slug}/data/coverage_document_weighted.json       (LR coverage)
  4_outputs/{slug}/data/semantic_gap_distances_lr.json           (LR raw gaps)
  4_outputs/{slug}/data/adjusted/semantic_gap_distances_lr.json  (LR adj gaps)
  4_outputs/{slug}/data/semantic_gap_distances_zeroshot.json               (ZS raw gaps)
  4_outputs/{slug}/data/adjusted/semantic_gap_distances_zeroshot.json      (ZS adj gaps, MPNet only)
  4_outputs/{slug}/data/semantic_gap_distances_mlp.json       (MLP raw gaps)
  4_outputs/{slug}/data/adjusted/semantic_gap_distances_mlp.json (MLP adj gaps)
  4_outputs/mpnet/data/concept/coverage_document_weighted.json (concept coverage)
  4_outputs/mpnet/data/concept/semantic_gap_distances_lr.json    (concept raw gaps)

Outputs:
  4_outputs/appendix/{model}/h1_register_correlation_table/data/register_correlation_table.json
  4_outputs/appendix/{model}/h1_register_correlation_table/tables/tab_h1_register_correlation.tex

Run from project root:
  python 1_code/7_main_analysis/0_shared/h1_register_correlation_table.py --embed-model mpnet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = ROOT / "1_code"
ANALYSIS_ROOT = Path(__file__).resolve().parents[1]
SHARED_DIR = ANALYSIS_ROOT / "0_shared"
for path in (CODE_ROOT, SHARED_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from model_utils import (
    DEFAULT_EMBED_MODEL,
    DEFAULT_OUTPUT_ROOT,
    N_SDG,
    model_slug,
    output_dir_for_model,
    resolve_model_alias,
    scored_dir_for_model,
)
from shared_utils import (
    ensure_dissertation_outputs,
    fingerprint_of,
    record_fingerprint,
    should_skip,
)
from shard_pipeline_utils import atomic_write_json, load_json


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------

ENCODERS = [
    ("MPNet", "all-mpnet-base-v2"),
    ("MiniLM", "all-MiniLM-L6-v2"),
    ("SciBERT", "allenai/scibert_scivocab_uncased"),
]

# Each config: (label, encoder_model, method, corpus, has_adj_data)
CONFIGS = [
    ("MPNet LR",   "all-mpnet-base-v2",              "LR",   "canon",   True),
    ("MPNet MLP",  "all-mpnet-base-v2",              "MLP",  "canon",   True),
    ("MPNet ZS",   "all-mpnet-base-v2",              "ZS",   "canon",   True),
    ("MiniLM LR",  "all-MiniLM-L6-v2",               "LR",   "subset",  True),
    ("MiniLM MLP", "all-MiniLM-L6-v2",               "MLP",  "subset",  True),
    ("SciBERT LR", "allenai/scibert_scivocab_uncased", "LR",  "subset",  True),
    ("SciBERT MLP","allenai/scibert_scivocab_uncased", "MLP", "subset",  True),
    ("Concept LR", "all-mpnet-base-v2",              "LR",   "concept", True),
    ("Concept MLP","all-mpnet-base-v2",              "MLP",  "concept", True),
]


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return load_json(path)


def _coverage_gaps(root: Path, model: str, corpus: str) -> dict[int, float] | None:
    """Load coverage_gap_hard keyed by SDG number."""
    if corpus == "concept":
        p = output_dir_for_model(model, root=root) / "data" / "concept" / "coverage_document_weighted.json"
    else:
        p = output_dir_for_model(model, root=root) / "data" / "coverage_document_weighted.json"
    data = _load_json(p)
    if data is None:
        return None
    cg = data.get("coverage_gap_hard")
    if not cg:
        return None
    return {int(k[3:]): v for k, v in cg.items()}


def _lr_raw_gaps(root: Path, model: str) -> dict[int, float] | None:
    p = output_dir_for_model(model, root=root) / "data" / "semantic_gap_distances_lr.json"
    data = _load_json(p)
    if data is None:
        return None
    return {row["sdg"]: row["semantic_gap"] for row in data["per_sdg"] if row["semantic_gap"] is not None}


def _lr_adj_gaps(root: Path, model: str) -> dict[int, float] | None:
    p = output_dir_for_model(model, root=root) / "data" / "adjusted" / "semantic_gap_distances_lr.json"
    data = _load_json(p)
    if data is None:
        return None
    return {row["sdg"]: row["semantic_gap"] for row in data["per_sdg"] if row["semantic_gap"] is not None}


def _zs_raw_gaps(root: Path, model: str) -> dict[int, float] | None:
    p = output_dir_for_model(model, root=root) / "data" / "semantic_gap_distances_zeroshot.json"
    data = _load_json(p)
    if data is None:
        return None
    return {row["sdg"]: row["semantic_gap"] for row in data["per_sdg"] if row["semantic_gap"] is not None}


def _zs_adj_gaps(root: Path, model: str) -> dict[int, float] | None:
    p = output_dir_for_model(model, root=root) / "data" / "adjusted" / "semantic_gap_distances_zeroshot.json"
    data = _load_json(p)
    if data is None:
        return None
    return {row["sdg"]: row["semantic_gap"] for row in data["per_sdg"] if row["semantic_gap"] is not None}


def _mlp_raw_gaps(root: Path, model: str) -> dict[int, float] | None:
    """Load MLP raw gaps from the new semantic_gap_distances_mlp.json."""
    p = output_dir_for_model(model, root=root) / "data" / "semantic_gap_distances_mlp.json"
    data = _load_json(p)
    if data is None:
        return None
    return {row["sdg"]: row["semantic_gap"] for row in data["per_sdg"] if row["semantic_gap"] is not None}


def _mlp_adj_gaps(root: Path, model: str) -> dict[int, float] | None:
    """Load MLP adjusted gaps from adjusted/semantic_gap_distances_mlp.json."""
    p = output_dir_for_model(model, root=root) / "data" / "adjusted" / "semantic_gap_distances_mlp.json"
    data = _load_json(p)
    if data is None:
        return None
    return {row["sdg"]: row["semantic_gap"] for row in data["per_sdg"] if row["semantic_gap"] is not None}


def _concept_raw_gaps(root: Path) -> dict[int, float] | None:
    p = output_dir_for_model("all-mpnet-base-v2", root=root) / "data" / "concept" / "semantic_gap_distances_lr.json"
    data = _load_json(p)
    if data is None:
        return None
    return {row["sdg"]: row["semantic_gap"] for row in data["per_sdg"] if row["semantic_gap"] is not None}


def _concept_adj_gaps(root: Path) -> dict[int, float] | None:
    p = output_dir_for_model("all-mpnet-base-v2", root=root) / "data" / "concept" / "adjusted" / "semantic_gap_distances_lr.json"
    data = _load_json(p)
    if data is None:
        return None
    return {row["sdg"]: row["semantic_gap"] for row in data["per_sdg"] if row["semantic_gap"] is not None}


def _concept_mlp_raw_gaps(root: Path) -> dict[int, float] | None:
    p = output_dir_for_model("all-mpnet-base-v2", root=root) / "data" / "concept" / "semantic_gap_distances_mlp.json"
    data = _load_json(p)
    if data is None:
        return None
    return {row["sdg"]: row["semantic_gap"] for row in data["per_sdg"] if row["semantic_gap"] is not None}


def _concept_mlp_adj_gaps(root: Path) -> dict[int, float] | None:
    p = output_dir_for_model("all-mpnet-base-v2", root=root) / "data" / "concept" / "adjusted" / "semantic_gap_distances_mlp.json"
    data = _load_json(p)
    if data is None:
        return None
    return {row["sdg"]: row["semantic_gap"] for row in data["per_sdg"] if row["semantic_gap"] is not None}


# ---------------------------------------------------------------------------
# Correlation
# ---------------------------------------------------------------------------

def _spearman(x: np.ndarray, y: np.ndarray) -> dict:
    rho, p = stats.spearmanr(x, y)
    return {"rho": round(float(rho), 6), "p": round(float(p), 6)}


# ---------------------------------------------------------------------------
# Table builder
# ---------------------------------------------------------------------------

def _build_config_row(
    label: str,
    model: str,
    method: str,
    corpus: str,
    has_adj: bool,
    root: Path,
) -> dict | None:
    """Compute correlations for one config. Returns None if coverage data missing."""
    cov = _coverage_gaps(root, model, corpus)
    if cov is None:
        return None

    # Raw gap
    if method == "LR":
        raw = _lr_raw_gaps(root, model) if corpus != "concept" else _concept_raw_gaps(root)
    elif method == "ZS":
        raw = _zs_raw_gaps(root, model)
    elif method == "MLP":
        if corpus == "concept":
            raw = _concept_mlp_raw_gaps(root)
        else:
            raw = _mlp_raw_gaps(root, model)
    else:
        return None

    if raw is None:
        return None

    # Adjusted gap
    adj = None
    if has_adj:
        if method == "LR":
            adj = _lr_adj_gaps(root, model) if corpus != "concept" else _concept_adj_gaps(root)
        elif method == "ZS":
            adj = _zs_adj_gaps(root, model)
        elif method == "MLP":
            if corpus == "concept":
                adj = _concept_mlp_adj_gaps(root)
            else:
                adj = _mlp_adj_gaps(root, model)

    # Align on common SDGs
    common_raw = sorted(set(cov) & set(raw))
    cov_arr = np.array([cov[sdg] for sdg in common_raw])
    raw_arr = np.array([raw[sdg] for sdg in common_raw])

    n_valid = len(common_raw)

    # Raw correlation
    rho_cov_raw = _spearman(cov_arr, raw_arr)

    # Adjusted + register correlations (only if adj data available)
    rho_cov_adj = None
    rho_cov_register = None
    if adj is not None:
        common_adj = sorted(set(common_raw) & set(adj))
        if len(common_adj) >= 3:
            mask = np.array([sdg in adj for sdg in common_raw])
            adj_arr = np.array([adj.get(sdg, np.nan) for sdg in common_raw])
            reg_arr = raw_arr - adj_arr

            valid = np.isfinite(raw_arr) & np.isfinite(adj_arr) & mask
            n_valid_adj = int(valid.sum())
            if n_valid_adj >= 3:
                rho_cov_adj = _spearman(cov_arr[valid], adj_arr[valid])
                rho_cov_register = _spearman(cov_arr[valid], reg_arr[valid])
                n_valid = n_valid_adj

    return {
        "label": label,
        "encoder": model,
        "method": method,
        "corpus": corpus,
        "n_valid": n_valid,
        "rho_cov_raw": rho_cov_raw,
        "rho_cov_adj": rho_cov_adj,
        "rho_cov_register": rho_cov_register,
    }


# ---------------------------------------------------------------------------
# TeX writer
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


def _fmt_rho(v: dict | None) -> str:
    if v is None:
        return "--"
    return f"{v['rho']:+.3f}{_sig_stars(v['p'])}"


def _write_tex(path: Path, configs: list[dict]) -> None:
    lines = [
        "% Auto-generated by 1_code/7_main_analysis/0_shared/h1_register_correlation_table.py",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r" & Raw gap & Adj.\ gap & Register \\",
        r"\midrule",
    ]

    for c in configs:
        row = [
            c["label"],
            _fmt_rho(c["rho_cov_raw"]),
            _fmt_rho(c["rho_cov_adj"]),
            _fmt_rho(c["rho_cov_register"]),
        ]
        lines.append(" & ".join(row) + r" \\")

    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Written {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    root = Path(args.output_dir)
    model = args.embed_model

    layout = ensure_dissertation_outputs(
        root, subdir="appendix/h1_register_correlation_table", model=model,
    )
    out_json = layout.data_dir / "register_correlation_table.json"
    out_tex = layout.tables_dir / "tab_h1_register_correlation.tex"
    outputs = [out_json, out_tex]

    # Fingerprint all input files
    fp_paths: list[Path] = []
    for _, m, method, corpus, _ in CONFIGS:
        if corpus == "concept":
            fp_paths.append(output_dir_for_model(m, root=root) / "data" / "concept" / "coverage_document_weighted.json")
            if method == "LR":
                fp_paths.append(output_dir_for_model(m, root=root) / "data" / "concept" / "semantic_gap_distances_lr.json")
                fp_paths.append(output_dir_for_model(m, root=root) / "data" / "concept" / "adjusted" / "semantic_gap_distances_lr.json")
            elif method == "MLP":
                fp_paths.append(output_dir_for_model(m, root=root) / "data" / "concept" / "semantic_gap_distances_mlp.json")
                fp_paths.append(output_dir_for_model(m, root=root) / "data" / "concept" / "adjusted" / "semantic_gap_distances_mlp.json")
        else:
            fp_paths.append(output_dir_for_model(m, root=root) / "data" / "coverage_document_weighted.json")
            if method == "LR":
                fp_paths.append(output_dir_for_model(m, root=root) / "data" / "semantic_gap_distances_lr.json")
                fp_paths.append(output_dir_for_model(m, root=root) / "data" / "adjusted" / "semantic_gap_distances_lr.json")
            elif method == "ZS":
                fp_paths.append(output_dir_for_model(m, root=root) / "data" / "semantic_gap_distances_zeroshot.json")
                fp_paths.append(output_dir_for_model(m, root=root) / "data" / "adjusted" / "semantic_gap_distances_zeroshot.json")
            elif method == "MLP":
                fp_paths.append(output_dir_for_model(m, root=root) / "data" / "semantic_gap_distances_mlp.json")
                fp_paths.append(output_dir_for_model(m, root=root) / "data" / "adjusted" / "semantic_gap_distances_mlp.json")
                # Also include legacy path for fingerprint stability
                fp_paths.append(scored_dir_for_model(m) / "mlp_scores" / "mlp_summary.json")

    fp = fingerprint_of(*fp_paths) + "h1_corr_v3"
    if should_skip(outputs, fp, args.overwrite, out_json):
        print(f"Skipping {out_json} -- inputs unchanged")
        return

    # Build rows
    rows = []
    for label, m, method, corpus, has_adj in CONFIGS:
        row = _build_config_row(label, m, method, corpus, has_adj, root)
        if row is not None:
            rows.append(row)
        else:
            print(f"WARNING: missing data for {label} -- skipping row")

    if not rows:
        print("ERROR: no config rows produced -- all inputs missing?")
        return

    # Write outputs
    payload = {
        "embedding_model": model,
        "n_configs": len(rows),
        "configs": rows,
    }
    atomic_write_json(out_json, payload)
    _write_tex(out_tex, rows)
    record_fingerprint(outputs, fp, out_json)

    # Summary
    print(f"\nCorrelation table: {len(rows)} configs")
    for c in rows:
        raw_rho = c["rho_cov_raw"]["rho"] if c["rho_cov_raw"] else "N/A"
        adj_rho = c["rho_cov_adj"]["rho"] if c["rho_cov_adj"] else "--"
        reg_rho = c["rho_cov_register"]["rho"] if c["rho_cov_register"] else "--"
        print(f"  {c['label']:15s}  raw={raw_rho:>7}  adj={adj_rho:>7}  reg={reg_rho:>7}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Consolidated register-correlation table (§6.5.3).",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL, type=resolve_model_alias, help=argparse.SUPPRESS)
    parser.add_argument("--overwrite", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()

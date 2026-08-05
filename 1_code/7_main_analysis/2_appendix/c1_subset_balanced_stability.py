"""
Measure whether the within-SDG semantic-gap ranking survives when the research
side is subsampled to sizes comparable to the policy corpus.

This stage addresses the corpus-asymmetry robustness check for the unit-count
imbalance (3,105,144 research segments from 2,536,771 papers vs 40,597 policy
segments, a ~76x segment-unit ratio): the full-corpus research centroids
over-represent the research side by ~76x at the segment level. We reuse the
existing sample-stability draws (Appendix C, c_sample_stability.py) — each draw
is a random 50k-paper research subset whose per-SDG semantic gaps were computed
against the fixed canonical policy centroids — and correlate each draw's
within-SDG semantic-gap ranking with the full-corpus ranking from
semantic_gap_distances_lr.json.

A high Spearman rho at the ~50k tier (comparable to the policy corpus size)
shows the ranking of which SDGs have large vs small semantic gaps is not an
artefact of research-side over-representation. This stage reads no embeddings
and re-scores nothing.

Outputs:
  4_outputs/appendix/{model}/c1_subset_balanced_stability/data/c1_subset_balanced_stability.json
  4_outputs/appendix/{model}/c1_subset_balanced_stability/tables/num_c1_subset_stability.tex
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = ROOT / "1_code"
ANALYSIS_ROOT = Path(__file__).resolve().parents[1]
SHARED_DIR = ANALYSIS_ROOT / "0_shared"
for path in (CODE_ROOT, SHARED_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from model_utils import DEFAULT_EMBED_MODEL, DEFAULT_OUTPUT_ROOT, output_dir_for_model, resolve_model_alias
from shared_utils import fingerprint_of, should_skip, record_fingerprint
from shared_utils import ensure_dissertation_outputs

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

CANONICAL_SEMANTIC_JSON = "semantic_gap_distances_lr.json"
DRAWS_JSONL = "c_sample_stability_draws.jsonl"
MIN_VALID_SDGS = 3
SCRIPT_VERSION = "1"

TIER_WORD = {
    "1k": "OneK",
    "2k": "TwoK",
    "5k": "FiveK",
    "10k": "TenK",
    "20k": "TwentyK",
    "50k": "FiftyK",
    "100k": "HundredK",
    "200k": "TwoHundredK",
    "500k": "FiveHundredK",
    "1m": "OneM",
    "2m": "TwoM",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the balanced-subset rank-stability stage.")
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_ROOT))
    p.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL, type=resolve_model_alias, help=argparse.SUPPRESS)
    p.add_argument("--overwrite", action="store_true", help=argparse.SUPPRESS)
    return p.parse_args()


def load_full_corpus_gaps(canonical_data_dir: Path) -> dict[str, float]:
    """Full-corpus per-SDG semantic gaps, keyed by 'SDG1'..'SDG17'."""
    payload = json.loads((canonical_data_dir / CANONICAL_SEMANTIC_JSON).read_text(encoding="utf-8"))
    gaps = {}
    for row in payload["per_sdg"]:
        sdg = int(row["sdg"])
        gap = row["semantic_gap"]
        if gap is not None:
            gaps[f"SDG{sdg}"] = float(gap)
    return gaps


def spearman_rho(full_gaps: dict[str, float], draw_gaps: dict[str, float | None]) -> float | None:
    """Spearman rho between a draw's per-SDG gaps and the full-corpus ranking."""
    common = [
        s for s in full_gaps if s in draw_gaps and draw_gaps[s] is not None
    ]
    if len(common) < MIN_VALID_SDGS:
        return None
    rho = spearmanr(
        [full_gaps[s] for s in common],
        [float(draw_gaps[s]) for s in common],
    ).statistic
    if not np.isfinite(rho):
        return None
    return float(rho)


def summarize_tiers(draws: list[dict[str, Any]], full_gaps: dict[str, float]) -> list[dict[str, Any]]:
    tiers: dict[str, list[dict[str, Any]]] = {}
    for draw in draws:
        tiers.setdefault(draw["tier_label"], []).append(draw)

    rows: list[dict[str, Any]] = []
    for tier_label in TIER_WORD:
        tier_draws = tiers.get(tier_label, [])
        rhos = [
            rho
            for draw in tier_draws
            if (rho := spearman_rho(full_gaps, draw["semantic_gap_by_sdg"])) is not None
        ]
        n_sdgs = [
            sum(
                1
                for s in full_gaps
                if s in draw["semantic_gap_by_sdg"] and draw["semantic_gap_by_sdg"][s] is not None
            )
            for draw in tier_draws
        ]
        rows.append(
            {
                "tier_label": tier_label,
                "sample_size": tier_draws[0]["sample_size"] if tier_draws else None,
                "n_draws": len(tier_draws),
                "n_usable_draws": len(rhos),
                "n_sdgs_used_median": float(np.median(n_sdgs)) if n_sdgs else None,
                "mean_rho": float(np.mean(rhos)) if rhos else None,
                "std_rho": float(np.std(rhos)) if rhos else None,
                "min_rho": float(np.min(rhos)) if rhos else None,
                "max_rho": float(np.max(rhos)) if rhos else None,
            }
        )
    return rows


def write_outputs(layout: Any, rows: list[dict[str, Any]], full_mean_gap: float | None) -> Path:
    payload = {
        "method": "subset_balanced_rank_stability",
        "note": (
            "Spearman rho between each sample-stability draw's within-SDG semantic-gap "
            "ranking and the full-corpus ranking (semantic_gap_distances_lr.json). "
            "The 50k tier is comparable to the 40,597-segment policy corpus."
        ),
        "full_corpus_mean_semantic_gap": full_mean_gap,
        "tiers": rows,
    }
    json_path = layout.data_dir / "c1_subset_balanced_stability.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    num_lines = [
        "% Auto-generated by 1_code/7_main_analysis/2_appendix/c1_subset_balanced_stability.py — do not edit manually",
    ]
    for row in rows:
        word = TIER_WORD[row["tier_label"]]
        if row["mean_rho"] is None:
            continue
        num_lines.append(
            rf"\newcommand{{\SubsetGapRho{word}}}{{{row['mean_rho']:.3f}}}"
        )
        num_lines.append(
            rf"\newcommand{{\SubsetGapRhoStd{word}}}{{{row['std_rho']:.3f}}}"
        )
        num_lines.append(
            rf"\newcommand{{\SubsetGapRhoDraws{word}}}{{{int(row['n_usable_draws'])}}}"
        )
        if row["tier_label"] == "50k":
            num_lines.append(
                rf"\newcommand{{\SubsetGapRhoFiftyKN}}{{{int(row['n_usable_draws'])}}}"
            )
            num_lines.append(
                rf"\newcommand{{\SubsetGapSdgsUsed}}{{{int(round(row['n_sdgs_used_median']))}}}"
            )
    num_path = layout.tables_dir / "num_c1_subset_stability.tex"
    num_path.write_text("\n".join(num_lines) + "\n", encoding="utf-8")
    return json_path


def run(args: argparse.Namespace) -> None:
    model = args.embed_model
    layout = ensure_dissertation_outputs(
        Path(args.output_dir), subdir="appendix/c1_subset_balanced_stability", model=model
    )
    canonical_data_dir = output_dir_for_model(model, root=Path(args.output_dir)) / "data"
    stability_data_dir = ensure_dissertation_outputs(
        Path(args.output_dir), subdir="appendix/c_sample_stability", model=model
    ).data_dir

    full_gap_path = canonical_data_dir / CANONICAL_SEMANTIC_JSON
    draws_path = stability_data_dir / DRAWS_JSONL
    for path in (full_gap_path, draws_path):
        if not path.exists():
            raise FileNotFoundError(
                f"C1 subset-balanced stability requires {path} "
                "(run semantic_gap + c_sample_stability first)."
            )

    primary = layout.data_dir / "c1_subset_balanced_stability.json"
    outputs = [
        primary,
        layout.tables_dir / "num_c1_subset_stability.tex",
    ]
    fp = fingerprint_of(full_gap_path, draws_path) + SCRIPT_VERSION
    if should_skip(outputs, fp, args.overwrite, primary):
        log.info("Skipping %s — inputs unchanged", primary)
        return

    full_gaps = load_full_corpus_gaps(canonical_data_dir)
    full_mean_gap = float(np.mean(list(full_gaps.values()))) if full_gaps else None
    draws = [
        json.loads(line)
        for line in draws_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows = summarize_tiers(draws, full_gaps)

    for row in rows:
        if row["mean_rho"] is None:
            log.info(
                "Tier %s: no usable draws (mean rho unavailable)",
                row["tier_label"],
            )
        else:
            log.info(
                "Tier %-5s n=%7d  rho=%.3f +/- %.3f (usable draws %d/%d, median %d SDGs)",
                row["tier_label"],
                row["sample_size"],
                row["mean_rho"],
                row["std_rho"],
                row["n_usable_draws"],
                row["n_draws"],
                int(round(row["n_sdgs_used_median"] or 0)),
            )

    write_outputs(layout, rows, full_mean_gap)
    log.info("Saved balanced-subset rank-stability outputs into %s", layout.data_dir)
    record_fingerprint(outputs, fp, primary)


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()

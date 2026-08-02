"""
Policy source-family sensitivity diagnostic.

This appendix-stage diagnostic checks whether the policy-side coverage and semantic-gap
patterns are broadly stable across the three existing policy source families:

  1. Curated AI/SDG policy documents (policy_scrape + policy_manual)
  2. SDGi VNR/VLR reports
  3. UN General Debate Corpus (UNGDC)

It does not alter the canonical hard-label results. It only reports how policy-side
profiles and within-SDG semantic gaps look when the policy corpus is split by source family.

It additionally replicates the canonical H25 interaction test (four coverage predictors
vs the within-SDG semantic gap) for each source family. The curated AI/SDG family is the
symmetric AI/SDG-vs-AI/SDG control for the research corpus's AI cap SDG scope (PLAN Item 1);
preserving the full-corpus null under this construction shows the dissociation is not an
artefact of the research/policy corpus asymmetry.

Run from project root:
    python 1_code/7_main_analysis/2_appendix/a2_policy_source_family_sensitivity.py
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from collections import defaultdict
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


from model_utils import DEFAULT_EMBED_MODEL, DEFAULT_OUTPUT_ROOT, embed_dir_for_model, model_slug, resolve_model_alias
from shard_pipeline_utils import load_json
from shared_utils import fingerprint_of, should_skip, record_fingerprint
import semantic_gap_shared
import register_utils
from semantic_gap_shared import (
    SEGMENT_CAP_PRIMARY,
    MIN_CLUSTER_SIZE,
    N_SDG,
    RANDOM_SEED,
    build_source_family_map,
    build_sub_centroid,
    cap_policy_indices_per_doc,
    get_cluster_assignments,
    write_csv,
)

SUMMARY_CSV = "policy_source_family_summary.csv"
COVERAGE_CSV = "policy_source_family_coverage.csv"
SEMANTIC_CSV = "policy_source_family_semantic_gaps.csv"
H25_CSV = "policy_source_family_h25.csv"
H25_JSON = "policy_source_family_h25.json"
H25_TEX = "tab_a2_policy_source_family_h25.tex"

CANONICAL_COVERAGE_FILE = "4_2_coverage_document_weighted.json"


FAMILY_ORDER = [
    "full_policy_corpus",
    "curated_ai_sdg",
    "sdgi_vnr_vlr",
    "ungdc_speeches",
]

FAMILY_LABELS = {
    "full_policy_corpus": "Full corpus",
    "curated_ai_sdg": "Curated AI/SDG",
    "sdgi_vnr_vlr": "SDGi VNR/VLR",
    "ungdc_speeches": "UNGDC speeches",
}

INTERPRETATION_NOTES = {
    "full_policy_corpus": "Full corpus",
    "curated_ai_sdg": "AI/SDG-specific",
    "sdgi_vnr_vlr": "VNR/VLR institutional reports",
    "ungdc_speeches": "UN speech discourse",
}

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run policy source-family sensitivity diagnostic.")
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_ROOT))
    p.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL, type=resolve_model_alias, help=argparse.SUPPRESS)
    p.add_argument("--embeddings", choices=["raw", "adjusted"], default="raw",
                   help="Use raw (default) or register-adjusted embeddings (project via G).")
    p.add_argument("--overwrite", action="store_true", help=argparse.SUPPRESS)
    return p.parse_args()


def document_weighted_policy_profile_subset(
    policy_scores: np.ndarray,
    policy_ids: list[dict],
    subset_indices: list[int],
) -> tuple[np.ndarray, dict[str, dict]]:
    # Delegates to the single source of truth for Assumption A19 document-weighting
    # (semantic_gap_shared.document_weighted_policy_profile), restricted to the
    # subset's policy row indices.
    if not subset_indices:
        raise RuntimeError("Subset has no policy documents.")
    return semantic_gap_shared.document_weighted_policy_profile(
        policy_scores, policy_ids, subset_indices=subset_indices
    )


def compute_family_semantic_rows(
    family: str,
    subset_indices: list[int],
    policy_emb: np.ndarray,
    policy_assignments: np.ndarray,
    policy_ids: list[dict],
    research_centroids: np.ndarray,
    research_meta: list[dict],
) -> list[dict]:
    subset_lookup = set(subset_indices)
    rows: list[dict] = []

    for sdg_idx in range(N_SDG):
        sdg = sdg_idx + 1
        family_sdg_idxs = [
            idx for idx in subset_indices
            if int(policy_assignments[idx]) == sdg_idx
        ]
        n_segments = len(family_sdg_idxs)
        capped = cap_policy_indices_per_doc(
            family_sdg_idxs,
            policy_ids,
            SEGMENT_CAP_PRIMARY,
            np.random.default_rng(RANDOM_SEED + sdg_idx),
        )
        n_segments_capped = len(capped)
        raw_docs = {policy_ids[i]["source_doc"] for i in family_sdg_idxs}
        capped_docs = {policy_ids[i]["source_doc"] for i in capped}
        n_papers = int(research_meta[sdg_idx]["n_papers_assigned"])
        unreliable = n_papers < MIN_CLUSTER_SIZE or n_segments_capped < MIN_CLUSTER_SIZE

        pol_centroid, pol_cohesion = build_sub_centroid(policy_emb, capped)
        if pol_centroid is None:
            sim = None
            gap = None
        else:
            sim = float(np.dot(research_centroids[sdg_idx], pol_centroid))
            gap = 1.0 - sim

        rows.append(
            {
                "family": family,
                "sdg": sdg,
                "n_research_papers": n_papers,
                "n_policy_segments": n_segments,
                "n_policy_segments_capped": n_segments_capped,
                "n_policy_docs": len(raw_docs),
                "n_policy_docs_capped": len(capped_docs),
                "segment_cap": SEGMENT_CAP_PRIMARY,
                "semantic_similarity": None if sim is None else round(sim, 6),
                "semantic_gap": None if gap is None else round(gap, 6),
                "policy_cohesion": round(pol_cohesion, 6),
                "unreliable": unreliable or gap is None,
            }
        )
    return rows


def top3_sdgs(profile: np.ndarray) -> str:
    order = np.argsort(profile)[::-1][:3]
    return " / ".join(f"SDG {i + 1}" for i in order)


def _spearman(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 3:
        return 1.0
    rx = np.argsort(np.argsort(xs))
    ry = np.argsort(np.argsort(ys))
    d2 = ((rx - ry) ** 2).sum()
    return 1 - (6 * d2) / (n * (n * n - 1))


# ---------------------------------------------------------------------------
# Per-family H25 replication (mirrors the canonical interaction test in
# 1_main_text/2_coverage_semantic_interaction.py): do the four coverage
# predictors (research share, policy share, |research-policy| coverage gap,
# signed dominance) correlate with the within-SDG semantic gap?
# ---------------------------------------------------------------------------
def _pearson_and_spearman(x: np.ndarray, y: np.ndarray) -> dict:
    """Pearson r and Spearman rho (with p-values) between x and y."""
    if len(x) < 3:
        return {"n": int(len(x)), "skipped": True, "reason": "fewer_than_3_valid_sdgs"}
    r, r_p = stats.pearsonr(x, y)
    rho, s_p = stats.spearmanr(x, y)
    return {
        "n": int(len(x)),
        "pearson_r": round(float(r), 6),
        "pearson_p": round(float(r_p), 6),
        "spearman_rho": round(float(rho), 6),
        "spearman_p": round(float(s_p), 6),
    }


def _correlation_or_skip(x: np.ndarray, y: np.ndarray, mask: np.ndarray, label: str) -> dict:
    if int(mask.sum()) < 3:
        return {
            "n": int(mask.sum()),
            "sdgs": [i + 1 for i, keep in enumerate(mask) if keep],
            "skipped": True,
            "reason": "fewer_than_3_valid_sdgs",
        }
    result = _pearson_and_spearman(x[mask], y[mask])
    result["sdgs"] = [i + 1 for i, keep in enumerate(mask) if keep]
    result["skipped"] = False
    return result


def _compute_four_tests(res: np.ndarray, pol: np.ndarray, covgap: np.ndarray, dom: np.ndarray,
                        sem_gap: np.ndarray, mask: np.ndarray) -> dict:
    return {
        "research": _correlation_or_skip(res, sem_gap, mask, "research share vs semantic gap"),
        "policy": _correlation_or_skip(pol, sem_gap, mask, "policy share vs semantic gap"),
        "covgap": _correlation_or_skip(covgap, sem_gap, mask, "coverage gap vs semantic gap"),
        "dominance": _correlation_or_skip(dom, sem_gap, mask, "dominance vs semantic gap"),
    }


def _fisher_ci(r: float, n: int) -> tuple[float, float]:
    """Fisher-z 95% CI for a Pearson r, matching the canonical interaction test."""
    if n < 4 or not np.isfinite(r):
        return float("nan"), float("nan")
    z = math.atanh(max(-0.999, min(0.999, r)))
    se = 1.0 / math.sqrt(n - 3)
    return math.tanh(z - 1.96 * se), math.tanh(z + 1.96 * se)


def _fmt2(v: float) -> str:
    s = f"{abs(v):.2f}"
    return f"-{s}" if v < 0 else s


def compute_family_h25(
    research_share: np.ndarray,
    policy_share: np.ndarray,
    sem_gap: np.ndarray,
    unreliable: np.ndarray,
) -> dict:
    """Run the four-predictor H25 test for one policy family."""
    available = np.isfinite(sem_gap) & ~unreliable
    covgap = np.abs(research_share - policy_share)
    dominance = research_share - policy_share
    tests_primary = _compute_four_tests(research_share, policy_share, covgap, dominance, sem_gap, available)
    excl4 = available.copy()
    excl4[3] = False  # SDG 4 is index 3
    tests_excl4 = _compute_four_tests(research_share, policy_share, covgap, dominance, sem_gap, excl4)
    primary = tests_primary["research"]
    n_primary = int(primary["n"]) if not primary.get("skipped") else 0
    lo, hi = _fisher_ci(primary.get("pearson_r", float("nan")), n_primary)
    return {
        "tests_primary": tests_primary,
        "tests_excl4": tests_excl4,
        "n_primary": n_primary,
        "primary_fisher_ci_lower": float(lo),
        "primary_fisher_ci_upper": float(hi),
    }


def write_table_h25(path: Path, h25_rows: list[dict]) -> None:
    lines = [
        "% Auto-generated by 1_code/7_main_analysis/2_appendix/a2_policy_source_family_sensitivity.py — do not edit manually",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Family & $n$ & Research share $(\rho,\,p)$ & Coverage gap $(\rho,\,p)$ & Research share excl.\ SDG 4 $(\rho,\,p)$ \\",
        r"\midrule",
    ]
    for row in h25_rows:
        r = row["research"]
        cg = row["covgap"]
        r4 = row["research_excl4"]
        if r.get("skipped"):
            r_cell = "--"
        else:
            r_cell = f"{_fmt2(r['spearman_rho'])} ({r['spearman_p']:.3f})"
        if cg.get("skipped"):
            cg_cell = "--"
        else:
            cg_cell = f"{_fmt2(cg['spearman_rho'])} ({cg['spearman_p']:.3f})"
        if r4.get("skipped"):
            r4_cell = "--"
        else:
            r4_cell = f"{_fmt2(r4['spearman_rho'])} ({r4['spearman_p']:.3f})"
        lines.append(
            rf"{row['family_label']} & {row['n_primary']} & {r_cell} & {cg_cell} & {r4_cell} \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_h25_macros(path: Path, h25_rows: list[dict]) -> None:
    by_family = {row["family"]: row for row in h25_rows}
    lines = [
        "% Auto-generated by 1_code/7_main_analysis/2_appendix/a2_policy_source_family_sensitivity.py — do not edit manually",
    ]
    prefix_map = {
        "full_policy_corpus": "FullPolicy",
        "curated_ai_sdg": "Curated",
        "sdgi_vnr_vlr": "Sdgi",
        "ungdc_speeches": "Ungdc",
    }
    for family in FAMILY_ORDER:
        if family not in by_family:
            continue
        row = by_family[family]
        prefix = prefix_map[family]
        r = row["research"]
        cg = row["covgap"]
        r4 = row["research_excl4"]
        rho = _fmt2(r["spearman_rho"]) if not r.get("skipped") else "--"
        p = f"{r['spearman_p']:.3f}" if not r.get("skipped") else "--"
        cg_rho = _fmt2(cg["spearman_rho"]) if not cg.get("skipped") else "--"
        cg_p = f"{cg['spearman_p']:.3f}" if not cg.get("skipped") else "--"
        r4_rho = _fmt2(r4["spearman_rho"]) if not r4.get("skipped") else "--"
        r4_p = f"{r4['spearman_p']:.3f}" if not r4.get("skipped") else "--"
        lines.append(rf"\newcommand{{\{prefix}HPrimaryN}}{{{row['n_primary']}}}")
        lines.append(rf"\newcommand{{\{prefix}HPrimaryResearchSpearmanRho}}{{{rho}}}")
        lines.append(rf"\newcommand{{\{prefix}HPrimaryResearchSpearmanP}}{{{p}}}")
        lines.append(rf"\newcommand{{\{prefix}HPrimaryCovgapSpearmanRho}}{{{cg_rho}}}")
        lines.append(rf"\newcommand{{\{prefix}HPrimaryCovgapSpearmanP}}{{{cg_p}}}")
        lines.append(rf"\newcommand{{\{prefix}HPrimaryResearchSpearmanRhoNoSdgFour}}{{{r4_rho}}}")
        lines.append(rf"\newcommand{{\{prefix}HPrimaryResearchSpearmanPNoSdgFour}}{{{r4_p}}}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_table_combined(path: Path, semantic_rows: list[dict], coverage_rows: list[dict]) -> None:
    sem_lookup: dict[tuple[str, int], dict] = {(r["family"], r["sdg"]): r for r in semantic_rows}
    cov_lookup: dict[tuple[str, int], dict] = {(r["family"], r["sdg"]): r for r in coverage_rows}
    lines = [
        "% Auto-generated by 1_code/7_main_analysis/2_appendix/a2_policy_source_family_sensitivity.py — do not edit manually",
        r"\begin{tabular}{lcccccccc}",
        r"\toprule",
        r"SDG & \multicolumn{2}{c}{Full Corpus} & \multicolumn{2}{c}{Curated AI/SDG} & \multicolumn{2}{c}{SDGi VNR/VLR} & \multicolumn{2}{c}{UNGDC} \\",
        r"\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7} \cmidrule(lr){8-9}",
        r"& pol.\% (n) & sem.\,(n) & pol.\% (n) & sem.\,(n) & pol.\% (n) & sem.\,(n) & pol.\% (n) & sem.\,(n) \\",
        r"\midrule",
    ]
    for sdg in range(1, N_SDG + 1):
        sdg_label = str(sdg)
        cells = []
        for family in ["full_policy_corpus", "curated_ai_sdg", "sdgi_vnr_vlr", "ungdc_speeches"]:
            sr = sem_lookup.get((family, sdg))
            cr = cov_lookup.get((family, sdg))
            share = f"{float(cr['document_weighted_share']) * 100:.1f}" if cr is not None else "--"
            raw_n = f"{int(cr['document_count_assigned']):,}" if cr is not None else "--"
            gap = f"{float(sr['semantic_gap']):.2f}" if sr is not None and sr["semantic_gap"] is not None else "--"
            capped_n = f"{int(sr['n_policy_segments_capped']):,}" if sr is not None else "--"
            cells.append(f"{share} ({raw_n})")
            cells.append(f"{gap} ({capped_n})")
        lines.append(f"{sdg_label} & " + " & ".join(cells) + r" \\")

    # Spearman ρ row: each sub-family vs Full Corpus
    families_order = ["full_policy_corpus", "curated_ai_sdg", "sdgi_vnr_vlr", "ungdc_speeches"]
    full_share = [float(cov_lookup[("full_policy_corpus", s)]['document_weighted_share']) for s in range(1, N_SDG + 1)]
    full_gap = [float(sem_lookup[("full_policy_corpus", s)]['semantic_gap']) for s in range(1, N_SDG + 1)]
    rho_cells = [r"$\rho$"]
    for fam in families_order:
        if fam == "full_policy_corpus":
            rho_cells.extend(["1.00", "1.00"])
        else:
            fv_share = [float(cov_lookup[(fam, s)]['document_weighted_share']) for s in range(1, N_SDG + 1)]
            fv_gap = [float(sem_lookup[(fam, s)]['semantic_gap']) for s in range(1, N_SDG + 1)]
            rho_cells.append(f"{_spearman(full_share, fv_share):.2f}")
            rho_cells.append(f"{_spearman(full_gap, fv_gap):.2f}")

    lines.append(r"\midrule")
    lines.append(" & ".join(rho_cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    _POLICY_EMB = semantic_gap_shared.get_policy_emb(args.embed_model)
    _POLICY_IDS = semantic_gap_shared.get_policy_ids(args.embed_model)
    _POLICY_SCORES = semantic_gap_shared.get_policy_scores(args.embed_model)
    _RESEARCH_CENTROIDS = semantic_gap_shared.get_research_centroids(args.embed_model)
    _RESEARCH_CENTROID_META = semantic_gap_shared.get_research_centroid_meta(args.embed_model)
    is_adjusted = args.embeddings == "adjusted"

    output_dir = Path(args.output_dir)
    out_root = output_dir / "appendix" / model_slug(args.embed_model) / "a2_source_family_sensitivity"
    if is_adjusted:
        data_dir = out_root / "data" / "adjusted"
        tables_dir = out_root / "tables" / "adjusted"
    else:
        data_dir = out_root / "data"
        tables_dir = out_root / "tables"
    for d in (data_dir, tables_dir):
        d.mkdir(parents=True, exist_ok=True)

    SCRIPT_VERSION = "2"
    PRIMARY = data_dir / SUMMARY_CSV
    OUTPUTS = [PRIMARY, data_dir / COVERAGE_CSV, data_dir / SEMANTIC_CSV,
               data_dir / H25_CSV, data_dir / H25_JSON]
    if not is_adjusted:
        OUTPUTS += [
            tables_dir / "tab_a2_policy_source_family_combined.tex",
            tables_dir / H25_TEX,
            tables_dir / "num_a2_policy_source_family_h25.tex",
        ]
    canonical_coverage = (Path(args.output_dir) / model_slug(args.embed_model)
                          / "data" / CANONICAL_COVERAGE_FILE)
    if not canonical_coverage.exists():
        raise FileNotFoundError(f"Canonical coverage input missing: {canonical_coverage}")
    fp = fingerprint_of(_POLICY_EMB, _POLICY_IDS, _POLICY_SCORES,
                        _RESEARCH_CENTROIDS, _RESEARCH_CENTROID_META,
                        canonical_coverage) + SCRIPT_VERSION
    if is_adjusted:
        g_path = register_utils.register_dir(args.embed_model) / "G.npy"
        fp += f"_adjusted_{register_utils.track_for_model(args.embed_model)}"
        fp += fingerprint_of(g_path)
    if should_skip(OUTPUTS, fp, args.overwrite, PRIMARY):
        log.info("Skipping %s \u2014 inputs unchanged", PRIMARY)
        return

    source_family_map = build_source_family_map(args.embed_model)

    policy_scores = np.load(_POLICY_SCORES)
    policy_emb = np.load(_POLICY_EMB)
    policy_ids = load_json(_POLICY_IDS)
    research_centroids = np.load(_RESEARCH_CENTROIDS).astype(np.float32)
    research_meta = load_json(_RESEARCH_CENTROID_META)

    if is_adjusted:
        G = register_utils.load_G(args.embed_model)
        log.info("Projecting policy embeddings and research centroids through G...")
        policy_emb = register_utils.project(policy_emb, G)
        research_centroids = register_utils.project(research_centroids, G)

    policy_assignments = get_cluster_assignments(policy_scores)

    row_family: list[str] = []
    family_to_indices: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(policy_ids):
        source_doc = row["source_doc"]
        family = source_family_map.get(source_doc)
        if family is None:
            raise RuntimeError(f"Missing source-family assignment for source_doc '{source_doc}'")
        row_family.append(family)
        family_to_indices[family].append(idx)

    family_to_indices["full_policy_corpus"] = list(range(len(policy_ids)))

    coverage_rows: list[dict] = []
    semantic_rows: list[dict] = []
    summary_rows: list[dict] = []

    for family in FAMILY_ORDER:
        subset_indices = family_to_indices[family]
        profile, doc_meta = document_weighted_policy_profile_subset(policy_scores, policy_ids, subset_indices)
        docs = len(doc_meta)
        segments = len(subset_indices)
        top3 = top3_sdgs(profile)

        assigned_counts = np.bincount(
            np.asarray([meta["sdg_assignment"] - 1 for meta in doc_meta.values()], dtype=np.int64),
            minlength=N_SDG,
        )
        for sdg_idx in range(N_SDG):
            coverage_rows.append(
                {
                    "family": family,
                    "family_label": FAMILY_LABELS[family],
                    "sdg": sdg_idx + 1,
                    "document_weighted_share": round(float(profile[sdg_idx]), 6),
                    "document_count_assigned": int(assigned_counts[sdg_idx]),
                    "n_documents_total": docs,
                    "n_segments_total": segments,
                }
            )

        family_semantic = compute_family_semantic_rows(
            family,
            subset_indices,
            policy_emb,
            policy_assignments,
            policy_ids,
            research_centroids,
            research_meta,
        )
        semantic_rows.extend(family_semantic)
        valid_gaps = [
            float(row["semantic_gap"])
            for row in family_semantic
            if row["semantic_gap"] is not None and not row["unreliable"]
        ]
        ranked_gaps = sorted(
            [
                (int(row["sdg"]), float(row["semantic_gap"]))
                for row in family_semantic
                if row["semantic_gap"] is not None and not row["unreliable"]
            ],
            key=lambda item: item[1],
            reverse=True,
        )[:3]
        summary_rows.append(
            {
                "family": family,
                "family_label": FAMILY_LABELS[family],
                "docs": docs,
                "segments": segments,
                "top3_sdgs_docweighted": top3,
                "mean_semantic_gap": None if not valid_gaps else round(float(np.mean(valid_gaps)), 6),
                "n_valid_semantic_gap_sdgs": len(valid_gaps),
                "top3_semantic_gaps": " / ".join(f"SDG {sdg}" for sdg, _ in ranked_gaps) if ranked_gaps else "N/A",
                "interpretation_note": INTERPRETATION_NOTES[family],
            }
        )

    write_csv(
        data_dir / SUMMARY_CSV,
        [
            "family",
            "family_label",
            "docs",
            "segments",
            "top3_sdgs_docweighted",
            "mean_semantic_gap",
            "n_valid_semantic_gap_sdgs",
            "top3_semantic_gaps",
            "interpretation_note",
        ],
        summary_rows,
    )
    write_csv(
        data_dir / COVERAGE_CSV,
        [
            "family",
            "family_label",
            "sdg",
            "document_weighted_share",
            "document_count_assigned",
            "n_documents_total",
            "n_segments_total",
        ],
        coverage_rows,
    )
    write_csv(
        data_dir / SEMANTIC_CSV,
        [
            "family",
            "sdg",
            "n_research_papers",
            "n_policy_segments",
            "n_policy_segments_capped",
            "n_policy_docs",
            "n_policy_docs_capped",
            "segment_cap",
            "semantic_similarity",
            "semantic_gap",
            "policy_cohesion",
            "unreliable",
        ],
        semantic_rows,
    )
    if not is_adjusted:
        write_table_combined(tables_dir / "tab_a2_policy_source_family_combined.tex", semantic_rows, coverage_rows)

    # ---- Per-family H25 replication (Item 1 symmetry control) ----
    cov_data = load_json(canonical_coverage)
    research_share = np.array(
        [cov_data["research_profile_hard"][f"SDG{i}"] for i in range(1, N_SDG + 1)],
        dtype=float,
    )
    cov_lookup = {(r["family"], r["sdg"]): r for r in coverage_rows}
    sem_lookup = {(r["family"], r["sdg"]): r for r in semantic_rows}
    h25_rows: list[dict] = []
    for family in FAMILY_ORDER:
        policy_share = np.array(
            [float(cov_lookup[(family, s)]["document_weighted_share"]) for s in range(1, N_SDG + 1)],
            dtype=float,
        )
        sem_gap = np.array(
            [
                np.nan if sem_lookup[(family, s)]["semantic_gap"] is None
                else float(sem_lookup[(family, s)]["semantic_gap"])
                for s in range(1, N_SDG + 1)
            ],
            dtype=float,
        )
        unreliable = np.array(
            [bool(sem_lookup[(family, s)]["unreliable"]) for s in range(1, N_SDG + 1)],
            dtype=bool,
        )
        result = compute_family_h25(research_share, policy_share, sem_gap, unreliable)
        row = {
            "family": family,
            "family_label": FAMILY_LABELS[family],
            "n_primary": result["n_primary"],
            "primary_fisher_ci_lower": result["primary_fisher_ci_lower"],
            "primary_fisher_ci_upper": result["primary_fisher_ci_upper"],
            "research": result["tests_primary"]["research"],
            "policy": result["tests_primary"]["policy"],
            "covgap": result["tests_primary"]["covgap"],
            "dominance": result["tests_primary"]["dominance"],
            "research_excl4": result["tests_excl4"]["research"],
        }
        h25_rows.append(row)
        log.info(
            "Family %-20s n=%d  research rho=%.3f (p=%.3f)  covgap rho=%.3f (p=%.3f)  excl4 rho=%.3f (p=%.3f)",
            family,
            row["n_primary"],
            row["research"].get("spearman_rho", float("nan")),
            row["research"].get("spearman_p", float("nan")),
            row["covgap"].get("spearman_rho", float("nan")),
            row["covgap"].get("spearman_p", float("nan")),
            row["research_excl4"].get("spearman_rho", float("nan")),
            row["research_excl4"].get("spearman_p", float("nan")),
        )

    h25_flat: list[dict] = []
    for row in h25_rows:
        for pred in ("research", "policy", "covgap", "dominance"):
            cell = row[pred]
            base = {"family": row["family"], "family_label": row["family_label"],
                    "n_primary": row["n_primary"], "predictor": pred}
            if cell.get("skipped"):
                base.update({"n": cell.get("n", 0), "skipped": True,
                             "pearson_r": "", "pearson_p": "", "spearman_rho": "", "spearman_p": ""})
            else:
                base.update({"n": cell["n"], "skipped": False,
                             "pearson_r": cell["pearson_r"], "pearson_p": cell["pearson_p"],
                             "spearman_rho": cell["spearman_rho"], "spearman_p": cell["spearman_p"]})
            h25_flat.append(base)

    write_csv(
        data_dir / H25_CSV,
        ["family", "family_label", "n_primary", "predictor", "n", "skipped",
         "pearson_r", "pearson_p", "spearman_rho", "spearman_p"],
        h25_flat,
    )
    h25_summary = [
        {
            "family": row["family"],
            "family_label": row["family_label"],
            "n_primary": row["n_primary"],
            "primary_fisher_ci_lower": row["primary_fisher_ci_lower"],
            "primary_fisher_ci_upper": row["primary_fisher_ci_upper"],
            "research_spearman_rho": row["research"]["spearman_rho"] if not row["research"].get("skipped") else None,
            "research_spearman_p": row["research"]["spearman_p"] if not row["research"].get("skipped") else None,
            "covgap_spearman_rho": row["covgap"]["spearman_rho"] if not row["covgap"].get("skipped") else None,
            "covgap_spearman_p": row["covgap"]["spearman_p"] if not row["covgap"].get("skipped") else None,
            "research_excl4_spearman_rho": row["research_excl4"]["spearman_rho"] if not row["research_excl4"].get("skipped") else None,
            "research_excl4_spearman_p": row["research_excl4"]["spearman_p"] if not row["research_excl4"].get("skipped") else None,
        }
        for row in h25_rows
    ]
    (data_dir / H25_JSON).write_text(json.dumps(h25_summary, indent=2) + "\n", encoding="utf-8")
    if not is_adjusted:
        write_table_h25(tables_dir / H25_TEX, h25_rows)
        write_h25_macros(tables_dir / "num_a2_policy_source_family_h25.tex", h25_rows)

    log.info("Saved: %s", data_dir / H25_CSV)
    log.info("Saved: %s", data_dir / H25_JSON)
    if not is_adjusted:
        log.info("Saved: %s", tables_dir / H25_TEX)
        log.info("Saved: %s", tables_dir / "num_a2_policy_source_family_h25.tex")

    log.info("Saved: %s", data_dir / SUMMARY_CSV)
    log.info("Saved: %s", data_dir / COVERAGE_CSV)
    log.info("Saved: %s", data_dir / SEMANTIC_CSV)
    if not is_adjusted:
        log.info("Saved: %s", tables_dir / "tab_a2_policy_source_family_combined.tex")
    record_fingerprint(OUTPUTS, fp, PRIMARY)


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()

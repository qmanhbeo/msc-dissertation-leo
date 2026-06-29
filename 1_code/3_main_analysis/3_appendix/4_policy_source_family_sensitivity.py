"""
Policy source-family sensitivity diagnostic.

This appendix-stage diagnostic checks whether the policy-side coverage and semantic-gap
patterns are broadly stable across the three existing policy source families:

  1. Curated AI/SDG policy documents (policy_scrape + policy_manual)
  2. SDGi VNR/VLR reports
  3. UN General Debate Corpus (UNGDC)

It does not alter the canonical hard-label results. It only reports how policy-side
profiles and within-SDG semantic gaps look when the policy corpus is split by source family.

Run from project root:
    python 1_code/3_main_analysis/3_appendix/4_policy_source_family_sensitivity.py
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = ROOT / "1_code"
ANALYSIS_ROOT = Path(__file__).resolve().parents[1]
SHARED_DIR = ANALYSIS_ROOT / "0_shared"
for path in (CODE_ROOT, SHARED_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from shared_utils import ensure_canonical_outputs
from semantic_gap_shared import (
    CHUNK_CAP_PRIMARY,
    MIN_CLUSTER_SIZE,
    N_SDG,
    POLICY_EMB,
    POLICY_IDS,
    POLICY_SCORES,
    RANDOM_SEED,
    RESEARCH_CENTROID_META,
    RESEARCH_CENTROIDS,
    build_sub_centroid,
    cap_policy_indices_per_doc,
    get_cluster_assignments,
    load_json,
)


DEFAULT_OUTPUT_ROOT = Path("outputs")
POLICY_PREPROCESSED_ROOT = Path("2_data/1_preprocessed/policy_all")

SUMMARY_CSV = "policy_source_family_summary.csv"
COVERAGE_CSV = "policy_source_family_coverage.csv"
SEMANTIC_CSV = "policy_source_family_semantic_gaps.csv"
TABLE_TEX = "tab_policy_source_family_sensitivity.tex"

FAMILY_FILE_MAP = {
    "curated_ai_sdg": [
        POLICY_PREPROCESSED_ROOT / "policy_scrape" / "policy_scrape_chunks.jsonl",
        POLICY_PREPROCESSED_ROOT / "policy_manual" / "policy_manual_chunks.jsonl",
    ],
    "sdgi_vnr_vlr": [
        POLICY_PREPROCESSED_ROOT / "sdgi_corpus" / "sdgi_chunks.jsonl",
    ],
    "ungdc_speeches": [
        POLICY_PREPROCESSED_ROOT / "ungdc_sdg" / "ungdc_sdg_chunks.jsonl",
    ],
}

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
    return p.parse_args()


def iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def build_source_family_map() -> dict[str, str]:
    source_family: dict[str, str] = {}
    for family, paths in FAMILY_FILE_MAP.items():
        for path in paths:
            for row in iter_jsonl(path):
                source_doc = str(row["source_doc"])
                existing = source_family.get(source_doc)
                if existing is not None and existing != family:
                    raise RuntimeError(
                        f"source_doc '{source_doc}' appears in multiple families: {existing} vs {family}"
                    )
                source_family[source_doc] = family
    if not source_family:
        raise RuntimeError("No source-family assignments were built from policy preprocessed files.")
    return source_family


def document_weighted_policy_profile_subset(
    policy_scores: np.ndarray,
    policy_ids: list[dict],
    subset_indices: list[int],
) -> tuple[np.ndarray, dict[str, dict]]:
    doc_to_rows: dict[str, list[int]] = defaultdict(list)
    for idx in subset_indices:
        doc_to_rows[policy_ids[idx]["source_doc"]].append(idx)

    n_docs = len(doc_to_rows)
    if n_docs == 0:
        raise RuntimeError("Subset has no policy documents.")

    doc_vectors = np.zeros((n_docs, N_SDG), dtype=np.float32)
    doc_meta: dict[str, dict] = {}
    for d_idx, (source_doc, row_idxs) in enumerate(doc_to_rows.items()):
        doc_vec = policy_scores[row_idxs].mean(axis=0)
        doc_vectors[d_idx] = doc_vec
        doc_meta[source_doc] = {
            "n_chunks": len(row_idxs),
            "sdg_assignment": int(doc_vec.argmax()) + 1,
        }

    doc_assignments = doc_vectors.argmax(axis=1)
    counts = np.bincount(doc_assignments, minlength=N_SDG).astype(np.float64)
    hard_profile = counts / counts.sum()
    return hard_profile, doc_meta


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
        n_chunks = len(family_sdg_idxs)
        capped = cap_policy_indices_per_doc(
            family_sdg_idxs,
            policy_ids,
            CHUNK_CAP_PRIMARY,
            np.random.default_rng(RANDOM_SEED + sdg_idx),
        )
        n_chunks_capped = len(capped)
        raw_docs = {policy_ids[i]["source_doc"] for i in family_sdg_idxs}
        capped_docs = {policy_ids[i]["source_doc"] for i in capped}
        n_papers = int(research_meta[sdg_idx]["n_papers_assigned"])
        unreliable = n_papers < MIN_CLUSTER_SIZE or n_chunks_capped < MIN_CLUSTER_SIZE

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
                "n_policy_chunks": n_chunks,
                "n_policy_chunks_capped": n_chunks_capped,
                "n_policy_docs": len(raw_docs),
                "n_policy_docs_capped": len(capped_docs),
                "chunk_cap": CHUNK_CAP_PRIMARY,
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


def latex_escape(text: str) -> str:
    return (
        text.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
    )


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_table(path: Path, summary_rows: list[dict]) -> None:
    lines = [
        "% Auto-generated by 1_code/3_main_analysis/3_appendix/4_policy_source_family_sensitivity.py — do not edit manually",
        r"\begin{tabular}{lrrp{2.8cm}c>{\centering\arraybackslash}p{1.6cm}p{3.2cm}}",
        r"\toprule",
        r"Policy source family & Docs & Chunks & \shortstack[l]{Top 3 SDGs\\(doc-weighted)} & \shortstack[l]{Mean semantic\\gap vs research} & \shortstack[c]{Largest\\gap SDG} & Interpretation note \\",
        r"\midrule",
    ]
    for row in summary_rows:
        mean_gap = "N/A" if row["mean_semantic_gap"] is None else f"{float(row['mean_semantic_gap']):.3f}"
        top_gaps = str(row.get("top3_semantic_gaps", ""))
        largest_sdg = top_gaps.split(" / ")[0] if top_gaps and top_gaps != "N/A" else "N/A"
        lines.append(
            f"{latex_escape(str(row['family_label']))} & "
            f"{int(row['docs'])} & "
            f"{int(row['chunks'])} & "
            f"{latex_escape(str(row['top3_sdgs_docweighted']))} & "
            f"{mean_gap} & "
            f"{largest_sdg} & "
            f"{latex_escape(str(row['interpretation_note']))} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    layout = ensure_canonical_outputs(output_dir)
    out_dir = output_dir / "appendix" / "source_family_sensitivity"
    out_dir.mkdir(parents=True, exist_ok=True)

    source_family_map = build_source_family_map()

    policy_scores = np.load(POLICY_SCORES)
    policy_emb = np.load(POLICY_EMB)
    policy_ids = load_json(POLICY_IDS)
    research_centroids = np.load(RESEARCH_CENTROIDS).astype(np.float32)
    research_meta = load_json(RESEARCH_CENTROID_META)
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
        chunks = len(subset_indices)
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
                    "n_chunks_total": chunks,
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
                "chunks": chunks,
                "top3_sdgs_docweighted": top3,
                "mean_semantic_gap": None if not valid_gaps else round(float(np.mean(valid_gaps)), 6),
                "n_valid_semantic_gap_sdgs": len(valid_gaps),
                "top3_semantic_gaps": " / ".join(f"SDG {sdg}" for sdg, _ in ranked_gaps) if ranked_gaps else "N/A",
                "interpretation_note": INTERPRETATION_NOTES[family],
            }
        )

    write_csv(
        out_dir / SUMMARY_CSV,
        [
            "family",
            "family_label",
            "docs",
            "chunks",
            "top3_sdgs_docweighted",
            "mean_semantic_gap",
            "n_valid_semantic_gap_sdgs",
            "top3_semantic_gaps",
            "interpretation_note",
        ],
        summary_rows,
    )
    write_csv(
        out_dir / COVERAGE_CSV,
        [
            "family",
            "family_label",
            "sdg",
            "document_weighted_share",
            "document_count_assigned",
            "n_documents_total",
            "n_chunks_total",
        ],
        coverage_rows,
    )
    write_csv(
        out_dir / SEMANTIC_CSV,
        [
            "family",
            "sdg",
            "n_research_papers",
            "n_policy_chunks",
            "n_policy_chunks_capped",
            "n_policy_docs",
            "n_policy_docs_capped",
            "chunk_cap",
            "semantic_similarity",
            "semantic_gap",
            "policy_cohesion",
            "unreliable",
        ],
        semantic_rows,
    )
    write_table(layout.tables_dir / TABLE_TEX, summary_rows)

    log.info("Saved: %s", out_dir / SUMMARY_CSV)
    log.info("Saved: %s", out_dir / COVERAGE_CSV)
    log.info("Saved: %s", out_dir / SEMANTIC_CSV)
    log.info("Saved: %s", layout.tables_dir / TABLE_TEX)


if __name__ == "__main__":
    main()

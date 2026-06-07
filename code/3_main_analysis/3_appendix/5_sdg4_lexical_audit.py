"""
SDG 4 lexical artefact audit.

This appendix-stage audit checks whether research records assigned to SDG 4 are dominated
by machine-learning vocabulary rather than education-specific vocabulary. It does not
reassign SDGs or alter the canonical hard-label analysis.

Run from project root:
    python code/3_main_analysis/3_appendix/5_sdg4_lexical_audit.py
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from shared_utils import ensure_canonical_outputs


DEFAULT_OUTPUT_ROOT = Path("outputs")
RESEARCH_TEXT_MANIFEST = Path("data/1_preprocessed/research_corpus/metadata/manifest.json")
RESEARCH_SCORE_MANIFEST = Path("data/3_scored/paper_scores_shards/metadata/manifest.json")

AUDIT_CSV = "sdg4_lexical_audit.csv"
AUDIT_JSON = "sdg4_lexical_audit_summary.json"
TABLE_TEX = "tab_sdg4_lexical_audit.tex"

ML_TERMS = [
    "machine learning",
    "deep learning",
    "reinforcement learning",
    "supervised learning",
    "unsupervised learning",
    "neural network",
    "model",
    "models",
    "training",
    "trained",
    "train",
    "algorithm",
    "classification",
    "prediction",
]

EDU_TERMS = [
    "education",
    "educational",
    "school",
    "schools",
    "student",
    "students",
    "teacher",
    "teachers",
    "classroom",
    "curriculum",
    "learning outcome",
    "pedagogy",
    "university",
    "universities",
    "course",
    "courses",
    "teaching",
]

SUBSET_LABELS = {
    "sdg4_assigned": "SDG 4-assigned research",
    "non_sdg4_sample": "Non-SDG4 research sample",
    "sdg9_assigned": "SDG 9-assigned research",
}

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run SDG 4 lexical artefact audit.")
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_ROOT))
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_json(path: Path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def compile_patterns(terms: list[str]) -> list[re.Pattern[str]]:
    return [re.compile(r"\b" + re.escape(term.lower()) + r"\b") for term in terms]


ML_PATTERNS = compile_patterns(ML_TERMS)
EDU_PATTERNS = compile_patterns(EDU_TERMS)


def classify_text(text: str) -> str:
    lower = text.lower()
    has_ml = any(pattern.search(lower) for pattern in ML_PATTERNS)
    has_edu = any(pattern.search(lower) for pattern in EDU_PATTERNS)
    if has_ml and not has_edu:
        return "ml_only"
    if has_edu and not has_ml:
        return "education_only"
    if has_ml and has_edu:
        return "both"
    return "neither_unclear"


def reservoir_sample_non_sdg4(target_n: int, seed: int, score_manifest: dict) -> dict[int, set[int]]:
    rng = random.Random(seed)
    reservoir: list[tuple[int, int]] = []
    seen = 0
    n_shards = len(score_manifest["shards"])
    for shard_idx, shard in enumerate(score_manifest["shards"], start=1):
        shard_id = int(shard["shard_id"])
        ids_path = ROOT / shard["ids_path"]
        for row in iter_jsonl(ids_path):
            assigned_sdg = int(row["assigned_sdg"])
            if assigned_sdg == 4:
                continue
            ref = (shard_id, int(row["row_in_shard"]))
            seen += 1
            if len(reservoir) < target_n:
                reservoir.append(ref)
            else:
                j = rng.randrange(seen)
                if j < target_n:
                    reservoir[j] = ref
        log.info(
            "Reservoir sampled non-SDG4 candidates from score shard %s/%s (%s seen so far)",
            shard_idx,
            n_shards,
            seen,
        )
    out: dict[int, set[int]] = defaultdict(set)
    for shard_id, row_idx in reservoir:
        out[shard_id].add(row_idx)
    return out


def scan_assignment_refs(score_manifest: dict) -> tuple[dict[int, set[int]], dict[int, set[int]], int]:
    sdg4_refs: dict[int, set[int]] = defaultdict(set)
    sdg9_refs: dict[int, set[int]] = defaultdict(set)
    n_sdg4 = 0
    n_shards = len(score_manifest["shards"])
    for shard_idx, shard in enumerate(score_manifest["shards"], start=1):
        shard_id = int(shard["shard_id"])
        ids_path = ROOT / shard["ids_path"]
        for row in iter_jsonl(ids_path):
            assigned_sdg = int(row["assigned_sdg"])
            row_in_shard = int(row["row_in_shard"])
            if assigned_sdg == 4:
                sdg4_refs[shard_id].add(row_in_shard)
                n_sdg4 += 1
            elif assigned_sdg == 9:
                sdg9_refs[shard_id].add(row_in_shard)
        log.info(
            "Scanned score shard %s/%s for SDG 4 and SDG 9 assignments (%s SDG4 rows found so far)",
            shard_idx,
            n_shards,
            n_sdg4,
        )
    if n_sdg4 == 0:
        raise RuntimeError("No SDG 4-assigned research records were found.")
    return sdg4_refs, sdg9_refs, n_sdg4


def audit_subsets(
    text_manifest: dict,
    subset_refs: dict[str, dict[int, set[int]]],
) -> dict[str, Counter]:
    counters = {subset: Counter() for subset in subset_refs}
    subset_sizes = {subset: 0 for subset in subset_refs}

    n_shards = len(text_manifest["shards"])
    for shard_idx, shard in enumerate(text_manifest["shards"], start=1):
        shard_id = int(shard["shard_id"])
        data_path = ROOT / shard["data_path"]
        targets = {
            subset: refs.get(shard_id, set())
            for subset, refs in subset_refs.items()
            if refs.get(shard_id)
        }
        if not targets:
            continue
        with data_path.open(encoding="utf-8") as f:
            for row_idx, line in enumerate(f):
                matching = [subset for subset, rows in targets.items() if row_idx in rows]
                if not matching:
                    continue
                payload = json.loads(line)
                text = str(payload.get("combined_text") or "")
                category = classify_text(text)
                for subset in matching:
                    counters[subset][category] += 1
                    subset_sizes[subset] += 1
        log.info(
            "Scanned text shard %s/%s for lexical audit (%s SDG4, %s non-SDG4 sample, %s SDG9 rows matched so far)",
            shard_idx,
            n_shards,
            subset_sizes["sdg4_assigned"],
            subset_sizes["non_sdg4_sample"],
            subset_sizes["sdg9_assigned"],
        )

    for subset, refs in subset_refs.items():
        expected = sum(len(rows) for rows in refs.values())
        observed = subset_sizes[subset]
        if observed != expected:
            raise RuntimeError(f"Subset '{subset}' row mismatch: expected {expected}, observed {observed}")
    return counters


def pct(n: int, d: int) -> float:
    if d == 0:
        return 0.0
    return 100.0 * float(n) / float(d)


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


def write_table(path: Path, rows: list[dict]) -> None:
    lines = [
        "% Auto-generated by code/3_main_analysis/3_appendix/5_sdg4_lexical_audit.py — do not edit manually",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Corpus subset & N & ML-only \% & Education-only \% & Both \% & \shortstack[l]{Neither /\\unclear \%} \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{latex_escape(str(row['subset_label']))} & "
            f"{int(row['n'])} & "
            f"{float(row['ml_only_pct']):.1f} & "
            f"{float(row['education_only_pct']):.1f} & "
            f"{float(row['both_pct']):.1f} & "
            f"{float(row['neither_unclear_pct']):.1f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    layout = ensure_canonical_outputs(Path(args.output_dir))
    out_dir = layout.root / "sdg4_audit"
    out_dir.mkdir(parents=True, exist_ok=True)

    score_manifest = load_json(RESEARCH_SCORE_MANIFEST)
    text_manifest = load_json(RESEARCH_TEXT_MANIFEST)
    log.info("Loaded research score manifest with %s shards", len(score_manifest["shards"]))
    log.info("Loaded research text manifest with %s shards", len(text_manifest["shards"]))

    sdg4_refs, sdg9_refs, n_sdg4 = scan_assignment_refs(score_manifest)
    log.info("Found %s SDG 4-assigned research records", n_sdg4)
    non_sdg4_refs = reservoir_sample_non_sdg4(n_sdg4, args.seed, score_manifest)
    log.info("Built matched non-SDG4 reservoir sample with %s records using seed %s", n_sdg4, args.seed)

    subset_refs = {
        "sdg4_assigned": sdg4_refs,
        "non_sdg4_sample": non_sdg4_refs,
        "sdg9_assigned": sdg9_refs,
    }
    counters = audit_subsets(text_manifest, subset_refs)

    rows: list[dict] = []
    summary = {
        "generated_from": "code/3_main_analysis/3_appendix/5_sdg4_lexical_audit.py",
        "random_seed": args.seed,
        "ml_terms": ML_TERMS,
        "education_terms": EDU_TERMS,
        "subsets": {},
        "note": (
            "This audit does not relabel research records. It only tests whether the SDG 4 lexical artefact concern is empirically plausible."
        ),
    }
    for subset_key in ["sdg4_assigned", "non_sdg4_sample", "sdg9_assigned"]:
        counts = counters[subset_key]
        n = int(sum(counts.values()))
        row = {
            "subset": subset_key,
            "subset_label": SUBSET_LABELS[subset_key],
            "n": n,
            "ml_only_n": int(counts["ml_only"]),
            "education_only_n": int(counts["education_only"]),
            "both_n": int(counts["both"]),
            "neither_unclear_n": int(counts["neither_unclear"]),
            "ml_only_pct": round(pct(counts["ml_only"], n), 3),
            "education_only_pct": round(pct(counts["education_only"], n), 3),
            "both_pct": round(pct(counts["both"], n), 3),
            "neither_unclear_pct": round(pct(counts["neither_unclear"], n), 3),
        }
        rows.append(row)
        summary["subsets"][subset_key] = row

    write_csv(
        out_dir / AUDIT_CSV,
        [
            "subset",
            "subset_label",
            "n",
            "ml_only_n",
            "education_only_n",
            "both_n",
            "neither_unclear_n",
            "ml_only_pct",
            "education_only_pct",
            "both_pct",
            "neither_unclear_pct",
        ],
        rows,
    )
    (out_dir / AUDIT_JSON).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_table(layout.tables_dir / TABLE_TEX, rows)

    log.info("Saved: %s", out_dir / AUDIT_CSV)
    log.info("Saved: %s", out_dir / AUDIT_JSON)
    log.info("Saved: %s", layout.tables_dir / TABLE_TEX)


if __name__ == "__main__":
    main()

"""
SDG 4 lexical artefact audit.

This appendix-stage audit checks whether research records assigned to SDG 4 are dominated
by machine-learning vocabulary rather than education-specific vocabulary. It does not
reassign SDGs or alter the canonical hard-label analysis.

Run from project root:
    python 1_code/7_main_analysis/2_appendix/a3_sdg4_lexical_audit.py
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import re
import sys
from multiprocessing import Pool
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = ROOT / "1_code"
ANALYSIS_ROOT = Path(__file__).resolve().parents[1]
SHARED_DIR = ANALYSIS_ROOT / "0_shared"
for path in (CODE_ROOT, SHARED_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))



from model_utils import DEFAULT_EMBED_MODEL, DEFAULT_OUTPUT_ROOT, RANDOM_SEED, embed_research_dir_for_model, model_slug, scored_dir_for_model, open_text, resolve_research_text_path, resolve_model_alias
from shared_utils import fingerprint_of, should_skip, record_fingerprint
from semantic_gap_shared import latex_escape, write_csv
from shard_pipeline_utils import iter_jsonl, load_json

AUDIT_CSV = "sdg4_lexical_audit.csv"
AUDIT_JSON = "sdg4_lexical_audit_summary.json"
TABLE_TEX = "tab_a3_sdg4_lexical_audit.tex"

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
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    p.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL, type=resolve_model_alias, help=argparse.SUPPRESS)
    p.add_argument("--overwrite", action="store_true", help=argparse.SUPPRESS)
    return p.parse_args()


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


def scan_and_sample_score_shards(score_manifest: dict, seed: int) -> tuple[dict[int, set[int]], dict[int, set[int]], dict[int, set[int]]]:
    rng = random.Random(seed)
    sdg4_refs: dict[int, set[int]] = defaultdict(set)
    sdg9_refs: dict[int, set[int]] = defaultdict(set)
    non_sdg4_candidates: list[tuple[int, int]] = []
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
            else:
                if assigned_sdg == 9:
                    sdg9_refs[shard_id].add(row_in_shard)
                non_sdg4_candidates.append((shard_id, row_in_shard))
        log.info(
            "Scanned score shard %s/%s (%s SDG4, %s non-SDG4 candidates so far)",
            shard_idx,
            n_shards,
            n_sdg4,
            len(non_sdg4_candidates),
        )
    if n_sdg4 == 0:
        raise RuntimeError("No SDG 4-assigned research records were found.")

    # Reservoir-sample non-SDG4 in memory (identical algorithm to original two-pass)
    reservoir: list[tuple[int, int]] = []
    for i, ref in enumerate(non_sdg4_candidates):
        seen = i + 1
        if len(reservoir) < n_sdg4:
            reservoir.append(ref)
        else:
            j = rng.randrange(seen)
            if j < n_sdg4:
                reservoir[j] = ref

    non_sdg4_refs: dict[int, set[int]] = defaultdict(set)
    for shard_id, row_idx in reservoir:
        non_sdg4_refs[shard_id].add(row_idx)
    return sdg4_refs, sdg9_refs, non_sdg4_refs


def _audit_single_shard(args: tuple[str, dict[str, set[int]]]) -> dict[str, Counter]:
    data_path_str, targets_per_subset = args
    """Process a single text shard. Returns per-subset category counters."""
    counters = {subset: Counter() for subset in targets_per_subset}
    data_path = Path(data_path_str)
    with open_text(data_path) as f:
        for row_idx, line in enumerate(f):
            matching = [subset for subset, rows in targets_per_subset.items() if row_idx in rows]
            if not matching:
                continue
            payload = json.loads(line)
            text = str(payload.get("text") or "")
            category = classify_text(text)
            for subset in matching:
                counters[subset][category] += 1
    return counters


def audit_subsets(
    research_dir: Path,
    text_manifest: dict,
    subset_refs: dict[str, dict[int, set[int]]],
    model: str,
) -> dict[str, Counter]:
    n_shards = len(text_manifest["shards"])

    jobs: list[tuple[str, dict[str, set[int]]]] = []
    for shard in text_manifest["shards"]:
        shard_id = int(shard["shard_id"])
        data_path = str(resolve_research_text_path(model, shard["name"]))
        targets = {
            subset: refs.get(shard_id, set())
            for subset, refs in subset_refs.items()
            if refs.get(shard_id)
        }
        if targets:
            jobs.append((data_path, targets))

    if not jobs:
        raise RuntimeError("No shards matched any subset")

    counters = {subset: Counter() for subset in subset_refs}
    completed = 0
    with Pool() as pool:
        for partial in pool.imap_unordered(_audit_single_shard, jobs):
            completed += 1
            for subset, c in partial.items():
                counters[subset] += c
            if "sdg4_assigned" in counters and "non_sdg4_sample" in counters and "sdg9_assigned" in counters:
                log.info(
                    "Processed shard %s/%s for lexical audit (%s SDG4, %s non-SDG4 sample, %s SDG9 rows matched so far)",
                    completed,
                    len(jobs),
                    sum(counters["sdg4_assigned"].values()),
                    sum(counters["non_sdg4_sample"].values()),
                    sum(counters["sdg9_assigned"].values()),
                )

    for subset, refs in subset_refs.items():
        expected = sum(len(rows) for rows in refs.values())
        observed = sum(counters[subset].values())
        if observed != expected:
            raise RuntimeError(f"Subset '{subset}' row mismatch: expected {expected}, observed {observed}")
    return counters


def pct(n: int, d: int) -> float:
    if d == 0:
        return 0.0
    return 100.0 * float(n) / float(d)


def write_table(path: Path, rows: list[dict]) -> None:
    lines = [
        "% Auto-generated by 1_code/7_main_analysis/2_appendix/a3_sdg4_lexical_audit.py — do not edit manually",
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


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    out_root = output_dir / "appendix" / model_slug(args.embed_model) / "a3_sdg4_audit"
    data_dir = out_root / "data"
    tables_dir = out_root / "tables"
    for d in (data_dir, tables_dir):
        d.mkdir(parents=True, exist_ok=True)

    SCRIPT_VERSION = "1"
    PRIMARY = data_dir / AUDIT_JSON
    OUTPUTS = [PRIMARY, data_dir / AUDIT_CSV, tables_dir / TABLE_TEX]
    fp = fingerprint_of(
        scored_dir_for_model(args.embed_model) / "paper_scores_shards" / "metadata" / "manifest.json",
        embed_research_dir_for_model(args.embed_model) / "metadata" / "manifest.json",
    ) + SCRIPT_VERSION
    if should_skip(OUTPUTS, fp, args.overwrite, PRIMARY):
        log.info("Skipping %s \u2014 inputs unchanged", PRIMARY)
        return

    scored_dir = scored_dir_for_model(args.embed_model)
    research_dir = embed_research_dir_for_model(args.embed_model)
    score_manifest = load_json(scored_dir / "paper_scores_shards" / "metadata" / "manifest.json")
    text_manifest = load_json(research_dir / "metadata" / "manifest.json")
    log.info("Loaded research score manifest with %s shards", len(score_manifest["shards"]))
    log.info("Loaded research text manifest with %s shards", len(text_manifest["shards"]))

    sdg4_refs, sdg9_refs, non_sdg4_refs = scan_and_sample_score_shards(score_manifest, args.seed)
    all_sdg4 = sum(len(v) for v in sdg4_refs.values())
    all_sample = sum(len(v) for v in non_sdg4_refs.values())
    log.info("Found %s SDG 4-assigned research records", all_sdg4)
    log.info("Built matched non-SDG4 reservoir sample with %s records using seed %s", all_sample, args.seed)

    subset_refs = {
        "sdg4_assigned": sdg4_refs,
        "non_sdg4_sample": non_sdg4_refs,
        "sdg9_assigned": sdg9_refs,
    }
    counters = audit_subsets(research_dir, text_manifest, subset_refs, args.embed_model)

    rows: list[dict] = []
    summary = {
        "generated_from": "1_code/7_main_analysis/2_appendix/a3_sdg4_lexical_audit.py",
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
        data_dir / AUDIT_CSV,
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
    (data_dir / AUDIT_JSON).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_table(tables_dir / TABLE_TEX, rows)

    log.info("Saved: %s", data_dir / AUDIT_CSV)
    log.info("Saved: %s", data_dir / AUDIT_JSON)
    log.info("Saved: %s", tables_dir / TABLE_TEX)
    record_fingerprint(OUTPUTS, fp, PRIMARY)


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()

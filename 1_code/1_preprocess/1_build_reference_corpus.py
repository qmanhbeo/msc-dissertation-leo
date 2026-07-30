"""
Build consolidated reference corpus from individual SDG-labeled sources.

Reads preprocessed JSONL from each reference source (osdg, benchmark,
sdg_knowledge_hub, aurora, sdgi), normalises fields to a common schema,
deduplicates by exact text (first occurrence wins), and writes
reference.json.

Output: preprocessed_dir() / "reference.json"  (JSON list of records)

Run from project root:
    python 1_code/1_preprocess/1_build_reference_corpus.py
"""

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
ANALYSIS_DIR = CODE_ROOT / "7_main_analysis" / "0_shared"
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from model_utils import preprocessed_dir

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

SOURCES = [
    {"name": "osdg", "path": lambda: preprocessed_dir() / "osdg" / "osdg_clean.jsonl",
     "id_field": "text_id"},
    {"name": "benchmark", "path": lambda: preprocessed_dir() / "sdg_benchmark" / "benchmark_clean.jsonl",
     "id_field": "id"},
    {"name": "sdg_knowledge_hub", "path": lambda: preprocessed_dir() / "sdg_knowledge_hub" / "sdg_knowledge_hub_clean.jsonl",
     "id_field": "id"},
    {"name": "aurora", "path": lambda: preprocessed_dir() / "aurora" / "aurora_texts.jsonl",
     "id_field": "doi"},
    {"name": "sdgi", "path": lambda: preprocessed_dir() / "sdgi" / "sdgi_clean.jsonl",
     "id_field": "id"},
]


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def normalise_record(rec: dict, source_cfg: dict) -> dict:
    """Map source-specific fields to the common reference schema."""
    source_name = source_cfg["name"]
    id_val = rec.get(source_cfg["id_field"], "")
    sdgs = rec.get("sdgs")
    if sdgs is None:
        sdg = rec.get("sdg")
        sdgs = [sdg] if sdg is not None else None
    if isinstance(sdgs, int):
        sdgs = [sdgs]

    return {
        "id": f"{source_name}_{id_val}",
        "text": rec.get("text", ""),
        "sdgs": sdgs or [],
        "source": source_name,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build consolidated reference corpus.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing reference.json.")
    args = parser.parse_args()

    output_path = preprocessed_dir() / "reference.json"
    if output_path.exists() and not args.overwrite:
        log.info("Skip — %s already exists", output_path)
        return

    all_records: list[dict] = []
    source_stats: dict[str, dict] = {}

    for cfg in SOURCES:
        path = cfg["path"]()
        if not path.exists():
            log.warning("  %s: input not found — %s", cfg["name"], path)
            source_stats[cfg["name"]] = {"raw": 0, "after_norm": 0}
            continue
        raw = load_jsonl(path)
        normed = [normalise_record(r, cfg) for r in raw]
        all_records.extend(normed)
        source_stats[cfg["name"]] = {"raw": len(raw), "after_norm": len(normed)}
        log.info("  %s: %d raw -> %d normalised", cfg["name"], len(raw), len(normed))

    total_raw = sum(s["raw"] for s in source_stats.values())
    log.info("Total raw records: %d", total_raw)

    # Dedup by exact text (first occurrence keeps)
    seen_texts: set[str] = set()
    deduped: list[dict] = []
    dedup_source_counts: Counter = Counter()
    for rec in all_records:
        text_key = rec.get("text", "").strip()
        if not text_key or text_key in seen_texts:
            continue
        seen_texts.add(text_key)
        deduped.append(rec)
        dedup_source_counts[rec.get("source", "unknown")] += 1

    total_deduped = len(deduped)
    total_removed = total_raw - total_deduped
    log.info(
        "Dedup: %d -> %d (%d removed, %.1f%%)",
        total_raw, total_deduped, total_removed,
        100.0 * total_removed / total_raw if total_raw else 0,
    )
    log.info("Per-source deduped counts: %s", dict(dedup_source_counts))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(deduped, f, ensure_ascii=False)
    log.info("Wrote %d records -> %s", total_deduped, output_path)


if __name__ == "__main__":
    main()

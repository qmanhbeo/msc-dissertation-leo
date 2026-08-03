"""
Build consolidated policy corpus from individual policy sources.

Reads preprocessed JSONL from individual_sources/ for each policy source
(policy_scrape, policy_manual, ungdc_sdg, sdgi), normalises fields,
adds source_family,
deduplicates by exact text (first occurrence wins), and writes policy.jsonl.

source_family mapping (moved from deleted 1_merge_policy_corpus.py):
  policy_scrape / policy_manual  → "curated_ai_sdg"
  ungdc_sdg                      → "ungdc_speeches"
  sdgi                           → "sdgi_vnr_vlr"

Output: preprocessed_dir() / "policy.jsonl"  (line-delimited JSON)

Run from project root:
    python 1_code/1_preprocess/1_build_policy_corpus.py
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

from model_utils import preprocessed_dir, individual_source_dir
from shard_pipeline_utils import atomic_write_json, ensure_dir, read_json

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

SOURCE_FAMILY_MAP = {
    "policy_scrape": "curated_ai_sdg",
    "policy_manual": "curated_ai_sdg",
    "ungdc_sdg": "ungdc_speeches",
    "sdgi": "sdgi_vnr_vlr",
}

SOURCES = [
    {"name": "policy_scrape", "path": lambda: individual_source_dir("policy_scrape") / "policy_scrape_clean.jsonl",
     "id_field": "id"},
    {"name": "policy_manual", "path": lambda: individual_source_dir("policy_manual") / "policy_manual_clean.jsonl",
     "id_field": "id"},
    {"name": "ungdc_sdg", "path": lambda: individual_source_dir("ungdc_sdg") / "ungdc_sdg_clean.jsonl",
     "id_field": "id"},
    {"name": "sdgi", "path": lambda: individual_source_dir("sdgi") / "sdgi_clean.jsonl",
     "id_field": "id"},
]

MERGED_ORDER = ["policy_scrape", "policy_manual", "ungdc_sdg", "sdgi"]


def load_jsonl(path: Path) -> list[dict]:
    records = []
    if not path.exists():
        log.warning("  Input not found: %s", path)
        return records
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def normalise_record(rec: dict, source_cfg: dict) -> dict:
    source_name = source_cfg["name"]
    id_val = rec.get(source_cfg["id_field"], "")
    return {
        "id": f"{source_name}_{id_val}",
        "text": rec.get("text", ""),
        "source": source_name,
        "source_family": SOURCE_FAMILY_MAP.get(source_name, source_name),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build consolidated policy corpus.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing policy.jsonl.")
    args = parser.parse_args()

    output_path = preprocessed_dir() / "policy.jsonl"
    if output_path.exists() and not args.overwrite:
        meta_path = preprocessed_dir() / "metadata" / "build_policy_corpus.json"
        kept = read_json(meta_path, {}).get("total_deduped", "?")
        log.info("skip: %s already exists (kept=%s from earlier run)", output_path, kept)
        return

    all_records: list[dict] = []
    source_stats: dict[str, int] = {}

    for cfg in SOURCES:
        path = cfg["path"]()
        if not path.exists():
            log.warning("  %s: input not found — %s", cfg["name"], path)
            source_stats[cfg["name"]] = 0
            continue
        raw = load_jsonl(path)
        normed = [normalise_record(r, cfg) for r in raw]
        all_records.extend(normed)
        source_stats[cfg["name"]] = len(normed)
        log.info("  %s: %d raw -> %d normalised", cfg["name"], len(raw), len(normed))

    total_raw = sum(source_stats.values())
    log.info("Total raw records: %d", total_raw)

    # Dedup by exact text, ordered by MERGED_ORDER so policy_scrape seeds first
    seen_texts: set[str] = set()
    deduped: list[dict] = []
    dedup_source_counts: Counter = Counter()
    for source_name in MERGED_ORDER:
        for rec in all_records:
            if rec.get("source") != source_name:
                continue
            text_key = rec.get("text", "").strip()
            if not text_key or text_key in seen_texts:
                continue
            seen_texts.add(text_key)
            deduped.append(rec)
            dedup_source_counts[source_name] += 1

    total_deduped = len(deduped)
    total_removed = total_raw - total_deduped
    log.info(
        "done: total_processed=%d kept=%d dropped=%d",
        total_raw, total_deduped, total_removed,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        for rec in deduped:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    log.info("%d rows written to %s", total_deduped, output_path)

    metadata = {
        "stage": "build_policy_corpus",
        "total_raw": total_raw,
        "total_deduped": total_deduped,
        "total_removed": total_removed,
        "per_source_raw": source_stats,
        "per_source_deduped": dict(dedup_source_counts),
    }
    meta_path = preprocessed_dir() / "metadata" / "build_policy_corpus.json"
    ensure_dir(meta_path.parent)
    atomic_write_json(meta_path, metadata)


if __name__ == "__main__":
    main()

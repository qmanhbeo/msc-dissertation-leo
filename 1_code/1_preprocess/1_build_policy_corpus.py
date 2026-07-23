"""
Merge all policy segment sources into a single model-specific policy corpus.

Reads model-agnostic policy sources (scrape, manual, UNGDC) from the preprocessed
directory and the model-specific SDGI from the segmented directory, producing one
policy corpus per embed model.

All paths resolved via model_utils helpers (preprocessed_dir, segmented_dir_for_model).

Run from project root:
    python 1_code/1_preprocess/1_build_policy_corpus.py --model all-mpnet-base-v2
"""

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
ANALYSIS_DIR = CODE_ROOT / "7_main_analysis" / "0_shared"
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from model_utils import preprocessed_dir, segmented_dir_for_model

MIN_WORD_COUNT = 20


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        print(f"  WARNING: {path} not found - skipping")
        return []
    segments = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                segments.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return segments


def csv_safe(row: dict) -> dict:
    safe = dict(row)
    if isinstance(safe.get("text"), str):
        safe["text"] = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", safe["text"])
    return safe


def main() -> None:
    parser = argparse.ArgumentParser(description="Build model-specific policy corpus.")
    parser.add_argument("--model", default="all-mpnet-base-v2", help="Embed model name (default: %(default)s)")
    args = parser.parse_args()
    SOURCES = [
        preprocessed_dir() / "policy_all" / "policy_scrape" / "policy_scrape_segments.jsonl",
        preprocessed_dir() / "policy_all" / "policy_manual" / "policy_manual_segments.jsonl",
        segmented_dir_for_model(args.model) / "sdgi.jsonl",
        preprocessed_dir() / "policy_all" / "ungdc_sdg" / "ungdc_sdg_segments.jsonl",
    ]

    output_dir = segmented_dir_for_model(args.model)
    OUTPUT_JSONL = output_dir / "policy.jsonl"
    OUTPUT_CSV = output_dir / "policy.csv"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_segments: list[dict] = []
    source_counts: Counter = Counter()

    for source_path in SOURCES:
        raw = load_jsonl(source_path)
        added = 0
        skipped_short = 0

        for segment in raw:
            text = segment.get("text", "").strip()
            if not text:
                continue
            word_count = segment.get("word_count", len(text.split()))
            if word_count < MIN_WORD_COUNT:
                skipped_short += 1
                continue
            all_segments.append(segment)
            added += 1

        label = source_path.name
        source_counts[label] = added
        print(f"  {label}: {len(raw)} raw -> {added} kept ({skipped_short} too short)")

    print(f"\nTotal segments (pre-dedup): {len(all_segments)}")

    seen_texts: set[str] = set()
    dedup_kept: list[dict] = []
    for segment in all_segments:
        text_key = segment.get("text", "").strip()
        if text_key in seen_texts:
            continue
        seen_texts.add(text_key)
        dedup_kept.append(segment)
    n_dedup = len(all_segments) - len(dedup_kept)
    all_segments = dedup_kept
    print(f"Deduplication: removed {n_dedup} exact-duplicate segments ({100 * n_dedup / (len(all_segments) + n_dedup):.1f}%)")

    for index, segment in enumerate(all_segments):
        segment["segment_id_merged"] = f"merged_{index:06d}"

    word_counts = sorted(segment.get("word_count", len(segment["text"].split())) for segment in all_segments)
    if word_counts:
        print(f"Word counts - min: {word_counts[0]}, median: {word_counts[len(word_counts)//2]}, max: {word_counts[-1]}")

    with OUTPUT_JSONL.open("w", encoding="utf-8") as handle:
        for segment in all_segments:
            handle.write(json.dumps(segment) + "\n")
    print(f"✓ JSONL -> {OUTPUT_JSONL}")

    csv_fields = ["segment_id_merged", "segment_id", "source_doc", "segment_index", "word_count", "text"]
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields, extrasaction="ignore", quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(csv_safe(segment) for segment in all_segments)
    print(f"✓ CSV  -> {OUTPUT_CSV}")

    print("\nSource breakdown:")
    for source, count in source_counts.items():
        pct = count / max(len(all_segments), 1) * 100
        print(f"  {source}: {count} segments ({pct:.1f}%)")


if __name__ == "__main__":
    main()

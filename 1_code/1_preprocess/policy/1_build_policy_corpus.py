"""
Merge all policy chunk sources into a single policy corpus.

Input sources:
  2_data/1_preprocessed/policy_all/policy_scrape/policy_scrape_chunks.jsonl
  2_data/1_preprocessed/policy_all/policy_manual/policy_manual_chunks.jsonl
  2_data/1_preprocessed/policy_all/sdgi_corpus/sdgi_chunks.jsonl
  2_data/1_preprocessed/policy_all/ungdc_sdg/ungdc_sdg_chunks.jsonl

Output:
  2_data/1_preprocessed/policy_all/policy_chunks_all.jsonl
  2_data/1_preprocessed/policy_all/policy_chunks_all.csv

Merge rule: concatenate source corpora in order, while keeping the existing
minimum word-count filter.

Run from project root:
    python 1_code/1_preprocess/policy/1_build_policy_corpus.py
"""

import csv
import json
import re
from collections import Counter
from pathlib import Path

SOURCES = [
    Path("2_data/1_preprocessed/policy_all/policy_scrape/policy_scrape_chunks.jsonl"),
    Path("2_data/1_preprocessed/policy_all/policy_manual/policy_manual_chunks.jsonl"),
    Path("2_data/1_preprocessed/policy_all/sdgi_corpus/sdgi_chunks.jsonl"),
    Path("2_data/1_preprocessed/policy_all/ungdc_sdg/ungdc_sdg_chunks.jsonl"),
]

OUTPUT_DIR = Path("2_data/1_preprocessed/policy_all")
OUTPUT_JSONL = OUTPUT_DIR / "policy_chunks_all.jsonl"
OUTPUT_CSV = OUTPUT_DIR / "policy_chunks_all.csv"
MIN_WORD_COUNT = 20


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        print(f"  WARNING: {path} not found - skipping")
        return []
    chunks = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                chunks.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return chunks


def csv_safe(row: dict) -> dict:
    safe = dict(row)
    if isinstance(safe.get("text"), str):
        safe["text"] = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", safe["text"])
    return safe


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_chunks: list[dict] = []
    source_counts: Counter = Counter()

    for source_path in SOURCES:
        raw = load_jsonl(source_path)
        added = 0
        skipped_short = 0

        for chunk in raw:
            text = chunk.get("text", "").strip()
            if not text:
                continue
            word_count = chunk.get("word_count", len(text.split()))
            if word_count < MIN_WORD_COUNT:
                skipped_short += 1
                continue
            all_chunks.append(chunk)
            added += 1

        label = source_path.name
        source_counts[label] = added
        print(f"  {label}: {len(raw)} raw -> {added} kept ({skipped_short} too short)")

    print(f"\nTotal chunks: {len(all_chunks)}")

    for index, chunk in enumerate(all_chunks):
        chunk["chunk_id_merged"] = f"merged_{index:06d}"

    word_counts = sorted(chunk.get("word_count", len(chunk["text"].split())) for chunk in all_chunks)
    if word_counts:
        print(f"Word counts - min: {word_counts[0]}, median: {word_counts[len(word_counts)//2]}, max: {word_counts[-1]}")

    with OUTPUT_JSONL.open("w", encoding="utf-8") as handle:
        for chunk in all_chunks:
            handle.write(json.dumps(chunk) + "\n")
    print(f"✓ JSONL -> {OUTPUT_JSONL}")

    csv_fields = ["chunk_id_merged", "chunk_id", "source_doc", "chunk_index", "word_count", "text"]
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields, extrasaction="ignore", quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(csv_safe(chunk) for chunk in all_chunks)
    print(f"✓ CSV  -> {OUTPUT_CSV}")

    print("\nSource breakdown:")
    for source, count in source_counts.items():
        pct = count / max(len(all_chunks), 1) * 100
        print(f"  {source}: {count} chunks ({pct:.1f}%)")


if __name__ == "__main__":
    main()

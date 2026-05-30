"""
Merge all policy chunk sources into a single extended corpus.

Input sources (each independently produced):
  data/1_preprocessed/policy_all/policy_scrape/policy_chunks.jsonl        — chunks from unified policy preprocessing
  data/1_preprocessed/policy_all/sdgi_corpus/sdgi_chunks.jsonl     — VNR/VLR corpus (integrate_sdgi.py)
  data/1_preprocessed/policy_all/ungdc_sdg/ungdc_sdg_chunks.jsonl  — UNGDC filtered passages (filter_ungdc_sdg.py)

Output:
  data/1_preprocessed/policy_all/policy_chunks_extended.jsonl  — merged, deduplicated
  data/1_preprocessed/policy_all/policy_chunks_extended.csv    — flat CSV for inspection

Deduplication: exact text match (after normalisation).

Run from project root:
    python code/preprocess/build_policy_corpus.py

After running, re-embed with:
    python code/embed/embeddings.py --corpus policy_all
"""

import csv
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

SOURCES = [
    Path("data/1_preprocessed/policy_all/policy_scrape/policy_chunks.jsonl"),
    Path("data/1_preprocessed/policy_all/sdgi_corpus/sdgi_chunks.jsonl"),
    Path("data/1_preprocessed/policy_all/ungdc_sdg/ungdc_sdg_chunks.jsonl"),
]

OUTPUT_DIR = Path("data/1_preprocessed/policy_all")
OUTPUT_JSONL = OUTPUT_DIR / "policy_chunks_extended.jsonl"
OUTPUT_CSV = OUTPUT_DIR / "policy_chunks_extended.csv"

# Minimum word count to include in the merged corpus
MIN_WORD_COUNT = 20


def normalise_text(text: str) -> str:
    """Lightweight normalisation for deduplication only (not stored)."""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        print(f"  WARNING: {path} not found — skipping")
        return []
    chunks = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    chunks.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return chunks


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_chunks: list[dict] = []
    seen_texts: set[str] = set()
    source_counts: Counter = Counter()

    for source_path in SOURCES:
        raw = load_jsonl(source_path)
        added = 0
        skipped_short = 0
        skipped_dup = 0

        for chunk in raw:
            text = chunk.get("text", "").strip()
            if not text:
                continue
            wc = chunk.get("word_count", len(text.split()))
            if wc < MIN_WORD_COUNT:
                skipped_short += 1
                continue
            key = normalise_text(text)
            if key in seen_texts:
                skipped_dup += 1
                continue
            seen_texts.add(key)
            all_chunks.append(chunk)
            added += 1

        label = source_path.name
        source_counts[label] = added
        print(
            f"  {label}: {len(raw)} raw → {added} kept "
            f"({skipped_short} too short, {skipped_dup} duplicates)"
        )

    print(f"\nTotal chunks: {len(all_chunks)}")

    # Reassign sequential chunk IDs to avoid collisions across sources
    for i, chunk in enumerate(all_chunks):
        chunk["chunk_id_merged"] = f"merged_{i:06d}"

    # Word count summary
    wcs = sorted(c.get("word_count", len(c["text"].split())) for c in all_chunks)
    print(
        f"Word counts — min: {wcs[0]}, median: {wcs[len(wcs)//2]}, max: {wcs[-1]}"
    )

    # Save JSONL
    with OUTPUT_JSONL.open("w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk) + "\n")
    print(f"✓ JSONL → {OUTPUT_JSONL}")

    # Save CSV — sanitize text field (null bytes / control chars from OCR crash csv)
    def _csv_safe(row: dict) -> dict:
        safe = {k: v for k, v in row.items()}
        if "text" in safe and isinstance(safe["text"], str):
            safe["text"] = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", safe["text"])
        return safe

    csv_fields = ["chunk_id_merged", "chunk_id", "source_doc", "chunk_index", "word_count", "text"]
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=csv_fields, extrasaction="ignore",
            quoting=csv.QUOTE_ALL,
        )
        writer.writeheader()
        writer.writerows(_csv_safe(c) for c in all_chunks)
    print(f"✓ CSV  → {OUTPUT_CSV}")

    print(f"\nSource breakdown:")
    for source, count in source_counts.items():
        pct = count / max(len(all_chunks), 1) * 100
        print(f"  {source}: {count} chunks ({pct:.1f}%)")

    print(f"\nNext: re-run embeddings.py pointing at {OUTPUT_JSONL}")
    print("  Update embeddings.py POLICY_CHUNKS path or pass as argument.")


if __name__ == "__main__":
    main()

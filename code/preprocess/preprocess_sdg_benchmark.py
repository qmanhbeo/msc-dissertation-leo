"""
Preprocess SDG Benchmark corpus (expert-verified SDG-labeled texts).

Input:  data/raw/sdg_benchmark/benchmark.csv  (1,251 rows, label True/False)
Output: data/preprocessed/sdg_benchmark/benchmark_clean.jsonl  — positive examples only
        data/preprocessed/sdg_benchmark/benchmark_clean.csv    — flat CSV for inspection

Filtering:
  - Keep only rows where label == True (expert-confirmed SDG relevance)
  - Drop rows with text shorter than MIN_WORDS

Role in pipeline:
  Higher-quality complement to OSDG — used to:
  1. Cross-validate SDG centroid embeddings built from OSDG
  2. Evaluate alignment scoring (ground truth for SDG relevance)
  3. Provide a held-out evaluation set if needed

Run from project root:
    python code/preprocess/preprocess_sdg_benchmark.py
"""

import csv
import json
import logging
import re
import unicodedata
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
INPUT_FILE = Path("data/raw/sdg_benchmark/benchmark.csv")
OUTPUT_JSONL = Path("data/preprocessed/sdg_benchmark/benchmark_clean.jsonl")
OUTPUT_CSV = Path("data/preprocessed/sdg_benchmark/benchmark_clean.csv")

MIN_WORDS = 10  # lower threshold — benchmark texts tend to be shorter

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------
_BOILERPLATE = [
    re.compile(r"©\s*\d{4}.*", re.IGNORECASE),
    re.compile(r"all rights reserved\.?", re.IGNORECASE),
    re.compile(r"https?://\S+"),
    re.compile(r"\S+@\S+\.\S+"),
    re.compile(r"\b(doi|DOI):\s*\S+"),
]
_MULTI_SPACE = re.compile(r"\s{2,}")


def normalize_unicode(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    return text


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = normalize_unicode(text)
    for pattern in _BOILERPLATE:
        text = pattern.sub(" ", text)
    text = _MULTI_SPACE.sub(" ", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    log.info("Loading %s", INPUT_FILE)

    raw_rows: list[dict] = []
    with INPUT_FILE.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_rows.append(row)

    log.info("Loaded %d raw rows (%d True, %d False)",
             len(raw_rows),
             sum(1 for r in raw_rows if r["label"].strip().lower() == "true"),
             sum(1 for r in raw_rows if r["label"].strip().lower() == "false"))

    kept, dropped_label, dropped_text = [], 0, 0

    for row in raw_rows:
        # Keep only expert-confirmed positive examples
        if row.get("label", "").strip().lower() != "true":
            dropped_label += 1
            continue

        text = clean_text(row.get("text", ""))

        if len(text.split()) < MIN_WORDS:
            dropped_text += 1
            log.warning("Dropping short text (id=%s, %d words)", row.get("id"), len(text.split()))
            continue

        kept.append({
            "id": row.get("id", ""),
            "text": text,
            "sdg": int(row["sdg"]),
            "word_count": len(text.split()),
        })

    log.info(
        "Kept: %d  |  Dropped (False label): %d  |  Dropped (short text): %d",
        len(kept), dropped_label, dropped_text,
    )

    # SDG distribution
    sdg_counts = Counter(r["sdg"] for r in kept)
    log.info("SDG distribution: %s", dict(sorted(sdg_counts.items())))

    # Word count summary
    wcs = sorted(r["word_count"] for r in kept)
    log.info(
        "Text length — min: %d  median: %d  max: %d words",
        wcs[0], wcs[len(wcs) // 2], wcs[-1],
    )

    # Save JSONL
    OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_JSONL.open("w") as f:
        for row in kept:
            f.write(json.dumps(row) + "\n")
    log.info("Saved JSONL → %s", OUTPUT_JSONL)

    # Save CSV
    csv_fields = ["id", "sdg", "word_count", "text"]
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(kept)
    log.info("Saved CSV  → %s", OUTPUT_CSV)

    print(f"\nDone. {len(kept)} rows written to {OUTPUT_JSONL}")
    print(f"  SDGs covered: {sorted(sdg_counts.keys())}")
    print(f"  Examples per SDG — min: {min(sdg_counts.values())}  max: {max(sdg_counts.values())}")


if __name__ == "__main__":
    main()

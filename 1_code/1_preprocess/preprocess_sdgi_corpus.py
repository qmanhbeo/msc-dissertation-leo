"""
Preprocess SDGi Corpus: extract multi-label texts for all 17 SDGs.

Input:  2_data/0_raw/sdgi_corpus/sdgi_corpus.parquet  (5,880 rows, multi-label)
Output: 2_data/1_preprocessed/sdgi_corpus/sdgi_clean.jsonl
        2_data/1_preprocessed/sdgi_corpus/sdgi_clean.csv

Filtering:
  - Keep all texts regardless of SDG count (multi-label preserved)
  - Drop texts shorter than MIN_WORDS

Cleaning:
  - Normalize Unicode, strip boilerplate (URLs, emails, copyright)
  - Normalize whitespace

Fields:
  - sdgs: list[int] — all active SDG labels for this text
  - Single-label texts are filtered at MLP training time, not here.

Run from project root:
    python 1_code/1_preprocess/preprocess_sdgi_corpus.py
"""

import csv
import json
import logging
import re
import unicodedata
from pathlib import Path

import numpy as np

INPUT_FILE = Path("2_data/0_raw/sdgi_corpus/sdgi_corpus.parquet")
OUTPUT_JSONL = Path("2_data/1_preprocessed/sdgi_corpus/sdgi_clean.jsonl")
OUTPUT_CSV = Path("2_data/1_preprocessed/sdgi_corpus/sdgi_clean.csv")

MIN_WORDS = 20

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

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

def main() -> None:
    log.info("Loading %s", INPUT_FILE)

    import pandas as pd
    df = pd.read_parquet(INPUT_FILE)

    log.info("Loaded %d raw rows", len(df))

    kept, dropped_text, multi_count = [], 0, 0

    for idx, row in df.iterrows():
        text = clean_text(row.get("text", ""))
        if len(text.split()) < MIN_WORDS:
            dropped_text += 1
            continue

        labels = row.get("labels")
        if not isinstance(labels, np.ndarray) or len(labels) == 0:
            continue

        active = sorted(int(l) for l in labels)
        if len(active) > 1:
            multi_count += 1

        metadata = row.get("metadata", {})
        country = metadata.get("country", "") if isinstance(metadata, dict) else ""

        kept.append({
            "id": f"sdgi_{idx}",
            "text": text,
            "sdgs": active,
            "country": country,
            "word_count": len(text.split()),
        })

    log.info(
        "Total: %d kept (%d multi-label)  |  Dropped (short text): %d  |  SDG distribution: %s",
        len(kept), multi_count, dropped_text,
        dict(sorted({n: sum(1 for r in kept if len(r["sdgs"]) == n) for n in range(1, 18)}.items())),
    )

    OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_JSONL.open("w") as f:
        for row in kept:
            f.write(json.dumps(row) + "\n")
    log.info("Saved JSONL -> %s", OUTPUT_JSONL)

    csv_fields = ["id", "sdgs", "word_count", "country", "text"]
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(kept)
    log.info("Saved CSV  -> %s", OUTPUT_CSV)

    print(f"\nDone. {len(kept)} rows written to {OUTPUT_JSONL}")

if __name__ == "__main__":
    main()

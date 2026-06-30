"""
Preprocess SDGi Corpus: extract single-label texts for all 17 SDGs.

Input:  2_data/0_raw/sdgi_corpus/sdgi_corpus.parquet  (5,880 rows, multi-label)
Output: 2_data/1_preprocessed/sdgi_corpus/sdgi_clean.jsonl
        2_data/1_preprocessed/sdgi_corpus/sdgi_clean.csv

Filtering:
  - Keep only single-label texts (labels == [sdg] for sdg in 1..17)
  - Drop texts shorter than MIN_WORDS

Cleaning:
  - Normalize Unicode, strip boilerplate (URLs, emails, copyright)
  - Normalize whitespace

Role in pipeline:
  Provides a within-genre (policy VNR/VLR) SDG reference corpus for supplementing
  the OSDG and Knowledge Hub corpora in building per-SDG centroids.

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

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
INPUT_FILE = Path("2_data/0_raw/sdgi_corpus/sdgi_corpus.parquet")
OUTPUT_JSONL = Path("2_data/1_preprocessed/sdgi_corpus/sdgi_clean.jsonl")
OUTPUT_CSV = Path("2_data/1_preprocessed/sdgi_corpus/sdgi_clean.csv")

MIN_WORDS = 20

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Text cleaning (shared pattern with preprocess_osdg.py)
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

    import pandas as pd
    df = pd.read_parquet(INPUT_FILE)

    log.info("Loaded %d raw rows", len(df))

    # Collect single-label texts for all 17 SDGs
    kept, dropped_text = [], 0

    for sdg in range(1, 18):
        single_mask = df["labels"].apply(
            lambda x, sdg=sdg: isinstance(x, np.ndarray) and len(x) == 1 and x[0] == sdg
        )
        n_single = int(single_mask.sum())
        if n_single == 0:
            log.info("  Single-label SDG %2d: 0 / %d rows", sdg, len(df))
            continue

        prev_kept = len(kept)
        for idx, row in df[single_mask].iterrows():
            text = clean_text(row.get("text", ""))
            if len(text.split()) < MIN_WORDS:
                dropped_text += 1
                continue

            metadata = row.get("metadata", {})
            country = metadata.get("country", "") if isinstance(metadata, dict) else ""

            kept.append({
                "id": f"sdgi_{idx}",
                "text": text,
                "sdg": sdg,
                "country": country,
                "word_count": len(text.split()),
            })

        n_kept = len(kept) - prev_kept
        log.info("  Single-label SDG %2d: %d kept from %d (dropped %d)", sdg, n_kept, n_single, n_single - n_kept)

    log.info(
        "Total single-label: %d kept across all SDGs  |  Dropped (short text): %d",
        len(kept), dropped_text,
    )

    OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_JSONL.open("w") as f:
        for row in kept:
            f.write(json.dumps(row) + "\n")
    log.info("Saved JSONL → %s", OUTPUT_JSONL)

    csv_fields = ["id", "sdg", "word_count", "country", "text"]
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(kept)
    log.info("Saved CSV  → %s", OUTPUT_CSV)

    print(f"\nDone. {len(kept)} rows written to {OUTPUT_JSONL}")


if __name__ == "__main__":
    main()

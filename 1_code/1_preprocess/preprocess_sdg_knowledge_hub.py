"""
Preprocess SDG Knowledge Hub corpus: extract single-label texts for all 17 SDGs.

Input:  2_data/0_raw/sdg_knowledge_hub/sdg_knowledge_hub.csv  (9,172 rows, multi-label)
Output: 2_data/1_preprocessed/sdg_knowledge_hub/sdg_knowledge_hub_clean.jsonl
        2_data/1_preprocessed/sdg_knowledge_hub/sdg_knowledge_hub_clean.csv

Filtering:
  - Keep only single-label texts (exactly one SDG flag = 1, all others = 0)
  - Drop texts shorter than MIN_WORDS (matching OSDG convention: 20)

Cleaning:
  - Normalize Unicode, strip boilerplate (URLs, emails, copyright)
  - Normalize whitespace

Role in pipeline:
  Provides an independent journalistic-domain SDG reference corpus for supplementing
  the OSDG and SDGi corpora in building per-SDG centroids.

Run from project root:
    python 1_code/1_preprocess/preprocess_sdg_knowledge_hub.py
"""

import csv
import hashlib
import json
import logging
import re
import unicodedata
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
INPUT_FILE = Path("2_data/0_raw/sdg_knowledge_hub/sdg_knowledge_hub.csv")
OUTPUT_JSONL = Path("2_data/1_preprocessed/sdg_knowledge_hub/sdg_knowledge_hub_clean.jsonl")
OUTPUT_CSV = Path("2_data/1_preprocessed/sdg_knowledge_hub/sdg_knowledge_hub_clean.csv")

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
    df = pd.read_csv(INPUT_FILE)

    log.info("Loaded %d raw rows", len(df))

    # Collect single-label texts for all 17 SDGs
    sdg_cols = [c for c in df.columns if c.startswith("SDG-")]
    kept, dropped_text = [], 0

    for col in sdg_cols:
        sdg_num = int(col.split("-")[1])
        other_cols = [c for c in sdg_cols if c != col]
        single_mask = (df[col] == 1) & (df[other_cols].sum(axis=1) == 0)
        n_single = int(single_mask.sum())
        prev_kept = len(kept)

        for _, row in df[single_mask].iterrows():
            text = clean_text(row.get("text", ""))
            if len(text.split()) < MIN_WORDS:
                dropped_text += 1
                continue
            kept.append({
                "id": hashlib.sha256(row["url"].encode()).hexdigest()[:16],
                "url": row["url"],
                "text": text,
                "sdg": sdg_num,
                "word_count": len(text.split()),
            })

        n_kept = len(kept) - prev_kept
        log.info("  Single-label SDG %2d: %d kept from %d (dropped %d)", sdg_num, n_kept, n_single, n_single - n_kept)

    log.info(
        "Total single-label: %d kept across all SDGs  |  Total dropped (short text): %d",
        len(kept), dropped_text,
    )

    OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_JSONL.open("w") as f:
        for row in kept:
            f.write(json.dumps(row) + "\n")
    log.info("Saved JSONL → %s", OUTPUT_JSONL)

    csv_fields = ["id", "sdg", "word_count", "url", "text"]
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(kept)
    log.info("Saved CSV  → %s", OUTPUT_CSV)

    print(f"\nDone. {len(kept)} rows written to {OUTPUT_JSONL}")


if __name__ == "__main__":
    main()

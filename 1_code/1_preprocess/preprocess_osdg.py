"""
Preprocess OSDG corpus for use as SDG classification training signal.

Input:  2_data/0_raw/osdg/osdg_dataset.csv  (TSV, 43,025 rows)
Output: 2_data/1_preprocessed/osdg/osdg_clean.jsonl  — filtered, cleaned records
        2_data/1_preprocessed/osdg/osdg_clean.csv    — flat CSV for inspection

Filtering:
  - Keep rows where agreement >= AGREEMENT_THRESHOLD (default 0.5)
  - Drop rows with empty or very short text (< 20 words)

Cleaning:
  - Normalize Unicode, strip boilerplate (URLs, emails, copyright)
  - Normalize whitespace

Role in pipeline:
  This corpus provides SDG-labeled text snippets used to:
  1. Validate SDG topic clusters from topic modeling
  2. Build a reference embedding per SDG (centroid of labeled texts)
  3. Provide a classification training signal if supervised labeling is needed

Run from project root:
    python 1_code/1_preprocess/preprocess_osdg.py
"""

import csv
import json
import logging
import re
import unicodedata
from collections import Counter
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
ANALYSIS_DIR = CODE_ROOT / "7_main_analysis" / "0_shared"
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))
from model_utils import raw_dir, preprocessed_dir

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
INPUT_FILE = raw_dir() / "osdg" / "osdg_dataset.csv"
OUTPUT_JSONL = preprocessed_dir() / "osdg" / "osdg_clean.jsonl"
OUTPUT_CSV = preprocessed_dir() / "osdg" / "osdg_clean.csv"

AGREEMENT_THRESHOLD = 0.5   # minimum annotator agreement to keep a row
MIN_WORDS = 20              # drop texts shorter than this

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Text cleaning (shared pattern with preprocess_papers.py)
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
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            raw_rows.append(row)

    log.info("Loaded %d raw rows", len(raw_rows))

    kept, dropped_agreement, dropped_text = [], 0, 0

    for row in raw_rows:
        agreement = float(row.get("agreement", 0))

        # Agreement filter
        if agreement < AGREEMENT_THRESHOLD:
            dropped_agreement += 1
            continue

        text = clean_text(row.get("text", ""))

        # Text quality filter
        if len(text.split()) < MIN_WORDS:
            dropped_text += 1
            continue

        kept.append({
            "text_id": row.get("text_id", ""),
            "doi": row.get("doi", ""),
            "text": text,
            "sdg": int(row["sdg"]),
            "agreement": agreement,
            "word_count": len(text.split()),
        })

    log.info(
        "Kept: %d  |  Dropped (low agreement): %d  |  Dropped (short text): %d",
        len(kept), dropped_agreement, dropped_text,
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
    csv_fields = ["text_id", "sdg", "agreement", "word_count", "doi", "text"]
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(kept)
    log.info("Saved CSV  → %s", OUTPUT_CSV)

    print(f"\nDone. {len(kept)} rows written to {OUTPUT_JSONL}")
    print(f"  Agreement threshold: >= {AGREEMENT_THRESHOLD}")
    print(f"  SDGs covered: {sorted(sdg_counts.keys())}")


if __name__ == "__main__":
    main()

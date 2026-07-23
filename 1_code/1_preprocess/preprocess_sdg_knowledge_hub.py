"""
Preprocess SDG Knowledge Hub corpus: extract multi-label texts for all 17 SDGs.

Input:  2_data/0_raw/sdg_knowledge_hub/sdg_knowledge_hub.csv  (9,172 rows, multi-label)
Output: 2_data/1_preprocessed/sdg_knowledge_hub/sdg_knowledge_hub_clean.jsonl
        2_data/1_preprocessed/sdg_knowledge_hub/sdg_knowledge_hub_clean.csv

Filtering:
  - Keep all texts regardless of SDG count (multi-label preserved)
  - Drop texts shorter than MIN_WORDS (matching OSDG convention: 20)

Cleaning:
  - Normalize Unicode, strip boilerplate (URLs, emails, copyright)
  - Normalize whitespace

Fields:
  - sdgs: list[int] — all active SDG labels for this text
  - Single-label texts are filtered at MLP training time, not here.

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

import sys

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
ANALYSIS_DIR = CODE_ROOT / "7_main_analysis" / "0_shared"
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))
from model_utils import raw_dir, preprocessed_dir

INPUT_FILE = raw_dir() / "sdg_knowledge_hub" / "sdg_knowledge_hub.csv"
OUTPUT_JSONL = preprocessed_dir() / "sdg_knowledge_hub" / "sdg_knowledge_hub_clean.jsonl"
OUTPUT_CSV = preprocessed_dir() / "sdg_knowledge_hub" / "sdg_knowledge_hub_clean.csv"

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
    df = pd.read_csv(INPUT_FILE)

    log.info("Loaded %d raw rows", len(df))

    sdg_cols = [c for c in df.columns if c.startswith("SDG-")]
    kept, dropped_text, multi_count = [], 0, 0

    for _, row in df.iterrows():
        text = clean_text(row.get("text", ""))
        if len(text.split()) < MIN_WORDS:
            dropped_text += 1
            continue

        active = [int(c.split("-")[1]) for c in sdg_cols if row[c] == 1]
        if not active:
            continue

        if len(active) > 1:
            multi_count += 1

        kept.append({
            "id": hashlib.sha256(row["url"].encode()).hexdigest()[:16],
            "url": row["url"],
            "text": text,
            "sdgs": active,
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

    csv_fields = ["id", "sdgs", "word_count", "url", "text"]
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(kept)
    log.info("Saved CSV  -> %s", OUTPUT_CSV)

    print(f"\nDone. {len(kept)} rows written to {OUTPUT_JSONL}")

if __name__ == "__main__":
    main()

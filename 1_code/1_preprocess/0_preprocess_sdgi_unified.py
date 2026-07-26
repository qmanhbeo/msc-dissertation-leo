"""
Unified SDGi preprocessor — clean text only, no segmentation.

Output per-document cleaned records to 1_preprocessed/sdgi/sdgi_clean.jsonl.
Segmentation is now handled by segment_corpus.py in the dedicated segment stage.

Preserves: English filter, text cleaning, SDG label extraction, metadata fields.

Run from project root:
    python 1_code/1_preprocess/0_preprocess_sdgi_unified.py
"""

import json
import logging
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
ANALYSIS_DIR = CODE_ROOT / "7_main_analysis" / "0_shared"
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from model_utils import raw_dir, preprocessed_dir

INPUT_PARQUET = raw_dir() / "sdgi_corpus" / "sdgi_corpus.parquet"
OUTPUT_DIR = preprocessed_dir() / "sdgi"
OUTPUT_JSONL = OUTPUT_DIR / "sdgi_clean.jsonl"
TARGET_LANGUAGE = "en"
MIN_WORDS = 20

_BOILERPLATE = [
    re.compile(r"©\s*\d{4}.*", re.IGNORECASE),
    re.compile(r"all rights reserved\.?", re.IGNORECASE),
    re.compile(r"https?://\S+"),
    re.compile(r"\S+@\S+\.\S+"),
    re.compile(r"\b(doi|DOI):\s*\S+"),
]
_MULTI_SPACE = re.compile(r"\s{2,}")

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


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


def sanitise_source_doc(value: str) -> str:
    return value.replace("/", "_").replace(" ", "_").replace("-", "_")


def main() -> None:
    log.info("Loading %s", INPUT_PARQUET)
    df = pd.read_parquet(INPUT_PARQUET)
    log.info("Loaded %d raw rows", len(df))

    # --- English filter ---
    def is_english(meta: object) -> bool:
        if isinstance(meta, dict):
            return meta.get("language", "") == TARGET_LANGUAGE
        return False

    n_before_lang = len(df)
    df = df[df["metadata"].apply(is_english)].copy()
    log.info("English rows: %d (dropped %d)", len(df), n_before_lang - len(df))

    records_out: list[dict] = []
    total_before_word = 0
    total_after_word = 0

    for idx, row in df.iterrows():
        meta = row["metadata"] if isinstance(row["metadata"], dict) else {}

        text = str(row.get("text", "") or "")
        text = clean_text(text)
        wc = len(text.split())
        total_before_word = max(total_before_word, wc)
        if wc < MIN_WORDS:
            continue
        total_after_word += wc

        labels = row.get("labels")
        if isinstance(labels, np.ndarray):
            sdgs = sorted(int(l) for l in labels)
        elif isinstance(labels, list):
            sdgs = sorted(int(l) for l in labels)
        elif labels is not None:
            sdgs = [int(labels)]
        else:
            continue

        country = meta.get("country", "unknown").upper()
        doc_type = meta.get("type", "policy").upper()
        locality = meta.get("locality", "")
        year = meta.get("year")
        file_id = meta.get("file_id", "")

        source_doc = (
            f"sdgi_{country}_{doc_type}"
            + (f"_{locality}" if locality else "")
            + (f"_{year}" if year else "")
        )
        source_doc = sanitise_source_doc(source_doc)

        records_out.append({
            "id": f"sdgi_{idx:05d}",
            "text": text,
            "source_doc": source_doc,
            "sdgs": sdgs,
            "institution": f"{country} ({doc_type})",
            "year": year,
            "original_file": file_id,
            "country": country,
        })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUTPUT_JSONL.open("w", encoding="utf-8") as f:
        for r in records_out:
            f.write(json.dumps(r) + "\n")

    log.info("Wrote %d documents -> %s", len(records_out), OUTPUT_JSONL)

    # --- Statistics ---
    n_multi = sum(1 for r in records_out if len(r["sdgs"]) > 1)
    n_single = len(records_out) - n_multi
    log.info(
        "Documents: %d total (%d single-label, %d multi-label)",
        len(records_out), n_single, n_multi,
    )

    from collections import Counter
    sdg_counts: Counter = Counter()
    for r in records_out:
        for sdg in r["sdgs"]:
            sdg_counts[sdg] += 1
    log.info("SDG label distribution: %s", dict(sorted(sdg_counts.items())))

    print(f"\nDone. {len(records_out)} documents -> {OUTPUT_JSONL}")


if __name__ == "__main__":
    main()

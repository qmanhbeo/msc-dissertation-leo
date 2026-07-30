"""
Unified SDGi preprocessor — clean text only, no segmentation.

Output per-document cleaned records to 1_preprocessed/individual_sources/sdgi/sdgi_clean.jsonl.
Segmentation is now handled by segment_corpus.py in the dedicated segment stage.

Preserves: English filter, text cleaning, SDG label extraction, metadata fields.

Run from project root:
    python 1_code/1_preprocess/0_preprocess_sdgi_unified.py
"""

import argparse
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

from model_utils import raw_dir, preprocessed_dir, individual_source_dir
from _resume import resumable_records

INPUT_PARQUET = raw_dir() / "sdgi_corpus" / "sdgi_corpus.parquet"
OUTPUT_DIR = individual_source_dir("sdgi")
OUTPUT_JSONL = OUTPUT_DIR / "sdgi_clean.jsonl"
STATE_PATH = OUTPUT_DIR / "sdgi_state.json"
STATUS_DIR = OUTPUT_DIR / "metadata"
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


def read_records():
    df = pd.read_parquet(INPUT_PARQUET)
    for idx, row in df.iterrows():
        yield idx, row


def transform(payload) -> dict | None:
    idx, row = payload

    meta = row["metadata"] if isinstance(row["metadata"], dict) else {}
    if not (isinstance(meta, dict) and meta.get("language", "") == TARGET_LANGUAGE):
        return None

    text = str(row.get("text", "") or "")
    text = clean_text(text)
    if len(text.split()) < MIN_WORDS:
        return None

    labels = row.get("labels")
    if isinstance(labels, np.ndarray):
        sdgs = sorted(int(l) for l in labels)
    elif isinstance(labels, list):
        sdgs = sorted(int(l) for l in labels)
    elif labels is not None:
        sdgs = [int(labels)]
    else:
        return None

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

    # NOTE: idx is the global parquet row index; it advances for EVERY input
    # row (including dropped ones) because read_records yields all rows and
    # resumable_records increments rows_done per yielded record. This
    # guarantees id = sdgi_{idx:05d} reproduces exactly across resumes.
    return {
        "id": f"sdgi_{idx:05d}",
        "text": text,
        "source_doc": source_doc,
        "sdgs": sdgs,
        "institution": f"{country} ({doc_type})",
        "year": year,
        "original_file": file_id,
        "country": country,
        "source": "sdgi",
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Preprocess SDGi corpus (resume-safe).")
    p.add_argument("--input", default=str(INPUT_PARQUET))
    p.add_argument("--out-jsonl", default=str(OUTPUT_JSONL))
    p.add_argument("--state", default=str(STATE_PATH))
    p.add_argument("--status-dir", default=str(STATUS_DIR))
    p.add_argument("--chunk-size", type=int, default=5000)
    p.add_argument("--reset", action="store_true", help="Delete checkpoint + output and start fresh.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    global INPUT_PARQUET, OUTPUT_JSONL, STATE_PATH, STATUS_DIR
    INPUT_PARQUET = Path(args.input)
    OUTPUT_JSONL = Path(args.out_jsonl)
    STATE_PATH = Path(args.state)
    STATUS_DIR = Path(args.status_dir)

    resumable_records(
        stage="preprocess_sdgi_unified",
        read_records=read_records,
        transform=transform,
        out_path=OUTPUT_JSONL,
        state_path=STATE_PATH,
        status_dir=STATUS_DIR,
        chunk_size=args.chunk_size,
        reset=args.reset,
        dumps=lambda r: json.dumps(r, ensure_ascii=False),
    )

    n = sum(1 for line in OUTPUT_JSONL.open(encoding="utf-8") if line.strip()) if OUTPUT_JSONL.exists() else 0
    print(f"\nDone. {n} documents -> {OUTPUT_JSONL}")


if __name__ == "__main__":
    main()

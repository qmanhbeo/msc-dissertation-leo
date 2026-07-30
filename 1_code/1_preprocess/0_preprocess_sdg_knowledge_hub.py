"""
Preprocess SDG Knowledge Hub corpus: extract multi-label texts for all 17 SDGs.

Input:  2_data/0_raw/sdg_knowledge_hub/sdg_knowledge_hub.csv  (9,172 rows, multi-label)
Output: 2_data/1_preprocessed/individual_sources/sdg_knowledge_hub/sdg_knowledge_hub_clean.jsonl

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
    python 1_code/1_preprocess/0_preprocess_sdg_knowledge_hub.py
"""

import argparse
import hashlib
import json
import logging
import re
import unicodedata
from pathlib import Path

import sys
import pandas as pd

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
ANALYSIS_DIR = CODE_ROOT / "7_main_analysis" / "0_shared"
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))
from model_utils import raw_dir, preprocessed_dir, individual_source_dir
from _resume import resumable_records

INPUT_FILE = raw_dir() / "sdg_knowledge_hub" / "sdg_knowledge_hub.csv"
OUTPUT_JSONL = individual_source_dir("sdg_knowledge_hub") / "sdg_knowledge_hub_clean.jsonl"
STATE_PATH = individual_source_dir("sdg_knowledge_hub") / "sdg_kh_state.json"
STATUS_DIR = individual_source_dir("sdg_knowledge_hub") / "metadata"

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

def read_records():
    df = pd.read_csv(INPUT_FILE)
    sdg_cols = [c for c in df.columns if c.startswith("SDG-")]
    for _, row in df.iterrows():
        yield row, sdg_cols


def transform(payload) -> dict | None:
    row, sdg_cols = payload
    text = clean_text(row.get("text", ""))
    if len(text.split()) < MIN_WORDS:
        return None
    active = [int(c.split("-")[1]) for c in sdg_cols if row[c] == 1]
    if not active:
        return None
    return {
        "id": hashlib.sha256(row["url"].encode()).hexdigest()[:16],
        "url": row["url"],
        "text": text,
        "sdgs": active,
        "word_count": len(text.split()),
        "source": "sdg_knowledge_hub",
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Preprocess SDG Knowledge Hub corpus (resume-safe).")
    p.add_argument("--input", default=str(INPUT_FILE))
    p.add_argument("--out-jsonl", default=str(OUTPUT_JSONL))
    p.add_argument("--state", default=str(STATE_PATH))
    p.add_argument("--status-dir", default=str(STATUS_DIR))
    p.add_argument("--chunk-size", type=int, default=5000)
    p.add_argument("--reset", action="store_true", help="Delete checkpoint + output and start fresh.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    global INPUT_FILE, OUTPUT_JSONL, STATE_PATH, STATUS_DIR
    INPUT_FILE = Path(args.input)
    OUTPUT_JSONL = Path(args.out_jsonl)
    STATE_PATH = Path(args.state)
    STATUS_DIR = Path(args.status_dir)

    resumable_records(
        stage="preprocess_sdg_knowledge_hub",
        read_records=read_records,
        transform=transform,
        out_path=OUTPUT_JSONL,
        state_path=STATE_PATH,
        status_dir=STATUS_DIR,
        chunk_size=args.chunk_size,
        reset=args.reset,
    )

    n = sum(1 for line in OUTPUT_JSONL.open(encoding="utf-8") if line.strip()) if OUTPUT_JSONL.exists() else 0
    log.info("%d rows written to %s", n, OUTPUT_JSONL)

if __name__ == "__main__":
    main()

"""
Preprocess OSDG corpus for use as SDG classification training signal.

Input:  2_data/0_raw/osdg/osdg_dataset.csv  (TSV, 43,025 rows)
Output: 2_data/1_preprocessed/individual_sources/osdg/osdg_clean.jsonl  — filtered, cleaned records

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
    python 1_code/1_preprocess/0_preprocess_osdg.py
"""

import argparse
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
from model_utils import raw_dir, preprocessed_dir, individual_source_dir
from _resume import resumable_records

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
INPUT_FILE = raw_dir() / "osdg" / "osdg_dataset.csv"
OUTPUT_JSONL = individual_source_dir("osdg") / "osdg_clean.jsonl"
STATE_PATH = individual_source_dir("osdg") / "osdg_state.json"
STATUS_DIR = individual_source_dir("osdg") / "metadata"

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
# Resume-safe driver
# ---------------------------------------------------------------------------
def read_records():
    with INPUT_FILE.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            yield row


def transform(row: dict) -> dict | None:
    agreement = float(row.get("agreement", 0))
    if agreement < AGREEMENT_THRESHOLD:
        return None
    text = clean_text(row.get("text", ""))
    if len(text.split()) < MIN_WORDS:
        return None
    return {
        "text_id": row.get("text_id", ""),
        "doi": row.get("doi", ""),
        "text": text,
        "sdgs": [int(row["sdg"])],
        "agreement": agreement,
        "word_count": len(text.split()),
        "source": "osdg",
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Preprocess OSDG corpus (resume-safe).")
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
        stage="preprocess_osdg",
        read_records=read_records,
        transform=transform,
        out_path=OUTPUT_JSONL,
        state_path=STATE_PATH,
        status_dir=STATUS_DIR,
        chunk_size=args.chunk_size,
        reset=args.reset,
    )

    if OUTPUT_JSONL.exists():
        sdg_counts = Counter(json.loads(line)["sdgs"][0]
                             for line in OUTPUT_JSONL.open(encoding="utf-8") if line.strip())
    else:
        sdg_counts = Counter()
    print(f"\nDone. {sum(sdg_counts.values())} rows written to {OUTPUT_JSONL}")
    print(f"  Agreement threshold: >= {AGREEMENT_THRESHOLD}")
    print(f"  SDGs covered: {sorted(sdg_counts.keys())}")


if __name__ == "__main__":
    main()

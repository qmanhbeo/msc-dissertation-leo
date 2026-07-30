"""
Preprocess SDG Benchmark corpus (expert-verified SDG-labeled texts).

Input:  2_data/0_raw/sdg_benchmark/benchmark.csv  (1,251 rows, label True/False)
Output: 2_data/1_preprocessed/individual_sources/sdg_benchmark/benchmark_clean.jsonl  — positive examples only

Filtering:
  - Keep only rows where label == True (expert-confirmed SDG relevance)
  - Drop rows with text shorter than MIN_WORDS

Role in pipeline:
  Higher-quality complement to OSDG — used to:
  1. Cross-validate SDG centroid embeddings built from OSDG
  2. Evaluate alignment scoring (ground truth for SDG relevance)
  3. Provide a held-out evaluation set if needed

Run from project root:
    python 1_code/1_preprocess/0_preprocess_sdg_benchmark.py
"""

import argparse
import json
import logging
import re
import unicodedata
from collections import Counter
from pathlib import Path

import sys

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
INPUT_FILE = raw_dir() / "sdg_benchmark" / "benchmark.csv"
OUTPUT_JSONL = individual_source_dir("sdg_benchmark") / "benchmark_clean.jsonl"
STATE_PATH = individual_source_dir("sdg_benchmark") / "sdg_benchmark_state.json"
STATUS_DIR = individual_source_dir("sdg_benchmark") / "metadata"

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
# Resume-safe driver
# ---------------------------------------------------------------------------
def read_records():
    with INPUT_FILE.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row


def transform(row: dict) -> dict | None:
    # Keep only expert-confirmed positive examples
    if row.get("label", "").strip().lower() != "true":
        return None
    text = clean_text(row.get("text", ""))
    if len(text.split()) < MIN_WORDS:
        return None
    return {
        "id": row.get("id", ""),
        "text": text,
        "sdgs": [int(row["sdg"])],
        "word_count": len(text.split()),
        "source": "benchmark",
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Preprocess SDG Benchmark corpus (resume-safe).")
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
        stage="preprocess_sdg_benchmark",
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
    print(f"  SDGs covered: {sorted(sdg_counts.keys())}")


if __name__ == "__main__":
    main()

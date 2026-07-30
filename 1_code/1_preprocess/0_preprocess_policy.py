"""
Preprocess policy documents into cleaned text for downstream segmentation.

Inputs:
  2_data/0_raw/policy_scrape/texts/*.txt
  2_data/0_raw/policy_manual/texts/*.txt

Outputs:
  2_data/1_preprocessed/policy_all/policy_scrape/policy_scrape_clean.jsonl
  2_data/1_preprocessed/policy_all/policy_manual/policy_manual_clean.jsonl

Cleaning:
  1. Normalise Unicode (NFKC, smart quotes/hyphens)
  2. Remove [PAGE BREAK] markers, standalone page numbers, pipe-prefixed artefacts
  3. Remove OCR duplicate lines
  4. Collapse multiple spaces and newlines

Run from project root:
    python 1_code/1_preprocess/0_preprocess_policy.py
"""

import argparse
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
from _resume import resumable_records


SOURCE_CONFIGS = [
    {
        "source_name": "policy_scrape",
        "input_dir": raw_dir() / "policy_scrape" / "texts",
        "output_jsonl": preprocessed_dir() / "policy_all" / "policy_scrape" / "policy_scrape_clean.jsonl",
        "state_path": preprocessed_dir() / "policy_all" / "policy_scrape" / "policy_scrape_state.json",
        "status_dir": preprocessed_dir() / "policy_all" / "policy_scrape" / "metadata",
    },
    {
        "source_name": "policy_manual",
        "input_dir": raw_dir() / "policy_manual" / "texts",
        "output_jsonl": preprocessed_dir() / "policy_all" / "policy_manual" / "policy_manual_clean.jsonl",
        "state_path": preprocessed_dir() / "policy_all" / "policy_manual" / "policy_manual_state.json",
        "status_dir": preprocessed_dir() / "policy_all" / "policy_manual" / "metadata",
    },
]


logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

_PAGE_BREAK = re.compile(r"\[PAGE BREAK\]", re.IGNORECASE)
_PAGE_NUMBER = re.compile(r"^\s*\d+\s*$", re.MULTILINE)
_PIPE_PREFIX = re.compile(r"^\|\s*\d+\s*$", re.MULTILINE)
_MULTI_SPACE = re.compile(r"[ \t]{2,}")
_MULTI_NL = re.compile(r"\n{3,}")


def discover_docs(input_dir: Path) -> dict[str, Path]:
    docs: dict[str, Path] = {}
    if not input_dir.exists():
        return docs
    for path in sorted(input_dir.glob("*.txt")):
        docs[path.stem] = path
    return docs


def remove_ocr_duplicates(text: str) -> str:
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        if cleaned and line.strip() and line.strip() == cleaned[-1].strip():
            continue
        cleaned.append(line)
    return "\n".join(cleaned)


def normalize_unicode(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    return text


def clean_document(raw: str) -> str:
    text = normalize_unicode(raw)
    text = _PAGE_BREAK.sub("\n\n", text)
    text = _PAGE_NUMBER.sub("", text)
    text = _PIPE_PREFIX.sub("", text)
    text = remove_ocr_duplicates(text)
    text = _MULTI_SPACE.sub(" ", text)
    text = _MULTI_NL.sub("\n\n", text)
    return text.strip()


def process_source(config: dict, reset: bool) -> None:
    source_name = config["source_name"]
    input_dir = Path(config["input_dir"])
    output_jsonl = Path(config["output_jsonl"])

    docs = discover_docs(input_dir)
    log.info("Source %s: %d documents", source_name, len(docs))

    def read_records():
        for doc_name, filepath in sorted(docs.items()):
            yield doc_name, filepath

    def transform(payload) -> dict | None:
        doc_name, filepath = payload
        raw = filepath.read_text(encoding="utf-8", errors="replace")
        cleaned = clean_document(raw)
        return {
            "id": doc_name,
            "text": cleaned,
            "source_doc": doc_name,
            "source": source_name,
        }

    resumable_records(
        stage=f"preprocess_policy_{source_name}",
        read_records=read_records,
        transform=transform,
        out_path=output_jsonl,
        state_path=Path(config["state_path"]),
        status_dir=Path(config["status_dir"]),
        chunk_size=2000,
        reset=reset,
    )

    n = sum(1 for line in output_jsonl.open(encoding="utf-8") if line.strip()) if output_jsonl.exists() else 0
    log.info("Wrote %d documents -> %s", n, output_jsonl)
    print(f"\nDone. {source_name}: {n} documents written to {output_jsonl}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Preprocess policy documents (resume-safe).")
    p.add_argument("--reset", action="store_true", help="Delete checkpoints + outputs and start fresh.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    for config in SOURCE_CONFIGS:
        process_source(config, args.reset)


if __name__ == "__main__":
    main()

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


SOURCE_CONFIGS = [
    {
        "source_name": "policy_scrape",
        "input_dir": raw_dir() / "policy_scrape" / "texts",
        "output_jsonl": preprocessed_dir() / "policy_all" / "policy_scrape" / "policy_scrape_clean.jsonl",
    },
    {
        "source_name": "policy_manual",
        "input_dir": raw_dir() / "policy_manual" / "texts",
        "output_jsonl": preprocessed_dir() / "policy_all" / "policy_manual" / "policy_manual_clean.jsonl",
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


def process_source(config: dict) -> None:
    source_name = config["source_name"]
    input_dir = Path(config["input_dir"])
    output_jsonl = Path(config["output_jsonl"])

    docs = discover_docs(input_dir)
    log.info("Source %s: %d documents", source_name, len(docs))

    records = []
    for doc_name, filepath in docs.items():
        raw = filepath.read_text(encoding="utf-8", errors="replace")
        cleaned = clean_document(raw)
        records.append({
            "id": doc_name,
            "text": cleaned,
            "source_doc": doc_name,
        })

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    log.info("Wrote %d documents -> %s", len(records), output_jsonl)
    print(f"\nDone. {source_name}: {len(records)} documents written to {output_jsonl}")
    for r in records:
        wc = len(r["text"].split())
        print(f"  {r['id']}: {wc} words")


def main() -> None:
    for config in SOURCE_CONFIGS:
        process_source(config)


if __name__ == "__main__":
    main()

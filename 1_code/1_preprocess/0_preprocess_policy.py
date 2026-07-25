"""
Preprocess policy documents into source-specific text segments for embedding.

Inputs:
  2_data/0_raw/policy_scrape/texts/*.txt
  2_data/0_raw/policy_manual/texts/*.txt

Outputs:
  2_data/1_preprocessed/policy_all/policy_scrape/policy_scrape_segments.jsonl
  2_data/1_preprocessed/policy_all/policy_scrape/policy_scrape_segments.csv
  2_data/1_preprocessed/policy_all/policy_manual/policy_manual_segments.jsonl
  2_data/1_preprocessed/policy_all/policy_manual/policy_manual_segments.csv

Segmentation strategy:
  1. Clean OCR artifacts and page-break markers
  2. Segment full document text using segment_text() (NLTK sent_tokenize +
     SentenceTransformer tokenizer budget, max_seq_length - 10)

Run from project root:
    python 1_code/1_preprocess/0_preprocess_policy.py
    python 1_code/1_preprocess/0_preprocess_policy.py --model all-MiniLM-L6-v2
"""

import argparse
import csv
import json
import logging
import re
import unicodedata
from pathlib import Path

import sys

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
SEGMENT_DIR = CODE_ROOT / "2_segment"
if str(SEGMENT_DIR) not in sys.path:
    sys.path.insert(0, str(SEGMENT_DIR))
ANALYSIS_DIR = CODE_ROOT / "7_main_analysis" / "0_shared"
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from sentence_transformers import SentenceTransformer

from segment_utils import segment_text
from model_utils import raw_dir, preprocessed_dir


SOURCE_CONFIGS = [
    {
        "source_name": "policy_scrape",
        "input_dir": raw_dir() / "policy_scrape" / "texts",
        "output_jsonl": preprocessed_dir() / "policy_all" / "policy_scrape" / "policy_scrape_segments.jsonl",
        "output_csv": preprocessed_dir() / "policy_all" / "policy_scrape" / "policy_scrape_segments.csv",
    },
    {
        "source_name": "policy_manual",
        "input_dir": raw_dir() / "policy_manual" / "texts",
        "output_jsonl": preprocessed_dir() / "policy_all" / "policy_manual" / "policy_manual_segments.jsonl",
        "output_csv": preprocessed_dir() / "policy_all" / "policy_manual" / "policy_manual_segments.csv",
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


def build_segments_for_docs(docs: dict[str, Path], model: SentenceTransformer) -> list[dict]:
    all_segments: list[dict] = []
    for doc_name, filepath in docs.items():
        log.info("Processing %s (%s)", doc_name, filepath)
        raw = filepath.read_text(encoding="utf-8", errors="replace")
        cleaned = clean_document(raw)
        segments = segment_text(cleaned, model)
        log.info("  %s -> %d segments", doc_name, len(segments))

        for index, text in enumerate(segments):
            all_segments.append(
                {
                    "segment_id": f"{doc_name}_{index:04d}",
                    "source_doc": doc_name,
                    "segment_index": index,
                    "text": text,
                    "word_count": len(text.split()),
                }
            )
    return all_segments


def write_segments(output_jsonl: Path, output_csv: Path, segments: list[dict]) -> None:
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as handle:
        for segment in segments:
            handle.write(json.dumps(segment) + "\n")
    log.info("Saved JSONL -> %s", output_jsonl)

    csv_fields = ["segment_id", "source_doc", "segment_index", "word_count", "text"]
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(segments)
    log.info("Saved CSV -> %s", output_csv)


def log_quality_summary(source_name: str, segments: list[dict]) -> None:
    if not segments:
        log.info("%s produced no segments", source_name)
        return
    word_counts = [segment["word_count"] for segment in segments]
    log.info(
        "%s total segments: %d | word count - min: %d median: %d max: %d",
        source_name,
        len(segments),
        min(word_counts),
        sorted(word_counts)[len(word_counts) // 2],
        max(word_counts),
    )


def process_source(config: dict[str, Path | str], model: SentenceTransformer) -> None:
    source_name = str(config["source_name"])
    input_dir = Path(config["input_dir"])
    output_jsonl = Path(config["output_jsonl"])
    output_csv = Path(config["output_csv"])

    docs = discover_docs(input_dir)
    log.info("Source %s: %d documents", source_name, len(docs))
    segments = build_segments_for_docs(docs, model)
    log_quality_summary(source_name, segments)
    write_segments(output_jsonl, output_csv, segments)

    print(f"\nDone. {source_name}: {len(segments)} segments written to {output_jsonl}")
    for doc_name in docs:
        count = sum(1 for segment in segments if segment["source_doc"] == doc_name)
        print(f"  {doc_name}: {count} segments")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess policy documents using token-count-aware segmentation."
    )
    parser.add_argument(
        "--model", default="all-mpnet-base-v2",
        help="Sentence-transformer model for tokenizer (default: %(default)s).",
    )
    args = parser.parse_args()

    log.info("Loading model: %s", args.model)
    model = SentenceTransformer(args.model)
    log.info("Max sequence length: %d", model.max_seq_length)

    for config in SOURCE_CONFIGS:
        process_source(config, model)


if __name__ == "__main__":
    main()

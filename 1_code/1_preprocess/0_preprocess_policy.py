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
  2. Split into paragraphs (blank-line delimited)
  3. Merge short paragraphs into windows of TARGET_WORDS +/- tolerance
  4. Each segment has a source label, sequential ID, and word count

Run from project root:
    python 1_code/1_preprocess/policy/0_preprocess_policy.py
"""

import csv
import json
import logging
import re
import unicodedata
from pathlib import Path


SOURCE_CONFIGS = [
    {
        "source_name": "policy_scrape",
        "input_dir": Path("2_data/0_raw/policy_scrape/texts"),
        "output_jsonl": Path("2_data/1_preprocessed/policy_all/policy_scrape/policy_scrape_segments.jsonl"),
        "output_csv": Path("2_data/1_preprocessed/policy_all/policy_scrape/policy_scrape_segments.csv"),
    },
    {
        "source_name": "policy_manual",
        "input_dir": Path("2_data/0_raw/policy_manual/texts"),
        "output_jsonl": Path("2_data/1_preprocessed/policy_all/policy_manual/policy_manual_segments.jsonl"),
        "output_csv": Path("2_data/1_preprocessed/policy_all/policy_manual/policy_manual_segments.csv"),
    },
]

TARGET_WORDS = 150
MAX_WORDS = 300


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


def split_paragraphs(text: str) -> list[str]:
    paras = [paragraph.strip() for paragraph in re.split(r"\n\s*\n", text)]
    return [paragraph for paragraph in paras if len(paragraph.split()) >= 5]


def split_long_paragraph(text: str, max_words: int) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    segments: list[str] = []
    current: list[str] = []
    count = 0
    for sentence in sentences:
        word_count = len(sentence.split())
        if count + word_count > max_words and current:
            segments.append(" ".join(current))
            current = [sentence]
            count = word_count
        else:
            current.append(sentence)
            count += word_count
    if current:
        segments.append(" ".join(current))
    return segments


def merge_paragraphs(paragraphs: list[str], target_words: int, max_words: int) -> list[str]:
    expanded: list[str] = []
    for paragraph in paragraphs:
        if len(paragraph.split()) > max_words:
            expanded.extend(split_long_paragraph(paragraph, max_words))
        else:
            expanded.append(paragraph)

    segments: list[str] = []
    current_parts: list[str] = []
    current_count = 0
    for paragraph in expanded:
        word_count = len(paragraph.split())
        if current_count + word_count > target_words and current_parts:
            segments.append(" ".join(current_parts))
            current_parts = [paragraph]
            current_count = word_count
        else:
            current_parts.append(paragraph)
            current_count += word_count
    if current_parts:
        segments.append(" ".join(current_parts))
    return segments


def build_segments_for_docs(docs: dict[str, Path]) -> list[dict]:
    all_segments: list[dict] = []
    for doc_name, filepath in docs.items():
        log.info("Processing %s (%s)", doc_name, filepath)
        raw = filepath.read_text(encoding="utf-8", errors="replace")
        cleaned = clean_document(raw)
        paragraphs = split_paragraphs(cleaned)
        segments = merge_paragraphs(paragraphs, TARGET_WORDS, MAX_WORDS)
        log.info("  %s -> %d paragraphs -> %d segments", doc_name, len(paragraphs), len(segments))

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


def process_source(config: dict[str, Path | str]) -> None:
    source_name = str(config["source_name"])
    input_dir = Path(config["input_dir"])
    output_jsonl = Path(config["output_jsonl"])
    output_csv = Path(config["output_csv"])

    docs = discover_docs(input_dir)
    log.info("Source %s: %d documents", source_name, len(docs))
    segments = build_segments_for_docs(docs)
    log_quality_summary(source_name, segments)
    write_segments(output_jsonl, output_csv, segments)

    print(f"\nDone. {source_name}: {len(segments)} segments written to {output_jsonl}")
    for doc_name in docs:
        count = sum(1 for segment in segments if segment["source_doc"] == doc_name)
        print(f"  {doc_name}: {count} segments")


def main() -> None:
    for config in SOURCE_CONFIGS:
        process_source(config)


if __name__ == "__main__":
    main()

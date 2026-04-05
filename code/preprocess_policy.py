"""
Preprocess all policy documents into text chunks for embedding.

Input:  data/un_sdg/texts/*.txt        (original 2 UN docs)
        data/policy_expanded/texts/*.txt  (expanded corpus: 11 docs)

Output: data/un_sdg/policy_chunks.jsonl  — one chunk per line
        data/un_sdg/policy_chunks.csv    — flat CSV for inspection

Chunking strategy:
  1. Clean OCR artifacts and page-break markers
  2. Split into paragraphs (blank-line delimited)
  3. Merge short paragraphs into windows of TARGET_WORDS ± tolerance
  4. Each chunk has a source label, sequential ID, and word count

Run from project root:
    python code/preprocess_policy.py
"""

import csv
import json
import logging
import re
import unicodedata
from pathlib import Path

# ---------------------------------------------------------------------------
# Config — discover all text files across both directories dynamically
# ---------------------------------------------------------------------------
TEXT_DIRS = [
    Path("data/un_sdg/texts"),
    Path("data/policy_expanded/texts"),
]


def discover_docs() -> dict[str, Path]:
    """Return {stem: path} for every .txt in TEXT_DIRS (later dirs win on name clash)."""
    docs = {}
    for d in TEXT_DIRS:
        if d.exists():
            for p in sorted(d.glob("*.txt")):
                docs[p.stem] = p
    return docs


DOCS = discover_docs()
OUTPUT_JSONL = Path("data/un_sdg/policy_chunks.jsonl")
OUTPUT_CSV = Path("data/un_sdg/policy_chunks.csv")

# Target chunk size in words; chunks will be merged until they exceed this
TARGET_WORDS = 150
# Hard cap — split any single paragraph longer than this
MAX_WORDS = 300

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cleaning helpers
# ---------------------------------------------------------------------------
_PAGE_BREAK = re.compile(r"\[PAGE BREAK\]", re.IGNORECASE)
_PAGE_NUMBER = re.compile(r"^\s*\d+\s*$", re.MULTILINE)   # lone page numbers
_PIPE_PREFIX = re.compile(r"^\|\s*\d+\s*$", re.MULTILINE) # "| 1" style page nums
_MULTI_SPACE = re.compile(r"[ \t]{2,}")
_MULTI_NL = re.compile(r"\n{3,}")

# OCR duplication: lines repeated consecutively (common in PDF-extracted text)
def remove_ocr_duplicates(text: str) -> str:
    """Remove consecutive duplicate lines (OCR artifact)."""
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
    """Full document-level cleaning before chunking."""
    text = normalize_unicode(raw)
    text = _PAGE_BREAK.sub("\n\n", text)
    text = _PAGE_NUMBER.sub("", text)
    text = _PIPE_PREFIX.sub("", text)
    text = remove_ocr_duplicates(text)
    text = _MULTI_SPACE.sub(" ", text)
    text = _MULTI_NL.sub("\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Chunking helpers
# ---------------------------------------------------------------------------
def split_paragraphs(text: str) -> list[str]:
    """Split on blank lines; strip and discard very short fragments."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text)]
    return [p for p in paras if len(p.split()) >= 5]


def split_long_paragraph(text: str, max_words: int) -> list[str]:
    """Split a long paragraph at sentence boundaries to stay under max_words."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks, current, count = [], [], 0
    for sent in sentences:
        wc = len(sent.split())
        if count + wc > max_words and current:
            chunks.append(" ".join(current))
            current, count = [sent], wc
        else:
            current.append(sent)
            count += wc
    if current:
        chunks.append(" ".join(current))
    return chunks


def merge_paragraphs(paras: list[str], target: int, max_words: int) -> list[str]:
    """
    Greedily merge consecutive paragraphs until reaching target_words,
    then start a new chunk. Long paragraphs are split first.
    """
    # Pre-split paragraphs that exceed max_words
    expanded = []
    for p in paras:
        if len(p.split()) > max_words:
            expanded.extend(split_long_paragraph(p, max_words))
        else:
            expanded.append(p)

    chunks, current_parts, current_wc = [], [], 0
    for para in expanded:
        wc = len(para.split())
        if current_wc + wc > target and current_parts:
            chunks.append(" ".join(current_parts))
            current_parts, current_wc = [para], wc
        else:
            current_parts.append(para)
            current_wc += wc
    if current_parts:
        chunks.append(" ".join(current_parts))
    return chunks


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    all_chunks: list[dict] = []
    chunk_id = 0

    for doc_name, filepath in DOCS.items():
        log.info("Processing %s (%s)", doc_name, filepath)
        raw = filepath.read_text(encoding="utf-8", errors="replace")
        cleaned = clean_document(raw)
        paragraphs = split_paragraphs(cleaned)
        chunks = merge_paragraphs(paragraphs, TARGET_WORDS, MAX_WORDS)

        log.info(
            "  %s → %d paragraphs → %d chunks",
            doc_name, len(paragraphs), len(chunks),
        )

        for i, text in enumerate(chunks):
            all_chunks.append({
                "chunk_id": f"{doc_name}_{i:04d}",
                "source_doc": doc_name,
                "chunk_index": i,
                "text": text,
                "word_count": len(text.split()),
            })
            chunk_id += 1

    # Quality summary
    word_counts = [c["word_count"] for c in all_chunks]
    log.info(
        "Total chunks: %d  |  word count — min: %d  median: %d  max: %d",
        len(all_chunks),
        min(word_counts),
        sorted(word_counts)[len(word_counts) // 2],
        max(word_counts),
    )

    # Save JSONL
    OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_JSONL.open("w") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk) + "\n")
    log.info("Saved JSONL → %s", OUTPUT_JSONL)

    # Save CSV
    csv_fields = ["chunk_id", "source_doc", "chunk_index", "word_count", "text"]
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_chunks)
    log.info("Saved CSV  → %s", OUTPUT_CSV)

    print(f"\nDone. {len(all_chunks)} chunks written to {OUTPUT_JSONL}")
    for doc in DOCS:
        n = sum(1 for c in all_chunks if c["source_doc"] == doc)
        print(f"  {doc}: {n} chunks")


if __name__ == "__main__":
    main()

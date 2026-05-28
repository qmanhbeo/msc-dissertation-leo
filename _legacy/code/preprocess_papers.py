"""
Preprocess OpenAlex papers for downstream embedding and topic modeling.

Input:  data/openalex/papers.jsonl
Output: data/openalex/papers_clean.jsonl  — one cleaned record per line
        data/openalex/papers_clean.csv    — flat CSV for quick inspection

Cleaning steps:
  1. Drop papers with no abstract
  2. Normalize whitespace and Unicode in title + abstract
  3. Remove boilerplate patterns (copyright notices, URLs, email addresses)
  4. Build a combined text field: title + ". " + abstract (used for embeddings)
  5. Extract top-3 concept labels per paper
  6. Validate: log any remaining quality issues

Run from project root:
    python code/preprocess_papers.py
"""

import csv
import json
import logging
import re
import unicodedata
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
INPUT_FILE = Path("data/openalex/papers.jsonl")
OUTPUT_JSONL = Path("data/openalex/papers_clean.jsonl")
OUTPUT_CSV = Path("data/openalex/papers_clean.csv")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s  %(message)s",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Text cleaning helpers
# ---------------------------------------------------------------------------
# Patterns to strip before embedding
_BOILERPLATE = [
    re.compile(r"©\s*\d{4}.*", re.IGNORECASE),          # copyright lines
    re.compile(r"all rights reserved\.?", re.IGNORECASE),
    re.compile(r"https?://\S+"),                          # URLs
    re.compile(r"\S+@\S+\.\S+"),                          # email addresses
    re.compile(r"\b(doi|DOI):\s*\S+"),                    # inline DOIs
]

_MULTI_SPACE = re.compile(r"\s{2,}")
_LEADING_PUNCT = re.compile(r"^[\s\-–—,.:;]+")


def normalize_unicode(text: str) -> str:
    """Replace fancy quotes, dashes, and non-ASCII spaces with ASCII equivalents."""
    text = unicodedata.normalize("NFKC", text)
    # Smart quotes → straight quotes
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    # En/em dash → hyphen
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    return text


def clean_text(text: str) -> str:
    """Full cleaning pipeline for a single text field."""
    if not text:
        return ""
    text = normalize_unicode(text)
    for pattern in _BOILERPLATE:
        text = pattern.sub(" ", text)
    text = _LEADING_PUNCT.sub("", text)
    text = _MULTI_SPACE.sub(" ", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Per-paper processing
# ---------------------------------------------------------------------------
def process_paper(raw: dict) -> dict | None:
    """
    Clean a single paper record.

    Returns a cleaned dict, or None if the paper should be dropped.
    """
    title = clean_text(raw.get("title", ""))
    abstract = clean_text(raw.get("abstract", ""))

    # Drop if no usable abstract
    if not abstract:
        log.warning("Dropping paper (no abstract): %s", raw.get("openalex_id"))
        return None

    # Drop if abstract is suspiciously short (< 30 chars — probably truncated)
    if len(abstract) < 30:
        log.warning(
            "Dropping paper (abstract too short: %d chars): %s",
            len(abstract),
            raw.get("openalex_id"),
        )
        return None

    # Combined text field for embedding
    combined_text = f"{title}. {abstract}" if title else abstract

    # Top-3 concept labels (sorted by score descending)
    concepts_sorted = sorted(
        raw.get("concepts", []),
        key=lambda c: c.get("score", 0),
        reverse=True,
    )
    top_concepts = [c["display_name"] for c in concepts_sorted[:3] if c.get("display_name")]

    return {
        "openalex_id": raw.get("openalex_id", ""),
        "title": title,
        "abstract": abstract,
        "combined_text": combined_text,
        "doi": raw.get("doi", ""),
        "publication_year": raw.get("publication_year"),
        "cited_by_count": raw.get("cited_by_count", 0),
        "author_count": raw.get("author_count", 0),
        "top_concepts": top_concepts,
        "top_concepts_str": "; ".join(top_concepts),  # flat string for CSV
        "source_url": raw.get("source_url", ""),
        "abstract_word_count": len(abstract.split()),
    }


# ---------------------------------------------------------------------------
# CSV columns (flat subset for quick inspection)
# ---------------------------------------------------------------------------
CSV_FIELDS = [
    "openalex_id",
    "publication_year",
    "cited_by_count",
    "author_count",
    "abstract_word_count",
    "top_concepts_str",
    "title",
    "abstract",
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    log.info("Loading %s", INPUT_FILE)

    raw_papers: list[dict] = []
    with INPUT_FILE.open() as f:
        for line in f:
            line = line.strip()
            if line:
                raw_papers.append(json.loads(line))

    log.info("Loaded %d raw papers", len(raw_papers))

    # Process and filter
    cleaned: list[dict] = []
    for raw in raw_papers:
        result = process_paper(raw)
        if result is not None:
            cleaned.append(result)

    n_dropped = len(raw_papers) - len(cleaned)
    log.info("Kept %d papers  |  Dropped %d", len(cleaned), n_dropped)

    # Quality summary
    word_counts = [p["abstract_word_count"] for p in cleaned]
    if word_counts:
        log.info(
            "Abstract length — min: %d  median: %d  max: %d words",
            min(word_counts),
            sorted(word_counts)[len(word_counts) // 2],
            max(word_counts),
        )

    # Save JSONL
    OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_JSONL.open("w") as f:
        for paper in cleaned:
            f.write(json.dumps(paper) + "\n")
    log.info("Saved JSONL → %s", OUTPUT_JSONL)

    # Save CSV
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(cleaned)
    log.info("Saved CSV  → %s", OUTPUT_CSV)

    print(f"\nDone. {len(cleaned)} clean papers written to {OUTPUT_JSONL}")


if __name__ == "__main__":
    main()

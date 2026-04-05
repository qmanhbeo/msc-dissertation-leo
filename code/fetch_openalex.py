"""
Fetch academic papers from OpenAlex API on AI + Sustainable Development.

Runs multiple search queries, paginates fully, deduplicates by openalex_id.
- Year range: 2018-2025
- Saves: title, abstract, DOI, year, concepts, cited-by-count

Output: data/openalex/papers.jsonl (one JSON per line, deduplicated)
        data/openalex/metadata.json (fetch metadata)
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Generator

import requests
from tqdm import tqdm

# Configuration
OPENALEX_BASE_URL = "https://api.openalex.org/works"
OUTPUT_DIR = Path("data/openalex")
PAPERS_FILE = OUTPUT_DIR / "papers.jsonl"
METADATA_FILE = OUTPUT_DIR / "metadata.json"

# User email for polite pool (faster rate limits)
USER_EMAIL = "dissertation@example.com"

# Multiple queries to diversify the corpus
QUERIES = [
    "artificial intelligence sustainable development",
    "machine learning sustainable development goals",
    "deep learning sustainability",
    "artificial intelligence SDG",
]

YEAR_FILTER = "publication_year:>2017,publication_year:<2026"


def reconstruct_abstract(abstract_inverted_index: dict) -> str:
    """Reconstruct plain text abstract from OpenAlex inverted index format."""
    if not abstract_inverted_index:
        return ""

    # Create a position->word map
    position_to_word = {}
    for word, positions in abstract_inverted_index.items():
        for pos in positions:
            position_to_word[pos] = word

    # Sort by position and join
    if not position_to_word:
        return ""

    max_pos = max(position_to_word.keys())
    words = [position_to_word.get(i, "") for i in range(max_pos + 1)]
    return " ".join(filter(None, words))


def fetch_papers(query: str) -> Generator[dict, None, None]:
    """
    Generator to fetch all papers for a given query string using cursor pagination.
    Yields one paper dict at a time.
    """
    params = {
        "search": query,
        "sort": "publication_year:desc",
        "per-page": 200,
        "mailto": USER_EMAIL,
        "filter": YEAR_FILTER,
        "cursor": "*",
    }

    page = 0
    while True:
        page += 1
        response = requests.get(OPENALEX_BASE_URL, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        results = data.get("results", [])
        meta = data.get("meta", {})

        for paper in results:
            yield paper

        next_cursor = meta.get("next_cursor")
        if not next_cursor or not results:
            break

        params["cursor"] = next_cursor


def extract_paper(paper: dict) -> dict:
    """Extract and return relevant fields from a raw OpenAlex paper record."""
    abstract_text = reconstruct_abstract(
        paper.get("abstract_inverted_index", {})
    )
    # Fall back to plain abstract field if inverted index is absent
    if not abstract_text:
        abstract_text = paper.get("abstract", "") or ""

    return {
        "openalex_id": paper.get("id", ""),
        "title": paper.get("title", ""),
        "abstract": abstract_text,
        "doi": paper.get("doi", ""),
        "publication_year": paper.get("publication_year"),
        "cited_by_count": paper.get("cited_by_count", 0),
        "concepts": [
            {
                "id": c.get("id"),
                "display_name": c.get("display_name"),
                "score": c.get("score")
            }
            for c in paper.get("concepts", [])[:10]
        ],
        "author_count": len(paper.get("authorships", [])),
        "source_url": paper.get("primary_location", {}).get("landing_page_url", ""),
    }


def main():
    """Main fetch and save pipeline — runs all queries, deduplicates, saves."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print("OpenAlex Paper Fetcher (multi-query, deduplicated)")
    print(f"{'='*70}")
    print(f"Queries: {len(QUERIES)}")
    print(f"Year range: 2018-2025")
    print(f"Output: {PAPERS_FILE}")
    print(f"{'='*70}\n")

    start_time = datetime.now()
    seen_ids: set[str] = set()
    all_records: list[dict] = []
    query_counts: dict[str, int] = {}

    for query in QUERIES:
        print(f"\nQuery: \"{query}\"")
        count_this_query = 0
        new_this_query = 0

        for paper in tqdm(fetch_papers(query), desc=f"  Fetching", unit=" papers"):
            record = extract_paper(paper)
            count_this_query += 1
            if record["openalex_id"] and record["openalex_id"] not in seen_ids:
                seen_ids.add(record["openalex_id"])
                all_records.append(record)
                new_this_query += 1

        query_counts[query] = new_this_query
        print(f"  → {count_this_query} fetched, {new_this_query} new after dedup")

    # Write deduplicated JSONL
    with open(PAPERS_FILE, "w") as f:
        for record in all_records:
            f.write(json.dumps(record) + "\n")

    elapsed = datetime.now() - start_time
    file_size_mb = PAPERS_FILE.stat().st_size / (1024 * 1024)

    metadata = {
        "source": "OpenAlex API",
        "url": OPENALEX_BASE_URL,
        "queries": QUERIES,
        "year_range": [2018, 2025],
        "fetched_at": start_time.isoformat(),
        "elapsed_seconds": elapsed.total_seconds(),
        "total_papers": len(all_records),
        "papers_per_query": query_counts,
        "output_file": str(PAPERS_FILE),
        "file_size_mb": round(file_size_mb, 2),
    }

    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'='*70}")
    print(f"✓ Total unique papers: {len(all_records)}")
    print(f"✓ File size: {file_size_mb:.2f} MB")
    print(f"✓ Time elapsed: {elapsed.total_seconds():.1f}s")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()

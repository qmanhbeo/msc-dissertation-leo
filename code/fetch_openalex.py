"""
Fetch academic papers from OpenAlex API on AI + Sustainable Development.

Uses cursor-based pagination to retrieve papers matching:
- Concepts: "sustainable development" AND "artificial intelligence"
- Year range: 2018-2025
- Saves: title, abstract, DOI, year, concepts, cited-by-count

Output: data/openalex/papers.jsonl (one JSON per line)
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

# Query parameters
QUERY_PARAMS = {
    "filter": "concepts.id:C15744967,concepts.id:C167923496",  # AI + sustainable dev
    "sort": "publication_year:desc",
    "per-page": 100,
    "mailto": USER_EMAIL,
}

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


def fetch_papers(cursor: str = None) -> Generator[tuple, None, None]:
    """
    Generator to fetch papers with cursor-based pagination.
    Yields: (paper_data, next_cursor)
    """
    params = QUERY_PARAMS.copy()
    params["filter"] = f"{QUERY_PARAMS['filter']},{YEAR_FILTER}"

    if cursor:
        params["cursor"] = cursor

    while True:
        print(f"Fetching page (cursor: {cursor or 'initial'})...")
        response = requests.get(OPENALEX_BASE_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        results = data.get("results", [])
        meta = data.get("meta", {})

        for paper in results:
            yield paper

        # Check for next page
        next_cursor = meta.get("next_cursor")
        if not next_cursor or not results:
            break

        cursor = next_cursor
        params["cursor"] = cursor


def save_paper(paper: dict, file_handle) -> None:
    """Extract and save relevant fields from a paper record."""
    abstract_text = reconstruct_abstract(
        paper.get("abstract_inverted_index", {})
    )

    record = {
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
            for c in paper.get("concepts", [])[:10]  # Top 10 concepts
        ],
        "author_count": len(paper.get("authorships", [])),
        "source_url": paper.get("primary_location", {}).get("landing_page_url", ""),
    }

    file_handle.write(json.dumps(record) + "\n")
    return record


def main():
    """Main fetch and save pipeline."""
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print("OpenAlex Paper Fetcher")
    print(f"{'='*70}")
    print(f"Query: AI + Sustainable Development (2018-2025)")
    print(f"Output: {PAPERS_FILE}")
    print(f"{'='*70}\n")

    # Count and save papers
    paper_count = 0
    start_time = datetime.now()

    try:
        with open(PAPERS_FILE, "w") as f:
            for paper in tqdm(fetch_papers(), desc="Papers", unit=" papers"):
                save_paper(paper, f)
                paper_count += 1

        elapsed = datetime.now() - start_time
        file_size_mb = PAPERS_FILE.stat().st_size / (1024 * 1024)

        # Save metadata
        metadata = {
            "source": "OpenAlex API",
            "url": OPENALEX_BASE_URL,
            "query": {
                "concepts": ["AI", "Sustainable Development"],
                "year_range": [2018, 2025],
            },
            "fetched_at": start_time.isoformat(),
            "elapsed_seconds": elapsed.total_seconds(),
            "total_papers": paper_count,
            "output_file": str(PAPERS_FILE),
            "file_size_mb": round(file_size_mb, 2),
        }

        with open(METADATA_FILE, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"\n{'='*70}")
        print(f"✓ Successfully fetched {paper_count} papers")
        print(f"✓ File size: {file_size_mb:.2f} MB")
        print(f"✓ Time elapsed: {elapsed.total_seconds():.1f}s")
        print(f"✓ Metadata saved to {METADATA_FILE}")
        print(f"{'='*70}\n")

    except Exception as e:
        print(f"\n✗ Error during fetch: {e}")
        raise


if __name__ == "__main__":
    main()

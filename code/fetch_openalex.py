"""
Fetch academic papers from OpenAlex API on AI + Sustainable Development.

Uses OpenAlex's native SDG classification combined with AI/ML text search.
Fetches sequentially, saves incrementally every K new papers.
Safe to interrupt and resume — seen IDs are persisted.

Output:
  - data/openalex/papers_sdg{{N}}.jsonl  — papers tagged as SDG N
  - data/openalex/seen_ids.json           — seen openalex_ids
  - data/openalex/metadata.json           — fetch metadata

Run from project root:
    python code/fetch_openalex.py
"""

import json
import time
from datetime import datetime
from pathlib import Path

import requests

OPENALEX_BASE_URL = "https://api.openalex.org/works"
OUTPUT_DIR = Path("data/openalex")
METADATA_FILE = OUTPUT_DIR / "metadata.json"
SEEN_IDS_FILE = OUTPUT_DIR / "seen_ids.json"
USER_EMAIL = "dissertation@example.com"
SDG_BASE = "https://metadata.un.org/sdg"
SAVE_EVERY = 100
REQUEST_DELAY = 0.1

AI_TERMS = [
    "machine learning",
    "deep learning",
    "artificial intelligence",
    "neural network",
]

QUERIES = []
for sdg_num in range(1, 18):
    sdg_filter = f"sustainable_development_goals.id:{SDG_BASE}/{sdg_num}"
    filter_parts = [sdg_filter, "publication_year:>2017", "has_abstract:true"]
    for term in AI_TERMS:
        QUERIES.append({
            "term": term,
            "sdg": sdg_num,
            "filter_parts": filter_parts,
            "description": f"SDG {sdg_num} + \"{term}\"",
        })


def reconstruct_abstract(abstract_inverted_index: dict) -> str:
    if not abstract_inverted_index:
        return ""
    position_to_word = {}
    for word, positions in abstract_inverted_index.items():
        for pos in positions:
            position_to_word[pos] = word
    if not position_to_word:
        return ""
    max_pos = max(position_to_word.keys())
    words = [position_to_word.get(i, "") for i in range(max_pos + 1)]
    return " ".join(filter(None, words))


def extract_paper(paper: dict) -> dict:
    abstract_text = reconstruct_abstract(paper.get("abstract_inverted_index", {}))
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
            {"id": c.get("id"), "display_name": c.get("display_name"), "score": c.get("score")}
            for c in paper.get("concepts", [])[:10]
        ],
        "author_count": len(paper.get("authorships", [])),
        "source_url": paper.get("primary_location", {}).get("landing_page_url", ""),
    }


def load_seen_ids() -> set[str]:
    if SEEN_IDS_FILE.exists():
        with SEEN_IDS_FILE.open() as f:
            return set(json.load(f))
    return set()


def save_seen_ids(seen_ids: set[str]) -> None:
    with SEEN_IDS_FILE.open("w") as f:
        json.dump(sorted(seen_ids), f)


def papers_file(sdg: int) -> Path:
    return OUTPUT_DIR / f"papers_sdg{sdg:02d}.jsonl"


def load_existing_count(sdg: int) -> int:
    path = papers_file(sdg)
    if not path.exists():
        return 0
    with path.open() as f:
        return sum(1 for line in f if line.strip())


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}", flush=True)
    print("OpenAlex Fetcher — AI/ML for SDGs", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"Queries: {len(QUERIES)} (17 SDGs × 4 AI terms), sequential", flush=True)
    print(f"Save: every {SAVE_EVERY} new papers", flush=True)
    print(f"{'='*60}\n", flush=True)

    start_time = datetime.now()
    seen_ids = load_seen_ids()
    sdg_counts = {sdg: load_existing_count(sdg) for sdg in range(1, 18)}
    total_new = sum(sdg_counts.values())

    print(f"Loaded {len(seen_ids):,} seen IDs", flush=True)
    print(f"Existing papers on disk: {total_new:,}\n", flush=True)

    query_results = []

    for q_idx, q in enumerate(QUERIES, 1):
        sdg = q["sdg"]
        desc = q["description"]
        out_path = papers_file(sdg)

        print(f"[{q_idx}/{len(QUERIES)}] {desc}...", flush=True)

        params = [
            ("search", q["term"]),
            ("per-page", "200"),
            ("mailto", USER_EMAIL),
            ("filter", ",".join(q["filter_parts"])),
            ("cursor", "*"),
        ]

        page_num = 0
        raw_total = 0
        local_new = 0
        buffer = []

        while True:
            page_num += 1
            response = requests.get(OPENALEX_BASE_URL, params=params, timeout=60)
            if response.status_code == 429:
                print(f"    Page {page_num}: rate limited, waiting 10s...", flush=True)
                time.sleep(10)
                continue
            response.raise_for_status()
            data = response.json()
            results = data.get("results", [])
            raw_total += len(results)

            for raw in results:
                record = extract_paper(raw)
                if not record["openalex_id"]:
                    continue
                if record["openalex_id"] in seen_ids:
                    continue
                seen_ids.add(record["openalex_id"])
                local_new += 1
                total_new += 1
                sdg_counts[sdg] += 1
                buffer.append(record)

                if len(buffer) >= SAVE_EVERY:
                    with out_path.open("a") as f:
                        for r in buffer:
                            f.write(json.dumps(r) + "\n")
                    buffer = []
                    save_seen_ids(seen_ids)
                    print(f"    → {total_new:,} saved", flush=True)

            print(f"    Page {page_num}: {raw_total} fetched, {local_new} new, {total_new:,} total", flush=True)

            meta = data.get("meta", {})
            next_cursor = meta.get("next_cursor")
            if not next_cursor or not results:
                break
            params[-1] = ("cursor", next_cursor)
            time.sleep(REQUEST_DELAY)

        if buffer:
            with out_path.open("a") as f:
                for r in buffer:
                    f.write(json.dumps(r) + "\n")
            save_seen_ids(seen_ids)
            print(f"    → {total_new:,} saved (final)", flush=True)

        query_results.append({
            "query_index": q_idx,
            "sdg": sdg,
            "term": q["term"],
            "raw_count": raw_total,
            "new_count": local_new,
        })

        print(f"  ✓ {local_new} new ({total_new:,} total)\n", flush=True)

    elapsed = datetime.now() - start_time
    total_size = sum(f.stat().st_size for f in OUTPUT_DIR.glob("papers_sdg*.jsonl"))

    combined_path = OUTPUT_DIR / "papers.jsonl"
    if combined_path.exists():
        combined_path.unlink()
    combined_count = 0
    with combined_path.open("w") as out:
        for sdg in range(1, 18):
            for pfile in sorted(OUTPUT_DIR.glob(f"papers_sdg{sgd:02d}.jsonl")):
                with pfile.open() as inp:
                    for line in inp:
                        if line.strip():
                            out.write(line)
                            combined_count += 1

    metadata = {
        "source": "OpenAlex API",
        "url": OPENALEX_BASE_URL,
        "fetched_at": datetime.now().isoformat(),
        "elapsed_seconds": round(elapsed.total_seconds(), 1),
        "total_unique_papers": combined_count,
        "total_size_mb": round(total_size / (1024 * 1024), 2),
        "year_range": [2018, 2025],
        "requires_abstract": True,
        "sdg_filter": "native OpenAlex sustainable_development_goals.id classification",
        "papers_per_sdg": {str(sdg): count for sdg, count in sdg_counts.items()},
        "query_results": query_results,
    }

    with METADATA_FILE.open("w") as f:
        json.dump(metadata, f, indent=2)

    print(f"{'='*60}", flush=True)
    print(f"Done! {combined_count:,} papers in {elapsed.total_seconds():.0f}s", flush=True)
    print(f"Size: {total_size / (1024*1024):.1f} MB", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()

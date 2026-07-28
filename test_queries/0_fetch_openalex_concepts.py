"""
Fetch academic papers from OpenAlex API on AI + Sustainable Development.

Uses OpenAlex's native SDG classification combined with AI/ML text search.
Processes one query at a time, sequentially. Pages within a query are fetched
sequentially (OpenAlex cursor pagination requires it), but we skip already-
completed queries and resume mid-query using progress.json.

Output:
  - 2_data/0_raw/openalex/papers_sdg{{N}}.jsonl  — papers tagged as SDG N
  - 2_data/0_raw/openalex/artifact/seen_ids.json   — seen openalex_ids
  - 2_data/0_raw/openalex/artifact/progress.json   — per-query page tracking
  - 2_data/0_raw/openalex/artifact/metadata.json   — fetch metadata

Run from project root:
    python 1_code/0_fetch/fetch_openalex.py
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path

import sys

from dotenv import load_dotenv
import requests

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
ANALYSIS_DIR = CODE_ROOT / "7_main_analysis" / "0_shared"
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))
from model_utils import raw_dir


class AllKeysExhaustedError(Exception):
    """Raised when all API keys have been rate-limited after max retries."""
    pass


load_dotenv()

OPENALEX_BASE_URL = "https://api.openalex.org/works"
OUTPUT_DIR = raw_dir() / "openalex"
ARTIFACT_DIR = OUTPUT_DIR / "artifact"
METADATA_FILE = ARTIFACT_DIR / "metadata.json"
SEEN_IDS_FILE = ARTIFACT_DIR / "seen_ids.json"
PROGRESS_FILE = ARTIFACT_DIR / "progress.json"


def require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


USER_EMAIL = ""
API_KEY = ""
RATE_LIMIT_USER_EMAIL = ""
RATE_LIMIT_API_KEY = ""
RATE_LIMIT_USER_EMAIL_2 = ""
RATE_LIMIT_API_KEY_2 = ""
RATE_LIMIT_USER_EMAIL_3 = ""
RATE_LIMIT_API_KEY_3 = ""


def _load_api_keys() -> None:
    """Lazily validate and bind API keys. Called on first use, not at import."""
    global USER_EMAIL, API_KEY
    global RATE_LIMIT_USER_EMAIL, RATE_LIMIT_API_KEY
    global RATE_LIMIT_USER_EMAIL_2, RATE_LIMIT_API_KEY_2
    global RATE_LIMIT_USER_EMAIL_3, RATE_LIMIT_API_KEY_3
    USER_EMAIL = require_env("OPENALEX_MAILTO")
    API_KEY = require_env("OPENALEX_API_KEY")
    RATE_LIMIT_USER_EMAIL = os.environ.get("OPENALEX_RATE_LIMIT_MAILTO", "").strip()
    RATE_LIMIT_API_KEY = os.environ.get("OPENALEX_RATE_LIMIT_API_KEY", "").strip()
    RATE_LIMIT_USER_EMAIL_2 = os.environ.get("OPENALEX_RATE_LIMIT_MAILTO_2", "").strip()
    RATE_LIMIT_API_KEY_2 = os.environ.get("OPENALEX_RATE_LIMIT_API_KEY_2", "").strip()
    RATE_LIMIT_USER_EMAIL_3 = os.environ.get("OPENALEX_RATE_LIMIT_MAILTO_3", "").strip()
    RATE_LIMIT_API_KEY_3 = os.environ.get("OPENALEX_RATE_LIMIT_API_KEY_3", "").strip()


_keys_loaded = False


def ensure_api_keys() -> None:
    """Ensure API keys are loaded. Safe to call multiple times."""
    global _keys_loaded
    if not _keys_loaded:
        _load_api_keys()
        _keys_loaded = True
MAX_RETRIES = 10
SDG_BASE = "https://metadata.un.org/sdg"
SAVE_EVERY = 1000
PER_PAGE = 200
REQUEST_DELAY = 1.0

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
            "key": f"sdg{sdg_num}_{term}",
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


# --- Persistence ---

def load_seen_ids() -> set[str]:
    if SEEN_IDS_FILE.exists():
        with SEEN_IDS_FILE.open() as f:
            return set(json.load(f))
    return set()


def save_seen_ids(seen_ids: set[str]) -> None:
    with SEEN_IDS_FILE.open("w") as f:
        json.dump(sorted(seen_ids), f)


def load_progress() -> dict:
    if PROGRESS_FILE.exists():
        with PROGRESS_FILE.open() as f:
            return json.load(f)
    return {}


def save_progress(progress: dict) -> None:
    with PROGRESS_FILE.open("w") as f:
        json.dump(progress, f, indent=2)


def papers_file(sdg: int) -> Path:
    return OUTPUT_DIR / f"papers_sdg{sdg:02d}.jsonl"


def load_existing_count(sdg: int) -> int:
    path = papers_file(sdg)
    if not path.exists():
        return 0
    with path.open() as f:
        return sum(1 for line in f if line.strip())


# --- Fetch one query, all pages ---

def fetch_query(q: dict, seen_ids: set[str], progress: dict) -> dict:
    key = q["key"]
    sdg = q["sdg"]
    desc = q["description"]
    out_path = papers_file(sdg)

    # Already done?
    qprog = progress.get(key, {})
    if qprog.get("done"):
        print(f"  [SKIP] {desc} (complete)", flush=True)
        return {"new": 0, "raw": 0, "pages": 0}

    # Resume from saved cursor
    start_page = qprog.get("page", 0)
    cursor = qprog.get("cursor", "*")

    credential_sets = [
        (USER_EMAIL, API_KEY),
        (RATE_LIMIT_USER_EMAIL, RATE_LIMIT_API_KEY),
        (RATE_LIMIT_USER_EMAIL_2, RATE_LIMIT_API_KEY_2),
        (RATE_LIMIT_USER_EMAIL_3, RATE_LIMIT_API_KEY_3),
    ]
    credential_index = 0

    def build_params(current_cursor: str) -> dict:
        email, api_key = credential_sets[credential_index]
        params = {
            "search": q["term"],
            "per-page": str(PER_PAGE),
            "mailto": email,
            "filter": ",".join(q["filter_parts"]),
            "cursor": current_cursor,
        }
        if api_key:
            params["api_key"] = api_key
        return params

    params = build_params(cursor)

    page = start_page
    raw_total = 0
    local_new = 0
    buffer = []
    consecutive_failures = 0

    while True:
        page += 1
        try:
            resp = requests.get(OPENALEX_BASE_URL, params=params, timeout=30)
        except requests.RequestException as e:
            consecutive_failures += 1
            print(f"    [{desc}] p{page}: request failed ({e}), retry {consecutive_failures}/{MAX_RETRIES}", flush=True)
            if consecutive_failures >= MAX_RETRIES:
                raise AllKeysExhaustedError(f"Too many failed requests ({MAX_RETRIES}), all keys likely exhausted")
            time.sleep(3)
            page -= 1
            continue

        if resp.status_code == 429:
            consecutive_failures += 1
            print(f"    [{desc}] p{page}: rate limited, retry {consecutive_failures}/{MAX_RETRIES}", flush=True)
            if consecutive_failures >= MAX_RETRIES:
                raise AllKeysExhaustedError(f"Too many rate limits ({MAX_RETRIES}), all keys likely exhausted")
            time.sleep(3)
            if credential_index < len(credential_sets) - 1:
                credential_index += 1
            params = build_params(cursor)
            page -= 1  # retry same page
            continue

        consecutive_failures = 0  # Reset on success
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])
        raw_total += len(results)

        for raw in results:
            record = extract_paper(raw)
            oid = record["openalex_id"]
            if not oid or oid in seen_ids:
                continue
            seen_ids.add(oid)
            local_new += 1
            buffer.append(record)

        # Flush buffer periodically
        if len(buffer) >= SAVE_EVERY:
            with out_path.open("a") as f:
                for r in buffer:
                    f.write(json.dumps(r) + "\n")
            buffer = []
            save_seen_ids(seen_ids)

        # Save progress after each page
        next_cursor = data.get("meta", {}).get("next_cursor")
        done = not next_cursor or not results
        progress[key] = {"page": page, "cursor": next_cursor or cursor, "done": done}
        save_progress(progress)

        print(f"    [{desc}] p{page}: {len(results)} fetched, {local_new} new", flush=True)

        if done:
            break
        params["cursor"] = next_cursor
        time.sleep(REQUEST_DELAY)

    # Flush remaining
    if buffer:
        with out_path.open("a") as f:
            for r in buffer:
                f.write(json.dumps(r) + "\n")
        save_seen_ids(seen_ids)

    return {"new": local_new, "raw": raw_total, "pages": page - start_page}


def main() -> None:
    ensure_api_keys()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    seen_ids = load_seen_ids()
    progress = load_progress()

    done_count = sum(1 for v in progress.values() if v.get("done"))
    total_on_disk = sum(load_existing_count(sdg) for sdg in range(1, 18))

    print(f"\n{'='*60}", flush=True)
    print("OpenAlex Fetcher -- AI/ML for SDGs", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"Queries: {len(QUERIES)} ({done_count} done, {len(QUERIES) - done_count} remaining)", flush=True)
    print(f"Seen IDs: {len(seen_ids):,} | Papers on disk: {total_on_disk:,}", flush=True)
    print(f"{'='*60}\n", flush=True)

    start_time = datetime.now()
    total_new = 0
    all_results = []

    for i, q in enumerate(QUERIES, 1):
        print(f"[{i}/{len(QUERIES)}] {q['description']}", flush=True)
        try:
            result = fetch_query(q, seen_ids, progress)
        except AllKeysExhaustedError as e:
            print(f"\n{'='*60}", flush=True)
            print(f"STOPPED: {e}", flush=True)
            print(f"Progress saved. Run again to resume from where we stopped.", flush=True)
            print(f"{'='*60}", flush=True)
            save_seen_ids(seen_ids)
            save_progress(progress)
            return
        total_new += result["new"]
        all_results.append({"query": q["description"], **result})

    # Final saves
    save_seen_ids(seen_ids)
    save_progress(progress)

    # Combine into single file
    combined_path = OUTPUT_DIR / "papers.jsonl"
    if combined_path.exists():
        combined_path.unlink()
    combined_count = 0
    with combined_path.open("w") as out:
        for sdg in range(1, 18):
            pfile = papers_file(sdg)
            if pfile.exists():
                with pfile.open() as inp:
                    for line in inp:
                        if line.strip():
                            out.write(line)
                            combined_count += 1

    elapsed = datetime.now() - start_time
    total_size = sum(f.stat().st_size for f in OUTPUT_DIR.glob("papers_sdg*.jsonl"))

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
        "papers_per_sdg": {str(sdg): load_existing_count(sdg) for sdg in range(1, 18)},
        "query_results": [r for r in all_results if r["pages"] > 0],
    }

    with METADATA_FILE.open("w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'='*60}", flush=True)
    print(f"Done! {combined_count:,} total papers ({total_new:,} new) in {elapsed.total_seconds():.0f}s", flush=True)
    print(f"Size: {total_size / (1024*1024):.1f} MB", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()

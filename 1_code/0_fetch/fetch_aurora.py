"""
Fetch Aurora survey accepted papers from OpenAlex by DOI.

The Aurora dataset (Vanderfeesten et al., 2020) provides 5,695 research-domain
expert-validated SDG labels. This script fetches their titles and abstracts
from OpenAlex, then saves as a JSONL corpus ready for embedding.

Resumable: writes incrementally to aurora_texts.jsonl and tracks fetched DOIs
in aurora_fetched.log for crash recovery.

Uses the same credential rotation as fetch_openalex.py.

Output:
  2_data/1_preprocessed/aurora/aurora_texts.jsonl   — {text, sdg, doi, title, has_abstract}
  2_data/1_preprocessed/aurora/aurora_fetched.log   — one DOI per line (already fetched)
  2_data/1_preprocessed/aurora/aurora_manifest.json — {n_total, n_with_abstract, per_sdg_counts}
"""

import csv
import io
import json
import logging
import os
import random
import sys
import time
import zipfile
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

load_dotenv()

AURORA_ZIP = Path("2_data/0_raw/aurora/aurora.zip")
OUTPUT_DIR = Path("2_data/1_preprocessed/aurora")
OUTPUT_JSONL = OUTPUT_DIR / "aurora_texts.jsonl"
FETCHED_LOG = OUTPUT_DIR / "aurora_fetched.log"

API_BASE = "https://api.openalex.org/works/doi/{}"
MAX_RETRIES_PER_KEY = 3


def require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


# Load credential sets (same pattern as fetch_openalex.py)
def load_credentials() -> list[dict]:
    raw = [
        ("OPENALEX_MAILTO", "OPENALEX_API_KEY"),
        ("OPENALEX_RATE_LIMIT_MAILTO", "OPENALEX_RATE_LIMIT_API_KEY"),
        ("OPENALEX_RATE_LIMIT_MAILTO_2", "OPENALEX_RATE_LIMIT_API_KEY_2"),
        ("OPENALEX_RATE_LIMIT_MAILTO_3", "OPENALEX_RATE_LIMIT_API_KEY_3"),
    ]
    creds = []
    for email_var, key_var in raw:
        email = os.environ.get(email_var, "").strip()
        api_key = os.environ.get(key_var, "").strip()
        if email:
            creds.append({"mailto": email, "api_key": api_key})
    if not creds:
        raise RuntimeError("No OpenAlex credentials found in environment. Set OPENALEX_MAILTO at minimum.")
    return creds


def extract_dois_from_zip(zip_path: Path) -> dict[int, list[dict]]:
    """Return {sdg: [{doi}]} from accepted-papers CSVs."""
    z = zipfile.ZipFile(zip_path)
    result = defaultdict(list)
    for sdg in range(1, 18):
        fname = f"04-processed-data/SDG{sdg:02d}/sdg{sdg:02d}-SDG-survey-selected-publications-accepted.csv"
        try:
            text = z.read(fname).decode("utf-8")
        except KeyError:
            continue
        reader = csv.DictReader(io.StringIO(text))
        for row in reader:
            doi = row.get("doi", "").strip().lower()
            if not doi:
                continue
            result[sdg].append({"doi": doi, "sdg": sdg})
    return dict(result)


def reconstruct_abstract(inverted_index: dict | None) -> str | None:
    if inverted_index is None:
        return None
    words = []
    for word, positions in inverted_index.items():
        for pos in positions:
            words.append((pos, word))
    words.sort()
    return " ".join(w for _, w in words)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load credentials
    credential_sets = load_credentials()
    log.info("Loaded %d credential sets", len(credential_sets))

    # Step 1: Extract all DOIs
    log.info("Extracting DOIs from Aurora zip...")
    dois_by_sdg = extract_dois_from_zip(AURORA_ZIP)
    all_entries = []
    for sdg in sorted(dois_by_sdg):
        for entry in dois_by_sdg[sdg]:
            all_entries.append(entry)
    total = len(all_entries)
    log.info("Total entries: %d", total)

    # Step 2: Load already-fetched DOIs for resume
    fetched_so_far = set()
    if FETCHED_LOG.exists():
        with open(FETCHED_LOG) as f:
            fetched_so_far = {line.strip().lower() for line in f if line.strip()}
        log.info("Resume mode: %d DOIs already fetched", len(fetched_so_far))

    # Step 3: Fetch missing DOIs with credential rotation
    out_fh = open(OUTPUT_JSONL, "a", encoding="utf-8")
    fetched_fh = open(FETCHED_LOG, "a", encoding="utf-8")

    n_new = 0
    n_with_abstract = 0
    cred_idx = 0
    consecutive_failures = 0

    for i, entry in enumerate(all_entries):
        doi = entry["doi"]

        if doi in fetched_so_far:
            continue

        # Rotate creds every request for load balancing
        cred = credential_sets[cred_idx % len(credential_sets)]
        cred_idx += 1

        url = f"{API_BASE.format(doi)}?mailto={cred['mailto']}"
        if cred["api_key"]:
            url += f"&api_key={cred['api_key']}"

        success = False
        for attempt in range(MAX_RETRIES_PER_KEY):
            try:
                resp = requests.get(url, timeout=15)
                if resp.status_code == 200:
                    data = resp.json()
                    title = data.get("title", "") or ""
                    abstract = reconstruct_abstract(data.get("abstract_inverted_index"))
                    has_abstract = abstract is not None
                    combined_text = title
                    if abstract:
                        combined_text = title + " " + abstract

                    rec = {
                        "doi": doi,
                        "sdg": entry["sdg"],
                        "title": title,
                        "abstract": abstract,
                        "has_abstract": has_abstract,
                        "text": combined_text,
                        "source": "aurora",
                    }
                    out_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    out_fh.flush()

                    if has_abstract:
                        n_with_abstract += 1

                    fetched_so_far.add(doi)
                    fetched_fh.write(doi + "\n")
                    fetched_fh.flush()

                    n_new += 1
                    consecutive_failures = 0
                    success = True
                    break

                elif resp.status_code == 404:
                    # DOI not found — still mark as fetched to avoid retry
                    fetched_so_far.add(doi)
                    fetched_fh.write(doi + "\n")
                    fetched_fh.flush()
                    success = True
                    break

                elif resp.status_code == 429:
                    consecutive_failures += 1
                    wait = 2 ** attempt
                    log.warning("Rate limited (429) on DOI %s (attempt %d/%d), waiting %ds",
                                doi, attempt + 1, MAX_RETRIES_PER_KEY, wait)
                    time.sleep(wait)
                    continue
                else:
                    log.warning("HTTP %d for DOI %s", resp.status_code, doi)
                    break

            except requests.RequestException as e:
                consecutive_failures += 1
                log.warning("Request error for DOI %s (attempt %d/%d): %s",
                            doi, attempt + 1, MAX_RETRIES_PER_KEY, str(e)[:80])
                time.sleep(2 ** attempt)

        if not success:
            log.warning("Giving up on DOI %s after %d attempts", doi, MAX_RETRIES_PER_KEY)

        if n_new % 200 == 0 and n_new > 0:
            log.info("  Fetched %d new (total seen: %d/%d, %.0f%%)",
                     n_new, len(fetched_so_far), total, len(fetched_so_far) / total * 100)

        if consecutive_failures >= 50:
            log.error("Too many consecutive failures (%d). Stopping. Resume by re-running.",
                      consecutive_failures)
            break

        # Polite delay between requests
        time.sleep(0.05 + random.uniform(0, 0.02))

    out_fh.close()
    fetched_fh.close()

    # Step 4: Build manifest
    log.info("Building manifest from saved JSONL...")
    per_sdg_counts = defaultdict(lambda: {"total": 0, "with_abstract": 0})
    n_total = 0
    n_abstract = 0
    with open(OUTPUT_JSONL) as f:
        for line in f:
            r = json.loads(line)
            sdg = r["sdg"]
            per_sdg_counts[sdg]["total"] += 1
            n_total += 1
            if r.get("has_abstract"):
                per_sdg_counts[sdg]["with_abstract"] += 1
                n_abstract += 1

    missing = total - n_total
    log.info("Total saved: %d / %d total Aurora DOIs (%d missing, %.1f%%)",
             n_total, total, missing, missing / total * 100 if total else 0)
    log.info("With abstracts: %d / %d (%.0f%%)", n_abstract, n_total, n_abstract / n_total * 100 if n_total else 0)

    manifest = {
        "n_total": n_total,
        "n_missing_from_openalex": missing,
        "n_with_abstract": n_abstract,
        "n_without_abstract": n_total - n_abstract,
        "per_sdg_counts": {str(k): v for k, v in sorted(per_sdg_counts.items())},
        "source": "Aurora survey accepted papers (Vanderfeesten et al., 2020), fetched from OpenAlex",
    }
    manifest_path = OUTPUT_DIR / "aurora_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    log.info("Saved: %s", manifest_path)

    log.info("Done. Per-SDG breakdown:")
    log.info("  %-4s  %-8s  %-8s", "SDG", "total", "abstract")
    log.info("  " + "-" * 24)
    for sdg in sorted(per_sdg_counts):
        c = per_sdg_counts[sdg]
        log.info("  %3d  %5d    %5d", sdg, c["total"], c["with_abstract"])


if __name__ == "__main__":
    main()

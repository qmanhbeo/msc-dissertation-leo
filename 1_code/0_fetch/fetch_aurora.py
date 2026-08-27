"""
Fetch Aurora survey accepted papers from OpenAlex by DOI.

The Aurora dataset (Vanderfeesten et al., 2020) provides 5,695 research-domain
expert-validated SDG labels. This script auto-downloads the raw survey ZIP from
Zenodo (doi:10.5281/zenodo.3813230) if not already present, then fetches titles
and abstracts from OpenAlex, and saves as raw JSONL.

Deduplicates by DOI (one API call per unique DOI). The full multi-label SDG
mapping (sdgs: list[int]) is recovered from the ZIP before fetching, so every
record carries all its SDGs even though the API is called only once.

Resumable: writes incrementally and tracks fetched DOIs for crash recovery.

Uses the same credential rotation as fetch_openalex.py.

Output:
   2_data/0_raw/aurora/aurora.zip              — downloaded from Zenodo
   2_data/0_raw/aurora/aurora_raw.jsonl        — {doi, sdgs, title, abstract, has_abstract, text, source}
   2_data/0_raw/aurora/aurora_fetched.log      — one DOI per line (resume tracking)
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
from tqdm import tqdm

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
ANALYSIS_DIR = CODE_ROOT / "7_main_analysis" / "0_shared"
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))
from model_utils import raw_dir

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

load_dotenv()

AURORA_ZIP = raw_dir() / "aurora" / "aurora.zip"
AURORA_ZENODO_RECORD = "https://zenodo.org/api/records/3813230"
OUTPUT_DIR = raw_dir() / "aurora"
OUTPUT_JSONL = OUTPUT_DIR / "aurora_raw.jsonl"
FETCHED_LOG = OUTPUT_DIR / "aurora_fetched.log"

API_BASE = "https://api.openalex.org/works/doi/{}"
MAX_RETRIES_PER_KEY = 3


def require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


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


def download_aurora_zip() -> Path:
    if AURORA_ZIP.exists():
        log.info("aurora.zip already exists at %s", AURORA_ZIP)
        return AURORA_ZIP

    log.info("Fetching Zenodo record metadata from %s ...", AURORA_ZENODO_RECORD)
    resp = requests.get(AURORA_ZENODO_RECORD, timeout=15)
    resp.raise_for_status()
    record = resp.json()

    zip_url = None
    for f in record.get("files", []):
        if f["key"] == "aurora-sdg-survey-result-data-public.zip":
            zip_url = f["links"]["self"]
            break

    if not zip_url:
        log.warning("Could not find aurora-sdg-survey-result-data-public.zip on Zenodo record %s", AURORA_ZENODO_RECORD)
        raise FileNotFoundError(
            f"Aurora ZIP not available for download from Zenodo. "
            f"Place aurora.zip manually at {AURORA_ZIP}"
        )

    AURORA_ZIP.parent.mkdir(parents=True, exist_ok=True)
    log.info("Downloading %s -> %s (31 MB) ...", zip_url, AURORA_ZIP)
    with requests.get(zip_url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(AURORA_ZIP, "wb") as f:
            with tqdm(total=total, unit="B", unit_scale=True, desc="Downloading aurora.zip") as pbar:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))

    log.info("Downloaded aurora.zip (%d MB)", AURORA_ZIP.stat().st_size // (1024 * 1024))
    return AURORA_ZIP


def build_doi_to_sdgs(zip_path: Path) -> dict[str, list[int]]:
    """Build {doi: sorted[sdgs]} mapping from the Aurora ZIP's per-SDG CSVs.

    The ZIP contains one CSV per SDG. A paper accepted for multiple SDGs
    appears in multiple CSVs, so we collapse across CSVs to recover the
    full multi-label mapping.
    """
    doi_to_sdgs: dict[str, set[int]] = defaultdict(set)
    try:
        z = zipfile.ZipFile(zip_path)
    except FileNotFoundError:
        log.warning("Aurora ZIP not found at %s", zip_path)
        return {}

    for sdg in range(1, 18):
        fname = f"04-processed-data/SDG{sdg:02d}/sdg{sdg:02d}-SDG-survey-selected-publications-accepted.csv"
        try:
            text = z.read(fname).decode("utf-8")
        except KeyError:
            continue
        reader = csv.DictReader(io.StringIO(text))
        for row in reader:
            doi = row.get("doi", "").strip().lower()
            if doi:
                doi_to_sdgs[doi].add(sdg)

    z.close()
    return {doi: sorted(sdgs) for doi, sdgs in doi_to_sdgs.items()}


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

    credential_sets = load_credentials()
    log.info("Loaded %d credential sets", len(credential_sets))

    # Step 0: Download Aurora ZIP from Zenodo if not present
    aurora_zip = download_aurora_zip()

    # Step 1: Build {doi -> sdgs} mapping from ZIP
    log.info("Building DOI-to-SDGs mapping from Aurora ZIP...")
    doi_to_sdgs = build_doi_to_sdgs(aurora_zip)
    all_dois = sorted(doi_to_sdgs.keys())
    log.info("Total unique DOIs: %d", len(all_dois))

    multi_count = sum(1 for sdgs in doi_to_sdgs.values() if len(sdgs) > 1)
    log.info("Multi-label DOIs: %d (%.1f%%)", multi_count, multi_count / len(all_dois) * 100 if all_dois else 0)

    # Step 2: Load already-fetched DOIs for resume
    fetched_dois: set[str] = set()
    if FETCHED_LOG.exists():
        with open(FETCHED_LOG) as f:
            for line in f:
                line = line.strip().lower()
                if line:
                    fetched_dois.add(line)
        log.info("Resume mode: %d DOIs already fetched", len(fetched_dois))

    # Step 3: Fetch missing DOIs
    out_fh = open(OUTPUT_JSONL, "a", encoding="utf-8")
    fetched_fh = open(FETCHED_LOG, "a", encoding="utf-8")

    n_new = 0
    n_with_abstract = 0
    cred_idx = 0
    consecutive_failures = 0

    for i, doi in enumerate(all_dois):
        if doi in fetched_dois:
            continue

        sdgs = doi_to_sdgs.get(doi, [])

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
                    # Benchmark-corpus convention, deliberately DIFFERENT from
                    # the research corpus's f"{title}. {abstract}": text =
                    # "title abstract" with no ". " separator. Do not "align"
                    # this without re-embedding the benchmark corpus.
                    # NB: abstract == "" (empty inverted index) counts as
                    # has_abstract=True but yields title-only text; only None
                    # means the paper truly has no abstract field.
                    combined_text = title
                    if abstract:
                        combined_text = title + " " + abstract

                    rec = {
                        "doi": doi,
                        "sdgs": sdgs,
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

                    fetched_fh.write(f"{doi}\n")
                    fetched_fh.flush()

                    n_new += 1
                    consecutive_failures = 0
                    success = True
                    break

                elif resp.status_code == 404:
                    # Permanent tombstone: a 404 writes an empty record AND
                    # marks the DOI fetched, so resume never retries it.
                    # Recovering a transient 404 requires deleting the DOI
                    # from aurora_fetched.log AND its record from the jsonl.
                    rec = {
                        "doi": doi,
                        "sdgs": sdgs,
                        "title": "",
                        "abstract": None,
                        "has_abstract": False,
                        "text": "",
                        "source": "aurora",
                    }
                    out_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    out_fh.flush()

                    fetched_fh.write(f"{doi}\n")
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
            log.info("  Fetched %d new (total: %d/%d, %.0f%%)",
                     n_new, len(fetched_dois) + n_new, len(all_dois),
                     (len(fetched_dois) + n_new) / len(all_dois) * 100)

        if consecutive_failures >= 50:
            log.error("Too many consecutive failures (%d). Stopping. Resume by re-running.",
                      consecutive_failures)
            break

        time.sleep(0.05 + random.uniform(0, 0.02))

    out_fh.close()
    fetched_fh.close()

    log.info(
        "Done. %d new DOIs fetched (%d with abstract). Total: %d DOIs.",
        n_new, n_with_abstract, len(fetched_dois) + n_new,
    )


if __name__ == "__main__":
    main()

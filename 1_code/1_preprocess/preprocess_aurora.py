"""
Preprocess Aurora corpus: extract multi-label texts with full SDG recovery.

Input:  2_data/0_raw/aurora/aurora.zip  (SDG survey ZIP)
        2_data/1_preprocessed/aurora/aurora_texts.jsonl  (produced by fetch_aurora.py)

Output: 2_data/1_preprocessed/aurora/aurora_texts.jsonl  (cleaned, multi-label SDGs restored)

Multi-label recovery:
  The Aurora survey ZIP contains per-SDG CSV files. A paper accepted for
  multiple SDGs appears in multiple CSVs. This script collapses across CSVs
  to recover the full multi-label mapping, restoring SDGs lost during the
  fetch step's DOI-deduplication.

Single-label texts are filtered at MLP training time, not here.

Run from project root:
    python 1_code/0_fetch/fetch_aurora.py          # fetch from OpenAlex first
    python 1_code/1_preprocess/preprocess_aurora.py
"""

import csv
import io
import json
import logging
import re
import unicodedata
import zipfile
from collections import defaultdict
from pathlib import Path

N_SDG = 17

INPUT_FILE = Path("2_data/1_preprocessed/aurora/aurora_texts.jsonl")
AURORA_ZIP = Path("2_data/0_raw/aurora/aurora.zip")
OUTPUT_JSONL = Path("2_data/1_preprocessed/aurora/aurora_texts.jsonl")

MIN_WORDS = 20

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

_BOILERPLATE = [
    re.compile(r"©\s*\d{4}.*", re.IGNORECASE),
    re.compile(r"all rights reserved\.?", re.IGNORECASE),
    re.compile(r"https?://\S+"),
    re.compile(r"\S+@\S+\.\S+"),
    re.compile(r"\b(doi|DOI):\s*\S+"),
]
_MULTI_SPACE = re.compile(r"\s{2,}")

def normalize_unicode(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    return text

def clean_text(text: str) -> str:
    if not text:
        return ""
    text = normalize_unicode(text)
    for pattern in _BOILERPLATE:
        text = pattern.sub(" ", text)
    text = _MULTI_SPACE.sub(" ", text)
    return text.strip()

def extract_doi_to_sdgs(zip_path: Path) -> dict[str, set[int]]:
    doi_to_sdgs: dict[str, set[int]] = defaultdict(set)
    try:
        z = zipfile.ZipFile(zip_path)
    except FileNotFoundError:
        log.warning("Aurora ZIP not found at %s — no multi-label recovery possible", zip_path)
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
    return dict(doi_to_sdgs)

def main() -> None:
    if not INPUT_FILE.exists():
        log.error(
            "Input not found: %s\n"
            "  Run the fetch script first:\n"
            "    python 1_code/0_fetch/fetch_aurora.py",
            INPUT_FILE,
        )
        return

    doi_to_all_sdgs = extract_doi_to_sdgs(AURORA_ZIP)
    n_sdg_restored = 0
    n_restored_texts = 0

    log.info("Loading %s", INPUT_FILE)
    with INPUT_FILE.open(encoding="utf-8") as f:
        raw = [json.loads(line) for line in f if line.strip()]
    log.info("Loaded %d raw rows", len(raw))

    doi_to_raw: dict[str, list[dict]] = {}
    for r in raw:
        doi_to_raw.setdefault(r.get("doi", ""), []).append(r)

    kept, dropped_text = [], 0
    multi_count = 0

    for doi, records in doi_to_raw.items():
        best = max(records, key=lambda x: (x.get("has_abstract", False), x.get("text", "")))

        text = clean_text(best.get("text", ""))
        if len(text.split()) < MIN_WORDS:
            dropped_text += 1
            continue

        if doi in doi_to_all_sdgs:
            sdgs = sorted(s for s in doi_to_all_sdgs[doi] if 1 <= s <= N_SDG)
            fetched_sdgs = set()
            for r in records:
                if "sdgs" in r:
                    fetched_sdgs.update(int(s) for s in r["sdgs"] if 1 <= s <= N_SDG)
                elif "sdg" in r:
                    sdg_val = int(r["sdg"])
                    if 1 <= sdg_val <= N_SDG:
                        fetched_sdgs.add(sdg_val)
            restored = set(sdgs) - fetched_sdgs
            if restored:
                n_sdg_restored += len(restored)
                n_restored_texts += 1
        else:
            if "sdgs" in best:
                sdgs = [s for s in best["sdgs"] if 1 <= s <= N_SDG]
            elif "sdg" in best:
                sdg_val = int(best["sdg"])
                sdgs = [sdg_val] if 1 <= sdg_val <= N_SDG else []
            else:
                sdgs = []

        if not sdgs:
            continue

        if len(sdgs) > 1:
            multi_count += 1

        kept.append({
            "doi": doi,
            "sdgs": sdgs,
            "title": best.get("title", ""),
            "abstract": best.get("abstract", ""),
            "has_abstract": best.get("has_abstract", False),
            "text": text,
            "word_count": len(text.split()),
            "source": "aurora",
        })

    log.info(
        "Total: %d kept (%d multi-label)  |  Dropped (short text): %d  |  "
        "SDGs restored from ZIP: %d labels across %d texts  |  "
        "SDG distribution: %s",
        len(kept), multi_count, dropped_text,
        n_sdg_restored, n_restored_texts,
        dict(sorted({n: sum(1 for r in kept if len(r["sdgs"]) == n) for n in range(1, 18)}.items())),
    )

    OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_JSONL.open("w", encoding="utf-8") as f:
        for row in kept:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    log.info("Saved -> %s", OUTPUT_JSONL)

    print(f"\nDone. {len(kept)} rows written to {OUTPUT_JSONL}")

if __name__ == "__main__":
    main()

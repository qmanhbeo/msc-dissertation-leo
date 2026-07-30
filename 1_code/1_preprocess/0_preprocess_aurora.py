"""
Preprocess Aurora corpus: clean text and filter short entries.

Input:  2_data/0_raw/aurora/aurora_raw.jsonl  (from fetch_aurora.py, has sdgs: list[int])
        2_data/0_raw/aurora/aurora.zip        (cross-check SDG mapping)

Output: 2_data/1_preprocessed/aurora/aurora_texts.jsonl
        2_data/1_preprocessed/aurora/aurora_manifest.json

Single-label texts are filtered at MLP training time, not here.

Run from project root:
    python 1_code/0_fetch/fetch_aurora.py
    python 1_code/1_preprocess/0_preprocess_aurora.py
"""

import argparse
import csv
import io
import json
import logging
import re
import unicodedata
import zipfile
from collections import defaultdict
from pathlib import Path

import sys

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
ANALYSIS_DIR = CODE_ROOT / "7_main_analysis" / "0_shared"
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))
from model_utils import raw_dir, preprocessed_dir
from _resume import resumable_records

N_SDG = 17
INPUT_FILE = raw_dir() / "aurora" / "aurora_raw.jsonl"
AURORA_ZIP = raw_dir() / "aurora" / "aurora.zip"
OUTPUT_JSONL = preprocessed_dir() / "aurora" / "aurora_texts.jsonl"
OUTPUT_DIR = preprocessed_dir() / "aurora"
MANIFEST_PATH = OUTPUT_DIR / "aurora_manifest.json"
STATE_PATH = OUTPUT_DIR / "aurora_state.json"
STATUS_DIR = OUTPUT_DIR / "metadata"

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


def build_doi_to_sdgs_from_zip(zip_path: Path) -> dict[str, list[int]]:
    """Cross-check: build {doi: sdgs} from ZIP to catch any missing SDGs."""
    try:
        z = zipfile.ZipFile(zip_path)
    except FileNotFoundError:
        return {}
    doi_to_sdgs: dict[str, set[int]] = defaultdict(set)
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


def read_records():
    if not INPUT_FILE.exists():
        log.error("Input not found: %s\n  Run fetch_aurora.py first.", INPUT_FILE)
        return
    with INPUT_FILE.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def transform(raw):
    doi = raw.get("doi", "")
    sdgs = raw.get("sdgs", [])
    if not isinstance(sdgs, list):
        sdgs = [int(sdgs)] if sdgs else []
    sdgs = [s for s in sdgs if 1 <= s <= N_SDG]

    # Cross-check: if ZIP has more SDGs than the raw record, restore them
    if doi in _DOI_TO_ALL_SDGS:
        zip_sdgs = set(_DOI_TO_ALL_SDGS[doi])
        fetched_sdgs = set(sdgs)
        restored = zip_sdgs - fetched_sdgs
        if restored:
            sdgs = sorted(zip_sdgs)

    if not sdgs:
        return None

    text = raw.get("text", "") or ""
    if not text.strip():
        return None

    text = clean_text(text)
    if len(text.split()) < MIN_WORDS:
        return None

    return {
        "doi": doi,
        "sdgs": sdgs,
        "title": raw.get("title", ""),
        "abstract": raw.get("abstract", ""),
        "has_abstract": raw.get("has_abstract", False),
        "text": text,
        "word_count": len(text.split()),
        "source": "aurora",
    }


def finalize(out_path: Path) -> None:
    per_sdg_counts = defaultdict(lambda: {"total": 0, "with_abstract": 0})
    n_total = 0
    n_abstract = 0
    multi_count = 0
    with out_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            n_total += 1
            if r.get("has_abstract"):
                n_abstract += 1
            if len(r["sdgs"]) > 1:
                multi_count += 1
            for sdg in r["sdgs"]:
                per_sdg_counts[sdg]["total"] += 1
                if r.get("has_abstract"):
                    per_sdg_counts[sdg]["with_abstract"] += 1

    manifest = {
        "n_total": n_total,
        "n_with_abstract": n_abstract,
        "n_without_abstract": n_total - n_abstract,
        "n_multi_label": multi_count,
        "n_single_label": n_total - multi_count,
        "per_sdg_counts": {str(k): dict(v) for k, v in sorted(per_sdg_counts.items())},
    }
    with MANIFEST_PATH.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    log.info("Saved -> %s", MANIFEST_PATH)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Preprocess Aurora corpus (resume-safe).")
    p.add_argument("--input", default=str(INPUT_FILE))
    p.add_argument("--zip", default=str(AURORA_ZIP))
    p.add_argument("--out-jsonl", default=str(OUTPUT_JSONL))
    p.add_argument("--manifest", default=str(MANIFEST_PATH))
    p.add_argument("--state", default=str(STATE_PATH))
    p.add_argument("--status-dir", default=str(STATUS_DIR))
    p.add_argument("--chunk-size", type=int, default=5000)
    p.add_argument("--reset", action="store_true", help="Delete checkpoint + output and start fresh.")
    return p.parse_args()


def main() -> None:
    global INPUT_FILE, AURORA_ZIP, OUTPUT_JSONL, MANIFEST_PATH, STATE_PATH, STATUS_DIR, _DOI_TO_ALL_SDGS
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    INPUT_FILE = Path(args.input)
    AURORA_ZIP = Path(args.zip)
    OUTPUT_JSONL = Path(args.out_jsonl)
    MANIFEST_PATH = Path(args.manifest)
    STATE_PATH = Path(args.state)
    STATUS_DIR = Path(args.status_dir)

    _DOI_TO_ALL_SDGS = build_doi_to_sdgs_from_zip(AURORA_ZIP)

    resumable_records(
        stage="preprocess_aurora",
        read_records=read_records,
        transform=transform,
        out_path=OUTPUT_JSONL,
        state_path=STATE_PATH,
        status_dir=STATUS_DIR,
        chunk_size=args.chunk_size,
        reset=args.reset,
        finalize=finalize,
        dumps=lambda r: json.dumps(r, ensure_ascii=False),
    )

    n = sum(1 for line in OUTPUT_JSONL.open(encoding="utf-8") if line.strip()) if OUTPUT_JSONL.exists() else 0
    print(f"\nDone. {n} rows written to {OUTPUT_JSONL}")


if __name__ == "__main__":
    main()

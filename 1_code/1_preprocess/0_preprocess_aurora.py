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

N_SDG = 17
INPUT_FILE = raw_dir() / "aurora" / "aurora_raw.jsonl"
AURORA_ZIP = raw_dir() / "aurora" / "aurora.zip"
OUTPUT_JSONL = preprocessed_dir() / "aurora" / "aurora_texts.jsonl"
OUTPUT_DIR = preprocessed_dir() / "aurora"
MANIFEST_PATH = OUTPUT_DIR / "aurora_manifest.json"

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


def main() -> None:
    if not INPUT_FILE.exists():
        log.error("Input not found: %s\n  Run fetch_aurora.py first.", INPUT_FILE)
        return

    # Load raw records (already have sdgs: list[int])
    log.info("Loading %s", INPUT_FILE)
    with INPUT_FILE.open(encoding="utf-8") as f:
        raw = [json.loads(line) for line in f if line.strip()]
    log.info("Loaded %d raw rows", len(raw))

    # Cross-check SDGs against ZIP
    doi_to_all_sdgs = build_doi_to_sdgs_from_zip(AURORA_ZIP)
    n_sdg_restored = 0
    n_restored_texts = 0

    kept, dropped_no_text, dropped_text = [], 0, 0
    multi_count = 0

    for r in raw:
        doi = r.get("doi", "")
        sdgs = r.get("sdgs", [])
        if not isinstance(sdgs, list):
            sdgs = [int(sdgs)] if sdgs else []
        sdgs = [s for s in sdgs if 1 <= s <= N_SDG]

        # Cross-check: if ZIP has more SDGs than the raw record, restore them
        if doi in doi_to_all_sdgs:
            zip_sdgs = set(doi_to_all_sdgs[doi])
            fetched_sdgs = set(sdgs)
            restored = zip_sdgs - fetched_sdgs
            if restored:
                sdgs = sorted(zip_sdgs)
                n_sdg_restored += len(restored)
                n_restored_texts += 1

        if not sdgs:
            continue

        text = r.get("text", "") or ""
        if not text.strip():
            dropped_no_text += 1
            continue

        text = clean_text(text)
        if len(text.split()) < MIN_WORDS:
            dropped_text += 1
            continue

        if len(sdgs) > 1:
            multi_count += 1

        kept.append({
            "doi": doi,
            "sdgs": sdgs,
            "title": r.get("title", ""),
            "abstract": r.get("abstract", ""),
            "has_abstract": r.get("has_abstract", False),
            "text": text,
            "word_count": len(text.split()),
            "source": "aurora",
        })

    log.info(
        "Total: %d kept (%d multi-label)  |  Dropped (no text): %d  |  "
        "Dropped (short text): %d  |  SDGs restored from ZIP: %d labels across %d texts",
        len(kept), multi_count, dropped_no_text, dropped_text,
        n_sdg_restored, n_restored_texts,
    )

    OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_JSONL.open("w", encoding="utf-8") as f:
        for row in kept:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    log.info("Saved -> %s", OUTPUT_JSONL)

    # Build manifest
    per_sdg_counts = defaultdict(lambda: {"total": 0, "with_abstract": 0})
    n_total = 0
    n_abstract = 0
    for r in kept:
        n_total += 1
        if r.get("has_abstract"):
            n_abstract += 1
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

    print(f"\nDone. {len(kept)} rows written to {OUTPUT_JSONL}")


if __name__ == "__main__":
    main()

"""
Unified SDGi preprocessor — replaces both preprocess_sdgi_corpus.py and 0_integrate_sdgi.py.

Single path: English filter → text cleaning → token-count-aware segmentation → one output
that feeds both training (centroid building) and the policy-side merge.

Outputs model-specific segmentations (MPNet vs MiniLM have different max_seq_length).

Run from project root:
    python 1_code/1_preprocess/preprocess_sdgi_unified.py
    python 1_code/1_preprocess/preprocess_sdgi_unified.py --model all-MiniLM-L6-v2
"""

import argparse
import json
import logging
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
SEGMENT_DIR = CODE_ROOT / "2_segment"
if str(SEGMENT_DIR) not in sys.path:
    sys.path.insert(0, str(SEGMENT_DIR))
ANALYSIS_DIR = CODE_ROOT / "7_main_analysis" / "0_shared"
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from sentence_transformers import SentenceTransformer

from segment_utils import segment_text
from model_utils import raw_dir, segmented_dir_for_model

INPUT_PARQUET = raw_dir() / "sdgi_corpus" / "sdgi_corpus.parquet"
TARGET_LANGUAGE = "en"
MIN_WORDS = 20

_BOILERPLATE = [
    re.compile(r"©\s*\d{4}.*", re.IGNORECASE),
    re.compile(r"all rights reserved\.?", re.IGNORECASE),
    re.compile(r"https?://\S+"),
    re.compile(r"\S+@\S+\.\S+"),
    re.compile(r"\b(doi|DOI):\s*\S+"),
]
_MULTI_SPACE = re.compile(r"\s{2,}")

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


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


def sanitise_source_doc(value: str) -> str:
    return value.replace("/", "_").replace(" ", "_").replace("-", "_")


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified SDGi preprocessor.")
    parser.add_argument("--model", default="all-mpnet-base-v2",
                        help="Model name for token-count-aware segmentation")
    args = parser.parse_args()

    output_path = segmented_dir_for_model(args.model) / "sdgi.jsonl"

    log.info("Loading %s", INPUT_PARQUET)
    df = pd.read_parquet(INPUT_PARQUET)
    log.info("Loaded %d raw rows", len(df))

    # --- English filter ---
    def is_english(meta: object) -> bool:
        if isinstance(meta, dict):
            return meta.get("language", "") == TARGET_LANGUAGE
        return False

    n_before_lang = len(df)
    df = df[df["metadata"].apply(is_english)].copy()
    log.info("English rows: %d (dropped %d)", len(df), n_before_lang - len(df))

    # --- Load segmentation model ---
    log.info("Loading model: %s", args.model)
    model = SentenceTransformer(args.model)

    segments_out: list[dict] = []
    doc_counts: dict[str, int] = {}

    for idx, row in df.iterrows():
        meta = row["metadata"] if isinstance(row["metadata"], dict) else {}

        text = str(row.get("text", "") or "")
        text = clean_text(text)
        if len(text.split()) < MIN_WORDS:
            continue

        labels = row.get("labels")
        if isinstance(labels, np.ndarray):
            sdgs = sorted(int(l) for l in labels)
        elif isinstance(labels, list):
            sdgs = sorted(int(l) for l in labels)
        elif labels is not None:
            sdgs = [int(labels)]
        else:
            continue

        country = meta.get("country", "unknown").upper()
        doc_type = meta.get("type", "policy").upper()
        locality = meta.get("locality", "")
        year = meta.get("year")
        file_id = meta.get("file_id", "")

        source_doc = (
            f"sdgi_{country}_{doc_type}"
            + (f"_{locality}" if locality else "")
            + (f"_{year}" if year else "")
        )
        source_doc = sanitise_source_doc(source_doc)

        sub_texts = segment_text(text, model)
        if source_doc not in doc_counts:
            doc_counts[source_doc] = 0

        for sub_i, sub_text in enumerate(sub_texts):
            segments_out.append({
                "segment_id": f"sdgi_{idx:05d}_{sub_i}",
                "source_doc": source_doc,
                "segment_index": doc_counts[source_doc],
                "text": sub_text,
                "word_count": len(sub_text.split()),
                "sdgs": sdgs,
                "institution": f"{country} ({doc_type})",
                "year": year,
                "original_file": file_id,
                "country": country,
            })
            doc_counts[source_doc] += 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for seg in segments_out:
            f.write(json.dumps(seg) + "\n")

    log.info("Wrote %d segments → %s", len(segments_out), output_path)

    # --- Statistics ---
    n_multi = sum(1 for s in segments_out if len(s["sdgs"]) > 1)
    n_single = len(segments_out) - n_multi
    log.info(
        "Segments: %d total (%d single-label, %d multi-label) | %d source docs",
        len(segments_out), n_single, n_multi, len(doc_counts),
    )
    word_counts = [s["word_count"] for s in segments_out]
    if word_counts:
        sorted_wc = sorted(word_counts)
        log.info(
            "Segment word counts: min=%d median=%d max=%d",
            sorted_wc[0], sorted_wc[len(sorted_wc) // 2], sorted_wc[-1],
        )

    # SDG label distribution
    from collections import Counter
    sdg_counts: Counter = Counter()
    for s in segments_out:
        for sdg in s["sdgs"]:
            sdg_counts[sdg] += 1
    log.info("SDG label distribution: %s", dict(sorted(sdg_counts.items())))

    print(f"\nDone. {len(segments_out)} segments → {output_path}")


if __name__ == "__main__":
    main()

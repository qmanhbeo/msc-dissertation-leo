"""
Convert SDGi corpus (parquet) → policy_chunks.jsonl format.

Input:  data/raw/sdgi_corpus/sdgi_corpus.parquet
        (~5,880 text excerpts from Voluntary National Reviews and
        Voluntary Local Reviews submitted to the UN HLPF)

Output: data/preprocessed/policy_all/sdgi_corpus/sdgi_chunks.jsonl

These VNR/VLR texts are the gold standard for SDG policy language —
authored by national and local governments reporting on SDG implementation.

Run from project root:
    python code/preprocess/integrate_sdgi.py
"""

import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

INPUT_PARQUET = Path("data/raw/sdgi_corpus/sdgi_corpus.parquet")
OUTPUT_JSONL = Path("data/preprocessed/policy_all/sdgi_corpus/sdgi_chunks.jsonl")

# Only English; SDGi has multilingual content — non-English would noise the embedding
TARGET_LANGUAGE = "en"
MIN_TEXT_LEN = 80  # characters — discard header-only fragments

# Chunking parameters — match preprocess_policy.py
TARGET_WORDS = 150
MAX_WORDS = 300


def split_long_text(text: str) -> list[str]:
    """Split a long text block into ~TARGET_WORDS chunks at sentence boundaries."""
    # Split into sentences
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks, current, count = [], [], 0
    for sent in sentences:
        wc = len(sent.split())
        if count + wc > MAX_WORDS and current:
            chunks.append(" ".join(current))
            current, count = [sent], wc
        else:
            current.append(sent)
            count += wc
        if count >= TARGET_WORDS:
            chunks.append(" ".join(current))
            current, count = [], 0
    if current:
        chunks.append(" ".join(current))
    return [c for c in chunks if len(c.split()) >= 10]


def main() -> None:
    print(f"Loading {INPUT_PARQUET} ...")
    df = pd.read_parquet(INPUT_PARQUET)
    print(f"  Total rows: {len(df)}")

    # Filter to English
    def is_english(meta: object) -> bool:
        if isinstance(meta, dict):
            return meta.get("language", "en") == TARGET_LANGUAGE
        return True

    df = df[df["metadata"].apply(is_english)].copy()
    print(f"  English rows: {len(df)}")

    # Filter short texts
    df = df[df["text"].str.len() >= MIN_TEXT_LEN].copy()
    print(f"  After length filter (≥{MIN_TEXT_LEN} chars): {len(df)}")

    chunks: list[dict] = []
    for idx, row in df.iterrows():
        meta = row["metadata"] if isinstance(row["metadata"], dict) else {}
        labels = row["labels"]
        if hasattr(labels, "tolist"):
            labels = [int(x) for x in labels.tolist()]
        else:
            labels = [int(x) for x in (labels or [])]

        text = str(row["text"]).strip()
        if not text:
            continue

        # Build a human-readable source label
        country = meta.get("country", "unknown").upper()
        doc_type = meta.get("type", "policy").upper()  # vnr or vlr
        locality = meta.get("locality", "")
        year = meta.get("year")
        file_id = meta.get("file_id", "")

        source_doc = (
            f"sdgi_{country}_{doc_type}"
            + (f"_{locality}" if locality else "")
            + (f"_{year}" if year else "")
        )
        # Sanitise for use as an ID
        source_doc = source_doc.replace("/", "_").replace(" ", "_").replace("-", "_")

        # Rechunk long texts (SDGi often stores full pages as one entry)
        if len(text.split()) > MAX_WORDS:
            sub_texts = split_long_text(text)
        else:
            sub_texts = [text]

        for sub_i, sub_text in enumerate(sub_texts):
            chunks.append(
                {
                    "chunk_id": f"sdgi_{idx:05d}_{sub_i}",
                    "source_doc": source_doc,
                    "chunk_index": sub_i,
                    "text": sub_text,
                    "word_count": len(sub_text.split()),
                    "sdg_labels": labels,
                    "institution": f"{country} ({doc_type})",
                    "year": year,
                    "original_file": file_id,
                }
            )

    OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_JSONL.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk) + "\n")

    print(f"\nDone. {len(chunks)} chunks → {OUTPUT_JSONL}")

    # SDG distribution
    sdg_counts: Counter = Counter()
    for c in chunks:
        for sdg in c["sdg_labels"]:
            sdg_counts[sdg] += 1
    print("\nSDG label distribution:")
    for sdg in sorted(sdg_counts):
        bar = "█" * (sdg_counts[sdg] // 20)
        print(f"  SDG {sdg:2d}: {sdg_counts[sdg]:4d}  {bar}")

    # Country distribution (top 15)
    country_counts: Counter = Counter(
        c["institution"].split("(")[0].strip() for c in chunks
    )
    print("\nTop 15 countries:")
    for country, count in country_counts.most_common(15):
        print(f"  {country}: {count}")


if __name__ == "__main__":
    main()

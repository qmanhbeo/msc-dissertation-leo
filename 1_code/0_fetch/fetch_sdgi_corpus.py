"""
Fetch the SDGi Corpus from Hugging Face.

Source: https://huggingface.co/datasets/UNDP/sdgi-corpus
Dataset: UNDP/sdgi-corpus

The SDGi Corpus is a comprehensive multilingual dataset of Voluntary National Reviews (VNRs)
and Voluntary Local Reviews (VLRs) labeled by SDG. This is authoritative policy language
directly from governments reporting on SDG implementation.

Output: 2_data/0_raw/sdgi_corpus/ (downloaded Parquet files)
        2_data/0_raw/sdgi_corpus/metadata.json

Citation:
    Skrynnyk, O. et al. (2024). SDGi Corpus: A Comprehensive Multilingual Dataset
    for Text Classification by Sustainable Development Goals.
    https://huggingface.co/datasets/UNDP/sdgi-corpus

Run from project root:
    python 1_code/0_fetch/fetch_sdgi_corpus.py
"""

import json
import sys
from datetime import datetime
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
ANALYSIS_DIR = CODE_ROOT / "7_main_analysis" / "0_shared"
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))
from model_utils import raw_dir

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
HUGGINGFACE_ID = "UNDP/sdgi-corpus"
OUTPUT_DIR = raw_dir() / "sdgi_corpus"
METADATA_FILE = OUTPUT_DIR / "metadata.json"

SOURCE_CITATION = (
    "Skrynnyk, O. et al. (2024). SDGi Corpus: A Comprehensive Multilingual Dataset "
    "for Text Classification by Sustainable Development Goals. "
    "United Nations Development Programme (UNDP). "
    "https://huggingface.co/datasets/UNDP/sdgi-corpus"
)

SOURCE_DESCRIPTION = (
    "The SDGi Corpus contains text excerpts from Voluntary National Reviews (VNRs) "
    "and Voluntary Local Reviews (VLRs) submitted by countries and cities to the UN "
    "High-Level Political Forum. Texts are labeled by SDG (1-17). "
    "Languages: English, Spanish, French. "
    "License: cc-by-nc-sa-4.0"
)


def check_existing() -> dict | None:
    """Check if data already exists. Returns metadata if found, None otherwise."""
    downloaded_files = list(OUTPUT_DIR.glob("*.parquet")) + list(OUTPUT_DIR.glob("*.json"))
    downloaded_files = [f for f in downloaded_files if f.name != "metadata.json"]

    if not downloaded_files:
        return None

    total_size_mb = sum(f.stat().st_size for f in downloaded_files) / (1024 * 1024)
    stat = downloaded_files[0].stat()

    metadata = {
        "source": "Hugging Face - UNDP/sdgi-corpus",
        "dataset_id": HUGGINGFACE_ID,
        "url": f"https://huggingface.co/datasets/{HUGGINGFACE_ID}",
        "citation": SOURCE_CITATION,
        "description": SOURCE_DESCRIPTION,
        "fetched_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        "file_size_mb": round(total_size_mb, 2),
        "status": "already_downloaded",
        "file_count": len(downloaded_files),
    }
    return metadata


def download_via_huggingface() -> bool:
    """Download dataset using the Hugging Face datasets library."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: 'datasets' package not installed.")
        print("Run: pip install datasets")
        return False

    print(f"Downloading dataset: {HUGGINGFACE_ID}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("This may take a few minutes on first run (downloads ~50-100 MB)...\n")

    try:
        dataset = load_dataset(HUGGINGFACE_ID, split="train", trust_remote_code=True)
        print(f"Loaded dataset: {dataset.num_rows} rows, {dataset.num_columns} columns")
        print(f"Features: {list(dataset.column_names)}")

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        output_parquet = OUTPUT_DIR / "sdgi_corpus.parquet"
        dataset.to_parquet(str(output_parquet))
        print(f"\n✓ Saved train split → {output_parquet}")

        try:
            test_ds = load_dataset(HUGGINGFACE_ID, split="test", trust_remote_code=True)
            output_test = OUTPUT_DIR / "sdgi_corpus_test.parquet"
            test_ds.to_parquet(str(output_test))
            print(f"✓ Saved test split → {output_test}")
        except Exception:
            print("(No test split available)")

        import pandas as pd
        df = dataset.to_pandas()

        sdg_counts = {}
        for labels in df["labels"]:
            for l in labels:
                sdg_counts[int(l)] = sdg_counts.get(int(l), 0) + 1

        stats = {
            "num_rows": dataset.num_rows,
            "num_columns": dataset.num_columns,
            "columns": list(dataset.column_names),
            "features": {k: str(v) for k, v in dataset.features.items()},
            "sdg_distribution": sdg_counts,
            "language_distribution": (
                df["metadata"]
                .apply(lambda x: x.get("language", "unknown") if isinstance(x, dict) else "unknown")
                .value_counts()
                .to_dict()
            ),
        }

        return stats

    except Exception as e:
        print(f"✗ Error downloading dataset: {e}")
        return False


def main() -> None:
    """Main download pipeline."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print("SDGi Corpus Fetcher (Hugging Face)")
    print(f"{'='*70}")
    print(f"Dataset: {HUGGINGFACE_ID}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Citation: {SOURCE_CITATION}")
    print(f"{'='*70}\n")

    start_time = datetime.now()

    existing = check_existing()
    if existing:
        print(f"✓ Data already exists ({existing['file_size_mb']:.2f} MB)")
        print(f"  Downloaded: {existing['fetched_at']}")
        with METADATA_FILE.open("w") as f:
            json.dump(existing, f, indent=2)
        print(f"✓ Metadata saved to {METADATA_FILE}\n")
        return

    print("Downloading from Hugging Face Hub...\n")
    stats = download_via_huggingface()

    if stats:
        elapsed = datetime.now() - start_time
        downloaded_files = list(OUTPUT_DIR.glob("*.parquet"))
        total_size_mb = sum(f.stat().st_size for f in downloaded_files) / (1024 * 1024)

        metadata = {
            "source": "Hugging Face - UNDP/sdgi-corpus",
            "dataset_id": HUGGINGFACE_ID,
            "url": f"https://huggingface.co/datasets/{HUGGINGFACE_ID}",
            "citation": SOURCE_CITATION,
            "description": SOURCE_DESCRIPTION,
            "fetched_at": datetime.now().isoformat(),
            "elapsed_seconds": round(elapsed.total_seconds(), 1),
            "file_size_mb": round(total_size_mb, 2),
            "status": "success",
            "file_count": len(downloaded_files),
            "num_rows": stats["num_rows"],
            "num_columns": stats["num_columns"],
            "sdg_distribution": stats.get("sdg_distribution", {}),
            "language_distribution": stats.get("language_distribution", {}),
        }

        with METADATA_FILE.open("w") as f:
            json.dump(metadata, f, indent=2, default=str)

        print(f"\n{'='*70}")
        print("✓ Successfully downloaded SDGi Corpus")
        print(f"✓ Rows: {stats['num_rows']}")
        print(f"✓ Columns: {stats['num_columns']}")
        if stats.get("sdg_distribution"):
            print(f"✓ SDG distribution: {stats['sdg_distribution']}")
        print(f"✓ Total size: {total_size_mb:.2f} MB")
        print(f"✓ Time elapsed: {elapsed.total_seconds():.1f}s")
        print(f"✓ Metadata saved to {METADATA_FILE}")
        print(f"{'='*70}\n")
    else:
        print("\n✗ Download failed")
        metadata = {
            "source": "Hugging Face - UNDP/sdgi-corpus",
            "dataset_id": HUGGINGFACE_ID,
            "url": f"https://huggingface.co/datasets/{HUGGINGFACE_ID}",
            "citation": SOURCE_CITATION,
            "fetched_at": datetime.now().isoformat(),
            "status": "failed",
        }
        with METADATA_FILE.open("w") as f:
            json.dump(metadata, f, indent=2)


if __name__ == "__main__":
    main()

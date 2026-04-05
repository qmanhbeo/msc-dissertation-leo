"""
Fetch Sustainable Development Report dataset from Kaggle.

Dataset: sazidthe1/sustainable-development-report
This dataset contains SDG progress metrics and related data.

Requires: Kaggle API credentials (~/.kaggle/kaggle.json)

Output: data/kaggle/ (downloaded CSV files)
        data/kaggle/metadata.json
"""

import json
import os
from datetime import datetime
from pathlib import Path

# Try to import kaggle API
try:
    from kaggle.api.kaggle_api_extended import KaggleApi
    HAS_KAGGLE = True
except ImportError:
    HAS_KAGGLE = False

# Configuration
KAGGLE_DATASET = "sazidthe1/sustainable-development-report"
OUTPUT_DIR = Path("data/kaggle")
METADATA_FILE = OUTPUT_DIR / "metadata.json"


def check_kaggle_credentials() -> bool:
    """Check if Kaggle credentials are available."""
    creds_path = Path.home() / ".kaggle" / "kaggle.json"
    return creds_path.exists()


def setup_kaggle_instructions() -> str:
    """Return instructions for setting up Kaggle credentials."""
    return """
╔═══════════════════════════════════════════════════════════════════╗
║  Kaggle Credentials Not Found                                     ║
╠═══════════════════════════════════════════════════════════════════╣
║                                                                   ║
║  To download from Kaggle, you need to:                           ║
║                                                                   ║
║  1. Go to https://www.kaggle.com/settings/account                ║
║  2. Click "Create New Token" to download kaggle.json             ║
║  3. Place the file at: ~/.kaggle/kaggle.json                     ║
║  4. Set permissions: chmod 600 ~/.kaggle/kaggle.json             ║
║  5. Run this script again                                        ║
║                                                                   ║
║  Note: This step is optional. The other 4 fetch_*.py scripts     ║
║  provide complete data without Kaggle credentials.               ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝
"""


def download_dataset() -> bool:
    """Download dataset from Kaggle using the API."""
    if not HAS_KAGGLE:
        print("Error: 'kaggle' package not installed.")
        print("Run: pip install -r requirements.txt")
        return False

    if not check_kaggle_credentials():
        print(setup_kaggle_instructions())
        return False

    try:
        api = KaggleApi()
        api.authenticate()

        print(f"Downloading dataset: {KAGGLE_DATASET}")
        print(f"Output directory: {OUTPUT_DIR}")

        api.dataset_download_files(KAGGLE_DATASET, path=str(OUTPUT_DIR), unzip=True)

        return True

    except Exception as e:
        print(f"✗ Error downloading dataset: {e}")
        return False


def count_files_by_type(output_dir: Path) -> dict:
    """Count files in the downloaded directory by type."""
    counts = {"csv": 0, "json": 0, "txt": 0, "other": 0}

    for file_path in output_dir.rglob("*"):
        if file_path.is_file() and file_path.name != "metadata.json":
            ext = file_path.suffix.lstrip(".").lower()
            if ext in counts:
                counts[ext] += 1
            else:
                counts["other"] += 1

    return counts


def main():
    """Main download pipeline."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print("Kaggle Sustainable Development Report Fetcher")
    print(f"{'='*70}")
    print(f"Dataset: {KAGGLE_DATASET}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"{'='*70}\n")

    start_time = datetime.now()

    # Check if files already exist
    existing_files = [f for f in OUTPUT_DIR.glob("*") if f.is_file() and f.name != "metadata.json"]
    if existing_files:
        print(f"✓ Dataset already downloaded ({len(existing_files)} files)")
        elapsed = datetime.now() - start_time
        total_size_mb = sum(f.stat().st_size for f in existing_files) / (1024 * 1024)

        metadata = {
            "source": "Kaggle - Sustainable Development Report",
            "dataset": KAGGLE_DATASET,
            "kaggle_url": f"https://www.kaggle.com/datasets/{KAGGLE_DATASET}",
            "fetched_at": start_time.isoformat(),
            "elapsed_seconds": elapsed.total_seconds(),
            "status": "already_downloaded",
            "file_count": len(existing_files),
            "file_types": count_files_by_type(OUTPUT_DIR),
            "total_size_mb": round(total_size_mb, 2),
        }

        with open(METADATA_FILE, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"✓ Metadata saved to {METADATA_FILE}\n")
        return

    # Download
    print("Checking Kaggle credentials...")
    if not check_kaggle_credentials():
        print(setup_kaggle_instructions())
        print("⚠ Skipping Kaggle download (credentials not found)")
        print("  This is optional — all other data sources are available.\n")

        metadata = {
            "source": "Kaggle - Sustainable Development Report",
            "dataset": KAGGLE_DATASET,
            "kaggle_url": f"https://www.kaggle.com/datasets/{KAGGLE_DATASET}",
            "fetched_at": start_time.isoformat(),
            "status": "skipped_no_credentials",
        }

        with open(METADATA_FILE, "w") as f:
            json.dump(metadata, f, indent=2)

        return

    print("✓ Credentials found\n")

    if download_dataset():
        elapsed = datetime.now() - start_time
        downloaded_files = [f for f in OUTPUT_DIR.glob("*") if f.is_file() and f.name != "metadata.json"]
        total_size_mb = sum(f.stat().st_size for f in downloaded_files) / (1024 * 1024)

        metadata = {
            "source": "Kaggle - Sustainable Development Report",
            "dataset": KAGGLE_DATASET,
            "kaggle_url": f"https://www.kaggle.com/datasets/{KAGGLE_DATASET}",
            "fetched_at": start_time.isoformat(),
            "elapsed_seconds": elapsed.total_seconds(),
            "status": "success",
            "file_count": len(downloaded_files),
            "file_types": count_files_by_type(OUTPUT_DIR),
            "total_size_mb": round(total_size_mb, 2),
        }

        with open(METADATA_FILE, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"\n{'='*70}")
        print(f"✓ Successfully downloaded dataset")
        print(f"✓ Total files: {len(downloaded_files)}")
        print(f"✓ File breakdown: {metadata['file_types']}")
        print(f"✓ Total size: {total_size_mb:.2f} MB")
        print(f"✓ Time elapsed: {elapsed.total_seconds():.1f}s")
        print(f"✓ Metadata saved to {METADATA_FILE}")
        print(f"{'='*70}\n")
    else:
        print("\n✗ Download failed")
        metadata = {
            "source": "Kaggle - Sustainable Development Report",
            "dataset": KAGGLE_DATASET,
            "kaggle_url": f"https://www.kaggle.com/datasets/{KAGGLE_DATASET}",
            "fetched_at": start_time.isoformat(),
            "status": "failed",
        }

        with open(METADATA_FILE, "w") as f:
            json.dump(metadata, f, indent=2)


if __name__ == "__main__":
    main()

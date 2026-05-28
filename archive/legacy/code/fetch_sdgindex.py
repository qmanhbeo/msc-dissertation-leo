"""
Fetch Sustainable Development Report 2025 data from SDG Index.

Source: https://dashboards.sdgindex.org/downloads/
Direct URL: https://dashboards.sdgindex.org/static/downloads/files/SDR2025-data.xlsx

Output: data/sdgindex/sdr2025_data.xlsx
        data/sdgindex/metadata.json

Run from project root:
    python code/fetch_sdgindex.py
"""

import json
import os
from datetime import datetime
from pathlib import Path

import requests

# Configuration
SDGINDEX_URL = "https://dashboards.sdgindex.org/static/downloads/files/SDR2025-data.xlsx"
OUTPUT_FILE = Path("data/sdgindex/sdr2025_data.xlsx")
METADATA_FILE = OUTPUT_FILE.parent / "metadata.json"
SOURCE_URL = "https://dashboards.sdgindex.org/downloads/"
SOURCE_NAME = "SDG Index - Sustainable Development Report 2025"
SOURCE_CITATION = (
    "Sachs, J.D., Lafortune, G., Fuller, G., Iablonovski, G. (2025). "
    "Financing Sustainable Development to 2030 and Mid-Century. "
    "Sustainable Development Report 2025. Paris: SDSN, Dublin: Dublin University Press. "
    "DOI: https://doi.org/10.25546/111909"
)


def check_existing() -> dict | None:
    """Check if data already exists. Returns metadata if found, None otherwise."""
    if not OUTPUT_FILE.exists():
        return None

    stat = OUTPUT_FILE.stat()
    size_mb = stat.st_size / (1024 * 1024)

    metadata = {
        "source": SOURCE_NAME,
        "source_url": SOURCE_URL,
        "direct_url": SDGINDEX_URL,
        "citation": SOURCE_CITATION,
        "fetched_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        "file_size_mb": round(size_mb, 2),
        "status": "already_downloaded",
        "file_count": 1,
    }
    return metadata


def download_file() -> bool:
    """Download the Excel file from SDG Index."""
    print(f"Downloading: {SDGINDEX_URL}")
    print(f"Output: {OUTPUT_FILE}")

    try:
        response = requests.get(SDGINDEX_URL, timeout=120)
        response.raise_for_status()

        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        with OUTPUT_FILE.open("wb") as f:
            f.write(response.content)

        size_mb = OUTPUT_FILE.stat().st_size / (1024 * 1024)
        print(f"✓ Downloaded {size_mb:.2f} MB")
        return True

    except requests.exceptions.Timeout:
        print("✗ Timeout - network may be slow, try again later")
        return False
    except requests.exceptions.HTTPError as e:
        print(f"✗ HTTP error: {e}")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def main() -> None:
    """Main download pipeline."""
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print("SDG Index Sustainable Development Report 2025 Fetcher")
    print(f"{'='*70}")
    print(f"Source: {SOURCE_URL}")
    print(f"Output: {OUTPUT_FILE}")
    print(f"{'='*70}\n")

    start_time = datetime.now()

    # Check if already downloaded
    existing = check_existing()
    if existing:
        print(f"✓ Data already exists ({existing['file_size_mb']:.2f} MB)")
        print(f"  Downloaded: {existing['fetched_at']}")
        with METADATA_FILE.open("w") as f:
            json.dump(existing, f, indent=2)
        print(f"✓ Metadata saved to {METADATA_FILE}\n")
        return

    # Download
    print("Downloading SDR 2025 data...\n")
    if download_file():
        elapsed = datetime.now() - start_time
        size_mb = OUTPUT_FILE.stat().st_size / (1024 * 1024)

        metadata = {
            "source": SOURCE_NAME,
            "source_url": SOURCE_URL,
            "direct_url": SDGINDEX_URL,
            "citation": SOURCE_CITATION,
            "fetched_at": datetime.now().isoformat(),
            "elapsed_seconds": round(elapsed.total_seconds(), 1),
            "file_size_mb": round(size_mb, 2),
            "status": "success",
            "file_count": 1,
        }

        with METADATA_FILE.open("w") as f:
            json.dump(metadata, f, indent=2)

        print(f"\n{'='*70}")
        print("✓ Successfully downloaded SDR 2025 data")
        print(f"✓ File size: {size_mb:.2f} MB")
        print(f"✓ Time elapsed: {elapsed.total_seconds():.1f}s")
        print(f"✓ Metadata saved to {METADATA_FILE}")
        print(f"{'='*70}\n")
    else:
        print("\n✗ Download failed")
        metadata = {
            "source": SOURCE_NAME,
            "source_url": SOURCE_URL,
            "direct_url": SDGINDEX_URL,
            "citation": SOURCE_CITATION,
            "fetched_at": datetime.now().isoformat(),
            "status": "failed",
        }
        with METADATA_FILE.open("w") as f:
            json.dump(metadata, f, indent=2)


if __name__ == "__main__":
    main()

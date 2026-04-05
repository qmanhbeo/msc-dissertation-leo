"""
Fetch OSDG Community Dataset from Zenodo.

The OSDG (Open Sustainable Development Goals) Community Dataset is a public dataset
of ~42,000 text excerpts validated against the UN SDGs by 1,400+ citizen scientists.

Source: https://zenodo.org/records/11441197
Output: data/osdg/ (extracted CSV files)
        data/osdg/metadata.json (fetch metadata)
"""

import json
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests
from tqdm import tqdm

# Configuration
ZENODO_API_URL = "https://zenodo.org/api/records/11441197"
OUTPUT_DIR = Path("data/osdg")
METADATA_FILE = OUTPUT_DIR / "metadata.json"


def get_download_url() -> Optional[str]:
    """Fetch the Zenodo record and extract the CSV download URL."""
    print(f"Fetching Zenodo record metadata from {ZENODO_API_URL}...")
    response = requests.get(ZENODO_API_URL, timeout=10)
    response.raise_for_status()

    record = response.json()
    files = record.get("files", [])

    # Find the CSV file (usually the largest/main file)
    for file_info in files:
        filename = file_info.get("key", "")
        if filename.endswith(".csv") or filename.endswith(".zip"):
            return file_info.get("links", {}).get("self")

    # If no CSV found, try the first downloadable file
    if files:
        return files[0].get("links", {}).get("self")

    return None


def download_file(url: str, output_path: Path) -> None:
    """Download file with progress bar."""
    response = requests.get(url, stream=True, timeout=30)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))

    with open(output_path, "wb") as f:
        with tqdm(
            total=total_size,
            unit="B",
            unit_scale=True,
            desc=f"Downloading {output_path.name}",
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))


def extract_archive(archive_path: Path, extract_to: Path) -> list:
    """Extract ZIP or handle CSV file. Returns list of extracted files."""
    extracted_files = []

    try:
        # Try as ZIP first
        if archive_path.suffix == ".zip" or zipfile.is_zipfile(archive_path):
            print(f"Extracting {archive_path.name}...")
            with zipfile.ZipFile(archive_path, "r") as zf:
                zf.extractall(extract_to)
                extracted_files = [extract_to / name for name in zf.namelist()]
        else:
            # Treat as CSV
            csv_name = "osdg_dataset.csv"
            final_path = extract_to / csv_name
            archive_path.rename(final_path)
            extracted_files = [final_path]
    except zipfile.BadZipFile:
        # Not a ZIP, treat as CSV
        csv_name = "osdg_dataset.csv"
        final_path = extract_to / csv_name
        archive_path.rename(final_path)
        extracted_files = [final_path]

    return extracted_files


def main():
    """Main fetch and extract pipeline."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print("OSDG Community Dataset Fetcher")
    print(f"{'='*70}")
    print(f"Source: {ZENODO_API_URL}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"{'='*70}\n")

    start_time = datetime.now()

    try:
        # Get download URL
        download_url = get_download_url()
        if not download_url:
            raise ValueError("Could not find download URL in Zenodo record")

        print(f"Download URL: {download_url}")

        # Download file
        temp_file = OUTPUT_DIR / "osdg_temp"
        download_file(download_url, temp_file)

        # Extract
        extracted = extract_archive(temp_file, OUTPUT_DIR)

        # Clean up temp file if it was a ZIP
        if temp_file.exists() and temp_file.suffix == ".zip":
            temp_file.unlink()

        # Count CSV files and estimate records
        csv_files = list(OUTPUT_DIR.glob("*.csv"))
        total_records = 0

        for csv_file in csv_files:
            # Quick line count for metadata
            with open(csv_file, "r", encoding="utf-8") as f:
                line_count = sum(1 for _ in f) - 1  # -1 for header
                total_records += line_count

        elapsed = datetime.now() - start_time
        total_size_mb = sum(f.stat().st_size for f in extracted) / (1024 * 1024)

        # Save metadata
        metadata = {
            "source": "Zenodo - OSDG Community Dataset",
            "zenodo_url": "https://zenodo.org/records/11441197",
            "download_url": download_url,
            "fetched_at": start_time.isoformat(),
            "elapsed_seconds": elapsed.total_seconds(),
            "extracted_files": [str(f.relative_to(OUTPUT_DIR)) for f in extracted],
            "csv_count": len(csv_files),
            "estimated_total_records": total_records,
            "total_size_mb": round(total_size_mb, 2),
        }

        with open(METADATA_FILE, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"\n{'='*70}")
        print(f"✓ Successfully downloaded OSDG dataset")
        print(f"✓ CSV files: {len(csv_files)}")
        print(f"✓ Estimated records: {total_records:,}")
        print(f"✓ Total size: {total_size_mb:.2f} MB")
        print(f"✓ Time elapsed: {elapsed.total_seconds():.1f}s")
        print(f"✓ Metadata saved to {METADATA_FILE}")
        print(f"{'='*70}\n")

    except Exception as e:
        print(f"\n✗ Error during fetch: {e}")
        raise


if __name__ == "__main__":
    main()

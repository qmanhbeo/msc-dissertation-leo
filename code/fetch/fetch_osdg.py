"""
Fetch OSDG Community Dataset from Zenodo + examples from GitHub.

The OSDG (Open Sustainable Development Goals) Community Dataset is a public dataset
of ~42,000 text excerpts validated against the UN SDGs by 1,400+ citizen scientists.

Sources:
- Dataset: https://zenodo.org/records/11441197
- Examples: https://github.com/osdg-ai/osdg-data/tree/main/examples

Output: data/raw/osdg/ (extracted CSV files)
        data/raw/osdg/examples/ (helpers.py, notebooks)
        data/raw/osdg/metadata.json (fetch metadata)

Run from project root:
    python code/fetch/fetch_osdg.py
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
GITHUB_RAW_BASE = "https://raw.githubusercontent.com/osdg-ai/osdg-data/main/examples"
GITHUB_REPO_URL = "https://github.com/osdg-ai/osdg-data"
OUTPUT_DIR = Path("data/raw/osdg")
EXAMPLES_DIR = OUTPUT_DIR / "examples"
METADATA_FILE = OUTPUT_DIR / "metadata.json"

EXAMPLES_FILES = [
    ("helpers.py", "Python utilities for SDG classification"),
    ("osdg-cd-example-classifier-sklearn.ipynb", "Sklearn classification example notebook"),
]


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


def download_examples() -> list[dict]:
    """Download example files from GitHub repo."""
    print("\nFetching examples from GitHub...")
    EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    for filename, description in EXAMPLES_FILES:
        output_path = EXAMPLES_DIR / filename
        url = f"{GITHUB_RAW_BASE}/{filename}"

        if output_path.exists():
            size_kb = output_path.stat().st_size / 1024
            print(f"  {filename} already exists ({size_kb:.1f} KB)")
            results.append({
                "filename": filename,
                "description": description,
                "path": str(output_path.relative_to(OUTPUT_DIR)),
                "status": "already_exists",
                "size_kb": round(size_kb, 1),
            })
            continue

        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            output_path.write_bytes(response.content)
            size_kb = output_path.stat().st_size / 1024
            print(f"  Downloaded {filename} ({size_kb:.1f} KB)")
            results.append({
                "filename": filename,
                "description": description,
                "path": str(output_path.relative_to(OUTPUT_DIR)),
                "status": "success",
                "size_kb": round(size_kb, 1),
            })
        except requests.RequestException as e:
            print(f"  Failed to download {filename}: {e}")
            results.append({
                "filename": filename,
                "description": description,
                "status": "error",
                "error": str(e),
            })

    return results


def main():
    """Main fetch and extract pipeline."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print("OSDG Community Dataset Fetcher")
    print(f"{'='*70}")
    print(f"Zenodo: https://zenodo.org/records/11441197")
    print(f"GitHub: {GITHUB_REPO_URL}")
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

        # Download examples from GitHub
        examples_results = download_examples()

        elapsed = datetime.now() - start_time

        # Save metadata
        metadata = {
            "source": "Zenodo - OSDG Community Dataset",
            "zenodo_url": "https://zenodo.org/records/11441197",
            "github_url": GITHUB_REPO_URL,
            "download_url": download_url,
            "fetched_at": start_time.isoformat(),
            "elapsed_seconds": elapsed.total_seconds(),
            "dataset_files": [str(f.relative_to(OUTPUT_DIR)) for f in extracted],
            "csv_count": len(csv_files),
            "estimated_total_records": total_records,
            "dataset_size_mb": round(total_size_mb, 2),
            "examples": examples_results,
        }

        with open(METADATA_FILE, "w") as f:
            json.dump(metadata, f, indent=2)

        success_examples = sum(1 for r in examples_results if r["status"] == "success")
        existing_examples = sum(1 for r in examples_results if r["status"] == "already_exists")

        print(f"\n{'='*70}")
        print(f"Successfully downloaded OSDG dataset")
        print(f"  CSV files: {len(csv_files)}")
        print(f"  Estimated records: {total_records:,}")
        print(f"  Dataset size: {total_size_mb:.2f} MB")
        print(f"  Examples: {success_examples} downloaded, {existing_examples} already existed")
        print(f"  Time elapsed: {elapsed.total_seconds():.1f}s")
        print(f"  Metadata: {METADATA_FILE}")
        print(f"{'='*70}\n")

    except Exception as e:
        print(f"\nError during fetch: {e}")
        raise


if __name__ == "__main__":
    main()

"""
Fetch IISD SDG Knowledge Hub Dataset from Zenodo.

A dataset of news articles published on the IISD SDG Knowledge Hub (sdg.iisd.org),
labeled by SDG. Articles are authored and validated by IISD editors.

Source: https://zenodo.org/records/7523032
DOI: 10.5281/zenodo.7523032

Output: data/sdg_news/
        data/sdg_news/sdg_knowledge_hub.csv
        data/sdg_news/metadata.json

Run from project root:
    python code/fetch_sdg_news.py

Requires: requests, pandas, tqdm (optional)
"""

import json
from datetime import datetime
from pathlib import Path

import requests
from tqdm import tqdm

ZENODO_API_URL = "https://zenodo.org/api/records/7523032"
OUTPUT_DIR = Path("data/sdg_news")
METADATA_FILE = OUTPUT_DIR / "metadata.json"


def get_download_url() -> str:
    """Fetch the Zenodo record and extract the CSV download URL."""
    print(f"Fetching Zenodo record metadata from {ZENODO_API_URL}...")
    response = requests.get(ZENODO_API_URL, timeout=30)
    response.raise_for_status()

    record = response.json()
    files = record.get("files", [])

    for file_info in files:
        filename = file_info.get("key", "")
        if filename.endswith(".csv"):
            return file_info.get("links", {}).get("self")

    if files:
        return files[0].get("links", {}).get("self")

    raise ValueError("Could not find CSV file in Zenodo record")


def download_file(url: str, output_path: Path) -> None:
    """Download file with progress bar."""
    print(f"Downloading: {url.split('/')[-1]}")
    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))

    with open(output_path, "wb") as f:
        with tqdm(
            total=total_size,
            unit="B",
            unit_scale=True,
            desc="Downloading",
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))


def count_records(csv_path: Path) -> int:
    """Count records in CSV file."""
    with open(csv_path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f) - 1


def main() -> None:
    """Main fetch pipeline."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print("IISD SDG Knowledge Hub Dataset Fetcher")
    print(f"{'='*70}")
    print(f"Source: https://zenodo.org/records/7523032")
    print(f"Output: {OUTPUT_DIR}")
    print(f"{'='*70}\n")

    start_time = datetime.now()

    try:
        download_url = get_download_url()
        output_file = OUTPUT_DIR / "sdg_knowledge_hub.csv"

        if output_file.exists():
            size_mb = output_file.stat().st_size / (1024 * 1024)
            print(f"File already exists ({size_mb:.2f} MB)")
            print(f"  Re-downloading to ensure freshness...")
            output_file.unlink()

        download_file(download_url, output_file)

        elapsed = datetime.now() - start_time
        size_mb = output_file.stat().st_size / (1024 * 1024)
        record_count = count_records(output_file)

        metadata = {
            "source": "IISD SDG Knowledge Hub",
            "zenodo_url": "https://zenodo.org/records/7523032",
            "doi": "10.5281/zenodo.7523032",
            "dataset_url": "http://sdg.iisd.org/",
            "download_url": download_url,
            "fetched_at": datetime.now().isoformat(),
            "elapsed_seconds": round(elapsed.total_seconds(), 1),
            "file": "sdg_knowledge_hub.csv",
            "file_size_mb": round(size_mb, 2),
            "record_count": record_count,
            "citation": (
                "Wulff, D. U., & Meier, D. S. (2024). "
                "SDG Knowledge Hub Dataset of SDG-labeled News Articles "
                "[Data set]. Zenodo. https://doi.org/10.5281/zenodo.7523032"
            ),
            "description": (
                "Dataset of articles published on the IISD SDG Knowledge Hub. "
                "Labels assigned by authors and validated by IISD editors. "
                "Columns include: url, title, type, text, date, sdgs, SDG-01 to SDG-17 (binary labels)."
            ),
            "license": "cc-by-4.0",
        }

        with open(METADATA_FILE, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"\n{'='*70}")
        print("Successfully downloaded IISD SDG Knowledge Hub dataset")
        print(f"  Records: {record_count:,}")
        print(f"  Size: {size_mb:.2f} MB")
        print(f"  Time: {elapsed.total_seconds():.1f}s")
        print(f"  Metadata: {METADATA_FILE}")
        print(f"{'='*70}\n")

    except Exception as e:
        print(f"\nError: {e}")
        raise


if __name__ == "__main__":
    main()

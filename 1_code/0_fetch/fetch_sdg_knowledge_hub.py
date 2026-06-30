"""
Fetch SDG Knowledge Hub Dataset from Zenodo.

Source: Wulff, D. U., Meier, D. S., & Mata, R. (2023).
        SDG Knowledge Hub Dataset of SDG-labeled News Articles.
        https://doi.org/10.5281/zenodo.7523032

The dataset contains ~9,172 news articles from the IISD SDG Knowledge Hub,
each labelled with relevant SDGs by expert editors. Articles span 2017--2022.

Output: 2_data/0_raw/sdg_knowledge_hub/sdg_knowledge_hub.csv
        2_data/0_raw/sdg_knowledge_hub/artifact/metadata.json

Run from project root:
    python 1_code/0_fetch/fetch_sdg_knowledge_hub.py
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests
from tqdm import tqdm

# Configuration
ZENODO_API_URL = "https://zenodo.org/api/records/7523032"
OUTPUT_DIR = Path("2_data/0_raw/sdg_knowledge_hub")
METADATA_FILE = OUTPUT_DIR / "artifact" / "metadata.json"


def get_download_url() -> Optional[str]:
    """Fetch the Zenodo record and extract the CSV download URL."""
    print(f"Fetching Zenodo record metadata from {ZENODO_API_URL}...")
    response = requests.get(ZENODO_API_URL, timeout=10)
    response.raise_for_status()

    record = response.json()
    files = record.get("files", [])

    for file_info in files:
        filename = file_info.get("key", "")
        if filename.endswith(".csv"):
            return file_info.get("links", {}).get("self")

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


def main():
    """Main fetch pipeline."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_FILE.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print("SDG Knowledge Hub Dataset Fetcher")
    print(f"{'='*70}")
    print(f"Zenodo: https://zenodo.org/records/7523032")
    print(f"DOI: 10.5281/zenodo.7523032")
    print(f"Output: {OUTPUT_DIR}")
    print(f"{'='*70}\n")

    start_time = datetime.now()

    try:
        download_url = get_download_url()
        if not download_url:
            raise ValueError("Could not find download URL in Zenodo record")

        print(f"Download URL: {download_url}")

        output_csv = OUTPUT_DIR / "sdg_knowledge_hub.csv"
        download_file(download_url, output_csv)

        # Count records (rows minus header)
        with open(output_csv, "r", encoding="utf-8") as f:
            total_records = sum(1 for _ in f) - 1

        elapsed = datetime.now() - start_time
        file_size_mb = output_csv.stat().st_size / (1024 * 1024)

        metadata = {
            "source": "Zenodo - SDG Knowledge Hub Dataset",
            "zenodo_url": "https://zenodo.org/records/7523032",
            "doi": "10.5281/zenodo.7523032",
            "citation": "Wulff, D. U., Meier, D. S., & Mata, R. (2023). SDG Knowledge Hub Dataset of SDG-labeled News Articles.",
            "download_url": download_url,
            "fetched_at": start_time.isoformat(),
            "elapsed_seconds": elapsed.total_seconds(),
            "dataset_file": str(output_csv.relative_to(OUTPUT_DIR)),
            "file_size_mb": round(file_size_mb, 2),
            "estimated_total_records": total_records,
        }

        with open(METADATA_FILE, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"\n{'='*70}")
        print(f"Successfully downloaded SDG Knowledge Hub dataset")
        print(f"  Records: {total_records:,}")
        print(f"  Size: {file_size_mb:.2f} MB")
        print(f"  Time elapsed: {elapsed.total_seconds():.1f}s")
        print(f"  Metadata: {METADATA_FILE}")
        print(f"{'='*70}\n")

    except Exception as e:
        print(f"\nError during fetch: {e}")
        raise


if __name__ == "__main__":
    main()

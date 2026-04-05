"""
Fetch AURORA SDG Dataset (1.4M research articles with SDG labels) from Zenodo.

A dataset of 1.4 million research article DOIs labeled at SDG Target level (169 targets)
and Goal level (17 goals), covering 2009-2020.

Source: https://zenodo.org/records/5224005
DOI: 10.5281/zenodo.5224005

Output: data/aurora/
        data/aurora/aurora_sdg_targets.csv      — CSV format (doi, date, sdg_target, sdg_goal)
        data/aurora/aurora_sdg_targets_wide.csv  — Wide format (doi, date, 169 targets, 17 goals)
        data/aurora/aurora_sdg_targets.xlsx      — Excel format
        data/aurora/metadata.json

Run from project root:
    python code/fetch_aurora.py

Requires: requests, pandas, tqdm (optional)
"""

import json
from datetime import datetime
from pathlib import Path

import requests
from tqdm import tqdm

ZENODO_API_URL = "https://zenodo.org/api/records/5224005"
OUTPUT_DIR = Path("data/aurora")
METADATA_FILE = OUTPUT_DIR / "metadata.json"


def get_files() -> list[dict]:
    """Fetch Zenodo record and return file list."""
    print(f"Fetching Zenodo record metadata...")
    response = requests.get(ZENODO_API_URL, timeout=30)
    response.raise_for_status()

    record = response.json()
    return record.get("files", []), record


def download_file(url: str, output_path: Path) -> None:
    """Download file with progress bar."""
    print(f"Downloading: {output_path.name}")
    response = requests.get(url, stream=True, timeout=300)
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


def main() -> None:
    """Main fetch pipeline."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print("AURORA SDG Dataset Fetcher")
    print(f"{'='*70}")
    print(f"Source: https://zenodo.org/records/5224005")
    print(f"Output: {OUTPUT_DIR}")
    print(f"{'='*70}\n")

    start_time = datetime.now()

    try:
        files, record = get_files()
        metadata_record = record.get("metadata", {})

        files_to_download = [
            ("aurora_sdg_v5_worldwide_set_doi_sdg_targets_2009-2020.csv", "Long format (doi, date, sdg_target, sdg_goal)"),
            ("aurora_sdg_v5_worldwide_set_doi_sdg_targets_2009-2020-in-columns.csv", "Wide format (169 targets, 17 goals as columns)"),
            ("aurora_sdg_v5_worldwide_set_doi_sdg_targets_2009-2020.xlsx", "Excel format"),
        ]

        downloaded = []
        total_size_mb = 0

        for filename, description in files_to_download:
            output_path = OUTPUT_DIR / filename

            file_info = next((f for f in files if f.get("key") == filename), None)
            if not file_info:
                print(f"  {filename}: not found in record")
                continue

            download_url = file_info.get("links", {}).get("self")

            if output_path.exists():
                size_mb = output_path.stat().st_size / (1024 * 1024)
                print(f"  {filename} already exists ({size_mb:.2f} MB)")
                total_size_mb += size_mb
                downloaded.append({
                    "filename": filename,
                    "description": description,
                    "status": "already_exists",
                    "size_mb": round(size_mb, 2),
                })
            else:
                download_file(download_url, output_path)
                size_mb = output_path.stat().st_size / (1024 * 1024)
                total_size_mb += size_mb
                downloaded.append({
                    "filename": filename,
                    "description": description,
                    "status": "success",
                    "size_mb": round(size_mb, 2),
                })

        elapsed = datetime.now() - start_time

        citation = (
            "Vanderfeesten, M. (2024). "
            "DOI's with SDG labels on Target level | 1.4M research articles (2009-2020) "
            "related to Sustainable Development Goals (Version 1.1) [Data set]. "
            "Zenodo. https://doi.org/10.5281/zenodo.5224005"
        )

        metadata = {
            "source": "AURORA SDG Dataset",
            "zenodo_url": "https://zenodo.org/records/5224005",
            "doi": "10.5281/zenodo.5224005",
            "citation": citation,
            "fetched_at": datetime.now().isoformat(),
            "elapsed_seconds": round(elapsed.total_seconds(), 1),
            "description": {
                "summary": "1.4 million research article DOIs labeled at SDG Target level (169 targets) and Goal level (17 goals)",
                "period": "2009-2020",
                "files": downloaded,
            },
            "license": "cc-by-4.0",
            "notes": [
                "aurora_sdg_targets.csv: Long format with doi, date, sdg_target, sdg_goal columns",
                "aurora_sdg_targets_wide.csv: Wide format with 169 target columns + 17 goal columns",
                "aurora_sdg_targets.xlsx: Excel version of the wide format",
                "Use DOI to cross-reference with OpenAlex or CrossRef for full article metadata",
            ],
        }

        with open(METADATA_FILE, "w") as f:
            json.dump(metadata, f, indent=2)

        success_count = sum(1 for d in downloaded if d["status"] == "success")
        existing_count = sum(1 for d in downloaded if d["status"] == "already_exists")

        print(f"\n{'='*70}")
        print("Successfully processed AURORA SDG Dataset")
        print(f"  Files: {success_count} downloaded, {existing_count} already existed")
        print(f"  Total size: {total_size_mb:.2f} MB")
        print(f"  Time: {elapsed.total_seconds():.1f}s")
        print(f"  Metadata: {METADATA_FILE}")
        print(f"{'='*70}\n")

    except Exception as e:
        print(f"\nError: {e}")
        raise


if __name__ == "__main__":
    main()

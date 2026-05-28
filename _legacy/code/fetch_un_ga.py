"""
Fetch UN General Assembly data from the UN Digital Library.

Two datasets:
1. GA Outcomes: UN General Assembly resolutions (1946-2025) - resolution text, subjects, voting
2. GA Voting: Country-level voting data on each resolution

Source: https://digitallibrary.un.org/
Records: 4060945 (outcomes), 4060887 (voting)

Output: data/un_ga/
        data/un_ga/ga_outcomes.csv     — Resolution outcomes
        data/un_ga/ga_voting.csv      — Country-level voting data
        data/un_ga/metadata.json

Run from project root:
    python code/fetch_un_ga.py

Requires: requests, pandas, tqdm (optional)
"""

import json
from datetime import datetime
from pathlib import Path

import requests
from tqdm import tqdm

OUTPUT_DIR = Path("data/un_ga")
METADATA_FILE = OUTPUT_DIR / "metadata.json"

RECORDS = {
    "outcomes": {
        "record_id": "4060945",
        "filename": "ga_outcomes.csv",
        "date_suffix": "2026_02_06",
        "description": "UN General Assembly resolution outcomes (1946-2025)",
    },
    "voting": {
        "record_id": "4060887",
        "filename": "ga_voting.csv",
        "date_suffix": "2026_02_06",
        "description": "Country-level voting data for GA resolutions (1946-2025)",
    },
}


def download_csv(record_id: str, filename: str, date_suffix: str) -> dict:
    """Download a CSV file from UN Digital Library."""
    base_url = f"https://digitallibrary.un.org/record/{record_id}/files/{date_suffix}_{filename}"
    
    output_path = OUTPUT_DIR / filename
    
    if output_path.exists():
        size_mb = output_path.stat().st_size / (1024 * 1024)
        return {
            "status": "already_exists",
            "path": str(output_path.relative_to(OUTPUT_DIR)),
            "size_mb": round(size_mb, 2),
        }
    
    print(f"Downloading: {filename}")
    response = requests.get(base_url, stream=True, timeout=300, allow_redirects=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get("content-length", 0))
    
    with open(output_path, "wb") as f:
        with tqdm(
            total=total_size,
            unit="B",
            unit_scale=True,
            desc=filename,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    
    size_mb = output_path.stat().st_size / (1024 * 1024)
    return {
        "status": "success",
        "path": str(output_path.relative_to(OUTPUT_DIR)),
        "size_mb": round(size_mb, 2),
    }


def count_records(csv_path: Path) -> int:
    """Count records in CSV file."""
    with open(csv_path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f) - 1


def main() -> None:
    """Main fetch pipeline."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print("UN General Assembly Data Fetcher")
    print(f"{'='*70}")
    print(f"Source: https://digitallibrary.un.org/")
    print(f"Output: {OUTPUT_DIR}")
    print(f"{'='*70}\n")

    start_time = datetime.now()

    try:
        results = []
        
        for key, info in RECORDS.items():
            result = download_csv(
                record_id=info["record_id"],
                filename=info["filename"],
                date_suffix=info["date_suffix"],
            )
            result["type"] = key
            result["description"] = info["description"]
            result["record_id"] = info["record_id"]
            results.append(result)
            
            if result["status"] == "success" or result["status"] == "already_exists":
                csv_path = OUTPUT_DIR / info["filename"]
                if csv_path.exists():
                    result["record_count"] = count_records(csv_path)

        elapsed = datetime.now() - start_time

        metadata = {
            "source": "UN Digital Library",
            "source_url": "https://digitallibrary.un.org/",
            "fetched_at": datetime.now().isoformat(),
            "elapsed_seconds": round(elapsed.total_seconds(), 1),
            "description": {
                "ga_outcomes": "UN General Assembly resolution outcomes - resolution text, subjects, vote counts",
                "ga_voting": "Country-level voting data - how each country voted on each resolution",
            },
            "citation": (
                "United Nations Dag Hammarskjöld Library. "
                "UN General Assembly resolutions and voting data, 1946-2025. "
                "https://digitallibrary.un.org/"
            ),
            "files": results,
            "notes": [
                "ga_outcomes.csv: Resolution-level data (7.4 MB, ~8,000 resolutions)",
                "ga_voting.csv: Country-vote level data (364 MB, ~1.5M rows) - large file!",
                "Use to track which SDGs get UNGA attention over time",
                "Vote data can be used for coalition analysis on development issues",
            ],
        }

        with open(METADATA_FILE, "w") as f:
            json.dump(metadata, f, indent=2)

        success_count = sum(1 for r in results if r["status"] == "success")
        existing_count = sum(1 for r in results if r["status"] == "already_exists")

        print(f"\n{'='*70}")
        print("Successfully processed UN GA data")
        print(f"  Files: {success_count} downloaded, {existing_count} already existed")
        print(f"  Time: {elapsed.total_seconds():.1f}s")
        print(f"  Metadata: {METADATA_FILE}")
        print(f"{'='*70}\n")

    except Exception as e:
        print(f"\nError: {e}")
        raise


if __name__ == "__main__":
    main()

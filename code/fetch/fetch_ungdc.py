"""
Fetch UN General Debate Corpus (UNGDC) from Harvard Dataverse.

A corpus of UN General Assembly General Debate speeches from 1946-2025.
Contains 11,141 speeches from 202 countries expressing government positions on global issues.

Source: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/0TJX8Y
DOI: 10.7910/DVN/0TJX8Y
Website: https://www.ungdc.bham.ac.uk

Output: data/ungdc/
        data/ungdc/TXT/              — extracted corpus (11,141 speeches by session)
        data/ungdc/Speakers_by_session.xlsx
        data/ungdc/README.txt
        data/ungdc/UNGDC_1946-2025.tar.gz
        data/ungdc/metadata.json

Run from project root:
    python code/fetch/fetch_ungdc.py

Requires: requests, pandas, tqdm (optional)
"""

import json
import tarfile
from datetime import datetime
from pathlib import Path

import requests
from tqdm import tqdm

DATASET_PERSISTENT_ID = "doi:10.7910/DVN/0TJX8Y"
DATASET_API_URL = f"https://dataverse.harvard.edu/api/datasets/:persistentId?persistentId={DATASET_PERSISTENT_ID}"
OUTPUT_DIR = Path("data/ungdc")
CORPUS_DIR = OUTPUT_DIR / "UNGDC_1946-2025"
METADATA_FILE = OUTPUT_DIR / "metadata.json"


def get_file_download_urls() -> list[dict]:
    """Fetch dataset metadata and extract file download URLs."""
    print(f"Fetching dataset metadata from Harvard Dataverse...")
    response = requests.get(DATASET_API_URL, timeout=30)
    response.raise_for_status()

    data = response.json().get("data", {})
    files = data.get("latestVersion", {}).get("files", [])

    result = []
    for f in files:
        file_id = f.get("dataFile", {}).get("id")
        label = f.get("label")
        size = f.get("dataFile", {}).get("filesize", 0)
        content_type = f.get("dataFile", {}).get("contentType", "")

        download_url = f"https://dataverse.harvard.edu/api/access/datafile/{file_id}"

        result.append({
            "file_id": file_id,
            "label": label,
            "size_bytes": size,
            "content_type": content_type,
            "download_url": download_url,
        })
        print(f"  Found: {label} ({size / 1024:.1f} KB)")

    return result


def download_file(url: str, output_path: Path, description: str = "Downloading") -> None:
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
            desc=description,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))


def extract_tarball(tar_path: Path, extract_to: Path) -> list[Path]:
    """Extract tar.gz archive."""
    print(f"Extracting: {tar_path.name}")
    extracted = []

    with tarfile.open(tar_path, "r:gz") as tf:
        members = tf.getmembers()
        for member in tqdm(members, desc="Extracting"):
            tf.extract(member, extract_to)
            extracted.append(extract_to / member.name)

    return extracted


def main() -> None:
    """Main fetch pipeline."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print("UN General Debate Corpus (UNGDC) Fetcher")
    print(f"{'='*70}")
    print(f"Source: https://dataverse.harvard.edu/dataset.xhtml?persistentId={DATASET_PERSISTENT_ID}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"{'='*70}\n")

    start_time = datetime.now()

    try:
        files = get_file_download_urls()

        downloaded = []

        for file_info in files:
            output_path = OUTPUT_DIR / file_info["label"]
            download_url = file_info["download_url"]

            if output_path.exists():
                size_mb = output_path.stat().st_size / (1024 * 1024)
                print(f"  {file_info['label']} already exists ({size_mb:.2f} MB)")
                downloaded.append({
                    **file_info,
                    "path": str(output_path.relative_to(OUTPUT_DIR)),
                    "status": "already_exists",
                })

                if file_info["label"].endswith(".tar.gz"):
                    if not CORPUS_DIR.exists():
                        extract_tarball(output_path, OUTPUT_DIR)
                    else:
                        print(f"  Corpus already extracted")

            else:
                download_file(download_url, output_path, file_info["label"])

                size_mb = output_path.stat().st_size / (1024 * 1024)
                downloaded.append({
                    **file_info,
                    "path": str(output_path.relative_to(OUTPUT_DIR)),
                    "status": "success",
                })

                if file_info["label"].endswith(".tar.gz"):
                    extract_tarball(output_path, OUTPUT_DIR)

        elapsed = datetime.now() - start_time

        metadata = {
            "source": "UN General Debate Corpus (UNGDC)",
            "dataverse_url": f"https://dataverse.harvard.edu/dataset.xhtml?persistentId={DATASET_PERSISTENT_ID}",
            "website_url": "https://www.ungdc.bham.ac.uk",
            "doi": "10.7910/DVN/0TJX8Y",
            "fetched_at": datetime.now().isoformat(),
            "elapsed_seconds": round(elapsed.total_seconds(), 1),
            "description": {
                "summary": "UN General Assembly General Debate speeches 1946-2025",
                "speeches": "11,141 speeches from 202 countries",
                "period": "1946-2025",
            },
            "citation": (
                "Jankin, S., Baturo, A., & Dasandi, N. (2025). "
                "Words to unite nations: The complete United Nations General Debate Corpus, 1946–present. "
                "Journal of Peace Research, 62(4), 1339-1351. "
                "https://doi.org/10.1177/00223433241275335"
            ),
            "authors": [
                {"name": "Jankin, Slava", "affiliation": "University of Birmingham"},
                {"name": "Baturo, Alexander", "affiliation": "Dublin City University"},
                {"name": "Dasandi, Niheer", "affiliation": "University of Birmingham"},
            ],
            "files": downloaded,
            "license": "Unknown - check README",
            "notes": [
                "Contains government speeches expressing national positions on global issues",
                "Valuable for tracking policy discourse and alignment over time",
                "Cross-reference with SDG coverage in UNGD to measure policy attention to SDGs",
            ],
        }

        with open(METADATA_FILE, "w") as f:
            json.dump(metadata, f, indent=2)

        success_count = sum(1 for d in downloaded if d["status"] == "success")
        existing_count = sum(1 for d in downloaded if d["status"] == "already_exists")

        print(f"\n{'='*70}")
        print("Successfully processed UN General Debate Corpus")
        print(f"  Files: {success_count} downloaded, {existing_count} already existed")
        print(f"  Time: {elapsed.total_seconds():.1f}s")
        print(f"  Metadata: {METADATA_FILE}")
        print(f"{'='*70}\n")

    except Exception as e:
        print(f"\nError: {e}")
        raise


if __name__ == "__main__":
    main()

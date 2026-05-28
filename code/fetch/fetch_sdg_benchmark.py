"""
Fetch SDG Classification Benchmark from GitHub.

This repository contains a benchmark dataset for SDG classification in academic papers.
Source: https://github.com/SDGClassification/benchmark

Output: data/raw/sdg_benchmark/ (extracted repo contents)
        data/raw/sdg_benchmark/metadata.json
"""

import json
import zipfile
from datetime import datetime
from io import BytesIO
from pathlib import Path

import requests
from tqdm import tqdm

# Configuration
GITHUB_REPO = "SDGClassification/benchmark"
GITHUB_API_URL = f"https://api.github.com/repos/{GITHUB_REPO}/zipball/main"
OUTPUT_DIR = Path("data/raw/sdg_benchmark")
METADATA_FILE = OUTPUT_DIR / "metadata.json"


def download_repo_zip(url: str) -> BytesIO:
    """Download repository as ZIP from GitHub API."""
    print(f"Downloading repository from GitHub: {GITHUB_REPO}")
    response = requests.get(url, stream=True, timeout=30)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))

    # Download to memory
    content = BytesIO()
    with tqdm(
        total=total_size,
        unit="B",
        unit_scale=True,
        desc="Downloading repository",
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                content.write(chunk)
                pbar.update(len(chunk))

    content.seek(0)
    return content


def extract_zip_to_dir(zip_buffer: BytesIO, output_dir: Path) -> list:
    """Extract ZIP contents to directory, skipping top-level folder created by GitHub."""
    extracted_files = []

    print(f"Extracting to {output_dir}...")

    with zipfile.ZipFile(zip_buffer, "r") as zf:
        # GitHub API creates a folder like "SDGClassification-benchmark-abc123"
        # We want to extract its contents directly to output_dir

        all_files = zf.namelist()
        if all_files:
            # Find the top-level folder
            top_folder = all_files[0].split("/")[0]

            # Extract each file, removing the top-level folder prefix
            with tqdm(total=len(all_files), desc="Extracting files", leave=False) as pbar:
                for file_info in zf.infolist():
                    # Remove top-level folder from path
                    path_parts = file_info.filename.split("/")
                    if len(path_parts) > 1:
                        relative_path = "/".join(path_parts[1:])
                    else:
                        continue  # Skip the top-level folder itself

                    if not relative_path:
                        continue

                    # Extract to output directory
                    target_path = output_dir / relative_path

                    if file_info.is_dir():
                        target_path.mkdir(parents=True, exist_ok=True)
                    else:
                        target_path.parent.mkdir(parents=True, exist_ok=True)
                        with zf.open(file_info) as source, open(target_path, "wb") as target:
                            target.write(source.read())

                    extracted_files.append(target_path)
                    pbar.update(1)

    return extracted_files


def count_files_by_type(output_dir: Path) -> dict:
    """Count files in the extracted directory by type."""
    counts = {
        "py": 0,
        "csv": 0,
        "json": 0,
        "md": 0,
        "txt": 0,
        "other": 0,
    }

    for file_path in output_dir.rglob("*"):
        if file_path.is_file():
            ext = file_path.suffix.lstrip(".").lower()
            if ext in counts:
                counts[ext] += 1
            else:
                counts["other"] += 1

    return counts


def main():
    """Main fetch and extract pipeline."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print("SDG Classification Benchmark Fetcher")
    print(f"{'='*70}")
    print(f"Repository: https://github.com/{GITHUB_REPO}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"{'='*70}\n")

    start_time = datetime.now()

    try:
        # Download
        zip_buffer = download_repo_zip(GITHUB_API_URL)

        # Extract
        extracted_files = extract_zip_to_dir(zip_buffer, OUTPUT_DIR)

        elapsed = datetime.now() - start_time
        total_size_mb = sum(
            f.stat().st_size for f in OUTPUT_DIR.rglob("*") if f.is_file()
        ) / (1024 * 1024)

        file_counts = count_files_by_type(OUTPUT_DIR)

        # Save metadata
        metadata = {
            "source": "GitHub - SDG Classification Benchmark",
            "repository": f"https://github.com/{GITHUB_REPO}",
            "branch": "main",
            "fetched_at": start_time.isoformat(),
            "elapsed_seconds": elapsed.total_seconds(),
            "extracted_file_count": len(extracted_files),
            "file_types": file_counts,
            "total_size_mb": round(total_size_mb, 2),
            "output_dir": str(OUTPUT_DIR),
        }

        with open(METADATA_FILE, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"\n{'='*70}")
        print(f"✓ Successfully extracted repository")
        print(f"✓ Total files: {len(extracted_files)}")
        print(f"✓ File breakdown: {file_counts}")
        print(f"✓ Total size: {total_size_mb:.2f} MB")
        print(f"✓ Time elapsed: {elapsed.total_seconds():.1f}s")
        print(f"✓ Metadata saved to {METADATA_FILE}")
        print(f"{'='*70}\n")

    except Exception as e:
        print(f"\n✗ Error during fetch: {e}")
        raise


if __name__ == "__main__":
    main()

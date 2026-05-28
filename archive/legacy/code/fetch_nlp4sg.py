"""
Fetch NLP4SGPapers dataset from Hugging Face.

A dataset of 5,000 NLP papers from the ACL Anthology labeled for NLP4SG
(Natural Language Processing for Social Good), with SDG mappings.

Source: https://huggingface.co/datasets/feradauto/NLP4SGPapers
Paper: https://aclanthology.org/2023.findings-emnlp.31/
Website: https://nlp4sg.vercel.app

Output: data/nlp4sg/
        data/nlp4sg/train.json
        data/nlp4sg/validation.json
        data/nlp4sg/test.json
        data/nlp4sg/metadata.json

Run from project root:
    python code/fetch_nlp4sg.py

Requires: requests, pandas (optional)
"""

import json
from datetime import datetime
from pathlib import Path

import requests
from tqdm import tqdm

HF_DATASET_URL = "https://huggingface.co/datasets/feradauto/NLP4SGPapers"
OUTPUT_DIR = Path("data/nlp4sg")
METADATA_FILE = OUTPUT_DIR / "metadata.json"

SPLITS = ["train", "validation", "test"]


def download_json(url: str, output_path: Path) -> None:
    """Download a JSON file."""
    response = requests.get(url, stream=True, timeout=120)
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


def main() -> None:
    """Main fetch pipeline."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print("NLP4SGPapers Dataset Fetcher")
    print(f"{'='*70}")
    print(f"Source: {HF_DATASET_URL}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"{'='*70}\n")

    start_time = datetime.now()

    try:
        downloaded = []

        for split in SPLITS:
            output_path = OUTPUT_DIR / f"{split}.json"
            url = f"https://huggingface.co/datasets/feradauto/NLP4SGPapers/resolve/main/data/{split}.json"

            if output_path.exists():
                size_kb = output_path.stat().st_size / 1024
                print(f"  {split}.json already exists ({size_kb:.1f} KB)")
                downloaded.append({
                    "split": split,
                    "filename": f"{split}.json",
                    "status": "already_exists",
                    "size_kb": round(size_kb, 1),
                })
            else:
                print(f"Downloading: {split}.json")
                download_json(url, output_path)
                size_kb = output_path.stat().st_size / 1024
                downloaded.append({
                    "split": split,
                    "filename": f"{split}.json",
                    "status": "success",
                    "size_kb": round(size_kb, 1),
                })

        elapsed = datetime.now() - start_time

        metadata = {
            "source": "NLP4SGPapers",
            "huggingface_url": HF_DATASET_URL,
            "paper_url": "https://aclanthology.org/2023.findings-emnlp.31/",
            "website_url": "https://nlp4sg.vercel.app",
            "code_url": "https://github.com/feradauto/nlp4sg",
            "fetched_at": datetime.now().isoformat(),
            "elapsed_seconds": round(elapsed.total_seconds(), 1),
            "description": {
                "summary": "5,000 NLP papers from ACL Anthology labeled for NLP4SG",
                "splits": {
                    "train": "2,500 rows",
                    "validation": "500 rows",
                    "test": "2,000 rows",
                },
                "columns": [
                    "id", "url", "title", "abstract", "label_nlp4sg (bool)",
                    "task", "method",
                    "sdg1-sdg17 (bool) - SDG labels",
                    "goal1, goal2, goal3 - top 3 SDGs",
                    "acknowledgments", "year"
                ],
            },
            "citation": (
                "Fernandez, F., et al. (2023). NLP4SGPapers: A Scientific Dataset for "
                "Identifying NLP Papers Addressing Social Problems and UN SDGs. "
                "Findings of EMNLP 2023. https://aclanthology.org/2023.findings-emnlp.31/"
            ),
            "files": downloaded,
            "license": "cc-by-nc-sa-4.0",
            "notes": [
                "Directly maps NLP papers to SDGs - very relevant for alignment analysis",
                "3 tasks: (1) identify social impact papers, (2) map to SDGs, (3) identify task/method",
                "Covers entire ACL Anthology",
                "Website provides visualization workspace of NLP4SG landscape",
            ],
        }

        with open(METADATA_FILE, "w") as f:
            json.dump(metadata, f, indent=2)

        success_count = sum(1 for d in downloaded if d["status"] == "success")
        existing_count = sum(1 for d in downloaded if d["status"] == "already_exists")

        print(f"\n{'='*70}")
        print("Successfully processed NLP4SGPapers dataset")
        print(f"  Files: {success_count} downloaded, {existing_count} already existed")
        print(f"  Time: {elapsed.total_seconds():.1f}s")
        print(f"  Metadata: {METADATA_FILE}")
        print(f"{'='*70}\n")

    except Exception as e:
        print(f"\nError: {e}")
        raise


if __name__ == "__main__":
    main()

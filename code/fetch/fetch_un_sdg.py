"""
Fetch UN SDG-related policy documents and indicators.

Part 1: Official UN SDG Indicators (from UN Statistics API)
Part 2: Key UN/AI policy PDFs with text extraction

Output: data/raw/un_sdg/artifact/sdg_indicators.json
        data/raw/policy_scrape/pdfs/*.pdf
        data/raw/policy_scrape/texts/*.txt
        data/raw/un_sdg/artifact/metadata.json
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests
from tqdm import tqdm

# Try to import pdfplumber; warn if not available
try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False
    print("Warning: pdfplumber not installed. PDFs will be downloaded but not extracted.")

# Configuration
INDICATOR_DIR = Path("data/raw/un_sdg/artifact")
INDICATORS_FILE = INDICATOR_DIR / "sdg_indicators.json"
POLICY_DIR = Path("data/raw/policy_scrape")
PDFS_DIR = POLICY_DIR / "pdfs"
TEXTS_DIR = POLICY_DIR / "texts"
METADATA_FILE = INDICATOR_DIR / "metadata.json"

# UN SDG Indicators API
UN_INDICATORS_API = "https://unstats.un.org/sdgs/api/v1/sdg"

# Policy documents: hardcoded list of public PDFs
POLICY_DOCUMENTS = [
    {
        "name": "UN_AI_Strategy_Resource_Guide.pdf",
        "url": "https://sdgs.un.org/sites/default/files/2021-06/Resource%20Guide%20on%20AI%20Strategies_June%202021.pdf",
        "source": "UN SDGS - Resource Guide on AI Strategies",
    },
    {
        "name": "UN_AI_Advisory_Body_Interim_Report.pdf",
        "url": "https://www.un.org/digital-emerging-technologies/sites/www.un.org.techenvoy/files/ai_advisory_body_interim_report.pdf",
        "source": "UN Advisory Body - Governing AI for Humanity",
    },
    {
        "name": "PARIS21_AI_for_SDGs.pdf",
        "url": "https://www.paris21.org/sites/default/files/related_documents/2024-04/the-potential-of-ai-for-the-sdgs-and-official-stats_working-paper_0.pdf",
        "source": "PARIS21 - AI Potential for SDGs and Official Statistics",
    },
    {
        "name": "UN_DESA_Policy_Brief_AI_Risk.pdf",
        "url": "https://desapublications.un.org/policy-briefs/un-desa-policy-brief-no-174-leveraging-strategic-foresight-mitigate-artificial",
        "source": "UN DESA - Policy Brief No. 174: AI Risk Mitigation",
    },
]


def fetch_un_indicators() -> Optional[dict]:
    """Fetch official UN SDG indicators structure."""
    print("Fetching UN SDG Indicators from API...")
    try:
        response = requests.get(UN_INDICATORS_API, timeout=15)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"  Warning: Could not fetch indicators: {e}")
        return None


def download_pdf(url: str, output_path: Path) -> bool:
    """Download a PDF file with progress bar. Returns True if successful."""
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))

        with open(output_path, "wb") as f:
            with tqdm(
                total=total_size,
                unit="B",
                unit_scale=True,
                desc=f"Downloading {output_path.name}",
                leave=False,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

        return True
    except Exception as e:
        print(f"  ✗ Failed to download {output_path.name}: {e}")
        return False


def extract_pdf_text(pdf_path: Path) -> Optional[str]:
    """Extract text from PDF using pdfplumber."""
    if not HAS_PDFPLUMBER:
        return None

    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text() or ""
                text += "\n[PAGE BREAK]\n"
            return text
    except Exception as e:
        print(f"  Warning: Could not extract text from {pdf_path.name}: {e}")
        return None


def main():
    """Main fetch and extract pipeline."""
    INDICATOR_DIR.mkdir(parents=True, exist_ok=True)
    POLICY_DIR.mkdir(parents=True, exist_ok=True)
    PDFS_DIR.mkdir(parents=True, exist_ok=True)
    TEXTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print("UN SDG Policy Documents & Indicators Fetcher")
    print(f"{'='*70}")
    print(f"Indicators output: {INDICATORS_FILE}")
    print(f"Policy output: {POLICY_DIR}")
    print(f"{'='*70}\n")

    start_time = datetime.now()
    metadata = {
        "source": "UN Statistics Division & OECD/UN Policy Documents",
        "fetched_at": start_time.isoformat(),
        "indicators": None,
        "policy_documents": [],
    }

    # Part 1: Fetch UN Indicators
    print("Part 1: UN SDG Indicators")
    print("-" * 70)
    indicators = fetch_un_indicators()
    if indicators:
        with open(INDICATORS_FILE, "w") as f:
            json.dump(indicators, f, indent=2)
        print(f"✓ Saved SDG indicators to {INDICATORS_FILE}")
        metadata["indicators"] = {
            "file": str(INDICATORS_FILE),
            "structure": "Official UN SDG goals, targets, and indicators",
        }
    else:
        print("✗ Could not fetch indicators (continuing with PDFs)")

    # Part 2: Fetch Policy PDFs
    print("\nPart 2: Policy Documents")
    print("-" * 70)

    successful_downloads = 0
    extracted_texts = 0

    for doc in POLICY_DOCUMENTS:
        pdf_path = PDFS_DIR / doc["name"]
        text_path = TEXTS_DIR / doc["name"].replace(".pdf", ".txt")

        print(f"\nDocument: {doc['source']}")

        # Download PDF
        if pdf_path.exists():
            print(f"  (already downloaded, skipping)")
        else:
            if download_pdf(doc["url"], pdf_path):
                successful_downloads += 1
                print(f"  ✓ Downloaded")
            else:
                continue  # Skip text extraction if download failed

        # Extract text
        if text_path.exists():
            print(f"  (text already extracted, skipping)")
            extracted_texts += 1
        else:
            text = extract_pdf_text(pdf_path)
            if text:
                with open(text_path, "w", encoding="utf-8") as f:
                    f.write(text)
                extracted_texts += 1
                print(f"  ✓ Text extracted ({len(text)} chars)")
            elif HAS_PDFPLUMBER:
                print(f"  ⚠ Could not extract text from PDF")

        # Add to metadata
        metadata["policy_documents"].append({
            "name": doc["name"],
            "source": doc["source"],
            "url": doc["url"],
            "pdf_path": str(pdf_path.relative_to(POLICY_DIR)),
            "text_path": str(text_path.relative_to(POLICY_DIR)) if text_path.exists() else None,
        })

    elapsed = datetime.now() - start_time
    total_size_mb = sum(f.stat().st_size for f in POLICY_DIR.rglob("*") if f.is_file()) / (1024 * 1024)

    metadata["elapsed_seconds"] = elapsed.total_seconds()
    metadata["total_size_mb"] = round(total_size_mb, 2)
    metadata["successful_downloads"] = successful_downloads
    metadata["extracted_texts"] = extracted_texts

    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'='*70}")
    print(f"✓ Downloaded {successful_downloads}/{len(POLICY_DOCUMENTS)} PDFs")
    if HAS_PDFPLUMBER:
        print(f"✓ Extracted text from {extracted_texts} PDFs")
    print(f"✓ Total size: {total_size_mb:.2f} MB")
    print(f"✓ Time elapsed: {elapsed.total_seconds():.1f}s")
    print(f"✓ Metadata saved to {METADATA_FILE}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()

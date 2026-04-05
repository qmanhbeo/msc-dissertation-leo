"""
Fetch expanded policy document corpus for dissertation.

Downloads PDFs from diverse international sources:
- UN family (SDG progress reports, AI advisory body)
- IPCC (climate policy)
- National AI strategies (UK, India, Singapore, US, AU)
- Regional frameworks (African Union)
- Intergovernmental (UNESCO, OECD)

Graceful failure: logs errors and continues — a 403/404 does not abort the run.

Output: data/policy_expanded/pdfs/<name>.pdf
        data/policy_expanded/texts/<name>.txt
        data/policy_expanded/metadata.json

Run from project root:
    python code/fetch_policy_expanded.py
"""

import json
from datetime import datetime
from pathlib import Path

import requests
from tqdm import tqdm

try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False
    print("Warning: pdfplumber not installed. PDFs cannot be extracted.")

OUTPUT_DIR = Path("data/policy_expanded")
PDFS_DIR = OUTPUT_DIR / "pdfs"
TEXTS_DIR = OUTPUT_DIR / "texts"
METADATA_FILE = OUTPUT_DIR / "metadata.json"

# ---------------------------------------------------------------------------
# Document list — confirmed accessible first, then try-or-skip
# ---------------------------------------------------------------------------
DOCUMENTS = [
    # --- Already downloaded successfully ---
    {
        "name": "UN_SDG_Progress_Report_2023",
        "url": "https://unstats.un.org/sdgs/report/2023/The-Sustainable-Development-Goals-Report-2023.pdf",
        "institution": "UN Statistics Division",
        "type": "SDG progress",
        "year": 2023,
        "confirmed": True,
    },
    {
        "name": "IPCC_AR6_Summary_for_Policymakers",
        "url": "https://www.ipcc.ch/report/ar6/syr/downloads/report/IPCC_AR6_SYR_SPM.pdf",
        "institution": "IPCC",
        "type": "climate policy",
        "year": 2023,
        "confirmed": True,
    },
    {
        "name": "UK_National_AI_Strategy_2021",
        "url": "https://assets.publishing.service.gov.uk/government/uploads/system/uploads/attachment_data/file/1020402/National_AI_Strategy_-_PDF_version.pdf",
        "institution": "UK Government",
        "type": "national AI strategy",
        "year": 2021,
        "confirmed": True,
    },
    # --- Updated URLs (verified via search) ---
    {
        "name": "UN_AI_Advisory_Body_Final_Report_2024",
        "url": "https://www.un.org/sites/un2.un.org/files/governing_ai_for_humanity_final_report_en.pdf",
        "institution": "UN AI Advisory Body",
        "type": "AI governance",
        "year": 2024,
        "confirmed": True,
    },
    {
        "name": "Singapore_National_AI_Strategy_2.0",
        "url": "https://file.go.gov.sg/nais2023.pdf",
        "institution": "Singapore MDDI",
        "type": "national AI strategy",
        "year": 2023,
        "confirmed": True,
    },
    {
        "name": "African_Union_Continental_AI_Strategy_2024",
        "url": "https://au.int/sites/default/files/documents/44004-doc-EN-_Continental_AI_Strategy_July_2024.pdf",
        "institution": "African Union",
        "type": "regional AI framework",
        "year": 2024,
        "confirmed": True,
    },
    {
        "name": "Germany_AI_Strategy_2020_Update",
        "url": "https://www.ki-strategie-deutschland.de/files/downloads/Fortschreibung_KI-Strategie_engl.pdf",
        "institution": "German Federal Government",
        "type": "national AI strategy",
        "year": 2020,
        "confirmed": True,
    },
    {
        "name": "UNESCO_Ethics_of_AI_2021",
        "url": "https://www.ohchr.org/sites/default/files/2022-03/UNESCO.pdf",
        "institution": "UNESCO",
        "type": "AI ethics",
        "year": 2021,
        "confirmed": True,
    },
    # --- Try or skip ---
    {
        "name": "US_Blueprint_AI_Bill_of_Rights",
        "url": "https://www.managementsolutions.com/sites/default/files/publicaciones/eng/blueprint-for-an-ai-bill-of-rights.pdf",
        "institution": "White House OSTP",
        "type": "national AI policy",
        "year": 2022,
        "confirmed": False,
    },
    {
        "name": "India_Responsible_AI_NITI_Aayog_2021",
        "url": "https://indiaai.gov.in/documents/pdf/RaiPolicyDocument.pdf",
        "institution": "NITI Aayog (India)",
        "type": "national AI strategy",
        "year": 2021,
        "confirmed": False,
    },
    {
        "name": "EU_AI_Ethics_Guidelines_HLEG",
        "url": "https://www.europarl.europa.eu/cmsdata/196377/AI%20HLEG_Ethics%20Guidelines%20for%20Trustworthy%20AI.pdf",
        "institution": "EU High-Level Expert Group on AI",
        "type": "AI ethics guidelines",
        "year": 2019,
        "confirmed": False,
    },
    {
        "name": "OECD_AI_Recommendation_2019",
        "url": "https://wecglobal.org/uploads/2019/07/2019_OECD_Recommendations-AI.pdf",
        "institution": "OECD",
        "type": "AI principles",
        "year": 2019,
        "confirmed": False,
    },
    {
        "name": "UNDP_Trustworthy_AI_2023",
        "url": "https://www.undp.org/sites/g/files/zskgke326/files/2023-11/UNDP_Trustworthy_AI_report_0.pdf",
        "institution": "UNDP",
        "type": "AI development",
        "year": 2023,
        "confirmed": False,
    },
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; dissertation-research-bot/1.0; "
        "academic use only)"
    )
}


def download_pdf(url: str, output_path: Path) -> bool:
    """Download PDF. Returns True on success, False on any failure."""
    try:
        response = requests.get(
            url, stream=True, timeout=60, headers=HEADERS, allow_redirects=True
        )
        if response.status_code == 403:
            print(f"  ✗ 403 Forbidden — skipping")
            return False
        if response.status_code == 404:
            print(f"  ✗ 404 Not Found — skipping")
            return False
        response.raise_for_status()

        content_type = response.headers.get("content-type", "")
        if "html" in content_type and "pdf" not in content_type:
            print(f"  ✗ Response is HTML not PDF — skipping")
            return False

        total_size = int(response.headers.get("content-length", 0))
        with open(output_path, "wb") as f:
            with tqdm(
                total=total_size, unit="B", unit_scale=True,
                desc=f"  Downloading", leave=False,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

        # Sanity check: must be at least 10KB to be a real PDF
        if output_path.stat().st_size < 10_000:
            print(f"  ✗ Downloaded file too small ({output_path.stat().st_size} bytes) — likely an error page")
            output_path.unlink()
            return False

        return True

    except Exception as e:
        print(f"  ✗ Download error: {e}")
        if output_path.exists():
            output_path.unlink()
        return False


def extract_text(pdf_path: Path) -> str | None:
    """Extract text from PDF using pdfplumber."""
    if not HAS_PDFPLUMBER:
        return None
    try:
        with pdfplumber.open(pdf_path) as pdf:
            pages = []
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    pages.append(text)
            return "\n[PAGE BREAK]\n".join(pages) if pages else None
    except Exception as e:
        print(f"  ✗ Text extraction failed: {e}")
        return None


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PDFS_DIR.mkdir(parents=True, exist_ok=True)
    TEXTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print("Expanded Policy Document Fetcher")
    print(f"{'='*70}")
    print(f"Documents to attempt: {len(DOCUMENTS)} ({sum(1 for d in DOCUMENTS if d['confirmed'])} confirmed, {sum(1 for d in DOCUMENTS if not d['confirmed'])} try-or-skip)")
    print(f"{'='*70}\n")

    start_time = datetime.now()
    results = []

    for doc in DOCUMENTS:
        pdf_path = PDFS_DIR / f"{doc['name']}.pdf"
        txt_path = TEXTS_DIR / f"{doc['name']}.txt"
        status = {"name": doc["name"], "institution": doc["institution"],
                  "type": doc["type"], "year": doc["year"],
                  "url": doc["url"], "confirmed": doc["confirmed"],
                  "downloaded": False, "text_extracted": False,
                  "text_chars": 0, "error": None}

        print(f"\n[{doc['institution']}] {doc['name']}")

        # Download
        if txt_path.exists():
            print(f"  (already processed — skipping)")
            status["downloaded"] = True
            status["text_extracted"] = True
            status["text_chars"] = txt_path.stat().st_size
            results.append(status)
            continue

        if not pdf_path.exists():
            if not download_pdf(doc["url"], pdf_path):
                status["error"] = "download failed"
                results.append(status)
                continue
        else:
            print(f"  (PDF already downloaded)")

        status["downloaded"] = True
        print(f"  ✓ Downloaded ({pdf_path.stat().st_size / 1024:.0f} KB)")

        # Extract text
        text = extract_text(pdf_path)
        if text:
            txt_path.write_text(text, encoding="utf-8")
            status["text_extracted"] = True
            status["text_chars"] = len(text)
            print(f"  ✓ Text extracted ({len(text):,} chars)")
        else:
            status["error"] = "text extraction failed"
            print(f"  ✗ Could not extract text")

        results.append(status)

    # Summary
    elapsed = datetime.now() - start_time
    n_success = sum(1 for r in results if r["text_extracted"])
    n_fail = len(results) - n_success

    metadata = {
        "fetched_at": start_time.isoformat(),
        "elapsed_seconds": elapsed.total_seconds(),
        "total_attempted": len(DOCUMENTS),
        "total_success": n_success,
        "total_failed": n_fail,
        "documents": results,
    }

    METADATA_FILE.write_text(json.dumps(metadata, indent=2))

    print(f"\n{'='*70}")
    print(f"✓ Successfully fetched: {n_success}/{len(DOCUMENTS)} documents")
    if n_fail:
        failed = [r["name"] for r in results if not r["text_extracted"]]
        print(f"✗ Failed ({n_fail}): {', '.join(failed)}")
    print(f"✓ Elapsed: {elapsed.total_seconds():.1f}s")
    print(f"✓ Metadata saved to {METADATA_FILE}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()

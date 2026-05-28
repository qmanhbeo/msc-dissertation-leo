"""
Fetch UN SDG publications from sdgs.un.org.

Source: https://sdgs.un.org/
Approach: Parse sitemap to get publication URLs, then scrape each for PDF links.

Output: data/sdg_publications/
        data/sdg_publications/metadata.json
        data/sdg_publications/urls.json       — discovered publication URLs
        data/sdg_publications/pdfs/          — downloaded PDFs

Note: The site is JavaScript-heavy (Drupal). We use the sitemap as entry point
to discover publication URLs, then scrape for PDF download links.

Run from project root:
    python code/fetch_sdg_publications.py

Requires: requests, beautifulsoup4, pdfplumber (optional, for text extraction)
"""

import json
import logging
import os
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SITE_BASE = "https://sdgs.un.org"
SITEMAP_URL = f"{SITE_BASE}/sitemap.xml"
OUTPUT_DIR = Path("data/sdg_publications")
PDF_DIR = OUTPUT_DIR / "pdfs"
METADATA_FILE = OUTPUT_DIR / "metadata.json"
URLS_FILE = OUTPUT_DIR / "urls.json"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; DissertationResearchBot/1.0; research project)",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

REQUEST_TIMEOUT = 30
DELAY_BETWEEN_REQUESTS = 1.0  # Be respectful to the server

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def get_session() -> requests.Session:
    """Create a requests session with retry logic."""
    session = requests.Session()
    session.headers.update(HEADERS)
    return session


def fetch_sitemap(session: requests.Session) -> list[str]:
    """Fetch and parse the sitemap to get all URLs (all pages)."""
    log.info("Fetching sitemap from %s", SITEMAP_URL)

    all_urls = []

    for page in range(1, 10):
        page_url = f"{SITEMAP_URL}?page={page}"
        log.info("Fetching sitemap page %d...", page)

        try:
            response = session.get(page_url, timeout=REQUEST_TIMEOUT)
            if response.status_code == 404:
                log.info("No more sitemap pages after page %d", page - 1)
                break
            response.raise_for_status()
        except requests.RequestException as e:
            log.error("Failed to fetch sitemap page %d: %s", page, e)
            break

        root = ET.fromstring(response.content)

        namespaces = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
        page_urls = []
        for url in root.findall(".//sm:loc", namespaces):
            loc = url.text
            if loc:
                page_urls.append(loc)

        if not page_urls:
            log.info("Empty sitemap page %d, stopping", page)
            break

        all_urls.extend(page_urls)
        log.info("  Found %d URLs on page %d", len(page_urls), page)

    log.info("Found %d total URLs in sitemap", len(all_urls))
    return all_urls


def filter_publication_urls(urls: list[str]) -> list[str]:
    """Filter URLs to identify publication pages."""
    publication_patterns = [
        "/publications/",
        "/gsdr",
        "/2026/",
        "/2025/",
        "/2024/",
        "/2023/",
        "/2022/",
        "/2021/",
        "/2020/",
        "/2019/",
        "/2018/",
        "/2017/",
        "/2016/",
    ]

    publication_urls = []
    for url in urls:
        parsed = urlparse(url)
        path = parsed.path.lower()

        if path.endswith(".pdf"):
            continue

        if any(pattern.lower() in path for pattern in publication_patterns):
            if "/ar/" not in path and "/zh/" not in path and "/fr/" not in path and "/ru/" not in path and "/es/" not in path:
                publication_urls.append(url)

    log.info("Filtered to %d publication URLs (English only)", len(publication_urls))
    return publication_urls


def find_pdf_links(session: requests.Session, url: str) -> list[dict]:
    """Scrape a publication page to find PDF download links."""
    try:
        response = session.get(url, timeout=REQUEST_TIMEOUT)
        if response.status_code != 200:
            return []
    except requests.RequestException:
        return []

    soup = BeautifulSoup(response.content, "html.parser")

    pdf_links = []

    for link in soup.find_all("a", href=True):
        href = link["href"]

        if href.endswith(".pdf") or ".pdf?" in href:
            full_url = urljoin(url, href)
            text = link.get_text(strip=True) or "Download PDF"

            pdf_links.append({
                "url": full_url,
                "label": text,
            })

    for link in soup.find_all("a", href=True):
        href = link["href"]
        text = (link.get_text(strip=True) or "").lower()

        if any(kw in text for kw in ["download", "pdf", "read", "report", "publication"]):
            if href.startswith("/"):
                full_url = urljoin(SITE_BASE, href)
            elif href.startswith("http"):
                full_url = href
            else:
                continue

            if full_url not in [p["url"] for p in pdf_links]:
                pdf_links.append({
                    "url": full_url,
                    "label": text,
                })

    return pdf_links


def download_pdf(session: requests.Session, pdf_info: dict, country: str = "") -> dict:
    """Download a PDF file."""
    url = pdf_info["url"]
    filename = url.split("/")[-1].split("?")[0]

    if not filename.endswith(".pdf"):
        filename += ".pdf"

    if country:
        safe_country = re.sub(r"[^a-zA-Z0-9_-]", "_", country)
        filename = f"{safe_country}_{filename}"

    output_path = PDF_DIR / filename

    if output_path.exists():
        return {"status": "already_exists", "path": str(output_path), "url": url}

    try:
        response = session.get(url, timeout=REQUEST_TIMEOUT, stream=True)
        response.raise_for_status()

        content_type = response.headers.get("Content-Type", "")
        if "pdf" not in content_type.lower() and not url.endswith(".pdf"):
            return {"status": "not_pdf", "url": url}

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        size_mb = output_path.stat().st_size / (1024 * 1024)
        return {"status": "success", "path": str(output_path), "size_mb": size_mb, "url": url}

    except requests.RequestException as e:
        return {"status": "error", "error": str(e), "url": url}


def save_urls(publication_urls: list[str], pdf_discoveries: list[dict]) -> None:
    """Save discovered URLs to file."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = {
        "publication_urls": publication_urls,
        "pdf_discoveries": pdf_discoveries,
        "scraped_at": datetime.now().isoformat(),
    }

    with URLS_FILE.open("w") as f:
        json.dump(data, f, indent=2)

    log.info("Saved %d publication URLs and %d PDFs to %s",
             len(publication_urls), len(pdf_discoveries), URLS_FILE)


def main() -> None:
    """Main scraping pipeline."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PDF_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print("UN SDG Publications Scraper")
    print(f"{'='*70}")
    print(f"Source: {SITE_BASE}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"{'='*70}\n")

    start_time = datetime.now()
    session = get_session()

    sitemap_urls = fetch_sitemap(session)
    if not sitemap_urls:
        log.error("No URLs found in sitemap. Aborting.")
        return

    publication_urls = filter_publication_urls(sitemap_urls)
    if not publication_urls:
        log.warning("No publication URLs found.")
        return

    existing_urls = set()
    if URLS_FILE.exists():
        with URLS_FILE.open() as f:
            existing_data = json.load(f)
            existing_urls = set(existing_data.get("publication_urls", []))
            log.info("Found %d previously scraped URLs", len(existing_urls))

    new_urls = [u for u in publication_urls if u not in existing_urls]
    urls_to_scrape = new_urls if new_urls else publication_urls[:10]

    log.info("Scraping %d publication pages for PDF links...", len(urls_to_scrape))

    pdf_discoveries = []
    scraped_count = 0

    for i, url in enumerate(urls_to_scrape, 1):
        if i % 20 == 0:
            log.info("Progress: %d/%d pages scraped", i, len(urls_to_scrape))

        log.debug("Scraping: %s", url)
        pdf_links = find_pdf_links(session, url)

        if pdf_links:
            for pdf in pdf_links:
                pdf_discoveries.append({
                    "source_url": url,
                    "pdf_url": pdf["url"],
                    "label": pdf["label"],
                })

        scraped_count += 1
        time.sleep(DELAY_BETWEEN_REQUESTS)

    log.info("Scraped %d pages, found %d PDF links", scraped_count, len(pdf_discoveries))

    save_urls(publication_urls, pdf_discoveries)

    unique_pdfs = list({p["pdf_url"]: p for p in pdf_discoveries}.values())
    log.info("Unique PDFs: %d", len(unique_pdfs))

    log.info("\nDownloading PDFs (first 50 to avoid excessive bandwidth)...")
    downloaded = []
    for i, pdf in enumerate(unique_pdfs[:50], 1):
        log.info("Downloading %d/%d: %s", i, min(50, len(unique_pdfs)), pdf["pdf_url"][-60:])
        result = download_pdf(session, {"url": pdf["pdf_url"], "label": pdf.get("label", "")})
        downloaded.append(result)
        time.sleep(DELAY_BETWEEN_REQUESTS)

    elapsed = datetime.now() - start_time
    success_count = sum(1 for d in downloaded if d["status"] == "success")
    existing_count = sum(1 for d in downloaded if d["status"] == "already_exists")

    metadata = {
        "source": "sdgs.un.org",
        "url": SITE_BASE,
        "scraped_at": datetime.now().isoformat(),
        "elapsed_seconds": round(elapsed.total_seconds(), 1),
        "sitemap_urls_found": len(sitemap_urls),
        "publication_urls": len(publication_urls),
        "pdf_links_found": len(pdf_discoveries),
        "unique_pdfs": len(unique_pdfs),
        "pdfs_downloaded": len(downloaded),
        "downloads": {
            "success": success_count,
            "already_exists": existing_count,
            "failed": len(downloaded) - success_count - existing_count,
        },
        "notes": [
            "Site is JavaScript-heavy (Drupal). Using sitemap for URL discovery.",
            "PDFs are in data/sdg_publications/pdfs/",
            "Full URL list saved to data/sdg_publications/urls.json",
            "Only first 50 PDFs downloaded to save bandwidth.",
            "Re-run to download more PDFs.",
        ],
    }

    with METADATA_FILE.open("w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'='*70}")
    print("✓ Scraping complete")
    print(f"✓ Sitemap URLs: {len(sitemap_urls)}")
    print(f"✓ Publication pages: {len(publication_urls)}")
    print(f"✓ PDF links found: {len(pdf_discoveries)}")
    print(f"✓ PDFs downloaded: {success_count} (plus {existing_count} already existed)")
    print(f"✓ Time: {elapsed.total_seconds():.1f}s")
    print(f"✓ Metadata: {METADATA_FILE}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()

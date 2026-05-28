"""
Fetch SDG metadata and reports from UN Statistics Division (UNSD).

Source: https://unstats.un.org/UNSDWebsite/
API: https://unstats.un.org/SDGAPI/

This script fetches:
1. SDG Goal, Target, Indicator, Series metadata from the official API
2. Geographic areas (countries, regions, groupings)
3. Global Indicator Framework Excel (official SDG indicator definitions)
4. Key reports (Statistical Annex PDFs, SG Progress Reports)

Output: data/unsd/
        data/unsd/goals.json         — 17 SDG goal definitions
        data/unsd/targets.json      — 169 targets
        data/unsd/indicators.json   — 234 indicators
        data/unsd/series.json       — 713 indicator series
        data/unsd/geoareas.json    — 460 geographic areas
        data/unsd/indicator_framework.xlsx — Official indicator definitions
        data/unsd/reports/         — Downloaded PDFs

Run from project root:
    python code/fetch_unsd.py

Requires: requests, pandas, openpyxl
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path

import requests

API_BASE = "https://unstats.un.org/SDGAPI/v1/sdg"
OUTPUT_DIR = Path("data/unsd")
REPORTS_DIR = OUTPUT_DIR / "reports"
METADATA_FILE = OUTPUT_DIR / "metadata.json"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; DissertationResearchBot/1.0; research project)",
    "Accept": "application/json",
}

REQUEST_TIMEOUT = 30
DELAY_BETWEEN_REQUESTS = 0.5


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def get_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(HEADERS)
    return session


def fetch_json(session: requests.Session, endpoint: str) -> list | dict | None:
    url = f"{API_BASE}/{endpoint}"
    try:
        response = session.get(url, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        log.error("Failed to fetch %s: %s", endpoint, e)
        return None


def fetch_all_goals(session: requests.Session) -> list[dict]:
    log.info("Fetching SDG Goals...")
    goals = fetch_json(session, "Goal/List")
    if goals:
        log.info("  Found %d goals", len(goals))
    return goals or []


def fetch_all_targets(session: requests.Session) -> list[dict]:
    log.info("Fetching SDG Targets...")
    targets = fetch_json(session, "Target/List")
    if targets:
        log.info("  Found %d targets", len(targets))
    return targets or []


def fetch_all_indicators(session: requests.Session) -> list[dict]:
    log.info("Fetching SDG Indicators...")
    indicators = fetch_json(session, "Indicator/List")
    if indicators:
        log.info("  Found %d indicators", len(indicators))
    return indicators or []


def fetch_all_series(session: requests.Session) -> list[dict]:
    log.info("Fetching SDG Series...")
    series = fetch_json(session, "Series/List")
    if series:
        log.info("  Found %d series", len(series))
    return series or []


def fetch_geoareas(session: requests.Session) -> list[dict]:
    log.info("Fetching Geographic Areas...")
    geoareas = fetch_json(session, "GeoArea/List")
    if geoareas:
        log.info("  Found %d geographic areas", len(geoareas))
    return geoareas or []


def download_indicator_framework(session: requests.Session) -> dict:
    url = "https://unstats.un.org/sdgs/indicators/Global-Indicator-Framework-after-2026-refinement-English.xlsx"
    output_path = OUTPUT_DIR / "indicator_framework.xlsx"
    
    if output_path.exists():
        size_mb = output_path.stat().st_size / (1024 * 1024)
        log.info("Indicator framework already exists (%.2f MB)", size_mb)
        return {"status": "already_exists", "path": str(output_path), "size_mb": size_mb}
    
    log.info("Downloading Global Indicator Framework...")
    download_headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "*/*",
    }
    try:
        response = session.get(url, headers=download_headers, timeout=REQUEST_TIMEOUT, stream=True)
        response.raise_for_status()
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        size_mb = output_path.stat().st_size / (1024 * 1024)
        log.info("  Downloaded: %.2f MB", size_mb)
        return {"status": "success", "path": str(output_path), "size_mb": size_mb}
    except requests.RequestException as e:
        log.error("  Failed: %s", e)
        return {"status": "error", "error": str(e)}


def download_reports(session: requests.Session) -> list[dict]:
    reports = [
        {
            "year": 2025,
            "type": "statistical_annex",
            "url": "https://unstats.un.org/sdgs/files/report/2025/E_2025_62_Statistical_Annex_I_and_II.pdf",
            "filename": "SDG_Statistical_Annex_2025.pdf",
        },
        {
            "year": 2024,
            "type": "statistical_annex", 
            "url": "https://unstats.un.org/sdgs/files/report/2024/E_2024_54_Statistical_Annex_I_and_II.pdf",
            "filename": "SDG_Statistical_Annex_2024.pdf",
        },
        {
            "year": 2025,
            "type": "sg_report",
            "url": "https://unstats.un.org/sdgs/files/report/2025/secretary-general-sdg-report-2025--EN.pdf",
            "filename": "SG_SDG_Report_2025.pdf",
        },
        {
            "year": 2024,
            "type": "sg_report",
            "url": "https://unstats.un.org/sdgs/files/report/2024/secretary-general-sdg-report-2024--EN.pdf",
            "filename": "SG_SDG_Report_2024.pdf",
        },
    ]
    
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    
    download_headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "*/*",
    }
    
    for report in reports:
        output_path = REPORTS_DIR / report["filename"]
        
        if output_path.exists():
            size_mb = output_path.stat().st_size / (1024 * 1024)
            log.info("  %s already exists (%.2f MB)", report["filename"], size_mb)
            results.append({
                **report,
                "status": "already_exists",
                "path": str(output_path),
                "size_mb": size_mb
            })
            continue
        
        log.info("  Downloading: %s", report["filename"])
        try:
            response = session.get(report["url"], headers=download_headers, timeout=120, stream=True)
            response.raise_for_status()
            
            with output_path.open("wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            size_mb = output_path.stat().st_size / (1024 * 1024)
            log.info("    Downloaded: %.2f MB", size_mb)
            results.append({
                **report,
                "status": "success",
                "path": str(output_path),
                "size_mb": size_mb
            })
            time.sleep(DELAY_BETWEEN_REQUESTS)
        except requests.RequestException as e:
            log.error("    Failed: %s", e)
            results.append({
                **report,
                "status": "error",
                "error": str(e)
            })
    
    return results


def save_json(data: list | dict, filepath: Path) -> None:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with filepath.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    log.info("  Saved: %s", filepath)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print("UNSD SDG Data Fetcher")
    print(f"{'='*70}")
    print(f"Source: https://unstats.un.org/UNSDWebsite/")
    print(f"API: https://unstats.un.org/SDGAPI/")
    print(f"Output: {OUTPUT_DIR}")
    print(f"{'='*70}\n")
    
    start_time = datetime.now()
    session = get_session()
    
    goals = fetch_all_goals(session)
    if goals:
        save_json(goals, OUTPUT_DIR / "goals.json")
    
    targets = fetch_all_targets(session)
    if targets:
        save_json(targets, OUTPUT_DIR / "targets.json")
    
    indicators = fetch_all_indicators(session)
    if indicators:
        save_json(indicators, OUTPUT_DIR / "indicators.json")
    
    series = fetch_all_series(session)
    if series:
        save_json(series, OUTPUT_DIR / "series.json")
    
    geoareas = fetch_geoareas(session)
    if geoareas:
        save_json(geoareas, OUTPUT_DIR / "geoareas.json")
    
    framework_result = download_indicator_framework(session)
    
    print("\nDownloading reports...")
    reports_results = download_reports(session)
    
    elapsed = datetime.now() - start_time
    
    metadata = {
        "source": "UN Statistics Division (UNSD)",
        "source_url": "https://unstats.un.org/UNSDWebsite/",
        "api_url": "https://unstats.un.org/SDGAPI/",
        "fetched_at": datetime.now().isoformat(),
        "elapsed_seconds": round(elapsed.total_seconds(), 1),
        "counts": {
            "goals": len(goals),
            "targets": len(targets),
            "indicators": len(indicators),
            "series": len(series),
            "geoareas": len(geoareas),
        },
        "indicator_framework": framework_result,
        "reports": reports_results,
        "notes": [
            "Goals: Official SDG goal definitions (17 goals)",
            "Targets: SDG targets mapped to goals (169 targets)",
            "Indicators: Official indicator list with tier classifications",
            "Series: 713 indicator series with metadata",
            "GeoAreas: 460 geographic areas (countries, regions, groupings)",
            "Indicator Framework: Official Excel with all indicator definitions",
            "Reports: Statistical Annex and SG Progress Reports",
        ],
    }
    
    with METADATA_FILE.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    success_reports = sum(1 for r in reports_results if r["status"] == "success")
    existing_reports = sum(1 for r in reports_results if r["status"] == "already_exists")
    
    print(f"\n{'='*70}")
    print("Fetch complete")
    print(f"  Goals: {len(goals)}")
    print(f"  Targets: {len(targets)}")
    print(f"  Indicators: {len(indicators)}")
    print(f"  Series: {len(series)}")
    print(f"  GeoAreas: {len(geoareas)}")
    print(f"  Reports: {success_reports} downloaded, {existing_reports} already existed")
    print(f"  Time: {elapsed.total_seconds():.1f}s")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()

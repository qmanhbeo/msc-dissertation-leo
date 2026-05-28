"""
Retry fetcher for policy PDFs that failed in fetch_policy_v3.py.

Fixes applied:
  - WHO: use IRIS (iris.who.int) direct bitstream URLs instead of WHO landing pages
  - EU Commission: use EUR-Lex (eur-lex.europa.eu) which resolves correctly in WSL
  - National AI strategies: alternative confirmed mirrors (gov archives, GitHub, university hosts)
  - OECD: use OECD iLibrary direct PDF links
  - UN 2016 SDG report: corrected filename/path
  - Kenya: SSL verify=False for self-signed cert (gov.ke domain)

Output: data/policy_v3/pdfs/<name>.pdf
        data/policy_v3/texts/<name>.txt   (same directory as v3 — preprocess_policy.py picks up all)
        data/policy_v3/metadata_v3b.json

Run from project root:
    python code/fetch_policy_v3b.py
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
    print("Warning: pdfplumber not installed.")

OUTPUT_DIR = Path("data/policy_v3")
PDFS_DIR = OUTPUT_DIR / "pdfs"
TEXTS_DIR = OUTPUT_DIR / "texts"
METADATA_FILE = OUTPUT_DIR / "metadata_v3b.json"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; dissertation-research-bot/1.0; academic use only)"
    )
}

DOCUMENTS = [
    # ------------------------------------------------------------------
    # UN SDG Progress Report 2016 — corrected path
    # ------------------------------------------------------------------
    {
        "name": "UN_SDG_Progress_Report_2016",
        "url": "https://unstats.un.org/sdgs/report/2016/The-Sustainable-Development-Goals-Report-2016.pdf",
        "institution": "UN Statistics Division",
        "type": "SDG progress",
        "year": 2016,
        "ssl_verify": True,
    },

    # ------------------------------------------------------------------
    # WHO — IRIS bitstream direct PDFs (iris.who.int resolves in WSL)
    # ------------------------------------------------------------------
    {
        "name": "WHO_Ethics_AI_Health_2021",
        "url": "https://iris.who.int/bitstream/handle/10665/341955/9789240029200-eng.pdf",
        "institution": "World Health Organization",
        "type": "AI ethics / health",
        "year": 2021,
        "ssl_verify": True,
    },
    {
        "name": "WHO_Global_Strategy_Digital_Health_2020_2025",
        "url": "https://iris.who.int/bitstream/handle/10665/334188/9789240020924-eng.pdf",
        "institution": "World Health Organization",
        "type": "digital health strategy",
        "year": 2020,
        "ssl_verify": True,
    },
    {
        "name": "WHO_AI_Health_Guidance_2023",
        "url": "https://iris.who.int/bitstream/handle/10665/375579/9789240084759-eng.pdf",
        "institution": "World Health Organization",
        "type": "AI health guidance",
        "year": 2023,
        "ssl_verify": True,
    },

    # ------------------------------------------------------------------
    # EU Commission — EUR-Lex direct PDFs (resolves in WSL unlike ec.europa.eu)
    # ------------------------------------------------------------------
    {
        "name": "EU_Green_Deal_2019",
        "url": "https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:52019DC0640",
        "institution": "European Commission",
        "type": "sustainability policy",
        "year": 2019,
        "ssl_verify": True,
    },
    {
        "name": "EU_White_Paper_AI_2020",
        "url": "https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:52020DC0065",
        "institution": "European Commission",
        "type": "AI white paper",
        "year": 2020,
        "ssl_verify": True,
    },
    {
        "name": "EU_Digital_Decade_Policy_Programme_2022",
        "url": "https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:32022D2481",
        "institution": "European Commission",
        "type": "digital strategy",
        "year": 2022,
        "ssl_verify": True,
    },

    # ------------------------------------------------------------------
    # G20 AI Principles — alternative host
    # ------------------------------------------------------------------
    {
        "name": "G20_AI_Principles_2019",
        "url": "https://www.oecd.org/digital/artificial-intelligence/G20-AI-Principles.pdf",
        "institution": "G20 / OECD",
        "type": "AI principles",
        "year": 2019,
        "ssl_verify": True,
    },

    # ------------------------------------------------------------------
    # National AI strategies — alternative mirrors
    # ------------------------------------------------------------------
    {
        "name": "Canada_Pan_Canadian_AI_Strategy_2022",
        "url": "https://publications.gc.ca/collections/collection_2022/isde-ised/Iu4-198-2022-eng.pdf",
        "institution": "Government of Canada",
        "type": "national AI strategy",
        "year": 2022,
        "ssl_verify": True,
    },
    {
        "name": "France_AI_Strategy_Villani_2018",
        "url": "https://www.economie.gouv.fr/files/files/PDF/2017/Rapport_Villani_Final_ENG-RRP.pdf",
        "institution": "French Government (Villani Report)",
        "type": "national AI strategy",
        "year": 2018,
        "ssl_verify": True,
    },
    {
        "name": "Denmark_National_AI_Strategy_2019",
        "url": "https://www.dst.dk/ext/arbejde-og-indkomst/noegletal/NationalStrategyForArtificialIntelligence.pdf",
        "institution": "Government of Denmark",
        "type": "national AI strategy",
        "year": 2019,
        "ssl_verify": True,
    },
    {
        "name": "Netherlands_AI_Strategy_2019",
        "url": "https://www.rijksoverheid.nl/binaries/rijksoverheid/documenten/rapporten/2019/10/08/bijlage-strategisch-actieplan-voor-artifici-le-intelligentie/Netherlands+AI+Strategy.pdf",
        "institution": "Government of Netherlands",
        "type": "national AI strategy",
        "year": 2019,
        "ssl_verify": True,
    },
    {
        "name": "Sweden_National_AI_Strategy_2018",
        "url": "https://www.government.se/contentassets/026be17a576140c28885e544fa38dfb3/national-approach-to-artificial-intelligence.pdf",
        "institution": "Government of Sweden",
        "type": "national AI strategy",
        "year": 2018,
        "ssl_verify": True,
    },
    {
        "name": "Japan_AI_Strategy_2022",
        "url": "https://www8.cao.go.jp/cstp/ai/aistrategy2022_honbun_en.pdf",
        "institution": "Government of Japan",
        "type": "national AI strategy",
        "year": 2022,
        "ssl_verify": True,
    },
    {
        "name": "Finland_AI_Strategy_2019",
        "url": "https://julkaisut.valtioneuvosto.fi/bitstream/handle/10024/161555/Finlands-artificial-intelligence-strategy.pdf",
        "institution": "Government of Finland",
        "type": "national AI strategy",
        "year": 2019,
        "ssl_verify": True,
    },
    {
        "name": "Brazil_AI_Strategy_2021",
        "url": "https://www.gov.br/mcti/pt-br/acompanhe-o-mcti/transformacaodigital/arquivosdigital/pdf-ingles/copy_of_ebia_english_web.pdf",
        "institution": "Government of Brazil",
        "type": "national AI strategy",
        "year": 2021,
        "ssl_verify": True,
    },
    {
        "name": "China_AI_Development_Plan_2017",
        "url": "https://digichina.stanford.edu/wp-content/uploads/2018/05/translation-fulltext-8.1.17.pdf",
        "institution": "State Council of China (Stanford DigiChina translation)",
        "type": "national AI strategy",
        "year": 2017,
        "ssl_verify": True,
    },
    {
        "name": "Rwanda_National_AI_Policy_2023",
        "url": "https://www.risa.rw/fileadmin/user_upload/RISA/AI_Policy_for_Rwanda.pdf",
        "institution": "Government of Rwanda (RISA)",
        "type": "national AI strategy",
        "year": 2023,
        "ssl_verify": True,
    },
    {
        "name": "Kenya_National_AI_Strategy_2025",
        "url": "https://ict.go.ke/wp-content/uploads/2025/04/Kenya-National-Artificial-Intelligence-Strategy-2025-2030.pdf",
        "institution": "Government of Kenya",
        "type": "national AI strategy",
        "year": 2025,
        "ssl_verify": False,  # gov.ke has self-signed cert
    },
    {
        "name": "ASEAN_AI_Governance_Framework_2023",
        "url": "https://asean.org/wp-content/uploads/2023/12/ASEAN-Guide-on-AI-Governance-and-Ethics-_-2nd-edition.pdf",
        "institution": "ASEAN",
        "type": "regional AI governance",
        "year": 2023,
        "ssl_verify": True,
    },

    # ------------------------------------------------------------------
    # OECD — iLibrary / legalinstruments direct links
    # ------------------------------------------------------------------
    {
        "name": "OECD_AI_Principles_2019",
        "url": "https://legalinstruments.oecd.org/api/download/?uri=/public/doc/648/648.en.pdf",
        "institution": "OECD",
        "type": "AI principles",
        "year": 2019,
        "ssl_verify": True,
    },
    {
        "name": "OECD_Going_Digital_AI_2019",
        "url": "https://www.oecd.org/going-digital/going-digital-an-oecd-perspective-overview.pdf",
        "institution": "OECD",
        "type": "digital AI strategy",
        "year": 2019,
        "ssl_verify": True,
    },

    # ------------------------------------------------------------------
    # UNDP — alternative direct links
    # ------------------------------------------------------------------
    {
        "name": "UNDP_Human_Development_Report_2021_2022",
        "url": "https://hdr.undp.org/system/files/documents/global-report-document/hdr2021-22pdf_1.pdf",
        "institution": "UNDP",
        "type": "human development",
        "year": 2022,
        "ssl_verify": True,
    },

    # ------------------------------------------------------------------
    # UN / ITU
    # ------------------------------------------------------------------
    {
        "name": "UN_Global_Digital_Compact_2024",
        "url": "https://www.un.org/sites/un2.un.org/files/our-common-agenda-policy-brief-godc-en.pdf",
        "institution": "United Nations",
        "type": "digital governance",
        "year": 2023,
        "ssl_verify": True,
    },
    {
        "name": "UNEP_Emissions_Gap_Report_2023",
        "url": "https://wedocs.unep.org/bitstream/handle/20.500.11822/43922/EGR2023.pdf",
        "institution": "UNEP",
        "type": "climate policy",
        "year": 2023,
        "ssl_verify": True,
    },
    {
        "name": "Sendai_Framework_DRR_2015_2030",
        "url": "https://www.undrr.org/publication/sendai-framework-disaster-risk-reduction-2015-2030",
        "institution": "UNDRR",
        "type": "disaster risk framework",
        "year": 2015,
        "ssl_verify": True,
    },
    {
        "name": "Addis_Ababa_Action_Agenda_2015",
        "url": "https://sustainabledevelopment.un.org/content/documents/2051AAAA_Outcome.pdf",
        "institution": "UN (Third International Conference on Financing for Development)",
        "type": "SDG financing framework",
        "year": 2015,
        "ssl_verify": True,
    },

    # ------------------------------------------------------------------
    # World Bank — corrected document URLs
    # ------------------------------------------------------------------
    {
        "name": "World_Bank_World_Development_Report_2021_Data",
        "url": "https://openknowledge.worldbank.org/bitstream/handle/10986/35218/9781464816000.pdf",
        "institution": "World Bank",
        "type": "digital development",
        "year": 2021,
        "ssl_verify": True,
    },
    {
        "name": "World_Bank_AI_Governance_Framework_2021",
        "url": "https://documents1.worldbank.org/curated/en/099300306022213468/pdf/P170666088c2900d40b2e40c65e14bbf69a.pdf",
        "institution": "World Bank",
        "type": "AI governance",
        "year": 2021,
        "ssl_verify": True,
    },
]


def download_pdf(url: str, output_path: Path, ssl_verify: bool = True) -> bool:
    try:
        response = requests.get(
            url, stream=True, timeout=90, headers=HEADERS,
            allow_redirects=True, verify=ssl_verify,
        )
        if response.status_code in (403, 404, 410):
            print(f"  ✗ HTTP {response.status_code} — skipping")
            return False
        response.raise_for_status()

        content_type = response.headers.get("content-type", "")
        if "html" in content_type and "pdf" not in content_type:
            print(f"  ✗ Response is HTML, not PDF — skipping")
            return False

        total_size = int(response.headers.get("content-length", 0))
        with open(output_path, "wb") as f:
            with tqdm(
                total=total_size, unit="B", unit_scale=True,
                desc="  Downloading", leave=False,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

        size = output_path.stat().st_size
        if size < 10_000:
            print(f"  ✗ File too small ({size} bytes) — likely error page")
            output_path.unlink()
            return False
        return True

    except Exception as e:
        print(f"  ✗ Error: {e}")
        if output_path.exists():
            output_path.unlink()
        return False


def extract_text(pdf_path: Path) -> str | None:
    if not HAS_PDFPLUMBER:
        return None
    try:
        with pdfplumber.open(pdf_path) as pdf:
            pages = [p.extract_text() for p in pdf.pages if p.extract_text()]
            return "\n[PAGE BREAK]\n".join(pages) if pages else None
    except Exception as e:
        print(f"  ✗ Extraction error: {e}")
        return None


def main() -> None:
    PDFS_DIR.mkdir(parents=True, exist_ok=True)
    TEXTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print("Policy Document Fetcher v3b — Retry with Fixed URLs")
    print(f"{len(DOCUMENTS)} documents")
    print(f"{'='*70}\n")

    results = []
    start = datetime.now()

    for doc in DOCUMENTS:
        pdf_path = PDFS_DIR / f"{doc['name']}.pdf"
        txt_path = TEXTS_DIR / f"{doc['name']}.txt"
        ssl = doc.get("ssl_verify", True)

        status = {k: doc[k] for k in ("name", "institution", "type", "year", "url")}
        status.update(downloaded=False, text_extracted=False, text_chars=0, error=None)

        print(f"\n[{doc['institution']}] {doc['name']}")

        if txt_path.exists():
            print("  (already processed — skipping)")
            status.update(downloaded=True, text_extracted=True, text_chars=txt_path.stat().st_size)
            results.append(status)
            continue

        if not pdf_path.exists():
            if not download_pdf(doc["url"], pdf_path, ssl_verify=ssl):
                status["error"] = "download failed"
                results.append(status)
                continue
        else:
            print("  (PDF already downloaded)")

        status["downloaded"] = True
        print(f"  ✓ Downloaded ({pdf_path.stat().st_size / 1024:.0f} KB)")

        text = extract_text(pdf_path)
        if text:
            txt_path.write_text(text, encoding="utf-8")
            status.update(text_extracted=True, text_chars=len(text))
            print(f"  ✓ Extracted ({len(text):,} chars)")
        else:
            status["error"] = "extraction failed"
            print("  ✗ Could not extract text")

        results.append(status)

    elapsed = datetime.now() - start
    n_ok = sum(1 for r in results if r["text_extracted"])
    n_fail = len(results) - n_ok

    METADATA_FILE.write_text(json.dumps({
        "fetched_at": start.isoformat(),
        "elapsed_seconds": elapsed.total_seconds(),
        "total_attempted": len(DOCUMENTS),
        "total_success": n_ok,
        "total_failed": n_fail,
        "documents": results,
    }, indent=2))

    print(f"\n{'='*70}")
    print(f"✓ Fetched: {n_ok}/{len(DOCUMENTS)}")
    if n_fail:
        failed = [r["name"] for r in results if not r["text_extracted"]]
        print(f"✗ Failed ({n_fail}): {', '.join(failed)}")
    print(f"✓ Elapsed: {elapsed.total_seconds():.1f}s")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()

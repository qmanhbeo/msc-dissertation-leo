"""
Fetch unified policy document corpus for dissertation.

This is the single active policy fetcher.

Output: data/0_raw/policy_scrape/pdfs/<name>.pdf
        data/0_raw/policy_scrape/texts/<name>.txt
        data/0_raw/policy_scrape/artifact/metadata.json

Run from project root:
    python code/fetch/fetch_policy.py
"""

import json
import argparse
from collections import Counter
from datetime import datetime
from pathlib import Path

import requests
from tqdm import tqdm

try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False
    print("Warning: pdfplumber not installed. Install with: pip install pdfplumber")

OUTPUT_DIR = Path("data/0_raw/policy_scrape")
PDFS_DIR = OUTPUT_DIR / "pdfs"
TEXTS_DIR = OUTPUT_DIR / "texts"
ARTIFACT_DIR = OUTPUT_DIR / "artifact"
METADATA_FILE = ARTIFACT_DIR / "metadata.json"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; dissertation-research-bot/1.0; academic use only)"
    )
}

# Merged registry from legacy policy fetchers.
DOCUMENTS = [{'name': 'UN_SDG_Progress_Report_2023',
  'url': 'https://unstats.un.org/sdgs/report/2023/The-Sustainable-Development-Goals-Report-2023.pdf',
  'institution': 'UN Statistics Division',
  'type': 'SDG progress',
  'year': 2023,
  'confirmed': True},
 {'name': 'IPCC_AR6_Summary_for_Policymakers',
  'url': 'https://www.ipcc.ch/report/ar6/syr/downloads/report/IPCC_AR6_SYR_SPM.pdf',
  'institution': 'IPCC',
  'type': 'climate policy',
  'year': 2023,
  'confirmed': True},
 {'name': 'UK_National_AI_Strategy_2021',
  'url': 'https://assets.publishing.service.gov.uk/government/uploads/system/uploads/attachment_data/file/1020402/National_AI_Strategy_-_PDF_version.pdf',
  'institution': 'UK Government',
  'type': 'national AI strategy',
  'year': 2021,
  'confirmed': True},
 {'name': 'UN_AI_Advisory_Body_Final_Report_2024',
  'url': 'https://www.un.org/sites/un2.un.org/files/governing_ai_for_humanity_final_report_en.pdf',
  'institution': 'UN AI Advisory Body',
  'type': 'AI governance',
  'year': 2024,
  'confirmed': True},
 {'name': 'Singapore_National_AI_Strategy_2.0',
  'url': 'https://file.go.gov.sg/nais2023.pdf',
  'institution': 'Singapore MDDI',
  'type': 'national AI strategy',
  'year': 2023,
  'confirmed': True},
 {'name': 'African_Union_Continental_AI_Strategy_2024',
  'url': 'https://au.int/sites/default/files/documents/44004-doc-EN-_Continental_AI_Strategy_July_2024.pdf',
  'institution': 'African Union',
  'type': 'regional AI framework',
  'year': 2024,
  'confirmed': True},
 {'name': 'Germany_AI_Strategy_2020_Update',
  'url': 'https://www.ki-strategie-deutschland.de/files/downloads/Fortschreibung_KI-Strategie_engl.pdf',
  'institution': 'German Federal Government',
  'type': 'national AI strategy',
  'year': 2020,
  'confirmed': True},
 {'name': 'UNESCO_Ethics_of_AI_2021',
  'url': 'https://www.ohchr.org/sites/default/files/2022-03/UNESCO.pdf',
  'institution': 'UNESCO',
  'type': 'AI ethics',
  'year': 2021,
  'confirmed': True},
 {'name': 'US_Blueprint_AI_Bill_of_Rights',
  'url': 'https://www.managementsolutions.com/sites/default/files/publicaciones/eng/blueprint-for-an-ai-bill-of-rights.pdf',
  'institution': 'White House OSTP',
  'type': 'national AI policy',
  'year': 2022,
  'confirmed': False},
 {'name': 'India_Responsible_AI_NITI_Aayog_2021',
  'url': 'https://indiaai.gov.in/documents/pdf/RaiPolicyDocument.pdf',
  'institution': 'NITI Aayog (India)',
  'type': 'national AI strategy',
  'year': 2021,
  'confirmed': False},
 {'name': 'EU_AI_Ethics_Guidelines_HLEG',
  'url': 'https://www.europarl.europa.eu/cmsdata/196377/AI%20HLEG_Ethics%20Guidelines%20for%20Trustworthy%20AI.pdf',
  'institution': 'EU High-Level Expert Group on AI',
  'type': 'AI ethics guidelines',
  'year': 2019,
  'confirmed': False},
 {'name': 'OECD_AI_Recommendation_2019',
  'url': 'https://wecglobal.org/uploads/2019/07/2019_OECD_Recommendations-AI.pdf',
  'institution': 'OECD',
  'type': 'AI principles',
  'year': 2019,
  'confirmed': False},
 {'name': 'UNDP_Trustworthy_AI_2023',
  'url': 'https://www.undp.org/sites/g/files/zskgke326/files/2023-11/UNDP_Trustworthy_AI_report_0.pdf',
  'institution': 'UNDP',
  'type': 'AI development',
  'year': 2023,
  'confirmed': False},
 {'name': 'UN_SDG_Progress_Report_2024',
  'url': 'https://unstats.un.org/sdgs/report/2024/The-Sustainable-Development-Goals-Report-2024.pdf',
  'institution': 'UN Statistics Division',
  'type': 'SDG progress',
  'year': 2024,
  'confirmed': True},
 {'name': 'UN_SDG_Progress_Report_2022',
  'url': 'https://unstats.un.org/sdgs/report/2022/The-Sustainable-Development-Goals-Report-2022.pdf',
  'institution': 'UN Statistics Division',
  'type': 'SDG progress',
  'year': 2022,
  'confirmed': True},
 {'name': 'UN_SDG_Progress_Report_2021',
  'url': 'https://unstats.un.org/sdgs/report/2021/The-Sustainable-Development-Goals-Report-2021.pdf',
  'institution': 'UN Statistics Division',
  'type': 'SDG progress',
  'year': 2021,
  'confirmed': True},
 {'name': 'UN_SDG_Progress_Report_2020',
  'url': 'https://unstats.un.org/sdgs/report/2020/The-Sustainable-Development-Goals-Report-2020.pdf',
  'institution': 'UN Statistics Division',
  'type': 'SDG progress',
  'year': 2020,
  'confirmed': True},
 {'name': 'UN_SDG_Progress_Report_2019',
  'url': 'https://unstats.un.org/sdgs/report/2019/The-Sustainable-Development-Goals-Report-2019.pdf',
  'institution': 'UN Statistics Division',
  'type': 'SDG progress',
  'year': 2019,
  'confirmed': True},
 {'name': 'UN_SDG_Progress_Report_2018',
  'url': 'https://unstats.un.org/sdgs/files/report/2018/TheSustainableDevelopmentGoalsReport2018-EN.pdf',
  'institution': 'UN Statistics Division',
  'type': 'SDG progress',
  'year': 2018,
  'confirmed': True},
 {'name': 'UN_SDG_Progress_Report_2017',
  'url': 'https://unstats.un.org/sdgs/files/report/2017/TheSustainableDevelopmentGoalsReport2017.pdf',
  'institution': 'UN Statistics Division',
  'type': 'SDG progress',
  'year': 2017,
  'confirmed': True},
 {'name': 'UN_SDG_Progress_Report_2016',
  'url': 'https://unstats.un.org/sdgs/files/report/2016/TheSustainableDevelopmentGoalsReport2016.pdf',
  'institution': 'UN Statistics Division',
  'type': 'SDG progress',
  'year': 2016,
  'confirmed': True},
 {'name': 'WHO_Ethics_AI_Health_2021',
  'url': 'https://www.who.int/publications/i/item/9789240029200',
  'institution': 'World Health Organization',
  'type': 'AI ethics / health',
  'year': 2021,
  'confirmed': False},
 {'name': 'WHO_Global_Strategy_Digital_Health_2020_2025',
  'url': 'https://www.who.int/docs/default-source/documents/gs4dhdaa2a9f352b0445bafbc79ca03f227.pdf',
  'institution': 'World Health Organization',
  'type': 'digital health strategy',
  'year': 2020,
  'confirmed': True},
 {'name': 'WHO_AI_Health_Guidance_2023',
  'url': 'https://www.who.int/publications/i/item/9789240084759',
  'institution': 'World Health Organization',
  'type': 'AI health guidance',
  'year': 2023,
  'confirmed': False},
 {'name': 'G20_AI_Principles_2019',
  'url': 'https://www.oecd.org/going-digital/ai/G20-AI-Principles.pdf',
  'institution': 'G20 / OECD',
  'type': 'AI principles',
  'year': 2019,
  'confirmed': False},
 {'name': 'G7_Hiroshima_AI_Process_2023',
  'url': 'https://www.meti.go.jp/press/2023/10/20231030002/20231030002-1.pdf',
  'institution': 'G7 (Japan Presidency)',
  'type': 'AI governance',
  'year': 2023,
  'confirmed': False},
 {'name': 'EU_AI_Act_2024',
  'url': 'https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=OJ:L_202401689',
  'institution': 'European Union',
  'type': 'AI regulation',
  'year': 2024,
  'confirmed': False},
 {'name': 'EU_Green_Deal_2019',
  'url': 'https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:52019DC0640',
  'institution': 'European Commission',
  'type': 'sustainability policy',
  'year': 2019,
  'confirmed': False},
 {'name': 'EU_Digital_Strategy_2020',
  'url': 'https://ec.europa.eu/info/sites/default/files/communication-shaping-europes-digital-future-feb2020_en_4.pdf',
  'institution': 'European Commission',
  'type': 'digital strategy',
  'year': 2020,
  'confirmed': False},
 {'name': 'EU_White_Paper_AI_2020',
  'url': 'https://ec.europa.eu/info/sites/default/files/commission-white-paper-artificial-intelligence-feb2020_en.pdf',
  'institution': 'European Commission',
  'type': 'AI white paper',
  'year': 2020,
  'confirmed': False},
 {'name': 'Canada_Pan_Canadian_AI_Strategy_2022',
  'url': 'https://ised-isde.canada.ca/site/ai-strategy/sites/default/files/attachments/2022/canada-ai-national-strategy-en.pdf',
  'institution': 'Government of Canada (ISED)',
  'type': 'national AI strategy',
  'year': 2022,
  'confirmed': False},
 {'name': 'France_AI_Strategy_Villani_2018',
  'url': 'https://www.aiforhumanity.fr/pdfs/9782111457089_Rapport_Villani_accessible.pdf',
  'institution': 'French Government (Villani Report)',
  'type': 'national AI strategy',
  'year': 2018,
  'confirmed': False},
 {'name': 'Japan_AI_Strategy_2022',
  'url': 'https://www8.cao.go.jp/cstp/ai/aistrategy2022_honbun_en.pdf',
  'institution': 'Government of Japan (CSTP)',
  'type': 'national AI strategy',
  'year': 2022,
  'confirmed': False},
 {'name': 'South_Korea_National_AI_Strategy_2019',
  'url': 'https://english.msit.go.kr/cms/www/m_bbs/index.do?bbsSeqNo=42&nttSeqNo=2',
  'institution': 'Government of South Korea (MSIT)',
  'type': 'national AI strategy',
  'year': 2019,
  'confirmed': False},
 {'name': 'UAE_AI_Strategy_2031',
  'url': 'https://ai.gov.ae/wp-content/uploads/2021/07/UAE-National-AI-Strategy-2031.pdf',
  'institution': 'UAE Ministry of AI',
  'type': 'national AI strategy',
  'year': 2017,
  'confirmed': False},
 {'name': 'India_National_AI_Strategy_NITI_2018',
  'url': 'https://indiaai.gov.in/documents/pdf/NationalStrategy-for-AI-Discussion-Paper.pdf',
  'institution': 'NITI Aayog (India)',
  'type': 'national AI strategy',
  'year': 2018,
  'confirmed': False},
 {'name': 'Finland_AI_Strategy_2019',
  'url': 'https://julkaisut.valtioneuvosto.fi/bitstream/handle/10024/161555/Finlands-artificial-intelligence-strategy.pdf',
  'institution': 'Government of Finland',
  'type': 'national AI strategy',
  'year': 2019,
  'confirmed': False},
 {'name': 'Denmark_National_AI_Strategy_2019',
  'url': 'https://em.dk/media/13081/denmarks-national-strategy-for-artificial-intelligence.pdf',
  'institution': 'Government of Denmark',
  'type': 'national AI strategy',
  'year': 2019,
  'confirmed': False},
 {'name': 'Brazil_AI_Strategy_2021',
  'url': 'https://www.gov.br/mcti/pt-br/acompanhe-o-mcti/transformacaodigital/arquivosdigital/pdf-ingles/copy_of_ebia_english_web.pdf',
  'institution': 'Government of Brazil (MCTI)',
  'type': 'national AI strategy',
  'year': 2021,
  'confirmed': False},
 {'name': 'Netherlands_AI_Strategy_2019',
  'url': 'https://www.government.nl/binaries/government/documenten/reports/2019/10/08/strategic-action-plan-for-artificial-intelligence/ai-policy-strategy-netherlands.pdf',
  'institution': 'Government of Netherlands',
  'type': 'national AI strategy',
  'year': 2019,
  'confirmed': False},
 {'name': 'Spain_AI_Strategy_2020',
  'url': 'https://www.lamoncloa.gob.es/presidente/actividades/Documents/2020/030620-estrategia-nacional-inteligencia-artificial.pdf',
  'institution': 'Government of Spain',
  'type': 'national AI strategy',
  'year': 2020,
  'confirmed': False},
 {'name': 'Sweden_National_AI_Strategy_2018',
  'url': 'https://www.government.se/4a7451/contentassets/26bc69/national-approach-to-ai.pdf',
  'institution': 'Government of Sweden',
  'type': 'national AI strategy',
  'year': 2018,
  'confirmed': False},
 {'name': 'China_AI_Development_Plan_2017',
  'url': 'https://www.newamerica.org/documents/1959/translation-fulltext-8.1.17.pdf',
  'institution': 'State Council of China',
  'type': 'national AI strategy',
  'year': 2017,
  'confirmed': False},
 {'name': 'OECD_AI_Policy_Observatory_2021',
  'url': 'https://www.oecd.org/digital/artificial-intelligence/OECD-AI-Principles-Overview.pdf',
  'institution': 'OECD',
  'type': 'AI policy',
  'year': 2021,
  'confirmed': False},
 {'name': 'OECD_SDG_Action_Report_2022',
  'url': 'https://www.oecd.org/stories/sustainable-development/OECD-and-the-SDGs.pdf',
  'institution': 'OECD',
  'type': 'SDG action',
  'year': 2022,
  'confirmed': False},
 {'name': 'OECD_Responsible_AI_Toolkit_2023',
  'url': 'https://www.oecd.org/innovation/innovative-business/ai/OECD-Responsible-AI-Toolkit.pdf',
  'institution': 'OECD',
  'type': 'AI ethics',
  'year': 2023,
  'confirmed': False},
 {'name': 'UNDP_Human_Development_Report_2021_2022',
  'url': 'https://hdr.undp.org/content/dam/india/docs/HDR2021-22pdf.pdf',
  'institution': 'UNDP',
  'type': 'human development',
  'year': 2022,
  'confirmed': False},
 {'name': 'UNDP_Strategy_Technology_2022',
  'url': 'https://www.undp.org/sites/g/files/zskgke326/files/2022-11/UNDP-Strategy-2022-2025-R-Smart-Technologies.pdf',
  'institution': 'UNDP',
  'type': 'technology strategy',
  'year': 2022,
  'confirmed': False},
 {'name': 'ITU_AI_for_Good_Global_Summit_2023',
  'url': 'https://www.itu.int/en/ITU-T/AI/Documents/AI4G_Summit_2023_Outcome_Document.pdf',
  'institution': 'ITU / UN',
  'type': 'AI for good',
  'year': 2023,
  'confirmed': False},
 {'name': 'UN_Secretary_General_Roadmap_Digital_Cooperation_2020',
  'url': 'https://www.un.org/en/content/digital-cooperation-roadmap/assets/pdf/Roadmap_for_Digital_Cooperation_EN.pdf',
  'institution': 'UN Secretary-General',
  'type': 'digital cooperation',
  'year': 2020,
  'confirmed': False},
 {'name': 'UN_Global_Digital_Compact_2024',
  'url': 'https://www.un.org/sites/un2.un.org/files/our-common-agenda-policy-brief-godc-en.pdf',
  'institution': 'UN / OECD',
  'type': 'digital governance',
  'year': 2023,
  'confirmed': False},
 {'name': 'UNEP_Emissions_Gap_Report_2023',
  'url': 'https://www.unep.org/resources/emissions-gap-report-2023',
  'institution': 'UNEP',
  'type': 'climate policy',
  'year': 2023,
  'confirmed': False},
 {'name': 'SDSN_Sustainable_Development_Report_2024',
  'url': 'https://s3.amazonaws.com/sustainabledevelopment.report/2024/sustainable-development-report-2024.pdf',
  'institution': 'SDSN / Bertelsmann Stiftung',
  'type': 'SDG index report',
  'year': 2024,
  'confirmed': False},
 {'name': 'SDSN_Sustainable_Development_Report_2025',
  'url': 'https://s3.amazonaws.com/sustainabledevelopment.report/2025/sustainable-development-report-2025.pdf',
  'institution': 'SDSN / Bertelsmann Stiftung',
  'type': 'SDG index report',
  'year': 2025,
  'confirmed': False},
 {'name': 'World_Bank_Digital_Economy_Compass_2023',
  'url': 'https://documents1.worldbank.org/curated/en/099120123014531212/pdf/P17831700f94050b00afc101e7a5c7ae0be.pdf',
  'institution': 'World Bank',
  'type': 'digital economy',
  'year': 2023,
  'confirmed': False},
 {'name': 'World_Bank_AI_Development_2021',
  'url': 'https://documents1.worldbank.org/curated/en/099300306022213468/pdf/P170666088c2900d40b2e40c65e14bbf69a.pdf',
  'institution': 'World Bank',
  'type': 'AI development',
  'year': 2021,
  'confirmed': False},
 {'name': 'Paris_Agreement_2015',
  'url': 'https://unfccc.int/sites/default/files/english_paris_agreement.pdf',
  'institution': 'UNFCCC',
  'type': 'climate treaty',
  'year': 2015,
  'confirmed': True},
 {'name': 'IPCC_AR6_WG2_Summary_Policymakers_2022',
  'url': 'https://www.ipcc.ch/report/ar6/wg2/downloads/report/IPCC_AR6_WGII_SummaryForPolicymakers.pdf',
  'institution': 'IPCC',
  'type': 'climate policy',
  'year': 2022,
  'confirmed': True},
 {'name': 'IPCC_AR6_WG3_Summary_Policymakers_2022',
  'url': 'https://www.ipcc.ch/report/ar6/wg3/downloads/report/IPCC_AR6_WGIII_SummaryForPolicymakers.pdf',
  'institution': 'IPCC',
  'type': 'climate mitigation policy',
  'year': 2022,
  'confirmed': True},
 {'name': 'ASEAN_AI_Governance_Framework_2023',
  'url': 'https://asean.org/wp-content/uploads/2023/12/ASEAN-Guide-on-AI-Governance-and-Ethics_updated_-2023.pdf',
  'institution': 'ASEAN',
  'type': 'regional AI governance',
  'year': 2023,
  'confirmed': False},
 {'name': 'Rwanda_National_AI_Policy_2023',
  'url': 'https://www.minict.gov.rw/fileadmin/user_upload/AI_Policy_for_Rwanda.pdf',
  'institution': 'Government of Rwanda (MINICT)',
  'type': 'national AI strategy',
  'year': 2023,
  'confirmed': False},
 {'name': 'Kenya_National_AI_Strategy_2025',
  'url': 'https://ict.go.ke/wp-content/uploads/2025/04/Kenya-National-Artificial-Intelligence-Strategy-2025-2030.pdf',
  'institution': 'Government of Kenya',
  'type': 'national AI strategy',
  'year': 2025,
  'confirmed': False},
 {'name': 'UN_2030_Agenda_Sustainable_Development_2015',
  'url': 'https://sdgs.un.org/sites/default/files/publications/21252030%20Agenda%20for%20Sustainable%20Development%20web.pdf',
  'institution': 'United Nations',
  'type': 'SDG foundational document',
  'year': 2015,
  'confirmed': True},
 {'name': 'Addis_Ababa_Action_Agenda_2015',
  'url': 'https://sustainabledevelopment.un.org/content/documents/2051AAAA_Outcome.pdf',
  'institution': 'UN (Addis Ababa)',
  'type': 'SDG financing framework',
  'year': 2015,
  'confirmed': False},
 {'name': 'Sendai_Framework_DRR_2015_2030',
  'url': 'https://www.undrr.org/publication/sendai-framework-disaster-risk-reduction-2015-2030',
  'institution': 'UNDRR',
  'type': 'disaster risk framework',
  'year': 2015,
  'confirmed': False}]


def download_pdf(url: str, output_path: Path) -> bool:
    """Download PDF. Returns True on success, False on any failure."""
    try:
        response = requests.get(
            url, stream=True, timeout=90, headers=HEADERS, allow_redirects=True
        )
        if response.status_code in (403, 404, 410):
            print(f"  ✗ HTTP {response.status_code} - skipping")
            return False
        response.raise_for_status()

        content_type = response.headers.get("content-type", "")
        if "html" in content_type and "pdf" not in content_type:
            print("  ✗ Response is HTML, not PDF - skipping")
            return False

        total_size = int(response.headers.get("content-length", 0))
        with output_path.open("wb") as f:
            with tqdm(
                total=total_size,
                unit="B",
                unit_scale=True,
                desc="  Downloading",
                leave=False,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

        size = output_path.stat().st_size
        if size < 10_000:
            print(f"  ✗ File too small ({size} bytes) - likely error page")
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
        print(f"  ✗ Text extraction error: {e}")
        return None


def corpus_inventory() -> dict:
    """Return current on-disk inventory for policy raw artifacts."""
    pdf_files = list(PDFS_DIR.glob("*.pdf"))
    txt_files = list(TEXTS_DIR.glob("*.txt"))
    pdf_size_bytes = sum(p.stat().st_size for p in pdf_files if p.exists())
    txt_size_bytes = sum(p.stat().st_size for p in txt_files if p.exists())
    return {
        "pdf_count": len(pdf_files),
        "text_count": len(txt_files),
        "pdf_size_bytes": pdf_size_bytes,
        "text_size_bytes": txt_size_bytes,
    }


def main(local_files_only: bool = False) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PDFS_DIR.mkdir(parents=True, exist_ok=True)
    TEXTS_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    n_confirmed = sum(1 for d in DOCUMENTS if d.get("confirmed"))
    n_try = len(DOCUMENTS) - n_confirmed

    print(f"\n{'=' * 70}")
    print("Unified Policy Document Fetcher")
    print(f"{'=' * 70}")
    print(f"  {len(DOCUMENTS)} documents total:")
    print(f"    {n_confirmed} confirmed-accessible")
    print(f"    {n_try} try-or-skip")
    print(f"  Local-files-only: {local_files_only}")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"{'=' * 70}\n")

    results = []
    start_time = datetime.now()

    for doc in DOCUMENTS:
        pdf_path = PDFS_DIR / f"{doc['name']}.pdf"
        txt_path = TEXTS_DIR / f"{doc['name']}.txt"

        status = {
            **{k: doc[k] for k in ("name", "institution", "type", "year", "url", "confirmed")},
            "pdf_path": str(pdf_path.relative_to(OUTPUT_DIR)),
            "text_path": str(txt_path.relative_to(OUTPUT_DIR)),
            "status": "unknown",
            "downloaded": False,
            "text_extracted": False,
            "downloaded_now": False,
            "text_extracted_now": False,
            "pdf_exists": False,
            "text_exists": False,
            "pdf_bytes": 0,
            "text_bytes": 0,
            "error": None,
        }

        print(f"\n[{doc['institution']}] {doc['name']}")

        if txt_path.exists():
            print("  (already processed - skipping)")
            status.update(
                status="already_present",
                downloaded=True,
                text_extracted=True,
                pdf_exists=pdf_path.exists(),
                text_exists=True,
                pdf_bytes=pdf_path.stat().st_size if pdf_path.exists() else 0,
                text_bytes=txt_path.stat().st_size,
            )
            results.append(status)
            continue

        if local_files_only:
            status["status"] = "missing_local_artifacts"
            status["error"] = "local-files-only enabled and text file missing"
            status["pdf_exists"] = pdf_path.exists()
            status["pdf_bytes"] = pdf_path.stat().st_size if pdf_path.exists() else 0
            results.append(status)
            continue

        if not pdf_path.exists():
            ok = download_pdf(doc["url"], pdf_path)
            if not ok:
                status["status"] = "download_failed"
                status["error"] = "download failed"
                results.append(status)
                continue
            status["downloaded_now"] = True
        else:
            print("  (PDF already downloaded)")

        status["downloaded"] = True
        status["pdf_exists"] = pdf_path.exists()
        status["pdf_bytes"] = pdf_path.stat().st_size if pdf_path.exists() else 0
        print(f"  ✓ Downloaded ({pdf_path.stat().st_size / 1024:.0f} KB)")

        text = extract_text(pdf_path)
        if text:
            txt_path.write_text(text, encoding="utf-8")
            status["text_extracted"] = True
            status["text_extracted_now"] = True
            status["text_exists"] = True
            status["text_bytes"] = txt_path.stat().st_size
            status["status"] = (
                "downloaded_and_extracted"
                if status["downloaded_now"]
                else "extracted_from_existing_pdf"
            )
            print(f"  ✓ Text extracted ({len(text):,} chars)")
        else:
            status["status"] = "text_extraction_failed"
            status["error"] = "text extraction failed"
            print("  ✗ Could not extract text")

        results.append(status)

    end_time = datetime.now()
    elapsed = end_time - start_time
    status_counts = Counter(r["status"] for r in results)
    n_ok = sum(1 for r in results if r["text_extracted"])
    n_fail = len(results) - n_ok
    inventory = corpus_inventory()

    metadata = {
        "schema_version": "policy_fetch_metadata_v2",
        "generated_by": "code/fetch/fetch_policy.py",
        "run_started_at": start_time.isoformat(),
        "run_finished_at": end_time.isoformat(),
        "generated_at": end_time.isoformat(),
        "elapsed_seconds": elapsed.total_seconds(),
        "registry": {
            "total_documents": len(DOCUMENTS),
            "confirmed_count": n_confirmed,
            "try_or_skip_count": n_try,
        },
        "run_summary": {
            "total_attempted": len(DOCUMENTS),
            "total_success": n_ok,
            "total_failed": n_fail,
            "already_present": status_counts.get("already_present", 0),
            "downloaded_new": status_counts.get("downloaded_and_extracted", 0),
            "extracted_from_existing_pdf": status_counts.get("extracted_from_existing_pdf", 0),
            "download_failed": status_counts.get("download_failed", 0),
            "text_extraction_failed": status_counts.get("text_extraction_failed", 0),
        },
        "outputs": {
            "root_dir": str(OUTPUT_DIR),
            "pdf_dir": str(PDFS_DIR),
            "text_dir": str(TEXTS_DIR),
            "metadata_file": str(METADATA_FILE),
            **inventory,
        },
        "documents": results,
    }
    METADATA_FILE.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"\n{'=' * 70}")
    print(f"✓ Successfully fetched: {n_ok}/{len(DOCUMENTS)} documents")
    if n_fail:
        failed = [r["name"] for r in results if not r["text_extracted"]]
        print(f"✗ Failed ({n_fail}): {', '.join(failed)}")
    print(f"✓ Elapsed: {elapsed.total_seconds():.1f}s")
    print(f"✓ Metadata saved to {METADATA_FILE}")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch unified policy corpus")
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Do not download missing files; summarize local artifacts only.",
    )
    args = parser.parse_args()
    main(local_files_only=args.local_files_only)

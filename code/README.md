# Data Fetching Scripts

Python scripts to download data sources for dissertation Topic 2: "Semantic Alignment Between AI Sustainability Research and Policy Frameworks"

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run any script
python code/fetch_openalex.py
python code/fetch_osdg.py
python code/fetch_un_sdg.py
python code/fetch_sdg_benchmark.py
python code/fetch_sdg_publications.py  # scrapes sdgs.un.org
python code/fetch_sdgi_corpus.py  # (requires: pip install datasets)
python code/fetch_sdgindex.py
python code/fetch_unsd.py  # UNSD official SDG metadata + reports
python code/fetch_aurora.py  # AURORA 1.4M DOIs with SDG labels
python code/fetch_sdg_news.py  # IISD SDG news articles
python code/fetch_ungdc.py  # UN General Debate Corpus
python code/fetch_nlp4sg.py  # NLP papers mapped to SDGs
python code/fetch_un_ga.py  # UN General Assembly data
python code/fetch_kaggle.py  # (optional, requires Kaggle credentials)
```

---

## Scripts Overview

### 1. `fetch_openalex.py`
**Source:** [OpenAlex API](https://openalex.org/) (free, no authentication required)

Downloads academic papers on AI + Sustainable Development.

**Output:** 
- `data/openalex/papers.jsonl` — papers with title, abstract, DOI, year, concepts, citation count
- `data/openalex/metadata.json` — fetch metadata

**Time:** ~2-5 minutes | **Size:** ~50-200 MB

---

### 2. `fetch_osdg.py`
**Source:** [Zenodo Record #11441197](https://zenodo.org/records/11441197) + [GitHub examples](https://github.com/osdg-ai/osdg-data)

Downloads the OSDG Community Dataset from Zenodo, plus classification examples from GitHub.

**Output:**
- `data/osdg/` — CSV files with text excerpts and SDG labels
- `data/osdg/examples/helpers.py` — Python utilities (SDG mappings, confusion matrix, feature extraction)
- `data/osdg/examples/osdg-cd-example-classifier-sklearn.ipynb` — sklearn classification example
- `data/osdg/metadata.json` — fetch metadata

**Time:** ~1-3 minutes | **Size:** ~30-50 MB

---

### 3. `fetch_un_sdg.py`
**Source:** [UN Statistics API](https://unstats.un.org/sdgs/) + Public UN/AI policy PDFs (free, no authentication required)

Downloads two types of data:
1. Official UN SDG indicators (goals, targets, indicators taxonomy)
2. Key policy documents: UN AI strategies, AI advisory reports, policy briefs

**Output:**
- `data/un_sdg/sdg_indicators.json` — official SDG structure
- `data/un_sdg/pdfs/` — downloaded PDF documents
- `data/un_sdg/texts/` — extracted text from PDFs
- `data/un_sdg/metadata.json` — fetch metadata

**Requirements:** `pdfplumber` (installed via `requirements.txt`)

**Time:** ~2-5 minutes | **Size:** ~30-80 MB

---

### 4. `fetch_sdg_benchmark.py`
**Source:** [GitHub: SDGClassification/benchmark](https://github.com/SDGClassification/benchmark) (free, no authentication required)

Downloads the SDG Classification Benchmark dataset from GitHub.

**Output:**
- `data/sdg_benchmark/` — benchmark files, code, data
- `data/sdg_benchmark/metadata.json` — fetch metadata

**Time:** ~1-2 minutes | **Size:** ~10-30 MB

---

### 5. `fetch_sdg_publications.py`
**Source:** https://sdgs.un.org (scrapes sitemap for publication URLs)

Scrapes the UN SDG website to discover and download publications. Uses the XML sitemap for URL discovery, then scrapes publication pages for PDF download links.

**Output:**
- `data/sdg_publications/urls.json` — discovered publication URLs and PDF links
- `data/sdg_publications/pdfs/` — downloaded PDF documents
- `data/sdg_publications/metadata.json` — scrape metadata

**What it found:**
- 6,354 URLs in sitemap
- 202 unique PDF links discovered
- 37 PDFs downloaded (first 50 attempted)

**Requirements:** `requests`, `beautifulsoup4`

**Time:** ~1-2 minutes | **Size:** Varies by number of PDFs downloaded

**Notes:**
- Site is JavaScript-heavy (Drupal CMS); sitemap used for URL discovery
- PDF links include both sdgs.un.org documents and external references (WEF, academic papers)
- Re-run to download more PDFs (progress is saved)
- See `data/sdg_publications/urls.json` for full list of 202 PDF URLs

---

### 7. `fetch_sdgi_corpus.py`
**Source:** [Hugging Face: UNDP/sdgi-corpus](https://huggingface.co/datasets/UNDP/sdgi-corpus) (free, requires `datasets` library)

Downloads the SDGi Corpus: authoritative policy language from Voluntary National Reviews (VNRs) and Voluntary Local Reviews (VLRs) labeled by SDG.

**Output:**
- `data/sdgi_corpus/sdgi_corpus.parquet` — train split (5,880 rows)
- `data/sdgi_corpus/sdgi_corpus_test.parquet` — test split (1,470 rows)
- `data/sdgi_corpus/metadata.json` — fetch metadata with corpus statistics

**Dataset stats:**
- 7,350 total rows (5,880 train + 1,470 test)
- Columns: `text`, `embedding`, `labels`, `metadata`
- Multi-label (texts can belong to multiple SDGs)
- Languages: English (4,225), Spanish (935), French (720)
- Document types: VNR (4,001), VLR (1,879)
- License: cc-by-nc-sa-4.0

**Requirements:** `pip install datasets`

**Time:** ~1-2 minutes | **Size:** ~120 MB

**Citation:**
> Skrynnyk, O. et al. (2024). SDGi Corpus: A Comprehensive Multilingual Dataset for Text Classification by Sustainable Development Goals. UNDP. https://huggingface.co/datasets/UNDP/sdgi-corpus

**Use:** This is the most authoritative policy corpus — directly from governments reporting SDG implementation. Use to validate or supplement the policy corpus built from the expanded policy fetch.

---

### 8. `fetch_sdg_news.py`
**Source:** [Zenodo Record #7523032](https://zenodo.org/records/7523032) (free, no authentication required)

Downloads the IISD SDG Knowledge Hub dataset: news articles from sdg.iisd.org labeled by SDG.

**Output:**
- `data/sdg_news/sdg_knowledge_hub.csv` — news articles with SDG labels
- `data/sdg_news/metadata.json` — fetch metadata

**Dataset stats:**
- 9,172 news articles
- Columns: `url`, `title`, `type`, `text`, `date`, `sdgs`, `SDG-01` to `SDG-17` (binary)
- Source: IISD SDG Knowledge Hub (sdg.iisd.org)
- License: CC-BY 4.0

**Requirements:** `requests`, `pandas`, `tqdm` (optional)

**Time:** ~5 seconds | **Size:** ~42 MB

**Citation:**
> Wulff, D. U., & Meier, D. S. (2024). SDG Knowledge Hub Dataset of SDG-labeled News Articles. Zenodo. https://doi.org/10.5281/zenodo.7523032

**Use:** Provides current news coverage of SDG topics — useful for tracking which SDGs are most covered in international development news.

---

### 9. `fetch_aurora.py`
**Source:** [Zenodo Record #5224005](https://zenodo.org/records/5224005) (free, no authentication required)

Downloads the AURORA SDG Dataset: 1.4 million research article DOIs labeled at SDG Target level (169 targets) and Goal level (17 goals), covering 2009-2020.

**Output:**
- `data/aurora/aurora_sdg_targets.csv` — Long format (doi, date, sdg_target, sdg_goal)
- `data/aurora/aurora_sdg_targets_wide.csv` — Wide format (doi, date, 169 targets, 17 goals as binary columns)
- `data/aurora/aurora_sdg_targets.xlsx` — Excel format
- `data/aurora/metadata.json` — fetch metadata

**Dataset stats:**
- 1.1M unique DOIs (wide format)
- 1.4M DOIs with labels (long format, one row per DOI-target pair)
- Period: 2009-2020
- Labels: 17 SDGs + 169 targets
- Source: UN SDG Metadata ontology
- License: CC-BY 4.0

**Requirements:** `requests`, `pandas`, `tqdm` (optional)

**Time:** ~10 seconds | **Size:** ~344 MB

**Citation:**
> Vanderfeesten, M. (2024). DOI's with SDG labels on Target level | 1.4M research articles (2009-2020) related to Sustainable Development Goals. Zenodo. https://doi.org/10.5281/zenodo.5224005

**Use:** Cross-reference DOIs with OpenAlex or CrossRef for full article metadata. Excellent for analyzing research output by SDG over time.

---

### 10. `fetch_ungdc.py`
**Source:** [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/0TJX8Y) (free, no authentication required)

Downloads the UN General Debate Corpus (UNGDC): speeches from UN General Assembly General Debates 1946-2025.

**Output:**
- `data/ungdc/TXT/` — Extracted corpus (11,141 speeches organized by session/year)
- `data/ungdc/Speakers_by_session.xlsx` — Speaker metadata
- `data/ungdc/README.txt` — Documentation
- `data/ungdc/metadata.json` — fetch metadata

**Dataset stats:**
- 11,141 speeches from 202 countries
- Period: 1946-2025 (80 sessions)
- Format: Plain text (UTF-8), named by ISO country code, session, year (e.g., USA_75_2020.txt)
- Source: UN Library transcripts + audio transcription for 2025

**Requirements:** `requests`, `pandas`, `tqdm` (optional)

**Time:** ~15 seconds | **Size:** ~71 MB compressed

**Citation:**
> Jankin, S., Baturo, A., & Dasandi, N. (2025). Words to unite nations: The complete United Nations General Debate Corpus, 1946–present. Journal of Peace Research, 62(4), 1339-1351. https://doi.org/10.1177/00223433241275335

**Use:** Government policy discourse — track how countries' stated positions on global issues evolve. Can cross-reference with SDG coverage in UNGD to measure policy attention to different goals over time.

---

### 11. `fetch_nlp4sg.py`
**Source:** [Hugging Face: feradauto/NLP4SGPapers](https://huggingface.co/datasets/feradauto/NLP4SGPapers) (free)

Maps NLP papers to SDGs from the ACL Anthology — directly relevant to your dissertation!

**Output:**
- `data/nlp4sg/train.json` — Training split (2,500 rows)
- `data/nlp4sg/validation.json` — Validation split (500 rows)
- `data/nlp4sg/test.json` — Test split (2,000 rows)
- `data/nlp4sg/metadata.json` — fetch metadata

**Dataset stats:**
- 5,000 NLP papers from ACL Anthology
- Columns: `id`, `url`, `title`, `abstract`, `label_nlp4sg`, `task`, `method`, `goal1-3`, `sdg1-sdg17` (bool)
- Train: 2,500 | Val: 500 | Test: 2,000

**Requirements:** `requests`, `pandas` (optional)

**Time:** ~2 seconds | **Size:** ~7 MB

**Citation:**
> Fernandez, F., et al. (2023). NLP4SGPapers: A Scientific Dataset for Identifying NLP Papers Addressing Social Problems and UN SDGs. Findings of EMNLP 2023. https://aclanthology.org/2023.findings-emnlp.31/

**Use:** **Goldmine for your dissertation** — directly maps NLP papers to SDGs. Compare AI/sustainability research coverage with policy priorities. Website: https://nlp4sg.vercel.app

---

### 12. `fetch_un_ga.py`
**Source:** [UN Digital Library](https://digitallibrary.un.org/) (free, no authentication required)

Downloads UN General Assembly resolutions and voting data from the Dag Hammarskjöld Library.

**Output:**
- `data/un_ga/ga_outcomes.csv` — Resolution outcomes with subjects, vote counts
- `data/un_ga/ga_voting.csv` — Country-level voting data
- `data/un_ga/metadata.json` — fetch metadata

**Dataset stats:**
- GA Outcomes: 20,761 resolutions (1946-2025), 7.4 MB
- GA Voting: 947,434 votes (country-level), 364 MB
- Columns: resolution, date, session, subjects, vote counts, member state votes

**Requirements:** `requests`, `pandas`, `tqdm` (optional)

**Time:** ~20 seconds | **Size:** ~372 MB

**Citation:**
> United Nations Dag Hammarskjöld Library. UN General Assembly resolutions and voting data, 1946-2025. https://digitallibrary.un.org/

**Use:** Track UN policy attention to SDGs over time. Correlate with UNGDC speeches and AURORA research output for alignment analysis.

---

### 13. `fetch_kaggle.py`
**Source:** [Kaggle Dataset](https://www.kaggle.com/datasets/sazidthe1/sustainable-development-report) (free dataset, requires Kaggle account)

Downloads Sustainable Development Report data (SDG progress metrics).

**Output:**
- `data/kaggle/` — CSV files with SDG metrics
- `data/kaggle/metadata.json` — fetch metadata

**Requirements:** 
- Kaggle account (free)
- Credentials file at `~/.kaggle/kaggle.json`
  - [Get credentials here](https://www.kaggle.com/settings/account)
  - Click "Create New Token" to download `kaggle.json`
  - Place in `~/.kaggle/kaggle.json`
  - Run `chmod 600 ~/.kaggle/kaggle.json`

**Time:** ~1-2 minutes | **Size:** ~10-20 MB

**Note:** This script is optional. The other sources provide complete data for the dissertation.

---

### 9. `fetch_sdgindex.py`
**Source:** [SDG Index Downloads](https://dashboards.sdgindex.org/downloads/) (free, no authentication required)

Downloads the official Sustainable Development Report 2025 Excel database directly from the SDG Index.

**Output:**
- `data/sdgindex/sdr2025_data.xlsx` — full database (208 countries, 685 columns)
- `data/sdgindex/metadata.json` — fetch metadata

**Preprocess:**
- `preprocess_sdgindex.py` — extracts per-country SDG scores and summary statistics
  - Output: `data/sdgindex/sdr2025_overview.csv` (208 countries × 17 SDG scores)
  - Output: `data/sdgindex/sdr2025_summary.json` (aggregate statistics)

**Requirements:** `pandas`, `openpyxl`

**Time:** ~30 seconds | **Size:** ~9 MB

**Citation:**
> Sachs, J.D., Lafortune, G., Fuller, G., Iablonovski, G. (2025). *Financing Sustainable Development to 2030 and Mid-Century. Sustainable Development Report 2025.* Paris: SDSN, Dublin: Dublin University Press. DOI: https://doi.org/10.25546/111909

**Use:** Provides real-world SDG performance context for gap analysis (H3: are alignment gaps largest where global progress is weakest?)

---

### 10. `fetch_unsd.py`
**Source:** [UN Statistics Division (UNSD)](https://unstats.un.org/UNSDWebsite/) + [SDG API](https://unstats.un.org/SDGAPI/)

Fetches official SDG metadata from the UN Statistics Division via their public API, plus key reports.

**Output:**
- `data/unsd/goals.json` — 17 SDG goal definitions
- `data/unsd/targets.json` — 169 SDG targets
- `data/unsd/indicators.json` — 251 indicator definitions (234 unique)
- `data/unsd/series.json` — 713 indicator series with metadata
- `data/unsd/geoareas.json` — 460 geographic areas (countries, regions, groupings)
- `data/unsd/indicator_framework.xlsx` — Official Global Indicator Framework Excel
- `data/unsd/reports/` — Downloaded PDFs (Statistical Annex, SG Reports)
- `data/unsd/metadata.json` — fetch metadata

**Requirements:** `requests`, `pandas`, `openpyxl`

**Time:** ~15 seconds | **Size:** ~16 MB

**API Endpoints Used:**
- `GET /v1/sdg/Goal/List` — Goal definitions
- `GET /v1/sdg/Target/List` — Target definitions
- `GET /v1/sdg/Indicator/List` — Indicator definitions
- `GET /v1/sdg/Series/List` — Series definitions
- `GET /v1/sdg/GeoArea/List` — Geographic areas

**Use:** Provides authoritative official SDG taxonomy for validating research-policy alignment. The indicator framework Excel and API metadata are the most comprehensive source for SDG classification.

---

## Running Scripts

### Run individually:
```bash
python code/fetch_openalex.py
python code/fetch_sdgindex.py  # fetches SDR 2025
```

### Run all at once:
```bash
for script in code/fetch_*.py; do python "$script"; done
```

### Run specific scripts:
```bash
python code/fetch_openalex.py && python code/fetch_osdg.py && python code/fetch_un_sdg.py
```

### Preprocess SDR 2025 data:
```bash
python code/preprocess_sdgindex.py
```

---

## What You Get

Each script downloads to its own folder:

```
data/
├── openalex/          # Academic papers (~50-200 MB)
│   ├── papers.jsonl
│   └── metadata.json
├── osdg/              # OSDG Community Dataset (~20-50 MB)
│   ├── *.csv
│   └── metadata.json
├── un_sdg/            # UN indicators + policy documents (~30-80 MB)
│   ├── sdg_indicators.json
│   ├── pdfs/
│   ├── texts/
│   └── metadata.json
├── sdg_benchmark/     # SDG classification benchmark (~10-30 MB)
│   ├── ** (repo files)
│   └── metadata.json
├── sdg_publications/  # Scraped from sdgs.un.org
│   ├── urls.json               # discovered URLs + PDF links
│   ├── pdfs/                  # downloaded PDFs
│   └── metadata.json
├── sdgi_corpus/       # UNDP SDGi Corpus (~120 MB)
│   ├── sdgi_corpus.parquet      # train (5,880 rows)
│   ├── sdgi_corpus_test.parquet # test (1,470 rows)
│   └── metadata.json
├── sdgindex/          # SDR 2025 (~9 MB)
│   ├── sdr2025_data.xlsx
│   ├── sdr2025_overview.csv     # (from preprocess_sdgindex.py)
│   ├── sdr2025_summary.json     # (from preprocess_sdgindex.py)
│   └── metadata.json
├── unsd/              # UNSD official SDG metadata (~16 MB)
│   ├── goals.json             # 17 SDG goals
│   ├── targets.json           # 169 targets
│   ├── indicators.json        # 251 indicators
│   ├── series.json            # 713 series
│   ├── geoareas.json          # 460 geographic areas
│   ├── indicator_framework.xlsx
│   ├── reports/               # PDFs
│   └── metadata.json
├── aurora/            # AURORA SDG Dataset (~344 MB)
│   ├── aurora_sdg_targets.csv       # Long format (doi, date, sdg_target, sdg_goal)
│   ├── aurora_sdg_targets_wide.csv  # Wide format (169 targets + 17 goals as columns)
│   ├── aurora_sdg_targets.xlsx      # Excel format
│   └── metadata.json
├── sdg_news/          # IISD SDG Knowledge Hub news (~42 MB)
│   ├── sdg_knowledge_hub.csv
│   └── metadata.json
├── ungdc/             # UN General Debate Corpus (~71 MB)
│   ├── TXT/                        # 11,141 speeches by session
│   ├── Speakers_by_session.xlsx
│   ├── README.txt
│   └── metadata.json
├── nlp4sg/           # NLP papers mapped to SDGs (~7 MB)
│   ├── train.json
│   ├── validation.json
│   ├── test.json
│   └── metadata.json
├── un_ga/            # UN General Assembly data (~372 MB)
│   ├── ga_outcomes.csv       # Resolution outcomes (20k resolutions)
│   ├── ga_voting.csv        # Country voting data (947k votes)
│   └── metadata.json
└── kaggle/            # SDG progress metrics (~10-20 MB, optional)
    ├── *.csv
    └── metadata.json
```

**Total size:** ~150-450 MB (depending on data size variations)

---

## Reproducibility

Each script saves a `metadata.json` file with:
- Source URL
- Fetch date & time
- Version information
- File counts and sizes

This allows you to document exact data sources in your dissertation methodology section.

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'requests'"
```bash
pip install -r requirements.txt
```

### "No module named 'pdfplumber'" (fetch_un_sdg.py)
The script will still work — PDFs download but text extraction is skipped.
```bash
pip install pdfplumber
```

### Kaggle authentication fails
Ensure credentials are set up:
```bash
# Check if file exists
ls -la ~/.kaggle/kaggle.json

# Set correct permissions
chmod 600 ~/.kaggle/kaggle.json
```

### Network timeouts
Most scripts have 30-second timeouts. If you're on a slow connection, edit the timeout values in the script:
```python
response = requests.get(url, timeout=60)  # Increase to 60 seconds
```

---

## Next Steps

After downloading, you can:

1. **Explore the data** — check file sizes and structure
2. **Build your corpus** — combine papers and policy docs
3. **Extract text** — prepare for topic modeling
4. **Analyze** — run topic modeling, semantic similarity, etc.

See `/dissertation/writing/` for dissertation writing and analysis scripts (coming next).

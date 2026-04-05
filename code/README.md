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
python code/fetch_sdgi_corpus.py  # (requires: pip install datasets)
python code/fetch_sdgindex.py
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
**Source:** [Zenodo Record #11441197](https://zenodo.org/records/11441197) (free, no authentication required)

Downloads the OSDG Community Dataset: ~42,000 text excerpts tagged by SDG by 1,400+ citizen scientists.

**Output:**
- `data/osdg/` — CSV files with text excerpts and SDG labels
- `data/osdg/metadata.json` — fetch metadata

**Time:** ~1-3 minutes | **Size:** ~20-50 MB

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

### 5. `fetch_sdgi_corpus.py`
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

### 6. `fetch_kaggle.py`
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

### 7. `fetch_sdgindex.py`
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
├── sdgi_corpus/       # UNDP SDGi Corpus (~120 MB)
│   ├── sdgi_corpus.parquet      # train (5,880 rows)
│   ├── sdgi_corpus_test.parquet # test (1,470 rows)
│   └── metadata.json
├── sdgindex/          # SDR 2025 (~9 MB)
│   ├── sdr2025_data.xlsx
│   ├── sdr2025_overview.csv     # (from preprocess_sdgindex.py)
│   ├── sdr2025_summary.json     # (from preprocess_sdgindex.py)
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

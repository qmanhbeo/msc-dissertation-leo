# Bibliography Management Setup — Complete

**Date:** 11 April 2026  
**Task:** Clone marker, extract dissertation citations, organize as structured data  
**Status:** ✅ Complete

---

## What Was Done

### 1. ✅ Cloned Marker Tool
- **Location:** `/dissertation/marker/` (27 files, latest commit v1.10.2-8-g2085e10)
- **Purpose:** High-fidelity PDF-to-Markdown conversion for papers
- **Use case:** Future expansion if full-text semantic analysis is needed
- **Not currently used:** Dissertation analysis uses abstracts only

**To use marker on a single PDF:**
```bash
cd marker
python convert_single.py /path/to/paper.pdf
```

### 2. ✅ Extracted Dissertation Citations
- **Source:** `writing/references.bib` (22 BibTeX entries)
- **Script:** `code/fetch_cited_papers.py` (created; 170 lines with docstring)

**Generated outputs:**

| File | Format | Size | Contents |
|------|--------|------|----------|
| `data/cited_papers.json` | JSON | 9.4 KB | 22 papers with full metadata (authors, year, journal, DOI, arXiv links) |
| `data/cited_papers.md` | Markdown | 6.8 KB | Human-readable index organized by type (articles, conference papers, reports) |

**Sample entry (JSON):**
```json
{
  "key": "Vinuesa2020",
  "type": "article",
  "title": "The role of artificial intelligence in achieving the Sustainable Development Goals",
  "author": "Vinuesa, Ricardo and Azizpour, Hossein and Leite, Iolanda...",
  "year": "2020",
  "journal": "Nature Communications",
  "doi": "10.1038/s41467-019-14108-y",
  "doi_url": "https://doi.org/10.1038/s41467-019-14108-y"
}
```

### 3. ✅ Updated Documentation

**Updated files:**

| File | Changes |
|------|---------|
| `README.md` | Added "Bibliography Management and Paper Organization" section with details on generated indices and marker tool usage |
| `code/README.md` | Added "Bibliography Management" section documenting `fetch_cited_papers.py` script, usage, and outputs |
| `code/fetch_cited_papers.py` | **New file** — module-level docstring with inputs/outputs/run command per hard rule |

**Key documentation features:**
- Both output files are documented in the main rebuild guide (`README.md`)
- Script is listed in `code/README.md` with full usage instructions
- Rebuild-from-scratch compliance: script has module docstring, outputs are documented, can be regenerated anytime
- Markdown output is human-readable and includes clickable DOI/arXiv links

---

## Citation Summary

**Total papers cited:** 22

**Breakdown by type:**
- **Journal Articles:** 15 (2019–2025)
- **Conference Papers:** 2 (2024)
- **Books:** 1 (1994)
- **Reports & Documents:** 4 (2015–2025)

**Coverage by discipline:**
- AI and Sustainable Development: 8 papers
- NLP and text classification (BERT, Sentence-BERT): 3 papers
- SDG measurement and benchmarking: 4 papers
- Research-policy alignment: 2 papers
- General sustainability governance: 2 papers

**Key papers:**
- Vinuesa et al. (2020) — Nature Communications — foundational AI-SDG alignment
- Reimers & Gurevych (2019) — EMNLP — Sentence-BERT methodology
- Strauss et al. (2025) — AI & Society — research-policy gap analysis
- Ingram et al. (2025) — Scientometrics — SDG classification landscape

---

## How to Use These Files

### In dissertation
- **Forward references:** Link to `data/cited_papers.md` in methodology or supplementary materials
- **Appendix:** Include `data/cited_papers.json` as structured data export

### For paper fetching (future)
```bash
# Extract DOI from JSON, fetch full paper via DOI API
cat data/cited_papers.json | jq -r '.[] | select(.doi) | .doi'

# Or use marker to convert PDFs to markdown when available
cd marker
python convert_single.py paper.pdf
```

### For versioning
- Both `cited_papers.json` and `cited_papers.md` are tracked in git
- Regenerate if `writing/references.bib` changes:
  ```bash
  python code/fetch_cited_papers.py
  ```

---

## Rebuild from Scratch (Hard Rule Compliance)

✅ All outputs are reproducible:

1. **Script exists:** `/dissertation/code/fetch_cited_papers.py`
2. **Script is documented:** Module docstring with inputs, outputs, run command
3. **Source is documented:** `README.md` and `code/README.md` list the script
4. **Outputs are documented:** Both JSON and Markdown are listed in rebuild guide
5. **Can regenerate:** Run `python code/fetch_cited_papers.py` at any time

**If machine is lost:**
- Clone the repo → `git checkout` → `python code/fetch_cited_papers.py` → outputs regenerated

---

## Next Steps

1. **Optional:** If you collect full PDFs of cited papers, organize in `literature/cited_papers/` and use marker to convert to markdown
2. **Integration:** Consider linking `cited_papers.md` in dissertation appendix or supplementary materials
3. **Automation:** If bibliography grows, re-run the script before final submission (regenerates indices)

---

**Created by:** Claude Code  
**Dependencies:** `bibtexparser` (installed automatically)  
**Status:** Ready for use

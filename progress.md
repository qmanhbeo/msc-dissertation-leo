# Progress Log

## 2026-03-27
- **Proposal submitted** to supervisor

## 2026-04-05
- **Data collection began** (non-primary data to avoid ethical approval)
  
  ### Data Sources (13 fetch scripts in /code)
  | Script | Source | Purpose |
  |--------|--------|---------|
  | `fetch_openalex.py` | OpenAlex API | AI + SDG academic papers |
  | `fetch_osdg.py` | Zenodo #11441197 | OSDG Community Dataset |
  | `fetch_un_sdg.py` | UN Statistics | SDG indicators + policy PDFs |
  | `fetch_sdg_benchmark.py` | GitHub | SDG classification benchmark |
  | `fetch_sdg_publications.py` | sdgs.un.org | UN SDG publications |
  | `fetch_sdgi_corpus.py` | Hugging Face (UNDP) | SDGi Corpus (VNRs/VLRs) |
  | `fetch_sdgindex.py` | SDG Index | Sustainable Development Report 2025 |
  | `fetch_unsd.py` | UN SDG API | Official SDG taxonomy |
  | `fetch_aurora.py` | Zenodo #5224005 | 1.4M DOIs with SDG labels |
  | `fetch_sdg_news.py` | Zenodo #7523032 | IISD SDG news articles |
  | `fetch_ungdc.py` | Harvard Dataverse | UN General Debate Corpus |
  | `fetch_nlp4sg.py` | Hugging Face | NLP papers mapped to SDGs |
  | `fetch_un_ga.py` | UN Digital Library | UNGA resolutions + voting |

  ### Data Outputs (in /data)
  - `openalex/papers.jsonl` — 100 AI+SDG papers
  - `osdg/osdg_dataset.csv` — 43,025 labeled text excerpts
  - `sdgi_corpus/sdgi_corpus.parquet` — 5,880 VNR/VLR texts
  - `sdgindex/sdr2025_data.xlsx` — 208 countries × 17 SDGs
  - `unsd/goals.json`, `targets.json`, `indicators.json` — Official SDG taxonomy
  - `aurora/aurora_sdg_targets.csv` — 1.4M DOIs with SDG labels
  - `ungdc/TXT/` — 11,141 UNGA speeches (1946-2025)
  - `un_sdg/texts/` — UN AI Strategy + PARIS21 policy texts

- **Literature review performed** (SciSpace comprehensive mapping)
  
  ### Key Findings
  - ~1,684 papers synthesised across 9 thematic areas
  - **Established pattern:** AI research concentrates on SDGs 3, 7, 9, 11, 13; consistently neglects SDGs 5, 10, 16, 17
  - **Gap identified:** No existing study has systematically compared semantic content of AI research papers to policy documents at the SDG level
  - **Method validated:** SBERT validated for cross-domain semantic similarity (Bergman 2023, Justino 2025)
  - **Geographic bias:** AI research biased toward high-income country priorities
  
  ### Theoretical Frameworks
  - Epistemic communities (Haas 1992) — semantic alignment operationalises shared causal beliefs
  - Multiple Streams (Kingdon) — semantic alignment necessary but not sufficient for policy windows
  - Mode 1/2 knowledge production (Gibbons et al.) — supply-driven research → structural misalignment

  ### Key Papers to Cite
  - Vinuesa et al. (2020) Nature Communications — AI enabling/inhibiting 169 SDG targets
  - Cowls et al. (2021) Nature Machine Intelligence — 108 AI4SG projects
  - Singh et al. (2023) Sustainable Development — 20 years bibliometric analysis
  - Toney et al. (2024) FAccT — "Trust Issues": high-level agreement masks operational divergence

---

### Research Hypotheses (34 total, H1-H34)

| Category | Focus | Key Hypotheses |
|----------|-------|----------------|
| H1-H5 | Coverage Gap | Research over-indexes on SDGs 7, 9, 13; Policy on SDGs 1, 2, 3, 5, 10 |
| H6-H10 | Semantic Gap | Within-SDG divergence (e.g., research technical vs policy contextual) |
| H11-H15 | SDG-Specific | SDG 14 most neglected; SDG 13 most over-represented |
| H16-H20 | Structural | Papers narrower than policy chunks; temporal trends |
| H21-H24 | Contextual | Gap correlation with SDG Index performance |
| H25-H27 | New from Lit Review | Coverage/semantic gap correlation; bidirectional asymmetry |
| H28-H30 | Second Wave | AI-for vs AI-in framing gap; shared blind spot |
| H31-H34 | Extended Data | NLP4SG validation; VNR vs AI policy corpus differences |

**Key structural hypotheses:**
- **H25:** Coverage gap and semantic gap are *negatively* correlated — SDGs with most research show largest semantic divergence
- **H26:** Research ignores policy more than policy ignores research (asymmetric alignment)
- **H28:** "AI in sustainability" policy chunks semantically distant from research (framing gap)

---

### Assumptions Framework (key assumptions, A1-A25)

| ID | Assumption | Risk Level |
|----|-----------|------------|
| A1 | 6,172 papers representative of AI-for-sustainability field | Low-Medium |
| A2 | 13 policy docs represent "global AI/SDG policy discourse" | Medium |
| A15 | OSDG centroids may inflate policy alignment (calibration bias) | Medium-High |
| A16 | Cosine similarity measures topical overlap, not substantive alignment | **High** |
| A19 | Unit-of-analysis asymmetry: paper abstracts vs policy chunks | **High** |
| A20 | High alignment may = shared blind spot, not responsive research | **High** |

**Critical framing:** We measure *topical overlap* (necessary but not sufficient for alignment), not whether research actually influences policy.

---

### Methodology Pipeline

```
OSDG (30k texts) → SDG centroids (17) → Validate on Benchmark
                                                    ↓
OpenAlex papers + Policy chunks → SBERT embeddings → Cosine similarity
                                                    ↓
                    ┌──────────┴──────────┐
              Coverage Gap          Semantic Gap
         (SDG proportion profiles)  (intra-SDG cluster similarity)
                                                    ↓
                                      Kaggle SDG Index (contextualise)
```

**Two gap types:**
1. **Coverage Gap:** Which SDGs receive how much attention in research vs policy
2. **Semantic Gap:** Even where both discuss the same SDG, are they discussing the same aspects?

**Validation:** Centroid quality tested on SDG Benchmark (616 expert-verified texts) before applying to corpora.

---

*Last updated: 2026-04-05*

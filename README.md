# MSc AI and Sustainable Development — Dissertation

**University of Birmingham | School of Government**

This repository contains the dissertation research for the MSc AI and Sustainable Development programme at the University of Birmingham. The dissertation is a 60-credit independent research project (600 hours total) that culminates the MSc programme.

---

## Program Overview

The dissertation module enables demonstration of professional competence in a substantial AI and data science research project, applying material learned across the degree programme. Students utilize research methods, data science methods, and appropriate AI tools to design solutions that account for sustainable development frameworks.

### Learning Outcomes

By the end of this dissertation, I will be able to:

- Apply core machine learning techniques and algorithms to address sustainable development challenges
- Utilise research and data science skills and methods to answer a substantial research question
- Examine practical, ethical, social and cultural considerations in deploying AI solutions in international development contexts
- Design an end-to-end AI/ML solution for a sustainable development issue, considering technical constraints and local stakeholder perspectives
- Communicate effectively about the risks, limitations and unintended consequences of AI innovations for sustainable development

---

## Assessment

### Components

| Component | Weight | Format |
|-----------|--------|--------|
| Written Research Report | 75% | 8,000 words |
| Project Presentation | 25% | 10-minute recorded video (max 15 slides) |

### Report Structure & Word Counts

1. **Introduction** (~500 words) — Research topic, questions/objectives, and significance
2. **Literature Review** (~2,500 words) — Academic literature, theoretical frameworks, and existing research gaps
3. **Methodology** (~1,000 words) — Research approach, data collection, and analysis methods
4. **Findings/Results** (~2,000 words) — Research findings with supporting data, tables, and figures
5. **Discussion** (~1,500 words) — Interpretation of findings and implications for theory/practice
6. **Conclusion** (~500 words) — Key findings, limitations, and directions for future research

### Marking Criteria

Dissertations are assessed on:

- **Research Design and Methods** — Clear aims, appropriate methods, understanding of methodologies and limitations
- **Theoretical and Conceptual Perspectives** — Clear research questions, comprehensive literature review, understanding of strengths/weaknesses
- **Analysis and Originality** — Contextualisation, addressing research questions, original thinking and new insights
- **Relevance to Practice** — Linkage with practice, theory-practice integration, practical implications for sustainable development
- **Conclusions & Recommendations** — Clear, sound conclusions that address all key issues
- **Structure and Presentation** — Logical structure, coherent argument, clarity, Harvard referencing

### Grade Bands

- **Distinction:** 72%, 75%, 78% and above
- **Merit:** 62%, 65%, 68%
- **Pass:** 52%, 55%, 58%
- **Fail:** 42%, 45%, 48% and below

### Presentation Assessment

**Content:**
- Clear and easy-to-follow structure
- Helps audience understand the essence of the written report
- Intrigues audience to read the full report

**Presentation Skills:**
- Clear voice with good pace
- Effective use of time
- Professional body language and eye contact

---

## Project Timeline

| Milestone | Deadline | Description |
|-----------|----------|-------------|
| Initial Development | Feb–Mar 2026 | Develop preliminary ideas and topics |
| Project Proposal | 2 Mar 2026 | Submit draft proposal on Canvas for supervisor allocation |
| Supervisor Allocation | Mar 2026 | Receive allocated supervisor and initiate contact |
| Project Work | Mar–Jun 2026 | Active research and development; discuss ethics requirements with supervisor |
| Finalisation | July 2026 | Complete chapters; submit for supervisor feedback by 1 Aug |
| Independent Work | Aug 2026 | Refine project independently, proofread, record presentation |
| **Final Submission** | **1 Sep 2026, 12 pm** | **Report + Recorded Presentation due** |

### Supervisor Feedback Deadline

The final deadline for submitting draft chapters for supervisor feedback is **1 August**. No feedback will be provided on drafts submitted after this date. Supervisors take annual leave during summer — confirm their availability and last feedback date early.

---

## Supervision

### Supervisor Role

Your allocated supervisor will provide **advice and guidance** on:

- Defining and organising your research project
- Asking productive questions
- Tailoring project approach and coverage
- Making suitable methodological choices
- Improving structure and presentation

### What Supervisors Do NOT Do

- Conceive, direct, or manage your research project (you are the lead)
- Ensure you pass
- Proofread the final draft before submission
- Do line-by-line editing of drafts
- Debug programming code

### Supervision Requirements

- **Minimum:** 2 supervision meetings throughout the project period
- Additional meetings can be arranged based on individual needs
- Prepare in advance and send work to your supervisor in good time for review before meetings

### Your Responsibilities as a Student

- Set realistic deadlines and maintain regular communication
- Prepare for supervision meetings and provide work in advance
- Take notes during meetings for future reference
- Utilise Canvas resources from the research project module and other programme modules
- Ensure the quality and content of your dissertation is your responsibility

---

## Research Proposal Ideas

### Topic 1: Mapping Fair Reinforcement Learning for Resource Allocation

**Research Questions:**
- What structural clusters define fair RL for resource allocation research (2022–2026)?
- How is fairness operationalized across methodological streams?
- Are theoretical (e.g., RMAB) and deep RL approaches integrated or fragmented?
- Which application domains dominate the field?
- What recurring methodological or governance gaps appear across clusters?

**Motivation:**
Resource allocation lies at the core of sustainability, determining how scarce resources are distributed under constraints of uncertainty, efficiency, and equity. RL has been increasingly proposed for dynamic allocation in smart grids, humanitarian logistics, and disaster response. However, the structural organization of fair RL for resource allocation remains unclear.

**Method:**
- Collect corpus using OpenAlex/Scopus
- Construct citation network with community detection (e.g., Louvain algorithm)
- Generate embeddings (Sentence-BERT)
- Apply topic modeling (e.g., BERTopic)
- Overlay semantic themes onto citation clusters
- Analyze centrality, fragmentation, and thematic distribution

**Contribution:**
Provides a structural and semantic mapping of fair RL for resource allocation, identifying dominant paradigms, fragmentation patterns, and underexplored intersections between fairness, sustainability, and governance.

---

### Topic 2: Semantic Alignment Between AI Sustainability Research and Policy Frameworks ⭐ **CHOSEN**

**Core Question:**
To what extent does academic AI-for-sustainability research align with sustainability policy priorities?

**Motivation:**
AI governance emphasizes that technical development and policy objectives must co-evolve. However, academic AI research and sustainability policy documents may prioritize different themes, potentially leading to technically sophisticated systems that fail to address real-world sustainability needs. Previous work has applied topic modeling to map AI/ML to SDGs in literature, but no study has comprehensively compared academic literature with government/IO policy documents.

**Method:**
- Collect AI-for-sustainability research abstracts (OpenAlex/Scopus)
- Collect sustainability policy documents (UN SDGs, national AI strategies, climate reports)
- Generate embeddings (Sentence-BERT)
- Measure semantic similarity (cosine similarity)
- Apply topic modeling to both corpora
- Compare thematic prominence and divergence
- Identify underrepresented policy themes in research literature

**Contribution:**
Provides a quantitative assessment of alignment between AI research and sustainability governance priorities, revealing thematic gaps and structural mismatches.

---

## Dissertation Analysis Plan

### Research Question
**To what extent does academic AI-for-sustainability research align with policy priorities for sustainable development?**

Which SDGs do researchers emphasize vs. which do policymakers prioritize? Where are the gaps?

### Analysis Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                    RESEARCH CORPUS                                  │
│                  (What academics publish)                           │
├─────────────────────────────────────────────────────────────────────┤
│  • OpenAlex Papers (100 papers on AI + sustainability, 2025)        │
│  • OSDG Dataset (43,025 text excerpts, pre-labeled with SDGs)       │
│  • GitHub Benchmark (1,251 expert-verified texts for validation)    │
└───────────────────────────┬──────────────────────────────────────────┘
                            │
                            ↓
                 ┌──────────────────────┐
                 │  Topic Modeling 1    │
                 │  (Research themes)   │
                 │  + Extract SDG focus │
                 └──────────────────────┘
                            │
┌───────────────────────────┴──────────────────────────────────────────┐
│                          COMPARATIVE ANALYSIS                        │
├─────────────────────────────────────────────────────────────────────┤
│  • Semantic Similarity (cosine distance between embeddings)          │
│  • Theme Comparison (which topics overlap? which diverge?)           │
│  • SDG Alignment (% research vs % policy per SDG)                   │
│  • Gap Analysis (critical research gaps, ignored policy priorities) │
└───────────────────────────┬──────────────────────────────────────────┘
                            │
                            ↓
┌─────────────────────────────────────────────────────────────────────┐
│                     POLICY CORPUS                                   │
│               (What policymakers prioritize)                        │
├─────────────────────────────────────────────────────────────────────┤
│  • UN AI Strategy Guide (June 2021, 298k chars)                     │
│  • PARIS21 Report (April 2024, 61k chars)                          │
└───────────────────────────┬──────────────────────────────────────────┘
                            │
                            ↓
                 ┌──────────────────────┐
                 │  Topic Modeling 2    │
                 │  (Policy priorities) │
                 │  + Extract SDG focus │
                 └──────────────────────┘
                            │
┌───────────────────────────┴──────────────────────────────────────────┐
│                       CONTEXTUAL DATA                                │
├─────────────────────────────────────────────────────────────────────┤
│  • Kaggle SDG Index (2000–2022 country performance scores)          │
│    → Which SDGs have worst progress?                                │
│    → Is research focused on the most critical goals?                │
└─────────────────────────────────────────────────────────────────────┘
                            │
                            ↓
        ┌───────────────────────────────────────┐
        │   FINDINGS & INTERPRETATION           │
        ├───────────────────────────────────────┤
        │ ✓ Alignment scores (quantified)       │
        │ ✓ Which SDGs are aligned/misaligned  │
        │ ✓ Critical research gaps              │
        │ ✓ Policy priorities under-researched  │
        │ ✓ Heatmaps and visualizations         │
        └───────────────────────────────────────┘
```

### Datasets & Their Role

| Dataset | Records | Years | Role in Analysis |
|---------|---------|-------|-----------------|
| **OpenAlex** | 100 papers | 2025 | Extract contemporary research themes |
| **OSDG** | 43,025 excerpts | Multi-year | Identify which SDGs research addresses (pre-labeled) |
| **UN Policy** | 2 PDFs | 2021–2024 | Extract official policy priorities & themes |
| **GitHub Benchmark** | 1,251 texts | 2024+ | Validate topic modeling accuracy |
| **Kaggle SDG Index** | 4,140 records | 2000–2022 | Provide context: which SDGs have poorest progress? |

### Research Questions Answered

1. **What themes dominate academic AI research?**
   - Method: Topic modeling on OpenAlex + OSDG corpus
   - Output: Topic clusters, word frequencies, semantic embeddings

2. **What themes dominate policy documents?**
   - Method: Topic modeling on UN policy texts
   - Output: Policy priorities, governance themes, SDG focus

3. **How aligned are they?**
   - Method: Cosine similarity between research & policy embeddings
   - Output: Alignment score, semantic distance matrix

4. **Which SDGs get mismatched attention?**
   - Method: Compare SDG distribution in OSDG (research) vs policy texts
   - Output: Table/heatmap of alignment by SDG

5. **Is misalignment problematic?**
   - Method: Cross-reference with Kaggle SDG performance data
   - Output: Which neglected SDGs have worst development progress?

---

## Time Commitment

**Total: 600 hours** including:

- Dissertation seminars
- Supervision meetings
- Research question development
- Chapter planning
- Literature sourcing
- Data sourcing
- Writing
- Revisions

---

## Support Resources

### Academic Support

- **Personal Tutor:** Advise on academic skills and support services
- **Birmingham International Academy:** English language support
- **Academic Skills Gateway:** Wide range of academic resources and skills training

### Personal Issues & Wellbeing

The **Wellbeing Department** offers support for personal circumstances affecting your studies:

- **Location:** Muirhead Tower, Rooms 641 & 642
- **Email:** gov.wellbeing@contacts.bham.ac.uk
- **Phone:** 0121 4148060 or 0121 414 8452

**Services:**
- Drop-in appointments
- Assignment extensions (where appropriate)
- Removal of late submission penalties for extenuating circumstances
- Leave of absence authorisation
- Referral to relevant support services

**Student Services** also provides support on health, wellbeing, funding, graduation, and postgraduate study.

---

## Key Documents & Resources

**Project Planning:**
- `proposal-2026-03-27.md` — Research proposal ideas and initial planning
- `CLAUDE.md` — Guidelines for Claude collaboration on dissertation
- `README.md` — This file

**Research Support:**
- `notes/DATA_SUMMARY.md` — Complete data exploration results and field descriptions
- `code/README.md` — Data fetching scripts documentation
- `code/fetch_*.py` — 5 reproducible data pipeline scripts
- `Dissertation_Handbook_26.pdf` — Full university dissertation handbook

**Data:**
- `data/openalex/` — Academic papers (100 records, 2025)
- `data/osdg/` — SDG-labeled text excerpts (43,025 records)
- `data/un_sdg/` — Policy documents and extracted text (2 PDFs)
- `data/sdg_benchmark/` — Expert-verified benchmark dataset (1,251 texts)
- `data/kaggle/` — Historical SDG performance data (4,140 records, 2000–2022)

---

## Current Status

*Last updated: 2026-04-05*

### ✅ Done

| Area | Output |
|------|--------|
| Topic selection | Topic 2: AI–Sustainability Research–Policy Alignment |
| Data fetching | All 5 sources fetched (87,000+ records, 48.7 MB) |
| Preprocessing — papers | `data/openalex/papers_clean.jsonl` (94 papers) |
| Preprocessing — policy | `data/un_sdg/policy_chunks.jsonl` (253 chunks) |
| Preprocessing — OSDG | `data/osdg/osdg_clean.jsonl` (30,534 rows, agreement ≥ 0.5) |
| Preprocessing — benchmark | `data/sdg_benchmark/benchmark_clean.jsonl` (616 rows) |
| Embeddings | `data/embeddings/*.npy` — all 4 corpora embedded (all-MiniLM-L6-v2, 384-dim) |
| Methodology design | `notes/METHODOLOGY_DECISIONS.md` — pipeline, gap types, validation approach |
| Assumptions | `notes/ASSUMPTIONS.md` — 14 assumptions documented with risk levels |
| Hypotheses | `notes/HYPOTHESES.md` — 24 pre-registered hypotheses across 5 categories |

### 🔄 Current Focus

**Literature review** — reading key papers to ground hypotheses and strengthen the theoretical framework before running analysis.

Key papers to engage:
- Vinuesa et al. (2020) — AI and the SDGs (Nature Communications)
- Sachs et al. (annual) — Sustainable Development Report
- Reimers & Gurevych (2019) — Sentence-BERT
- OSDG dataset paper
- SDG Benchmark dataset paper
- Literature on research-policy gaps in AI governance

### 📋 Remaining Analysis (parked — resume after literature review)

1. **SDG centroids** — `code/sdg_centroids.py` — compute per-SDG mean embeddings from OSDG
2. **Centroid validation** — `code/validate_centroids.py` — accuracy/F1 on benchmark; validates measurement instrument
3. **Alignment scoring** — `code/alignment_score.py` — cosine similarity of papers + policy chunks vs. all 17 centroids
4. **Coverage gap** — `code/coverage_gap.py` — SDG proportion profiles, bar/radar charts
5. **Semantic gap** — `code/semantic_gap.py` — intra-SDG similarity between research and policy clusters
6. **Kaggle context** — `code/kaggle_context.py` — correlate gaps with global SDG performance scores
7. **Topic modeling** — `code/topic_model.py` *(optional)* — surface themes within high-scoring SDG clusters

### 🗓 Timeline

| Phase | When | Status |
|-------|------|--------|
| Data & preprocessing | Mar–Apr 2026 | ✅ Done |
| Literature review | Apr–May 2026 | 🔄 Active |
| Analysis (steps 1–7 above) | May–Jun 2026 | ⏸ Parked |
| Visualization | Jun–Jul 2026 | Not started |
| Writing | Jul–Aug 2026 | Not started |
| Supervisor feedback deadline | 1 Aug 2026 | — |
| Final submission | 1 Sep 2026 | — |

---

**Programme:** MSc AI and Sustainable Development
**Submission Deadline:** 1 September 2026, 12:00 pm
**Last Updated:** 2026-04-05

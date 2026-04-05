# Methodology Decisions & Research Paper Notes

Last updated: 2026-04-05

---

## Core Research Question

> To what extent does academic AI-for-sustainability research **align** with sustainability policy priorities — and where do the gaps lie?

---

## Two Types of Gap (Critical Distinction)

### 1. Coverage Gap (what we measure first)
- **Definition:** Difference in the *proportion* of attention each SDG receives in research vs. policy
- **How measured:** For each SDG, compare share of research papers vs. policy chunks that score highly on that SDG centroid
- **What it tells you:** Which SDGs are overrepresented in research vs. policy, and vice versa
- **Limitation to name in paper:** This is a quantitative/distributional finding — it doesn't tell you *what* is being said within each SDG

### 2. Semantic Gap (what we measure second — stronger contribution)
- **Definition:** Even where both corpora engage the same SDG, they may be discussing fundamentally different aspects of it
- **Example:** Both research and policy engage SDG 13 (Climate Action), but research discusses ML for emissions forecasting while policy discusses carbon pricing and adaptation frameworks
- **How measured:** For each SDG, collect high-scoring research chunks and high-scoring policy chunks → compute **intra-SDG cosine similarity** between the two clusters
  - High similarity → both corpora are talking about the same things within this SDG
  - Low similarity → semantic divergence even within shared SDG engagement
- **Optional extension:** Run topic modeling *within* each SDG cluster to surface "research theme vs. policy theme" pairs
- **Why this matters for the paper:** Allows you to say not just "research neglects SDG 1" but "even where research engages SDG 13, it addresses different aspects than policy does" — a more nuanced and publishable finding

---

## Pipeline Architecture

```
OSDG (30,534 labeled texts)
    → compute SDG centroid embeddings (one per SDG, SDGs 1–16)

SDG Benchmark (616 expert-verified texts)
    → evaluate centroid quality: nearest-centroid accuracy/F1
    → also provides SDG 17 reference (OSDG missing SDG 17)

OpenAlex papers (94, cleaned)        ──┐
Policy chunks (253, cleaned)          ──┤── Sentence-BERT embeddings
                                         │
                                         ▼
                              SDG alignment scores (cosine similarity to each centroid)
                                         │
                              ┌──────────┴──────────┐
                         Coverage gap          Semantic gap
                    (SDG proportion profiles)  (intra-SDG cluster similarity)
                                         │
                              Kaggle SDG Index (country performance)
                              → contextualise: are gaps worst where world is most behind?
```

---

## Measurement Instrument Validation (Important for Methodology Chapter)

- **No classifier is trained** — Sentence-BERT is pre-trained and frozen; SDG centroids are computed as mean embeddings, not trained parameters
- **No train/val/test split needed** in the ML sense — nothing can overfit
- **What IS needed:** centroid validation
  - Use OSDG to build centroids
  - Use SDG Benchmark as held-out evaluation: does nearest-centroid assignment predict correct SDG label?
  - Report accuracy and macro-F1 as a methodological sanity check
  - Frame in paper as: "We validate our SDG measurement instrument before applying it to the research and policy corpora"
- **Implication for writing:** Be precise — this is not a supervised ML study; it is a semantic similarity study with a validated measurement instrument

---

## Hypotheses to Test

1. **H1 (Coverage):** Research over-indexes on technical SDGs (SDG 9 Industry/Innovation, SDG 13 Climate, SDG 7 Energy) while policy emphasizes human development SDGs (SDG 1 Poverty, SDG 2 Hunger, SDG 3 Health, SDG 5 Gender Equality)
2. **H2 (Semantic):** Even within shared SDGs, research and policy discuss different aspects — research is more technical/methodological, policy is more applied/contextual
3. **H3 (World performance):** The alignment gaps are largest for SDGs where global progress (Kaggle SDG Index) is weakest — suggesting research effort is misallocated relative to real-world need

---

## TODOs — Implementation

- [x] `code/embeddings.py` — all 4 corpora embedded with `all-MiniLM-L6-v2` (384-dim); saved to `data/embeddings/*.npy`
  - Note: switched from `all-mpnet-base-v2` to `all-MiniLM-L6-v2` for CPU/WSL practicality
  - Sanity check showed same-SDG vs diff-SDG similarity was marginal; centroid validation will be the real test
  - Inter-centroid analysis confirmed sensible SDG cluster structure (SDG 1↔10: 0.887, SDG 7↔16: 0.183)
- [ ] `code/sdg_centroids.py` — compute per-SDG mean embeddings from OSDG; handle SDG 17 using benchmark
- [ ] `code/validate_centroids.py` — evaluate centroid quality on SDG benchmark; report accuracy + macro-F1
- [ ] `code/alignment_score.py` — for each paper/chunk, compute cosine similarity to all 17 SDG centroids
- [ ] `code/coverage_gap.py` — compare SDG proportion profiles (research vs. policy); produce bar/radar charts
- [ ] `code/semantic_gap.py` — per-SDG: compute intra-SDG cosine similarity between research cluster and policy cluster
- [ ] `code/topic_model.py` — optional: topic modeling within high-scoring SDG clusters to surface theme pairs
- [ ] `code/kaggle_context.py` — join SDG Index scores with gap scores; correlation analysis

⏸ **Analysis parked as of 2026-04-05 — focus shifted to literature review.**

---

## TODOs — Writing

- [ ] **Methodology chapter:** Explain why cosine similarity + centroids is appropriate; cite Sentence-BERT (Reimers & Gurevych, 2019); note that no classifier is trained
- [ ] **Methodology chapter:** Include centroid validation results as a subsection ("Validating the SDG Measurement Instrument")
- [ ] **Methodology chapter:** Clearly define the two gap types (coverage gap vs. semantic gap) with examples
- [ ] **Findings chapter:** Present coverage gap first (SDG proportion profiles) with visualizations
- [ ] **Findings chapter:** Present semantic gap second (intra-SDG similarity scores) — this is the novel contribution
- [ ] **Discussion chapter:** Interpret gaps in light of Kaggle SDG Index — which neglected SDGs are also most off-track globally?
- [ ] **Discussion chapter:** Acknowledge limitation — 94 papers is a small corpus; results are indicative, not definitive; could be extended
- [ ] **Discussion chapter:** Suggest implications — research funding priorities, AI-for-SDG agenda-setting

---

## Key Citations to Track Down

- Reimers & Gurevych (2019) — Sentence-BERT paper
- OSDG dataset paper (cite the dataset source)
- SDG Benchmark paper (cite the dataset source)
- Sachs et al. (annual) — Sustainable Development Report (source of Kaggle SDG Index data)
- UN 2030 Agenda for Sustainable Development (2015) — foundational SDG document

---

## OSDG Coverage Note

- OSDG corpus covers SDGs 1–16 only (SDG 17 missing from labeled data)
- SDG 17 centroid will be built from SDG Benchmark data instead
- **Flag in Methodology chapter:** "SDG 17 centroid derived from benchmark corpus (n=31) rather than OSDG due to absence of SDG 17 labels in OSDG dataset"

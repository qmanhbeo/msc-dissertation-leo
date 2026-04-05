# Methodology Decisions & Research Paper Notes

Last updated: 2026-04-05

---

## What We Are Actually Measuring (precise statement — critical)

**We are NOT measuring:**
- Whether research actually influences policy
- Whether research findings are adopted by policymakers
- Substantive alignment (whether research and policy solve the same problem in compatible ways)

**We ARE measuring:**
- **Topical overlap** — whether research and policy discuss the same subject areas, as proxied by semantic similarity in embedding space
- Topical overlap is the *necessary but not sufficient* condition for research-policy dialogue. You cannot align on what you don't discuss. But shared vocabulary does not guarantee shared meaning or direction.

**Critical framing (from literature review):**
Policy and research are different speech acts. Policy uses performative modal language to articulate commitments ("must," "should"). Research uses hedged indicative language to report findings ("we show," "results suggest"). High cosine similarity reflects shared subject matter — not that they are "saying the same thing." All findings must be framed as "research and policy show [high/low] *topical overlap* on SDG X," never simply "alignment." This is not a weakness — topical overlap is exactly the right first diagnostic.

---

## Core Research Question

> To what extent does academic AI-for-sustainability research show *topical overlap* with sustainability policy priorities — and where do the gaps lie?

---

## Positioning Statement (how to describe our contribution precisely)

We are advancing measurement from **Level 1 to Level 2** of a three-level alignment hierarchy:

| Level | What it measures | Approach | Who does it |
|-------|-----------------|----------|-------------|
| 1 — Lexical | Same keywords/terms for an SDG | Keyword dictionaries, bibliometrics | Most prior work (Armitage 2020, Singh 2023) |
| 2 — Topical | Same subject areas semantically | Embedding similarity, centroids | **This study** |
| 3 — Operational | Same approach to the problem | Requires structured comparison of proposed actions | Toney et al. (2024) approximates this |

Our within-SDG semantic gap analysis (H6–H10) approximates Level 3 without claiming to reach it. This framing should appear in the Introduction ("we advance beyond keyword mapping"), Methodology ("we measure topical alignment"), and Discussion ("even topical overlap may mask operational divergence, per Toney et al. 2024").

---

## Corpus Concentration Asymmetry (methodological note)

Research corpus: 94 papers from 94 independent authors → high diversity of perspectives
Policy corpus: 253 chunks from 2 documents from 2 authoring teams → high autocorrelation

Implication: Policy SDG profile reflects 2 editorial decisions amplified across 253 chunks. A document that heavily emphasises SDG 9 produces dozens of SDG-9 chunks. This can inflate apparent policy emphasis on certain SDGs.

**Required mitigations:**
- Report per-document profiles (PARIS21 vs UN AI Strategy) alongside combined figures
- When interpreting coverage gaps, weight by document not chunk count
- Consistently frame as "in these two UN documents" not "in policy discourse"

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

OpenAlex papers (6,172, cleaned)     ──┐
Policy chunks (1,211 from 13 docs)   ──┤── Sentence-BERT embeddings
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

- [x] **Corpus expanded (2026-04-05):**
  - Papers: 94 → 6,172 (4 OpenAlex queries, 2,000 cap each, deduped, abstracts required)
  - Policy: 2 docs/253 chunks → 13 docs/1,211 chunks (UN SDG Progress Report, IPCC AR6, UK/Singapore/Germany/AU national AI strategies, UN AI Advisory Body 2024, UNESCO, EU HLEG, OECD, US AI Bill of Rights)
- [x] `code/embeddings.py` — all 4 corpora embedded with `all-MiniLM-L6-v2` (384-dim); saved to `data/embeddings/*.npy`
  - papers.npy: (6172, 384); policy.npy: (1211, 384)
  - Sanity check showed same-SDG vs diff-SDG similarity was marginal (expected — 2-sample artifact); centroid validation will be the real test
- [ ] `code/sdg_centroids.py` — compute per-SDG mean embeddings from OSDG; handle SDG 17 using benchmark
- [ ] `code/validate_centroids.py` — evaluate centroid quality on SDG benchmark; report accuracy + macro-F1
- [ ] `code/alignment_score.py` — for each paper/chunk, compute cosine similarity to all 17 SDG centroids
- [ ] `code/coverage_gap.py` — compare SDG proportion profiles (research vs. policy); produce bar/radar charts
- [ ] `code/semantic_gap.py` — per-SDG: compute intra-SDG cosine similarity between research cluster and policy cluster
- [ ] `code/topic_model.py` — optional: topic modeling within high-scoring SDG clusters to surface theme pairs
- [ ] `code/kaggle_context.py` — join SDG Index scores with gap scores; correlation analysis

- [ ] `code/alignment_score.py` — add **bidirectional scoring**: also build research-side centroids (per inferred SDG) and score policy chunks against them (needed for H26)
- [ ] `code/coverage_semantic_interaction.py` — correlate coverage gap magnitude vs semantic gap magnitude per SDG (needed for H25 — the key structural finding)
- [ ] OSDG circularity check: after scoring, compare mean alignment scores for research vs policy to detect systematic calibration bias (see A15)

⏸ **Analysis parked as of 2026-04-05 — focus shifted to literature review.**

---

## TODOs — Writing

- [ ] **Methodology chapter:** Explain why cosine similarity + centroids is appropriate; cite Sentence-BERT (Reimers & Gurevych, 2019); note that no classifier is trained
- [ ] **Methodology chapter:** Include centroid validation results as a subsection ("Validating the SDG Measurement Instrument")
- [ ] **Methodology chapter:** Clearly define the two gap types (coverage gap vs. semantic gap) with examples
- [ ] **Findings chapter:** Present coverage gap first (SDG proportion profiles) with visualizations
- [ ] **Findings chapter:** Present semantic gap second (intra-SDG similarity scores) — this is the novel contribution
- [ ] **Discussion chapter:** Interpret gaps in light of Kaggle SDG Index — which neglected SDGs are also most off-track globally?
- [ ] **Discussion chapter:** Acknowledge limitations — unit-of-analysis asymmetry (paper abstracts vs policy chunks); corpus scope ("international institutional discourse", not all policy); English-language bias; topical overlap ≠ substantive alignment
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

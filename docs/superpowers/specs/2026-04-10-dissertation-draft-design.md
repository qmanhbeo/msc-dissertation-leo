# Dissertation Draft Design

**Date:** 2026-04-10
**Topic:** Semantic alignment between AI-for-sustainability research and SDG policy frameworks
**Format:** LaTeX (.tex + .bib), IMRaD structure
**Word limit:** 8,000 words (no hard trim on first draft)
**Programme:** MSc AI and Sustainable Development, University of Birmingham, School of Government
**Deadline:** 1 September 2026 (supervisor draft: 1 August 2026)

---

## Structure: Classic IMRaD (Approach A)

### Abstract (~250 words, not counted)
- RQ, method in two sentences, four bullet findings, one-sentence implication

### 1. Introduction (~700 words)
- Hook: AI-sustainability coupling vs. research-policy tracking
- Gap: bibliometrics shows *what* gets researched; no semantic distance from policy at SDG level
- Contribution: Level 1 (lexical) → Level 2 (topical) measurement; first bidirectional analysis; first coverage × semantic gap interaction
- RQ: *To what extent does AI-for-sustainability research show topical overlap with SDG policy priorities, and where do the gaps lie?*
- Roadmap sentence

### 2. Literature Review (~1,600 words)
- 2.1 AI and SDGs: bibliometric consensus (Vinuesa 2020, Singh 2023, Cowls 2021, Nedungadi 2024)
- 2.2 Research-policy alignment: temporal gap (Strauss 2025), metric divergence (Sioumalas-Christodoulou 2025), operational divergence (Toney 2024)
- 2.3 Measurement: keyword limits (Armitage 2020), SBERT for policy (Bergman 2023, Gjorgjevikj 2025)
- 2.4 Theory: epistemic communities (Haas 1992), Mode 1/2 (Gibbons), productive misalignment
- 2.5 Gap: no within-SDG semantic comparison, no bidirectional, no coverage × semantic interaction

### 3. Methodology (~2,000 words)
- 3.1 Conceptual framing: topical overlap vs substantive alignment; three-level hierarchy
- 3.2 Research corpus: 6,172 OpenAlex abstracts, 2018–2025, 17 SDGs × 4 AI terms
- 3.3 Policy corpus: 47,005 chunks from 2,392 documents (curated AI/SDG docs, SDGi VNR/VLR, UNGDC speeches)
- 3.4 Embedding model: all-MiniLM-L6-v2 (384-dim, L2-normalised)
- 3.5 SDG measurement instrument: OSDG centroids (SDGs 1–16), benchmark centroid (SDG 17); validation F1=0.733; SDG 8 (F1=0.531) and SDG 11 (F1=0.519) flagged
- 3.6 Coverage gap: hard-assignment profiles; document-weighting (A19)
- 3.7 Semantic gap: centroid-to-centroid cosine distance; chunk cap 50; three-cap sensitivity
- 3.8 Coverage × semantic interaction; bidirectional asymmetry (H26)
- 3.9 SDG Index contextualisation
- 3.10 Assumptions and mitigations: A15 circularity, A19 weighting, SDG 4 artefact, unit-of-analysis asymmetry

### 4. Results (~2,500 words)
- 4.1 Instrument validation: macro-F1=0.733; per-SDG F1; SDG 8 (0.531), SDG 11 (0.519) flagged; SDG 16 (0.857) highlight
- 4.2 Coverage gap: policy → SDG 13 (36%), SDG 17 (35%), SDG 16 (20%); research → SDG 4 (22%), SDG 9 (17%); SDG 4 artefact caveat; doc-weighted vs chunk-level; A15 flag
- 4.3 Semantic gap: SDG 8 (0.292), SDG 3 (0.285), SDG 16 (0.284) largest; SDG 17 (0.182), SDG 9 (0.201) smallest; sensitivity robustness
- 4.4 Coverage × semantic interaction: H25 null (r=0.145, p=0.578); 2×2 typology; SDG 9 vs SDG 3 contrasting cases
- 4.5 Directional asymmetry (H26): 0.472 vs 0.353; A15 caveat; partial support
- 4.6 SDG Index context: H21/H22 null; H23 confirmed (SDG 13 = 82.98)

### 5. Discussion (~1,600 words)
- 5.1 Two independent dimensions: H25 null as structural finding
- 5.2 SDG 9 paradox: high coverage + low semantic gap (H29 — shared blind spot?)
- 5.3 Productive vs problematic misalignment (H27): SDG 3/8 problematic; SDG 9 potentially productive
- 5.4 SDG 13 anomaly: policy-dominated coverage + index inflation + moderate semantic gap
- 5.5 Implications: funding priorities, SDG-aware publication incentives, bidirectional governance framing
- 5.6 Limitations: A15, SDG 4 artefact, English-language bias, topical ≠ substantive, corpus scope, single time-point

### 6. Conclusion (~500 words)
- Two-dimension finding as central contribution
- H25 null reframed as positive result
- Future work: longitudinal, fine-tuned models, 169-target level, Global South corpus

---

## Key Data Points (from analysis, 2026-04-10)

| Metric | Value |
|--------|-------|
| Research corpus | 6,172 papers |
| Policy corpus | 47,005 chunks, 2,392 documents |
| Centroid validation macro-F1 | 0.733 (PASS) |
| Largest coverage gap | SDG 13 (policy 36.1% vs research 5.0%) |
| Largest semantic gap | SDG 8 (0.292) |
| H25 correlation | r=0.145, p=0.578 (null) |
| H26 asymmetry | 0.472 vs 0.353 (policy engages research more) |
| H23 SDG 13 index score | 82.98 (highest of 16 SDGs) |

## Output Files
- `writing/dissertation.tex`
- `writing/references.bib`

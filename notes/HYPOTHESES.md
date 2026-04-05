# Research Hypotheses

Last updated: 2026-04-05

These hypotheses are stated **before** running alignment analysis to prevent HARKing (Hypothesising After Results are Known). Each is testable with the pipeline outputs. Marked with expected direction and rationale.

---

## H1–H5: Coverage Gap (Which SDGs get how much attention?)

### H1 — Research over-indexes on technical/environmental SDGs
> **Research corpus will show disproportionately high alignment scores for SDGs 7, 9, and 13 relative to the policy corpus.**

- SDG 7 (Affordable and Clean Energy): AI for grid optimisation, renewable forecasting are mainstream research topics
- SDG 9 (Industry, Innovation, Infrastructure): AI is inherently an "innovation" topic; many papers frame AI itself as SDG 9 progress
- SDG 13 (Climate Action): The largest AI+sustainability research vertical; climate ML is a well-funded subfield
- **Rationale:** These are technically tractable SDGs where AI has obvious, demonstrable applications. They also attract funding and publication incentives.
- **Measured by:** Research SDG proportion profile vs. policy SDG proportion profile

### H2 — Policy over-indexes on human development SDGs
> **Policy corpus will show disproportionately higher alignment scores for SDGs 1, 2, 3, 5, and 10 than the research corpus.**

- SDG 1 (No Poverty): Core UN mandate; policy documents constantly reference poverty reduction
- SDG 2 (Zero Hunger): Food security is a primary policy priority, especially for developing nations
- SDG 3 (Good Health): Health AI is a research topic too, but policy framing is broader (access, equity)
- SDG 5 (Gender Equality): Prominent in UN policy discourse; underrepresented in technical AI papers
- SDG 10 (Reduced Inequalities): A central policy concern; less naturally framed as an AI research problem
- **Rationale:** These SDGs require institutional, political, and social interventions — not just technical AI solutions. Policy documents naturally emphasise them; researchers may find them less tractable.

### H3 — SDG 16 is a policy-exclusive concern
> **SDG 16 (Peace, Justice, Strong Institutions) will appear significantly more in policy than in research.**

- SDG 16 had the lowest inter-centroid similarity to other SDGs (max 0.47 with SDG 9) — it is semantically isolated
- UN policy frameworks frequently reference governance, rule of law, and institution-building
- AI research on SDG 16 exists (AI for governance, bias in legal systems) but is niche
- **Rationale:** SDG 16 is fundamentally a governance SDG. Its language does not naturally overlap with technical AI research framing.

### H4 — SDG 17 (Partnerships) will be high in policy, low in research
> **SDG 17 (Partnerships for the Goals) will be much more prominent in policy texts than in research abstracts.**

- Policy documents routinely call for "multi-stakeholder partnerships" and "international cooperation"
- Research papers rarely frame their contributions as SDG 17 unless specifically studying partnership frameworks
- **Rationale:** SDG 17 is a meta-SDG about how to achieve the others — it is naturally expressed in policy language, not technical research language.

### H5 — The environmental SDG cluster (6, 12, 13, 14, 15) will split between research and policy
> **Research will emphasise SDGs 13 and 7 within the environmental cluster; policy will more evenly spread across SDGs 6, 12, 14, and 15.**

- Inter-centroid analysis shows SDGs 6, 7, 12, 13, 14, 15 form a tight cluster (similarities 0.57–0.74)
- Research gravitates toward AI-tractable environmental problems (climate prediction, energy systems)
- Policy must address the full range: water (6), consumption (12), oceans (14), biodiversity (15)
- **Rationale:** Research funding follows tractability and impact; policy responsibility is broader.

---

## H6–H10: Semantic Gap (What is said within each SDG?)

### H6 — Within SDG 13 (Climate), research is technical; policy is contextual
> **Research chunks aligned to SDG 13 will discuss modelling, forecasting, and ML methods; policy chunks will discuss adaptation, finance, and equity.**

- Expected research themes: emissions modelling, climate downscaling, extreme weather prediction, carbon footprint optimisation
- Expected policy themes: climate finance, loss and damage, adaptation frameworks, vulnerable populations
- **Measured by:** Intra-SDG cosine similarity between research and policy clusters; topic modelling within each cluster
- **Rationale:** This is the most studied SDG in AI research. The semantic divergence — if large — is a strong finding.

### H7 — Within SDG 3 (Health), research focuses on diagnosis; policy focuses on access
> **Research chunks aligned to SDG 3 will cluster around AI diagnostics and drug discovery; policy chunks will emphasise health system access, equity, and prevention.**

- AI health research is dominated by medical imaging, genomics, and clinical prediction
- Policy health discourse centres on universal health coverage, maternal mortality, infectious disease control
- **Rationale:** The "AI for health" research agenda and the global health policy agenda are well-documented to be misaligned (see existing literature on AI and global health equity).

### H8 — Within SDG 4 (Education), research is about personalisation; policy is about access
> **Research chunks will focus on adaptive learning systems and AI tutors; policy chunks will emphasise out-of-school children, teacher training, and foundational literacy.**

- EdTech AI research focuses on optimising learning for connected, enrolled students
- Policy focuses on the 250+ million children not in school, and low-resource settings
- **Rationale:** A classic "last mile" gap between what AI can optimise and what policy needs.

### H9 — The semantic gap will be largest for SDGs where the research corpus is thinnest
> **SDGs with few highly-aligned research papers (e.g. SDG 2, SDG 14, SDG 16) will show the largest intra-SDG semantic divergence from policy.**

- When few research papers address an SDG, the ones that do may be highly specialised or tangentially related
- This produces high semantic distance from the policy framing of the same SDG
- **Rationale:** Coverage gap and semantic gap are expected to be positively correlated.

### H10 — SDG 9 (Innovation) will show the smallest semantic gap
> **Research and policy will be most semantically aligned within SDG 9.**

- Both research abstracts and policy documents discussing SDG 9 naturally use similar language: innovation, infrastructure, industrial transformation, technology
- SDG 9 is the SDG most natively expressed in research/technology language
- **Rationale:** SDG 9 is the point of natural convergence between the two corpora.

---

## H11–H15: SDG-Specific Directional Hypotheses

### H11 — SDG 14 (Life Below Water) will be the most neglected in research
> **SDG 14 will have the lowest average alignment score across research papers.**

- SDG 14 scored lowest globally in 2022 (50.5) AND is likely under-researched
- Ocean science AI exists but is a small niche relative to climate or health AI
- SDG 14 centroid is well-separated from most others (inter-centroid sims mostly 0.3–0.6)
- **Implication if confirmed:** The most off-track SDG is also the most neglected in AI research — a strong policy-relevance finding.

### H12 — SDG 13 (Climate) will be the most over-represented in research
> **SDG 13 will have the highest average alignment score across research papers, and a higher research proportion than policy proportion.**

- Climate ML is the dominant AI+sustainability research area
- Note: SDG 13 global performance score (82.5) is misleadingly high due to index methodology (developing countries score well by emitting little); the real-world urgency is not captured by this score
- **Caveat for paper:** Discuss SDG Index limitations when interpreting H12 + H21 together.

### H13 — SDG 10 (Reduced Inequalities) will be a high-priority policy topic but low-priority research topic
> **SDG 10 will score higher in policy alignment than research alignment.**

- SDG 10 global score is 56.0 — one of the lowest, indicating a real need
- UN policy documents frequently address inequality, especially post-COVID
- AI research rarely frames contributions as addressing inequality per se (even if it does)
- **Rationale:** Inequality is a cross-cutting policy concern that doesn't map neatly to specific AI methods.

### H14 — SDG 1 and SDG 10 will be nearly indistinguishable in embedding space but diverge in their research/policy profiles
> **Despite centroids SDG 1 ↔ SDG 10 similarity of 0.887, their research vs. policy proportions will differ — SDG 10 will be more policy-dominant.**

- Both SDGs address poverty/inequality but SDG 10 has a more explicitly political/redistributive framing
- This tests whether high centroid similarity translates to similar coverage patterns — expected answer: no.

### H15 — SDG 2 (Zero Hunger) will show moderate research presence due to precision agriculture AI
> **SDG 2 will not be completely absent from research — AI for crop yield, food security prediction, and agricultural monitoring will create measurable alignment.**

- Precision agriculture and AI-driven food systems are active research areas
- But the alignment will be weaker than SDG 13 or SDG 7
- **Rationale:** Tests whether AI research engages food security through a technical lens even if not framed in SDG language.

---

## H16–H20: Structural Hypotheses

### H16 — Research papers will have narrower SDG profiles than policy chunks
> **Each research paper will score highly on fewer SDGs (more concentrated alignment) than each policy chunk.**

- Abstracts are focused; they address one or two problems
- Policy document chunks span multiple SDGs within a single passage
- **Measured by:** Average number of SDGs scoring above threshold per document type; entropy of SDG score distribution per text

### H17 — Policy chunks from PARIS21 and UN AI Strategy will have different SDG profiles
> **PARIS21 chunks will align more strongly with SDGs 9, 17, and data-related SDGs; UN AI Strategy chunks will spread more evenly across governance SDGs.**

- PARIS21 is about AI for official statistics — naturally SDG 9 and 17
- UN AI Strategy is broader — covers ethics, national strategies, governance
- **Measured by:** Mean SDG scores separately for each source document

### H18 — Papers published after 2021 will show stronger SDG 13 alignment than earlier papers
> **More recent papers (2022–2025) will be more likely to explicitly engage with climate AI, reflecting the growth of the climate ML field.**

- NeurIPS Climate Change AI workshop began 2019; field accelerated 2021+
- **Measured by:** Mean SDG 13 alignment score split by publication year

### H19 — Highly cited papers will be more narrowly aligned than lowly cited papers
> **Papers with higher citation counts will show more concentrated SDG profiles (higher max SDG score, fewer SDGs above threshold).**

- Highly cited papers tend to make focused, methodological contributions
- Lower-cited papers may be broader, exploratory, or cross-cutting
- **Rationale:** This is speculative but testable with the `cited_by_count` field.

### H20 — The research corpus will be dominated by the "environmental + innovation" cluster
> **The combined attention to SDGs 7, 9, 12, 13, 14, 15 in research will exceed 50% of total alignment mass.**

- These SDGs form a coherent "technical sustainability" cluster in embedding space
- They represent the natural intersection of AI methods and sustainability framing in academic literature

---

## H21–H24: Contextual Hypotheses (Kaggle SDG Index)

### H21 — Research-policy alignment gaps will not correlate with global SDG performance scores
> **The SDGs with the largest research-policy gaps will not systematically be the worst-performing SDGs globally.**

- This is a null hypothesis to test against H22
- If confirmed: the research agenda is doubly misaligned — not just diverging from policy, but also ignoring the most off-track SDGs
- Global scores: worst are SDG 14 (50.5), SDG 9 (51.8), SDG 10 (56.0), SDG 2 (59.8)

### H22 (Alternative to H21) — The research-policy gap is largest for low-performing SDGs
> **SDGs with the lowest global performance scores will show the largest divergence between research and policy emphasis.**

- If confirmed: this is a more alarming finding — the world's biggest challenges are where research and policy are most misaligned
- **Expected pattern:** SDG 14 and SDG 10 are both low-performing and expected to be under-researched

### H23 — SDG 13 is an outlier: high research attention despite high global performance score
> **SDG 13 will be over-represented in research relative to its global performance score (82.5), making it an outlier in a research-attention vs. performance plot.**

- SDG 13 performance score is inflated (developing countries emit little CO2 → score well)
- Research attention to SDG 13 is high regardless
- **Implication:** If plotted as (global performance vs. research attention), SDG 13 appears as "over-researched relative to reported need" — but this is a methodological artefact worth discussing

### H24 — SDGs 9 and 14 will represent the most critical gap: low global performance + low research attention
> **SDG 9 and SDG 14 will appear in the bottom-left quadrant of a (global performance × research attention) scatter plot.**

- SDG 9: global score 51.8 (low) AND likely under-represented in AI research despite being the "innovation" SDG
- SDG 14: global score 50.5 (lowest) AND likely the most neglected SDG in AI research
- **This is the dissertation's most policy-relevant potential finding.**

---

## Summary: Expected Findings Matrix

| SDG | Research attention | Policy attention | Global performance | Key hypothesis |
|-----|-------------------|-----------------|-------------------|----------------|
| 1 | Low | High | Medium (68.6) | H2 |
| 2 | Low-Medium | High | Low (59.8) | H2, H15 |
| 3 | Medium | High | Medium (69.5) | H2, H7 |
| 4 | Low | Medium | High (76.4) | H8 |
| 5 | Low | High | Medium (63.1) | H2 |
| 6 | Low-Medium | Medium | Medium (66.2) | H5 |
| 7 | **High** | Medium | Medium (61.2) | H1, H5 |
| 8 | Low-Medium | Medium | High (72.0) | — |
| 9 | **High** | Low-Medium | **Low (51.8)** | H1, H24 |
| 10 | Low | **High** | Low (56.0) | H2, H13, H14 |
| 11 | Low-Medium | Medium | High (71.9) | — |
| 12 | Medium | Medium | High (80.2) | H5 |
| 13 | **Very high** | Medium | High* (82.5) | H1, H6, H12, H23 |
| 14 | **Very low** | Low-Medium | **Lowest (50.5)** | H11, H24 |
| 15 | Medium | Medium | Medium (66.1) | H5 |
| 16 | **Very low** | **High** | Medium (61.2) | H3 |
| 17 | Very low | **Very high** | Medium (60.8) | H4 |

*SDG 13 global score is methodologically inflated — see H23.

---

## Notes on Testing

- All hypotheses tested at the level of the full corpus (94 papers, 253 policy chunks)
- With 94 papers, many findings will be descriptive rather than statistically significant — acknowledge this
- Hypotheses H6–H10 (semantic gap) require qualitative inspection of representative texts to confirm interpretation
- H18 and H19 require splitting by paper metadata (year, citation count) — sample sizes will be small per cell

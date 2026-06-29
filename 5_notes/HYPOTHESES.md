# Research Hypotheses

Last updated: 2026-04-10

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

## H25–H27: New Hypotheses from Literature Review (added 2026-04-05)

### H25 — Coverage gap and semantic gap are negatively correlated across SDGs
> **SDGs with the highest research attention will show the largest within-SDG semantic gaps — not the smallest.**

- This is the single most important structural hypothesis. Two scenarios are possible:
  - *Positive correlation*: SDGs that research ignores in coverage also have the most divergent language when research does engage them → simple neglect story
  - *Negative correlation*: SDGs that research engages with *most* (SDG 13, SDG 7, SDG 3) are precisely where research and policy talk past each other, because researchers go deep into technical sub-problems while policy maintains a broad systemic framing → "talking past each other" story
- **The negative correlation story is more alarming and more novel.** It means the problem isn't just absence — it's active divergence at the point of engagement
- **Measured by:** Pearson/Spearman correlation between (a) SDG coverage proportion in research corpus and (b) intra-SDG cosine similarity between research and policy clusters
- **Prediction:** Negative correlation, particularly driven by SDG 13 (high research, divergent language) and SDG 10 (low research, possibly convergent language when it appears)

### H26 — Research-policy alignment is asymmetric: research ignores policy more than policy ignores research
> **Research papers will score lower against policy-derived SDG profiles than policy chunks score against research-derived SDG profiles.**

- Bidirectional alignment test: measure not just "does research align with policy" but "does policy align with research"
- If research scores low against policy centroids but policy scores high against research centroids → research is ignoring policy priorities, but policy is aware of / using research vocabulary
- If both score low → mutual incomprehension
- If research scores higher → policy is setting a narrower agenda than research is addressing
- **Measured by:** Building centroids from research corpus (paper embeddings averaged per inferred SDG) and scoring policy chunks against them; comparing direction of asymmetry
- **Rationale:** Policy documents (UN AI Strategy, PARIS21) explicitly cite and reference AI research. They may use research language. But research papers rarely cite UN policy documents in the same way.

### H27 — The productive/problematic misalignment divide follows technical vs social SDG lines
> **Misalignment on technical SDGs (7, 9, 13) will tend to be "productive" (research going deeper than policy); misalignment on social SDGs (1, 5, 10, 16) will tend to be "problematic" (research ignoring policy-urgent dimensions).**

- *Productive misalignment*: Research engages an SDG at greater technical depth than policy; it addresses dimensions policy hasn't yet articulated
- *Problematic misalignment*: Research is absent from or superficial on dimensions of an SDG that policy considers urgent
- **Operationalisation:** For each misaligned SDG, qualitatively inspect the highest-scoring research chunks and policy chunks; classify the divergence type
- **This is the core interpretive contribution of the Discussion chapter** — not just measuring gaps but explaining what kind of gap each one is

---

## H28–H30: Second Wave Hypotheses from Literature Review (added 2026-04-05)

### H28 — "AI in sustainability" policy chunks will score lowest against research embeddings
> **Policy chunks discussing AI's own governance, ethics, energy use, or societal risks will be the most semantically distant from research paper embeddings — more distant than even the most neglected SDG topic chunks.**

- UN policy documents discuss AI *in* sustainability (AI as actor with its own impacts) as well as AI *for* sustainability (AI as tool)
- Our research corpus almost entirely captures AI *for* sustainability — papers applying AI to solve problems
- This is not just a topical gap but a *categorical framing gap*: they are not just discussing different problems, they are having different conversations about what AI even is in a sustainability context
- **Measured by:** After scoring, rank policy chunks by their similarity to research embeddings; qualitatively examine the bottom decile — do they cluster around governance/ethics/risk?
- **Implication if confirmed:** The framing gap (AI as tool vs AI as actor) is a distinct, additional dimension of misalignment beyond SDG coverage and within-SDG semantic gaps. Deserves its own subsection in Results

### H29 — Topical overlap on SDG 9 (Innovation) reflects a shared blind spot, not responsive research
> **SDG 9 will show high topical overlap between research and policy, but this reflects both corpora being embedded in a techno-innovation paradigm rather than research genuinely responding to policy's SDG 9 priorities.**

- This is the "shared blind spot" hypothesis operationalised for the most likely candidate SDG
- Both AI research and UN policy naturally gravitate to innovation/infrastructure framing because both are produced in institutional contexts that valorise technological progress
- **Cannot be proven with our data** — but can be argued through: (a) examining *what specific aspects* of SDG 9 each corpus discusses (if research focuses on AI performance while policy focuses on equitable access to technology, this is operational divergence despite surface topical alignment); (b) connecting to the North-South critique (SDG 9 is a Global North priority)
- **Measured by:** Semantic gap analysis within SDG 9; qualitative inspection of top-scoring research and policy chunks

### H30 — Papers from 2022–2025 will show higher overall topical overlap with policy than 2018–2021 papers
> **More recent papers will be more aligned with current policy priorities, indicating that the field is responding — slowly — to the SDG framework adopted in 2015.**

- The SDG framework was adopted in 2015; research community uptake is gradual
- The climate ML boom (NeurIPS Climate Change AI workshop from 2019, growth 2021+) may have increased SDG 13 alignment post-2021
- **Measured by:** Mean SDG alignment scores split by two periods: 2018–2021 vs 2022–2025
- **Alternative:** Alignment is stable across time → research agenda is driven by supply (researcher interests and funding) rather than demand (policy), and the SDG framework has not successfully reoriented research

---

## H31–H34: Third Wave Hypotheses — Extended Data Sources (added 2026-04-05)

### H31 — NLP4SG papers show stronger policy topical overlap than the broader AI-for-sustainability corpus
> **Papers that explicitly target SDGs (NLP4SG, n≈5,000 ACL Anthology papers) will score higher on average against policy-aligned SDG centroids than the broader OpenAlex AI-for-sustainability corpus.**

- Rationale: Intentional SDG framing forces researchers to use more policy-adjacent vocabulary; incidental SDG engagement in the broader OpenAlex corpus may never explicitly invoke SDG language at all
- **Measured by:** Mean max-SDG alignment score for NLP4SG embeddings vs. OpenAlex paper embeddings; also compare SDG distribution profiles
- **Implication if confirmed:** Awareness of the SDG framework in research is itself a predictor of research-policy alignment — a finding with direct policy implications (e.g. SDG-aware publication incentives, conference SDG tracks)
- **Implication if not confirmed:** The SDG label is cosmetic; explicitly SDG-framed papers are not substantively different from the broader field → more damning

### H32 — VNR policy texts (SDGi corpus) over-index on development SDGs; AI policy docs over-index on governance SDGs
> **SDGi Voluntary National Review texts will show higher proportional emphasis on SDGs 1, 2, 3, and 10 relative to the 13 institutional AI policy documents, which will over-index on SDGs 9, 16, and 17.**

- Rationale: VNRs are countries reporting on progress toward human development goals; AI policy docs are institutions framing AI's governance role — different institutional speech acts with structurally different SDG vocabularies
- **Measured by:** Per-corpus SDG proportion profiles compared between a SDGi sample and the existing 13-doc policy corpus; visualise as side-by-side bar chart
- **Implication:** The choice of policy corpus is not neutral — it materially shapes which SDGs appear "prioritised by policy." This should be reported as a substantive finding, not just a limitation. If confirmed, it vindicates treating the two corpora separately (see A23)

### H33 — Within well-researched SDGs, AURORA target-level analysis reveals systematic neglect of equity-related targets
> **Within SDGs 7 and 13, AI research will cluster on performance/technical targets (e.g. SDG 7.2 renewable energy share, 13.1 resilience modelling) while equity-related targets (7.1 universal access, 13.3 climate education and awareness) will be systematically underrepresented.**

- Rationale: AI methods are more naturally applied to measurable, optimisable targets than to access and equity targets that require social/institutional intervention rather than algorithmic solutions
- **Measured by:** AURORA target-level alignment scores per target within SDGs 7 and 13; classify each of the 169 targets as technical/optimisation-oriented vs equity/access-oriented; compare average representation
- **Implication:** The coverage gap operates not just between SDGs but *within* SDGs — a subtler, more novel finding than aggregate SDG-level analysis. Only available via AURORA's 169-target labels. This directly challenges any "SDG 13 is well-researched" claim.

### H34 — Government VNRs from lower-income countries show larger semantic gaps from AI research than high-income country VNRs
> **SDGi texts from countries in the bottom tercile of GDP per capita will be semantically further from the AI-for-sustainability research corpus than VNRs from high-income countries.**

- Rationale: AI research is predominantly produced in high-income countries; the problems, framings, priorities, and vocabulary of their VNRs will naturally share more with AI research than those from lower-income countries facing different developmental challenges
- **Measured by:** Mean cosine similarity of SDGi texts to paper embeddings, grouped by country World Bank income classification (requires joining SDGi country metadata with WB income data — available)
- **Implication if confirmed:** Research-policy semantic gaps are not uniformly distributed globally — they are structurally larger for the countries that most need AI-assisted development progress. This connects the dissertation to the North-South critique of AI governance.

---

## H35–H36: Hypotheses from Centroid Validation (added 2026-04-10)

These hypotheses emerged directly from the centroid nearest-neighbour structure revealed by `validate_centroids.py` (centroid_similarity_matrix.csv). They are pre-registered before running alignment_score.py.

### H35 — AI research classified as SDG 17 will predominantly address climate/tech partnerships, not global governance
> **Papers assigned high SDG-17 scores will cluster around technology transfer and climate action mechanisms, not around development finance, multilateralism, or global institutional frameworks.**

- Empirical basis: SDG-17 centroid sits unexpectedly close to SDG-13 (cosine sim = 0.860) and SDG-9 (0.813) in embedding space — far closer than to SDGs about governance/finance (e.g. SDG 16: 0.414)
- This means "partnerships" language in the training corpus is overwhelmingly framed through climate and technology lenses. Research papers scoring high on SDG 17 will have been selected for that framing, not for multilateralism or financing content
- **Implication:** Apparent SDG-17 research coverage will be inflated by climate/tech-partnership papers that belong semantically to SDG 13 or SDG 9. The genuine SDG-17 dimension — global governance, development finance, data for development — will be under-counted
- **Measured by:** After alignment scoring, qualitatively inspect the top-20 papers by SDG-17 score; classify as climate/tech-partnerships vs governance/financing; report proportion
- **Connects to:** H4 (SDG 17 low in research), H28 (AI governance framing gap)

### H36 — SDG 11 (Sustainable Cities) research coverage will be systematically understated due to classifier leakage into SDG 9
> **Smart city and urban AI papers will score higher on SDG-9 (Innovation) than SDG-11 (Sustainable Cities), causing SDG-11 to appear under-researched as a measurement artefact.**

- Empirical basis: SDG-11 centroid's nearest neighbours are SDG-9 (0.716) and SDG-17 (0.691); SDG-11 achieved the lowest per-SDG F1 in validation (0.519), and its primary confusion is with SDG-9
- Urban AI / smart city research uses language of innovation, infrastructure, and technology — vocabulary that is closer to SDG-9's centroid than SDG-11's
- **Implication:** Any finding that SDG 11 shows a large research gap should be reported as a *lower bound* on coverage; the gap may be partly an artefact of classifier leakage. SDG-9 coverage, conversely, may be *inflated* by urban AI papers
- **Measured by:** After alignment scoring, examine papers with top SDG-9 scores that also score highly on SDG-11; report the overlap; qualitatively check whether these are urban AI papers
- **Connects to:** H1 (SDG 9 over-indexed in research), H20 (environmental+innovation cluster dominance)

---

## Notes on Testing

- All hypotheses tested at the level of the full corpus (6,172 papers, 47,005 policy chunks)
- Hypotheses H6–H10 (semantic gap) require qualitative inspection of representative texts to confirm interpretation
- H18 and H19 require splitting by paper metadata (year, citation count) — sample sizes will be small per cell
- H25 is the critical structural hypothesis — a negative correlation finding would be the headline result
- H26 requires building research-side centroids, not just OSDG-derived centroids — add to implementation TODOs
- H27 is qualitative; do not over-quantify it
- H35 and H36 are measurement-layer hypotheses — test them *during* alignment scoring as robustness checks, before reporting coverage gaps
- **Macro-cluster robustness check:** SDGs 1, 8, and 10 have near-collinear centroids (sim 0.780–0.887). Report their coverage as a group alongside individual scores as a sensitivity check. Individual SDG-level findings within this cluster are potentially unreliable due to classifier leakage between them.

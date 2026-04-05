# Literature Review Insights

Source: SciSpace comprehensive mapping, 2026-04-05
File: `literature/comprehensive-mapping-scispace-2026-04-05/literature_review_semantic_alignment_ai_sdg.md`
~1,684 papers synthesised across 9 thematic areas.

---

## What Is Already Known (we will replicate, not discover)

- **SDG distribution pattern is well-established:** AI research consistently concentrates on SDGs 3 (health), 7 (energy), 9 (industry), 11 (cities), 13 (climate). SDGs 5 (gender), 10 (inequalities), 16 (peace/justice), 17 (partnerships) are consistently neglected. This finding holds across bibliometrics, topic modeling, expert surveys, and project databases (Vinuesa 2020, Singh 2023, Cowls 2021, Nedungadi 2024).
- **Geographic bias is documented:** AI research is biased toward SDGs relevant to high-income countries where most AI researchers are based (Vinuesa 2020, Ferreira 2025, Chavarro 2022).
- **Research-policy gaps are pervasive:** Research concentrates on pre-deployment, technically tractable problems; policy emphasises deployment-stage societal impacts (Strauss 2025, Sioumalas-Christodoulou 2025, Toney 2024).
- **SBERT is validated for semantic similarity across document types:** Including regulatory documents (Bergman 2023), legal precedents (Justino 2025), and cross-domain applications.

---

## What Has NOT Been Done (our genuine contribution)

> Direct quote from the review: *"no existing study has systematically compared the semantic content of AI research papers to the semantic content of policy documents at the SDG level."*

- No study has measured **within-SDG semantic gaps** — whether research and policy discuss the *same aspects* of an SDG or different ones
- No study has done **bidirectional alignment** — measuring both whether research aligns with policy AND whether policy aligns with research
- No study has examined the **relationship between coverage gap and semantic gap** — whether they are correlated, uncorrelated, or inversely related across SDGs
- The **productive vs problematic misalignment** distinction has not been operationalised at SDG level

---

## Key Papers to Read and Cite

### Foundational (must cite)
- **Vinuesa et al. (2020)** Nature Communications — AI enabling/inhibiting 169 SDG targets; expert consensus; most cited paper in the space. Note limitations: expert perception, not empirical bibliometrics; treats AI as monolithic
- **Cowls et al. (2021)** Nature Machine Intelligence — 108 AI for Social Good projects; SDG 3 dominant; SDGs 5, 16, 17 under-addressed. Important because it covers deployed projects, not just papers
- **Singh et al. (2023)** Sustainable Development — 20 years of AI/ML/DL papers; SDGs 3 and 7 most common; bibliometric
- **Armitage et al. (2020)** Quantitative Science Studies — "Do independent bibliometric approaches get the same results?" Critical methodological challenge: method choice affects SDG distributions. Must address this
- **Toney et al. (2024)** FAccT — "Trust Issues": apparent high-level agreement between research and policy masks operational divergence. Directly validates our within-SDG semantic gap analysis

### Methodology (must cite for our methods)
- **Reimers & Gurevych (2019)** EMNLP — Sentence-BERT paper; cite for our embedding method
- **Gjorgjevikj et al. (2025)** — domain-specific benchmarking of sentence encoders for SDG association; fine-tuning helps; validates our centroid approach
- **Bergman et al. (2023)** — BERT-based comparison of regulatory documents; methodological precedent for cross-corpus semantic comparison
- **Hajikhani et al. (2022)** Scientometrics — ML mapping of publications to SDGs; notes SDGs 8, 14, 15 difficult to classify from text; relevant to our per-SDG reliability caveats

### Research-policy gap (for lit review framing)
- **Strauss et al. (2025)** — corporate AI research focuses on pre-deployment; policy focuses on deployment. The temporal/stage gap
- **Sioumalas-Christodoulou et al. (2025)** AI & Society — AI metrics focus on technical performance; policy emphasises ethical/social SDG-aligned concerns; directly relevant
- **Mejía (2025)** ISSI — research-policy alignment and EU AI Act; bibliometric; direct precedent

### Theoretical frameworks (for Methodology and Discussion)
- **Haas (1992)** — epistemic communities; our semantic alignment operationalises "shared causal beliefs" dimension
- **Kingdon (MSF)** — multiple streams; semantic alignment is necessary but not sufficient for policy window coupling
- **Gibbons et al.** — Mode 1 vs Mode 2 knowledge production; supply-driven research naturally misaligns with policy

### SDG datasets and tools (for Methodology)
- **Pukelis et al. (2022)** — OSDG 2.0 paper; how OSDG works; cite when using OSDG
- **Ingram et al. (2025)** — OSDG relies on binary crowd validation; "high-quality graded relevance sets for SDG classification do not exist"; cite when discussing OSDG limitations
- **Adauto et al. (2023)** — NLP4SGPAPERS: 5,000 papers annotated for SDG; inter-annotator agreement 88.67% kappa; SDGs 1 (poverty) and 2 (hunger) largely unaddressed in research papers — confirms H2 and H11

### Tensions and critiques (for Discussion)
- **Heilinger et al. (2023)** — "Beware of sustainable AI!"; greenwashing risk; alignment with SDG rhetoric can be performative
- **Sætra (2021)** — AI may be fundamentally unsustainable; need to examine the sociotechnical system, not just AI's SDG applications
- **Rehak et al. (2025)** — "On the (im)possibility of sustainable AI"; provocative; useful for Discussion's critical section

---

## Key Methodological Challenges Raised (must address)

### 1. Domain mismatch (most critical)
Research papers and policy documents have fundamentally different linguistic properties:
- Policy: modal verbs (should/must/will), performative, goal-articulating, high abstraction
- Research: hedged indicative claims, methods-reporting, finding-announcing, technical vocabulary
- Consequence: cosine similarity may detect vocabulary overlap without substantive alignment
- Mitigation: reframe all claims as "topical overlap"; validate with human judgments on a sample; cite Hassan (2022) on modality in policy text

### 2. OSDG circularity (medium risk)
- OSDG was labeled using a tool trained on UN-related documents
- Our policy corpus is UN documents
- Risk: policy chunks score higher against OSDG centroids by design, not by genuine alignment
- Mitigation: post-hoc check — compare mean research vs policy scores; if systematically different, flag

### 3. Method choice dependence (Armitage et al. 2020)
- Different bibliometric methods produce different SDG distributions
- If our embedding approach produces radically different results from Singh et al. / Nedungadi et al., must explain why
- Mitigation: treat convergence as validation; divergence as finding requiring explanation

### 4. Granularity mismatch
- We treat research papers at document level (1 abstract = 1 vector)
- We treat policy documents at chunk level (150-word chunks)
- Abstracts are already short and focused; policy chunks may cover multiple topics
- This asymmetry may inflate semantic precision for papers vs policy
- Flag in methodology; potentially run paper-level chunking as robustness check

---

## Theoretical Frameworks to Use

| Framework | Author | How it applies |
|-----------|--------|----------------|
| Epistemic communities | Haas (1992) | Shared causal beliefs manifest as semantic similarity; our method operationalises this |
| Multiple Streams | Kingdon | Semantic alignment is necessary for research-policy coupling but political timing also needed |
| Mode 1/2 knowledge | Gibbons et al. | Supply-driven Mode 1 research → misalignment is structural, not accidental |
| Knowledge utilisation | Rose et al. (2017) | Semantic alignment measures the "relevance" dimension; credibility and legitimacy also needed |
| Productive misalignment | (synthesised here) | Not all misalignment is bad; some reflects critical distance or frontier exploration |

---

## What This Means for the Dissertation's Contribution Claims

**Claim 1 (safe):** First study to systematically measure within-SDG semantic gaps between AI research and policy across all 17 SDGs

**Claim 2 (safe):** First bidirectional alignment analysis (research→policy AND policy→research)

**Claim 3 (safe):** First examination of whether coverage gaps and semantic gaps are correlated (H25)

**Claim 4 (conditional):** Operationalisation of productive vs problematic misalignment distinction — conditional on finding clear enough patterns for interpretation

**Claim 5 (do NOT make):** That we have measured actual research-policy alignment — we have measured topical overlap

---

## Second-Wave Insights (deeper reading, added 2026-04-05)

### Insight A: The three-level alignment structure — our precise positioning

Nobody in the literature names this but the review implies it clearly:

| Level | What it measures | Who does it |
|-------|-----------------|-------------|
| **Level 1 — Lexical** | Same terms/keywords for an SDG | Armitage (2020), keyword mapping studies, most bibliometrics |
| **Level 2 — Topical** | Same subject areas (semantic/embedding) | **Us** |
| **Level 3 — Operational** | Same approach to the problem | Toney et al. (2024) find this diverges even when Level 1 agrees |

Our contribution is advancing measurement from Level 1 to Level 2. Our within-SDG semantic gap analysis (H6–H10) approximates Level 3 without claiming to fully reach it.

**Use this in:**
- Introduction: "We advance beyond keyword-based mapping to semantic topical analysis"
- Methodology: "We measure topical alignment (Level 2); prior work measures lexical alignment (Level 1); operational alignment (Level 3) remains a frontier"
- Discussion: "Even where we find topical alignment, Toney et al. (2024) warn that operational divergence can persist"

### Insight B: "AI for" vs "AI in" sustainability — a categorical framing gap

Two distinct conversations are happening under the same umbrella:
- **AI for sustainability:** Research using AI as a tool to address SDG-labelled problems (dominates our research corpus)
- **AI in sustainability:** Examining AI's own sustainability impacts — energy consumption, governance, ethics, e-waste, power concentration (dominates parts of the policy corpus)

These are not just different topics within the same conversation. They are different framings of what AI *is* in relation to sustainability. A policy document saying "AI systems must be governed to avoid exacerbating inequalities" and a research paper on "AI for poverty prediction" are both nominally about SDG 10 but are categorically different in how they position AI.

**Testable:** After scoring, identify policy chunks that score lowest against all research embeddings. If these cluster around governance/ethics/risk language → confirmed framing gap. This would be a standalone finding worth ~1 paragraph in Results.

**Cite:** Sætra (2021); Heilinger et al. (2023); Ghamisi et al. (2024)

### Insight C: The "shared blind spot" — high alignment ≠ responsive research

High topical overlap between research and policy can mean two opposite things:
1. **Responsive:** Research is addressing what policymakers actually need → desirable
2. **Shared blind spot:** Both are embedded in the same Global North, techno-solutionist, growth-oriented paradigm → both over-index on the same SDGs for the same ideological reasons

**Cannot be distinguished with our data.** Must be named honestly in Discussion.

Most plausible shared blind spots: SDG 9 (Innovation — valorised by both academic and UN institutional cultures), SDG 13 (Climate — high-visibility, well-funded, attracting both research and policy attention post-Paris).

Most revealing negative test: if SDGs 10 (Inequalities) and 16 (Peace/Justice) are consistently misaligned, and these are precisely the SDGs most critical to structural transformation (Global South priorities, power redistribution), this suggests the shared frame breaks down at the structural edges — where the questions are too politically uncomfortable for either research or policy to centre.

**Cite:** Ferreira et al. (2025) on Global South underrepresentation; Wall et al. (2021) on AI4D; Section 9.3 of the review on SDG critiques

### Insight D: Corpus concentration asymmetry — a methodological correction needed

**The asymmetry:**
- Research corpus: 94 papers × 94 independent authorships → high diversity
- Policy corpus: 253 chunks × 2 documents × 2 authorships → high autocorrelation

The SDG profile of the policy corpus reflects editorial decisions by 2 documents, not 253 independent observations. A single editorial choice ("this report will emphasise SDG 9 heavily") produces dozens of SDG-9-heavy chunks. This could amplify apparent policy emphasis on certain SDGs far beyond their true importance in the broader policy landscape.

**Mitigation steps:**
1. When reporting coverage gap, show per-document SDG profiles (PARIS21 vs UN AI Strategy separately) alongside the combined figure
2. Consider weighting policy chunks by document (each document = 1 observation for coverage purposes, not each chunk)
3. Frame findings as "in these two UN documents" rather than "in policy discourse generally" — which we should already be doing per A2

**Cite:** No specific citation needed; this is an observation about corpus construction. But Wang et al. (2023) on heterogeneity across national AI strategies implicitly supports the need for caution

---

## Open Questions Raised by the Literature

1. **Does AI research address AI's own sustainability costs (SDG 12/13 via energy use, e-waste)?** Our corpus likely focuses on AI *for* sustainability, not AI's *own* sustainability impacts. Is this gap in our corpus or in the field?

2. **What explains the SDG 16 anomaly?** SDG 16 has low centroid coherence (0.477), high inter-centroid isolation, expected high policy presence but very low research presence. Is this because SDG 16 (peace/justice) is genuinely not tractable for AI, or because it's not funded, or because researchers don't frame governance work in SDG 16 terms even when they do it?

3. **Does alignment improve over time?** We have year metadata on papers (2018-2025). Post-2015 SDG adoption — has research become more aligned? Post-2020 climate ML boom — has SDG 13 alignment changed?

4. **Is the UN AI Strategy/PARIS21 a fair proxy for "policy"?** These are multilateral UN documents. National AI strategies may tell a different story. The review notes substantial heterogeneity across national policies (Wang 2023). We are measuring alignment with *UN-level multilateral* policy discourse, which is a specific thing.

# Research Assumptions

Last updated: 2026-04-05

Assumptions made (explicitly or implicitly) at each layer of the research. Each entry notes the assumption, its risk level, and what to say about it in the paper.

---

## Data Assumptions

### A1 — 6,172 papers are a representative sample of AI-for-sustainability research
- **Assumption:** Four OpenAlex queries ("artificial intelligence sustainable development", "machine learning sustainable development goals", "deep learning sustainability", "artificial intelligence SDG"; 2018–2025; 2,000 per query cap; deduplicated to 6,172 with abstracts) return papers broadly representative of the field
- **Risk:** Low-medium. 6,172 is a substantial corpus. Query terms may still miss niche adjacent work (e.g. papers framed around specific SDGs without using "sustainable development" as a term); English-language bias applies
- **Mitigation:** Corpus concentration is no longer a primary weakness; frame any residual gaps as limitations for future multi-lingual or citation-network-based expansion
- **Where to address:** Methodology → Data Sources; Discussion → Limitations

### A2 — Thirteen policy documents represent "global AI and SDG policy discourse"
- **Assumption:** 13 documents from diverse institutions (2 original UN docs, UN SDG Progress Report 2023, IPCC AR6 SPM, UK/Singapore/Germany national AI strategies, African Union Continental AI Strategy, UN AI Advisory Body 2024, UNESCO Ethics of AI 2021, EU HLEG Ethics Guidelines, OECD AI Recommendation 2019, US AI Bill of Rights) are sufficient proxies for international policy discourse on AI and sustainability
- **Risk:** Medium. National strategies from Global South (India failed download; no Latin America, Southeast Asia outside Singapore) are underrepresented. All documents are English-language. The corpus mixes AI-governance documents with sustainability-focused documents — institutional type is itself a variable
- **Mitigation:** Report per-document SDG profiles; flag geographic and institutional gaps; describe corpus explicitly as "international institutional policy discourse" rather than "global policy"
- **Where to address:** Methodology → Data Sources; Discussion → Limitations

### A3 — OSDG labels at agreement ≥ 0.5 are reliable ground truth
- **Assumption:** Texts where ≥50% of annotators agreed on an SDG label are correctly labeled
- **Risk:** Low-medium. Agreement ≥ 0.5 is a relatively permissive threshold (keeps 71% of data). Some noise will remain
- **Mitigation:** Centroid validation against SDG benchmark (which is expert-verified) will empirically test this; report validation accuracy
- **Where to address:** Methodology → SDG Reference Embeddings

### A4 — English-language bias
- **Assumption:** All corpora are in English; this is treated as acceptable scope
- **Risk:** Medium. Research and policy from non-English-speaking countries is underrepresented. SDG gaps identified may partially reflect language bias rather than genuine research gaps
- **Where to address:** Discussion → Limitations

---

## Methodological Assumptions

### A5 — Sentence-BERT embeddings capture SDG-relevant semantics
- **Assumption:** The semantic space learned by Sentence-BERT (trained on NLI/STS tasks) generalises to SDG-related discourse well enough for cosine similarity to be meaningful
- **Risk:** Low-medium. Sentence-BERT is general-purpose; it was not fine-tuned on SDG or sustainability text. A domain-specific model might produce better-separated SDG clusters
- **Mitigation:** Validate empirically via centroid classification accuracy on benchmark; if accuracy is low (<60%), consider fine-tuning or switching to a domain-adapted model
- **Where to address:** Methodology → Embedding Model; Results → Centroid Validation

### A6 — Mean embedding (centroid) is a valid SDG representation
- **Assumption:** Averaging all OSDG embeddings for a given SDG yields a centroid that meaningfully represents "what that SDG is about" in semantic space
- **Risk:** Medium. If an SDG covers very heterogeneous topics (e.g. SDG 16: Peace, Justice, and Strong Institutions covers rule of law, corruption, violence, institutions — quite different themes), the centroid may be a poor representation of any of them
- **Mitigation:** Check intra-SDG embedding variance; SDGs with high variance may need sub-cluster treatment; flag in methodology
- **Where to address:** Methodology → SDG Reference Embeddings

### A7 — Cosine similarity is an appropriate alignment measure
- **Assumption:** Cosine similarity between a text embedding and an SDG centroid is a valid measure of how much that text "belongs to" or "addresses" that SDG
- **Risk:** Low. Cosine similarity is standard for semantic similarity tasks. The main risk is that high-dimensional spaces can produce spuriously high similarities; mitigated by comparing relative scores rather than absolute thresholds
- **Where to address:** Brief justification in Methodology

### A8 — Text content reflects actual research/policy priorities
- **Assumption:** What a paper's abstract discusses reflects what the research actually contributes; what a policy document says reflects actual policy priorities
- **Risk:** Medium. Abstracts may emphasise framing over content. Policy documents may be aspirational rather than operational. The analysis is of *discourse*, not *action*
- **Mitigation:** Be explicit that the study analyses discourse alignment, not implementation alignment — this is a feature, not a bug (discourse shapes agenda)
- **Where to address:** Introduction → Scope; Discussion → Interpretation of Findings

---

## Analytical Assumptions

### A9 — Coverage gap (proportion difference) is a meaningful proxy for attention allocation
- **Assumption:** If 30% of research papers score highly on SDG 13 but only 5% of policy chunks do, this reflects a genuine difference in emphasis, not an artefact of corpus size or document style
- **Risk:** Medium. Policy documents tend to be broader and cover more SDGs per chunk; research abstracts tend to be narrower and more focused. This could inflate apparent research concentration on specific SDGs
- **Mitigation:** Normalise scores; report per-document SDG profiles, not raw counts; use relative ranking rather than absolute proportions where appropriate
- **Where to address:** Methodology → Alignment Scoring

### A10 — Intra-SDG cosine similarity reflects thematic divergence
- **Assumption:** Low cosine similarity between the research cluster and policy cluster within an SDG means they are discussing genuinely different aspects of that SDG
- **Risk:** Low-medium. It could also reflect stylistic differences (academic writing vs. policy writing) rather than substantive thematic differences
- **Mitigation:** Qualitatively inspect representative chunks from each low-similarity cluster to confirm the divergence is substantive; use topic modeling to surface themes
- **Where to address:** Methodology → Semantic Gap Analysis; Results → Semantic Gap

### A11 — Kaggle SDG Index scores are a valid proxy for "real-world need"
- **Assumption:** Countries with low SDG Index scores for a given goal indicate that SDG is a greater priority / area of need, and that research-policy gaps in those SDGs are more consequential
- **Risk:** Medium. SDG Index scores aggregate diverse indicators and may not reflect research/policy relevance. A low score on SDG 14 (Life Below Water) may reflect geography rather than neglect
- **Mitigation:** Use SDG Index as contextual framing, not as a causal claim; present as correlation, not causation
- **Where to address:** Methodology → Contextual Analysis; Discussion → Interpretation

---

## Scope Assumptions (Implicit)

### A12 — 2018–2025 is the relevant time window
- Research and policy on AI + sustainability accelerated post-2018 (partly driven by the 2015 SDG adoption and 2017 AI strategy wave); this window is defensible but excludes earlier foundational work

### A13 — Paper-level analysis is sufficient (not citation network or author network)
- No graph-based analysis; alignment is purely semantic/text-based

### A15 — OSDG-derived centroids are neutral SDG reference points
- **Assumption:** Building SDG centroids from OSDG (which was labeled using a tool trained on UN-related documents) produces reference vectors that are independent of our policy corpus
- **Risk:** Medium-high. OSDG's labeling tool was trained primarily on UN-related sources. Our policy corpus *is* UN documents. The centroids may be implicitly calibrated toward UN-style language, inflating policy alignment scores relative to research alignment scores — not because policy is genuinely more aligned, but because the measurement instrument and the policy corpus share a common linguistic ancestor
- **Diagnostic test:** After scoring, check whether policy chunks score systematically higher than research papers against OSDG centroids on average. A large systematic gap would indicate calibration bias, not genuine alignment difference
- **Mitigation:** If bias detected, consider building separate centroids from (a) OSDG only and (b) SDG indicator metadata, and comparing results; flag in methodology chapter
- **Where to address:** Methodology → SDG Reference Embeddings; Results → Centroid Validation

### A16 — Semantic (cosine) similarity measures substantive alignment
- **Assumption:** High cosine similarity between a research paper and an SDG centroid indicates the paper substantively addresses that SDG's policy priorities
- **Risk:** High. This is the deepest assumption in the study. Research papers and policy documents are different *speech acts* with different communicative functions: policy articulates commitments and goals using performative, modal language ("must," "should," "will"); research reports findings using hedged, indicative language ("we show," "results suggest"). Two texts can share substantial vocabulary while performing entirely different functions. High cosine similarity may reflect *topical overlap* — they discuss the same subject area — without reflecting *substantive alignment* — they address the same problem in compatible ways
- **Implication:** We are measuring *topical overlap as a proxy for alignment*, not alignment directly. This must be stated explicitly in every claim derived from similarity scores. The contribution is still valid — topical overlap is the necessary (if not sufficient) condition for research-policy alignment — but the language must be precise
- **Where to address:** Introduction → Scope; Methodology → Alignment Scoring; Discussion → Limitations

### A17 — Misalignment between research and policy is normatively undesirable
- **Assumption:** Semantic gaps between research and policy indicate a problem to be addressed
- **Risk:** Medium. Sometimes research *should* be misaligned with policy — when it challenges flawed policy assumptions, explores domains policy has not yet reached, or critically examines the SDG framework itself (which has been critiqued as reflecting Global North priorities). Uncritical alignment with policy could mean research that validates existing blind spots rather than expanding the frontier
- **Implication:** Findings must be interpreted using a "productive vs problematic misalignment" lens:
  - *Problematic misalignment*: Research ignores policy-urgent needs due to tractability bias, funding concentration, or researcher demographics (e.g., AI researchers not engaging with SDG 10 inequality because it resists reduction to optimization problems)
  - *Productive misalignment*: Research addresses dimensions of a problem that policy has not yet framed, or challenges assumptions embedded in policy discourse (e.g., research questioning whether AI is net-positive for SDG 13 while policy assumes it is)
- **Where to address:** Discussion → Interpretation of Findings; Conclusion

### A19 — Unit-of-analysis asymmetry: papers vs chunks
- **Assumption:** Comparing 6,172 paper-level vectors (1 abstract = 1 vector) against 1,211 chunk-level vectors (~150 words per chunk, many chunks per document) is a valid basis for SDG profiling
- **Risk:** High. This is a fundamental unit-of-analysis mismatch with three distinct problems:
  1. **Unequal granularity:** A research paper is represented by its abstract — a dense, author-curated summary of the whole work. A policy chunk is a 150-word passage that may cover only one corner of a 300-page document. These are not the same kind of object semantically.
  2. **Coverage inflation:** A policy document that heavily discusses SDG 9 generates ~50 SDG-9-heavy chunks. This inflates the apparent policy emphasis on SDG 9 — not because SDG 9 is more important to policy, but because one editorial decision produces dozens of correlated data points.
  3. **Autocorrelation:** The 1,211 chunks are not 1,211 independent observations. They are ~13 clusters of highly correlated passages. Treating them as independent observations in any statistical test would be invalid. The research corpus (6,172 papers from thousands of independent authors) is genuinely diverse.
- **Why chunking is necessary anyway:** Policy documents have no abstract. A 300-page report covers many topics; treating it as one vector would lose all internal structure. Chunking is the standard NLP approach for long-document comparison.
- **Mitigations (must apply all three):**
  1. **Per-document SDG profiles:** Always report which document drives which SDG signal; never report aggregate policy scores without showing the document breakdown
  2. **Document-weighted comparisons:** For the coverage gap analysis, weight by document (each of 13 docs = 1 observation) not by chunk count (which would let large documents dominate)
  3. **Fixed-sample per document:** For semantic gap analysis (within-SDG cluster similarity), cap the number of chunks contributed per document to avoid single-document dominance
- **Where to address:** Methodology → Data Sources (explain chunking rationale); Methodology → Limitations (name the asymmetry explicitly); Results → Coverage Gap (show per-document breakdown alongside aggregate)

### A20 — High topical overlap indicates research is responding to policy (not a shared blind spot)
- **Assumption:** When research and policy show high topical overlap on an SDG, this means research is addressing what policymakers need
- **Risk:** High. Both AI research and UN SDG policy are produced within overlapping institutional contexts — predominantly Global North, growth-oriented, techno-solutionist. High alignment could indicate (a) research responsively addressing policy priorities, OR (b) both being embedded in the same paradigmatic frame, which causes them to both over-emphasise certain SDGs (e.g. SDG 9 Innovation) and under-emphasise others (e.g. SDG 10 Inequalities) for the same ideological reasons. There is no way to distinguish (a) from (b) with our data
- **Implication:** This is a fundamental interpretive ambiguity that must be named in the Discussion, not apologised for. It changes what our study can claim: we identify *what* is aligned and *what* isn't; we cannot determine *why*. The North-South asymmetry pattern (if social SDGs 1, 5, 10 are consistently misaligned) is where the shared blind spot hypothesis is most plausible
- **Where to address:** Discussion → Interpretation; Conclusion → Limitations

### A21 — Our research corpus captures AI *for* sustainability, not AI *in* sustainability
- **Assumption:** Papers retrieved via "artificial intelligence" + "sustainable development" query are studying AI as a tool applied to sustainability challenges (AI for sustainability)
- **Risk:** Medium. UN policy documents discuss *both* AI for sustainability AND AI's own governance, ethics, energy use, and societal risks (AI in sustainability). If policy chunks about AI governance score low against our research-paper-derived topics, this reflects a categorical framing difference — not just a topical gap
- **Diagnostic:** After scoring, examine which policy chunks score *lowest* against research embeddings. If these cluster around AI governance/ethics/risk topics, it confirms the framing gap
- **Where to address:** Methodology → Corpus Definition; Discussion → "AI for vs AI in" as an additional gap dimension

### A18 — Our corpus is representative of the AI-for-sustainability research field
- **Assumption:** 94 papers from a single OpenAlex query reproduce the SDG distribution patterns found in larger bibliometric studies
- **Risk:** Medium-high. Armitage et al. (2020) showed that independent bibliometric approaches can produce different SDG distributions, suggesting method choice matters as much as the underlying reality. Our specific query terms, corpus size, and time window may produce a non-representative sample. If our SDG distribution diverges sharply from Singh et al. (2023) or Nedungadi et al. (2024), this requires explanation
- **Mitigation:** After running alignment scoring, compare our SDG distribution for the research corpus against findings from established bibliometric studies. Convergence validates; divergence must be explained
- **Where to address:** Methodology → Data Sources; Discussion → Limitations

### A14 — SDG categories are treated as independent dimensions
- In reality SDGs are highly interdependent (the "SDG interlinkages" literature). Treating them as independent scoring dimensions is a simplification
- **Where to address:** Discussion → Limitations

---

## Summary Table

| ID | Assumption | Risk | Mitigated by |
|----|-----------|------|--------------|
| A1 | 94 papers is representative | Medium-high | Acknowledge as limitation |
| A2 | 2 policy docs represent policy | High | Narrow scope claim |
| A3 | OSDG agreement ≥ 0.5 is reliable | Low-medium | Centroid validation |
| A4 | English-language only | Medium | Acknowledge as limitation |
| A5 | SBERT captures SDG semantics | Low-medium | Validate accuracy on benchmark |
| A6 | Centroid = valid SDG representation | Medium | Check intra-SDG variance |
| A7 | Cosine similarity is appropriate | Low | Standard in literature |
| A8 | Text = priorities | Medium | Frame as discourse analysis |
| A9 | Coverage gap is meaningful | Medium | Normalise; use relative ranking |
| A10 | Low intra-SDG similarity = thematic gap | Low-medium | Qualitative inspection |
| A11 | SDG Index = real-world need | Medium | Frame as correlation only |
| A12 | 2018–2025 time window | Low | Defensible; note exclusions |
| A13 | Paper-level only | Low | Scope choice |
| A14 | SDGs are independent | Medium | Acknowledge SDG interlinkages |
| A15 | OSDG centroids are neutral reference points | Medium-high | Check policy/research score distributions |
| A16 | Cosine similarity = substantive alignment | High | Reframe as "topical overlap"; qualify all claims |
| A17 | Misalignment is always bad | Medium | Introduce productive vs problematic misalignment lens |
| A18 | Research corpus represents current AI-for-sustainability field | Medium-high | Cite Armitage et al. (2020) on method-dependence |
| A19 | Unit-of-analysis asymmetry: paper abstracts vs 150-word chunks | **High** | Per-doc profiles; document-weighted comparisons; fixed sample per doc in semantic gap |
| A20 | High alignment = research responding to policy (not shared blind spot) | High | Name ambiguity explicitly; cannot distinguish with our data |
| A21 | Our corpus captures AI *for* sustainability, not AI *in* sustainability | Medium | Check which policy chunks score lowest; flag framing gap |

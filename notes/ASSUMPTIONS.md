# Research Assumptions

Last updated: 2026-04-05

Assumptions made (explicitly or implicitly) at each layer of the research. Each entry notes the assumption, its risk level, and what to say about it in the paper.

---

## Data Assumptions

### A1 — 94 papers are a representative sample of AI-for-sustainability research
- **Assumption:** The OpenAlex query ("artificial intelligence" + "sustainable development", 2018–2025) returns papers that are broadly representative of the field
- **Risk:** Medium-high. 94 is a small corpus. Query terms may miss adjacent work (e.g. papers using "machine learning" + "SDGs", or work framed around specific SDGs without using "sustainable development" as a term)
- **Mitigation:** Acknowledge as a limitation; frame findings as indicative rather than definitive; suggest future work could expand to 1,000+ papers with broader query terms
- **Where to address:** Methodology → Data Sources; Discussion → Limitations

### A2 — Two policy documents represent "policy priorities"
- **Assumption:** The UN AI Strategy Resource Guide (2021) and PARIS21 report (2024) are sufficient proxies for global AI-for-sustainability policy discourse
- **Risk:** High. Two documents is a very thin policy corpus. These are also both UN-affiliated — they may not reflect national-level or private-sector policy priorities
- **Mitigation:** Be explicit about scope ("UN-level multilateral policy discourse"); flag that national AI strategies, OECD reports, etc. are out of scope; suggest extensions
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

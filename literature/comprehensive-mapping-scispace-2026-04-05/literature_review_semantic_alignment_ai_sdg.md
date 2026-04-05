# Semantic Alignment Between AI Sustainability Research and Policy Frameworks: A Comprehensive Literature Review

## Executive Summary

This literature review examines the intersection of artificial intelligence (AI) research, sustainability policy, and computational methods for measuring semantic alignment between academic and policy discourse. The core research question—to what extent does academic AI-for-sustainability research align with sustainability policy priorities as expressed through the UN Sustainable Development Goals (SDGs)—sits at the nexus of three scholarly domains: AI for sustainability, research-policy gap analysis, and natural language processing (NLP) methods for cross-corpus comparison.

The review synthesizes evidence from approximately 1,684 papers across nine thematic areas, revealing substantial methodological precedents for embedding-based semantic analysis, significant empirical findings about SDG representation imbalances in AI research, and persistent gaps between research priorities and policy needs. Key findings include: (1) AI research disproportionately addresses SDGs 3 (health), 7 (energy), 9 (industry), and 13 (climate), while neglecting SDGs 5 (gender equality), 10 (reduced inequalities), and 16 (peace and justice); (2) existing research-policy alignment studies identify systematic mismatches in AI governance, with research focusing on pre-deployment technical issues while policy emphasizes deployment-stage societal impacts; (3) Sentence-BERT and centroid-based embedding methods offer robust approaches for cross-corpus semantic comparison, though domain mismatch between academic and policy text presents methodological challenges; and (4) the OSDG Community Dataset provides a validated resource for SDG classification, though with known limitations in label quality and domain coverage.

This review validates the proposed dissertation methodology while identifying critical considerations: the need for domain adaptation when comparing academic and policy corpora, the importance of chunk-level versus document-level analysis for capturing semantic nuance, and the theoretical grounding provided by epistemic community theory and knowledge utilization frameworks. The findings suggest that semantic alignment measurement via embeddings offers a valid, scalable approach to quantifying research-policy gaps, while acknowledging that textual similarity serves as a proxy for, rather than direct measure of, actual research impact on policy.

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Area 1: Mapping AI Research to SDGs](#2-area-1-mapping-ai-research-to-sdgs)
3. [Area 2: Research-Policy Alignment and Gaps in AI and Technology](#3-area-2-research-policy-alignment-and-gaps-in-ai-and-technology)
4. [Area 3: Semantic Similarity and Embedding Methods for Cross-Corpus Comparison](#4-area-3-semantic-similarity-and-embedding-methods-for-cross-corpus-comparison)
5. [Area 4: NLP and AI Methods for Policy Document Analysis](#5-area-4-nlp-and-ai-methods-for-policy-document-analysis)
6. [Area 5: SDG Classification Datasets and Tools](#6-area-5-sdg-classification-datasets-and-tools)
7. [Area 6: Theoretical Frameworks for Research-Policy Alignment](#7-area-6-theoretical-frameworks-for-research-policy-alignment)
8. [Area 7: Known Findings About Which SDGs AI Research Neglects](#8-area-7-known-findings-about-which-sdgs-ai-research-neglects)
9. [Area 8: Tensions, Debates, and Open Questions](#9-area-8-tensions-debates-and-open-questions)
10. [Area 9: Methodological Precedents](#10-area-9-methodological-precedents)
11. [Synthesis and Implications for the Dissertation](#11-synthesis-and-implications-for-the-dissertation)
12. [Conclusion](#12-conclusion)
13. [References](#references)

---

## 1. Introduction

The relationship between academic research and policy implementation has long been characterized by what scholars term the "valley of death"—a persistent gap between knowledge production and knowledge utilization [1]. In the domain of artificial intelligence (AI) for sustainability, this gap takes on particular urgency given the accelerating climate crisis and the 2030 deadline for achieving the United Nations Sustainable Development Goals (SDGs). While AI research has proliferated in recent years, with thousands of papers claiming contributions to sustainability objectives, fundamental questions remain about whether this research addresses the priorities articulated in policy frameworks, and whether researchers and policymakers are even discussing the same aspects of sustainability challenges when they invoke the same SDG labels.

This literature review examines three interconnected scholarly domains to establish the theoretical, empirical, and methodological foundations for measuring semantic alignment between AI sustainability research and policy frameworks. First, it surveys the landscape of efforts to map AI research to the 17 SDGs, examining methods ranging from keyword matching to machine learning and the empirical findings about which goals receive disproportionate attention. Second, it explores the broader literature on research-policy gaps in AI governance and technology domains, identifying patterns of misalignment and their causes. Third, it investigates computational methods for cross-corpus semantic comparison, with particular attention to embedding-based approaches using Sentence-BERT and centroid construction.

The review is structured around nine thematic areas, each addressing a specific dimension of the research question. For each area, the analysis identifies: (a) what exists in the current scholarly landscape, (b) key papers with methodological or empirical contributions, (c) what remains missing or contested, and (d) implications for the dissertation methodology. This comprehensive approach ensures that the proposed research builds on established precedents while addressing genuine gaps in knowledge.

The central methodological innovation under examination—using Sentence-BERT embeddings to construct per-SDG semantic centroids from policy documents, then scoring research papers against these centroids—represents a novel application of established NLP techniques to the research-policy alignment problem. This review therefore pays particular attention to precedents for centroid-based classification, domain adaptation challenges when comparing academic and policy text, and the validity of semantic similarity as a proxy for substantive alignment.

---

## 2. Area 1: Mapping AI Research to SDGs

### 2.1 Current Landscape

The systematic mapping of AI research to the UN Sustainable Development Goals emerged as a distinct research area following the 2015 adoption of the 2030 Agenda. Early efforts relied primarily on keyword-based approaches, but the field has evolved to incorporate sophisticated bibliometric analysis, topic modeling, and machine learning classification. This body of work seeks to answer fundamental questions about the distribution of AI research effort across the 17 goals, identify over- and under-represented domains, and assess whether AI's potential contributions align with the most pressing sustainability challenges.

The seminal work in this area is Vinuesa et al.'s 2020 Nature Communications paper "The role of artificial intelligence in achieving the Sustainable Development Goals" [1]. This consensus-based expert study assessed AI's potential impact on all 169 SDG targets, concluding that AI could enable 134 targets (79%) but might inhibit 59 targets (35%), with 14% overlap. The methodology involved literature review and expert elicitation rather than systematic bibliometric analysis, and the paper explicitly noted that "AI research is biased towards SDGs relevant to nations where most AI researchers live and work" [1]. This observation about geographic and thematic bias has become a recurring theme in subsequent research.

Following Vinuesa et al., multiple bibliometric studies have attempted to quantify SDG representation in AI literature. Singh et al. (2023) analyzed 20 years of AI, machine learning, and deep learning publications, finding that SDGs 3 (good health and well-being) and 7 (affordable and clean energy) received the most AI applications, followed by SDGs 4, 13, 11, and 16 [2]. Meitei et al. (2023) conducted a bibliometric study mapping AI/ML techniques to SDGs, though their analysis did not specify which goals were over- or under-represented by number [3]. Abdalkareem et al. (2025) analyzed 1,349 Scopus-indexed documents from 2003-2025, revealing "a rapidly emerging field with high thematic dispersion, moderate growth, and early citation impact," with India leading in research output [4].

### 2.2 Methodological Approaches

The methods used to map AI research to SDGs vary considerably in sophistication and validity. Keyword matching remains the most common approach, typically using SDG-specific dictionaries or ontologies. The OSDG (Open Source SDG) tool, for instance, integrates keywords, ontology items, and machine learning features linked to topics in the Microsoft Academic Graph [5]. However, keyword approaches face well-documented limitations: they struggle with polysemy (words with multiple meanings), miss semantically related content that uses different terminology, and cannot capture the nuanced ways researchers discuss SDG-relevant topics.

Topic modeling approaches, particularly Latent Dirichlet Allocation (LDA) and more recently BERTopic, offer an alternative by discovering latent thematic structure in research corpora. Nedungadi et al. (2024) used BERTopic to analyze big data and AI literature from 2013-2024, identifying healthcare (SDG 3), sustainable energy (SDG 7), and industry/infrastructure (SDG 9) as dominant themes [6]. Tashakori et al. (2025) combined bibliometric co-word mapping with BERTopic to derive "complementary structural and semantic views" of sustainability research, revealing thematic frontiers and making SDG linkages explicit [7].

Machine learning classification represents the most sophisticated approach, training supervised models on labeled datasets to predict SDG relevance. Hajikhani et al. (2022) applied machine learning to map publications and patents to SDGs, achieving "acceptable accuracy (above 60%) for most SDGs" but noting that some goals (SDGs 8, 14, 15) were difficult to identify from text alone [8]. Yin et al. (2025) developed an AI-based tool for mapping publications to SDGs using similarity measures, benchmarking the accuracy of automated mapping [9]. The challenge for all supervised approaches is the availability of high-quality labeled training data—a gap that the OSDG Community Dataset attempts to address.

### 2.3 Empirical Findings on SDG Distribution

Across multiple studies, a consistent pattern emerges: AI research concentrates on a subset of SDGs related to health, energy, climate, and infrastructure, while systematically neglecting goals related to social equity, governance, and peace. Cowls et al. (2021) surveyed 108 AI for Social Good projects and found SDG 3 (health) leading in representation, while SDGs 5 (gender equality), 16 (peace and justice), and 17 (partnerships) were under-addressed [10]. This finding is particularly significant because it represents actual deployed AI projects rather than just published research.

Chavarro et al. (2022) conducted a scientometric analysis of IEEE Xplore papers from 2000-2019, finding that "a small share of papers explicitly focused on Sustainable Development" and that "inter-regional and inter-income group collaboration were limited, with network power concentrated in a few countries" [11]. This geographic concentration reinforces Vinuesa et al.'s observation about bias toward wealthy-nation priorities.

Domain-specific analyses reveal similar patterns. Hoyas (2023) analyzed approximately 820,000 aerospace engineering papers from 2011-2020 using an AI-based model (ASDG), finding SDGs 7 (clean energy), 9 (industry), 11 (sustainable cities), and 13 (climate action) most contributed [12]. Ramezani et al. (2024) focused on health-related SDGs, documenting "a significant increase in AI research in health-related SDGs during 2015-2022" [13]. Filho et al. (2022) surveyed researchers who believed AI is most profitable for SDGs 7, 11, 13, and 17 [14].

### 2.4 Critiques and Limitations

The Vinuesa et al. (2020) paper, while highly influential (published in Nature Communications), has faced methodological critiques. Its reliance on expert consensus rather than systematic evidence synthesis means findings reflect expert perceptions rather than empirical analysis of research output. The paper also lacks specificity about which AI techniques enable or inhibit which targets, treating "AI" as a monolithic category. Varelas et al. (2024) extended Vinuesa's approach by applying AI to map funded research to SDGs, revealing "unbalanced sustainability domains" but noting "the lack of explicit expert validation" as a limitation [15].

Armitage et al. (2020) raised a fundamental methodological question: "Do independent bibliometric approaches get the same results?" when mapping scholarly publications to SDGs [16]. This concern about methodological consistency is critical for the present dissertation, which proposes yet another approach (embedding-based semantic scoring). If different methods produce substantially different SDG distributions, this suggests that the choice of method may matter more than the underlying research landscape.

Several studies note the challenge of distinguishing between AI for sustainability (using AI to address sustainability challenges) and AI in sustainability (studying AI's own sustainability impacts). This conceptual ambiguity complicates efforts to map research to SDGs, as papers may discuss AI's environmental costs (relevant to SDG 13) while ostensibly contributing to other goals.

### 2.5 Implications for the Dissertation

The existing literature on mapping AI research to SDGs validates several aspects of the proposed dissertation methodology while highlighting critical considerations:

**Validation**: The consistent finding across multiple studies that certain SDGs (particularly 3, 7, 9, 13) dominate AI research while others (5, 10, 16) are neglected provides a baseline expectation. If the dissertation's embedding-based approach produces radically different distributions, this would require explanation. The convergence of findings across different methods (bibliometrics, topic modeling, expert surveys) suggests a genuine pattern rather than methodological artifact.

**Challenge**: The dissertation must address the domain mismatch problem more explicitly than prior work. Most existing studies map research papers to SDG labels derived from the goals themselves or from other research papers. The proposed approach—comparing research papers to policy document centroids—introduces an additional layer of complexity: policy documents may discuss SDGs using different vocabulary, framing, and level of abstraction than academic papers. This is precisely the semantic gap the dissertation aims to measure, but it requires careful methodological justification.

**Opportunity**: No existing study has systematically compared the semantic content of AI research papers to the semantic content of policy documents at the SDG level. Most work assumes that if a paper addresses SDG 13 (climate action), it aligns with policy priorities for SDG 13. The dissertation's innovation is to test this assumption by measuring whether papers and policies discuss the same aspects of each SDG, even when both are labeled with the same goal number.

**Methodological consideration**: The literature reveals tension between coverage (mapping all AI research to all SDGs) and depth (understanding what aspects of each SDG are addressed). The dissertation's chunk-level analysis of policy documents, rather than document-level classification, represents a move toward depth. This allows detection of within-SDG semantic gaps—cases where research and policy both address SDG 13, for example, but focus on different climate-related topics.

---

## 3. Area 2: Research-Policy Alignment and Gaps in AI and Technology

### 3.1 The Research-Policy Gap in AI Governance

The gap between AI research priorities and policy needs has emerged as a critical concern in AI governance scholarship. Mejía (2025) conducted a bibliometric study of research-policy alignment in the context of the EU AI Act, finding "gaps in regulatory implementation research and domain-specific applications" [17]. Notably, the study identified "limited academic engagement with regulatory bodies and oversight mechanisms, contrasting with substantial research on cultural heritage and medical applications lacking direct regulatory correspondence" [17]. This suggests that researchers pursue topics of academic interest or technical tractability rather than topics of regulatory urgency.

Sioumalas-Christodoulou et al. (2025) identified a fundamental misalignment between "the technical and economic focus of global AI metrics and the broader societal and ethical priorities emphasized in National Artificial Intelligence Strategies" [18]. Using topic modeling and qualitative content analysis, they found that evaluation frameworks emphasize measurable technical performance while policy documents stress ethical and social considerations aligned with UN SDGs. This metric-policy gap means that AI systems optimized for research benchmarks may not serve policy objectives.

Toney et al. (2024) examined "Trust Issues: Discrepancies in Trustworthy AI Keywords Use in Policy and Research," analyzing over 322,000 scientific papers and national policy documents from five countries [19]. While they found "broad agreement" on trustworthy AI principles, "substantive and relevant differences" existed in how these principles were operationalized. This suggests that apparent alignment at the level of high-level concepts (e.g., "fairness," "transparency") may mask deeper semantic divergence in what these concepts mean in practice.

### 3.2 Deployment-Stage Gaps

Strauss et al. (2025) documented a critical temporal gap in AI governance research: "Corporate AI research increasingly focuses on pre-deployment issues like model alignment and testing & evaluation, neglecting deployment-stage concerns such as model bias" [20]. Analyzing 1,178 safety and reliability papers from 9,439 generative AI papers (January 2020 – March 2025), they found "significant research gaps in high-risk domains, including healthcare, finance, misinformation, persuasive features, hallucinations, and copyright" [20]. This pre-deployment bias in research is particularly problematic because policy concerns center on real-world harms that emerge only after deployment.

Kim et al. (2025) identified "a significant lag between AI technological advancement and the development of policy and regulation, especially concerning specific AI systems categorized as high-risk by the EU AI Act" [21]. Their bibliometric analysis revealed "critical gaps in research concerning regulated AI systems, highlighting the need for more focused research aligned with the Act's regulatory framework" [21]. This suggests that research follows technological possibility rather than regulatory need.

### 3.3 Fragmentation and Coordination Failures

Hong et al. (2025) characterized existing governance responses for AI sustainability as "fragmented and reactive, lacking proactive, lifecycle-wide risk management" [22]. They identified "persistent limitations including the absence of operational tools, inconsistent standards, and insufficient cross-sectoral coordination" [22]. This fragmentation means that even when research addresses policy-relevant topics, the lack of coordination prevents effective knowledge transfer.

Agarwal et al. (2025) proposed a five-layer framework for AI governance, identifying "a critical gap between high-level regulatory mandates and specific, actionable guidance" [23]. For instance, "the EU AI Act mandates fairness assessments but lacks specific methodologies" [23]. This implementation gap means that researchers developing fairness metrics may not be addressing the specific fairness concerns that policymakers need operationalized.

Jiang et al. (2025) surveyed AI governance comprehensively, finding that "fragmented regulatory landscapes produce inconsistent oversight and enforcement, attributed to governance treated as an afterthought rather than foundational design" [24]. They noted that "evaluation protocols insufficiently reflect real-world deployment risks" and that "existing studies tend to isolate technical safety from broader governance or narrowly focus on specific risks" [24]. This siloing of research prevents the integrated, systems-level analysis that policy requires.

### 3.4 Domain-Specific Gaps

In climate change innovation, a study (2023) quantified "the gap between climate research and innovation action in Europe using AI methods and network science," finding "significant differences between and within these two layers" [25]. Loose research-action connections were identified in "bioproducts, biotechnologies, and risk assessment practices, where applications are still too few compared to research insights" [25]. This suggests that even in domains with strong policy urgency (climate), research does not translate to action.

Makhura (2025) examined AI in South African public policy implementation, identifying "a governance gap between policy design and implementation, exacerbated by challenges like inefficiency, poor resource allocation, and weak interdepartmental coordination" [26]. The study revealed "AI's potential to streamline processes and enhance decision-making in healthcare, education, and public safety, but also highlighted significant infrastructural and ethical challenges, including algorithmic bias, data privacy, and the digital divide" [26]. This highlights how research-policy gaps are compounded by implementation capacity gaps in resource-constrained settings.

### 3.5 Epistemic and Structural Causes

Giordano (2023) identified structural causes of research-policy misalignment in AI for science: "AI research often focuses on performance metrics over scientific problems, with AutoML researchers evaluating methods on technical benchmarks instead of impact" [27]. Furthermore, "policymakers need more expertise to decide on technology initiatives, and current NLP research is driven by standardized metrics and quick publications, limiting high-risk, speculative ideation" [27]. This suggests that academic incentive structures (publication metrics, benchmark competitions) systematically misalign research with policy needs.

Zysman et al. (2020) noted that "decision-makers often focus on artificial general intelligence (AGI), while policies should address narrow AI's distinctive problems like misconceived benefits, distribution, autonomous weapons, and algorithmic bias" [28]. This conceptual misalignment—policymakers worrying about future AGI while researchers build narrow AI systems with immediate societal impacts—represents a fundamental gap in problem framing.

Delic (2019) highlighted "a research gap in political science regarding AI, particularly on ethical issues beyond economic impacts," noting that "both states and scholars are 'lagging behind' in knowledge of fast-growing AI technology" [29]. The study found that "governments' lack of AI expertise and regulation is a major concern" and that "policy evaluation research on AI is scarce" [29]. This bidirectional knowledge gap—researchers not understanding policy processes, policymakers not understanding AI technology—creates structural barriers to alignment.

### 3.6 Implications for the Dissertation

The research-policy gap literature provides crucial context for the dissertation's focus on semantic alignment:

**Validation**: The consistent finding across domains that research and policy priorities diverge validates the dissertation's premise that alignment cannot be assumed. The specific finding that research focuses on pre-deployment technical issues while policy emphasizes deployment-stage societal impacts suggests that semantic gaps may be particularly pronounced in certain SDG domains (e.g., SDG 16 on justice and institutions, where deployment context is critical).

**Challenge**: The literature reveals that research-policy gaps have multiple causes—incentive structures, expertise asymmetries, fragmented governance, temporal lags—that cannot be addressed by better semantic alignment alone. The dissertation must be careful not to imply that measuring semantic gaps will solve research-policy misalignment. Rather, semantic gap measurement is a diagnostic tool that can reveal where misalignment exists and potentially inform interventions.

**Theoretical grounding**: The finding that gaps exist at multiple levels—high-level principles (where apparent agreement masks operational divergence) and specific implementations (where research addresses tractable problems rather than policy-urgent problems)—suggests that the dissertation's two-level analysis (coverage gaps across SDGs and semantic gaps within SDGs) is well-motivated. The literature predicts that even when research and policy both address the same SDG, they may discuss different aspects.

**Methodological consideration**: Several studies used text analysis methods (topic modeling, keyword analysis, semantic similarity) to detect research-policy gaps, providing methodological precedents. However, most compared research to policy at a coarse-grained level (e.g., comparing research topics to policy themes). The dissertation's innovation—using embeddings to measure semantic distance between research papers and policy document chunks at the SDG level—offers finer-grained analysis that can detect within-SDG divergence.

---

## 4. Area 3: Semantic Similarity and Embedding Methods for Cross-Corpus Comparison

### 4.1 Sentence-BERT and Transformer-Based Embeddings

The development of Sentence-BERT (SBERT) by Reimers and Gurevych (2019) represented a breakthrough for semantic similarity tasks, enabling efficient computation of semantically meaningful sentence embeddings using siamese BERT networks [30]. Unlike standard BERT, which requires feeding sentence pairs through the network (computationally expensive for large-scale comparison), SBERT produces fixed-size embeddings that can be compared using cosine similarity, reducing retrieval time from 65 hours to 5 seconds for 10,000 sentences [30].

Gjorgjevikj et al. (2025) conducted domain-specific benchmarking of sentence encoders for associating indicators with SDG targets, finding that "fine-tuning improved predictive performance over baselines and reduced sensitivity to changes in indicator description length" [31]. This suggests that while general-purpose SBERT models provide strong baselines, domain-specific fine-tuning can improve performance for specialized tasks like SDG classification.

Justino et al. (2025) compared BERT-based embedding models for semantic retrieval of Brazilian legal precedents, finding that "task-specific SBERT-pt model fine-tuned for similarity achieved the highest performance" compared to general-purpose and domain-specific models [32]. Evaluation using Precision@k, MRR@15, and MAP@15 showed that "domain adaptation offered marginal benefits, while task-specific finetuning was most influential" [32]. This finding is significant for the dissertation: it suggests that fine-tuning SBERT on SDG-specific similarity tasks may improve performance more than simply using domain-adapted models.

### 4.2 Centroid-Based and Aggregation Methods

The dissertation's proposed approach—constructing per-SDG centroids from policy document embeddings—has precedents in the literature, though not specifically for research-policy comparison. Yuan et al. (2022) used BERT embeddings with "prototypes determined by sentence embeddings, with mean aggregation for initialization" in cross-domain few-shot relation extraction [33]. Their approach employed "domain adaptation using Wasserstein distance to bridge domain gaps" [33], suggesting that explicit domain adaptation may be necessary when comparing research and policy corpora.

Bergman et al. (2023) conducted "a full-document analysis of the semantic relation between European Public Assessment Reports and EMA guidelines using a BERT language model" [34]. While their specific application differs, the methodological approach—comparing two distinct document types (regulatory reports vs. guidelines) using BERT embeddings—parallels the dissertation's comparison of research papers and policy documents. Their success in detecting semantic relationships across document types validates the feasibility of cross-corpus comparison.

The question of document-level versus sentence-level embeddings is addressed by several studies. A 2023 paper asked "Are the Best Multilingual Document Embeddings simply Based on Sentence Embeddings?" and found that "clever combinations of sentence embeddings are often better than full document encoding" [35]. Specifically, "sentence average is strong for classification, while more complex combinations are needed for semantic tasks" [35]. This suggests that the dissertation's approach of chunking policy documents and averaging chunk embeddings to create centroids is methodologically sound.

### 4.3 Domain Adaptation and Cross-Domain Challenges

A critical challenge for the dissertation is the domain mismatch between academic research papers and policy documents. Tang et al. (2024) evaluated seven state-of-the-art embedding models on FinMTEB, a finance-specific benchmark, observing "a significant performance drop compared to general-purpose benchmarks, indicating these models struggle with domain-specific linguistic and semantic patterns" [36]. Importantly, "general-purpose model performance on MTEB does not correlate with FinMTEB performance, suggesting a need for domain-specific models and benchmarks" [36].

Bollegala et al. (2015) proposed an unsupervised method for cross-domain word representation learning, using "frequent words as 'pivots' and optimizing an objective function enforcing two constraints: pivots predict co-occurring non-pivots, and pivot representations are similar across domains" [37]. While their focus was sentiment classification, the principle—identifying shared vocabulary that bridges domains—is relevant for comparing research and policy text, which may share SDG-related terminology while using it in different contexts.

Hu et al. (2023) advanced domain adaptation of BERT by "learning domain term semantics," encoding term definitions and injecting semantics into BERT's vocabulary through contrastive learning [38]. This approach "narrows the semantic gap between original vocabulary and domain terms" [38], achieving "significant improvement on biomedical NLP tasks without affecting general tasks" [38]. For the dissertation, this suggests that explicitly modeling SDG-specific terminology could improve semantic alignment measurement.

### 4.4 Evaluation Metrics and Validation

The choice of evaluation metrics for semantic similarity is consequential. Nemani et al. (2022) used "an average Pearson correlation score of 0.79 on the U.S Patent Phrase to Phrase Matching Dataset" to evaluate DeBERTa variants for semantic similarity [39]. Gusdevi et al. (2025) compared SBERT against TF-IDF for fact verification, finding "SBERT achieves higher similarity with 'SUPPORTS' (0.65) and stronger negative similarity with 'NOT ENOUGH INFO' (-0.90) than TF-IDF (0.49 and -0.62)" [40]. These studies demonstrate that cosine similarity on SBERT embeddings provides meaningful semantic discrimination.

However, Kramer (2024) found that "BERT showed lower performance in challenging domains, likely due to insufficient domain-specific fine-tuning" when comparing Shakespeare sonnets and Taylor Swift lyrics [41]. This cautionary finding suggests that SBERT's performance may degrade when comparing texts from very different domains or genres—precisely the challenge the dissertation faces when comparing academic papers and policy documents.

### 4.5 Alternative Approaches

While the dissertation proposes centroid-based SBERT embeddings, alternative approaches exist. Cadeddu et al. (2025) compared task adaptation techniques for LLMs in SDG text classification, finding that "prompt optimization with flan-t5-large achieved macro F1-scores up to 0.75, closely matching gpt-3.5's 0.77" [42]. This suggests that zero-shot or few-shot LLM approaches could potentially classify SDG relevance without requiring large labeled datasets.

Zarrieß et al. (2025) proposed SemCSE-Multi, "an unsupervised framework that produces aspect-specific summarizing sentences and trains embedding models to map semantically related summaries to nearby positions" [43]. Their "decoding pipeline translates embeddings back into natural language descriptions, effective even for unoccupied regions, offering improved interpretability" [43]. This interpretability feature could be valuable for the dissertation: rather than just reporting that research and policy have low semantic similarity for a given SDG, the method could generate natural language descriptions of what each corpus emphasizes.

### 4.6 Implications for the Dissertation

The semantic similarity literature provides strong methodological foundations while highlighting critical challenges:

**Validation**: SBERT's proven effectiveness for semantic similarity tasks, including cross-domain applications, validates its use for comparing research and policy text. The finding that sentence-level embeddings can be effectively aggregated to document-level representations supports the centroid-based approach. Multiple studies demonstrate that cosine similarity on SBERT embeddings provides meaningful semantic discrimination.

**Challenge**: The domain adaptation literature reveals that embedding models trained on general text may struggle with domain-specific content. The dissertation must address whether SBERT, trained primarily on general English text, adequately captures the specialized vocabulary and discourse patterns of both AI research papers and policy documents. The finding that domain-specific fine-tuning often outperforms general models suggests that fine-tuning on SDG-specific text could improve results.

**Methodological consideration**: The literature suggests several enhancements to the basic centroid approach: (1) explicit domain adaptation using techniques like pivot features or contrastive learning to bridge the research-policy domain gap; (2) aspect-specific embeddings that capture different dimensions of SDG relevance (e.g., problem framing, proposed solutions, evaluation metrics); (3) interpretability mechanisms that translate semantic distances back into natural language descriptions of what differs between research and policy.

**Validation strategy**: The dissertation should validate SBERT's performance on SDG-specific semantic similarity tasks before applying it to research-policy comparison. This could involve: (1) comparing SBERT similarity scores to human judgments of SDG relevance for a sample of papers and policy chunks; (2) testing whether SBERT correctly identifies known cases of alignment and misalignment; (3) comparing SBERT-based results to alternative methods (keyword matching, topic modeling) to assess convergent validity.

---

## 5. Area 4: NLP and AI Methods for Policy Document Analysis

### 5.1 Computational Analysis of Policy Documents

The application of NLP methods to policy document analysis has grown substantially in recent years, driven by the availability of large policy corpora and advances in text analysis techniques. Cheng et al. (2025) conducted a "quantitative study on artificial intelligence governance policy texts under the framework of the United Nations," providing a precedent for analyzing policy documents through a UN framework lens [44]. Navaratna et al. (2025) performed "keyword and topic modelling analysis" of national AI policies, demonstrating the feasibility of extracting thematic structure from policy text [45].

Papadopoulos et al. (2020) asked "What do governments plan in the field of artificial intelligence?" and analyzed national AI strategies using NLP [46]. Their work revealed that computational text analysis can uncover patterns in policy priorities that may not be apparent from manual reading. Wang et al. (2023) conducted "Artificial Intelligence Policy Frameworks in China, the EU and the US: An Analysis Based on Structure Topic Model," demonstrating that topic modeling can reveal cross-national differences in policy emphasis [47].

Golpayegani et al. (2025) used "BERTopic and Thematic Analysis" to uncover "AI Governance Themes in EU Policies" [48]. This combination of computational (BERTopic) and qualitative (thematic analysis) methods represents a methodological best practice: using NLP to identify patterns at scale, then validating and interpreting these patterns through close reading. For the dissertation, this suggests that purely computational semantic alignment scores should be supplemented with qualitative analysis of selected cases.

### 5.2 Linguistic Properties of Policy Text

Policy documents have distinctive linguistic and rhetorical properties that differentiate them from academic text. Hassan (2022) investigated "Modality in Policy Texts: Corpus-assisted Critical Discourse Analysis of Modals in the 2030 Agenda for Sustainable Development," finding that policy documents use modal verbs (should, must, will) to express different degrees of obligation and commitment [49]. This performative dimension of policy language—where text doesn't just describe but enacts commitments—has no direct parallel in academic writing.

Torres (2021) examined "The Role of Modals in Policies: The US Opioid Crisis as a Case Study," finding that modal verb choice reflects policy stance and urgency [50]. Agbeleoba (2025) analyzed "Textual Cohesion and Inter-connectedness in Sustainable Development Goals (SDGs)-Related Speeches and Reports," revealing how policy documents create coherence across multiple goals through linguistic devices [51]. These studies suggest that policy text operates at a different level of abstraction than research papers: policies articulate goals and commitments, while research papers report methods and findings.

Baturo et al. (2017) introduced "the UN General Debate corpus," demonstrating that UN policy speeches can be analyzed computationally to "understand state preferences" [52]. Arias (2024) analyzed "The Textual Dynamics of International Policymaking: A New Corpus of UN Resolutions, 1946-2018," revealing how policy language evolves over time [53]. For the dissertation, these corpora provide potential sources of policy text beyond the SDG framework documents themselves.

### 5.3 Domain-Specific Challenges

Strelkovskii et al. (2025) used "a text analysis approach" to examine "Integration of UN sustainable development goals in national hydrogen strategies," finding that national policies vary substantially in how they incorporate SDG language [54]. This heterogeneity in policy text—ranging from high-level aspirational statements to specific implementation plans—poses challenges for semantic analysis. The dissertation must decide whether to focus on high-level SDG framework documents, national implementation strategies, or both.

Hung (2025) explored "China's cyber sovereignty concept and artificial intelligence governance model: a machine learning approach," demonstrating that policy analysis must account for different governance paradigms [55]. What counts as "policy" varies across political systems: in some contexts, policy is codified in legislation; in others, it emerges from party documents, white papers, or regulatory guidance. The dissertation must clearly define its policy corpus and justify this choice.

### 5.4 Methodological Approaches

Saheb (2024) conducted "Mapping Ethical Artificial Intelligence Policy Landscape: A Mixed Method Analysis," combining quantitative text analysis with qualitative interpretation [56]. This mixed-methods approach is particularly valuable for policy analysis, where context and nuance matter. Silva (2024) performed "Decoding Global AI Governance: A Computational Linguistic Analysis of National Regulations," demonstrating that computational methods can reveal patterns across multiple policy documents [57].

Chakraborti et al. (2024) introduced "NLP4Gov: A Comprehensive Library for Computational Policy Analysis," providing tools specifically designed for policy text [58]. This library includes functions for extracting policy actors, actions, and targets—structural elements that may be relevant for understanding policy priorities. For the dissertation, such tools could supplement embedding-based semantic analysis by identifying what specific actions policies propose for each SDG.

### 5.5 Implications for the Dissertation

The policy document analysis literature reveals both opportunities and challenges:

**Validation**: The successful application of NLP methods (topic modeling, keyword analysis, embedding-based classification) to policy documents validates the feasibility of computational policy analysis. The existence of established policy corpora (UN resolutions, national AI strategies, SDG framework documents) provides potential sources for the dissertation's policy text.

**Challenge**: Policy text has distinctive linguistic properties (modal verbs, performative language, high-level abstraction) that differentiate it from academic text. This domain mismatch is not just a technical challenge for embedding models but a conceptual challenge: policies and research papers serve different communicative functions. Policies articulate goals and commitments; research papers report findings and methods. Measuring semantic similarity between these text types requires careful interpretation.

**Methodological consideration**: The literature suggests that policy analysis benefits from mixed methods—computational analysis to identify patterns, qualitative analysis to interpret them. The dissertation should not rely solely on semantic similarity scores but should examine specific cases where scores indicate alignment or misalignment, reading the actual text to understand what drives these patterns.

**Corpus definition**: The dissertation must clearly define its policy corpus. Options include: (1) the SDG framework documents themselves (the 2030 Agenda, SDG indicator metadata); (2) national SDG implementation strategies; (3) AI governance policies that reference SDGs; (4) UN reports on SDG progress. Each choice has implications for what "policy priorities" means. The literature suggests that using multiple policy sources and comparing results could reveal whether alignment varies by policy type.

---

## 6. Area 5: SDG Classification Datasets and Tools

### 6.1 The OSDG Community Dataset

The OSDG (Open Source SDG) Community Dataset represents the most comprehensive publicly available resource for SDG text classification. The dataset contains 42,065 text excerpts and 303,643 assigned labels, validated by over 1,400 citizen scientists from more than 140 countries [59]. Texts are paragraph-length (3-6 sentences, approximately 90 words on average) and derived from publicly available documents, including over 3,000 UN-related sources [59].

The OSDG validation methodology is distinctive: rather than asking volunteers to select which of the 17 SDGs a text relates to (which would be cognitively demanding and time-consuming), volunteers make binary accept/reject decisions on suggested SDG labels [59]. Each text is validated by at least 3, and up to 9, different volunteers [59]. This approach balances label quality with scalability, though it introduces potential bias: the suggested labels come from an initial classification system, so the validation process can only reject false positives, not identify false negatives.

Pukelis et al. (2022) described OSDG 2.0 as "a multilingual tool for classifying text data by UN Sustainable Development Goals," supporting content in 15 languages [60]. The tool integrates "existing research and previous classifications into a robust framework," linking "features from various approaches, such as ontology items, keywords, or machine-learning model features, to topics in Microsoft Academic Graph" [60]. This integration of multiple classification approaches (keyword, ontology, machine learning) represents a strength, but also makes the system complex and potentially difficult to interpret.

### 6.2 Label Quality and Known Limitations

Ingram et al. (2025) noted a critical limitation of the OSDG Community Dataset: it "relies on binary crowd-validated annotations of pre-assigned labels, rather than open-ended, expert-generated relevance judgments" [61]. This means "high-quality graded relevance sets for SDG classification do not exist," limiting "the stability of traditional evaluation frameworks" [61]. For the dissertation, this suggests that using OSDG as ground truth for SDG classification may introduce systematic biases.

Tamagnone et al. (2025) addressed "the absence of a large, labeled dataset for patent-SDG classification, noting limitations of existing methods like keyword searches and transfer learning in scalability and generalizability" [62]. They developed a "silver-standard, soft multi-label dataset" using weak supervision, which "outperformed transformer-based models and zero-shot LLMs in internal validation" [62]. This work highlights that even state-of-the-art datasets like OSDG may not generalize well to all text types (e.g., patents vs. academic papers vs. policy documents).

Hajikhani et al. (2022) used "an existing classification of scientific publications' relevance to SDGs as a gold standard," noting that "this training data is limited and incorporated from lexical-based search queries" [63]. Their classification models showed "acceptable accuracy (above 60%) for most SDGs, but some (like SDG 8, 14, 15) were difficult to identify from text," indicating that "unbalanced performance between SDG classes" is a persistent challenge [63]. This suggests that certain SDGs may be inherently more difficult to classify from text, either because their scope is broader or because relevant text uses less distinctive vocabulary.

### 6.3 Alternative SDG Classification Resources

Beyond OSDG, several alternative SDG classification datasets and tools exist. Wulff et al. (2023) introduced the "SDG Knowledge Hub Dataset of SDG-labeled News Articles," containing 9,172 articles with labels "assigned by authors and validated by SDG Knowledge Hub editors" [64]. This expert-validated dataset provides higher label quality than crowd-sourced approaches but is limited to news articles, which may not generalize to academic or policy text.

Skrynnyk et al. (2023) developed the "SDGi Corpus: A Comprehensive Multilingual Dataset for Text Classification by Sustainable Development Goals," providing "baselines" for multilingual SDG classification [65]. The emphasis on multilingual coverage is important for global policy analysis, though the dissertation focuses on English-language text.

Clematide et al. (2025) organized the "SwissText 2024 Shared Task: Automatic Classification of the United Nations' Sustainable Development Goals (SDGs) and Their Targets in English Scientific Abstracts" [66]. This shared task provides a benchmark for comparing SDG classification methods on scientific abstracts specifically, which is directly relevant to the dissertation's focus on research papers.

Adauto et al. (2023) introduced "NLP4SGPAPERS, a scientific dataset of 5,000 papers annotated for SDG classification" [67]. This dataset revealed that "healthcare, education, and peace are popular, while poverty and hunger are largely unaddressed" [67]. Importantly, "the dataset has a skewed distribution, prompting upsampling for low-occurrence classes" [67], with "inter-annotator agreement for Task 2 at 88.67% Cohen's kappa" [67]. This high inter-annotator agreement suggests that expert annotators can reliably classify papers to SDGs, though the skewed distribution reflects the underlying imbalance in research output.

### 6.4 Domain Bias and Coverage Issues

Li et al. (2024) compared "Sustainable Development Goals Labeling Systems based on Topic Coverage," revealing "substantial discrepancies among SDG labeling systems" and emphasizing "the need for improved methodologies to enhance accuracy" [68]. The study highlighted "the crucial role of contextual information in keyword-based labeling systems, noting that overlooking context can introduce bias in the retrieval of papers" [68]. This finding is critical for the dissertation: different SDG classification systems may produce different results not because they measure different things, but because they handle context differently.

Li et al. (2024) trained "a novel SDG-related BERT model on the OSDG-CD corpus, which was extended by labeling approximately 10,000 sentences based on SDGs content" [69]. The model's "classification capabilities appear very effective," and "analysts using this methodology can make faster decisions about financial institutions' sustainability claims" [69]. However, the focus on financial institutions suggests potential domain bias: a model trained on OSDG (which includes UN documents and academic text) and fine-tuned on financial reports may not generalize well to AI research papers or policy documents.

### 6.5 Practitioner Perspectives and Evaluation

Gjorgjevikj et al. (2025) conducted "domain-specific benchmarking of a diverse sentence encoder portfolio" for "associating indicators with Sustainable Development Goals and Targets" [70]. Their finding that "fine-tuning improved predictive performance over baselines and reduced sensitivity to changes in indicator description length" [70] suggests that off-the-shelf SDG classification tools may not perform optimally without domain-specific adaptation.

Kannan et al. (2024) provided "a curated dataset of Indian sustainability startup texts labeled with UN SDG taxonomy," including "a detailed discussion of label bias and quality issues" [71]. This work highlights that SDG classification challenges vary by domain and geography: what counts as relevant to SDG 1 (poverty) may differ between Indian startups and European research institutions.

### 6.6 Implications for the Dissertation

The SDG classification literature reveals both the value and limitations of existing resources:

**Validation**: The OSDG Community Dataset provides a large-scale, validated resource for SDG classification that can serve as training data for the dissertation's centroid construction. The dataset's focus on paragraph-length text (90 words on average) aligns well with the proposed chunk-level analysis of policy documents. The multilingual coverage and diverse source documents (including UN materials) suggest reasonable generalizability.

**Challenge**: The OSDG dataset's reliance on binary validation of pre-assigned labels means it can only confirm or reject suggested classifications, not discover novel SDG associations. This may introduce systematic bias toward conventional understandings of what each SDG encompasses. For the dissertation, this means that policy documents discussing SDGs in unconventional ways may not be well-represented in OSDG-derived centroids.

**Methodological consideration**: The finding that certain SDGs (8, 14, 15) are difficult to classify from text suggests that semantic alignment measurement may be more reliable for some goals than others. The dissertation should report per-SDG confidence or reliability metrics, acknowledging that alignment scores for text-ambiguous SDGs should be interpreted cautiously.

**Alternative approach**: Rather than relying solely on OSDG-derived centroids, the dissertation could construct policy centroids directly from policy documents labeled with SDG tags (e.g., using SDG indicator metadata, which explicitly links indicators to goals and targets). This would ensure that policy centroids reflect how policies actually discuss SDGs, rather than how OSDG's crowd-sourced validators understand SDGs.

**Validation strategy**: The dissertation should validate its SDG classification approach against multiple benchmarks: OSDG, the SDG Knowledge Hub dataset, and the NLP4SGPAPERS dataset. If the embedding-based approach produces substantially different classifications than these established resources, this requires investigation: is the embedding method capturing something these resources miss, or is it introducing error?

---

## 7. Area 6: Theoretical Frameworks for Research-Policy Alignment

### 7.1 Multiple Streams Framework and Agenda-Setting

The Multiple Streams Framework (MSF), developed by John Kingdon and extended by subsequent scholars, provides a foundational theory for understanding how research enters policy agendas [72]. The framework posits three independent streams—problems, policies, and politics—that must converge for policy change to occur. Blum (2018) examined "the multiple-streams framework and knowledge utilization," focusing on "argumentative couplings of problem, policy, and politics issues" [73]. This work suggests that research influences policy not through direct knowledge transfer but through strategic coupling when policy windows open.

Taghizadeh et al. (2021) applied MSF to "childhood obesity prevention policies in Iran," demonstrating the framework's utility for analyzing "agenda-setting using Kingdon's multiple streams" [74]. For the dissertation, MSF suggests that semantic alignment between research and policy is necessary but not sufficient for research to influence policy: even perfectly aligned research may not affect policy if political conditions are unfavorable.

Knaggård (2015) introduced "the problem broker" as an extension to MSF, arguing that certain actors specialize in framing problems in ways that facilitate coupling across streams [75]. This concept is relevant for understanding how SDG framing functions: the SDGs provide a shared vocabulary that potentially enables research-policy coupling, but only if researchers and policymakers use this vocabulary to discuss the same problems.

### 7.2 Epistemic Communities

Haas's (1992) theory of epistemic communities—"networks of professionals with recognized expertise and competence in a particular domain and an authoritative claim to policy-relevant knowledge"—offers another lens for understanding research-policy relationships [76]. Epistemic communities influence policy by defining problems, identifying solutions, and establishing causal beliefs that policymakers adopt.

Dunlop (2009) examined "policy transfer as learning: capturing variation in what decision-makers learn from epistemic communities" [77]. This work highlights that policymakers don't simply adopt expert knowledge wholesale but selectively learn from epistemic communities based on political context and prior beliefs. For the dissertation, this suggests that semantic alignment may vary depending on whether AI researchers constitute a recognized epistemic community for sustainability policy.

Martínez (2025) rethought "the Role of Epistemic Communities in the International Response to Pandemics," noting that epistemic communities' influence depends on their credibility, legitimacy, and relevance to policymakers [78]. Sarkki et al. (2014) analyzed "trade-offs in science-policy interfaces," identifying tensions between "credibility, relevance and legitimacy" [79]. For the dissertation, this suggests that semantic alignment (relevance) is only one dimension of research-policy fit; credibility and legitimacy also matter.

### 7.3 Knowledge Utilization Theory

Rose et al. (2017) examined "policy windows for the environment: Tips for improving the uptake of scientific knowledge," identifying barriers to knowledge utilization including "timing, framing, and accessibility" [80]. Schut et al. (2013) analyzed "boundary arrangements at research-stakeholder interfaces in the policy debate on biofuel sustainability in Mozambique," revealing how "boundary organizations" mediate between research and policy [81].

Varisco (2018) explored "Policy Networks and Research Utilisation in Policy," arguing that research utilization depends on network structures that connect researchers and policymakers [82]. For the dissertation, this suggests that semantic alignment may be mediated by institutional structures: even when research and policy discuss the same topics, knowledge transfer may not occur without appropriate boundary-spanning mechanisms.

### 7.4 Mode 1 vs Mode 2 Knowledge Production

Gibbons et al.'s distinction between Mode 1 (disciplinary, academic) and Mode 2 (transdisciplinary, problem-focused) knowledge production provides a framework for understanding different research orientations [83]. Mode 1 research follows academic disciplinary logic and may not align with policy needs; Mode 2 research is explicitly problem-focused and involves stakeholders from the outset.

For the dissertation, this framework suggests that semantic misalignment between AI research and sustainability policy may reflect Mode 1 research orientation: researchers pursue questions of academic interest (e.g., improving model accuracy on benchmarks) rather than policy-relevant questions (e.g., whether AI systems exacerbate inequality). The SDG framework represents an attempt to orient research toward Mode 2 problem-focus, but the dissertation's empirical analysis will reveal whether this reorientation has occurred.

### 7.5 Demand-Driven vs Supply-Driven Research

Raina (2003) critiqued "supply driven science," arguing that research agendas driven by researcher interests rather than societal needs produce knowledge that remains unused [84]. This supply-driven vs demand-driven distinction maps onto the research-policy gap: supply-driven research may be semantically misaligned with policy because it addresses questions researchers find interesting rather than questions policymakers need answered.

For AI and sustainability, the literature suggests that much research is supply-driven: researchers apply AI techniques to sustainability problems that are technically tractable (e.g., image classification for species identification) rather than policy-urgent (e.g., addressing structural inequalities that drive biodiversity loss). The dissertation's semantic alignment analysis can potentially distinguish supply-driven research (low semantic similarity to policy) from demand-driven research (high semantic similarity).

### 7.6 Implications for the Dissertation

The theoretical frameworks literature provides essential conceptual grounding:

**Validation**: The existence of established theories explaining research-policy gaps validates the dissertation's premise that alignment cannot be assumed. These theories predict that research and policy will often diverge due to different institutional logics, incentive structures, and problem framings. The dissertation's empirical contribution is to measure the extent and nature of this divergence in the AI-sustainability domain.

**Challenge**: The theories reveal that semantic alignment is only one dimension of research-policy fit. Epistemic community theory emphasizes credibility and legitimacy; MSF emphasizes political timing; knowledge utilization theory emphasizes accessibility and framing. The dissertation must be careful not to imply that semantic alignment alone determines research impact on policy. Rather, semantic misalignment is a barrier to impact, but alignment is not sufficient.

**Theoretical contribution**: The dissertation can contribute to these theoretical frameworks by providing a novel measurement approach for one dimension of research-policy fit (semantic alignment). Existing applications of these theories rely primarily on qualitative case studies or surveys. The dissertation's computational approach enables large-scale, systematic measurement of alignment across all 17 SDGs, potentially revealing patterns not visible in case studies.

**Interpretation framework**: The theories provide frameworks for interpreting empirical findings. If the dissertation finds low semantic alignment for certain SDGs, possible explanations include: (1) epistemic community failure—AI researchers are not recognized as authoritative for these SDGs; (2) supply-driven research—researchers pursue technically interesting problems rather than policy-urgent ones; (3) problem framing divergence—researchers and policymakers conceptualize the SDG differently; (4) temporal mismatch—research addresses long-term technical challenges while policy focuses on immediate implementation.

---

## 8. Area 7: Known Findings About Which SDGs AI Research Neglects

### 8.1 Consistent Patterns of Neglect

Across multiple independent studies using different methodologies, a consistent pattern emerges: AI research systematically neglects SDGs related to social equity, governance, and peace. Cowls et al. (2021) found that SDGs 5 (gender equality), 16 (peace, justice and strong institutions), and 17 (partnerships for the goals) were under-addressed in their survey of 108 AI for Social Good projects [85]. This finding is particularly significant because it examines actual deployed projects rather than just published research, suggesting that the neglect extends beyond academic publication to real-world applications.

Vinuesa et al. (2020) noted that "AI research is biased towards SDGs relevant to nations where most AI researchers live and work," with "AI in agriculture often in wealthy nations, with only a handful of examples in less wealthy nations" [86]. This geographic bias suggests that SDG neglect reflects not just technical tractability but also the priorities and contexts of researchers, who are disproportionately located in high-income countries.

Ferreira et al. (2025) synthesized evidence on Human-Centered AI and SDGs for 2020-2024, finding "underrepresentation of the Global South, particularly Brazil" [87]. The study concluded that "HCAI needs to integrate ethical, regional, and impact-assessment dimensions more systematically to achieve global targets effectively" [87]. This finding reinforces that SDG neglect has a geographic dimension: research concentrates on problems and contexts familiar to researchers in high-income countries.

### 8.2 Over-Represented SDGs

The flip side of neglect is over-representation. Singh et al. (2023) found that "SDGs 3 (good health & well-being) and 7 (affordable and clean energy) have the most AI applications," followed by SDGs 4, 13, 11, and 16 [88]. Nedungadi et al. (2024) similarly identified "healthcare (SDG3), sustainable energy (SDG7), and industry and infrastructure (SDG9)" as dominant themes in big data and AI literature [89].

Raman et al. (2025) found that "AGI research has diversified into human-centered domains like healthcare (SDG 3), education (SDG 4), clean energy (SDG 7), industrial innovation (SDG 9), and public governance (SDG 16)" [90]. Filho et al. (2022) surveyed researchers who believed "AI is most profitable for SDGs 7 (Affordable and Clean Energy), 11 (Sustainable Cities), 13 (Climate Action), and 17 (Partnerships)" [91].

This concentration on SDGs 3, 7, 9, 11, and 13 reflects several factors: (1) technical tractability—these SDGs involve problems amenable to data-driven approaches (e.g., medical diagnosis, energy optimization, climate modeling); (2) data availability—health, energy, and climate domains have substantial digitized data; (3) funding priorities—these domains attract significant research funding; (4) researcher expertise—AI researchers often have backgrounds in engineering and computer science, making technical SDGs more accessible than social SDGs.

### 8.3 Explanations for Neglect

The literature offers several explanations for why certain SDGs are neglected:

**Tractability**: Greif et al. (2024) conducted "a systematic review of current AI techniques used in the context of the SDGs," noting that AI applications concentrate in domains where problems can be formulated as prediction or optimization tasks [92]. SDGs like 5 (gender equality) or 16 (peace and justice) involve complex social and political dynamics that resist reduction to technical problems.

**Data availability**: Hajikhani et al. (2022) noted that "some (like SDG 8, 14, 15) were difficult to identify from text," suggesting that these SDGs may lack distinctive textual markers or that relevant research doesn't explicitly invoke SDG framing [93]. More fundamentally, domains like SDG 14 (life below water) or SDG 15 (life on land) may lack the large-scale digitized datasets that AI methods require.

**Researcher demographics**: The finding that AI research concentrates on wealthy-nation priorities suggests that researcher demographics shape research agendas. If most AI researchers are male, from high-income countries, and trained in technical disciplines, they may not recognize or prioritize problems related to gender equality (SDG 5), reduced inequalities (SDG 10), or peace and justice (SDG 16).

**Funding structures**: Isoieva et al. (2024) examined "Threats and Benefits of AI in the context of targeting SDGs: A Youth Perception Approach," finding that "youth perceive AI as more beneficial for technical SDGs than social SDGs" [94]. If funders share this perception, they may preferentially fund AI research for technical SDGs, creating a self-reinforcing cycle of neglect for social SDGs.

### 8.4 Domain-Specific Patterns

Domain-specific analyses reveal additional patterns. Hoyas (2023) found that aerospace engineering research contributes most to "SDGs 7 (clean energy), 9 (industry), 11 (sustainable cities), and 13 (climate action)" [95], reflecting the field's focus on energy efficiency, transportation, and climate monitoring. Ramezani et al. (2024) documented "a significant increase in AI research in health-related SDGs during 2015-2022" [96], suggesting that SDG 3 (health) has become increasingly dominant.

Raghavendra et al. (2023) focused specifically on "AI's role in poverty alleviation (SDG-1)," finding limited research despite poverty being the first SDG [97]. This suggests that even SDGs that might seem amenable to AI approaches (e.g., using AI to target poverty interventions) remain neglected, possibly because poverty research requires interdisciplinary expertise that AI researchers lack.

### 8.5 Consensus and Disagreement

There is strong consensus across studies that SDGs 3, 7, 9, 11, and 13 dominate AI research, while SDGs 5, 10, 16, and 17 are neglected. However, some disagreement exists about intermediate SDGs. Singh et al. (2023) listed SDG 16 (peace and justice) among the more-addressed goals [98], while Cowls et al. (2021) found it under-addressed [99]. This discrepancy may reflect different corpora (published research vs. deployed projects) or different time periods.

The explanations for neglect are less settled. While technical tractability is widely cited, some scholars argue that framing certain SDGs as "technically intractable" is itself a political choice that naturalizes neglect. Sætra (2021) argued that "AI in context and the sustainable development goals" requires examining "the unsustainability of the sociotechnical system" [100], suggesting that technical approaches to SDGs may miss systemic issues.

### 8.6 Implications for the Dissertation

The literature on SDG neglect provides clear empirical expectations:

**Validation**: If the dissertation's embedding-based approach finds that AI research concentrates on SDGs 3, 7, 9, 11, and 13 while neglecting SDGs 5, 10, 16, and 17, this would validate the method by replicating established findings. Conversely, if the method produces substantially different patterns, this requires explanation: is the embedding approach capturing something previous methods missed, or is it introducing error?

**Coverage gap hypothesis**: The consistent finding of SDG neglect supports the dissertation's hypothesis that a coverage gap exists—certain SDGs receive disproportionate research attention relative to policy emphasis. The dissertation can quantify this gap by comparing the distribution of research papers across SDGs to the distribution of policy document chunks across SDGs.

**Semantic gap hypothesis**: The literature suggests that even for well-researched SDGs (3, 7, 9, 11, 13), research may focus on technically tractable sub-problems rather than policy-urgent aspects. For example, AI research on SDG 13 (climate action) may concentrate on climate modeling and prediction (technically tractable) while neglecting climate adaptation and justice (policy-urgent but technically complex). The dissertation's semantic gap analysis can test this hypothesis by examining whether research and policy discuss different aspects of the same SDG.

**Explanatory framework**: The literature's explanations for SDG neglect (tractability, data availability, researcher demographics, funding) provide a framework for interpreting the dissertation's findings. If certain SDGs show low semantic alignment, the dissertation can explore whether this reflects technical intractability (by examining the types of AI methods used), data constraints (by examining whether research cites data limitations), or other factors.

---

## 9. Area 8: Tensions, Debates, and Open Questions

### 9.1 AI as Enabler vs Disabler of SDG Progress

A fundamental tension in the literature concerns whether AI is a net enabler or disabler of SDG progress. Vinuesa et al. (2020) found that AI could enable 79% of SDG targets but might inhibit 35%, with 14% overlap [101]. This dual-use nature of AI means that the same technology can simultaneously advance and undermine sustainability goals. For example, AI-powered precision agriculture (enabling SDG 2, zero hunger) may increase energy consumption and e-waste (inhibiting SDG 12, responsible consumption).

Sætra (2021) argued that "AI in context and the sustainable development goals" requires examining "the unsustainability of the sociotechnical system" [102]. This critique suggests that focusing on AI's potential contributions to SDGs may obscure AI's own sustainability costs—energy consumption, resource extraction for hardware, labor exploitation in data annotation, and concentration of power in tech companies. Rehak et al. (2025) provocatively titled their paper "On the (im)possibility of sustainable artificial intelligence. Why it does not make sense to move faster when heading the wrong way" [103], arguing that current AI development trajectories are fundamentally incompatible with sustainability.

Heilinger et al. (2023) warned "Beware of sustainable AI! Uses and abuses of a worthy goal" [104], noting that "sustainable AI" rhetoric can serve as greenwashing, allowing companies to claim sustainability credentials while continuing harmful practices. This debate has direct implications for the dissertation: if AI research claims to address SDGs while actually undermining them, semantic alignment with policy priorities may be misleading.

### 9.2 Validity of Semantic Alignment as Proxy for Impact

A second tension concerns whether measuring semantic alignment via text analysis is a valid proxy for actual research-policy impact. The dissertation proposes that semantic similarity between research papers and policy documents indicates alignment, but this assumption can be questioned on several grounds.

First, semantic similarity may reflect superficial linguistic overlap rather than substantive alignment. Research and policy documents may both use SDG vocabulary ("climate action," "gender equality") while meaning different things. For example, a research paper on "gender equality" might focus on algorithmic fairness in hiring algorithms, while a policy document on "gender equality" might focus on structural barriers to women's political participation. These are both relevant to SDG 5, but they address different aspects and may not inform each other.

Second, research impact on policy may occur through mechanisms other than semantic alignment. Breakthrough research may introduce entirely new concepts or framings that don't initially align with policy language but eventually reshape policy discourse. Conversely, research that perfectly aligns with current policy language may be redundant, telling policymakers what they already know.

Third, the direction of influence matters. If research and policy show high semantic alignment, this could mean: (1) research is responding to policy priorities (demand-driven research); (2) policy is adopting research framings (research-driven policy); or (3) both are responding to external events (e.g., a climate disaster that focuses both research and policy attention on SDG 13). Semantic alignment measurement alone cannot distinguish these scenarios.

### 9.3 Critique of SDGs as Framework

A third debate concerns whether the SDG framework itself is the right lens for analyzing AI and sustainability. Several scholars have critiqued the SDGs as overly broad, internally contradictory, and insufficiently attentive to power dynamics and structural inequalities.

The SDGs encompass 17 goals and 169 targets, covering virtually all aspects of human development and environmental sustainability. This breadth makes the framework inclusive but also vague: almost any research or policy can be framed as relevant to some SDG. For the dissertation, this raises the question: if everything is relevant to SDGs, does SDG alignment become a meaningless metric?

Some SDGs are in tension with each other. SDG 8 (economic growth) may conflict with SDG 12 (responsible consumption) and SDG 13 (climate action), as economic growth historically correlates with resource consumption and emissions. If research addresses these tensions while policy documents treat SDGs as mutually reinforcing, semantic misalignment may reflect intellectual honesty rather than failure to align with policy.

The SDGs have been critiqued as reflecting Global North priorities and development paradigms. If AI research aligns with SDG policy frameworks developed primarily by high-income countries, this alignment may actually indicate misalignment with Global South needs and priorities. The dissertation must be attentive to this possibility: high semantic alignment might indicate problematic consensus around a flawed framework rather than desirable research-policy fit.

### 9.4 North-South Asymmetry

Related to the SDG critique is the question of North-South asymmetry in AI for sustainability research. Ferreira et al. (2025) found "underrepresentation of the Global South, particularly Brazil" in human-centered AI research [105]. Wall et al. (2021) examined "Artificial Intelligence in the Global South (AI4D): Potential and Risks," noting that AI development concentrates in high-income countries while deployment often occurs in low- and middle-income countries [106].

This geographic asymmetry raises questions about whose priorities shape both AI research and sustainability policy. If AI researchers are predominantly in the Global North and sustainability policies reflect Global North priorities, high semantic alignment between research and policy may indicate a shared blind spot rather than desirable convergence. The dissertation should consider whether semantic alignment varies by SDG in ways that reflect North-South priorities: for example, research might align well with policy on SDG 9 (industry and innovation, a Global North priority) but poorly on SDG 1 (poverty, a Global South priority).

### 9.5 AI for Sustainability vs AI in Sustainability

A conceptual distinction exists between "AI for sustainability" (using AI to address sustainability challenges) and "AI in sustainability" (studying AI's own sustainability impacts). Much research focuses on the former—applying AI to climate modeling, precision agriculture, smart cities—while neglecting the latter—AI's energy consumption, e-waste, labor conditions, and concentration of power.

Ghamisi et al. (2024) called for "Responsible AI for Earth Observation," arguing that AI applications for environmental monitoring must themselves be sustainable [107]. This distinction is relevant for the dissertation: if policy documents emphasize AI's sustainability impacts (AI in sustainability) while research focuses on AI's sustainability applications (AI for sustainability), this would constitute a semantic gap even if both invoke the same SDGs.

### 9.6 Implications for the Dissertation

The tensions and debates literature reveals fundamental challenges for the dissertation:

**Conceptual clarity**: The dissertation must clearly define what "alignment" means and why it matters. Is alignment desirable in all cases, or might misalignment sometimes reflect productive tension or critical distance? The literature suggests that alignment is not inherently good: research that uncritically aligns with policy may fail to challenge problematic assumptions.

**Interpretation framework**: When the dissertation finds semantic misalignment, multiple interpretations are possible: (1) research is failing to address policy priorities (problematic); (2) research is addressing aspects of SDGs that policy neglects (potentially valuable); (3) research and policy are using SDG language differently (conceptual divergence); (4) research is critically examining SDG assumptions that policy takes for granted (productive tension). The dissertation should examine specific cases to distinguish these scenarios.

**Limitations acknowledgment**: The dissertation must acknowledge that semantic alignment is an imperfect proxy for research-policy fit. High alignment doesn't guarantee research impact, and low alignment doesn't necessarily indicate research failure. The semantic alignment metric is a diagnostic tool that reveals patterns requiring further investigation, not a definitive measure of research quality or policy relevance.

**Critical perspective**: The dissertation should maintain critical distance from both AI research and SDG policy frameworks. Rather than assuming that research should align with policy, the analysis can reveal where alignment exists, where it doesn't, and what these patterns suggest about the relationship between AI research and sustainability governance.

---

## 10. Area 9: Methodological Precedents

### 10.1 Cross-Corpus Semantic Comparison

The dissertation's core methodological innovation—using embeddings to measure semantic alignment between two distinct text corpora (research papers and policy documents)—has precedents in several domains. Yang-liu et al. (2025) examined "Beyond Citations: Measuring Idea-level Knowledge Diffusion from Research to Journalism and Policy-making," using computational methods to track how research ideas propagate into policy documents [108]. While their specific approach differs, the conceptual framework—measuring research-policy connection through text analysis—directly parallels the dissertation.

Bergman et al. (2023) conducted "a full-document analysis of the semantic relation between European Public Assessment Reports and EMA guidelines using a BERT language model" [109]. This study compared two types of regulatory documents (reports vs. guidelines) using BERT embeddings to measure semantic similarity. The finding that BERT embeddings can detect meaningful semantic relationships across document types validates the dissertation's approach.

Zimmerman (2026) analyzed "Semantic Novelty Trajectories in 80,000 Books: A Cross-Corpus Embedding Analysis," demonstrating that embedding-based methods can track semantic change across large text corpora over time [110]. While focused on literary analysis rather than research-policy comparison, the methodological approach—using embeddings to measure semantic distance between text collections—is directly relevant.

### 10.2 Document-Level Alignment Methods

Wang et al. (2025) introduced "BiMax: Bidirectional MaxSim Score for Document-Level Alignment," proposing a method for aligning documents based on semantic similarity [111]. Their approach uses bidirectional maximum similarity scoring, which could be adapted for research-policy alignment: rather than comparing each research paper to a single policy centroid, the method could identify which policy chunks each paper is most similar to, and vice versa.

Ganguly et al. (2018) developed "Word Embedding based Semantic Cross-Lingual Document Alignment in Comparable Corpora," demonstrating that embedding-based methods can align documents across languages [112]. While the dissertation focuses on English-language text, the cross-lingual alignment problem is analogous to the cross-domain alignment problem: research papers and policy documents use different vocabularies and discourse conventions, much like different languages.

Pial et al. (2023) introduced "GNAT: A General Narrative Alignment Tool," which aligns narratives based on semantic similarity [113]. This tool could potentially be adapted for research-policy alignment, treating research papers and policy documents as different narratives about sustainability challenges.

### 10.3 Centroid-Based Classification

The dissertation's use of centroids—averaging embeddings of policy document chunks to create per-SDG centroids, then measuring research paper similarity to these centroids—has precedents in few-shot learning and prototype-based classification. Yuan et al. (2022) used "prototypes determined by sentence embeddings, with mean aggregation for initialization" in cross-domain few-shot relation extraction [114]. Their approach of using mean embeddings as prototypes directly parallels the dissertation's centroid construction.

Hematialam et al. (2021) developed "A Method for Computing Conceptual Distances between Medical Recommendations: Experiments in Modeling Medical Disagreement," using embeddings to measure semantic distance between medical guidelines [115]. This application—measuring disagreement between expert recommendations—is conceptually similar to measuring misalignment between research and policy.

### 10.4 Domain Adaptation for Cross-Corpus Comparison

A critical methodological challenge is domain adaptation: research papers and policy documents come from different domains with different linguistic conventions. Beyer et al. (2020) examined "Embedding Space Correlation as a Measure of Domain Similarity," proposing methods to quantify how similar two domains are in embedding space [116]. This could be used to validate whether research and policy corpora are sufficiently similar for meaningful comparison.

Bianchi et al. (2020) developed "Compass-aligned Distributional Embeddings for Studying Semantic Differences across Corpora," proposing a method to align embeddings from different corpora to enable comparison [117]. Their approach ensures that embeddings from different corpora occupy comparable regions of semantic space, addressing the domain mismatch problem.

Vera et al. (2025) introduced "MOSAIC: Masked Objective with Selective Adaptation for In-domain Contrastive Learning," a framework for domain adaptation of sentence embedding models [118]. Their method "achieves improvements up to 13.4% in NDCG@10 over strong general-domain baselines" [118], suggesting that domain-specific adaptation could substantially improve the dissertation's semantic alignment measurement.

### 10.5 Evaluation and Validation Methods

A key methodological question is how to validate semantic alignment measurements. Jurgens et al. (2015) examined "Semantic Similarity Frontiers: From Concepts to Documents," proposing evaluation methods for semantic similarity at different granularities [119]. Their framework could be adapted to validate the dissertation's approach: do document-level similarity scores correlate with human judgments of research-policy alignment?

Hassan et al. (2024) introduced "UESTS: An Unsupervised Ensemble Semantic Textual Similarity Method," proposing an ensemble approach that combines multiple similarity metrics [120]. This suggests that the dissertation could improve robustness by using multiple embedding models and aggregating their similarity scores, rather than relying on a single model.

A 2022 paper examined "Measuring the Measuring Tools: An Automatic Evaluation of Semantic Metrics for Text Corpora," proposing meta-evaluation methods for semantic similarity metrics [121]. This work highlights that different similarity metrics may produce different results, and the choice of metric should be justified based on the specific task.

### 10.6 Applications to Policy and Governance

Several studies have applied semantic similarity methods specifically to policy analysis. Keating (2014) examined "Cross-Border Policy Alignment Using Transformer-Based Representations and Unsupervised Clustering Models," demonstrating that transformer embeddings can identify policy alignment across jurisdictions [122]. This application—measuring policy alignment—is closely related to the dissertation's goal of measuring research-policy alignment.

Yang et al. (2019) developed "Bi-directional Relevance Matching between Medical Corpora," proposing a method to measure relevance between different medical text collections [123]. Their bidirectional approach—measuring both how relevant corpus A is to corpus B and vice versa—could reveal asymmetries in research-policy alignment: research might address policy priorities more than policy addresses research findings.

### 10.7 Implications for the Dissertation

The methodological precedents literature provides strong validation while suggesting enhancements:

**Validation**: Multiple studies have successfully used embedding-based methods for cross-corpus semantic comparison in domains ranging from regulatory documents to medical guidelines to literary analysis. This validates the feasibility of the dissertation's approach. The specific precedent of comparing regulatory documents (Bergman et al. 2023) is particularly relevant.

**Methodological enhancement**: The literature suggests several ways to strengthen the basic centroid approach:
1. **Bidirectional alignment** (Wang et al. 2025): Rather than just measuring how well research aligns with policy, also measure how well policy aligns with research. Asymmetries could reveal whether research is ignoring policy priorities or policy is ignoring research findings.
2. **Domain adaptation** (Vera et al. 2025, Bianchi et al. 2020): Explicitly adapt embeddings to bridge the research-policy domain gap, potentially improving alignment measurement accuracy.
3. **Ensemble methods** (Hassan et al. 2024): Use multiple embedding models and aggregate results to improve robustness.
4. **Interpretability** (Zarrieß et al. 2025): Develop methods to translate semantic distances back into natural language descriptions of what differs between research and policy.

**Validation strategy**: The literature emphasizes the importance of validating semantic similarity measurements against human judgments. The dissertation should include a validation study where human experts rate research-policy alignment for a sample of papers and SDGs, then assess whether embedding-based similarity scores correlate with expert judgments.

**Granularity considerations**: The literature reveals tension between document-level and chunk-level analysis. Document-level embeddings (averaging all sentences in a document) may obscure important details; chunk-level embeddings (analyzing paragraphs or sections separately) provide finer granularity but increase complexity. The dissertation's choice to analyze policy documents at chunk level while treating research papers as documents represents a compromise that should be justified.

---

## 11. Synthesis and Implications for the Dissertation

### 11.1 Convergent Findings Across Areas

Several findings emerge consistently across the nine thematic areas, providing strong empirical and theoretical foundations for the dissertation:

**SDG distribution patterns**: Multiple independent studies using different methods (bibliometrics, topic modeling, expert surveys, project databases) find that AI research concentrates on SDGs 3, 7, 9, 11, and 13 while neglecting SDGs 5, 10, 16, and 17. This convergence suggests a genuine pattern rather than methodological artifact, providing a baseline expectation for the dissertation's findings.

**Research-policy gaps**: Across AI governance, climate policy, health policy, and other domains, studies consistently find that research priorities diverge from policy needs. Common patterns include research focusing on technically tractable problems while policy emphasizes deployment-stage societal impacts, and research following academic incentives (publications, benchmarks) rather than policy urgency.

**Embedding method validity**: Multiple studies demonstrate that Sentence-BERT and related transformer-based embedding methods can effectively measure semantic similarity across different text types, including regulatory documents, medical guidelines, and policy texts. The centroid-based approach (averaging embeddings to create prototypes) has precedents in few-shot learning and prototype-based classification.

**Domain adaptation necessity**: Studies consistently find that general-purpose embedding models struggle with domain-specific text, and that domain adaptation or fine-tuning improves performance. This suggests that the dissertation should not assume that off-the-shelf SBERT will optimally capture research-policy semantic similarity without adaptation.

### 11.2 Methodological Validation and Challenges

The literature validates the dissertation's core methodological approach while identifying critical challenges:

**Validated**: The use of Sentence-BERT embeddings for semantic similarity measurement, the construction of centroids from policy document chunks, the comparison of research papers to these centroids via cosine similarity, and the analysis of both coverage gaps (SDG distribution) and semantic gaps (within-SDG divergence) all have precedents in the literature.

**Challenged**: The domain mismatch between research papers and policy documents is more severe than the literature typically addresses. Most cross-corpus comparison studies examine corpora that are more similar (e.g., two types of regulatory documents, or research papers from different fields) than research papers and policy documents. The dissertation must validate that SBERT embeddings can bridge this domain gap.

**Enhanced**: The literature suggests several methodological enhancements: bidirectional alignment measurement, explicit domain adaptation, ensemble methods combining multiple embedding models, and interpretability mechanisms that translate semantic distances into natural language descriptions.

### 11.3 Theoretical Contributions

The dissertation can make several theoretical contributions by integrating insights from multiple literatures:

**Operationalizing epistemic community theory**: Epistemic community theory predicts that research influences policy when researchers constitute a recognized expert community with shared causal beliefs. The dissertation's semantic alignment measurement operationalizes one dimension of this theory: shared causal beliefs should manifest as semantic similarity in how research and policy discuss problems and solutions.

**Extending knowledge utilization theory**: Knowledge utilization theory identifies multiple barriers to research uptake in policy, including relevance, credibility, and legitimacy. The dissertation's semantic alignment metric specifically measures relevance—whether research addresses topics that policy prioritizes. By quantifying relevance at scale across all 17 SDGs, the dissertation can test knowledge utilization theory's predictions about when and why research-policy gaps emerge.

**Bridging computational and qualitative policy analysis**: The dissertation bridges computational text analysis (embeddings, semantic similarity) and qualitative policy analysis (close reading, interpretation). This mixed-methods approach can reveal patterns at scale while providing rich interpretation of specific cases, advancing methodological integration in policy studies.

### 11.4 Empirical Contributions

The dissertation will make several empirical contributions:

**Comprehensive SDG coverage**: While existing studies examine AI research distribution across SDGs, none systematically compare research to policy at the semantic level for all 17 goals. The dissertation will provide the first comprehensive semantic alignment analysis across the full SDG framework.

**Coverage vs semantic gaps**: The dissertation's two-level analysis—coverage gaps (which SDGs get disproportionate attention) and semantic gaps (within-SDG divergence in what aspects are discussed)—will reveal whether apparent alignment at the coverage level masks deeper semantic misalignment. This distinction has not been systematically examined in prior work.

**Temporal dynamics**: If the dissertation includes temporal analysis (comparing alignment over time), it can reveal whether research-policy alignment is improving, worsening, or remaining stable. This would test whether the SDG framework's adoption in 2015 has successfully oriented research toward policy priorities.

**Domain-specific patterns**: By analyzing alignment separately for each SDG, the dissertation can reveal whether research-policy gaps vary by domain in predictable ways (e.g., larger gaps for social SDGs than technical SDGs), testing explanations based on tractability, data availability, and researcher expertise.

### 11.5 Practical Implications

The dissertation's findings will have practical implications for multiple stakeholders:

**For researchers**: Semantic alignment analysis can reveal which SDGs and which aspects of SDGs are under-researched relative to policy priorities, potentially guiding research agenda-setting. However, the dissertation should not imply that research should uncritically align with policy; critical distance may be valuable.

**For policymakers**: Understanding where research does and doesn't align with policy priorities can inform evidence-based policymaking. If research concentrates on certain SDGs or certain aspects of SDGs, policymakers should be aware of these gaps when seeking research evidence.

**For funders**: Research funders seeking to support policy-relevant research can use semantic alignment analysis to identify gaps and target funding accordingly. However, funders should balance alignment with policy priorities against support for exploratory research that may not initially align with policy but could reshape policy discourse.

**For SDG governance**: The UN and other SDG governance bodies can use semantic alignment analysis to assess whether the SDG framework is successfully orienting research toward global priorities, or whether adjustments to the framework or its communication are needed.

### 11.6 Limitations and Caveats

The literature review reveals several limitations that the dissertation must acknowledge:

**Semantic alignment as proxy**: Semantic similarity between research and policy text is an imperfect proxy for actual research impact on policy. High alignment doesn't guarantee impact, and low alignment doesn't necessarily indicate research failure. The metric is diagnostic, not evaluative.

**SDG framework limitations**: The SDGs themselves have been critiqued as overly broad, internally contradictory, and reflecting Global North priorities. High alignment with SDG policy frameworks may not indicate alignment with actual sustainability needs, particularly in the Global South.

**Domain mismatch**: Research papers and policy documents serve different communicative functions (reporting findings vs. articulating commitments) and use different discourse conventions. Measuring semantic similarity across this domain gap requires careful interpretation.

**Causality**: Semantic alignment measurement cannot determine causality. If research and policy show high alignment, this could mean research is responding to policy, policy is adopting research framings, or both are responding to external events. The dissertation cannot distinguish these scenarios without additional evidence.

**Temporal lag**: Research-policy alignment may exhibit temporal lag: research may address today's policy priorities, or it may address problems that will become policy priorities in the future. The dissertation's snapshot analysis may miss these temporal dynamics.

---

## 12. Conclusion

This comprehensive literature review has examined nine thematic areas relevant to measuring semantic alignment between AI sustainability research and policy frameworks. The review reveals a mature scholarly landscape with substantial empirical findings, established theoretical frameworks, and validated methodological approaches that provide strong foundations for the dissertation.

The empirical literature consistently finds that AI research concentrates on a subset of SDGs (particularly 3, 7, 9, 11, 13) while neglecting others (particularly 5, 10, 16, 17), and that research-policy gaps are pervasive across domains. These findings validate the dissertation's premise that alignment cannot be assumed and must be empirically measured. The theoretical literature provides frameworks—epistemic communities, knowledge utilization, multiple streams—for understanding why research-policy gaps emerge and persist. The methodological literature demonstrates that Sentence-BERT embeddings and centroid-based approaches can effectively measure semantic similarity across different text types, though domain adaptation may be necessary for optimal performance.

The dissertation's proposed methodology—using Sentence-BERT to construct per-SDG centroids from policy documents, then measuring research paper similarity to these centroids—represents a novel application of established techniques to the research-policy alignment problem. This approach enables large-scale, systematic measurement of both coverage gaps (SDG distribution) and semantic gaps (within-SDG divergence), advancing beyond prior work that typically examines only coverage.

However, the literature also reveals critical challenges and limitations. The domain mismatch between research papers and policy documents is substantial, and semantic similarity is an imperfect proxy for actual research impact. The SDG framework itself has limitations, and high alignment with SDG policy may not indicate alignment with actual sustainability needs. The dissertation must interpret findings carefully, acknowledging that semantic alignment is diagnostic rather than evaluative, and that both alignment and misalignment can be productive depending on context.

The review identifies several opportunities for the dissertation to make theoretical, empirical, and methodological contributions. Theoretically, the dissertation operationalizes dimensions of epistemic community and knowledge utilization theories at scale. Empirically, it provides the first comprehensive semantic alignment analysis across all 17 SDGs, distinguishing coverage from semantic gaps. Methodologically, it advances cross-corpus semantic comparison by addressing the research-policy domain gap and developing interpretability mechanisms.

Ultimately, this literature review establishes that the dissertation addresses a genuine gap in knowledge—we do not currently know the extent to which AI sustainability research aligns with policy priorities at the semantic level—using validated methods adapted from multiple scholarly traditions. The proposed research is well-grounded in existing literature while offering novel contributions that advance understanding of research-policy relationships in the critical domain of AI and sustainability.

---

## References

[1] Vinuesa, R., Azizpour, H., Leite, I., Balaam, M., Dignum, V., Domisch, S., Felländer, A., Langhans, S. D., Tegmark, M., & Fuso Nerini, F. (2020). The role of artificial intelligence in achieving the Sustainable Development Goals. *Nature Communications*, 11(1), 233. https://doi.org/10.1038/S41467-019-14108-Y

[2] Singh, P., Dwivedi, Y. K., Kahlon, K. S., Sawhney, R. S., Alalwan, A. A., & Rana, N. P. (2023). Artificial intelligence for Sustainable Development Goals: Bibliometric patterns and concept evolution trajectories. *Sustainable Development*, 31(6), 3915-3939. https://doi.org/10.1002/sd.2706

[3] Meitei, L. S., Singh, T. R., & Devi, T. P. (2023). Application of AI/ML techniques in achieving SDGs: a bibliometric study. *Environment, Development and Sustainability*, 26(11), 27221-27249. https://doi.org/10.1007/s10668-023-03935-1

[4] Abdalkareem, Z. A., Amir, A., Al-Betar, M. A., Hammouri, A. I., & Alkhawaldeh, R. S. (2025). Mapping the intersection of artificial intelligence and the sustainable development goals: A bibliometric and scientometric analysis (2003–2025). *ESTIDAMAA*, 1(1), 1-28. https://doi.org/10.70470/estidamaa/2025/001

[5] Pukelis, L., Bautista-Puig, N., Skrynik, M., & Stančiauskas, V. (2020). OSDG - Open-Source Approach to Classify Text Data by UN Sustainable Development Goals (SDGs). arXiv preprint. https://doi.org/10.48550/arxiv.2005.14569

[6] Nedungadi, P., Menon, R., Gutjahr, G., Erickson, L., & Raman, R. (2024). Big data and AI algorithms for sustainable development goals: a topic modeling analysis. *IEEE Access*, 12, 180516-180532. https://doi.org/10.1109/access.2024.3516500

[7] Tashakori, M., Hassani, H., Huang, X., Silva, E. S., & Ghodsi, M. (2025). Uncovering Semantic Patterns in Sustainability Research: A Systematic NLP Review. *Sustainable Development*, 33(2), e70319. https://doi.org/10.1002/sd.70319

[8] Hajikhani, A., Suominen, A., & Kässi, T. (2022). Mapping the sustainable development goals (SDGs) in science, technology and innovation: application of machine learning in SDG-oriented artefact detection. *Scientometrics*, 127(11), 6661-6693. https://doi.org/10.1007/s11192-022-04358-x

[9] Yin, Y., Dong, Y., Wang, K., Wang, D., & Jones, B. F. (2025). Leveraging artificial intelligence technology for mapping publications to sustainable development goals. *Array*, 25, 100419. https://doi.org/10.1016/j.array.2025.100419

[10] Cowls, J., Tsamados, A., Taddeo, M., & Floridi, L. (2021). A definition, benchmark and database of AI for social good initiatives. *Nature Machine Intelligence*, 3(2), 111-115. https://doi.org/10.1038/S42256-021-00296-0

[11] Chavarro, D., Tang, P., & Ràfols, I. (2022). Connecting brain and heart: artificial intelligence for sustainable development. *Scientometrics*, 127(4), 1829-1862. https://doi.org/10.1007/s11192-022-04299-5

[12] Hoyas, S., Sánchez-Roncero, A., Martín-Alcántara, A., Sanmiguel-Rojas, E., & Fernández-Feria, R. (2023). The Sustainable Development Goals and Aerospace Engineering: A critical note through Artificial Intelligence. *Results in Engineering*, 18, 100940. https://doi.org/10.1016/j.rineng.2023.100940

[13] Ramezani, M., Bashiri, A., Atashi, A., & Khodamoradi, F. (2024). Bibliometric Analysis of Artificial Intelligence Revolutions in Health-related Sustainable Development Goals. *Health Technology Assessment in Action*, 7(4), e14654. https://doi.org/10.18502/htaa.v7i4.14654

[14] Filho, W. L., Yang, P., Eustachio, J. H. P. P., Azul, A. M., Gellers, J. C., Gielczyk, A., Dinis, M. A. P., & Kozlova, V. (2022). Deploying digitalisation and artificial intelligence in sustainable development research. *Environment, Development and Sustainability*, 25(6), 4957-4988. https://doi.org/10.1007/s10668-022-02252-3

[15] Varelas, S., Georgiou, A., & Karlis, D. (2024). Artificial intelligence reveals unbalanced sustainability domains in funded research. *SSRN Electronic Journal*. https://doi.org/10.2139/ssrn.5031355

[16] Armitage, C. S., Lorenz, M., & Mikki, S. (2020). Mapping scholarly publications related to the Sustainable Development Goals: Do independent bibliometric approaches get the same results? *Quantitative Science Studies*, 1(3), 1092-1108. https://doi.org/10.1162/QSS_A_00071

[17] Mejía, C. (2025). Research-Policy Alignment in AI: A Bibliometric Study of the EU AI Act. *Proceedings of ISSI 2025*, 058. https://doi.org/10.51408/issi2025_058

[18] Sioumalas-Christodoulou, K., Kapitsaki, G. M., & Korfiatis, N. (2025). AI metrics and policymaking: assumptions and challenges in the shaping of AI. *AI & Society*. https://doi.org/10.1007/s00146-025-02181-5

[19] Toney, A., Pethig, F., Krügel, S., & Lütge, C. (2024). Trust Issues: Discrepancies in Trustworthy AI Keywords Use in Policy and Research. *Proceedings of the 2024 ACM Conference on Fairness, Accountability, and Transparency*, 2659-2670. https://doi.org/10.1145/3630106.3659035

[20] Strauss, B., Ilan, G., Shen, J., & Hooker, S. (2025). Real-World Gaps in AI Governance Research. *Scientia et Innovatio*, 2(3), 15163. https://doi.org/10.70777/si.v2i3.15163

[21] Kim, M., Zhu, J., Akata, Z., & Kasirzadeh, A. (2025). AI Governance in the Context of the EU AI Act (Extended Abstract). *Proceedings of the 2025 AAAI/ACM Conference on AI, Ethics, and Society*, 8(2), 36642. https://doi.org/10.1609/aies.v8i2.36642

[22] Hong, Y., Jiang, Y., & Zhang, H. (2025). Governing AI's sustainability: risks, current responses, and pathways for improved governance. *Zenodo*. https://doi.org/10.5281/zenodo.16730439

[23] Agarwal, R., Gao, G., DesRoches, C., & Jha, A. K. (2025). A five-layer framework for AI governance: integrating regulation, standards, and certification. *Transforming Government: People, Process and Policy*. https://doi.org/10.1108/tg-03-2025-0065

[24] Jiang, Y., Zhang, H., Chan, L., & Lyu, S. (2025). Never Compromise to Vulnerabilities: A Comprehensive Survey on AI Governance. *arXiv preprint*. https://doi.org/10.48550/arxiv.2508.08789

[25] Closing the gap between research and projects in climate change innovation in Europe. (2023). arXiv preprint. https://doi.org/10.48550/arxiv.2303.17560

[26] Makhura, M. (2025). Bridging the Governance Gap: Integrating Artificial Intelligence in South Africa's Public Policy Implementation. *Deleted Journal*. https://doi.org/10.62019/qq39na69

[27] Giordano, R. (2023). Artificial intelligence in science: Overview and policy proposals. *OECD Science, Technology and Industry Policy Papers*, 147. https://doi.org/10.1787/a2817e1f-en

[28] Zysman, J., Feldman, S., Murray, J., Kushida, K. E., & Breznitz, D. (2020). Governing AI: Understanding the Limits, Possibility, and Risks of AI in an Era of Intelligent Tools and Systems. *SSRN Electronic Journal*. https://doi.org/10.2139/SSRN.3681088

[29] Delic, A. (2019). ARTIFICIAL INTELLIGENCE: HOW AI IS UNDERSTOOD IN THE LIGHT OF DEMOCRACY AND HUMAN RIGHTS. A comparative case study of Sweden, France and the European Commission. [Master's thesis].

[30] Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)*, 3982-3992.

[31] Gjorgjevikj, A., Madjarov, G., Gjorgjevikj, D., & Lameski, P. (2025). Benchmarking Sentence Encoders in Associating Indicators with Sustainable Development Goals and Targets. *IEEE Access*, 13, 3595894. https://doi.org/10.1109/access.2025.3595894

[32] Justino, G., Oliveira, W., & Bittencourt, I. I. (2025). A Comparative Study of BERT Models for Semantic Retrieval of Brazilian Legal Precedents. *Proceedings of KDMiLe 2025*, 247782. https://doi.org/10.5753/kdmile.2025.247782

[33] Yuan, H., Yuan, Z., & Tan, C. (2022). Cross-Domain Few-Shot Relation Extraction via Representation Learning and Domain Adaptation. *arXiv preprint*. https://doi.org/10.48550/arXiv.2212.02560

[34] Bergman, U., Gustafsson, L. L., Höglund, P., Wettermark, B., & Vég, A. (2023). A full-document analysis of the semantic relation between European Public Assessment Reports and EMA guidelines using a BERT language model. *PLOS ONE*, 18(12), e0294560. https://doi.org/10.1371/journal.pone.0294560

[35] Are the Best Multilingual Document Embeddings simply Based on Sentence Embeddings? (2023). arXiv preprint. https://doi.org/10.48550/arxiv.2304.14796

[36] Tang, X., Luo, Y., Shen, Y., Ouyang, Z., Xiong, H., Zhu, Y., & Ding, J. (2024). Do We Need Domain-Specific Embedding Models? An Empirical Investigation. arXiv preprint. https://doi.org/10.48550/arxiv.2409.18511

[37] Bollegala, D., Mu, T., & Goulermas, J. Y. (2015). Unsupervised Cross-Domain Word Representation Learning. *Proceedings of the 53rd Annual Meeting of the Association for Computational Linguistics and the 7th International Joint Conference on Natural Language Processing*, 730-740. https://doi.org/10.3115/V1/P15-1071

[38] Hu, Y., Chen, Q., & Zuo, W. (2023). Advancing Domain Adaptation of BERT by Learning Domain Term Semantics. In *Natural Language Processing and Chinese Computing* (pp. 15-27). Springer. https://doi.org/10.1007/978-3-031-40292-0_2

[39] Nemani, V., Biggio, L., Huan, X., Hu, Z., Fink, O., Tran, A., Wang, Y., Zhang, X., & Hu, C. (2022). A Cognitive Study on Semantic Similarity Analysis of Large Corpora: A Transformer-based Approach. *2022 IEEE 19th India Council International Conference (INDICON)*, 1-6. https://doi.org/10.1109/INDICON56171.2022.10039840

[40] Gusdevi, R. A., Wibowo, A. T., & Fauzi, M. A. (2025). Cosine Similarity-Based Evidences Selection for Fact Verification Using SBERT on the FEVER Dataset. *Cogito Smart Journal*, 11(1), 52-66. https://doi.org/10.31154/cogito.v11i1.917.52-66

[41] Kramer, O. (2024). Comparative Analysis of Document-Level Embedding Methods for Similarity Scoring on Shakespeare Sonnets and Taylor Swift Lyrics. arXiv preprint. https://doi.org/10.48550/arxiv.2412.17552

[42] Cadeddu, A., Chessa, M., Fanni, S. C., Marcialis, G. L., & Meloni, P. (2025). A Comparative Study of Task Adaptation Techniques of Large Language Models for Identifying Sustainable Development Goals. *IEEE Access*, 13, 3618017. https://doi.org/10.1109/access.2025.3618017

[43] Zarrieß, S., Eger, S., & Ponzetto, S. P. (2025). SemCSE-Multi: Multifaceted and Decodable Embeddings for Aspect-Specific and Interpretable Scientific Domain Mapping. arXiv preprint. https://doi.org/10.48550/arxiv.2510.11599

[44] Cheng, X., Wang, Y., & Li, J. (2025). Quantitative Study on Artificial Intelligence Governance Policy Texts Under the Framework of the United Nations. *2025 IEEE International Conference on Artificial Intelligence and Next-Generation Information Technology (AINIT)*, 11035026. https://doi.org/10.1109/ainit65432.2025.11035026

[45] Navaratna, R., Lokuge, S., Sedera, D., & Dootson, P. (2025). National AI Policy: Keyword and Topic Modelling Analysis. *2025 IEEE International Conference on Global Communications (GCON)*, 11173368. https://doi.org/10.1109/gcon65540.2025.11173368

[46] Papadopoulos, S., Kompatsiaris, I., & Vakali, A. (2020). What do governments plan in the field of artificial intelligence?: Analysing national AI strategies using NLP. *Proceedings of the 13th International Conference on Theory and Practice of Electronic Governance*, 26-35. https://doi.org/10.1145/3428502.3428514

[47] Wang, X., Li, Y., & Zhang, J. (2023). Artificial Intelligence Policy Frameworks in China, the EU and the US: An Analysis Based on Structure Topic Model. *SSRN Electronic Journal*. https://doi.org/10.2139/ssrn.4547428

[48] Golpayegani, D., Pandit, H. J., & Lewis, D. (2025). Uncovering AI Governance Themes in EU Policies using BERTopic and Thematic Analysis. arXiv preprint. https://doi.org/10.48550/arxiv.2509.13387

[49] Hassan, M. M. (2022). Investigating Modality in Policy Texts: Corpus-assisted Critical Discourse Analysis of Modals in the 2030 Agenda for Sustainable Development. *Textual Turnings: An International Peer-Reviewed Journal in English Studies*, 4(1), 1-28. https://doi.org/10.21608/ttaip.2022.277139

[50] Torres, J. (2021). The Role of Modals in Policies: The US Opioid Crisis as a Case Study. *Applied Corpus Linguistics*, 1(2), 100008. https://doi.org/10.1016/J.ACORP.2021.100008

[51] Agbeleoba, O. A. (2025). Textual Cohesion and Inter-connectedness in Sustainable Development Goals (SDGs)-Related Speeches and Reports. *Journal of Research in Humanities and Social Science*, 13(9), 51-60. https://doi.org/10.35629/9467-13095160

[52] Baturo, A., Dasandi, N., & Mikhaylov, S. J. (2017). Understanding state preferences with text as data: Introducing the UN General Debate corpus. *Research & Politics*, 4(2), 2053168017712821. https://doi.org/10.1177/2053168017712821

[53] Arias, E. (2024). The Textual Dynamics of International Policymaking: A New Corpus of UN Resolutions, 1946-2018. *Journal of Peace Research*. https://doi.org/10.1177/00223433241280152

[54] Strelkovskii, N., Rovenskaya, E., Ilmola-Sheppard, L., & Abramzon, S. (2025). Integration of UN sustainable development goals in national hydrogen strategies: A text analysis approach. *International Journal of Hydrogen Energy*, 103, 1134-1147. https://doi.org/10.1016/j.ijhydene.2025.01.134

[55] Hung, C. F. (2025). Exploring China's cyber sovereignty concept and artificial intelligence governance model: a machine learning approach. *Journal of Computational Social Science*, 8(1), 346-378. https://doi.org/10.1007/s42001-024-00346-8

[56] Saheb, T. (2024). Mapping Ethical Artificial Intelligence Policy Landscape: A Mixed Method Analysis. *Science and Engineering Ethics*, 30(3), 472-506. https://doi.org/10.1007/s11948-024-00472-6

[57] Silva, R. (2024). Decoding Global AI Governance: A Computational Linguistic Analysis of National Regulations. *Proceedings of the 2024 AAAI/ACM Conference on AI, Ethics, and Society*, 7(2), 31909. https://doi.org/10.1609/aies.v7i2.31909

[58] Chakraborti, T., Isahagian, V., Khalaf, R., Khazaeni, Y., Muthusamy, V., Rizk, Y., & Unuvar, M. (2024). NLP4Gov: A Comprehensive Library for Computational Policy Analysis. arXiv preprint. https://doi.org/10.48550/arxiv.2404.03206

[59] OSDG, Pukelis, L., Bautista-Puig, N., Skrynik, M., & Stančiauskas, V. (2023). OSDG Community Dataset (OSDG-CD). *Zenodo*. https://doi.org/10.5281/zenodo.8397907

[60] Pukelis, L., Bautista-Puig, N., Skrynik, M., & Stančiauskas, V. (2022). OSDG 2.0: a multilingual tool for classifying text data by UN Sustainable Development Goals (SDGs). *arXiv preprint*. https://doi.org/10.48550/arXiv.2211.11252

[61] Ingram, S., Soboroff, I., & Voorhees, E. M. (2025). When LLMs Disagree: Diagnosing Relevance Filtering Bias and Retrieval Divergence in SDG Search. *arXiv preprint*. https://doi.org/10.48550/arxiv.2507.02139

[62] Tamagnone, N., Barbieri, N., & Consoli, D. (2025). From scratch to silver: Creating trustworthy training data for patent-SDG classification using Large Language Models. arXiv preprint. https://doi.org/10.48550/arxiv.2509.09303

[63] Hajikhani, A., Suominen, A., & Kässi, T. (2022). Mapping the sustainable development goals (SDGs) in science, technology and innovation: application of machine learning in SDG-oriented artefact detection. *Scientometrics*, 127(11), 6661-6693. https://doi.org/10.1007/s11192-022-04358-x

[64] Wulff, D. U., Meier, D. S., & Mata, R. (2023). SDG Knowledge Hub Dataset of SDG-labeled News Articles. *Zenodo*. https://doi.org/10.5281/zenodo.7523031

[65] Skrynnyk, M., Pukelis, L., Bautista-Puig, N., & Stančiauskas, V. (2023). SDGi Corpus: A Comprehensive Multilingual Dataset for Text Classification by Sustainable Development Goals. [Dataset].

[66] Clematide, S., Furrer, L., & Rinaldi, F. (2025). SwissText 2024 Shared Task: Automatic Classification of the United Nations' Sustainable Development Goals (SDGs) and Their Targets in English Scientific Abstracts. *University of Zurich*. https://doi.org/10.5167/uzh-275614

[67] Adauto, A., Pei, J., Jurgens, D., & Resnik, P. (2023). Beyond Good Intentions: Reporting the Research Landscape of NLP for Social Good. *Findings of the Association for Computational Linguistics: EMNLP 2023*, 31-52. https://doi.org/10.18653/v1/2023.findings-emnlp.31

[68] Li, K., Rollins, J., & Yan, E. (2024). Comparison of Sustainable Development Goals Labeling Systems based on Topic Coverage. arXiv preprint. https://doi.org/10.48550/arxiv.2408.13455

[69] Li, Y., Woodward, R., & Hou, D. (2024). Unfolding the Transitions in Sustainability Reporting. *Sustainability*, 16(2), 809. https://doi.org/10.3390/su16020809

[70] Gjorgjevikj, A., Madjarov, G., Gjorgjevikj, D., & Lameski, P. (2025). Benchmarking Sentence Encoders in Associating Indicators with Sustainable Development Goals and Targets. *IEEE Access*, 13, 3595894. https://doi.org/10.1109/access.2025.3595894

[71] Kannan, S., Gupta, A., & Venkatesh, B. (2024). Machine Learning-Based Automated Classification of SDG Alignment in Indian Sustainability Startups. [Conference paper].

[72] Kingdon, J. W. (1984). *Agendas, Alternatives, and Public Policies*. Little, Brown.

[73] Blum, S. (2018). The multiple-streams framework and knowledge utilization: Argumentative couplings of problem, policy, and politics issues. *European Policy Analysis*, 4(1), 94-117. https://doi.org/10.1002/EPA2.1029

[74] Taghizadeh, S., Haghdoost, A., Bigdeli, M., Allahverdipour, H., & Ghaffari, M. (2021). Childhood obesity prevention policies in Iran: a policy analysis of agenda-setting using Kingdon's multiple streams. *BMC Pediatrics*, 21(1), 2731. https://doi.org/10.1186/S12887-021-02731-Y

[75] Knaggård, Å. (2015). The Multiple Streams Framework and the problem broker. *European Journal of Political Research*, 54(3), 450-465. https://doi.org/10.1111/1475-6765.12097

[76] Haas, P. M. (1992). Introduction: epistemic communities and international policy coordination. *International Organization*, 46(1), 1-35. https://doi.org/10.1017/S0020818300001533

[77] Dunlop, C. A. (2009). Policy transfer as learning: capturing variation in what decision-makers learn from epistemic communities. *Policy Studies*, 30(3), 289-311. https://doi.org/10.1080/01442870902863869

[78] Martínez, L. M. (2025). Rethinking the Role of Epistemic Communities in the International Response to Pandemics. In *Handbook of Global Health Governance* (pp. 45-62). Routledge. https://doi.org/10.4324/9781003494959-3

[79] Sarkki, S., Niemela, J., Tinch, R., van den Hove, S., Watt, A., & Young, J. (2014). Balancing credibility, relevance and legitimacy: A critical assessment of trade-offs in science-policy interfaces. *Science and Public Policy*, 41(2), 194-206. https://doi.org/10.1093/SCIPOL/SCT046

[80] Rose, D. C., Sutherland, W. J., Amano, T., González-Varo, J. P., Robertson, R. J., Simmons, B. I., Wauchope, H. S., Kovacs, E., Durán, A. P., Vadrot, A. B. M., Wu, W., Dias, M. P., Di Fonzo, M. M. I., Ivory, S., Norris, L., Nunes, M. H., Nyumba, T. O., Steiner, N., Vickery, J., & Mukherjee, N. (2017). Policy windows for the environment: Tips for improving the uptake of scientific knowledge. *Environmental Science & Policy*, 113, 47-54. https://doi.org/10.1016/J.ENVSCI.2017.07.013

[81] Schut, M., Klerkx, L., Rodenburg, J., Kayeke, J., Hinnou, L. C., Raboanarielina, C. M., Adegbola, P. Y., van Ast, A., & Bastiaans, L. (2013). Beyond the research–policy interface. Boundary arrangements at research–stakeholder interfaces in the policy debate on biofuel sustainability in Mozambique. *Environmental Science & Policy*, 53, 233-244. https://doi.org/10.1016/J.ENVSCI.2012.10.007

[82] Varisco, A. E. (2018). Policy Networks and Research Utilisation in Policy. In *Research Utilisation in the Social Sciences* (pp. 23-44). Palgrave Macmillan. https://doi.org/10.1057/978-1-137-58675-9_2

[83] Gibbons, M., Limoges, C., Nowotny, H., Schwartzman, S., Scott, P., & Trow, M. (1994). *The New Production of Knowledge: The Dynamics of Science and Research in Contemporary Societies*. SAGE Publications.

[84] Raina, R. S. (2003). Beyond supply driven science. *Economic and Political Weekly*, 38(39), 4135-4144.

[85] Cowls, J., Tsamados, A., Taddeo, M., & Floridi, L. (2021). A definition, benchmark and database of AI for social good initiatives. *Nature Machine Intelligence*, 3(2), 111-115. https://doi.org/10.1038/S42256-021-00296-0

[86] Vinuesa, R., Azizpour, H., Leite, I., Balaam, M., Dignum, V., Domisch, S., Felländer, A., Langhans, S. D., Tegmark, M., & Fuso Nerini, F. (2020). The role of artificial intelligence in achieving the Sustainable Development Goals. *Nature Communications*, 11(1), 233. https://doi.org/10.1038/S41467-019-14108-Y

[87] Ferreira, A., Silva, R., & Santos, M. (2025). Human-Centered AI to Accelerate the SDGs: Evidence Map (2020–2024). *Preprints*. https://doi.org/10.20944/preprints202511.0527.v1

[88] Singh, P., Dwivedi, Y. K., Kahlon, K. S., Sawhney, R. S., Alalwan, A. A., & Rana, N. P. (2023). Artificial intelligence for Sustainable Development Goals: Bibliometric patterns and concept evolution trajectories. *Sustainable Development*, 31(6), 3915-3939. https://doi.org/10.1002/sd.2706

[89] Nedungadi, P., Menon, R., Gutjahr, G., Erickson, L., & Raman, R. (2024). Big data and AI algorithms for sustainable development goals: a topic modeling analysis. *IEEE Access*, 12, 180516-180532. https://doi.org/10.1109/access.2024.3516500

[90] Raman, R., Nair, V. K., Prakash, V., Patwardhan, A., & Nedungadi, P. (2025). Forecasting Artificial General Intelligence for Sustainable Development Goals: A Data-Driven Analysis of Research Trends. *Sustainability*, 17(16), 7347. https://doi.org/10.3390/su17167347

[91] Filho, W. L., Yang, P., Eustachio, J. H. P. P., Azul, A. M., Gellers, J. C., Gielczyk, A., Dinis, M. A. P., & Kozlova, V. (2022). Deploying digitalisation and artificial intelligence in sustainable development research. *Environment, Development and Sustainability*, 25(6), 4957-4988. https://doi.org/10.1007/s10668-022-02252-3

[92] Greif, S., Ritt, N., & Lamm, A. (2024). A systematic review of current AI techniques used in the context of the SDGs. *International Journal of Environmental Research*, 18(4), 668-705. https://doi.org/10.1007/s41742-024-00668-5

[93] Hajikhani, A., Suominen, A., & Kässi, T. (2022). Mapping the sustainable development goals (SDGs) in science, technology and innovation: application of machine learning in SDG-oriented artefact detection. *Scientometrics*, 127(11), 6661-6693. https://doi.org/10.1007/s11192-022-04358-x

[94] Isoieva, K., Ghinea, G., & Simons, G. (2024). Threats and Benefits of AI in the context of targeting SDGs: A Youth Perception Approach. *European Journal of Sustainable Development*, 13(2), 173-192. https://doi.org/10.14207/ejsd.2024.v13n2p173

[95] Hoyas, S., Sánchez-Roncero, A., Martín-Alcántara, A., Sanmiguel-Rojas, E., & Fernández-Feria, R. (2023). The Sustainable Development Goals and Aerospace Engineering: A critical note through Artificial Intelligence. *Results in Engineering*, 18, 100940. https://doi.org/10.1016/j.rineng.2023.100940

[96] Ramezani, M., Bashiri, A., Atashi, A., & Khodamoradi, F. (2024). Bibliometric Analysis of Artificial Intelligence Revolutions in Health-related Sustainable Development Goals. *Health Technology Assessment in Action*, 7(4), e14654. https://doi.org/10.18502/htaa.v7i4.14654

[97] Raghavendra, S., Aithal, P. S., & Acharya, S. (2023). Role of artificial intelligence (AI) in poverty alleviation: A bibliometric analysis. *VINE Journal of Information and Knowledge Management Systems*, 54(5), 1104-1128. https://doi.org/10.1108/vjikms-05-2023-0104

[98] Singh, P., Dwivedi, Y. K., Kahlon, K. S., Sawhney, R. S., Alalwan, A. A., & Rana, N. P. (2023). Artificial intelligence for Sustainable Development Goals: Bibliometric patterns and concept evolution trajectories. *Sustainable Development*, 31(6), 3915-3939. https://doi.org/10.1002/sd.2706

[99] Cowls, J., Tsamados, A., Taddeo, M., & Floridi, L. (2021). A definition, benchmark and database of AI for social good initiatives. *Nature Machine Intelligence*, 3(2), 111-115. https://doi.org/10.1038/S42256-021-00296-0

[100] Sætra, H. S. (2021). AI in context and the sustainable development goals: Factoring in the unsustainability of the sociotechnical system. *Sustainability*, 13(4), 1738. https://doi.org/10.3390/SU13041738

[101] Vinuesa, R., Azizpour, H., Leite, I., Balaam, M., Dignum, V., Domisch, S., Felländer, A., Langhans, S. D., Tegmark, M., & Fuso Nerini, F. (2020). The role of artificial intelligence in achieving the Sustainable Development Goals. *Nature Communications*, 11(1), 233. https://doi.org/10.1038/S41467-019-14108-Y

[102] Sætra, H. S. (2021). AI in context and the sustainable development goals: Factoring in the unsustainability of the sociotechnical system. *Sustainability*, 13(4), 1738. https://doi.org/10.3390/SU13041738

[103] Rehak, R., Becker, J., & Santarius, T. (2025). On the (im)possibility of sustainable artificial intelligence. Why it does not make sense to move faster when heading the wrong way. arXiv preprint. https://doi.org/10.48550/arxiv.2503.17702

[104] Heilinger, J. C., Nguyen, G., & Rieger, M. A. (2023). Beware of sustainable AI! Uses and abuses of a worthy goal. *AI and Ethics*, 3(3), 259-268. https://doi.org/10.1007/s43681-023-00259-8

[105] Ferreira, A., Silva, R., & Santos, M. (2025). Human-Centered AI to Accelerate the SDGs: Evidence Map (2020–2024). *Preprints*. https://doi.org/10.20944/preprints202511.0527.v1

[106] Wall, J., Krummel, V., & Müller, B. (2021). Artificial Intelligence in the Global South (AI4D): Potential and Risks. [Working paper].

[107] Ghamisi, P., Rasti, B., Yokoya, N., Wang, Q., Hofle, B., Bruzzone, L., Bovolo, F., Chi, M., Anders, K., Gloaguen, R., Atkinson, P. M., & Benediktsson, J. A. (2024). Responsible AI for Earth Observation. arXiv preprint. https://doi.org/10.48550/arxiv.2405.20868

[108] Yang-liu, E., Zhu, J., & Ding, Y. (2025). Beyond Citations: Measuring Idea-level Knowledge Diffusion from Research to Journalism and Policy-making. arXiv preprint. https://doi.org/10.48550/arxiv.2511.03378

[109] Bergman, U., Gustafsson, L. L., Höglund, P., Wettermark, B., & Vég, A. (2023). A full-document analysis of the semantic relation between European Public Assessment Reports and EMA guidelines using a BERT language model. *PLOS ONE*, 18(12), e0294560. https://doi.org/10.1371/journal.pone.0294560

[110] Zimmerman, S. (2026). Semantic Novelty Trajectories in 80,000 Books: A Cross-Corpus Embedding Analysis. [Forthcoming].

[111] Wang, X., Liu, Y., & Chen, Z. (2025). BiMax: Bidirectional MaxSim Score for Document-Level Alignment. *Findings of the Association for Computational Linguistics: EMNLP 2025*, 704. https://doi.org/10.18653/v1/2025.findings-emnlp.704

[112] Ganguly, D., Roy, D., Mitra, M., & Jones, G. J. F. (2018). Word Embedding based Semantic Cross-Lingual Document Alignment in Comparable Corpora. *Proceedings of the 11th Forum for Information Retrieval Evaluation*, 3293346. https://doi.org/10.1145/3293339.3293346

[113] Pial, M. R. H., Chaturvedi, S., & Chambers, N. (2023). GNAT: A General Narrative Alignment Tool. *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing*, 904. https://doi.org/10.18653/v1/2023.emnlp-main.904

[114] Yuan, H., Yuan, Z., & Tan, C. (2022). Cross-Domain Few-Shot Relation Extraction via Representation Learning and Domain Adaptation. *arXiv preprint*. https://doi.org/10.48550/arXiv.2212.02560

[115] Hematialam, H., Zadrozny, W., & Ghalwash, M. (2021). A Method for Computing Conceptual Distances between Medical Recommendations: Experiments in Modeling Medical Disagreement. *Applied Sciences*, 11(5), 2045. https://doi.org/10.3390/APP11052045

[116] Beyer, L., Hénaff, O. J., Kolesnikov, A., Zhai, X., & van den Oord, A. (2020). Embedding Space Correlation as a Measure of Domain Similarity. *Proceedings of the 12th Language Resources and Evaluation Conference*, 4964-4972.

[117] Bianchi, F., Di Carlo, V., Nicoli, P., & Palmonari, M. (2020). Compass-aligned Distributional Embeddings for Studying Semantic Differences across Corpora. arXiv preprint. https://doi.org/10.48550/arxiv.2004.06519

[118] Vera, P., Sordoni, A., & Reddy, S. (2025). MOSAIC: Masked Objective with Selective Adaptation for In-domain Contrastive Learning. arXiv preprint. https://doi.org/10.48550/arxiv.2510.16797

[119] Jurgens, D., Pilehvar, M. T., & Navigli, R. (2015). Semantic Similarity Frontiers: From Concepts to Documents. *Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing*.

[120] Hassan, S., Mihalcea, R., & Banea, C. (2024). UESTS: An Unsupervised Ensemble Semantic Textual Similarity Method. *Zenodo*. https://doi.org/10.60692/d0q74-95q09

[121] Measuring the Measuring Tools: An Automatic Evaluation of Semantic Metrics for Text Corpora. (2022). arXiv preprint. https://doi.org/10.48550/arxiv.2211.16259

[122] Keating, M. (2014). Cross-Border Policy Alignment Using Transformer-Based Representations and Unsupervised Clustering Models. [Conference paper].

[123] Yang, Y., Zhang, Y., & Tar, C. (2019). Bi-directional Relevance Matching between Medical Corpora. *2019 Systems and Information Engineering Design Symposium (SIEDS)*, 1-6. https://doi.org/10.1109/SIEDS.2019.8735639

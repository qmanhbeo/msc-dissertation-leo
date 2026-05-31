# Data

- OSDG
- SDGi (https://huggingface.co/datasets/UNDP/sdgi-corpus) 


# I won't use SDGi initially
I will not use the SDGi Corpus to create the initial semantic clusters for each SDG. The labels in this dataset come from high-level chapter titles in government reports, meaning a single paragraph labeled "Zero Hunger" might actually spend most of its time talking about water infrastructure or economic policy. Some specific reasons:

- Self-Reporting and Political Context: VNRs and VLRs are self-reported documents produced by governments. Bureaucrats and policy writers often frame achievements in a way that aligns with specific political goals or funding requirements.

- Author Biases: As the dataset card explicitly states under Labelling Bias, "The biases of the authors of the source documents might be reflected in... the labels they assigned to it." A government might categorize an economic policy under SDG 8 (Decent Work and Economic Growth) to highlight job creation, while an independent linguist or environmental scientist might argue the text is actually about infrastructure (SDG 9) or is actively detrimental to climate goals (SDG 13).

Using these texts first would create messy, overlapping clusters in the embedding space. Instead, I will build the initial, trustworthy "centroids" (cluster centers) using cleaner datasets like OSDG and the SDG Benchmark, which were manually labeled sentence-by-sentence by human annotators. Once those accurate boundaries are set, I can safely use the SDGi Corpus texts to test and expand the clusters without ruining the core definitions.

# Two

Policy corpus includes:
- SDGi
- UNGDC
- policy_scrape

# Three

I used OpenAlex built-in SDG filters to fetch papers. This would have likely filtered out the more deeply abstract fields like theoretical mathematics, particle physics, or niche historical texts, etc. This, hopefully, isolated the corpus to research dedicated to human and planetary survival.

# Four

Even though OpenAlex indexes over 474 million entries, the final research corpus in this paper has 2.54M titles + abstracts (I call SDG-related corpus), which is more than a human or any institution could ever read in a lifetime, and hopefully enough to be representative of the actual distribution


# Five

but the bias is, of course AI-driven SDG research is gonna focus heavily on SDG 9 and SDG 4 (due to terms like machine learning)

It is critical to acknowledge that querying a research corpus using explicit computational nomenclature ('machine learning', 'neural networks') inherently conditions the dataset toward domains with high concentrations of technical and pedagogical literature, specifically SDG 4 (Quality Education) and SDG 9 (Industry, Innovation, and Infrastructure).Rather than viewing this as an uncontrolled confounding variable, this dissertation explicitly operationalizes this distribution as a baseline for Computational Sustainability. To ensure this vocabulary bias did not compromise our structural insights, an A15 Calibration Bias control was implemented. Because the baseline calibration bias ($0.326$) exceeded the observed directional asymmetry ($0.144$), the H26 hypothesis was conservatively treated as inconclusive, preventing vocabulary artifacting from driving false positive claims.

## For main text:

Methodological Justification for Calibration Bias Controls (Assumption A15)To ensure that the observed semantic alignments reflect genuine conceptual engagement rather than mere stylistic mimicry, the analysis implements a strict calibration bias safeguard (A15). Because both the international policy corpus and the underlying OSDG training text are written in a highly stylized, institutional "UN-speak," policy documents enjoy an artificial linguistic head-start when evaluated against SDG centroids. Our pipeline mathematically isolates this baseline vocabulary inflation, establishing a structural noise threshold of $0.326$. While the raw analysis for Hypothesis 26 indicates a directional asymmetry of $0.144$—suggesting that policy frameworks actively engage with scientific research vectors more than vice versa—this observed signal is completely swallowed by the much larger $0.326$ dialect bias. Consequently, this relationship is conservatively treated as inconclusive, preventing an artifact of institutional vocabulary from being misconstrued as a genuine cross-disciplinary alignment.
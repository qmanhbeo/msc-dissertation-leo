# Formal Research Question

**To what extent does academic AI-for-sustainability research show semantic overlap with SDG priorities, and where are the gaps?**

---

**To what extent do AI-for-sustainability research and SDG policy discourse overlap with each other in semantic space, and do coverage priorities and semantic framing move together or independently across the 17 goals?**

---

**How can AI-for-sustainability research/policy be characterised in semantic space — and do coverage gaps and semantic gaps represent productive versus problematic misalignment?**

---

**To what extent do AI-for-sustainability research and SDG policy discourse align/diverge in semantic space, and what systematic gaps in coverage and semantic representation can be identified?**

---

**To what extent do AI-for-sustainability research and SDG policy discourse align in their focus across and within the Sustainable Development Goals?**





To what extent do academic research and global policy discourse align in their sustainable development priorities, and how do macro-level attention distribution and micro-level semantic framing interact across the 17 SDGs?

How can the misalignment between sustainability research and policy frameworks be characterized in high-dimensional semantic space, and does asymmetrical attention allocation predict conceptual divergence within specific Sustainable Development Goals?

What systematic variations exist in the alignment of research and policy across the Sustainable Development Goals, and to what extent do structural focus across goals and conceptual representation within goals diverge?



-------------
but the thing is, tho, my research corpora is fetched by querying "SDG + AI/ML/DS/...". What I wanted to do in the beginning was "AI-for-SDG" research vs policy, not just SDG research vs policy alone. But then I realized the policy world doesn't actually talk about this a lot. So SDG-only might be the better thing to do. But the thing is my research corpus is already that... Would it matter?

| Component        | Value                                                                                            |
| ---------------- | ------------------------------------------------------------------------------------------------ |
| SDG filter       | `sustainable_development_goals.id:https://metadata.un.org/sdg/{1..17}`                           |
| Publication year | `publication_year:>2017`                                                                         |
| Has abstract     | `has_abstract:true`                                                                              |
| Text search      | One of: `"machine learning"`, `"deep learning"`, `"artificial intelligence"`, `"neural network"` |


but the bias is, of course AI-driven SDG research is gonna focus heavily on SDG 9 and SDG 4 (due to terms like machine learning)

It is critical to acknowledge that querying a research corpus using explicit computational nomenclature ('machine learning', 'neural networks') inherently conditions the dataset toward domains with high concentrations of technical and pedagogical literature, specifically SDG 4 (Quality Education) and SDG 9 (Industry, Innovation, and Infrastructure).Rather than viewing this as an uncontrolled confounding variable, this dissertation explicitly operationalizes this distribution as a baseline for Computational Sustainability. To ensure this vocabulary bias did not compromise our structural insights, an A15 Calibration Bias control was implemented. Because the baseline calibration bias ($0.326$) exceeded the observed directional asymmetry ($0.144$), the H26 hypothesis was conservatively treated as inconclusive, preventing vocabulary artifacting from driving false positive claims.

## For main text:

Methodological Justification for Calibration Bias Controls (Assumption A15)To ensure that the observed semantic alignments reflect genuine conceptual engagement rather than mere stylistic mimicry, the analysis implements a strict calibration bias safeguard (A15). Because both the international policy corpus and the underlying OSDG training text are written in a highly stylized, institutional "UN-speak," policy documents enjoy an artificial linguistic head-start when evaluated against SDG centroids. Our pipeline mathematically isolates this baseline vocabulary inflation, establishing a structural noise threshold of $0.326$. While the raw analysis for Hypothesis 26 indicates a directional asymmetry of $0.144$—suggesting that policy frameworks actively engage with scientific research vectors more than vice versa—this observed signal is completely swallowed by the much larger $0.326$ dialect bias. Consequently, this relationship is conservatively treated as inconclusive, preventing an artifact of institutional vocabulary from being misconstrued as a genuine cross-disciplinary alignment.

--------------

# AI-focused:

In high-dimensional semantic space, to what extent do AI-for-sustainability research and global SDG policy align, and how do macro-level coverage priorities and micro-level semantic framing diverge across the 17 SDGs? 
(This is the general idea)

In semantic space, to what extent do AI-for-sustainability research and SDG policy discussion overlap, and do coverage priorities and semantic framing move together or independently across the 17 goals?
(This puts some focus on the correlation test between the two types of differences)

How can the (possible) misalignment between AI-for-sustainability research and global SDG policy discussion be characterized in semantic space, and do macro-coverage gaps and micro-semantic framing differences move together or independently across the 17 goals?
(This puts some more focus on 1. the problem of misalignment, 2. the methodology of semantic embeddings, and 3. the correlation test between the two types of differences)

In semantic space, to what extent do AI-for-sustainability research and global SDG policy discussion overlap, and what systematic differences can be identified in their macro-attention distribution across goals versus their micro-semantic representation within goals?
(This focuses on more detailed explanation of macro and micro gaps)


In general, the most important ideas to be conveyed are just:
- The framing: (possible misalignment between) AI-for-sustainability research VS global SDG policy
- The method: semantic embeddings
- The analysis: 
+ Coverage (macro attention) gaps and within-goal semantics differences between research and policy
+ And if those two are correlated in any way
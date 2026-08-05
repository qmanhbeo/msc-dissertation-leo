# Independent Scientific Architecture Verification Audit — Dissertation Argument Reconstruction

**Title examined:** *Measuring the Gap: Semantic Alignment Between AI-for-Sustainability Research and SDG Policy Frameworks*
**Role:** independent senior-researcher / journal-editor reviewer, second pass.
**Materials read in full:** `3_writing/dissertation.tex` (978 lines; abstract, intro, literature review, methodology, results, discussion, conclusion, all appendices).
**Date:** 2026-08-05
**Repo:** `/home/manh/dissertation`

This is a **verification pass**. Before making editorial recommendations, it reconstructs the dissertation's actual scientific argument and audits whether the previous editorial audit (commit d6ebed8) misread the role of the methods, hypotheses, or findings. Where the previous report was wrong it is corrected; where it was missing content (hypothesis verification, the measure/decompose/explain distinction, the validation-vs-hypothesis distinction, a corrected dependency graph) it is added. Numerical correctness is out of scope except where it bears on structure.

---

## Corrections summary (previous audit, d6ebed8)

| Previous claim | Status | Corrected verdict |
|---|---|---|
| "The paper's stated innovation is separating topic from register, introduced mid-methods" | **Wrong** | The stated innovation is a *measurement framework* separating coverage from framing (tex:111); register decomposition is an instrument *within* it, and it is foreshadowed in the intro (tex:111, 113). |
| "Register is never foreshadowed in the intro roadmap" | **Wrong** | The intro previews register removal explicitly (tex:113). |
| "H1a–H1d are introduced after the instruments that motivate them; methods-before-motivation" | **Valid but exaggerated** | The RQ itself contains the H1 relation question (tex:115); the formal H1a–H1d statement at §3.9 still precedes the tests (§4.3). Placement is presentation, not a logic error. |
| "Robustness grid reads as reviewer-driven additions" | **Wrong** | The cross-sensitivity grid is a *designed feature* of the measurement framework (tex:496) — its evidence, not its defense. |
| "Zero-shot is vestigial developmental residue" | **Wrong** | A deliberate scoping decision (single supervised-vs-unsupervised comparison under the canonical encoder), consistently applied. |
| "The SDG 4 audit must be promoted to the main text" | **Wrong** | The main text already flags SDG 4 as artefact-affected at every point of use (tex:330, 336, 377, 404); detail-in-appendix is correct caveat handling. |
| "The paper built a cathedral on a p = 0.054 finding" | **Valid but exaggerated** | The Conclusion names the *adjusted-gap ranking* as "the primary result" (tex:488), not the cancellation. The cancellation is prominent in the abstract/intro but consistently hedged ("may reflect"). The real issue is front-matter/body *emphasis inconsistency*, not a structurally overloaded claim. |
| "'Naive baseline' proves retroactive demotion" | **Exaggerated / speculative** | It is a consistent primary-vs-reference labeling convention used from the intro onward; whether it encodes history is unverifiable and irrelevant. |
| "Appendix G is process history; the trust evidence must move into the main flow" | **Valid but exaggerated** | Appendix G validates a *method* (the register reading), not a research hypothesis; method validation in an appendix is standard. The main text already discloses the partial support (tex:478, §5.4). A one-sentence validation summary in §4.2 would help; nothing more. |
| "The cancellation is the paper's headline and the structure disagrees with its own sentence at tex:439" | **Partly wrong** | tex:439 (framework = main contribution) and tex:488 (adjusted ranking = primary result) *agree* with each other; it is the abstract that previews the cancellation disproportionately. The fix is abstract/intro density, not re-weighting the whole structure. |

The three highest-impact recommendations of the previous audit survive in weakened form: re-weighting is only needed in the abstract; hypothesis placement is presentation; the appendices need compression of style (A.3, C.3, F.1, G.5 prose), not relocation of substance.

---

# Phase 1 — Reconstruct the actual research logic

## 1.1 The central research question(s)

The dissertation states a single RQ with two parts (tex:115):

> "To what extent do academic AI-for-sustainability research and international institutional SDG policy discourse differ in SDG coverage and semantic framing across the 17 SDGs, and how are these two dimensions related?"

Part 1 is a **measurement** question (how far apart are the two corpora, on two dimensions, across all 17 SDGs). Part 2 is a **relation** question (are the two dimensions themselves associated?). The two parts are not symmetric in weight: Part 1 is the framework's application; Part 2 is where the empirical interest lands (the H1 family, §3.9).

## 1.2 The theoretical framework

Three components:

1. **The two-dimension decomposition** (tex:107–111): research–policy alignment is not a single quantity; it has a *coverage* dimension (which SDGs each corpus prioritises) and a *semantic framing* dimension (how differently they discuss the same SDG).
2. **The three-level alignment construct** (tex:129–139, §2.2): Level 1 lexical/bibliometric correspondence; Level 2 observed textual-semantic proximity in embedding space; Level 3 substantive comparison of implied problems, actors, interventions. The study "is positioned at this level" (Level 2) and deliberately does not cross to Level 3.
3. **Register as the boundary of Level-2 measurement** (tex:160–164, §2.4): embedding distance mixes topic, register, and contextual convention; "semantic proximity is a distributional measurement, not evidence of substantive framing alignment" — which motivates the decomposition.

The framework is introduced in the literature review (§2.2, §2.4), before any method, and is referenced by the intro, methodology, results, discussion, and conclusion. It is not a post-hoc device.

## 1.3 The hypotheses tested

H1a–H1d, formally stated only at §3.9 (tex:285–287):

> "The first hypothesis (H1) decomposes into four sub-hypotheses, each correlating a distinct coverage predictor with the within-SDG semantic gap across the 17 SDGs: H1a, coverage gap ↔ semantic gap; H1b, research–policy dominance ↔ semantic gap; H1c, research coverage ↔ semantic gap; and H1d, policy coverage ↔ semantic gap. H1a is the motivating hypothesis; H1b–H1d examine complementary coverage-structure predictions."

The narrative antecedent is in the RQ itself ("how are these two dimensions related", tex:115). There is no H2 anywhere in the text.

## 1.4 Role of each empirical component

| Component | Role in the argument | Introduced |
|---|---|---|
| **Supervised LR classifier** (§3.5) | The *assignment instrument*. Produces the shared 17-SDG coordinate system on both corpora. It does not measure the gaps; it labels texts, which is what makes coverage profiles (§3.6) and within-SDG clusters (§3.7) well-defined. Chosen (over MLP) for transparency, not accuracy (tex:675). | Methodology §3.5 |
| **Coverage gap** (§3.6) | Measures **dimension 1**. `CoverageGap_j = \|Research_j − Policy^docwt_j\|`; signed version = research-minus-policy dominance. Document-weighted. | Methodology §3.6 (tex:234–243) |
| **Semantic gap** (§3.7) | Measures **dimension 2**. `SemanticGap_j = 1 − (c^res_j · c^pol_j)`, the raw centroid distance within each SDG. | Methodology §3.7 (tex:250) |
| **INLP register adjustment** (§3.8) | **Decomposes** the raw semantic gap into a register-adjusted (topic) component and a register component. It does not add a new measure; it refines the existing one. "After register removal, the adjusted gap is the primary estimate of within-SDG topic divergence" (tex:259). | Methodology §3.8 |
| **Adjusted semantic gap** | The **primary DV** for reporting which SDGs diverge most (SDG 17 most, SDG 15 least) and for the H1 tests. Explicitly named "the primary result" (tex:488). | Methodology §3.8; Results §4.2 |
| **Coverage–semantic interaction** (§3.9 stated; §4.3 tested) | The **H1a–H1d relation tests**: correlates each coverage predictor against the semantic gap, computed on raw, adjusted, and register versions of the DV. | Methodology §3.9; Results §4.3 |

The three roles are cleanly separable, and the dissertation keeps them separate:

1. **Measuring** semantic divergence = semantic gap (§3.7).
2. **Decomposing** semantic divergence = INLP (§3.8) → topic + register.
3. **Explaining** semantic divergence = H1a–H1d (§3.9/§4.3) + Discussion.

## 1.5 The primary contribution claimed

Two statements, and they agree:

- tex:111: "Its contribution is a measurement framework. I advance AI–SDG mapping from Level 1 (lexical correspondence) to Level 2 (observed textual-semantic proximity in embedding space)... treating semantic proximity as a register-sensitive proxy for framing."
- tex:439: "The framework is the main contribution; the alignment findings are a first application to be checked against a larger, more diverse policy corpus, not settled results."

The framework's portability claim is concretely embodied in the cross-sensitivity grid, presented as "a reusable template for bounding uncertainty" (tex:496). The empirical AI–SDG findings — the SDG 17/15 ranking and the coverage–framing relation — are applications, not the contribution.

---

# Phase 2 — Hypothesis verification

## Where H1a–H1d are introduced

Formally, in Methodology §3.9, subsection titled "Coverage–Semantic Interaction" (tex:285–287). Conceptually, in the intro's RQ ("how are these two dimensions related", tex:115). Tested in Results §4.3 (tex:381–408). This is the only hypothesis apparatus in the document.

## What exactly their mathematical relationships are

Across the 17 SDGs (j = 1..17), each is a bivariate association between one coverage predictor (defined §3.6, tex:234–243) and the within-SDG semantic gap (defined §3.7, tex:250), evaluated with Spearman ρ (canonical) and Pearson r (metric-sensitivity appendix), and computed against **three versions** of the DV (tex:383): raw gap, adjusted (topic) gap, and register component (raw − adjusted).

- **H1a**: corr(CoverageGap_j , SemanticGap_j) — the motivating hypothesis.
- **H1b**: corr(SignedCoverageGap_j , SemanticGap_j) — research-vs-policy *dominance*.
- **H1c**: corr(ResearchCoverage_j , SemanticGap_j) — research attention.
- **H1d**: corr(PolicyCoverage_j , SemanticGap_j) — policy attention.

The four predictors are the "four predictors reported in Table 4" (tex:243). The interaction grid (§4.3) reports all four against all three DV versions across every encoder–classifier config, so H1a–H1d are at once coverage-relation hypotheses *and* decomposition-sensitive tests — but the *object* of each hypothesis is the coverage-structure → gap association.

## Are they (a) coverage predictors predicting semantic gaps? (b) register-decomposition hypotheses? (c) something else?

**(a) Yes.** Each hypothesis relates a coverage-structure predictor to within-SDG semantic divergence. They are correlational (n = 17; explicitly underpowered, tex:287), not causal claims; the regression machinery (Appendix J) adds controls across configs but the hypotheses themselves are bivariate.

**(b) No.** H1a–H1d are not hypotheses about the INLP register decomposition. The decomposition is not the object of any H1 test. Its role is entirely instrumental: it produces the *adjusted* DV column (and the register-component DV column) that the H1 correlations are then computed on.

**(c) No.** They are not framing of anything more abstract (e.g., "knowledge-system divergence"); the interpretation of the results in those terms is Discussion (§5.1), downstream of the tests.

### Explicit answer

**"Would it be scientifically accurate to describe H1a–H1d as hypotheses about the INLP register decomposition?"**

**No.**

The decomposition is neither the predictor nor the predicted in any H1 test; it is the instrument that *produces a cleaner version of the DV*. Three asymmetries make the description inaccurate:

1. **Direction of dependence.** The hypotheses depend on the decomposition only through the DV's adjusted/register columns; the decomposition depends on nothing from H1a–H1d. A hypothesis "about" a method would test the method's properties; here the method's properties are validated separately (Appendix G), not hypothesized.
2. **What is vs. is not claimed.** H1a–H1d claim nothing about whether the removed subspace is register, whether it is well-estimated, or whether the projection preserves topic. Those are Appendix G questions. Mixing them into H1 would misstate both the hypotheses and the validation.
3. **Consistency with the text's own framing.** The text says the decomposition "suggests this may reflect cancellation" (abstract, tex:92) — the cancellation is an *interpretation of the H1 results*, not a hypothesis about the decomposition.

The previous audit did not explicitly make this error, but it blurred the boundary in one place: it framed V1 as "methods explained before their motivation," which implied the decomposition *motivated* the hypotheses. The true relation is the reverse: the RQ's relation question motivates H1a–H1d; the decomposition *refines the DV* they are tested on.

---

# Phase 3 — Corrected dependency graph

The dissertation's logic is (departing from the previous audit's graph only in where the hypotheses sit):

```
Research question (tex:115: measure coverage & framing divergence; how are the dimensions related?)
        │
        ▼
Conceptual framework (tex:107–139: coverage vs. framing; three-level construct; register as Level-2 boundary)
        │
        ▼
Hypotheses — H1a–H1d (conceptually implied by RQ part 2; formal statement §3.9)
        │
        ▼
Measurement construction
   ├─ assignment instrument: supervised LR classifier (§3.5) → 17-SDG labels
   ├─ dimension 1: coverage gap (§3.6)
   ├─ dimension 2: semantic gap, raw (§3.7)
   └─ decomposition instrument: INLP register adjustment → adjusted (topic) gap = PRIMARY DV (§3.8)
        │
        ▼
Empirical tests (Results)
   ├─ instrument validity: classifier held-out F1, coverage profiles (§4.1)
   ├─ measurement: semantic gap, raw vs. adjusted, register decomposition (§4.2)
   ├─ H1a–H1d tests on raw/adjusted/register DV (§4.3)
   └─ robustness of rankings: encoder, cross-method, sample-stability, distributional (§4.4 + appendices)
        │
        ▼
Interpretation (Discussion: cancellation as knowledge-system structure; robust patterns; implications; limitations)
```

Key corrections to the previous graph:

1. **Hypotheses precede measurement in dependency**, even though they are *textually* stated after measurement (§3.9). The RQ contains them; the measurement is built to test them. The previous audit's V1 ("methods-before-motivation") treated textual order as logical order.
2. **The decomposition is a measurement instrument, not a hypothesis and not a finding.** It sits on the measurement-construction row, below the semantic gap it refines. The previous audit's dependency graph placed it correctly as a child of the semantic gap but then treated it as the paper's center of gravity; it is an instrument whose *primary output* (the adjusted gap) is what the findings use.
3. **Validation of the instrument (Appendix G) is method validation, downstream of measurement, parallel to — not feeding — H1a–H1d.** It validates that "adjusted" can be read as "topic." It is not a test of any hypothesis.
4. **The "cancellation" is an interpretation** produced at the analysis/interpretation boundary (§4.3 result → §5 interpretation), not a pre-registered expectation and not a step in the graph. Its correctness is the only legitimate robustness concern, and the text discloses its tentativeness (tex:92) and its underpowering (tex:396).

The three-way role distinction the audit must respect throughout:

- **(1) Explaining** semantic divergence — H1a–H1d + Discussion (why is the gap what it is, and what does the gap mean).
- **(2) Measuring** semantic divergence — semantic gap (§3.7).
- **(3) Decomposing** semantic divergence — INLP (§3.8).

---

# Phase 4 — Architecture audit (reclassified)

## 4.1 Which sections genuinely feel added later

Only a small subset, and most are *style* problems, not architecture problems:

| Section | Genuine issue | Severity |
|---|---|---|
| Appendix A.3 "Truncation Fix" | Documents a mid-development bug fix as an appendix (tex:524–527: "During pipeline development, a truncation issue was discovered and corrected"). This is developmental residue; it belongs as a sentence in §3.4 or a footnote. | Low (genuine) |
| Appendix C.3 "Implications for Interpreting the Main Estimates" | Discussion-prose inside an appendix (tex:608–612) that re-litigates the classifier choice. Belongs in §3.5 or §5.4. | Low (genuine) |
| Appendix F.1 | Duplicates the §4.2 SDG-17 register-component story nearly verbatim (tex:366 vs. tex:697). Redundancy. | Low (genuine) |
| Appendix G.5 | Style: reports "the robustness investigation as it unfolded" and "This history is reported deliberately" (tex:756, 762). The *content* is legitimate draw-stability robustness analysis; only the narration is unusual. | Low (style only) |
| Appendix H opening | "Advisor review flagged the four-term keyword query" (tex:798) is a transparency time-stamp; the appendix content (retrieval-sensitivity) is a legitimate scope-condition test. | None (note) |

## 4.2 Which sections are necessary consequences of the design (not bolt-ons)

These were flagged or implicitly suspected by the previous audit but are in fact *required by the design*:

- **The register decomposition (§3.8) and Appendices F/G.** The framework commits to treating semantic proximity as "a register-sensitive proxy" (tex:111). A reader-credible Level-2 measure therefore *must* deal with register; the decomposition and its validation are consequences of the framework's own promise, not afterthoughts.
- **The cross-sensitivity grid (Appendix I) and its synthesis (§4.4).** The framework's portability claim is operationalised as "the stability of every position is transparent across the measurement choices analysts control" (tex:496). A measurement framework that did not report this would be incomplete. This is evidence *for* the framework, not defense against a reviewer.
- **Sample-stability (Appendix D).** Corpus size is a design choice (1,000 → 3.1M papers); demonstrating the plateau is required for any size-dependent claim.
- **Encoder sensitivity (Appendices C/I).** The embedding space is the coordinate system; its choice is a principal axis of the framework.
- **Concept-retrieval sensitivity (Appendix H).** Retrieval defines the corpus boundary, and the paper's thesis is partly that coverage is method-contingent (Armitage-style); testing the boundary is part of the argument.
- **Model selection (Appendix E).** The classifier is the assignment instrument; its selection rationale is needed because the instrument is load-bearing.
- **Pooled regression (Appendix J).** Provides the config-pooled corroboration that the n=17 bivariate tests cannot.

The previous audit's Phase-4 criticism that the "robustness story is fragmented across 4–5 appendix sections" conflated *volume* with *bolt-on-ness*: a framework whose contribution is measurement stability is *supposed* to be robustness-heavy. The real (minor) problem is repetition across §4.4, §5.2, and §5.4, not the existence of the grid.

## 4.3 Which sections are misplaced

Genuinely misplaced (small set):

1. **§3.9 title "Coverage–Semantic Interaction" hides the hypotheses.** A ToC scan will not reveal that the paper has hypotheses at all. The subsection is really "hypotheses + analysis plan." Rename (presentation).
2. **Appendix C.3** — discussion in appendix (as above).
3. **Appendix F.1** — redundant narrative (as above).
4. **§3.10 "Methodology Summary"** at the end of methods — a pipeline figure is a natural orientation device; moving it to the opening of Methodology is *mildly* better but the current placement is defensible. Not a real problem.

Not misplaced (previous audit flagged, now rejected):

- §3.1/§3.5 scope-condition statements (asymmetry, probability gap) are appropriately placed where the constructs are defined; they are scope conditions, not premature defenses.
- §4.2's placement of the PCA figure before the gap figure is fine (visual motivation of the decomposition that produces the primary DV).
- Hypotheses at §3.9 rather than end of literature review: defensible for a methods-heavy dissertation; both placements work. Presentation only.

## 4.4 Classification of every previous criticism

| # | Previous criticism | Classification |
|---|---|---|
| V1 | "Methods explained before their motivation; hypotheses arrive after instruments" | **Valid but exaggerated** — real only as a presentation point; the RQ contains the hypotheses' motivation (tex:115), and they still precede the tests. |
| V2 | "Register validation evidence provided after the claim it supports" | **Valid but exaggerated** — method validation in an appendix is standard; the main text discloses the partial support (§5.4). A one-sentence summary in §4.2 would suffice. |
| V3 | "§3.9 is hypotheses masquerading as an analysis-plan section" | **Valid structural issue** (minor) — the title hides the hypotheses. Rename. |
| V4 | "Register concept introduced after it is load-bearing; not foreshadowed in intro" | **Based on incorrect interpretation** — register is defined in §2.4, operationalised in §3.8, and previewed in the intro (tex:111, 113). |
| V5 | "Robustness defenses embedded in §3.1/§3.5 before the reader needs them" | **Valid but exaggerated** — they are scope conditions where the constructs are defined; the defensive phrasing is tone, not placement. |
| V6 | "Pipeline summary at end of methods" | **Not a problem** (or presentation at most) — a summary figure at the end of a long methods chapter is standard. |
| V7 | "Distributional-robustness motivation appears after its contradiction (C.2)" | **Valid but exaggerated** — the content is a legitimate measurement clarification; the narrative framing is style. |
| A1 | "INLP is the paper's central method but arrives as one subsection, validation exiled to appendix" | **Valid but exaggerated** — INLP is an *instrument* of the framework, not the framework; the framework is the contribution and it is foregrounded. |
| A2 | "Robustness grid reads as reviewer-driven additions" | **Based on incorrect interpretation** — the grid is a designed feature and the framework's core evidence (tex:496). |
| A3 | "Zero-shot nearest-centroid is vestigial residue" | **Based on incorrect interpretation** — a deliberate, consistently applied scoping decision (MPNet-only, one supervised-vs-unsupervised comparison). |
| B1 | "Register concept's interpretive payoff only established in Discussion" | **Valid but exaggerated** — §3.8 already declares the adjusted gap primary; Discussion interprets, which is its job. |
| B2 | "Limitations duplicated in Methodology" | **Valid but exaggerated** — presentation; the facts are needed in methods, the adjudication in §5.4. |
| B3 | "Appendix C.3 is a Discussion chapter inside an appendix" | **Valid structural issue** (minor) — genuine. |
| C1 | "Cancellation previewed before the reader is equipped to evaluate it" | **Valid but exaggerated** — abstracts preview results by convention; §4.2 delivers the machinery before §4.3 delivers the claim. The real issue is *front-matter density*, not order. |
| C2 | "SDG 4 audit must be promoted to the main text" | **Based on incorrect interpretation** — the caveat is already flagged at every point of use; detail-in-appendix is correct. |
| C3 | "Multiple appendices are process diaries (A.3, C.2, F.1, G.5)" | **Partly valid** — A.3 (genuine residue), F.1 (duplication), C.3 (misplaced prose). C.2 and G.5 are robustness content with a narrative *style*; their existence is correct. |
| C4 | "The paper built a cathedral on a p = 0.054 finding" | **Valid but exaggerated** — the Conclusion's "primary result" is the adjusted-gap *ranking* (tex:488), and the cancellation is consistently hedged. The genuine residual issue is that the abstract previews the cancellation with numbers while the body weighs the framework — an emphasis mismatch to smooth, not a structural overload. |

---

# Phase 5 — ToC audit (corrected)

The previous audit's ToC ratings were largely right and are retained where noted. Only genuine improvements are listed; each is classified as meaning-changing or presentation-only.

**Every change below is presentation-only. None affects the scientific meaning of any claim.**

| # | Proposed change | Problem it solves | Why the current order fails | Presentation / meaning |
|---|---|---|---|---|
| 1 | Rename §3.9 from "Coverage–Semantic Interaction" to "Hypotheses: The Coverage–Semantic Interaction" (or similar) | The paper's only hypotheses are invisible to a ToC scan; a reader cannot find where the research questions are operationalised. | The current heading describes the object of study, not the scientific claim, so the hypotheses look like a byproduct. | Presentation |
| 2 | Trim the abstract's cancellation emphasis (move the p = 0.054/0.045 preview to one sentence, or to "the front matter previews the ranking as primary" instead) | The front matter and the Conclusion disagree on what is primary: abstract/intro weight the cancellation; Conclusion weights the adjusted ranking (tex:488). | Inconsistent emphasis forces the reader to re-derive which finding is load-bearing. | Presentation |
| 3 | Compress Appendix A.3 ("Truncation Fix") to one sentence in §3.4 | Removes the most visible developmental-residue fingerprint. | A bug-fix narrative is not supporting material. | Presentation |
| 4 | Fold Appendix C.3 into §3.5 (design rationale) or §5.4 (limitations) | Removes a discussion chapter from inside an appendix. | Its content is argumentative self-critique, not supplementary evidence. | Presentation |
| 5 | Cut the F.1 duplication of the §4.2 SDG-17 narrative | Removes verbatim redundancy. | The same register-component story is told twice (tex:366, tex:697). | Presentation |
| 6 | Add a one-sentence validation summary in §4.2 ("Appendix G evaluates the register reading against independent linguistic markers and finds partial, not complete, support") | Lets the reader judge the primary DV's trustworthiness where it is first used, without opening the appendix. | Currently the reader meets the primary estimate before any statement of how well "register" is identified. | Presentation |
| 7 | Reword Appendix G.5 from "the investigation as it unfolded" to a results-shaped narrative | Removes process-history narration from a robustness section. | A section structured as "what I checked, then what I found" reads as a lab notebook. | Presentation |
| 8 | (Optional) Move the formal H1a–H1d statement from §3.9 to the end of the literature review | Would place hypotheses where journals expect them. | Current placement (end of methodology) is defensible for a methods-heavy dissertation; both work. | Presentation |

No changes to the Results order (instrument → coverage → semantic gap → interaction → robustness) are needed; it already follows main-finding → interpretation → robustness correctly. No change to the Discussion order is needed. No appendix needs to be deleted wholesale except A.3's content.

---

# Phase 6 — Final verdict

## 1. The true intellectual spine (one paragraph)

The dissertation builds a **measurement framework** that separates research–policy alignment into two independent dimensions — SDG coverage and within-SDG semantic framing — and moves AI–SDG mapping from Level 1 (lexical correspondence) to Level 2 (observed textual-semantic proximity in embedding space). A frozen embedding space plus a transparent supervised logistic-regression classifier supplies a shared 17-SDG coordinate system; from it, the coverage gap measures the first dimension and the semantic gap measures the second; Iterative Nullspace Projection then **decomposes** the semantic gap into a register-adjusted (topic) component and a register component, so that the adjusted gap can serve as the primary estimate of within-SDG topic divergence. Applied to AI-for-sustainability research versus international institutional SDG policy discourse across all 17 SDGs, the framework finds that the two dimensions appear unrelated on the raw gap but positively related on the adjusted (topic) gap while the register component is negatively related — a possible cancellation — and that the adjusted ranking (SDG 17 most divergent, SDG 15 least) is stable across encoders, retrieval, segment caps, and policy-source families. The contribution is the framework and its portability template; the AI–SDG findings are a first application, explicitly not settled results.

## 2. The three highest-impact structural improvements

1. **Make the front matter agree with the Conclusion about what is primary.** Reduce the abstract/intro emphasis on the cancellation (with its p-values) and let the adjusted-gap ranking and the framework carry the preview. This is the single biggest credibility lever: it removes the impression that a tentative n=17 decomposition finding is the headline, and it makes tex:439, tex:488, and the abstract tell one story.
2. **Surface the hypotheses.** Rename §3.9 so H1a–H1d are findable; optionally move the formal statement to the end of the literature review. The hypotheses are cleanly stated and correctly placed before their tests; they are simply invisible in the ToC.
3. **Compress the developmental-residue and redundancy (A.3 → one sentence in §3.4; C.3 → fold into §5.4; F.1 → cut duplication; G.5 → results-shaped prose; add a one-sentence G-summary in §4.2).** These changes remove the visible "patched things together" fingerprints without touching a single result.

## 3. Three things that should not be changed

1. **The framework-first design** — the coverage/framing decomposition and the three-level construct (§2.2) are the genuine contribution and are consistently carried through every chapter. Do not touch.
2. **The results arc** — instrument validation → coverage → semantic gap → register decomposition → interaction → robustness. Its internal order is the correct one: establish the measure, report the gap, decompose it, test the relation, stress-test. Do not reorder.
3. **The measurement-stability architecture** — the cross-sensitivity grid, the raw/adjusted Panel (a)/(b) convention, the sample-stability ladder, and the register validation in Appendix G. This is the framework's evidence, not its defense; method validation in an appendix is standard practice. Keep it intact.

## 4. Earlier editorial recommendations that should be rejected

1. **Reject "re-weight the thesis around the framework and demote the cancellation" as a structural change.** The body already does this (tex:439, tex:488). The only action needed is abstract/intro emphasis, not restructuring.
2. **Reject "promote the SDG 4 audit to the main text."** The caveat is already flagged at every point of use; detail-in-appendix is correct handling.
3. **Reject "the zero-shot method is vestigial residue and should be cut or minimized further."** It is a deliberate scoping decision, already minimized to one comparison.
4. **Reject "the robustness grid is reviewer-driven and should be consolidated."** It is a designed feature and the framework's core evidence. Only the *repetition* of its conclusions across §4.4/§5.2/§5.4 should be trimmed, and that is cosmetic.
5. **Reject the strong form of "move the register validation into the main flow."** A one-sentence summary in §4.2 is enough; the appendix's existence and detail are correct.
6. **Reject "the paper is built on a p = 0.054 cathedral."** The Conclusion's primary result is the adjusted-gap ranking; the cancellation is a hedged, secondary interpretation with its power limitation disclosed up front.

---

**Bottom line.** The argument, reconstructed without prejudice, is a designed measurement-framework study: RQ → framework → hypotheses → instrument construction (including the register decomposition as a measurement refinement) → tests → interpretation. The structure supports this logic; the previous audit's strongest criticisms were either presentation-level (abstract emphasis, hypothesis visibility, appendix style) or based on misreading the register decomposition as a hypothesis/headline rather than an instrument. The science should not be restructured; the front matter should be made to state its own Conclusion's priorities, and the visible residue (A.3, C.3, F.1, G.5's narration) should be smoothed.
